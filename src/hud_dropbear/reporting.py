"""Reproducible cohort gates and explicitly bounded latency measurements."""

import json
import math
from collections import defaultdict
from pathlib import Path
from uuid import UUID


def distribution(values):
    """Linear-interpolated quantiles; retain sample count and range."""
    values = sorted(float(v) for v in values)
    if any(not math.isfinite(v) or v < 0 for v in values):
        raise ValueError("Timing samples must be finite and nonnegative")
    if not values:
        return {"n": 0, "min": None, "p50": None, "p95": None, "max": None}

    def percentile(q):
        index = (len(values) - 1) * q
        lower, upper = math.floor(index), math.ceil(index)
        return values[lower] + (values[upper] - values[lower]) * (index - lower)

    return {
        "n": len(values),
        "min": values[0],
        "p50": percentile(0.5),
        "p95": percentile(0.95),
        "max": values[-1],
    }


def timing_report(events):
    """Do not add overlapping SDK/server/GPU spans or subtract different clocks."""
    episode_starts, first_inference = {}, {}
    sim_starts, lease_acquire, control_ready = {}, [], []
    setup, first_action, steady, all_inference = [], [], [], []
    first_model_input, command_first_action = [], []
    provider_ready, encoding, server = [], [], defaultdict(list)
    provider_create, discovery = [], []
    attempts, posts, outcomes = [], [], defaultdict(int)
    journal_before_post, journal_terminal = [], []
    transport_uncertain, recovery, terminal_failures, fenced = set(), set(), set(), set()
    for row in events:
        event, episode = row["event"], row.get("episode_id")
        elapsed = row.get("elapsed_s")
        if event == "inference_post" and "journal_before_post_s" in row:
            journal_before_post.append(row["journal_before_post_s"])
        if (
            event
            in {"provider_inference", "inference_failed", "inference_cancelled", "slot_fenced"}
            and "journal_terminal_s" in row
        ):
            journal_terminal.append(row["journal_terminal_s"])
        if event == "environment_starting" and episode:
            episode_starts[episode] = elapsed
        elif event == "simulator_starting":
            sim_starts[row.get("lane_id")] = elapsed
        elif event == "runtime_lease_acquired" and row.get("lane_id") in sim_starts:
            lease_acquire.append(elapsed - sim_starts[row.get("lane_id")])
        elif event == "simulator_control_ready" and row.get("lane_id") in sim_starts:
            control_ready.append(elapsed - sim_starts[row.get("lane_id")])
        elif event in {"environment_ready", "first_action_confirmed"}:
            if episode in episode_starts:
                target = setup if event == "environment_ready" else first_action
                target.append(elapsed - episode_starts[episode])
            if event == "first_action_confirmed":
                command_first_action.append(elapsed)
        elif event == "episode_first_model_input" and episode in episode_starts:
            first_model_input.append(elapsed - episode_starts[episode])
        elif event == "inference":
            duration = row["duration_s"]
            all_inference.append(duration)
            if episode is not None and episode not in first_inference:
                first_inference[episode] = duration
            elif episode is not None:
                steady.append(duration)
        elif event == "provider_ready":
            provider_ready.append(row["duration_s"])
        elif event == "provider_created":
            provider_create.append(row["duration_s"])
        elif event == "provider_connections_ready":
            discovery.append(row["duration_s"])
        elif event == "inference_encode":
            encoding.append(row["duration_s"])
        elif event == "provider_inference":
            for key, value in row.get("timing", {}).items():
                server[key].append(value)
        elif event == "inference_attempt_finished":
            attempts.append(row["duration_s"])
            outcomes[row["outcome"]] += 1
        elif event == "inference_post":
            posts.append(row["duration_s"])
            if not row.get("response_received"):
                transport_uncertain.add(row.get("request_id"))
        elif event == "inference_recovery":
            recovery.add(row.get("request_id"))
        elif event == "inference_failed":
            terminal_failures.add(row.get("request_id"))
        elif event == "slot_fenced":
            fenced.add(row.get("request_id"))
    return {
        "units": {"client": "seconds", "server": "milliseconds"},
        "runtime_lease_acquire_s": distribution(lease_acquire),
        "runtime_request_to_control_ready_s": distribution(control_ready),
        "inference_session_readiness_s": distribution(provider_ready),
        "deployment_create_s": distribution(provider_create),
        "inference_connection_discovery_s": distribution(discovery),
        "episode_task_setup_s": distribution(setup),
        "episode_start_to_first_model_input_s": distribution(first_model_input),
        "client_entry_to_first_executed_action_s": min(command_first_action)
        if command_first_action
        else None,
        "episode_start_to_first_action_s": distribution(first_action),
        "episode_first_inference_s": distribution(first_inference.values()),
        "steady_inference_s": distribution(steady),
        "successful_model_calls_s": distribution(all_inference),
        "all_model_attempts_s": distribution(attempts),
        "model_attempt_outcomes": dict(outcomes),
        "sdk_post_attempts_s": distribution(posts),
        "journal_before_post_s": distribution(journal_before_post),
        "journal_terminal_s": distribution(journal_terminal),
        "request_counts": {
            "post_attempts": len(posts),
            "unknown_post_outcomes": len(transport_uncertain),
            "recovery_requests": len(recovery),
            "terminal_failures": len(terminal_failures),
            "fenced_requests": len(fenced),
        },
        "encoding_and_client_queue_s": distribution(encoding),
        "server_reported_ms": {key: distribution(vals) for key, vals in sorted(server.items())},
        "limitations": [
            "Provider readiness includes allocation and initialization; separate phases require "
            "provider-side evidence. No allocation time is inferred by subtraction.",
            "Client, gateway, scheduler and GPU measurements may overlap; do not sum them.",
            "Each episode's first inference is excluded from steady-state latency.",
            "The client-entry clock starts in the CLI module before SDK imports; Python "
            "interpreter launch and earlier standard-library imports are excluded. Full command "
            "startup requires an external parent-process timestamp.",
            "Successful model calls include client encoding, durable request journaling and "
            "any recovery. SDK POST attempts exclude those outer costs. Failed and cancelled "
            "attempts are retained separately, never treated as fast successful responses.",
            "Journal phases include shared-lock wait, write, flush and fsync, and overlap the "
            "outer model call. A cancellation before submission has no POST event; its journal "
            "time remains unseparated inside the cancelled model-attempt duration.",
        ],
    }


def concurrency_report(events):
    """Prove client overlap on one clock; this is not simultaneous GPU execution.

    Active episodes begin after task setup and end when the robot loop exits.
    POST intervals cover HTTP requests only; their overlap does not establish GPU
    batch width, worker count, or GPU utilization.
    """
    starts, episodes, posts, invalid = {}, [], [], []
    ready_capacity = 0
    for row in events:
        event, episode = row["event"], row.get("episode_id")
        if event == "provider_ready":
            capacity = row.get("ready_robots", 0)
            if type(capacity) is int and capacity >= 0:
                ready_capacity = max(ready_capacity, capacity)
        if event == "environment_ready":
            if not episode or episode in starts:
                invalid.append("Missing or duplicate active episode identity")
            else:
                starts[episode] = row
        elif event in {"episode_driven", "episode_error"} and episode in starts:
            first = starts.pop(episode)
            episodes.append((first.get("elapsed_s"), row.get("elapsed_s"), first.get("lane_id")))
        elif event == "inference_post":
            end, duration = row.get("elapsed_s"), row.get("duration_s")
            if type(end) in (int, float) and type(duration) in (int, float):
                posts.append((end - duration, end, row.get("robot_slot")))
            else:
                invalid.append("Missing HTTP attempt interval")
    if starts:
        invalid.append("Active episode intervals were not closed")

    def peak(intervals):
        boundaries = []
        for begin, end, identity in intervals:
            if (
                type(identity) is not int
                or type(begin) not in (int, float)
                or type(end) not in (int, float)
                or not math.isfinite(begin)
                or not math.isfinite(end)
                or begin < 0
                or end <= begin
            ):
                invalid.append("Invalid concurrency interval")
                continue
            # Ends sort before starts: touching intervals do not overlap.
            boundaries.extend(((begin, 1, identity), (end, -1, identity)))
        active, maximum, maximum_seconds = defaultdict(int), 0, 0.0
        previous = None
        for stamp, delta, identity in sorted(boundaries):
            if previous is not None and stamp > previous:
                count = sum(value > 0 for value in active.values())
                if count > maximum:
                    maximum, maximum_seconds = count, stamp - previous
                elif count == maximum:
                    maximum_seconds += stamp - previous
            active[identity] += delta
            previous = stamp
        return {"peak_distinct_lanes": maximum, "seconds_at_peak": maximum_seconds}

    episode_overlap, post_overlap = peak(episodes), peak(posts)
    return {
        "active_episodes": {**episode_overlap, "closed_intervals": len(episodes)},
        "sdk_posts": {**post_overlap, "attempts": len(posts)},
        "provider_ready_robots": ready_capacity,
        "errors": sorted(set(invalid)),
        "limitation": "Client overlap and advertised ready capacity do not prove GPU worker "
        "count, batch size, or utilization; use server and provider evidence for those claims.",
    }


def cohort_gate(summary, *, concurrency, minimum_success_rate=0.5):
    if type(concurrency) is not int or not 1 <= concurrency <= 64:
        raise ValueError("concurrency must be from 1 to 64")
    if not 0 <= minimum_success_rate <= 1:
        raise ValueError("minimum_success_rate must be between zero and one")
    runs = summary.get("runs", [])
    expected = summary.get("expected_episodes", 0)
    errors, reasons = [], []
    if summary.get("error_type"):
        reasons.append("The cohort ended with an operational error")
    if type(expected) is not int or expected < concurrency:
        reasons.append("Missing or invalid expected episode count")
        expected = max(len(runs), concurrency)
    identities = [r.get("case_index") for r in runs]
    if any(i is None for i in identities) or len(set(identities)) != len(identities):
        reasons.append("Missing or duplicate case identities")
    elif set(identities) != set(range(expected)):
        reasons.append("Case identities differ from the frozen cohort")
    if len(runs) != expected:
        reasons.append("Recorded episode count differs from the frozen cohort")
    if summary.get("concurrency") != concurrency:
        reasons.append("Concurrency differs from the frozen cohort")
    fixed_task_coverage = None
    if concurrency == 1:
        required = {(task_id, init_state_id) for task_id in range(3) for init_state_id in range(2)}
        observed, mismatched = set(), []
        for run in runs:
            args, case = run.get("args"), run.get("case_index")
            valid_args = isinstance(args, dict) and all(
                type(args.get(field)) is int for field in ("task_id", "init_state_id", "seed")
            )
            pair = (args["task_id"], args["init_state_id"]) if valid_args else None
            if pair in required:
                observed.add(pair)
            if (
                not valid_args
                or args["seed"] != 0
                or type(case) is not int
                or case < 0
                or pair != divmod(case % 6, 2)
            ):
                mismatched.append(case)
        missing = required - observed
        fixed_task_coverage = {
            "required_task_initial_states": [list(pair) for pair in sorted(required)],
            "observed_task_initial_states": [list(pair) for pair in sorted(observed)],
            "missing_task_initial_states": [list(pair) for pair in sorted(missing)],
            "mismatched_case_indices": mismatched,
        }
        if missing:
            reasons.append("Single-lane campaign must cover all six fixed task/initial-state pairs")
        if mismatched:
            reasons.append("Single-lane task arguments differ from frozen case assignments")
    try:
        trace_ids = [str(UUID(run.get("trace_id") or "")) for run in runs]
        if len(set(trace_ids)) != len(trace_ids):
            reasons.append("Episodes reuse a HUD trace identity")
    except (ValueError, TypeError, AttributeError):
        reasons.append("Missing or invalid HUD trace identities")
    episode_ids = [run.get("episode_id") for run in runs]
    if any(not isinstance(i, str) or not i for i in episode_ids) or len(set(episode_ids)) != len(
        runs
    ):
        reasons.append("Missing or duplicate episode identities")
    successes = 0
    for run in runs:
        reward = run.get("reward")
        if (
            run.get("integration_error")
            or not isinstance(reward, (int, float))
            or not math.isfinite(reward)
            or reward not in (0, 1)
            or not run.get("trace_id")
        ):
            errors.append(run.get("case_index"))
        elif isinstance(reward, (int, float)) and math.isfinite(reward) and reward == 1:
            successes += 1
    rate = successes / expected
    if errors:
        reasons.append("Integration errors or missing episode grades/traces")
    if rate < minimum_success_rate:
        reasons.append("Success rate is below the cohort threshold")
    lanes = {r.get("lane_id") for r in runs}
    if lanes != set(range(concurrency)):
        reasons.append("Not every requested simulator lane produced an episode")
    overlap = summary.get("concurrency_evidence", {})
    if overlap.get("errors"):
        reasons.append("Concurrency evidence contains incomplete or invalid intervals")
    if overlap.get("active_episodes", {}).get("peak_distinct_lanes", 0) != concurrency:
        reasons.append("Requested simulator concurrency was not observed")
    if overlap.get("provider_ready_robots", 0) < concurrency:
        reasons.append("Requested inference capacity was not observed ready")
    provenance = summary.get("provenance", {})
    if provenance.get("runtime", summary.get("runtime")) != "hud":
        reasons.append("Simulation did not run on HUD")
    if provenance.get("max_steps", summary.get("max_steps")) != 600:
        reasons.append("Episode action limit differs from the evaluation contract")
    platform = summary.get("platform_evidence", summary.get("platform", {}))
    if not isinstance(platform, dict) or platform.get("verified") is not True:
        reasons.append("HUD platform grades and camera traces are not verified")
    if summary.get("provider_cleanup_confirmed") is not True:
        reasons.append("Dropbear deployment termination is not confirmed")
    if summary.get("simulator_cleanup", {}).get("verified") is not True:
        reasons.append("HUD simulator termination is not confirmed")
    return {
        "passed": not reasons,
        "expected_episodes": expected,
        "recorded_episodes": len(runs),
        "successes": successes,
        "success_rate": rate,
        "minimum_success_rate": minimum_success_rate,
        "fixed_task_coverage": fixed_task_coverage,
        "integration_error_cases": errors,
        "reasons": reasons,
    }


def write_campaign_report(output_dir: Path, *, concurrency: int, minimum_success_rate: float = 0.5):
    directory = Path(output_dir)
    summary = json.loads((directory / "results.json").read_text())
    path = directory / "timings.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    overlap = concurrency_report(events)
    cost_path = directory / "cost.json"
    report = {
        "schema_version": 1,
        "cohort_sha256": summary.get("cohort_sha256"),
        "job_url": summary.get("job_url"),
        "concurrency": concurrency,
        "gate": cohort_gate(
            {**summary, "concurrency_evidence": overlap},
            concurrency=concurrency,
            minimum_success_rate=minimum_success_rate,
        ),
        "concurrency_evidence": overlap,
        "timings": timing_report(events),
        "cost": json.loads(cost_path.read_text())
        if cost_path.exists()
        else {
            "status": "not_reconciled",
            "actual_usd": None,
            "note": "No cost is invented from inference-only GPU time; startup, idle and cleanup "
            "must be included and provider billing may lag.",
        },
    }
    (directory / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
