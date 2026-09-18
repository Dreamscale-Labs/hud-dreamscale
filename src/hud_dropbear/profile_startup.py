"""Measure simulation startup without allocating inference or sending policy actions."""

import argparse
import asyncio
import json
import statistics
import time
from pathlib import Path
from uuid import uuid4

from .contract import (
    ENV_NAME,
    LEGACY_PROFILE,
    POOLED_ENV_NAME,
    POOLED_PROFILE,
    PROFILES,
    build_contract,
)
from .startup import StartupProfile


def validate_observation(observation, profile):
    import numpy as np

    data = observation.get("data", {})
    for key, feature in build_contract(profile=profile)["features"].items():
        if feature["role"] != "observation":
            continue
        array = np.asarray(data.get(key))
        if array.shape != tuple(feature["shape"]) or array.dtype != np.dtype(feature["dtype"]):
            raise ValueError(f"Unexpected initial observation contract: {key}")
        if not np.isfinite(array).all():
            raise ValueError(f"Nonfinite initial observation: {key}")
    if observation.get("terminated"):
        raise ValueError("Fresh diagnostic episode is already terminated")
    if profile == POOLED_PROFILE and not np.isclose(
        np.linalg.norm(data["robot0_eef_quat_xyzw"]), 1.0, atol=1e-3
    ):
        raise ValueError("Initial quaternion must be normalized")


def validate_episode_evidence(info, episode_id, profile):
    if info.get("steps") != 0:
        raise RuntimeError("Startup diagnostic executed unexpected policy actions")
    if info.get("observation_profile") != profile:
        raise ValueError("Simulator profile differs from the requested startup profile")
    if info.get("control_hz") != 20 or info.get("settling_steps") != 10:
        raise ValueError("Simulator cadence/settling differs from the profiling protocol")
    diagnostic = info.get("startup_profile", {})
    if len(json.dumps(diagnostic).encode()) > 65536:
        raise ValueError("Startup diagnostics exceed the bounded evidence size")
    for role in ("bridge_boot", "bridge_reset", "environment_boot", "environment_episode"):
        part = diagnostic.get(role, {})
        if (
            part.get("role") != role
            or not part.get("complete")
            or part.get("dropped_events") != 0
            or not part.get("process_id")
        ):
            raise ValueError("Missing or incomplete startup diagnostics")
    for role in ("bridge_reset", "environment_episode"):
        if diagnostic[role].get("episode_id") != episode_id:
            raise ValueError("Startup diagnostics belong to another episode")
    if diagnostic["bridge_boot"]["process_id"] != diagnostic["bridge_reset"]["process_id"]:
        raise ValueError("Bridge diagnostics mix process identities")
    if (
        diagnostic["environment_boot"]["process_id"]
        != diagnostic["environment_episode"]["process_id"]
    ):
        raise ValueError("Environment diagnostics mix process identities")


async def _bounded_cleanup(awaitable, timeout):
    """Let owned cleanup finish after cancellation, with a separate finite deadline."""
    task = asyncio.create_task(asyncio.wait_for(awaitable, timeout))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await task
        raise


async def _episode(client, *, episode_id, profile, timeout, cleanup_timeout, robot_connect):
    timings = StartupProfile("startup_client_episode", episode_id=episode_id)
    robot = None
    task_started = False
    row = {"episode_id": episode_id, "kind": "startup_diagnostic", "policy_actions": 0}
    error = None
    task_finished = False
    try:
        async with asyncio.timeout(timeout):
            # Mark attempted before await: cancellation can follow server-side reset.
            task_started = True
            with timings.span("task_setup"):
                started = await client.start_task(
                    "libero_spatial",
                    {
                        "task_id": 0,
                        "init_state_id": 0,
                        "seed": 0,
                        "startup_id": episode_id,
                    },
                )
            row["_setup_monotonic_ns"] = time.monotonic_ns()
            cap = client.binding("openpi/0")
            token = started["bindings"][cap.name]["token"]
            with timings.span("robot_connect_claim_first_observation"):
                robot = await robot_connect(cap, token=token)
            with timings.span("first_observation_consume"):
                observation = await robot.get_observation()
                validate_observation(observation, profile)
            row["observation_ready"] = True
            row["ready_after_start_s"] = (time.monotonic_ns() - timings.started) / 1e9
            row["_ready_monotonic_ns"] = time.monotonic_ns()
    except BaseException as exc:
        error = exc
        row["error_type"] = type(exc).__name__
    finally:
        try:
            if robot is not None:
                with timings.span("robot_close"):
                    await _bounded_cleanup(robot.close(), cleanup_timeout)
            if task_started:
                if error is None:
                    # Closing the websocket parks the slot; result() still owns its token.
                    with timings.span("diagnostic_grade"):
                        result = await _bounded_cleanup(client.grade({}), cleanup_timeout)
                    task_finished = True
                    info = result.get("info", {})
                    validate_episode_evidence(info, episode_id, profile)
                    row["environment"] = info
                    # Zero is diagnostic interruption, not a failed model evaluation.
                    row["diagnostic_score"] = result.get("score")
                else:
                    with timings.span("task_cancel"):
                        await _bounded_cleanup(client.cancel(), cleanup_timeout)
                    task_finished = True
        except BaseException as exc:
            row["cleanup_error_type"] = type(exc).__name__
            if error is None:
                error = exc
        if task_started and not task_finished:
            try:
                with timings.span("task_cancel_after_cleanup_failure"):
                    await _bounded_cleanup(client.cancel(), cleanup_timeout)
            except BaseException as exc:
                row["cancel_error_type"] = type(exc).__name__
        row["timing"] = timings.snapshot()
    if error is not None:
        if isinstance(error, asyncio.CancelledError):
            error.startup_episode = row
            raise error
        raise EpisodeFailure(row, error) from error
    return row


class EpisodeFailure(Exception):
    def __init__(self, row, cause):
        self.row, self.cause = row, cause
        super().__init__(type(cause).__name__)


async def profile_startup(
    provider,
    *,
    profile=LEGACY_PROFILE,
    trials=5,
    warm_resets=3,
    startup_timeout=900,
    warm_timeout=120,
    cleanup_timeout=30,
    overall_timeout=1800,
    connect=None,
    robot_connect=None,
    verify_cleanup=None,
    on_update=None,
):
    """Sequential fresh leases and same-task resets; no jobs or inference dependencies.

    ``verify_cleanup`` runs after provider exit, with its yielded runtime identity.
    It must return a dict containing ``verified: True`` before another lease starts.
    SDK context exit alone is not proof of remote resource termination.
    """
    import hud
    from hud import Task
    from hud.capabilities.robot import RobotClient

    if profile not in PROFILES or not 1 <= trials <= 20 or not 0 <= warm_resets <= 20:
        raise ValueError("Invalid startup profile or bounded trial count")
    if min(startup_timeout, warm_timeout, cleanup_timeout, overall_timeout) <= 0:
        raise ValueError("Timeouts must be positive")
    connect = connect or hud.connect
    robot_connect = robot_connect or RobotClient.connect
    report = {
        "schema_version": 1,
        "campaign_id": uuid4().hex,
        "kind": "startup_diagnostic",
        "profile": profile,
        "requested_trials": trials,
        "warm_resets": warm_resets,
        "policy_actions": 0,
        "inference_allocations": 0,
        "trials": [],
        "cold_definition": "fresh runtime lease; simulator process checked; host cache unknown",
        "settings": {
            "suite": "libero_spatial",
            "task_id": 0,
            "init_state_id": 0,
            "seed": 0,
            "control_hz": 20,
            "settling_steps": 10,
        },
    }
    task = Task(env=POOLED_ENV_NAME if profile == POOLED_PROFILE else ENV_NAME, id="libero_spatial")
    begun = time.monotonic()
    deadline = begun + overall_timeout
    prior_processes = set()
    for trial in range(trials):
        entry = {"trial_id": uuid4().hex, "ordinal": trial, "episodes": []}
        report["trials"].append(entry)
        timing = StartupProfile("startup_client_lease", episode_id=entry["trial_id"])
        runtime_cm, client_cm = provider(task), None
        runtime, client = None, None
        failure = None
        active_reset_ordinal = 0
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Campaign deadline reached")
            async with asyncio.timeout(min(startup_timeout, remaining)):
                with timing.span("runtime_acquire"):
                    runtime = await runtime_cm.__aenter__()
                entry["session_id"] = runtime.params.get("session_id")
                entry["container"] = getattr(runtime, "container", None)
                with timing.span("control_connect_ready"):
                    client_cm = connect(runtime, ready_timeout=startup_timeout)
                    client = await client_cm.__aenter__()
                with timing.span("initial_episode_diagnostic"):
                    episode = await _episode(
                        client,
                        episode_id=uuid4().hex,
                        profile=profile,
                        timeout=startup_timeout,
                        cleanup_timeout=cleanup_timeout,
                        robot_connect=robot_connect,
                    )
                episode["reset_ordinal"] = 0
                entry["runtime_request_to_first_observation_s"] = (
                    episode.pop("_ready_monotonic_ns") - timing.started
                ) / 1e9
                entry["runtime_request_to_task_setup_ready_s"] = (
                    episode.pop("_setup_monotonic_ns") - timing.started
                ) / 1e9
                entry["episodes"].append(episode)
                child_process = episode["environment"]["startup_profile"]["bridge_boot"][
                    "process_id"
                ]
                if child_process in prior_processes:
                    raise ValueError("Fresh lease reused a previously measured simulator process")
                prior_processes.add(child_process)
                if episode["environment"]["reset_ordinal"] != 1:
                    raise ValueError("Initial episode is not a fresh simulator reset")
                episode["validated"] = True
                entry["initial_validated"] = True
            for warm in range(warm_resets):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Campaign deadline reached")
                active_reset_ordinal = warm + 1
                episode = await _episode(
                    client,
                    episode_id=uuid4().hex,
                    profile=profile,
                    timeout=min(warm_timeout, remaining),
                    cleanup_timeout=cleanup_timeout,
                    robot_connect=robot_connect,
                )
                episode["reset_ordinal"] = warm + 1
                episode.pop("_ready_monotonic_ns", None)
                episode.pop("_setup_monotonic_ns", None)
                entry["episodes"].append(episode)
                child = episode["environment"]["startup_profile"]["bridge_boot"]["process_id"]
                if child != child_process:
                    raise ValueError("Warm reset changed simulator process")
                if episode["environment"]["reset_ordinal"] != warm + 2:
                    raise ValueError("Warm reset ordinal differs from the sequential protocol")
                episode["validated"] = True
        except EpisodeFailure as exc:
            exc.row.pop("_ready_monotonic_ns", None)
            exc.row.pop("_setup_monotonic_ns", None)
            exc.row["reset_ordinal"] = active_reset_ordinal
            entry["episodes"].append(exc.row)
            failure = exc.cause
            entry["error_type"] = type(failure).__name__
            entry["failure_phase"] = "warm_reset" if active_reset_ordinal else "initial"
        except BaseException as exc:
            failure = exc
            entry["error_type"] = type(exc).__name__
            entry["failure_phase"] = "warm_reset" if active_reset_ordinal else "initial"
            partial = getattr(exc, "startup_episode", None) or getattr(
                exc.__cause__, "startup_episode", None
            )
            if partial:
                partial.pop("_ready_monotonic_ns", None)
                partial.pop("_setup_monotonic_ns", None)
                partial["reset_ordinal"] = active_reset_ordinal
                entry["episodes"].append(partial)
        finally:
            cleanup_errors = []
            for label, cm in (
                ("control_close", client_cm if client is not None else None),
                ("runtime_close", runtime_cm if runtime is not None else None),
            ):
                if cm is not None:
                    try:
                        with timing.span(label):
                            await _bounded_cleanup(cm.__aexit__(None, None, None), cleanup_timeout)
                    except BaseException as exc:
                        cleanup_errors.append({"stage": label, "error_type": type(exc).__name__})
            if verify_cleanup is not None:
                try:
                    with timing.span("cleanup_verify"):
                        entry["cleanup"] = await _bounded_cleanup(
                            verify_cleanup(runtime), max(120, cleanup_timeout)
                        )
                except BaseException as exc:
                    cleanup_errors.append(
                        {"stage": "cleanup_verify", "error_type": type(exc).__name__}
                    )
            else:
                entry["cleanup"] = {"verified": False, "reason": "verification_unavailable"}
            entry["cleanup_errors"] = cleanup_errors
            entry["timing"] = timing.snapshot()
            if on_update:
                on_update(report)
        if isinstance(failure, (asyncio.CancelledError, KeyboardInterrupt, SystemExit)):
            raise failure
        if failure or cleanup_errors or not entry.get("cleanup", {}).get("verified"):
            report["stopped_early"] = True
            break
    report["elapsed_s"] = time.monotonic() - begun
    report["completed"] = len(report["trials"]) == trials and not report.get("stopped_early", False)
    report["summary"] = summarize(report)
    report["attempts"] = {
        "cold_leases": len(report["trials"]),
        "validated_cold_observations": sum(
            bool(row.get("initial_validated")) for row in report["trials"]
        ),
        "warm_resets": sum(
            episode.get("reset_ordinal", 0) > 0
            for row in report["trials"]
            for episode in row["episodes"]
        ),
        "validated_warm_observations": sum(
            episode.get("reset_ordinal", 0) > 0 and bool(episode.get("validated"))
            for row in report["trials"]
            for episode in row["episodes"]
        ),
        "failures": [
            {
                "trial_id": row["trial_id"],
                "phase": row["failure_phase"],
                "error_type": row["error_type"],
            }
            for row in report["trials"]
            if "error_type" in row
        ],
    }
    if on_update:
        on_update(report)
    return report


def summarize(report):
    values = {
        "initial_setup_s": [],
        "warm_setup_s": [],
        "claim_first_observation_s": [],
        "runtime_request_to_task_setup_ready_s": [],
        "runtime_request_to_first_observation_s": [],
        "runtime_acquire_s": [],
        "control_connect_ready_s": [],
        "warm_ready_s": [],
    }
    for trial in report["trials"]:
        # A later reset failure cannot invalidate an already verified cold observation.
        if trial.get("initial_validated"):
            for key in (
                "runtime_request_to_task_setup_ready_s",
                "runtime_request_to_first_observation_s",
            ):
                if key in trial:
                    values[key].append(trial[key])
        for event in trial["timing"]["events"]:
            if event["outcome"] == "ok" and event["name"] in (
                "runtime_acquire",
                "control_connect_ready",
            ):
                values[event["name"] + "_s"].append(event["duration_ms"] / 1000)
        for episode in trial["episodes"]:
            if not episode.get("validated"):
                continue
            if episode.get("reset_ordinal", 0) > 0:
                values["warm_ready_s"].append(episode["ready_after_start_s"])
            for event in episode["timing"]["events"]:
                if event["outcome"] != "ok":
                    continue
                if event["name"] == "task_setup":
                    key = "initial_setup_s" if episode.get("reset_ordinal") == 0 else "warm_setup_s"
                    values[key].append(event["duration_ms"] / 1000)
                elif event["name"] == "robot_connect_claim_first_observation":
                    values["claim_first_observation_s"].append(event["duration_ms"] / 1000)
    return {
        key: {"n": len(v), "min": min(v), "median": statistics.median(v), "max": max(v)}
        for key, v in values.items()
        if v
    }


async def verify_local_cleanup(runtime):
    container = getattr(runtime, "container", None)
    if not container:
        return {"verified": False, "reason": "container_identity_missing"}
    proc = await asyncio.create_subprocess_exec(
        "docker",
        "container",
        "inspect",
        container,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        _, stderr = await proc.communicate()
    finally:
        if proc.returncode is None:
            proc.kill()
            await proc.wait()
    absent = proc.returncode == 1 and b"No such container" in stderr
    return {"verified": absent, "reason": "container_absent" if absent else "container_unresolved"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("hud", "local-container"), default="hud")
    parser.add_argument("--image")
    parser.add_argument("--registry-id", help="Dedicated HUD profiling registry UUID")
    parser.add_argument("--build-id", help="Expected immutable HUD build UUID")
    parser.add_argument(
        "--exclusive-registry",
        action="store_true",
        help="Confirm this HUD registry is reserved for this campaign",
    )
    parser.add_argument("--profile", choices=PROFILES, default=LEGACY_PROFILE)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--warm-resets", type=int, default=3)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--warm-timeout", type=float, default=120)
    parser.add_argument("--cleanup-timeout", type=float, default=30)
    parser.add_argument("--overall-timeout", type=float, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    from hud import DockerRuntime, HUDRuntime

    if args.runtime == "local-container" and not args.image:
        parser.error("--image is required for local-container")
    if args.runtime == "hud" and not (args.registry_id and args.exclusive_registry):
        parser.error("HUD profiling requires --registry-id and --exclusive-registry for cleanup")
    provider = HUDRuntime() if args.runtime == "hud" else DockerRuntime(args.image)
    verify_cleanup = verify_local_cleanup
    if args.runtime == "hud":
        from hud.utils.platform import PlatformClient

        from .startup_cleanup import ExclusiveRegistryRuntime

        provider = ExclusiveRegistryRuntime(
            provider, PlatformClient.from_settings(), args.registry_id, build_id=args.build_id
        )
        verify_cleanup = provider.verify_cleanup

    def save(report):
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    report = asyncio.run(
        profile_startup(
            provider,
            profile=args.profile,
            trials=args.trials,
            warm_resets=args.warm_resets,
            startup_timeout=args.startup_timeout,
            warm_timeout=args.warm_timeout,
            cleanup_timeout=args.cleanup_timeout,
            overall_timeout=args.overall_timeout,
            verify_cleanup=verify_cleanup,
            on_update=save,
        )
    )
    print(json.dumps({"completed": report["completed"], "summary": report["summary"]}))
    if not report["completed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
