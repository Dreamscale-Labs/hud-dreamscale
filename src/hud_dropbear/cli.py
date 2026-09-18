"""Run the fixed LIBERO demo locally or on HUD, with one inference connection."""
# ruff: noqa: E402 -- measure startup before importing either SDK.

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

CLI_STARTED = time.monotonic()

from hud import DockerRuntime, HUDRuntime, Runtime, Task, Taskset
from hud.settings import settings
from hud.utils.platform import canonical_record_id

from .contract import CONTROL_HZ, ENV_NAME, MAX_STEPS, TASK_SUITES
from .platform import verify_platform
from .telemetry import CURRENT_TASK, Evidence
from .video import export_videos


def source_revision():
    """Record an editable checkout's identity; wheels need not have Git installed."""
    root = Path(__file__).resolve().parents[2]
    if not (root / ".git").exists():
        return None
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, timeout=5
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return {"revision": revision, "modified_tracked_files": bool(dirty)}


class MeasuredRuntime:
    def __init__(self, provider, emit):
        self.provider = provider
        self.emit = emit

    @asynccontextmanager
    async def __call__(self, task):
        previous = CURRENT_TASK.get()
        CURRENT_TASK.set(task.slug)
        try:
            self.emit("environment_starting", task=task.slug)
            async with self.provider(task) as runtime:
                self.emit("environment_acquired", task=task.slug)
                yield runtime
        finally:
            # HUD may exit the runtime in a shielded cleanup task after an
            # exception. ContextVar tokens cannot be reset in that copied context.
            CURRENT_TASK.set(previous)


@asynccontextmanager
async def job_runtime(provider, first_task, emit):
    # Lease outside an individual trace: HUD associates a lease created inside
    # a trace with that trace's completion, even when wrapped in Shared.
    emit("simulator_starting")
    try:
        async with provider(first_task) as runtime:
            emit("simulator_acquired", runtime_session_id=runtime.params.get("session_id"))
            yield runtime
    finally:
        emit("simulator_closed")


def startup_metrics(events):
    first = [r["elapsed_s"] for r in events if r["event"] == "first_action_confirmed"]
    # Read chronologically. A last-start-by-task dict silently paired an earlier
    # episode with a later reset when HUD repeated a task slug.
    starts, active, occurrences = {}, {}, {}
    intervals = {"environment_ready": {}, "first_action_confirmed": {}}
    inferences = [r for r in events if r["event"] == "inference"]
    providers = [r for r in events if r["event"] == "provider_ready"]
    first_by_task = {}
    steady = []
    for row in events:
        task = row.get("task")
        if row["event"] == "environment_starting":
            occurrences[task] = occurrences.get(task, 0) + 1
            key = row.get("episode_id") or (
                task if occurrences[task] == 1 else f"{task}#{occurrences[task]}"
            )
            active[task] = key
            starts[key] = row["elapsed_s"]
        key = row.get("episode_id") or active.get(task, task)
        if row["event"] in intervals and key in starts:
            intervals[row["event"]][key] = row["elapsed_s"] - starts[key]
        if row["event"] == "inference":
            if key in first_by_task:
                steady.append(row["duration_s"])
            else:
                first_by_task[key] = row["duration_s"]
    return {
        "cli_entry_to_first_action_s": first[0] if first else None,
        "provider_readiness_s": providers[0]["duration_s"] if providers else None,
        "simulation_setup_s": intervals["environment_ready"],
        "episode_start_to_first_action_s": intervals["first_action_confirmed"],
        "first_inference_s": inferences[0]["duration_s"] if inferences else None,
        "episode_first_inference_s": first_by_task,
        "steady_state_inference_samples_s": steady,
        "inference_samples_s": [r["duration_s"] for r in inferences],
    }


def tasks(
    task_ids=(0, 1, 2), init_state_ids=(0, 1), *, max_steps=MAX_STEPS, suite="libero_spatial"
):
    return [
        Task(
            env=ENV_NAME,
            id=suite,
            args={"task_id": tid, "init_state_id": sid, "seed": 0, "max_steps": max_steps},
            columns={"task_name": TASK_SUITES[suite][tid], "model": "molmoact2-libero"},
        )
        for tid in task_ids
        for sid in init_state_ids
    ]


def selected_tasks(args):
    if args.taskset is None:
        suites = list(TASK_SUITES) if args.all_suites else [args.suite]
        return [
            row
            for suite in suites
            for row in tasks(
                args.task_ids, args.init_state_ids, max_steps=args.max_steps, suite=suite
            )
        ]
    if args.taskset.suffix not in {".json", ".jsonl"}:
        raise ValueError("--taskset requires HUD JSON or JSONL data")
    rows = list(Taskset.from_file(args.taskset))
    if not rows:
        raise ValueError("The taskset is empty")
    for row in rows:
        if row.env != ENV_NAME or row.id not in TASK_SUITES:
            raise ValueError("Every row must select a pinned dropbear-libero suite")
        if row.runtime_config is not None or row.verifier is not None:
            raise ValueError("Taskset placement and grading must use the demo environment")
        if set(row.args) != {"task_id", "init_state_id", "seed", "max_steps"}:
            raise ValueError("Each row requires task_id, init_state_id, seed and max_steps")
        if any(type(value) is not int for value in row.args.values()):
            raise ValueError("Task selection values must be integers")
        tid = row.args["task_id"]
        if not 0 <= tid < len(TASK_SUITES[row.id]) or row.args["init_state_id"] not in (0, 1):
            raise ValueError("Task or initial state is outside the pinned demo selection")
        if row.args["seed"] != 0 or row.args["max_steps"] != args.max_steps:
            raise ValueError("Taskset seed and action limit must match the demo contract")
        if row.columns.get("task_name") != TASK_SUITES[row.id][tid]:
            raise ValueError("Taskset task_name does not match the pinned numeric ID")
    return rows


def summarize(job, task_rows=()):
    selections = {task.slug: task.args for task in task_rows}
    runs = []
    for run in job.runs:
        runs.append(
            {
                "trace_id": run.trace_id,
                "trace_url": f"{settings.hud_web_url}/trace/{canonical_record_id(run.trace_id)}",
                "task": run.slug,
                "args": selections.get(run.slug, {}),
                "reward": run.reward,
                "status": run.trace.status,
                "integration_error": run.trace.is_error or run.grade.is_error,
                "grade": run.grade.raw,
            }
        )
    return {
        "job_id": job.id,
        "job_url": f"{settings.hud_web_url}/jobs/{canonical_record_id(job.id)}",
        "runs": runs,
        "successes": sum(r["reward"] > 0 for r in runs),
        "integration_errors": sum(r["integration_error"] for r in runs),
    }


async def evaluate(args):
    from .agent import DropbearRobotAgent

    if args.runtime == "hud" and not settings.api_key:
        raise ValueError("HUD-hosted simulation requires a configured HUD API key")
    if args.runtime == "docker":
        runtime = DockerRuntime(args.image, env_vars={"LIBERO_CONTROL_HZ": str(args.control_hz)})
    elif args.runtime == "hud":
        runtime = HUDRuntime()
    else:
        if not args.env_url:
            raise ValueError("--env-url is required for an already-running environment")
        runtime = Runtime(args.env_url)
    rows = selected_tasks(args)
    suites = list(dict.fromkeys(row.id for row in rows))
    evidence = Evidence(args.output / "timings.jsonl", started=CLI_STARTED)
    previous_trace_dir = settings.telemetry_local_dir
    settings.telemetry_local_dir = str((args.output / "traces").resolve())
    summary = None
    try:
        evidence.emit(
            "job_start",
            runtime=args.runtime,
            suites=suites,
            source=source_revision(),
            task_count=len(rows),
            taskset_sha256=hashlib.sha256(args.taskset.read_bytes()).hexdigest()
            if args.taskset
            else None,
            max_steps=args.max_steps,
            image=args.image if args.runtime == "docker" else None,
            hud_revision="0b63b4d3b9acb6d095e0886e18b2c905219e1e5a",
            simulator={
                "hf_libero": "0.1.3",
                "assets_revision": "0b3ea86be5fe169d0fd036ae63d1070ec09e90f6",
                "mujoco": "3.3.7",
                "control_hz": args.control_hz,
                "settling_steps": 10,
                "chunk_size": 10,
            },
            versions={
                n: importlib.metadata.version(n)
                for n in ("hud-dropbear", "hud", "dropbear", "numpy")
            },
        )
        async with (
            DropbearRobotAgent(
                region=args.region,
                control_hz=args.control_hz,
                emit=evidence.emit,
                connect_in_background=True,
            ) as agent,
            job_runtime(runtime, rows[0], evidence.emit) as shared_runtime,
        ):
            job = await Taskset(
                f"dropbear-{'all-suites' if len(suites) == 5 else suites[0]}-{args.runtime}", rows
            ).run(
                agent,
                runtime=MeasuredRuntime(shared_runtime, evidence.emit),
                max_concurrent=1,
                rollout_timeout=900,
            )
            summary = summarize(job, rows)
            summary["provider"] = agent.identity
            summary["runtime"] = args.runtime
            summary["expected_episodes"] = len(rows)
            summary["timings"] = startup_metrics(evidence.rows)
            summary["provenance"] = evidence.rows[0]
            summary["suite_results"] = {
                suite: {
                    "episodes": sum(
                        r["grade"].get("info", {}).get("suite") == suite for r in summary["runs"]
                    ),
                    "successes": sum(
                        r["grade"].get("info", {}).get("suite") == suite
                        and r["grade"].get("success") is True
                        for r in summary["runs"]
                    ),
                }
                for suite in suites
            }
            summary["timing_sidecar"] = "timings.jsonl"
            summary["demo_passed"] = (
                args.max_steps == MAX_STEPS
                and len(rows) == 6
                and len(job.runs) == 6
                and summary["integration_errors"] == 0
                and summary["successes"] >= 1
                and args.runtime == "hud"
                and args.control_hz == CONTROL_HZ
                and args.suite == "libero_spatial"
                and sorted(args.task_ids) == [0, 1, 2]
                and sorted(args.init_state_ids) == [0, 1]
            )
            if set(suites) == set(TASK_SUITES):
                summary["demo_passed"] = (
                    args.runtime == "hud"
                    and args.control_hz == CONTROL_HZ
                    and args.max_steps == MAX_STEPS
                    and len(job.runs) == len(rows)
                    and summary["integration_errors"] == 0
                    and all(row["successes"] >= 1 for row in summary["suite_results"].values())
                )
        # Release billable inference before checking best-effort platform uploads.
        summary["videos"] = export_videos(args.output / "traces", args.output / "videos")
        if settings.api_key and settings.telemetry_enabled:
            summary["platform_evidence"] = await verify_platform(summary)
        else:
            summary["platform_evidence"] = {"verified": False, "reason": "telemetry_disabled"}
        summary["demo_passed"] = summary["demo_passed"] and summary["platform_evidence"]["verified"]
        summary["local_traces"] = "traces/"
        evidence.emit("job_result", **summary)
    except BaseException as exc:
        evidence.emit("job_error", error_type=type(exc).__name__)
        raise
    finally:
        evidence.close()
        settings.telemetry_local_dir = previous_trace_dir
        if summary is not None:
            (args.output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 1 if summary["integration_errors"] else 0


def parser():
    p = argparse.ArgumentParser(description="Evaluate MolmoAct2-LIBERO with HUD and Dropbear")
    p.add_argument("--runtime", choices=("docker", "hud", "attached"), default="docker")
    p.add_argument("--image", default="hud-dropbear-libero:local")
    p.add_argument("--env-url", help="HUD control URL for --runtime attached")
    p.add_argument("--suite", choices=TASK_SUITES, default="libero_spatial")
    p.add_argument("--all-suites", action="store_true", help="Evaluate all five LIBERO suites")
    p.add_argument(
        "--taskset", type=Path, help="Pinned HUD JSON/JSONL rows; replaces numeric selections"
    )
    p.add_argument(
        "--control-hz",
        type=int,
        choices=(10, 20),
        default=CONTROL_HZ,
        help="Actual simulator control rate; attached environments must use the same rate",
    )
    p.add_argument("--region", default="ap-southeast-2", choices=("ap-southeast-2", "us-west-2"))
    p.add_argument("--task-ids", type=int, nargs="+", choices=range(90), default=[0, 1, 2])
    p.add_argument("--init-state-ids", type=int, nargs="+", choices=(0, 1), default=[0, 1])
    p.add_argument(
        "--max-steps",
        type=int,
        choices=range(1, MAX_STEPS + 1),
        default=MAX_STEPS,
        metavar="1..600",
        help="Lower values are smoke tests only",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts") / datetime.now(UTC).strftime("%Y%m%dT%H%M%S.%fZ"),
    )
    return p


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in {"pooled", "profile-startup"}:
        if argv[0] == "pooled":
            from .pooled_cli import main as command
        else:
            from .profile_startup import main as command
        return command(argv[1:])
    args = parser().parse_args(argv)
    suites = list(TASK_SUITES) if args.all_suites else [args.suite]
    if args.taskset is None and any(
        tid >= len(TASK_SUITES[suite]) for suite in suites for tid in args.task_ids
    ):
        raise SystemExit("Task ID is outside the supported suite's pinned task manifest")
    if args.taskset is None and (
        len(set(args.task_ids)) != len(args.task_ids)
        or len(set(args.init_state_ids)) != len(args.init_state_ids)
    ):
        raise SystemExit("Task and initial-state selections must not contain duplicates")
    try:
        raise SystemExit(asyncio.run(evaluate(args)))
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
