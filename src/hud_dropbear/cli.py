"""Run the fixed LIBERO demo locally or on HUD, with one inference connection."""
# ruff: noqa: E402 -- measure startup before importing either SDK.

import argparse
import asyncio
import importlib.metadata
import json
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

CLI_STARTED = time.monotonic()

from hud import DockerRuntime, HUDRuntime, Runtime, Task, Taskset
from hud.settings import settings
from hud.utils.platform import canonical_record_id

from .agent import DropbearRobotAgent
from .contract import ENV_NAME, MAX_STEPS, TASK_NAMES
from .telemetry import CURRENT_TASK, Evidence


class MeasuredRuntime:
    def __init__(self, provider, emit):
        self.provider = provider
        self.emit = emit

    @asynccontextmanager
    async def __call__(self, task):
        token = CURRENT_TASK.set(task.slug)
        try:
            self.emit("environment_starting", task=task.slug)
            async with self.provider(task) as runtime:
                self.emit("environment_acquired", task=task.slug)
                yield runtime
        finally:
            CURRENT_TASK.reset(token)


def startup_metrics(events):
    first = [r["elapsed_s"] for r in events if r["event"] == "first_action_confirmed"]
    starts = {r["task"]: r["elapsed_s"] for r in events if r["event"] == "environment_starting"}

    def intervals(event):
        return {
            r["task"]: r["elapsed_s"] - starts[r["task"]]
            for r in events
            if r["event"] == event and r.get("task") in starts
        }

    inferences = [r for r in events if r["event"] == "inference"]
    providers = [r for r in events if r["event"] == "provider_ready"]
    return {
        "cli_entry_to_first_action_s": first[0] if first else None,
        "provider_readiness_s": providers[0]["duration_s"] if providers else None,
        "simulation_setup_s": intervals("environment_ready"),
        "episode_start_to_first_action_s": intervals("first_action_confirmed"),
        "first_inference_s": inferences[0]["duration_s"] if inferences else None,
        "inference_samples_s": [r["duration_s"] for r in inferences],
    }


def tasks(task_ids=(0, 1, 2), init_state_ids=(0, 1), *, max_steps=MAX_STEPS):
    return [
        Task(
            env=ENV_NAME,
            id="libero_spatial",
            args={"task_id": tid, "init_state_id": sid, "seed": 0, "max_steps": max_steps},
            columns={"task_name": TASK_NAMES[tid], "model": "molmoact2-libero"},
        )
        for tid in task_ids
        for sid in init_state_ids
    ]


def summarize(job, task_rows=()):
    selections = {task.slug: task.args for task in task_rows}
    runs = []
    for run in job.runs:
        runs.append(
            {
                "trace_id": run.trace_id,
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
    if args.runtime == "hud" and not settings.api_key:
        raise ValueError("HUD-hosted simulation requires a configured HUD API key")
    if args.runtime == "docker":
        runtime = DockerRuntime(args.image)
    elif args.runtime == "hud":
        runtime = HUDRuntime()
    else:
        if not args.env_url:
            raise ValueError("--env-url is required for an already-running environment")
        runtime = Runtime(args.env_url)
    rows = tasks(args.task_ids, args.init_state_ids, max_steps=args.max_steps)
    evidence = Evidence(args.output / "timings.jsonl", started=CLI_STARTED)
    summary = None
    try:
        evidence.emit(
            "job_start",
            runtime=args.runtime,
            task_count=len(rows),
            max_steps=args.max_steps,
            versions={
                n: importlib.metadata.version(n)
                for n in ("hud-dropbear", "hud", "dropbear", "numpy")
            },
        )
        async with DropbearRobotAgent(region=args.region, emit=evidence.emit) as agent:
            job = await Taskset("dropbear-libero-demo", rows).run(
                agent,
                runtime=MeasuredRuntime(runtime, evidence.emit),
                max_concurrent=1,
                rollout_timeout=900,
            )
            summary = summarize(job, rows)
            summary["provider"] = agent.identity
            summary["runtime"] = args.runtime
            summary["expected_episodes"] = len(rows)
            summary["timings"] = startup_metrics(evidence.rows)
            summary["demo_passed"] = (
                args.max_steps == MAX_STEPS
                and len(rows) == 6
                and len(job.runs) == 6
                and summary["integration_errors"] == 0
                and summary["successes"] >= 1
                and args.runtime == "hud"
            )
            evidence.emit("job_result", **summary)
    except BaseException as exc:
        evidence.emit("job_error", error_type=type(exc).__name__)
        raise
    finally:
        evidence.close()
        if summary is not None:
            (args.output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 1 if summary["integration_errors"] else 0


def parser():
    p = argparse.ArgumentParser(description="Evaluate MolmoAct2-LIBERO with HUD and Dropbear")
    p.add_argument("--runtime", choices=("docker", "hud", "attached"), default="docker")
    p.add_argument("--image", default="hud-dropbear-libero:local")
    p.add_argument("--env-url", help="HUD control URL for --runtime attached")
    p.add_argument("--region", default="ap-southeast-2", choices=("ap-southeast-2", "us-west-2"))
    p.add_argument("--task-ids", type=int, nargs="+", choices=range(3), default=[0, 1, 2])
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


def main():
    args = parser().parse_args()
    if len(set(args.task_ids)) != len(args.task_ids) or len(set(args.init_state_ids)) != len(
        args.init_state_ids
    ):
        raise SystemExit("Task and initial-state selections must not contain duplicates")
    try:
        raise SystemExit(asyncio.run(evaluate(args)))
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
