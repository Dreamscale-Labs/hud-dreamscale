"""Run RoboLab DROID episodes on HUD with Dreamscale cloud policies, as one wave.

Each episode is its own Modal L40S sandbox (HUD ``ModalRuntime``) running the
RoboLab sim; the agent runs here and calls Dreamscale through the public SDK.
The wave is one HUD job (``Taskset.run`` with ``max_concurrent``). Afterwards
``<output>/summary.json`` joins every HUD grade with the agent's timing and the
sandbox lifetimes.

    uv run --project environments/robolab python environments/robolab/run_waves.py \\
        --model cosmos3-nano-policy-droid --tasks all --episodes-per-task 5 \\
        --concurrency 10 --env prod --output runs/cosmos3-prod-wave1

``--model hold`` runs the same harness with no inference (a hold-pose smoke).
Credentials: ``HUD_API_KEY`` (or ``hud set``), a Modal token (``modal token set``
or ``MODAL_TOKEN_ID``/``MODAL_TOKEN_SECRET``), and for real models a Dreamscale
key: ``DREAMSCALE_API_KEY``/``DROPBEAR_API_KEY`` in the environment, or
``dreamscale login`` state whose control plane matches ``--env``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import AsyncIterator, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from dreamscale_agent import (  # noqa: E402
    HOLD_MODEL,
    MODELS,
    DreamscalePolicy,
    HoldPolicy,
    RobolabDreamscaleAgent,
)
from robolab_tasks import horizon_steps, select_tasks, task_set, task_spec  # noqa: E402
from summary import aggregate, episode_record, job_url, read_inference_rows  # noqa: E402

ENV_NAME = "dreamscale-robolab"
#: Image built from Dockerfile.hud (RoboLab ad45d4f) with ``build_image.py``.
#: Override with --image, or pass --build-image to build Dockerfile.hud now.
IMAGE = os.environ.get("ROBOLAB_MODAL_IMAGE", "modal://im-p0jN9zrruXkRAg7r9Uql1g")
CONTROL_PLANES = {
    "prod": "https://api.dreamscalelabs.com",
    "dev": "https://api-dev.dreamscalelabs.com",
}
# Must exceed runtime.json's startup (1800 s) and run (3600 s) limits.
ROLLOUT_TIMEOUT_S = 7200
SERVE_COMMAND = [
    # Matches Dockerfile.hud. Modal replaces the image CMD, so name it here.
    # --no-sync: the image already ran `uv sync`.
    "uv", "run", "--no-sync", "hud", "serve", "env.py", "--host", "0.0.0.0", "--port", "8765",
]  # fmt: skip


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--model", required=True, choices=[*MODELS, HOLD_MODEL])
    parser.add_argument("--tasks", default="all", help="'all' or comma-separated task names")
    parser.add_argument("--episodes-per-task", type=int, default=1)
    parser.add_argument("--episode-offset", type=int, default=0, help="first episode index")
    parser.add_argument("--concurrency", type=int, default=1, help="max concurrent sandboxes")
    parser.add_argument("--env", choices=sorted(CONTROL_PLANES), default="prod")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-warm", type=int, default=300, help="SDK keep_warm seconds")
    parser.add_argument("--startup-timeout", type=float, default=900.0)
    parser.add_argument("--instruction-type", default="default")
    parser.add_argument("--scene-seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=None, help="cap steps (smoke only)")
    parser.add_argument("--no-local-video", action="store_true")
    parser.add_argument(
        "--image", default=IMAGE, help="modal://im-... (default: $ROBOLAB_MODAL_IMAGE)"
    )
    parser.add_argument("--build-image", action="store_true", help="build Dockerfile.hud on Modal")
    parser.add_argument(
        "--sandbox-cloud", default="oci",
        help="Modal cloud for simulator sandboxes ('' to let Modal choose); see pin_sandbox_cloud",
    )
    parser.add_argument("--dreamscale-home", type=Path, help="use this DREAMSCALE_HOME as-is")
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    args = parser.parse_args(argv)
    if args.episodes_per_task < 1 or args.concurrency < 1 or args.episode_offset < 0:
        parser.error("--episodes-per-task and --concurrency must be >= 1; offset >= 0")
    if not args.image and not args.build_image and not args.dry_run:
        parser.error("pass --image modal://im-... (or set ROBOLAB_MODAL_IMAGE) or --build-image")
    try:
        args.task_names = select_tasks(args.tasks)
    except ValueError as error:
        parser.error(str(error))
    return args


# ── Dreamscale credentials ───────────────────────────────────────────────────


@contextlib.contextmanager
def dreamscale_credentials(env: str, explicit_home: Path | None = None) -> Iterator[dict[str, Any]]:
    """Point ``dreamscale.aconnect`` at the chosen control plane without printing the key.

    ``connect`` reads only ``$DREAMSCALE_HOME/config.toml``. A key in
    ``DREAMSCALE_API_KEY``/``DROPBEAR_API_KEY`` is written to a private
    temporary home (0600, removed afterwards); otherwise the existing login is
    used when its control plane matches ``env``.
    """
    from dreamscale.config import Config, load_config, save_config

    url = CONTROL_PLANES[env]
    previous = os.environ.get("DREAMSCALE_HOME")
    if explicit_home is not None:
        os.environ["DREAMSCALE_HOME"] = str(explicit_home)
        try:
            config = load_config()
            if not config.api_key:
                raise SystemExit(f"{explicit_home}/config.toml has no API key")
            yield {"source": "explicit_home", "control_plane_url": config.control_plane_url}
        finally:
            _restore_env("DREAMSCALE_HOME", previous)
        return
    key = os.environ.get("DREAMSCALE_API_KEY") or os.environ.get("DROPBEAR_API_KEY")
    if key:
        home = tempfile.mkdtemp(prefix="robolab-dreamscale-")
        os.environ["DREAMSCALE_HOME"] = home
        try:
            save_config(Config(api_key=key, control_plane_url=url, preferred_region="us-west-2"))
            yield {"source": "environment_key", "control_plane_url": url}
        finally:
            _restore_env("DREAMSCALE_HOME", previous)
            shutil.rmtree(home, ignore_errors=True)
        return
    config = load_config()
    if config.api_key and config.control_plane_url.rstrip("/") == url:
        yield {"source": "dreamscale_login", "control_plane_url": url}
        return
    raise SystemExit(
        f"No Dreamscale key for {env} ({url}). Export DROPBEAR_API_KEY, e.g. "
        f"`set -a; . ~/.config/dreamscale/{'test' if env == 'prod' else 'dev'}.env; set +a`, "
        "or `dreamscale login` against that control plane."
    )


def _restore_env(name: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(name, None)
    else:
        os.environ[name] = value


# ── Modal placement with timing ──────────────────────────────────────────────


def pin_sandbox_cloud(cloud: str | None) -> None:
    """Place every simulator sandbox on one Modal cloud.

    Isaac Sim 5.0 needs the 580 NVIDIA driver. On 2026-10-09 Modal's AWS L40S hosts
    ran 610.57.04 and crashed Kit at RTX startup, while OCI L40S hosts ran 580.95.05.
    ``ModalRuntime`` has no placement option, so wrap ``Sandbox.create.aio``.
    """
    if not cloud:
        return
    import modal

    original = modal.Sandbox.create

    class _PinnedCreate:
        def __call__(self, *args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("cloud", cloud)
            return original(*args, **kwargs)

        async def aio(self, *args: Any, **kwargs: Any) -> Any:
            kwargs.setdefault("cloud", cloud)
            return await original.aio(*args, **kwargs)

    modal.Sandbox.create = _PinnedCreate()


def make_runtime(image: str, *, build: bool):
    """``ModalRuntime`` that also records each sandbox's request/ready/release times."""
    from hud.eval import ModalRuntime

    class TimedModalRuntime(ModalRuntime):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.sandboxes: dict[str, dict[str, Any]] = {}

        @contextlib.asynccontextmanager
        async def __call__(self, task: Any) -> AsyncIterator[Any]:
            entry: dict[str, Any] = {"requested_unix": time.time()}
            self.sandboxes[task.slug] = entry
            try:
                async with super().__call__(task) as endpoint:
                    entry["ready_unix"] = time.time()
                    entry["instance_id"] = (getattr(endpoint, "params", None) or {}).get(
                        "instance_id"
                    )
                    yield endpoint
            finally:
                entry["released_unix"] = time.time()

    config = json.loads((HERE / "runtime.json").read_text())
    modal_image = None
    if build:
        import modal

        modal_image = modal.Image.from_dockerfile(HERE / "Dockerfile.hud", context_dir=HERE)
    else:
        config["image"] = image
    return TimedModalRuntime(
        image=modal_image,
        command=SERVE_COMMAND,
        port=8765,
        app_name="hud-dreamscale-robolab",
        runtime_config=config,
    )


async def ensure_sandboxes_stopped(
    sandboxes: dict[str, dict[str, Any]], *, grace_s: float = 60.0
) -> list[str]:
    """Belt and braces after ModalRuntime's own teardown: terminate any survivor.

    ``terminate`` is asynchronous on Modal's side, so give each sandbox a grace
    period to report an exit code before terminating it again.
    """
    import modal

    survivors = []
    for entry in sandboxes.values():
        sandbox_id = entry.get("instance_id")
        if not sandbox_id:
            continue
        try:
            sandbox = await modal.Sandbox.from_id.aio(sandbox_id)
            deadline = time.monotonic() + grace_s
            code = await sandbox.poll.aio()
            while code is None and time.monotonic() < deadline:
                await asyncio.sleep(2.0)
                code = await sandbox.poll.aio()
            entry["exit_code"] = code
            if code is None:
                survivors.append(sandbox_id)
                await sandbox.terminate.aio()
                entry["terminated_by_runner"] = True
        except Exception as error:  # report, never mask the wave result
            entry["stop_check_error"] = repr(error)
    return survivors


# ── the wave ─────────────────────────────────────────────────────────────────


def build_tasks(args: argparse.Namespace, model: str) -> list[Any]:
    from hud.eval import Task

    tasks = []
    for task_name in args.task_names:
        spec = task_spec(task_name)
        for episode in range(args.episode_offset, args.episode_offset + args.episodes_per_task):
            tasks.append(
                Task(
                    env=ENV_NAME,
                    id="episode",
                    args={
                        "task_name": task_name,
                        "episode": episode,
                        "instruction_type": args.instruction_type,
                        "scene_seed": args.scene_seed,
                    },
                    columns={
                        "task_name": task_name,
                        "episode": episode,
                        "model": model,
                        "difficulty": spec.get("difficulty_label"),
                    },
                )
            )
    return tasks


def run_view(run: Any) -> dict[str, Any]:
    grade = getattr(run, "grade", None)
    trace = getattr(run, "trace", None)
    return {
        "trace_id": getattr(run, "trace_id", None),
        "reward": getattr(run, "reward", None),
        "grade_info": dict(getattr(grade, "info", None) or {}),
        "grade_is_error": bool(getattr(grade, "is_error", False)),
        "trace_status": getattr(trace, "status", None),
        "error": getattr(trace, "error", None) if getattr(trace, "is_error", False) else None,
    }


def _trace_key(trace_id: str) -> str:
    return trace_id.replace("-", "").lower()


def load_agent_summaries(output: Path) -> dict[str, dict[str, Any]]:
    found = {}
    for path in sorted((output / "episodes").glob("*/episode.json")):
        data = json.loads(path.read_text())
        if data.get("trace_id"):
            found[_trace_key(data["trace_id"])] = data
    return found


async def run_wave(args: argparse.Namespace) -> dict[str, Any]:
    from hud.eval import Taskset
    from hud.eval.job import Job

    output: Path = args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists():
        raise SystemExit(f"{output}/summary.json exists; choose a new --output")
    model = args.model
    tasks = build_tasks(args, model)
    policy = (
        HoldPolicy()
        if model == HOLD_MODEL
        else DreamscalePolicy(model, keep_warm=args.keep_warm, startup_timeout=args.startup_timeout)
    )
    agent = RobolabDreamscaleAgent(
        policy,
        output_dir=output,
        local_video=not args.no_local_video,
        max_steps_cap=args.max_steps,
    )
    pin_sandbox_cloud(args.sandbox_cloud)
    runtime = make_runtime(args.image, build=args.build_image)
    name = args.job_name or (
        f"RoboLab DROID {model} [{args.env}] {len(args.task_names)} tasks x "
        f"{args.episodes_per_task} ep, c={args.concurrency}"
    )
    meta: dict[str, Any] = {
        "model": model,
        "env": args.env,
        "tasks": args.task_names,
        "horizon_steps": {t: horizon_steps(t) for t in args.task_names},
        "episodes_per_task": args.episodes_per_task,
        "episode_offset": args.episode_offset,
        "concurrency": args.concurrency,
        "instruction_type": args.instruction_type,
        "scene_seed": args.scene_seed,
        "max_steps_cap": args.max_steps,
        "image": "built:Dockerfile.hud" if args.build_image else args.image,
        "sandbox_cloud": args.sandbox_cloud or None,
        "robolab_revision": task_set()["robolab_revision"],
        "keep_warm_s": None if model == HOLD_MODEL else args.keep_warm,
        "sdk_connect": None if model == HOLD_MODEL else policy.connect_kwargs(),
    }
    job = await Job.start(name)
    meta["job"] = {"id": job.id, "name": name, "url": job_url(job.id)}
    (output / "wave.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"[wave] {len(tasks)} episodes, concurrency {args.concurrency}", flush=True)
    print(f"[wave] job {meta['job']['url']}", flush=True)

    started_unix = time.time()
    started = time.monotonic()
    try:
        job = await Taskset(name, tasks).run(
            agent,
            runtime=runtime,
            max_concurrent=args.concurrency,
            job=job,
            rollout_timeout=ROLLOUT_TIMEOUT_S,
        )
    finally:
        survivors = await ensure_sandboxes_stopped(runtime.sandboxes)
    wall = time.monotonic() - started

    by_slug = {task.slug: task for task in tasks}
    agents = load_agent_summaries(output)
    records = []
    rows: list[dict[str, Any]] = []
    for run in job.runs:
        task = by_slug.get(run.slug)
        if task is None:
            print(f"[wave] run with unknown slug {run.slug!r} skipped", flush=True)
            continue
        view = run_view(run)
        agent_summary = agents.get(_trace_key(view["trace_id"] or ""))
        records.append(
            episode_record(
                task_name=task.args["task_name"],
                episode=task.args["episode"],
                run=view,
                agent=agent_summary,
                sandbox=runtime.sandboxes.get(task.slug),
            )
        )
        if agent_summary and agent_summary.get("timing_file"):
            rows.extend(read_inference_rows(output / agent_summary["timing_file"]))
    records.sort(key=lambda r: (args.task_names.index(r["task_name"]), r["episode"]))
    meta.update(
        {
            "started_at": datetime.fromtimestamp(started_unix, UTC).isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "wall_clock_s": wall,
            "sandboxes_terminated_by_runner": survivors,
        }
    )
    summary = aggregate(records, inference_rows=rows, meta=meta)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    overall = summary["overall"]
    print(
        f"[wave] {overall['successes']}/{overall['graded']} successes "
        f"({overall['errors']} errors), wall {summary['wall_clock_s']:.0f}s",
        flush=True,
    )
    for task, stats in summary["per_task"].items():
        print(f"[wave]   {task}: {stats['successes']}/{stats['graded']}", flush=True)
    rtt = summary["latency_ms"]["sdk_rtt"]
    if rtt.get("n"):
        print(f"[wave] sdk rtt p50 {rtt['p50']:.0f} ms, p95 {rtt['p95']:.0f} ms", flush=True)
    print(f"[wave] job {summary['job']['url']}", flush=True)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.dry_run:
        for task in build_tasks(args, args.model):
            print(json.dumps(task.args))
        return
    import modal

    with contextlib.ExitStack() as stack:
        if args.model != HOLD_MODEL:
            creds = stack.enter_context(dreamscale_credentials(args.env, args.dreamscale_home))
            print(f"[wave] dreamscale {creds['control_plane_url']} ({creds['source']})", flush=True)
        stack.enter_context(modal.enable_output())
        summary = asyncio.run(run_wave(args))
    print_summary(summary)
    print(f"[wave] wrote {args.output / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
