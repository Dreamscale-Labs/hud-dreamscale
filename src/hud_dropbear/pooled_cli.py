"""Fixed, fully reported parallel LIBERO cohorts using Dropbear's inference API."""

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import os
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

from hud import DockerRuntime, HUDRuntime, Task, Taskset
from hud.eval.job import Job
from hud.settings import settings
from hud.utils.platform import PlatformClient, canonical_record_id

from .cli import CLI_STARTED, source_revision, startup_metrics
from .contract import CONTROL_HZ, MAX_STEPS, MODEL, POOLED_ENV_NAME, POOLED_PROFILE, TASK_SUITES
from .platform import verify_platform
from .pooled import PooledProvider
from .pooled_agent import PooledRobotAgent
from .provenance import package_provenance
from .runtime_pool import RuntimePool
from .startup_cleanup import ExclusiveRegistryCampaign
from .telemetry import Evidence
from .video import export_videos

# The environment/build baseline is separate from the installed client revision.
HUD_BASE_REVISION = "0b63b4d3b9acb6d095e0886e18b2c905219e1e5a"
_HUD_INSTALLED_PROVENANCE = package_provenance("hud")
HUD_REVISION = _HUD_INSTALLED_PROVENANCE.get("source_commit")
# A mutable local checkout cannot identify installed bytes without an exact match.
if "source_dirty" in _HUD_INSTALLED_PROVENANCE and (
    _HUD_INSTALLED_PROVENANCE["source_dirty"]
    or _HUD_INSTALLED_PROVENANCE.get("installed_files_match_source") is not True
):
    HUD_REVISION = None
COHORT_VERSION = "libero-pooled-v1"


def cohort_tasks(concurrency, *, episodes_per_lane=2, max_steps=MAX_STEPS):
    """Predeclare every attempt: six fixed selections cycled across lane waves."""
    if type(concurrency) is not int or not 1 <= concurrency <= 64:
        raise ValueError("concurrency must be between 1 and 64")
    if type(episodes_per_lane) is not int or episodes_per_lane < 2:
        raise ValueError("At least two episodes per lane are required to measure reuse")
    if type(max_steps) is not int or not 1 <= max_steps <= MAX_STEPS:
        raise ValueError("max_steps must be between 1 and 600")
    rows = []
    for episode_index in range(episodes_per_lane):
        for lane_id in range(concurrency):
            case_index = episode_index * concurrency + lane_id
            task_id, init_state_id = divmod(case_index % 6, 2)
            rows.append(
                Task(
                    env=POOLED_ENV_NAME,
                    id="libero_spatial",
                    slug=f"pooled-t{task_id}-i{init_state_id}-lane{lane_id:02d}-ep{episode_index:03d}",
                    args={
                        "task_id": task_id,
                        "init_state_id": init_state_id,
                        "seed": 0,
                        "max_steps": max_steps,
                    },
                    columns={
                        "task_name": TASK_SUITES["libero_spatial"][task_id],
                        "model": MODEL,
                        "cohort_version": COHORT_VERSION,
                        "case_index": case_index,
                        "lane_id": lane_id,
                        "lane_episode_index": episode_index,
                        "noise_seed": case_index * 1000,
                        "observation_profile": POOLED_PROFILE,
                    },
                )
            )
    return rows


def cohort_manifest(rows, *, concurrency, episodes_per_lane):
    manifest = {
        "version": COHORT_VERSION,
        "concurrency": concurrency,
        "episodes_per_lane": episodes_per_lane,
        "control_hz": CONTROL_HZ,
        "chunk_size": 10,
        "profile": POOLED_PROFILE,
        "tasks": [row.model_dump(mode="json", exclude_none=True) for row in rows],
    }
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return manifest, digest


def summarize_cohort(job, rows, events):
    completed = {run.slug: run for run in job.runs} if job else {}
    starts = {row["task"]: row for row in events if row["event"] == "environment_starting"}
    results = []
    for task in rows:
        run, start = completed.get(task.slug), starts.get(task.slug, {})
        trace_id = run.trace_id if run else start.get("trace_id")
        results.append(
            {
                "task": task.slug,
                "args": task.args,
                **task.columns,
                "episode_id": start.get("episode_id"),
                "trace_id": trace_id,
                "trace_url": f"{settings.hud_web_url}/trace/{canonical_record_id(trace_id)}"
                if trace_id
                else None,
                "reward": run.reward if run else None,
                "status": run.trace.status if run else "not_completed",
                "integration_error": bool(run.trace.is_error or run.grade.is_error)
                if run
                else True,
                "grade": run.grade.raw if run else {},
            }
        )
    return {
        "job_id": job.id if job else None,
        "job_url": f"{settings.hud_web_url}/jobs/{canonical_record_id(job.id)}" if job else None,
        "runs": results,
        "successes": sum(row["reward"] == 1 and not row["integration_error"] for row in results),
        "integration_errors": sum(row["integration_error"] for row in results),
        "expected_episodes": len(rows),
        "completed_episodes": len(completed),
    }


async def run_cohort(
    agent, rows, *, runtime, job, concurrency, rollout_timeout=900, cleanup_timeout=60
):
    """Keep completed grades if another episode is interrupted, using public HUD APIs.

    HUD appends a Taskset batch's results only after its whole gather completes.
    Single-row scheduler calls append independently to the same job. One worker
    per lane advances its own episodes without waiting behind another busy lane.
    """
    if type(concurrency) is not int or not 1 <= concurrency <= 64:
        raise ValueError("concurrency must be between 1 and 64")
    if cleanup_timeout <= 0:
        raise ValueError("cleanup_timeout must be positive")
    lanes = [[] for _ in range(concurrency)]
    for row in rows:
        slot = (row.columns or {}).get("lane_id")
        if type(slot) is not int or not 0 <= slot < concurrency:
            raise ValueError("Every cohort row must select a valid lane")
        lanes[slot].append(row)

    async def drive_lane(lane_rows):
        for row in lane_rows:
            await Taskset(job.name, [row]).run(
                agent,
                runtime=runtime,
                max_concurrent=1,
                rollout_timeout=rollout_timeout,
                job=job,
            )

    pending = [asyncio.create_task(drive_lane(lane)) for lane in lanes]
    group = asyncio.gather(*pending)
    try:
        # Cancel each scheduler once. Re-cancelling while HUD is cancelling its
        # control client can interrupt that handler before it stops the driver.
        await asyncio.shield(group)
    except BaseException as error:
        for task in pending:
            if not task.done() and not task.cancelling():
                task.cancel()
        drained = asyncio.gather(*pending, return_exceptions=True)
        deadline = asyncio.get_running_loop().time() + cleanup_timeout
        while (
            not drained.done() and (remaining := deadline - asyncio.get_running_loop().time()) > 0
        ):
            try:
                # wait() leaves children alone on cancellation or timeout. A
                # wedged SDK cancellation handler cannot defer paid teardown.
                await asyncio.wait({drained}, timeout=remaining)
            except asyncio.CancelledError:
                pass
        if not drained.done():
            error.add_note("HUD scheduler cleanup exceeded its bounded deadline")
            emit = getattr(runtime, "emit", lambda event, **fields: None)
            emit(
                "cohort_cleanup_error",
                error_type="TimeoutError",
                timeout_s=cleanup_timeout,
                pending_lanes=[slot for slot, task in enumerate(pending) if not task.done()],
            )

        # The shielded group may have failed after its original waiter left.
        # Retrieve its eventual result even when its handler misses our deadline.
        def consume(future):
            try:
                future.exception()
            except asyncio.CancelledError:
                pass

        group.add_done_callback(consume)
        raise
    return job


async def evaluate(args):
    if args.runtime == "hud" and not settings.api_key:
        raise ValueError("HUD-hosted simulation requires a configured HUD API key")
    if args.runtime == "hud" and not args.registry_id:
        raise ValueError("--registry-id must identify a dedicated, otherwise idle HUD registry")
    rows = cohort_tasks(
        args.concurrency, episodes_per_lane=args.episodes_per_lane, max_steps=args.max_steps
    )
    manifest, digest = cohort_manifest(
        rows, concurrency=args.concurrency, episodes_per_lane=args.episodes_per_lane
    )
    evidence = Evidence(args.output / "timings.jsonl", started=CLI_STARTED)
    (args.output / "cohort.json").write_text(json.dumps(manifest, indent=2) + "\n")
    previous_trace_dir = settings.telemetry_local_dir
    settings.telemetry_local_dir = str((args.output / "traces").resolve())
    placement = (
        HUDRuntime()
        if args.runtime == "hud"
        else DockerRuntime(args.image, env_vars={"LIBERO_CONTROL_HZ": str(CONTROL_HZ)})
    )
    provider = PooledProvider(
        concurrency=args.concurrency,
        api_key=os.environ.get("DROPBEAR_API_KEY") or None,
        api_base=args.api_base,
        emit=evidence.emit,
        journal_path=args.output / "requests.jsonl",
        ready_timeout=args.startup_timeout,
        expected_release_id=args.expected_release_id,
        expected_release_sha256=args.expected_release_sha256,
    )
    runtimes = RuntimePool(
        placement,
        rows,
        concurrency=args.concurrency,
        emit=evidence.emit,
        startup_timeout=args.startup_timeout,
    )
    registry = (
        ExclusiveRegistryCampaign(
            PlatformClient.from_settings(),
            args.registry_id,
            environment_name=POOLED_ENV_NAME,
            expected_instances=args.concurrency,
            build_id=args.build_id,
        )
        if args.runtime == "hud"
        else None
    )
    job, failure, provider_identity = None, None, {}
    evidence.emit(
        "job_start",
        runtime=args.runtime,
        concurrency=args.concurrency,
        expected_episodes=len(rows),
        episodes_per_lane=args.episodes_per_lane,
        cohort_sha256=digest,
        hud_revision=HUD_REVISION,
        hud_base_revision=HUD_BASE_REVISION,
        source=source_revision(),
        control_hz=CONTROL_HZ,
        profile=POOLED_PROFILE,
        max_steps=args.max_steps,
        versions={
            name: importlib.metadata.version(name)
            for name in ("hud-dropbear", "hud", "dropbear", "numpy")
        },
        packages={name: package_provenance(name) for name in ("dropbear", "hud")},
    )
    try:
        job = await Job.start(f"dropbear-pooled-{args.concurrency}-{args.runtime}")
        async with asyncio.timeout(args.overall_timeout):
            async with registry if registry else nullcontext():
                async with provider, runtimes:
                    provider_identity = provider.identity
                    if registry:
                        receipt = await registry.validate_ready()
                        evidence.emit("simulator_instances_verified", **receipt)
                    agent = PooledRobotAgent(
                        provider=provider,
                        runtimes=runtimes,
                        max_steps=args.max_steps,
                        emit=evidence.emit,
                    )
                    await run_cohort(
                        agent,
                        rows,
                        runtime=runtimes,
                        concurrency=args.concurrency,
                        rollout_timeout=args.rollout_timeout,
                        job=job,
                    )
    except BaseException as exc:
        failure = exc
        evidence.emit("job_error", error_type=type(exc).__name__)
    finally:
        # Context managers release all paid resources before video/report work.
        summary = summarize_cohort(job, rows, evidence.rows)
        summary.update(
            {
                "runtime": args.runtime,
                "concurrency": args.concurrency,
                "cohort_sha256": digest,
                "episodes_per_lane": args.episodes_per_lane,
                "provider": provider_identity or provider.identity,
                "provider_cleanup_confirmed": provider.cleanup_confirmed,
                "provider_cleanup_error": provider.cleanup_error,
                "simulator_cleanup": registry.cleanup_receipt
                if registry
                else {
                    "verified": runtimes.cleanup_confirmed,
                    "reason": runtimes.cleanup_error or "local_runtime_closed",
                },
                "hud_revision": HUD_REVISION,
                "hud_base_revision": HUD_BASE_REVISION,
                "max_steps": args.max_steps,
                "control_hz": CONTROL_HZ,
                "profile": POOLED_PROFILE,
                "timings": startup_metrics(evidence.rows),
                "timing_sidecar": "timings.jsonl",
                "provenance": evidence.rows[0],
                "local_traces": "traces/",
                "error_type": type(failure).__name__ if failure else None,
            }
        )
        try:
            summary["videos"] = export_videos(args.output / "traces", args.output / "videos")
        except Exception as exc:
            summary["videos"] = []
            summary["video_error"] = type(exc).__name__
        if job and settings.api_key and settings.telemetry_enabled and not failure:
            summary["platform_evidence"] = await verify_platform(summary)
        else:
            summary["platform_evidence"] = {
                "verified": False,
                "reason": "incomplete_or_telemetry_disabled",
            }
        summary["success_rate"] = summary["successes"] / len(rows)
        summary["demo_passed"] = (
            not failure
            and summary["integration_errors"] == 0
            and summary["completed_episodes"] == len(rows)
            and summary["success_rate"] >= 0.5
            and args.max_steps == MAX_STEPS
            and args.runtime == "hud"
            and summary["platform_evidence"]["verified"]
            and summary["provider_cleanup_confirmed"]
            and summary["simulator_cleanup"]["verified"]
        )
        candidate_pass = summary["demo_passed"]
        summary["demo_passed"] = False  # Acceptance is unconfirmed until the report gate completes.
        try:
            (args.output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
            from .reporting import write_campaign_report

            report = write_campaign_report(
                args.output, concurrency=args.concurrency, minimum_success_rate=0.5
            )
            summary["demo_passed"] = candidate_pass and report["gate"]["passed"]
            evidence.emit("job_result", **summary)
        finally:
            evidence.close()
            settings.telemetry_local_dir = previous_trace_dir
            (args.output / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: summary[key]
                for key in (
                    "job_url",
                    "concurrency",
                    "expected_episodes",
                    "successes",
                    "integration_errors",
                    "success_rate",
                    "demo_passed",
                )
            },
            indent=2,
        )
    )
    if failure:
        raise failure
    return 0 if summary["demo_passed"] else 1


class _CohortParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        parsed = super().parse_args(args, namespace)
        if parsed.episodes_per_lane is None:
            parsed.episodes_per_lane = 6 if parsed.concurrency == 1 else 2
        return parsed


def parser():
    p = _CohortParser(description="Run a fixed parallel MolmoAct2-LIBERO cohort")
    p.add_argument("--runtime", choices=("hud", "local-container"), default="hud")
    p.add_argument("--concurrency", type=int, choices=range(1, 65), default=8, metavar="1..64")
    p.add_argument(
        "--episodes-per-lane",
        type=int,
        help="Default: 6 for one lane, 2 otherwise; short one-lane runs are smoke tests",
    )
    p.add_argument("--image", default="hud-dropbear-libero-pooled:local")
    p.add_argument("--registry-id", help="Dedicated idle HUD environment registry ID")
    p.add_argument("--build-id", help="Expected immutable HUD environment build ID")
    p.add_argument("--api-base", help="Dropbear inference API base (otherwise SDK configuration)")
    p.add_argument("--expected-release-id")
    p.add_argument("--expected-release-sha256")
    p.add_argument(
        "--max-steps",
        type=int,
        choices=range(1, MAX_STEPS + 1),
        default=MAX_STEPS,
        metavar="1..600",
        help="Lower limits are smoke tests, not demo acceptance",
    )
    p.add_argument("--startup-timeout", type=float, default=1000)
    p.add_argument("--rollout-timeout", type=float, default=900)
    p.add_argument("--overall-timeout", type=float, default=3600)
    p.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts") / datetime.now(UTC).strftime("pooled-%Y%m%dT%H%M%S.%fZ"),
    )
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.runtime == "hud" and not args.registry_id:
        raise SystemExit("--registry-id is required for safe HUD runtime ownership and cleanup")
    if args.episodes_per_lane < 2:
        raise SystemExit("--episodes-per-lane must be at least 2")
    if any(
        not 0 < value < float("inf")
        for value in (
            args.startup_timeout,
            args.rollout_timeout,
            args.overall_timeout,
        )
    ):
        raise SystemExit("Timeouts must be finite and positive")
    try:
        raise SystemExit(asyncio.run(evaluate(args)))
    except KeyboardInterrupt:
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
