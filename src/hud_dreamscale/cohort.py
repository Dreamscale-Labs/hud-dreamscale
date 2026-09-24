"""Small public-HUD cohort helpers; no deployment, cost or profiling CLI."""

import asyncio
import hashlib
import json

from hud import Task, Taskset
from hud.settings import settings
from hud.utils.platform import canonical_record_id

from .contract import CONTROL_HZ, MAX_STEPS, MODEL, POOLED_ENV_NAME, POOLED_PROFILE, TASK_SUITES

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
