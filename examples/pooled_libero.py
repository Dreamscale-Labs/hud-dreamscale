"""A small pooled evaluation example; simulator placement remains the caller's choice.

Requires a funded Dreamscale pooled grant and a deployed raw360 HUD environment.
The existing scalar environment/CLI is a different path and is unchanged.
"""

import argparse
import asyncio

from hud import HUDRuntime
from hud.eval.job import Job

from hud_dreamscale import PooledProvider, PooledRobotAgent, RuntimePool
from hud_dreamscale.cohort import cohort_tasks, run_cohort, summarize_cohort


async def evaluate(runtime, *, release_id, release_sha256, concurrency=8, api_base=None):
    rows = cohort_tasks(concurrency)
    async with (
        PooledProvider(
            concurrency=concurrency,
            api_base=api_base,
            expected_release_id=release_id,
            expected_release_sha256=release_sha256,
        ) as provider,
        RuntimePool(runtime, rows, concurrency=concurrency) as runtimes,
    ):
        agent = PooledRobotAgent(provider=provider, runtimes=runtimes)
        job = await Job.start("dreamscale-libero-pooled")
        await run_cohort(agent, rows, runtime=runtimes, job=job, concurrency=concurrency)
    return summarize_cohort(job, rows, [])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-id", required=True)
    parser.add_argument("--release-sha256", required=True)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--api-base")
    args = parser.parse_args()
    result = asyncio.run(evaluate(HUDRuntime(), **vars(args)))
    print(f"{result['successes']}/{result['expected_episodes']} successful episodes")
    print(f"Integration errors: {result['integration_errors']}; HUD job {result['job_id']}")
