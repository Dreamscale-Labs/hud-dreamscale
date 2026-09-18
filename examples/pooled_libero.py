"""Eight independent HUD environments sharing Dropbear HTTP inference.

Configure HUD_API_KEY and the Dropbear SDK credentials, then run this file with
--registry-id selecting a dedicated, otherwise idle HUD environment registry.
For a saved cohort manifest, timing report and videos, use:
    hud-dropbear pooled --runtime hud --concurrency 8 --registry-id YOUR_REGISTRY_ID
"""

import argparse
import asyncio
import os

from hud import HUDRuntime
from hud.eval.job import Job
from hud.utils.platform import PlatformClient

from hud_dropbear import PooledProvider, PooledRobotAgent, RuntimePool
from hud_dropbear.pooled_cli import cohort_tasks, run_cohort
from hud_dropbear.startup_cleanup import ExclusiveRegistryCampaign


async def main(registry_id, *, api_base=None):
    rows = cohort_tasks(8)  # Two distinct episodes on each reusable scalar runtime.
    async with (
        ExclusiveRegistryCampaign(
            PlatformClient.from_settings(),
            registry_id,
            environment_name=rows[0].env,
            expected_instances=8,
        ) as registry,
        PooledProvider(
            concurrency=8,
            api_key=os.environ.get("DROPBEAR_API_KEY") or None,
            api_base=api_base,
        ) as provider,
        RuntimePool(HUDRuntime(), rows, concurrency=8) as runtimes,
    ):
        await registry.validate_ready()
        agent = PooledRobotAgent(provider=provider, runtimes=runtimes)
        job = await Job.start("dropbear-libero-parallel")
        await run_cohort(
            agent,
            rows,
            runtime=runtimes,
            job=job,
            concurrency=8,
            rollout_timeout=900,
        )
        # HUD's job.reward excludes some infrastructure failures. Keep the fixed
        # denominator for this demo and inspect every run, including failures.
        successes = sum(
            run.reward == 1 and not run.trace.is_error and not run.grade.is_error
            for run in job.runs
        )
        print(f"{successes}/{len(rows)} successful episodes; HUD job {job.id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-id", required=True)
    parser.add_argument("--api-base", help="Dropbear management origin; defaults to saved config")
    args = parser.parse_args()
    asyncio.run(main(args.registry_id, api_base=args.api_base))
