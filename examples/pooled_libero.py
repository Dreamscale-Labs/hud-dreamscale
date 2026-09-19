"""Eight independent HUD environments sharing Dropbear HTTP inference.

Configure HUD_API_KEY and the Dropbear SDK credentials, then run this file with
--registry-id selecting a dedicated, otherwise idle HUD environment registry,
--build-id, --expected-release-id and --expected-release-sha256.
An operator must first provision the funded service and matching account grant.
This example owns one cohort and stops its inference deployment on exit.
With an ongoing grant, normal stop releases its GPU workers and retains the
service apps for another deployment; finite grants still terminate the GPU app
and require operator reprovisioning. Ongoing service reuse and renewal are under
live development qualification; earlier finite results do not establish them.
Episodes reuse the open provider and remain billable until deployment cleanup.
The operator separately owns service app funding and final retirement, including
the CPU gateway/sweeper app. Use the approved SDK source checkout while the
published package lacks the pooled API; see docs/pooled-inference.md.
For a saved cohort manifest, timing report and videos, use:
    hud-dropbear pooled --runtime hud --concurrency 8 --registry-id YOUR_REGISTRY_ID
        --build-id YOUR_BUILD_ID --expected-release-id YOUR_RELEASE_ID
        --expected-release-sha256 YOUR_RELEASE_MANIFEST_SHA256
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


async def main(
    registry_id, *, build_id, expected_release_id, expected_release_sha256, api_base=None
):
    rows = cohort_tasks(8)  # Two distinct episodes on each reusable scalar runtime.
    async with (
        ExclusiveRegistryCampaign(
            PlatformClient.from_settings(),
            registry_id,
            environment_name=rows[0].env,
            expected_instances=8,
            build_id=build_id,
        ) as registry,
        PooledProvider(
            concurrency=8,
            api_key=os.environ.get("DROPBEAR_API_KEY") or None,
            api_base=api_base,
            expected_release_id=expected_release_id,
            expected_release_sha256=expected_release_sha256,
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
        run.reward == 1 and not run.trace.is_error and not run.grade.is_error for run in job.runs
    )
    print(f"{successes}/{len(rows)} successful episodes; HUD job {job.id}")
    # An acceptance claim additionally requires platform/video verification.


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry-id", required=True)
    parser.add_argument("--build-id", required=True, help="Expected immutable HUD build ID")
    parser.add_argument("--expected-release-id", required=True)
    parser.add_argument("--expected-release-sha256", required=True)
    parser.add_argument("--api-base", help="Dropbear management origin; defaults to saved config")
    args = parser.parse_args()
    asyncio.run(
        main(
            args.registry_id,
            build_id=args.build_id,
            expected_release_id=args.expected_release_id,
            expected_release_sha256=args.expected_release_sha256,
            api_base=args.api_base,
        )
    )
