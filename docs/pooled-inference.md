# Pooled MolmoAct2-LIBERO evaluations

This adapter runs independent HUD simulator environments against Dropbear's
pooled inference API. Configure a ready-made agent; HUD still executes actions,
grades tasks and records traces. Dropbear owns native model preprocessing, GPU
batching and inference. The client needs no GPU or model weights.

**Qualification status:** local contract and real HUD protocol tests pass. The new
pooled adapter's live HUD cohorts, success rates, startup distributions and costs
are pending. The older six-episode demo uses the exclusive-session API and does
not qualify this serving path. Do not present mock tests as GPU capacity evidence.

## Installation during development

The pooled API must be present in `dropbear.inference`. The repository's published
dependency pin still supports the existing exclusive-session integration. The
published Dropbear `0.1.0a16` does **not** contain the merged inference API, even
though the merged SDK source currently has the same version string. Until the new
SDK is published, internal developers can install the exact approved SDK checkout
through a local uv override. This is a development path, not a public release.

From a checked-out integration repository, with Python 3.12:

```bash
uv venv --python 3.12 .venv
# Create a private overrides.txt with these two lines, using your actual SDK path:
# numpy==2.2.6
# dropbear @ file:///absolute/path/to/mvp/mvp/sdk
uv pip install --python .venv/bin/python --override /private/path/overrides.txt -e .
.venv/bin/python -c 'from dropbear.inference import AsyncInferenceClient'
```

The integration records installed SDK bytes and source commit alongside package
versions. NumPy 2.2.6 remains an explicit repository-local override of OpenPI's
NumPy<2 constraint; wire round trips are tested under that combination. Simulator
dependencies remain isolated in the locked CPU environment.

## Run a cohort

First deploy the separate raw360 environment using the
[remote-build instructions](startup-profiling.md#reproducible-pooled-environment-build).
Keep this registry exclusive to the campaign; the legacy `dropbear-libero`
deployment retains its original observation contract. Configure HUD and Dropbear
credentials outside this repository using their normal SDK configuration.
The CLI accepts `DROPBEAR_API_KEY` from the environment, with saved SDK
configuration as the fallback; credentials are never written to run evidence.

```bash
.venv/bin/hud-dropbear pooled \
  --runtime hud \
  --concurrency 8 \
  --registry-id YOUR_DEDICATED_REGISTRY_UUID \
  --build-id YOUR_DEPLOYED_BUILD_UUID \
  --expected-release-id YOUR_QUALIFIED_RELEASE_ID \
  --expected-release-sha256 YOUR_RELEASE_MANIFEST_SHA256 \
  --output artifacts/pooled-8-attempt-001
```

`--api-base` selects the management origin when it differs from saved Dropbear
configuration. Action traffic uses the returned regional inference origin and
SDK connection discovery. The server chooses actual compute placement; record it
from provider evidence rather than assuming the control-plane region is the GPU
region. Do not use a private diagnostic route for the evaluation.

`--concurrency` accepts any integer 1–64. Eight active slots fit each whole-model
H100 worker; reserved capacity is rounded up to a multiple of eight. Requesting
three environments therefore still reserves one worker. These are independent
closed-loop simulators, not one observation copied into multiple requests.

The default is two sequential episodes per lane. Frozen cases cycle through
`libero_spatial` task-order0, tasks0–2 and initial states0–1. Each row records task
name, seed, lane, episode identity and request noise seed before allocation. For
one lane, use `--episodes-per-lane 6` to cover all six selections. A new output
directory is required for every attempt.

The contract is raw agent-view and wrist-view RGB360×360, position3, normalized
XYZW quaternion4, gripper positions2, and returned action chunks10×7. Preserve
raw image orientation: the server applies the checkpoint's native preprocessing
once. Physics runs at20Hz with ten settling steps and a600-action limit. Shorter
`--max-steps` runs are smoke tests and cannot pass evaluation acceptance.

For local CPU simulation, replace `--runtime hud` with
`--runtime local-container --image YOUR_POOLED_IMAGE`. Local runs exercise the
same agent and action contract but cannot pass the HUD-hosted acceptance gate.

## Full Python example

The CLI is the recommended reproducible evaluation entry point because it saves
the manifest, timings, every grade, video and platform verification. This smaller
example shows the public composition and is also provided in
[`examples/pooled_libero.py`](../examples/pooled_libero.py):

```python
import asyncio
import os

from hud import HUDRuntime
from hud.eval.job import Job
from hud.utils.platform import PlatformClient

from hud_dropbear import PooledProvider, PooledRobotAgent, RuntimePool
from hud_dropbear.pooled_cli import cohort_tasks, run_cohort
from hud_dropbear.startup_cleanup import ExclusiveRegistryCampaign


async def main():
    concurrency = 8
    tasks = cohort_tasks(concurrency, episodes_per_lane=2)
    job = await Job.start("dropbear-libero-parallel")

    async with (
        ExclusiveRegistryCampaign(
            PlatformClient.from_settings(),
            os.environ["HUD_ENVIRONMENT_ID"],
            expected_instances=concurrency,
            environment_name=tasks[0].env,
        ) as registry,
        PooledProvider(
            concurrency=concurrency,
            api_key=os.environ.get("DROPBEAR_API_KEY") or None,
            # Omit api_base to use saved SDK configuration.
            api_base=os.environ.get("DROPBEAR_API_BASE") or None,
            expected_release_id=os.environ["DROPBEAR_RELEASE_ID"],
            expected_release_sha256=os.environ["DROPBEAR_RELEASE_SHA256"],
        ) as provider,
        RuntimePool(HUDRuntime(), tasks, concurrency=concurrency) as runtimes,
    ):
        await registry.validate_ready()
        agent = PooledRobotAgent(provider=provider, runtimes=runtimes)
        await run_cohort(
            agent,
            tasks,
            runtime=runtimes,
            job=job,
            concurrency=concurrency,
            rollout_timeout=900,
        )

    successes = sum(
        run.reward == 1 and not run.trace.is_error and not run.grade.is_error for run in job.runs
    )
    print(f"HUD job {job.id}: {successes}/{len(tasks)} successes")
    # An acceptance claim additionally requires platform/video verification.


asyncio.run(main())
```

`run_cohort` uses HUD's public `Taskset.run` under one shared job, with one
sequential worker per lane. Free lanes immediately start their next episode.
Each completed row is retained immediately, so cancelling
another row cannot discard its grade from the returned job. No HUD fork or private
runtime override is needed.

## Similarities to language agents, and differences

| Concern | This robotics integration |
|---|---|
| Ready-made provider agent | Configure `PooledRobotAgent`; no subclassing required by users |
| Inference API | Native async authenticated HTTP SDK, independent of simulator hosting |
| Evaluation harness | Standard HUD tasks, `Taskset.run`, job, traces and simulator grades |
| Model abstraction | A `Model.ainfer` wrapper plus an explicit observation/action adapter |
| Payload | Images, geometry and instructions in; action chunks out. Not Chat Completions |
| Resource lifetime | Explicit async context owns a paid deployment and reusable simulators |
| Request recovery | Stable slot, monotonic sequence and request ID; uncertain calls are resolved, never blindly replayed |
| Episode state | Fresh adapter and action queue per episode; transport clients persist |
| Batching | Dropbear's prefill/action scheduler owns batching; do not add HUD `BatchedModel` |
| Model attribution | Immutable provider identity is recorded in trace metadata; generic HUD model attribution remains an upstream API request |

The additional context managers make billable resource ownership visible. Closing
an HTTP client alone does not stop a deployment. The integration waits for the
owned deployment to report `stopped` and independently verifies termination of
the exact HUD instances. It never adopts or stops someone else's deployment.

## Evidence and timing

Each output directory contains `cohort.json`, `results.json`, append-only
`timings.jsonl`, a private `requests.jsonl` recovery journal, `report.json`, local
HUD spans and videos. Keep run artifacts out of the public repository.

The gate requires every planned row, distinct episode and trace identities, every
lane, zero integration errors, full600-action limits, HUD hosting, verified grades
and both camera streams, confirmed cleanup and at least50% success. Every attempt
retains its own outcome; successful reruns do not erase failures.
It also requires observed overlap across all requested simulator lanes and
advertised ready inference capacity. The report separately measures peak
concurrent HTTP requests and time at each peak. These client measurements do not
prove GPU worker count, batch size or utilization.

Timing distinguishes lease acquisition, control readiness, task setup, first usable
model input, first executed action, inference deployment readiness, per-episode
first inference, steady successful inference, and failed/cancelled attempts. Client
encoding includes queue wait; SDK POST time excludes outer journaling and recovery.
The existing journal's lock wait, write, flush and fsync are timed separately
before submission and at terminal persistence. These spans overlap the outer
model call; durability has not been removed to improve latency numbers.
Server-reported GPU/scheduler phases use separate clocks and can overlap. Do not
sum them or compare end-to-end RTT directly to GPU-only execution time.
The client-entry clock starts before SDK imports but excludes Python interpreter
launch and preceding standard-library imports. Full command startup needs a
parent-process timestamp; the report does not label the narrower interval as such.

Use the [GPU-free startup profiler](startup-profiling.md) for repeated simulation
measurements. A new lease proves a new simulator process, not an empty host image
cache. Compute-allocation and server-initialization phases require corresponding
provider evidence; unmeasured phases remain explicitly unattributed.

Internal cost accounting is separate from model success. `CampaignBudget` records
an approved cap, startup/idle/run/cleanup reservations, exact owned app/instance
IDs and posted billing. It retains liability until termination and billing coverage
are confirmed. App-level costs divided among cohorts by time are estimates. GPU
kernel time alone cannot establish operating cost.
The ledger bounds declared reservations; it is not a provider-side spending
switch. Rates must include the actual CPU/RAM envelope and placement charges,
with finite resource lifetimes, verified teardown and billing-lag reserves.
Requested resources alone may not bound a provider's billable resource use.
