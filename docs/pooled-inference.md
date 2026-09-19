# Pooled MolmoAct2-LIBERO evaluations

This adapter runs independent HUD simulator environments against Dropbear's
pooled inference API. Configure a ready-made agent; HUD still executes actions,
grades tasks and records traces. Dropbear owns native model preprocessing, GPU
batching and inference. The client needs no GPU or model weights.

**Qualification status (19 September 2026, 20:20 UTC):** on the current reusable-service
release, HUD-hosted cohorts passed with zero episode integration errors and
100% simulator success at widths 8 (16/16), 3 (6/6), 1 (6/6, all six fixed
task/initial-state pairs), 32 (64/64), 9 (18/18) and 33 (66/66); every planned
row ran and is retained, every job was re-read through the HUD Platform API, and
the exact provider Apps were confirmed stopped afterwards. Widths 1–33 therefore
validate arbitrary concurrency below the eight-worker ceiling; 64 lanes (eight
H100 workers) has not yet passed acceptance. Its attempts are all recorded:
68/128, 97/128 and 120/128 successes with 60, 31 and 8 integration errors, then
62/128 with 66 errors, each above or near the 50% floor but failing the
zero-integration-error gate. Those runs exposed three client defects that are
now fixed with tests: a leased HUD runtime whose control connection never became
ready stalled the whole pool (now released and replaced within a per-lane bound);
a wave of episode completions saturated the event loop's default thread pool
with recorder finalization and starved the durable request journal that precedes
every inference POST (journal writes now use a dedicated worker and the default
pool is sized per lane); and a transient inability to open new connections fenced
slots after single connect timeouts (connection-phase failures are now retried
before any request bytes are sent, with lane-sized keepalive pools). The 33-lane
pass is the first cohort run with all three fixes. The dev workers for these
results were placed on GCP `us-west4` under an explicit, dev-only relaxation of
the AWS cloud pin after AWS `us-west` H100 capacity was unavailable for several
hours; the region selector and the model contract were unchanged, and each
worker records its actual placement. From an operator host in AWS `us-west-2`
the steady client-observed model call is p50 about 0.52 s against an 87 ms
replica wall; from Sydney it was about 1.08 s. Earlier finite-release results
(two eight-lane passes, a three-lane pass, a failed one-lane attempt and a
failed 64-lane attempt with 78/128 successes and 50 integration errors) remain
recorded but do not qualify this release. The older six-episode demo uses the
separate exclusive-session API. Local contract tests are not GPU capacity
evidence.

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

Wide camera recording additionally requires the explicit
[HUD recording patches](../patches/README.md). They bound per-camera codec threads
and move recorder finalization off the asyncio event loop while retaining
ownership through repeated cancellation. The original SDK exhausted native
threads in a 64-lane fixture; its synchronous finalization also blocked unrelated
async peers in a controlled exporter-delay reproduction. The cumulative patch
at local revision `b31a73935084b8e56740f4787462798484e4f77c` passed the real HUD wire
and ordinary recording lifecycle for 128 synthetic episodes: all 256 camera
streams and 768 frames decoded correctly, with no encoder warnings or remaining
camera threads. This is a recording test, not a live GPU or policy-quality result.
The patch changes only agent-side recording; the hosted simulator uses the
original HUD build. Follow the included immutable base, tree/hash checks and
installation steps when reproducing the wide run. The existing best-effort
15-second per-camera join timeout remains: a stuck encoder or exporter can still
leave incomplete recording. The qualification reports observed complete streams,
not a guarantee under arbitrary encoder failure.

## Run a cohort

First deploy the separate raw360 environment using the
[remote-build instructions](startup-profiling.md#reproducible-pooled-environment-build).
Keep this registry exclusive to the campaign; the legacy `dropbear-libero`
deployment retains its original observation contract. Configure HUD and Dropbear
credentials outside this repository using their normal SDK configuration.
The CLI accepts `DROPBEAR_API_KEY` from the environment, with saved SDK
configuration as the fallback; credentials are never written to run evidence.

An operator first provisions the funded service and matching account grant.
Credentials alone do not provision this capacity. The CLI creates one owned
inference deployment and stops it when the run exits. With an ongoing grant,
normal stop releases that deployment's GPU workers and retains the service apps.
A later deployment with a new creation key can use those same apps, subject to
the unchanged account grant and cumulative funding cap.

The combined server source supporting this lifecycle is deployed in development;
live qualification of ongoing renewal, worker rollover and stop/recreate reuse is
still pending. The earlier finite results do not establish those behaviors, and
they are not a production qualification or a published SDK release.

The historical qualification campaign used finite grants. Finite mode still
terminates the granted GPU app on stop, so another finite run needs operator
reprovisioning and a matching fresh grant. A new creation key or unused grant time
does not recreate a stopped app. Do not change an active finite grant into an
ongoing grant in place.

`PooledProvider` checks deployment authority in the background, at most 60 seconds
apart and earlier near expiry. It accepts a new horizon only from authenticated
status for the same deployment, release, origin and capacity. Temporary transport
failures retain only the last verified unexpired horizon; expiry, identity change
or lost authority fences inference and triggers owned-deployment cleanup. These
checks do not add status requests to ordinary inference calls or renew funding
themselves. The finite campaign runner still enforces its original fixed deadline
and budget; it does not become an ongoing runner through provider status refresh.

Normal deployment stop and final service app retirement are separate operations.
The operator owns the retained service apps and the separately billable CPU
gateway/sweeper app. Campaign-owned apps need explicit final retirement; shared
apps need their own funded lifetime and cleanup owner. Do not stop a shared app
belonging to another run.

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

`--max-not-admitted-resubmissions 1` opts into one new request only after an
identity-checked atomic resolution confirms that the original was never admitted.
The default is zero; the maximum is two. Uncertain or pending work is never
replayed. Each failed attempt remains in the timing sidecar and durable request
journal, and logical inference latency includes the entire recovery interval.
Python callers use the same `PooledProvider(max_not_admitted_resubmissions=1)`
option.

`--concurrency` accepts any integer 1–64. Eight active slots fit each whole-model
H100 worker; reserved capacity is rounded up to a multiple of eight. Requesting
three environments therefore still reserves one worker. These are independent
closed-loop simulators, not one observation copied into multiple requests.

The CLI defaults to six sequential episodes for one lane and two episodes per
lane for every other width. Frozen cases cycle through `libero_spatial`
task-order0, tasks0–2 and initial states0–1. Each row records task name, seed,
lane, episode identity and request noise seed before allocation. An explicit
shorter one-lane run remains a smoke test: campaign acceptance requires all six
task/initial-state pairs at seed0 in their frozen case assignments. A new output
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
example shows the public composition. Set `HUD_ENVIRONMENT_ID` to the dedicated
registry UUID, `HUD_BUILD_ID` to its expected immutable build UUID, and
`DROPBEAR_RELEASE_ID` / `DROPBEAR_RELEASE_SHA256` to the operator's qualified
release. [`examples/pooled_libero.py`](../examples/pooled_libero.py) provides the
same composition with required `--registry-id`, `--build-id`,
`--expected-release-id` and `--expected-release-sha256` arguments.

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

    async with (
        ExclusiveRegistryCampaign(
            PlatformClient.from_settings(),
            os.environ["HUD_ENVIRONMENT_ID"],
            expected_instances=concurrency,
            environment_name=tasks[0].env,
            build_id=os.environ["HUD_BUILD_ID"],
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
        job = await Job.start("dropbear-libero-parallel")
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
another row cannot discard its grade from the returned job. Task execution uses
HUD's public extension interfaces; the separate recording patch above is required
for the wide camera fixture.

The generic Python helper `cohort_tasks()` retains its two-episode default for
small lifecycle tests. For one-lane campaign acceptance, call
`cohort_tasks(1, episodes_per_lane=6)` explicitly; the CLI applies this default
automatically. The eight-lane example above keeps two episodes per lane.

## Similarities to language agents, and differences

This follows the ready-made provider-agent pattern discussed as “LLMAgent-style.”
In the pinned HUD SDK, the language-agent base is named `ToolAgent`.

| Concern | This robotics integration |
|---|---|
| Ready-made provider agent | Configure `PooledRobotAgent`; no subclassing required by users |
| Inference API | Native async authenticated HTTP SDK, independent of simulator hosting |
| Evaluation harness | Standard HUD tasks, `Taskset.run`, job, traces and simulator grades |
| Model abstraction | A `Model.ainfer` wrapper plus an explicit observation/action adapter |
| Payload | Images, geometry and instructions in; action chunks out. Not Chat Completions |
| Resource lifetime | Explicit async context owns a paid deployment and reusable simulators |
| Request recovery | Stable slot, monotonic sequence and request ID; uncertain calls are resolved, never blindly replayed |
| Transport retries | Connection-phase failures (DNS, TCP, TLS) are retried because nothing was sent; a language-agent SDK retries the same class, but here a sent inference request is never retried |
| Simulator readiness | A leased simulator whose control connection never becomes ready is released and replaced within a bounded number of attempts, the way an agent framework replaces a dead tool sandbox |
| Shared executors | Durable request journaling uses its own worker and the default thread pool is sized per lane, so recorder finalization cannot delay the next model call |
| Episode state | Fresh adapter and action queue per episode; transport clients persist |
| Batching | Dropbear's prefill/action scheduler owns batching; do not add HUD `BatchedModel` |
| Model attribution | Immutable provider identity is recorded in trace metadata; generic HUD model attribution remains an upstream API request |

`PooledRobotAgent` is configured directly; it is not registered with HUD's
`create_agent()` factory or model gateway catalog. `HUDRuntime` hosts the
simulators while the Python agent runs in the caller's process. Full-agent
execution through `HostedRuntime` is unsupported: this adapter does not provide
the language agents' serializable `hosted_spec()` contract.

The additional context managers make billable resource ownership visible. Closing
an HTTP client alone does not stop a deployment. The integration waits for the
owned deployment to report `stopped` and independently verifies termination of
the exact HUD instances. It never adopts or stops someone else's deployment.

Warm episodes within a run reuse the open provider, including its clients and
slot sequences. A custom Python campaign can retain that same provider across
**sequential** cohorts within its original funded lifetime. Create a fresh
`RuntimePool` and `PooledRobotAgent` for each job, with an integer runtime
concurrency from 1 through `provider.concurrency`. Runtime lanes use the first
N provider slots; smaller cohorts neither reset sequences nor remap fenced
slots. Each episode still receives a fresh adapter and action queue. Do not run
overlapping runtime pools against the same slots.

Active cohort width, exposed provider slots and funded capacity are distinct:

| Provider configuration | Sequential active cohort widths | Retained funded capacity |
|---|---|---|
| `PooledProvider(concurrency=8)` | 8, then 3, then 1 | 8 robot slots on one H100 throughout |
| `PooledProvider(concurrency=64)` | 64, 32, 1, 9, 33, 63 | 64 robot slots on eight H100s throughout |
| `PooledProvider(concurrency=3)` | 1–3 only | 8 robot slots on one H100; only three slots exposed |

These are supported composition patterns, not live GPU qualification results.
The unused capacity remains allocated and billable; this is warm reuse, not
capacity resizing or an inference cold start. Trace metadata records active and
provider concurrency alongside the deployment's reserved capacity. At each cohort
boundary, `provider_reused_capacity` can record a fresh authenticated deployment
status check; reporting uses its capacity without adding an inference cold-start
latency sample. Give each
cohort a separate HUD job, frozen task manifest, timing sidecar and grades; apply
the success gate to each cohort independently. All cohorts reference the same
provider creation and final cleanup receipt. Close each simulator pool before
the next cohort, keep the provider's outer context open until all cohorts finish,
and close it reliably on completion, failure or cancellation. Shared reuse does
not itself extend the deployment/grant lifetime or budget reservation.
For an ongoing service, only a fresh authenticated status response can establish
the control plane's renewed authority; the operator must fund that service
lifetime separately.

The CLI still owns one cohort per invocation and has no cross-cohort mode.
The historical finite qualification protocol provisioned fresh apps and grants
for its three independent eight-lane cold trials. Ongoing grants allow a new
deployment after normal stop without replacing the service apps. A cold-start
claim must still establish a fresh worker initialization; reusing an open
provider across 8/3/1 cohorts measures warm reuse instead.

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
encoding includes queue wait. The legacy `sdk_post_attempts_s` field measures the
awaited SDK `predict()` call, including connection discovery/refresh, JSON
serialization and response handling. It excludes outer encoding, journaling and
recovery; it is not an isolated HTTP POST RTT measurement.
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
