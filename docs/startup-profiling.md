# Simulation startup diagnostics

The startup profiler opens no Dropbear deployment or inference connection. It
creates a HUD runtime, starts a fixed LIBERO task, claims the robot capability,
reads its first observation, closes the robot client, then requests the diagnostic
grade. It sends **zero policy actions**; the environment retains its standard ten
settling steps. These are diagnostics, with no HUD evaluation job or model score.

The default campaign is five sequential fresh runtime leases, each with an initial
reset and three same-task warm resets: `libero_spatial`, task 0, initial state 0,
seed 0, 20 Hz, ten settling steps, 600-action limit. Defaults retain the historical
256-pixel `libero-legacy-v1` profile. Select `libero-raw360-v1` for the pooled API;
results from these profiles belong to separate cohorts.

```bash
python -m hud_dropbear.profile_startup \
  --runtime hud --profile libero-raw360-v1 \
  --registry-id REGISTRY_UUID --build-id BUILD_UUID --exclusive-registry \
  --output /private/path/startup-raw360.json
```

Use an otherwise idle registry dedicated to the campaign. The runtime SDK resolves
by environment name, so freeze the deployed build throughout the campaign. The
optional `--build-id` verifies the actual instance build identity. A complete
before/after instance inventory establishes ownership only under that exclusive
registry assumption. Cleanup stops only the exact new instances and verifies
termination. Ambiguity, unexpected processes, incomplete diagnostics, a failed
episode or unresolved cleanup stops further trials and preserves partial evidence.

Local-container mode uses `--runtime local-container --image IMAGE` and verifies
that its exact Docker container was removed. It does not build an image.

## Measurement boundaries

Each episode and lease has its own ID, independently of repeated task names.
Every summary reports its own sample count. A failed later warm reset remains in
the attempt log and does not discard the already validated cold observation;
incomplete phases are excluded from completed-duration summaries.
Durations use a monotonic clock within one process; process IDs and UTC timestamps
provide correlation. Never subtract monotonic values from separate processes or
add overlapping span durations.

| Measurement | Boundary |
|---|---|
| Runtime acquire | Provider context entry through its yielded runtime endpoint |
| Control connect ready | HUD control connection and hello handshake |
| Task setup | `start_task` request through the task's first yield |
| Claim / first observation | `RobotClient.connect`, including its claim handshake and initial observation |
| Request to first observation | Runtime acquisition start through validated initial observation; excludes grade and cleanup |
| Warm ready | Subsequent `start_task` through validated first observation on the same runtime |
| Parent startup | Endpoint spawn/connect envelope and capability discovery |
| Child reset | Previous sim close, physics check, LIBERO imports, task/assets selection, simulator construction/reset, initial state and settling |

Parent and child diagnostics are returned in `info.startup_profile` and emitted as
bounded JSON lines on stderr. Stdout remains available to HUD's port discovery
protocol. No startup instrumentation runs on the inference/action hot loop.
Module/interpreter startup before instrumentation and the platform's scheduler,
image pull and allocation phases are **not individually attributed**. A fresh
lease and new simulator process do not prove a cold host or empty image cache.

Startup waits are bounded to 900 seconds initially, 120 seconds per warm reset,
and 1,800 seconds for the campaign's work. Cleanup has separate bounded time
allowances: 30 seconds per connection/runtime close and up to 120 seconds for
instance verification. Cleanup may therefore extend beyond the work deadline.

## Reproducible pooled environment build

The legacy Dockerfile copies both entrypoints but retains its original default.
For local builds, build `Dockerfile.hud`, then use its immutable repository digest
as the pooled wrapper's base:

```bash
docker build -f Dockerfile.hud -t libero-base .
docker build -f Dockerfile.pooled \
  --build-arg LIBERO_BASE_IMAGE=REGISTRY/IMAGE@sha256:DIGEST -t libero-pooled .
```

For HUD remote builds, prepare the minimal context with the same locked dependency
layers and only the entrypoint changed. This does not build or deploy anything:

```bash
python scripts/prepare_pooled_build.py /private/path/empty-pooled-context
```

The generated `Dockerfile.hud` serves `pooled_env.py`; `build-context.json` records
the SHA256 of every copied source. Credentials, private artifacts, docs, tests,
Git metadata and local environments are excluded. Deploy this as
`dropbear-libero-pooled`, preserving the legacy `dropbear-libero` deployment.
