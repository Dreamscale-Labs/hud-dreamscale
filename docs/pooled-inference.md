# Pooled LIBERO inference

This is additive to the scalar `DreamscaleRobotAgent` integration. Use
`PooledProvider`, `PooledRobotAgent` and `RuntimePool` for several independent
simulators sharing Dreamscale's HTTP inference service. The source is reconciled
onto the renamed mainline; historical profiling, cost-analysis and campaign
orchestration code is deliberately not imported.

## Ownership and contract

- The operator enables an owner-scoped funded grant. The job creates one fresh
  deployment, retains it across episodes, and stops only that deployment on exit.
  Ongoing service definitions remain available after its GPU workers stop.
- Declare concurrency once (1–64). Capacity is rounded up to eight robots per
  H100 and requested before inference. The provider retains one client and one
  monotonically sequenced, single-in-flight lane per robot. A restart starts a
  fresh deployment; it never guesses the state of someone else's request.
- `RuntimePool` leases one independent simulator per lane and reuses it across
  sequential episodes. Runtime placement remains HUD's public runtime interface.
  HUD owns simulation, grading and traces; Dreamscale owns inference/batching.
- Use **`dreamscale-libero-pooled`**, profile **`libero-raw360-v1`** at 20 Hz,
  raw 360×360 agent/wrist RGB and raw position/XYZW quaternion/gripper state.
  The server owns orientation, resize and axis-angle conversion. The scalar
  `dreamscale-libero` environment remains 256×256 with its original state format.
- Ten returned actions are applied through the normal HUD robot driver. Initial
  inference is staggered automatically by 0,125,…,875 ms within each group of
  eight, once per lane per job. Subsequent requests are simulator-driven: this
  is neither a recurring rate limiter nor a guaranteed 1 Hz wall-clock cadence.
- The selected backend remains P2/A4, BF16 with FP32 text SwiGLU. This integration
  does not change engines, precision, checkpoint, smoothing or scalar QUIC.

## Client and environment

Install the pinned project with `uv sync --locked`. SDK 0.1.0a27 supports both
Modal gateway discovery and discovery at the exact configured Dreamscale API
origin. Management uses `https://api.dreamscalelabs.com`; inference uses the
deployment's returned `api_base`. Do not hardcode a worker tunnel or bypass
`connect()`. Credentials use the SDK's normal login/configuration.

The example [pooled_libero.py](../examples/pooled_libero.py) composes the public
interfaces without changing the scalar CLI. Supply the operator's exact release
ID and manifest SHA256. `HUDRuntime()` requires the separate pooled environment
to have been deployed in your HUD workspace first. The example does not deploy
it, assume that a legacy registry is renamed, or claim that it is already hosted.
An attached/local runtime provider can be passed to `evaluate()` instead.

For local native simulation, follow the README's simulator setup but serve
`pooled_env.py` instead of `env.py`. For containers, `Dockerfile.hud` includes
both entrypoints; `Dockerfile.pooled` selects the pooled one from a digest-pinned
base built from this revision. Never reuse an old scalar-only image as that base.

## Failure and cleanup

POSTs are never automatically replayed after uncertain completion. The provider
first recovers the exact request identity through the public result/resolve API;
an unresolved lane is fenced. Cancellation and partial startup still close owned
resources with bounded cleanup. An optional private journal preserves request
identities before submission. Treat failed cleanup as actionable, not successful
completion. Do not reuse the provider object after close.

Unit tests cover the real HUD socket/claim/grade loop with a fake simulator and
mock inference, lane reuse, cancellation, initial staggering, authority renewal,
request recovery and teardown. Those tests are not real model or policy-quality
evidence; live qualification is recorded separately in the product repository.
