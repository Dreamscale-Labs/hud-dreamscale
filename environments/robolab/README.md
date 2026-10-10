# RoboLab DROID on HUD with Dreamscale policies

RoboLab (Isaac Lab) DROID tasks on HUD: the simulator runs in a Modal L40S
sandbox per episode (HUD `ModalRuntime`), the agent runs locally and calls a
Dreamscale cloud policy through the public `dreamscale` SDK. Supported models:
`cosmos3-nano-policy-droid` and `flux-3-action-droid` (same wire), plus `hold`
for a no-inference smoke.

## Run a wave

```sh
# The sim image (RoboLab ad45d4f, Isaac Sim 5.0) is prebuilt; the default id is
# in run_waves.py. Only after changing env.py / sim.py / Dockerfile.hud:
#   uv run --project environments/robolab python environments/robolab/build_image.py
#   export ROBOLAB_MODAL_IMAGE=modal://im-...

# Dreamscale key for the chosen control plane (never printed or written to the repo).
set -a; . ~/.config/dreamscale/test.env; set +a      # prod; dev.env for --env dev

uv run --project environments/robolab python environments/robolab/run_waves.py \
  --model cosmos3-nano-policy-droid --tasks all --episodes-per-task 5 \
  --concurrency 10 --env prod --output runs/cosmos3-prod-w1
```

Useful flags: `--tasks BananaInBowlTask,PickDrillTask`, `--episode-offset K`
(continue a wave), `--keep-warm 300` (SDK `keep_warm`, 0–3600 s),
`--max-steps N` (smoke cap), `--no-local-video`, `--dry-run` (print episodes),
`--dreamscale-home DIR` (use an existing SDK home as-is). HUD uses
`HUD_API_KEY` (or `hud set`), Modal its usual token.

Output (`--output`):

- `summary.json`: per-episode success, RoboLab score, steps, HUD trace URL,
  session id/transport, latency (SDK round trip, server inference, sim step) and
  startup timings (sandbox boot, ready → agent, session connect, first
  observation); per-task and overall success; pooled p50/p95 latency; Modal
  sandbox-seconds with a rough cost estimate; the HUD job URL.
- `wave.json`: the plan and job URL, written before the first sandbox starts.
- `episodes/<task>__ep<k>__<trace>/timing.jsonl`: one row per SDK request
  (`chunk_index`, `step`, `sdk_rtt_ms`, `server_inference_ms`, SDK timing
  breakdown, transport), plus session and end rows; `episode.json`;
  `policy_views.mp4` (left | wrist | right at 15 fps).

HUD keeps the canonical trace with per-camera video and every executed chunk.

## What the harness does

- **Scene and cameras** follow RoboLab's own Cosmos runner and Dreamscale's
  `integrations/cosmos3-robolab` client: `auto_register_droid_envs(task, cameras=WRIST_LEFT_RIGHT_HEAD)`,
  `create_env(seed=scene_seed=0)`. The env publishes `exterior_1_left`
  (over-shoulder left), `exterior_2_left` (over-shoulder right) and `wrist_left`
  after RoboLab's `resize_with_pad(…, 360, 640)` — exactly the arrays
  `Cosmos3Client` packs. RoboLab renders 1280x720, so this is a 2x downscale
  with no padding, and it cuts the per-step transfer to the local agent 4x.
  Frames are upright as rendered (no flip).
- **State** is the 7 Franka joint positions (rad) plus RoboLab's gripper
  closed fraction in [0, 1]. Actions are 7 absolute joint targets plus the
  gripper, thresholded at 0.5.
- **Task template** `episode(task_name, episode, instruction_type, scene_seed)`.
  `episode` seeds the scene reset (`env.reset(seed=episode)`), done twice as
  `robolab.eval.run_episode` does. `tasks.json` holds the ten-task screen from
  `mvp/infra/benchmarks/cosmos3_tasks.json`; their `episode_s` values are the
  task classes' own `episode_length_s`, so RoboLab truncates at
  `episode_s x 15` steps (750 for BananaInBowlTask, 1800 for BlackItemsInBin).
- **Grade**: reward is RoboLab's success (1/0), from `env.get_env_results()`
  once RoboLab freezes the env (success term or time limit). `info` adds
  RoboLab's subtask score, its termination step, steps executed, the time
  limit and the last event reason.
- **Agent**: per episode, one `dreamscale.aconnect(model, acceleration="pytorch",
  rtc="off", calibration="off", region="us-west-2", control_hz=15,
  keep_warm=…)` session, opened before dialing the sim so a cold start cannot
  trip the bridge's action-wait timeout (raised to 300 s on the sim side for
  slow replies). Each request is `dreamscale.franka.observe(left, right, wrist,
  joints, gripper)` plus the instruction; the reply must be a finite 32x8
  chunk; all 32 actions execute open-loop, then it replans. The sim only steps
  when an action arrives, so simulated time is independent of latency.

## Notes and limits

- This directory is its own uv project. Its `dev` group (the SDK, Modal,
  pytest) is local only: `Dockerfile.hud` runs `uv sync --no-dev`. The repo
  root keeps its older SDK pin for LIBERO.
- `dreamscale==0.1.0a56` has no client-side FLUX contract entry; the server
  is authoritative and the agent checks 32x8 and finiteness itself.
- Every observation (3 x 640x360 RGB) travels from the sandbox to this machine
  each step (~2 MB, ~1.5 GB for a 750-step episode). Measured with the hold
  smoke: Isaac step ~171 ms + packing ~20 ms in the sandbox, ~700 ms per step
  as seen by the agent, so transfer dominates wall-clock (a 750-step episode
  takes ~9 min, plus ~5 min sandbox boot and scene build). It never affects
  simulated time or scores. For big waves, run the runner on a well-connected
  host (e.g. us-west-2, next to inference) and keep concurrency within the
  downlink (~10 episodes ≈ 200+ Mbit/s).
- Modal cost is roughly $2.7 per sandbox-hour (L40S + 8 CPU + 48 GiB); the
  summary's `modal.estimated_usd` uses list prices, the dashboard is the bill.
  One pass over all ten tasks is ~8,550 steps, about 2.5 sandbox-hours here.
- The default image id lives in `run_waves.py` (`ROBOLAB_MODAL_IMAGE` overrides);
  rebuild with `build_image.py` after changing `env.py` / `sim.py`.
- The sandbox is terminated by `ModalRuntime` at the end of each episode; the
  runner re-checks every sandbox id it saw and terminates any survivor.

Tests (no GPU, no network): `cd environments/robolab && uv run pytest && uv run ruff check .`
