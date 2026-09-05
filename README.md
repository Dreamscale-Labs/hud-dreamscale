# hud-dropbear

Run [HUD robotics evaluations](https://docs.hud.ai/v6/advanced/robots) with
Dropbear-hosted [MolmoAct2-LIBERO](https://huggingface.co/allenai/MolmoAct2-LIBERO)
inference. HUD owns simulation, action execution, grading and video traces;
Dropbear owns the model server. The agent needs no local GPU or model weights.

## Install

Use Python 3.12 and [uv](https://docs.astral.sh/uv/):

```sh
uv sync --locked
```

Configure Dropbear with `uv run dropbear login` and HUD with `uv run hud login`.
Existing SDK credentials are reused. Keep credentials outside the repository.
Dropbear usage and HUD-hosted simulation incur their respective service charges.

The agent lock explicitly overrides NumPy to 2.2.6: OpenPI 0.1.2 declares NumPy
<2, while Dropbear requires NumPy 2. We test the array codec and HUD protocol
under this override. Ordinary `pip install` is not currently supported by those
upstream dependency constraints. No global Python environment is changed.

## Local simulation

```sh
docker build -f Dockerfile.hud -t hud-dropbear-libero:local .
uv run hud-dropbear --runtime docker --task-ids 0 --init-state-ids 0
```

The container runs CPU MuJoCo physics and OSMesa rendering. Its separately
locked dependencies include CPU PyTorch only for LIBERO's initial-state loading;
no policy weights are installed there. Assets are pinned and downloaded during
the build, so episode startup does not download them.

To run all six episodes, omit the task-selection options. For an already-served
HUD environment use `--runtime attached --env-url tcp://127.0.0.1:8765`.

## HUD-hosted simulation

After local validation:

```sh
uv run hud deploy . --no-env --runtime hud
uv run hud-dropbear --runtime hud
```

This uses [`HUDRuntime`](https://docs.hud.ai/v6/reference/runtime): the CPU
environment runs on HUD, while the agent client runs on your machine and talks
to Dropbear. The default inference region is Sydney; `--region us-west-2` selects
Oregon. One connection is retained across the sequential episodes and closed at
the end. Retaining it is billable; `keep_warm` is zero after close.

Results go to a new directory under `artifacts/` (or `--output <new-directory>`):

- `results.json`: every HUD trace ID, task selection, score, errors, serving
  identity, job link and startup measurements.
- `timings.jsonl`: per-episode startup and per-inference timing joined by task,
  trace, session and observation IDs.

With HUD credentials, HUD records camera videos, state and action chunks.
Local sidecars contain no camera payloads or credentials. CLI startup timing
begins at entry to the Python CLI module; first-action timing ends when the
simulator acknowledges the action with its next observation. Simulator-side
Unix timestamps are recorded separately and never subtracted from local clocks.

## Evaluation contract

The default taskset contains `libero_spatial` task-order 0, task IDs 0, 1 and 2,
with initial-state IDs 0 and 1. Names are checked against the pinned package:
pick up the black bowl between the plate and ramekin, next to the ramekin, and
at table center, placing it on the plate.

The environment uses 10 Hz actual control, ten settling steps, ten-action
chunks and a 600-action limit. State is EEF XYZ, axis-angle radians and two
gripper positions. Actions use LIBERO's native seven-value delta-EEF control.
Raw 256×256 RGB agent/wrist cameras are passed to Dropbear, which performs its
rotation, resize and encoding once. HUD traces preserve the raw camera orientation.
RTC and calibration are off because HUD drives a synchronous chunk loop.

The demo passes when the six HUD-hosted episodes have no integration errors and
at least one simulator-confirmed success. Every attempt is retained. A lower
`--max-steps` value is a smoke test and cannot pass demo acceptance. These settings
test integration, not reproduction of the checkpoint's published benchmark score.

Startup below two to three minutes is a target, not a pass/fail threshold. Warm
repeats reuse the inference session; they do not prove parked-session reclaim.

## Use the provider in Python

```python
from hud import HUDRuntime, Taskset
from hud_dropbear import DropbearRobotAgent
from hud_dropbear.cli import tasks

async def evaluate():
    async with DropbearRobotAgent() as agent:
        return await Taskset("libero-demo", tasks()).run(
            agent, runtime=HUDRuntime(), max_concurrent=1,
        )
```

The provider reuses HUD's `RobotAgent` loop and overrides native async inference.
It does not support `BatchedModel`, concurrent episodes, arbitrary checkpoints,
or DreamZero's stateful episode lifecycle. Unknown model artifacts and mismatched
camera/state/action contracts fail before actions are sent. The small
`serving_contracts.json` registry ties verified TensorRT artifact fingerprints to
checkpoint revisions; verify new build manifests before adding entries.

## Verify

```sh
uv run ruff check .
uv run pytest -q
```

Tests exercise the real HUD robot socket, claim/grade/release lifecycle, input
conversion, episode reuse and failure/cancellation cleanup with a fake policy.
They do not count as real-model evaluation. Live acceptance evidence is tracked
separately; no score or startup result is claimed before a completed run.

Integration code is MIT licensed. Dependencies, model checkpoints and simulator
assets retain their respective upstream licenses.
