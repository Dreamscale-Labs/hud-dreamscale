# MolmoAct2-LIBERO integration demo

On 6 September 2026, the fixed six-episode HUD-hosted taskset completed with
**six simulator-confirmed successes and zero integration errors**. HUD ran CPU
LIBERO simulation, executed actions, graded outcomes and recorded both camera
streams. Dropbear served the TensorRT model; the agent client needed no GPU.

This is an integration demonstration on three selected tasks and two initial
states per task. It does not estimate accuracy across LIBERO or reproduce the
checkpoint's published benchmark score. The qualification used a development
deployment; it is not a production latency guarantee.

## Reproduce the workflow

Follow the [installation and native-container preflight](../README.md) first.
Then deploy the environment and run the default taskset:

```sh
uv run hud deploy . --no-env --runtime hud
uv run hud-dropbear --runtime hud --suite libero_spatial \
  --task-ids 0 1 2 --init-state-ids 0 1 --control-hz 20 --max-steps 600
```

Use `--region us-west-2` to select Oregon instead of the default Sydney target.
This command uses your configured accounts. Check `results.json` for the resolved
checkpoint, TensorRT artifact and region; region selection alone does not pin a
particular model-server release. The measured artifact is recorded below and in
[the portable result](demo-result.json).

Success is determined by the simulator, independently of the model. Check
`demo_passed`, all six `runs`, `platform_evidence`, and both saved camera videos
per episode. The CLI exit status reports integration errors; it is not a claim
that all tasks succeeded. A run with a reduced action limit is only a smoke test.

## Results and protocol

All tasks pick up the black bowl and place it on the plate. Task IDs refer to
`libero_spatial`, task order 0; the runner asserts the full task names.

| Task ID | Bowl location | Initial state | Simulator success | Executed actions |
| --- | --- | ---: | --- | ---: |
| 0 | Between the plate and ramekin | 0 | Yes | 77 |
| 0 | Between the plate and ramekin | 1 | Yes | 106 |
| 1 | Next to the ramekin | 0 | Yes | 117 |
| 1 | Next to the ramekin | 1 | Yes | 102 |
| 2 | Table center | 0 | Yes | 98 |
| 2 | Table center | 1 | Yes | 94 |

Simulation seed 0, actual control rate 20 Hz, ten settling steps, ten-action
chunks, 600-action limit. One inference session and one HUD runtime were reused
sequentially, with fresh episode state and action queues. RTC, calibration and
post-close `keep_warm` were disabled. Model sampling was not reseeded per episode,
so an identical task selection does not promise identical actions.

The simulator used native Linux/x86-64 CPU physics and OSMesa rendering. Inputs
were raw 256×256 agent-view and wrist-view RGB frames plus EEF XYZ, axis-angle
rotation and two gripper positions. Dropbear applied its LIBERO image transform
once. HUD videos retain the raw simulator orientation; their displayed rotation
does not indicate the model's final preprocessed orientation.

The preceding final-qualification attempt ended with six integration errors
before any model actions: the identity check rejected a disconnected worker
registration alongside the ready worker. The corrected check uses the single
ready worker and active session. Both job attempts remain in the private evidence
ledger; the failed job is not a model-quality failure or part of the six-success
denominator. Earlier development runs include failed 10 Hz rollouts and an
emulated-container physics failure; those informed the corrected protocol and
remain separate from this final qualification sequence.

## Version pins

| Component | Version or revision |
| --- | --- |
| Integration source used for the run | `26873b5602db869a447613527e029539c21c3a51` |
| Agent Python | 3.12 |
| Dropbear SDK | `0.1.0a15` |
| HUD SDK | `0b63b4d3b9acb6d095e0886e18b2c905219e1e5a` |
| Agent NumPy override | `2.2.6` |
| hf-libero / MuJoCo / robosuite | `0.1.3` / `3.3.7` / `1.4.1` |
| LIBERO assets | `0b3ea86be5fe169d0fd036ae63d1070ec09e90f6` |
| Checkpoint | `allenai/MolmoAct2-LIBERO` |
| Checkpoint revision | `0d24a92bd1faf321ef497c3bbd5681af97c65aa2` |
| TensorRT artifact | `714f89f13e8d3ede87af7049` |

The model artifact uses full TensorRT neural execution, BF16, ten denoising steps
and a 128-token profile. CPU preprocessing and action normalization remain
outside TensorRT. No model weights or TensorRT engines are distributed here.

The [portable result](demo-result.json) contains only protocol and score data.
New runs produce their own authenticated HUD links, full traces, videos and
timing sidecars under `artifacts/`. Those outputs are ignored by Git.
