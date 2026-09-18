"""Opt-in real CPU LIBERO action-wire test; inference is explicitly fake and local."""

import os
import time

import numpy as np
import pytest
from hud import DockerRuntime
from hud.eval.job import Job
from hud.settings import settings

from hud_dropbear.contract import CAMERAS, CONTROL_HZ, POOLED_PROFILE
from hud_dropbear.pooled_agent import PooledRobotAgent
from hud_dropbear.pooled_cli import cohort_tasks, run_cohort
from hud_dropbear.runtime_pool import RuntimePool


class HoldProvider:
    """No SDK client or deployment: return an open-gripper, zero-motion action chunk."""

    concurrency = 1
    identity = {"id": "fake-hold-provider", "model": "fake-hold-actions"}

    def __init__(self):
        self.calls = []

    def is_slot_available(self, slot):
        return slot == 0

    async def predict(
        self, *, slot, observation, instruction, episode_id, noise_seed, trace_id=None
    ):
        assert slot == 0
        self.calls.append(
            {
                "call_index": len(self.calls),
                "monotonic_s": time.monotonic(),
                "slot": slot,
                "episode_id": episode_id,
                "trace_id": trace_id,
                "noise_seed": noise_seed,
                "instruction": instruction,
                "observation": {key: value.copy() for key, value in observation.items()},
            }
        )
        return np.tile(np.array([0.0] * 6 + [-1.0], dtype=np.float32), (10, 1))


@pytest.mark.skipif(
    not os.getenv("HUD_DROPBEAR_POOLED_IMAGE"), reason="set HUD_DROPBEAR_POOLED_IMAGE"
)
async def test_pooled_cpu_container_reuses_runtime_with_fresh_episode_action_queues(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(settings, "telemetry_local_dir", str(tmp_path / "traces"))
    rows = cohort_tasks(1, episodes_per_lane=2, max_steps=2)
    provider = HoldProvider()
    events = []

    def emit(event, **fields):
        events.append((event, fields))

    placement = DockerRuntime(
        os.environ["HUD_DROPBEAR_POOLED_IMAGE"],
        env_vars={"LIBERO_CONTROL_HZ": str(CONTROL_HZ)},
    )
    job = await Job.start("pooled-container-hold-preflight")
    async with RuntimePool(placement, rows, concurrency=1, emit=emit) as pool:
        runtime_url = pool.lanes[0].runtime.url
        await run_cohort(
            PooledRobotAgent(provider=provider, runtimes=pool, max_steps=2, emit=emit),
            rows,
            runtime=pool,
            job=job,
            concurrency=1,
            rollout_timeout=600,
        )
        assert pool.lanes[0].episodes == 2
        assert pool.lanes[0].current_task is None
    assert pool.cleanup_confirmed

    assert len(job.runs) == 2
    assert [run.slug for run in job.runs] == [row.slug for row in rows]
    infos = []
    for run in job.runs:
        assert not run.trace.is_error, run.trace.error
        assert not run.grade.is_error, run.grade.raw
        assert run.runtime == runtime_url
        assert run.reward == 0  # The simulator independently grades these hold actions.
        info = run.grade.raw["info"]
        assert info["steps"] == 2 and info["termination"] == "action_limit"
        assert info["observation_profile"] == POOLED_PROFILE
        assert info["render_resolution"] == 360 and info["control_hz"] == CONTROL_HZ
        assert info["first_action_unix_s"] is not None
        infos.append(info)
    assert [info["reset_ordinal"] for info in infos] == [1, 2]
    assert len({info["startup_profile"]["bridge_boot"]["process_id"] for info in infos}) == 1
    assert infos[1]["first_action_unix_s"] > infos[0]["first_action_unix_s"]

    # Two actions consume only part of a ten-action chunk. A fresh second call
    # proves the second episode did not inherit the first episode's queued actions.
    assert len(provider.calls) == 2
    assert [call["call_index"] for call in provider.calls] == [0, 1]
    assert provider.calls[1]["monotonic_s"] > provider.calls[0]["monotonic_s"]
    assert [call["noise_seed"] for call in provider.calls] == [0, 1000]
    assert len({call["episode_id"] for call in provider.calls}) == 2
    assert [call["trace_id"] for call in provider.calls] == [run.trace_id for run in job.runs]
    for call in provider.calls:
        observation = call["observation"]
        assert call["instruction"]
        assert set(observation) == {
            *CAMERAS,
            "robot0_eef_pos",
            "robot0_eef_quat",
            "robot0_gripper_qpos",
        }
        for camera in CAMERAS:
            frame = observation[camera]
            assert frame.shape == (360, 360, 3) and frame.dtype == np.uint8
            assert frame.std() > 1  # Real rendering, rather than an empty placeholder.
        assert not np.array_equal(observation[CAMERAS[0]], observation[CAMERAS[1]])
        for key, shape in (
            ("robot0_eef_pos", (3,)),
            ("robot0_eef_quat", (4,)),
            ("robot0_gripper_qpos", (2,)),
        ):
            assert observation[key].shape == shape and np.isfinite(observation[key]).all()
        assert np.isclose(np.linalg.norm(observation["robot0_eef_quat"]), 1, atol=1e-3)
    successful_calls = [fields for event, fields in events if event == "inference"]
    assert [call["inference_index"] for call in successful_calls] == [0, 0]
    assert len([event for event, _ in events if event == "first_action_confirmed"]) == 2
