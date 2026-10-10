"""Fakes for the Isaac env, the HUD robot socket and the Dreamscale SDK (no GPU, no network)."""

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

LEFT, RIGHT, WRIST = 10, 20, 30
JOINTS = (0.0, -0.5, 0.1, -1.5, 0.2, 1.5, 0.7)


def isaac_obs(*, gripper: float = 0.25, height: int = 720, width: int = 1280) -> dict:
    """An Isaac observation for one env, shaped like RoboLab's DROID registration."""

    def frame(value: int) -> np.ndarray:
        image = np.full((1, height, width, 3), value, dtype=np.uint8)
        image[0, 0, 0] = (1, 2, 3)  # top-left marker survives a 2x stride
        return image

    return {
        "image_obs": {
            "over_shoulder_left_camera": frame(LEFT),
            "over_shoulder_right_camera": frame(RIGHT),
            "head_camera": frame(99),
            "wrist_cam": frame(WRIST),
        },
        "proprio_obs": {
            "arm_joint_pos": np.asarray([JOINTS], dtype=np.float32),
            "gripper_pos": np.asarray([[gripper]], dtype=np.float32),
            "ee_pos": np.zeros((1, 3), dtype=np.float32),
        },
        "viewport_cam": {"egocentric_mirrored_camera": np.zeros((1, 48, 86, 3), np.uint8)},
    }


class FakeIsaac:
    """Mimics ``robolab.core.environments.env.RobolabEnv`` freezing semantics."""

    device = "cpu"

    def __init__(self, *, terminate_at: int = 3, success: bool = True) -> None:
        self.action_manager = SimpleNamespace(total_action_dim=8)
        self.max_episode_length = 750
        self.terminate_at = terminate_at
        self.success = success
        self.actions: list[np.ndarray] = []
        self.resets: list[int | None] = []
        self.eval_resets = 0
        self.all_terminated = False
        self.closed = False

    def reset_eval_state(self) -> None:
        self.eval_resets += 1
        self.all_terminated = False

    def reset(self, seed=None):
        self.resets.append(seed)
        self.actions.clear()
        return isaac_obs(), {}

    def step(self, action):
        self.actions.append(np.asarray(action))
        done = len(self.actions) >= self.terminate_at
        self.all_terminated = done
        term = np.asarray([done and self.success])
        trunc = np.asarray([done and not self.success])
        return isaac_obs(gripper=0.75), np.asarray([0.5]), term, trunc, {}

    def get_env_results(self):
        return [
            {
                "env_id": 0,
                "success": self.success if self.all_terminated else None,
                "step": len(self.actions) if self.all_terminated else None,
            }
        ]

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_isaac_runtime(monkeypatch):
    """Patch the Isaac-only seams of ``sim``: torch, Kit timeline, RoboLab helpers."""
    import sim

    torch = types.ModuleType("torch")
    torch.as_tensor = lambda data, device=None: np.asarray(data)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setattr(sim, "_robolab_resize", lambda image, h, w: image[::2, ::2])
    monkeypatch.setattr(sim.RobolabEnv, "_wait_until_playing", staticmethod(lambda: None))
    infos = iter(
        [{"status": 0, "info": "", "completed": 0, "total": 1, "score": 0.0}]
        + [{"status": 1, "info": "picked", "completed": 1, "total": 2, "score": 0.5}] * 1000
    )
    monkeypatch.setattr(sim.RobolabEnv, "_current_subtask_info", lambda self: next(infos))
    return sim
