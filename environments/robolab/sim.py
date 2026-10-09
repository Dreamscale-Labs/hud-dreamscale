"""RoboLab (Isaac Lab) DROID gym factory and bridge for ``env.gym``.

The scene, cameras, observation preprocessing and scoring follow RoboLab's own
Cosmos DROID evaluation at the pinned revision (``policies/cosmos3/run.py`` and
``policies/cosmos3/client.py``), which is also what Dreamscale's
``integrations/cosmos3-robolab`` client sends to the policy:

- the task is registered with ``auto_register_droid_envs(cameras=WRIST_LEFT_RIGHT_HEAD)``;
- the two over-shoulder views and the wrist view are published as
  ``exterior_1_left`` / ``exterior_2_left`` / ``wrist_left``, each passed
  through RoboLab's ``resize_with_pad`` to 640x360 exactly as ``Cosmos3Client``
  does before packing a request (RoboLab renders 1280x720, so this is a plain
  2x downscale, no padding);
- state is the 7 Franka joint positions (rad) plus RoboLab's gripper closed
  fraction in [0, 1];
- an episode ends when RoboLab freezes the env (success term or its own time
  limit); success and the termination step come from ``env.get_env_results()``
  and the score from RoboLab's subtask state machine, as in ``robolab.eval``.

Kit boots inside the factory, on the sim process main thread. Importing this
module does not start the simulator, so the env server and the tests can load
it without Omniverse.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from hud.environment.robot.gym import GymBridge

_APP = None
_REGISTERED: set[str] = set()

#: Contract feature -> RoboLab camera term (WRIST_LEFT_RIGHT_HEAD preset). The
#: feature names are the DROID policy slots Dreamscale maps them to.
CAMERAS = {
    "exterior_1_left": "over_shoulder_left_camera",
    "exterior_2_left": "over_shoulder_right_camera",
    "wrist_left": "wrist_cam",
}
#: ``Cosmos3Client.IMAGE_H`` / ``IMAGE_W``.
IMAGE_H, IMAGE_W = 360, 640
STATE_DIM = 8
POLICY_LABEL = "dreamscale_hud"

Resize = Callable[[np.ndarray, int, int], np.ndarray]


def _boot() -> None:
    """Start headless Kit once, before any isaaclab or robolab import."""
    global _APP
    if _APP is not None:
        return
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "Y")
    import cv2  # noqa: F401  RoboLab requires cv2 before isaaclab
    from isaaclab.app import AppLauncher

    _APP = AppLauncher(
        headless=True,
        enable_cameras=True,
        device=os.environ.get("ROBOLAB_DEVICE", "cuda:0"),
    )


def _register(task_name: str) -> str:
    """Register one task exactly as RoboLab's Cosmos runner and Dreamscale's client do."""
    import robolab.constants
    from robolab.core.environments.factory import get_envs
    from robolab.registrations.droid.auto_env_registrations_jointpos import (
        auto_register_droid_envs,
    )
    from robolab.registrations.droid.camera_presets import WRIST_LEFT_RIGHT_HEAD

    # The pinned default, set explicitly: the score comes from subtask tracking.
    robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = True
    if task_name not in _REGISTERED:
        auto_register_droid_envs(task=[task_name], cameras=WRIST_LEFT_RIGHT_HEAD)
        _REGISTERED.add(task_name)
    return get_envs(task=[task_name])[0]


def _robolab_resize(image: np.ndarray, height: int, width: int) -> np.ndarray:
    from robolab.core.utils.image_utils import resize_with_pad

    return resize_with_pad(image, height, width)


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def pack_observation(raw_obs: dict[str, Any], *, resize: Resize | None = None) -> dict[str, Any]:
    """Isaac observation (batch of one) -> the three cameras and 8-d state.

    Mirrors ``Cosmos3Client._extract_observation`` for env 0: the same camera
    terms, the same ``resize_with_pad`` to 640x360, and the same proprio terms.
    """
    resize = resize or _robolab_resize
    images = raw_obs["image_obs"]
    proprio = raw_obs["proprio_obs"]
    out: dict[str, Any] = {}
    for feature, camera in CAMERAS.items():
        frame = _to_numpy(images[camera][0])
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"{camera} must be HxWx3, got {frame.shape}")
        frame = resize(np.ascontiguousarray(frame.astype(np.uint8, copy=False)), IMAGE_H, IMAGE_W)
        out[feature] = np.ascontiguousarray(frame, dtype=np.uint8)
    joints = _to_numpy(proprio["arm_joint_pos"][0]).astype(np.float32).reshape(-1)
    gripper = _to_numpy(proprio["gripper_pos"][0]).astype(np.float32).reshape(-1)
    if joints.size != 7 or gripper.size != 1:
        raise ValueError(f"expected 7 joints + 1 gripper, got {joints.size} + {gripper.size}")
    out["state"] = np.concatenate([joints, gripper])
    return out


def episode_report(
    *,
    env_result: dict[str, Any] | None,
    subtask_info: dict[str, Any] | None,
    events: list[dict[str, Any]] | None,
    steps_executed: int,
    max_episode_length: int | None,
    task_name: str,
    instruction: str,
) -> dict[str, Any]:
    """RoboLab's per-episode verdict, shaped like ``robolab.eval.summarize`` fields."""
    env_result = env_result or {}
    success = bool(env_result.get("success"))
    score = None if subtask_info is None else subtask_info.get("score")
    events = list(events or [])
    reason = None
    if events:
        reason = events[-1].get("info")
        if success:
            reason = next(
                (
                    e.get("info")
                    for e in reversed(events)
                    if e.get("info") and e.get("code", 0) < 200
                ),
                reason,
            )
    return {
        "task_name": task_name,
        "instruction": instruction,
        "success": success,
        "terminated_by_robolab": env_result.get("success") is not None,
        "robolab_score": None if score is None else float(score),
        "robolab_episode_step": env_result.get("step"),
        "steps": int(steps_executed),
        "max_episode_length": max_episode_length,
        "reason": reason,
        "subtask": None
        if subtask_info is None
        else {k: subtask_info.get(k) for k in ("status", "info", "completed", "total", "score")},
        "num_events": len(events),
    }


class RobolabEnv(gym.Env):
    """One RoboLab task as a plain gym env (a batch of one, no ``num_envs``)."""

    metadata = {"render_fps": 15}

    def __init__(self, isaac: Any, instruction: str, task_name: str = "") -> None:
        self._isaac = isaac
        self.task_name = task_name
        self.task_description = instruction
        dim = int(isaac.action_manager.total_action_dim)
        self.action_space = gym.spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)
        # Shapes are advisory; the contract is taken from contract.json.
        image = gym.spaces.Box(0, 255, shape=(IMAGE_H, IMAGE_W, 3), dtype=np.uint8)
        self.observation_space = gym.spaces.Dict(
            {
                **{name: image for name in CAMERAS},
                "state": gym.spaces.Box(-np.inf, np.inf, shape=(STATE_DIM,), dtype=np.float32),
            }
        )
        self._steps = 0
        self._subtask_info: dict[str, Any] | None = None
        self._env_result: dict[str, Any] | None = None
        # Sim-side wall time per step (Isaac physics + render, then packing).
        self._step_ms: list[float] = []
        self._pack_ms: list[float] = []

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        del options
        reset_eval_state = getattr(self._isaac, "reset_eval_state", None)
        if reset_eval_state is not None:
            reset_eval_state()
        # robolab.eval.episode.run_episode resets twice so the tiled cameras
        # publish a frame from this scene; do the same.
        self._isaac.reset(seed=seed)
        obs, _info = self._isaac.reset(seed=seed)
        self._steps = 0
        self._subtask_info = None
        self._env_result = None
        self._step_ms.clear()
        self._pack_ms.clear()
        return pack_observation(obs), {"is_success": False}

    def step(self, action):
        import torch

        self._wait_until_playing()
        act = torch.as_tensor(np.asarray(action, dtype=np.float32), device=self._isaac.device)
        if act.ndim == 1:
            act = act[None]
        started = time.perf_counter()
        obs, reward, _terminated, _truncated, _info = self._isaac.step(act)
        self._step_ms.append((time.perf_counter() - started) * 1000.0)
        self._steps += 1
        self._subtask_info = self._current_subtask_info()
        # RoboLab freezes an env when it terminates (success term or time limit)
        # and records success/step itself; a <=2-step physics artifact is reset
        # instead of frozen. End the HUD episode exactly when RoboLab would.
        done = bool(getattr(self._isaac, "all_terminated", False))
        success = False
        if done:
            self._env_result = self._isaac.get_env_results()[0]
            success = bool(self._env_result.get("success"))
        step_reward = float(np.asarray(_to_numpy(reward)).reshape(-1)[0])
        info = {"is_success": success}
        started = time.perf_counter()
        packed = pack_observation(obs)
        self._pack_ms.append((time.perf_counter() - started) * 1000.0)
        return packed, step_reward, done and success, done and not success, info

    def report(self) -> dict[str, Any]:
        report = self._verdict()
        report["sim_compute_ms"] = {
            "isaac_step": timing_stats(self._step_ms),
            "pack_observation": timing_stats(self._pack_ms),
        }
        return report

    def _verdict(self) -> dict[str, Any]:
        events = None
        try:
            from robolab.core.logging.results import get_all_env_events

            per_env = get_all_env_events(self._isaac)
            events = per_env[0] if per_env else None
        except Exception:
            events = None
        return episode_report(
            env_result=self._env_result,
            subtask_info=self._subtask_info,
            events=events,
            steps_executed=self._steps,
            max_episode_length=_int_or_none(getattr(self._isaac, "max_episode_length", None)),
            task_name=self.task_name,
            instruction=self.task_description,
        )

    def close(self):
        self._isaac.close()

    def _current_subtask_info(self) -> dict[str, Any] | None:
        from robolab.core.logging.results import get_all_env_subtask_infos

        infos = get_all_env_subtask_infos(self._isaac)
        return dict(infos[0]) if infos else None

    @staticmethod
    def _wait_until_playing():
        """RoboLab steps only while the Omniverse timeline is playing."""
        import omni.kit.app
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        app = omni.kit.app.get_app()
        while not timeline.is_playing():
            app.update()


def timing_stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"n": 0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
    }


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class RobolabBridge(GymBridge):
    """``GymBridge`` that grades with RoboLab's own verdict and tolerates slow policies.

    The default barrier fails a slot that is silent for 30 s. A cloud policy's
    first chunk (and any network stall) can take longer, and the sim must wait
    for actions rather than drop the episode, so allow five minutes.
    """

    step_timeout = 300.0

    def result_slots(self) -> list[dict[str, Any]]:
        slots = super().result_slots()
        report = getattr(self._unwrapped, "report", None)
        if report is not None and slots:
            slots[0].update(report())
        return slots


def make_env(
    task_name: str = "BananaInBowlTask",
    instruction_type: str = "default",
    scene_seed: int = 0,
):
    """One RoboLab task. ``GymBridge`` rebuilds when any of these args change."""
    _boot()
    from robolab.core.environments.runtime import create_env

    env_name = _register(task_name)
    isaac, cfg = create_env(
        env_name,
        device=os.environ.get("ROBOLAB_DEVICE", "cuda:0"),
        num_envs=1,
        use_fabric=True,
        # Dreamscale's RoboLab runner binds create_env's seed to --scene-seed (0).
        seed=int(scene_seed),
        instruction_type=instruction_type,
        policy=POLICY_LABEL,
    )
    text = getattr(cfg, "instruction", "")
    if isinstance(text, dict):
        text = text.get(instruction_type) or text.get("default") or next(iter(text.values()), "")
    return RobolabEnv(isaac, str(text), task_name=task_name)
