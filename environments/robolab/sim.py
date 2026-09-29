"""RoboLab (Isaac Lab) gym factory for ``env.gym``.

Kit boots inside this factory, on the sim process main thread. Importing this
module does not start the simulator, so the agent can load the task template
without launching Omniverse.
"""

import os

import gymnasium as gym
import numpy as np

_APP = None
_REGISTERED = set()

# Contract feature -> RoboLab camera term (WRIST_LEFT preset).
EXTERIOR_CAM = "over_shoulder_left_camera"
WRIST_CAM = "wrist_cam"


def _boot():
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


def _register(task_name):
    """Register one task against the native DROID joint-position action."""
    import robolab.constants
    from robolab.core.environments.factory import auto_discover_and_create_cfgs, get_envs
    from robolab.core.observations.observation_utils import (
        generate_image_obs_from_cameras,
        generate_obs_cfg,
    )
    from robolab.registrations.droid.camera_presets import WRIST_LEFT
    from robolab.robots.droid import (
        DroidCfg,
        DroidJointPositionActionCfg,
        ProprioceptionObservationCfg,
        WristCameraCfg,
        contact_gripper,
    )
    from robolab.variations.backgrounds import HomeOfficeBackgroundCfg
    from robolab.variations.camera import EgocentricMirroredCameraCfg
    from robolab.variations.lighting import SphereLightCfg

    # Fractional subtask progress is the step reward. Off by default in RoboLab.
    robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = True
    if task_name not in _REGISTERED:
        image_obs = generate_image_obs_from_cameras(WRIST_LEFT)
        viewport = generate_image_obs_from_cameras([EgocentricMirroredCameraCfg])
        obs_cfg = generate_obs_cfg(
            {
                "image_obs": image_obs(),
                "proprio_obs": ProprioceptionObservationCfg(),
                "viewport_cam": viewport(),
            }
        )
        scene_cameras = [cam for cam in WRIST_LEFT if cam is not WristCameraCfg]
        auto_discover_and_create_cfgs(
            task_subdirs=robolab.constants.DEFAULT_TASK_SUBFOLDERS,
            tasks=[task_name],
            observations_cfg=obs_cfg(),
            actions_cfg=DroidJointPositionActionCfg(),
            robot_cfg=DroidCfg,
            camera_cfg=[*scene_cameras, EgocentricMirroredCameraCfg],
            lighting_cfg=SphereLightCfg,
            background_cfg=HomeOfficeBackgroundCfg,
            contact_gripper=contact_gripper,
            dt=1 / 120,
            render_interval=8,
            decimation=8,
            seed=1,
        )
        _REGISTERED.add(task_name)
    return get_envs(task=[task_name])[0]


def _pack(obs):
    """Isaac tensors -> the flat camera + joint layout the contract advertises."""
    from robolab.core.observations.observation_utils import unpack_image_obs, unpack_proprio_obs

    images = unpack_image_obs(obs, env_id=0)
    proprio = unpack_proprio_obs(obs, env_id=0)
    return {
        "exterior_image": np.asarray(images[EXTERIOR_CAM]),
        "wrist_image": np.asarray(images[WRIST_CAM]),
        "state": np.concatenate(
            [
                np.asarray(proprio["arm_joint_pos"], dtype=np.float32).reshape(-1),
                np.asarray(proprio["gripper_pos"], dtype=np.float32).reshape(-1),
            ]
        ),
    }


class RobolabEnv(gym.Env):
    """One RoboLab task as a plain gym env (a batch of one, no ``num_envs``)."""

    metadata = {"render_fps": 15}

    def __init__(self, isaac, instruction):
        self._isaac = isaac
        self.task_description = instruction
        dim = int(isaac.action_manager.total_action_dim)
        self.action_space = gym.spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)
        # Shapes are advisory; the contract is taken from the arrays the sim returns.
        self.observation_space = gym.spaces.Dict(
            {
                "exterior_image": gym.spaces.Box(0, 255, shape=(720, 1280, 3), dtype=np.uint8),
                "wrist_image": gym.spaces.Box(0, 255, shape=(720, 1280, 3), dtype=np.uint8),
                "state": gym.spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32),
            }
        )

    def reset(self, *, seed=None, options=None):
        # A second reset lets the tiled cameras publish a frame from this scene.
        self._isaac.reset(seed=seed)
        obs, _info = self._isaac.reset(seed=seed)
        return _pack(obs), {"is_success": False}

    def step(self, action):
        import torch

        self._wait_until_playing()
        act = torch.as_tensor(np.asarray(action, dtype=np.float32), device=self._isaac.device)
        if act.ndim == 1:
            act = act[None]
        obs, reward, terminated, truncated, _info = self._isaac.step(act)
        # terminated is the success term; truncated is the time limit.
        success = bool(np.asarray(terminated.detach().cpu()).reshape(-1)[0])
        timed_out = bool(np.asarray(truncated.detach().cpu()).reshape(-1)[0])
        step_reward = float(np.asarray(reward.detach().cpu()).reshape(-1)[0])
        return _pack(obs), step_reward, success, timed_out, {"is_success": success}

    def close(self):
        self._isaac.close()

    @staticmethod
    def _wait_until_playing():
        """RoboLab steps only while the Omniverse timeline is playing."""
        import omni.kit.app
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        app = omni.kit.app.get_app()
        while not timeline.is_playing():
            app.update()


def make_env(task_name: str = "RubiksCubeTask", instruction_type: str = "default"):
    """One RoboLab task. ``GymBridge`` rebuilds when the task or instruction changes."""
    _boot()
    from robolab.core.environments.runtime import create_env

    env_name = _register(task_name)
    isaac, cfg = create_env(
        env_name,
        device=os.environ.get("ROBOLAB_DEVICE", "cuda:0"),
        num_envs=1,
        use_fabric=True,
        seed=0,
        instruction_type=instruction_type,
    )
    text = getattr(cfg, "instruction", "")
    if isinstance(text, dict):
        text = text.get(instruction_type) or text.get("default") or next(iter(text.values()), "")
    return RobolabEnv(isaac, str(text))
