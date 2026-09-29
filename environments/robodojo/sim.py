"""RoboDojo gym factory for ``env.gym``. The task name rebuilds the env.

Point ``ROBODOJO_ROOT`` at a RoboDojo checkout that contains ``env/``, ``task/``,
and ``Assets/``. Kit boots inside this factory, on the sim process main thread.
"""

import importlib
import os
import sys

import gymnasium as gym
import numpy as np

_APP = None


def _joint_control_only():
    """Joint targets do not need cuRobo. End-effector planning does."""
    try:
        import curobo.batch_motion_planner  # noqa: F401

        return False
    except Exception:
        pass
    import types

    modules = {
        "curobo": {},
        "curobo.runtime": {"cuda_graph_reset": True},
        "curobo.batch_motion_planner": {"BatchMotionPlanner": object, "MotionPlannerCfg": object},
        "curobo.inverse_kinematics": {"InverseKinematics": object, "InverseKinematicsCfg": object},
        "curobo.motion_planner": {"MotionPlanner": object},
        "curobo.types": {
            "DeviceCfg": object,
            "GoalToolPose": object,
            "JointState": object,
            "Pose": object,
            "ToolPoseCriteria": object,
        },
    }
    for name, attrs in modules.items():
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        sys.modules[name] = module
    sys.modules["curobo"].runtime = sys.modules["curobo.runtime"]
    return True


def _boot():
    """Start headless Kit once, before any isaaclab import."""
    global _APP
    if _APP is not None:
        return _APP
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaaclab.app import AppLauncher

    # isaaclab.python.kit does not wait for its own frame, so the first camera
    # read is the clear color. These are read at startup and cannot be set later.
    kit_args = " ".join(
        [
            "--/app/updateOrder/checkForHydraRenderComplete=1000",
            "--/app/renderer/waitIdle=true",
            "--/app/hydraEngine/waitIdle=true",
        ]
    )
    _APP = AppLauncher(headless=True, enable_cameras=True, device="cuda:0", kit_args=kit_args).app
    return _APP


def _robodojo_path():
    """Put the checkout first. Kit and this directory's ``env.py`` would otherwise win."""
    root = os.path.abspath(os.environ["ROBODOJO_ROOT"])
    here = os.path.dirname(os.path.abspath(__file__))
    blocked = {root, here}
    sys.path[:] = [item for item in sys.path if os.path.abspath(item or root) not in blocked]
    sys.path.insert(0, root)
    cached = sys.modules.get("env")
    file = str(getattr(cached, "__file__", "") or "")
    if cached is not None and not file.startswith(root + os.sep):
        for name in list(sys.modules):
            if name == "env" or name.startswith("env."):
                del sys.modules[name]


def _build(task):
    """One RoboDojo task, the eval client's config path, without a policy server."""
    app = _boot()
    _robodojo_path()
    skip_planner = _joint_control_only()
    from env.global_configs import BENCHMARK, ENV_CONFIG_PATH, ROOT_DIR
    from omegaconf import OmegaConf
    from utils.load_file import load_yaml
    from utils.pipeline_utils import process_config, process_randomization

    registry = importlib.import_module(f"task.{BENCHMARK}.task_registry")
    task_name, task_cls = registry.load_task_class(task)
    eval_cfg = load_yaml(os.path.join(ENV_CONFIG_PATH, "arx_x5.yml"))
    eval_cfg["task_name"] = task_name
    eval_cfg["num_envs"] = 1

    def section(folder, name):
        return load_yaml(os.path.join(ENV_CONFIG_PATH, folder, f"{name}.yml"))

    chosen = eval_cfg["config"]
    config_dir = os.path.join(ROOT_DIR, "task", BENCHMARK, "config")
    env_cfg = OmegaConf.create(
        {
            "sim": section("sim", chosen["sim"]),
            "scene": section("scene", chosen["scene"]),
            "camera": section("camera", chosen["camera"]),
            "robot": section("robot", chosen["robot"]),
            "task_env": load_yaml(registry.task_config_path(config_dir, task_name)),
            "eval_cfg": eval_cfg,
        }
    )
    OmegaConf.update(env_cfg, "sim.scene.num_envs", 1, force_add=True)
    OmegaConf.update(env_cfg, "sim.device", "cuda:0", force_add=True)
    OmegaConf.update(env_cfg, "sim.use_fabric", True, force_add=True)
    env_cfg.sim.seed = [0]
    env_cfg = process_randomization(env_cfg)
    env_cfg, _eval_num = process_config(env_cfg, task_name=task_name)
    OmegaConf.update(
        env_cfg,
        "camera.default_frequency",
        eval_cfg["observation"].get("collect_freq", 0),
        force_add=True,
    )
    if skip_planner:
        OmegaConf.set_struct(env_cfg.robot, False)
        for robot in env_cfg.robot.robots:
            robot.need_planner = False
    return task_cls(env_cfg, app), eval_cfg["observation"]


def _host(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value


def _lit(image):
    """An unlit RTX frame is gray even when the geometry is already drawn."""
    if image.size == 0:
        return False
    span = np.max(image, axis=-1).astype(np.int16) - np.min(image, axis=-1).astype(np.int16)
    return float(span.mean()) >= 12.0


def _image(array):
    image = np.asarray(array)
    if image.dtype != np.uint8:
        scale = 255.0 if image.size and float(np.max(image)) <= 1.0 else 1.0
        image = np.clip(image * scale, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image[..., :3])


class RobodojoEnv(gym.Env):
    """One RoboDojo task as a plain gym env (a batch of one, no ``num_envs``)."""

    metadata = {"render_fps": 25}

    def __init__(self, task_env, observation_cfg, task):
        self._env = task_env
        self._task = task
        self._observation_cfg = observation_cfg
        self._obs = None
        self._arms = []
        dim = 14  # dual X5: 6 joints + gripper on each arm
        self.action_space = gym.spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict(
            {
                "cam_head": gym.spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8),
                "cam_left_wrist": gym.spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8),
                "cam_right_wrist": gym.spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8),
                "state": gym.spaces.Box(-np.inf, np.inf, shape=(dim,), dtype=np.float32),
            }
        )
        self.task_description = ""
        self._steps = 0

    def reset(self, *, seed=None, options=None):
        del options
        layout_seed = 0 if seed is None else int(seed)
        self._install_layout(layout_seed)
        # The scene was built before a layout existed, so force a reload.
        self._env.scene_manager.setup_scene = False
        self._env.reset(seed=[layout_seed])
        self._env.scene_manager.apply_saved_poses([0])
        if self._obs is None:
            from env.observation_manager.obs_manager import ObsManager

            freq = int(self._observation_cfg.get("collect_freq", 25) or 0)
            # 0.004 * 25 is not an exact binary fraction, so the manager's
            # integer check rejects the official dt. Set the interval after.
            obs_cfg = dict(self._observation_cfg)
            obs_cfg["collect_freq"] = 0
            self._obs = ObsManager(
                obs_config=obs_cfg,
                num_envs=1,
                dt=float(self._env.dt),
                task_name=type(self._env).__name__,
                description_cfg={},
                seeds_per_env=[0 if seed is None else int(seed)],
            )
            if freq:
                self._obs.collect_freq = freq
                self._obs.collect_interval = float(round(1.0 / (float(self._env.dt) * freq)))
            self._obs.initialize(self._env)
            self._arms = [
                robot for robot in self._env.robot_manager.robot_list if robot.type == "target"
            ]
        else:
            self._obs.reset()
        # The eval client uses this as "episode still valid", not as the task grade.
        self._env.success = [True]
        self._poses_as_numpy()
        self._env.reward_manager.init_state()
        if hasattr(self._env, "run_reward"):
            self._env.run_reward()
        if hasattr(self._env, "get_score"):
            self._env.get_score()
        self._steps = 0
        text = self._obs.instruction
        while isinstance(text, (list, tuple)) and text:
            text = text[0]
        self.task_description = str(text or type(self._env).__name__)
        self._warm_cameras()
        return self._observe(), {"is_success": False}

    def step(self, action):
        self._apply(np.asarray(action, dtype=np.float64).reshape(-1))
        self._env.reward_manager.step(env_idx_list=[0])
        reward = float(self._env.reward_manager.get_reward(final_check=False)[0])
        self._steps += 1
        success = reward > 1.0 - 1e-3
        timed_out = self._steps >= int(getattr(self._env, "step_lim", 800))
        return self._observe(), reward, success, timed_out and not success, {"is_success": success}

    def close(self):
        self._env.close()

    def _poses_as_numpy(self):
        """Reward checks concatenate poses with NumPy. Isaac returns CUDA tensors."""
        layout = self._env.scene_manager.layout_manager
        if getattr(layout, "_numpy_poses", False):
            return
        original = layout.get_instance_pose

        def get_instance_pose(*args, **kwargs):
            pos, rot = original(*args, **kwargs)
            return _host(pos), _host(rot)

        layout.get_instance_pose = get_instance_pose
        layout._numpy_poses = True

    def _install_layout(self, layout_seed):
        from env.seed_manager.seed_manager import SeedManager

        seeds = SeedManager(
            {"task_name": self._task, "config_name": "arx_x5", "num_envs": 1, "seed": 0}
        )
        seeds.init_eval()
        layout = seeds.get_seed_scene_info(layout_seed)
        self._env.scene_manager.layout_manager.set_saved_layout(0, layout)

    def _warm_cameras(self):
        """Render until the dome light and materials replace the gray first frame."""
        for _ in range(10):
            self._env.render()
        for _ in range(24):
            self._obs.render_for_capture()
            color = self._obs.get_obs(env_idx_list=[0])[0]["vision"]["cam_head"]["color"]
            if _lit(_image(color)):
                return

    def _observe(self):
        self._obs.render_for_capture()
        raw = self._obs.get_obs(env_idx_list=[0])[0]
        vision = raw.get("vision", {})
        packed = {}
        for name in ("cam_head", "cam_left_wrist", "cam_right_wrist"):
            color = vision.get(name, {}).get("color")
            if color is not None:
                packed[name] = _image(color)
        del raw
        joints = []
        manager = self._env.robot_manager
        for robot in self._arms:
            arm = manager.get_joint(robot, env_idx_list=[0])[0]
            joints.extend(np.asarray(arm, dtype=np.float32).reshape(-1))
            grip = manager.get_end_effector_real_val(robot, env_idx_list=[0])[0]
            opening = float(np.asarray(grip, dtype=np.float64).reshape(-1)[0])
            low, high = robot.gripper_scale
            if robot.gripper_move["sign"] == 1:
                opening = (opening - low) / (high - low)
            else:
                opening = (high - opening) / (high - low)
            joints.append(np.float32(opening))
        packed["state"] = np.asarray(joints, dtype=np.float32)
        return packed

    def _apply(self, action):
        # Absolute joint targets. Gripper is [0, 1], then blended like the eval client.
        target = {}
        cursor = 0
        for robot in self._arms:
            arm_key = self._env.robot_manager.process_name(robot.arm_name)
            width = len(robot.arm_joint_indices)
            position = action[cursor : cursor + width].tolist()
            target[arm_key] = {"position": position, "velocity": [0.0] * width}
            cursor += width
            grip = float(np.clip(action[cursor], 0.0, 1.0))
            cursor += 1
            grip_key = self._env.robot_manager.process_name(robot.gripper_name)
            if robot.ee_type == "gripper":
                low, high = robot.gripper_scale
                if robot.gripper_move["sign"] != 1:
                    grip = 1.0 - grip
                grip = grip * (high - low) + low
                mimic = robot.gripper_move["mimic"]
                position = [grip, grip * mimic[1] + mimic[2]]
                target[grip_key] = {"position": position, "velocity": [0.0] * len(position)}
            else:
                target[grip_key] = {"position": [grip], "velocity": [0.0]}
        steps = max(int(self._obs.collect_interval), 1)
        sequence = self._interpolate(target, steps)
        self._env.robot_manager.control_manager.push([0], [sequence])
        queue = self._env.robot_manager.control_manager.control_queue[0]
        while not queue.is_empty():
            meta = self._env.robot_manager.control_manager.pop([0])
            self._env.robot_manager.control_robot(meta_control_list=meta)
            self._env.sim_step(render=False)

    def _interpolate(self, target, steps):
        frames = []
        for _ in range(steps):
            frame = {
                key: {
                    "position": list(value["position"]),
                    "velocity": list(value["velocity"]),
                }
                for key, value in target.items()
            }
            frames.append(frame)
        blend = int(np.floor(steps * 0.8))
        for robot in self._arms:
            arm_key = self._env.robot_manager.process_name(robot.arm_name)
            current = self._env.robot_manager.get_joint(robot, env_idx_list=[0])[0]
            current = np.asarray(current, dtype=np.float64)
            goal = np.asarray(target[arm_key]["position"], dtype=np.float64)
            for index in range(blend):
                alpha = (index + 1) / (blend + 1)
                frames[index][arm_key]["position"] = ((1 - alpha) * current + alpha * goal).tolist()
            for index in range(blend, steps):
                frames[index][arm_key]["position"] = goal.tolist()
        return frames


def make_env(task: str = "stack_bowls"):
    """One RoboDojo task. ``GymBridge`` rebuilds when ``task`` changes."""
    root = os.environ.get("ROBODOJO_ROOT", "")
    if not root:
        raise RuntimeError("Set ROBODOJO_ROOT to a RoboDojo checkout")
    os.chdir(os.path.abspath(root))
    task_env, observation_cfg = _build(task)
    return RobodojoEnv(task_env, observation_cfg, task)
