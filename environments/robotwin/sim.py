"""RoboTwin 2.0 gym factory for ``env.gym``. The task name rebuilds the env.

LeRobot imports ``envs.<task>`` from a RoboTwin checkout. Point ``ROBOTWIN_ROOT``
at that checkout (the directory that contains ``envs/`` and ``assets/``). Task
code opens assets relative to the process cwd.
"""

import os
import sys


def allow_joint_control_without_curobo():
    """Joint steps use mplib. RoboTwin still imports cuRobo while building the robot.

    A real cuRobo is a CUDA extension and is what end-effector planning calls.
    When it is not installed, register the symbols that import needs so joint
    control can start.
    """
    import types

    try:
        import curobo  # noqa: F401
    except ModuleNotFoundError:
        pass
    else:
        return

    def mod(name):
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
        return module

    curobo = mod("curobo")
    types_mod = mod("curobo.types")
    math = mod("curobo.types.math")
    robot = mod("curobo.types.robot")
    wrap = mod("curobo.wrap")
    reacher = mod("curobo.wrap.reacher")
    motion = mod("curobo.wrap.reacher.motion_gen")
    util = mod("curobo.util")

    class _MotionGenConfig:
        @staticmethod
        def load_from_robot_config(*_args, **_kwargs):
            return None

    class _MotionGen:
        def __init__(self, _config):
            pass

        def warmup(self, batch=None):
            pass

    class _Logger:
        @staticmethod
        def setup_logger(**_kwargs):
            pass

    math.Pose = type("Pose", (), {})
    robot.JointState = type("JointState", (), {})
    motion.MotionGen = _MotionGen
    motion.MotionGenConfig = _MotionGenConfig
    motion.MotionGenPlanConfig = type("MotionGenPlanConfig", (), {})
    motion.PoseCostMetric = type("PoseCostMetric", (), {})
    util.logger = _Logger()
    curobo.types = types_mod
    curobo.wrap = wrap
    curobo.util = util
    types_mod.math = math
    types_mod.robot = robot
    wrap.reacher = reacher
    reacher.motion_gen = motion


def make_env(task: str = "beat_block_hammer"):
    """One RoboTwin task. ``GymBridge`` rebuilds when ``task`` changes."""
    root = os.environ.get("ROBOTWIN_ROOT", "")
    if root:
        # Task code opens ``./assets/...`` relative to the process cwd.
        os.chdir(root)
        if root not in sys.path:
            sys.path.insert(0, root)
    allow_joint_control_without_curobo()
    from lerobot.envs.configs import RoboTwinEnvConfig
    from lerobot.envs.factory import make_env as lerobot_make_env

    # Joint-space Aloha (14-d). Cameras stay at RoboTwin's D435 size.
    suites = lerobot_make_env(
        RoboTwinEnvConfig(task=task, action_mode="joint"),
        n_envs=1,
    )
    suite = next(iter(suites))
    task_id = next(iter(suites[suite]))
    return suites[suite][task_id].envs[0]
