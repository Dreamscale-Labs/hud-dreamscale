"""The explicit simulator-to-provider contract (no model or simulator imports)."""

import json
from copy import deepcopy
from importlib.resources import files

import numpy as np

MODEL = "molmoact2-libero"
CHECKPOINT = "allenai/MolmoAct2-LIBERO"
REVISION = "0d24a92bd1faf321ef497c3bbd5681af97c65aa2"
ENV_NAME = "dropbear-libero"
# LIBERO reference controller cadence. This is simulation time per action,
# independent of network latency and the published SDK sim profile's default.
CONTROL_HZ = 20
CHUNK_SIZE = 10
MAX_STEPS = 600
CAMERAS = ("agentview_image", "robot0_eye_in_hand_image")
STATE_NAMES = [
    "eef_x",
    "eef_y",
    "eef_z",
    "eef_rx",
    "eef_ry",
    "eef_rz",
    "gripper_qpos_0",
    "gripper_qpos_1",
]
ACTION_NAMES = ["delta_x", "delta_y", "delta_z", "delta_rx", "delta_ry", "delta_rz", "gripper"]
# Generated from hf-libero 0.1.3, task_order_index=0. The simulator independently
# checks every selected name before it starts an episode.
TASK_SUITES = {
    suite: tuple(names)
    for suite, names in json.loads(
        files("hud_dropbear").joinpath("task_manifest.json").read_text()
    ).items()
}
TASK_NAMES = TASK_SUITES["libero_spatial"]
GOAL_TASK_NAMES = TASK_SUITES["libero_goal"]


def build_contract(control_hz=CONTROL_HZ):
    if control_hz not in (10, 20):
        raise ValueError("Supported LIBERO control rates are 10 and 20 Hz")
    return {
        "robot_type": "libero_franka",
        "control_rate": control_hz,
        "features": {
            **{
                key: {
                    "role": "observation",
                    "type": "rgb",
                    "dtype": "uint8",
                    "shape": [256, 256, 3],
                    "orientation": "libero_raw",
                    "camera_role": role,
                }
                for key, role in zip(CAMERAS, ("agent", "wrist"), strict=True)
            },
            "state": {
                "role": "observation",
                "type": "state",
                "dtype": "float32",
                "shape": [8],
                "names": deepcopy(STATE_NAMES),
                "rotation": "axis_angle_radians",
                "position_unit": "meters",
            },
            "action": {
                "role": "action",
                "type": "action",
                "dtype": "float32",
                "shape": [7],
                "names": deepcopy(ACTION_NAMES),
                "control_mode": "libero_osc_pose_delta",
                "gripper_convention": "libero_native",
            },
        },
    }


def finite_array(value, shape, label):
    array = np.asarray(value, dtype=np.float32)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{label} must be a finite array of shape {shape}; got {array.shape}")
    return array


def validate_contact_height(height):
    """A settled 1 cm box must sit on a platform whose surface is at 0.9 m."""
    if not np.isfinite(height) or abs(float(height) - 0.905) > 0.001:
        raise RuntimeError(
            f"Invalid MuJoCo contact physics: box center is {height:.6f} m; "
            "expected 0.905 m within 1 mm. Run the simulator on its native CPU "
            "architecture; x86 Linux under Rosetta failed this check."
        )


def state_from_raw(obs):
    """LIBERO uses XYZW quaternions; match its axis-angle conversion convention."""
    pos = finite_array(obs["robot0_eef_pos"], (3,), "EEF position")
    quat = finite_array(obs["robot0_eef_quat"], (4,), "EEF quaternion")
    if not np.isclose(np.linalg.norm(quat), 1.0, atol=1e-3):
        raise ValueError("EEF quaternion must be normalized")
    w = float(np.clip(quat[3], -1.0, 1.0))
    denominator = np.sqrt(1.0 - w * w)
    rotation = (
        np.zeros(3, dtype=np.float32)
        if denominator < 1e-8
        else quat[:3] * (2 * np.arccos(w) / denominator)
    )
    grip = finite_array(obs["robot0_gripper_qpos"], (2,), "gripper positions")
    return finite_array(np.concatenate([pos, rotation, grip]), (8,), "LIBERO state")
