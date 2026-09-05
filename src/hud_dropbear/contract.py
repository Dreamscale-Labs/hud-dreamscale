"""The explicit simulator-to-provider contract (no model or simulator imports)."""

from copy import deepcopy

import numpy as np

MODEL = "molmoact2-libero"
CHECKPOINT = "allenai/MolmoAct2-LIBERO"
REVISION = "0d24a92bd1faf321ef497c3bbd5681af97c65aa2"
ENV_NAME = "dropbear-libero"
CONTROL_HZ = 10
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
TASK_NAMES = (
    "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_next_to_the_ramekin_and_place_it_on_the_plate",
    "pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate",
)
GOAL_TASK_NAMES = (
    "open_the_middle_drawer_of_the_cabinet",
    "put_the_bowl_on_the_stove",
    "put_the_wine_bottle_on_top_of_the_cabinet",
    "open_the_top_drawer_and_put_the_bowl_inside",
    "put_the_bowl_on_top_of_the_cabinet",
    "push_the_plate_to_the_front_of_the_stove",
    "put_the_cream_cheese_in_the_bowl",
    "turn_on_the_stove",
    "put_the_bowl_on_the_plate",
    "put_the_wine_bottle_on_the_rack",
)
TASK_SUITES = {"libero_spatial": TASK_NAMES, "libero_goal": GOAL_TASK_NAMES}


def build_contract():
    return {
        "robot_type": "libero_franka",
        "control_rate": CONTROL_HZ,
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
