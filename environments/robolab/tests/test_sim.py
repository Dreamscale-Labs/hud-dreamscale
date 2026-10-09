"""Sim side: camera/state packing, action application, RoboLab verdict and the bridge grade."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from conftest import JOINTS, LEFT, RIGHT, WRIST, FakeIsaac, isaac_obs

import sim

CONTRACT = Path(sim.__file__).with_name("contract.json")


def test_pack_observation_maps_robolab_cameras_and_state():
    seen = []

    def resize(image, height, width):
        seen.append((image.shape, height, width))
        return image[::2, ::2]

    packed = sim.pack_observation(isaac_obs(gripper=0.4), resize=resize)
    assert list(packed) == ["exterior_1_left", "exterior_2_left", "wrist_left", "state"]
    # Cosmos3Client: over_shoulder_left -> left, over_shoulder_right -> right, wrist_cam.
    assert packed["exterior_1_left"][5, 5, 0] == LEFT
    assert packed["exterior_2_left"][5, 5, 0] == RIGHT
    assert packed["wrist_left"][5, 5, 0] == WRIST
    for key in ("exterior_1_left", "exterior_2_left", "wrist_left"):
        frame = packed[key]
        assert frame.shape == (360, 640, 3) and frame.dtype == np.uint8
        assert frame.flags["C_CONTIGUOUS"]
        # Upright: no flip, the top-left pixel stays top-left.
        assert tuple(frame[0, 0]) == (1, 2, 3)
    assert seen == [((720, 1280, 3), 360, 640)] * 3
    np.testing.assert_allclose(packed["state"], [*JOINTS, 0.4], rtol=1e-6)
    assert packed["state"].dtype == np.float32


def test_pack_observation_rejects_wrong_proprio():
    obs = isaac_obs()
    obs["proprio_obs"]["arm_joint_pos"] = np.zeros((1, 6), np.float32)
    with pytest.raises(ValueError, match="7 joints"):
        sim.pack_observation(obs, resize=lambda image, h, w: image)


def test_contract_matches_packed_features():
    contract = json.loads(CONTRACT.read_text())
    features = contract["features"]
    assert contract["control_rate"] == 15
    images = [k for k, f in features.items() if f.get("type") == "rgb"]
    assert images == list(sim.CAMERAS)
    assert len(features["state"]["names"]) == 8
    assert len(features["action"]["names"]) == 8


def test_env_applies_action_and_ends_when_robolab_freezes(fake_isaac_runtime):
    isaac = FakeIsaac(terminate_at=2, success=True)
    env = sim.RobolabEnv(isaac, "Pick up the banana", task_name="BananaInBowlTask")
    obs, info = env.reset(seed=7)
    assert isaac.resets == [7, 7]  # run_episode's double reset, seeded by the episode
    assert isaac.eval_resets == 1
    assert obs["state"].shape == (8,) and info == {"is_success": False}

    action = np.arange(8, dtype=np.float64)
    _, _, terminated, truncated, info = env.step(action)
    assert (terminated, truncated, info["is_success"]) == (False, False, False)
    sent = isaac.actions[0]
    assert sent.shape == (1, 8) and sent.dtype == np.float32
    np.testing.assert_array_equal(sent[0], action.astype(np.float32))

    _, _, terminated, truncated, info = env.step(action)
    assert (terminated, truncated, info["is_success"]) == (True, False, True)
    report = env.report()
    assert report["success"] is True
    assert report["robolab_episode_step"] == 2
    assert report["steps"] == 2
    assert report["robolab_score"] == 0.5
    assert report["max_episode_length"] == 750
    assert report["task_name"] == "BananaInBowlTask"


def test_time_limit_is_truncation_not_success(fake_isaac_runtime):
    isaac = FakeIsaac(terminate_at=1, success=False)
    env = sim.RobolabEnv(isaac, "Stack the blocks", task_name="BlockStackingOrderAgnosticTask")
    env.reset(seed=0)
    _, _, terminated, truncated, info = env.step(np.zeros(8))
    assert (terminated, truncated, info["is_success"]) == (False, True, False)
    assert env.report()["success"] is False


def test_bridge_grade_carries_robolab_score_and_steps(fake_isaac_runtime):
    built = []

    def factory(task_name="BananaInBowlTask", instruction_type="default", scene_seed=0):
        built.append((task_name, instruction_type, scene_seed))
        return sim.RobolabEnv(FakeIsaac(terminate_at=3), "Pick up the banana", task_name=task_name)

    bridge = sim.RobolabBridge(factory, fps=15, contract=str(CONTRACT))
    assert bridge.step_timeout >= 300
    prompt = bridge.reset(task_name="PickDrillTask", scene_seed=0, seed=4)
    assert prompt == "Pick up the banana"
    assert built == [("PickDrillTask", "default", 0)]
    data, terminated = bridge.get_observation()
    assert data["exterior_1_left"].shape == (1, 360, 640, 3)
    assert data["state"].shape == (1, 8)
    assert not terminated[0]
    for _ in range(3):
        bridge.step(np.zeros((1, 8), np.float32))
    _, terminated = bridge.get_observation()
    assert terminated[0]
    (grade,) = bridge.result_slots()
    assert grade["score"] == 1.0 and grade["success"] is True
    assert grade["robolab_score"] == 0.5
    assert grade["steps"] == 3 and grade["robolab_episode_step"] == 3
    # A different task rebuilds the env (one task per build).
    bridge.reset(task_name="BananaInBowlTask", seed=0)
    assert built[-1] == ("BananaInBowlTask", "default", 0)


def test_episode_report_reason_prefers_success_event():
    report = sim.episode_report(
        env_result={"success": True, "step": 40},
        subtask_info={"status": 1, "info": "done", "completed": 1, "total": 1, "score": 1.0},
        events=[
            {"code": 101, "info": "banana in bowl", "score": 1.0},
            {"code": 301, "info": "drift", "score": 1.0},
        ],
        steps_executed=40,
        max_episode_length=750,
        task_name="BananaInBowlTask",
        instruction="Pick up the banana and place it in the bowl",
    )
    assert report["reason"] == "banana in bowl"
    assert report["num_events"] == 2
    assert report["robolab_score"] == 1.0
