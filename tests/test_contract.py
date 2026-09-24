import io

import numpy as np
import pytest
from hud.capabilities.robot import _packb, _unpackb
from PIL import Image

from hud_dreamscale.adapter import LiberoAdapter
from hud_dreamscale.contract import (
    CAMERAS,
    POOLED_PROFILE,
    build_contract,
    pooled_state_from_raw,
    state_from_raw,
)


def observation():
    agent = np.zeros((256, 256, 3), dtype=np.uint8)
    agent[:128, :128] = [255, 0, 0]
    wrist = np.zeros_like(agent)
    wrist[:128, :128] = [0, 255, 0]
    return {
        "data": {CAMERAS[0]: agent, CAMERAS[1]: wrist, "state": np.arange(8, dtype=np.float32)},
        "terminated": False,
    }


def test_wire_and_preprocessing_once():
    obs = observation()
    decoded = _unpackb(_packb(obs))
    adapter = LiberoAdapter()
    features = build_contract()["features"]
    adapter.bind(features["action"], {k: v for k, v in features.items() if k != "action"})
    prepared = adapter.adapt_observation(decoded, "put bowl on plate")
    wire = prepared.observation.to_wire(
        session_id="test", observation_id=0, instruction=prepared.instruction
    )
    assert wire.frame.libero_state == list(range(8))
    images = [np.array(Image.open(io.BytesIO(x))) for x in wire.camera_payloads.values()]
    assert images[0].shape == (224, 224, 3)
    # Exactly one 180-degree rotation: raw top-left colors arrive in bottom-right.
    assert images[0][180, 180, 0] > 240 and images[0][40, 40, 0] < 10
    assert images[1][180, 180, 1] > 240 and images[1][40, 40, 1] < 10


def test_input_digest_detects_camera_state_and_instruction_changes():
    adapter = LiberoAdapter()
    original = adapter.adapt_observation(observation(), "task").input_sha256
    assert original == adapter.adapt_observation(observation(), "task").input_sha256
    for key in (*CAMERAS, "state"):
        obs = observation()
        obs["data"][key].flat[0] += 1
        assert adapter.adapt_observation(obs, "task").input_sha256 != original
    assert adapter.adapt_observation(observation(), "changed task").input_sha256 != original


@pytest.mark.parametrize(
    "field,value", [("orientation", "upright"), ("camera_role", "wrist"), ("dtype", "float32")]
)
def test_contract_rejects_silent_camera_changes(field, value):
    f = build_contract()["features"]
    f[CAMERAS[0]][field] = value
    with pytest.raises(ValueError, match="contract mismatch"):
        LiberoAdapter().bind(f["action"], {k: v for k, v in f.items() if k != "action"})


@pytest.mark.parametrize("bad", [np.zeros((7,)), np.full((8,), np.nan)])
def test_bad_state_fails_before_inference(bad):
    obs = observation()
    obs["data"]["state"] = bad
    with pytest.raises(ValueError):
        LiberoAdapter().adapt_observation(obs, "task")


def test_quaternion_convention_and_units():
    obs = {
        "robot0_eef_pos": [0.1, 0.2, 0.3],
        "robot0_eef_quat": [0, 0, 1, 0],
        "robot0_gripper_qpos": [0.01, -0.01],
    }
    np.testing.assert_allclose(state_from_raw(obs), [0.1, 0.2, 0.3, 0, 0, np.pi, 0.01, -0.01])
    obs["robot0_eef_quat"] = [0, 0, 0, 1]
    np.testing.assert_array_equal(state_from_raw(obs)[3:6], [0, 0, 0])
    obs["robot0_eef_quat"] = [0, 0, 0, 0]
    with pytest.raises(ValueError):
        state_from_raw(obs)


def test_pooled_profile_keeps_raw_quaternion_and_scalar_profile_unchanged():
    raw = {
        "robot0_eef_pos": [0.1, 0.2, 0.3],
        "robot0_eef_quat": [0, 0, 1, 0],
        "robot0_gripper_qpos": [0.01, -0.01],
    }
    pooled = pooled_state_from_raw(raw)
    np.testing.assert_array_equal(pooled["robot0_eef_quat_xyzw"], [0, 0, 1, 0])
    assert "state" not in pooled and all(v.dtype == np.float32 for v in pooled.values())
    scalar = build_contract()
    contract = build_contract(profile=POOLED_PROFILE)
    assert scalar["features"][CAMERAS[0]]["shape"] == [256, 256, 3]
    assert contract["features"][CAMERAS[0]]["shape"] == [360, 360, 3]
    assert "state" in scalar["features"] and "state" not in contract["features"]
    assert contract["features"]["action"] == scalar["features"]["action"]
    with pytest.raises(ValueError, match="20 Hz"):
        build_contract(10, profile=POOLED_PROFILE)
    with pytest.raises(ValueError, match="Unsupported"):
        build_contract(profile="unknown")


@pytest.mark.parametrize("quat", [[0, 0, 0, 0], [0, 0, 0, 2], [0, 0, 0, np.nan]])
def test_pooled_profile_rejects_invalid_geometry(quat):
    with pytest.raises(ValueError):
        pooled_state_from_raw(
            {"robot0_eef_pos": [0] * 3, "robot0_eef_quat": quat, "robot0_gripper_qpos": [0] * 2}
        )
