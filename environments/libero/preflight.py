"""Run inside the simulator image: no GPU, server credentials or inference required."""

import json
import os

import numpy as np
import torch

from environments.libero.bridge import LiberoBridge
from hud_dropbear.contract import CAMERAS, TASK_NAMES


def main():
    assert os.environ["MUJOCO_GL"] == "osmesa"
    assert torch.version.cuda is None, "The simulator must use CPU-only PyTorch"
    bridge = LiberoBridge()
    results = []
    try:
        for task_id in range(3):
            prompt = bridge.reset(task_id=task_id, init_state_id=0, max_steps=2)
            observations, done = bridge.get_observation()
            assert prompt and not done.any()
            for camera in CAMERAS:
                frame = observations[camera]
                assert frame.shape == (1, 256, 256, 3) and frame.dtype == np.uint8
                assert frame.std() > 1, "Rendering produced a blank camera image"
            assert observations["state"].shape == (1, 8)
            assert np.isfinite(observations["state"]).all()
            assert bridge._env.env.control_freq == 10
            for _ in range(2):
                bridge.step(np.array([[0.0] * 6 + [-1.0]], dtype=np.float32))
            _, done = bridge.get_observation()
            result = bridge.result()
            assert done.all() and not result["success"] and result["score"] == 0
            assert result["info"]["termination"] == "action_limit"
            assert result["info"]["task_name"] == TASK_NAMES[task_id]
            results.append(result)
    finally:
        bridge._close_sim()
    print(json.dumps({"cpu_osmesa_preflight": "passed", "results": results}, indent=2))


if __name__ == "__main__":
    main()
