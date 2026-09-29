"""Local robot model against the hosted RoboTwin sim.

The Modal sandbox is only the environment. This process owns the policy.
A Dreamscale model replaces ``RandomJointModel``; it does not move into the sandbox.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import numpy as np
from hud.agents.robot import Model, RobotAgent
from hud.eval.runtime import ModalRuntime
from hud.utils.platform import PlatformClient

# This directory is the HUD env module. The uv project lives at the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from env import episode  # noqa: E402

ACTION_DIM = 14  # 7 joints per arm, Aloha joint mode


class RandomJointModel(Model):
    """Uniform joint targets in [-1, 1]. Stand-in for a Dreamscale policy."""

    def infer(self, obs_batch):
        # Wire observation (identity adapter). One tick, batch of one.
        del obs_batch
        return np.random.uniform(-1, 1, size=(1, 1, ACTION_DIM)).astype(np.float32)


class RandomJointAgent(RobotAgent):
    # Smoke budget. A full RoboTwin episode is far longer than this.
    max_steps = 8
    log_every = 4

    def __init__(self):
        # Policy chunk is already an env joint target.
        self.model = RandomJointModel()
        self.adapter = None


async def main():
    hud_config = Path(__file__).resolve().parent / ".hud" / "config.json"
    if not hud_config.is_file():
        raise RuntimeError("deploy the env first so .hud/config.json names the registry")
    registry_id = json.loads(hud_config.read_text())["registryId"]
    platform = PlatformClient.from_settings()
    registry = await platform.aget(f"/registry/{registry_id}")
    build = await platform.aget(f"/builds/{registry['latest_build_id']}/status")
    image = build.get("uri")
    if not isinstance(image, str) or not image.startswith("modal://"):
        raise RuntimeError(f"latest build has no Modal image: {image!r}")
    # GPU profile HUD stored at deploy. Boot that sandbox; the policy stays here.
    runtime_config = dict(build["runtime_config"])
    runtime_config["image"] = image
    runtime = ModalRuntime(
        command=(
            "uv",
            "run",
            "hud",
            "serve",
            "env.py",
            "--host",
            "0.0.0.0",
            "--port",
            "8765",
        ),
        runtime_config=runtime_config,
    )
    print(f"image={image}", flush=True)
    job = await episode(task="beat_block_hammer", seed=0).run(
        RandomJointAgent(),
        runtime=runtime,
    )
    print(f"reward={job.reward}", flush=True)
    if job.id:
        print(f"job=https://hud.ai/jobs/{job.id}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
