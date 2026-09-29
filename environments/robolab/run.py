"""Short RoboLab rollout on a local GPU. With HUD credentials, the job lands on hud.ai.

Isaac has to boot on this machine. Modal cannot run Isaac Lab yet.

    OMNI_KIT_ACCEPT_EULA=Y python run.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
from hud.agents.robot import Model, RobotAgent
from hud.eval import LocalRuntime

from env import episode

# A held Franka pose inside the joint limits. Enough to record a trace.
HOLD = np.array([0.0, -0.5, 0.0, -1.5, 0.0, 1.5, 0.7, 0.0], dtype=np.float32)


class HoldModel(Model):
    def infer(self, batch):
        del batch
        return HOLD.reshape(1, 1, -1).copy()


class HoldAgent(RobotAgent):
    max_steps = 4
    log_every = 2

    def __init__(self) -> None:
        self.model = HoldModel()
        self.adapter = None


async def main() -> None:
    job = await episode(task_name="RubiksCubeTask", seed=0).run(
        HoldAgent(),
        runtime=LocalRuntime(Path(__file__).parent / "env.py", ready_timeout=900.0),
    )
    run = job.runs[0]
    print(f"reward={run.reward}", flush=True)
    if getattr(job, "id", None):
        print(f"job=https://hud.ai/jobs/{job.id}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
