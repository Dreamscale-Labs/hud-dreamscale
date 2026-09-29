"""Short RoboDojo rollout. With HUD credentials, the job lands on hud.ai.

ROBODOJO_ROOT=/path/to/RoboDojo OMNI_KIT_ACCEPT_EULA=YES python run.py
"""

import asyncio
from pathlib import Path

import numpy as np
from hud.agents.robot import Model, RobotAgent
from hud.eval import LocalRuntime

from env import episode

ACTION_DIM = 14  # 6 joints + gripper, per arm


class HoldModel(Model):
    """Hold a zero joint target. Enough to record a trace."""

    def infer(self, batch):
        del batch
        return np.zeros((1, 1, ACTION_DIM), dtype=np.float32)


class HoldAgent(RobotAgent):
    def __init__(self):
        super().__init__()
        self.model = HoldModel()
        self.adapter = None
        self.max_steps = 4
        self.log_every = 2


async def main():
    job = await episode(task="stack_bowls", seed=0).run(
        HoldAgent(),
        runtime=LocalRuntime(Path(__file__).parent / "env.py", ready_timeout=1800),
    )
    print(f"reward={job.reward}")
    if job.id:
        print(f"job=https://hud.ai/jobs/{job.id}")


if __name__ == "__main__":
    asyncio.run(main())
