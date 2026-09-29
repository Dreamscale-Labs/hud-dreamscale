"""RoboTwin 2.0 on HUD. The simulator's success check is the grade.

This process is the sim. The robot model runs outside it and drives the robot socket.
"""

from pathlib import Path

from hud import Environment
from sim import make_env

env = Environment(name="dreamscale-robotwin")
sim = env.gym(
    make_env,
    fps=25,
    contract=str(Path(__file__).with_name("contract.json")),
)


@env.template()
async def episode(task: str = "beat_block_hammer", seed: int = 0):
    """One RoboTwin episode. The task instruction is the prompt."""
    episode = await sim.reset(task=task, seed=seed)
    yield {"prompt": episode["prompt"], "bindings": {"robot": {"token": episode["token"]}}}
    yield await sim.result(token=episode["token"])
