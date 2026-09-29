"""RoboDojo on HUD. Isaac owns the sim process; the task reward is the grade."""

from pathlib import Path

from hud import Environment
from sim import make_env

env = Environment(name="dreamscale-robodojo")
sim = env.gym(
    make_env,
    fps=25,
    contract=str(Path(__file__).with_name("contract.json")),
)


@env.template()
async def episode(task: str = "stack_bowls", seed: int = 0):
    """One RoboDojo episode. The task instruction is the prompt."""
    episode = await sim.reset(task=task, seed=seed)
    yield {"prompt": episode["prompt"], "bindings": {"robot": {"token": episode["token"]}}}
    yield await sim.result(token=episode["token"])
