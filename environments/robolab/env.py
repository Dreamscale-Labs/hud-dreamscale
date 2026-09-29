"""RoboLab on HUD. Isaac owns the sim process; the success term is the grade."""

from pathlib import Path

from hud import Environment
from sim import make_env

env = Environment(name="dreamscale-robolab")
sim = env.gym(
    make_env,
    fps=15,
    contract=str(Path(__file__).with_name("contract.json")),
)


@env.template()
async def episode(
    task_name: str = "RubiksCubeTask",
    seed: int = 0,
    instruction_type: str = "default",
):
    """One RoboLab episode. The task instruction is the prompt."""
    episode = await sim.reset(task_name=task_name, seed=seed, instruction_type=instruction_type)
    yield {"prompt": episode["prompt"], "bindings": {"robot": {"token": episode["token"]}}}
    yield await sim.result(token=episode["token"])
