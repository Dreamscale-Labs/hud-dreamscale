"""RoboLab DROID on HUD. Isaac owns the sim process; RoboLab's own verdict is the grade."""

from pathlib import Path

from hud import Environment

from robolab_tasks import CONTROL_HZ, max_agent_steps, task_names
from sim import RobolabBridge, make_env

# Tasks outside tasks.json still run; cap them at RoboLab's base 10 minute limit.
DEFAULT_MAX_STEPS = 600 * CONTROL_HZ

env = Environment(name="dreamscale-robolab")
sim = env.gym(
    make_env,
    fps=CONTROL_HZ,
    contract=str(Path(__file__).with_name("contract.json")),
    bridge=RobolabBridge,
)


@env.template()
async def episode(
    task_name: str = "BananaInBowlTask",
    episode: int = 0,
    instruction_type: str = "default",
    scene_seed: int = 0,
):
    """One RoboLab episode of ``task_name``; ``episode`` seeds the scene reset.

    Reward is RoboLab's binary success. ``info`` carries RoboLab's subtask
    score, its termination step and the step count the sim executed.
    """
    started = await sim.reset(
        task_name=task_name,
        instruction_type=instruction_type,
        scene_seed=scene_seed,
        seed=episode,
    )
    max_steps = max_agent_steps(task_name) if task_name in task_names() else DEFAULT_MAX_STEPS
    yield {
        "prompt": started["prompt"],
        "bindings": {
            "robot": {
                "token": started["token"],
                "task_name": task_name,
                "episode": episode,
                "max_steps": max_steps,
            }
        },
    }
    result = await sim.result(token=started["token"])
    success = bool(result.get("success"))
    score = result.get("robolab_score")
    yield {
        "score": 1.0 if success else 0.0,
        "content": (
            f"{task_name} episode {episode}: success={success} "
            f"robolab_score={score} steps={result.get('steps')}"
        ),
        "info": {**result, "episode": episode, "scene_seed": scene_seed},
    }
