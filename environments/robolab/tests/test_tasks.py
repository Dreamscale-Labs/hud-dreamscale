"""Task set, horizons and the HUD template's parameters and grade."""

from __future__ import annotations

import pytest

import robolab_tasks


def test_ten_task_screen_with_robolab_horizons():
    names = robolab_tasks.task_names()
    assert len(names) == 10
    assert names[0] == "BananaInBowlTask" and names[-1] == "BlackItemsInBinTask"
    assert robolab_tasks.task_set()["robolab_revision"].startswith("ad45d4f")
    assert robolab_tasks.horizon_steps("BananaInBowlTask") == 750
    assert robolab_tasks.horizon_steps("BowlStackingLeftOnRightTask") == 300
    assert robolab_tasks.horizon_steps("BlackItemsInBinTask") == 1800
    assert robolab_tasks.max_agent_steps("BananaInBowlTask") > 750


def test_select_tasks():
    assert robolab_tasks.select_tasks("all") == robolab_tasks.task_names()
    assert robolab_tasks.select_tasks("PickDrillTask, BananaInBowlTask") == [
        "PickDrillTask",
        "BananaInBowlTask",
    ]
    with pytest.raises(ValueError, match="unknown"):
        robolab_tasks.select_tasks("RubiksCubeTask")
    with pytest.raises(ValueError, match="duplicate"):
        robolab_tasks.select_tasks("PickDrillTask,PickDrillTask")


class FakeEndpoint:
    def __init__(self, result):
        self.calls = []
        self._result = result

    async def reset(self, **kwargs):
        self.calls.append(("reset", kwargs))
        return {"prompt": "Pick up the cordless drill.", "token": "slot-0"}

    async def result(self, **kwargs):
        self.calls.append(("result", kwargs))
        return self._result


async def test_template_parameterizes_reset_and_grades_with_robolab(monkeypatch):
    import env as env_module

    fake = FakeEndpoint(
        {
            "score": 1.0,
            "success": True,
            "total_reward": 3.0,
            "robolab_score": 1.0,
            "steps": 212,
            "robolab_episode_step": 212,
        }
    )
    monkeypatch.setattr(env_module, "sim", fake)
    gen = env_module.episode.func(task_name="PickDrillTask", episode=3)
    start = await gen.__anext__()
    assert fake.calls[0] == (
        "reset",
        {"task_name": "PickDrillTask", "instruction_type": "default", "scene_seed": 0, "seed": 3},
    )
    assert start["prompt"] == "Pick up the cordless drill."
    robot = start["bindings"]["robot"]
    assert robot["token"] == "slot-0"
    assert robot["task_name"] == "PickDrillTask" and robot["episode"] == 3
    assert robot["max_steps"] == robolab_tasks.max_agent_steps("PickDrillTask")
    grade = await gen.asend(None)
    assert fake.calls[1] == ("result", {"token": "slot-0"})
    assert grade["score"] == 1.0
    assert grade["info"]["robolab_score"] == 1.0
    assert grade["info"]["steps"] == 212 and grade["info"]["episode"] == 3


async def test_template_failure_scores_zero_and_unknown_task_gets_default_cap(monkeypatch):
    import env as env_module

    fake = FakeEndpoint({"score": 0.0, "success": False, "robolab_score": 0.33, "steps": 900})
    monkeypatch.setattr(env_module, "sim", fake)
    gen = env_module.episode.func(task_name="SomeOtherRoboLabTask", episode=0)
    start = await gen.__anext__()
    assert start["bindings"]["robot"]["max_steps"] == env_module.DEFAULT_MAX_STEPS
    grade = await gen.asend(None)
    assert grade["score"] == 0.0 and grade["info"]["robolab_score"] == 0.33
