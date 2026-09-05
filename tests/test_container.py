"""Opt-in real CPU container test; no inference credentials or GPU required."""

import os

import numpy as np
import pytest
from hud import DockerRuntime, Taskset
from hud.agents.robot.agent import RobotAgent
from hud.agents.robot.model import Model

from hud_dropbear.adapter import LiberoAdapter
from hud_dropbear.cli import tasks


class HoldModel(Model):
    def infer(self, batch):
        raise NotImplementedError

    async def ainfer(self, batch):
        return np.tile(np.array([0.0] * 6 + [-1.0], dtype=np.float32), (10, 1))


@pytest.mark.skipif(not os.getenv("HUD_DROPBEAR_IMAGE"), reason="set HUD_DROPBEAR_IMAGE")
async def test_cpu_container_claim_render_action_and_grade():
    agent = RobotAgent()
    agent.adapter = LiberoAdapter()
    agent.model = HoldModel()
    job = await Taskset("container-preflight", tasks([0], [0], max_steps=2)).run(
        agent, runtime=DockerRuntime(os.environ["HUD_DROPBEAR_IMAGE"]), rollout_timeout=600
    )
    assert len(job.runs) == 1
    run = job.runs[0]
    assert not run.trace.is_error, run.trace.error
    assert not run.grade.is_error
    assert run.reward == 0
    assert run.grade.raw["info"]["steps"] == 2
    assert run.grade.raw["info"]["termination"] == "action_limit"
