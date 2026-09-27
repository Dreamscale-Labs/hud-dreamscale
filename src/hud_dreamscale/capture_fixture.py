"""Capture a real HUD observation for release health qualification; apply no actions.

This is a fixture collection trace, excluded from scored evaluation results.
The operator must reserve its simulator cost before invoking this command.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path
from uuid import uuid4

from dreamscale.inference import action_request
from hud import HUDRuntime
from hud.agents.base import Agent
from hud.agents.robot.agent import RobotAgent
from hud.capabilities.robot import RobotClient

from .campaign import task_rows
from .campaign_run import verify_hud_build
from .pooled_agent import PooledLiberoAdapter
from .runtime_pool import RuntimePool
from .telemetry import Evidence


class Capture(Agent):
    def __init__(self, model, output):
        self.model, self.output = model, output
        self.captured = False

    async def __call__(self, run):
        cap = run.client.binding(RobotAgent.robot_protocol)
        token = run.bindings.get(cap.name, {}).get("token")
        robot = await RobotClient.connect(cap, token=token)
        try:
            adapter = PooledLiberoAdapter(self.model)
            adapter.bind(*robot.spaces())
            raw = await robot.get_observation()
            batch = adapter.adapt_observation(raw, run.prompt)
            request = action_request(
                deployment_id="health-fixture",
                robot_slot=0,
                sequence=0,
                request_id=uuid4(),
                episode_id="health-fixture",
                noise_seed=0,
                instruction=batch.instruction,
                observation=batch.observation,
            )
            self.output.write_text(
                json.dumps(
                    {
                        "purpose": "release_health_fixture_not_scored",
                        "model": self.model,
                        "hud_trace_id": run.trace_id,
                        "captured_at": time.time(),
                        "health_request": request,
                        "input_sha256": batch.input_sha256,
                        "actions_applied": 0,
                    },
                    indent=2,
                )
                + "\n"
            )
            self.captured = True
        finally:
            await robot.close()
        run.trace.status = "completed"
        run.trace.content = "Health observation captured; no model inference or actions applied."


async def collect(config, output):
    before = await asyncio.to_thread(verify_hud_build, config)
    output.mkdir(parents=True, exist_ok=False)
    (output / "hud-build.json").write_text(json.dumps(before, indent=2) + "\n")
    task = task_rows(config["model"], concurrency=1, scored=False)[0]
    task = task.model_copy(update={"env": config["hud_environment_name"]})
    evidence = Evidence(output / "events.jsonl")
    pool = RuntimePool(
        HUDRuntime(),
        [task],
        concurrency=1,
        emit=evidence.emit,
        startup_timeout=600,
        lane_ready_attempts=1,
    )
    agent = Capture(config["model"], output / "fixture.json")
    try:
        async with asyncio.timeout(720), pool:
            job = await task.run(agent, runtime=pool, rollout_timeout=120)
            (output / "job.json").write_text(
                json.dumps(
                    {
                        "job_id": job.id,
                        "fixture_captured": agent.captured,
                        "excluded_from_scored_study": True,
                    },
                    indent=2,
                )
                + "\n"
            )
        if not agent.captured:
            raise RuntimeError("HUD fixture capture did not complete")
        after = await asyncio.to_thread(verify_hud_build, config)
        if before != after:
            raise RuntimeError("HUD build changed during fixture capture")
    finally:
        evidence.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(collect(json.loads(args.config.read_text()), args.output))


if __name__ == "__main__":
    main()
