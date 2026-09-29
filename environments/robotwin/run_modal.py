"""Local agent, RoboTwin simulator on a Modal L4.

A Modal token is the only credential. Put it in `.env` next to this file:

    MODAL_TOKEN_ID=ak-...
    MODAL_TOKEN_SECRET=as-...

or paste it into the two constants below (those override `.env`). `HUD_API_KEY`
in the same file records the trace on hud.ai. From the repo root:

    uv run python environments/robotwin/run_modal.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from hud.agents.robot import Model, RobotAgent
from hud.eval import ModalRuntime, Task

# Paste a token here to override `.env` and the environment.
MODAL_TOKEN_ID = ""
MODAL_TOKEN_SECRET = ""

# Image from `hud deploy --runtime modal`. Blank builds Dockerfile.hud instead.
IMAGE = "modal://im-pTVoreTsL9wBSuTHakOOaF"

HERE = Path(__file__).parent
ACTION_DIM = 14  # 7 joints per arm

load_dotenv(HERE / ".env")
if MODAL_TOKEN_ID:
    os.environ["MODAL_TOKEN_ID"] = MODAL_TOKEN_ID
if MODAL_TOKEN_SECRET:
    os.environ["MODAL_TOKEN_SECRET"] = MODAL_TOKEN_SECRET


class HoldModel(Model):
    """Stand-in policy: hold a zero joint target so the socket stays exercised."""

    def infer(self, batch):
        del batch
        return np.zeros((1, 1, ACTION_DIM), dtype=np.float32)


class HoldAgent(RobotAgent):
    max_steps = 8
    log_every = 4

    def __init__(self) -> None:
        self.model = HoldModel()
        self.adapter = None


def make_runtime() -> ModalRuntime:
    config = json.loads((HERE / "runtime.json").read_text())
    image = None
    if IMAGE:
        config["image"] = IMAGE
    else:
        import modal

        image = modal.Image.from_dockerfile(HERE / "Dockerfile.hud", context_dir=HERE)
    return ModalRuntime(
        image=image,
        # Matches Dockerfile.hud. Modal replaces the image CMD, so name it here.
        command=["uv", "run", "hud", "serve", "env.py", "--host", "0.0.0.0", "--port", "8765"],
        port=8765,
        app_name="hud-dreamscale-robotwin",
        runtime_config=config,
    )


async def main() -> None:
    task = Task(
        env="dreamscale-robotwin",
        id="episode",
        args={"task": "beat_block_hammer", "seed": 0},
    )
    print("[modal] launching RoboTwin sandbox", flush=True)
    # Must exceed runtime.json run_timeout_s (1800). The hold episode is much shorter.
    job = await task.run(HoldAgent(), runtime=make_runtime(), rollout_timeout=2400)
    run = job.runs[0]
    print(f"reward={run.reward}", flush=True)
    print(f"status={run.trace.status}", flush=True)
    if getattr(job, "id", None):
        print(f"job=https://hud.ai/jobs/{job.id}", flush=True)


if __name__ == "__main__":
    import asyncio

    import modal

    with modal.enable_output():
        asyncio.run(main())
