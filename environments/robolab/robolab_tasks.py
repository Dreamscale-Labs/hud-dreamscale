"""The fixed RoboLab DROID task set and per-task horizons (no simulator imports).

``tasks.json`` is the ten-task screen from Dreamscale's Cosmos qualification
(``mvp/infra/benchmarks/cosmos3_tasks.json``). ``episode_s`` is RoboLab's own
``episode_length_s`` for each task class at the pinned revision, so the sim
truncates there by itself; the agent's step cap only adds a small margin.
"""

from __future__ import annotations

import json
import math
from functools import cache
from pathlib import Path
from typing import Any

TASKS_FILE = Path(__file__).with_name("tasks.json")
CONTROL_HZ = 15
# Extra ticks beyond RoboLab's own time limit. The env reports truncation at the
# horizon; this margin only keeps the agent from cutting an episode early.
STEP_MARGIN = 8


@cache
def task_set() -> dict[str, Any]:
    return json.loads(TASKS_FILE.read_text())


def task_names() -> list[str]:
    return [task["task_name"] for task in task_set()["tasks"]]


def task_spec(task_name: str) -> dict[str, Any]:
    for task in task_set()["tasks"]:
        if task["task_name"] == task_name:
            return task
    raise KeyError(f"{task_name!r} is not in {TASKS_FILE.name}; known: {', '.join(task_names())}")


def horizon_steps(task_name: str, *, control_hz: int = CONTROL_HZ) -> int:
    """RoboLab's time limit in control ticks (``episode_length_s`` x 15 Hz)."""
    return math.ceil(float(task_spec(task_name)["episode_s"]) * control_hz - 1e-9)


def max_agent_steps(task_name: str) -> int:
    return horizon_steps(task_name) + STEP_MARGIN


def select_tasks(spec: str) -> list[str]:
    """``all`` or a comma/space separated list of task class names, in file order."""
    known = task_names()
    if spec.strip().lower() == "all":
        return known
    wanted = [name for name in spec.replace(",", " ").split() if name]
    unknown = [name for name in wanted if name not in known]
    if unknown:
        raise ValueError(f"unknown RoboLab task(s) {unknown}; known: {', '.join(known)}")
    if len(set(wanted)) != len(wanted):
        raise ValueError("duplicate task names")
    return wanted
