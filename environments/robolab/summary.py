"""Wave summary: join HUD grades, agent evidence and sandbox timings per episode."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from dreamscale_agent import latency_stats

HUD_WEB = "https://hud.ai"

#: Rough Modal list prices for the runtime.json shape (1x L40S, 8 CPU, 48 GiB).
#: Only for an order-of-magnitude estimate; the Modal dashboard is the bill.
MODAL_USD_PER_HOUR = {"gpu_l40s": 1.95, "cpu_core": 0.0473, "memory_gib": 0.0080}


def modal_usd_per_sandbox_hour(cpu: float = 8, memory_gib: float = 48) -> float:
    rate = MODAL_USD_PER_HOUR
    return rate["gpu_l40s"] + cpu * rate["cpu_core"] + memory_gib * rate["memory_gib"]


def trace_url(trace_id: str | None) -> str | None:
    return f"{HUD_WEB}/trace/{trace_id}" if trace_id else None


def job_url(job_id: str | None) -> str | None:
    return f"{HUD_WEB}/jobs/{job_id}" if job_id else None


def read_inference_rows(timing_file: Path) -> list[dict[str, Any]]:
    if not timing_file.exists():
        return []
    rows = []
    for line in timing_file.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            if row.get("event") == "inference":
                rows.append(row)
    return rows


def episode_record(
    *,
    task_name: str,
    episode: int,
    run: dict[str, Any],
    agent: dict[str, Any] | None,
    sandbox: dict[str, Any] | None,
) -> dict[str, Any]:
    """One row of the summary. ``run`` is a plain view of the HUD ``Run``."""
    agent = agent or {}
    sandbox = sandbox or {}
    info = run.get("grade_info") or {}
    reward = run.get("reward")
    graded = bool(info) and not run.get("grade_is_error")
    success = bool(info.get("success")) if graded else None
    session = agent.get("session") or {}
    record: dict[str, Any] = {
        "task_name": task_name,
        "episode": episode,
        "trace_id": run.get("trace_id"),
        "trace_url": trace_url(run.get("trace_id")),
        "graded": graded,
        "success": success,
        "reward": reward,
        "robolab_score": info.get("robolab_score"),
        "steps": info.get("steps", agent.get("steps")),
        "robolab_episode_step": info.get("robolab_episode_step"),
        "max_episode_length": info.get("max_episode_length"),
        "reason": info.get("reason"),
        "trace_status": run.get("trace_status"),
        "error": run.get("error") or agent.get("error"),
        "session_id": session.get("session_id"),
        "transport": session.get("transport"),
        "requests": agent.get("requests"),
        "sdk_rtt_ms": agent.get("sdk_rtt_ms"),
        "server_inference_ms": agent.get("server_inference_ms"),
        "sim_step_ms": agent.get("sim_step_ms"),
        "timing": {
            "sandbox_boot_s": _delta(sandbox.get("requested_unix"), sandbox.get("ready_unix")),
            "sandbox_ready_to_agent_s": _delta(
                sandbox.get("ready_unix"), agent.get("agent_called_unix")
            ),
            "session_connect_s": agent.get("session_connect_s"),
            "agent_to_first_observation_s": agent.get("first_observation_s"),
            "episode_loop_s": agent.get("loop_wall_s"),
            "agent_episode_wall_s": agent.get("episode_wall_s"),
            "sandbox_lifetime_s": _delta(
                sandbox.get("requested_unix"), sandbox.get("released_unix")
            ),
        },
        "sandbox_id": sandbox.get("instance_id"),
        "local_video": agent.get("local_video"),
        "timing_file": agent.get("timing_file"),
        "observation_shapes": agent.get("observation_shapes"),
    }
    return record


def _delta(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return float(end) - float(start)


def _rate(records: list[dict[str, Any]]) -> dict[str, Any]:
    graded = [r for r in records if r["graded"]]
    successes = sum(1 for r in graded if r["success"])
    scores = [
        1.0 if r["success"] else float(r["robolab_score"])
        for r in graded
        if r["success"] or r["robolab_score"] is not None
    ]
    return {
        "episodes": len(records),
        "graded": len(graded),
        "errors": len(records) - len(graded),
        "successes": successes,
        "success_rate": successes / len(graded) if graded else None,
        # RoboLab's get_avg_score convention: a success counts as 1.0.
        "mean_score": sum(scores) / len(scores) if scores else None,
    }


def aggregate(
    records: list[dict[str, Any]],
    *,
    inference_rows: Iterable[dict[str, Any]],
    meta: dict[str, Any],
) -> dict[str, Any]:
    rows = list(inference_rows)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_task[record["task_name"]].append(record)
    timing_keys = (
        "sandbox_boot_s",
        "sandbox_ready_to_agent_s",
        "session_connect_s",
        "agent_to_first_observation_s",
        "episode_loop_s",
        "sandbox_lifetime_s",
    )
    sandbox_seconds = sum(r["timing"]["sandbox_lifetime_s"] or 0.0 for r in records)
    return {
        **meta,
        "overall": _rate(records),
        "per_task": {task: _rate(items) for task, items in by_task.items()},
        "latency_ms": {
            "sdk_rtt": latency_stats([r.get("sdk_rtt_ms") for r in rows]),
            "server_inference": latency_stats([r.get("server_inference_ms") for r in rows]),
            "data_plane_rtt": latency_stats(
                [(r.get("timing") or {}).get("data_plane_rtt_ms") for r in rows]
            ),
            "first_request_sdk_rtt": latency_stats(
                [r.get("sdk_rtt_ms") for r in rows if r.get("chunk_index") == 0]
            ),
        },
        "startup_s": {
            key: latency_stats([r["timing"][key] for r in records]) for key in timing_keys
        },
        "modal": {
            "sandbox_seconds": sandbox_seconds,
            "estimated_usd": sandbox_seconds / 3600.0 * modal_usd_per_sandbox_hour(),
            "estimate_basis": "list prices per sandbox-hour; check the Modal dashboard",
        },
        "episodes": records,
    }
