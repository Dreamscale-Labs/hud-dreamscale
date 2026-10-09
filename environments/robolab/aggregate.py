"""Combine a model's RoboLab waves (including infra retries) into one result.

Keeps one result per (task, episode): a graded result wins over an infra failure,
and a retry's graded result fills an earlier pre-launch failure. Latency is pooled
from every request row in the episodes' timing.jsonl files.

    uv run --project environments/robolab python environments/robolab/aggregate.py \
        runs/flux3-prod-ep0 runs/flux3-prod-ep0-retry1 ... --output runs/flux3-results.json
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

REQUEST_FIELDS = ("sdk_rtt_ms", "server_inference_ms")
TIMING_FIELDS = ("data_plane_rtt_ms", "client_overhead_ms", "transport_send_ms")
EPISODE_FIELDS = ("task_name", "episode", "success", "robolab_score", "steps", "max_episode_length",
                  "trace_url", "session_id", "run", "local_video", "error")


def wilson(successes: int, n: int, z: float = 1.96):
    if not n:
        return None
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


def stats(values):
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    return {
        "n": int(array.size), "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)), "p95": float(np.percentile(array, 95)),
        "max": float(array.max()), "min": float(array.min()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    chosen: dict[tuple[str, int], dict] = {}
    jobs, sandbox_seconds, estimated_usd, wall = [], 0.0, 0.0, []
    model = None
    for run in args.runs:
        summary = json.loads((run / "summary.json").read_text())
        model = model or summary["model"]
        if summary["model"] != model:
            raise SystemExit("runs mix models")
        jobs.append({"run": run.name, "url": summary["job"]["url"], "overall": summary["overall"]})
        modal = summary.get("modal") or {}
        sandbox_seconds += modal.get("sandbox_seconds") or 0
        estimated_usd += modal.get("estimated_usd") or 0
        wall.append(summary.get("wall_clock_s"))
        for episode in summary["episodes"]:
            key = (episode["task_name"], episode["episode"])
            current = chosen.get(key)
            if current is None or (episode.get("graded") and not current.get("graded")):
                chosen[key] = {**episode, "run": str(run)}
    episodes = sorted(chosen.values(), key=lambda e: (e["task_name"], e["episode"]))
    graded = [e for e in episodes if e.get("graded")]
    per_task = {}
    for e in graded:
        row = per_task.setdefault(e["task_name"], {"episodes": 0, "successes": 0, "steps": []})
        row["episodes"] += 1
        row["successes"] += bool(e.get("success"))
        row["steps"].append(e.get("steps"))
    requests = {k: [] for k in REQUEST_FIELDS + TIMING_FIELDS}
    startup = {k: [] for k in ("session_connect_s", "sandbox_boot_s", "sandbox_ready_to_agent_s",
                               "agent_to_first_observation_s", "agent_episode_wall_s")}
    first_rtt = []
    for e in graded:
        for key in startup:
            value = (e.get("timing") or {}).get(key)
            if value is not None:
                startup[key].append(value)
        path = Path(e["run"]) / e["timing_file"] if e.get("timing_file") else None
        if not path or not path.is_file():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        inference = [r for r in rows if r.get("event") == "inference"]
        for index, row in enumerate(inference):
            if index == 0:
                first_rtt.append(row.get("sdk_rtt_ms"))
                continue
            for key in REQUEST_FIELDS:
                if row.get(key) is not None:
                    requests[key].append(row[key])
            for key in TIMING_FIELDS:
                value = (row.get("timing") or {}).get(key)
                if value is not None:
                    requests[key].append(value)
    successes = sum(bool(e.get("success")) for e in graded)
    result = {
        "model": model,
        "episodes_planned": len(episodes),
        "episodes_graded": len(graded),
        "unresolved_infra_failures": [
            {"task_name": e["task_name"], "episode": e["episode"], "error": e.get("error")}
            for e in episodes if not e.get("graded")
        ],
        "successes": successes,
        "success_rate": successes / len(graded) if graded else None,
        "success_rate_wilson95": wilson(successes, len(graded)),
        "per_task": {k: {**v, "success_rate": v["successes"] / v["episodes"]}
                     for k, v in sorted(per_task.items())},
        "latency_ms_after_first_request": {k: stats(v) for k, v in requests.items()},
        "first_request_sdk_rtt_ms": stats([v for v in first_rtt if v is not None]),
        "startup_s": {k: stats(v) for k, v in startup.items()},
        "jobs": jobs,
        "modal_sandbox_seconds": sandbox_seconds,
        "modal_estimated_usd": estimated_usd,
        "wave_wall_clock_s": wall,
        "episodes": [{k: e.get(k) for k in EPISODE_FIELDS} for e in episodes],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    keys = ("model", "episodes_graded", "successes", "success_rate", "success_rate_wilson95")
    print(json.dumps({k: result[k] for k in keys}, indent=2))


if __name__ == "__main__":
    main()
