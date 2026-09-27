"""Retain startup, warm work and failures as separate evidence populations."""

from collections import Counter, defaultdict

import numpy as np


def distribution(values):
    if not values:
        return None
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all() or (array < 0).any():
        raise ValueError("invalid timing sample")
    return {
        "n": len(values),
        "mean": float(array.mean()),
        "min": float(array.min()),
        "p50": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def summarize(events):
    timings = defaultdict(list)
    counts = Counter()
    active = set()
    peak = 0
    malformed = []
    waves = {}
    for row in events:
        kind = row["event"]
        counts[kind] += 1
        for key, value in row.items():
            if (
                key.endswith(("_s", "_ms"))
                and key not in ("elapsed_s", "unix_s", "monotonic_s")
                and type(value) in (int, float)
            ):
                timings[f"{kind}.{key}"].append(value)
        for key, value in row.get("timing", {}).items():
            unit_key = key if key.endswith("_ms") else f"{key}_ms"
            timings[f"server.{unit_key}"].append(value)
        if kind == "episode_first_model_input":
            identity = (row["lane_id"], row["episode_id"])
            if identity in active:
                malformed.append({"event": kind, "reason": "duplicate episode start"})
            active.add(identity)
            peak = max(peak, len(active))
        elif kind in ("episode_driven", "episode_error"):
            active.discard((row["lane_id"], row["episode_id"]))
        elif kind == "wave_start":
            waves[row["wave"]] = row["monotonic_s"]
        elif kind == "wave_end" and row["wave"] in waves:
            timings["warm_wave.duration_s"].append(row["monotonic_s"] - waves.pop(row["wave"]))
    return {
        "event_counts": dict(counts),
        "timings": {k: distribution(v) for k, v in sorted(timings.items())},
        "actual_peak_episode_overlap": peak,
        "unterminated_episodes": len(active),
        "malformed_events": malformed,
        "unfinished_waves": sorted(waves),
        "gpu_hours": None,
        "provider_cost_usd": None,
        "cost_note": "Requires actual provider allocation intervals and billing.",
    }
