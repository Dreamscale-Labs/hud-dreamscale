"""Frozen VLA study design, spend gates and evidence summaries (no implicit launch)."""

import hashlib
import json
import math
from collections import Counter
from pathlib import Path

from .contract import CONTROL_HZ, POOLED_MODELS, TASK_SUITES

VERSION = "dreamscale-hud-vla-128-v1"
HORIZONS = {
    "libero_spatial": 220,
    "libero_object": 280,
    "libero_goal": 300,
    "libero_10": 520,
    "libero_90": 400,
}
SCALING = ((1, 1, 8), (8, 1, 8), (16, 1, 16), (32, 4, 8), (64, 8, 8), (128, 8, 16))
CAPS = {"qualification": 100.0, "vla": 250.0, "wam": 100.0, "recovery": 50.0}


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def design():
    excluded = sorted(TASK_SUITES["libero_90"])[-2:]
    tasks = [
        {"suite": suite, "task_id": index, "task_name": name, "max_steps": HORIZONS[suite]}
        for suite, names in TASK_SUITES.items()
        for index, name in enumerate(names)
        if suite != "libero_90" or name not in excluded
    ]
    if len(tasks) != 128 or len({t["task_name"] for t in tasks}) != 128:
        raise ValueError("task catalogue does not contain 128 distinct selected tasks")
    return {
        "version": VERSION,
        "tasks": tasks,
        "excluded_libero90": excluded,
        "initial_states": list(range(5)),
        "models": list(POOLED_MODELS),
        "control_hz": CONTROL_HZ,
        "settle_steps": 10,
        "sim_seed": 0,
        "task_order_index": 0,
        "task_catalog_sha256": digest(TASK_SUITES),
        "scored_episodes": 1280,
        "scaling_episodes": 1536,
        "headline_repeat_episodes": 512,
        "scaling": [{"concurrency": c, "h100s": g, "robots_per_replica": d} for c, g, d in SCALING],
        "budget_usd": CAPS,
        "total_cap_usd": 500,
        "headline_repeats": 3,
        "fallback_requires_human_approval": True,
        "models_contract": POOLED_MODELS,
    }


def task_rows(model, *, concurrency=128, scored=True):
    from hud import Task

    if model not in POOLED_MODELS or concurrency not in {c for c, _, _ in SCALING}:
        raise ValueError("unknown model or scale configuration")
    if scored and concurrency != 128:
        raise ValueError("lower headline concurrency requires a separately approved study revision")
    frozen = design()
    rows = []
    for initial_state in frozen["initial_states"] if scored else [0]:
        for index, case in enumerate(frozen["tasks"]):
            rows.append(
                Task(
                    env=POOLED_MODELS[model]["env"],
                    id=case["suite"],
                    slug=f"{model}-t{index:03d}-i{initial_state}",
                    args={
                        "task_id": case["task_id"],
                        "init_state_id": initial_state,
                        "seed": frozen["sim_seed"],
                        "max_steps": case["max_steps"],
                    },
                    columns={
                        "task_name": case["task_name"],
                        "suite": case["suite"],
                        "model": model,
                        "cohort_version": VERSION,
                        "design_sha256": digest(frozen),
                        "case_index": index,
                        "lane_id": index % concurrency,
                        "lane_episode_index": initial_state * (128 // concurrency)
                        + index // concurrency,
                        "noise_seed": (initial_state * 128 + index) * 10000,
                        "observation_profile": POOLED_MODELS[model]["profile"],
                    },
                )
            )
    return rows


def check_budget(ledger, *, category, estimated_usd):
    """Posted charges plus unresolved reservations count, including failed runs."""
    if category not in ("qualification", "vla", "recovery"):
        raise ValueError("WAM budget is reserved; no WAM execution in this phase")
    if (
        type(estimated_usd) not in (int, float)
        or not math.isfinite(estimated_usd)
        or estimated_usd <= 0
    ):
        raise ValueError("a positive measured cost estimate is required before launch")
    totals = Counter()
    for entry in ledger:
        amount = entry["liability_usd"]
        if entry["category"] not in CAPS or not math.isfinite(amount) or amount < 0:
            raise ValueError("invalid cost ledger")
        totals[entry["category"]] += amount
    if (
        totals[category] + estimated_usd > CAPS[category]
        or sum(totals.values()) + estimated_usd > 500
    ):
        raise ValueError("campaign budget shortfall; report it before changing samples or spending")
    return CAPS[category] - totals[category] - estimated_usd


def qualify(summary, *, expected=128):
    """Infrastructure completion is separate from policy success."""
    if (
        summary.get("identity_errors", 0)
        or summary.get("duplicate_actions", 0)
        or summary.get("cross_slot_errors", 0)
    ):
        return False
    return (
        summary.get("identity_audited") is True
        and summary.get("hardware_audited") is True
        and summary.get("hud_build_unchanged") is True
        and summary.get("expected_episodes") == expected
        and summary.get("completed_episodes", 0) >= math.ceil(0.99 * expected)
        and expected - summary.get("integration_errors", expected) >= math.ceil(0.99 * expected)
        and summary.get("actual_peak_overlap") == summary.get("requested_concurrency")
        and summary.get("actual_h100s") == summary.get("requested_h100s")
        and summary.get("cleanup_confirmed") is True
    )


def wilson(successes, total):
    if total == 0:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [max(0, center - half), min(1, center + half)]


def result_tables(rows):
    """Keep infrastructure errors in the planned denominator and visible separately."""
    results = {}
    for label, selected in {
        "overall": rows,
        "conventional_40": [r for r in rows if r["suite"] != "libero_90"],
        **{suite: [r for r in rows if r["suite"] == suite] for suite in HORIZONS},
    }.items():
        n = len(selected)
        successes = sum(
            r.get("reward") == 1 and not r.get("integration_error", True) for r in selected
        )
        results[label] = {
            "planned": n,
            "successes": successes,
            "infrastructure_errors": sum(r.get("integration_error", True) for r in selected),
            "success_rate": successes / n if n else None,
            "episode_wilson_95": wilson(successes, n),
        }
    return results


def write_design(path):
    path = Path(path)
    frozen = design()
    with path.open("x") as stream:
        json.dump({"sha256": digest(frozen), "design": frozen}, stream, indent=2)
        stream.write("\n")
    return digest(frozen)


def paired_difference(first, second, *, samples=10000, seed=20260927):
    """Matched task-cluster bootstrap; keep each task's five states together.

    This interval describes this frozen task collection, not generalization to
    unseen robot tasks. Infrastructure-missing outcomes remain zero in the
    planned-episode measure and must be reported alongside their error counts.
    """
    import numpy as np

    def keyed(rows):
        values = {}
        for row in rows:
            key = (row["suite"], row["task_name"], row["args"]["init_state_id"])
            if key in values:
                raise ValueError("duplicate scored episode")
            values[key] = int(row.get("reward") == 1 and not row.get("integration_error", True))
        return values

    a, b = keyed(first), keyed(second)
    if not a or a.keys() != b.keys():
        raise ValueError("model episodes are not matched")
    tasks = sorted({key[:2] for key in a})
    differences = []
    for task in tasks:
        if {k[2] for k in a if k[:2] == task} != set(range(5)):
            raise ValueError("each task requires all five initial states")
        differences.append(sum(b[(*task, i)] - a[(*task, i)] for i in range(5)) / 5)
    values = np.asarray(differences)
    bootstrap = np.random.default_rng(seed).choice(values, (samples, len(values))).mean(axis=1)
    return {
        "second_minus_first": float(values.mean()),
        "task_cluster_95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "task_count": len(tasks),
        "bootstrap_samples": samples,
    }
