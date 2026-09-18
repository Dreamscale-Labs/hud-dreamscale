"""Warm cohorts retain capacity evidence without inventing session-start samples."""

import json

import pytest

from hud_dropbear.reporting import write_campaign_report


@pytest.mark.parametrize("width", [1, 3, 63])
def test_report_uses_only_warm_cohort_events_and_capacity(width, tmp_path):
    count = 6 if width == 1 else width * 2
    runs = [
        {
            "case_index": case,
            "lane_id": case % width,
            "trace_id": f"{case:032x}",
            "episode_id": f"warm-{width}-{case}",
            "reward": case % 2,
            "args": {
                "task_id": (case % 6) // 2,
                "init_state_id": case % 2,
                "seed": 0,
            },
        }
        for case in range(count)
    ]
    summary = {
        "concurrency": width,
        "expected_episodes": count,
        "runtime": "hud",
        "max_steps": 600,
        "runs": runs,
        "platform_evidence": {"verified": True},
        "provider_cleanup_confirmed": True,
        "simulator_cleanup": {"verified": True},
    }
    events = [
        {
            "event": "provider_reused_capacity",
            "ready_robots": 64,
            "deployment_id": "shared-deployment",
            "source": "authenticated_deployment_get",
            "status_get_s": 0.02,
            "elapsed_s": 0,
        }
    ]
    for run in runs:
        begin = 1 + (run["case_index"] // width) * 4 + run["lane_id"] * 0.001
        fields = {key: run[key] for key in ("episode_id", "lane_id")}
        events.extend(
            [
                {"event": "environment_ready", "elapsed_s": begin, **fields},
                {
                    "event": "inference",
                    "elapsed_s": begin + 0.2,
                    "duration_s": 0.12,
                    **fields,
                },
                {"event": "episode_driven", "elapsed_s": begin + 3, **fields},
            ]
        )
    (tmp_path / "results.json").write_text(json.dumps(summary))
    (tmp_path / "timings.jsonl").write_text("\n".join(map(json.dumps, events)))
    report = write_campaign_report(tmp_path, concurrency=width)
    assert report["gate"]["passed"]
    assert report["gate"]["success_rate"] == 0.5
    assert report["concurrency_evidence"]["provider_ready_robots"] == 64
    assert report["concurrency_evidence"]["active_episodes"]["peak_distinct_lanes"] == width
    assert report["timings"]["inference_session_readiness_s"]["n"] == 0
    assert report["timings"]["deployment_create_s"]["n"] == 0
    assert report["timings"]["episode_first_inference_s"]["n"] == count
    assert report["timings"]["episode_first_inference_s"]["p50"] == 0.12
