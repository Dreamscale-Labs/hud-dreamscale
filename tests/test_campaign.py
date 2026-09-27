import json

import pytest

from hud_dreamscale.campaign import (
    check_budget,
    design,
    digest,
    paired_difference,
    qualify,
    task_rows,
)
from hud_dreamscale.campaign_metrics import summarize
from hud_dreamscale.campaign_run import preflight


def test_frozen_128_unique_tasks_and_five_matched_waves():
    frozen = design()
    assert len(frozen["tasks"]) == 128
    assert len({t["task_name"] for t in frozen["tasks"]}) == 128
    assert sum(t["suite"] != "libero_90" for t in frozen["tasks"]) == 40
    one, two = [task_rows(model) for model in frozen["models"]]
    assert len(one) == len(two) == 640
    for a, b in zip(one, two, strict=True):
        assert a.args == b.args
        assert a.columns["lane_id"] == b.columns["lane_id"]
    for state in range(5):
        wave = one[state * 128 : (state + 1) * 128]
        assert {t.args["init_state_id"] for t in wave} == {state}
        assert {t.columns["lane_id"] for t in wave} == set(range(128))


def test_lower_scaling_controls_allowed_but_no_silent_headline_fallback():
    assert len(task_rows("molmoact2-libero", concurrency=64, scored=False)) == 128
    with pytest.raises(ValueError, match="approved"):
        task_rows("molmoact2-libero", concurrency=64, scored=True)


def test_cost_gate_counts_failed_attempts_and_preserves_reserves():
    failed = [{"category": "qualification", "liability_usd": 99, "outcome": "failed"}]
    with pytest.raises(ValueError, match="shortfall"):
        check_budget(failed, category="qualification", estimated_usd=2)
    assert check_budget(failed, category="vla", estimated_usd=250) == 0
    with pytest.raises(ValueError, match="reserved"):
        check_budget([], category="wam", estimated_usd=1)


def test_qualification_requires_real_overlap_hardware_identity_audit_and_cleanup():
    base = dict(
        expected_episodes=128,
        completed_episodes=128,
        integration_errors=1,
        identity_audited=True,
        hardware_audited=True,
        hud_build_unchanged=True,
        actual_peak_overlap=128,
        requested_concurrency=128,
        actual_h100s=8,
        requested_h100s=8,
        cleanup_confirmed=True,
    )
    assert qualify(base)
    for fields in (
        {"integration_errors": 2},
        {"duplicate_actions": 1},
        {"actual_h100s": 16},
        {"actual_peak_overlap": 64},
        {"identity_audited": False},
        {"hardware_audited": False},
        {"hud_build_unchanged": False},
        {"cleanup_confirmed": False},
    ):
        assert not qualify({**base, **fields})
    # A missing episode is already in integration_errors; do not count it twice.
    assert qualify({**base, "completed_episodes": 127})


def test_measurement_requires_matching_audited_qualification(tmp_path):
    config = dict(
        design_sha256=digest(design()),
        api_base="https://api.dreamscalelabs.com",
        concurrency=128,
        h100s=8,
        robots_per_replica=16,
        kind="scored",
        model="pi0.5-libero",
        release_id="release",
        release_sha256="a" * 64,
        hud_environment_revision="build",
        simulator_image_digest="b" * 64,
        hud_environment_id="env-id",
        hud_environment_name="env-name",
        sdk_version="0.1.0a30",
        max_run_seconds=600,
        hourly_infrastructure_usd=50,
        cleanup_reserve_usd=10,
        remaining_study_estimate_usd=200,
    )
    with pytest.raises(ValueError, match="evidence required"):
        preflight(config, [])
    proof = dict(
        config=config.copy(),
        summary=dict(
            expected_episodes=128,
            completed_episodes=128,
            integration_errors=0,
            identity_audited=True,
            hardware_audited=True,
            hud_build_unchanged=True,
            actual_peak_overlap=128,
            requested_concurrency=128,
            actual_h100s=8,
            requested_h100s=8,
            cleanup_confirmed=True,
        ),
    )
    path = tmp_path / "qualification.json"
    path.write_text(json.dumps(proof))
    config.update(qualification_evidence=str(path), qualification_evidence_sha256=digest(proof))
    assert preflight(config, [])[0] == "vla"
    with pytest.raises(ValueError, match="differs from measured"):
        preflight({**config, "model": "molmoact2-libero"}, [])
    proof["summary"]["identity_audited"] = False
    path.write_text(json.dumps(proof))
    with pytest.raises(ValueError, match="digest differs"):
        preflight(config, [])
    with pytest.raises(ValueError, match="infrastructure gates"):
        preflight({**config, "qualification_evidence_sha256": digest(proof)}, [])


def test_paired_intervals_match_tasks_and_keep_resets_together():
    rows = [
        dict(
            suite="libero_goal",
            task_name=f"task{task}",
            args={"init_state_id": state},
            reward=task,
            integration_error=False,
        )
        for task in (0, 1)
        for state in range(5)
    ]
    result = paired_difference(rows, list(reversed(rows)), samples=100)
    assert result["second_minus_first"] == 0
    assert result["task_cluster_95"] == [0, 0]
    with pytest.raises(ValueError, match="not matched"):
        paired_difference(rows, rows[:-1])
    with pytest.raises(ValueError, match="duplicate"):
        paired_difference(rows + rows[:1], rows)


def test_overlap_uses_real_intervals_and_keeps_incomplete_episodes():
    events = [
        dict(event="wave_start", wave=0, monotonic_s=1),
        dict(event="episode_first_model_input", lane_id=0, episode_id="a"),
        dict(event="episode_first_model_input", lane_id=1, episode_id="b"),
        dict(event="episode_driven", lane_id=0, episode_id="a", duration_s=2),
        dict(event="inference_response", timing={"queue": 10, "replica_wall_ms": 25}),
        dict(event="wave_end", wave=0, monotonic_s=4),
    ]
    report = summarize(events)
    assert report["actual_peak_episode_overlap"] == 2
    assert report["unterminated_episodes"] == 1
    assert report["timings"]["warm_wave.duration_s"]["mean"] == 3
    assert report["timings"]["server.queue_ms"]["mean"] == 10
    assert report["timings"]["server.replica_wall_ms"]["mean"] == 25
    assert report["gpu_hours"] is None


@pytest.mark.parametrize(
    "kind,concurrency,count",
    [
        ("scored", 128, 128),
        ("scaling", 1, 1),
        ("qualification", 8, 1),
        ("qualification", 1, 0),
        ("qualification", 1, 129),
        ("qualification", 1, True),
    ],
)
def test_smoke_subset_cannot_reduce_measurements_or_skip_lanes(kind, concurrency, count):
    config = dict(
        design_sha256=digest(design()),
        api_base="https://api.dreamscalelabs.com",
        concurrency=concurrency,
        h100s=8 if concurrency == 128 else 1,
        robots_per_replica=16 if concurrency == 128 else 8,
        kind=kind,
        qualification_tasks=count,
    )
    with pytest.raises(ValueError, match="qualification task subset"):
        preflight(config, [])


@pytest.mark.parametrize("web", ["https://hud.ai", "https://www.hud.ai/"])
def test_production_hud_accepts_default_and_www_alias(web):
    from types import SimpleNamespace

    from hud_dreamscale.campaign_run import verify_production_hud

    verify_production_hud(
        SimpleNamespace(
            hud_web_url=web, hud_api_url="https://api.hud.ai", hud_runtime_url="https://mcp.hud.ai"
        )
    )


@pytest.mark.parametrize(
    "web,api",
    [
        ("http://hud.ai", "https://api.hud.ai"),
        ("https://dev.hud.ai", "https://api.hud.ai"),
        ("https://hud.ai", "http://api.hud.ai"),
        ("https://hud.ai", "https://api-dev.hud.ai"),
    ],
)
def test_production_hud_rejects_nonproduction_backend(web, api):
    from types import SimpleNamespace

    from hud_dreamscale.campaign_run import verify_production_hud

    with pytest.raises(ValueError, match="production platform"):
        verify_production_hud(
            SimpleNamespace(hud_web_url=web, hud_api_url=api, hud_runtime_url="https://mcp.hud.ai")
        )


@pytest.mark.parametrize("value", [True, -1, 3, 1.5])
def test_resubmission_policy_is_bounded(value):
    with pytest.raises(ValueError, match="resubmissions"):
        preflight(
            {
                "design_sha256": digest(design()),
                "api_base": "https://api.dreamscalelabs.com",
                "max_not_admitted_resubmissions": value,
            },
            [],
        )
