from decimal import Decimal

import pytest

from hud_dropbear.costs import (
    BudgetExceededError,
    CampaignBudget,
    allocate_app_cost,
    amount,
    estimate_stage,
)


def estimate(concurrency=8, **kwargs):
    return estimate_stage(
        concurrency=concurrency,
        **{
            "startup_seconds": "60",
            "idle_seconds": "60",
            "run_seconds": "60",
            "cleanup_seconds": "60",
            "modal_replica_usd_per_second": "0.001",
            "modal_overhead_usd_per_second": "0.0001",
            "hud_hourly_rate": "0.36",
            **kwargs,
        },
    )


def reserve(budget, stage="eight", **kwargs):
    budget.reserve_stage(
        stage,
        estimate=estimate(**kwargs),
        modal_lag_usd="0.2",
        hud_lag_usd="0.01",
        evidence_ref="private-current-provider-rate-receipt",
    )


def resources(budget, stage="eight"):
    for provider, resource in (("modal", "ap-owned"), ("hud", "instance-owned")):
        budget.register_owned_resource(
            provider, resource, baseline_usd="1", evidence_ref="ownership"
        )
        budget.bind_resource(stage, provider, resource)


def settle(budget, *, confirm=True, settled_through="1000"):
    for provider, resource in (("modal", "ap-owned"), ("hud", "instance-owned")):
        if confirm:
            budget.confirm_termination(
                provider, resource, terminated_at="1000", evidence_ref="stop"
            )
        budget.record_billing(
            provider,
            resource,
            total_usd="1.1",
            evidence_ref="provider-bill",
            settled_through=settled_through,
        )


def test_stage_cost_counts_every_phase_and_partial_replica_without_float_math():
    result = estimate(concurrency=9)
    assert result["replicas"] == 2
    assert set(result["seconds"]) == {"startup", "idle", "run", "cleanup"}
    assert sum(map(Decimal, result["estimated_usd"]["modal"].values())) == Decimal("0.504")
    assert sum(map(Decimal, result["estimated_usd"]["hud"].values())) == Decimal("0.216")
    assert Decimal(result["gpu_only_estimated_usd"]) == Decimal("0.48")
    assert result["rates_usd_per_second"]["modal_overhead"] == "0.0001"


def test_unknown_hud_balance_fails_closed_before_stage_creation(tmp_path):
    budget = CampaignBudget.create(tmp_path / "cost.jsonl", modal_cap_usd="200")
    before = budget.path.read_bytes()
    with pytest.raises(BudgetExceededError, match="hud.*unknown"):
        reserve(budget)
    assert budget.path.read_bytes() == before


def test_reservations_accumulate_across_reopen_and_new_cohorts(tmp_path):
    path = tmp_path / "cost.jsonl"
    budget = CampaignBudget.create(path, modal_cap_usd="1", hud_cap_usd="1")
    reserve(budget, "one", concurrency=1)
    first = budget.report()
    reopened = CampaignBudget(path)
    reserve(reopened, "eight")
    assert Decimal(first["modal"]["conservative_liability_usd"]) == Decimal("0.464")
    assert Decimal(reopened.report()["modal"]["conservative_liability_usd"]) == Decimal("0.928")
    before = path.read_bytes()
    with pytest.raises(BudgetExceededError, match="modal"):
        reserve(reopened, "third")
    assert path.read_bytes() == before
    with pytest.raises(FileExistsError):
        CampaignBudget.create(path, modal_cap_usd="200", hud_cap_usd="100")
    assert path.stat().st_mode & 0o777 == 0o600


def test_stopping_alone_does_not_release_a_reservation(tmp_path):
    budget = CampaignBudget.create(tmp_path / "cost.jsonl", modal_cap_usd="200", hud_cap_usd="10")
    reserve(budget)
    resources(budget)
    settle(budget, settled_through=None)
    with pytest.raises(ValueError, match="complete posted billing"):
        budget.reconcile_stage("eight", evidence_ref="app-stopped")
    report = budget.report()
    assert report["modal"]["provider_posted_usd"] == "0.1"
    assert Decimal(report["modal"]["unreconciled_holds_usd"]) == Decimal("0.464")
    assert Decimal(report["modal"]["conservative_liability_usd"]) == Decimal("0.564")


@pytest.mark.parametrize("confirmed,coverage", [(False, "1000"), (True, "999")])
def test_incomplete_termination_or_billing_coverage_preserves_liability(
    tmp_path, confirmed, coverage
):
    budget = CampaignBudget.create(tmp_path / "cost.jsonl", modal_cap_usd="200", hud_cap_usd="10")
    reserve(budget)
    resources(budget)
    settle(budget, confirm=confirmed, settled_through=coverage)
    with pytest.raises(ValueError, match="not confirmed"):
        budget.reconcile_stage("eight", evidence_ref="insufficient-receipt")
    assert Decimal(budget.report()["modal"]["unreconciled_holds_usd"]) == Decimal("0.464")


def test_complete_reconciliation_releases_hold_but_never_spend(tmp_path):
    budget = CampaignBudget.create(tmp_path / "cost.jsonl", modal_cap_usd="200", hud_cap_usd="10")
    reserve(budget)
    resources(budget)
    settle(budget)
    budget.reconcile_stage("eight", evidence_ref="final-provider-receipts")
    assert budget.report()["modal"]["conservative_liability_usd"] == "0.1"
    assert budget.report()["modal"]["unreconciled_holds_usd"] == "0"
    reserve(CampaignBudget(budget.path), "thirty-two", concurrency=32)
    assert budget.report()["modal"]["provider_posted_usd"] == "0.1"
    assert Decimal(budget.report()["modal"]["conservative_liability_usd"]) == Decimal("1.284")


def test_api_app_cost_is_counted_once_when_shared_by_sequential_stages(tmp_path):
    budget = CampaignBudget.create(tmp_path / "cost.jsonl", modal_cap_usd="200", hud_cap_usd="10")
    reserve(budget, "first")
    reserve(budget, "second")
    resources(budget, "first")
    for provider, resource in (("modal", "ap-owned"), ("hud", "instance-owned")):
        budget.bind_resource("second", provider, resource)
    settle(budget)
    budget.reconcile_stage("first", evidence_ref="shared-app-complete")
    budget.reconcile_stage("second", evidence_ref="shared-app-complete")
    assert budget.report()["modal"]["provider_posted_usd"] == "0.1"


def test_scope_and_baseline_cannot_drift(tmp_path):
    budget = CampaignBudget.create(tmp_path / "cost.jsonl", modal_cap_usd="200", hud_cap_usd="10")
    reserve(budget)
    resources(budget)
    with pytest.raises(ValueError, match="exact registered"):
        budget.record_billing("modal", "another-app", total_usd="5", evidence_ref="wrong-scope")
    with pytest.raises(ValueError, match="cannot reset"):
        budget.register_owned_resource("modal", "ap-owned", baseline_usd="5", evidence_ref="reset")
    with pytest.raises(ValueError, match="decreased"):
        budget.record_billing("modal", "ap-owned", total_usd="0.5", evidence_ref="backwards")


def test_actual_overrun_is_recorded_and_blocks_further_reservations(tmp_path):
    budget = CampaignBudget.create(tmp_path / "cost.jsonl", modal_cap_usd="1", hud_cap_usd="10")
    reserve(budget)
    resources(budget)
    budget.record_billing("modal", "ap-owned", total_usd="3", evidence_ref="actual-overrun")
    assert budget.report()["modal"]["provider_posted_usd"] == "2"
    assert budget.report()["modal"]["over_cap"]
    with pytest.raises(BudgetExceededError):
        reserve(budget, "do-not-start")


def test_app_cost_allocation_is_explicitly_estimated_and_keeps_unattributed_time():
    report = allocate_app_cost(
        posted_usd="10", app_lifetime_seconds="100", cohort_seconds={"one": "20", "eight": "30"}
    )
    assert report["basis"] == "estimated_time_allocation_of_app_billing"
    assert report["estimated_cohort_usd"] == {"one": "2", "eight": "3", "unattributed": "5"}
    thirds = allocate_app_cost(
        posted_usd="1", app_lifetime_seconds="3", cohort_seconds={"one": "1", "eight": "1"}
    )
    assert sum(map(Decimal, thirds["estimated_cohort_usd"].values())) == Decimal("1")
    with pytest.raises(ValueError, match="overlap"):
        allocate_app_cost(posted_usd="1", app_lifetime_seconds="2", cohort_seconds={"a": "3"})


def test_ledger_modification_and_truncation_are_detected(tmp_path):
    path = tmp_path / "cost.jsonl"
    budget = CampaignBudget.create(path, modal_cap_usd="200", hud_cap_usd="10")
    reserve(budget)
    content = path.read_text()
    path.write_text(content.replace('"eight"', '"wrong"'))
    with pytest.raises(ValueError, match="digest"):
        CampaignBudget(path)
    path.write_text(content[:-12])
    with pytest.raises(ValueError):
        CampaignBudget(path)


@pytest.mark.parametrize("value", [0.1, True, "-1", "NaN", "Infinity"])
def test_invalid_or_float_amounts_are_rejected(value):
    with pytest.raises(ValueError):
        amount(value)


def test_modal_cap_is_explicit_and_generic(tmp_path):
    with pytest.raises(TypeError):
        CampaignBudget.create(tmp_path / "missing-cap.jsonl", hud_cap_usd="1")
    with pytest.raises(ValueError, match="positive"):
        CampaignBudget.create(tmp_path / "zero-cap.jsonl", modal_cap_usd="0")
    budget = CampaignBudget.create(tmp_path / "cost.jsonl", modal_cap_usd="201", hud_cap_usd="1")
    assert budget.report()["modal"]["cap_usd"] == "201"
