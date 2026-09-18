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


def reserve_single_provider(budget, provider):
    budget.reserve_stage(
        "single",
        estimate=estimate(
            modal_replica_usd_per_second="0.001" if provider == "modal" else "0",
            modal_overhead_usd_per_second="0",
            hud_hourly_rate="0.36" if provider == "hud" else "0",
        ),
        modal_lag_usd="0.2" if provider == "modal" else "0",
        hud_lag_usd="0.01" if provider == "hud" else "0",
        evidence_ref="single-provider-rate-receipt",
    )


def bind_owned(budget, provider, resource, *, stage="single"):
    budget.register_owned_resource(provider, resource, baseline_usd="0", evidence_ref="ownership")
    budget.bind_resource(stage, provider, resource)


def settle_owned(budget, provider, resource, *, confirm=True, coverage="1000"):
    if confirm:
        budget.confirm_termination(provider, resource, terminated_at="1000", evidence_ref="stop")
    budget.record_billing(
        provider, resource, total_usd="0.1", evidence_ref="posted-bill", settled_through=coverage
    )


@pytest.mark.parametrize("provider", ["hud", "modal"])
def test_single_provider_stage_reconciles_without_dummy_resources(tmp_path, provider):
    budget = CampaignBudget.create(tmp_path / "cost.jsonl", modal_cap_usd="200", hud_cap_usd="10")
    reserve_single_provider(budget, provider)
    bind_owned(budget, provider, "only-owned-resource")
    settle_owned(budget, provider, "only-owned-resource")
    budget.reconcile_stage("single", evidence_ref="complete-single-provider-receipt")
    reopened = CampaignBudget(budget.path).report()
    assert reopened[provider]["unreconciled_holds_usd"] == "0"
    assert reopened[provider]["conservative_liability_usd"] == "0.1"
    unused = "modal" if provider == "hud" else "hud"
    assert reopened[unused]["conservative_liability_usd"] == "0"


@pytest.mark.parametrize("missing", ["hud", "modal"])
def test_mixed_stage_requires_bound_evidence_from_every_funded_provider(tmp_path, missing):
    budget = CampaignBudget.create(tmp_path / "cost.jsonl", modal_cap_usd="200", hud_cap_usd="10")
    reserve(budget)
    present = "modal" if missing == "hud" else "hud"
    bind_owned(budget, present, "bound", stage="eight")
    settle_owned(budget, present, "bound")
    # A fully settled global resource is insufficient until explicitly bound to this stage.
    budget.register_owned_resource(missing, "unbound", baseline_usd="0", evidence_ref="ownership")
    settle_owned(budget, missing, "unbound")
    before = budget.path.read_bytes()
    with pytest.raises(ValueError, match=f"Funded providers.*{missing}"):
        budget.reconcile_stage("eight", evidence_ref="missing-stage-binding")
    assert budget.path.read_bytes() == before
    assert all(Decimal(row["unreconciled_holds_usd"]) > 0 for row in budget.report().values())


def test_every_bound_resource_must_settle_even_with_same_provider(tmp_path):
    budget = CampaignBudget.create(tmp_path / "cost.jsonl", modal_cap_usd="200", hud_cap_usd="10")
    reserve_single_provider(budget, "hud")
    for resource in ("instance-first", "instance-second"):
        bind_owned(budget, "hud", resource)
        settle_owned(
            budget, "hud", resource, coverage="1000" if resource.endswith("first") else None
        )
    before = budget.path.read_bytes()
    with pytest.raises(ValueError, match="complete posted billing"):
        budget.reconcile_stage("single", evidence_ref="one-resource-unsettled")
    assert budget.path.read_bytes() == before
    settle_owned(budget, "hud", "instance-second", confirm=False)
    budget.reconcile_stage("single", evidence_ref="both-resource-bills-complete")
    assert budget.report()["hud"]["conservative_liability_usd"] == "0.2"


@pytest.mark.parametrize("confirm,coverage", [(False, "1000"), (True, None), (True, "999")])
def test_zero_hold_never_exempts_a_bound_provider_resource(tmp_path, confirm, coverage):
    budget = CampaignBudget.create(tmp_path / "cost.jsonl", modal_cap_usd="200", hud_cap_usd="10")
    reserve_single_provider(budget, "hud")
    bind_owned(budget, "hud", "funded-instance")
    settle_owned(budget, "hud", "funded-instance")
    bind_owned(budget, "modal", "unexpected-owned-app")
    settle_owned(budget, "modal", "unexpected-owned-app", confirm=confirm, coverage=coverage)
    assert budget.report()["modal"]["unreconciled_holds_usd"] == "0"
    before = budget.path.read_bytes()
    with pytest.raises(ValueError, match="Termination and complete posted billing"):
        budget.reconcile_stage("single", evidence_ref="zero-hold-does-not-prove-settlement")
    assert budget.path.read_bytes() == before
    settle_owned(budget, "modal", "unexpected-owned-app", confirm=not confirm)
    budget.reconcile_stage("single", evidence_ref="every-bound-resource-complete")
    assert budget.report()["modal"]["conservative_liability_usd"] == "0.1"
    assert budget.report()["hud"]["unreconciled_holds_usd"] == "0"


def test_lag_only_funding_still_requires_provider_evidence(tmp_path):
    budget = CampaignBudget.create(tmp_path / "cost.jsonl", modal_cap_usd="200", hud_cap_usd="10")
    reserve(budget, modal_replica_usd_per_second="0", modal_overhead_usd_per_second="0")
    bind_owned(budget, "hud", "instance", stage="eight")
    settle_owned(budget, "hud", "instance")
    before = budget.path.read_bytes()
    with pytest.raises(ValueError, match="Funded providers.*modal"):
        budget.reconcile_stage("eight", evidence_ref="modal-lag-reservation-is-still-funded")
    assert budget.path.read_bytes() == before


def test_reconciliation_without_any_bound_resource_still_fails(tmp_path):
    budget = CampaignBudget.create(tmp_path / "cost.jsonl", modal_cap_usd="200", hud_cap_usd="10")
    reserve_single_provider(budget, "hud")
    before = budget.path.read_bytes()
    with pytest.raises(ValueError, match="Bound resource evidence"):
        budget.reconcile_stage("single", evidence_ref="unproven-no-allocation")
    assert budget.path.read_bytes() == before


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


def completed_lifetime_fixture(tmp_path, *, terminate=True, shared=False):
    """Synthetic provider evidence; never an assertion about actual billing caps."""
    import hashlib
    import json

    budget = CampaignBudget.create(
        tmp_path / "lifetime.jsonl", modal_cap_usd="200", hud_cap_usd="100"
    )
    planned = estimate(
        concurrency=64,
        startup_seconds="900",
        idle_seconds="420",
        run_seconds="4000",
        cleanup_seconds="450",
        modal_replica_usd_per_second="0.00235179",
        modal_overhead_usd_per_second="0.00018417",
        hud_hourly_rate="0",
    )
    budget.reserve_stage(
        "wide",
        estimate=planned,
        modal_lag_usd="3.37871270",
        hud_lag_usd="2",
        evidence_ref="original planned lifetime",
    )
    if shared:
        reserve(budget, "another")
    bills = {"ap-gpu": "9.51503124", "ap-gateway": "1.14253052"}
    for provider, identity in (
        ("modal", "ap-gpu"),
        ("modal", "ap-gateway"),
        ("hud", "instance-owned"),
    ):
        budget.register_owned_resource(provider, identity, baseline_usd="1", evidence_ref="owned")
        budget.bind_resource("wide", provider, identity)
        if shared and identity == "ap-gpu":
            budget.bind_resource("another", provider, identity)
        if terminate:
            budget.confirm_termination(
                provider, identity, terminated_at="1000", evidence_ref="stop"
            )
        budget.record_billing(
            provider, identity, total_usd=bills.get(identity, "1.20"), evidence_ref="posted"
        )
    source = tmp_path / "reviewed-lifecycle-and-billing.json"
    source.write_text(
        '{"fixture":"complete owned inventory, terminal lifetimes and posted billing"}'
    )
    proof = {
        "schema_version": 1,
        "purpose": "completed_compute_lifetime_repricing",
        "stage": "wide",
        "provider": "modal",
        "resource_inventory_complete": True,
        "unknown_resource_creation": False,
        "additional_billable_resources": [],
        "lifecycle_evidence_complete": True,
        "basis": "Reviewed terminal lifetime estimate and unpaid uncertainty; not settled",
        "resources": [
            {
                "resource_id": key,
                "terminated_at": "1000",
                "baseline_usd": "1",
                "recorded_total_usd": value,
            }
            for key, value in bills.items()
        ],
        "conservative_lifecycle_estimate_usd": "17.5246",
        "unbilled_lag_usd": "2.4754",
        "original_planned_estimate_usd": "109.62128730",
        "evidence": [
            {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        ],
    }
    path = tmp_path / "completed-proof.json"
    path.write_text(json.dumps(proof))
    return budget, path, proof


def reprice_completed(budget, path, *, hold="20", digest=None):
    import hashlib

    budget.reprice_completed_compute_hold(
        "wide",
        "modal",
        hold_usd=hold,
        evidence_path=path,
        evidence_sha256=digest or hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def test_completed_repricing_keeps_original_plan_posted_cost_and_unsettled_history(tmp_path):
    import json

    from hud_dropbear.costs import _read, _state

    budget, path, _ = completed_lifetime_fixture(tmp_path)
    before = budget.path.read_bytes()
    reprice_completed(budget, path)
    assert budget.path.read_bytes().startswith(before)
    rows = [json.loads(line) for line in budget.path.read_text().splitlines()]
    assert len(rows) == len(before.splitlines()) + 1
    assert rows[1]["holds_usd"]["modal"] == "113.00000000"
    assert rows[-1]["event"] == "reprice_completed_hold"
    assert rows[-1]["previous_hold_usd"] == "113.00000000"
    assert rows[-1]["lag_usd"] == "2.4754"
    report = CampaignBudget(budget.path).report()
    assert report["modal"]["provider_posted_usd"] == "8.65756176"
    assert report["modal"]["unreconciled_holds_usd"] == "20"
    assert report["modal"]["conservative_liability_usd"] == "28.65756176"
    assert Decimal(report["hud"]["unreconciled_holds_usd"]) == Decimal("2")
    with budget.path.open() as file:
        state = _state(_read(file))
    assert not state["stages"]["wide"]["released"]
    assert all(row["settled_through"] is None for row in state["resources"].values())
    with pytest.raises(ValueError, match="complete posted billing"):
        budget.reconcile_stage("wide", evidence_ref="not settlement")
    budget.record_billing("modal", "ap-gpu", total_usd="10.51503124", evidence_ref="later bill")
    assert budget.report()["modal"]["conservative_liability_usd"] == "29.65756176"
    assert budget.report()["modal"]["unreconciled_holds_usd"] == "20"
    # Even a surprising late bill is retained; it blocks further allocations.
    budget.record_billing("modal", "ap-gpu", total_usd="201", evidence_ref="late overrun")
    before = budget.path.read_bytes()
    with pytest.raises(BudgetExceededError, match="modal"):
        reserve(budget, "next")
    assert budget.path.read_bytes() == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("purpose", "billing_settlement"),
        ("stage", "different"),
        ("provider", "hud"),
        ("resource_inventory_complete", False),
        ("unknown_resource_creation", True),
        ("additional_billable_resources", ["unknown"]),
        ("lifecycle_evidence_complete", False),
        ("basis", ""),
        ("resources", []),
        ("evidence", []),
        ("original_planned_estimate_usd", "109"),
        ("conservative_lifecycle_estimate_usd", "0"),
        ("unbilled_lag_usd", "0"),
        ("unbilled_lag_usd", "NaN"),
        ("unbilled_lag_usd", "2.4755"),
    ],
)
def test_completed_repricing_incomplete_or_inconsistent_proof_is_nonmutating(
    tmp_path, field, value
):
    import json

    budget, path, proof = completed_lifetime_fixture(tmp_path)
    proof[field] = value
    path.write_text(json.dumps(proof))
    before = budget.path.read_bytes()
    with pytest.raises(ValueError):
        reprice_completed(budget, path)
    assert budget.path.read_bytes() == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("resource_id", "ap-other"),
        ("resource_id", "vo-owned"),
        ("terminated_at", "999"),
        ("baseline_usd", "0"),
        ("recorded_total_usd", "9.5"),
    ],
)
def test_completed_repricing_exact_current_accounting_identity_required(tmp_path, field, value):
    import json

    budget, path, proof = completed_lifetime_fixture(tmp_path)
    proof["resources"][0][field] = value
    path.write_text(json.dumps(proof))
    before = budget.path.read_bytes()
    with pytest.raises(ValueError):
        reprice_completed(budget, path)
    assert budget.path.read_bytes() == before


@pytest.mark.parametrize(
    "condition",
    [
        "unterminated",
        "shared",
        "duplicate",
        "extra",
        "active_volume",
        "terminated_volume",
        "hud_unterminated",
        "later_bill",
    ],
)
def test_completed_repricing_ineligible_inventory_is_nonmutating(tmp_path, condition):
    import json

    budget, path, proof = completed_lifetime_fixture(
        tmp_path,
        terminate=condition not in {"unterminated", "hud_unterminated"},
        shared=condition == "shared",
    )
    if condition == "hud_unterminated":
        for identity in ("ap-gpu", "ap-gateway"):
            budget.confirm_termination("modal", identity, terminated_at="1000", evidence_ref="stop")
    if condition == "duplicate":
        proof["resources"].append(proof["resources"][0])
    if condition in {"extra", "active_volume", "terminated_volume"}:
        identity = "ap-extra" if condition == "extra" else "vo-owned"
        budget.register_owned_resource("modal", identity, baseline_usd="0", evidence_ref="owned")
        budget.bind_resource("wide", "modal", identity)
        if condition != "active_volume":
            budget.confirm_termination("modal", identity, terminated_at="1000", evidence_ref="stop")
    if condition == "later_bill":
        budget.record_billing("modal", "ap-gpu", total_usd="10", evidence_ref="new bill")
    path.write_text(json.dumps(proof))
    before = budget.path.read_bytes()
    with pytest.raises(ValueError):
        reprice_completed(budget, path)
    assert budget.path.read_bytes() == before


@pytest.mark.parametrize("target", ["proof", "reference"])
def test_completed_repricing_evidence_hashes_are_verified(tmp_path, target):
    import hashlib
    from pathlib import Path

    budget, path, proof = completed_lifetime_fixture(tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    changed = path if target == "proof" else Path(proof["evidence"][0]["path"])
    changed.write_text(changed.read_text() + " ")
    before = budget.path.read_bytes()
    with pytest.raises(ValueError, match="differs"):
        reprice_completed(budget, path, digest=digest)
    assert budget.path.read_bytes() == before


def test_completed_repricing_cannot_repeat_refine_or_reopen_stage(tmp_path):
    import hashlib
    import json

    budget, path, proof = completed_lifetime_fixture(tmp_path)
    reprice_completed(budget, path)
    before = budget.path.read_bytes()
    with pytest.raises(ValueError, match="already refined/repriced"):
        reprice_completed(budget, path)
    ordinary = {**proof, "resource_ids": ["ap-gpu", "ap-gateway"]}
    path.write_text(json.dumps(ordinary))
    with pytest.raises(ValueError, match="already refined"):
        budget.refine_stage_hold(
            "wide",
            "modal",
            hold_usd="15",
            evidence_path=path,
            evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    assert budget.path.read_bytes() == before
    budget.register_owned_resource("modal", "ap-future", baseline_usd="0", evidence_ref="owned")
    before = budget.path.read_bytes()
    with pytest.raises(ValueError, match="refined"):
        budget.bind_resource("wide", "modal", "ap-future")
    assert budget.path.read_bytes() == before
