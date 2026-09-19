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


def completed_lifetime_fixture(tmp_path, *, terminate=True, shared=False, post_modal_billing=True):
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
    if not post_modal_billing:
        bills = dict.fromkeys(bills, "1")
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
        if provider != "modal" or post_modal_billing:
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


def remaining_lifetime_proof(path, proof):
    import json

    posted = sum(
        Decimal(row["recorded_total_usd"]) - Decimal(row["baseline_usd"])
        for row in proof["resources"]
    )
    remaining = max(Decimal(proof["conservative_lifecycle_estimate_usd"]) - posted, Decimal(0))
    proof.update(
        schema_version=2,
        liability_basis="remaining_after_exact_posted_cost",
        posted_cost_deduction_usd=str(posted),
        remaining_lifecycle_allowance_usd=str(remaining),
    )
    path.write_text(json.dumps(proof))
    return str(remaining + Decimal(proof["unbilled_lag_usd"]))


def test_remaining_liability_deducts_only_posted_above_baseline_and_preserves_lag(tmp_path):
    import json

    budget, path, proof = completed_lifetime_fixture(tmp_path)
    hold = remaining_lifetime_proof(path, proof)
    reprice_completed(budget, path, hold=hold)
    row = json.loads(budget.path.read_text().splitlines()[-1])
    assert row["posted_cost_deduction_usd"] == "8.65756176"  # Excludes two $1 baselines.
    assert row["remaining_lifecycle_allowance_usd"] == "8.86703824"
    assert row["lag_usd"] == "2.4754"
    reopened = CampaignBudget(budget.path)
    report = reopened.report()["modal"]
    assert report["provider_posted_usd"] == "8.65756176"
    assert report["unreconciled_holds_usd"] == "11.34243824"
    assert report["conservative_liability_usd"] == "20.00000000"
    with pytest.raises(ValueError, match="complete posted billing"):
        reopened.reconcile_stage("wide", evidence_ref="still not settlement")
    reopened.record_billing("modal", "ap-gpu", total_usd="10.51503124", evidence_ref="later bill")
    report = reopened.report()["modal"]
    assert report["unreconciled_holds_usd"] == "11.34243824"
    assert report["conservative_liability_usd"] == "21.00000000"
    before = budget.path.read_bytes()
    with pytest.raises(ValueError, match="already refined/repriced"):
        reprice_completed(reopened, path, hold=hold)
    assert budget.path.read_bytes() == before


def test_remaining_liability_keeps_positive_lag_when_posted_exceeds_estimate(tmp_path):
    budget, path, proof = completed_lifetime_fixture(tmp_path)
    proof["conservative_lifecycle_estimate_usd"] = "1"
    hold = remaining_lifetime_proof(path, proof)
    assert hold == "2.4754"
    reprice_completed(budget, path, hold=hold)
    report = budget.report()["modal"]
    assert report["provider_posted_usd"] == "8.65756176"
    assert report["unreconciled_holds_usd"] == "2.4754"
    assert report["conservative_liability_usd"] == "11.13296176"


def test_remaining_liability_missing_bills_do_not_reduce_the_allowance(tmp_path):
    budget, path, proof = completed_lifetime_fixture(tmp_path, post_modal_billing=False)
    hold = remaining_lifetime_proof(path, proof)
    assert proof["posted_cost_deduction_usd"] == "0"
    assert hold == "20.0000"
    reprice_completed(budget, path, hold=hold)
    assert budget.report()["modal"]["unreconciled_holds_usd"] == "20.0000"
    with pytest.raises(ValueError, match="complete posted billing"):
        budget.reconcile_stage("wide", evidence_ref="missing billing is not free usage")


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 3),
        ("liability_basis", "billing_settled"),
        ("posted_cost_deduction_usd", "NaN"),
        ("posted_cost_deduction_usd", "-1"),
        ("posted_cost_deduction_usd", None),
        ("remaining_lifecycle_allowance_usd", "8"),
        ("remaining_lifecycle_allowance_usd", "Infinity"),
        ("unbilled_lag_usd", "0"),
    ],
)
def test_remaining_liability_invalid_arithmetic_or_schema_is_nonmutating(tmp_path, field, value):
    import json

    budget, path, proof = completed_lifetime_fixture(tmp_path)
    hold = remaining_lifetime_proof(path, proof)
    proof[field] = value
    path.write_text(json.dumps(proof))
    before = budget.path.read_bytes()
    with pytest.raises(ValueError):
        reprice_completed(budget, path, hold=hold)
    assert budget.path.read_bytes() == before


@pytest.mark.parametrize("deduction", ["0", "9.65756176"])
def test_remaining_liability_rechecks_deduction_against_locked_accounting(tmp_path, deduction):
    import json

    budget, path, proof = completed_lifetime_fixture(tmp_path)
    remaining_lifetime_proof(path, proof)
    proof["posted_cost_deduction_usd"] = deduction
    remaining = Decimal(proof["conservative_lifecycle_estimate_usd"]) - Decimal(deduction)
    proof["remaining_lifecycle_allowance_usd"] = str(remaining)
    path.write_text(json.dumps(proof))
    before = budget.path.read_bytes()
    with pytest.raises(ValueError, match="exact attributable posted cost"):
        reprice_completed(budget, path, hold=str(remaining + Decimal(proof["unbilled_lag_usd"])))
    assert budget.path.read_bytes() == before


@pytest.mark.parametrize("condition", ["later_bill", "baseline", "shared", "active", "missing"])
def test_remaining_liability_preserves_inventory_and_current_bill_requirements(tmp_path, condition):
    import json

    budget, path, proof = completed_lifetime_fixture(
        tmp_path, terminate=condition != "active", shared=condition == "shared"
    )
    hold = remaining_lifetime_proof(path, proof)
    if condition == "later_bill":
        budget.record_billing("modal", "ap-gpu", total_usd="10", evidence_ref="new bill")
    elif condition == "baseline":
        proof["resources"][0]["baseline_usd"] = "0"
        hold = remaining_lifetime_proof(path, proof)
    elif condition == "missing":
        proof["resources"].pop()
        hold = remaining_lifetime_proof(path, proof)
    path.write_text(json.dumps(proof))
    before = budget.path.read_bytes()
    with pytest.raises(ValueError):
        reprice_completed(budget, path, hold=hold)
    assert budget.path.read_bytes() == before


def migration_fixture(tmp_path, *, prior="schema1", post_modal_billing=True):
    import copy
    import hashlib
    import json

    budget, old_path, old = completed_lifetime_fixture(
        tmp_path, post_modal_billing=post_modal_billing
    )
    if prior == "schema1":
        reprice_completed(budget, old_path)
    elif prior == "schema2":
        hold = remaining_lifetime_proof(old_path, old)
        reprice_completed(budget, old_path, hold=hold)
    elif prior == "slack":
        slack = dict(old, resource_ids=[r["resource_id"] for r in old["resources"]])
        slack_path = tmp_path / "slack.json"
        slack_path.write_text(json.dumps(slack))
        budget.refine_stage_hold(
            "wide",
            "modal",
            hold_usd="110",
            evidence_path=slack_path,
            evidence_sha256=hashlib.sha256(slack_path.read_bytes()).hexdigest(),
        )
    event = json.loads(budget.path.read_text().splitlines()[-1])
    proof = copy.deepcopy(old)
    proof["migration"] = {
        "previous_event_sha256": event["sha256"],
        "previous_evidence_sha256": hashlib.sha256(old_path.read_bytes()).hexdigest(),
    }
    proof["evidence"].append(
        {
            "path": str(old_path.resolve()),
            "sha256": hashlib.sha256(old_path.read_bytes()).hexdigest(),
        }
    )
    path = tmp_path / "remaining-migration.json"
    hold = remaining_lifetime_proof(path, proof)
    return budget, path, proof, hold


def migrate_completed(budget, path, hold):
    import hashlib

    budget.migrate_completed_compute_hold_to_remaining_liability(
        "wide",
        "modal",
        hold_usd=hold,
        evidence_path=path,
        evidence_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def test_schema1_migration_preserves_original_allowance_history_and_unsettled_cost(tmp_path):
    import json

    budget, path, proof, hold = migration_fixture(tmp_path)
    prefix = budget.path.read_bytes()
    hud_before = budget.report()["hud"]
    migrate_completed(budget, path, hold)
    assert budget.path.read_bytes().startswith(prefix)
    event = json.loads(budget.path.read_text().splitlines()[-1])
    assert event["event"] == "migrate_completed_hold"
    assert event["previous_hold_usd"] == "20"
    assert event["conservative_lifecycle_estimate_usd"] == "17.5246"
    assert event["lag_usd"] == "2.4754"
    assert event["migration"] == proof["migration"]
    reopened = CampaignBudget(budget.path)
    assert reopened.report()["hud"] == hud_before
    report = reopened.report()["modal"]
    assert report["provider_posted_usd"] == "8.65756176"
    assert report["unreconciled_holds_usd"] == "11.34243824"
    assert report["conservative_liability_usd"] == "20.00000000"
    with pytest.raises(ValueError, match="complete posted billing"):
        reopened.reconcile_stage("wide", evidence_ref="not settlement")
    reopened.record_billing("modal", "ap-gpu", total_usd="10.51503124", evidence_ref="late")
    assert reopened.report()["modal"]["unreconciled_holds_usd"] == hold
    assert reopened.report()["modal"]["conservative_liability_usd"] == "21.00000000"
    before = budget.path.read_bytes()
    with pytest.raises(ValueError, match="unmigrated schema-1"):
        migrate_completed(reopened, path, hold)
    with pytest.raises(ValueError, match="explicit completed-hold schema migration"):
        reprice_completed(reopened, path, hold=hold)
    with pytest.raises(ValueError, match="already refined/repriced"):
        reprice_completed(reopened, tmp_path / "completed-proof.json")
    assert budget.path.read_bytes() == before


@pytest.mark.parametrize("prior", ["none", "slack", "schema2"])
def test_migration_requires_a_prior_completed_schema1_proof(tmp_path, prior):
    budget, path, _, hold = migration_fixture(tmp_path, prior=prior)
    before = budget.path.read_bytes()
    with pytest.raises(ValueError, match="unmigrated schema-1"):
        migrate_completed(budget, path, hold)
    assert budget.path.read_bytes() == before


@pytest.mark.parametrize(
    "mutation",
    [
        "event_hash",
        "proof_hash",
        "schema",
        "migration_keys",
        "estimate",
        "margin",
        "original",
        "old_reference",
        "proof_reference",
        "ownership",
        "termination",
        "baseline",
        "lower_bill",
        "evidence_changed",
        "unknown_creation",
        "extra_resource",
    ],
)
def test_migration_rejects_changed_original_proof_or_inventory(tmp_path, mutation):
    import json
    from pathlib import Path

    budget, path, proof, hold = migration_fixture(tmp_path)
    if mutation in {"event_hash", "proof_hash"}:
        key = "previous_event_sha256" if mutation == "event_hash" else "previous_evidence_sha256"
        proof["migration"][key] = "0" * 64
    elif mutation == "schema":
        proof["schema_version"] = 1
    elif mutation == "migration_keys":
        proof["migration"]["allow_repeat"] = True
    elif mutation in {"estimate", "margin", "original"}:
        key = {
            "estimate": "conservative_lifecycle_estimate_usd",
            "margin": "unbilled_lag_usd",
            "original": "original_planned_estimate_usd",
        }[mutation]
        proof[key] = str(Decimal(proof[key]) - Decimal("0.1"))
        hold = remaining_lifetime_proof(path, proof)
    elif mutation == "old_reference":
        proof["evidence"].pop(0)
    elif mutation == "proof_reference":
        proof["evidence"].pop()
    elif mutation == "ownership":
        proof["resources"][0]["resource_id"] = "ap-another"
    elif mutation == "termination":
        proof["resources"][0]["terminated_at"] = "999"
    elif mutation == "baseline":
        proof["resources"][0]["baseline_usd"] = "0"
        hold = remaining_lifetime_proof(path, proof)
    elif mutation == "lower_bill":
        proof["resources"][0]["recorded_total_usd"] = "9"
        hold = remaining_lifetime_proof(path, proof)
    elif mutation == "evidence_changed":
        Path(proof["evidence"][0]["path"]).write_text("changed source evidence")
    elif mutation == "unknown_creation":
        proof["unknown_resource_creation"] = True
    elif mutation == "extra_resource":
        proof["additional_billable_resources"] = ["unpriced-storage"]
    path.write_text(json.dumps(proof))
    before = budget.path.read_bytes()
    with pytest.raises(ValueError):
        migrate_completed(budget, path, hold)
    assert budget.path.read_bytes() == before


def test_migration_rechecks_locked_current_billing_and_allows_exact_later_bill(tmp_path):
    budget, path, proof, hold = migration_fixture(tmp_path)
    budget.record_billing("modal", "ap-gpu", total_usd="10.51503124", evidence_ref="new bill")
    before = budget.path.read_bytes()
    with pytest.raises(ValueError, match="stale or differs"):
        migrate_completed(budget, path, hold)
    assert budget.path.read_bytes() == before
    proof["resources"][0]["recorded_total_usd"] = "10.51503124"
    hold = remaining_lifetime_proof(path, proof)
    migrate_completed(budget, path, hold)
    assert budget.report()["modal"]["unreconciled_holds_usd"] == "10.34243824"


def test_migration_does_not_claim_missing_billing_is_free_or_release_unchanged_hold(tmp_path):
    budget, path, proof, hold = migration_fixture(tmp_path, post_modal_billing=False)
    before = budget.path.read_bytes()
    with pytest.raises(ValueError, match="hold must decrease"):
        migrate_completed(budget, path, hold)
    assert budget.path.read_bytes() == before
    budget.record_billing("modal", "ap-gpu", total_usd="2", evidence_ref="only one bill")
    proof["resources"][0]["recorded_total_usd"] = "2"
    hold = remaining_lifetime_proof(path, proof)
    migrate_completed(budget, path, hold)
    assert budget.report()["modal"]["unreconciled_holds_usd"] == "19.0000"
    with pytest.raises(ValueError, match="complete posted billing"):
        budget.reconcile_stage("wide", evidence_ref="other App bill still missing")


def test_migration_rejects_a_settled_stage(tmp_path):
    budget, path, _, hold = migration_fixture(tmp_path)
    for provider, identity, total in [
        ("modal", "ap-gpu", "9.51503124"),
        ("modal", "ap-gateway", "1.14253052"),
        ("hud", "instance-owned", "1.20"),
    ]:
        budget.record_billing(
            provider, identity, total_usd=total, settled_through="1000", evidence_ref="final"
        )
    budget.reconcile_stage("wide", evidence_ref="synthetic settlement")
    before = budget.path.read_bytes()
    with pytest.raises(ValueError, match="settled"):
        migrate_completed(budget, path, hold)
    assert budget.path.read_bytes() == before


def revision_fixture(tmp_path, *, prior="schema2", post_modal_billing=True, lag="0.50"):
    """Second reviewed revision of a completed compute hold: 2x -> 1.25x terminal lifetime."""
    import copy
    import hashlib
    import json

    budget, old_path, old = completed_lifetime_fixture(
        tmp_path, post_modal_billing=post_modal_billing
    )
    if prior == "schema1":
        reprice_completed(budget, old_path)
    elif prior == "schema2":
        hold = remaining_lifetime_proof(old_path, old)
        reprice_completed(budget, old_path, hold=hold)
    elif prior == "slack":
        slack = dict(old, resource_ids=[r["resource_id"] for r in old["resources"]])
        slack_path = tmp_path / "slack.json"
        slack_path.write_text(json.dumps(slack))
        budget.refine_stage_hold(
            "wide",
            "modal",
            hold_usd="110",
            evidence_path=slack_path,
            evidence_sha256=hashlib.sha256(slack_path.read_bytes()).hexdigest(),
        )
    event = json.loads(budget.path.read_text().splitlines()[-1])
    captures = []
    for index, captured in enumerate(("5000", "9000")):
        capture = tmp_path / f"billing-capture-{index}.json"
        capture.write_text(json.dumps({"fixture": "provider billing report", "at": captured}))
        captures.append(
            {
                "path": str(capture.resolve()),
                "sha256": hashlib.sha256(capture.read_bytes()).hexdigest(),
                "captured_unix_s": captured,
                "app_totals_usd": {
                    row["resource_id"]: row["recorded_total_usd"] for row in old["resources"]
                },
            }
        )
    proof = copy.deepcopy(old)
    proof.pop("migration", None)
    previous = Decimal(old["conservative_lifecycle_estimate_usd"])
    one_times = previous / Decimal("2")
    revised = one_times * Decimal("1.25")
    posted = sum(
        Decimal(row["recorded_total_usd"]) - Decimal(row["baseline_usd"])
        for row in old["resources"]
    )
    remaining = max(revised - posted, Decimal(0))
    proof.update(
        schema_version=3,
        liability_basis="remaining_after_exact_posted_cost",
        revision={
            "previous_event_sha256": event["sha256"],
            "previous_evidence_sha256": hashlib.sha256(old_path.read_bytes()).hexdigest(),
        },
        previous_conservative_lifecycle_estimate_usd=str(previous),
        previous_lifecycle_multiplier="2",
        one_times_lifetime_estimate_usd=str(one_times),
        revised_lifecycle_multiplier="1.25",
        conservative_lifecycle_estimate_usd=str(revised),
        posted_cost_deduction_usd=str(posted),
        remaining_lifecycle_allowance_usd=str(remaining),
        unbilled_lag_usd=lag,
        billing_captures=captures,
    )
    proof["evidence"] = old["evidence"] + [
        {
            "path": str(old_path.resolve()),
            "sha256": hashlib.sha256(old_path.read_bytes()).hexdigest(),
        }
    ]
    path = tmp_path / "revised-proof.json"
    path.write_text(json.dumps(proof))
    return budget, path, proof, str(remaining + Decimal(lag))


def revise_completed(budget, path, hold, *, digest=None):
    import hashlib

    budget.revise_completed_compute_hold(
        "wide",
        "modal",
        hold_usd=hold,
        evidence_path=path,
        evidence_sha256=digest or hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def test_revision_reduces_completed_hold_once_and_keeps_history(tmp_path):
    import json

    budget, path, proof, hold = revision_fixture(tmp_path)
    assert hold == "2.79531324"  # 8.7623 x 1.25 = 10.952875; minus posted 8.65756176; plus 0.50
    prefix = budget.path.read_bytes()
    hud_before = budget.report()["hud"]
    revise_completed(budget, path, hold)
    assert budget.path.read_bytes().startswith(prefix)
    event = json.loads(budget.path.read_text().splitlines()[-1])
    assert event["event"] == "revise_completed_hold"
    assert event["previous_hold_usd"] == "11.34243824"
    assert event["hold_usd"] == hold
    assert event["lag_usd"] == "0.50"
    assert event["conservative_lifecycle_estimate_usd"] == "10.952875"
    assert event["previous_conservative_lifecycle_estimate_usd"] == "17.5246"
    assert event["revised_lifecycle_multiplier"] == "1.25"
    assert event["posted_cost_deduction_usd"] == "8.65756176"
    assert event["revision"] == proof["revision"]
    assert [c["sha256"] for c in event["billing_captures"]] == [
        c["sha256"] for c in proof["billing_captures"]
    ]
    reopened = CampaignBudget(budget.path)
    assert reopened.report()["hud"] == hud_before
    report = reopened.report()["modal"]
    assert report["provider_posted_usd"] == "8.65756176"
    assert report["unreconciled_holds_usd"] == hold
    assert report["conservative_liability_usd"] == "11.45287500"
    with pytest.raises(ValueError, match="complete posted billing"):
        reopened.reconcile_stage("wide", evidence_ref="not settlement")
    reopened.record_billing("modal", "ap-gpu", total_usd="10.51503124", evidence_ref="late")
    assert reopened.report()["modal"]["unreconciled_holds_usd"] == hold
    assert reopened.report()["modal"]["conservative_liability_usd"] == "12.45287500"
    before = budget.path.read_bytes()
    with pytest.raises(ValueError, match="already revised"):
        revise_completed(reopened, path, hold)
    with pytest.raises(ValueError, match="already refined/repriced"):
        reprice_completed(reopened, tmp_path / "completed-proof.json", hold="11.34243824")
    with pytest.raises(ValueError, match="schema"):
        migrate_completed(reopened, path, hold)
    assert budget.path.read_bytes() == before


@pytest.mark.parametrize("prior", ["none", "slack", "schema1"])
def test_revision_requires_a_prior_exact_posted_cost_proof(tmp_path, prior):
    budget, path, _, hold = revision_fixture(tmp_path, prior=prior)
    before = budget.path.read_bytes()
    with pytest.raises(ValueError, match="prior exact-posted-cost"):
        revise_completed(budget, path, hold)
    assert budget.path.read_bytes() == before


def test_revision_accepts_a_migrated_schema1_stage_once(tmp_path):
    import hashlib
    import json

    budget, migrate_path, _, migrate_hold = migration_fixture(tmp_path)
    migrate_completed(budget, migrate_path, migrate_hold)
    old = json.loads(migrate_path.read_text())
    event = json.loads(budget.path.read_text().splitlines()[-1])
    _, path, proof, hold = revision_fixture(tmp_path / "second", prior="schema2")
    proof["revision"] = {
        "previous_event_sha256": event["sha256"],
        "previous_evidence_sha256": hashlib.sha256(migrate_path.read_bytes()).hexdigest(),
    }
    proof["evidence"] = old["evidence"] + [
        {
            "path": str(migrate_path.resolve()),
            "sha256": hashlib.sha256(migrate_path.read_bytes()).hexdigest(),
        }
    ]
    path.write_text(json.dumps(proof))
    revise_completed(budget, path, hold)
    assert budget.report()["modal"]["unreconciled_holds_usd"] == hold


@pytest.mark.parametrize(
    "mutation",
    [
        "event_hash",
        "proof_hash",
        "previous_estimate",
        "original",
        "one_times",
        "multiplier_not_lower",
        "multiplier_below_one",
        "revised_arithmetic",
        "remaining_arithmetic",
        "hold_arithmetic",
        "zero_lag",
        "hold_not_lower",
        "ownership",
        "termination",
        "baseline",
        "lower_bill",
        "missing_prior_reference",
        "single_capture",
        "capture_before_buffer",
        "captures_too_close",
        "capture_hash",
        "capture_total_differs",
        "capture_missing_app",
        "unknown_creation",
        "extra_resource",
        "schema",
        "basis",
    ],
)
def test_revision_rejects_changed_prior_evidence_or_inventory(tmp_path, mutation):
    import hashlib
    import json
    from pathlib import Path

    budget, path, proof, hold = revision_fixture(tmp_path)
    digest = None
    if mutation == "event_hash":
        proof["revision"]["previous_event_sha256"] = "0" * 64
    elif mutation == "proof_hash":
        proof["revision"]["previous_evidence_sha256"] = "0" * 64
    elif mutation == "previous_estimate":
        proof["previous_conservative_lifecycle_estimate_usd"] = "17.5245"
    elif mutation == "original":
        proof["original_planned_estimate_usd"] = "109.62128731"
    elif mutation == "one_times":
        proof["one_times_lifetime_estimate_usd"] = "8.7622"
    elif mutation == "multiplier_not_lower":
        proof["revised_lifecycle_multiplier"] = "2"
        proof["conservative_lifecycle_estimate_usd"] = "17.5246"
        proof["remaining_lifecycle_allowance_usd"] = "8.86703824"
        hold = "9.36703824"
    elif mutation == "multiplier_below_one":
        proof["revised_lifecycle_multiplier"] = "0.5"
        proof["conservative_lifecycle_estimate_usd"] = "4.38115"
        proof["remaining_lifecycle_allowance_usd"] = "0"
        hold = "0.50"
    elif mutation == "revised_arithmetic":
        proof["conservative_lifecycle_estimate_usd"] = "10.95"
    elif mutation == "remaining_arithmetic":
        proof["remaining_lifecycle_allowance_usd"] = "2.2"
    elif mutation == "hold_arithmetic":
        hold = "2.80"
    elif mutation == "zero_lag":
        proof["unbilled_lag_usd"] = "0"
        hold = "2.29531324"
    elif mutation == "hold_not_lower":
        proof["unbilled_lag_usd"] = "9.50"
        hold = "11.79531324"
    elif mutation == "ownership":
        proof["resources"] = proof["resources"][:1]
        for capture in proof["billing_captures"]:
            capture["app_totals_usd"] = {"ap-gpu": capture["app_totals_usd"]["ap-gpu"]}
    elif mutation == "termination":
        proof["resources"][0]["terminated_at"] = "999"
    elif mutation == "baseline":
        proof["resources"][0]["baseline_usd"] = "0"
    elif mutation == "lower_bill":
        proof["resources"][0]["recorded_total_usd"] = "9"
        for capture in proof["billing_captures"]:
            capture["app_totals_usd"]["ap-gpu"] = "9"
    elif mutation == "missing_prior_reference":
        proof["evidence"] = proof["evidence"][:1]
    elif mutation == "single_capture":
        proof["billing_captures"] = proof["billing_captures"][:1]
    elif mutation == "capture_before_buffer":
        proof["billing_captures"][0]["captured_unix_s"] = "4000"
    elif mutation == "captures_too_close":
        proof["billing_captures"][1]["captured_unix_s"] = "8000"
    elif mutation == "capture_hash":
        proof["billing_captures"][1]["sha256"] = "0" * 64
    elif mutation == "capture_total_differs":
        proof["billing_captures"][1]["app_totals_usd"]["ap-gpu"] = "9.51503125"
    elif mutation == "capture_missing_app":
        del proof["billing_captures"][1]["app_totals_usd"]["ap-gateway"]
    elif mutation == "unknown_creation":
        proof["unknown_resource_creation"] = True
    elif mutation == "extra_resource":
        proof["resources"].append(dict(proof["resources"][0], resource_id="ap-other"))
    elif mutation == "schema":
        proof["schema_version"] = 2
    elif mutation == "basis":
        proof["liability_basis"] = "settled"
    path.write_text(json.dumps(proof))
    if mutation == "proof_hash":
        pass
    before = budget.path.read_bytes()
    with pytest.raises(ValueError):
        revise_completed(budget, path, hold, digest=digest)
    assert budget.path.read_bytes() == before
    assert Path(path).exists()


def test_revision_rechecks_locked_current_billing(tmp_path):
    budget, path, _, hold = revision_fixture(tmp_path)
    budget.record_billing("modal", "ap-gpu", total_usd="9.6", evidence_ref="later bill")
    before = budget.path.read_bytes()
    with pytest.raises(ValueError, match="stale or differs"):
        revise_completed(budget, path, hold)
    assert budget.path.read_bytes() == before


def test_revision_keeps_full_revised_allowance_when_billing_is_missing(tmp_path):
    budget, path, _, hold = revision_fixture(tmp_path, post_modal_billing=False)
    assert hold == "11.452875"  # No posted cost above baseline: full 1.25x allowance plus lag.
    revise_completed(budget, path, hold)
    report = budget.report()["modal"]
    assert report["provider_posted_usd"] == "0"
    assert report["unreconciled_holds_usd"] == hold


def test_revision_rejects_a_settled_stage(tmp_path):
    budget, path, _, hold = revision_fixture(tmp_path)
    for identity in ("ap-gpu", "ap-gateway"):
        budget.record_billing(
            "modal", identity, total_usd=budget_total(budget, identity), evidence_ref="final", settled_through="1000"
        )
    budget.record_billing("hud", "instance-owned", total_usd="1.20", evidence_ref="final", settled_through="1000")
    budget.reconcile_stage("wide", evidence_ref="settled")
    before = budget.path.read_bytes()
    with pytest.raises(ValueError, match="settled"):
        revise_completed(budget, path, hold)
    assert budget.path.read_bytes() == before


def budget_total(budget, identity):
    import json

    total = "0"
    for line in budget.path.read_text().splitlines():
        row = json.loads(line)
        if row.get("event") == "billing" and row["resource"] == f"modal:{identity}":
            total = row["total_usd"]
    return total
