"""Hold refinement preserves spending, unsettled liability and evidence boundaries."""

import hashlib
import json
from decimal import Decimal

import pytest

from hud_dropbear.costs import CampaignBudget, _read, _state, estimate_stage


def raw_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def setup(tmp_path, *, terminated=True):
    ledger = CampaignBudget.create(
        tmp_path / "ledger.jsonl", modal_cap_usd="200", hud_cap_usd="100"
    )
    estimate = estimate_stage(
        concurrency=1,
        startup_seconds=240,
        idle_seconds=0,
        run_seconds=0,
        cleanup_seconds=120,
        modal_replica_usd_per_second="0",
        modal_overhead_usd_per_second="0.00005362",
        hud_hourly_rate="0",
    )
    ledger.reserve_stage(
        "cpu",
        estimate=estimate,
        modal_lag_usd="1.98069680",
        hud_lag_usd="0",
        evidence_ref="original-rate-receipt",
    )
    ledger.register_owned_resource("modal", "ap-owned", baseline_usd="0", evidence_ref="ownership")
    ledger.bind_resource("cpu", "modal", "ap-owned")
    if terminated:
        ledger.confirm_termination("modal", "ap-owned", terminated_at="1000", evidence_ref="stop")
    ledger.record_billing("modal", "ap-owned", total_usd="0.00312024", evidence_ref="posted-cost")
    provider = tmp_path / "provider-ack-and-stop.json"
    provider.write_text('{"fixture":"acknowledged hard CPU/RAM limits and exact termination"}')
    proof = dict(
        schema_version=1,
        stage="cpu",
        provider="modal",
        resource_ids=["ap-owned"],
        resource_inventory_complete=True,
        unknown_resource_creation=False,
        additional_billable_resources=[],
        basis="Reviewed full-lifetime bounded CPU-only operation",
        evidence=[{"path": str(provider), "sha256": raw_sha(provider)}],
    )
    path = tmp_path / "proof.json"
    path.write_text(json.dumps(proof))
    return ledger, path, proof


def refine(ledger, path, **kwargs):
    ledger.refine_stage_hold(
        "cpu",
        "modal",
        hold_usd=kwargs.pop("hold_usd", "0.25"),
        evidence_path=path,
        evidence_sha256=kwargs.pop("evidence_sha256", raw_sha(path)),
        **kwargs,
    )


def test_refinement_only_appends_and_never_settles_spend(tmp_path):
    ledger, path, _ = setup(tmp_path)
    original = ledger.path.read_bytes()
    refine(ledger, path)
    assert ledger.path.read_bytes().startswith(original)
    rows = [json.loads(line) for line in ledger.path.read_text().splitlines()]
    assert len(rows) == len(original.splitlines()) + 1
    assert rows[1]["holds_usd"]["modal"] == "2.00000000"
    assert rows[-1]["event"] == "refine_hold"
    assert rows[-1]["previous_hold_usd"] == "2.00000000"
    assert rows[-1]["lag_usd"] == "0.23069680"
    assert rows[-1]["evidence_sha256"] == raw_sha(path)
    report = CampaignBudget(ledger.path).report()["modal"]
    assert report["provider_posted_usd"] == "0.00312024"
    assert report["unreconciled_holds_usd"] == "0.25"
    assert report["conservative_liability_usd"] == "0.25312024"
    with ledger.path.open() as f:
        state = _state(_read(f))
    assert not state["stages"]["cpu"]["released"]
    assert state["resources"]["modal:ap-owned"]["settled_through"] is None
    with pytest.raises(ValueError, match="complete posted billing"):
        ledger.reconcile_stage("cpu", evidence_ref="not settled")
    # Future billing is still added in full; refinement is not a billing credit.
    ledger.record_billing("modal", "ap-owned", total_usd="0.10", evidence_ref="later posted")
    assert Decimal(ledger.report()["modal"]["conservative_liability_usd"]) == Decimal("0.35")


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("stage", "other"),
        ("provider", "hud"),
        ("resource_ids", []),
        ("resource_ids", ["ap-other"]),
        ("resource_ids", ["ap-owned", "ap-owned"]),
        ("resource_inventory_complete", False),
        ("unknown_resource_creation", True),
        ("additional_billable_resources", ["volume-unaccounted"]),
        ("basis", ""),
        ("evidence", []),
    ],
)
def test_unknown_or_incomplete_scope_cannot_reduce_hold(tmp_path, field, value):
    ledger, path, proof = setup(tmp_path)
    proof[field] = value
    path.write_text(json.dumps(proof))
    before = ledger.path.read_bytes()
    with pytest.raises(ValueError):
        refine(ledger, path)
    assert ledger.path.read_bytes() == before


@pytest.mark.parametrize("hold", ["0.01930319", "2", "3", "-1", "NaN"])
def test_original_full_lifetime_estimate_is_a_floor(tmp_path, hold):
    ledger, path, _ = setup(tmp_path)
    before = ledger.path.read_bytes()
    with pytest.raises(ValueError):
        refine(ledger, path, hold_usd=hold)
    assert ledger.path.read_bytes() == before


def test_unterminated_or_newly_bound_resource_rejects_refinement(tmp_path):
    ledger, path, _ = setup(tmp_path, terminated=False)
    before = ledger.path.read_bytes()
    with pytest.raises(ValueError, match="termination"):
        refine(ledger, path)
    assert ledger.path.read_bytes() == before
    ledger.confirm_termination("modal", "ap-owned", terminated_at="1000", evidence_ref="stop")
    ledger.register_owned_resource("modal", "ap-extra", baseline_usd="0", evidence_ref="ownership")
    ledger.bind_resource("cpu", "modal", "ap-extra")
    ledger.confirm_termination("modal", "ap-extra", terminated_at="1000", evidence_ref="stop")
    before = ledger.path.read_bytes()
    with pytest.raises(ValueError, match="Exact bound"):
        refine(ledger, path)
    assert ledger.path.read_bytes() == before


@pytest.mark.parametrize("target", ["proof", "referenced"])
def test_changed_content_addressed_evidence_is_rejected(tmp_path, target):
    ledger, path, proof = setup(tmp_path)
    expected = raw_sha(path)
    if target == "proof":
        path.write_text(path.read_text() + " ")
    else:
        from pathlib import Path

        Path(proof["evidence"][0]["path"]).write_text("changed evidence")
    before = ledger.path.read_bytes()
    with pytest.raises(ValueError, match="differs"):
        refine(ledger, path, evidence_sha256=expected)
    assert ledger.path.read_bytes() == before


def test_refined_stage_cannot_be_reopened_or_rebound(tmp_path):
    ledger, path, _ = setup(tmp_path)
    refine(ledger, path)
    before = ledger.path.read_bytes()
    with pytest.raises(ValueError, match="already refined"):
        refine(ledger, path, hold_usd="0.20")
    assert ledger.path.read_bytes() == before
    ledger.register_owned_resource("modal", "ap-new", baseline_usd="0", evidence_ref="ownership")
    before = ledger.path.read_bytes()
    with pytest.raises(ValueError, match="refined"):
        ledger.bind_resource("cpu", "modal", "ap-new")
    assert ledger.path.read_bytes() == before
    with ledger.path.open() as f:
        original = _state(_read(f))["stages"]["cpu"]["estimate"]
    with pytest.raises(ValueError, match="already has a reservation"):
        ledger.reserve_stage(
            "cpu", estimate=original, modal_lag_usd="0", hud_lag_usd="0", evidence_ref="reuse"
        )
    assert ledger.path.read_bytes() == before
