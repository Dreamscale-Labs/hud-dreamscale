"""Private campaign accounting from explicit rates and provider billing evidence.

No cloud calls or grants are made here. The server's finite operator reservation
still applies independently. App billing is authoritative only for that app;
allocating it among deployment lifetimes produces an estimate, not billed rows.
"""

import fcntl
import hashlib
import json
import os
from decimal import Decimal, InvalidOperation
from pathlib import Path

PROVIDERS = ("modal", "hud")
PHASES = ("startup", "idle", "run", "cleanup")


class BudgetExceededError(RuntimeError):
    pass


def amount(value):
    """Use API decimal strings; never silently import binary floating-point money."""
    if isinstance(value, bool) or not isinstance(value, (str, int, Decimal)):
        raise ValueError("Amounts and durations must be Decimal, decimal strings or integers")
    try:
        result = Decimal(value)
    except InvalidOperation as exc:
        raise ValueError("Invalid decimal amount") from exc
    if not result.is_finite() or result < 0:
        raise ValueError("Amounts and durations must be finite and nonnegative")
    return result


def estimate_stage(
    *,
    concurrency,
    startup_seconds,
    idle_seconds,
    run_seconds,
    cleanup_seconds,
    modal_replica_usd_per_second,
    modal_overhead_usd_per_second,
    hud_hourly_rate,
):
    """Reserve full rounded GPU capacity and every HUD environment for all phases.

    Supply HUD's current instance.hourly_rate; no price is embedded here. These
    rate-duration products are estimates. They are never relabeled provider spend.
    """
    if type(concurrency) is not int or not 1 <= concurrency <= 64:
        raise ValueError("concurrency must be an integer from 1 through 64")
    seconds = dict(
        zip(
            PHASES,
            map(
                amount,
                (
                    startup_seconds,
                    idle_seconds,
                    run_seconds,
                    cleanup_seconds,
                ),
            ),
            strict=True,
        )
    )
    if sum(seconds.values()) <= 0:
        raise ValueError("A stage must reserve a positive lifetime")
    replicas = (concurrency + 7) // 8
    gpu_rate = amount(modal_replica_usd_per_second) * replicas
    overhead_rate = amount(modal_overhead_usd_per_second)
    modal_rate = gpu_rate + overhead_rate
    hud_rate = amount(hud_hourly_rate) * concurrency / Decimal(3600)
    return {
        "concurrency": concurrency,
        "replicas": replicas,
        "seconds": {key: str(value) for key, value in seconds.items()},
        "rates_usd_per_second": {
            "modal_gpu": str(gpu_rate),
            "modal_overhead": str(overhead_rate),
            "hud_all_environments": str(hud_rate),
        },
        "gpu_only_estimated_usd": str(gpu_rate * sum(seconds.values())),
        "estimated_usd": {
            provider: {phase: str(duration * rate) for phase, duration in seconds.items()}
            for provider, rate in (("modal", modal_rate), ("hud", hud_rate))
        },
    }


def allocate_app_cost(*, posted_usd, app_lifetime_seconds, cohort_seconds):
    """Time-weighted estimates, retaining lifetime outside cohorts as unattributed.

    This does not model autoscaled replica counts or infer actual deployment bills.
    Include startup, idle, execution and cleanup in each supplied cohort lifetime.
    """
    total, lifetime = amount(posted_usd), amount(app_lifetime_seconds)
    if lifetime <= 0 or "unattributed" in cohort_seconds:
        raise ValueError("A positive app lifetime and unreserved cohort names are required")
    weights = {key: amount(value) for key, value in cohort_seconds.items()}
    if sum(weights.values()) > lifetime:
        raise ValueError("Cohort lifetimes exceed the billing window or overlap")
    weights["unattributed"] = lifetime - sum(weights.values())
    allocations = {key: total * duration / lifetime for key, duration in weights.items()}
    # Preserve the exact posted total despite Decimal division rounding.
    allocations["unattributed"] += total - sum(allocations.values())
    return {
        "basis": "estimated_time_allocation_of_app_billing",
        "posted_app_usd": str(total),
        "estimated_cohort_usd": {key: str(value) for key, value in allocations.items()},
    }


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _read(file):
    file.seek(0)
    rows, previous = [], ""
    for line in file:
        row = json.loads(line)
        digest = row.pop("sha256")
        if row.get("sequence") != len(rows) or row.get("previous_sha256") != previous:
            raise ValueError("Campaign ledger sequence was modified")
        if hashlib.sha256(_json(row).encode()).hexdigest() != digest:
            raise ValueError("Campaign ledger digest mismatch")
        previous = digest
        rows.append({**row, "sha256": digest})
    if not rows or rows[0].get("event") != "campaign":
        raise ValueError("Campaign ledger has no immutable header")
    return rows


def _state(rows):
    state = {"caps": rows[0]["caps_usd"], "stages": {}, "resources": {}}
    for row in rows[1:]:
        event = row["event"]
        if event == "reserve":
            state["stages"][row["stage"]] = {**row, "released": False, "resources": []}
        elif event == "resource":
            state["resources"][row["resource"]] = {
                **row,
                "total_usd": row["baseline_usd"],
                "terminated_at": None,
                "settled_through": None,
            }
        elif event == "bind":
            state["stages"][row["stage"]]["resources"].append(row["resource"])
        elif event == "billing":
            state["resources"][row["resource"]].update(row)
        elif event == "terminated":
            state["resources"][row["resource"]]["terminated_at"] = row["terminated_at"]
        elif event == "reconcile":
            state["stages"][row["stage"]]["released"] = True
    return state


def _report(state):
    result = {}
    for provider in PROVIDERS:
        posted = sum(
            (
                amount(row["total_usd"]) - amount(row["baseline_usd"])
                for row in state["resources"].values()
                if row["provider"] == provider
            ),
            Decimal(0),
        )
        held = sum(
            (
                amount(row["holds_usd"][provider])
                for row in state["stages"].values()
                if not row["released"]
            ),
            Decimal(0),
        )
        cap = state["caps"][provider]
        liability = posted + held
        result[provider] = {
            "cap_usd": cap,
            "provider_posted_usd": str(posted),
            "unreconciled_holds_usd": str(held),
            "conservative_liability_usd": str(liability),
            "remaining_usd": None if cap is None else str(amount(cap) - liability),
            "over_cap": cap is not None and liability > amount(cap),
        }
    return result


class CampaignBudget:
    """Hash-linked append-only evidence; reopening preserves every prior liability.

    Until billing is settled, posted cost plus the full pending reservation is a
    conservative bound and may overlap. Stopping resources alone never releases
    that hold. Unknown HUD credit balance prevents reserving any paid stage.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.report()  # Reject missing/truncated/edited ledgers before use.

    @classmethod
    def create(cls, path, *, modal_cap_usd, hud_cap_usd=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        modal = amount(modal_cap_usd)
        if modal <= 0:
            raise ValueError("Modal campaign cap must be positive")
        caps = {
            "modal": str(modal),
            "hud": None if hud_cap_usd is None else str(amount(hud_cap_usd)),
        }
        row = {"event": "campaign", "caps_usd": caps, "sequence": 0, "previous_sha256": ""}
        row["sha256"] = hashlib.sha256(_json(row).encode()).hexdigest()
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w") as file:
            file.write(_json(row) + "\n")
            file.flush()
            os.fsync(file.fileno())
        return cls(path)

    def _append(self, row, validate):
        fd = os.open(self.path, os.O_RDWR | os.O_APPEND)
        with os.fdopen(fd, "a+") as file:
            fcntl.flock(file, fcntl.LOCK_EX)
            rows = _read(file)
            validate(_state(rows))
            row = {**row, "sequence": len(rows), "previous_sha256": rows[-1]["sha256"]}
            row["sha256"] = hashlib.sha256(_json(row).encode()).hexdigest()
            file.write(_json(row) + "\n")
            file.flush()
            os.fsync(file.fileno())

    def report(self):
        with self.path.open() as file:
            fcntl.flock(file, fcntl.LOCK_SH)
            return _report(_state(_read(file)))

    def reserve_stage(self, stage, *, estimate, modal_lag_usd, hud_lag_usd, evidence_ref):
        if not stage or not evidence_ref:
            raise ValueError("Stage and rate evidence are required")
        holds = {}
        for provider, lag in (("modal", modal_lag_usd), ("hud", hud_lag_usd)):
            components = estimate["estimated_usd"][provider]
            if set(components) != set(PHASES):
                raise ValueError("Reservation must include startup, idle, run and cleanup")
            holds[provider] = str(sum(map(amount, components.values())) + amount(lag))

        def validate(state):
            if stage in state["stages"]:
                raise ValueError("Stage already has a reservation; use a new cohort identity")
            for provider, current in _report(state).items():
                if current["cap_usd"] is None:
                    raise BudgetExceededError(f"{provider} balance/cap is unknown")
                if amount(current["conservative_liability_usd"]) + amount(holds[provider]) > amount(
                    current["cap_usd"]
                ):
                    raise BudgetExceededError(f"{provider} campaign liability would exceed its cap")

        self._append(
            {
                "event": "reserve",
                "stage": stage,
                "estimate": estimate,
                "holds_usd": holds,
                "lag_usd": {"modal": str(amount(modal_lag_usd)), "hud": str(amount(hud_lag_usd))},
                "evidence_ref": evidence_ref,
            },
            validate,
        )

    def register_owned_resource(self, provider, resource_id, *, baseline_usd, evidence_ref):
        """Register an exact owned app/instance ID and pre-campaign billing baseline."""
        if provider not in PROVIDERS or not resource_id or not evidence_ref:
            raise ValueError(
                "Provider, exact resource identity and ownership evidence are required"
            )
        key = f"{provider}:{resource_id}"

        def validate(state):
            if key in state["resources"]:
                raise ValueError("Resource already registered; its billing baseline cannot reset")

        self._append(
            {
                "event": "resource",
                "provider": provider,
                "resource": key,
                "baseline_usd": str(amount(baseline_usd)),
                "evidence_ref": evidence_ref,
            },
            validate,
        )

    def bind_resource(self, stage, provider, resource_id):
        key = f"{provider}:{resource_id}"

        def validate(state):
            if key not in state["resources"] or stage not in state["stages"]:
                raise ValueError("Register resource and reserve stage before binding")
            row = state["stages"][stage]
            if row["released"] or key in row["resources"]:
                raise ValueError("Stage is reconciled or resource already bound")
            if state["resources"][key]["terminated_at"] is not None:
                raise ValueError("Cannot bind a resource already confirmed terminated")

        self._append({"event": "bind", "stage": stage, "resource": key}, validate)

    def record_billing(
        self, provider, resource_id, *, total_usd, evidence_ref, settled_through=None
    ):
        """Append provider-posted cumulative billing, even if it exposes an overrun.

        settled_through is an explicitly evidenced billing-coverage timestamp, NOT
        the time an eventually-consistent usage endpoint happened to be queried.
        Leave it None when the provider has not established complete coverage.
        """
        key, total = f"{provider}:{resource_id}", amount(total_usd)
        settled = None if settled_through is None else str(amount(settled_through))
        if not evidence_ref:
            raise ValueError("Provider billing evidence is required")

        def validate(state):
            resource = state["resources"].get(key)
            if resource is None:
                raise ValueError("Billing must belong to an exact registered resource")
            if total < amount(resource["total_usd"]):
                raise ValueError("Cumulative posted cost decreased; reconcile the source first")

        self._append(
            {
                "event": "billing",
                "resource": key,
                "total_usd": str(total),
                "settled_through": settled,
                "evidence_ref": evidence_ref,
            },
            validate,
        )

    def confirm_termination(self, provider, resource_id, *, terminated_at, evidence_ref):
        key = f"{provider}:{resource_id}"
        if not evidence_ref:
            raise ValueError("Confirmed termination evidence is required")

        def validate(state):
            if key not in state["resources"]:
                raise ValueError("Cannot terminate an unregistered resource")
            if state["resources"][key]["terminated_at"] is not None:
                raise ValueError("Termination was already recorded")

        self._append(
            {
                "event": "terminated",
                "resource": key,
                "terminated_at": str(amount(terminated_at)),
                "evidence_ref": evidence_ref,
            },
            validate,
        )

    def reconcile_stage(self, stage, *, evidence_ref):
        """Release holds only after every funded provider and bound resource settles.

        Single-provider stages need no dummy resource for an unused provider.
        A zero hold never exempts an actually bound resource from verification.
        """
        if not evidence_ref:
            raise ValueError("Reconciliation evidence is required")

        def validate(state):
            row = state["stages"].get(stage)
            if row is None or row["released"]:
                raise ValueError("Stage is missing or already reconciled")
            resources = [state["resources"][key] for key in row["resources"]]
            if not resources:
                raise ValueError("Bound resource evidence is required")
            funded = {provider for provider in PROVIDERS if amount(row["holds_usd"][provider]) > 0}
            missing = funded - {resource["provider"] for resource in resources}
            if missing:
                raise ValueError(
                    "Funded providers require bound resource evidence: "
                    + ", ".join(sorted(missing))
                )
            for resource in resources:
                stopped, settled = resource["terminated_at"], resource["settled_through"]
                if stopped is None or settled is None or amount(settled) < amount(stopped):
                    raise ValueError("Termination and complete posted billing are not confirmed")

        self._append({"event": "reconcile", "stage": stage, "evidence_ref": evidence_ref}, validate)
