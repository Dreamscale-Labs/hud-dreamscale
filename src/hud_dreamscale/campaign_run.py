"""Run one reserved production VLA campaign cell; never choose a fallback.

python -m hud_dreamscale.campaign_run --config cell.json --ledger costs.json --output new-directory
The caller supplies immutable release pins and a cost estimate based on qualification.
Liability remains reserved until provider billing and exact cleanup are reconciled.
"""

import argparse
import asyncio
import fcntl
import json
import math
import os
import time
from pathlib import Path
from uuid import uuid4

from .campaign import SCALING, check_budget, design, digest, qualify, result_tables, task_rows
from .campaign_metrics import summarize as summarize_metrics
from .cohort import run_cohort, summarize_cohort
from .pooled import PooledProvider
from .pooled_agent import PooledRobotAgent
from .runtime_pool import RuntimePool
from .telemetry import Evidence

PRODUCTION = "https://api.dreamscalelabs.com"


def verify_hud_build(config):
    from hud.utils.platform import PlatformClient

    row = PlatformClient.from_settings().get("/registry/" + config["hud_environment_id"])
    build = row.get("latest_build") or {}
    if (
        row.get("name") != config["hud_environment_name"]
        or build.get("id") != config["hud_environment_revision"]
        or build.get("status") != "succeeded"
    ):
        raise ValueError("production HUD environment build differs from the frozen cell")
    return {
        "id": row["id"],
        "name": row["name"],
        "build_id": build["id"],
        "manifest_sha256": digest(build.get("manifest")),
    }


def preflight(config, ledger):
    if config["design_sha256"] != digest(design()):
        raise ValueError("cell does not match the frozen study")
    if config.get("api_base") != PRODUCTION:
        raise ValueError("this campaign requires production Dreamscale")
    allocation = (config["concurrency"], config["h100s"], config["robots_per_replica"])
    if allocation not in SCALING:
        raise ValueError("unplanned concurrency/GPU allocation")
    if config["kind"] not in ("qualification", "scaling", "scored"):
        raise ValueError("unknown cell kind")
    if config["kind"] == "scored" and config["concurrency"] != 128:
        raise ValueError("headline fallback requires human approval and a new frozen design")
    for field in (
        "release_id",
        "release_sha256",
        "hud_environment_revision",
        "simulator_image_digest",
        "sdk_version",
        "hud_environment_id",
        "hud_environment_name",
    ):
        if not config.get(field) or "pending" in config[field].lower():
            raise ValueError(f"immutable pin required: {field}")
    for field in ("max_run_seconds", "hourly_infrastructure_usd", "cleanup_reserve_usd"):
        value = config[field]
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"positive finite bound required: {field}")
    liability = (
        config["max_run_seconds"] * config["hourly_infrastructure_usd"] / 3600
        + config["cleanup_reserve_usd"]
    )
    category = "qualification" if config["kind"] == "qualification" else "vla"
    check_budget(ledger, category=category, estimated_usd=liability)
    if category == "vla":
        proof_path = config.get("qualification_evidence")
        if not proof_path:
            raise ValueError("qualified configuration evidence required before measurement")
        proof = json.loads(Path(proof_path).read_text())
        if digest(proof) != config.get("qualification_evidence_sha256"):
            raise ValueError("qualification evidence digest differs")
        if not qualify(proof["summary"]):
            raise ValueError("qualification evidence does not pass infrastructure gates")
        for key in (
            "model",
            "release_id",
            "release_sha256",
            "concurrency",
            "h100s",
            "robots_per_replica",
            "hud_environment_revision",
            "simulator_image_digest",
        ):
            if proof["config"].get(key) != config.get(key):
                raise ValueError(f"qualification differs from measured cell: {key}")
        # Re-estimate the WHOLE remaining study, not only a cheap first cell.
        if config.get("remaining_study_estimate_usd", 0) < liability:
            raise ValueError("remaining study estimate must include this cell")
        check_budget(ledger, category="vla", estimated_usd=config["remaining_study_estimate_usd"])
    return category, liability


async def run(config, output):
    from importlib.metadata import distribution, version

    from dreamscale.config import load_config
    from hud import HUDRuntime
    from hud.eval.job import Job
    from hud.settings import settings

    direct = distribution("dreamscale").read_text("direct_url.json")
    if direct and json.loads(direct).get("url", "").startswith("file:"):
        raise ValueError("live campaign requires the published SDK, not a local source override")
    if version("dreamscale") != config["sdk_version"]:
        raise ValueError("installed SDK differs from the pinned campaign release")
    if str(settings.hud_web_url).rstrip("/") != "https://www.hud.ai":
        raise ValueError("HUD must use the production platform")
    account = load_config()
    if not account.api_key or account.control_plane_url.rstrip("/") != PRODUCTION:
        raise ValueError("sign in with the production Dreamscale account")
    rows = task_rows(
        config["model"], concurrency=config["concurrency"], scored=config["kind"] == "scored"
    )
    build_before = await asyncio.to_thread(verify_hud_build, config)
    (output / "hud-build-before.json").write_text(json.dumps(build_before, indent=2) + "\n")
    rows = [row.model_copy(update={"env": config["hud_environment_name"]}) for row in rows]
    evidence = Evidence(output / "events.jsonl")
    provider = PooledProvider(
        model=config["model"],
        concurrency=config["concurrency"],
        robots_per_replica=config["robots_per_replica"],
        api_key=account.api_key,
        api_base=PRODUCTION,
        emit=evidence.emit,
        journal_path=output / "ownership.json",
        expected_release_id=config["release_id"],
        expected_release_sha256=config["release_sha256"],
        request_timeout=15,
    )
    runtimes = RuntimePool(
        HUDRuntime(),
        rows,
        concurrency=config["concurrency"],
        emit=evidence.emit,
        lane_ready_attempts=1,
    )
    job = None
    error = None
    started = time.monotonic()
    try:
        async with asyncio.timeout(config["max_run_seconds"]):
            async with provider, runtimes:
                job = await Job.start(f"dreamscale-hud-{config['kind']}-{config['model']}")
                agent = PooledRobotAgent(provider=provider, runtimes=runtimes, emit=evidence.emit)
                # Scored cells have an explicit barrier between initial-state waves.
                waves = (
                    [rows[i : i + 128] for i in range(0, len(rows), 128)]
                    if config["kind"] == "scored"
                    else [rows]
                )
                for index, wave in enumerate(waves):
                    evidence.emit("wave_start", wave=index, episodes=len(wave))
                    await run_cohort(
                        agent, wave, runtime=runtimes, job=job, concurrency=config["concurrency"]
                    )
                    evidence.emit("wave_end", wave=index)
    except BaseException as exc:
        error = type(exc).__name__
        raise
    finally:
        summary = summarize_cohort(job, rows, evidence.rows)
        summary.update(
            model=config["model"],
            kind=config["kind"],
            requested_concurrency=config["concurrency"],
            requested_h100s=config["h100s"],
            elapsed_seconds=time.monotonic() - started,
            error=error,
            identity_audited=False,
            hardware_audited=False,
            simulator_contexts_closed=runtimes.cleanup_confirmed,
            simulator_cleanup_confirmed=False,
            inference_cleanup_confirmed=provider.cleanup_confirmed,
            # HUDRuntime currently suppresses DELETE errors. Returning from its
            # context manager alone cannot prove simulator/provider termination.
            cleanup_confirmed=False,
        )
        try:
            after = await asyncio.to_thread(verify_hud_build, config)
            summary["hud_build_unchanged"] = after == build_before
        except Exception as exc:
            summary["hud_build_unchanged"] = False
            summary["hud_build_audit_error"] = type(exc).__name__
        (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        (output / "results.json").write_text(
            json.dumps(result_tables(summary["runs"]), indent=2) + "\n"
        )
        (output / "metrics.json").write_text(
            json.dumps(summarize_metrics(evidence.rows), indent=2) + "\n"
        )
        evidence.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    # Cross-process campaign lock prevents overlapping fleets/spend reservations.
    with args.ledger.with_suffix(".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        ledger = json.loads(args.ledger.read_text()) if args.ledger.exists() else []
        category, liability = preflight(config, ledger)
        args.output.mkdir(parents=True, exist_ok=False)
        ledger.append(
            {
                "attempt": uuid4().hex,
                "category": category,
                "liability_usd": liability,
                "config_sha256": digest(config),
                "output": str(args.output.resolve()),
                "status": "reserved",
                "created_at": time.time(),
            }
        )
        temporary = args.ledger.with_suffix(".new")
        with temporary.open("w") as stream:
            json.dump(ledger, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, args.ledger)
        (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
        asyncio.run(run(config, args.output))


if __name__ == "__main__":
    main()
