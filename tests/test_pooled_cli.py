import asyncio
import json
from collections import Counter
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest

from hud_dropbear.cli import main, startup_metrics
from hud_dropbear.pooled_cli import (
    cohort_manifest,
    cohort_tasks,
    parser,
    run_cohort,
    summarize_cohort,
)


@pytest.mark.parametrize("width", [1, 3, 8, 17, 32, 64])
def test_fixed_cohort_has_unique_slugs_and_two_episodes_per_lane(width):
    rows = cohort_tasks(width)
    assert len(rows) == len({row.slug for row in rows}) == 2 * width
    assert Counter(row.columns["lane_id"] for row in rows) == dict.fromkeys(range(width), 2)
    assert [row.columns["case_index"] for row in rows] == list(range(2 * width))
    manifest, digest = cohort_manifest(rows, concurrency=width, episodes_per_lane=2)
    assert (manifest, digest) == cohort_manifest(
        cohort_tasks(width), concurrency=width, episodes_per_lane=2
    )
    assert len(digest) == 64


def test_missing_results_remain_in_denominator():
    rows = cohort_tasks(8)
    result = summarize_cohort(
        SimpleNamespace(id="12345678123412341234123456781234", runs=[]), rows, []
    )
    assert result["expected_episodes"] == len(result["runs"]) == 16
    assert result["integration_errors"] == 16 and result["successes"] == 0
    assert all(row["status"] == "not_completed" for row in result["runs"])


def test_repeated_task_timing_is_not_joined_to_the_last_start():
    events = [
        {"event": "environment_starting", "task": "same", "elapsed_s": 1},
        {"event": "environment_ready", "task": "same", "elapsed_s": 4},
        {"event": "inference", "task": "same", "duration_s": 2},
        {"event": "inference", "task": "same", "duration_s": 1},
        {"event": "first_action_confirmed", "task": "same", "elapsed_s": 6},
        {"event": "environment_starting", "task": "same", "elapsed_s": 10},
        {"event": "environment_ready", "task": "same", "elapsed_s": 11},
        {"event": "inference", "task": "same", "duration_s": 0.5},
        {"event": "first_action_confirmed", "task": "same", "elapsed_s": 12},
    ]
    metrics = startup_metrics(events)
    assert metrics["simulation_setup_s"] == {"same": 3, "same#2": 1}
    assert metrics["episode_start_to_first_action_s"] == {"same": 5, "same#2": 2}
    assert metrics["episode_first_inference_s"] == {"same": 2, "same#2": 0.5}
    assert metrics["steady_state_inference_samples_s"] == [1]


def test_dispatch_preserves_both_new_and_legacy_commands(monkeypatch):
    import hud_dropbear.pooled_cli

    received = []
    monkeypatch.setattr(hud_dropbear.pooled_cli, "main", received.append)
    main(["pooled", "--concurrency", "3"])
    assert received == [["--concurrency", "3"]]
    args = parser().parse_args([])
    assert args.runtime == "hud" and args.concurrency == 8 and args.episodes_per_lane == 2


def test_cohort_requires_warm_repeats():
    with pytest.raises(ValueError, match="At least two episodes"):
        cohort_tasks(8, episodes_per_lane=1)


@pytest.mark.parametrize("width", [1, 2, 3, 8, 17, 32, 64])
def test_cli_episode_default_covers_six_cases_for_one_lane_only(width):
    args = parser().parse_args(["--concurrency", str(width)])
    assert args.episodes_per_lane == (6 if width == 1 else 2)
    if width == 1:
        rows = cohort_tasks(width, episodes_per_lane=args.episodes_per_lane)
        assert {(row.args["task_id"], row.args["init_state_id"]) for row in rows} == {
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (2, 0),
            (2, 1),
        }


@pytest.mark.parametrize(
    "arguments",
    [
        ["--concurrency", "1", "--episodes-per-lane", "2"],
        ["--episodes-per-lane", "2", "--concurrency", "1"],
    ],
)
def test_cli_preserves_explicit_single_lane_smoke_episode_count(arguments):
    assert parser().parse_args(arguments).episodes_per_lane == 2


async def test_each_free_lane_advances_without_waiting_behind_another_lane(monkeypatch):
    import hud_dropbear.pooled_cli as cli

    slow_lane_release, fast_lane_second_episode = asyncio.Event(), asyncio.Event()
    locks = [asyncio.Lock(), asyncio.Lock()]
    starts = []

    class Taskset:
        def __init__(self, name, rows):
            self.row = rows[0]

        async def run(self, agent, *, job, **kwargs):
            lane = self.row.columns["lane_id"]
            async with locks[lane]:
                case = self.row.columns["case_index"]
                starts.append(case)
                if case == 0:
                    await slow_lane_release.wait()
                if case == 3:
                    fast_lane_second_episode.set()
                job.runs.append(self.row)
            return job

    monkeypatch.setattr(cli, "Taskset", Taskset)
    job = SimpleNamespace(name="independent-lanes", runs=[])
    running = asyncio.create_task(
        run_cohort(
            None,
            cohort_tasks(2),
            runtime=None,
            job=job,
            concurrency=2,
        )
    )
    try:
        await asyncio.wait_for(fast_lane_second_episode.wait(), 1)
        assert starts == [0, 1, 3]
    finally:
        slow_lane_release.set()
        await running
    assert starts == [0, 1, 3, 2] and len(job.runs) == 4


async def test_stuck_scheduler_cleanup_is_bounded_without_a_second_cancellation(monkeypatch):
    import hud_dropbear.pooled_cli as cli

    entered, cleaning, release, finished = (asyncio.Event() for _ in range(4))
    events = []

    class StuckTaskset:
        def __init__(self, *args):
            pass

        async def run(self, *args, **kwargs):
            entered.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleaning.set()
                # Models a HUD cancellation handler waiting indefinitely on IO.
                # A second cancellation would interrupt this owned cleanup.
                await release.wait()
                finished.set()
                raise

    monkeypatch.setattr(cli, "Taskset", StuckTaskset)
    runtime = SimpleNamespace(emit=lambda event, **fields: events.append((event, fields)))
    running = asyncio.create_task(
        run_cohort(
            None,
            cohort_tasks(1),
            runtime=runtime,
            job=SimpleNamespace(name="stuck-cleanup", runs=[]),
            concurrency=1,
            cleanup_timeout=0.03,
        )
    )
    try:
        await entered.wait()
        running.cancel()
        await cleaning.wait()
        running.cancel()  # Repeated user cancellation must not cancel the child again.
        with pytest.raises(asyncio.CancelledError) as error:
            await asyncio.wait_for(running, 1)
        assert "bounded deadline" in " ".join(error.value.__notes__)
        assert not finished.is_set()
        assert events == [
            (
                "cohort_cleanup_error",
                {"error_type": "TimeoutError", "timeout_s": 0.03, "pending_lanes": [0]},
            )
        ]
    finally:
        release.set()
        await asyncio.wait_for(finished.wait(), 1)


@pytest.mark.parametrize("width,expected", [(1, 6), (8, 16)])
async def test_runner_saves_every_planned_failure_after_provider_startup_error(
    tmp_path, monkeypatch, width, expected
):
    import hud_dropbear.pooled_cli as cli

    class FailingProvider:
        def __init__(self, **kwargs):
            self.identity = {}
            self.cleanup_confirmed = True
            self.cleanup_error = None

        async def __aenter__(self):
            raise TimeoutError("test provider startup")

        async def __aexit__(self, *exc):
            pass

    monkeypatch.setattr(cli, "PooledProvider", FailingProvider)
    args = parser().parse_args(
        ["--runtime", "local-container", "--output", str(tmp_path), "--concurrency", str(width)]
    )
    with pytest.raises(TimeoutError, match="test provider startup"):
        await cli.evaluate(args)
    result = json.loads((tmp_path / "results.json").read_text())
    assert result["expected_episodes"] == len(result["runs"]) == expected
    assert result["integration_errors"] == expected and not result["demo_passed"]
    assert result["provider_cleanup_confirmed"]
    assert (tmp_path / "cohort.json").exists() and (tmp_path / "report.json").exists()


@pytest.mark.parametrize("environment_key", ["test-env-key-should-never-be-logged", "", None])
async def test_cli_environment_key_precedence_and_evidence_redaction(
    environment_key, tmp_path, monkeypatch, capsys
):
    import dropbear.config
    from dropbear.inference import AsyncInferenceClient

    import hud_dropbear.pooled_cli as cli

    saved_key = "test-saved-key-should-never-be-logged"
    monkeypatch.setattr(
        dropbear.config,
        "load_config",
        lambda: SimpleNamespace(api_key=saved_key, control_plane_url="https://saved.invalid"),
    )
    if environment_key is None:
        monkeypatch.delenv("DROPBEAR_API_KEY", raising=False)
    else:
        monkeypatch.setenv("DROPBEAR_API_KEY", environment_key)
    effective_keys = []

    class OfflineProvider:
        def __init__(self, **kwargs):
            self.identity = {}
            self.cleanup_confirmed = True
            self.cleanup_error = None
            # Exercise real SDK option resolution without sending any request.
            self.client = AsyncInferenceClient(
                api_key=kwargs["api_key"], api_base=kwargs["api_base"]
            )
            effective_keys.append(self.client._http.headers["Authorization"])

        async def __aenter__(self):
            await self.client.close()
            raise RuntimeError("offline credential selection verified")

        async def __aexit__(self, *exc):
            pass

    monkeypatch.setattr(cli, "PooledProvider", OfflineProvider)
    args = parser().parse_args(["--runtime", "local-container", "--output", str(tmp_path)])
    with pytest.raises(RuntimeError, match="offline credential selection"):
        await cli.evaluate(args)
    assert effective_keys == [f"Bearer {environment_key or saved_key}"]
    output = capsys.readouterr()
    evidence = (
        output.out + output.err + "".join(path.read_text() for path in tmp_path.glob("*.json*"))
    )
    assert saved_key not in evidence
    if environment_key:
        assert environment_key not in evidence


@pytest.mark.parametrize("runtime", ["local-container", "hud"])
async def test_runner_writes_results_and_cleanup_on_normal_return(runtime, tmp_path, monkeypatch):
    from hud import Runtime

    import hud_dropbear.pooled_cli as cli

    class Provider:
        def __init__(self, **kwargs):
            self.concurrency = kwargs["concurrency"]
            self.identity = {"id": "test"}
            self.cleanup_confirmed = False
            self.cleanup_error = None

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            self.cleanup_confirmed = True

    @asynccontextmanager
    async def placement(task):
        yield Runtime(f"tcp://test-{task.columns['lane_id']}")

    @asynccontextmanager
    async def connect(runtime, **kwargs):
        yield SimpleNamespace(
            manifest=SimpleNamespace(server_info=SimpleNamespace(name="dropbear-libero-pooled"))
        )

    class FakeTaskset:
        def __init__(self, name, rows):
            self.rows = rows

        async def run(self, agent, *, runtime, job, **kwargs):
            for row in self.rows:
                async with runtime(row):
                    job.runs.append(
                        SimpleNamespace(
                            slug=row.slug,
                            trace_id=f"{row.columns['case_index']:032x}",
                            reward=1,
                            trace=SimpleNamespace(status="completed", is_error=False),
                            grade=SimpleNamespace(is_error=False, raw={"score": 1}),
                        )
                    )
            return job

    monkeypatch.setattr(cli, "PooledProvider", Provider)
    monkeypatch.setattr(cli, "DockerRuntime", lambda *args, **kwargs: placement)
    monkeypatch.setattr(cli, "Taskset", FakeTaskset)
    monkeypatch.setattr("hud_dropbear.runtime_pool.connect", connect)
    if runtime == "hud":

        class Guard:
            def __init__(self, *args, **kwargs):
                self.cleanup_receipt = {"verified": True}

            async def __aenter__(self):
                return self

            async def __aexit__(self, *exc):
                pass

            async def validate_ready(self):
                return {"instance_ids": [f"test-{i}" for i in range(8)]}

        async def job_start(name):
            return SimpleNamespace(id="a" * 32, name=name, runs=[])

        async def verify_platform(summary):
            return {"verified": True}

        def report(output_dir, **kwargs):
            # A report-only failure must also update the timing sidecar receipt.
            return {"gate": {"passed": False}}

        monkeypatch.setattr(cli.settings, "api_key", "test-hud-key")
        monkeypatch.setattr(cli.settings, "telemetry_enabled", True)
        monkeypatch.setattr(cli, "HUDRuntime", lambda: placement)
        monkeypatch.setattr(cli, "ExclusiveRegistryCampaign", Guard)
        monkeypatch.setattr(cli.Job, "start", job_start)
        monkeypatch.setattr(cli, "verify_platform", verify_platform)
        monkeypatch.setattr("hud_dropbear.reporting.write_campaign_report", report)
    args = parser().parse_args(
        [
            "--runtime",
            runtime,
            "--output",
            str(tmp_path),
            "--registry-id",
            "test-registry",
        ]
    )
    assert await cli.evaluate(args) == 1
    result = json.loads((tmp_path / "results.json").read_text())
    assert result["expected_episodes"] == result["completed_episodes"] == result["successes"] == 16
    assert result["integration_errors"] == 0
    assert result["provider_cleanup_confirmed"]
    assert len({run["episode_id"] for run in result["runs"]}) == 16
    receipts = [json.loads(line) for line in (tmp_path / "timings.jsonl").read_text().splitlines()]
    final = next(row for row in receipts if row["event"] == "job_result")
    assert final["demo_passed"] == result["demo_passed"] is False


def test_default_executor_workers_scale_with_lanes():
    from hud_dropbear.pooled_cli import default_executor_workers

    assert default_executor_workers(1) == 32
    assert default_executor_workers(8) == 48
    assert default_executor_workers(64) == 272
    with pytest.raises(ValueError):
        default_executor_workers(0)
