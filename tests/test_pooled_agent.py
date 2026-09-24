import asyncio
from contextlib import asynccontextmanager

import numpy as np
import pytest
from hud import Environment, LocalRuntime, Taskset
from hud.environment.robot import RobotEndpoint

from env import create_environment
from hud_dreamscale.cohort import cohort_tasks, run_cohort, summarize_cohort
from hud_dreamscale.contract import CAMERAS, POOLED_ENV_NAME, POOLED_PROFILE, build_contract
from hud_dreamscale.pooled_agent import PooledLiberoAdapter, PooledRobotAgent
from hud_dreamscale.runtime_pool import RuntimePool
from hud_dreamscale.telemetry import Evidence
from hud_dreamscale.video import export_videos
from tests.test_lifecycle import FakeBridge


class RawBridge(FakeBridge):
    def __init__(self, slot):
        super().__init__()
        self.contract = build_contract(profile=POOLED_PROFILE)
        self.slot = slot

    def get_observation(self):
        data = {key: np.full((1, 360, 360, 3), self.slot, dtype=np.uint8) for key in CAMERAS}
        data[CAMERAS[0]][:, 0, 0] = [10, 20, 30]  # Asymmetric orientation/color sentinel.
        data["robot0_eef_pos"] = np.zeros((1, 3), np.float32)
        data["robot0_eef_quat_xyzw"] = np.array([[0, 0, 0, 1]], np.float32)
        data["robot0_gripper_qpos"] = np.zeros((1, 2), np.float32)
        return data, np.array([self.terminated])


class FakeProvider:
    def __init__(self, concurrency, *, bad_slot=None, wait=False):
        self.concurrency = concurrency
        self.identity = {"id": "test-deployment", "model": "molmoact2-libero"}
        self.calls = []
        self.counts = [0] * concurrency
        self.active = self.peak = 0
        self.bad_slot, self.wait = bad_slot, wait
        self.entered = asyncio.Event()

    def is_slot_available(self, slot):
        return True

    async def predict(self, **kwargs):
        self.calls.append(kwargs)
        self.active += 1
        self.peak = max(self.peak, self.active)
        self.entered.set()
        try:
            if self.wait:
                await asyncio.Event().wait()
            await asyncio.sleep(0.01)
            slot = kwargs["slot"]
            self.counts[slot] += 1
            actions = np.full((10, 7), self.counts[slot], np.float32)
            if slot == self.bad_slot:
                actions[0, 0] = np.nan
            return actions
        finally:
            self.active -= 1


@asynccontextmanager
async def local_pool(rows, *, emit=None):
    bridges = {}

    @asynccontextmanager
    async def placement(task):
        slot = task.columns["lane_id"]
        bridge = RawBridge(slot)
        bridges[slot] = bridge
        await bridge.start()
        server = await bridge.serve_control("127.0.0.1", 0)
        endpoint = RobotEndpoint.remote("127.0.0.1", server.sockets[0].getsockname()[1])
        environment = create_environment(
            endpoint, profile=POOLED_PROFILE, environment=Environment(name=POOLED_ENV_NAME)
        )
        try:
            async with LocalRuntime(environment)(task) as runtime:
                yield runtime
        finally:
            await endpoint.stop()
            server.close()
            await server.wait_closed()
            await bridge.stop()

    concurrency = max(task.columns["lane_id"] for task in rows) + 1
    async with RuntimePool(placement, rows, concurrency=concurrency, emit=emit) as pool:
        yield pool, bridges


@pytest.mark.parametrize("stagger_step_s", [0.0, 0.125])
async def test_two_parallel_lanes_reuse_with_real_hud_wire_grading_and_video(
    tmp_path, monkeypatch, stagger_step_s
):
    import av
    from hud.settings import settings

    monkeypatch.setattr(settings, "telemetry_local_dir", str(tmp_path / "traces"))
    monkeypatch.setattr("hud_dreamscale.pooled_agent._INITIAL_STAGGER_STEP_S", stagger_step_s)
    evidence = Evidence(tmp_path / "timings.jsonl")
    rows = cohort_tasks(2, max_steps=3)
    provider = FakeProvider(2)
    async with local_pool(rows, emit=evidence.emit) as (pool, bridges):
        agent = PooledRobotAgent(provider=provider, runtimes=pool, max_steps=3, emit=evidence.emit)
        job = await Taskset("pooled-wire", rows).run(agent, runtime=pool, max_concurrent=2)
        summary = summarize_cohort(job, rows, evidence.rows)
        assert summary["successes"] == 4 and summary["integration_errors"] == 0
        assert provider.counts == [2, 2]
        if stagger_step_s == 0:
            assert provider.peak == 2  # Preserve the unstaggered concurrent execution check.
        else:
            assert 1 <= provider.peak <= 2
        waits = [row for row in evidence.rows if row["event"] == "initial_inference_stagger"]
        assert len(waits) == 2  # Two episodes per lane do not repeat the delay.
        assert {row["lane_id"]: row["requested_delay_s"] for row in waits} == {
            0: 0.0,
            1: stagger_step_s,
        }
        assert all(row["outcome"] == "completed" for row in waits)
        assert len({call["episode_id"] for call in provider.calls}) == 4
        assert {call["trace_id"] for call in provider.calls} == {run.trace_id for run in job.runs}
        for slot, bridge in bridges.items():
            assert [float(ep[0][0, 0]) for ep in bridge.episodes] == [1, 2]
            assert bridge._registry.all_free
            for call in (c for c in provider.calls if c["slot"] == slot):
                obs = call["observation"]
                assert np.array_equal(obs[CAMERAS[0]][0, 0], [10, 20, 30])
                assert np.array_equal(obs[CAMERAS[1]][0, 0], [slot, slot, slot])
                assert np.array_equal(obs["robot0_eef_quat"], [0, 0, 0, 1])
                assert "state" not in obs and "robot0_eef_quat_xyzw" not in obs
    evidence.close()
    videos = export_videos(tmp_path / "traces", tmp_path / "videos")
    assert len(videos) == 8
    for video in videos:
        with av.open(str(tmp_path / "videos" / video["file"])) as recording:
            assert len(list(recording.decode(video=0))) == 3


async def test_one_failed_lane_does_not_contaminate_other_lane():
    rows = cohort_tasks(2, max_steps=3)
    provider = FakeProvider(2, bad_slot=0)
    events = []
    async with local_pool(rows) as (pool, bridges):
        job = await Taskset("pooled-malformed", rows).run(
            PooledRobotAgent(
                provider=provider,
                runtimes=pool,
                max_steps=3,
                emit=lambda event, **fields: events.append((event, fields)),
            ),
            runtime=pool,
            max_concurrent=2,
        )
        assert [run.trace.is_error for run in job.runs] == [True, False, True, False]
        assert all(not episode for episode in bridges[0].episodes)
        assert all(bridge._registry.all_free for bridge in bridges.values())
        assert [float(ep[0][0, 0]) for ep in bridges[1].episodes] == [1, 2]
    attempts = [fields for event, fields in events if event == "inference_attempt_finished"]
    assert sorted(row["outcome"] for row in attempts) == [
        "completed",
        "completed",
        "error",
        "error",
    ]
    assert all(row["duration_s"] > 0 and row["inference_index"] == 0 for row in attempts)
    assert len({row["episode_id"] for row in attempts}) == 4


@pytest.mark.parametrize("stagger_step_s", [0.0, 0.125])
async def test_eight_lane_control_and_stagger_with_real_hud_wire(
    tmp_path, monkeypatch, stagger_step_s
):
    from hud.eval.job import Job
    from hud.settings import settings

    monkeypatch.setattr(settings, "telemetry_local_dir", str(tmp_path / "traces"))
    monkeypatch.setattr("hud_dreamscale.pooled_agent._INITIAL_STAGGER_STEP_S", stagger_step_s)
    evidence = Evidence(tmp_path / "timings.jsonl")
    rows = cohort_tasks(8, max_steps=3)
    provider = FakeProvider(8)
    original_predict = provider.predict

    async def predict(**kwargs):
        evidence.emit(
            "test_provider_submit", lane_id=kwargs["slot"], episode_id=kwargs["episode_id"]
        )
        return await original_predict(**kwargs)

    provider.predict = predict
    job = await Job.start("eight-lane-startup")
    try:
        async with local_pool(rows, emit=evidence.emit) as (pool, bridges):
            agent = PooledRobotAgent(
                provider=provider, runtimes=pool, max_steps=3, emit=evidence.emit
            )
            await run_cohort(agent, rows, runtime=pool, job=job, concurrency=8)
            summary = summarize_cohort(job, rows, evidence.rows)
            assert summary["successes"] == 16 and summary["integration_errors"] == 0
            assert provider.counts == [2] * 8
        assert all(bridge._registry.all_free for bridge in bridges.values())
        waits = [row for row in evidence.rows if row["event"] == "initial_inference_stagger"]
        assert len(waits) == 8
        inputs = {
            row["episode_id"]: row["monotonic_s"]
            for row in evidence.rows
            if row["event"] == "episode_first_model_input"
        }
        posts = {
            row["episode_id"]: row["monotonic_s"]
            for row in evidence.rows
            if row["event"] == "test_provider_submit"
        }
        for row in waits:
            delay = row["lane_id"] * stagger_step_s
            assert row["requested_delay_s"] == delay and row["outcome"] == "completed"
            assert posts[row["episode_id"]] - inputs[row["episode_id"]] >= delay
    finally:
        evidence.close()


async def test_cancelled_episode_releases_every_claim():
    rows = cohort_tasks(2, max_steps=3)
    provider = FakeProvider(2, wait=True)
    async with local_pool(rows) as (pool, bridges):
        agent = PooledRobotAgent(provider=provider, runtimes=pool, max_steps=3)
        task = asyncio.create_task(
            Taskset("pooled-cancel", rows).run(
                agent,
                runtime=pool,
                max_concurrent=2,
            )
        )
        await asyncio.wait_for(provider.entered.wait(), 10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert provider.active == 0
    # Cancellation during task setup can precede RobotAgent.connect. The job's
    # owner must close the runtime to release that not-yet-connected claim too.
    assert all(bridge._registry.all_free for bridge in bridges.values())


async def test_cancel_during_initial_stagger_releases_hud_claim_without_submission(monkeypatch):
    monkeypatch.setattr("hud_dreamscale.pooled_agent._INITIAL_STAGGER_STEP_S", 100.0)
    rows = cohort_tasks(2, max_steps=3)
    provider = FakeProvider(2, wait=True)
    events = []
    async with local_pool(rows) as (pool, bridges):
        agent = PooledRobotAgent(
            provider=provider,
            runtimes=pool,
            max_steps=3,
            emit=lambda event, **fields: events.append((event, fields)),
        )
        task = asyncio.create_task(
            Taskset("stagger-cancel", rows).run(agent, runtime=pool, max_concurrent=2)
        )
        async with asyncio.timeout(10):
            while not any(
                event == "episode_first_model_input" and fields["lane_id"] == 1
                for event, fields in events
            ):
                await asyncio.sleep(0.001)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert provider.active == 0
        assert all(call["slot"] == 0 for call in provider.calls)
    assert all(bridge._registry.all_free for bridge in bridges.values())
    waits = [fields for event, fields in events if event == "initial_inference_stagger"]
    assert any(row["lane_id"] == 1 and row["outcome"] == "cancelled" for row in waits)


def test_profile_rejects_legacy_resolution_and_invalid_quaternion():
    adapter = PooledLiberoAdapter()
    legacy = build_contract()["features"]
    with pytest.raises(ValueError, match="contract mismatch"):
        adapter.bind(legacy["action"], {k: v for k, v in legacy.items() if k != "action"})
    bridge = RawBridge(0)
    data, _ = bridge.get_observation()
    obs = {"data": {key: value[0] for key, value in data.items()}}
    obs["data"]["robot0_eef_quat_xyzw"][:] = 0
    with pytest.raises(ValueError, match="normalized"):
        adapter.adapt_observation(obs, "task")


async def test_cancelled_cohort_preserves_completed_grades_with_same_hud_job(tmp_path):
    from hud.eval.job import Job

    rows = cohort_tasks(2, max_steps=3)
    provider = FakeProvider(2)
    original_predict = provider.predict

    async def predict(**kwargs):
        if kwargs["slot"] == 1:
            await asyncio.Event().wait()
        return await original_predict(**kwargs)

    provider.predict = predict
    evidence = Evidence(tmp_path / "timings.jsonl")
    job = await Job.start("partial-preservation")
    async with local_pool(rows, emit=evidence.emit) as (pool, bridges):
        agent = PooledRobotAgent(provider=provider, runtimes=pool, max_steps=3, emit=evidence.emit)
        running = asyncio.create_task(
            run_cohort(
                agent,
                rows,
                runtime=pool,
                job=job,
                concurrency=2,
            )
        )
        async with asyncio.timeout(10):
            while not job.runs:
                await asyncio.sleep(0.001)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        summary = summarize_cohort(job, rows, evidence.rows)
        assert summary["runs"][0]["reward"] == 1
        assert summary["runs"][0]["status"] == "completed"
        assert summary["runs"][0]["trace_id"]
        assert len(summary["runs"]) == 4  # Includes incomplete attempts in the denominator.
        assert all(run.job_id == job.id for run in job.runs)
    assert all(bridge._registry.all_free for bridge in bridges.values())
    evidence.close()


async def test_runtime_owner_waits_for_hud_background_timeout_cleanup():
    rows = cohort_tasks(1, max_steps=3)
    cleanup_finished = asyncio.Event()

    async def slow_cleanup_agent(run):
        try:
            await asyncio.Event().wait()
        finally:
            await asyncio.sleep(0.15)
            cleanup_finished.set()

    async with local_pool(rows) as (pool, _bridges):
        job = await Taskset("background-timeout-cleanup", rows[:1]).run(
            slow_cleanup_agent,
            runtime=pool,
            rollout_timeout=0.03,
        )
        assert job.runs[0].trace.is_error
        assert not cleanup_finished.is_set()  # HUD returns before its driver cleanup completes.
    assert cleanup_finished.is_set()
