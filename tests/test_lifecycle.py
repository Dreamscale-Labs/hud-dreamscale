import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from dreamscale.policy.types import ActionChunkResult, ActionTiming
from hud import LocalRuntime, Taskset
from hud.environment.robot import RobotBridge, RobotEndpoint
from hud.eval import Shared

from env import create_environment
from hud_dreamscale.agent import DreamscaleRobotAgent
from hud_dreamscale.cli import MeasuredRuntime, job_runtime, startup_metrics, summarize, tasks
from hud_dreamscale.contract import CAMERAS, CONTROL_HZ, MODEL, build_contract
from hud_dreamscale.video import export_videos


class FakePolicy:
    model = MODEL
    session_id = "test-session"
    region = "ap-southeast-2"
    action_hz = CONTROL_HZ
    chunk_size = 10
    transport_mode = "quic"

    def __init__(self, bad=False, wait=False):
        self.resolved_optimization_config = SimpleNamespace(
            backend="tensorrt",
            rtc="off",
            calibration="off",
            tensorrt_artifact_fingerprint=(
                "014c358f8761e5f1391c931b3f358b0eed202e6422c4eaee4ae876b83613f70a"
            ),
            tensorrt_artifact_id="b2405523e67019934dd81be1",
        )
        self.calls = 0
        self.closed = 0
        self.bad = bad
        self.wait = wait
        self.entered = asyncio.Event()

    async def predict(self, observation, *, instruction, timeout_s):
        self.calls += 1
        self.entered.set()
        if self.wait:
            await asyncio.Event().wait()
        rows = np.full((10, 7), self.calls, dtype=np.float32)
        if self.bad:
            rows[0, 0] = np.nan
        return ActionChunkResult(self.calls - 1, self.calls, rows, ActionTiming(1.0))

    async def close(self):
        self.closed += 1


class FakeBridge(RobotBridge):
    def __init__(self, control_hz=CONTROL_HZ):
        super().__init__()
        self.contract = build_contract(control_hz)
        self.actions = []
        self.episodes = []
        self.selections = []
        self.limit = 2

    def reset(self, **kwargs):
        self.selections.append(kwargs)
        self.actions = []
        self.episodes.append(self.actions)
        return "test task"

    def step(self, action):
        self.actions.append(action.copy())
        self.success = len(self.actions) == self.limit
        self.terminated = self.success

    def get_observation(self):
        data = {key: np.zeros((1, 256, 256, 3), dtype=np.uint8) for key in CAMERAS}
        data["state"] = np.zeros((1, 8), dtype=np.float32)
        return data, np.array([self.terminated])


@asynccontextmanager
async def environment(control_hz=CONTROL_HZ):
    bridge = FakeBridge(control_hz)
    await bridge.start()
    server = await bridge.serve_control("127.0.0.1", 0)
    endpoint = RobotEndpoint.remote("127.0.0.1", server.sockets[0].getsockname()[1])
    try:
        yield create_environment(endpoint), bridge
    finally:
        await endpoint.stop()
        server.close()
        await server.wait_closed()
        await bridge.stop()


async def test_real_hud_wire_grading_reuse_and_fresh_chunks(tmp_path, monkeypatch):
    import av
    from hud.settings import settings

    monkeypatch.setattr(settings, "telemetry_local_dir", str(tmp_path / "traces"))
    policy = FakePolicy()
    connects = []
    events = []

    async def connect(**kwargs):
        connects.append(kwargs)
        return policy

    async with environment() as (env, bridge):
        async with DreamscaleRobotAgent(
            connector=connect, emit=lambda e, **f: events.append((e, f))
        ) as a:
            job = await Taskset("test", tasks([0], [0, 1], max_steps=3)).run(
                a,
                runtime=MeasuredRuntime(LocalRuntime(env), lambda e, **f: events.append((e, f))),
                max_concurrent=1,
            )
        assert [r.reward for r in job.runs] == [1.0, 1.0]
        assert summarize(job)["integration_errors"] == 0
        assert all(not r.trace.is_error for r in job.runs)
        assert len(connects) == 1 and policy.calls == 2 and policy.closed == 1
        assert connects[0]["control_hz"] == 20
        assert connects[0]["keep_warm"] == 0
        assert [float(ep[0][0, 0]) for ep in bridge.episodes] == [1.0, 2.0]
        assert bridge._registry.all_free
        assert len([e for e, _ in events if e == "first_action_confirmed"]) == 2
        inference_ids = [fields["trace_id"] for event, fields in events if event == "inference"]
        assert inference_ids == [run.trace_id for run in job.runs]
        assert all(inference_ids)
        videos = export_videos(tmp_path / "traces", tmp_path / "videos")
        assert len(videos) == 4
        for video in videos:
            with av.open(str(tmp_path / "videos" / video["file"])) as recording:
                # Initial observation plus the observations after both actions.
                assert len(list(recording.decode(video=0))) == 3


@pytest.mark.parametrize("action_limit", [1, 2])
async def test_action_cap_records_final_frame_without_an_extra_action(
    action_limit, tmp_path, monkeypatch
):
    import av
    from hud.settings import settings

    monkeypatch.setattr(settings, "telemetry_local_dir", str(tmp_path / "traces"))
    policy = FakePolicy()

    async def connect(**kwargs):
        return policy

    async with environment() as (env, bridge):
        async with DreamscaleRobotAgent(connector=connect) as agent:
            agent.max_steps = action_limit
            job = await Taskset("action-cap", tasks([0], [0], max_steps=3)).run(
                agent, runtime=LocalRuntime(env)
            )
        assert len(bridge.actions) == action_limit
        assert job.runs[0].reward == (1.0 if action_limit == bridge.limit else 0.0)
        assert summarize(job)["integration_errors"] == 0
        videos = export_videos(tmp_path / "traces", tmp_path / "videos")
        assert len(videos) == 2
        for video in videos:
            with av.open(str(tmp_path / "videos" / video["file"])) as recording:
                assert sum(1 for _ in recording.decode(video=0)) == action_limit + 1


async def test_malformed_model_chunk_never_reaches_simulator():
    policy = FakePolicy(bad=True)

    async def connect(**kwargs):
        return policy

    async with environment() as (env, bridge):
        async with DreamscaleRobotAgent(connector=connect) as a:
            job = await Taskset("test", tasks([0], [0], max_steps=3)).run(
                a, runtime=LocalRuntime(env)
            )
        assert job.runs[0].trace.is_error
        assert bridge.actions == [] and bridge._registry.all_free
        assert policy.closed == 1


async def test_goal_template_preserves_distinct_task_selections():
    policy = FakePolicy()

    async def connect(**kwargs):
        return policy

    selected = tasks([0, 5, 7], [0], suite="libero_goal", max_steps=3)
    async with environment() as (env, bridge):
        async with DreamscaleRobotAgent(connector=connect) as agent:
            job = await Taskset("goal-contract", selected).run(agent, runtime=LocalRuntime(env))
        assert len(job.runs) == 3 and all(not run.trace.is_error for run in job.runs)
        assert [row["suite_name"] for row in bridge.selections] == ["libero_goal"] * 3
        assert [row["task_id"] for row in bridge.selections] == [0, 5, 7]
        assert len({task.columns["task_name"] for task in selected}) == 3
        assert policy.calls == 3 and policy.closed == 1
        assert [float(ep[0][0, 0]) for ep in bridge.episodes] == [1.0, 2.0, 3.0]


@pytest.mark.parametrize("sim_hz", [10, 20])
async def test_explicit_control_rate_must_match_simulator(sim_hz):
    policy = FakePolicy()
    policy.action_hz = 20

    async def connect(**kwargs):
        assert kwargs["control_hz"] == 20
        return policy

    async with environment(control_hz=sim_hz) as (env, bridge):
        async with DreamscaleRobotAgent(connector=connect, control_hz=20) as agent:
            job = await Taskset("cadence", tasks([0], [0], max_steps=3)).run(
                agent, runtime=LocalRuntime(env)
            )
        assert job.runs[0].trace.is_error == (sim_hz != 20)
        assert len(bridge.actions) == (2 if sim_hz == 20 else 0)
        assert policy.closed == 1


async def test_inference_failure_releases_claim_and_session():
    policy = FakePolicy()

    async def predict(*args, **kwargs):
        raise ConnectionError("test transport failure")

    policy.predict = predict

    async def connect(**kwargs):
        return policy

    async with environment() as (env, bridge):
        async with DreamscaleRobotAgent(connector=connect) as agent:
            job = await Taskset("test", tasks([0], [0])).run(agent, runtime=LocalRuntime(env))
        assert summarize(job)["integration_errors"] == 1
        assert bridge.actions == [] and bridge._registry.all_free
        assert policy.closed == 1


async def test_cancellation_releases_inference_session():
    policy = FakePolicy(wait=True)

    async def connect(**kwargs):
        return policy

    async with environment() as (env, bridge):

        async def run():
            async with DreamscaleRobotAgent(connector=connect) as agent:
                await Taskset("test", tasks([0], [0])).run(agent, runtime=LocalRuntime(env))

        task = asyncio.create_task(run())
        await asyncio.wait_for(policy.entered.wait(), timeout=10)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert policy.closed == 1
        assert bridge._registry.all_free


async def test_wrong_backend_closes_before_any_actions():
    policy = FakePolicy()
    policy.resolved_optimization_config.backend = "pytorch"

    async def connect(**kwargs):
        return policy

    with pytest.raises(ValueError, match="TensorRT"):
        async with DreamscaleRobotAgent(connector=connect):
            pass
    assert policy.calls == 0 and policy.closed == 1


async def test_unknown_artifact_is_rejected():
    policy = FakePolicy()
    policy.resolved_optimization_config.tensorrt_artifact_fingerprint = "different-engine"

    async def connect(**kwargs):
        return policy

    with pytest.raises(ValueError, match="Unverified TensorRT"):
        async with DreamscaleRobotAgent(connector=connect):
            pass
    assert policy.calls == 0 and policy.closed == 1


def test_timing_joins_by_task_after_a_failed_environment():
    rows = [
        {"event": "environment_starting", "task": "failed", "elapsed_s": 1},
        {"event": "environment_starting", "task": "working", "elapsed_s": 20},
        {"event": "environment_ready", "task": "working", "elapsed_s": 25},
        {"event": "first_action_confirmed", "task": "working", "elapsed_s": 27},
    ]
    result = startup_metrics(rows)
    assert result["simulation_setup_s"] == {"working": 5}
    assert result["episode_start_to_first_action_s"] == {"working": 7}


async def test_all_five_suite_templates_keep_selection_and_reset_queues():
    from hud_dreamscale.contract import TASK_SUITES

    policy = FakePolicy()

    async def connect(**kwargs):
        return policy

    rows = [row for suite in TASK_SUITES for row in tasks([0], [0], suite=suite, max_steps=3)]
    async with environment() as (env, bridge):
        async with DreamscaleRobotAgent(connector=connect) as agent:
            job = await Taskset("five-suite-contract", rows).run(agent, runtime=LocalRuntime(env))
        assert len(job.runs) == 5
        assert all(not run.trace.is_error and run.reward == 1 for run in job.runs)
        assert [selection["suite_name"] for selection in bridge.selections] == list(TASK_SUITES)
        assert policy.calls == 5 and policy.closed == 1
        assert [float(episode[0][0, 0]) for episode in bridge.episodes] == [1, 2, 3, 4, 5]


def test_libero_90_selection_preserves_last_pinned_task():
    from hud_dreamscale.cli import parser
    from hud_dreamscale.contract import TASK_SUITES

    args = parser().parse_args(["--suite", "libero_90", "--task-ids", "89"])
    assert args.task_ids == [89]
    row = tasks([89], [1], suite="libero_90")[0]
    assert row.columns["task_name"] == TASK_SUITES["libero_90"][89]


async def test_background_connection_overlaps_shared_environment_and_resets_episodes():
    from hud_dreamscale.contract import TASK_SUITES

    policy = FakePolicy()
    environment_ready = asyncio.Event()
    opened = closed = 0

    async def connect(**kwargs):
        # This would deadlock if the runner awaited inference before provisioning.
        await asyncio.wait_for(environment_ready.wait(), 5)
        return policy

    def emit(event, **fields):
        if event == "environment_ready":
            environment_ready.set()

    async with environment() as (env, bridge):

        @asynccontextmanager
        async def provider(task):
            nonlocal opened, closed
            from hud.telemetry.context import get_current_trace_id

            assert get_current_trace_id() is None
            opened += 1
            try:
                async with LocalRuntime(env)(task) as address:
                    yield address
            finally:
                closed += 1

        rows = [row for suite in TASK_SUITES for row in tasks([0], [0], suite=suite)]
        async with (
            DreamscaleRobotAgent(connector=connect, emit=emit, connect_in_background=True) as agent,
            job_runtime(provider, rows[0], emit) as shared,
        ):
            job = await Taskset("overlapped-reuse", rows).run(
                agent, runtime=MeasuredRuntime(shared, emit), max_concurrent=1
            )
            assert opened == 1 and closed == 0
            assert len(job.runs) == 5
            assert all(run.reward == 1 and not run.trace.is_error for run in job.runs)
        assert closed == 1 and policy.closed == 1
        assert policy.calls == 5 and bridge._registry.all_free
        assert [float(ep[0][0, 0]) for ep in bridge.episodes] == [1, 2, 3, 4, 5]


async def test_cancel_during_background_connection_finishes_connector_cleanup():
    entered, cleaned = asyncio.Event(), asyncio.Event()

    async def connect(**kwargs):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    async def run():
        async with DreamscaleRobotAgent(connector=connect, connect_in_background=True):
            await asyncio.Event().wait()

    running = asyncio.create_task(run())
    await asyncio.wait_for(entered.wait(), 5)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert cleaned.is_set()


async def test_background_identity_failure_closes_without_sending_actions():
    policy = FakePolicy()
    policy.resolved_optimization_config.backend = "pytorch"

    async def connect(**kwargs):
        return policy

    async with environment() as (env, bridge):
        async with DreamscaleRobotAgent(connector=connect, connect_in_background=True) as agent:
            job = await Taskset("bad-background-identity", tasks([0], [0])).run(
                agent, runtime=LocalRuntime(env)
            )
        assert job.runs[0].trace.is_error
        assert policy.closed == 1 and policy.calls == 0
        assert bridge.actions == [] and bridge._registry.all_free


async def test_measured_shared_cleanup_preserves_the_original_inference_error():
    from hud_dreamscale.telemetry import CURRENT_TASK

    policy = FakePolicy(bad=True)

    async def connect(**kwargs):
        return policy

    async with environment() as (env, bridge):
        async with (
            DreamscaleRobotAgent(connector=connect) as agent,
            Shared(LocalRuntime(env), width=1) as shared,
        ):
            job = await Taskset("malformed-shared-response", tasks([0], [0])).run(
                agent, runtime=MeasuredRuntime(shared, lambda *a, **kw: None)
            )
        assert job.runs[0].trace.is_error
        assert "Context" not in str(job.runs[0].trace.error)
        assert "finite" in str(job.runs[0].trace.error)
        assert bridge._registry.all_free and policy.closed == 1
        assert CURRENT_TASK.get() is None
