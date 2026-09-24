import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from hud import Runtime
from hud.telemetry.context import get_current_trace_id

from hud_dreamscale.cohort import cohort_tasks
from hud_dreamscale.runtime_pool import RuntimePool
from hud_dreamscale.telemetry import CURRENT_TASK


@pytest.fixture(autouse=True)
def mock_control_connection(monkeypatch):
    @asynccontextmanager
    async def ready(runtime, **kwargs):
        yield SimpleNamespace(
            manifest=SimpleNamespace(server_info=SimpleNamespace(name="dreamscale-libero-pooled"))
        )

    monkeypatch.setattr("hud_dreamscale.runtime_pool.connect", ready)


async def test_pool_exclusive_stable_lane_and_nested_borrow():
    opened, closed = [], []

    @asynccontextmanager
    async def provider(task):
        assert get_current_trace_id() is None
        slot = task.columns["lane_id"]
        opened.append(slot)
        try:
            yield Runtime(f"tcp://lane-{slot}")
        finally:
            closed.append(slot)

    rows = cohort_tasks(2)
    async with RuntimePool(provider, rows, concurrency=2) as pool:
        async with pool:  # Taskset.run's public context-manager handling.
            assert opened == [0, 1]
            async with pool(rows[0]) as first:
                lane = pool.lane_for(first.url)
                first_episode = lane.episode_id
                assert lane.slot == 0 and lane.episodes == 1
                waiting = asyncio.create_task(pool(rows[2]).__aenter__())
                await asyncio.sleep(0)
                assert not waiting.done()
                waiting.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await waiting
            async with pool(rows[2]) as second:
                assert first is second
                assert lane.episode_id != first_episode and lane.episodes == 2
        assert closed == []
    assert sorted(closed) == [0, 1]


async def test_partial_startup_failure_closes_successes_and_cancels_other_starts():
    opened, closed, cancelled = [], [], []
    events = []
    ready = asyncio.Event()

    @asynccontextmanager
    async def provider(task):
        slot = task.columns["lane_id"]
        if slot == 1:
            await ready.wait()
            raise RuntimeError("startup failed")
        if slot == 2:
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.append(slot)
        opened.append(slot)
        ready.set()
        try:
            yield Runtime(f"tcp://lane-{slot}")
        finally:
            closed.append(slot)

    with pytest.raises(RuntimeError, match="startup failed"):
        async with RuntimePool(
            provider,
            cohort_tasks(3),
            concurrency=3,
            emit=lambda event, **fields: events.append({"event": event, **fields}),
        ):
            pytest.fail("Startup should fail")
    assert opened == closed == [0]
    assert cancelled == [2]
    failure = next(e for e in events if e["event"] == "simulator_startup_failed")
    cancellation = next(e for e in events if e["event"] == "simulator_startup_cancelled")
    assert (failure["lane_id"], failure["phase"], failure["error_type"]) == (
        1,
        "runtime_lease",
        "RuntimeError",
    )
    assert (cancellation["lane_id"], cancellation["phase"]) == (2, "runtime_lease")
    assert "startup failed" not in json.dumps(events)


async def test_cancellation_waits_for_all_runtime_cleanup():
    entered, released = asyncio.Event(), []

    @asynccontextmanager
    async def provider(task):
        try:
            yield Runtime(f"tcp://lane-{task.columns['lane_id']}")
        finally:
            await asyncio.sleep(0.01)
            released.append(task.columns["lane_id"])

    async def run():
        async with RuntimePool(provider, cohort_tasks(2), concurrency=2):
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(run())
    await asyncio.wait_for(entered.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sorted(released) == [0, 1]


async def test_duplicate_runtime_addresses_fail_closed():
    closed = []

    @asynccontextmanager
    async def provider(task):
        try:
            yield Runtime("tcp://same")
        finally:
            closed.append(task.columns["lane_id"])

    with pytest.raises(ValueError, match="distinct scalar runtime"):
        async with RuntimePool(provider, cohort_tasks(2), concurrency=2):
            pass
    assert sorted(closed) == [0, 1]


async def test_lazy_hud_runtime_is_initialized_before_pool_becomes_ready():
    leases, instances, clients_closed = [], [], []
    events = []

    @asynccontextmanager
    async def provider(task):
        slot = task.columns["lane_id"]
        leases.append(slot)
        yield Runtime(f"tcp://lazy-{slot}", params={"session_id": str(slot)})

    @asynccontextmanager
    async def connect(runtime, **kwargs):
        assert get_current_trace_id() is None
        slot = int(runtime.params["session_id"])
        assert slot in leases and slot not in instances
        instances.append(slot)  # Platform boot begins only on the first control connection.
        try:
            yield SimpleNamespace(
                manifest=SimpleNamespace(
                    server_info=SimpleNamespace(name="dreamscale-libero-pooled")
                )
            )
        finally:
            clients_closed.append(slot)

    async with RuntimePool(
        provider,
        cohort_tasks(8),
        concurrency=8,
        connector=connect,
        emit=lambda event, **fields: events.append((event, fields)),
    ):
        assert len(instances) == len(leases) == len(clients_closed) == 8
        for slot in range(8):
            names = [event for event, fields in events if fields["lane_id"] == slot]
            assert names == [
                "simulator_starting",
                "runtime_lease_acquired",
                "simulator_control_ready",
                "simulator_acquired",
            ]


@pytest.mark.parametrize("drain_timeout", [False, True])
async def test_cleanup_logging_failures_preserve_all_owner_teardown_and_errors(drain_timeout):
    closed, cleanup_events = [], []

    @asynccontextmanager
    async def provider(task):
        slot = task.columns["lane_id"]
        try:
            yield Runtime(f"tcp://lane-{slot}")
        finally:
            closed.append(slot)
            if slot == 1:
                raise RuntimeError("owner cleanup failed")

    def emit(event, **fields):
        if event in ("episode_cleanup_error", "simulator_cleanup_error", "simulator_closed"):
            cleanup_events.append((event, fields["lane_id"]))
            raise OSError("evidence disk full")

    pool = RuntimePool(provider, cohort_tasks(3), concurrency=3, emit=emit, cleanup_timeout=0.01)
    await pool.__aenter__()
    if drain_timeout:
        await pool.lanes[0].lock.acquire()
    try:
        with pytest.raises(ExceptionGroup, match="Simulator cleanup failed") as failure:
            await pool.close()
    finally:
        if drain_timeout:
            pool.lanes[0].lock.release()

    assert sorted(closed) == [0, 1, 2]
    assert cleanup_events == ([("episode_cleanup_error", 0)] if drain_timeout else []) + [
        ("simulator_closed", 0),
        ("simulator_cleanup_error", 1),
        ("simulator_closed", 2),
    ]
    errors = failure.value.exceptions
    assert sum(type(error) is OSError for error in errors) == 3 + drain_timeout
    assert sum(isinstance(error, RuntimeError) for error in errors) == 1
    assert sum(isinstance(error, TimeoutError) for error in errors) == drain_timeout
    assert not pool.cleanup_confirmed and "OSError" in pool.cleanup_error
    assert pool._depth == 0


@pytest.mark.parametrize(
    "failed_event", ["environment_starting", "environment_acquired", "episode_released"]
)
async def test_episode_logging_failure_always_clears_lane_and_task_context(failed_event):
    closed = []

    @asynccontextmanager
    async def provider(task):
        try:
            yield Runtime("tcp://lane-0")
        finally:
            closed.append(0)

    def emit(event, **fields):
        if event == failed_event:
            raise OSError("evidence disk full")

    rows = cohort_tasks(1)
    token = CURRENT_TASK.set("outer-task")
    try:
        async with RuntimePool(provider, rows, concurrency=1, emit=emit) as pool:
            with pytest.raises(OSError, match="evidence disk full"):
                async with pool(rows[0]):
                    assert CURRENT_TASK.get() == rows[0].slug
            lane = pool.lanes[0]
            assert lane.current_task is None and lane.episode_id is None
            assert not lane.lock.locked()
            assert CURRENT_TASK.get() == "outer-task"
        assert pool.cleanup_confirmed
    finally:
        CURRENT_TASK.reset(token)
    assert closed == [0]


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True, "900", None])
def test_startup_timeout_requires_positive_finite_number(timeout):
    with pytest.raises(ValueError, match="positive finite"):
        RuntimePool(None, cohort_tasks(1), concurrency=1, startup_timeout=timeout)


async def test_actual_hud_connect_uses_pool_timeout_without_mutating_provider_descriptor():
    from hud.clients import connect as hud_connect
    from hud.environment.utils import read_frame, send_frame

    attempts, closed, handlers = 0, [], set()

    async def control(reader, writer):
        nonlocal attempts
        handlers.add(asyncio.current_task())
        try:
            request = await read_frame(reader)
            assert request["method"] == "hello"
            attempts += 1
            if attempts <= 2:
                await asyncio.sleep(0.04)
                return  # Accepted before the remote control server is ready.
            await send_frame(
                writer,
                {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": {
                        "session_id": "control-session",
                        "bindings": [],
                        "env": {"name": "dreamscale-libero-pooled", "version": "1"},
                    },
                },
            )
            await reader.read()
        finally:
            writer.close()
            await writer.wait_closed()
            handlers.remove(asyncio.current_task())

    server = await asyncio.start_server(control, "127.0.0.1", 0)
    original = Runtime(
        f"tcp://127.0.0.1:{server.sockets[0].getsockname()[1]}",
        params={"ready_timeout": 0.02, "session_id": "lease", "opaque": "preserved"},
    )
    original_params = original.params.copy()

    @asynccontextmanager
    async def provider(task):
        try:
            yield original
        finally:
            closed.append(original)

    try:
        # Reproduce the actual SDK precedence: an explicit keyword alone loses.
        with pytest.raises(EOFError):
            async with hud_connect(original, ready_timeout=2):
                pytest.fail("The provider's shorter default still overrides the keyword")
        assert attempts == 1
        attempts = 0
        async with RuntimePool(
            provider,
            cohort_tasks(1),
            concurrency=1,
            startup_timeout=2,
            connector=hud_connect,
        ) as pool:
            effective = pool.lanes[0].runtime
            assert attempts == 3  # Real HUD retry loop survives both early EOFs.
            assert effective is not original
            assert effective.params == {**original_params, "ready_timeout": 2}
            assert effective.url == original.url and effective.config is original.config
            assert original.params == original_params
        assert closed == [original]
    finally:
        server.close()
        await server.wait_closed()
        if handlers:
            await asyncio.wait_for(asyncio.gather(*handlers), 2)


async def test_hanging_actual_hello_keeps_outer_startup_bound_and_cancelled_diagnostic():
    from hud.clients import connect as hud_connect
    from hud.environment.utils import read_frame

    handlers, closed, events = set(), [], []

    async def control(reader, writer):
        handlers.add(asyncio.current_task())
        try:
            await read_frame(reader)
            await reader.read()  # An accepted connection that never answers hello.
        finally:
            writer.close()
            await writer.wait_closed()
            handlers.remove(asyncio.current_task())

    server = await asyncio.start_server(control, "127.0.0.1", 0)
    original = Runtime(
        f"tcp://127.0.0.1:{server.sockets[0].getsockname()[1]}",
        params={"ready_timeout": 900},
    )

    @asynccontextmanager
    async def provider(task):
        try:
            yield original
        finally:
            closed.append(True)

    try:
        with pytest.raises(TimeoutError):
            async with (
                asyncio.timeout(2),
                RuntimePool(
                    provider,
                    cohort_tasks(1),
                    concurrency=1,
                    startup_timeout=0.05,
                    connector=hud_connect,
                    emit=lambda event, **fields: events.append({"event": event, **fields}),
                ),
            ):
                pytest.fail("A hanging hello must remain bounded")
        row = next(e for e in events if e["event"] == "simulator_startup_cancelled")
        assert row["phase"] == "control_hello" and row["error_type"] == "CancelledError"
        assert row["duration_s"] < 1
        assert closed == [True] and original.params == {"ready_timeout": 900}
    finally:
        server.close()
        await server.wait_closed()
        if handlers:
            await asyncio.wait_for(asyncio.gather(*handlers), 2)


async def test_startup_error_telemetry_is_bounded_and_cannot_suppress_cleanup():
    events, closed = [], []

    @asynccontextmanager
    async def provider(task):
        try:
            yield Runtime("tcp://private.invalid", params={"auth": "SECRET"})
        finally:
            closed.append(True)

    @asynccontextmanager
    async def failing_connection(runtime, **kwargs):
        raise EOFError("private.invalid SECRET")
        yield

    def emit(event, **fields):
        events.append({"event": event, **fields})
        if event == "simulator_startup_failed":
            raise OSError("telemetry unavailable")

    with pytest.raises(EOFError, match="SECRET"):
        async with RuntimePool(
            provider,
            cohort_tasks(1),
            concurrency=1,
            connector=failing_connection,
            emit=emit,
            lane_ready_attempts=1,
        ):
            pass
    row = next(e for e in events if e["event"] == "simulator_startup_failed")
    assert row["phase"] == "control_hello" and row["error_type"] == "EOFError"
    assert row["duration_s"] >= 0 and row["startup_timeout_s"] == 900
    assert "SECRET" not in json.dumps(events) and "private.invalid" not in json.dumps(events)
    assert closed == [True]


async def test_unready_lane_is_released_and_replaced_within_bounded_attempts():
    """A leased simulator whose control never becomes ready is released, then replaced."""
    events, opened, closed = [], [], []
    leases = {0: 0, 1: 0}

    @asynccontextmanager
    async def provider(task):
        slot = task.columns["lane_id"]
        leases[slot] += 1
        name = f"tcp://lane-{slot}-{leases[slot]}"
        opened.append(name)
        try:
            yield Runtime(name, params={"session_id": name})
        finally:
            closed.append(name)

    @asynccontextmanager
    async def connector(runtime, **kwargs):
        if runtime.url == "tcp://lane-1-1":
            await asyncio.sleep(10)  # never becomes ready inside the lane bound
        yield SimpleNamespace(
            manifest=SimpleNamespace(server_info=SimpleNamespace(name="dreamscale-libero-pooled"))
        )

    async with RuntimePool(
        provider,
        cohort_tasks(2),
        concurrency=2,
        connector=connector,
        emit=lambda event, **fields: events.append({"event": event, **fields}),
        startup_timeout=5,
        lane_ready_timeout=0.05,
        lane_ready_attempts=2,
    ) as pool:
        assert [lane.runtime.url for lane in pool.lanes] == ["tcp://lane-0-1", "tcp://lane-1-2"]
        assert closed == ["tcp://lane-1-1"]  # the unready lease was released before replacement
    assert (
        sorted(closed) == sorted(opened) == ["tcp://lane-0-1", "tcp://lane-1-1", "tcp://lane-1-2"]
    )
    retry = [e for e in events if e["event"] == "simulator_control_retry"]
    assert len(retry) == 1 and retry[0]["lane_id"] == 1 and retry[0]["attempt"] == 1
    assert retry[0]["phase"] == "control_hello" and retry[0]["error_type"] == "TimeoutError"
    ready = {e["lane_id"]: e["attempts"] for e in events if e["event"] == "simulator_control_ready"}
    assert ready == {0: 1, 1: 2}
    assert [
        e["event"] for e in events if e["lane_id"] == 1 and e["event"] == "simulator_starting"
    ] == ["simulator_starting"]


async def test_lane_readiness_retries_are_bounded_and_then_fail_closed():
    events, closed = [], []

    @asynccontextmanager
    async def provider(task):
        try:
            yield Runtime(f"tcp://lane-{task.columns['lane_id']}-{len(closed)}")
        finally:
            closed.append(task.columns["lane_id"])

    @asynccontextmanager
    async def never_ready(runtime, **kwargs):
        await asyncio.sleep(10)
        yield

    with pytest.raises(TimeoutError):
        async with RuntimePool(
            provider,
            cohort_tasks(1),
            concurrency=1,
            connector=never_ready,
            emit=lambda event, **fields: events.append({"event": event, **fields}),
            startup_timeout=5,
            lane_ready_timeout=0.05,
            lane_ready_attempts=2,
        ):
            pytest.fail("Startup should fail")
    assert closed == [0, 0]
    lifecycle = [
        e["event"]
        for e in events
        if e["event"]
        in ("simulator_starting", "simulator_control_retry", "simulator_startup_failed")
    ]
    assert lifecycle == [
        "simulator_starting",
        "simulator_control_retry",
        "simulator_startup_failed",
    ]
    failed = next(e for e in events if e["event"] == "simulator_startup_failed")
    assert failed["phase"] == "control_hello" and failed["attempt"] == 2


async def test_identity_mismatch_and_cancellation_are_never_retried():
    events, closed = [], []

    @asynccontextmanager
    async def provider(task):
        try:
            yield Runtime("tcp://lane-0")
        finally:
            closed.append(True)

    @asynccontextmanager
    async def wrong_environment(runtime, **kwargs):
        yield SimpleNamespace(manifest=SimpleNamespace(server_info=SimpleNamespace(name="other")))

    with pytest.raises(ValueError, match="wrong HUD environment"):
        async with RuntimePool(
            provider,
            cohort_tasks(1),
            concurrency=1,
            connector=wrong_environment,
            emit=lambda event, **fields: events.append({"event": event, **fields}),
            lane_ready_attempts=3,
        ):
            pytest.fail("Startup should fail")
    assert closed == [True]
    assert not [e for e in events if e["event"] == "simulator_control_retry"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"lane_ready_timeout": 0},
        {"lane_ready_timeout": 1000},
        {"lane_ready_timeout": float("inf")},
        {"lane_ready_attempts": 0},
        {"lane_ready_attempts": 1.5},
    ],
)
def test_lane_readiness_parameters_are_validated(kwargs):
    with pytest.raises(ValueError):
        RuntimePool(lambda task: None, cohort_tasks(1), concurrency=1, **kwargs)
