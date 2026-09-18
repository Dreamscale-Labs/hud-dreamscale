import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
from hud import Runtime
from hud.telemetry.context import get_current_trace_id

from hud_dropbear.pooled_cli import cohort_tasks
from hud_dropbear.runtime_pool import RuntimePool


@pytest.fixture(autouse=True)
def mock_control_connection(monkeypatch):
    @asynccontextmanager
    async def ready(runtime, **kwargs):
        yield SimpleNamespace(
            manifest=SimpleNamespace(server_info=SimpleNamespace(name="dropbear-libero-pooled"))
        )

    monkeypatch.setattr("hud_dropbear.runtime_pool.connect", ready)


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
        async with RuntimePool(provider, cohort_tasks(3), concurrency=3):
            pytest.fail("Startup should fail")
    assert opened == closed == [0]
    assert cancelled == [2]


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
                manifest=SimpleNamespace(server_info=SimpleNamespace(name="dropbear-libero-pooled"))
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
