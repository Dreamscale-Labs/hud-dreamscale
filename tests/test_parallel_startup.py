import asyncio
from contextlib import asynccontextmanager

import pytest

from hud_dreamscale.campaign_run import parallel_startup


@pytest.mark.parametrize("body_fails", [False, True])
async def test_independent_startups_overlap_with_readiness_barrier_and_ordered_cleanup(body_fails):
    started = [asyncio.Event(), asyncio.Event()]
    ready = []
    closed = []
    events = []

    @asynccontextmanager
    async def resource(index):
        started[index].set()
        # A sequential implementation deadlocks: each side needs the other started.
        await started[1 - index].wait()
        ready.append(index)
        try:
            yield
        finally:
            closed.append(index)

    async def exercise():
        async with parallel_startup(
            resource(0), resource(1), emit=lambda event, **kw: events.append((event, kw))
        ):
            assert set(ready) == {0, 1}
            assert not closed
            if body_fails:
                raise ValueError("rollout failed")

    if body_fails:
        with pytest.raises(ValueError, match="rollout failed"):
            await asyncio.wait_for(exercise(), 1)
    else:
        await asyncio.wait_for(exercise(), 1)
    assert closed == [1, 0]  # Simulator shutdown always precedes compute shutdown.
    assert [e[0] for e in events] == ["resources_starting", "resources_ready"]
    assert events[1][1]["duration_s"] >= 0


@pytest.mark.parametrize("failed", [0, 1])
@pytest.mark.parametrize("sibling_ready", [False, True])
async def test_startup_failure_joins_sibling_and_cleans_partial_or_completed_entry(
    failed, sibling_ready
):
    allocated = set()
    released = []
    sibling_started = asyncio.Event()
    sibling_cancelled = asyncio.Event()
    events = []

    @asynccontextmanager
    async def resource(index):
        allocated.add(index)
        try:
            if index == failed:
                await sibling_started.wait()
                raise ValueError("startup failed")
            sibling_started.set()
            if not sibling_ready:
                try:
                    await asyncio.Event().wait()
                except asyncio.CancelledError:
                    sibling_cancelled.set()
                    raise
            yield
        finally:
            allocated.remove(index)
            released.append(index)

    with pytest.raises(ExceptionGroup, match="TaskGroup"):
        async with asyncio.timeout(1):
            async with parallel_startup(
                resource(0), resource(1), emit=lambda event, **kw: events.append(event)
            ):
                pytest.fail("rollouts must not start")
    assert not allocated
    assert sorted(released) == [0, 1]
    assert sibling_cancelled.is_set() is not sibling_ready
    assert events == ["resources_starting"]


async def test_external_cancellation_waits_for_both_partial_startups_to_clean_up():
    started = [asyncio.Event(), asyncio.Event()]
    closed = []

    @asynccontextmanager
    async def resource(index):
        started[index].set()
        try:
            await asyncio.Event().wait()
            yield
        finally:
            await asyncio.sleep(0)
            closed.append(index)

    async def run():
        async with parallel_startup(resource(0), resource(1), emit=lambda *a, **kw: None):
            pytest.fail("rollouts must not start")

    task = asyncio.create_task(run())
    await asyncio.wait_for(asyncio.gather(*(event.wait() for event in started)), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sorted(closed) == [0, 1]


async def test_compute_failure_during_hud_post_receives_and_closes_remote_lease(monkeypatch):
    from hud import HUDRuntime

    from hud_dreamscale.cohort import cohort_tasks
    from hud_dreamscale.runtime_pool import RuntimePool

    allocated, response_ready = asyncio.Event(), asyncio.Event()
    deleted, evidence = [], []

    async def create(*args):
        allocated.set()
        await response_ready.wait()
        return "owned-session"

    async def delete(*args):
        deleted.append(args[-1])

    monkeypatch.setattr(HUDRuntime, "_create_runtime_session", create)
    monkeypatch.setattr(HUDRuntime, "_delete_runtime_session", delete)
    monkeypatch.setattr("hud.settings.settings.api_key", "test-key")

    @asynccontextmanager
    async def failed_compute():
        await allocated.wait()
        raise ValueError("admission denied")
        yield

    pool = RuntimePool(
        HUDRuntime(),
        cohort_tasks(1),
        concurrency=1,
        lease_acquisition_timeout=1,
        emit=lambda event, **fields: evidence.append({"event": event, **fields}),
    )

    async def exercise():
        async with parallel_startup(failed_compute(), pool, emit=lambda *a, **kw: None):
            pytest.fail("must not start rollout")

    task = asyncio.create_task(exercise())
    await allocated.wait()
    for _ in range(10):
        await asyncio.sleep(0)
    assert not task.done()
    response_ready.set()
    with pytest.raises(ExceptionGroup):
        await asyncio.wait_for(task, 1)
    assert deleted == ["owned-session"]
    leases = [e for e in evidence if e["event"] == "runtime_lease_acquired"]
    assert [e["runtime_session_id"] for e in leases] == ["owned-session"]
    assert pool.cleanup_confirmed


async def test_protected_lease_acquisition_still_has_a_deadline():
    from hud_dreamscale.cohort import cohort_tasks
    from hud_dreamscale.runtime_pool import RuntimePool

    cleaned = []

    @asynccontextmanager
    async def stalled_provider(task):
        try:
            await asyncio.Event().wait()
            yield
        finally:
            cleaned.append(True)

    pool = RuntimePool(
        stalled_provider,
        cohort_tasks(1),
        concurrency=1,
        lease_acquisition_timeout=0.01,
        lane_ready_attempts=1,
    )
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(pool.__aenter__(), 1)
    assert cleaned == [True]
