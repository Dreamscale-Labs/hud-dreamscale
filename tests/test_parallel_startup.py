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
