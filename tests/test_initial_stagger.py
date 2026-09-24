import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest

from hud_dreamscale import pooled_agent
from hud_dreamscale.pooled_agent import PooledInput, PooledModel, PooledRobotAgent


def make_agent(width):
    events = []
    agent = PooledRobotAgent(
        provider=SimpleNamespace(concurrency=width),
        runtimes=SimpleNamespace(concurrency=width),
        emit=lambda event, **fields: events.append((event, fields)),
    )
    return agent, events


@pytest.mark.parametrize("width", [1, 3, 8, 9, 16, 64])
async def test_offsets_repeat_per_gpu_and_only_once_per_job(monkeypatch, width):
    sleep = AsyncMock()
    monkeypatch.setattr(pooled_agent.asyncio, "sleep", sleep)
    agent, events = make_agent(width)
    for slot in range(width):
        await agent._stagger_first_inference("job-a", slot, {"lane_id": slot})
    assert [call.args[0] for call in sleep.await_args_list] == [
        (slot % 8) * 0.125 for slot in range(width) if slot % 8
    ]
    assert [fields["requested_delay_s"] for _, fields in events] == [
        (slot % 8) * 0.125 for slot in range(width)
    ]
    assert all(event == "initial_inference_stagger" for event, _ in events)
    assert all(fields["outcome"] == "completed" for _, fields in events)
    count = sleep.await_count
    for slot in range(width):
        await agent._stagger_first_inference("job-a", slot, {"lane_id": slot})
    assert len(events) == width and sleep.await_count == count
    # Even reusing the agent/provider for a different job gets one initial offset.
    for slot in range(width):
        await agent._stagger_first_inference("job-b", slot, {"lane_id": slot})
    assert len(events) == 2 * width and sleep.await_count == 2 * count


async def test_first_call_includes_wait_without_changing_payload_or_later_calls(monkeypatch):
    now = 0.0

    async def sleep(seconds):
        nonlocal now
        now += seconds

    monkeypatch.setattr(pooled_agent.asyncio, "sleep", sleep)
    monkeypatch.setattr(pooled_agent, "time", SimpleNamespace(monotonic=lambda: now))
    agent, events = make_agent(8)
    calls = []
    actions = np.arange(70, dtype=np.float32).reshape(10, 7)

    async def predict(**kwargs):
        nonlocal now
        calls.append(kwargs)
        now += 0.01
        return actions

    batch = PooledInput({"sentinel": object()}, "unchanged instruction", "input-hash")
    for episode in ("one", "two"):
        fields = {"lane_id": 7, "episode_id": episode}
        model = PooledModel(
            SimpleNamespace(predict=predict),
            slot=7,
            episode_id=episode,
            trace_id=f"trace-{episode}",
            seed=197,
            emit=agent.emit,
            fields=fields,
            before_first=lambda fields=fields: agent._stagger_first_inference("job", 7, fields),
        )
        for _ in range(2):
            np.testing.assert_array_equal(await model.ainfer(batch), actions)
    assert [call["noise_seed"] for call in calls] == [197, 198, 197, 198]
    assert [call["episode_id"] for call in calls] == ["one", "one", "two", "two"]
    assert all(call["observation"] is batch.observation for call in calls)
    assert all(call["instruction"] == batch.instruction and call["slot"] == 7 for call in calls)
    assert [call["trace_id"] for call in calls] == ["trace-one"] * 2 + ["trace-two"] * 2
    waits = [fields for event, fields in events if event == "initial_inference_stagger"]
    assert len(waits) == 1 and waits[0]["duration_s"] == 0.875
    for name in ("inference", "inference_attempt_finished"):
        durations = [fields["duration_s"] for event, fields in events if event == name]
        assert durations == pytest.approx([0.885, 0.01, 0.01, 0.01])


async def test_cancellation_during_wait_submits_nothing_and_does_not_block_peer(monkeypatch):
    entered = asyncio.Event()

    async def sleep(_seconds):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(pooled_agent.asyncio, "sleep", sleep)
    agent, events = make_agent(8)
    predict = AsyncMock(return_value=np.zeros((10, 7), np.float32))
    model = PooledModel(
        SimpleNamespace(predict=predict),
        slot=7,
        episode_id="episode",
        trace_id="trace",
        seed=73,
        emit=agent.emit,
        fields={"lane_id": 7},
        before_first=lambda: agent._stagger_first_inference("job", 7, {"lane_id": 7}),
    )
    task = asyncio.create_task(model.ainfer(PooledInput({}, "instruction", "digest")))
    await asyncio.wait_for(entered.wait(), 1)
    await asyncio.wait_for(agent._stagger_first_inference("job", 0, {"lane_id": 0}), 1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    predict.assert_not_awaited()
    assert model.calls == 0
    waits = [fields for event, fields in events if event == "initial_inference_stagger"]
    assert [(row["lane_id"], row["outcome"]) for row in waits] == [
        (0, "completed"),
        (7, "cancelled"),
    ]
    assert events[-1][0] == "inference_attempt_finished"
    assert events[-1][1]["outcome"] == "cancelled"
    # A cancelled initial attempt does not turn subsequent attempts into a rate limiter.
    await asyncio.wait_for(agent._stagger_first_inference("job", 7, {"lane_id": 7}), 1)
    assert len([event for event, _ in events if event == "initial_inference_stagger"]) == 2


async def test_internal_zero_delay_control_never_sleeps(monkeypatch):
    sleep = AsyncMock()
    monkeypatch.setattr(pooled_agent.asyncio, "sleep", sleep)
    monkeypatch.setattr(pooled_agent, "_INITIAL_STAGGER_STEP_S", 0.0)
    agent, _ = make_agent(8)
    for slot in range(8):
        await agent._stagger_first_inference("control", slot, {})
    sleep.assert_not_awaited()
