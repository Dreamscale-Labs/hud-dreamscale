"""Atomic non-admission permits a new identity, never an uncertain POST replay."""

import asyncio
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from hud_dreamscale import pooled
from hud_dreamscale.pooled import PooledProvider, RequestFailedError, SlotFencedError
from tests.test_pooled import Server, observation, predict


class RejectionServer(Server):
    def __init__(self, rejections=1):
        super().__init__()
        self.rejections = rejections
        self.code = "request_not_admitted"
        self.resolution_mode = "terminal"
        self.block_second = False
        self.second_entered = asyncio.Event()
        self.resolution_delay = 0

    async def handle(self, request):
        if request.url.path == "/v1/actions":
            self.mode = "unadmitted" if len(self.posts) < self.rejections else "success"
            if self.block_second and len(self.posts) == 1:
                self.mode = "block"
                self.second_entered.set()
            if self.journal and self.posts and len(self.posts) <= self.rejections:
                rows = [json.loads(line) for line in self.journal.read_text().splitlines()]
                previous = self.posts[-1]["request_id"]
                assert any(r["phase"] == "failed" and r["request_id"] == previous for r in rows)
        if request.url.path.endswith("/resolve"):
            await asyncio.sleep(self.resolution_delay)
            if self.resolution_mode == "uncertain":
                raise httpx.ReadTimeout("resolution acknowledgment unavailable")
            response = await super().handle(request)
            value = response.json()
            if self.resolution_mode == "pending":
                return httpx.Response(
                    202, json={**json.loads(request.content), "status": "pending"}
                )
            value["error"]["code"] = self.code
            if self.resolution_mode == "wrong_identity":
                value["episode_id"] = "other"
            return httpx.Response(422, json=value)
        return await super().handle(request)


@pytest.mark.parametrize("limit", [1, 2])
async def test_atomic_resubmission_preserves_input_and_records_every_attempt(tmp_path, limit):
    server, events = RejectionServer(rejections=limit), []
    server.journal = tmp_path / "requests.jsonl"
    server.resolution_delay = 0.002
    raw = observation()

    def emit(name, **fields):
        events.append((name, fields))
        if name == "inference_resubmission":
            raw["agentview_image"][:] = 255  # Caller mutation cannot change the owned input.

    async with server.provider(
        concurrency=1,
        max_not_admitted_resubmissions=limit,
        journal_path=server.journal,
        emit=emit,
    ) as provider:
        result = await provider.predict(0, raw, "pick up the bowl", "first", 197, trace_id="trace")
        assert np.all(result == limit + 1)
        assert provider.is_slot_available(0)
        assert len(server.posts) == limit + 1
        await predict(provider, episode="next")
    assert [row["sequence"] for row in server.posts] == list(range(limit + 2))
    assert len({row["request_id"] for row in server.posts}) == limit + 2
    original = {k: v for k, v in server.posts[0].items() if k not in {"request_id", "sequence"}}
    assert all(
        {k: v for k, v in row.items() if k not in {"request_id", "sequence"}} == original
        for row in server.posts[: limit + 1]
    )
    assert [r["request_id"] for r in server.resolutions] == [
        r["request_id"] for r in server.posts[:limit]
    ]
    rows = [json.loads(line) for line in server.journal.read_text().splitlines()]
    attempts = [r for r in rows if r["phase"] == "before_post"]
    assert [r["attempt_index"] for r in attempts] == [*range(limit + 1), 0]
    assert len({r["logical_call_id"] for r in attempts[: limit + 1]}) == 1
    assert attempts[-1]["logical_call_id"] != attempts[0]["logical_call_id"]
    failures = [r for name, r in events if name == "inference_failed"]
    assert len(failures) == limit and all(r["resolution_source"] == "resolve" for r in failures)
    final = next(r for name, r in events if name == "provider_inference")
    assert final["duration_s"] >= limit * server.resolution_delay
    assert final["duration_s"] >= sum(
        r["duration_s"]
        for name, r in events
        if name == "inference_post" and r["logical_call_id"] == final["logical_call_id"]
    )
    assert server.stops == 1


@pytest.mark.parametrize("limit", [0, 1, 2])
async def test_resubmission_exhaustion_is_terminal_and_attempts_remain_visible(limit):
    server, events = RejectionServer(rejections=10), []
    async with server.provider(
        concurrency=1,
        max_not_admitted_resubmissions=limit,
        emit=lambda name, **fields: events.append((name, fields)),
    ) as provider:
        with pytest.raises(RequestFailedError, match="request_not_admitted"):
            await predict(provider)
        assert provider.is_slot_available(0)
    assert len(server.posts) == len(server.resolutions) == limit + 1
    assert len([1 for name, _ in events if name == "inference_failed"]) == limit + 1
    assert not any(name == "provider_inference" for name, _ in events)


@pytest.mark.parametrize("mode", ["pending", "uncertain", "wrong_identity"])
async def test_uncertain_claimed_or_mismatched_resolution_never_resubmits(mode):
    server = RejectionServer()
    server.resolution_mode = mode
    async with server.provider(concurrency=1, max_not_admitted_resubmissions=2) as provider:
        with pytest.raises(ValueError if mode == "wrong_identity" else TimeoutError):
            await predict(provider)
        assert provider.fenced_slots == {0}
    assert len(server.posts) == 1


@pytest.mark.parametrize("code", ["expired", "preprocessing_failed"])
async def test_other_terminal_failures_never_resubmit(code):
    server = RejectionServer()
    server.code = code
    async with server.provider(concurrency=1, max_not_admitted_resubmissions=2) as provider:
        with pytest.raises(RequestFailedError, match=code):
            await predict(provider)
    assert len(server.posts) == len(server.resolutions) == 1


async def test_lookup_failure_alone_cannot_authorize_resubmission(monkeypatch):
    server = RejectionServer()
    server.resolution_mode = "pending"
    async with server.provider(concurrency=1, max_not_admitted_resubmissions=1) as provider:

        async def lookup(**identity):
            return {
                **identity,
                "episode_id": "first",
                "status": "failed",
                "error": {"code": "request_not_admitted"},
            }

        monkeypatch.setattr(provider._clients[0], "get_result", lookup)
        with pytest.raises(TimeoutError):
            await predict(provider)
    assert len(server.posts) == 1 and server.resolutions


async def test_completed_original_is_returned_without_resubmission():
    server = Server()
    server.mode = "response_lost"
    async with server.provider(concurrency=1, max_not_admitted_resubmissions=2) as provider:
        assert np.all(await predict(provider) == 1)
    assert len(server.posts) == 1 and not server.resolutions


@pytest.mark.parametrize("phase", ["failed", "before_post"])
async def test_journal_failure_prevents_resubmission(tmp_path, monkeypatch, phase):
    server = RejectionServer()
    server.journal = tmp_path / "journal.jsonl"
    async with server.provider(
        concurrency=1,
        max_not_admitted_resubmissions=1,
        journal_path=server.journal,
    ) as provider:
        checkpoint = provider._checkpoint

        async def fail(phase_name, **fields):
            if phase_name == phase and (phase == "failed" or fields.get("attempt_index") == 1):
                raise OSError("test write failure")
            return await checkpoint(phase_name, **fields)

        monkeypatch.setattr(provider, "_checkpoint", fail)
        with pytest.raises(OSError):
            await predict(provider)
    assert len(server.posts) == 1


async def test_cancellation_between_attempts_sends_no_new_identity():
    server = RejectionServer()

    def emit(name, **fields):
        if name == "inference_resubmission":
            asyncio.current_task().cancel()

    async with server.provider(
        concurrency=1, max_not_admitted_resubmissions=2, emit=emit
    ) as provider:
        task = asyncio.create_task(predict(provider))
        with pytest.raises(asyncio.CancelledError):
            await task
        assert provider.is_slot_available(0)
    assert len(server.posts) == len(server.resolutions) == 1


async def test_cancellation_during_second_attempt_resolves_only_its_identity():
    server = RejectionServer()
    server.block_second = True
    async with server.provider(concurrency=1, max_not_admitted_resubmissions=2) as provider:
        task = asyncio.create_task(predict(provider))
        await asyncio.wait_for(server.second_entered.wait(), 1)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert provider.is_slot_available(0)
    assert len(server.posts) == len(server.resolutions) == 2
    assert [r["request_id"] for r in server.posts] == [r["request_id"] for r in server.resolutions]


async def test_cancellation_with_uncertain_second_attempt_fences_and_stops():
    server = RejectionServer()
    server.block_second = True
    async with server.provider(concurrency=1, max_not_admitted_resubmissions=2) as provider:
        task = asyncio.create_task(predict(provider))
        await asyncio.wait_for(server.second_entered.wait(), 1)
        server.resolution_mode = "pending"
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert provider.fenced_slots == {0}
    assert len(server.posts) == 2 and server.stops == 1


async def test_expired_grant_stops_between_attempts():
    server = RejectionServer()

    def emit(name, **fields):
        if name == "inference_resubmission":
            server.deployment["expires_at"] = 1
            provider.deployment["expires_at"] = 1

    async with server.provider(
        concurrency=1, max_not_admitted_resubmissions=1, emit=emit
    ) as provider:
        with pytest.raises(SlotFencedError):
            await predict(provider)
    assert len(server.posts) == 1


@pytest.mark.parametrize("value", [-1, 3, True, 1.0, None, "1"])
def test_resubmission_limit_requires_bounded_integer(value):
    with pytest.raises(ValueError, match="max_not_admitted_resubmissions"):
        PooledProvider(max_not_admitted_resubmissions=value)


async def test_journal_phase_timings_separate_queue_io_and_resume(monkeypatch):
    clock, events, encoded = [100.0], [], asyncio.Event()
    provider = PooledProvider(
        concurrency=1, emit=lambda name, **fields: events.append((name, fields))
    )
    monkeypatch.setattr(pooled, "time", SimpleNamespace(time=lambda: 1, monotonic=lambda: clock[0]))

    def advance(seconds):
        clock[0] += seconds

    provider._journal = SimpleNamespace(
        write=lambda line: advance(1),
        flush=lambda: advance(1),
        fileno=lambda: 123,
    )
    monkeypatch.setattr(pooled.os, "fsync", lambda fd: advance(2))
    original = provider._protected
    to_thread = pooled.asyncio.to_thread

    async def queued_write(write):
        advance(1)  # Controlled executor queue interval.
        return await to_thread(write)

    monkeypatch.setattr(pooled.asyncio, "to_thread", queued_write)

    async def protected(*, write):
        advance(2)  # Event-loop scheduling before the thread-pool submission.
        result = await original(write=write)
        advance(3)  # Callback/event-loop resume delay after the worker completed.
        return result

    monkeypatch.setattr(provider, "_protected", protected)

    async def checkpoint():
        encoded.set()
        await provider._checkpoint("before_post", request_id="request")

    try:
        async with provider._journal_lock:
            task = asyncio.create_task(checkpoint())
            await encoded.wait()
            advance(5)
        await task
    finally:
        provider._encoder.shutdown(wait=False)
    row = events[0][1]
    assert row["lock_wait_s"] == 5
    assert row["dispatch_delay_s"] == 2
    assert row["executor_queue_s"] == 1
    assert row["write_flush_fsync_s"] == 4
    assert row["resume_delay_s"] == 3
    assert row["duration_s"] == 15 and row["durable"] is True


async def test_journal_io_failure_has_measured_duration_but_no_durability(monkeypatch):
    events = []
    provider = PooledProvider(
        concurrency=1, emit=lambda name, **fields: events.append((name, fields))
    )
    provider._journal = SimpleNamespace(
        write=lambda line: None, flush=lambda: None, fileno=lambda: 1
    )

    def failed_fsync(fd):
        raise OSError("test fsync failure")

    monkeypatch.setattr(pooled.os, "fsync", failed_fsync)
    try:
        with pytest.raises(OSError, match="test fsync failure"):
            await provider._checkpoint("before_post", request_id="request")
    finally:
        provider._encoder.shutdown(wait=False)
    assert events[0][0] == "journal_checkpoint"
    row = events[0][1]
    assert row["durable"] is False
    assert row["write_flush_fsync_s"] >= 0 and row["resume_delay_s"] >= 0
    assert row["duration_s"] >= row["write_flush_fsync_s"]
