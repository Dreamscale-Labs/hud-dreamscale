"""Renewable horizons use bounded management reads, outside the model request path."""

import asyncio
import time
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from dreamscale.errors import DreamscaleError

from hud_dreamscale import pooled
from hud_dreamscale.pooled import SlotFencedError
from tests.test_pooled import Server, predict


class RenewalServer(Server):
    def __init__(self):
        super().__init__()
        self.status_reads = 0
        self.override = None
        self.read_started = asyncio.Event()
        self.release_read = asyncio.Event()
        self.block_read = False
        self.fail_read = False
        self.auth_failure = False

    async def handle(self, request):
        status = request.method == "GET" and request.url.path.startswith(
            "/v1/inference/deployments/"
        )
        if status:
            self.status_reads += 1
            if not self.stops and self.status_reads > 1:
                self.read_started.set()
                if self.block_read:
                    await self.release_read.wait()
                if self.fail_read:
                    raise httpx.ConnectTimeout("test control read unavailable")
                if self.auth_failure:
                    return httpx.Response(401, json={"error": {"code": "invalid_api_key"}})
                if self.override:
                    return httpx.Response(200, json={**self.deployment, **self.override})
        return await super().handle(request)


@pytest.fixture
def clock(monkeypatch):
    value = [1000.0]
    monkeypatch.setattr(
        pooled, "time", SimpleNamespace(time=lambda: value[0], monotonic=time.monotonic)
    )
    return value


@pytest.fixture
def fast_refresh(monkeypatch):
    monkeypatch.setattr(pooled, "STATUS_REFRESH_INTERVAL_S", 0.005)
    monkeypatch.setattr(pooled, "STATUS_REFRESH_RETRY_S", 0.005)


async def wait_until(predicate):
    async with asyncio.timeout(1):
        while not predicate():
            await asyncio.sleep(0.001)


async def test_background_refresh_allows_confirmed_renewal_beyond_initial_horizon(
    clock, fast_refresh
):
    server, events = RenewalServer(), []
    server.deployment.update(expires_at=1300, control_lease_until=1300)
    async with server.provider(
        concurrency=1, emit=lambda name, **fields: events.append((name, fields))
    ) as provider:
        clock[0] = 1100
        server.override = {"expires_at": 1400, "control_lease_until": 1400}
        await wait_until(lambda: provider.identity["expires_at"] == 1400)
        clock[0] = 1301
        assert np.all(await predict(provider) == 1)
        assert provider.identity["expires_at"] == 1400
    assert provider._status_task.done() and provider.cleanup_confirmed
    assert any(name == "provider_authority_refreshed" for name, _ in events)
    assert server.creates and len(server.creates) == 1 and server.stops == 1


async def test_normal_predictions_do_not_fetch_management_status():
    server = RenewalServer()
    async with server.provider(concurrency=1) as provider:
        reads = server.status_reads
        for _ in range(3):
            await predict(provider)
        assert server.status_reads == reads == 1
    assert provider._status_task.done()


async def test_finite_grant_cannot_be_extended_by_a_renewed_control_lease(clock, fast_refresh):
    server = RenewalServer()
    server.deployment.update(expires_at=1300, control_lease_until=1200)
    async with server.provider(concurrency=1) as provider:
        server.override = {"control_lease_until": 1300}
        await wait_until(lambda: provider.identity["control_lease_until"] == 1300)
        clock[0] = 1301
        with pytest.raises(SlotFencedError):
            await predict(provider)
        # Stop the context before the background monitor's next scheduled read.
    assert len(server.posts) == 0 and server.stops == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("id", "some-other-deployment"),
        ("release_id", "different-release"),
        ("release_sha256", "b" * 64),
        ("api_base", "https://different.test"),
        ("max_robots", 64),
        ("warm_robots", 64),
        ("model", "different-model"),
        ("precision", "different-precision"),
        ("prefill_batch", 1),
        ("action_batch", 2),
        ("expires_at", None),
        ("control_lease_until", 1),
    ],
)
async def test_changed_authority_fails_closed_and_stops_owned_deployment(
    field, value, fast_refresh
):
    server = RenewalServer()
    provider = server.provider(concurrency=1)
    await provider.__aenter__()
    server.override = {field: value}
    await wait_until(lambda: provider._status_task.done())
    assert provider.fenced_slots == {0}
    with pytest.raises(SlotFencedError):
        await predict(provider)
    with pytest.raises(ValueError):
        await provider.close()
    assert provider.cleanup_confirmed and server.stop_ids == ["deployment-test"]
    assert not server.posts


@pytest.mark.parametrize("status", ["draining", "stopped", "failed", "unrecognized"])
async def test_nonserving_status_fences_without_waiting_for_cached_expiry(status, fast_refresh):
    server = RenewalServer()
    provider = server.provider(concurrency=1)
    await provider.__aenter__()
    server.override = {"status": status}
    await wait_until(lambda: provider._status_task.done())
    with pytest.raises(SlotFencedError):
        await provider.close()
    assert server.stops == 1 and not server.posts


async def test_transient_read_loss_preserves_only_unexpired_verified_authority(clock, fast_refresh):
    server, events = RenewalServer(), []
    server.deployment.update(expires_at=1300, control_lease_until=1300)
    provider = server.provider(
        concurrency=1, emit=lambda name, **fields: events.append((name, fields))
    )
    await provider.__aenter__()
    server.fail_read = True
    await wait_until(lambda: any(n == "provider_authority_refresh_failed" for n, _ in events))
    assert np.all(await predict(provider) == 1)
    assert provider.identity["expires_at"] == 1300
    clock[0] = 1301
    await wait_until(lambda: provider._status_task.done())
    with pytest.raises(SlotFencedError):
        await provider.close()
    assert provider.cleanup_confirmed and server.stops == 1
    assert len(server.posts) == 1


async def test_auth_failure_is_not_treated_as_transient_renewal(fast_refresh):
    server = RenewalServer()
    provider = server.provider(concurrency=1)
    await provider.__aenter__()
    server.auth_failure = True
    await wait_until(lambda: provider._status_task.done())
    with pytest.raises(DreamscaleError):
        await provider.close()
    assert server.stops == 1 and not server.posts


async def test_close_cancels_inflight_status_read_before_closing_client(fast_refresh):
    server = RenewalServer()
    provider = server.provider(concurrency=1)
    await provider.__aenter__()
    server.block_read = True
    await asyncio.wait_for(server.read_started.wait(), 1)
    await provider.close()
    assert provider._status_task.done() and provider._status_task.cancelled()
    assert provider.cleanup_confirmed and server.stops == 1
    assert len(server.closed) == len(server.clients)


async def test_status_get_is_bounded_and_timeout_does_not_mint_authority(fast_refresh):
    server, events = RenewalServer(), []
    provider = server.provider(
        concurrency=1,
        request_timeout=0.01,
        emit=lambda name, **fields: events.append((name, fields)),
    )
    async with provider:
        original = provider.identity
        server.block_read = True
        await wait_until(lambda: any(n == "provider_authority_refresh_failed" for n, _ in events))
        assert provider.identity == original and provider._status_error is None
    assert provider.cleanup_confirmed
