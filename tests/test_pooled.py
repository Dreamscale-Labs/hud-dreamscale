"""Exercise the merged SDK's real JSON/PNG codec and client over an in-memory HTTP boundary."""

import asyncio
import base64
import io
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from dropbear.inference import AsyncInferenceClient
from PIL import Image

from hud_dropbear.pooled import PooledProvider, RequestFailedError, SlotFencedError


def observation():
    return {
        "agentview_image": np.full((360, 360, 3), [1, 5, 9], dtype=np.uint8),
        "robot0_eye_in_hand_image": np.full((360, 360, 3), [20, 30, 40], dtype=np.uint8),
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.array([0, 0, 0, 1]),
        "robot0_gripper_qpos": np.array([0.02, -0.02]),
    }


class Server:
    def __init__(self):
        self.deployment = {
            "id": "deployment-test",
            "model": "molmoact2-libero",
            "status": "ready",
            "api_base": "https://inference.test",
            "release_id": "qualified-release",
            "release_sha256": "a" * 64,
            "precision": "bf16_swiglu_fp32",
            "prefill_batch": 2,
            "action_batch": 4,
            "ready_robots": 64,
            "warm_robots": 8,
            "max_robots": 8,
            "expires_at": 4102444800,
            "control_lease_until": 4102444800,
        }
        self.creates = []
        self.posts = []
        self.results = {}
        self.resolutions = []
        self.clients = []
        self.closed = []
        self.stops = 0
        self.stop_ids = []
        self.mode = "success"
        self.creation_lost = False
        self.block_creation = False
        self.creation_rejected = False
        self.allow_stop = True
        self.stop_errors = []
        self.status_errors = []
        self.permanent_stop_error = None
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.journal = None

    def client(self, **kwargs):
        server = self

        class Client(AsyncInferenceClient):
            async def close(self):
                server.closed.append(self)
                await super().close()

        client = Client(transport=httpx.MockTransport(self.handle), **kwargs)
        self.clients.append(client)
        return client

    async def handle(self, request):
        path = request.url.path
        data = json.loads(request.content) if request.content else None
        if path == "/v1/inference/deployments":
            self.creates.append(request.headers["idempotency-key"])
            if self.creation_rejected:
                return httpx.Response(409, json={"error": {"code": "deployment_active"}})
            self.deployment.update({k: data[k] for k in ("warm_robots", "max_robots")})
            if self.creation_lost and len(self.creates) == 1:
                raise httpx.ReadTimeout("creation response lost")
            if self.block_creation and len(self.creates) == 1:
                await self.release.wait()
            return httpx.Response(200, json=self.deployment)
        if path.startswith("/v1/inference/deployments/"):
            if request.method == "DELETE":
                self.stops += 1
                self.stop_ids.append(path.rsplit("/", 1)[-1])
                code = self.stop_errors.pop(0) if self.stop_errors else self.permanent_stop_error
                if code:
                    return httpx.Response(409, json={"error": {"code": code}})
                self.deployment["status"] = "stopped" if self.allow_stop else "draining"
            elif self.stops and self.status_errors:
                return httpx.Response(409, json={"error": {"code": self.status_errors.pop(0)}})
            return httpx.Response(200, json=self.deployment)
        if path == "/v1/connection":
            return httpx.Response(
                200,
                json={
                    "api_base": "https://qualified-gateway.modal.host",
                    "expires_at": 4102444800,
                    "instance_id": "instance-test",
                },
            )
        if path == "/v1/actions":
            self.posts.append(data)
            if self.journal is not None:
                rows = [json.loads(line) for line in self.journal.read_text().splitlines()]
                assert any(
                    r["phase"] == "before_post" and r["request_id"] == data["request_id"]
                    for r in rows
                )
            self.entered.set()
            if self.mode == "block":
                await self.release.wait()
            if self.mode in {"unadmitted", "pending_forever"}:
                raise httpx.ReadTimeout("original outcome unknown")
            result = self.result(data)
            self.results[data["request_id"]] = result
            if self.mode == "response_lost":
                raise httpx.ReadTimeout("completed response lost")
            if self.mode == "wrong_release":
                result["release_id"] = "another-release"
            if self.mode == "wrong_episode":
                result["episode_id"] = "someone-else"
            if self.mode == "malformed":
                result["actions"] = [[0.0] * 6] * 10
            return httpx.Response(200, json=result, headers={"server-timing": "gpu;dur=150"})
        if path.endswith("/resolve"):
            self.resolutions.append(data)
            if self.mode == "pending_forever":
                return httpx.Response(202, json={**data, "status": "pending"})
            return httpx.Response(
                422,
                json={
                    **data,
                    "status": "failed",
                    "error": {"code": "request_not_admitted"},
                },
            )
        if request.method == "GET" and path.startswith("/v1/actions/"):
            request_id = path.rsplit("/", 1)[-1]
            if request_id in self.results:
                return httpx.Response(200, json=self.results[request_id])
            query = request.url.params
            return httpx.Response(
                202,
                json={
                    "deployment_id": query["deployment_id"],
                    "request_id": request_id,
                    "robot_slot": int(query["robot_slot"]),
                    "sequence": int(query["sequence"]),
                    "status": "pending",
                },
            )
        raise AssertionError((request.method, path))

    def result(self, data):
        return {
            **{
                key: data[key]
                for key in ("deployment_id", "robot_slot", "request_id", "sequence", "episode_id")
            },
            "model": "molmoact2-libero",
            "release_id": "qualified-release",
            "actions": np.full((10, 7), data["robot_slot"] + data["sequence"] + 1).tolist(),
            "timings_ms": {"gpu": 150.0},
        }

    def provider(self, **kwargs):
        return PooledProvider(
            api_key="db_test",
            api_base="https://control.test",
            client_factory=self.client,
            poll_interval=0.001,
            recovery_timeout=0.03,
            close_timeout=0.15,
            **kwargs,
        )


async def predict(provider, slot=0, episode="first"):
    return await provider.predict(
        slot, observation(), "pick up the bowl", episode, 197, trace_id=f"trace-{episode}"
    )


@pytest.mark.parametrize(
    "concurrency,capacity",
    [
        (width, capacity)
        for capacity in range(8, 65, 8)
        for width in range(capacity - 7, capacity + 1)
    ],
)
async def test_capacity_rounds_to_full_replicas_and_cleanup_confirms_stop(concurrency, capacity):
    """All supported widths over the mock HTTP boundary, not live GPU capacity."""
    server = Server()
    async with server.provider(concurrency=concurrency) as provider:
        assert provider.capacity == capacity
        assert provider.identity["max_robots"] == capacity
        assert provider.identity["warm_robots"] == capacity
        assert len(server.creates) == 1
        assert len(server.clients) == concurrency + 1
        assert all(provider.is_slot_available(slot) for slot in range(concurrency))
        assert not provider.is_slot_available(concurrency)
        # The highest requested slot must work even at a partial replica width.
        # Rounded capacity must not expose additional robot slots to the caller.
        actions = await predict(provider, concurrency - 1)
        assert actions.shape == (10, 7) and np.all(actions == concurrency)
        assert server.posts[0]["robot_slot"] == concurrency - 1
        assert server.posts[0]["sequence"] == 0
        with pytest.raises(SlotFencedError):
            await predict(provider, concurrency)
        assert len(server.posts) == 1
    assert server.stops == 1 and server.deployment["status"] == "stopped"
    assert provider.cleanup_confirmed and provider.cleanup_error is None
    assert len(server.closed) == len(server.clients)
    await provider.close()
    assert server.stops == 1


async def test_real_sdk_png_state_journal_and_sequences_across_episodes(tmp_path):
    server, events = Server(), []
    server.journal = tmp_path / "requests.jsonl"
    async with server.provider(
        journal_path=server.journal, emit=lambda name, **fields: events.append((name, fields))
    ) as provider:
        first = await predict(provider, episode="first")
        second = await predict(provider, episode="second")
        assert first.shape == (10, 7) and np.all(first == 1) and np.all(second == 2)
        assert [p["sequence"] for p in server.posts] == [0, 1]
        assert [p["episode_id"] for p in server.posts] == ["first", "second"]
        wire = server.posts[0]["observation"]
        assert wire["robot0_eef_quat_xyzw"] == [0, 0, 0, 1]
        for camera in ("agentview_image", "robot0_eye_in_hand_image"):
            decoded = np.array(Image.open(io.BytesIO(base64.b64decode(wire[camera]["data"]))))
            assert np.array_equal(decoded, observation()[camera])
    assert server.journal.stat().st_mode & 0o777 == 0o600
    journal = server.journal.read_text()
    assert "db_test" not in journal and "agentview_image" not in journal
    assert json.loads(journal.splitlines()[-1])["phase"] == "stopped"
    inference = [fields for name, fields in events if name == "provider_inference"]
    assert [row["trace_id"] for row in inference] == ["trace-first", "trace-second"]
    assert all(row["duration_s"] >= row["encode_s"] for row in inference)
    assert any(
        name == "provider_http_response" and fields["server_timing"] == "gpu;dur=150"
        for name, fields in events
    )


@pytest.mark.parametrize(
    "mode,terminal_event",
    [
        ("success", "provider_inference"),
        ("unadmitted", "inference_failed"),
        ("block", "inference_cancelled"),
        ("wrong_release", "slot_fenced"),
    ],
)
async def test_journal_timing_includes_lock_wait_without_changing_writes(
    monkeypatch, tmp_path, mode, terminal_event
):
    from hud_dropbear import pooled

    server, events, encoded = Server(), [], asyncio.Event()
    server.mode = mode
    server.journal = tmp_path / "requests.jsonl"
    clock = [100.0]
    # A controlled provider clock tests attribution, never filesystem speed.
    monkeypatch.setattr(
        pooled,
        "time",
        SimpleNamespace(
            time=pooled.time.time,
            monotonic=lambda: clock[0],
        ),
    )

    def emit(name, **fields):
        events.append((name, fields))
        if name == "inference_encode":
            encoded.set()

    async with server.provider(concurrency=1, journal_path=server.journal, emit=emit) as provider:
        checkpoint = provider._checkpoint

        async def delayed_checkpoint(phase, **fields):
            if phase in {"completed", "failed", "cancelled_resolved", "fenced"}:
                await asyncio.sleep(0)
                clock[0] += 3.0
            await checkpoint(phase, **fields)

        monkeypatch.setattr(provider, "_checkpoint", delayed_checkpoint)
        async with provider._journal_lock:
            running = asyncio.create_task(predict(provider))
            await encoded.wait()
            # The before_post checkpoint is now blocked on the real shared lock.
            assert not server.posts
            clock[0] += 2.0
        if mode == "block":
            await server.entered.wait()
            running.cancel()
        if mode == "success":
            await running
        else:
            error = (
                asyncio.CancelledError
                if mode == "block"
                else (RequestFailedError if mode == "unadmitted" else ValueError)
            )
            with pytest.raises(error):
                await running
        post = next(fields for name, fields in events if name == "inference_post")
        terminal = next(fields for name, fields in events if name == terminal_event)
        assert post["journal_before_post_s"] == terminal["journal_before_post_s"] == 2.0
        assert post["duration_s"] == 0.0  # Journal waiting is outside the HTTP interval.
        assert terminal["journal_terminal_s"] == 3.0
        if mode == "success":
            assert terminal["duration_s"] == 5.0
        assert len(server.posts) == 1
    phases = [json.loads(line)["phase"] for line in server.journal.read_text().splitlines()]
    assert phases.count("before_post") == 1
    terminal_phase = {
        "success": "completed",
        "unadmitted": "failed",
        "block": "cancelled_resolved",
        "wrong_release": "fenced",
    }[mode]
    assert phases.count(terminal_phase) == 1


async def test_independent_slots_progress_without_a_batch_barrier():
    server = Server()
    server.mode = "block"
    async with server.provider(concurrency=8) as provider:
        tasks = [asyncio.create_task(predict(provider, slot)) for slot in range(8)]
        async with asyncio.timeout(3):
            while len(server.posts) != 8:
                await asyncio.sleep(0.001)
        with pytest.raises(SlotFencedError, match="busy"):
            await predict(provider, 0)
        server.release.set()
        rows = await asyncio.gather(*tasks)
        assert all(np.all(row == slot + 1) for slot, row in enumerate(rows))
        assert len({r["request_id"] for r in server.posts}) == 8


async def test_sixty_four_mock_wire_clients_reuse_slots_across_two_episodes_and_close():
    """Protocol concurrency only: this does not qualify live GPU or simulator capacity."""
    server = Server()
    server.mode = "block"
    async with server.provider(concurrency=64) as provider:
        clients = tuple(server.clients)
        for round_index, episode in enumerate(("first", "second")):
            server.release = asyncio.Event()
            tasks = [asyncio.create_task(predict(provider, slot, episode)) for slot in range(64)]
            async with asyncio.timeout(10):
                while len(server.posts) < 64 * (round_index + 1):
                    await asyncio.sleep(0.001)
            assert all(not task.done() for task in tasks)
            assert not provider.is_slot_available(63)
            server.release.set()
            chunks = await asyncio.gather(*tasks)
            assert all(np.all(chunk == slot + round_index + 1) for slot, chunk in enumerate(chunks))
            assert all(provider.is_slot_available(slot) for slot in range(64))
            assert tuple(server.clients) == clients
        assert provider.capacity == 64
        assert len(server.creates) == 1 and len(clients) == 65
        assert len({request["request_id"] for request in server.posts}) == 128
        for slot in range(64):
            requests = [row for row in server.posts if row["robot_slot"] == slot]
            assert [row["sequence"] for row in requests] == [0, 1]
            assert [row["episode_id"] for row in requests] == ["first", "second"]
    assert server.stops == 1 and provider.cleanup_confirmed
    assert len(server.closed) == len(clients)


async def test_unknown_completed_post_recovers_without_replay():
    server = Server()
    server.mode = "response_lost"
    async with server.provider() as provider:
        assert np.all(await predict(provider) == 1)
        assert provider.is_slot_available(0)
    assert len(server.posts) == 1 and not server.resolutions


async def test_unadmitted_request_resolves_fails_and_next_episode_uses_higher_sequence():
    server = Server()
    server.mode = "unadmitted"
    async with server.provider() as provider:
        with pytest.raises(RequestFailedError, match="request_not_admitted"):
            await predict(provider)
        assert provider.is_slot_available(0)
        server.mode = "success"
        assert np.all(await predict(provider, episode="second") == 2)
    assert [p["sequence"] for p in server.posts] == [0, 1]
    assert len(server.resolutions) == 1
    assert server.resolutions[0]["request_id"] == server.posts[0]["request_id"]


async def test_pending_claim_remains_fenced_after_bounded_recovery():
    server = Server()
    server.mode = "pending_forever"
    async with server.provider() as provider:
        with pytest.raises(TimeoutError):
            await predict(provider)
        assert provider.fenced_slots == {0}
        with pytest.raises(SlotFencedError):
            await predict(provider, episode="second")
    assert len(server.posts) == 1 and server.stops == 1


@pytest.mark.parametrize("mode", ["wrong_release", "wrong_episode", "malformed"])
async def test_invalid_result_never_returns_actions_and_fences_slot(mode):
    server = Server()
    server.mode = mode
    async with server.provider() as provider:
        with pytest.raises(ValueError):
            await predict(provider)
        assert provider.fenced_slots == {0}
    assert server.stops == 1


async def test_cancelled_post_resolves_original_identity_and_never_returns_actions():
    server = Server()
    server.mode = "block"
    async with server.provider() as provider:
        running = asyncio.create_task(predict(provider))
        await asyncio.wait_for(server.entered.wait(), 3)
        running.cancel()
        with pytest.raises(asyncio.CancelledError):
            await running
        assert len(server.posts) == len(server.resolutions) == 1
        assert provider.is_slot_available(0)
    assert server.stops == 1


async def test_context_cancellation_confirms_deployment_stop():
    server = Server()
    entered = asyncio.Event()

    async def run():
        async with server.provider():
            entered.set()
            await asyncio.Event().wait()

    running = asyncio.create_task(run())
    await asyncio.wait_for(entered.wait(), 3)
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    assert server.stops == 1 and len(server.closed) == len(server.clients)


async def test_creation_retry_uses_same_idempotency_key():
    server = Server()
    server.creation_lost = True
    async with server.provider():
        pass
    assert len(server.creates) == 2 and len(set(server.creates)) == 1
    assert server.stops == 1


async def test_another_owners_active_deployment_is_never_adopted_or_stopped():
    from dropbear.errors import DropbearError

    server = Server()
    server.creation_rejected = True
    provider = server.provider()
    with pytest.raises(DropbearError):
        async with provider:
            pass
    assert len(server.creates) == 1 and server.stops == 0
    assert provider.deployment is None and provider.cleanup_confirmed


async def test_wrong_selected_release_cleans_up_without_any_prediction():
    server = Server()
    with pytest.raises(ValueError, match="selected release"):
        async with server.provider(expected_release_sha256="b" * 64):
            pass
    assert server.stops == 1 and not server.posts


async def test_readiness_timeout_stops_owned_deployment():
    server = Server()
    server.deployment["status"] = "starting"
    with pytest.raises(TimeoutError):
        async with server.provider(ready_timeout=0.01):
            pass
    assert server.stops == 1 and not server.posts


async def test_cleanup_failure_is_visible_and_connections_close():
    server = Server()
    events = []
    server.allow_stop = False
    with pytest.raises(TimeoutError):
        async with server.provider(
            emit=lambda name, **fields: events.append((name, fields))
        ) as provider:
            pass
    assert len(server.closed) == len(server.clients)
    assert not provider.cleanup_confirmed and provider.cleanup_error == "TimeoutError"
    assert any(name == "provider_cleanup_uncertain" for name, _ in events)


@pytest.mark.parametrize("phase", ["stop_intent", "stopped"])
async def test_cleanup_journal_failure_does_not_prevent_owned_stop(phase, monkeypatch):
    server = Server()
    provider = server.provider(concurrency=1)
    await provider.__aenter__()
    checkpoint = provider._checkpoint

    async def failing_checkpoint(current_phase, **fields):
        if current_phase == phase:
            raise OSError("disk full while recording cleanup")
        await checkpoint(current_phase, **fields)

    monkeypatch.setattr(provider, "_checkpoint", failing_checkpoint)
    with pytest.raises(ExceptionGroup) as error:
        await provider.close()
    assert any(isinstance(exc, OSError) for exc in error.value.exceptions)
    assert provider.cleanup_confirmed and provider.cleanup_error == "ExceptionGroup"
    assert server.stop_ids == ["deployment-test"] and server.deployment["status"] == "stopped"
    assert len(server.closed) == len(server.clients)


@pytest.mark.parametrize(
    "failed_event", ["provider_stopping", "provider_stopped", "provider_http_response"]
)
async def test_cleanup_event_failure_does_not_prevent_owned_stop_or_confirmation(failed_event):
    server = Server()
    provider = server.provider(concurrency=1)
    await provider.__aenter__()

    def failing_emit(event, **fields):
        if event == failed_event:
            raise OSError("timing sidecar is unavailable")

    provider.emit = failing_emit
    with pytest.raises(ExceptionGroup) as error:
        await provider.close()
    assert all(isinstance(exc, OSError) for exc in error.value.exceptions)
    assert provider.cleanup_confirmed and provider.cleanup_error == "ExceptionGroup"
    assert server.stop_ids == ["deployment-test"] and server.deployment["status"] == "stopped"
    assert len(server.closed) == len(server.clients)


async def test_cleanup_diagnostic_failures_preserve_original_unconfirmed_stop(monkeypatch):
    server = Server()
    provider = server.provider(concurrency=1)
    await provider.__aenter__()
    server.allow_stop = False

    async def failing_checkpoint(*args, **kwargs):
        raise OSError("journal disk full")

    def failing_emit(*args, **kwargs):
        raise RuntimeError("timing writer failed")

    monkeypatch.setattr(provider, "_checkpoint", failing_checkpoint)
    provider.emit = failing_emit
    with pytest.raises(TimeoutError) as error:
        await provider.close()
    assert not provider.cleanup_confirmed and provider.cleanup_error == "TimeoutError"
    assert server.stops > 0 and set(server.stop_ids) == {"deployment-test"}
    assert any("cleanup evidence" in note.lower() for note in error.value.__notes__)
    assert len(server.closed) == len(server.clients)


async def test_cleanup_journal_close_failure_is_reported_after_owned_stop(tmp_path):
    server = Server()
    provider = server.provider(concurrency=1, journal_path=tmp_path / "requests.jsonl")
    await provider.__aenter__()

    class FailingClose:
        def __init__(self, stream):
            self.stream = stream

        def __getattr__(self, name):
            return getattr(self.stream, name)

        def close(self):
            self.stream.close()
            raise OSError("final journal flush failed")

    provider._journal = FailingClose(provider._journal)
    with pytest.raises(ExceptionGroup):
        await provider.close()
    assert provider.cleanup_confirmed and provider.cleanup_error == "ExceptionGroup"
    assert server.stop_ids == ["deployment-test"] and provider._journal.closed
    assert len(server.closed) == len(server.clients)


@pytest.mark.parametrize("operation", ["stop", "status"])
async def test_transient_cleanup_state_conflict_retries_owned_deployment(operation):
    server = Server()
    getattr(server, f"{operation}_errors").append("state_changed")
    async with server.provider() as provider:
        deployment_id = provider.deployment_id
    assert provider.deployment_id == deployment_id and provider.cleanup_confirmed
    assert server.stops == 2 and server.deployment["status"] == "stopped"
    assert len(server.creates) == 1 and len(server.closed) == len(server.clients)


async def test_permanent_cleanup_state_conflict_is_bounded_and_remains_uncertain():
    server = Server()
    server.permanent_stop_error = "state_changed"
    events = []
    async with asyncio.timeout(2):
        with pytest.raises(TimeoutError):
            async with server.provider(
                emit=lambda name, **fields: events.append((name, fields))
            ) as provider:
                pass
    assert server.stops > 1 and not provider.cleanup_confirmed
    assert provider.cleanup_error == "TimeoutError"
    assert any(name == "provider_cleanup_uncertain" for name, _ in events)
    assert len(server.closed) == len(server.clients)


@pytest.mark.parametrize("code", ["invalid_api_key", "not_granted"])
async def test_cleanup_never_retries_auth_or_ownership_failure(code):
    from dropbear.errors import DropbearError

    server = Server()
    server.permanent_stop_error = code
    with pytest.raises(DropbearError):
        async with server.provider() as provider:
            pass
    assert server.stops == 1 and not provider.cleanup_confirmed
    assert provider.cleanup_error == "DropbearError"
    assert len(server.closed) == len(server.clients)


async def test_expired_grant_prevents_new_actions_and_cleanup_is_still_attempted():
    server = Server()
    async with server.provider() as provider:
        provider.deployment["expires_at"] = 1
        with pytest.raises(SlotFencedError, match="expired"):
            await predict(provider)
        assert provider.fenced_slots == set(range(8)) and not server.posts
    assert provider.cleanup_confirmed


async def test_cancel_during_startup_recovers_same_creation_and_stops():
    server = Server()
    server.block_creation = True
    provider = server.provider()
    starting = asyncio.create_task(provider.__aenter__())
    async with asyncio.timeout(3):
        while not server.creates:
            await asyncio.sleep(0.001)
    starting.cancel()
    with pytest.raises(asyncio.CancelledError):
        await starting
    assert provider.cleanup_confirmed and server.stops == 1
    assert len(server.creates) == 2 and len(set(server.creates)) == 1


async def test_legacy_camera_size_is_rejected_before_post():
    server = Server()
    async with server.provider() as provider:
        obs = observation()
        obs["agentview_image"] = np.zeros((256, 256, 3), np.uint8)
        with pytest.raises(ValueError, match="raw360"):
            await provider.predict(0, obs, "task", "episode", 0)
        assert provider.is_slot_available(0) and not server.posts


@pytest.mark.parametrize("concurrency", [0, 65, True, 8.5])
def test_invalid_concurrency_is_rejected(concurrency):
    with pytest.raises(ValueError, match="concurrency"):
        PooledProvider(concurrency=concurrency)
