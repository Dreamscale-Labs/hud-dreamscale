"""Job-owned pooled HTTP inference; each simulator retains independent request ownership."""

import asyncio
import functools
import json
import math
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

import httpx
import numpy as np
from dropbear.errors import DropbearError
from dropbear.inference import AsyncInferenceClient, action_request

IDENTITY_KEYS = ("deployment_id", "robot_slot", "request_id", "sequence", "episode_id")
TERMINAL_FAILURES = {"request_not_admitted", "expired", "preprocessing_failed"}
CREATION_REJECTIONS = {
    "deployment_active",
    "budget_exhausted",
    "not_granted",
    "invalid_api_key",
    "invalid_request",
    "creation_key_required",
}
MODEL = "molmoact2-libero"


class SlotFencedError(RuntimeError):
    """An uncertain or invalid result prevents further use of the affected slot."""


class RequestFailedError(RuntimeError):
    """A resolved request failed before inference; it produced no executable actions."""

    def __init__(self, code, identity):
        self.code, self.identity = code, identity
        super().__init__(f"Slot {identity['robot_slot']} request failed: {code}")


class PooledProvider:
    """Own one fresh deployment and one persistent SDK client per active robot slot.

    This object is single-use. It never attaches to somebody else's deployment or
    guesses a sequence after a restart. The optional private JSONL journal records
    identities before submission, allowing an operator to reconcile interrupted jobs.
    Journal timing fields measure existing checkpoint calls, including lock wait and
    fsync; they do not remove durability or optimize inference latency. Cancellation
    before submission has no POST event; its total time remains in the agent metric.
    """

    def __init__(
        self,
        *,
        concurrency=8,
        api_key=None,
        api_base=None,
        emit=None,
        journal_path=None,
        ready_timeout=1000,
        recovery_timeout=10,
        close_timeout=120,
        poll_interval=1,
        request_timeout=5,
        encoding_workers=4,
        expected_release_id=None,
        expected_release_sha256=None,
        client_factory=AsyncInferenceClient,
    ):
        if type(concurrency) is not int or not 1 <= concurrency <= 64:
            raise ValueError("concurrency must be an integer from 1 to 64")
        for name, value in {
            "ready_timeout": ready_timeout,
            "recovery_timeout": recovery_timeout,
            "close_timeout": close_timeout,
            "poll_interval": poll_interval,
            "request_timeout": request_timeout,
        }.items():
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if type(encoding_workers) is not int or not 1 <= encoding_workers <= 64:
            raise ValueError("encoding_workers must be an integer from 1 to 64")
        self.concurrency = concurrency
        self.capacity = 8 * math.ceil(concurrency / 8)
        self.emit = emit or (lambda event, **fields: None)
        self.api_key, self.api_base = api_key, api_base
        self.ready_timeout, self.recovery_timeout = ready_timeout, recovery_timeout
        self.close_timeout, self.poll_interval = close_timeout, poll_interval
        self.request_timeout = request_timeout
        self.expected_release_id = expected_release_id
        self.expected_release_sha256 = expected_release_sha256
        self._factory = client_factory
        self._journal_path = Path(journal_path) if journal_path is not None else None
        self._journal = None
        self._journal_lock = asyncio.Lock()
        self._encoder = ThreadPoolExecutor(max_workers=min(encoding_workers, concurrency))
        self._encoding_gate = asyncio.Semaphore(min(encoding_workers, concurrency))
        self._key = f"hud-{uuid4().hex}"
        self._control = None
        self._clients = []
        self.deployment = None
        self._create_attempted = False
        self._creation_calls = 0
        self._entered = self._ready = self._closing = False
        self._close_task = None
        self._active = set()
        self._busy = set()
        self._fenced = set()
        self._sequences = [0] * concurrency
        self._pending = {}
        self.cleanup_confirmed = False
        self.cleanup_error = None

    @property
    def deployment_id(self):
        return self.deployment["id"] if self.deployment else None

    @property
    def identity(self):
        return dict(self.deployment or {})

    @property
    def fenced_slots(self):
        return frozenset(self._fenced)

    def is_slot_available(self, slot):
        return (
            type(slot) is int
            and 0 <= slot < self.concurrency
            and self._ready
            and not self._closing
            and slot not in self._busy | self._fenced
        )

    def _event(self, event, **fields):
        self.emit(event, deployment_id=self.deployment_id, **fields)

    async def _checkpoint(self, phase, **fields):
        if self._journal is None:
            return
        row = {
            "phase": phase,
            "unix_s": time.time(),
            "deployment_id": self.deployment_id,
            **fields,
        }
        line = json.dumps(row, allow_nan=False) + "\n"

        def write():
            self._journal.write(line)
            self._journal.flush()
            os.fsync(self._journal.fileno())

        async with self._journal_lock:
            # A cancelled write must finish before another write or file close.
            await self._protected(write=asyncio.to_thread(write))

    @staticmethod
    async def _protected(*, write):
        task = asyncio.ensure_future(write)
        cancelled = False
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                cancelled = True
        result = task.result()
        if cancelled:
            raise asyncio.CancelledError
        return result

    async def _response_hook(self, response):
        self._event(
            "provider_http_response",
            request_id=response.request.extensions.get("dropbear_request_id"),
            http_status=response.status_code,
            server_timing=response.headers.get("server-timing"),
        )

    def _client(self, api_base):
        return self._factory(
            api_key=self.api_key,
            api_base=api_base,
            timeout=self.request_timeout,
            event_hooks={"response": [self._response_hook]},
        )

    async def _create(self):
        self._create_attempted = True
        while True:
            try:
                self._creation_calls += 1
                value = await self._control.create_deployment(
                    idempotency_key=self._key,
                    warm_robots=self.capacity,
                    max_robots=self.capacity,
                )
                if not isinstance(value, dict) or not isinstance(value.get("id"), str):
                    raise ValueError("Deployment creation returned no identity")
                self.deployment = value
                return
            except httpx.TransportError:
                # Retrying creation with this exact key cannot allocate another job.
                await asyncio.sleep(self.poll_interval)
            except DropbearError as exc:
                if self._creation_calls == 1 and exc.info.code in CREATION_REJECTIONS:
                    # Explicit first-attempt rejection never acquired ownership. Do
                    # not turn closing a rejected job into a fresh paid allocation.
                    self._create_attempted = False
                raise

    def _validate_deployment(self, value):
        expected = {
            "id": self.deployment_id,
            "model": MODEL,
            "precision": "bf16_swiglu_fp32",
            "prefill_batch": 2,
            "action_batch": 4,
            "max_robots": self.capacity,
            "warm_robots": self.capacity,
        }
        if any(value.get(k) != v for k, v in expected.items()):
            raise ValueError("Deployment identity, capacity or qualified backend changed")
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,96}", str(value.get("release_id", ""))):
            raise ValueError("Deployment has no valid release ID")
        if not re.fullmatch(r"[0-9a-f]{64}", str(value.get("release_sha256", ""))):
            raise ValueError("Deployment has no immutable release digest")
        for key, selected in (
            ("release_id", self.expected_release_id),
            ("release_sha256", self.expected_release_sha256),
        ):
            if selected is not None and value[key] != selected:
                raise ValueError(f"Deployment {key} does not match the selected release")
            if self.deployment is not None and value[key] != self.deployment.get(key):
                raise ValueError(f"Deployment {key} changed while waiting for readiness")
        if not isinstance(value.get("api_base"), str):
            raise ValueError("Deployment has no action API origin")
        for key in ("expires_at", "control_lease_until"):
            stamp = value.get(key)
            if type(stamp) not in (int, float) or not math.isfinite(stamp) or stamp <= time.time():
                raise ValueError(f"Deployment {key} is missing or expired")

    async def __aenter__(self):
        if self._entered or self._closing:
            raise RuntimeError("Use a fresh pooled provider for each evaluation job")
        self._entered = True
        started = time.monotonic()
        try:
            if self._journal_path is not None:
                self._journal_path.parent.mkdir(parents=True, exist_ok=True)
                fd = os.open(self._journal_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                self._journal = os.fdopen(fd, "w")
            self._control = self._client(self.api_base)
            await self._checkpoint("creation_intent", idempotency_key=self._key)
            self._event("provider_starting", concurrency=self.concurrency, capacity=self.capacity)
            async with asyncio.timeout(self.ready_timeout):
                create_started = time.monotonic()
                await self._create()
                self._event("provider_created", duration_s=time.monotonic() - create_started)
                await self._checkpoint("deployment_created", identity=self.identity)
                self._validate_deployment(self.deployment)
                while True:
                    query_started = time.monotonic()
                    value = await self._control.deployment(self.deployment_id)
                    self._validate_deployment(value)
                    self.deployment = value
                    self._event(
                        "provider_status",
                        duration_s=time.monotonic() - query_started,
                        status=value["status"],
                        ready_robots=value.get("ready_robots", 0),
                    )
                    if value["status"] == "ready" and value.get("ready_robots", 0) >= self.capacity:
                        break
                    if value["status"] in {"failed", "stopped", "draining"}:
                        raise RuntimeError(f"Deployment became {value['status']} before readiness")
                    await asyncio.sleep(self.poll_interval)
                for _ in range(self.concurrency):
                    self._clients.append(self._client(value["api_base"]))
                discovery_started = time.monotonic()
                connections = await asyncio.gather(
                    *(client.connect() for client in self._clients), return_exceptions=True
                )
                for result in connections:
                    if isinstance(result, BaseException):
                        raise result
                self._event(
                    "provider_connections_ready",
                    duration_s=time.monotonic() - discovery_started,
                    client_count=len(self._clients),
                )
            self._ready = True
            await self._checkpoint("ready", identity=self.identity)
            self._event(
                "provider_ready",
                duration_s=time.monotonic() - started,
                **{key: value[key] for key in ("release_id", "release_sha256", "ready_robots")},
            )
            return self
        except BaseException as exc:
            try:
                await self.close()
            except BaseException as cleanup:
                exc.add_note(f"Pooled deployment cleanup was not confirmed: {cleanup}")
            raise

    def _check_identity(self, result, identity):
        # An unseen GET status can only echo its four query fields. A completed
        # or failed result must still carry the full original episode identity.
        check = identity
        if (
            isinstance(result, dict)
            and result.get("status") == "pending"
            and "episode_id" not in result
        ):
            check = {k: v for k, v in identity.items() if k != "episode_id"}
        if not isinstance(result, dict) or any(result.get(k) != v for k, v in check.items()):
            raise ValueError("Inference response identity mismatch")
        if result.get("status") == "pending":
            if "actions" in result:
                raise ValueError("Pending inference unexpectedly contained actions")
            return "pending"
        if result.get("status") == "failed":
            code = result.get("error", {}).get("code")
            if code not in TERMINAL_FAILURES or "actions" in result:
                raise ValueError("Invalid terminal inference failure")
            return "failed"
        if (
            result.get("model") != MODEL
            or result.get("release_id") != self.deployment["release_id"]
        ):
            raise ValueError("Inference model/release differs from the owned deployment")
        actions = np.asarray(result.get("actions"))
        if (
            actions.shape != (10, 7)
            or actions.dtype.kind not in "fi"
            or not np.isfinite(actions).all()
        ):
            raise ValueError("Expected finite numeric 10x7 actions")
        if np.any(np.abs(actions) > np.finfo(np.float32).max):
            raise ValueError("Actions are outside the finite float32 simulator range")
        return "completed"

    async def _recover(self, client, identity):
        """GET first, then atomically resolve unknown work. Never replay the inference POST."""
        self._event(
            "inference_recovery", **{k: v for k, v in identity.items() if k != "deployment_id"}
        )
        async with asyncio.timeout(self.recovery_timeout):
            while True:
                try:
                    try:
                        result = await client.get_result(
                            **{k: v for k, v in identity.items() if k != "episode_id"}
                        )
                    except (httpx.TransportError, DropbearError):
                        result = None
                    if result is not None and self._check_identity(result, identity) != "pending":
                        return result
                    result = await client.resolve_request(**identity)
                    if self._check_identity(result, identity) != "pending":
                        return result
                except (httpx.TransportError, DropbearError):
                    # Even an uncertain resolution must retain this same identity.
                    pass
                await asyncio.sleep(self.poll_interval)

    async def predict(
        self, slot, observation, instruction, episode_id, noise_seed, *, trace_id=None
    ):
        if not self.is_slot_available(slot):
            raise SlotFencedError(f"Slot {slot} is unavailable, busy or fenced")
        if time.time() >= self.deployment["expires_at"]:
            self._fenced.update(range(self.concurrency))
            raise SlotFencedError("The owned deployment's finite grant expired")
        self._busy.add(slot)
        current = asyncio.current_task()
        self._active.add(current)
        identity = None
        submitted = False
        journal_before_post_s = journal_terminal_s = 0.0
        started = time.monotonic()

        async def terminal_checkpoint(phase):
            nonlocal journal_terminal_s
            checkpoint_started = time.monotonic()
            try:
                await self._checkpoint(phase, **identity)
            finally:
                journal_terminal_s += time.monotonic() - checkpoint_started

        try:
            for key in ("agentview_image", "robot0_eye_in_hand_image"):
                frame = np.asarray(observation[key])
                if frame.shape != (360, 360, 3) or frame.dtype != np.uint8:
                    raise ValueError("Pooled LIBERO requires native raw360 RGB uint8 cameras")
            # Own the arrays while PNG encoding runs outside the asyncio event loop.
            raw = {key: np.array(value, copy=True) for key, value in observation.items()}
            identity = {
                "deployment_id": self.deployment_id,
                "robot_slot": slot,
                "request_id": str(uuid4()),
                "sequence": self._sequences[slot],
                "episode_id": episode_id,
            }
            correlation = {k: v for k, v in identity.items() if k != "deployment_id"}
            correlation["trace_id"] = trace_id
            encoding_started = time.monotonic()
            async with self._encoding_gate:
                request = await asyncio.get_running_loop().run_in_executor(
                    self._encoder,
                    functools.partial(
                        action_request,
                        **identity,
                        noise_seed=noise_seed,
                        instruction=instruction,
                        observation=raw,
                    ),
                )
            encode_s = time.monotonic() - encoding_started
            self._event("inference_encode", duration_s=encode_s, **correlation)
            checkpoint_started = time.monotonic()
            try:
                await self._checkpoint("before_post", **identity, trace_id=trace_id)
            finally:
                journal_before_post_s = time.monotonic() - checkpoint_started
            self._sequences[slot] += 1
            self._pending[slot] = identity
            submitted = True
            post_started = time.monotonic()
            result = None
            try:
                result = await self._clients[slot].predict(request)
            except (httpx.TransportError, DropbearError):
                pass
            finally:
                self._event(
                    "inference_post",
                    duration_s=time.monotonic() - post_started,
                    response_received=result is not None,
                    journal_before_post_s=journal_before_post_s,
                    **correlation,
                )
            if result is None or self._check_identity(result, identity) == "pending":
                result = await self._recover(self._clients[slot], identity)
            status = self._check_identity(result, identity)
            await terminal_checkpoint(status)
            del self._pending[slot]
            if status == "failed":
                self._event(
                    "inference_failed",
                    code=result["error"]["code"],
                    journal_before_post_s=journal_before_post_s,
                    journal_terminal_s=journal_terminal_s,
                    **correlation,
                )
                raise RequestFailedError(result["error"]["code"], identity)
            self._event(
                "provider_inference",
                duration_s=time.monotonic() - started,
                encode_s=encode_s,
                journal_before_post_s=journal_before_post_s,
                journal_terminal_s=journal_terminal_s,
                timing=result.get("timings_ms", {}),
                **correlation,
            )
            return np.asarray(result["actions"], dtype=np.float32)
        except asyncio.CancelledError:
            if submitted and slot in self._pending:
                try:
                    result = await self._protected(
                        write=self._recover(self._clients[slot], identity)
                    )
                    await terminal_checkpoint("cancelled_resolved")
                    self._pending.pop(slot, None)
                    self._event(
                        "inference_cancelled",
                        result_status=self._check_identity(result, identity),
                        journal_before_post_s=journal_before_post_s,
                        journal_terminal_s=journal_terminal_s,
                        **{k: v for k, v in identity.items() if k != "deployment_id"},
                    )
                except BaseException:
                    self._fenced.add(slot)
            raise
        except BaseException:
            if submitted and slot in self._pending:
                self._fenced.add(slot)
                await terminal_checkpoint("fenced")
                self._event(
                    "slot_fenced",
                    robot_slot=slot,
                    request_id=identity["request_id"],
                    journal_before_post_s=journal_before_post_s,
                    journal_terminal_s=journal_terminal_s,
                )
            raise
        finally:
            self._busy.discard(slot)
            self._active.discard(current)

    async def _cleanup(self):
        self._closing = True
        self._ready = False
        try:
            async with asyncio.timeout(self.close_timeout):
                active = list(self._active)
                for task in active:
                    task.cancel()
                if active:
                    await asyncio.gather(*active, return_exceptions=True)
                if self._control is not None and self._create_attempted:
                    if self.deployment is None:
                        # Creation may have succeeded before the response was lost.
                        await self._create()
                    await self._checkpoint("stop_intent")
                    self._event("provider_stopping")
                    while True:
                        try:
                            await self._control.stop(self.deployment_id)
                            value = await self._control.deployment(self.deployment_id)
                            if value.get("id") != self.deployment_id:
                                raise ValueError("Cleanup returned another deployment")
                            if value.get("status") == "stopped":
                                self.deployment = value
                                self.cleanup_confirmed = True
                                self._event("provider_stopped")
                                await self._checkpoint("stopped")
                                break
                        except httpx.TransportError:
                            pass
                        except DropbearError as exc:
                            # The reconciler can win the stop/update CAS. Repeating
                            # this owned, idempotent stop is safe within the same
                            # deadline; auth/ownership errors are never retried.
                            if exc.info.code != "state_changed":
                                raise
                        await asyncio.sleep(self.poll_interval)
                else:
                    self.cleanup_confirmed = True
        except BaseException as exc:
            self.cleanup_error = type(exc).__name__
            self._fenced.update(range(self.concurrency))
            self._event(
                "provider_cleanup_uncertain",
                error_type=self.cleanup_error,
                pending_requests=list(self._pending.values()),
            )
            await self._checkpoint(
                "cleanup_uncertain",
                error_type=self.cleanup_error,
                pending_requests=list(self._pending.values()),
            )
            raise
        finally:
            try:
                async with asyncio.timeout(min(10, self.close_timeout)):
                    await asyncio.gather(
                        *(c.close() for c in [*self._clients, self._control] if c is not None),
                        return_exceptions=True,
                    )
            finally:
                self._encoder.shutdown(wait=False, cancel_futures=True)
                if self._journal is not None:
                    self._journal.close()

    async def close(self):
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._cleanup())
        await self._protected(write=self._close_task)

    async def __aexit__(self, exc_type, exc, tb):
        try:
            await self.close()
        except BaseException as cleanup:
            if exc is None:
                raise
            exc.add_note(f"Pooled deployment cleanup was not confirmed: {cleanup}")
