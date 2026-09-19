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
STATUS_REFRESH_INTERVAL_S = 60.0
STATUS_REFRESH_MARGIN_S = 30.0
STATUS_REFRESH_RETRY_S = 5.0


def _http_operation(request):
    """Bounded route classification; never retain signed origins, paths or query strings."""
    method, path = request.method, request.url.path
    if method == "POST" and path == "/v1/actions":
        return "actions"
    if method == "GET" and re.fullmatch(r"/v1/actions/[^/]+", path):
        return "result_lookup"
    if method == "POST" and re.fullmatch(r"/v1/actions/[^/]+/resolve", path):
        return "resolve"
    return "management"


def _error_fields(exc):
    """Retain diagnostic categories only, excluding exception text and response content."""
    name = type(exc).__name__
    code = exc.info.code if isinstance(exc, DropbearError) else None
    return {
        "error_type": name if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name) else "Exception",
        "error_code": (
            code
            if isinstance(code, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,63}", code)
            else "unrecognized"
            if code is not None
            else None
        ),
    }


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
    Optional resubmissions require atomic, identity-checked non-admission and use a
    fresh request identity for the unchanged input. Uncertain work is never replayed.
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
        max_not_admitted_resubmissions=0,
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
        if (
            type(max_not_admitted_resubmissions) is not int
            or not 0 <= max_not_admitted_resubmissions <= 2
        ):
            raise ValueError("max_not_admitted_resubmissions must be an integer from 0 to 2")
        self.concurrency = concurrency
        self.capacity = 8 * math.ceil(concurrency / 8)
        self.emit = emit or (lambda event, **fields: None)
        self.api_key, self.api_base = api_key, api_base
        self.ready_timeout, self.recovery_timeout = ready_timeout, recovery_timeout
        self.close_timeout, self.poll_interval = close_timeout, poll_interval
        self.request_timeout = request_timeout
        self.max_not_admitted_resubmissions = max_not_admitted_resubmissions
        self.expected_release_id = expected_release_id
        self.expected_release_sha256 = expected_release_sha256
        self._factory = client_factory
        self._journal_path = Path(journal_path) if journal_path is not None else None
        self._journal = None
        # Durable journal writes get their own single worker: the loop's shared
        # default executor also carries recorder finalization and DNS lookups,
        # and at wide cohorts a burst of episode completions queued journal
        # writes (which precede every POST) behind it for over a minute.
        self._journal_executor = None
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
        self._status_task = None
        self._status_error = None
        self._active = set()
        self._busy = set()
        self._fenced = set()
        self._sequences = [0] * concurrency
        self._pending = {}
        self.cleanup_confirmed = False
        self.cleanup_error = None
        self._cleanup_evidence_errors = []

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

    def _cleanup_event(self, event, **fields):
        try:
            self._event(event, **fields)
        except Exception as exc:
            self._cleanup_evidence_errors.append((f"event:{event}", exc))

    async def _cleanup_checkpoint(self, phase, **fields):
        try:
            await self._checkpoint(phase, **fields)
        except Exception as exc:
            self._cleanup_evidence_errors.append((f"journal:{phase}", exc))

    async def _checkpoint(self, phase, **fields):
        if self._journal is None:
            return
        started = time.monotonic()
        row = {
            "phase": phase,
            "unix_s": time.time(),
            "deployment_id": self.deployment_id,
            **fields,
        }
        line = json.dumps(row, allow_nan=False) + "\n"

        stamps = {}
        failure = None

        def write():
            stamps["write_started"] = time.monotonic()
            try:
                self._journal.write(line)
                self._journal.flush()
                os.fsync(self._journal.fileno())
                stamps["durable"] = True
            finally:
                stamps["write_finished"] = time.monotonic()

        async def dispatch_write():
            stamps["executor_submitted"] = time.monotonic()
            executor = self._journal_executor
            if executor is None:
                await asyncio.to_thread(write)
            else:
                await asyncio.get_running_loop().run_in_executor(executor, write)

        lock_started = time.monotonic()
        try:
            async with self._journal_lock:
                stamps["lock_acquired"] = time.monotonic()
                stamps["submitted"] = time.monotonic()
                # A cancelled write must finish before another write or file close.
                await self._protected(write=dispatch_write())
        except BaseException as exc:
            failure = exc
            raise
        finally:
            finished = time.monotonic()
            emit = self._cleanup_event if self._closing else self._event
            try:
                emit(
                    "journal_checkpoint",
                    phase=phase,
                    duration_s=finished - started,
                    lock_wait_s=stamps.get("lock_acquired", finished) - lock_started,
                    dispatch_delay_s=(
                        stamps["executor_submitted"] - stamps["submitted"]
                        if "executor_submitted" in stamps
                        else None
                    ),
                    executor_queue_s=(
                        stamps["write_started"] - stamps["executor_submitted"]
                        if "write_started" in stamps
                        else None
                    ),
                    write_flush_fsync_s=(
                        stamps["write_finished"] - stamps["write_started"]
                        if "write_finished" in stamps
                        else None
                    ),
                    resume_delay_s=(
                        finished - stamps["write_finished"] if "write_finished" in stamps else None
                    ),
                    durable=stamps.get("durable", False),
                    **{k: v for k, v in fields.items() if k != "deployment_id"},
                )
            except BaseException as exc:
                if failure is None:
                    raise
                failure.add_note(f"Journal timing also failed: {type(exc).__name__}")

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
        # A timing writer failure during DELETE/GET must not prevent the SDK
        # from processing the response and confirming owned resource teardown.
        emit = self._cleanup_event if self._closing else self._event
        emit(
            "provider_http_response",
            request_id=response.request.extensions.get("dropbear_request_id"),
            operation=_http_operation(response.request),
            http_method=(
                response.request.method
                if response.request.method in {"GET", "POST", "PATCH", "DELETE"}
                else "other"
            ),
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
        if self.deployment is not None and value["api_base"] != self.deployment.get("api_base"):
            raise ValueError("Deployment action API origin changed")
        for key in ("expires_at", "control_lease_until"):
            stamp = value.get(key)
            if type(stamp) not in (int, float) or not math.isfinite(stamp) or stamp <= time.time():
                raise ValueError(f"Deployment {key} is missing or expired")

    def _authority_remaining(self):
        return (
            min(self.deployment["expires_at"], self.deployment["control_lease_until"]) - time.time()
        )

    async def _refresh_deployment_status(self):
        """Read renewed authority; never extend funding or infer an unobserved renewal."""
        remaining = self._authority_remaining()
        if remaining <= 0:
            raise SlotFencedError("The last verified deployment authority expired")
        started = time.monotonic()
        async with asyncio.timeout(min(self.request_timeout, remaining)):
            value = await self._control.deployment(self.deployment_id)
        self._validate_deployment(value)
        if value.get("status") not in {"starting", "ready", "degraded"}:
            raise SlotFencedError("The owned deployment is no longer serving")
        self.deployment = value
        self._event(
            "provider_authority_refreshed",
            duration_s=time.monotonic() - started,
            status=value["status"],
            ready_robots=value.get("ready_robots", 0),
            expires_at=value["expires_at"],
            control_lease_until=value["control_lease_until"],
        )

    async def _watch_deployment_status(self):
        """Keep renewal reads outside model RTT; stale authority never permits a POST."""
        retry = False
        try:
            while not self._closing:
                remaining = self._authority_remaining()
                if remaining <= 0:
                    raise SlotFencedError("The last verified deployment authority expired")
                delay = min(
                    STATUS_REFRESH_RETRY_S if retry else STATUS_REFRESH_INTERVAL_S,
                    max(0.1, remaining - STATUS_REFRESH_MARGIN_S),
                    remaining,
                )
                await asyncio.sleep(delay)
                if self._closing:
                    return
                try:
                    await self._refresh_deployment_status()
                    retry = False
                except (httpx.TransportError, TimeoutError) as exc:
                    retry = True
                    self._event("provider_authority_refresh_failed", **_error_fields(exc))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._status_error = exc
            self._ready = False
            self._fenced.update(range(self.concurrency))
            self._cleanup_event("provider_authority_lost", **_error_fields(exc))
            # Stop owned compute even if the client is idle. The next predict or
            # context exit still reports this failure; cleanup is never acceptance.
            if self._close_task is None:
                self._close_task = asyncio.create_task(self._cleanup())
                self._close_task.add_done_callback(
                    lambda task: None if task.cancelled() else task.exception()
                )

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
                self._journal_executor = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="hud-pooled-journal"
                )
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
            self._status_task = asyncio.create_task(self._watch_deployment_status())
            return self
        except BaseException as exc:
            try:
                await self.close()
            except BaseException as cleanup:
                exc.add_note(
                    f"Pooled cleanup error (resource cleanup confirmed={self.cleanup_confirmed}): "
                    f"{cleanup}"
                )
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

    async def _recovery_call(self, client, identity, *, operation):
        started = time.monotonic()
        try:
            if operation == "result_lookup":
                return await client.get_result(
                    **{k: v for k, v in identity.items() if k != "episode_id"}
                )
            return await client.resolve_request(**identity)
        except (httpx.TransportError, DropbearError) as exc:
            self._event(
                "provider_http_error",
                operation=operation,
                duration_s=time.monotonic() - started,
                **_error_fields(exc),
                **{k: v for k, v in identity.items() if k != "deployment_id"},
            )
            raise

    async def _recover(self, client, identity, *, require_atomic_resolution=False):
        """GET first, then atomically resolve unknown work. Never replay the inference POST."""
        self._event(
            "inference_recovery", **{k: v for k, v in identity.items() if k != "deployment_id"}
        )
        async with asyncio.timeout(self.recovery_timeout):
            while True:
                try:
                    try:
                        result = await self._recovery_call(
                            client, identity, operation="result_lookup"
                        )
                    except (httpx.TransportError, DropbearError):
                        result = None
                    if result is not None and self._check_identity(result, identity) != "pending":
                        if not (
                            require_atomic_resolution
                            and result.get("status") == "failed"
                            and result["error"]["code"] == "request_not_admitted"
                        ):
                            return result, "result_lookup"
                    result = await self._recovery_call(client, identity, operation="resolve")
                    if self._check_identity(result, identity) != "pending":
                        return result, "resolve"
                except (httpx.TransportError, DropbearError):
                    # Even an uncertain resolution must retain this same identity.
                    pass
                await asyncio.sleep(self.poll_interval)

    async def predict(
        self, slot, observation, instruction, episode_id, noise_seed, *, trace_id=None
    ):
        if not self.is_slot_available(slot):
            raise SlotFencedError(
                f"Slot {slot} is unavailable, busy or fenced"
            ) from self._status_error
        if self._authority_remaining() <= 0:
            self._fenced.update(range(self.concurrency))
            raise SlotFencedError("The owned deployment's finite grant expired")
        self._busy.add(slot)
        current = asyncio.current_task()
        self._active.add(current)
        identity = None
        submitted = False
        attempt_index = 0
        logical_call_id = str(uuid4())
        journal_before_post_s = journal_terminal_s = 0.0
        started = time.monotonic()

        async def terminal_checkpoint(phase, **details):
            nonlocal journal_terminal_s
            checkpoint_started = time.monotonic()
            try:
                await self._checkpoint(
                    phase,
                    **identity,
                    logical_call_id=logical_call_id,
                    attempt_index=attempt_index,
                    **details,
                )
            finally:
                journal_terminal_s += time.monotonic() - checkpoint_started

        try:
            for key in ("agentview_image", "robot0_eye_in_hand_image"):
                frame = np.asarray(observation[key])
                if frame.shape != (360, 360, 3) or frame.dtype != np.uint8:
                    raise ValueError("Pooled LIBERO requires native raw360 RGB uint8 cameras")
            # Own the arrays while PNG encoding runs outside the asyncio event loop.
            raw = {key: np.array(value, copy=True) for key, value in observation.items()}
            for attempt_index in range(self.max_not_admitted_resubmissions + 1):
                if self._closing or not self._ready or self._authority_remaining() <= 0:
                    raise SlotFencedError("The owned deployment cannot accept another request")
                submitted = False
                journal_before_post_s = journal_terminal_s = 0.0
                identity = {
                    "deployment_id": self.deployment_id,
                    "robot_slot": slot,
                    "request_id": str(uuid4()),
                    "sequence": self._sequences[slot],
                    "episode_id": episode_id,
                }
                correlation = {k: v for k, v in identity.items() if k != "deployment_id"}
                correlation.update(
                    trace_id=trace_id, logical_call_id=logical_call_id, attempt_index=attempt_index
                )
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
                    await self._checkpoint(
                        "before_post",
                        **identity,
                        trace_id=trace_id,
                        logical_call_id=logical_call_id,
                        attempt_index=attempt_index,
                    )
                finally:
                    journal_before_post_s = time.monotonic() - checkpoint_started
                if self._closing or self._authority_remaining() <= 0:
                    raise SlotFencedError("The owned deployment cannot accept another request")
                self._sequences[slot] += 1
                self._pending[slot] = identity
                submitted = True
                post_started = time.monotonic()
                result = None
                error_fields = {"error_type": None, "error_code": None}
                try:
                    result = await self._clients[slot].predict(request)
                except (httpx.TransportError, DropbearError) as exc:
                    error_fields = _error_fields(exc)
                finally:
                    self._event(
                        "inference_post",
                        duration_s=time.monotonic() - post_started,
                        # This legacy field means a decoded SDK envelope was returned;
                        # an HTTP error can still have a provider_http_response event.
                        response_received=result is not None,
                        **error_fields,
                        journal_before_post_s=journal_before_post_s,
                        **correlation,
                    )
                resolution_source = "predict"
                may_resubmit = attempt_index < self.max_not_admitted_resubmissions
                if result is None or self._check_identity(result, identity) == "pending":
                    result, resolution_source = await self._recover(
                        self._clients[slot], identity, require_atomic_resolution=may_resubmit
                    )
                elif (
                    may_resubmit
                    and result.get("status") == "failed"
                    and result["error"]["code"] == "request_not_admitted"
                ):
                    # Even a decoded failed POST is not authority to submit again.
                    result, resolution_source = await self._recover(
                        self._clients[slot], identity, require_atomic_resolution=True
                    )
                status = self._check_identity(result, identity)
                await terminal_checkpoint(
                    status,
                    resolution_source=resolution_source,
                    error_code=result["error"]["code"] if status == "failed" else None,
                )
                del self._pending[slot]
                if status == "failed":
                    self._event(
                        "inference_failed",
                        code=result["error"]["code"],
                        resolution_source=resolution_source,
                        journal_before_post_s=journal_before_post_s,
                        journal_terminal_s=journal_terminal_s,
                        **correlation,
                    )
                    if (
                        may_resubmit
                        and result["error"]["code"] == "request_not_admitted"
                        and resolution_source == "resolve"
                    ):
                        # The resolver's CAS fences a late original admission. The
                        # next loop sends a new identity for this same logical input.
                        self._event(
                            "inference_resubmission",
                            reason="request_not_admitted",
                            resolution_source=resolution_source,
                            **correlation,
                        )
                        await asyncio.sleep(0)  # Observe cancellation before any new POST.
                        continue
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
                    result, _ = await self._protected(
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
        failure = None
        try:
            async with asyncio.timeout(self.close_timeout):
                if self._status_task is not None:
                    self._status_task.cancel()
                    await asyncio.gather(self._status_task, return_exceptions=True)
                active = list(self._active)
                for task in active:
                    task.cancel()
                if active:
                    await asyncio.gather(*active, return_exceptions=True)
                if self._control is not None and self._create_attempted:
                    if self.deployment is None:
                        # Creation may have succeeded before the response was lost.
                        await self._create()
                    # Durability failures still fail the job, after paid teardown.
                    await self._cleanup_checkpoint("stop_intent")
                    self._cleanup_event("provider_stopping")
                    while True:
                        try:
                            await self._control.stop(self.deployment_id)
                            value = await self._control.deployment(self.deployment_id)
                            if value.get("id") != self.deployment_id:
                                raise ValueError("Cleanup returned another deployment")
                            if value.get("status") == "stopped":
                                self.deployment = value
                                self.cleanup_confirmed = True
                                self._cleanup_event("provider_stopped")
                                await self._cleanup_checkpoint("stopped")
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
            failure = exc
            self.cleanup_error = type(exc).__name__
            self._fenced.update(range(self.concurrency))
            self._cleanup_event(
                "provider_cleanup_uncertain",
                error_type=self.cleanup_error,
                pending_requests=list(self._pending.values()),
            )
            await self._cleanup_checkpoint(
                "cleanup_uncertain",
                error_type=self.cleanup_error,
                pending_requests=list(self._pending.values()),
            )
        finally:
            try:
                async with asyncio.timeout(min(10, self.close_timeout)):
                    closed = await asyncio.gather(
                        *(c.close() for c in [*self._clients, self._control] if c is not None),
                        return_exceptions=True,
                    )
                    for result in closed:
                        if isinstance(result, Exception):
                            self._cleanup_evidence_errors.append(("client_close", result))
            except BaseException as exc:
                if failure is None:
                    failure = exc
                else:
                    failure.add_note(f"SDK client closure also failed: {type(exc).__name__}")
            finally:
                self._encoder.shutdown(wait=False, cancel_futures=True)
                if self._journal is not None:
                    try:
                        self._journal.close()
                    except Exception as exc:
                        self._cleanup_evidence_errors.append(("journal:close", exc))
                if self._journal_executor is not None:
                    self._journal_executor.shutdown(wait=False)
        if failure is not None:
            self.cleanup_error = type(failure).__name__
            if self._cleanup_evidence_errors:
                failure.add_note(
                    "Local cleanup evidence also failed at: "
                    + ", ".join(stage for stage, _ in self._cleanup_evidence_errors)
                )
            raise failure
        if self._cleanup_evidence_errors:
            self._cleanup_event(
                "provider_cleanup_evidence_failed",
                stages=[stage for stage, _ in self._cleanup_evidence_errors],
                error_types=[type(exc).__name__ for _, exc in self._cleanup_evidence_errors],
            )
            self.cleanup_error = "ExceptionGroup"
            raise ExceptionGroup(
                "Resource cleanup completed, but local cleanup evidence failed",
                [exc for _, exc in self._cleanup_evidence_errors],
            )

    async def close(self):
        self._closing = True
        self._ready = False
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._cleanup())
        await self._protected(write=self._close_task)
        if self._status_error is not None:
            raise self._status_error

    async def __aexit__(self, exc_type, exc, tb):
        try:
            await self.close()
        except BaseException as cleanup:
            if exc is None:
                raise
            exc.add_note(
                f"Pooled cleanup error (resource cleanup confirmed={self.cleanup_confirmed}): "
                f"{cleanup}"
            )
