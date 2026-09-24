"""Job-owned scalar simulators, with stable inference-slot identity across resets."""

import asyncio
import math
import time
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field, replace

from hud.clients import connect
from hud.telemetry.context import get_current_trace_id

from .telemetry import CURRENT_TASK


@dataclass
class Lane:
    slot: int
    runtime: object
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    episodes: int = 0
    current_task: object = None
    episode_id: str | None = None


class RuntimePool:
    """Own one scalar runtime per lane, outside every episode's HUD trace.

    Rows must contain ``columns.lane_id``. Explicit placement guarantees each
    lane receives its planned warm repeats even when episodes finish unevenly.
    A borrow spans HUD's setup, agent execution, grading and claim cleanup.
    """

    DEFAULT_LANE_READY_ATTEMPTS = 3

    def __init__(
        self,
        provider,
        tasks,
        *,
        concurrency,
        emit=None,
        startup_timeout=900,
        cleanup_timeout=60,
        connector=None,
        lane_ready_timeout=None,
        lane_ready_attempts=None,
    ):
        if type(concurrency) is not int or not 1 <= concurrency <= 64:
            raise ValueError("concurrency must be between 1 and 64")
        if (
            type(startup_timeout) not in (int, float)
            or not math.isfinite(startup_timeout)
            or startup_timeout <= 0
        ):
            raise ValueError("startup_timeout must be a positive finite number")
        if lane_ready_timeout is None:
            lane_ready_timeout = min(startup_timeout, 300)
        if lane_ready_attempts is None:
            lane_ready_attempts = self.DEFAULT_LANE_READY_ATTEMPTS
        if (
            type(lane_ready_timeout) not in (int, float)
            or not math.isfinite(lane_ready_timeout)
            or not 0 < lane_ready_timeout <= startup_timeout
        ):
            raise ValueError("lane_ready_timeout must be positive and within startup_timeout")
        if type(lane_ready_attempts) is not int or lane_ready_attempts < 1:
            raise ValueError("lane_ready_attempts must be a positive integer")
        self.provider = provider
        self.concurrency = concurrency
        self.emit = emit or (lambda event, **fields: None)
        self.startup_timeout = startup_timeout
        # A leased simulator whose control connection never becomes ready is
        # released and replaced, a bounded number of times, inside the overall
        # startup bound; the remote's own readiness bound is set to the same value.
        self.lane_ready_timeout = lane_ready_timeout
        self.lane_ready_attempts = lane_ready_attempts
        self.cleanup_timeout = cleanup_timeout
        self._connector = connector or connect
        self.tasks = list(tasks)
        self._representatives = {}
        for task in self.tasks:
            slot = (task.columns or {}).get("lane_id")
            if type(slot) is not int or not 0 <= slot < concurrency:
                raise ValueError("Every task requires a valid columns.lane_id")
            if task.runtime_config is not None or task.verifier is not None:
                raise ValueError("Pooled tasks use the pool's placement and simulator grading")
            self._representatives.setdefault(slot, task)
        if set(self._representatives) != set(range(concurrency)):
            raise ValueError("The cohort must assign at least one episode to every lane")
        if len({task.env for task in self.tasks}) != 1:
            raise ValueError("A runtime pool serves one environment")
        self.lanes = []
        self._by_url = {}
        self._owners = []
        self._depth = 0
        self._closed = False
        self.cleanup_confirmed = False
        self.cleanup_error = None

    async def __aenter__(self):
        # Taskset.run enters context-manager providers too. It borrows this open
        # pool; only the outer job owner releases the billable simulators.
        if self._closed:
            raise RuntimeError("Use a fresh RuntimePool for each evaluation job")
        if self._depth:
            self._depth += 1
            return self
        if get_current_trace_id() is not None:
            raise RuntimeError("Open RuntimePool outside episode traces")
        self._owners = [AsyncExitStack() for _ in range(self.concurrency)]

        async def open_lane(slot):
            started = time.monotonic()
            self.emit("simulator_starting", lane_id=slot)
            attempt = 0
            while True:
                attempt += 1
                phase = "runtime_lease"
                attempt_started = time.monotonic()
                # Each attempt owns its lease in a fresh stack so a released
                # simulator never lingers on the lane's final owner.
                owner = AsyncExitStack()
                await owner.__aenter__()
                self._owners[slot] = owner
                try:
                    runtime = await owner.enter_async_context(
                        self.provider(self._representatives[slot])
                    )
                    # HUD connect prioritizes Runtime.params over its timeout keyword.
                    # Copy the public descriptor so the pool's bound takes effect without
                    # changing the provider's descriptor or its ownership/cleanup context.
                    runtime = replace(
                        runtime,
                        params={**runtime.params, "ready_timeout": self.lane_ready_timeout},
                    )
                    self.emit(
                        "runtime_lease_acquired",
                        lane_id=slot,
                        attempt=attempt,
                        duration_s=time.monotonic() - attempt_started,
                        runtime_session_id=runtime.params.get("session_id"),
                        ready_timeout_s=self.lane_ready_timeout,
                    )
                    # A lease can precede remote readiness. Hello initializes control
                    # without starting a task or claiming a robot slot.
                    phase = "control_hello"
                    connect_started = time.monotonic()
                    async with asyncio.timeout(self.lane_ready_timeout):
                        async with self._connector(
                            runtime, ready_timeout=self.lane_ready_timeout
                        ) as client:
                            phase = "control_identity"
                            if client.manifest.server_info.name != self._representatives[slot].env:
                                raise ValueError("Runtime initialized the wrong HUD environment")
                    fields = dict(
                        lane_id=slot,
                        duration_s=time.monotonic() - started,
                        control_connect_s=time.monotonic() - connect_started,
                        runtime_session_id=runtime.params.get("session_id"),
                        boundary="control_ready",
                        attempts=attempt,
                    )
                    self.emit("simulator_control_ready", **fields)
                    self.emit("simulator_acquired", **fields)  # Saved timing reader compat.
                    return Lane(slot, runtime)
                except BaseException as exc:
                    elapsed = time.monotonic() - started
                    retry = (
                        not isinstance(exc, (asyncio.CancelledError, ValueError))
                        and phase != "control_identity"
                        and attempt < self.lane_ready_attempts
                        and elapsed + self.lane_ready_timeout <= self.startup_timeout
                    )
                    if retry:
                        # Release the unready simulator before leasing a replacement;
                        # a release failure is not retried into a second paid lease.
                        try:
                            await owner.aclose()
                        except Exception as release_exc:
                            retry = False
                            exc = release_exc
                    if retry:
                        try:
                            self.emit(
                                "simulator_control_retry",
                                lane_id=slot,
                                attempt=attempt,
                                phase=phase,
                                duration_s=time.monotonic() - attempt_started,
                                lane_ready_timeout_s=self.lane_ready_timeout,
                                error_type=type(exc).__name__[:64],
                            )
                        except Exception:
                            pass
                        continue
                    event = (
                        "simulator_startup_cancelled"
                        if isinstance(exc, asyncio.CancelledError)
                        else "simulator_startup_failed"
                    )
                    try:
                        self.emit(
                            event,
                            lane_id=slot,
                            phase=phase,
                            attempt=attempt,
                            duration_s=elapsed,
                            startup_timeout_s=self.startup_timeout,
                            error_type=type(exc).__name__[:64],
                        )
                    except Exception:
                        # Preserve the original failure and unconditional paid teardown.
                        pass
                    raise exc

        pending = [asyncio.create_task(open_lane(i)) for i in range(self.concurrency)]
        try:
            async with asyncio.timeout(self.startup_timeout):
                self.lanes = list(await asyncio.gather(*pending))
            self._by_url = {lane.runtime.url: lane for lane in self.lanes}
            if len(self._by_url) != self.concurrency:
                raise ValueError("Each lane must own a distinct scalar runtime URL")
        except BaseException:
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            await self.close()
            raise
        self._depth = 1
        return self

    def lane_for(self, runtime_url):
        if runtime_url not in self._by_url:
            raise ValueError("Run does not belong to this runtime pool")
        lane = self._by_url[runtime_url]
        if lane.current_task is None:
            raise RuntimeError("Runtime lane is not leased to an episode")
        return lane

    @asynccontextmanager
    async def __call__(self, task):
        if not self._depth or self._closed:
            raise RuntimeError("Open RuntimePool around Taskset.run")
        slot = (task.columns or {}).get("lane_id")
        if type(slot) is not int or not 0 <= slot < self.concurrency:
            raise ValueError("Task does not select a valid runtime lane")
        lane = self.lanes[slot]
        async with lane.lock:
            if self._closed:
                raise RuntimeError("RuntimePool is closing; queued episodes cannot start")
            previous = CURRENT_TASK.get()
            CURRENT_TASK.set(task.slug)
            lane.current_task = task
            lane.episode_id = str(uuid.uuid4())
            lane.episodes += 1
            fields = dict(
                lane_id=slot,
                episode_id=lane.episode_id,
                trace_id=get_current_trace_id(),
                task=task.slug,
                lane_episode_index=lane.episodes - 1,
                case_index=(task.columns or {}).get("case_index"),
            )
            try:
                self.emit("environment_starting", **fields)
                self.emit("environment_acquired", **fields)
                yield lane.runtime
            finally:
                try:
                    self.emit("episode_released", **fields)
                finally:
                    lane.current_task = None
                    lane.episode_id = None
                    # HUD's shielded cleanup can run in a copied context: tokens
                    # cannot be reset there, so restore the previous value directly.
                    CURRENT_TASK.set(previous)

    async def close(self):
        if self._closed:
            return
        self._closed = True

        async def cleanup():
            async def drain_lane(lane):
                # HUD returns timed-out Run objects before the cancelled driver's
                # cleanup completes. Keep the control channel alive until its
                # borrow (including grading/recorder cleanup) has unwound.
                async with asyncio.timeout(self.cleanup_timeout):
                    async with lane.lock:
                        pass

            async def close_owner(owner):
                async with asyncio.timeout(self.cleanup_timeout):
                    await owner.aclose()

            drained = await asyncio.gather(
                *(drain_lane(lane) for lane in self.lanes), return_exceptions=True
            )
            errors = []

            def emit_cleanup(event, **fields):
                try:
                    self.emit(event, **fields)
                except BaseException as exc:
                    # Failed evidence writes still fail the run, but must never
                    # prevent paid teardown or processing another owner's receipt.
                    errors.append(exc)

            for lane, result in zip(self.lanes, drained, strict=True):
                if isinstance(result, BaseException):
                    errors.append(result)
                    emit_cleanup(
                        "episode_cleanup_error", lane_id=lane.slot, error_type=type(result).__name__
                    )
            # Even a wedged borrowed driver must not skip paid runtime teardown.
            results = await asyncio.gather(
                *(close_owner(owner) for owner in self._owners), return_exceptions=True
            )
            for slot, result in enumerate(results):
                if isinstance(result, BaseException):
                    errors.append(result)
                    emit_cleanup(
                        "simulator_cleanup_error", lane_id=slot, error_type=type(result).__name__
                    )
                else:
                    emit_cleanup("simulator_closed", lane_id=slot)
            if errors:
                self.cleanup_error = ", ".join(type(error).__name__ for error in errors)
                raise BaseExceptionGroup("Simulator cleanup failed", errors)
            self.cleanup_confirmed = True

        task = asyncio.create_task(cleanup())
        cancelled = False
        try:
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    cancelled = True
            task.result()
            if cancelled:
                raise asyncio.CancelledError
        finally:
            self._depth = 0

    async def __aexit__(self, *exc):
        self._depth -= 1
        if self._depth <= 0:
            await self.close()
