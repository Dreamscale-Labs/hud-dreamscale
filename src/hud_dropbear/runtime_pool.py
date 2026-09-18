"""Job-owned scalar simulators, with stable inference-slot identity across resets."""

import asyncio
import time
import uuid
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass, field

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
    ):
        if type(concurrency) is not int or not 1 <= concurrency <= 64:
            raise ValueError("concurrency must be between 1 and 64")
        self.provider = provider
        self.concurrency = concurrency
        self.emit = emit or (lambda event, **fields: None)
        self.startup_timeout = startup_timeout
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
            runtime = await self._owners[slot].enter_async_context(
                self.provider(self._representatives[slot])
            )
            self.emit(
                "runtime_lease_acquired",
                lane_id=slot,
                duration_s=time.monotonic() - started,
                runtime_session_id=runtime.params.get("session_id"),
            )
            # HUDRuntime can yield a local tunnel before the environment has
            # started. A public hello completes initialization without starting
            # any task or claiming a robot slot. Closing this idle client leaves
            # the job-owned runtime available for the real episode connection.
            connect_started = time.monotonic()
            async with self._connector(runtime, ready_timeout=self.startup_timeout) as client:
                if client.manifest.server_info.name != self._representatives[slot].env:
                    raise ValueError("Runtime initialized the wrong HUD environment")
            fields = dict(
                lane_id=slot,
                duration_s=time.monotonic() - started,
                control_connect_s=time.monotonic() - connect_started,
                runtime_session_id=runtime.params.get("session_id"),
                boundary="control_ready",
            )
            self.emit("simulator_control_ready", **fields)
            self.emit("simulator_acquired", **fields)  # Compatibility with saved timing readers.
            return Lane(slot, runtime)

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
            self.emit("environment_starting", **fields)
            self.emit("environment_acquired", **fields)
            try:
                yield lane.runtime
            finally:
                self.emit("episode_released", **fields)
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
            for lane, result in zip(self.lanes, drained, strict=True):
                if isinstance(result, BaseException):
                    errors.append(result)
                    self.emit(
                        "episode_cleanup_error", lane_id=lane.slot, error_type=type(result).__name__
                    )
            # Even a wedged borrowed driver must not skip paid runtime teardown.
            results = await asyncio.gather(
                *(close_owner(owner) for owner in self._owners), return_exceptions=True
            )
            for slot, result in enumerate(results):
                if isinstance(result, BaseException):
                    errors.append(result)
                    self.emit(
                        "simulator_cleanup_error", lane_id=slot, error_type=type(result).__name__
                    )
                else:
                    self.emit("simulator_closed", lane_id=slot)
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
