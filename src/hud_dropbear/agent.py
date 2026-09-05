"""One reusable inference session; HUD owns each episode's execution loop."""

import asyncio
import json
import time
from dataclasses import asdict
from importlib.resources import files

import dropbear
from hud.agents.robot.agent import RobotAgent
from hud.agents.robot.model import Model

from .adapter import LiberoAdapter
from .contract import CHECKPOINT, CHUNK_SIZE, CONTROL_HZ, MAX_STEPS, MODEL, REVISION, finite_array


def serving_identity(policy):
    config = policy.resolved_optimization_config
    if policy.model != MODEL or config.backend != "tensorrt":
        raise ValueError("The demo requires MolmoAct2-LIBERO served with TensorRT")
    if config.rtc != "off" or config.calibration != "off":
        raise ValueError("The synchronous HUD baseline requires RTC and calibration off")
    if (policy.action_hz, policy.chunk_size) != (CONTROL_HZ, CHUNK_SIZE):
        raise ValueError("Dropbear's resolved LIBERO timing contract has changed")
    if not config.tensorrt_artifact_fingerprint:
        raise ValueError("Dropbear did not report its TensorRT artifact fingerprint")
    contracts = json.loads(files("hud_dropbear").joinpath("serving_contracts.json").read_text())
    artifact = contracts.get(config.tensorrt_artifact_id)
    if artifact is None or artifact["fingerprint"] != config.tensorrt_artifact_fingerprint:
        raise ValueError("Unverified TensorRT artifact; verify its checkpoint manifest before use")
    if artifact["checkpoint"] != CHECKPOINT or artifact["revision"] != REVISION:
        raise ValueError("The TensorRT artifact targets a different checkpoint revision")
    return {
        "session_id": policy.session_id,
        "model": policy.model,
        "region": policy.region,
        "backend": config.backend,
        "transport": policy.transport_mode,
        "checkpoint": CHECKPOINT,
        "checkpoint_revision": REVISION,
        "checkpoint_verification": "manifest_and_engine_fingerprint",
        "tensorrt_artifact_fingerprint": config.tensorrt_artifact_fingerprint,
        "tensorrt_artifact_id": config.tensorrt_artifact_id,
        "action_hz": policy.action_hz,
        "chunk_size": policy.chunk_size,
    }


class DropbearModel(Model):
    def __init__(self, policy, emit):
        self.policy = policy
        self.emit = emit
        self.trace_id = None

    def infer(self, batch):
        raise TypeError("DropbearModel is async-only; use ainfer, without HUD BatchedModel")

    async def ainfer(self, batch):
        started = time.monotonic()
        result = await self.policy.predict(
            batch.observation, instruction=batch.instruction, timeout_s=60.0
        )
        actions = finite_array(result.actions, (CHUNK_SIZE, 7), "model action chunk")
        self.emit(
            "inference",
            trace_id=self.trace_id,
            session_id=self.policy.session_id,
            observation_id=result.observation_id,
            chunk_id=result.chunk_id,
            duration_s=time.monotonic() - started,
            timing=asdict(result.timing),
        )
        return actions


class DropbearRobotAgent(RobotAgent):
    max_steps = MAX_STEPS
    log_every = 50

    def __init__(self, *, region="ap-southeast-2", emit=None, connector=None):
        self.region = region
        self.emit = emit or (lambda event, **fields: None)
        self._connector = connector or dropbear.aconnect
        self._policy = None
        self._running = False
        self._closed = False
        self._trace_id = None
        self.adapter = LiberoAdapter()
        self.model = None
        self.identity = None

    async def __aenter__(self):
        if self._policy is not None or self._closed:
            raise RuntimeError("Use a fresh agent for each evaluation job")
        self.emit("provider_starting", model=MODEL, region=self.region)
        started = time.monotonic()
        try:
            self._policy = await self._connector(
                model=MODEL,
                region=self.region,
                acceleration="tensorrt",
                rtc="off",
                calibration="off",
                control_hz=CONTROL_HZ,
                keep_warm=0,
                idle_timeout=900,
                startup_timeout=900,
            )
            self.identity = serving_identity(self._policy)
            self.model = DropbearModel(self._policy, self.emit)
            self.emit("provider_ready", duration_s=time.monotonic() - started, **self.identity)
            return self
        except BaseException:
            await self.close()
            raise

    async def close(self):
        if self._closed:
            return
        self._closed = True
        if self._policy is not None:
            # A cancellation must not abandon the billable session cleanup.
            task = asyncio.create_task(self._policy.close())
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                await task
                raise
            self.emit("provider_closed", session_id=self._policy.session_id)

    async def __aexit__(self, *exc):
        await self.close()

    async def __call__(self, run, *, max_steps=None):
        if self.model is None or self._closed:
            raise RuntimeError("Use 'async with DropbearRobotAgent() as agent' around the job")
        if self._running:
            raise RuntimeError("This provider supports one rollout at a time; max_concurrent=1")
        self._running = True
        self._trace_id = run.trace_id
        self.model.trace_id = self._trace_id
        started = time.monotonic()
        try:
            cap = run.client.binding(self.robot_protocol)
            contract = cap.params.get("contract") or {}
            if contract.get("control_rate") != CONTROL_HZ:
                raise ValueError("The actual LIBERO environment control rate must be 10 Hz")
            self.emit("environment_ready", trace_id=self._trace_id)
            run.trace.extra["dropbear"] = self.identity
            await super().__call__(run, max_steps=max_steps)
            self.emit(
                "episode_driven", trace_id=self._trace_id, duration_s=time.monotonic() - started
            )
        except BaseException as exc:
            self.emit("episode_error", trace_id=self._trace_id, error_type=type(exc).__name__)
            raise
        finally:
            self._running = False
            self.model.trace_id = None

    def should_stop(self, obs, *, step, max_steps):
        if step == 1:
            # The first action has been sent and the simulator's next observation received.
            self.emit("first_action_confirmed", trace_id=self._trace_id)
        return super().should_stop(obs, step=step, max_steps=max_steps)
