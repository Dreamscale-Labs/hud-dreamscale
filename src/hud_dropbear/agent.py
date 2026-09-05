"""One reusable inference session; HUD owns each episode's execution loop."""

import asyncio
import json
import time
from dataclasses import asdict
from importlib.resources import files

import dropbear
from dropbear.config import load_config
from dropbear.control import ControlPlaneClient
from hud.agents.robot.agent import RobotAgent
from hud.agents.robot.model import Model
from hud.telemetry.context import get_current_trace_id

from .adapter import LiberoAdapter
from .contract import CHECKPOINT, CHUNK_SIZE, CONTROL_HZ, MAX_STEPS, MODEL, REVISION, finite_array


async def ready_target_artifact(policy, *, client_factory=ControlPlaneClient):
    """Resolve cold-session metadata against an unambiguous live worker snapshot.

    SDK 0.1.0a15 can retain the planned configuration after cold startup. Never
    guess from a model name or an aggregate multi-worker status response.
    """
    config = load_config()
    client = client_factory(config.control_plane_url, config.api_key)
    try:
        session = await client.get_session(policy.session_id)
        status = await client.status()
    finally:
        await client.close()
    if (
        session.status != "ready"
        or session.model != policy.model
        or session.region != policy.region
        or not session.target_key
    ):
        raise ValueError("Cannot verify the active Dropbear session's target")
    targets = status.get(policy.model, {}).get("targets", {}).values()
    matches = [t for t in targets if t.get("target_key") == session.target_key]
    if len(matches) != 1:
        raise ValueError("Cannot identify one Dropbear target for this session")
    target = matches[0]
    if not target.get("ready") or any(
        target.get(key) != 1 for key in ("worker_count", "ready_worker_count", "active_sessions")
    ):
        raise ValueError("Artifact fallback requires exactly one ready worker and active session")
    caps = target.get("worker_capabilities") or {}
    artifact = caps.get("tensorrt_artifact") or {}
    if caps.get("backends") != ["tensorrt"]:
        raise ValueError("Artifact fallback requires a dedicated TensorRT worker")
    return {
        "fingerprint": caps.get("tensorrt_artifact_fingerprint"),
        "artifact_id": artifact.get("tensorrt_artifact_id"),
    }


def serving_identity(policy, *, resolved_artifact=None, control_hz=CONTROL_HZ):
    config = policy.resolved_optimization_config
    if policy.model != MODEL or config.backend != "tensorrt":
        raise ValueError("The demo requires MolmoAct2-LIBERO served with TensorRT")
    if config.rtc != "off" or config.calibration != "off":
        raise ValueError("The synchronous HUD baseline requires RTC and calibration off")
    if (policy.action_hz, policy.chunk_size) != (control_hz, CHUNK_SIZE):
        raise ValueError("Dropbear's resolved LIBERO timing contract has changed")
    fingerprint = (
        resolved_artifact["fingerprint"]
        if resolved_artifact is not None
        else config.tensorrt_artifact_fingerprint
    )
    artifact_id = (
        resolved_artifact["artifact_id"]
        if resolved_artifact is not None
        else config.tensorrt_artifact_id
    )
    if not fingerprint:
        raise ValueError("Dropbear did not report its TensorRT artifact fingerprint")
    contracts = json.loads(files("hud_dropbear").joinpath("serving_contracts.json").read_text())
    artifact = contracts.get(artifact_id)
    if artifact is None or artifact["fingerprint"] != fingerprint:
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
        "artifact_identity_source": "single_ready_worker_status"
        if resolved_artifact is not None
        else "session_configuration",
        "session_artifact_id": config.tensorrt_artifact_id,
        "tensorrt_artifact_fingerprint": fingerprint,
        "tensorrt_artifact_id": artifact_id,
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
            transport=self.policy.transport_mode,
            fallback_reason=getattr(self.policy, "fallback_reason", None),
            duration_s=time.monotonic() - started,
            timing=asdict(result.timing),
        )
        return actions


class DropbearRobotAgent(RobotAgent):
    max_steps = MAX_STEPS
    log_every = 50

    def __init__(
        self, *, region="ap-southeast-2", control_hz=CONTROL_HZ, emit=None, connector=None
    ):
        if control_hz not in (10, 20):
            raise ValueError("Supported LIBERO control rates are 10 and 20 Hz")
        self.control_hz = control_hz
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
                control_hz=self.control_hz,
                keep_warm=0,
                idle_timeout=900,
                startup_timeout=900,
            )
            self.emit(
                "provider_connected",
                duration_s=time.monotonic() - started,
                sdk_connection_milestones={
                    key: value
                    for key, value in getattr(
                        self._policy, "_connection_timing_monotonic", {}
                    ).items()
                    if key
                    in {
                        "session_create_started_at",
                        "session_id_received_at",
                        "session_ready_at",
                        "transport_connected_at",
                    }
                    and isinstance(value, (int, float))
                },
            )
            resolved_artifact = None
            if not self._policy.resolved_optimization_config.tensorrt_artifact_fingerprint:
                resolved_artifact = await ready_target_artifact(self._policy)
            self.identity = serving_identity(
                self._policy, resolved_artifact=resolved_artifact, control_hz=self.control_hz
            )
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
        # HUD assigns run.trace_id at rollout exit; the active ID lives in context.
        self._trace_id = run.trace_id or get_current_trace_id()
        self.model.trace_id = self._trace_id
        started = time.monotonic()
        try:
            cap = run.client.binding(self.robot_protocol)
            contract = cap.params.get("contract") or {}
            if contract.get("control_rate") != self.control_hz:
                raise ValueError(
                    f"The actual LIBERO environment control rate must be {self.control_hz} Hz"
                )
            self.emit("environment_ready", trace_id=self._trace_id)
            run.trace.extra["dropbear"] = dict(self.identity)
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
            self.identity["final_transport"] = self._policy.transport_mode
            self.identity["fallback_reason"] = getattr(self._policy, "fallback_reason", None)
            run.trace.extra["dropbear"] = dict(self.identity)

    def should_stop(self, obs, *, step, max_steps):
        if step == 1:
            # The first action has been sent and the simulator's next observation received.
            self.emit("first_action_confirmed", trace_id=self._trace_id)
        return super().should_stop(obs, step=step, max_steps=max_steps)
