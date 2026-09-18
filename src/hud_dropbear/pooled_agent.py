"""A provider-configured HUD agent: independent episodes, shared HTTP inference."""

import asyncio
import hashlib
import time
from dataclasses import dataclass

import numpy as np
from hud.agents.base import Agent
from hud.agents.robot.adapter import Adapter
from hud.agents.robot.agent import RobotAgent
from hud.agents.robot.model import Model
from hud.telemetry.context import get_current_trace_id

from .contract import (
    CAMERAS,
    CHUNK_SIZE,
    CONTROL_HZ,
    MAX_STEPS,
    POOLED_PROFILE,
    POOLED_RESOLUTION,
    build_contract,
    finite_array,
)


@dataclass(frozen=True)
class PooledInput:
    observation: dict
    instruction: str
    input_sha256: str


class PooledLiberoAdapter(Adapter):
    """Preserve raw RGB and quaternion geometry for server-owned preprocessing."""

    def __init__(self):
        super().__init__(chunk_size=CHUNK_SIZE)

    def bind(self, action_space, observation_space):
        expected = build_contract(profile=POOLED_PROFILE)["features"]
        for name, contract in expected.items():
            actual = action_space if name == "action" else observation_space.get(name)
            if actual is None or any(
                actual.get(key) != value for key, value in contract.items() if key != "role"
            ):
                raise ValueError(f"Pooled LIBERO contract mismatch: {name}")
        super().bind(action_space, observation_space)

    def adapt_observation(self, obs, prompt):
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("A LIBERO task instruction is required")
        data = obs["data"]
        raw = {}
        for key in CAMERAS:
            frame = np.asarray(data[key])
            if frame.dtype != np.uint8 or frame.shape != (POOLED_RESOLUTION, POOLED_RESOLUTION, 3):
                raise ValueError(f"{key} must be raw 360x360 RGB uint8")
            raw[key] = frame
        for key, shape in (
            ("robot0_eef_pos", (3,)),
            ("robot0_eef_quat_xyzw", (4,)),
            ("robot0_gripper_qpos", (2,)),
        ):
            raw[key] = finite_array(data[key], shape, key)
        quat = raw.pop("robot0_eef_quat_xyzw")
        if not np.isclose(np.linalg.norm(quat), 1.0, atol=1e-3):
            raise ValueError("EEF quaternion must be normalized")
        raw["robot0_eef_quat"] = quat
        digest = hashlib.sha256(prompt.encode("utf-8") + b"\0")
        for key, value in raw.items():
            digest.update(key.encode() + b"\0" + value.tobytes(order="C"))
        return PooledInput(raw, prompt, digest.hexdigest())

    def adapt_chunk(self, chunk, obs):
        return finite_array(chunk, (CHUNK_SIZE, 7), "model action chunk")

    def adapt_action(self, action, obs):
        return finite_array(action, (7,), "LIBERO action")


class PooledModel(Model):
    def __init__(self, provider, *, slot, episode_id, trace_id, seed, emit, fields):
        self.provider = provider
        self.slot, self.episode_id, self.trace_id = slot, episode_id, trace_id
        self.seed, self.emit, self.fields = seed, emit, fields
        self.calls = 0

    def infer(self, batch):
        raise TypeError("PooledModel is async-only; HUD batching is not required")

    async def ainfer(self, batch):
        if self.calls == 0:
            self.emit("episode_first_model_input", **self.fields)
        started = time.monotonic()
        inference_index, outcome = self.calls, "error"
        try:
            actions = await self.provider.predict(
                slot=self.slot,
                observation=batch.observation,
                instruction=batch.instruction,
                episode_id=self.episode_id,
                noise_seed=self.seed + self.calls,
                trace_id=self.trace_id,
            )
            actions = finite_array(actions, (CHUNK_SIZE, 7), "model action chunk")
            self.emit(
                "inference",
                **self.fields,
                duration_s=time.monotonic() - started,
                inference_index=self.calls,
                input_sha256=batch.input_sha256,
            )
            self.calls += 1
            outcome = "completed"
            return actions
        except asyncio.CancelledError:
            outcome = "cancelled"
            raise
        finally:
            self.emit(
                "inference_attempt_finished",
                **self.fields,
                duration_s=time.monotonic() - started,
                outcome=outcome,
                inference_index=inference_index,
            )


class _EpisodeAgent(RobotAgent):
    log_every = 0

    def __init__(self, model, *, max_steps, emit, fields):
        self.model = model
        self.adapter = PooledLiberoAdapter()
        self.action_limit = max_steps
        self.emit, self.fields = emit, fields

    def should_stop(self, obs, *, step, max_steps):
        if step == 1:
            self.emit("first_action_confirmed", **self.fields)
        return step >= self.action_limit or super().should_stop(obs, step=step, max_steps=max_steps)


class PooledRobotAgent(Agent):
    """Ready-made, concurrency-safe facade accepted directly by ``Taskset.run``.

    Configure one runtime pool per job, using no more lanes than the provider's
    slots. Sequential jobs can share one open provider. HUD creates and grades
    each episode; this facade creates a fresh model, adapter and action queue.
    The provider alone owns persistent HTTP clients, slot sequences and batching.
    """

    def __init__(self, *, provider, runtimes, max_steps=MAX_STEPS, emit=None):
        for name, width in (("Provider", provider.concurrency), ("Runtime", runtimes.concurrency)):
            if type(width) is not int or not 1 <= width <= 64:
                raise ValueError(f"{name} concurrency must be an integer from 1 to 64")
        if runtimes.concurrency > provider.concurrency:
            raise ValueError("Runtime concurrency must not exceed the provider's slot count")
        if type(max_steps) is not int or not 1 <= max_steps <= MAX_STEPS:
            raise ValueError("max_steps must be between 1 and 600")
        self.provider, self.runtimes, self.max_steps = provider, runtimes, max_steps
        self.emit = emit or (lambda event, **fields: None)

    async def __call__(self, run):
        lane = self.runtimes.lane_for(run.runtime)
        columns = lane.current_task.columns or {}
        trace_id = run.trace_id or get_current_trace_id()
        fields = dict(
            lane_id=lane.slot,
            episode_id=lane.episode_id,
            trace_id=trace_id,
            task=lane.current_task.slug,
            case_index=columns.get("case_index"),
            lane_episode_index=lane.episodes - 1,
        )
        started = time.monotonic()
        try:
            contract = run.client.binding(RobotAgent.robot_protocol).params.get("contract") or {}
            if (
                contract.get("control_rate") != CONTROL_HZ
                or contract.get("observation_profile") != POOLED_PROFILE
            ):
                raise ValueError("Pooled inference requires the raw360 LIBERO profile at 20 Hz")
            if not self.provider.is_slot_available(lane.slot):
                raise RuntimeError("Inference slot is fenced after an unresolved request")
            self.emit("environment_ready", **fields)
            run.trace.extra["dropbear"] = {
                **self.provider.identity,
                **fields,
                "active_concurrency": self.runtimes.concurrency,
                "provider_concurrency": self.provider.concurrency,
                "observation_profile": POOLED_PROFILE,
                "control_hz": CONTROL_HZ,
            }
            model = PooledModel(
                self.provider,
                slot=lane.slot,
                episode_id=lane.episode_id,
                trace_id=trace_id,
                seed=int(columns.get("noise_seed", 0)),
                emit=self.emit,
                fields=fields,
            )
            agent = _EpisodeAgent(model, max_steps=self.max_steps, emit=self.emit, fields=fields)
            # A final observation tick records the state after action 600.
            await agent(run, max_steps=self.max_steps + 1)
            self.emit(
                "episode_driven",
                **fields,
                duration_s=time.monotonic() - started,
                inference_calls=model.calls,
            )
        except BaseException as exc:
            self.emit("episode_error", **fields, error_type=type(exc).__name__)
            raise
