"""RoboLab DROID agent: one Dreamscale session per episode, 32-action open-loop chunks.

Mirrors Dreamscale's ``integrations/cosmos3-robolab/dreamscale_client.py`` (the
RoboLab ``Cosmos3Client`` with its websocket replaced by an SDK session):

- one ``dreamscale.aconnect(model, acceleration="pytorch", rtc="off",
  calibration="off", region="us-west-2", control_hz=15, keep_warm=...)``
  session per episode, closed when the episode ends;
- each request is ``dreamscale.franka.observe(left, right, wrist, joints,
  gripper)`` plus the RoboLab instruction; the reply must be a finite 32x8
  absolute joint-position chunk;
- the gripper column is thresholded at 0.5 (``Cosmos3Client._postprocess_chunk``;
  the server applies the same transform) and all 32 actions execute
  open-loop at the sim's 15 Hz before the next request.

The HUD bridge steps the sim only when an action arrives, so simulated time is
independent of inference latency; the latency is recorded, not hidden. The
session is opened *before* dialing the robot socket so a cold start never trips
the bridge's action-wait timeout.

Per episode the agent writes ``timing.jsonl`` (session, every request, the end),
``episode.json`` (local summary) and, optionally, a local MP4 of the three
policy views. HUD keeps the canonical trace and per-camera video.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import math
import re
import sys
import time
from collections import deque
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from hud.agents.robot.agent import RobotAgent
from hud.capabilities.robot import RobotClient
from hud.telemetry.robot import TraceRecorder

MODELS = ("cosmos3-nano-policy-droid", "flux-3-action-droid")
HOLD_MODEL = "hold"
CHUNK_SIZE = 32
ACTION_DIM = 8
CONTROL_HZ = 15
CAMERA_KEYS = ("exterior_1_left", "exterior_2_left", "wrist_left")
STATE_KEY = "state"
#: RoboLab's ``Cosmos3Client._build_visualization`` order: left | wrist | right.
VIDEO_ORDER = ("exterior_1_left", "wrist_left", "exterior_2_left")


# ── observation / action mapping (pure) ──────────────────────────────────────


def split_observation(data: dict[str, Any]) -> dict[str, Any]:
    """Validate one HUD RoboLab observation and split it into SDK inputs."""
    frames = {}
    for key in CAMERA_KEYS:
        if key not in data:
            raise KeyError(f"observation is missing camera {key!r}; got {sorted(data)}")
        frame = np.asarray(data[key])
        if frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError(
                f"{key} must be an HxWx3 uint8 RGB frame, got {frame.dtype} {frame.shape}"
            )
        frames[key] = np.ascontiguousarray(frame)
    state = np.asarray(data[STATE_KEY], dtype=np.float64).reshape(-1)
    if state.size != 8 or not np.isfinite(state).all():
        raise ValueError(f"state must be 8 finite values (7 joints + gripper), got {state}")
    # RoboLab clips its noisy gripper observation to [0, 1]; guard float edges.
    gripper = float(np.clip(state[7], 0.0, 1.0))
    return {
        "exterior_frame": frames["exterior_1_left"],
        "second_exterior_frame": frames["exterior_2_left"],
        "wrist_frame": frames["wrist_left"],
        "joint_positions": [float(v) for v in state[:7]],
        "gripper": gripper,
    }


def build_observation(data: dict[str, Any]):
    """The ``dreamscale.franka`` observation Dreamscale's RoboLab client sends."""
    from dreamscale.franka import observe

    return observe(**split_observation(data))


def postprocess_chunk(actions: Any) -> np.ndarray:
    """Validate a 32x8 chunk and binarize the gripper like ``Cosmos3Client``."""
    chunk = np.asarray(actions, dtype=np.float32)
    if chunk.shape != (CHUNK_SIZE, ACTION_DIM) or not np.isfinite(chunk).all():
        raise ValueError(
            f"policy returned an invalid action chunk {chunk.shape}; expected finite "
            f"({CHUNK_SIZE}, {ACTION_DIM}) absolute joint positions + gripper"
        )
    chunk = chunk.copy()
    chunk[:, -1] = (chunk[:, -1] > 0.5).astype(np.float32)
    return chunk


def hold_chunk(data: dict[str, Any]) -> np.ndarray:
    """A chunk that holds the observed joints and gripper (the no-model smoke policy)."""
    state = np.asarray(data[STATE_KEY], dtype=np.float32).reshape(-1)
    if state.size != ACTION_DIM:
        raise ValueError(f"state must have {ACTION_DIM} values, got {state.size}")
    return np.repeat(state[None], CHUNK_SIZE, axis=0)


def composite_frame(data: dict[str, Any]) -> np.ndarray:
    """Side-by-side policy views for the local MP4 (RoboLab's visualization order)."""
    frames = [np.asarray(data[key]) for key in VIDEO_ORDER]
    height = min(f.shape[0] for f in frames)
    return np.ascontiguousarray(np.concatenate([f[:height] for f in frames], axis=1))


def _timing_dict(timing: Any) -> dict[str, Any] | None:
    if timing is None:
        return None
    if dataclasses.is_dataclass(timing) and not isinstance(timing, type):
        return dataclasses.asdict(timing)
    if isinstance(timing, dict):
        return dict(timing)
    return None


def server_inference_ms(timing: dict[str, Any] | None) -> float | None:
    """Model compute as the server reports it (worker figure when present)."""
    if not timing:
        return None
    for key in ("worker_inference_ms", "server_inference_ms"):
        value = timing.get(key)
        if isinstance(value, (int, float)) and math.isfinite(value):
            return float(value)
    return None


# ── policy sessions ──────────────────────────────────────────────────────────


@dataclasses.dataclass
class ChunkReply:
    chunk: np.ndarray
    record: dict[str, Any]


class PolicySession(Protocol):
    info: dict[str, Any]

    async def predict(self, data: dict[str, Any], instruction: str) -> ChunkReply: ...

    async def close(self) -> None: ...


class PolicyFactory(Protocol):
    model: str

    async def open(self, *, label: str) -> PolicySession: ...


class HoldSession:
    """No inference: hold the pose observed at each replan (smoke tests only)."""

    def __init__(self) -> None:
        self.info: dict[str, Any] = {"model": HOLD_MODEL, "session_id": None}

    async def predict(self, data: dict[str, Any], instruction: str) -> ChunkReply:
        del instruction
        return ChunkReply(postprocess_chunk(hold_chunk(data)), {})

    async def close(self) -> None:
        return None


class HoldPolicy:
    model = HOLD_MODEL

    async def open(self, *, label: str) -> PolicySession:
        del label
        return HoldSession()


class DreamscaleSession:
    def __init__(self, policy: Any, *, predict_timeout_s: float) -> None:
        self._policy = policy
        self._timeout = predict_timeout_s
        config = getattr(policy, "resolved_optimization_config", None)
        self.info = {
            "model": getattr(policy, "model", None),
            "session_id": getattr(policy, "session_id", None),
            "region": getattr(policy, "region", None),
            "transport": getattr(policy, "transport_mode", None),
            "fallback_reason": getattr(policy, "fallback_reason", None),
            "action_hz": getattr(policy, "action_hz", None),
            "chunk_size": getattr(policy, "chunk_size", None),
            "backend": getattr(config, "backend", None),
            "rtc": getattr(config, "rtc", None),
        }

    async def predict(self, data: dict[str, Any], instruction: str) -> ChunkReply:
        observation = build_observation(data)
        result = await self._policy.predict(
            observation, instruction=instruction, timeout_s=self._timeout
        )
        chunk = postprocess_chunk(result.actions)
        timing = _timing_dict(getattr(result, "timing", None))
        return ChunkReply(
            chunk,
            {
                "observation_id": getattr(result, "observation_id", None),
                "chunk_id": getattr(result, "chunk_id", None),
                "transport": getattr(result, "transport_mode", None)
                or getattr(self._policy, "transport_mode", None),
                "server_inference_ms": server_inference_ms(timing),
                "timing": timing,
            },
        )

    async def close(self) -> None:
        await self._policy.close()


class DreamscalePolicy:
    """Opens the same session shape as Dreamscale's RoboLab client, per episode."""

    def __init__(
        self,
        model: str,
        *,
        keep_warm: int = 300,
        idle_timeout: int = 300,
        startup_timeout: float = 900.0,
        predict_timeout_s: float = 120.0,
        region: str = "us-west-2",
        connect: Callable[..., Any] | None = None,
    ) -> None:
        if model not in MODELS:
            raise ValueError(f"model must be one of {MODELS}, got {model!r}")
        if not 0 <= keep_warm <= 3600:
            raise ValueError("keep_warm must be from 0 to 3600 seconds")
        self.model = model
        self.keep_warm = keep_warm
        self.idle_timeout = idle_timeout
        self.startup_timeout = startup_timeout
        self.predict_timeout_s = predict_timeout_s
        self.region = region
        self._connect = connect

    def connect_kwargs(self) -> dict[str, Any]:
        return {
            "acceleration": "pytorch",
            "rtc": "off",
            "calibration": "off",
            "region": self.region,
            "control_hz": CONTROL_HZ,
            "keep_warm": self.keep_warm,
            "idle_timeout": self.idle_timeout,
            "startup_timeout": self.startup_timeout,
        }

    async def open(self, *, label: str) -> PolicySession:
        connect = self._connect
        if connect is None:
            import dreamscale

            connect = dreamscale.aconnect

        def progress(message: str) -> None:
            print(f"[dreamscale {label}] {message}", file=sys.stderr, flush=True)

        policy = await connect(self.model, **self.connect_kwargs(), on_progress=progress)
        return DreamscaleSession(policy, predict_timeout_s=self.predict_timeout_s)


# ── local evidence ───────────────────────────────────────────────────────────


class TimingLog:
    """Append-only per-episode JSONL; one row per event, flushed immediately."""

    def __init__(self, path: Path, *, started: float) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._started = started
        self._file = path.open("x")

    def emit(self, event: str, **fields: Any) -> dict[str, Any]:
        now = time.monotonic()
        row = {
            "event": event,
            "t_s": round(now - self._started, 6),
            "unix_s": time.time(),
            **fields,
        }
        self._file.write(json.dumps(row, default=_json_default) + "\n")
        self._file.flush()
        return row

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


class LocalVideo:
    """H.264 MP4 of the composite policy views, written as frames arrive."""

    def __init__(self, path: Path, *, fps: int = CONTROL_HZ) -> None:
        self.path = path
        self._fps = fps
        self._container: Any = None
        self._stream: Any = None
        self.frames = 0

    def write(self, frame: np.ndarray) -> None:
        import av

        frame = np.ascontiguousarray(frame[: frame.shape[0] // 2 * 2, : frame.shape[1] // 2 * 2])
        if self._container is None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._container = av.open(str(self.path), mode="w")
            self._stream = self._container.add_stream("libx264", rate=self._fps)
            self._stream.width = frame.shape[1]
            self._stream.height = frame.shape[0]
            self._stream.pix_fmt = "yuv420p"
            self._stream.options = {"crf": "23", "preset": "veryfast"}
        for packet in self._stream.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
            self._container.mux(packet)
        self.frames += 1

    def close(self) -> None:
        if self._container is None:
            return
        for packet in self._stream.encode():
            self._container.mux(packet)
        self._container.close()
        self._container = None


def episode_slug(task_name: str, episode: int, trace_id: str | None) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", task_name)
    suffix = f"__{trace_id.replace('-', '')[:8]}" if trace_id else ""
    return f"{safe}__ep{int(episode):03d}{suffix}"


def latency_stats(values: Sequence[float]) -> dict[str, Any]:
    clean = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not clean:
        return {"n": 0}

    def pct(p: float) -> float:
        # Linear interpolation between closest ranks (numpy's default).
        return float(np.percentile(clean, p))

    return {
        "n": len(clean),
        "mean": float(np.mean(clean)),
        "p50": pct(50),
        "p95": pct(95),
        "max": clean[-1],
        "min": clean[0],
    }


# ── the agent ────────────────────────────────────────────────────────────────


class RobolabDreamscaleAgent(RobotAgent):
    """HUD ``RobotAgent`` for the RoboLab env; stateless across concurrent runs."""

    max_steps = 9008
    log_every = 150

    def __init__(
        self,
        policy: PolicyFactory,
        *,
        output_dir: Path,
        local_video: bool = True,
        max_steps_cap: int | None = None,
    ) -> None:
        self.policy = policy
        self.output_dir = Path(output_dir)
        self.local_video = local_video
        self.max_steps_cap = max_steps_cap
        # RobotAgent's base contract; this agent drives its own loop.
        self.model = None
        self.adapter = None

    async def __call__(self, run: Any, *, max_steps: int | None = None) -> None:
        called_unix = time.time()
        started = time.monotonic()
        prompt = run.prompt
        if not isinstance(prompt, str):
            raise TypeError(f"run.prompt must be a str, got {type(prompt).__name__}")
        cap = run.client.binding(self.robot_protocol)
        binding = dict(run.bindings.get(cap.name, {}) or {})
        task_name = str(binding.get("task_name") or "unknown")
        episode = int(binding.get("episode") or 0)
        limit = int(binding.get("max_steps") or max_steps or self.max_steps)
        if max_steps is not None:
            limit = min(limit, max_steps)
        if self.max_steps_cap is not None:
            limit = min(limit, self.max_steps_cap)

        slug = episode_slug(task_name, episode, run.trace_id)
        episode_dir = self.output_dir / "episodes" / slug
        log = TimingLog(episode_dir / "timing.jsonl", started=started)
        log.emit(
            "agent_started",
            task_name=task_name,
            episode=episode,
            trace_id=run.trace_id,
            model=self.policy.model,
            max_steps=limit,
            instruction=prompt,
        )
        summary: dict[str, Any] = {
            "task_name": task_name,
            "episode": episode,
            "trace_id": run.trace_id,
            "model": self.policy.model,
            "instruction": prompt,
            "max_steps": limit,
            "agent_called_unix": called_unix,
            "timing_file": str((episode_dir / "timing.jsonl").relative_to(self.output_dir)),
        }
        session: PolicySession | None = None
        video = LocalVideo(episode_dir / "policy_views.mp4") if self.local_video else None
        try:
            # Open the policy session before dialing the robot socket: a cold
            # start must not trip the bridge's action-wait timeout.
            connect_started = time.monotonic()
            session = await self.policy.open(label=f"{task_name} ep{episode}")
            summary["session_connect_s"] = time.monotonic() - connect_started
            summary["session"] = session.info
            log.emit("session_ready", connect_s=summary["session_connect_s"], **session.info)

            robot = await RobotClient.connect(cap, token=binding.get("token"))
            try:
                action_space, obs_space = robot.spaces()
                obs = await robot.get_observation()
                summary["first_observation_s"] = time.monotonic() - started
                summary["observation_shapes"] = {
                    k: list(np.asarray(v).shape) for k, v in obs["data"].items()
                }
                log.emit("first_observation", shapes=summary["observation_shapes"])
                recorder = TraceRecorder(
                    run=run,
                    fps=robot.get_control_rate(),
                    obs_space=obs_space,
                    action_names=action_space.get("names"),
                )
                print(f"[agent] {task_name} ep{episode}: {prompt!r}", flush=True)
                try:
                    await self._episode(
                        robot, obs, prompt, recorder, session, video, log, summary, limit
                    )
                finally:
                    await robot.close()
                    recorder.close()
            finally:
                await robot.close()
            run.trace.status = "completed"
            run.trace.content = "done"
        except BaseException as error:
            summary["error"] = f"{type(error).__name__}: {error}"
            log.emit("error", error=summary["error"])
            raise
        finally:
            if video is not None:
                try:
                    await asyncio.to_thread(video.close)
                    if video.frames:
                        summary["local_video"] = str(video.path.relative_to(self.output_dir))
                        summary["local_video_frames"] = video.frames
                except Exception as error:  # evidence only; never fail the run on it
                    summary["local_video_error"] = repr(error)
            if session is not None:
                close_started = time.monotonic()
                try:
                    await session.close()
                    summary["session_close_s"] = time.monotonic() - close_started
                except Exception as error:
                    summary["session_close_error"] = repr(error)
            summary["episode_wall_s"] = time.monotonic() - started
            log.emit(
                "episode_end",
                **{k: summary.get(k) for k in ("steps", "terminated", "episode_wall_s")},
            )
            log.close()
            (episode_dir / "episode.json").write_text(
                json.dumps(summary, indent=2, default=_json_default) + "\n"
            )

    async def _episode(
        self,
        robot: RobotClient,
        obs: dict[str, Any],
        prompt: str,
        recorder: TraceRecorder,
        session: PolicySession,
        video: LocalVideo | None,
        log: TimingLog,
        summary: dict[str, Any],
        limit: int,
    ) -> None:
        chunk: deque[np.ndarray] = deque()
        rtts: list[float] = []
        server_ms: list[float] = []
        step_ms: list[float] = []
        requests = 0
        steps = 0
        terminated = False
        loop_started = time.monotonic()
        for tick in range(limit):
            recorder.record_observation(obs["data"], tick=tick)
            if video is not None:
                await asyncio.to_thread(video.write, composite_frame(obs["data"]))
            if bool(np.asarray(obs["terminated"]).reshape(-1)[0]):
                terminated = True
                break
            if not chunk:
                sent = time.monotonic()
                reply = await session.predict(obs["data"], prompt)
                rtt_ms = (time.monotonic() - sent) * 1000.0
                rows = reply.chunk
                chunk.extend(rows)
                recorder.record_inference(rows, tick=tick)
                rtts.append(rtt_ms)
                if reply.record.get("server_inference_ms") is not None:
                    server_ms.append(reply.record["server_inference_ms"])
                log.emit(
                    "inference", chunk_index=requests, step=tick, sdk_rtt_ms=rtt_ms, **reply.record
                )
                requests += 1
            action = chunk.popleft()
            stepped = time.monotonic()
            await robot.send_action(action)
            obs = await robot.get_observation()
            step_ms.append((time.monotonic() - stepped) * 1000.0)
            steps += 1
            if self.log_every and tick % self.log_every == 0:
                print(
                    f"[agent] {summary['task_name']} ep{summary['episode']} step {tick}", flush=True
                )
        else:
            # Record the observation that followed the last executed action.
            recorder.record_observation(obs["data"], tick=limit)
            if video is not None:
                await asyncio.to_thread(video.write, composite_frame(obs["data"]))
            terminated = bool(np.asarray(obs["terminated"]).reshape(-1)[0])
        summary.update(
            {
                "steps": steps,
                "terminated": terminated,
                "requests": requests,
                "loop_wall_s": time.monotonic() - loop_started,
                "sdk_rtt_ms": latency_stats(rtts),
                "server_inference_ms": latency_stats(server_ms),
                "sim_step_ms": latency_stats(step_ms),
            }
        )
