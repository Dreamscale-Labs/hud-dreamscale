"""Agent side: SDK observation mapping, chunk handling, session lifecycle and evidence files."""

from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest
from conftest import JOINTS

import dreamscale_agent as da


def hud_obs(*, gripper: float = 0.3, terminated: bool = False, h: int = 36, w: int = 64) -> dict:
    data = {
        "exterior_1_left": np.full((h, w, 3), 10, np.uint8),
        "exterior_2_left": np.full((h, w, 3), 20, np.uint8),
        "wrist_left": np.full((h, w, 3), 30, np.uint8),
        "state": np.asarray([*JOINTS, gripper], np.float32),
    }
    return {"data": data, "terminated": terminated}


def test_split_observation_matches_dreamscale_robolab_client():
    inputs = da.split_observation(hud_obs(gripper=1.0000001)["data"])
    assert inputs["exterior_frame"][0, 0, 0] == 10  # over-shoulder left
    assert inputs["second_exterior_frame"][0, 0, 0] == 20  # over-shoulder right
    assert inputs["wrist_frame"][0, 0, 0] == 30
    assert inputs["joint_positions"] == pytest.approx(list(JOINTS))
    assert inputs["gripper"] == 1.0


def test_build_observation_uses_the_sdk_franka_surface():
    observation = da.build_observation(hud_obs(gripper=0.25)["data"])
    from dreamscale.franka import FrankaObservation

    assert isinstance(observation, FrankaObservation)
    assert observation.exterior_frame[0, 0, 0] == 10
    assert observation.second_exterior_frame[0, 0, 0] == 20
    assert observation.wrist_frame[0, 0, 0] == 30
    assert observation.joint_positions == pytest.approx(JOINTS)
    assert observation.gripper == pytest.approx(0.25)


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda d: d.pop("wrist_left"), "missing camera"),
        (lambda d: d.__setitem__("exterior_1_left", np.zeros((4, 4, 3), np.float32)), "uint8"),
        (lambda d: d.__setitem__("state", np.zeros(7, np.float32)), "8 finite"),
        (lambda d: d.__setitem__("state", np.full(8, np.nan, np.float32)), "8 finite"),
    ],
)
def test_split_observation_rejects_malformed(mutate, match):
    data = hud_obs()["data"]
    mutate(data)
    with pytest.raises((KeyError, ValueError), match=match):
        da.split_observation(data)


def test_postprocess_chunk_thresholds_gripper_and_validates():
    raw = np.zeros((32, 8))
    raw[:, -1] = np.linspace(0, 1, 32)
    chunk = da.postprocess_chunk(raw)
    assert chunk.dtype == np.float32
    assert set(np.unique(chunk[:, -1])) == {0.0, 1.0}
    assert chunk[15, -1] == 0.0 and chunk[16, -1] == 1.0  # > 0.5 closes
    with pytest.raises(ValueError, match="invalid action chunk"):
        da.postprocess_chunk(np.zeros((16, 8)))
    bad = np.zeros((32, 8))
    bad[3, 2] = np.inf
    with pytest.raises(ValueError, match="invalid action chunk"):
        da.postprocess_chunk(bad)


def test_hold_chunk_repeats_state():
    chunk = da.hold_chunk(hud_obs(gripper=0.8)["data"])
    assert chunk.shape == (32, 8)
    np.testing.assert_allclose(chunk[0], [*JOINTS, 0.8], rtol=1e-6)
    np.testing.assert_array_equal(chunk[0], chunk[-1])


class FakeSDKPolicy:
    def __init__(self, actions=None):
        self.model = "cosmos3-nano-policy-droid"
        self.session_id = "sess-1"
        self.region = "us-west-2"
        self.transport_mode = "quic"
        self.fallback_reason = None
        self.action_hz = 15
        self.chunk_size = 32
        self.resolved_optimization_config = SimpleNamespace(backend="pytorch", rtc="off")
        self.requests = []
        self.closed = 0
        self._actions = actions

    async def predict(self, observation, *, instruction, timeout_s):
        self.requests.append((observation, instruction, timeout_s))
        actions = self._actions
        if actions is None:
            actions = np.zeros((32, 8))
            actions[:, 0] = np.arange(32) + 100 * len(self.requests)
            actions[:, -1] = 0.7
        timing = da.dataclasses.make_dataclass(
            "Timing", [("obs_to_action_ms", float), ("server_inference_ms", float)]
        )(obs_to_action_ms=180.0, server_inference_ms=95.0)
        return SimpleNamespace(
            actions=tuple(map(tuple, actions)),
            timing=timing,
            observation_id=len(self.requests),
            chunk_id=len(self.requests),
            transport_mode="quic",
        )

    async def close(self):
        self.closed += 1


async def test_dreamscale_policy_opens_reference_session_shape():
    calls = []
    sdk = FakeSDKPolicy()

    async def connect(model, **kwargs):
        calls.append((model, kwargs))
        return sdk

    factory = da.DreamscalePolicy("flux-3-action-droid", keep_warm=600, connect=connect)
    session = await factory.open(label="BananaInBowlTask ep0")
    model, kwargs = calls[0]
    assert model == "flux-3-action-droid"
    assert {k: kwargs[k] for k in ("acceleration", "rtc", "calibration", "region")} == {
        "acceleration": "pytorch",
        "rtc": "off",
        "calibration": "off",
        "region": "us-west-2",
    }
    assert kwargs["control_hz"] == 15 and kwargs["keep_warm"] == 600
    assert callable(kwargs["on_progress"])
    assert session.info["session_id"] == "sess-1" and session.info["backend"] == "pytorch"

    reply = await session.predict(hud_obs()["data"], "Pick up the banana")
    observation, instruction, timeout_s = sdk.requests[0]
    assert instruction == "Pick up the banana" and timeout_s == 120.0
    assert observation.second_exterior_frame[0, 0, 0] == 20
    assert reply.chunk.shape == (32, 8) and set(reply.chunk[:, -1]) == {1.0}
    assert reply.record["server_inference_ms"] == 95.0
    assert reply.record["timing"]["obs_to_action_ms"] == 180.0
    await session.close()
    assert sdk.closed == 1


async def test_dreamscale_session_rejects_bad_chunk():
    async def connect(model, **kwargs):
        return FakeSDKPolicy(actions=np.zeros((16, 8)))

    session = await da.DreamscalePolicy("cosmos3-nano-policy-droid", connect=connect).open(
        label="x"
    )
    with pytest.raises(ValueError, match="invalid action chunk"):
        await session.predict(hud_obs()["data"], "Pick up the banana")


def test_dreamscale_policy_rejects_unknown_model_and_keep_warm():
    with pytest.raises(ValueError, match="model must be"):
        da.DreamscalePolicy("molmoact2-libero")
    with pytest.raises(ValueError, match="keep_warm"):
        da.DreamscalePolicy("cosmos3-nano-policy-droid", keep_warm=4000)


# ── the loop ─────────────────────────────────────────────────────────────────


class FakeRobot:
    """Serves observations; terminates after ``terminate_after`` actions."""

    instances: list[FakeRobot] = []

    def __init__(self, events, terminate_after):
        self.events = events
        self.terminate_after = terminate_after
        self.actions = []
        self.closed = False

    @classmethod
    def factory(cls, events, terminate_after):
        async def connect(cap, *, token=None):
            events.append(("robot_connect", token))
            robot = cls(events, terminate_after)
            cls.instances.append(robot)
            return robot

        return connect

    def spaces(self):
        return {"names": ["j1", "j2", "j3", "j4", "j5", "j6", "j7", "gripper"]}, {}

    def get_control_rate(self):
        return 15

    async def get_observation(self):
        return hud_obs(terminated=len(self.actions) >= self.terminate_after)

    async def send_action(self, action):
        self.actions.append(np.asarray(action))

    async def close(self):
        self.closed = True


class FakeRecorder:
    instances: list[FakeRecorder] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.observations = []
        self.inferences = []
        self.closed = False
        FakeRecorder.instances.append(self)

    def record_observation(self, data, *, tick):
        self.observations.append(tick)

    def record_inference(self, chunk, *, tick):
        self.inferences.append((tick, np.asarray(chunk).shape))

    def close(self):
        self.closed = True


class RecordingPolicy:
    model = "cosmos3-nano-policy-droid"

    def __init__(self, events):
        self.events = events
        self.sdk = FakeSDKPolicy()

    async def open(self, *, label):
        self.events.append(("session_open", label))
        return da.DreamscaleSession(self.sdk, predict_timeout_s=120.0)


def fake_run(max_steps=200):
    return SimpleNamespace(
        prompt="Pick up the banana and place it in the bowl",
        client=SimpleNamespace(binding=lambda ref: SimpleNamespace(name="robot")),
        bindings={
            "robot": {
                "token": "slot-0",
                "task_name": "BananaInBowlTask",
                "episode": 2,
                "max_steps": max_steps,
            }
        },
        trace_id="0123456789abcdef",
        trace=SimpleNamespace(status=None, content=None),
    )


@pytest.fixture
def patched_io(monkeypatch):
    events = []
    FakeRobot.instances.clear()
    FakeRecorder.instances.clear()

    def install(terminate_after):
        monkeypatch.setattr(
            da.RobotClient, "connect", staticmethod(FakeRobot.factory(events, terminate_after))
        )
        monkeypatch.setattr(da, "TraceRecorder", FakeRecorder)
        return events

    return install


async def test_agent_executes_full_chunks_open_loop_and_writes_evidence(tmp_path, patched_io):
    events = patched_io(terminate_after=40)
    policy = RecordingPolicy(events)
    agent = da.RobolabDreamscaleAgent(policy, output_dir=tmp_path, local_video=True)
    run = fake_run()
    await agent(run)

    # The session is open before the robot socket is dialed.
    assert [e[0] for e in events] == ["session_open", "robot_connect"]
    robot = FakeRobot.instances[0]
    assert len(robot.actions) == 40
    # Two requests: 32 actions open-loop, then a replan at tick 32.
    assert len(policy.sdk.requests) == 2
    assert [a[0] for a in robot.actions[:32]] == [100 + i for i in range(32)]
    assert robot.actions[32][0] == 200
    assert all(a[-1] == 1.0 for a in robot.actions)
    assert policy.sdk.closed == 1 and robot.closed
    recorder = FakeRecorder.instances[0]
    assert recorder.inferences == [(0, (32, 8)), (32, (32, 8))]
    assert recorder.observations[-1] == 40 and recorder.closed
    assert run.trace.status == "completed"

    (episode_dir,) = (tmp_path / "episodes").iterdir()
    assert episode_dir.name == "BananaInBowlTask__ep002__01234567"
    rows = [json.loads(line) for line in (episode_dir / "timing.jsonl").read_text().splitlines()]
    kinds = [row["event"] for row in rows]
    assert kinds[:3] == ["agent_started", "session_ready", "first_observation"]
    assert kinds.count("inference") == 2 and kinds[-1] == "episode_end"
    inference = [row for row in rows if row["event"] == "inference"]
    assert [row["chunk_index"] for row in inference] == [0, 1]
    assert [row["step"] for row in inference] == [0, 32]
    assert all(row["sdk_rtt_ms"] >= 0 and row["server_inference_ms"] == 95.0 for row in inference)
    summary = json.loads((episode_dir / "episode.json").read_text())
    assert summary["steps"] == 40 and summary["terminated"] is True
    assert summary["requests"] == 2
    assert summary["sdk_rtt_ms"]["n"] == 2
    assert summary["server_inference_ms"]["p50"] == 95.0
    assert summary["session"]["session_id"] == "sess-1"
    assert summary["observation_shapes"]["exterior_1_left"] == [36, 64, 3]
    assert summary["local_video_frames"] == 41
    assert (tmp_path / summary["local_video"]).stat().st_size > 0


async def test_agent_respects_step_cap_and_closes_on_error(tmp_path, patched_io):
    events = patched_io(terminate_after=10_000)
    policy = RecordingPolicy(events)
    agent = da.RobolabDreamscaleAgent(
        policy, output_dir=tmp_path, local_video=False, max_steps_cap=5
    )
    await agent(fake_run())
    assert len(FakeRobot.instances[0].actions) == 5
    summary = json.loads(next((tmp_path / "episodes").glob("*/episode.json")).read_text())
    assert summary["terminated"] is False and summary["max_steps"] == 5

    class Exploding(RecordingPolicy):
        async def open(self, *, label):
            session = await super().open(label=label)

            async def boom(data, instruction):
                raise RuntimeError("worker lost")

            session.predict = boom
            return session

    failing = Exploding(events)
    agent = da.RobolabDreamscaleAgent(failing, output_dir=tmp_path / "b", local_video=False)
    with pytest.raises(RuntimeError, match="worker lost"):
        await agent(fake_run())
    assert failing.sdk.closed == 1
    summary = json.loads(next((tmp_path / "b" / "episodes").glob("*/episode.json")).read_text())
    assert "worker lost" in summary["error"]


async def test_hold_policy_needs_no_sdk(tmp_path, patched_io):
    patched_io(terminate_after=3)
    agent = da.RobolabDreamscaleAgent(da.HoldPolicy(), output_dir=tmp_path, local_video=False)
    await agent(fake_run())
    robot = FakeRobot.instances[0]
    np.testing.assert_allclose(robot.actions[0][:7], JOINTS, rtol=1e-6)
    assert robot.actions[0][-1] == 0.0  # observed 0.3 -> open


def test_latency_stats():
    stats = da.latency_stats([10, 20, 30, 40, None, float("nan")])
    assert stats["n"] == 4 and stats["p50"] == 25.0 and stats["max"] == 40.0
    assert stats["p95"] == pytest.approx(38.5)
    assert da.latency_stats([]) == {"n": 0}
