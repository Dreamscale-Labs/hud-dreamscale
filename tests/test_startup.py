import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace

import numpy as np
import pytest
from hud import Environment, LocalRuntime
from hud.capabilities.robot import _packb, _unpackb
from hud.environment.robot import RobotEndpoint

from env import create_environment
from environments.libero.bridge import LiberoBridge
from hud_dropbear.contract import (
    CAMERAS,
    LEGACY_PROFILE,
    POOLED_ENV_NAME,
    POOLED_PROFILE,
    build_contract,
    pooled_state_from_raw,
)
from hud_dropbear.profile_startup import (
    profile_startup,
    validate_episode_evidence,
    validate_observation,
)
from hud_dropbear.startup import StartupProfile
from hud_dropbear.startup_cleanup import (
    ExclusiveRegistryCampaign,
    ExclusiveRegistryRuntime,
    instance_inventory,
    verify_owned_instances,
)


def raw_observation():
    agent = np.zeros((360, 360, 3), dtype=np.uint8)
    agent[:180, :180, 0] = 255
    wrist = np.zeros_like(agent)
    wrist[:180, :180, 1] = 255
    return {
        CAMERAS[0]: agent,
        CAMERAS[1]: wrist,
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "robot0_eef_quat": np.array([0, 0, 1, 0], dtype=np.float32),
        "robot0_gripper_qpos": np.array([0.01, -0.01], dtype=np.float32),
    }


class DiagnosticBridge(LiberoBridge):
    def __init__(self):
        super().__init__(profile=POOLED_PROFILE)
        self.actions = []

    def _reset(self, suite_name, task_id, init_state_id, seed, max_steps):
        with self.reset_profile.span("fake_reset"):
            self._obs = raw_observation()
            self.steps = 0
            self.success = self.terminated = False
            self.total_reward = 0.0
        return "diagnostic"

    def step(self, action):
        self.actions.append(action)
        raise AssertionError("No policy actions are allowed")


@asynccontextmanager
async def wire_environment():
    bridge = DiagnosticBridge()
    await bridge.start()
    server = await bridge.serve_control("127.0.0.1", 0)
    endpoint = RobotEndpoint.remote("127.0.0.1", server.sockets[0].getsockname()[1])
    env = create_environment(
        endpoint, profile=POOLED_PROFILE, environment=Environment(name=POOLED_ENV_NAME)
    )
    try:
        yield env, bridge
    finally:
        await endpoint.stop()
        server.close()
        await server.wait_closed()
        await bridge.stop()


def test_native_profile_is_lossless_and_legacy_unchanged():
    bridge = DiagnosticBridge()
    bridge.reset(startup_id="identity")
    data, _ = bridge.get_observation()
    decoded = _unpackb(_packb({key: value[0] for key, value in data.items()}))
    raw = raw_observation()
    for camera in CAMERAS:
        np.testing.assert_array_equal(decoded[camera], raw[camera])
    np.testing.assert_array_equal(decoded["robot0_eef_quat_xyzw"], raw["robot0_eef_quat"])
    assert "state" not in decoded
    assert build_contract(profile=POOLED_PROFILE)["observation_profile"] == POOLED_PROFILE
    legacy = build_contract(profile=LEGACY_PROFILE)
    assert "observation_profile" not in legacy
    assert legacy["features"]["state"]["shape"] == [8]
    assert legacy["features"][CAMERAS[0]]["shape"] == [256, 256, 3]


@pytest.mark.parametrize(
    "field,value",
    [
        (CAMERAS[0], np.zeros((256, 256, 3), dtype=np.uint8)),
        ("robot0_eef_pos", np.array([np.nan, 0, 0], dtype=np.float32)),
        ("robot0_eef_quat_xyzw", np.zeros(4, dtype=np.float32)),
    ],
)
def test_startup_observation_contract_rejects_bad_payload(field, value):
    raw = raw_observation()
    data = {key: raw[key] for key in CAMERAS} | pooled_state_from_raw(raw)
    data[field] = value
    with pytest.raises(ValueError):
        validate_observation({"data": data, "terminated": False}, POOLED_PROFILE)


async def test_actual_hud_wire_zero_actions_and_repeated_reset_identity():
    async def verified(_):
        return {"verified": True}

    async with wire_environment() as (env, bridge):
        report = await profile_startup(
            LocalRuntime(env),
            profile=POOLED_PROFILE,
            trials=1,
            warm_resets=3,
            verify_cleanup=verified,
        )
        assert report["completed"], report
        episodes = report["trials"][0]["episodes"]
        assert len(episodes) == 4
        assert len({row["episode_id"] for row in episodes}) == 4
        assert [row["environment"]["reset_ordinal"] for row in episodes] == [1, 2, 3, 4]
        for episode in episodes:
            diagnostic = episode["environment"]["startup_profile"]
            assert diagnostic["bridge_reset"]["episode_id"] == episode["episode_id"]
            assert diagnostic["environment_episode"]["episode_id"] == episode["episode_id"]
            assert len(diagnostic["bridge_reset"]["events"]) == 2
            assert episode["diagnostic_score"] == 0
        assert report["summary"]["initial_setup_s"]["n"] == 1
        assert report["summary"]["warm_setup_s"]["n"] == 3
        assert bridge.actions == []
        assert bridge._registry.all_free


async def test_later_warm_failure_retains_valid_cold_sample(monkeypatch):
    async def verified(_):
        return {"verified": True}

    async with wire_environment() as (env, bridge):
        original = bridge._reset

        def fail_second_reset(*args):
            if bridge.reset_ordinal == 2:
                raise RuntimeError("simulated warm reset failure")
            return original(*args)

        monkeypatch.setattr(bridge, "_reset", fail_second_reset)
        report = await profile_startup(
            LocalRuntime(env),
            profile=POOLED_PROFILE,
            trials=1,
            warm_resets=3,
            verify_cleanup=verified,
        )
        assert not report["completed"]
        assert report["trials"][0]["failure_phase"] == "warm_reset"
        for key in (
            "runtime_request_to_first_observation_s",
            "runtime_request_to_task_setup_ready_s",
            "runtime_acquire_s",
            "control_connect_ready_s",
            "initial_setup_s",
            "claim_first_observation_s",
        ):
            assert report["summary"][key]["n"] == 1
        assert "warm_setup_s" not in report["summary"]
        assert "warm_ready_s" not in report["summary"]
        assert report["attempts"]["warm_resets"] == 1
        assert report["attempts"]["validated_cold_observations"] == 1
        assert report["attempts"]["validated_warm_observations"] == 0
        assert report["attempts"]["failures"][0]["phase"] == "warm_reset"
        assert bridge.actions == [] and bridge._registry.all_free


async def test_cancellation_during_setup_cancels_task_and_closes_runtime():
    entered, cancelled, closed, checked = (asyncio.Event() for _ in range(4))

    class Client:
        async def start_task(self, *args):
            entered.set()
            await asyncio.Event().wait()

        async def cancel(self):
            cancelled.set()

    @asynccontextmanager
    async def connect(*args, **kwargs):
        yield Client()

    @asynccontextmanager
    async def provider(task):
        try:
            yield SimpleNamespace(params={"session_id": "owned"})
        finally:
            closed.set()

    async def verified(runtime):
        checked.set()
        return {"verified": True}

    operation = asyncio.create_task(
        profile_startup(
            provider,
            trials=1,
            connect=connect,
            verify_cleanup=verified,
        )
    )
    await asyncio.wait_for(entered.wait(), 3)
    operation.cancel()
    with pytest.raises(asyncio.CancelledError):
        await operation
    assert cancelled.is_set() and closed.is_set() and checked.is_set()


async def test_startup_timeout_preserves_partial_evidence_and_finishes_cleanup():
    cancelled = []

    class Client:
        async def start_task(self, *args):
            await asyncio.Event().wait()

        async def cancel(self):
            cancelled.append(True)

    @asynccontextmanager
    async def connect(*args, **kwargs):
        yield Client()

    @asynccontextmanager
    async def provider(task):
        yield SimpleNamespace(params={"session_id": "owned"})

    async def verified(runtime):
        return {"verified": True}

    report = await profile_startup(
        provider,
        trials=5,
        connect=connect,
        verify_cleanup=verified,
        startup_timeout=0.01,
    )
    assert not report["completed"] and report["stopped_early"]
    assert len(report["trials"]) == 1
    assert report["trials"][0]["error_type"] == "TimeoutError"
    assert report["trials"][0]["episodes"][0]["timing"]["events"]
    assert cancelled == [True]
    assert "initial_setup_s" not in report["summary"]
    assert "runtime_request_to_first_observation_s" not in report["summary"]
    assert report["summary"]["runtime_acquire_s"]["n"] == 1


def test_diagnostic_evidence_rejects_wrong_episode_and_dropped_events():
    bridge = DiagnosticBridge()
    bridge.reset(startup_id="right")
    info = bridge.result()["info"]
    info["startup_profile"].update(
        {
            "environment_boot": StartupProfile("environment_boot").snapshot(),
            "environment_episode": StartupProfile(
                "environment_episode", episode_id="right"
            ).snapshot(),
        }
    )
    validate_episode_evidence(info, "right", POOLED_PROFILE)
    with pytest.raises(ValueError, match="another episode"):
        validate_episode_evidence(info, "wrong", POOLED_PROFILE)
    info["startup_profile"]["bridge_reset"]["dropped_events"] = 1
    with pytest.raises(ValueError, match="incomplete"):
        validate_episode_evidence(info, "right", POOLED_PROFILE)


def test_startup_diagnostics_bounded_and_only_stderr(capsys):
    profile = StartupProfile("test", max_events=1)
    with profile.span("first"):
        pass
    with profile.span("second"):
        pass
    snapshot = profile.snapshot()
    assert not snapshot["complete"] and snapshot["dropped_events"] == 1
    assert len(snapshot["events"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert json.loads(captured.err)["hud_startup"]["name"] == "first"


async def test_cleanup_stops_only_single_owned_instance():
    class Platform:
        def __init__(self):
            self.stopped = []

        async def aget(self, path, *, params):
            return {
                "instances": [
                    {"id": "old", "registry_id": "registry", "status": "terminated"},
                    {
                        "id": "owned",
                        "registry_id": "registry",
                        "build_id": "build",
                        "status": "terminated" if self.stopped else "running",
                    },
                ],
                "has_more": False,
            }

        async def apost(self, path, *, json):
            self.stopped += json["instance_ids"]
            return {"stopped": 1}

    platform = Platform()
    receipt = await verify_owned_instances(platform, "registry", {"old"}, build_id="build")
    assert receipt["verified"] and platform.stopped == ["owned"]
    platform = Platform()
    receipt = await verify_owned_instances(platform, "registry", set())
    assert not receipt["verified"] and platform.stopped == []


async def test_pooled_campaign_cleans_partial_start_without_touching_existing_history():
    class Platform:
        rows = {"old": {"id": "old", "registry_id": "registry", "status": "terminated"}}
        stopped = []

        async def aget(self, path, *, params=None):
            if path == "/registry/registry":
                return {"id": "registry", "name": "environment", "latest_build_id": "build"}
            return {"instances": list(self.rows.values()), "has_more": False}

        async def apost(self, path, *, json):
            self.stopped += json["instance_ids"]
            for identity in json["instance_ids"]:
                self.rows[identity]["status"] = "terminated"
            return {"stopped": len(json["instance_ids"])}

    platform = Platform()
    guard = ExclusiveRegistryCampaign(
        platform, "registry", environment_name="environment", expected_instances=8, build_id="build"
    )
    async with guard:
        for index in range(3):
            platform.rows[str(index)] = {
                "id": str(index),
                "registry_id": "registry",
                "status": "running",
                "build_id": "build",
            }
        with pytest.raises(RuntimeError, match="every expected"):
            await guard.validate_ready()
    assert guard.cleanup_receipt["verified"]
    assert platform.stopped == ["0", "1", "2"]


@pytest.mark.parametrize(
    "changed",
    [{"name": "different-environment"}, {"id": "different-registry"}, {"latest_build_id": "old"}],
)
@pytest.mark.parametrize("mode", ["campaign", "single-runtime"])
async def test_registry_selection_rejected_before_inventory_or_allocation(changed, mode):
    paths = []

    class Platform:
        async def aget(self, path, **kwargs):
            paths.append(path)
            assert path == "/registry/registry"
            return {
                "id": "registry",
                "name": "environment",
                "latest_build_id": "build",
                **changed,
            }

    @asynccontextmanager
    async def provider(task):
        pytest.fail("A mismatched registry must be rejected before runtime allocation")
        yield

    if mode == "campaign":
        scope = ExclusiveRegistryCampaign(
            Platform(),
            "registry",
            environment_name="environment",
            expected_instances=8,
            build_id="build",
        )
    else:
        scope = ExclusiveRegistryRuntime(provider, Platform(), "registry", build_id="build")(
            SimpleNamespace(env="environment")
        )
    with pytest.raises(RuntimeError, match="Dedicated registry"):
        async with scope:
            pytest.fail("The paid-resource scope must not be entered")
    assert paths == ["/registry/registry"]


def test_pooled_campaign_width_bounds():
    for width in (1, 8, 32, 64):
        ExclusiveRegistryCampaign(
            None, "registry", environment_name="environment", expected_instances=width
        )
    for width in (0, 65, True):
        with pytest.raises(ValueError):
            ExclusiveRegistryCampaign(
                None, "registry", environment_name="environment", expected_instances=width
            )


async def test_no_instance_evidence_preserves_original_provider_failure():
    class Platform:
        async def aget(self, path, *, params=None):
            if path == "/registry/registry":
                return {"id": "registry", "name": "environment"}
            return {"instances": [], "has_more": False}

    guard = ExclusiveRegistryCampaign(
        Platform(), "registry", environment_name="environment", expected_instances=1
    )
    with pytest.raises(ConnectionError, match="creation failed"):
        async with guard:
            raise ConnectionError("creation failed")
    assert not guard.cleanup_receipt["verified"]
    assert guard.cleanup_receipt["reason"] == "no_created_instance_evidence"


async def test_wrong_build_rejects_readiness_but_still_cleans_owned_instance():
    class Platform:
        rows = []
        stopped = []

        async def aget(self, path, *, params=None):
            if path == "/registry/registry":
                return {"id": "registry", "name": "environment", "latest_build_id": "pinned"}
            return {"instances": self.rows, "has_more": False}

        async def apost(self, path, *, json):
            self.stopped += json["instance_ids"]
            self.rows[0]["status"] = "terminated"
            return {"stopped": 1}

    platform = Platform()
    guard = ExclusiveRegistryCampaign(
        platform,
        "registry",
        environment_name="environment",
        expected_instances=1,
        build_id="pinned",
    )
    with pytest.raises(RuntimeError, match="build differs"):
        async with guard:
            platform.rows = [
                {"id": "owned", "registry_id": "registry", "build_id": "wrong", "status": "running"}
            ]
            await guard.validate_ready()
    assert platform.stopped == ["owned"] and guard.cleanup_receipt["verified"]


async def test_inventory_change_still_stops_previously_verified_owned_instances():
    class Platform:
        rows = []
        stopped = []

        async def aget(self, path, *, params=None):
            if path == "/registry/registry":
                return {"id": "registry", "name": "environment"}
            return {"instances": self.rows, "has_more": False}

        async def apost(self, path, *, json):
            self.stopped += json["instance_ids"]
            for row in self.rows:
                if row["id"] in json["instance_ids"]:
                    row["status"] = "terminated"
            return {"stopped": len(json["instance_ids"])}

    platform = Platform()
    guard = ExclusiveRegistryCampaign(
        platform, "registry", environment_name="environment", expected_instances=1
    )
    with pytest.raises(RuntimeError, match="owned_instance_inventory_changed"):
        async with guard:
            platform.rows = [{"id": "owned", "registry_id": "registry", "status": "running"}]
            await guard.validate_ready()
            platform.rows.append({"id": "unowned", "registry_id": "registry", "status": "running"})
    assert platform.stopped == ["owned"]
    assert platform.rows[1]["status"] == "running"
    assert guard.cleanup_receipt["termination_confirmed"]
    assert not guard.cleanup_receipt["verified"]
    assert guard.cleanup_receipt["unexpected_instance_ids"] == ["unowned"]


async def test_repeated_cancellation_cannot_interrupt_owned_instance_stop():
    stop_started, allow_stop = asyncio.Event(), asyncio.Event()

    class Platform:
        rows = []

        async def aget(self, path, *, params=None):
            if path == "/registry/registry":
                return {"id": "registry", "name": "environment"}
            return {"instances": self.rows, "has_more": False}

        async def apost(self, path, *, json):
            assert json["instance_ids"] == ["owned"]
            stop_started.set()
            await allow_stop.wait()
            self.rows[0]["status"] = "terminated"
            return {"stopped": 1}

    platform = Platform()
    guard = ExclusiveRegistryCampaign(
        platform, "registry", environment_name="environment", expected_instances=1
    )
    await guard.__aenter__()
    platform.rows = [{"id": "owned", "registry_id": "registry", "status": "running"}]
    await guard.validate_ready()
    closing = asyncio.create_task(guard.__aexit__(None, None, None))
    await asyncio.wait_for(stop_started.wait(), 1)
    try:
        closing.cancel()
        await asyncio.sleep(0)
        closing.cancel()
        await asyncio.sleep(0)
        allow_stop.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(closing, 1)
        assert platform.rows[0]["status"] == "terminated"
        assert guard.cleanup_receipt["verified"]
    finally:
        allow_stop.set()
        if not closing.done():
            await asyncio.gather(closing, return_exceptions=True)


async def test_inventory_pagination_reads_all_and_rejects_shifted_pages():
    class Platform:
        offsets = []
        duplicate = False

        async def aget(self, path, *, params):
            offset = params["offset"]
            self.offsets.append(offset)
            rows = [{"id": str(i), "registry_id": "registry"} for i in range(201)]
            if self.duplicate and offset:
                return {"instances": [rows[0]], "has_more": False}
            return {"instances": rows[offset : offset + 200], "has_more": offset == 0}

    platform = Platform()
    assert len(await instance_inventory(platform, "registry")) == 201
    assert platform.offsets == [0, 200]
    platform.duplicate = True
    with pytest.raises(RuntimeError, match="changed during pagination"):
        await instance_inventory(platform, "registry")


@pytest.mark.parametrize("transport_error", [False, True])
async def test_pinned_hud_delete_swallows_failure_reproduction(monkeypatch, transport_error):
    """Current upstream bug: context exit alone cannot establish successful deletion."""
    from hud.eval.runtime import hud as runtime_module

    seen = []

    class Http:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def delete(self, *args, **kwargs):
            seen.append("delete")
            if transport_error:
                raise ConnectionError("simulated transport failure")
            return SimpleNamespace(status_code=500, raise_for_status=lambda: seen.append("checked"))

    monkeypatch.setattr(runtime_module.httpx, "AsyncClient", Http)
    await runtime_module.HUDRuntime()._delete_runtime_session(
        "https://unused.invalid", "fake-test-key", "owned-test-session"
    )
    assert seen == ["delete"]  # No error surfaces and status is never checked.


def test_pooled_build_context_contains_only_runtime_sources(tmp_path):
    from hud.cli.utils.source import EnvironmentSource

    from scripts.prepare_pooled_build import prepare

    manifest = prepare(tmp_path)
    assert '"pooled_env.py"' in (tmp_path / "Dockerfile.hud").read_text()
    assert (tmp_path / "environments/libero/uv.lock").is_file()
    assert (tmp_path / "src/hud_dropbear/startup.py").is_file()
    assert not (tmp_path / ".git").exists()
    assert not (tmp_path / "tests").exists()
    assert not (tmp_path / "docs").exists()
    assert manifest["entrypoint"] == "pooled_env.py"
    source = EnvironmentSource.open(tmp_path)
    assert source.is_environment
    assert source.served_environment_module() == "pooled_env.py"
    assert source.served_environment_name() == POOLED_ENV_NAME
    assert source.validate() == []


class _RetryPlatform:
    """Registry with an idle baseline; rows are set by the test after entry."""

    def __init__(self):
        self.rows = {}
        self.stopped = []

    async def aget(self, path, *, params=None):
        if path == "/registry/registry":
            return {"id": "registry", "name": "environment", "latest_build_id": "build"}
        return {"instances": list(self.rows.values()), "has_more": False}

    async def apost(self, path, *, json):
        self.stopped += json["instance_ids"]
        for identity in json["instance_ids"]:
            self.rows[identity]["status"] = "terminated"
        return {"stopped": len(json["instance_ids"])}

    def add(self, identity, status="running", build_id="build"):
        self.rows[identity] = {
            "id": identity,
            "registry_id": "registry",
            "status": status,
            "build_id": build_id,
        }


async def test_released_retry_instance_is_owned_but_not_counted_as_live():
    platform = _RetryPlatform()
    guard = ExclusiveRegistryCampaign(
        platform,
        "registry",
        environment_name="environment",
        expected_instances=2,
        build_id="build",
        retry_allowance=2,
    )
    async with guard:
        # A lane readiness retry released its first simulator before leasing another.
        platform.add("retired", status="terminated")
        platform.add("a")
        platform.add("b")
        receipt = await guard.validate_ready()
        assert receipt["instance_ids"] == ["a", "b"]
        assert receipt["retired_instance_ids"] == ["retired"]
    assert guard.cleanup_receipt["verified"]
    assert platform.stopped == ["a", "b"]
    assert guard.cleanup_receipt["instance_ids"] == ["a", "b", "retired"]


async def test_live_created_instances_beyond_expected_are_rejected_despite_allowance():
    platform = _RetryPlatform()
    guard = ExclusiveRegistryCampaign(
        platform,
        "registry",
        environment_name="environment",
        expected_instances=2,
        build_id="build",
        retry_allowance=2,
    )
    with pytest.raises(RuntimeError, match="Live created instance count exceeds"):
        async with guard:
            for identity in ("a", "b", "c"):
                platform.add(identity)
            await guard.validate_ready()
    assert guard.cleanup_receipt["verified"]
    assert sorted(platform.stopped) == ["a", "b", "c"]


async def test_created_instances_beyond_retry_allowance_are_rejected():
    platform = _RetryPlatform()
    guard = ExclusiveRegistryCampaign(
        platform,
        "registry",
        environment_name="environment",
        expected_instances=1,
        build_id="build",
        retry_allowance=1,
    )
    with pytest.raises(RuntimeError, match="Created instance count exceeds"):
        async with guard:
            platform.add("r1", status="terminated")
            platform.add("r2", status="terminated")
            platform.add("a")
            await guard.validate_ready()
    # Ownership beyond the bound is ambiguous: refuse exact-ID cleanup, as before.
    assert platform.stopped == []
    assert guard.cleanup_receipt == {"verified": False, "reason": "RuntimeError"}


async def test_default_retry_allowance_keeps_exact_ownership():
    platform = _RetryPlatform()
    guard = ExclusiveRegistryCampaign(
        platform, "registry", environment_name="environment", expected_instances=1
    )
    with pytest.raises(RuntimeError, match="Created instance count exceeds"):
        async with guard:
            platform.add("retired", status="terminated")
            platform.add("a")
            await guard.validate_ready()
    assert platform.stopped == []
    assert guard.cleanup_receipt == {"verified": False, "reason": "RuntimeError"}


def test_retry_allowance_bounds():
    for allowance in (0, 1, 128):
        ExclusiveRegistryCampaign(
            None,
            "registry",
            environment_name="environment",
            expected_instances=64,
            retry_allowance=allowance,
        )
    for allowance in (-1, True, 1.5, None):
        with pytest.raises(ValueError):
            ExclusiveRegistryCampaign(
                None,
                "registry",
                environment_name="environment",
                expected_instances=64,
                retry_allowance=allowance,
            )
