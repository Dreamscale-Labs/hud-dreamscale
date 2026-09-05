from types import SimpleNamespace

import pytest
from test_lifecycle import FakePolicy

from hud_dropbear.agent import ready_target_artifact, serving_identity


@pytest.mark.parametrize(
    "worker_count,active_sessions,allowed", [(1, 1, True), (2, 1, False), (1, 2, False)]
)
async def test_cold_session_identity_requires_unambiguous_worker(
    worker_count, active_sessions, allowed
):
    policy = FakePolicy()
    config = policy.resolved_optimization_config
    fingerprint = config.tensorrt_artifact_fingerprint
    config.tensorrt_artifact_fingerprint = None
    config.tensorrt_artifact_id = "ad661627af0e95fd3151e342"
    closed = []

    class Client:
        def __init__(self, *args):
            pass

        async def get_session(self, session_id):
            assert session_id == policy.session_id
            return SimpleNamespace(
                status="ready", model=policy.model, region=policy.region, target_key="test-target"
            )

        async def status(self):
            return {
                policy.model: {
                    "targets": {
                        "test": {
                            "target_key": "test-target",
                            "ready": True,
                            "worker_count": worker_count,
                            "ready_worker_count": worker_count,
                            "active_sessions": active_sessions,
                            "worker_capabilities": {
                                "backends": ["tensorrt"],
                                "tensorrt_artifact_fingerprint": fingerprint,
                                "tensorrt_artifact": {
                                    "tensorrt_artifact_id": "b2405523e67019934dd81be1"
                                },
                            },
                        }
                    }
                }
            }

        async def close(self):
            closed.append(True)

    if allowed:
        artifact = await ready_target_artifact(policy, client_factory=Client)
        identity = serving_identity(policy, resolved_artifact=artifact)
        assert identity["artifact_identity_source"] == "single_ready_worker_status"
        assert identity["tensorrt_artifact_id"] == "b2405523e67019934dd81be1"
        assert identity["session_artifact_id"] == "ad661627af0e95fd3151e342"
    else:
        with pytest.raises(ValueError, match="exactly one"):
            await ready_target_artifact(policy, client_factory=Client)
    assert closed == [True]
