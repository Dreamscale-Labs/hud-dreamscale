from types import SimpleNamespace

import pytest

from hud_dropbear.contract import CAMERAS
from hud_dropbear.platform import verify_platform


@pytest.mark.parametrize(
    "cameras,reward,expected", [(CAMERAS, 0, True), (CAMERAS[:1], 0, False), (CAMERAS, 1, False)]
)
async def test_platform_proof_requires_both_videos_and_matching_grade(cameras, reward, expected):
    trace_id = "00000000-0000-0000-0000-000000000001"

    async def get(path, **kwargs):
        if path.endswith("/traces"):
            return {"items": [{"id": trace_id, "status": "completed", "reward": reward}]}
        return {
            "events": [
                {"kind": "robot_observation"},
                {"kind": "robot_inference"},
                *[
                    {
                        "kind": "robot_video_segment",
                        "camera": camera,
                        "url": "https://example.test/video",
                    }
                    for camera in cameras
                ],
            ],
            "has_more": True,
        }

    result = await verify_platform(
        {"job_id": "0" * 32, "runs": [{"trace_id": trace_id, "reward": 0}]},
        client=SimpleNamespace(aget=get),
    )
    assert result["verified"] is expected
