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
                        "index": index,
                        "camera": camera,
                        "url": "https://example.test/video",
                    }
                    for camera in cameras
                    for index in (0, 1)
                ],
            ],
            "has_more": True,
        }

    result = await verify_platform(
        {"job_id": "0" * 32, "runs": [{"trace_id": trace_id, "reward": 0}]},
        client=SimpleNamespace(aget=get),
    )
    assert result["verified"] is expected


async def test_traces_and_camera_events_beyond_first_page_are_verified():
    trace_id = "00000000-0000-0000-0000-000000000128"
    offsets, cursors = [], []

    async def get(path, *, params):
        if path.endswith("/traces"):
            offsets.append(params["offset"])
            if params["offset"] == 0:
                return {"items": [{"id": f"{i:032x}"} for i in range(100)], "total": 101}
            return {"items": [{"id": trace_id, "status": "completed", "reward": 1}], "total": 101}
        cursors.append(params["since_seq"])
        if params["since_seq"] == -1:
            return {"events": [{"kind": "robot_observation"}], "has_more": True, "next_seq": 20}
        return {
            "events": [
                {"kind": "robot_inference"},
                *[
                    {
                        "kind": "robot_video_segment",
                        "camera": c,
                        "index": index,
                        "url": "https://test/v",
                    }
                    for c in CAMERAS
                    for index in (0, 1)
                ],
            ],
            "has_more": False,
            "next_seq": 40,
        }

    result = await verify_platform(
        {
            "job_id": "0" * 32,
            "runs": [
                *[{"trace_id": f"{i:032x}", "reward": None} for i in range(100)],
                {"trace_id": trace_id, "reward": 1},
            ],
        },
        client=SimpleNamespace(aget=get),
    )
    assert result["traces"][-1]["verified"]
    assert offsets == [0, 100]
    assert cursors == [-1, 20] * 101


async def test_duplicate_trace_cannot_prove_two_episodes():
    async def get(*args, **kwargs):
        raise AssertionError("duplicate identities must fail before platform requests")

    result = await verify_platform(
        {"job_id": "0" * 32, "runs": [{"trace_id": "1" * 32, "reward": 1}] * 2},
        client=SimpleNamespace(aget=get),
    )
    assert not result["verified"]


async def test_media_without_initialization_cannot_prove_playable_video():
    trace_id = "1" * 32

    async def get(path, **kwargs):
        if path.endswith("/traces"):
            return {"items": [{"id": trace_id, "status": "completed", "reward": 1}]}
        return {
            "events": [
                {"kind": "robot_observation"},
                {"kind": "robot_inference"},
                *[
                    {
                        "kind": "robot_video_segment",
                        "camera": camera,
                        "index": 1,
                        "url": "https://example.test/media",
                    }
                    for camera in CAMERAS
                ],
            ],
            "has_more": False,
        }

    result = await verify_platform(
        {"job_id": "0" * 32, "runs": [{"trace_id": trace_id, "reward": 1}]},
        client=SimpleNamespace(aget=get),
    )
    assert not result["verified"]
