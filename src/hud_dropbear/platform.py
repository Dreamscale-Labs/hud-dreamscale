"""Confirm that HUD received the completed grades and robot camera traces."""

from hud.utils.platform import PlatformClient, canonical_record_id

from .contract import CAMERAS


async def verify_platform(summary, *, client=None):
    client = client or PlatformClient.from_settings()
    checked = []
    try:
        job_id = canonical_record_id(summary["job_id"])
        data = await client.aget(f"/jobs/{job_id}/traces", params={"limit": 100})
        items = data if isinstance(data, list) else data.get("items", [])
        traces = {row["id"]: row for row in items}
        for run in summary["runs"]:
            trace_id = canonical_record_id(run["trace_id"])
            trace = traces.get(trace_id, {})
            # The first page contains observations, inference and both camera streams.
            # Full local spans and the platform viewer retain all later segments.
            page = await client.aget(f"/trace/{trace_id}/events")
            events = page.get("events", [])
            cameras = sorted(
                {
                    event.get("camera")
                    for event in events
                    if event.get("kind") == "robot_video_segment" and event.get("url")
                }
            )
            kinds = {event.get("kind") for event in events}
            verified = (
                trace.get("status") == "completed"
                and not trace.get("error")
                and trace.get("reward") == run["reward"]
                and set(CAMERAS).issubset(cameras)
                and {"robot_observation", "robot_inference"}.issubset(kinds)
            )
            checked.append(
                {
                    "trace_id": trace_id,
                    "verified": verified,
                    "cameras": cameras,
                    "platform_status": trace.get("status"),
                    "platform_reward": trace.get("reward"),
                    "sample_has_more_events": page.get("has_more", False),
                }
            )
    except Exception as exc:
        return {"verified": False, "traces": checked, "error_type": type(exc).__name__}
    return {
        "verified": bool(checked) and all(row["verified"] for row in checked),
        "traces": checked,
    }
