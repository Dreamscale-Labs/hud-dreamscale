"""Confirm that HUD received the completed grades and robot camera traces."""

from hud.utils.platform import PlatformClient, canonical_record_id

from .contract import CAMERAS


async def verify_platform(summary, *, client=None):
    client = client or PlatformClient.from_settings()
    checked = []
    try:
        job_id = canonical_record_id(summary["job_id"])
        requested = [canonical_record_id(run["trace_id"]) for run in summary["runs"]]
        wanted = set(requested)
        if len(wanted) != len(requested):
            raise ValueError("Each episode must have a distinct HUD trace")
        traces, offset = {}, 0
        while True:
            data = await client.aget(
                f"/jobs/{job_id}/traces", params={"limit": 100, "offset": offset}
            )
            items = data if isinstance(data, list) else data.get("items", [])
            before = len(traces)
            traces.update({canonical_record_id(row["id"]): row for row in items})
            offset += len(items)
            if not items or len(traces) == before:
                break
            if isinstance(data, dict) and offset >= data.get("total", float("inf")):
                break
            if len(items) < 100:
                break
            if offset >= 10000:
                raise ValueError("Job trace inventory exceeds the verification bound")
        if set(traces) != wanted:
            raise ValueError("HUD job trace membership differs from the frozen cohort")
        for run in summary["runs"]:
            trace_id = canonical_record_id(run["trace_id"])
            trace = traces.get(trace_id, {})
            cameras, initialized, kinds, cursor = set(), set(), set(), -1
            for _ in range(100):
                page = await client.aget(
                    f"/trace/{trace_id}/events", params={"since_seq": cursor, "limit": 200}
                )
                events = page.get("events", [])
                cameras.update(
                    event.get("camera")
                    for event in events
                    if event.get("kind") == "robot_video_segment"
                    and event.get("url")
                    and event.get("index", 0) > 0
                )
                initialized.update(
                    event.get("camera")
                    for event in events
                    if event.get("kind") == "robot_video_segment"
                    and event.get("url")
                    and event.get("index") == 0
                )
                kinds.update(event.get("kind") for event in events)
                if set(CAMERAS).issubset(cameras & initialized) and {
                    "robot_observation",
                    "robot_inference",
                }.issubset(kinds):
                    break
                next_cursor = page.get("next_seq", cursor)
                if not page.get("has_more") or next_cursor <= cursor:
                    break
                cursor = next_cursor
            verified = (
                trace.get("status") == "completed"
                and not trace.get("error")
                and trace.get("reward") == run["reward"]
                and set(CAMERAS).issubset(cameras & initialized)
                and {"robot_observation", "robot_inference"}.issubset(kinds)
            )
            checked.append(
                {
                    "trace_id": trace_id,
                    "verified": verified,
                    "cameras": sorted(cameras),
                    "initialized_cameras": sorted(initialized),
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
