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
        traces, offset, expected_total = {}, 0, None
        while True:
            data = await client.aget(
                f"/jobs/{job_id}/traces", params={"limit": 100, "offset": offset}
            )
            items = data if isinstance(data, list) else data.get("items")
            total = data.get("total") if isinstance(data, dict) else None
            more = data.get("has_more") if isinstance(data, dict) else None
            if not isinstance(items, list) or len(items) > 100:
                raise ValueError("Invalid HUD trace inventory page")
            if total is not None:
                if type(total) is not int or total < 0:
                    raise ValueError("Invalid HUD trace inventory total")
                if expected_total is not None and total != expected_total:
                    raise ValueError("HUD trace inventory changed during pagination")
                expected_total = total
            if more is not None and type(more) is not bool:
                raise ValueError("Invalid HUD trace pagination flag")
            for row in items:
                key = canonical_record_id(row["id"])
                if key in traces:
                    raise ValueError("HUD trace inventory repeated an identity")
                traces[key] = row
            offset += len(items)
            if expected_total is not None:
                if offset > expected_total or (more is False and offset < expected_total):
                    raise ValueError("HUD trace pagination contradicts its total")
                if offset == expected_total:
                    if more is True:
                        raise ValueError("HUD trace pagination contradicts its total")
                    break
            elif more is False or (more is None and len(items) < 100):
                break
            if not items:
                raise ValueError("HUD trace inventory stopped before declared completion")
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
