"""Reassemble locally recorded HUD CMAF segments into playable MP4 files."""

import base64
import json
from pathlib import Path

from .contract import CAMERAS


def export_videos(trace_directory: Path, destination: Path):
    videos = []
    for source in sorted(trace_directory.glob("*.jsonl")):
        segments = {}
        for line in source.read_text().splitlines():
            span = json.loads(line)
            payload = span.get("attributes", {}).get("hud.payload", {})
            if payload.get("source") != "video_segment" or payload.get("camera") not in CAMERAS:
                continue
            camera = payload["camera"]
            index = payload["index"]
            data = base64.b64decode(payload["segment"]["data"], validate=True)
            chunks = segments.setdefault(camera, {})
            if index in chunks and chunks[index] != data:
                raise ValueError("Conflicting HUD video segment indices")
            chunks[index] = data
        for camera, chunks in segments.items():
            indices = sorted(chunks)
            if len(indices) < 2 or indices != list(range(len(indices))):
                raise ValueError("Incomplete HUD camera video segment sequence")
            destination.mkdir(parents=True, exist_ok=True)
            target = destination / f"{source.stem}.{camera}.mp4"
            with target.open("xb") as output:
                for index in indices:
                    output.write(chunks[index])
            videos.append({"trace_id": source.stem, "camera": camera, "file": target.name})
    return videos
