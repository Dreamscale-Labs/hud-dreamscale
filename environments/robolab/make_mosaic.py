"""Render a wave's episodes as one synchronized grid video (no labels).

Each tile is one episode's left exterior view (or wrist / all three views), aligned
at simulation step 0 and played at ``--speed`` x the 15 Hz control rate. When an
episode ends, its tile holds the last frame; successful episodes get a green tint
and a check mark at the step RoboLab scored success.

    uv run --project environments/robolab python environments/robolab/make_mosaic.py \
        runs/flux3-prod-w1 runs/flux3-prod-w2 --output runs/flux3-mosaic.mp4
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

VIEWS = {"left": 0, "wrist": 1, "right": 2}


def episodes(run_dirs: list[Path]) -> list[dict]:
    rows = []
    for run in run_dirs:
        summary = json.loads((run / "summary.json").read_text())
        for episode in summary["episodes"]:
            video = episode.get("local_video")
            if not video or not (run / video).is_file():
                continue
            rows.append({**episode, "video_path": run / video})
    # One tile per (task, episode): retries replace pre-launch failures.
    best: dict[tuple[str, int], dict] = {}
    for row in rows:
        key = (row["task_name"], row["episode"])
        if key not in best or (row.get("steps") or 0) > (best[key].get("steps") or 0):
            best[key] = row
    return sorted(best.values(), key=lambda e: (e["task_name"], e["episode"]))


def reader(path: Path, view: str, width: int, height: int):
    crop = "" if view == "all" else f"crop=iw/3:ih:{VIEWS[view]}*iw/3:0,"
    return subprocess.Popen(
        ["ffmpeg", "-v", "error", "-i", str(path), "-vf", f"{crop}scale={width}:{height}",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        stdout=subprocess.PIPE,
    )


def check_tile(tile: np.ndarray) -> np.ndarray:
    image = Image.fromarray(tile).convert("RGBA")
    h, w = tile.shape[:2]
    image = Image.alpha_composite(image, Image.new("RGBA", image.size, (16, 185, 95, 105)))
    draw = ImageDraw.Draw(image)
    draw.rectangle((2, 2, w - 3, h - 3), outline="#34e98b", width=4)
    r = min(w, h) // 6
    cx, cy = w // 2, h // 2
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill="#129852", outline="#a7f3d0", width=2)
    draw.line([(cx - r * 0.55, cy), (cx - r * 0.15, cy + r * 0.4), (cx + r * 0.6, cy - r * 0.4)],
              fill="white", width=max(4, r // 4), joint="curve")
    return np.array(image.convert("RGB"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--view", choices=("left", "wrist", "right", "all"), default="left")
    parser.add_argument("--columns", type=int, default=0)
    parser.add_argument("--tile-width", type=int, default=384)
    parser.add_argument("--speed", type=float, default=2.0)
    parser.add_argument("--hold-s", type=float, default=2.0)
    args = parser.parse_args()
    rows = episodes(args.runs)
    if not rows:
        raise SystemExit("no episodes with local video")
    columns = args.columns or math.ceil(math.sqrt(len(rows) * 16 / 9 / (16 / 9)))
    columns = min(columns, len(rows))
    grid_rows = math.ceil(len(rows) / columns)
    tw = args.tile_width - args.tile_width % 2
    th = int(tw * 9 / 16) if args.view != "all" else int(tw * 360 / 1920)
    th -= th % 2
    width, height = columns * tw, grid_rows * th
    fps = 15 * args.speed
    readers = [reader(r["video_path"], args.view, tw, th) for r in rows]
    frames = [np.zeros((th, tw, 3), np.uint8) for _ in rows]
    done = [False] * len(rows)
    marked = [None] * len(rows)
    writer = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s",
         f"{width}x{height}", "-r", str(fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
         "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         str(args.output)],
        stdin=subprocess.PIPE,
    )
    step = 0
    hold_frames = int(args.hold_s * fps)
    tail = 0
    while tail < hold_frames:
        canvas = np.zeros((height, width, 3), np.uint8)
        for i, row in enumerate(rows):
            if not done[i]:
                data = readers[i].stdout.read(tw * th * 3)
                if len(data) == tw * th * 3:
                    frames[i] = np.frombuffer(data, np.uint8).reshape(th, tw, 3)
                else:
                    done[i] = True
            tile = frames[i]
            if row.get("success") and (done[i] or step >= (row.get("steps") or 0)):
                if marked[i] is None:
                    marked[i] = check_tile(tile)
                tile = marked[i]
            y, x = divmod(i, columns)
            canvas[y * th:(y + 1) * th, x * tw:(x + 1) * tw] = tile
        writer.stdin.write(canvas.tobytes())
        step += 1
        if all(done):
            tail += 1
    writer.stdin.close()
    if writer.wait() != 0:
        raise SystemExit("ffmpeg failed")
    for proc in readers:
        proc.wait()
    report = {"output": str(args.output), "episodes": len(rows), "grid": [columns, grid_rows],
              "size": [width, height], "fps": fps, "frames": step,
              "successes": sum(bool(r.get("success")) for r in rows)}
    print(json.dumps(report))


if __name__ == "__main__":
    main()
