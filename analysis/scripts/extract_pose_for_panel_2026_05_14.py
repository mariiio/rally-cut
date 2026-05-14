"""Extract YOLO-Pose cache ONLY for our 12-video panel (Pose v2 probe, 2026-05-14).

Wraps `extract_pose_cache._process_video` but restricts to the GT rallies on the
12 panel videos used by the per-action-type learned-attribution experiment.

Usage:
    cd analysis
    uv run python scripts/extract_pose_for_panel_2026_05_14.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
from rich.console import Console

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from rallycut.evaluation.tracking.db import get_video_path  # noqa: E402
from rallycut.tracking.pose_attribution.pose_cache import (  # noqa: E402
    load_pose_cache,
    save_pose_cache,
)
from scripts.eval_action_detection import load_rallies_with_action_gt  # noqa: E402
from scripts.extract_pose_cache import _process_video  # noqa: E402

VIDEO_NAMES = [
    "titi", "toto", "lulu", "wawa", "caco", "cece",
    "cici", "cuco", "gaga", "kaka", "juju", "yeye",
]

console = Console()


def main() -> int:
    # We need video_name → video_id mapping
    from rallycut.evaluation.tracking.db import get_connection
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, name FROM videos WHERE name = ANY(%s)", [VIDEO_NAMES])
        panel_video_ids = {str(r[0]) for r in cur.fetchall()}
    console.print(f"Panel video IDs: {len(panel_video_ids)}")

    console.print("Loading rallies with action GT...")
    all_rallies = load_rallies_with_action_gt()
    panel_rallies = [r for r in all_rallies if str(r.video_id) in panel_video_ids]
    console.print(f"  Panel rallies: {len(panel_rallies)} / {len(all_rallies)} total")

    # Skip already cached
    missing = [r for r in panel_rallies if load_pose_cache(r.rally_id) is None]
    console.print(f"  Missing pose cache: {len(missing)}")

    if not missing:
        console.print("[green]All panel rallies already cached[/green]")
        return 0

    # Group by video
    by_video: dict[str, list] = defaultdict(list)
    for r in missing:
        by_video[r.video_id].append(r)

    console.print(f"  Videos to process: {len(by_video)}")
    total_contacts = sum(len(r.gt_labels) for r in missing)
    console.print(f"  Total GT contacts: {total_contacts}")

    # Load pose model
    console.print("[dim]Loading yolo11s-pose model...[/dim]")
    from ultralytics import YOLO
    pose_model = YOLO("yolo11s-pose.pt")

    processed_rallies = 0
    total_detections = 0
    t0 = time.time()

    for vid_idx, (video_id, video_rallies) in enumerate(by_video.items()):
        video_path = get_video_path(video_id)
        if video_path is None:
            console.print(f"  [yellow]Skipping video {video_id[:8]}: not found[/yellow]")
            continue

        n_contacts = sum(len(r.gt_labels) for r in video_rallies)
        console.print(
            f"[{vid_idx + 1}/{len(by_video)}] Video {video_id[:8]}: "
            f"{len(video_rallies)} rallies, {n_contacts} contacts",
        )

        vt0 = time.time()
        results = _process_video(
            video_path=str(video_path),
            video_rallies=video_rallies,
            pose_model=pose_model,
            window_half=10,
            imgsz=960,
            batch_size=8,
        )

        for rally in video_rallies:
            data = results.get(rally.rally_id)
            if data is None:
                data = {
                    "frames": np.array([], dtype=np.int32),
                    "track_ids": np.array([], dtype=np.int32),
                    "keypoints": np.zeros((0, 17, 3), dtype=np.float32),
                    "bboxes": np.zeros((0, 4), dtype=np.float32),
                }
            n_det = len(data["frames"])
            total_detections += n_det
            save_pose_cache(rally.rally_id, data)
            processed_rallies += 1

        vt = time.time() - vt0
        console.print(f"  -> cached in {vt:.1f}s ({total_detections} detections total)")

    elapsed = time.time() - t0
    console.print(f"\n[bold]Done.[/bold] {processed_rallies} rallies, {total_detections} detections")
    console.print(f"  Time: {elapsed:.0f}s ({elapsed / max(len(by_video), 1):.1f}s/video)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
