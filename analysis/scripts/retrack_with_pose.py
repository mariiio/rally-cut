"""Re-track rallies with yolo11s-pose and update DB positions_json.

Re-runs player tracking with yolo11s-pose to get native keypoints in
positions_json. Updates the DB directly for the specified rallies.

Usage:
    cd analysis
    uv run python scripts/retrack_with_pose.py --video 44e89f6c 601d4a69 0a383519
    uv run python scripts/retrack_with_pose.py --video 44e89f6c --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time

from rich.console import Console

from rallycut.evaluation.db import get_connection
from rallycut.evaluation.tracking.db import get_video_path
from rallycut.tracking.player_tracker import PlayerTracker
from scripts.eval_action_detection import RallyData, load_rallies_with_action_gt

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-track rallies with yolo11s-pose")
    parser.add_argument("--video", nargs="+", required=True,
                        help="Video ID prefixes to re-track")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be updated without writing to DB")
    args = parser.parse_args()

    rallies = load_rallies_with_action_gt()
    console.print(f"Loaded {len(rallies)} rallies")

    target = [r for r in rallies
              if any(r.video_id.startswith(v) for v in args.video)]
    console.print(f"Target: {len(target)} rallies from {len(args.video)} video prefix(es)")

    if not target:
        console.print("[red]No matching rallies[/red]")
        sys.exit(1)

    by_video: dict[str, list[RallyData]] = {}
    for r in target:
        by_video.setdefault(r.video_id, []).append(r)

    tracker = PlayerTracker()
    console.print(f"Tracker model: {tracker.yolo_model}")

    total_updated = 0
    t0 = time.time()

    for vid_idx, (video_id, video_rallies) in enumerate(by_video.items()):
        video_path = get_video_path(video_id)
        if video_path is None:
            console.print(f"  [yellow]Video {video_id[:8]} not found[/yellow]")
            continue

        console.print(
            f"\n[{vid_idx+1}/{len(by_video)}] Video {video_id[:8]}: "
            f"{len(video_rallies)} rallies"
        )

        for rally in video_rallies:
            rt0 = time.time()

            # Compute end_ms from start_ms + frame_count
            end_ms = rally.start_ms + int((rally.frame_count or 300) / rally.fps * 1000)

            result = tracker.track_video(
                video_path=str(video_path),
                start_ms=rally.start_ms,
                end_ms=end_ms,
            )

            positions = [p.to_dict() for p in result.positions]
            n_with_kps = sum(1 for p in positions if p.get("keypoints"))
            n_total = len(positions)
            elapsed = time.time() - rt0

            console.print(
                f"  {rally.rally_id[:8]}: {n_total} positions, "
                f"{n_with_kps} with keypoints ({elapsed:.1f}s)"
            )

            if args.dry_run:
                continue

            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE player_tracks SET positions_json = %s WHERE rally_id = %s",
                        [json.dumps(positions), rally.rally_id],
                    )
                conn.commit()

            total_updated += 1

    elapsed = time.time() - t0
    console.print(f"\n[bold]Done.[/bold] {total_updated} rallies updated in {elapsed:.0f}s")

    # Verify keypoints are stored
    if total_updated > 0:
        r0 = target[0]
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT positions_json FROM player_tracks WHERE rally_id = %s",
                    [r0.rally_id],
                )
                row = cur.fetchone()
                if row and row[0]:
                    pos = row[0]
                    has_kps = sum(1 for p in pos if "keypoints" in p)
                    console.print(f"\nVerification ({r0.rally_id[:8]}): {has_kps}/{len(pos)} have keypoints")


if __name__ == "__main__":
    main()
