"""Backfill ball_positions_json for rallies that have player tracking but no ball data.

Runs WASB ball tracking on each rally and saves the results to the database
without overwriting existing player tracking data.

Usage:
    cd analysis
    uv run python scripts/backfill_ball_tracking.py
    uv run python scripts/backfill_ball_tracking.py --dry-run  # preview only
"""

from __future__ import annotations

import argparse
import json
import logging
import time

from rallycut.evaluation.db import get_connection
from rallycut.evaluation.tracking.db import get_video_path
from rallycut.tracking import create_ball_tracker
from rallycut.tracking.ball_tracker import BallPosition

logger = logging.getLogger(__name__)


def find_rallies_missing_ball_data() -> list[dict]:
    """Find rallies with action GT but no ball_positions_json."""
    query = """
        SELECT r.id, r.video_id, r.start_ms, r.end_ms,
               pt.frame_count, pt.fps
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE pt.action_ground_truth_json IS NOT NULL
          AND pt.ball_positions_json IS NULL
        ORDER BY r.video_id, r.start_ms
    """
    rallies = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            for row in cur.fetchall():
                rallies.append({
                    "rally_id": row[0],
                    "video_id": row[1],
                    "start_ms": row[2],
                    "end_ms": row[3],
                    "frame_count": row[4],
                    "fps": row[5] or 30.0,
                })
    return rallies


def save_ball_positions(rally_id: str, ball_positions: list[BallPosition]) -> None:
    """Save only ball_positions_json for a rally (preserves all other fields)."""
    ball_json = json.dumps([bp.to_dict() for bp in ball_positions])
    query = """
        UPDATE player_tracks
        SET ball_positions_json = %s::jsonb
        WHERE rally_id = %s
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, [ball_json, rally_id])
        conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill ball tracking data")
    parser.add_argument("--dry-run", action="store_true", help="Preview without saving")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("rallycut.tracking.ball_filter").setLevel(logging.WARNING)

    rallies = find_rallies_missing_ball_data()
    if not rallies:
        print("All rallies already have ball tracking data.")
        return

    print(f"Found {len(rallies)} rallies missing ball data:\n")
    for r in rallies:
        dur = (r["end_ms"] - r["start_ms"]) / 1000
        print(f"  {r['rally_id'][:12]}  video={r['video_id'][:10]}  {dur:.1f}s")

    if args.dry_run:
        print("\nDry run — no tracking performed.")
        return

    # Group rallies by video to avoid re-downloading
    by_video: dict[str, list[dict]] = {}
    for r in rallies:
        by_video.setdefault(r["video_id"], []).append(r)

    tracker = create_ball_tracker()
    total_saved = 0

    for video_id, video_rallies in by_video.items():
        video_path = get_video_path(video_id)
        if video_path is None:
            print(f"\n  SKIP video {video_id[:10]} — not available")
            continue

        print(f"\nProcessing video {video_id[:10]} ({len(video_rallies)} rallies)...")

        for r in video_rallies:
            rally_short = r["rally_id"][:12]
            t0 = time.time()

            result = tracker.track_video(
                video_path=str(video_path),
                start_ms=r["start_ms"],
                end_ms=r["end_ms"],
                enable_filtering=True,
            )

            # WASB track_video with start_ms/end_ms already returns
            # rally-relative frame numbers (0-indexed), no adjustment needed.

            elapsed = time.time() - t0
            n_pos = len(result.positions)

            save_ball_positions(r["rally_id"], result.positions)
            total_saved += 1

            print(f"  {rally_short}  {n_pos:>4} positions  {elapsed:.1f}s  [SAVED]")

    print(f"\nDone. Saved ball data for {total_saved}/{len(rallies)} rallies.")


if __name__ == "__main__":
    main()
