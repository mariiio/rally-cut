"""Re-run ball tracking for all rallies with action ground truth.

Updates ball_positions_json in the database so downstream eval
(eval_action_detection.py) uses the latest ball filter config.

Usage:
    uv run python scripts/resave_ball_for_action_gt.py           # Dry run
    uv run python scripts/resave_ball_for_action_gt.py --save    # Save to DB
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from rallycut.evaluation.db import get_connection
from rallycut.evaluation.tracking.db import get_video_path
from rallycut.tracking.ball_tracker import create_ball_tracker

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-run ball tracking for action GT rallies"
    )
    parser.add_argument("--save", action="store_true", help="Save to DB")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("rallycut.tracking.ball_filter").setLevel(logging.WARNING)

    # Load all rallies with action GT
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT r.id, r.video_id, r.start_ms, r.end_ms
                FROM rallies r
                JOIN player_tracks pt ON pt.rally_id = r.id
                WHERE pt.action_ground_truth_json IS NOT NULL
                AND pt.ball_positions_json IS NOT NULL
                ORDER BY r.video_id, r.start_ms
            """)
            rows = cur.fetchall()

    if not rows:
        print("No rallies with action GT found")
        sys.exit(1)

    print(f"Re-tracking ball for {len(rows)} rallies with action GT")
    if not args.save:
        print("(dry run — add --save to update DB)")

    ball_tracker = create_ball_tracker()
    total_time = 0.0

    for i, (rally_id, video_id, start_ms, end_ms) in enumerate(rows):
        video_path = get_video_path(str(video_id))
        if video_path is None:
            print(f"[{i + 1}/{len(rows)}] {str(rally_id)[:10]} MISSING VIDEO")
            continue

        t0 = time.time()
        ball_result = ball_tracker.track_video(
            Path(video_path), start_ms=start_ms, end_ms=end_ms,
        )
        elapsed = time.time() - t0
        total_time += elapsed

        n_detected = sum(1 for bp in ball_result.positions if bp.confidence > 0)

        status = ""
        if args.save:
            ball_json = json.dumps([bp.to_dict() for bp in ball_result.positions])
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE player_tracks SET ball_positions_json = %s::jsonb "
                        "WHERE rally_id = %s",
                        [ball_json, str(rally_id)],
                    )
                conn.commit()
            status = " [SAVED]"

        print(
            f"[{i + 1}/{len(rows)}] {str(rally_id)[:10]} "
            f"{n_detected} detections ({elapsed:.1f}s){status}"
        )

    print(f"\nDone. {len(rows)} rallies in {total_time:.0f}s")


if __name__ == "__main__":
    main()
