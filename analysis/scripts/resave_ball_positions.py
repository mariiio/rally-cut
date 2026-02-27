"""Re-run ball tracking only and save ball positions to DB.

Much faster than full retrack — skips YOLO/BoT-SORT player tracking.
Only updates ball_positions_json column; leaves player data untouched.

Usage:
    uv run python scripts/resave_ball_positions.py         # Dry run (show counts)
    uv run python scripts/resave_ball_positions.py --save   # Save to DB
    uv run python scripts/resave_ball_positions.py --rally <id> --save
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from rallycut.evaluation.tracking.db import (
    get_connection,
    get_video_path,
    load_labeled_rallies,
)
from rallycut.tracking.ball_tracker import create_ball_tracker

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-run ball tracking and save ball positions to DB"
    )
    parser.add_argument("--rally", help="Specific rally ID")
    parser.add_argument("--save", action="store_true", help="Save to DB")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logging.getLogger("rallycut.tracking.ball_filter").setLevel(logging.WARNING)

    rallies = load_labeled_rallies(rally_id=args.rally)
    if not rallies:
        print("No labeled rallies found")
        sys.exit(1)

    # Filter to rallies with ball GT
    rallies = [r for r in rallies if r.ground_truth.ball_positions]
    print(f"Loaded {len(rallies)} rally(s) with ball GT")

    ball_tracker = create_ball_tracker()

    for i, rally in enumerate(rallies):
        rally_short = rally.rally_id[:12]
        video_path = get_video_path(rally.video_id)
        if video_path is None:
            print(f"[{i + 1}/{len(rallies)}] {rally_short} MISSING VIDEO")
            continue

        t0 = time.time()
        ball_result = ball_tracker.track_video(
            video_path, start_ms=rally.start_ms, end_ms=rally.end_ms,
        )
        ball_positions = ball_result.positions
        elapsed = time.time() - t0

        # WASB returns 0-indexed rally-relative frames already — no adjustment needed

        status = ""
        if args.save:
            ball_json = json.dumps([bp.to_dict() for bp in ball_positions])
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "UPDATE player_tracks SET ball_positions_json = %s::jsonb "
                        "WHERE rally_id = %s",
                        [ball_json, rally.rally_id],
                    )
                conn.commit()
            status = " [SAVED]"

        print(
            f"[{i + 1}/{len(rallies)}] {rally_short} "
            f"{len(ball_positions)} ball positions ({elapsed:.1f}s){status}"
        )

    print("Done.")


if __name__ == "__main__":
    main()
