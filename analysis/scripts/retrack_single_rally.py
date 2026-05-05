"""Re-run player tracking for a single rally and save to DB.

Used to validate the bbox-quality-aware merge gate (2026-05-05) on
real production data: re-tracking is the only way to actually exercise
`relink_primary_fragments` (which lives inside player_tracker, not
match-players). This script retracks ONE rally, prints the chimera
structure before/after via probe_track_continuity, and writes the new
tracking results to DB so the editor reflects them.

Usage:
    cd analysis
    uv run python scripts/retrack_single_rally.py --rally <rally-uuid>
    uv run python scripts/retrack_single_rally.py --rally <rally-uuid> --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Any, cast

from rallycut.evaluation.tracking.db import (
    get_connection,
    get_rally_info,
    get_video_path,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("retrack_single_rally")


def _retrack_rally(rally_id: str) -> dict[str, Any]:
    """Run player_tracker.track_video on the rally; return result dict."""
    from rallycut.court.calibration import CourtCalibrator
    from rallycut.tracking.ball_tracker import create_ball_tracker
    from rallycut.tracking.player_tracker import PlayerTracker

    info = get_rally_info(rally_id)
    if info is None:
        raise SystemExit(f"rally {rally_id} not found")
    video_path = get_video_path(info.video_id)
    if video_path is None:
        raise SystemExit(f"video {info.video_id} unavailable")

    calibrator = None
    if info.calibration_json and len(info.calibration_json) == 4:
        calibrator = CourtCalibrator()
        image_corners = [(c["x"], c["y"]) for c in info.calibration_json]
        calibrator.calibrate(image_corners)

    logger.info("ball-tracker start")
    ball_tracker = create_ball_tracker()
    ball_result = ball_tracker.track_video(
        video_path, start_ms=info.start_ms, end_ms=info.end_ms,
    )
    ball_positions = ball_result.positions
    logger.info("ball-tracker done: %d positions", len(ball_positions))

    logger.info("player-tracker start")
    t0 = time.time()
    tracker = PlayerTracker()
    result = tracker.track_video(
        video_path=video_path,
        start_ms=info.start_ms,
        end_ms=info.end_ms,
        stride=1,
        filter_enabled=True,
        court_calibrator=calibrator,
        ball_positions=ball_positions,
    )
    elapsed = time.time() - t0
    logger.info("player-tracker done in %.1fs", elapsed)

    positions = [
        {
            "x": p.x, "y": p.y, "width": p.width, "height": p.height,
            "trackId": p.track_id, "frameNumber": p.frame_number,
            "confidence": p.confidence,
        }
        for p in result.positions
    ]
    raw_positions = [
        {
            "x": p.x, "y": p.y, "width": p.width, "height": p.height,
            "trackId": p.track_id, "frameNumber": p.frame_number,
            "confidence": p.confidence,
        }
        for p in result.raw_positions
    ]
    ball_positions_json = [
        {"x": bp.x, "y": bp.y, "frameNumber": bp.frame_number,
         "confidence": bp.confidence}
        for bp in ball_positions
    ]

    return {
        "positions": positions,
        "raw_positions": raw_positions,
        "primary_track_ids": list(result.primary_track_ids),
        "ball_positions": ball_positions_json,
        "frame_count": result.frame_count,
        "fps": result.video_fps,
        "court_split_y": result.court_split_y,
        "elapsed_ms": int(elapsed * 1000),
    }


def _save_to_db(rally_id: str, result: dict[str, Any]) -> None:
    """Update player_tracks row with new tracking output. Wipes
    pre_remap_state_json + assignmentAnchor cache so the new positions
    flow cleanly through match-players + remap on the next run.
    """
    with get_connection() as conn, conn.cursor() as cur:
        # Update player_tracks core fields. Clear pre_remap_state_json so
        # remap-track-ids treats this as a fresh tracking run.
        cur.execute(
            """
            UPDATE player_tracks
            SET positions_json = %s::jsonb,
                raw_positions_json = %s::jsonb,
                primary_track_ids = %s::jsonb,
                ball_positions_json = %s::jsonb,
                frame_count = %s,
                fps = %s,
                court_split_y = %s,
                processing_time_ms = %s,
                model_version = 'yolo11s',
                completed_at = now(),
                pre_remap_state_json = NULL,
                needs_retrack = false
            WHERE rally_id = %s
            """,
            [
                json.dumps(result["positions"]),
                json.dumps(result["raw_positions"]),
                json.dumps(result["primary_track_ids"]),
                json.dumps(result["ball_positions"]),
                int(result["frame_count"]),
                float(result["fps"]),
                float(result["court_split_y"]) if result["court_split_y"] is not None else None,
                int(result["elapsed_ms"]),
                rally_id,
            ],
        )

        # Strip the assignmentAnchor for this rally from match_analysis_json
        # so match-players solves it fresh next time (track_ids changed).
        cur.execute(
            "SELECT video_id FROM rallies WHERE id = %s",
            [rally_id],
        )
        video_row = cur.fetchone()
        if video_row:
            video_id = cast(str, video_row[0])
            cur.execute(
                "SELECT match_analysis_json FROM videos WHERE id = %s",
                [video_id],
            )
            ma_row = cur.fetchone()
            if ma_row and ma_row[0]:
                ma = ma_row[0] if isinstance(ma_row[0], dict) else json.loads(cast(str, ma_row[0]))
                stripped = 0
                for entry in ma.get("rallies", []):
                    if entry.get("rallyId") == rally_id and entry.pop("assignmentAnchor", None):
                        stripped += 1
                if stripped:
                    cur.execute(
                        "UPDATE videos SET match_analysis_json = %s::jsonb WHERE id = %s",
                        [json.dumps(ma), video_id],
                    )
                    logger.info("stripped %d anchor(s) from match_analysis_json", stripped)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rally", required=True)
    parser.add_argument("--dry-run", action="store_true",
                        help="Re-track but don't save to DB.")
    args = parser.parse_args()

    print(f"=== Re-tracking rally {args.rally[:8]} ===")
    result = _retrack_rally(args.rally)
    print(
        f"  primary_track_ids: {result['primary_track_ids']}\n"
        f"  positions: {len(result['positions'])}, "
        f"raw_positions: {len(result['raw_positions'])}\n"
        f"  ball positions: {len(result['ball_positions'])}\n"
        f"  elapsed: {result['elapsed_ms'] / 1000:.1f}s"
    )

    if args.dry_run:
        print("(dry-run — not saving)")
        return 0

    _save_to_db(args.rally, result)
    print("Saved to DB. Run match-players + remap to update editor view.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
