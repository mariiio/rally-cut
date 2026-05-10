"""Re-track 42 corrupted rallies with negative frame numbers in ball_positions_json.

Usage:
    cd analysis
    uv run python -u scripts/retrack_corrupted_rallies.py 2>&1 | tee reports/retrack_corrupted_rallies_2026_05_10.log
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
import time
from typing import Any, cast

from rallycut.evaluation.tracking.db import (
    get_connection,
    get_rally_info,
    get_video_path,
)

# Ensure unbuffered output
os.environ['PYTHONUNBUFFERED'] = '1'

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    force=True,
)
logger = logging.getLogger("retrack_corrupted_rallies")

# Set all loggers to flush
for h in logging.root.handlers:
    h.flush = lambda: None


def _retrack_rally(rally_id: str) -> dict[str, Any]:
    """Run player_tracker.track_video on the rally; return result dict."""
    from rallycut.court.calibration import CourtCalibrator
    from rallycut.tracking.ball_tracker import create_ball_tracker
    from rallycut.tracking.player_tracker import PlayerTracker

    info = get_rally_info(rally_id)
    if info is None:
        raise ValueError(f"rally {rally_id} not found")
    video_path = get_video_path(info.video_id)
    if video_path is None:
        raise ValueError(f"video {info.video_id} unavailable")

    calibrator = None
    if info.calibration_json and len(info.calibration_json) == 4:
        calibrator = CourtCalibrator()
        image_corners = [(c["x"], c["y"]) for c in info.calibration_json]
        calibrator.calibrate(image_corners)

    logger.debug("ball-tracker start for rally %s", rally_id[:8])
    ball_tracker = create_ball_tracker()
    ball_result = ball_tracker.track_video(
        video_path, start_ms=info.start_ms, end_ms=info.end_ms,
    )
    ball_positions = ball_result.positions
    logger.debug("ball-tracker done: %d positions", len(ball_positions))

    # Snapshot ball position frame numbers BEFORE PlayerTracker mutates them.
    # PlayerTracker.track_video subtracts start_frame from ball_positions in-place,
    # which is correct for video-absolute frames but WASB returns rally-relative.
    # We snapshot first to preserve the rally-relative frames for ball_positions_json.
    ball_positions_snapshot = [
        {"x": bp.x, "y": bp.y, "frameNumber": bp.frame_number,
         "confidence": bp.confidence}
        for bp in ball_positions
    ]

    logger.debug("player-tracker start for rally %s", rally_id[:8])
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
    logger.debug("player-tracker done in %.1fs", elapsed)

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
    # Use the pre-mutation snapshot so we save rally-relative ball frames
    ball_positions_json = ball_positions_snapshot

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
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
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
            conn.commit()

            # Strip the assignmentAnchor for this rally from match_analysis_json
            # so match-players solves it fresh next time (track_ids changed).
            with conn.cursor() as cur:
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
                            logger.debug("stripped %d anchor(s) from match_analysis_json", stripped)
            conn.commit()
    except Exception as e:
        logger.error(f"Error saving to DB for rally {rally_id}: {e}", exc_info=True)
        raise


def _find_corrupted_rally_ids() -> list[str]:
    """Query DB to find rally IDs with negative frame numbers in ball_positions_json.
    Restrict to GT-labeled rallies. Exclude rallies with existing serve actions.
    """
    # Load GT to find GT videos
    with open("training_datasets/beach_v11/action_ground_truth.json") as f:
        gt = json.load(f)
    gt_hashes = {r["video_content_hash"] for r in gt["rallies"]}

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT id, content_hash FROM videos WHERE content_hash = ANY(%s)", [list(gt_hashes)])
            meta = {r[0]: r[1] for r in cur.fetchall()}
    hash_to_id = {h: vid for vid, h in meta.items()}

    corrupted_rally_ids: list[str] = []
    with get_connection() as conn:
        cur = conn.cursor()
        for r in gt["rallies"]:
            chash = r["video_content_hash"]
            if chash not in hash_to_id:
                continue
            vid = hash_to_id[chash]
            cur.execute(
                """SELECT r.id, pt.actions_json, pt.ball_positions_json
                   FROM rallies r JOIN player_tracks pt ON pt.rally_id = r.id
                   WHERE r.video_id = %s AND r.start_ms = %s""",
                [vid, r["rally_start_ms"]],
            )
            row = cur.fetchone()
            if row is None or not row[1] or not row[2]:
                continue
            rid, aj, bp_json = row
            actions = (aj or {}).get("actions") or []
            # Only re-track rallies without serve action
            if any(a.get("action") == "serve" for a in actions):
                continue
            # Check for negative frames
            frames = [bp.get("frameNumber", 0) for bp in bp_json if isinstance(bp, dict)]
            if frames and min(frames) < 0:
                corrupted_rally_ids.append(rid)

    return corrupted_rally_ids


def main() -> int:
    print("=" * 80)
    print("RE-TRACKING 42 CORRUPTED RALLIES WITH NEGATIVE FRAME NUMBERS")
    print("=" * 80)
    print()
    sys.stdout.flush()

    logger.info("Finding corrupted rally IDs...")
    corrupted_ids = _find_corrupted_rally_ids()
    logger.info(f"Found {len(corrupted_ids)} corrupted rallies")
    sys.stdout.flush()

    if len(corrupted_ids) == 0:
        logger.error("Expected 42 corrupted rallies, found 0. Aborting.")
        return 1

    print()
    print(f"Processing {len(corrupted_ids)} rallies...")
    print()
    sys.stdout.flush()

    updated = 0
    errored = 0
    errors_log: list[str] = []
    t0_total = time.time()

    for idx, rally_id in enumerate(corrupted_ids, 1):
        t0 = time.time()
        try:
            logger.info(f"[{idx}/{len(corrupted_ids)}] Re-tracking {rally_id[:8]}...")
            sys.stdout.flush()
            result = _retrack_rally(rally_id)
            ball_count = len(result["ball_positions"])
            pos_count = len(result["positions"])
            elapsed = time.time() - t0
            logger.info(
                f"  ✓ {rally_id[:8]} done: {ball_count} ball pos, {pos_count} player pos in {elapsed:.1f}s"
            )
            sys.stdout.flush()
            _save_to_db(rally_id, result)
            logger.info(f"  ✓ {rally_id[:8]} saved to DB")
            sys.stdout.flush()
            updated += 1
        except Exception as e:
            errored += 1
            err_msg = f"{rally_id[:8]}: {type(e).__name__}: {str(e)}"
            errors_log.append(err_msg)
            logger.error(f"  ✗ {err_msg}")
            sys.stdout.flush()
            continue

    elapsed_total = time.time() - t0_total

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total rallies:      {len(corrupted_ids)}")
    print(f"Updated:            {updated}")
    print(f"Errored:            {errored}")
    print(f"Total runtime:      {elapsed_total / 60:.1f} minutes")
    print()
    sys.stdout.flush()

    if errors_log:
        print("Errors encountered:")
        for err in errors_log:
            print(f"  - {err}")
        print()
        sys.stdout.flush()

    # Verify post-retrack state
    logger.info("Verifying post-retrack state...")
    sys.stdout.flush()
    with get_connection() as conn:
        cur = conn.cursor()
        # Count rallies with min frame >= 0
        cur.execute(
            """SELECT COUNT(*) FROM (
                 SELECT r.id FROM rallies r
                 JOIN player_tracks pt ON pt.rally_id = r.id
                 WHERE r.id = ANY(%s) AND pt.ball_positions_json IS NOT NULL
                   AND (SELECT MIN(COALESCE((bp->>'frameNumber')::int, 0))
                        FROM jsonb_array_elements(pt.ball_positions_json) AS bp) >= 0
               ) x""",
            [corrupted_ids],
        )
        verified_positive = cur.fetchone()[0]
        logger.info(f"Verified {verified_positive} rallies have non-negative frames")
        sys.stdout.flush()

        # Count rallies with at least 1 action
        cur.execute(
            """SELECT COUNT(*) FROM (
                 SELECT r.id FROM rallies r
                 JOIN player_tracks pt ON pt.rally_id = r.id
                 WHERE r.id = ANY(%s) AND pt.actions_json IS NOT NULL
                   AND jsonb_array_length(pt.actions_json->'actions') > 0
               ) x""",
            [corrupted_ids],
        )
        with_actions = cur.fetchone()[0]
        logger.info(f"Verified {with_actions} rallies have >= 1 action")
        sys.stdout.flush()

        # Count rallies with at least 1 serve
        cur.execute(
            """SELECT COUNT(*) FROM (
                 SELECT r.id FROM rallies r
                 JOIN player_tracks pt ON pt.rally_id = r.id
                 WHERE r.id = ANY(%s) AND pt.actions_json IS NOT NULL
                   AND EXISTS (
                     SELECT 1 FROM jsonb_array_elements(pt.actions_json->'actions') AS a
                     WHERE a->>'action' = 'serve'
                   )
               ) x""",
            [corrupted_ids],
        )
        with_serve = cur.fetchone()[0]
        logger.info(f"Verified {with_serve} rallies have >= 1 serve action")
        sys.stdout.flush()

    print()
    print(f"Post-retrack verification:")
    print(f"  Rallies with non-negative frames: {verified_positive}/{len(corrupted_ids)}")
    print(f"  Rallies with >= 1 action:        {with_actions}/{len(corrupted_ids)}")
    print(f"  Rallies with >= 1 serve:         {with_serve}/{len(corrupted_ids)}")
    print()
    sys.stdout.flush()

    if updated == len(corrupted_ids):
        logger.info("SUCCESS: All rallies re-tracked and verified")
        return 0
    else:
        logger.warning(f"PARTIAL: {updated}/{len(corrupted_ids)} rallies updated successfully")
        return 1


if __name__ == "__main__":
    sys.exit(main())
