"""Retrack stale panel rallies in bulk and save to DB.

Runs retrack_single_rally logic for each stale rally in the 4 panel videos.
Prints per-rally progress: [N/total] rally_id: primary_track_ids=... elapsed=...s
Stops on first error.

Usage:
    cd analysis
    uv run python scripts/retrack_panel_stale.py
    uv run python scripts/retrack_panel_stale.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from typing import Any, cast

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger("retrack_panel_stale")

# 28 stale rally UUIDs across the 4 panel videos.
# Resumed run 2026-05-07: rallies 1-4 already retracked successfully in a
# prior session (commented below); restart from rally 5.
STALE_RALLIES = [
    # 5c756c41 (8 stale; first 4 already done)
    # "7094136a-6b71-438e-aaee-e6ddd5cf1589",
    # "8c49e480-407e-4118-8a9e-c4ed5172a7ce",
    # "fb6e37bf-0f0d-4c38-a8c0-3a5f81e2998d",
    # "11adfc2f-69da-458a-9c9b-0743466752b0",
    "82d549a6-c99f-4f92-8575-f77781d71348",
    "50ece37b-2a6b-4105-94f3-d94d1ff3c5b9",
    "1099f2d3-0ac5-4eca-b483-8c56be5e4481",
    "21d4cdf6-ca17-4a4a-a1a8-d567f4fa1d76",
    # b5fb0594 (4 stale, be3134ba already done)
    "cc2b967b-51dd-43e1-a824-481b5903eccf",
    "9d5f5152-7772-4e09-b677-7cb4c8f9862b",
    "2e95067d-099f-4818-a55c-6d429f96c3ce",
    "26165b51-ea99-4a43-8380-edb3a4c959ed",
    # 854bb250 (1 stale)
    "55c2c6e5-1e13-4146-bac1-470565781a38",
    # 7d77980f (15 stale)
    "8b0b9e13-0694-4697-b131-745ee79a81d8",
    "09553ef1-1b53-4756-9c29-a74db1f1a29e",
    "ad7cccbf-657f-4a49-8069-e33575be537d",
    "97c42190-ec9a-480d-b0d4-bcbbcdd89bdb",
    "e50f127e-3952-4ba2-9047-39fff14a2e25",
    "8c802c26-fb67-4a0f-b3a9-315a48cc6a2d",
    "724ead56-4093-48b6-8a78-fabc1a18fca1",
    "03144243-8fd8-4926-a2bc-3c4a8ba6b0a3",
    "4bfafd6d-cada-4279-801f-76b4de7db8a1",
    "fdc4375b-513e-4a5b-a67c-e3fc27620e33",
    "d5c51d52-6e57-4739-a86c-aef3ce2b5f80",
    "4e7e589c-8e64-4872-958b-45aeefab3ced",
    "d934f57a-5555-4695-b4fb-3372aa3f9d7e",
    "9f46f74e-809f-4322-8c94-db288cb479bc",
    "db841666-5900-422e-afd9-2e74a7754534",
]


def _retrack_rally(rally_id: str) -> dict[str, Any]:
    """Run player_tracker.track_video on the rally; return result dict."""
    from rallycut.court.calibration import CourtCalibrator
    from rallycut.tracking.ball_tracker import create_ball_tracker
    from rallycut.tracking.player_tracker import PlayerTracker
    from rallycut.evaluation.tracking.db import get_rally_info, get_video_path

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

    ball_tracker = create_ball_tracker()
    ball_result = ball_tracker.track_video(
        video_path, start_ms=info.start_ms, end_ms=info.end_ms,
    )
    ball_positions = ball_result.positions

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
    from rallycut.evaluation.tracking.db import get_connection

    with get_connection() as conn, conn.cursor() as cur:
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

        cur.execute("SELECT video_id FROM rallies WHERE id = %s", [rally_id])
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    total = len(STALE_RALLIES)
    wall_start = time.time()

    print(f"Retracking {total} stale panel rallies{'  (DRY RUN)' if args.dry_run else ''}")
    print(f"Estimated time: ~{total * 3.5 / 60:.0f} min at 3.5 min/rally\n")

    for idx, rally_id in enumerate(STALE_RALLIES, 1):
        t0 = time.time()
        print(f"[{idx}/{total}] {rally_id[:8]}  retracking...", flush=True)
        try:
            result = _retrack_rally(rally_id)
        except Exception as e:
            print(f"[{idx}/{total}] {rally_id[:8]}  ERROR: {e}", flush=True)
            return 1

        elapsed = time.time() - t0
        pti = result["primary_track_ids"]
        print(
            f"[{idx}/{total}] {rally_id[:8]}  primary_track_ids={pti}  "
            f"positions={len(result['positions'])}  elapsed={elapsed:.0f}s",
            flush=True,
        )

        if not args.dry_run:
            _save_to_db(rally_id, result)

    wall = time.time() - wall_start
    print(f"\nDone. {total} rallies in {wall / 60:.1f} min total.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
