#!/usr/bin/env python3
"""Update court calibration in DB using YOLO-pose keypoint detection.

Runs keypoint detection on all videos with existing court GT, compares
against stored calibration, and optionally updates the DB with better
keypoint predictions.

Usage:
    uv run python scripts/update_court_calibrations.py --dry-run   # Compare only
    uv run python scripts/update_court_calibrations.py --apply      # Update DB
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Update court calibrations with keypoint detection"
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Apply updates to database (default: dry-run comparison only)",
    )
    parser.add_argument(
        "--model", type=Path,
        default=Path("weights/court_keypoint/court_keypoint_best.pt"),
        help="Keypoint model path",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.5,
        help="Minimum keypoint confidence to accept (default: 0.5)",
    )
    args = parser.parse_args()

    from rallycut.court.keypoint_detector import CourtKeypointDetector
    from rallycut.evaluation.db import get_connection
    from rallycut.evaluation.tracking.db import get_video_path

    detector = CourtKeypointDetector(model_path=args.model)
    if not detector.model_exists:
        print(f"Model not found: {args.model}")
        return

    # Load all videos with court calibration
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, court_calibration_json, court_calibration_source
                   FROM videos
                   WHERE court_calibration_json IS NOT NULL
                   ORDER BY id"""
            )
            rows = cur.fetchall()

    print(f"Found {len(rows)} videos with court calibration\n")

    header = f"{'Video':<14} {'Source':<8} {'KP Conf':>8} {'DB MCD':>8} {'KP MCD':>8} {'Action':<12}"
    print(header)
    print("-" * len(header))

    updated = 0
    skipped_manual = 0
    skipped_low_conf = 0
    skipped_no_video = 0

    for vid_id, cal_json, cal_source in rows:
        vid_short = vid_id[:12]

        video_path = get_video_path(vid_id)
        if video_path is None:
            print(f"{vid_short:<14} {'':>8} {'':>8} {'':>8} {'':>8} no video")
            skipped_no_video += 1
            continue

        # Run keypoint detection
        result = detector.detect(video_path)

        if len(result.corners) != 4 or result.confidence < args.min_confidence:
            print(
                f"{vid_short:<14} {(cal_source or '-'):<8} "
                f"{result.confidence:>7.3f} {'':>8} {'':>8} low conf"
            )
            skipped_low_conf += 1
            continue

        # Compare with stored calibration
        gt_corners = cal_json if isinstance(cal_json, list) else json.loads(cal_json)
        if len(gt_corners) != 4:
            continue

        # Mean corner distance between keypoint detection and stored calibration
        kp_mcd = sum(
            ((r["x"] - g["x"]) ** 2 + (r["y"] - g["y"]) ** 2) ** 0.5
            for r, g in zip(result.corners, gt_corners)
        ) / 4

        source = cal_source or "-"

        # Skip manual calibrations (user-provided, protected)
        if cal_source == "manual":
            print(
                f"{vid_short:<14} {source:<8} "
                f"{result.confidence:>7.3f} {'manual':>8} {kp_mcd:>8.4f} skip manual"
            )
            skipped_manual += 1
            continue

        action = "update" if args.apply else "would update"

        print(
            f"{vid_short:<14} {source:<8} "
            f"{result.confidence:>7.3f} {'stored':>8} {kp_mcd:>8.4f} {action}"
        )

        if args.apply:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """UPDATE videos
                           SET court_calibration_json = %s,
                               court_calibration_source = 'keypoint'
                           WHERE id = %s""",
                        (json.dumps(result.corners), vid_id),
                    )
                conn.commit()
            updated += 1

    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"  Total videos:     {len(rows)}")
    print(f"  Updated:          {updated}")
    print(f"  Skipped (manual): {skipped_manual}")
    print(f"  Skipped (low cf): {skipped_low_conf}")
    print(f"  Skipped (no vid): {skipped_no_video}")

    if not args.apply and updated == 0 and skipped_manual < len(rows):
        eligible = len(rows) - skipped_manual - skipped_low_conf - skipped_no_video
        print(f"\nDry run: {eligible} videos eligible for update.")
        print("Run with --apply to update the database.")


if __name__ == "__main__":
    main()
