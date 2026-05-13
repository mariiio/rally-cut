#!/usr/bin/env python3
"""Backfill courtCalibrationJson for videos missed by the broken preflight path.

Between commit b8ceeedd (2026-04-14) and the auto-save restoration, the
preflight rewrite stopped persisting detected court corners into
`videos.court_calibration_json`. This script re-runs `CourtDetector` on
candidate videos and writes the result with `court_calibration_source = 'auto'`,
matching what the live qualityService now does on every preflight run.

Selection (default): videos where `court_calibration_json IS NULL`,
`court_calibration_source IS NULL`, `deleted_at IS NULL`. Pass `--video-ids
<csv>` to target specific UUIDs instead.

Confidence + on-screen gates match qualityService.ts:
    confidence >= 0.7
    every corner within ±0.3 of [0, 1] on both axes
    not overwriting `manual` source (extra guard; default filter excludes those)

Usage:
    uv run python scripts/backfill_court_auto_calibration.py --dry-run
    uv run python scripts/backfill_court_auto_calibration.py --apply
    uv run python scripts/backfill_court_auto_calibration.py --apply --video-ids 38f65800-...,06f0b063-...
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

MIN_CONFIDENCE = 0.7
MAX_OFFSCREEN_MARGIN = 0.3


def corners_reasonable(corners: list[dict[str, float]]) -> bool:
    if len(corners) != 4:
        return False
    for c in corners:
        if (
            c["x"] < -MAX_OFFSCREEN_MARGIN
            or c["x"] > 1 + MAX_OFFSCREEN_MARGIN
            or c["y"] < -MAX_OFFSCREEN_MARGIN
            or c["y"] > 1 + MAX_OFFSCREEN_MARGIN
        ):
            return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--apply", action="store_true", help="Write to DB")
    group.add_argument("--dry-run", action="store_true", help="Detect + report only")
    parser.add_argument(
        "--video-ids",
        type=str,
        default=None,
        help="Comma-separated UUIDs; overrides the default NULL-calibration selection",
    )
    args = parser.parse_args()

    from rallycut.court.detector import CourtDetectionConfig, CourtDetector
    from rallycut.evaluation.db import get_connection
    from rallycut.evaluation.tracking.db import get_video_path

    if args.video_ids:
        ids = [s.strip() for s in args.video_ids.split(",") if s.strip()]
        sql = (
            "SELECT id, filename, court_calibration_source FROM videos "
            "WHERE id = ANY(%s) AND deleted_at IS NULL ORDER BY created_at"
        )
        params: tuple = (ids,)
    else:
        sql = (
            "SELECT id, filename, court_calibration_source FROM videos "
            "WHERE court_calibration_json IS NULL "
            "  AND court_calibration_source IS NULL "
            "  AND deleted_at IS NULL "
            "ORDER BY created_at"
        )
        params = ()

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    if not rows:
        print("No candidate videos found.")
        return 0

    print(f"Found {len(rows)} candidate video(s)\n", flush=True)
    header = f"{'video':<14} {'filename':<20} {'conf':>6} {'reasonable':>10} {'action':<14}"
    print(header)
    print("-" * len(header), flush=True)

    detector = CourtDetector(CourtDetectionConfig())
    applied = 0
    skipped_low = 0
    skipped_off = 0
    skipped_manual = 0
    skipped_no_video = 0

    for idx, (video_id, filename, source) in enumerate(rows, start=1):
        short = video_id[:12]
        name = (filename or "?")[:20]

        if source == "manual":
            print(f"[{idx}/{len(rows)}] {short:<14} {name:<20} {'':>6} {'':>10} skip:manual", flush=True)
            skipped_manual += 1
            continue

        video_path = get_video_path(video_id)
        if video_path is None:
            print(f"[{idx}/{len(rows)}] {short:<14} {name:<20} {'':>6} {'':>10} skip:no-video", flush=True)
            skipped_no_video += 1
            continue

        result = detector.detect(video_path)
        conf = float(result.confidence)
        ok_shape = result.corners and len(result.corners) == 4
        reasonable = ok_shape and corners_reasonable(result.corners)

        if not reasonable:
            print(
                f"[{idx}/{len(rows)}] {short:<14} {name:<20} {conf:>6.3f} "
                f"{'False':>10} skip:off-screen",
                flush=True,
            )
            skipped_off += 1
            continue

        if conf < MIN_CONFIDENCE:
            print(
                f"[{idx}/{len(rows)}] {short:<14} {name:<20} {conf:>6.3f} "
                f"{'True':>10} skip:low-conf",
                flush=True,
            )
            skipped_low += 1
            continue

        action = "apply" if args.apply else "would-apply"
        print(
            f"[{idx}/{len(rows)}] {short:<14} {name:<20} {conf:>6.3f} "
            f"{'True':>10} {action:<14}",
            flush=True,
        )

        if args.apply:
            with get_connection() as conn, conn.cursor() as cur:
                cur.execute(
                    "UPDATE videos SET court_calibration_json = %s, "
                    "court_calibration_source = 'auto' "
                    "WHERE id = %s "
                    "  AND court_calibration_source IS DISTINCT FROM 'manual'",
                    (json.dumps(result.corners), video_id),
                )
                conn.commit()
            applied += 1

    print(f"\nSummary: total={len(rows)} applied={applied} "
          f"skipped(manual={skipped_manual}, no-video={skipped_no_video}, "
          f"off-screen={skipped_off}, low-conf={skipped_low})",
          flush=True)
    if args.dry_run:
        eligible = len(rows) - skipped_manual - skipped_no_video - skipped_off - skipped_low
        print(f"Dry run: {eligible} video(s) would be updated. Re-run with --apply.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
