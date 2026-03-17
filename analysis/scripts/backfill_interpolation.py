#!/usr/bin/env python3
"""Backfill gap interpolation on existing tracked rallies.

Reads positions_json from DB, runs interpolate_player_gaps, writes back.
No retracking needed — applies the same interpolation that new tracking runs
would get automatically.

Usage:
    uv run python scripts/backfill_interpolation.py --dry-run   # Preview changes
    uv run python scripts/backfill_interpolation.py --apply      # Apply to DB
    uv run python scripts/backfill_interpolation.py --apply --video <video-id>
"""

import argparse
import json
import sys
from collections import Counter
from typing import Any

import numpy as np

from rallycut.evaluation.tracking.db import get_connection
from rallycut.tracking.player_filter import PlayerFilterConfig, interpolate_player_gaps
from rallycut.tracking.player_tracker import PlayerPosition


def load_rallies(video_id: str | None = None) -> list[dict[str, Any]]:
    """Load rallies with tracking data."""
    query = """
        SELECT r.id, r.video_id, v.filename, pt.positions_json,
               pt.primary_track_ids, pt.frame_count, pt.quality_report_json
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        JOIN videos v ON v.id = r.video_id
        WHERE pt.status = 'COMPLETED' AND pt.positions_json IS NOT NULL
          AND r.rejection_reason IS NULL
    """
    params: list[Any] = []
    if video_id:
        query += " AND r.video_id = %s"
        params.append(video_id)
    query += " ORDER BY v.filename, r.start_ms"

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            cols = ["rally_id", "video_id", "filename", "positions_json",
                    "primary_track_ids", "frame_count", "quality_report_json"]
            return [dict(zip(cols, row)) for row in cur.fetchall()]


def compute_4player_pct(positions_json: list[dict[str, Any]], primary_ids: list[int]) -> float:
    """Compute % of frames with 4 primary players."""
    primary_set = set(primary_ids)
    frame_counts: dict[int, int] = Counter()
    for p in positions_json:
        if p["trackId"] in primary_set:
            frame_counts[p["frameNumber"]] += 1
    if not frame_counts:
        return 0.0
    return sum(1 for c in frame_counts.values() if c >= 4) / len(frame_counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backfill gap interpolation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dry-run", action="store_true", help="Preview without changes")
    group.add_argument("--apply", action="store_true", help="Write interpolated positions to DB")
    parser.add_argument("--video", type=str, help="Limit to specific video ID")
    args = parser.parse_args()

    config = PlayerFilterConfig()
    print(f"Config: max_interpolation_gap={config.max_interpolation_gap}, "
          f"confidence={config.interpolated_confidence}")

    rallies = load_rallies(video_id=args.video)
    print(f"Loaded {len(rallies)} rallies\n")

    if not rallies:
        sys.exit(0)

    before_pcts = []
    after_pcts = []
    updates: list[tuple[list[dict[str, Any]], dict[str, Any] | None, str]] = []
    total_interpolated = 0
    skipped = 0

    for i, rally in enumerate(rallies):
        if (i + 1) % 100 == 0:
            print(f"  Processing {i + 1}/{len(rallies)}...")

        positions_json = rally["positions_json"]
        primary_ids = rally["primary_track_ids"] or []
        if not positions_json or not primary_ids:
            skipped += 1
            continue

        pct_before = compute_4player_pct(positions_json, primary_ids)

        positions = [
            PlayerPosition(
                frame_number=p["frameNumber"], track_id=p["trackId"],
                x=p["x"], y=p["y"], width=p["width"], height=p["height"],
                confidence=p["confidence"],
            )
            for p in positions_json
        ]

        new_positions, n_interp = interpolate_player_gaps(positions, primary_ids, config)
        total_interpolated += n_interp

        if n_interp == 0:
            skipped += 1
            continue

        new_json = [p.to_dict() for p in new_positions]
        pct_after = compute_4player_pct(new_json, primary_ids)
        before_pcts.append(pct_before)
        after_pcts.append(pct_after)

        # Update quality report with interpolation count
        qr = rally["quality_report_json"]
        if isinstance(qr, dict):
            qr = {**qr, "interpolatedPositionCount": n_interp}

        updates.append((new_json, qr, rally["rally_id"]))

    print("\nResults:")
    print(f"  Rallies to update: {len(updates)}")
    print(f"  Skipped (no gaps): {skipped}")
    print(f"  Total interpolated positions: {total_interpolated}")
    if before_pcts:
        print(f"  Frames with 4 players: {np.mean(before_pcts):.1%} -> {np.mean(after_pcts):.1%} "
              f"(+{np.mean(np.array(after_pcts) - np.array(before_pcts)):.1%})")

    if args.dry_run:
        print("\n--dry-run: no changes written.")
        return

    print(f"\nWriting {len(updates)} updates to DB...")
    with get_connection() as conn:
        with conn.cursor() as cur:
            for new_json, qr, rally_id in updates:
                cur.execute(
                    """UPDATE player_tracks
                       SET positions_json = %s::jsonb, quality_report_json = %s::jsonb
                       WHERE rally_id = %s""",
                    (json.dumps(new_json),
                     json.dumps(qr) if qr else None,
                     rally_id),
                )
        conn.commit()
    print("Done.")


if __name__ == "__main__":
    main()
