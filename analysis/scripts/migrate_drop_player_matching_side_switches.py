#!/usr/bin/env python3
"""Drop the legacy ``sideSwitches`` key from videos.player_matching_gt_json.

The Score-tracking UI is the source of truth for side switches via
per-rally ``rallies.gt_side_switch``.  The pre-existing
``videos.player_matching_gt_json["sideSwitches"]`` list is a stale
duplicate; nothing reads it after the 2026-04-27 migration.

This script:
  - Walks every video where ``player_matching_gt_json IS NOT NULL``.
  - Removes both ``sideSwitches`` and ``side_switches`` from the JSON.
  - Reports per-video whether the legacy list agreed with the
    chronological ``gt_side_switch`` indices, so disagreements are visible
    at run time (those rows are the spot-check candidates).
  - Idempotent: re-running is a no-op (only ``UPDATE``-s rows where at
    least one of the legacy keys was present).

Usage:
    cd analysis
    uv run python scripts/migrate_drop_player_matching_side_switches.py --dry-run
    uv run python scripts/migrate_drop_player_matching_side_switches.py
    uv run python scripts/migrate_drop_player_matching_side_switches.py \
        --video-id <prefix>            # restrict to one video
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rallycut.evaluation.db import get_connection  # noqa: E402
from rallycut.evaluation.gt_loader import load_side_switches_from_db  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Drop legacy sideSwitches key from player_matching_gt_json"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report diffs without updating the DB",
    )
    parser.add_argument(
        "--video-id", type=str, default=None,
        help="Restrict to one video id prefix",
    )
    args = parser.parse_args()

    with get_connection() as conn, conn.cursor() as cur:
        query = (
            "SELECT id::text, player_matching_gt_json FROM videos "
            "WHERE player_matching_gt_json IS NOT NULL "
            "  AND deleted_at IS NULL"
        )
        params: list[str] = []
        if args.video_id:
            query += " AND id::text LIKE %s"
            params.append(f"{args.video_id}%")
        query += " ORDER BY id::text"
        cur.execute(query, params)
        rows = cur.fetchall()

        if not rows:
            print("No videos with player_matching_gt_json found")
            return 0

        video_ids = [str(r[0]) for r in rows]
        switches_by_video = load_side_switches_from_db(cur, video_ids)

        print(f"Videos with player_matching_gt_json: {len(rows)}")
        print(f"Mode: {'DRY RUN' if args.dry_run else 'APPLY'}\n")
        print(f"{'video_id':10s}  {'legacy':>14s}  {'current_gt':>14s}  {'match':>5s}  {'action':>10s}")
        print("-" * 70)

        n_to_update = 0
        n_disagree = 0
        rows_to_update: list[tuple[str, str]] = []

        for vid_text, gt_json in rows:
            vid = str(vid_text)
            data = gt_json if isinstance(gt_json, dict) else json.loads(str(gt_json))
            had_legacy = (
                "sideSwitches" in data or "side_switches" in data
            )
            legacy_list = list(
                data.get("sideSwitches", data.get("side_switches", []))
            )
            current = list(switches_by_video.get(vid, []))
            matches = sorted(legacy_list) == sorted(current)
            if not matches:
                n_disagree += 1

            if had_legacy:
                action = "drop"
                cleaned = {k: v for k, v in data.items()
                           if k not in ("sideSwitches", "side_switches")}
                if not args.dry_run:
                    rows_to_update.append((vid, json.dumps(cleaned)))
                n_to_update += 1
            else:
                action = "noop"

            match_str = "OK" if matches else "DIFF"
            print(
                f"{vid[:10]:10s}  {str(legacy_list):>14s}  "
                f"{str(current):>14s}  {match_str:>5s}  {action:>10s}"
            )

        print("-" * 70)
        print(f"Disagreements (legacy != current): {n_disagree}/{len(rows)}")
        print(f"Rows that would be updated:        {n_to_update}/{len(rows)}")

        if args.dry_run:
            if n_to_update > 0:
                print("\nRe-run without --dry-run to apply.")
            return 0

        for vid, payload in rows_to_update:
            cur.execute(
                "UPDATE videos SET player_matching_gt_json = %s::jsonb "
                "WHERE id::text = %s",
                (payload, vid),
            )
        conn.commit()
        print(f"\nUpdated {len(rows_to_update)} videos.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
