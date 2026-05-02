"""Repair stale primary_track_ids on cached pre_remap_state_json.

Older tracking runs persisted `primaryTrackIds` lists containing
sentinels (e.g. -1 for BoT-SORT's unmatched-detection placeholder) or
otherwise-invalid IDs. Today's player_filter excludes negative IDs and
emits a clean 4-element set drawn from real BoT-SORT tracks.

For every player_tracks row of a video, this script:
  1. Loads `pre_remap_state_json.positions` (the cached raw BoT-SORT
     output — never touched after the first remap).
  2. Re-runs `PlayerFilter.analyze_tracks` to derive what today's code
     would call primary.
  3. If the cached `primaryTrackIds` differs from today's derivation,
     writes the fresh set into `pre_remap_state_json.primaryTrackIds`
     AND resets `pt.primary_track_ids` to the same value (the
     subsequent `match-players` + `remap-track-ids` re-run remaps both
     fields back into PID space).

Idempotent. Safe to re-run. `--dry-run` shows what would change.

Usage:
    uv run python scripts/repair_primary_track_ids.py <video_id> [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, cast

from rallycut.evaluation.tracking.db import get_connection
from rallycut.tracking.player_filter import PlayerFilter, PlayerFilterConfig
from rallycut.tracking.player_tracker import PlayerPosition


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video_id")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    config = PlayerFilterConfig()

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT pt.rally_id, pt.frame_count, pt.primary_track_ids,
                   pt.pre_remap_state_json
            FROM player_tracks pt
            JOIN rallies r ON pt.rally_id = r.id
            WHERE r.video_id = %s AND pt.pre_remap_state_json IS NOT NULL
            ORDER BY r.start_ms
            """,
            [args.video_id],
        )
        rows = cur.fetchall()

    if not rows:
        sys.exit(f"no player_tracks rows with pre_remap_state_json for {args.video_id}")

    repairs: list[tuple[str, list[int] | None, list[int] | None, list[int]]] = []
    for rally_id, frame_count, current_primary, snap in rows:
        if isinstance(snap, str):
            snap = json.loads(snap)
        positions_data = (snap or {}).get("positions") or []
        cached_primary = (snap or {}).get("primaryTrackIds")

        positions = [
            PlayerPosition(
                p["frameNumber"], p["trackId"],
                p["x"], p["y"], p["width"], p["height"], p["confidence"],
            )
            for p in positions_data
        ]
        if not positions:
            continue

        pf = PlayerFilter(
            ball_positions=None,
            total_frames=frame_count or max(p.frame_number for p in positions) + 1,
            config=config,
        )
        pf.analyze_tracks(positions)
        fresh = sorted(pf.primary_tracks)

        cached = sorted(cached_primary) if cached_primary else None
        if fresh == cached:
            continue
        repairs.append((str(rally_id), cached, list(current_primary or []), fresh))

    if not repairs:
        print("All rallies already have current-filter primary_track_ids. "
              "Nothing to repair.")
        return

    print(f"Found {len(repairs)} rallies with stale primary_track_ids:\n")
    for rid, cached, current, fresh in repairs:
        print(f"  {rid[:8]}: cached(pre)={cached}  db(post-remap)={current}  "
              f"→ fresh={fresh}")

    if args.dry_run:
        print("\n[dry-run] no changes written.")
        return

    with get_connection() as conn, conn.cursor() as cur:
        for rally_id, _, _, fresh in repairs:
            # Re-load the snapshot, replace primaryTrackIds, write back.
            cur.execute(
                "SELECT pre_remap_state_json FROM player_tracks WHERE rally_id = %s",
                [rally_id],
            )
            row = cur.fetchone()
            snap = row[0] if row else None
            if isinstance(snap, str):
                snap = json.loads(snap)
            snap_dict = cast(dict[str, Any], snap or {})
            snap_dict["primaryTrackIds"] = fresh
            cur.execute(
                """
                UPDATE player_tracks
                SET pre_remap_state_json = %s,
                    primary_track_ids = %s
                WHERE rally_id = %s
                """,
                [json.dumps(snap_dict), json.dumps(fresh), rally_id],
            )
        conn.commit()

    print(f"\nRepaired {len(repairs)} rallies. "
          f"Re-run `match-players` + `remap-track-ids` to propagate the "
          f"fix into match_analysis_json and post-remap positions.")


if __name__ == "__main__":
    main()
