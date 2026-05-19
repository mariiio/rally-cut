"""Probe: WHY do some Serve contacts have empty playerCandidates on 60fps?

For each EMPTY-candidates Serve failure, check:
  1. Is the GT player's track_id in the rally's primary_track_ids?
  2. Does the GT player's track appear ANYWHERE in positions_json for this rally?
  3. If yes, at which frames? How close to the contact frame?
  4. If the GT player's track exists but is filtered out by primary_track_ids,
     we have a match_tracker assignment issue, not a contact_detector issue.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import psycopg

DB_DSN = "postgresql://postgres:postgres@localhost:5436/rallycut"

# 5 specific EMPTY failure cases from probe_serve_attribution
FAILURES = [
    ("haha", "18175bae", 203, 3),
    ("kaka", "ca5d5f57", 169, 3),
    ("kaka", "f33d7ac8", 213, 3),
    ("ruru", "cc2b967b", 224, 3),
    ("yoyo", "9d24aa93", 90, 3),
]


def main() -> int:
    with psycopg.connect(DB_DSN) as conn:
        for vname, rally_prefix, gt_frame, gt_tid in FAILURES:
            cur = conn.execute(
                """
                SELECT r.id, pt.positions_json, pt.primary_track_ids
                FROM rallies r
                JOIN videos v ON r.video_id = v.id
                JOIN player_tracks pt ON pt.rally_id = r.id
                WHERE v.name = %s AND r.id::text LIKE %s || '%%'
                """,
                (vname, rally_prefix),
            )
            row = cur.fetchone()
            if not row:
                print(f"  rally not found: {vname}/{rally_prefix}")
                continue
            rid, positions_json, primary_tids_json = row
            primary_tids = primary_tids_json if isinstance(primary_tids_json, list) else (
                json.loads(primary_tids_json) if primary_tids_json else None
            )
            positions = positions_json if isinstance(positions_json, list) else (
                json.loads(positions_json) if positions_json else []
            )
            in_primary = primary_tids is not None and gt_tid in primary_tids

            # Find all frames where gt_tid appears
            frames_with_gt: list[int] = []
            for p in positions:
                if p.get("trackId") == gt_tid:
                    frames_with_gt.append(p.get("frameNumber", -1))
            frames_with_gt.sort()
            n_gt_frames = len(frames_with_gt)
            min_dist = min(
                (abs(f - gt_frame) for f in frames_with_gt), default=None,
            )
            in_range_15 = sum(1 for f in frames_with_gt if abs(f - gt_frame) <= 15)
            in_range_30 = sum(1 for f in frames_with_gt if abs(f - gt_frame) <= 30)
            in_range_60 = sum(1 for f in frames_with_gt if abs(f - gt_frame) <= 60)
            in_range_120 = sum(1 for f in frames_with_gt if abs(f - gt_frame) <= 120)
            in_range_240 = sum(1 for f in frames_with_gt if abs(f - gt_frame) <= 240)

            # All distinct track IDs in the rally
            all_tids = sorted({p.get("trackId") for p in positions if p.get("trackId") is not None})
            # Tracks near the GT frame
            tids_within_15 = sorted({
                p.get("trackId") for p in positions
                if p.get("trackId") is not None and abs(p.get("frameNumber", -1) - gt_frame) <= 15
            })
            tids_within_30 = sorted({
                p.get("trackId") for p in positions
                if p.get("trackId") is not None and abs(p.get("frameNumber", -1) - gt_frame) <= 30
            })

            print(f"=== {vname} rally={rally_prefix} GT_frame={gt_frame} GT_tid={gt_tid} ===")
            print(f"  primary_track_ids = {primary_tids}")
            print(f"  GT_tid in primary? {in_primary}")
            print(f"  GT_tid appears in positions_json at {n_gt_frames} frames total")
            print(f"  Min frame-distance from GT_frame: {min_dist}")
            print(f"  GT_tid in window ±15: {in_range_15}, ±30: {in_range_30}, "
                  f"±60: {in_range_60}, ±120: {in_range_120}, ±240: {in_range_240}")
            print(f"  All track IDs in rally: {all_tids}")
            print(f"  Tracks within ±15f of GT: {tids_within_15}")
            print(f"  Tracks within ±30f of GT: {tids_within_30}")
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
