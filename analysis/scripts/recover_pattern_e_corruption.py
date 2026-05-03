"""Recover videos stuck in Pattern E corruption (frozen first-remap state).

# What's broken

Some production videos were processed with older code that applied
remap-track-ids but did NOT persist `appliedFullMapping` on
`match_analysis_json.rallies[].appliedFullMapping`. The remap rewrote
`positions_json` (renaming raw BoT-SORT IDs to player IDs 1-4), but the
inverse mapping was lost.

Subsequent `match-players` runs see:
  - `positions_json` already in PID space (e.g., tracks {1,2,3,4})
  - `pre_remap_state_json.positions` still in raw BoT-SORT space (e.g.,
    tracks {1,2,3,5})
  - `appliedFullMapping` is NULL → reverse-remap path skipped

The matcher operates on the post-remap positions_json, produces identity
AFM, remap applies identity (no-op), and the FIRST remap's stale
assignment stays frozen forever. ID-stability across rallies is
arbitrary — whatever the first remap set up.

Fleet scan (2026-05-03): 10 of 11 videos with `pre_remap_state_json` +
`positions_json` have at least one corrupted rally. Affected baseline
fixtures: 7d77980f (15/21 rallies), 5c756c41 (6/10), 854bb250 (1/5).
The supposed 7d77980f baseline of 98.5% PERMUTED was an artifact;
the honest blind-mode number is 88.4%.

# What this script does

For each rally with corruption (positions_json track IDs ≠
pre_remap_state_json.positions track IDs):

  1. Restore `positions_json` from `pre_remap_state_json.positions`
     (giving back the original BoT-SORT track IDs).
  2. Restore `primary_track_ids` from
     `pre_remap_state_json.primaryTrackIds`.
  3. Strip `assignmentAnchor` from `match_analysis_json.rallies[]`
     (anchor is keyed on a stale stats hash now).
  4. Strip the rally's `appliedFullMapping` and `remapApplied` flags
     so the next match-players + remap cycle writes a fresh AFM.
  5. NULL `videos.canonical_pid_map_json` (canonical map references
     dead track IDs from the lost remap).

After running, the next `match-players <video>` + `remap-track-ids
<video>` cycle will produce a clean assignment with appliedFullMapping
persisted.

Usage:
    uv run python scripts/recover_pattern_e_corruption.py <video_id>
    uv run python scripts/recover_pattern_e_corruption.py --all-corrupted
    uv run python scripts/recover_pattern_e_corruption.py --dry-run <video_id>
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, cast

from rallycut.evaluation.tracking.db import get_connection


def _detect_corruption(
    video_id: str,
) -> list[tuple[str, list[int], list[int]]]:
    """Return list of (rally_id, pre_track_ids, post_track_ids) for
    rallies stuck in Pattern E corruption.

    The corruption signature is TWO conditions together:
      1. positions_json track IDs differ from pre_remap_state_json track
         IDs — i.e., positions has been remapped at some point.
      2. match_analysis_json.rallies[<rid>].appliedFullMapping is
         missing — i.e., the inverse mapping wasn't preserved, so
         match-players' reverse-remap path can't unwind the rename.

    A healthy post-remap rally satisfies (1) but NOT (2): its
    appliedFullMapping is populated, so the next match-players run can
    reverse-remap to the pristine snapshot before solving. Detecting
    only on (1) would also flag healthy rallies and incorrectly roll
    them back."""
    corrupted: list[tuple[str, list[int], list[int]]] = []
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT match_analysis_json FROM videos WHERE id = %s",
            [video_id],
        )
        ma_row = cur.fetchone()
        ma_dict: dict[str, Any] = {}
        if ma_row and ma_row[0] is not None:
            ma = ma_row[0] if isinstance(ma_row[0], dict) else json.loads(cast(str, ma_row[0]))
            ma_dict = cast(dict, ma)
        afm_present_by_rally: dict[str, bool] = {}
        for entry in ma_dict.get("rallies", []):
            rid = entry.get("rallyId") or entry.get("rally_id")
            if rid:
                afm_present_by_rally[str(rid)] = bool(
                    entry.get("appliedFullMapping")
                )

        cur.execute(
            "SELECT pt.rally_id, pt.positions_json, pt.pre_remap_state_json "
            "FROM player_tracks pt JOIN rallies r ON pt.rally_id = r.id "
            "WHERE r.video_id = %s "
            "  AND pt.positions_json IS NOT NULL "
            "  AND pt.pre_remap_state_json IS NOT NULL",
            [video_id],
        )
        for rid, pj, pre in cur.fetchall():
            rid_str = str(rid)
            pj_data = pj if isinstance(pj, list) else json.loads(cast(str, pj))
            pre_data = pre if isinstance(pre, dict) else json.loads(cast(str, pre))
            pre_pos = pre_data.get("positions") or []
            post_ids = sorted({int(p["trackId"]) for p in pj_data
                              if int(p.get("trackId", -1)) >= 0})
            pre_ids = sorted({int(p["trackId"]) for p in pre_pos
                             if int(p.get("trackId", -1)) >= 0})
            ids_mismatch = pre_ids != post_ids
            afm_missing = not afm_present_by_rally.get(rid_str, False)
            if ids_mismatch and afm_missing:
                corrupted.append((rid_str, pre_ids, post_ids))
    return corrupted


def _list_corrupted_videos() -> list[str]:
    """Videos with at least one rally matching the Pattern E shape.

    Same criterion as `_detect_corruption`: track-ID mismatch AND
    `appliedFullMapping IS NULL`. Healthy post-remap rallies (mismatch
    but AFM populated) are excluded so this is safe to run repeatedly."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            WITH per_rally AS (
              SELECT v.id AS vid,
                     pt.rally_id::text AS rid,
                     (SELECT array_agg(DISTINCT (q ->> 'trackId')::int ORDER BY (q ->> 'trackId')::int)
                      FROM jsonb_array_elements(pt.pre_remap_state_json::jsonb -> 'positions') AS q
                      WHERE (q ->> 'trackId')::int >= 0) AS pre_ids,
                     (SELECT array_agg(DISTINCT (q ->> 'trackId')::int ORDER BY (q ->> 'trackId')::int)
                      FROM jsonb_array_elements(pt.positions_json::jsonb) AS q
                      WHERE (q ->> 'trackId')::int >= 0) AS post_ids,
                     (SELECT (elem -> 'appliedFullMapping' IS NULL)
                      FROM jsonb_array_elements(v.match_analysis_json::jsonb -> 'rallies') AS elem
                      WHERE elem ->> 'rallyId' = pt.rally_id::text) AS afm_null
              FROM player_tracks pt
              JOIN rallies r ON pt.rally_id = r.id
              JOIN videos v ON r.video_id = v.id
              WHERE pt.pre_remap_state_json IS NOT NULL
                AND pt.positions_json IS NOT NULL
                AND v.match_analysis_json IS NOT NULL
            )
            SELECT DISTINCT vid::text FROM per_rally
            WHERE pre_ids IS DISTINCT FROM post_ids
              AND afm_null IS TRUE
            ORDER BY 1
        """)
        return [str(row[0]) for row in cur.fetchall()]


def _recover_video(
    video_id: str,
    *,
    dry_run: bool,
) -> tuple[int, int]:
    """Returns (rallies_recovered, anchors_stripped)."""
    short = video_id[:8]
    corrupted = _detect_corruption(video_id)
    if not corrupted:
        print(f"  {short}: no corruption detected — skipping")
        return 0, 0

    print(f"  {short}: {len(corrupted)} corrupted rallies")
    for rid, pre_ids, post_ids in corrupted:
        print(f"    {rid[:8]}: pre={pre_ids} post={post_ids}")

    if dry_run:
        print(f"  [dry-run] {short}: would recover {len(corrupted)} rallies")
        return len(corrupted), 0

    rally_ids = [rid for rid, _, _ in corrupted]
    rallies_recovered = 0
    anchors_stripped = 0

    with get_connection() as conn, conn.cursor() as cur:
        for rid in rally_ids:
            # Read pre_remap state
            cur.execute(
                "SELECT pre_remap_state_json FROM player_tracks WHERE rally_id = %s",
                [rid],
            )
            row = cur.fetchone()
            if row is None or row[0] is None:
                continue
            pre = row[0] if isinstance(row[0], dict) else json.loads(cast(str, row[0]))
            pre_positions = pre.get("positions") or []
            pre_primary = pre.get("primaryTrackIds") or []
            if not pre_positions:
                print(f"    {rid[:8]}: pre_remap has empty positions — skipping")
                continue

            # Restore positions_json + primary_track_ids
            cur.execute(
                "UPDATE player_tracks SET positions_json = %s, primary_track_ids = %s "
                "WHERE rally_id = %s",
                [
                    json.dumps(pre_positions),
                    json.dumps([int(t) for t in pre_primary]),
                    rid,
                ],
            )
            rallies_recovered += 1

        # Strip assignment anchors + appliedFullMapping + remapApplied for the
        # affected rallies (and clear canonical_pid_map_json on the video).
        cur.execute(
            "SELECT match_analysis_json FROM videos WHERE id = %s",
            [video_id],
        )
        ma_row = cur.fetchone()
        if ma_row and ma_row[0] is not None:
            ma = ma_row[0] if isinstance(ma_row[0], dict) else json.loads(cast(str, ma_row[0]))
            ma_dict: dict[str, Any] = cast(dict, ma)
            for entry in ma_dict.get("rallies", []):
                rid = entry.get("rallyId") or entry.get("rally_id")
                if rid in rally_ids:
                    if entry.pop("assignmentAnchor", None) is not None:
                        anchors_stripped += 1
                    entry.pop("appliedFullMapping", None)
                    entry.pop("remapApplied", None)
            cur.execute(
                "UPDATE videos SET match_analysis_json = %s, "
                "canonical_pid_map_json = NULL WHERE id = %s",
                [json.dumps(ma_dict), video_id],
            )
        conn.commit()

    print(
        f"  {short}: recovered {rallies_recovered} rallies; "
        f"stripped {anchors_stripped} anchors + cleared canonical_pid_map_json"
    )
    return rallies_recovered, anchors_stripped


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("video_ids", nargs="*",
                   help="Video IDs to recover (or use --all-corrupted)")
    p.add_argument("--all-corrupted", action="store_true",
                   help="Recover every video with corruption")
    p.add_argument("--dry-run", action="store_true",
                   help="Report what would change; don't write")
    args = p.parse_args()

    if args.all_corrupted:
        if args.video_ids:
            sys.exit("--all-corrupted is exclusive with explicit video_ids")
        video_ids = _list_corrupted_videos()
        print(f"Recovering {len(video_ids)} corrupted video(s):")
    elif args.video_ids:
        video_ids = list(args.video_ids)
        print(f"Recovering {len(video_ids)} video(s):")
    else:
        sys.exit("Provide video_ids or use --all-corrupted")

    total_rallies = 0
    total_anchors = 0
    for vid in video_ids:
        r, a = _recover_video(vid, dry_run=args.dry_run)
        total_rallies += r
        total_anchors += a

    verb = "would recover" if args.dry_run else "recovered"
    print(
        f"\nSummary: {verb} {total_rallies} rallies; "
        f"stripped {total_anchors} anchors."
    )
    if not args.dry_run and total_rallies > 0:
        print(
            "\nNext step: run match-players + remap-track-ids for each "
            "video to write fresh appliedFullMapping with the new "
            "(clean-state) matcher decisions."
        )


if __name__ == "__main__":
    main()
