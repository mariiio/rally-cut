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
    uv run python scripts/repair_primary_track_ids.py --all [--dry-run]
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
    ap.add_argument("video_id", nargs="?", default=None)
    ap.add_argument("--all", action="store_true",
                    help="Repair every video that has cached pre_remap_state_json")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.video_id and not args.all:
        sys.exit("specify <video_id> or --all")

    config = PlayerFilterConfig()

    with get_connection() as conn, conn.cursor() as cur:
        if args.all:
            cur.execute(
                """
                SELECT pt.rally_id, pt.frame_count, pt.primary_track_ids,
                       pt.pre_remap_state_json, r.video_id
                FROM player_tracks pt
                JOIN rallies r ON pt.rally_id = r.id
                WHERE pt.pre_remap_state_json IS NOT NULL
                ORDER BY r.video_id, r.start_ms
                """,
            )
        else:
            cur.execute(
                """
                SELECT pt.rally_id, pt.frame_count, pt.primary_track_ids,
                       pt.pre_remap_state_json, r.video_id
                FROM player_tracks pt
                JOIN rallies r ON pt.rally_id = r.id
                WHERE r.video_id = %s AND pt.pre_remap_state_json IS NOT NULL
                ORDER BY r.start_ms
                """,
                [args.video_id],
            )
        rows = cur.fetchall()

    if not rows:
        sys.exit(
            "no player_tracks rows with pre_remap_state_json"
            + (f" for {args.video_id}" if args.video_id else "")
        )

    repairs: list[
        tuple[str, str, list[int] | None, list[int] | None, list[int]]
    ] = []
    for rally_id, frame_count, current_primary, snap, video_id in rows:
        if isinstance(snap, str):
            snap = json.loads(snap)
        snap_obj: dict[str, Any] = (
            cast(dict[str, Any], snap) if isinstance(snap, dict) else {}
        )
        positions_data: list[dict[str, Any]] = (
            snap_obj.get("positions") or []
        )
        cached_primary = snap_obj.get("primaryTrackIds")

        positions = [
            PlayerPosition(
                p["frameNumber"], p["trackId"],
                p["x"], p["y"], p["width"], p["height"], p["confidence"],
            )
            for p in positions_data
        ]
        if not positions:
            continue

        total_frames = (
            int(cast(int, frame_count)) if frame_count
            else max(p.frame_number for p in positions) + 1
        )
        pf = PlayerFilter(
            ball_positions=None,
            total_frames=total_frames,
            config=config,
        )
        pf.analyze_tracks(positions)
        fresh = sorted(pf.primary_tracks)

        cached_list_for_sort = (
            cast(list[int], cached_primary) if cached_primary else None
        )
        cached = sorted(cached_list_for_sort) if cached_list_for_sort else None
        if fresh == cached:
            continue

        # Only auto-repair the unambiguous corruption shapes:
        #   - negative sentinel ids (BoT-SORT's "unmatched" placeholder)
        #   - duplicate ids
        #   - fewer than max_players valid ids when fresh has more
        #   - primary id that doesn't exist in the cached positions data
        #     (silent corruption — matcher will drop the track and emit
        #     fewer than 4 PIDs)
        #
        # When cached and fresh are BOTH 4 distinct positive ids that
        # disagree AND every cached id exists in positions, that's a
        # filter-behavior diff, not corruption — leave it alone.
        cached_list = list(cached_primary) if cached_primary else []
        cached_valid = [t for t in cached_list if isinstance(t, int) and t >= 0]
        positions_track_ids = {p["trackId"] for p in positions_data}
        orphan_primaries = [
            t for t in cached_valid if t not in positions_track_ids
        ]
        has_corruption = (
            any(t < 0 for t in cached_list if isinstance(t, int))
            or len(cached_list) != len(set(cached_list))
            or len(cached_valid) < len(fresh)
            or bool(orphan_primaries)
        )
        if not has_corruption:
            print(f"  skipping {str(video_id)[:8]}/{str(rally_id)[:8]}: "
                  f"cached={cached} vs fresh={fresh} differs but neither "
                  f"is corrupted — manual review needed")
            continue
        if orphan_primaries:
            print(f"  {str(video_id)[:8]}/{str(rally_id)[:8]}: cached "
                  f"primaries {sorted(orphan_primaries)} have no positions "
                  f"in the snapshot — corruption, will repair")
        current_list = (
            list(cast(list[int], current_primary)) if current_primary else []
        )
        repairs.append(
            (str(video_id), str(rally_id), cached, current_list, fresh),
        )

    if not repairs:
        print("No corrupted primary_track_ids to repair.")
        return

    affected_videos = sorted({r[0] for r in repairs})
    print(f"Found {len(repairs)} stale rallies across "
          f"{len(affected_videos)} video(s):\n")
    for vid, rid, cached, current, fresh in repairs:
        print(f"  {vid[:8]}/{rid[:8]}: cached(pre)={cached}  "
              f"db(post-remap)={current}  → fresh={fresh}")

    if args.dry_run:
        print("\n[dry-run] no changes written.")
        return

    with get_connection() as conn, conn.cursor() as cur:
        for _, rally_id, _, _, fresh in repairs:
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

        # Strip the assignmentAnchor for each repaired rally from
        # videos.match_analysis_json. Without this, the next match-players
        # run would see the prior anchor (computed against the OLD broken
        # primary_track_ids) and pin the bad assignment instead of
        # re-solving on the freshly-corrected input. Without this step
        # the user has to remember `--reset-anchors`; with it, the next
        # run "just works."
        affected_rids = {rid for _, rid, _, _, _ in repairs}
        for vid in affected_videos:
            cur.execute(
                "SELECT match_analysis_json FROM videos WHERE id = %s", [vid],
            )
            row = cur.fetchone()
            ma = row[0] if row else None
            if isinstance(ma, str):
                ma = json.loads(ma)
            ma_dict = cast(dict[str, Any], ma or {})
            stripped = 0
            for entry in ma_dict.get("rallies", []):
                rid = entry.get("rallyId") or entry.get("rally_id")
                if rid in affected_rids and entry.pop("assignmentAnchor", None):
                    stripped += 1
            if stripped:
                cur.execute(
                    "UPDATE videos SET match_analysis_json = %s WHERE id = %s",
                    [json.dumps(ma_dict), vid],
                )
                print(f"  Stripped {stripped} stale assignmentAnchor "
                      f"entry/entries from videos {vid[:8]}.")

        conn.commit()

    print(f"\nRepaired {len(repairs)} rallies across "
          f"{len(affected_videos)} video(s). Affected anchors stripped, "
          f"so the next match-players run will re-solve cleanly without "
          f"--reset-anchors:")
    for vid in affected_videos:
        print(f"  uv run rallycut match-players {vid}")
        print(f"  uv run rallycut remap-track-ids {vid}")


if __name__ == "__main__":
    main()
