"""Repair video(s) corrupted by the 2026-05-11 remap track-drop bug.

Symptom: rallies whose raw BoT-SORT track IDs landed outside ``{1..max_players}``
(e.g. ``{1, 3, 9, 10}`` after mid-rally ID switches) saw the secondary tracks
silently dropped from ``positions_json`` and ``primary_track_ids`` on the
second remap of the pipeline. ``pre_remap_state_json`` still holds the
correct raw-ID snapshot. The matcher's ``assignmentAnchor`` still holds the
correct raw→canonical mapping. The break was in ``remap-track-ids``: it
sourced its mapping from ``trackToPlayer`` (overwritten to canonical-space
identity at the end of the prior run) instead of ``appliedFullMapping`` —
fixed in the same PR as this script.

What this script does, per affected rally:

1. Reads the rally's ``assignmentAnchor.assignment`` (the matcher's correct
   raw→canonical decision, persisted on the prior run).
2. Overwrites the rally entry's ``appliedFullMapping`` with that mapping AND
   sets ``remapApplied=true`` so that the next ``remap-track-ids`` run picks
   it up via the new source-priority logic.
3. After applying #2 for every affected rally in the video, invokes
   ``remap-track-ids`` via the CLI entrypoint (NOT spawning a subprocess —
   in-process so we share the DB connection pool and so errors surface
   immediately). Remap restores positions from ``pre_remap_state_json`` and
   re-applies the correct mapping.

Idempotent: if the rally entry already has a non-identity
``appliedFullMapping`` covering the snapshot's raw IDs, no rewrite needed.

Usage::

    uv run python scripts/repair_remap_track_to_player_drop.py <video_id>
    uv run python scripts/repair_remap_track_to_player_drop.py <video_id> --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, cast

from rallycut.evaluation.db import get_connection


def _load_video(
    video_id: str,
) -> tuple[dict[str, Any], dict[str, list[int]]]:
    """Return ``(match_analysis_json, pre_remap_primary_ids_by_rally)``.

    ``pre_remap_primary_ids_by_rally`` is the snapshot's ``primaryTrackIds``
    per rally — the set of raw tracker IDs that match-players assigned to
    canonical players (and that the anchor must cover for the repair to
    work). Non-primary tracks in the snapshot's positions are noise that the
    remap pipeline handles via collision-shift / drop independently.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT match_analysis_json FROM videos WHERE id = %s",
                [video_id],
            )
            row = cur.fetchone()
            if not row or not row[0]:
                raise SystemExit(
                    f"Video {video_id} has no match_analysis_json — "
                    "run match-players first."
                )
            match_analysis = cast(dict[str, Any], row[0])

            cur.execute(
                """
                SELECT r.id, pt.pre_remap_state_json
                FROM rallies r
                JOIN player_tracks pt ON pt.rally_id = r.id
                WHERE r.video_id = %s
                """,
                [video_id],
            )
            pre_remap_primary_ids: dict[str, list[int]] = {}
            for rid, pre_remap in cur.fetchall():
                if not pre_remap:
                    continue
                pre_remap_dict = cast(dict[str, Any], pre_remap)
                primary = pre_remap_dict.get("primaryTrackIds") or []
                tids = sorted({int(t) for t in primary if int(t) > 0})
                pre_remap_primary_ids[str(rid)] = tids
    return match_analysis, pre_remap_primary_ids


def _needs_repair(
    rally_entry: dict[str, Any],
    snapshot_primary_ids: list[int],
) -> tuple[bool, str]:
    """Decide whether this rally's appliedFullMapping needs to be rewritten.

    The mapping is broken iff any snapshot ``primaryTrackIds`` entry is NOT
    a key in ``appliedFullMapping`` — those primaries would be dropped from
    positions and primary_track_ids on the next remap.
    """
    afm = rally_entry.get("appliedFullMapping") or {}
    afm_keys = {int(k) for k in afm.keys() if int(k) > 0}
    snapshot_real = {tid for tid in snapshot_primary_ids if tid > 0}
    missing = snapshot_real - afm_keys
    if not missing:
        return False, "afm covers all snapshot primary ids"
    return True, f"afm missing {sorted(missing)} from snapshot primaries {sorted(snapshot_real)}"


def _build_repair_mapping(
    rally_entry: dict[str, Any],
    snapshot_primary_ids: list[int],
) -> dict[str, int] | None:
    """Build the appliedFullMapping repair from assignmentAnchor.assignment.

    Returns ``None`` when the anchor is missing or doesn't cover the snapshot.
    """
    anchor = rally_entry.get("assignmentAnchor") or {}
    assignment = anchor.get("assignment") or {}
    if not assignment:
        return None
    anchor_mapping = {int(k): int(v) for k, v in assignment.items() if int(k) > 0}
    snapshot_real = {tid for tid in snapshot_primary_ids if tid > 0}
    if not snapshot_real.issubset(anchor_mapping.keys()):
        # Anchor was computed for a different primary set than the snapshot
        # — not safe to use blindly. (Synthetic / negative IDs are resolved
        # through subTracks separately and aren't expected here.)
        return None
    # Restrict to the snapshot's primary track IDs. Anchor may carry extras
    # from a prior solve on a different track topology that we shouldn't
    # blindly re-apply.
    return {
        str(k): v for k, v in anchor_mapping.items()
        if k in snapshot_real
    }


def _find_affected_videos() -> list[str]:
    """Return video IDs where at least one rally's current
    ``primary_track_ids`` is shorter than the ``primaryTrackIds`` captured
    in ``pre_remap_state_json`` — the signature of the remap track-drop
    bug. Comparing ``primaryTrackIds`` (not snapshot positions trackIds)
    avoids false positives from non-primary noise tracks that the remap
    legitimately drops.
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH rally_pre AS (
                    SELECT pt.rally_id, r.video_id,
                        pt.pre_remap_state_json->'primaryTrackIds' AS pre_pri,
                        pt.primary_track_ids AS cur_pri
                    FROM player_tracks pt
                    JOIN rallies r ON r.id = pt.rally_id
                    WHERE pt.pre_remap_state_json IS NOT NULL
                )
                SELECT DISTINCT video_id FROM rally_pre
                WHERE pre_pri IS NOT NULL
                  AND jsonb_array_length(pre_pri) > jsonb_array_length(cur_pri)
                ORDER BY video_id;
                """
            )
            return [str(row[0]) for row in cur.fetchall()]


def _repair_video(video_id: str, dry_run: bool, quiet: bool = False) -> tuple[int, int, int]:
    """Repair one video. Returns (rallies_repaired, rallies_ok, rallies_skipped)."""
    match_analysis, pre_remap_raw_ids = _load_video(video_id)

    rallies = match_analysis.get("rallies") or []
    repairs: list[tuple[int, str, dict[str, int]]] = []
    ok_count = 0
    skip_count = 0
    for idx, rally_entry in enumerate(rallies):
        rid = rally_entry.get("rallyId") or rally_entry.get("rally_id")
        if not rid:
            continue
        snapshot_primary_ids = pre_remap_raw_ids.get(rid)
        if snapshot_primary_ids is None:
            if not quiet:
                print(f"  [{idx}] {rid[:8]}: SKIP — no pre_remap_state_json snapshot")
            skip_count += 1
            continue
        needs, reason = _needs_repair(rally_entry, snapshot_primary_ids)
        if not needs:
            if not quiet:
                print(f"  [{idx}] {rid[:8]}: OK ({reason})")
            ok_count += 1
            continue
        repair_mapping = _build_repair_mapping(rally_entry, snapshot_primary_ids)
        if repair_mapping is None:
            if not quiet:
                print(f"  [{idx}] {rid[:8]}: SKIP — assignmentAnchor missing "
                      f"or doesn't cover snapshot {sorted(snapshot_primary_ids)}")
            skip_count += 1
            continue
        repairs.append((idx, rid, repair_mapping))
        if not quiet:
            print(f"  [{idx}] {rid[:8]}: REPAIR — {reason}")
            print(f"           new appliedFullMapping = {repair_mapping}")

    if not repairs:
        if not quiet:
            print(f"  {video_id[:8]}: nothing to repair.")
        return 0, ok_count, skip_count

    if dry_run:
        if not quiet:
            print(f"  {video_id[:8]}: would repair {len(repairs)} rallies "
                  f"(dry-run).")
        return len(repairs), ok_count, skip_count

    # Apply repairs to match_analysis_json in-memory then write back.
    for idx, _, repair_mapping in repairs:
        rallies[idx]["appliedFullMapping"] = repair_mapping
        rallies[idx]["remapApplied"] = True

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE videos SET match_analysis_json = %s WHERE id = %s",
                [json.dumps(match_analysis), video_id],
            )
        conn.commit()

    # Invoke remap-track-ids to apply the corrected mappings. Restore-from-
    # snapshot + new source-priority logic produces the correct canonical
    # positions on a single pass.
    from rallycut.cli.commands.remap_track_ids import remap_track_ids_cmd
    remap_track_ids_cmd(
        video_id=video_id,
        dry_run=False,
        quiet=True,  # quiet to keep fleet runs readable
        rally_ids=None,
        reset_snapshot=False,
    )
    return len(repairs), ok_count, skip_count


def main() -> int:
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("video_id", nargs="?",
                   help="Repair a single video by ID.")
    g.add_argument("--all-affected", action="store_true",
                   help="Find all affected videos by DB query and repair "
                        "each in sequence.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned repairs without writing to DB.")
    args = p.parse_args()

    if args.all_affected:
        video_ids = _find_affected_videos()
        print(f"Found {len(video_ids)} affected videos.")
        if not video_ids:
            return 0
        total_repaired = 0
        for i, vid in enumerate(video_ids, 1):
            print(f"\n[{i}/{len(video_ids)}] {vid}")
            try:
                repaired, _, _ = _repair_video(vid, args.dry_run, quiet=False)
                total_repaired += repaired
            except Exception as e:
                print(f"  ERROR repairing {vid}: {e}")
                continue
        print(f"\n=== Fleet repair complete: {total_repaired} rallies "
              f"repaired across {len(video_ids)} videos. ===")
        return 0

    if not args.video_id:
        p.error("Either video_id or --all-affected is required.")
    repaired, ok, skipped = _repair_video(args.video_id, args.dry_run)
    print(f"\n{repaired} repaired, {ok} OK, {skipped} skipped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
