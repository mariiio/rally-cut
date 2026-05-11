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


def _load_video(video_id: str) -> tuple[dict[str, Any], dict[str, list[int]]]:
    """Return ``(match_analysis_json, pre_remap_raw_ids_by_rally)``.

    ``pre_remap_raw_ids_by_rally`` is the set of raw tracker IDs that lived in
    ``pre_remap_state_json.positions`` for each rally — the ground-truth set
    of IDs the mapping must cover.
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
            pre_remap_raw_ids: dict[str, list[int]] = {}
            for rid, pre_remap in cur.fetchall():
                if not pre_remap:
                    continue
                pre_remap_dict = cast(dict[str, Any], pre_remap)
                positions = pre_remap_dict.get("positions") or []
                tids = sorted({int(p["trackId"]) for p in positions
                               if p.get("trackId") is not None})
                pre_remap_raw_ids[str(rid)] = tids
    return match_analysis, pre_remap_raw_ids


def _needs_repair(
    rally_entry: dict[str, Any],
    snapshot_raw_ids: list[int],
) -> tuple[bool, str]:
    """Decide whether this rally's appliedFullMapping needs to be rewritten.

    The mapping is broken iff any real (non-synthetic) raw track ID from
    the snapshot is NOT a key in ``appliedFullMapping``. Synthetic sub-track
    IDs (negative numbers) are resolved through ``subTracks``, not via
    ``trackToPlayer`` / ``appliedFullMapping``.
    """
    afm = rally_entry.get("appliedFullMapping") or {}
    afm_keys = {int(k) for k in afm.keys() if int(k) > 0}
    snapshot_real = {tid for tid in snapshot_raw_ids if tid > 0}
    missing = snapshot_real - afm_keys
    if not missing:
        return False, "afm covers all snapshot raw ids"
    return True, f"afm missing {sorted(missing)} from snapshot {sorted(snapshot_real)}"


def _build_repair_mapping(
    rally_entry: dict[str, Any],
    snapshot_raw_ids: list[int],
) -> dict[str, int] | None:
    """Build the appliedFullMapping repair from assignmentAnchor.assignment.

    Returns ``None`` when the anchor is missing or doesn't cover the snapshot.
    """
    anchor = rally_entry.get("assignmentAnchor") or {}
    assignment = anchor.get("assignment") or {}
    if not assignment:
        return None
    anchor_mapping = {int(k): int(v) for k, v in assignment.items() if int(k) > 0}
    snapshot_real = {tid for tid in snapshot_raw_ids if tid > 0}
    if not snapshot_real.issubset(anchor_mapping.keys()):
        # Anchor was computed for a different track set than the snapshot —
        # not safe to use blindly. Negative (synthetic) IDs in the snapshot
        # are flow through subTracks separately and are not expected to
        # appear as keys here.
        return None
    # Restrict to keys that exist in the snapshot (anchor may carry extras
    # from synthetic sub-tracks).
    return {
        str(k): v for k, v in anchor_mapping.items()
        if k in snapshot_real
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("video_id")
    p.add_argument("--dry-run", action="store_true",
                   help="Print planned repairs without writing to DB.")
    args = p.parse_args()

    match_analysis, pre_remap_raw_ids = _load_video(args.video_id)

    rallies = match_analysis.get("rallies") or []
    repairs: list[tuple[int, str, dict[str, int]]] = []  # (idx, rid, mapping)
    for idx, rally_entry in enumerate(rallies):
        rid = rally_entry.get("rallyId") or rally_entry.get("rally_id")
        if not rid:
            continue
        snapshot_raw_ids = pre_remap_raw_ids.get(rid)
        if snapshot_raw_ids is None:
            print(f"  [{idx}] {rid[:8]}: SKIP — no pre_remap_state_json snapshot")
            continue
        needs, reason = _needs_repair(rally_entry, snapshot_raw_ids)
        if not needs:
            print(f"  [{idx}] {rid[:8]}: OK ({reason})")
            continue
        repair_mapping = _build_repair_mapping(rally_entry, snapshot_raw_ids)
        if repair_mapping is None:
            print(f"  [{idx}] {rid[:8]}: SKIP — assignmentAnchor missing or "
                  f"doesn't cover snapshot {sorted(snapshot_raw_ids)}")
            continue
        repairs.append((idx, rid, repair_mapping))
        print(f"  [{idx}] {rid[:8]}: REPAIR — {reason}")
        print(f"           new appliedFullMapping = {repair_mapping}")

    if not repairs:
        print("\nNothing to repair.")
        return 0

    print(f"\nPlanned: {len(repairs)} rallies will get appliedFullMapping "
          f"rewritten + remapApplied=true.")
    if args.dry_run:
        print("(dry-run; no DB writes)")
        return 0

    # Apply repairs to match_analysis_json in-memory then write back.
    for idx, _, repair_mapping in repairs:
        rallies[idx]["appliedFullMapping"] = repair_mapping
        rallies[idx]["remapApplied"] = True

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE videos SET match_analysis_json = %s WHERE id = %s",
                [json.dumps(match_analysis), args.video_id],
            )
        conn.commit()
    print(f"\nWrote {len(repairs)} appliedFullMapping repairs to "
          f"videos.match_analysis_json.")

    print("\nNow invoking remap-track-ids to apply the corrected mappings...")
    # Lazy import to keep CLI startup lean — pulls heavy deps.
    from rallycut.cli.commands.remap_track_ids import remap_track_ids_cmd
    # Direct call (skip Typer's argument parsing); remap_track_ids_cmd will
    # restore positions from pre_remap_state_json and apply the repaired
    # appliedFullMapping via the new source-priority logic.
    remap_track_ids_cmd(
        video_id=args.video_id,
        dry_run=False,
        quiet=False,
        rally_ids=None,
        reset_snapshot=False,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
