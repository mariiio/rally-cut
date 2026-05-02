"""Remap per-rally track IDs to consistent match-level player IDs.

After match-players assigns consistent player IDs (1-4) across rallies,
this command remaps all stored track IDs in positions_json, contacts_json,
actions_json, primaryTrackIds, and action_ground_truth_json so the UI shows
consistent identities.

Handles collisions: unmapped tracks that would collide with remapped player
IDs are shifted to high IDs (101+) to avoid conflicts.

Usage:
    rallycut remap-track-ids <video-id>
    rallycut remap-track-ids <video-id> --dry-run
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, cast

import typer
from rich.console import Console

from rallycut.cli.utils import handle_errors

console = Console()


# Sentinel value written to `trackId` / `playerTrackId` when a position or
# action falls inside a parent track that was split by within-track
# segmentation BUT outside any kept sub-track segment (i.e. its sub-track
# lost the pid conflict in `_apply_subtrack_assignments`). Matches the
# existing convention in `tracking/global_identity.py`,
# `tracking/contact_detector.py`, and `tracking/player_tracker.py` where
# `track_id < 0` means "drop / unknown".
UNLABELED_TRACK_ID = -1


@dataclass
class RallyRemapPlan:
    """Per-rally remap plan with optional frame-conditional sub-track overrides.

    `flat_mapping` is the un-split, real track→pid mapping (synthetic
    sub-track ids stripped). `sub_track_overrides` carries the
    frame-conditional pid resolution for parent tracks that were split by
    within-track segmentation: each entry routes a parent_track_id, when
    f_start ≤ frame ≤ f_end, to its winning pid. Frames inside a split
    parent that fall outside every kept sub-track segment are written as
    `UNLABELED_TRACK_ID` (the lost-conflict sub-track's frames have no pid
    resolution by design — north-star: miss > wrong).
    """

    flat_mapping: dict[int, int]
    sub_track_overrides: list[dict[str, int]] = field(default_factory=list)


def _build_remap_plan_for_rally(rally_entry: dict[str, Any]) -> RallyRemapPlan:
    """Build a RallyRemapPlan from a `match_analysis_json` rally entry.

    Reads `trackToPlayer` (real tracks → pid) and the optional `subTracks`
    (parent + frame range → pid). Strips negative (synthetic) ids from the
    flat mapping defensively — `match_players.py` already strips them, but
    this guards against legacy rally entries from before that change.
    """
    track_to_player = rally_entry.get("trackToPlayer") or rally_entry.get(
        "track_to_player", {}
    )
    flat: dict[int, int] = {}
    for k, v in track_to_player.items():
        tid = int(k)
        if tid > 0:
            flat[tid] = int(v)
    sub_tracks = rally_entry.get("subTracks") or []
    overrides: list[dict[str, int]] = [
        {
            "parent_track_id": int(s["parentTrackId"]),
            "f_start": int(s["fStart"]),
            "f_end": int(s["fEnd"]),
            "pid": int(s["pid"]),
        }
        for s in sub_tracks
        if s.get("pid") is not None
    ]
    return RallyRemapPlan(flat_mapping=flat, sub_track_overrides=overrides)


def _invert_mapping(mapping: dict[int, int]) -> dict[int, int]:
    """Invert a bijective {old: new} mapping. Raises ValueError if non-bijective."""
    inverse: dict[int, int] = {}
    for k, v in mapping.items():
        if v in inverse:
            raise ValueError(
                f"Non-bijective mapping: both {inverse[v]} and {k} map to {v}"
            )
        inverse[v] = k
    return inverse


def _should_reverse(
    positions: list[dict[str, Any]],
    applied_mapping: dict[int, int],
) -> bool:
    """Check if current position IDs match the output of applied_mapping.

    If they do, the positions are remapped and should be reversed.
    If they don't (e.g. after re-tracking), positions already have original IDs.

    Uses subset check: current IDs must be a subset of the mapping's output values
    AND must NOT be a subset of the mapping's input values (to distinguish remapped
    data from fresh tracker data that happens to use the same low IDs).
    """
    if not applied_mapping:
        return False
    mapped_output_ids = set(applied_mapping.values())
    mapped_input_ids = set(applied_mapping.keys())
    current_ids: set[int] = set()
    for p in positions:
        tid = p.get("trackId")
        if tid is not None:
            tid_int = int(tid)
            # Skip the unlabeled sentinel — it's never an output of any
            # legitimate forward mapping, so including it would always
            # break the subset check.
            if tid_int == UNLABELED_TRACK_ID:
                continue
            current_ids.add(tid_int)
    if not current_ids:
        return False
    is_subset_of_output = current_ids.issubset(mapped_output_ids)
    is_subset_of_input = current_ids.issubset(mapped_input_ids)
    # If current IDs match output but NOT input, data was remapped.
    # If they match both (e.g. identity mapping or overlapping ranges), rely on
    # the remapApplied flag (checked by callers) to disambiguate.
    if is_subset_of_output and not is_subset_of_input:
        return True
    if is_subset_of_output and is_subset_of_input:
        # Ambiguous — could be either. Return True only if the mapping is
        # non-trivial (not identity), since identity mappings don't need reversal.
        return any(k != v for k, v in applied_mapping.items())
    return False


def _build_full_mapping(
    track_to_player: dict[int, int],
    all_track_ids: set[int],
    sub_track_pids: set[int] | None = None,
) -> dict[int, int]:
    """Build collision-safe mapping for ALL track IDs in a rally.

    Mapped tracks get their player IDs (1-4).
    Unmapped tracks keep their ID if no collision, otherwise shift to 101+.

    `sub_track_pids` carries pids claimed by sub-tracks (from within-track
    segmentation). When provided, an unmapped track whose ID equals one of
    those pids would render as a phantom duplicate label alongside the
    sub-track's bbox at the same frame, so we treat its ID as colliding
    and shift it (or, more practically, the caller passes
    `unmapped_to_unlabeled=True` to `_resolve_with_overrides` which writes
    `UNLABELED_TRACK_ID` for them — see below). To make that path fire,
    such tracks must NOT be added to `mapping` here. We exclude them.
    """
    mapping: dict[int, int] = {}
    used_ids = set(track_to_player.values())
    sub_track_pids = sub_track_pids or set()

    # First, add the explicit track→player mappings
    for tid, pid in track_to_player.items():
        mapping[tid] = pid

    # Then handle unmapped tracks
    next_shifted = 101
    for tid in sorted(all_track_ids):
        if tid in mapping:
            continue  # Already mapped
        if tid in sub_track_pids:
            # Pid claimed by a sub-track. Don't map this real track to its
            # identity (or anything) — leave it absent from `mapping` so
            # `_resolve_with_overrides` falls through to `unmapped_to_unlabeled`
            # and writes UNLABELED_TRACK_ID for these positions. Otherwise
            # they'd render as a duplicate pid label at the same frames.
            continue
        if tid in used_ids:
            # Collision: this track ID conflicts with a mapped player ID
            while next_shifted in all_track_ids or next_shifted in used_ids:
                next_shifted += 1
            mapping[tid] = next_shifted
            next_shifted += 1
        else:
            # No collision, keep original ID
            mapping[tid] = tid

    return mapping


def _index_overrides_by_parent(
    overrides: list[dict[str, int]],
) -> dict[int, list[dict[str, int]]]:
    """Group sub-track overrides by parent_track_id."""
    out: dict[int, list[dict[str, int]]] = {}
    for ov in overrides:
        out.setdefault(ov["parent_track_id"], []).append(ov)
    return out


def _resolve_with_overrides(
    old_id_int: int,
    frame: int | None,
    mapping: dict[int, int],
    overrides_by_parent: dict[int, list[dict[str, int]]],
    unmapped_to_unlabeled: bool = False,
) -> int | None:
    """Resolve a track id to its post-remap value.

    Order of precedence:
    1. If the id is a split parent (in `overrides_by_parent`), search for a
       sub-track segment whose [f_start, f_end] contains `frame`. Returns
       the segment's pid on hit; returns `UNLABELED_TRACK_ID` when frame
       falls outside every kept segment.
    2. Otherwise, look up the flat mapping. Returns the new id, or `None`
       when the id is absent from the mapping AND `unmapped_to_unlabeled`
       is False (caller leaves it untouched). When `unmapped_to_unlabeled`
       is True (set by remap_positions when sub-tracks exist), absent
       tracks resolve to `UNLABELED_TRACK_ID` — this prevents real tracks
       whose pid was claimed by a sub-track from rendering as a phantom
       duplicate label in positions_json.

    A `None` `frame` for a split-parent id is treated as "frame unknown" →
    returns `UNLABELED_TRACK_ID` (we can't resolve sub-tracks without a
    frame).
    """
    if old_id_int in overrides_by_parent:
        if frame is None:
            return UNLABELED_TRACK_ID
        for ov in overrides_by_parent[old_id_int]:
            if ov["f_start"] <= frame <= ov["f_end"]:
                return ov["pid"]
        return UNLABELED_TRACK_ID
    if old_id_int in mapping:
        return mapping[old_id_int]
    if unmapped_to_unlabeled:
        return UNLABELED_TRACK_ID
    return None


def _remap_positions(
    positions: list[dict[str, Any]],
    mapping: dict[int, int],
    sub_track_overrides: list[dict[str, int]] | None = None,
) -> int:
    """Remap `trackId` in positions list. Returns count of remapped entries.

    When `sub_track_overrides` is provided, the resolution becomes
    frame-conditional for parent tracks that were split by within-track
    segmentation. Frames inside a split parent but outside every kept
    sub-track segment are written as `UNLABELED_TRACK_ID`.
    """
    overrides_by_parent = _index_overrides_by_parent(sub_track_overrides or [])
    # When sub-tracks are present, an unmapped real track means its pid was
    # claimed by a sub-track and Hungarian had no remaining pid to assign it.
    # Mark those positions as unlabeled so they don't render as a phantom
    # duplicate label alongside the sub-track's bbox at the same frame.
    unmapped_to_unlabeled = bool(overrides_by_parent)
    count = 0
    for p in positions:
        old_id = p.get("trackId")
        if old_id is None:
            continue
        old_id_int = int(old_id)
        frame = p.get("frameNumber")
        frame_int = int(frame) if frame is not None else None
        new_id = _resolve_with_overrides(
            old_id_int, frame_int, mapping, overrides_by_parent,
            unmapped_to_unlabeled=unmapped_to_unlabeled,
        )
        if new_id is None:
            continue
        if new_id != old_id_int:
            p["trackId"] = new_id
            count += 1
    return count


def _remap_contacts(
    contacts_data: dict[str, Any],
    mapping: dict[int, int],
    sub_track_overrides: list[dict[str, int]] | None = None,
) -> int:
    """Remap `playerTrackId` and `playerCandidates` in contacts.

    Frame-conditional override of split-parent ids uses each contact's
    `frame` field. Candidates lack per-frame context; when a candidate id
    is a split parent, we use the contact's own `frame` (best available
    proxy). Returns count of remapped main `playerTrackId` entries (matches
    pre-existing return semantics).
    """
    overrides_by_parent = _index_overrides_by_parent(sub_track_overrides or [])
    count = 0
    for c in contacts_data.get("contacts", []):
        frame = c.get("frame")
        frame_int = int(frame) if frame is not None else None
        old_id = c.get("playerTrackId")
        if old_id is not None:
            old_id_int = int(old_id)
            new_id = _resolve_with_overrides(
                old_id_int, frame_int, mapping, overrides_by_parent
            )
            if new_id is not None and new_id != old_id_int:
                c["playerTrackId"] = new_id
                count += 1
        # Also remap candidates — use the contact's frame as the resolution
        # context (candidates don't carry their own frame).
        for cand in c.get("playerCandidates", []):
            if isinstance(cand, list) and len(cand) >= 1:
                cand_id = cand[0]
                if cand_id is None:
                    continue
                resolved = _resolve_with_overrides(
                    int(cand_id), frame_int, mapping, overrides_by_parent
                )
                if resolved is not None:
                    cand[0] = resolved
    return count


def _remap_actions(
    actions_data: dict[str, Any],
    mapping: dict[int, int],
    sub_track_overrides: list[dict[str, int]] | None = None,
) -> int:
    """Remap `playerTrackId` in actions and `teamAssignments`.

    Each action carries a `frame` field — used for frame-conditional
    resolution of split-parent ids. `teamAssignments` is keyed on track id
    with no frame context: split-parent keys are dropped (a team label
    can't be assigned to a parent whose pid varies per-segment); per-segment
    pids inherit their team via the position remap downstream.
    """
    overrides_by_parent = _index_overrides_by_parent(sub_track_overrides or [])
    count = 0
    for a in actions_data.get("actions", []):
        frame = a.get("frame")
        frame_int = int(frame) if frame is not None else None
        old_id = a.get("playerTrackId")
        if old_id is None:
            continue
        old_id_int = int(old_id)
        new_id = _resolve_with_overrides(
            old_id_int, frame_int, mapping, overrides_by_parent
        )
        if new_id is not None and new_id != old_id_int:
            a["playerTrackId"] = new_id
            count += 1

    # Remap teamAssignments keys. Split parents (in overrides_by_parent)
    # are dropped — their team is per-segment, not constant.
    old_ta = actions_data.get("teamAssignments")
    if old_ta and isinstance(old_ta, dict):
        new_ta: dict[str, str] = {}
        for tid_str, team_label in old_ta.items():
            tid = int(tid_str)
            if tid in overrides_by_parent:
                # Skip — pid (and therefore team) varies by segment for
                # split parents. Per-segment pids will appear directly in
                # the remapped action stream and inherit team via existing
                # team_assignment downstream paths.
                continue
            new_tid = mapping.get(tid, tid)
            new_ta[str(new_tid)] = team_label
        actions_data["teamAssignments"] = new_ta

    return count


_SNAPSHOT_FIELDS = (
    "positions",
    "contacts",
    "actions",
    "primaryTrackIds",
    "actionGroundTruth",
)


def _capture_snapshot(
    positions_json: Any,
    contacts_json: Any,
    actions_json: Any,
    primary_ids: Any,
    action_gt_json: Any,
) -> dict[str, Any]:
    """Bundle the five remap-mutated fields into one snapshot dict.

    Storing them together (rather than in separate columns) keeps the
    snapshot atomic: we never half-restore. None values are preserved as
    None so the restore round-trip is exact.
    """
    return {
        "positions": positions_json,
        "contacts": contacts_json,
        "actions": actions_json,
        "primaryTrackIds": primary_ids,
        "actionGroundTruth": action_gt_json,
    }


def _restore_snapshot(
    snapshot: dict[str, Any],
) -> tuple[Any, Any, Any, Any, Any]:
    """Inverse of _capture_snapshot.

    Returns (positions, contacts, actions, primary_ids, action_gt) in the
    same order the SELECT uses, so the caller can reuse it directly.
    """
    return (
        snapshot.get("positions"),
        snapshot.get("contacts"),
        snapshot.get("actions"),
        snapshot.get("primaryTrackIds"),
        snapshot.get("actionGroundTruth"),
    )


def _deepcopy_json(value: Any) -> Any:
    """Round-trip through JSON to deep-copy a snapshotted value.

    Faster than copy.deepcopy for plain JSON structures and guarantees the
    working copy is independent of the snapshot dict (so mutations during
    remap don't leak back into pre_remap_state_json on write).
    """
    if value is None:
        return None
    return json.loads(json.dumps(value))


@handle_errors
def remap_track_ids_cmd(
    video_id: str = typer.Argument(
        ...,
        help="Video ID to remap track IDs for",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would change without updating DB",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet", "-q",
        help="Suppress progress output",
    ),
    rally_ids: str | None = typer.Option(
        None,
        "--rally-ids",
        help="Comma-separated rally UUIDs to process. If omitted, all tracked rallies in the video are processed.",
    ),
    reset_snapshot: bool = typer.Option(
        False,
        "--reset-snapshot",
        help=(
            "Re-snapshot pre_remap_state_json from current row state on this "
            "run. Use after match-players produces a new canonical permutation "
            "and you want subsequent remap-track-ids runs to treat the current "
            "values as the new pristine baseline."
        ),
    ),
) -> None:
    """Remap per-rally track IDs to consistent match-level player IDs (1-4).

    Uses the trackToPlayer mapping from match-players to remap all stored
    track IDs so the UI shows consistent player identities across rallies.

    Example:
        rallycut remap-track-ids abc123
        rallycut remap-track-ids abc123 --dry-run
    """
    from rallycut.evaluation.db import get_connection

    # Load match analysis
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT match_analysis_json, canonical_pid_map_json "
                "FROM videos WHERE id = %s",
                [video_id],
            )
            row = cur.fetchone()

    if not row or not row[0]:
        console.print(
            "[red]Error:[/red] No match analysis found. "
            "Run 'rallycut match-players' first."
        )
        raise typer.Exit(1)

    match_analysis = cast(dict[str, Any], row[0])
    canonical_pid_map_json = cast(dict[str, Any] | None, row[1])

    # Per-rally canonical map (ref-crop sourced). When present for a rally,
    # it WINS over trackToPlayer — `appliedFullMapping` becomes a synonym
    # for `canonicalPidMap.rallies[rid]`. This collapses the two pid sources
    # into one and prevents the legacy Hungarian permutation from drifting
    # the displayed pid badge across re-runs after the user has uploaded
    # ref crops.
    canonical_per_rally: dict[str, dict[int, int]] = {}
    if isinstance(canonical_pid_map_json, dict):
        for rid_c, m in canonical_pid_map_json.get("rallies", {}).items():
            if rid_c and m:
                canonical_per_rally[rid_c] = {int(k): int(v) for k, v in m.items()}

    # Build per-rally track→player mappings and index rally entries.
    # Source priority: canonical_per_rally → matchAnalysisJson.trackToPlayer.
    # Sub-track overrides (frame-conditional pid resolution for split parents)
    # come from the rally entry's `subTracks` field regardless of which
    # mapping source wins.
    raw_mappings: dict[str, dict[int, int]] = {}
    rally_overrides: dict[str, list[dict[str, int]]] = {}
    rally_entries_by_id: dict[str, dict[str, Any]] = {}
    for rally_entry in match_analysis.get("rallies", []):
        rid = rally_entry.get("rallyId") or rally_entry.get("rally_id", "")
        track_to_player = rally_entry.get("trackToPlayer") or rally_entry.get(
            "track_to_player", {}
        )
        if rid:
            rally_entries_by_id[rid] = rally_entry
            canonical_for_rally = canonical_per_rally.get(rid)
            if canonical_for_rally:
                raw_mappings[rid] = canonical_for_rally
            elif track_to_player:
                # Strip negative (synthetic sub-track) ids defensively. New
                # match_analysis_json never contains them in trackToPlayer
                # but legacy entries from before the strip may.
                raw_mappings[rid] = {
                    int(k): int(v) for k, v in track_to_player.items()
                    if int(k) > 0
                }
            plan = _build_remap_plan_for_rally(rally_entry)
            if plan.sub_track_overrides:
                rally_overrides[rid] = plan.sub_track_overrides

    if not raw_mappings:
        console.print("[yellow]No track mappings found in match analysis[/yellow]")
        return

    # Parse --rally-ids filter
    rally_id_filter: list[str] | None = (
        [s.strip() for s in rally_ids.split(",") if s.strip()]
        if rally_ids else None
    )
    if rally_id_filter is not None:
        raw_mappings = {k: v for k, v in raw_mappings.items() if k in rally_id_filter}
        rally_entries_by_id = {k: v for k, v in rally_entries_by_id.items() if k in rally_id_filter}

    if not quiet:
        console.print(
            f"[bold]Remapping track IDs[/bold] for video {video_id[:8]}..."
        )
        console.print(f"  {len(raw_mappings)} rallies with mappings")

    # Load all player tracks for this video. pre_remap_state_json is the
    # idempotency anchor: when present, it carries the pristine pre-remap
    # snapshot of {positions, contacts, actions, primaryTrackIds,
    # actionGroundTruth}; we restore from it on every run so the mapping is
    # always applied to the same input.
    rally_ids_list = list(raw_mappings.keys())
    placeholders = ", ".join(["%s"] * len(rally_ids_list))
    query = f"""
        SELECT r.id, pt.id, pt.positions_json, pt.contacts_json,
               pt.actions_json, pt.primary_track_ids,
               pt.action_ground_truth_json, pt.pre_remap_state_json
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE r.id IN ({placeholders})
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, rally_ids_list)
            rows = cur.fetchall()

    total_remapped = 0
    updates: list[tuple[int, dict[str, Any]]] = []  # (pt_id, {column: value})

    for (
        rally_id_val,
        pt_id_val,
        pos_json,
        contacts_json,
        actions_json,
        primary_ids,
        action_gt_json,
        pre_remap_state,
    ) in rows:
        rally_id = str(rally_id_val)
        pt_id = cast(int, pt_id_val)
        raw_mapping = raw_mappings.get(rally_id, {})
        if not raw_mapping:
            continue
        sub_track_overrides = rally_overrides.get(rally_id, [])
        rally_entry = rally_entries_by_id.get(rally_id, {})

        # --- Step 1: Snapshot or restore. ---
        # First run (or --reset-snapshot): capture current row values as the
        # pristine snapshot. Subsequent runs: restore working values from the
        # snapshot so the forward mapping always applies to the same input.
        # All field assignments below are "working copies" — independent of
        # the snapshot dict, so later mutations don't leak back into
        # pre_remap_state_json on write.
        snapshot_payload: dict[str, Any] | None = (
            cast(dict[str, Any], pre_remap_state) if pre_remap_state else None
        )
        write_snapshot = False
        if snapshot_payload is None or reset_snapshot:
            snapshot_payload = _capture_snapshot(
                pos_json, contacts_json, actions_json, primary_ids, action_gt_json,
            )
            write_snapshot = True

        pos_json, contacts_json, actions_json, primary_ids, action_gt_json = (
            _deepcopy_json(snapshot_payload.get("positions")),
            _deepcopy_json(snapshot_payload.get("contacts")),
            _deepcopy_json(snapshot_payload.get("actions")),
            _deepcopy_json(snapshot_payload.get("primaryTrackIds")),
            _deepcopy_json(snapshot_payload.get("actionGroundTruth")),
        )

        # --- Step 2: Collect all track IDs (now original IDs) ---
        all_track_ids: set[int] = set()
        if pos_json:
            for p in cast(list[dict[str, Any]], pos_json):
                tid = p.get("trackId")
                if tid is not None:
                    all_track_ids.add(int(tid))
        if primary_ids:
            primary_ids_list = cast(list[int], primary_ids)
            for pid in primary_ids_list:
                all_track_ids.add(pid)

        # --- Step 3: Build and apply new mapping ---
        # Sub-track-claimed pids must NOT identity-map other real tracks
        # to the same number, or we'd get duplicate pid labels rendered at
        # the same frame. See `_build_full_mapping` docstring.
        sub_track_pids: set[int] = {
            int(ov["pid"]) for ov in sub_track_overrides
        }
        mapping = _build_full_mapping(
            raw_mapping, all_track_ids, sub_track_pids=sub_track_pids,
        )

        # Initialize changes early so first-run snapshot capture is
        # persisted even on identity-mapping rallies (where no other field
        # would otherwise change).
        changes: dict[str, Any] = {}
        rally_count = 0
        if write_snapshot:
            changes["pre_remap_state_json"] = json.dumps(snapshot_payload)

        # Check if mapping is all identity AND there are no sub-track
        # overrides (which always force a remap). Clear stale
        # appliedFullMapping/remapApplied so they don't trigger spurious
        # reversals on future runs (e.g. after re-tracking).
        if not any(k != v for k, v in mapping.items()) and not sub_track_overrides:
            rally_entry.pop("appliedFullMapping", None)
            rally_entry.pop("remapApplied", None)
            if changes:
                updates.append((pt_id, changes))
            if not quiet:
                console.print(f"  [dim]{rally_id[:8]}: already using player IDs[/dim]")
            continue

        # Remap positions
        if pos_json:
            positions = cast(list[dict[str, Any]], pos_json)
            n = _remap_positions(positions, mapping, sub_track_overrides)
            if n > 0:
                changes["positions_json"] = json.dumps(positions)
                rally_count += n

        # Remap contacts
        if contacts_json:
            contacts = cast(dict[str, Any], contacts_json)
            n = _remap_contacts(contacts, mapping, sub_track_overrides)
            if n > 0:
                changes["contacts_json"] = json.dumps(contacts)
                rally_count += n

        # Remap actions
        if actions_json:
            actions = cast(dict[str, Any], actions_json)
            n = _remap_actions(actions, mapping, sub_track_overrides)
            if n > 0:
                changes["actions_json"] = json.dumps(actions)
                rally_count += n

        # Remap primaryTrackIds. Primary track ids have no per-frame
        # context; the array represents "which 4 players are primary in
        # this rally" for editor display. For split parents, use the
        # flat (parent-level) mapping rather than UNLABELED — the
        # parent's frames carry per-segment overrides in positions, but
        # the editor still needs all 4 primaries listed to render the
        # full player overlay. The parent's flat-mapped pid (its
        # MatchSolver assignment) is the natural representative.
        if primary_ids:
            old_ids = cast(list[int], primary_ids)
            new_ids = [mapping.get(tid, tid) for tid in old_ids]
            if new_ids != old_ids:
                changes["primary_track_ids"] = json.dumps(new_ids)
                rally_count += 1

        # Remap action ground truth labels. GT labels have a `frame` field;
        # apply the same frame-conditional resolution as the action stream.
        if action_gt_json:
            gt_labels = cast(list[dict[str, Any]], action_gt_json)
            overrides_by_parent = _index_overrides_by_parent(sub_track_overrides)
            gt_changed = False
            for label in gt_labels:
                old_tid = label.get("playerTrackId")
                if old_tid is None:
                    continue
                old_tid_int = int(old_tid)
                gt_frame = label.get("frame")
                gt_frame_int = int(gt_frame) if gt_frame is not None else None
                resolved = _resolve_with_overrides(
                    old_tid_int, gt_frame_int, mapping, overrides_by_parent,
                )
                if resolved is not None and resolved != old_tid_int:
                    label["playerTrackId"] = resolved
                    gt_changed = True
                    rally_count += 1
            if gt_changed:
                changes["action_ground_truth_json"] = json.dumps(gt_labels)

        # --- Step 4: Store appliedFullMapping + remapApplied flag ---
        rally_entry["appliedFullMapping"] = {
            str(k): v for k, v in mapping.items()
        }
        rally_entry["remapApplied"] = True

        if changes:
            updates.append((pt_id, changes))
            total_remapped += rally_count
            # Show only the player-mapped changes, not collision shifts
            mapping_str = ", ".join(
                f"T{k}→P{v}" for k, v in sorted(raw_mapping.items()) if k != v
            )
            if not quiet:
                console.print(
                    f"  {rally_id[:8]}: {rally_count} remapped ({mapping_str})"
                )
        elif not quiet:
            console.print(f"  [dim]{rally_id[:8]}: no changes needed[/dim]")

    if not quiet:
        console.print(f"\n  Total: {total_remapped} track ID references remapped")

    # Set trackToPlayer to identity for remapped rallies (downstream consumers need this).
    # appliedFullMapping/remapApplied already set on rally_entry objects above.
    match_analysis_changed = False
    for rally_entry in match_analysis.get("rallies", []):
        rid = (
            rally_entry.get("rallyId")
            or rally_entry.get("rally_id", "")
        )
        raw = raw_mappings.get(rid, {})
        if raw and any(k != v for k, v in raw.items()):
            identity = {str(v): v for v in raw.values()}
            rally_entry["trackToPlayer"] = identity
            if "track_to_player" in rally_entry:
                rally_entry["track_to_player"] = identity
            match_analysis_changed = True

    # Update DB
    if not dry_run and (updates or match_analysis_changed):
        with get_connection() as conn:
            with conn.cursor() as cur:
                for update_pt_id, cols in updates:
                    set_clauses = []
                    values: list[Any] = []
                    for col, val in cols.items():
                        set_clauses.append(f"{col} = %s")
                        values.append(val)
                    values.append(update_pt_id)
                    cur.execute(
                        f"UPDATE player_tracks SET {', '.join(set_clauses)} "
                        f"WHERE id = %s",
                        values,
                    )

                cur.execute(
                    "UPDATE videos SET match_analysis_json = %s WHERE id = %s",
                    [json.dumps(match_analysis), video_id],
                )
            conn.commit()

        if not quiet:
            if updates:
                console.print(
                    f"  [green]Updated {len(updates)} player tracks in DB[/green]"
                )
            else:
                console.print(
                    "  [green]Cleared stale remap metadata in DB[/green]"
                )
    elif dry_run and updates:
        if not quiet:
            console.print(
                f"  [yellow]Dry run: would update {len(updates)} player tracks[/yellow]"
            )
