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
from typing import Any, cast

import typer
from rich.console import Console

from rallycut.cli.utils import handle_errors

console = Console()


def _build_full_mapping(
    track_to_player: dict[int, int],
    all_track_ids: set[int],
) -> dict[int, int]:
    """Build collision-safe mapping for ALL track IDs in a rally.

    Mapped tracks get their player IDs (1-4).
    Unmapped tracks keep their ID if no collision, otherwise shift to 101+.
    """
    mapping: dict[int, int] = {}
    # Player IDs that are targets of the mapping
    used_ids = set(track_to_player.values())

    # First, add the explicit track→player mappings
    for tid, pid in track_to_player.items():
        mapping[tid] = pid

    # Then handle unmapped tracks
    next_shifted = 101
    for tid in sorted(all_track_ids):
        if tid in mapping:
            continue  # Already mapped
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


def _remap_positions(
    positions: list[dict[str, Any]],
    mapping: dict[int, int],
) -> int:
    """Remap trackId in positions list. Returns count of remapped entries."""
    count = 0
    for p in positions:
        old_id = p.get("trackId")
        if old_id is not None and old_id in mapping:
            new_id = mapping[old_id]
            if new_id != old_id:
                p["trackId"] = new_id
                count += 1
    return count


def _remap_contacts(
    contacts_data: dict[str, Any],
    mapping: dict[int, int],
) -> int:
    """Remap playerTrackId and playerCandidates in contacts. Returns count."""
    count = 0
    for c in contacts_data.get("contacts", []):
        old_id = c.get("playerTrackId")
        if old_id is not None and old_id in mapping:
            new_id = mapping[old_id]
            if new_id != old_id:
                c["playerTrackId"] = new_id
                count += 1
        # Also remap candidates
        for cand in c.get("playerCandidates", []):
            if isinstance(cand, list) and len(cand) >= 1:
                cand_id = cand[0]
                if cand_id in mapping:
                    cand[0] = mapping[cand_id]
    return count


def _remap_actions(
    actions_data: dict[str, Any],
    mapping: dict[int, int],
) -> int:
    """Remap playerTrackId in actions and teamAssignments. Returns count."""
    count = 0
    for a in actions_data.get("actions", []):
        old_id = a.get("playerTrackId")
        if old_id is not None and old_id in mapping:
            new_id = mapping[old_id]
            if new_id != old_id:
                a["playerTrackId"] = new_id
                count += 1

    # Remap teamAssignments keys
    old_ta = actions_data.get("teamAssignments")
    if old_ta and isinstance(old_ta, dict):
        new_ta: dict[str, str] = {}
        for tid_str, team_label in old_ta.items():
            tid = int(tid_str)
            new_tid = mapping.get(tid, tid)
            new_ta[str(new_tid)] = team_label
        actions_data["teamAssignments"] = new_ta

    return count


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
                "SELECT match_analysis_json FROM videos WHERE id = %s",
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

    # Build per-rally track→player mappings (from match analysis)
    raw_mappings: dict[str, dict[int, int]] = {}
    for rally_entry in match_analysis.get("rallies", []):
        rid = rally_entry.get("rallyId") or rally_entry.get("rally_id", "")
        track_to_player = rally_entry.get("trackToPlayer") or rally_entry.get(
            "track_to_player", {}
        )
        if rid and track_to_player:
            raw_mappings[rid] = {int(k): int(v) for k, v in track_to_player.items()}

    if not raw_mappings:
        console.print("[yellow]No track mappings found in match analysis[/yellow]")
        return

    if not quiet:
        console.print(
            f"[bold]Remapping track IDs[/bold] for video {video_id[:8]}..."
        )
        console.print(f"  {len(raw_mappings)} rallies with mappings")

    # Load all player tracks for this video
    rally_ids = list(raw_mappings.keys())
    placeholders = ", ".join(["%s"] * len(rally_ids))
    query = f"""
        SELECT r.id, pt.id, pt.positions_json, pt.contacts_json,
               pt.actions_json, pt.primary_track_ids,
               pt.action_ground_truth_json
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE r.id IN ({placeholders})
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, rally_ids)
            rows = cur.fetchall()

    total_remapped = 0
    updates: list[tuple[int, dict[str, Any]]] = []  # (pt_id, {column: value})

    for rally_id_val, pt_id_val, pos_json, contacts_json, actions_json, primary_ids, action_gt_json in rows:
        rally_id = str(rally_id_val)
        pt_id = cast(int, pt_id_val)
        raw_mapping = raw_mappings.get(rally_id, {})
        if not raw_mapping:
            continue

        # Collect ALL track IDs present in this rally's data
        all_track_ids: set[int] = set()
        if pos_json:
            for p in cast(list[dict[str, Any]], pos_json):
                tid = p.get("trackId")
                if tid is not None:
                    all_track_ids.add(int(tid))
        if primary_ids:
            for pid in cast(list[int], primary_ids):
                all_track_ids.add(pid)

        # Build collision-safe full mapping
        mapping = _build_full_mapping(raw_mapping, all_track_ids)

        # Check if already remapped (all mapped tracks already using target IDs)
        needs_remap = any(k != v for k, v in mapping.items())
        if not needs_remap:
            if not quiet:
                console.print(f"  [dim]{rally_id[:8]}: already using player IDs[/dim]")
            continue

        changes: dict[str, Any] = {}
        rally_count = 0

        # Remap positions
        if pos_json:
            positions = cast(list[dict[str, Any]], pos_json)
            n = _remap_positions(positions, mapping)
            if n > 0:
                changes["positions_json"] = json.dumps(positions)
                rally_count += n

        # Remap contacts
        if contacts_json:
            contacts = cast(dict[str, Any], contacts_json)
            n = _remap_contacts(contacts, mapping)
            if n > 0:
                changes["contacts_json"] = json.dumps(contacts)
                rally_count += n

        # Remap actions
        if actions_json:
            actions = cast(dict[str, Any], actions_json)
            n = _remap_actions(actions, mapping)
            if n > 0:
                changes["actions_json"] = json.dumps(actions)
                rally_count += n

        # Remap primaryTrackIds
        if primary_ids:
            old_ids = cast(list[int], primary_ids)
            new_ids = [mapping.get(tid, tid) for tid in old_ids]
            if new_ids != old_ids:
                changes["primary_track_ids"] = json.dumps(new_ids)
                rally_count += 1

        # Remap action ground truth labels
        if action_gt_json:
            gt_labels = cast(list[dict[str, Any]], action_gt_json)
            gt_changed = False
            for label in gt_labels:
                old_tid = label.get("playerTrackId")
                if old_tid is not None and old_tid in mapping:
                    new_tid = mapping[old_tid]
                    if new_tid != old_tid:
                        label["playerTrackId"] = new_tid
                        gt_changed = True
                        rally_count += 1
            if gt_changed:
                changes["action_ground_truth_json"] = json.dumps(gt_labels)

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

    # Update DB
    if updates and not dry_run:
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
            conn.commit()

        if not quiet:
            console.print(
                f"  [green]Updated {len(updates)} player tracks in DB[/green]"
            )
    elif dry_run and updates:
        if not quiet:
            console.print(
                f"  [yellow]Dry run: would update {len(updates)} player tracks[/yellow]"
            )
