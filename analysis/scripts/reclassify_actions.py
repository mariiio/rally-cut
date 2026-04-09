"""Re-classify actions from stored contacts to apply server exclusion fix.

Loads raw contacts (with player_candidates) from DB, re-runs action
classification through classify_rally_actions(), and updates actions_json.
This applies the server-can't-receive-own-serve fix without re-tracking.

Usage:
    cd analysis
    uv run python scripts/reclassify_actions.py --dry-run   # Preview
    uv run python scripts/reclassify_actions.py              # Apply
"""

from __future__ import annotations

import json
import sys
from typing import Any, cast

from rich.console import Console

from rallycut.cli.commands.reattribute_actions import _reconstruct_contacts
from rallycut.evaluation.db import get_connection
from rallycut.statistics.match_stats import _validate_contact_sequence
from rallycut.tracking.action_classifier import (
    ActionType,
    RallyActions,
    classify_rally_actions,
)
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_detector import ContactSequence

console = Console()


def _load_rallies() -> list[dict[str, Any]]:
    """Load all rallies with contacts and actions from DB."""
    query = """
        SELECT
            r.id AS rally_id,
            r.video_id,
            v.filename,
            pt.id AS pt_id,
            pt.contacts_json,
            pt.actions_json,
            pt.ball_positions_json
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        JOIN videos v ON v.id = r.video_id
        WHERE pt.contacts_json IS NOT NULL
          AND pt.actions_json IS NOT NULL
        ORDER BY v.filename, r.start_ms
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

    return [
        {
            "rally_id": str(r[0]),
            "video_id": str(r[1]),
            "filename": str(r[2] or "?"),
            "pt_id": r[3],
            "contacts_json": r[4],
            "actions_json": r[5],
            "ball_positions_json": r[6],
        }
        for r in rows
    ]


def _build_contact_sequence(row: dict[str, Any]) -> ContactSequence:
    """Reconstruct a ContactSequence from stored DB data."""
    contacts_data = cast(dict[str, Any], row["contacts_json"])
    contacts = _reconstruct_contacts(contacts_data)

    net_y = contacts_data.get("netY", 0.5)
    rally_start_frame = contacts_data.get("rallyStartFrame", 0)

    # Reconstruct ball positions
    ball_positions: list[BallPosition] = []
    bp_json = row["ball_positions_json"]
    if bp_json and isinstance(bp_json, list):
        for bp in bp_json:
            ball_positions.append(BallPosition(
                frame_number=bp.get("frame", 0),
                x=bp.get("x", 0.0),
                y=bp.get("y", 0.0),
                confidence=bp.get("confidence", 0.0),
            ))

    return ContactSequence(
        contacts=contacts,
        net_y=net_y,
        rally_start_frame=rally_start_frame,
        ball_positions=ball_positions,
    )


def _get_team_assignments(actions_data: dict[str, Any]) -> dict[int, int]:
    """Extract team assignments from stored actions_json."""
    ta = actions_data.get("teamAssignments", {})
    result: dict[int, int] = {}
    for tid_str, team_label in ta.items():
        team_int = 0 if team_label == "A" else 1 if team_label == "B" else -1
        if team_int >= 0:
            result[int(tid_str)] = team_int
    return result


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    rows = _load_rallies()
    console.print(f"[bold]Re-classifying actions for {len(rows)} rallies[/bold]")
    if dry_run:
        console.print("[yellow]Dry run — no DB changes[/yellow]")

    # First pass: identify rallies with invalid contact sequences
    invalid_rally_ids: set[str] = set()
    for row in rows:
        actions_data = cast(dict[str, Any], row["actions_json"])
        team_assignments = _get_team_assignments(actions_data)

        # Build RallyActions to validate
        actions_list = []
        for a in actions_data.get("actions", []):
            try:
                action_type = ActionType(a["action"])
            except (ValueError, KeyError):
                action_type = ActionType.UNKNOWN
            from rallycut.tracking.action_classifier import ClassifiedAction
            actions_list.append(ClassifiedAction(
                action_type=action_type,
                frame=a.get("frame", 0),
                ball_x=a.get("ballX", 0.0),
                ball_y=a.get("ballY", 0.0),
                velocity=a.get("velocity", 0.0),
                player_track_id=a.get("playerTrackId", -1),
                court_side=a.get("courtSide", "unknown"),
                confidence=a.get("confidence", 0.0),
                is_synthetic=a.get("isSynthetic", False),
                team=a.get("team", "unknown"),
            ))

        ra = RallyActions(
            actions=actions_list,
            rally_id=row["rally_id"],
            team_assignments=team_assignments,
        )
        if _validate_contact_sequence(ra) is False:
            invalid_rally_ids.add(row["rally_id"])

    console.print(f"  Invalid contact sequences: {len(invalid_rally_ids)}")

    # Second pass: re-classify invalid rallies
    updated: list[tuple[Any, str]] = []
    fixed = 0
    still_invalid = 0

    for row in rows:
        if row["rally_id"] not in invalid_rally_ids:
            continue

        actions_data = cast(dict[str, Any], row["actions_json"])
        team_assignments = _get_team_assignments(actions_data)

        # Check if contacts have player_candidates
        contacts_data = cast(dict[str, Any], row["contacts_json"])
        has_candidates = any(
            c.get("playerCandidates") for c in contacts_data.get("contacts", [])
        )
        if not has_candidates:
            console.print(
                f"  [dim]{row['rally_id'][:8]} ({row['filename']}): "
                f"no player_candidates — skipped[/dim]"
            )
            continue

        # Reconstruct ContactSequence and re-classify
        contact_seq = _build_contact_sequence(row)
        new_ra = classify_rally_actions(
            contact_seq,
            rally_id=row["rally_id"],
            team_assignments=team_assignments,
        )

        # Check if the new classification is valid
        new_valid = _validate_contact_sequence(new_ra)

        # Compare old vs new serve/receive track IDs
        old_actions = actions_data.get("actions", [])
        old_serve_tid = next(
            (a.get("playerTrackId", -1) for a in old_actions if a.get("action") == "serve"),
            -1,
        )
        old_recv_tid = next(
            (a.get("playerTrackId", -1) for a in old_actions
             if a.get("action") in ("receive", "dig") and a.get("playerTrackId") != old_serve_tid),
            -1,
        )
        new_serve_tid = next(
            (a.player_track_id for a in new_ra.actions if a.action_type == ActionType.SERVE),
            -1,
        )
        new_recv_action = next(
            (a for a in new_ra.actions if a.action_type == ActionType.RECEIVE),
            None,
        )
        new_recv_tid = new_recv_action.player_track_id if new_recv_action else -1

        changed = False
        changes: list[str] = []

        # Check for any track ID changes across all actions
        for old_a, new_a in zip(old_actions, new_ra.actions):
            if old_a.get("playerTrackId", -1) != new_a.player_track_id:
                changed = True
                changes.append(
                    f"{new_a.action_type.value}: tid {old_a.get('playerTrackId', -1)}→{new_a.player_track_id}"
                )

        status = "[green]FIXED[/green]" if new_valid is True else "[yellow]still invalid[/yellow]"
        if new_valid is True:
            fixed += 1
        else:
            still_invalid += 1

        if changed:
            console.print(
                f"  {row['rally_id'][:8]} ({row['filename']}): "
                f"{status}  changes: {', '.join(changes)}"
            )

            # Serialize new actions back
            new_actions_data = dict(actions_data)
            new_actions_data["actions"] = [a.to_dict() for a in new_ra.actions]
            # Update teamAssignments from new RallyActions
            if new_ra.team_assignments:
                new_actions_data["teamAssignments"] = {
                    str(tid): ("A" if team == 0 else "B")
                    for tid, team in new_ra.team_assignments.items()
                }
            updated.append((row["pt_id"], json.dumps(new_actions_data)))
        else:
            console.print(
                f"  [dim]{row['rally_id'][:8]} ({row['filename']}): "
                f"{status}  no attribution changes[/dim]"
            )

    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Fixed:         {fixed}")
    console.print(f"  Still invalid: {still_invalid}")
    console.print(f"  DB updates:    {len(updated)}")

    if updated and not dry_run:
        with get_connection() as conn:
            with conn.cursor() as cur:
                for pt_id, new_json in updated:
                    cur.execute(
                        "UPDATE player_tracks SET actions_json = %s WHERE id = %s",
                        [new_json, pt_id],
                    )
            conn.commit()
        console.print(f"  [green]Updated {len(updated)} player tracks in DB[/green]")
    elif dry_run and updated:
        console.print(f"  [yellow]Dry run: would update {len(updated)} tracks[/yellow]")


if __name__ == "__main__":
    main()
