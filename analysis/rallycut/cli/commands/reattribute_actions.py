"""Re-attribute player actions using match-level team assignments.

Runs after match-players to improve player attribution on stored actions
by leveraging cross-rally team identity (which isn't available during
initial per-rally tracking).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, cast

import typer
from rich.console import Console

from rallycut.cli.utils import handle_errors

if TYPE_CHECKING:
    from rallycut.tracking.action_classifier import ClassifiedAction
    from rallycut.tracking.contact_detector import Contact

console = Console()
logger = logging.getLogger(__name__)


def _build_match_team_assignments(
    match_analysis: dict[str, Any],
    min_confidence: float = 0.70,
) -> dict[str, dict[int, int]]:
    """Build per-rally team assignments from match_analysis_json.

    Returns:
        Map of rally_id → {track_id → team (0=near, 1=far)}.
        Only includes rallies with assignment confidence >= min_confidence.
    """
    rallies = match_analysis.get("rallies", [])
    if not isinstance(rallies, list):
        return {}

    result: dict[str, dict[int, int]] = {}
    side_switch_count = 0

    for rally_entry in rallies:
        if rally_entry.get("sideSwitchDetected") or rally_entry.get(
            "side_switch_detected"
        ):
            side_switch_count += 1

        track_to_player = rally_entry.get("trackToPlayer") or rally_entry.get(
            "track_to_player", {}
        )
        rid = rally_entry.get("rallyId") or rally_entry.get("rally_id", "")
        if not rid or not track_to_player:
            continue

        conf = rally_entry.get("assignmentConfidence") or rally_entry.get(
            "assignment_confidence", 0
        )
        if conf < min_confidence:
            continue

        teams: dict[int, int] = {}
        for tid_str, player_id in track_to_player.items():
            pid = int(player_id)
            base_team = 0 if pid <= 2 else 1
            team = base_team if side_switch_count % 2 == 0 else 1 - base_team
            teams[int(tid_str)] = team

        if teams:
            result[rid] = teams

    return result


def _reconstruct_contacts(
    contacts_data: dict[str, Any],
) -> list[Contact]:
    """Reconstruct Contact objects from stored contacts_json."""
    from rallycut.tracking.contact_detector import Contact as ContactCls

    contacts: list[Contact] = []
    for c in contacts_data.get("contacts", []):
        candidates = c.get("playerCandidates", [])
        contacts.append(ContactCls(
            frame=c.get("frame", 0),
            ball_x=c.get("ballX", 0.0),
            ball_y=c.get("ballY", 0.0),
            velocity=c.get("velocity", 0.0),
            direction_change_deg=c.get("directionChangeDeg", 0.0),
            player_track_id=c.get("playerTrackId", -1),
            player_distance=c["playerDistance"] if c.get("playerDistance") is not None else float("inf"),
            player_candidates=[
                (int(p[0]), float(p[1])) for p in candidates if p[1] is not None
            ],
            court_side=c.get("courtSide", "unknown"),
            is_at_net=c.get("isAtNet", False),
            is_validated=c.get("isValidated", False),
            confidence=c.get("confidence", 0.0),
            arc_fit_residual=c.get("arcFitResidual", 0.0),
        ))
    return contacts


def _reconstruct_actions(
    actions_data: dict[str, Any],
) -> list[ClassifiedAction]:
    """Reconstruct ClassifiedAction objects from stored actions_json."""
    from rallycut.tracking.action_classifier import ActionType
    from rallycut.tracking.action_classifier import ClassifiedAction as ActionCls

    actions: list[ClassifiedAction] = []
    for a in actions_data.get("actions", []):
        try:
            action_type = ActionType(a["action"])
        except (ValueError, KeyError):
            action_type = ActionType.UNKNOWN

        actions.append(ActionCls(
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
    return actions


def _serialize_actions(
    actions: list[ClassifiedAction],
    original_actions_data: dict[str, Any],
) -> dict[str, Any]:
    """Serialize actions back, preserving other fields from original data."""
    result = dict(original_actions_data)
    result["actions"] = [a.to_dict() for a in actions]
    return result


@handle_errors
def reattribute_actions_cmd(
    video_id: str = typer.Argument(
        ...,
        help="Video ID to re-attribute actions for",
    ),
    min_confidence: float = typer.Option(
        0.70,
        "--min-confidence",
        help="Minimum match confidence to use team assignments",
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
    """Re-attribute player actions using match-level team assignments.

    After match-players has run, this command uses the cross-rally team
    identity to fix player attributions where the nearest player is on
    the wrong team for the court side.

    Example:
        rallycut reattribute-actions abc123
        rallycut reattribute-actions abc123 --dry-run
    """
    from rallycut.evaluation.tracking.db import get_connection
    from rallycut.tracking.action_classifier import reattribute_players

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
    teams_by_rally = _build_match_team_assignments(match_analysis, min_confidence)

    if not teams_by_rally:
        if not quiet:
            console.print(
                f"[yellow]No rallies with confidence >= {min_confidence:.2f}[/yellow]"
            )
        return

    if not quiet:
        console.print(
            f"[bold]Re-attributing actions[/bold] for video {video_id[:8]}..."
        )
        console.print(
            f"  {len(teams_by_rally)} rallies with match teams "
            f"(conf >= {min_confidence:.2f})"
        )

    # Load contacts + actions for eligible rallies
    rally_ids = list(teams_by_rally.keys())
    placeholders = ", ".join(["%s"] * len(rally_ids))
    query = f"""
        SELECT r.id, pt.id, pt.contacts_json, pt.actions_json
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE r.id IN ({placeholders})
          AND pt.contacts_json IS NOT NULL
          AND pt.actions_json IS NOT NULL
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, rally_ids)
            rows = cur.fetchall()

    total_reattributed = 0
    total_actions = 0
    updated_tracks: list[tuple[int, str]] = []  # (pt_id, new_actions_json)

    for rally_id_val, pt_id_val, contacts_json_val, actions_json_val in rows:
        rally_id = str(rally_id_val)
        pt_id = cast(int, pt_id_val)
        contacts_data = cast(dict[str, Any], contacts_json_val)
        actions_data = cast(dict[str, Any], actions_json_val)

        team_assignments = teams_by_rally.get(rally_id)
        if not team_assignments:
            continue

        contacts = _reconstruct_contacts(contacts_data)
        actions = _reconstruct_actions(actions_data)

        if not contacts or not actions:
            continue

        # Check if any contacts have player_candidates
        has_candidates = any(c.player_candidates for c in contacts)
        if not has_candidates:
            if not quiet:
                console.print(
                    f"  [dim]{rally_id[:8]}: no player_candidates stored "
                    "(re-track to populate)[/dim]"
                )
            continue

        # Count changes by comparing before/after
        original_track_ids = [a.player_track_id for a in actions]
        reattribute_players(actions, contacts, team_assignments)
        n_changed = sum(
            1 for orig, act in zip(original_track_ids, actions)
            if orig != act.player_track_id
        )

        total_actions += len(actions)
        total_reattributed += n_changed

        if n_changed > 0:
            new_actions_data = _serialize_actions(actions, actions_data)
            updated_tracks.append((pt_id, json.dumps(new_actions_data)))

            if not quiet:
                console.print(
                    f"  {rally_id[:8]}: {n_changed}/{len(actions)} actions re-attributed"
                )
        elif not quiet:
            console.print(f"  [dim]{rally_id[:8]}: no changes[/dim]")

    # Summary
    if not quiet:
        console.print(
            f"\n  Total: {total_reattributed}/{total_actions} actions "
            f"re-attributed across {len(rows)} rallies"
        )

    # Update DB
    if updated_tracks and not dry_run:
        with get_connection() as conn:
            with conn.cursor() as cur:
                for update_pt_id, new_json in updated_tracks:
                    cur.execute(
                        "UPDATE player_tracks SET actions_json = %s WHERE id = %s",
                        [new_json, update_pt_id],
                    )
            conn.commit()

        if not quiet:
            console.print(
                f"  [green]Updated {len(updated_tracks)} player tracks in DB[/green]"
            )
    elif dry_run and updated_tracks:
        if not quiet:
            console.print(
                f"  [yellow]Dry run: would update "
                f"{len(updated_tracks)} player tracks[/yellow]"
            )
