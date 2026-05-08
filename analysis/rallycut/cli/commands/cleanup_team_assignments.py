"""CLI: rallycut cleanup-team-assignments <video-id>

One-shot cleanup that rebuilds persisted `teamAssignments` per rally
using `classify_teams` precomputed-passthrough. Clears I-6 violations
on legacy data persisted before the player_tracker fix landed.

The CLI is idempotent: re-running on already-clean rallies is a no-op.
"""

from __future__ import annotations

import json
from typing import Any, cast

import typer
from rich.console import Console

from rallycut.evaluation.tracking.db import get_connection
from rallycut.tracking.player_filter import classify_teams
from rallycut.tracking.player_tracker import PlayerPosition

console = Console()

# Convention from action_classifier.py:138 — team 0 (near) = "A", team 1 (far) = "B".
_LABEL_TO_INT = {"A": 0, "B": 1}
_INT_TO_LABEL = {0: "A", 1: "B"}


def _convert_str_to_int(team_assignments: dict[str, str]) -> dict[int, int]:
    """Convert legacy {str: 'A'|'B'} → {int: 0|1}. Skip corrupt entries."""
    result: dict[int, int] = {}
    for tid_str, label in team_assignments.items():
        if label not in _LABEL_TO_INT:
            continue
        try:
            result[int(tid_str)] = _LABEL_TO_INT[label]
        except (ValueError, TypeError):
            continue
    return result


def _convert_int_to_str(team_assignments: dict[int, int]) -> dict[str, str]:
    """Convert classify_teams output {int: 0|1} → {str: 'A'|'B'}."""
    return {
        str(tid): _INT_TO_LABEL[team]
        for tid, team in team_assignments.items()
        if team in _INT_TO_LABEL
    }


def _reconstruct_positions(positions_json: list[dict[str, Any]]) -> list[PlayerPosition]:
    """Build PlayerPosition objects from persisted JSON.

    classify_teams only reads frame_number, track_id, and y; other fields
    use safe defaults.
    """
    out: list[PlayerPosition] = []
    for p in positions_json:
        try:
            out.append(
                PlayerPosition(
                    frame_number=int(p.get("frameNumber", 0)),
                    track_id=int(p.get("trackId", -1)),
                    x=float(p.get("x", 0.0)),
                    y=float(p.get("y", 0.0)),
                    width=float(p.get("width", 0.0)),
                    height=float(p.get("height", 0.0)),
                    confidence=float(p.get("confidence", 0.0)),
                )
            )
        except (TypeError, ValueError):
            continue
    return out


def cleanup_team_assignments_cmd(
    video_id: str = typer.Argument(..., help="Video UUID"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress per-rally info"),
) -> None:
    """Rebuild persisted teamAssignments per rally to close I-6 violations on legacy data."""
    if not quiet:
        console.print(f"[dim]Cleaning up teamAssignments for video {video_id}…[/dim]")

    rally_query = """
        SELECT
            r.id AS rally_id,
            pt.positions_json,
            pt.actions_json,
            pt.court_split_y
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE r.video_id = %s
        ORDER BY r.start_ms
    """

    n_updated = 0
    n_skipped = 0
    n_no_change = 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(rally_query, [video_id])
            rows = cur.fetchall()

        for row in rows:
            rally_id = cast(str, row[0])
            positions_json = row[1]
            actions_json = row[2]
            court_split_y = row[3]

            if (
                court_split_y is None
                or not isinstance(positions_json, list)
                or not positions_json
                or not isinstance(actions_json, dict)
            ):
                n_skipped += 1
                if not quiet:
                    console.print(
                        f"  [yellow][skip][/yellow] rally {rally_id}: missing court_split_y, "
                        f"positions, or actions_json"
                    )
                continue

            old_team_assignments_str = actions_json.get("teamAssignments")
            if not isinstance(old_team_assignments_str, dict):
                n_skipped += 1
                if not quiet:
                    console.print(
                        f"  [yellow][skip][/yellow] rally {rally_id}: no teamAssignments dict"
                    )
                continue

            int_precomputed = _convert_str_to_int(old_team_assignments_str)
            positions = _reconstruct_positions(positions_json)
            if not positions:
                n_skipped += 1
                if not quiet:
                    console.print(
                        f"  [yellow][skip][/yellow] rally {rally_id}: no usable positions"
                    )
                continue

            new_team_assignments_int = classify_teams(
                positions,
                float(cast(float, court_split_y)),
                precomputed_assignments=int_precomputed,
            )
            new_team_assignments_str = _convert_int_to_str(new_team_assignments_int)

            if new_team_assignments_str == old_team_assignments_str:
                n_no_change += 1
                if not quiet:
                    console.print(
                        f"  [dim][noop][/dim] rally {rally_id}: already clean"
                    )
                continue

            actions_json["teamAssignments"] = new_team_assignments_str

            try:
                with conn.cursor() as wcur:
                    wcur.execute(
                        "UPDATE player_tracks SET actions_json = %s WHERE rally_id = %s",
                        [json.dumps(actions_json), rally_id],
                    )
                conn.commit()
                n_updated += 1
                if not quiet:
                    console.print(
                        f"  [green][fix][/green]  rally {rally_id}: "
                        f"{old_team_assignments_str} → {new_team_assignments_str}"
                    )
            except Exception as exc:
                conn.rollback()
                if not quiet:
                    console.print(
                        f"  [red][err][/red]  rally {rally_id}: write failed: {exc}"
                    )

    console.print(
        f"\n[bold]Summary:[/bold] {n_updated} updated · "
        f"{n_no_change} no-change · {n_skipped} skipped"
    )
