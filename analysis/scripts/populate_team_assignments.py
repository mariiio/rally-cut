"""Populate missing teamAssignments in actions_json from match_analysis.

Many rallies were tracked before reattribute-actions existed, so their
actions_json has empty teamAssignments. This script derives teams from
match_analysis_json (which accounts for side switches) and stamps them
into actions_json + per-action team labels.

Usage:
    cd analysis
    uv run python scripts/populate_team_assignments.py --dry-run
    uv run python scripts/populate_team_assignments.py
"""

from __future__ import annotations

import json
import sys
from typing import Any, cast

from rich.console import Console

from rallycut.evaluation.db import get_connection
from rallycut.tracking.action_classifier import _team_label
from rallycut.tracking.match_tracker import build_match_team_assignments

console = Console()


def main() -> None:
    dry_run = "--dry-run" in sys.argv

    # Load all rallies with empty teamAssignments but match_analysis available
    query = """
        SELECT
            r.id AS rally_id,
            r.video_id,
            v.filename,
            pt.id AS pt_id,
            pt.actions_json,
            v.match_analysis_json
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        JOIN videos v ON v.id = r.video_id
        WHERE pt.actions_json IS NOT NULL
          AND v.match_analysis_json IS NOT NULL
        ORDER BY v.filename, r.start_ms
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

    # Filter to rallies with empty teamAssignments
    needs_teams: list[dict[str, Any]] = []
    for r in rows:
        actions_data = cast(dict[str, Any], r[4])
        ta = actions_data.get("teamAssignments", {})
        if not ta:
            needs_teams.append({
                "rally_id": str(r[0]),
                "video_id": str(r[1]),
                "filename": str(r[2] or "?"),
                "pt_id": r[3],
                "actions_json": r[4],
                "match_analysis": r[5],
            })

    if not needs_teams:
        console.print("[green]All rallies already have teamAssignments.[/green]")
        return

    console.print(
        f"[bold]Populating teamAssignments[/bold] for {len(needs_teams)} rallies"
    )
    if dry_run:
        console.print("[yellow]Dry run — no DB changes[/yellow]")

    # Build team assignments per video from match_analysis
    video_teams: dict[str, dict[str, dict[int, int]]] = {}
    for row in needs_teams:
        vid = row["video_id"]
        if vid not in video_teams:
            match_analysis = cast(dict[str, Any], row["match_analysis"])
            teams_by_rally = build_match_team_assignments(
                match_analysis, min_confidence=0.0,
            )
            video_teams[vid] = teams_by_rally

    updated: list[tuple[Any, str]] = []
    populated = 0
    skipped = 0

    for row in needs_teams:
        rally_id = row["rally_id"]
        vid = row["video_id"]
        teams_by_rally = video_teams.get(vid, {})
        team_assignments = teams_by_rally.get(rally_id)

        if not team_assignments:
            console.print(
                f"  [dim]{rally_id[:8]} ({row['filename']}): "
                f"no match teams for this rally[/dim]"
            )
            skipped += 1
            continue

        # Stamp teamAssignments and per-action team labels
        actions_data = cast(dict[str, Any], row["actions_json"])
        new_data = dict(actions_data)
        new_data["teamAssignments"] = {
            str(tid): ("A" if team == 0 else "B")
            for tid, team in team_assignments.items()
        }

        # Update per-action team labels
        for a in new_data.get("actions", []):
            tid = a.get("playerTrackId", -1)
            if tid >= 0:
                a["team"] = _team_label(tid, team_assignments)
            elif a.get("courtSide") in ("near", "far"):
                a["team"] = "A" if a["courtSide"] == "near" else "B"

        n_known = sum(
            1 for a in new_data.get("actions", [])
            if a.get("team") in ("A", "B")
        )
        n_total = len(new_data.get("actions", []))

        console.print(
            f"  {rally_id[:8]} ({row['filename']}): "
            f"{len(team_assignments)} tracks, "
            f"{n_known}/{n_total} actions with teams"
        )

        updated.append((row["pt_id"], json.dumps(new_data)))
        populated += 1

    console.print(f"\n[bold]Summary:[/bold]")
    console.print(f"  Populated: {populated}")
    console.print(f"  Skipped:   {skipped}")

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
        console.print(
            f"  [yellow]Dry run: would update {len(updated)} tracks[/yellow]"
        )


if __name__ == "__main__":
    main()
