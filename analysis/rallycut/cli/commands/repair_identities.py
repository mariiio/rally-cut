"""Post-occlusion identity repair using match-level player profiles.

Runs after match-players to detect and fix within-rally identity switches
by comparing temporal windows of each track against match-level profiles.

Usage:
    rallycut repair-identities <video-id>
    rallycut repair-identities <video-id> --dry-run
"""

from __future__ import annotations

import json
from typing import Any, cast

import typer
from rich.console import Console

from rallycut.cli.utils import handle_errors

console = Console()


@handle_errors
def repair_identities_cmd(
    video_id: str = typer.Argument(
        ...,
        help="Video ID to repair identities for",
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
    """Detect and fix within-rally identity switches using match-level profiles.

    After match-players has built strong player profiles from many rallies,
    this command checks each rally for appearance shifts that indicate
    BoT-SORT swapped two players during a net interaction.

    Example:
        rallycut repair-identities abc123
        rallycut repair-identities abc123 --dry-run
    """
    from rallycut.evaluation.db import get_connection
    from rallycut.evaluation.tracking.db import get_video_path
    from rallycut.tracking.identity_repair import (
        apply_repairs,
        repair_rally_identities,
    )
    from rallycut.tracking.player_features import PlayerAppearanceProfile
    from rallycut.tracking.player_tracker import PlayerPosition

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

    # Load profiles
    profiles_data = match_analysis.get("playerProfiles", {})
    if not profiles_data:
        console.print("[red]Error:[/red] No player profiles in match analysis.")
        raise typer.Exit(1)

    profiles: dict[int, PlayerAppearanceProfile] = {}
    for pid_str, pdata in profiles_data.items():
        profiles[int(pid_str)] = PlayerAppearanceProfile.from_dict(pdata)

    # Get video path
    video_path = get_video_path(video_id)
    if not video_path:
        console.print(f"[red]Error:[/red] Video {video_id[:8]} not found locally.")
        raise typer.Exit(1)

    # Build per-rally info
    rally_entries = match_analysis.get("rallies", [])
    if not rally_entries:
        console.print("[yellow]No rallies in match analysis[/yellow]")
        return

    if not quiet:
        console.print(
            f"[bold]Repairing identities[/bold] for video {video_id[:8]}..."
        )
        console.print(f"  {len(rally_entries)} rallies, {len(profiles)} player profiles")

    # Load rally data
    rally_ids = [
        r.get("rallyId") or r.get("rally_id", "")
        for r in rally_entries
    ]
    placeholders = ", ".join(["%s"] * len(rally_ids))

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                SELECT r.id, r.start_ms, pt.id, pt.positions_json
                FROM rallies r
                JOIN player_tracks pt ON pt.rally_id = r.id
                WHERE r.id IN ({placeholders})
                """,
                rally_ids,
            )
            rows = cur.fetchall()

    rally_data: dict[str, tuple[int, int, Any]] = {}
    for r in rows:
        rally_data[str(r[0])] = (
            int(cast(int, r[1])),
            int(cast(int, r[2])),
            r[3],
        )

    total_repairs = 0
    updates: list[tuple[int, str]] = []  # (pt_id, new_positions_json)

    for rally_entry in rally_entries:
        rid = rally_entry.get("rallyId") or rally_entry.get("rally_id", "")
        if rid not in rally_data:
            continue

        start_ms, pt_id, pos_json = rally_data[rid]

        ttp = rally_entry.get("trackToPlayer") or rally_entry.get(
            "track_to_player", {}
        )
        if not ttp:
            continue
        ttp = {int(k): int(v) for k, v in ttp.items()}

        if not pos_json:
            continue

        # Convert positions
        positions = [
            PlayerPosition(
                frame_number=p["frameNumber"],
                track_id=p["trackId"],
                x=p["x"],
                y=p["y"],
                width=p["width"],
                height=p["height"],
                confidence=p.get("confidence", 1.0),
            )
            for p in cast(list[dict[str, Any]], pos_json)
        ]

        # Run repair
        repair_result = repair_rally_identities(
            positions=positions,
            track_to_player=ttp,
            player_profiles=profiles,
            video_path=video_path,
            start_ms=start_ms,
            rally_id=rid,
        )

        if repair_result.skipped:
            if not quiet:
                console.print(
                    f"  [dim]{rid[:8]}: skipped ({repair_result.skip_reason})[/dim]"
                )
            continue

        if repair_result.num_repairs > 0:
            n_modified = apply_repairs(positions, repair_result.decisions)
            total_repairs += repair_result.num_repairs

            # Serialize back
            new_pos_json = json.dumps([p.to_dict() for p in positions])
            updates.append((pt_id, new_pos_json))

            if not quiet:
                accepted = [d for d in repair_result.decisions if d.accepted]
                for dec in accepted:
                    direction = "before" if dec.is_backward else "after"
                    console.print(
                        f"  [green]{rid[:8]}: swapped p{dec.player_a}↔p{dec.player_b} "
                        f"{direction} frame {dec.swap_frame} "
                        f"(Δ={dec.improvement:+.3f}, "
                        f"{'cross' if dec.is_cross_team else 'same'}-team)[/green]"
                    )
                console.print(
                    f"  [green]  {n_modified} positions modified[/green]"
                )
        elif not quiet:
            n_cands = repair_result.num_candidates
            console.print(
                f"  [dim]{rid[:8]}: {n_cands} shift{'s' if n_cands != 1 else ''} "
                f"detected, none confident enough[/dim]"
            )

    if not quiet:
        console.print(f"\n  Total repairs: {total_repairs}")

    # Update DB
    if not dry_run and updates:
        with get_connection() as conn:
            with conn.cursor() as cur:
                for update_pt_id, new_json in updates:
                    cur.execute(
                        "UPDATE player_tracks SET positions_json = %s WHERE id = %s",
                        [new_json, update_pt_id],
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
