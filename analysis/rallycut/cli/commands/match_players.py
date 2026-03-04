"""Match players across rallies for consistent IDs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import typer
from rich.console import Console
from rich.table import Table

from rallycut.cli.utils import handle_errors

console = Console()


def _reverse_rally_positions(
    rally: Any,
    inverse: dict[int, int],
) -> None:
    """Reverse remapped positions and primaryTrackIds in a RallyTrackData in place."""
    for p in rally.positions:
        if p.track_id in inverse:
            p.track_id = inverse[p.track_id]
    rally.primary_track_ids = [
        inverse.get(tid, tid) for tid in rally.primary_track_ids
    ]


@handle_errors
def match_players(
    video_id: str = typer.Argument(
        ...,
        help="Video ID to match players across rallies",
    ),
    output: Path | None = typer.Option(
        None,
        "--output", "-o",
        help="Output JSON file for player matching results",
    ),
    num_samples: int = typer.Option(
        12,
        "--num-samples",
        help="Number of frames to sample per track for appearance extraction",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet", "-q",
        help="Suppress progress output",
    ),
) -> None:
    """Match players across rallies for consistent player IDs (1-4).

    Uses appearance features (skin tone, jersey color, body proportions)
    to assign consistent player IDs across all rallies in a video.
    Detects side switches based on appearance mismatch.

    Example:
        rallycut match-players abc123
        rallycut match-players abc123 -o result.json
    """
    from rallycut.cli.commands.remap_track_ids import _invert_mapping, _should_reverse
    from rallycut.evaluation.db import get_connection
    from rallycut.evaluation.tracking.db import get_video_path, load_rallies_for_video
    from rallycut.tracking.match_tracker import MatchPlayersResult, match_players_across_rallies

    if not quiet:
        console.print(f"[bold]Cross-Rally Player Matching:[/bold] video {video_id[:8]}...")

    # Load rallies from DB
    rallies = load_rallies_for_video(video_id)
    if not rallies:
        console.print("[red]Error:[/red] No tracked rallies found for this video.")
        console.print("[dim]Hint: Run player tracking first with 'rallycut track-players'[/dim]")
        raise typer.Exit(1)

    if not quiet:
        console.print(f"  Loaded {len(rallies)} rallies")

    # Load existing match analysis to check for appliedFullMapping
    old_match_analysis: dict[str, Any] | None = None
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT match_analysis_json FROM videos WHERE id = %s",
                [video_id],
            )
            row = cur.fetchone()
            if row and row[0]:
                old_match_analysis = cast(dict[str, Any], row[0])

    # Reverse remapped positions so matching always sees original tracker IDs
    if old_match_analysis:
        old_entries_by_id: dict[str, dict[str, Any]] = {}
        for entry in old_match_analysis.get("rallies", []):
            rid = entry.get("rallyId") or entry.get("rally_id", "")
            if rid:
                old_entries_by_id[rid] = entry

        reversed_count = 0
        for rally in rallies:
            old_entry = old_entries_by_id.get(rally.rally_id)
            if not old_entry:
                continue
            if not old_entry.get("remapApplied", False):
                continue
            applied_raw = old_entry.get("appliedFullMapping")
            if not applied_raw:
                continue
            applied = {int(k): int(v) for k, v in applied_raw.items()}
            # Build position dicts for _should_reverse check
            pos_dicts = [{"trackId": p.track_id} for p in rally.positions]
            if _should_reverse(pos_dicts, applied):
                inverse = _invert_mapping(applied)
                _reverse_rally_positions(rally, inverse)
                reversed_count += 1

        if reversed_count > 0 and not quiet:
            console.print(
                f"  Reversed previous remap on {reversed_count} rallies"
            )

    # Resolve video path
    video_path = get_video_path(video_id)
    if video_path is None:
        console.print("[red]Error:[/red] Could not resolve video file.")
        raise typer.Exit(1)

    if not quiet:
        console.print(f"  Video: {video_path.name}")
        console.print()

    # Run matching
    match_result: MatchPlayersResult = match_players_across_rallies(
        video_path=video_path,
        rallies=rallies,
        num_samples=num_samples,
    )
    results = match_result.rally_results

    # Print summary table
    if not quiet:
        table = Table(title="Player Matching Results", show_header=True, header_style="bold")
        table.add_column("Rally", style="dim")
        table.add_column("Confidence")
        table.add_column("Side Switch")
        table.add_column("Assignments")
        table.add_column("Server")

        for rally, result in zip(rallies, results):
            # Format assignments
            assign_str = ", ".join(
                f"T{tid}→P{pid}" for tid, pid in sorted(result.track_to_player.items())
            )

            # Confidence color
            conf = result.assignment_confidence
            if conf >= 0.7:
                conf_str = f"[green]{conf:.2f}[/green]"
            elif conf >= 0.5:
                conf_str = f"[yellow]{conf:.2f}[/yellow]"
            else:
                conf_str = f"[red]{conf:.2f}[/red]"

            switch_str = "[yellow]YES[/yellow]" if result.side_switch_detected else "no"
            server_str = f"P{result.server_player_id}" if result.server_player_id else "-"

            table.add_row(
                rally.rally_id[:8],
                conf_str,
                switch_str,
                assign_str,
                server_str,
            )

        console.print(table)

        # Summary stats
        n_switches = sum(1 for r in results if r.side_switch_detected)
        avg_conf = sum(r.assignment_confidence for r in results) / len(results) if results else 0
        console.print(f"\n  Rallies: {len(results)}")
        console.print(f"  Avg confidence: {avg_conf:.2f}")
        console.print(f"  Side switches: {n_switches}")

    # Serialize player profiles
    player_profiles_data = {
        str(pid): profile.to_dict()
        for pid, profile in match_result.player_profiles.items()
        if profile.rally_count > 0
    }

    # Build per-rally entries, carrying forward remap state from old analysis
    old_remap_state: dict[str, dict[str, Any]] = {}
    if old_match_analysis:
        for entry in old_match_analysis.get("rallies", []):
            rid = entry.get("rallyId") or entry.get("rally_id", "")
            afm = entry.get("appliedFullMapping")
            if rid and afm:
                old_remap_state[rid] = {
                    "appliedFullMapping": afm,
                    "remapApplied": entry.get("remapApplied", False),
                }

    rally_entries = []
    for rally, result in zip(rallies, results):
        rally_entry: dict[str, Any] = {
            "rallyId": rally.rally_id,
            "rallyIndex": result.rally_index,
            "trackToPlayer": {
                str(k): v for k, v in result.track_to_player.items()
            },
            "assignmentConfidence": result.assignment_confidence,
            "sideSwitchDetected": result.side_switch_detected,
            "serverPlayerId": result.server_player_id,
        }
        # Carry forward remap state so remap-track-ids can reverse
        old_state = old_remap_state.get(rally.rally_id)
        if old_state:
            rally_entry["appliedFullMapping"] = old_state["appliedFullMapping"]
            rally_entry["remapApplied"] = old_state["remapApplied"]
        rally_entries.append(rally_entry)

    # Build result JSON (same format used by batch_match_players and API)
    result_json: dict[str, Any] = {
        "videoId": video_id,
        "numRallies": len(results),
        "rallies": rally_entries,
        "playerProfiles": player_profiles_data,
    }

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE videos SET match_analysis_json = %s WHERE id = %s",
                [json.dumps(result_json), video_id],
            )
        conn.commit()

    if not quiet:
        console.print("  Saved to DB")

    # Write JSON file output
    if output:
        output_data = {
            "video_id": video_id,
            "num_rallies": len(results),
            "playerProfiles": player_profiles_data,
            "rallies": [
                {
                    "rally_id": rally.rally_id,
                    "rally_index": result.rally_index,
                    "start_ms": rally.start_ms,
                    "end_ms": rally.end_ms,
                    "track_to_player": {
                        str(k): v for k, v in result.track_to_player.items()
                    },
                    "assignment_confidence": result.assignment_confidence,
                    "side_switch_detected": result.side_switch_detected,
                    "server_player_id": result.server_player_id,
                }
                for rally, result in zip(rallies, results)
            ],
        }
        with open(output, "w") as f:
            json.dump(output_data, f, indent=2)

        if not quiet:
            console.print(f"\n[green]Results saved to {output}[/green]")
