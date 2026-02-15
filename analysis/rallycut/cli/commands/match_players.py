"""Match players across rallies for consistent IDs."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from rallycut.cli.utils import handle_errors

console = Console()


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
    from rallycut.evaluation.tracking.db import get_video_path, load_rallies_for_video
    from rallycut.tracking.match_tracker import match_players_across_rallies

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

    # Resolve video path
    video_path = get_video_path(video_id)
    if video_path is None:
        console.print("[red]Error:[/red] Could not resolve video file.")
        raise typer.Exit(1)

    if not quiet:
        console.print(f"  Video: {video_path.name}")
        console.print()

    # Run matching
    results = match_players_across_rallies(
        video_path=video_path,
        rallies=rallies,
        num_samples=num_samples,
    )

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

    # Write JSON output
    if output:
        output_data = {
            "video_id": video_id,
            "num_rallies": len(results),
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
