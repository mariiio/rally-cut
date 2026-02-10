"""Evaluate player tracking against ground truth from database."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from rich.console import Console
from rich.table import Table

from rallycut.cli.utils import handle_errors

if TYPE_CHECKING:
    from rallycut.cli.commands.compare_tracking import BallMetrics, MOTMetrics
    from rallycut.evaluation.tracking.error_analysis import ErrorEvent
    from rallycut.evaluation.tracking.metrics import PerPlayerMetrics

console = Console()


def _status_icon(value: float, target: float, higher_better: bool = True) -> str:
    """Return status icon based on whether value meets target."""
    if higher_better:
        return "[green]OK[/green]" if value >= target else "[yellow]Below[/yellow]"
    else:
        return "[green]OK[/green]" if value <= target else "[yellow]Above[/yellow]"


@handle_errors
def evaluate_tracking(
    rally_id: str = typer.Option(
        None,
        "--rally-id", "-r",
        help="Evaluate specific rally by ID",
    ),
    video_id: str = typer.Option(
        None,
        "--video-id", "-v",
        help="Evaluate all labeled rallies in a video",
    ),
    all_rallies: bool = typer.Option(
        False,
        "--all", "-a",
        help="Evaluate all labeled rallies in database",
    ),
    per_player: bool = typer.Option(
        False,
        "--per-player", "-p",
        help="Show per-player breakdown",
    ),
    analyze_errors: bool = typer.Option(
        False,
        "--analyze-errors", "-e",
        help="Show detailed error analysis",
    ),
    iou_threshold: float = typer.Option(
        0.5,
        "--iou", "-i",
        help="Minimum IoU for matching predictions to ground truth",
    ),
    output: Path = typer.Option(
        None,
        "--output", "-o",
        help="Output metrics to JSON file",
    ),
) -> None:
    """Evaluate player tracking predictions against ground truth.

    Loads ground truth and predictions from the database and computes
    MOT (Multi-Object Tracking) metrics including MOTA, precision, recall,
    F1, and ID switches.

    Examples:

        # Evaluate specific rally
        rallycut evaluate-tracking --rally-id abc123

        # Evaluate all labeled rallies in a video
        rallycut evaluate-tracking --video-id a7ee3d38-...

        # Evaluate all labeled data
        rallycut evaluate-tracking --all

        # Show per-player breakdown
        rallycut evaluate-tracking --rally-id abc123 --per-player

        # Show detailed error analysis
        rallycut evaluate-tracking --rally-id abc123 --analyze-errors

        # Export to JSON
        rallycut evaluate-tracking --rally-id abc123 -o metrics.json
    """
    from rallycut.evaluation.tracking.db import load_labeled_rallies
    from rallycut.evaluation.tracking.error_analysis import analyze_errors as get_errors
    from rallycut.evaluation.tracking.metrics import aggregate_results, evaluate_rally

    # Validate input
    if not rally_id and not video_id and not all_rallies:
        console.print(
            "[red]Error:[/red] Specify --rally-id, --video-id, or --all"
        )
        raise typer.Exit(1)

    # Load rallies from database
    console.print("[bold]Loading labeled rallies from database...[/bold]")

    rallies = load_labeled_rallies(
        video_id=video_id,
        rally_id=rally_id,
    )

    if not rallies:
        console.print("[yellow]No rallies with ground truth found[/yellow]")
        if rally_id:
            console.print(f"  Rally ID: {rally_id}")
        if video_id:
            console.print(f"  Video ID: {video_id}")
        raise typer.Exit(1)

    console.print(f"  Found {len(rallies)} rally(s) with ground truth\n")

    # Evaluate each rally
    results = []
    all_errors = []

    for rally in rallies:
        if rally.predictions is None:
            console.print(
                f"[yellow]Rally {rally.rally_id[:8]}... has no predictions, skipping[/yellow]"
            )
            continue

        result = evaluate_rally(
            rally_id=rally.rally_id,
            ground_truth=rally.ground_truth,
            predictions=rally.predictions,
            iou_threshold=iou_threshold,
            video_width=rally.video_width,
            video_height=rally.video_height,
        )
        results.append(result)

        if analyze_errors:
            errors = get_errors(
                rally.ground_truth,
                rally.predictions,
                iou_threshold,
            )
            all_errors.extend(errors)

    if not results:
        console.print("[red]No rallies with predictions found[/red]")
        raise typer.Exit(1)

    # Display results
    if len(results) == 1:
        # Single rally - show detailed output
        result = results[0]
        rally = rallies[0]

        console.print(f"[bold]Player Tracking Evaluation - Rally {result.rally_id[:8]}...[/bold]")
        console.print("=" * 50)

        _display_aggregate_metrics(result.aggregate)

        if result.ball_metrics:
            _display_ball_metrics(result.ball_metrics)

        if per_player and result.per_player:
            _display_per_player_metrics(result.per_player)

        # Error frames summary
        error_count = len(result.error_frames)
        total_frames = len(result.per_frame)
        if error_count > 0:
            console.print(f"\n[bold]Error Frames:[/bold] {error_count} of {total_frames} frames")
            if error_count <= 10:
                console.print(f"  Frames: {', '.join(str(f) for f in result.error_frames)}")
            else:
                first_five = result.error_frames[:5]
                last_five = result.error_frames[-5:]
                console.print(
                    f"  First 5: {', '.join(str(f) for f in first_five)}"
                )
                console.print(
                    f"  Last 5: {', '.join(str(f) for f in last_five)}"
                )
        else:
            console.print("\n[green]No error frames![/green]")

        if analyze_errors and all_errors:
            _display_error_analysis(all_errors)

    else:
        # Multiple rallies - show summary
        console.print(f"[bold]Player Tracking Evaluation - {len(results)} Rallies[/bold]")
        console.print("=" * 50)

        # Per-rally table
        rally_table = Table(show_header=True, header_style="bold")
        rally_table.add_column("Rally")
        rally_table.add_column("MOTA", justify="right")
        rally_table.add_column("Precision", justify="right")
        rally_table.add_column("Recall", justify="right")
        rally_table.add_column("F1", justify="right")
        rally_table.add_column("ID Sw", justify="right")
        rally_table.add_column("Errors", justify="right")

        for result in results:
            error_count = len(result.error_frames)
            rally_table.add_row(
                result.rally_id[:8] + "...",
                f"{result.aggregate.mota:.1%}",
                f"{result.aggregate.precision:.1%}",
                f"{result.aggregate.recall:.1%}",
                f"{result.aggregate.f1:.1%}",
                str(result.aggregate.num_id_switches),
                str(error_count),
            )

        console.print(rally_table)

        # Aggregate metrics
        console.print("\n[bold]Aggregate Metrics[/bold]")
        combined = aggregate_results(results)
        _display_aggregate_metrics(combined)

        if analyze_errors and all_errors:
            _display_error_analysis(all_errors)

    # Save to file if requested
    if output:
        if len(results) == 1:
            output_data = results[0].to_dict()
        else:
            combined = aggregate_results(results)
            output_data = {
                "rallies": [r.to_dict() for r in results],
                "aggregate": {
                    "mota": combined.mota,
                    "precision": combined.precision,
                    "recall": combined.recall,
                    "f1": combined.f1,
                    "idSwitches": combined.num_id_switches,
                },
            }

        with open(output, "w") as f:
            json.dump(output_data, f, indent=2)
        console.print(f"\n[green]Metrics saved to {output}[/green]")


def _display_aggregate_metrics(metrics: MOTMetrics) -> None:
    """Display aggregate metrics table."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_column("Target", justify="right")
    table.add_column("Status")

    table.add_row(
        "MOTA",
        f"{metrics.mota:.2%}",
        ">80%",
        _status_icon(metrics.mota, 0.80),
    )
    table.add_row(
        "Precision",
        f"{metrics.precision:.2%}",
        ">85%",
        _status_icon(metrics.precision, 0.85),
    )
    table.add_row(
        "Recall",
        f"{metrics.recall:.2%}",
        ">80%",
        _status_icon(metrics.recall, 0.80),
    )
    table.add_row(
        "F1",
        f"{metrics.f1:.2%}",
        ">80%",
        _status_icon(metrics.f1, 0.80),
    )
    table.add_row(
        "ID Switches",
        str(metrics.num_id_switches),
        "<5",
        _status_icon(metrics.num_id_switches, 5, higher_better=False),
    )

    console.print("\n[bold]Aggregate Metrics[/bold]")
    console.print(table)

    # Detection breakdown
    console.print("\n[dim]Detection breakdown:[/dim]")
    console.print(f"  Ground truth: {metrics.num_gt} objects")
    console.print(f"  Predictions: {metrics.num_pred} objects")
    console.print(f"  Matches (TP): {metrics.num_matches}")
    console.print(f"  Misses (FN): {metrics.num_misses}")
    console.print(f"  False positives (FP): {metrics.num_false_positives}")


def _display_ball_metrics(ball_metrics: BallMetrics) -> None:
    """Display ball tracking metrics."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_column("Target", justify="right")
    table.add_column("Status")

    table.add_row(
        "Detection Rate",
        f"{ball_metrics.detection_rate:.2%}",
        ">60%",
        _status_icon(ball_metrics.detection_rate, 0.60),
    )
    table.add_row(
        "Mean Error",
        f"{ball_metrics.mean_error_px:.1f} px",
        "<20px",
        _status_icon(ball_metrics.mean_error_px, 20, higher_better=False),
    )

    console.print("\n[bold]Ball Tracking Metrics[/bold]")
    console.print(table)


def _display_per_player_metrics(per_player: list[PerPlayerMetrics]) -> None:
    """Display per-player metrics table."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("Player")
    table.add_column("Recall", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("ID Swaps", justify="right")
    table.add_column("GT", justify="right")
    table.add_column("Matches", justify="right")

    # Find worst player for highlighting
    worst = min(per_player, key=lambda p: p.f1) if per_player else None

    for player in per_player:
        is_worst = player == worst
        suffix = " [yellow]<- worst[/yellow]" if is_worst else ""

        table.add_row(
            player.label + suffix,
            f"{player.recall:.1%}",
            f"{player.precision:.1%}",
            f"{player.f1:.1%}",
            str(player.id_switches),
            str(player.gt_count),
            str(player.matches),
        )

    console.print("\n[bold]Per-Player Breakdown[/bold]")
    console.print(table)


def _display_error_analysis(errors: list[ErrorEvent]) -> None:
    """Display error analysis summary."""
    from rallycut.evaluation.tracking.error_analysis import ErrorType, summarize_errors

    summary = summarize_errors(errors)

    console.print("\n[bold]Error Analysis[/bold]")
    console.print(f"  Total errors: {summary.total_errors}")

    # By type
    console.print("\n  [dim]By type:[/dim]")
    for error_type in ErrorType:
        count = summary.by_type.get(error_type, 0)
        if count > 0:
            console.print(f"    {error_type.value}: {count}")

    # By player (top 5)
    if summary.by_player:
        console.print("\n  [dim]By player (most errors):[/dim]")
        sorted_players = sorted(
            summary.by_player.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        for label, count in sorted_players:
            console.print(f"    {label}: {count}")

    # Consecutive error frames
    if summary.consecutive_error_frames > 1:
        console.print(
            f"\n  [yellow]Max consecutive error frames: "
            f"{summary.consecutive_error_frames}[/yellow]"
        )
