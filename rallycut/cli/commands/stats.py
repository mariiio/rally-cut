"""Stats command - analyze volleyball video for action statistics."""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table

from rallycut.core.config import get_config
from rallycut.core.video import Video
from rallycut.analysis.action_detector import ActionAnalyzer
from rallycut.statistics.aggregator import StatisticsAggregator

console = Console()


def format_time(seconds: float) -> str:
    """Format seconds as MM:SS or HH:MM:SS."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    if hours > 0:
        return f"{hours}:{minutes:02d}:{secs:05.2f}"
    return f"{minutes}:{secs:05.2f}"


def stats(
    video: Path = typer.Argument(
        ...,
        help="Path to input video file",
        exists=True,
        dir_okay=False,
        readable=True,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "--output", "-o",
        help="Output JSON file path",
    ),
    stride: int = typer.Option(
        8,
        "--stride", "-s",
        help="Frames to skip between detections (higher = faster, less accurate)",
    ),
    confidence: float = typer.Option(
        0.25,
        "--confidence", "-c",
        help="Minimum confidence threshold for detections",
    ),
    gpu: Optional[bool] = typer.Option(
        None,
        "--gpu/--no-gpu",
        help="Enable/disable GPU acceleration (auto-detected by default)",
    ),
    limit: Optional[float] = typer.Option(
        None,
        "--limit", "-l",
        help="Only analyze first N seconds of video (for testing)",
    ),
    segments_file: Optional[Path] = typer.Option(
        None,
        "--segments",
        help="Use play segments from JSON file for rally detection",
    ),
):
    """
    Analyze volleyball video to detect and count actions.

    Detects serves, receptions, sets, attacks, and blocks using YOLO-based
    action detection. Outputs statistics in JSON format.

    Examples:
        rallycut stats match.mp4                    # Full analysis
        rallycut stats match.mp4 -o stats.json     # Save to JSON
        rallycut stats match.mp4 --limit 120       # First 2 minutes
    """
    # Determine device
    config = get_config()
    if gpu is True:
        device = "cuda" if config.device == "cuda" else "mps"
    elif gpu is False:
        device = "cpu"
    else:
        device = config.device

    console.print(f"\n[bold]RallyCut Stats[/bold] - Action Analysis")
    console.print(f"Input: [cyan]{video}[/cyan]")
    console.print(f"Device: [yellow]{device}[/yellow]")
    console.print(f"Stride: [yellow]{stride} frames[/yellow]")
    console.print(f"Confidence: [yellow]{confidence}[/yellow]")
    if limit:
        console.print(f"Limit: [yellow]{format_time(limit)} ({limit:.1f}s)[/yellow]")
    console.print()

    # Get video info
    with Video(video) as v:
        video_info = v.info
        console.print(f"Duration: {format_time(video_info.duration)} ({video_info.duration:.1f}s)")
        console.print(f"Resolution: {video_info.width}x{video_info.height}")
        console.print(f"FPS: {video_info.fps:.1f}")
        console.print()

    # Load segments if provided
    segments = None
    if segments_file:
        from rallycut.core.models import GameState, TimeSegment

        with open(segments_file) as f:
            data = json.load(f)

        raw_segments = data if isinstance(data, list) else data.get("segments", [])
        segments = []

        for s in raw_segments:
            if isinstance(s, dict) and "start_time" in s:
                segments.append(TimeSegment(
                    start_frame=int(s.get("start_frame", s["start_time"] * video_info.fps)),
                    end_frame=int(s.get("end_frame", s["end_time"] * video_info.fps)),
                    start_time=float(s["start_time"]),
                    end_time=float(s["end_time"]),
                    state=GameState(s.get("state", "play")),
                ))

        console.print(f"Loaded [cyan]{len(segments)}[/cyan] segments from file")
        console.print()

    # Create analyzer
    analyzer = ActionAnalyzer(
        device=device,
        confidence_threshold=confidence,
    )

    # Progress tracking
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Detecting actions...", total=100)

        def update_progress(pct: float, msg: str):
            progress.update(task, completed=int(pct * 100), description=msg)

        try:
            with Video(video) as v:
                actions = analyzer.analyze_video(
                    v,
                    stride=stride,
                    progress_callback=update_progress,
                    limit_seconds=limit,
                )

            progress.update(task, completed=100, description="Complete!")

        except Exception as e:
            console.print(f"\n[red]Error:[/red] {e}")
            raise typer.Exit(1)

    # Aggregate statistics
    aggregator = StatisticsAggregator(video_info)
    stats_result = aggregator.compute_statistics(actions, segments)

    # Print results
    console.print()
    console.print("[bold green]Analysis Complete![/bold green]")
    console.print()

    # Actions table
    if actions:
        table = Table(title="Action Counts")
        table.add_column("Action", style="cyan")
        table.add_column("Count", style="green", justify="right")

        table.add_row("Serves", str(stats_result.serves.count))
        table.add_row("Receptions", str(stats_result.receptions.count))
        table.add_row("Sets", str(stats_result.sets.count))
        table.add_row("Attacks", str(stats_result.attacks.count))
        table.add_row("Blocks", str(stats_result.blocks.count))
        table.add_row("─" * 10, "─" * 5)
        table.add_row("Total", str(len(actions)), style="bold")

        console.print(table)
        console.print()

    # Summary
    console.print("[bold]Summary:[/bold]")
    console.print(f"  Total rallies:    {stats_result.total_rallies}")
    console.print(f"  Video duration:   {format_time(stats_result.total_duration)}")
    console.print(f"  Play time:        {format_time(stats_result.play_duration)}")
    console.print(f"  Dead time:        {format_time(stats_result.dead_time_duration)} ({stats_result.dead_time_percentage:.1f}%)")

    if stats_result.total_rallies > 0:
        console.print(f"  Avg rally:        {stats_result.avg_rally_duration:.1f}s")
        console.print(f"  Longest rally:    {stats_result.longest_rally_duration:.1f}s")
        console.print(f"  Touches/rally:    {stats_result.touches_per_rally:.1f}")

    # Output JSON
    if output:
        stats_dict = aggregator.to_dict(stats_result)
        stats_dict["actions_detail"] = [
            {
                "type": a.action_type.value,
                "timestamp": a.timestamp,
                "frame": a.frame_idx,
                "confidence": a.confidence,
            }
            for a in actions
        ]

        with open(output, "w") as f:
            json.dump(stats_dict, f, indent=2)
        console.print(f"\nStatistics saved to: [cyan]{output}[/cyan]")
