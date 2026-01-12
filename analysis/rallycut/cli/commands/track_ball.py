"""Track ball command - detect ball positions in volleyball videos."""

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn

from rallycut.cli.utils import handle_errors, validate_video_file
from rallycut.tracking.ball_tracker import BallTracker

console = Console()


@handle_errors
def track_ball(
    video: Path = typer.Argument(
        ...,
        exists=True,
        help="Input video file to track ball positions",
    ),
    output: Path = typer.Option(
        None,
        "--output", "-o",
        help="Output JSON file for ball positions (default: video_ball_track.json)",
    ),
    start_ms: int | None = typer.Option(
        None,
        "--start",
        help="Start time in milliseconds",
    ),
    end_ms: int | None = typer.Option(
        None,
        "--end",
        help="End time in milliseconds",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet", "-q",
        help="Suppress progress output",
    ),
) -> None:
    """Track ball positions in a volleyball video.

    Uses a lightweight ONNX model optimized for CPU (~100 FPS).
    Outputs ball coordinates with confidence scores for each frame.

    Example:
        rallycut track-ball game.mp4 --start 5000 --end 15000 -o rally_ball.json
    """
    validate_video_file(video)

    # Default output path
    if output is None:
        output = video.with_name(f"{video.stem}_ball_track.json")

    if not quiet:
        console.print(f"[bold]Ball Tracking:[/bold] {video.name}")
        if start_ms is not None or end_ms is not None:
            time_range = f"{start_ms or 0}ms - {end_ms or 'end'}ms"
            console.print(f"[dim]Time range: {time_range}[/dim]")

    # Create tracker
    tracker = BallTracker()

    # Track with progress
    if quiet:
        result = tracker.track_video(
            video,
            start_ms=start_ms,
            end_ms=end_ms,
        )
    else:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Tracking ball...", total=100)

            def update_progress(p: float) -> None:
                progress.update(task, completed=int(p * 100))

            result = tracker.track_video(
                video,
                start_ms=start_ms,
                end_ms=end_ms,
                progress_callback=update_progress,
            )

    # Save results
    result.to_json(output)

    # Print summary
    if not quiet:
        console.print(f"\n[green]Ball tracking complete![/green]")
        console.print(f"  Frames processed: {result.frame_count}")
        console.print(f"  Detection rate: {result.detection_rate * 100:.1f}%")
        console.print(f"  Processing time: {result.processing_time_ms / 1000:.2f}s")
        console.print(f"  Output: {output}")

        # Performance stats
        if result.frame_count > 0 and result.processing_time_ms > 0:
            fps = result.frame_count / (result.processing_time_ms / 1000)
            console.print(f"  Speed: {fps:.1f} FPS")

        # Warning for low detection rate
        if result.detection_rate < 0.25:
            console.print(
                "\n[yellow]Warning:[/yellow] Low detection rate. "
                "Ball may be out of frame or occluded in many frames."
            )
