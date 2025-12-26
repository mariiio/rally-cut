"""Overlay command for ball trajectory visualization."""

from pathlib import Path

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

from rallycut.cli.utils import handle_errors
from rallycut.core.video import Video
from rallycut.output.overlay import OverlayRenderer
from rallycut.tracking.ball_tracker import BallTracker
from rallycut.tracking.trajectory import TrajectoryProcessor

console = Console()


@handle_errors
def overlay(
    video_path: Path = typer.Argument(
        ...,
        help="Path to input video file",
        exists=True,
        dir_okay=False,
    ),
    output: Path | None = typer.Option(
        None,
        "-o", "--output",
        help="Output video path (default: input_overlay.mp4)",
    ),
    start: float | None = typer.Option(
        None,
        "-s", "--start",
        help="Start time in seconds",
    ),
    end: float | None = typer.Option(
        None,
        "-e", "--end",
        help="End time in seconds",
    ),
    confidence: float = typer.Option(
        0.3,
        "-c", "--confidence",
        help="Ball detection confidence threshold",
    ),
    trail: int = typer.Option(
        15,
        "-t", "--trail",
        help="Trail length in frames",
    ),
    smooth: float = typer.Option(
        1.5,
        "--smooth",
        help="Trajectory smoothing factor (0 = none)",
    ),
    stride: int = typer.Option(
        1,
        "--stride",
        help="Frame stride for tracking (1 = every frame)",
    ),
    gpu: bool | None = typer.Option(
        None,
        "--gpu/--no-gpu",
        help="Force GPU/CPU for inference",
    ),
) -> None:
    """
    Add ball trajectory overlay to video.

    Tracks the volleyball using ML detection and Kalman filtering,
    then renders a visual trail showing ball movement.

    Examples:
        rallycut overlay match.mp4
        rallycut overlay match.mp4 -s 12 -e 20 -o rally1.mp4
        rallycut overlay match.mp4 --trail 20 --smooth 2.0
    """
    # Determine output path
    if output is None:
        output = video_path.with_stem(f"{video_path.stem}_overlay")

    # Determine device
    if gpu is None:
        device = None  # Auto-detect
    elif gpu:
        device = "cuda"
    else:
        device = "cpu"

    console.print("[bold]Ball Tracking Overlay[/bold]")
    console.print(f"Input: {video_path}")
    console.print(f"Output: {output}")
    console.print()

    # Load video
    video = Video(str(video_path))
    duration = video.info.duration

    # Determine time range
    start_time = start if start is not None else 0.0
    end_time = end if end is not None else duration

    console.print(f"Segment: {start_time:.1f}s - {end_time:.1f}s ({end_time - start_time:.1f}s)")
    console.print()

    # Initialize components
    tracker = BallTracker(
        device=device,
        confidence_threshold=confidence,
        use_predictions=False,  # Only show detected positions
    )
    processor = TrajectoryProcessor(
        max_gap_frames=5,  # Split trajectory if gap > 5 frames (ball likely out of frame)
        smooth_sigma=smooth,
        trail_length=trail,
    )
    renderer = OverlayRenderer(trail_length=trail)

    # Track ball
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        # Tracking phase
        track_task = progress.add_task("Tracking ball...", total=100)

        def track_progress(pct: float, msg: str) -> None:
            progress.update(track_task, completed=int(pct * 100))

        result = tracker.track_segment(
            video,
            start_time=start_time,
            end_time=end_time,
            stride=stride,
        )
        progress.update(track_task, completed=100)

    # Report tracking results
    console.print()
    console.print(f"Detection rate: {result.detection_rate:.1%}")
    console.print(f"Frames tracked: {len(result.positions)}/{result.total_frames}")

    if result.detection_rate < 0.3:
        console.print("[yellow]Warning: Low detection rate - trajectory may be incomplete[/yellow]")

    # Process trajectory
    console.print()
    console.print("Processing trajectory...")
    segments = processor.process(
        result.positions,
        interpolate=True,
        smooth=smooth > 0,
    )
    console.print(f"Trajectory segments: {len(segments)}")

    if not segments:
        console.print("[red]Error: No ball trajectory detected[/red]")
        raise typer.Exit(1)

    # Render overlay
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    ) as progress:
        render_task = progress.add_task("Rendering overlay...", total=100)

        def render_progress(pct: float, msg: str) -> None:
            progress.update(render_task, completed=int(pct * 100))

        renderer.render_segment(
            video,
            segments,
            processor,
            output,
            start_time=start_time,
            end_time=end_time,
            progress_callback=render_progress,
        )
        progress.update(render_task, completed=100)

    console.print()
    console.print(f"[green]Output saved to: {output}[/green]")
