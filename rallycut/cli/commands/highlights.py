"""Highlights command for generating highlight reels."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.table import Table

from rallycut.cli.utils import handle_errors
from rallycut.core.cache import AnalysisCache
from rallycut.core.video import Video
from rallycut.processing.cutter import VideoCutter
from rallycut.processing.highlights import HighlightGenerator

console = Console()


@handle_errors
def highlights(
    video_path: Path = typer.Argument(
        ...,
        help="Path to input video file",
        exists=True,
        dir_okay=False,
    ),
    output: Optional[Path] = typer.Option(
        None,
        "-o", "--output",
        help="Output video path (default: input_highlights.mp4)",
    ),
    count: int = typer.Option(
        5,
        "-n", "--count",
        help="Number of highlights to include",
    ),
    individual: bool = typer.Option(
        False,
        "-i", "--individual",
        help="Export as individual clips instead of one video",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show rankings without generating video",
    ),
    by_score: bool = typer.Option(
        False,
        "--by-score",
        help="Order clips by score instead of chronologically",
    ),
    min_duration: float = typer.Option(
        8.0,
        "--min-duration",
        help="Minimum rally duration to consider (seconds)",
    ),
    padding_start: float = typer.Option(
        1.0,
        "--padding-start",
        help="Padding before each rally (seconds)",
    ),
    padding_end: float = typer.Option(
        2.0,
        "--padding-end",
        help="Padding after each rally (seconds)",
    ),
    quick: bool = typer.Option(
        False,
        "--quick",
        help="Use fast motion detection instead of ML",
    ),
    stride: int = typer.Option(
        8,
        "--stride",
        help="Frame stride for analysis (lower = more accurate, slower)",
    ),
    gpu: Optional[bool] = typer.Option(
        None,
        "--gpu/--no-gpu",
        help="Force GPU/CPU for ML inference",
    ),
    limit: Optional[float] = typer.Option(
        None,
        "--limit",
        help="Only analyze first N seconds (for testing)",
    ),
    no_cache: bool = typer.Option(
        False,
        "--no-cache",
        help="Force re-analysis even if cached",
    ),
    proxy: bool = typer.Option(
        True,
        "--proxy/--no-proxy",
        help="Use low-res proxy for faster ML analysis (default: on)",
    ),
    two_pass: bool = typer.Option(
        True,
        "--two-pass/--full-scan",
        help="Use two-pass: motion scan then ML on motion only (2-3x faster)",
    ),
    min_gap: float = typer.Option(
        3.5,
        "--min-gap",
        help="Min NO_PLAY gap (seconds) before ending a rally (higher = longer rallies)",
    ),
):
    """
    Generate highlight reel from top rallies.

    Analyzes the video to find rallies, ranks them by duration/excitement,
    and exports the top N as a highlight video or individual clips.

    Examples:
        rallycut highlights match.mp4
        rallycut highlights match.mp4 -n 10 -o best_rallies.mp4
        rallycut highlights match.mp4 --individual -o clips/
        rallycut highlights match.mp4 --by-score
    """
    # Determine output path
    if output is None:
        if individual:
            output = video_path.parent / f"{video_path.stem}_highlights"
        else:
            output = video_path.with_stem(f"{video_path.stem}_highlights")

    # Determine device
    if gpu is None:
        device = None
    elif gpu:
        device = "cuda"
    else:
        device = "cpu"

    console.print("[bold]Highlight Generation[/bold]")
    console.print(f"Input: {video_path}")
    console.print(f"Output: {output}")
    console.print(f"Top {count} rallies")
    console.print()

    # Load video info
    video = Video(str(video_path))

    # Initialize cutter for analysis (no padding - we'll add it ourselves)
    cutter = VideoCutter(
        device=device,
        stride=stride,
        use_quick_mode=quick,
        use_two_pass=two_pass and not quick,  # Two-pass only when not quick mode
        limit_seconds=limit,
        padding_seconds=0,  # We apply asymmetric padding below
        proxy_height=360 if proxy else None,
        min_gap_seconds=min_gap,
    )

    # Check cache first (only if not using --limit which affects results)
    cache = AnalysisCache()
    segments = None

    if not no_cache and limit is None:
        segments = cache.get(video_path, stride, quick, proxy)
        if segments:
            console.print("[dim]Using cached analysis[/dim]")

    # Analyze video if not cached
    if segments is None:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Analyzing video...", total=100)

            def analysis_progress(pct: float, msg: str):
                progress.update(task, completed=int(pct * 100))

            segments = cutter.analyze_only(video_path, analysis_progress)
            progress.update(task, completed=100)

        # Cache results (only if not using --limit)
        if limit is None and segments:
            cache.set(video_path, stride, quick, segments, proxy)

    if not segments:
        console.print("[red]No rallies detected in video[/red]")
        raise typer.Exit(1)

    console.print(f"Found {len(segments)} rallies")

    # Apply asymmetric padding to segments
    from rallycut.core.models import TimeSegment

    fps = video.info.fps
    duration = video.info.duration
    padded_segments = []
    for seg in segments:
        new_start = max(0, seg.start_time - padding_start)
        new_end = min(duration, seg.end_time + padding_end)
        padded_segments.append(TimeSegment(
            start_frame=int(new_start * fps),
            end_frame=int(new_end * fps),
            start_time=new_start,
            end_time=new_end,
            state=seg.state,
        ))

    # Merge overlapping segments to avoid jumps in output
    padded_segments.sort(key=lambda s: s.start_time)
    merged_segments = []
    for seg in padded_segments:
        if merged_segments and seg.start_time <= merged_segments[-1].end_time:
            # Overlapping - merge
            last = merged_segments[-1]
            merged_segments[-1] = TimeSegment(
                start_frame=last.start_frame,
                end_frame=max(last.end_frame, seg.end_frame),
                start_time=last.start_time,
                end_time=max(last.end_time, seg.end_time),
                state=last.state,
            )
        else:
            merged_segments.append(seg)

    segments = merged_segments

    # Generate highlights
    generator = HighlightGenerator()
    generator.scorer.min_duration = min_duration

    # Get scored highlights
    top_highlights = generator.scorer.get_top_highlights(segments, count)

    if not top_highlights:
        console.print("[red]No rallies found matching criteria[/red]")
        raise typer.Exit(1)

    if dry_run:
        # Just show the rankings
        table = Table(title=f"Top {len(top_highlights)} Highlights")
        table.add_column("Rank", style="cyan")
        table.add_column("Duration", style="green")
        table.add_column("Time", style="yellow")
        table.add_column("Score", style="magenta")

        total_duration = 0
        for highlight in top_highlights:
            total_duration += highlight.duration
            table.add_row(
                f"#{highlight.rank}",
                f"{highlight.duration:.1f}s",
                f"{highlight.start_time:.1f}s - {highlight.end_time:.1f}s",
                f"{highlight.score:.2f}",
            )

        console.print(table)
        console.print()
        console.print(f"Total highlight duration: {total_duration:.1f}s")
        return

    if individual:
        # Export individual clips
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Exporting clips...", total=100)

            def export_progress(pct: float, msg: str):
                progress.update(task, completed=int(pct * 100))

            results = generator.export_individual_clips(
                input_path=video_path,
                output_dir=output,
                segments=segments,
                count=count,
                progress_callback=export_progress,
            )
            progress.update(task, completed=100)

        # Show results table
        console.print()
        table = Table(title="Exported Highlights")
        table.add_column("Rank", style="cyan")
        table.add_column("Duration", style="green")
        table.add_column("Time", style="yellow")
        table.add_column("File")

        for clip_path, highlight in results:
            table.add_row(
                f"#{highlight.rank}",
                f"{highlight.duration:.1f}s",
                f"{highlight.start_time:.1f}s - {highlight.end_time:.1f}s",
                clip_path.name,
            )

        console.print(table)
        console.print()
        console.print(f"[green]Exported {len(results)} clips to: {output}/[/green]")

    else:
        # Export single highlights video
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            task = progress.add_task("Generating highlights...", total=100)

            def export_progress(pct: float, msg: str):
                progress.update(task, completed=int(pct * 100))

            _, top_highlights = generator.generate_highlights(
                input_path=video_path,
                output_path=output,
                segments=segments,
                count=count,
                chronological=not by_score,
                progress_callback=export_progress,
            )
            progress.update(task, completed=100)

        # Show results table
        console.print()
        table = Table(title="Included Highlights")
        table.add_column("Rank", style="cyan")
        table.add_column("Duration", style="green")
        table.add_column("Time", style="yellow")
        table.add_column("Score", style="magenta")

        total_duration = 0
        for highlight in sorted(top_highlights, key=lambda h: h.start_time):
            total_duration += highlight.duration
            table.add_row(
                f"#{highlight.rank}",
                f"{highlight.duration:.1f}s",
                f"{highlight.start_time:.1f}s - {highlight.end_time:.1f}s",
                f"{highlight.score:.2f}",
            )

        console.print(table)
        console.print()
        console.print(f"Total highlight duration: {total_duration:.1f}s")
        console.print(f"[green]Output saved to: {output}[/green]")
