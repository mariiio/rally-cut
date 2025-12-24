"""Profile command - analyze performance bottlenecks."""

import json
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from rallycut.cli.utils import handle_errors
from rallycut.core.config import get_config
from rallycut.core.profiler import enable_profiling, get_profiler
from rallycut.processing.cutter import VideoCutter

console = Console()


@handle_errors
def profile(
    video: Path = typer.Argument(
        ...,
        exists=True,
        help="Input video file to profile",
    ),
    output_json: Optional[Path] = typer.Option(
        None,
        "--output-json", "-o",
        help="Save profile results to JSON file",
    ),
    use_proxy: bool = typer.Option(
        True,
        "--proxy/--no-proxy",
        help="Use proxy video for ML analysis",
    ),
) -> None:
    """Profile video analysis to identify performance bottlenecks.

    Runs the full two-pass analysis pipeline with detailed timing instrumentation.
    Use this to understand where time is spent and identify optimization opportunities.
    """
    config = get_config()
    console.print(f"[bold]Profiling:[/bold] {video.name}")
    console.print(f"[dim]Device: {config.device}[/dim]")

    # Enable profiling
    profiler = enable_profiling()

    # Run analysis with profiling enabled
    console.print("\n[yellow]Running analysis with profiling...[/yellow]\n")

    cutter = VideoCutter(
        use_two_pass=True,
        device=config.device,
        use_proxy=use_proxy,
    )

    with profiler.time("pipeline", "total"):
        segments = cutter.analyze_only(video)

    # Generate report
    report = profiler.report()

    # Display results
    console.print(f"\n[green]Analysis complete![/green]")
    console.print(f"Found {len(segments)} segments\n")

    # Print formatted report
    _print_profile_table(report)

    # Save to JSON if requested
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(report, f, indent=2)
        console.print(f"\n[dim]Profile saved to: {output_json}[/dim]")


def _print_profile_table(report: dict) -> None:
    """Print a formatted profile table."""
    total = report["total_seconds"]

    console.print(f"[bold]Total time:[/bold] {total:.2f}s")
    console.print(f"[dim]Entries: {report['entries_count']}[/dim]\n")

    if not report["by_component"]:
        console.print("[yellow]No profiling data collected[/yellow]")
        return

    # Create table
    table = Table(title="Time Breakdown by Component")
    table.add_column("Component", style="cyan")
    table.add_column("Operation", style="white")
    table.add_column("Time (s)", justify="right")
    table.add_column("%", justify="right")
    table.add_column("Count", justify="right")
    table.add_column("Avg (ms)", justify="right")

    # Sort components by time
    sorted_components = sorted(
        report["by_component"].items(),
        key=lambda x: x[1]["total_seconds"],
        reverse=True,
    )

    for component, data in sorted_components:
        # Add component row
        table.add_row(
            f"[bold]{component}[/bold]",
            "",
            f"[bold]{data['total_seconds']:.2f}[/bold]",
            f"[bold]{data['percentage']:.1f}%[/bold]",
            "",
            "",
        )

        # Add operation rows
        sorted_ops = sorted(
            data["operations"].items(),
            key=lambda x: x[1]["total_seconds"],
            reverse=True,
        )
        for op, op_data in sorted_ops:
            pct = 100 * op_data["total_seconds"] / total if total > 0 else 0
            table.add_row(
                "",
                op,
                f"{op_data['total_seconds']:.2f}",
                f"{pct:.1f}%",
                str(op_data["count"]),
                f"{op_data['avg_seconds'] * 1000:.1f}",
            )

    console.print(table)
