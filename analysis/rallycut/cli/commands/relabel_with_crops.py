"""relabel-with-crops CLI: removed in ref-crop cleanup (phase 1).

The command body now raises immediately; the whole file is deleted in
phase 4. Kept here only because main.py still registers it via Typer at
import time — removing the registration is also part of phase 4.
"""

from __future__ import annotations

import typer
from rich.console import Console

from rallycut.cli.utils import handle_errors

console = Console()


@handle_errors
def relabel_with_crops_cmd(
    video_id: str = typer.Argument(
        ...,
        help="Video ID to relabel using current DB reference crops",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet", "-q",
        help="Suppress progress output",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would change without writing match_analysis_json",
    ),
) -> None:
    """Removed in ref-crop cleanup; the whole command is deleted in phase 4."""
    del video_id, quiet, dry_run
    console.print(
        "[red]Error[/red]: relabel-with-crops is removed. "
        "The ref-crop matcher path has been dropped."
    )
    raise typer.Exit(1)
