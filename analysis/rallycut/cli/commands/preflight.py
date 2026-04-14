"""`rallycut preflight <video>` — run all quality checks, emit JSON."""
from __future__ import annotations

import json
from pathlib import Path

import typer

from rallycut.quality.runner import run_full_preflight


def preflight(
    video: Path = typer.Argument(..., exists=True, dir_okay=False, readable=True),
    sample_seconds: int = typer.Option(60, "--sample-seconds"),
    as_json: bool = typer.Option(True, "--json/--no-json"),
    quiet: bool = typer.Option(False, "--quiet"),
):
    """Run the full preflight quality check and print a JSON QualityReport."""
    report = run_full_preflight(str(video), sample_seconds=sample_seconds)
    d = report.to_dict()
    if as_json:
        typer.echo(json.dumps(d, indent=None if quiet else 2))
    else:
        for issue in d["issues"]:
            typer.echo(f"[{issue['tier'].upper()}] {issue['message']}")
