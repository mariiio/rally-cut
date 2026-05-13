"""CLI: rallycut audit-coherence-invariants <video-id>

Eval-time game-rule audit. Sibling to audit-pid-invariants. Exits non-zero
on any violation.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from rallycut.tracking.coherence_invariants import run_all
from rallycut.tracking.pid_invariants import StaleVersionReport

console = Console()


def audit_coherence_invariants_cmd(
    video_id: str = typer.Argument(..., help="Video UUID"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress info output"),
) -> None:
    """Audit volleyball-rule coherence for a video's action sequences."""
    if not quiet:
        console.print(
            f"[dim]Running coherence-invariant audit on {video_id}…[/dim]"
        )

    violations, stale = run_all(video_id=video_id)

    _render_stale_header(stale)

    if not violations:
        if not quiet:
            console.print("[green]✓ All coherence invariants hold[/green]")
        raise typer.Exit(code=0)

    table = Table(title=f"Coherence-invariant violations — video {video_id}")
    table.add_column("Invariant", style="bold")
    table.add_column("Rally", style="cyan")
    table.add_column("Severity", style="yellow")
    table.add_column("Detail")
    for v in violations:
        table.add_row(v.invariant, v.rally_id, v.severity, v.detail)
    console.print(table)

    n_errors = sum(1 for v in violations if v.severity == "error")
    n_warns = len(violations) - n_errors
    console.print(
        f"[red]{n_errors} error[/red] · [yellow]{n_warns} warn[/yellow] · "
        f"{len(violations)} total"
    )

    raise typer.Exit(code=1 if n_errors else 0)


def _render_stale_header(stale: StaleVersionReport) -> None:
    """Print the stale-version block at the top of the report."""
    if not stale.has_stale:
        return
    n_stale = len(stale.skipped_stale_actions | stale.skipped_stale_contacts)
    console.print(
        f"\n[yellow]⚠ {n_stale} of {stale.total_rallies} rallies skipped due to stale pipeline version[/yellow]"
    )
    if stale.skipped_stale_actions:
        observed = ", ".join(
            f"{k}:{v}" for k, v in sorted(stale.observed_actions_versions.items())
        )
        console.print(
            f"  - {len(stale.skipped_stale_actions)} stale actions_pipeline_version "
            f"(observed: {{{observed}}}; current: {stale.current_actions_version})"
        )
    if stale.skipped_stale_contacts:
        observed = ", ".join(
            f"{k}:{v}" for k, v in sorted(stale.observed_contacts_versions.items())
        )
        console.print(
            f"  - {len(stale.skipped_stale_contacts)} stale contacts_pipeline_version "
            f"(observed: {{{observed}}}; current: {stale.current_contacts_version})"
        )
    console.print("  Run: uv run python scripts/redetect_all_actions.py --apply\n")
