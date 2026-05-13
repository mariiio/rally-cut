"""Shared rendering helpers for the audit CLIs.

Used by both `audit-pid-invariants` and `audit-coherence-invariants` to
print the StaleVersionReport header in a uniform shape. Lives in
`cli/commands/` (alongside its callers) so it's not picked up by
production paths.
"""

from __future__ import annotations

from rich.console import Console

from rallycut.tracking.pid_invariants import StaleVersionReport


def render_stale_header(console: Console, stale: StaleVersionReport) -> None:
    """Print the stale-version block at the top of an audit report.

    Silent when nothing is stale. When non-empty, lists per-category
    skip counts + observed-version histogram + call-to-action. The
    refresh script (`redetect_all_actions.py`) requires non-null
    `ball_positions_json` on each rally — rallies missing ball positions
    cannot be refreshed by the script and will keep surfacing as stale
    until they are re-tracked end-to-end. The CTA line acknowledges this.
    """
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
    console.print(
        "  Run: uv run python scripts/redetect_all_actions.py --apply"
    )
    console.print(
        "  [dim](rallies without ball_positions_json cannot be refreshed by the script; "
        "they need an end-to-end re-track first)[/dim]\n"
    )
