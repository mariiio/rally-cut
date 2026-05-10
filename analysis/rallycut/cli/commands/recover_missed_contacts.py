"""CLI: rallycut recover-missed-contacts <video-id> [--dry-run]

Coherence-driven post-processing pass. For each rally that fires a
coherence-invariant violation, attempts a constrained second-pass
contact detection inside the implied gap window and inserts at most
one recovered contact per gap.

Conservative by design:
  - Only fires on rallies the audit already flags.
  - Hard team-match gate: a recovered contact's nearest player must be
    on the team whose absence caused the violation.
  - Two-signal agreement: GBM conf >= 0.10 AND seq peak >= 0.80.
  - Audit-after-injection guard: commit only if the rally's coherence-
    violation count strictly decreases.

Spec: docs/superpowers/specs/2026-05-10-coherence-driven-contact-recovery-design.md
"""

from __future__ import annotations

import json
from typing import Any

import typer
from rich.console import Console

from rallycut.evaluation.tracking.db import get_connection
from rallycut.tracking.contact_recovery import (
    load_rally_inputs,
    recover_rally,
)

console = Console()


def recover_missed_contacts_cmd(
    video_id: str = typer.Argument(..., help="Video UUID"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview changes without writing to DB"
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Suppress per-rally info"
    ),
) -> None:
    """Recover coherence-flagged missed contacts on a video's persisted state."""
    prefix = "[DRY RUN] " if dry_run else ""
    if not quiet:
        console.print(
            f"[dim]{prefix}Coherence-driven contact recovery for video {video_id}…[/dim]"
        )

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT r.id FROM rallies r
                   WHERE r.video_id = %s
                     AND (r.status = 'CONFIRMED' OR r.status IS NULL)
                   ORDER BY r.start_ms""",
                [video_id],
            )
            rally_ids = [str(row[0]) for row in cur.fetchall()]

    n_rallies = len(rally_ids)
    n_recovered = 0
    n_rejected_gate = 0
    n_rejected_audit = 0
    n_skipped = 0

    for rid in rally_ids:
        try:
            inputs = load_rally_inputs(rid)
        except ValueError:
            n_skipped += 1
            if not quiet:
                console.print(f"  [dim][skip][/dim]   rally {rid[:8]}: no player_tracks row")
            continue

        result = recover_rally(inputs)
        if not result.has_changes:
            if result.gaps_attempted == 0:
                n_skipped += 1
                if not quiet:
                    console.print(
                        f"  [dim][noop][/dim]   rally {rid[:8]}: no gaps"
                    )
                continue
            n_rejected_gate += result.rejected_by_gate
            n_rejected_audit += result.rejected_by_audit
            if not quiet:
                console.print(
                    f"  [yellow][noop][/yellow] rally {rid[:8]}: "
                    f"{result.gaps_attempted} gaps · "
                    f"{result.rejected_by_gate} gate-rejects · "
                    f"{result.rejected_by_audit} audit-rejects"
                )
            continue

        # Build new actions_json with the recovered contacts merged in.
        existing_actions: list[dict[str, Any]] = list(inputs.actions_json.get("actions") or [])
        merged: list[dict[str, Any]] = sorted(
            existing_actions + result.recovered_actions,
            key=lambda a: int(a.get("frame", 0)),
        )
        new_aj: dict[str, Any] = dict(inputs.actions_json)
        new_aj["actions"] = merged
        new_aj["numContacts"] = sum(
            1 for a in merged if a.get("action") and a.get("action") != "unknown"
        )
        new_aj["actionSequence"] = [a.get("action") for a in merged]

        n_recovered_here = len(result.recovered_actions)
        n_recovered += n_recovered_here
        n_rejected_gate += result.rejected_by_gate
        n_rejected_audit += result.rejected_by_audit

        recovery_summary = ", ".join(
            f"{a['action']}@f{a['frame']}({a.get('team','?')})"
            for a in result.recovered_actions
        )

        if dry_run:
            if not quiet:
                console.print(
                    f"  [cyan][DRY][/cyan]  rally {rid[:8]}: would recover "
                    f"{n_recovered_here} contact(s) — {recovery_summary}"
                )
            continue

        try:
            with get_connection() as conn:
                with conn.cursor() as wcur:
                    wcur.execute(
                        "UPDATE player_tracks SET actions_json = %s WHERE rally_id = %s",
                        [json.dumps(new_aj), rid],
                    )
                conn.commit()
            if not quiet:
                console.print(
                    f"  [green][fix][/green]  rally {rid[:8]}: recovered "
                    f"{n_recovered_here} — {recovery_summary}"
                )
        except Exception as exc:  # pragma: no cover — DB error path
            if not quiet:
                console.print(
                    f"  [red][err][/red]  rally {rid[:8]}: write failed: {exc}"
                )

    summary_label = "Dry-run" if dry_run else "Summary"
    console.print(
        f"\n[bold]{summary_label}:[/bold] {n_rallies} rallies · "
        f"{n_recovered} recovered · {n_rejected_gate} gate-rejects · "
        f"{n_rejected_audit} audit-rejects · {n_skipped} skipped"
    )
