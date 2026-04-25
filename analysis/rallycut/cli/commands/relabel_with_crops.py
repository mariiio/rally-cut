"""relabel-with-crops CLI: re-run Pass 2 stages 1+2 against new ref crops.

Phase 1.1 of the identity-layer rebuild. Reads the rallyScratchpad +
playerProfiles persisted by `match-players`, rebuilds frozen profiles
from the current player_reference_crops in the DB, runs
`replay_refine_from_scratchpad`, and writes the new trackToPlayer
back to match_analysis_json.

The user-facing flow:
    1. `match-players <video-id>`  — blind pass, writes rallyScratchpad
    2. user uploads ref crops      — populates player_reference_crops table
    3. `relabel-with-crops <video-id>`  — this command
    4. `reattribute-actions <video-id>` — propagates new pids to actions
"""

from __future__ import annotations

import json
from typing import Any

import typer
from rich.console import Console
from rich.table import Table

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
    """Relabel rally pids by replaying Pass 2 with current reference crops.

    Requires that `match-players` has already run (so match_analysis_json
    has rallyScratchpad + playerProfiles) AND that at least one row exists
    in player_reference_crops for this video.

    Example:
        rallycut relabel-with-crops abc123
        rallycut relabel-with-crops abc123 --dry-run
    """
    import time

    from rallycut.cli.commands.match_players import _load_db_reference_crops
    from rallycut.evaluation.db import get_connection
    from rallycut.evaluation.tracking.db import get_video_path
    from rallycut.tracking.match_tracker import replay_refine_from_scratchpad
    from rallycut.tracking.relabel import (
        apply_relabel_to_rally_entries,
        reconstruct_initial_results,
        reconstruct_profiles,
    )

    if not quiet:
        console.print(f"[bold]Relabel-with-crops[/bold] for {video_id}")

    # 1. Load match_analysis_json — must have scratchpad + profiles.
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT match_analysis_json FROM videos WHERE id = %s",
                [video_id],
            )
            row = cur.fetchone()

    if not row or not row[0]:
        console.print(
            "[red]Error[/red]: video has no match_analysis_json. "
            "Run `rallycut match-players` first."
        )
        raise typer.Exit(1)

    raw = row[0]
    if isinstance(raw, dict):
        match_analysis: dict[str, Any] = raw
    elif isinstance(raw, (str, bytes, bytearray)):
        match_analysis = json.loads(raw)
    else:
        console.print(
            f"[red]Error[/red]: unexpected match_analysis_json type {type(raw).__name__}"
        )
        raise typer.Exit(1)

    scratchpad = match_analysis.get("rallyScratchpad")
    if not scratchpad:
        console.print(
            "[red]Error[/red]: match_analysis_json has no rallyScratchpad. "
            "Re-run `rallycut match-players` after upgrading; older runs "
            "predate Phase 0 and lack the per-rally scratchpad."
        )
        raise typer.Exit(1)

    profiles_dict = match_analysis.get("playerProfiles", {})
    if not profiles_dict:
        console.print(
            "[red]Error[/red]: match_analysis_json has no playerProfiles."
        )
        raise typer.Exit(1)

    rally_entries: list[dict[str, Any]] = match_analysis.get("rallies", [])
    if not rally_entries:
        console.print("[yellow]Warning[/yellow]: no rallies in match_analysis_json — nothing to relabel.")
        return

    # 2. Load DB reference crops + build frozen profiles.
    video_path = get_video_path(video_id)
    if video_path is None:
        console.print(f"[red]Error[/red]: could not resolve video path for {video_id}")
        raise typer.Exit(1)

    if not quiet:
        console.print("Loading reference crops from DB...")
    _, reference_profiles, _ = _load_db_reference_crops(video_id, video_path, quiet=quiet)
    if not reference_profiles:
        console.print(
            "[red]Error[/red]: no rows found in player_reference_crops for "
            f"{video_id}. Upload reference crops before running relabel."
        )
        raise typer.Exit(1)

    # 3. Reconstruct typed objects from match_analysis_json.
    initial_results = reconstruct_initial_results(rally_entries)
    blind_profiles = reconstruct_profiles(profiles_dict)

    # Merge: anchored pids come from reference_profiles (frozen); any pid
    # the user did NOT provide a crop for inherits the blind profile.
    merged_profiles = dict(blind_profiles)
    for pid, prof in reference_profiles.items():
        merged_profiles[pid] = prof

    # 4. Replay Pass 2 stages 1+2.
    t0 = time.time()
    refined = replay_refine_from_scratchpad(
        scratchpad=scratchpad,
        player_profiles=merged_profiles,
        initial_results=initial_results,
    )
    elapsed = time.time() - t0
    if not quiet:
        console.print(f"Replay complete in {elapsed:.2f}s ({len(refined)} rallies)")

    # 5. Build the new rally_entries.
    new_entries = apply_relabel_to_rally_entries(rally_entries, refined)

    # 6. Show diff summary.
    n_changed = 0
    for orig, new in zip(rally_entries, new_entries):
        if orig.get("trackToPlayer") != new.get("trackToPlayer"):
            n_changed += 1
    if not quiet:
        table = Table(title=f"Relabel diff: {n_changed}/{len(new_entries)} rallies changed")
        table.add_column("Rally", style="cyan")
        table.add_column("Conf", justify="right")
        table.add_column("Δ trackToPlayer")
        for orig, new in zip(rally_entries, new_entries):
            if orig.get("trackToPlayer") == new.get("trackToPlayer"):
                continue
            rally_id = (orig.get("rallyId") or "")[:8]
            conf = f"{new['assignmentConfidence']:.2f}"
            delta_str = (
                f"{orig.get('trackToPlayer', {})} → {new['trackToPlayer']}"
            )
            table.add_row(rally_id, conf, delta_str)
        console.print(table)

    # 7. Write back (unless dry run).
    if dry_run:
        if not quiet:
            console.print("[yellow]Dry run — match_analysis_json NOT updated[/yellow]")
        return

    new_match_analysis = dict(match_analysis)
    new_match_analysis["rallies"] = new_entries

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE videos SET match_analysis_json = %s WHERE id = %s",
                [json.dumps(new_match_analysis), video_id],
            )
        conn.commit()

    if not quiet:
        console.print("  [green]Saved.[/green] Next: rallycut reattribute-actions " + video_id)
