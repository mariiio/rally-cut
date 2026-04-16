"""Retro-fix: linearly interpolate stored primary tracks to every rally frame.

Older tracking pipeline versions sometimes left *secondary* primary tracks
(tracks promoted into the primary set after Stage 10 `interpolate_player_gaps`
ran) at YOLO's native stride (e.g. even frames only). When Ground Truth is
annotated on the opposite parity, Hungarian-at-exact-frame can never match,
driving false `filter_drop` / `severe_loss` signals in the audit.

Current pipeline code (`player_tracker.py:1758`) runs interpolation after
primary selection is finalized — so fresh tracking runs are clean. This
command is a DB-only retro-fix for the 10 legacy rallies that pre-date the
change. It rewrites `player_tracks.positions_json` in place; track IDs
are NOT modified, so downstream couplings (actions, contacts, teamTemplates,
pose cache) are preserved.

Usage:
    rallycut reinterpolate-primary --video-id <video-id>
    rallycut reinterpolate-primary --rally-id <rally-id>
    rallycut reinterpolate-primary --all-affected
    rallycut reinterpolate-primary --rally-id <id> --dry-run
"""

from __future__ import annotations

import json
from typing import Any, cast

import typer
from rich.console import Console
from rich.table import Table

from rallycut.cli.utils import handle_errors

console = Console()


def _detect_affected(positions: list[dict[str, Any]], primary: set[int]) -> bool:
    """Return True if any primary track is stuck on single-parity frames.

    Criterion: the track has ≥ 30 positions, 0 frames on the opposite parity,
    AND > 80 % of its frame gaps are exactly 2 (stride-2 cadence).
    """
    per_track: dict[int, list[int]] = {}
    for p in positions:
        tid = p["trackId"]
        if tid in primary:
            per_track.setdefault(tid, []).append(p["frameNumber"])
    for fs in per_track.values():
        if len(fs) < 30:
            continue
        even = sum(1 for f in fs if f % 2 == 0)
        odd = len(fs) - even
        if even > 0 and odd > 0:
            continue
        fs_sorted = sorted(fs)
        gap2 = sum(
            1 for i in range(len(fs_sorted) - 1)
            if fs_sorted[i + 1] - fs_sorted[i] == 2
        )
        if gap2 > len(fs_sorted) * 0.8:
            return True
    return False


def _interpolate(
    positions: list[dict[str, Any]],
    primary: set[int],
    max_gap: int = 10,
) -> tuple[list[dict[str, Any]], int]:
    """Fill single-frame gaps for primary tracks. Returns (new_positions, n_added)."""
    per_track: dict[int, dict[int, dict[str, Any]]] = {}
    for p in positions:
        if p["trackId"] in primary:
            per_track.setdefault(p["trackId"], {})[p["frameNumber"]] = p

    new_positions: list[dict[str, Any]] = []
    for tid, frame_map in per_track.items():
        frames = sorted(frame_map)
        for i in range(len(frames) - 1):
            f1, f2 = frames[i], frames[i + 1]
            gap = f2 - f1
            if gap <= 1 or gap > max_gap:
                continue
            p1, p2 = frame_map[f1], frame_map[f2]
            for f in range(f1 + 1, f2):
                t = (f - f1) / gap
                new_positions.append({
                    "frameNumber": f,
                    "trackId": tid,
                    "x": p1["x"] + t * (p2["x"] - p1["x"]),
                    "y": p1["y"] + t * (p2["y"] - p1["y"]),
                    "width": p1["width"] + t * (p2["width"] - p1["width"]),
                    "height": p1["height"] + t * (p2["height"] - p1["height"]),
                    "confidence": 0.5,
                })
    if not new_positions:
        return positions, 0
    combined = positions + new_positions
    combined.sort(key=lambda p: (p["frameNumber"], p["trackId"]))
    return combined, len(new_positions)


@handle_errors
def reinterpolate_primary_cmd(
    video_id: str = typer.Option(
        None, "--video-id", "-v",
        help="Process every rally in this video",
    ),
    rally_id: str = typer.Option(
        None, "--rally-id", "-r",
        help="Process a single rally",
    ),
    all_affected: bool = typer.Option(
        False, "--all-affected",
        help="Process every rally in the DB that trips the affected-detector",
    ),
    dry_run: bool = typer.Option(
        False, "--dry-run",
        help="Report changes without writing to DB",
    ),
    max_gap: int = typer.Option(
        10, "--max-gap",
        help="Skip gaps larger than this (frames)",
    ),
) -> None:
    """Retrofit stride-parity interpolation for primary tracks in stored rallies.

    Targets stale tracking data from pre-2026-04 pipeline revisions where a
    secondary primary track was promoted after the final interpolation step
    and left at YOLO's native stride. Writes positions_json only; track IDs
    and all downstream JSON are untouched.
    """
    from rallycut.evaluation.db import get_connection

    if not video_id and not rally_id and not all_affected:
        console.print(
            "[red]Error:[/red] pass --video-id, --rally-id, or --all-affected"
        )
        raise typer.Exit(1)

    where_clauses = ["pt.status = 'COMPLETED'",
                     "pt.positions_json IS NOT NULL",
                     "pt.primary_track_ids IS NOT NULL"]
    params: list[Any] = []
    if rally_id:
        where_clauses.append("pt.rally_id = %s")
        params.append(rally_id)
    elif video_id:
        where_clauses.append("r.video_id = %s")
        params.append(video_id)
    where_sql = " AND ".join(where_clauses)

    sql = f"""
        SELECT pt.rally_id, pt.primary_track_ids, pt.positions_json
        FROM player_tracks pt
        JOIN rallies r ON r.id = pt.rally_id
        WHERE {where_sql}
    """

    table = Table(
        title="Primary-track stride reinterpolation",
        show_header=True,
        header_style="bold",
    )
    table.add_column("Rally", style="cyan")
    table.add_column("Affected?", justify="center")
    table.add_column("Positions +", justify="right")
    table.add_column("New total", justify="right")
    table.add_column("Status")

    rally_rows: list[tuple[str, list[int], list[dict[str, Any]]]] = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            for rid, primary_ids, pos_json in cur.fetchall():
                rally_rows.append((
                    str(rid),
                    list(cast(list[int], primary_ids or [])),
                    cast(list[dict[str, Any]], pos_json or []),
                ))

    if not rally_rows:
        console.print("[yellow]No rallies matched the filter.[/yellow]")
        raise typer.Exit(0)

    total_affected = 0
    total_added = 0
    updates: list[tuple[str, list[dict[str, Any]]]] = []

    for rid, primary_ids, positions in rally_rows:
        primary = set(primary_ids)
        affected = _detect_affected(positions, primary)
        if all_affected and not affected:
            continue
        if not affected:
            table.add_row(rid[:8], "no", "0", str(len(positions)), "skipped")
            continue

        new_positions, n_added = _interpolate(positions, primary, max_gap=max_gap)
        total_affected += 1
        total_added += n_added
        status = "[yellow]dry-run[/yellow]" if dry_run else "[green]will write[/green]"
        table.add_row(rid[:8], "yes", f"{n_added}", str(len(new_positions)), status)
        if not dry_run:
            updates.append((rid, new_positions))

    console.print(table)
    console.print(
        f"\n[bold]{total_affected}[/bold] rally(s) affected · "
        f"+[bold]{total_added}[/bold] interpolated positions total"
    )

    if not updates:
        return

    with get_connection() as conn:
        with conn.cursor() as cur:
            for rid, new_positions in updates:
                cur.execute(
                    "UPDATE player_tracks SET positions_json = %s::jsonb "
                    "WHERE rally_id = %s",
                    [json.dumps(new_positions), rid],
                )
        conn.commit()
    console.print(f"[green]Wrote updates for {len(updates)} rally(s).[/green]")
