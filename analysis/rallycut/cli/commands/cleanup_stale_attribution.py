"""CLI: rallycut cleanup-stale-attribution <video-id> [--dry-run]

One-shot cleanup that filters per-rally `positions_json` and
`actions_json["actions"]` to drop entries referencing track IDs not in
`primary_track_ids`. Closes I-2 (positions leakage), I-3 and I-7 (action
attribution leakage) on legacy data persisted before Task 12's silent-skip
fix in compute_match_stats.

Idempotent. Includes `--dry-run` mode for safe preview before mutating.
"""

from __future__ import annotations

import json
from typing import Any, cast

import typer
from rich.console import Console

from rallycut.evaluation.tracking.db import get_connection

console = Console()


def _filter_positions(
    positions_json: list[dict[str, Any]], primary: set[int]
) -> tuple[list[dict[str, Any]], int]:
    """Keep only positions whose trackId is in primary. Return (filtered, n_dropped)."""
    kept: list[dict[str, Any]] = []
    dropped = 0
    for p in positions_json:
        try:
            tid = int(p.get("trackId", -1))
        except (TypeError, ValueError):
            dropped += 1
            continue
        if tid in primary:
            kept.append(p)
        else:
            dropped += 1
    return kept, dropped


def _filter_actions(
    actions: list[dict[str, Any]], primary: set[int]
) -> tuple[list[dict[str, Any]], int]:
    """Keep only actions whose playerTrackId is in primary ∪ {-1}. Return (filtered, n_dropped)."""
    allowed = primary | {-1}
    kept: list[dict[str, Any]] = []
    dropped = 0
    for a in actions:
        try:
            tid = int(a.get("playerTrackId", -1))
        except (TypeError, ValueError):
            dropped += 1
            continue
        if tid in allowed:
            kept.append(a)
        else:
            dropped += 1
    return kept, dropped


def cleanup_stale_attribution_cmd(
    video_id: str = typer.Argument(..., help="Video UUID"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview changes without writing to DB"
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Suppress per-rally info"
    ),
) -> None:
    """Drop legacy positions/actions referencing non-primary track IDs (I-2/I-3/I-7)."""
    prefix = "[DRY RUN] " if dry_run else ""
    if not quiet:
        console.print(
            f"[dim]{prefix}Cleaning up stale attribution for video {video_id}…[/dim]"
        )

    rally_query = """
        SELECT
            r.id AS rally_id,
            pt.primary_track_ids,
            pt.positions_json,
            pt.actions_json
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE r.video_id = %s
        ORDER BY r.start_ms
    """

    n_updated = 0
    n_no_change = 0
    n_skipped = 0
    total_positions_dropped = 0
    total_actions_dropped = 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(rally_query, [video_id])
            rows = cur.fetchall()

        for row in rows:
            rally_id = cast(str, row[0])
            primary_raw = row[1]
            positions_json = row[2]
            actions_json = row[3]

            if (
                not isinstance(primary_raw, list)
                or not primary_raw
                or not isinstance(positions_json, list)
                or not isinstance(actions_json, dict)
            ):
                n_skipped += 1
                if not quiet:
                    console.print(
                        f"  [yellow][skip][/yellow] rally {rally_id}: "
                        f"missing primary_track_ids, positions, or actions_json"
                    )
                continue

            primary = {int(t) for t in primary_raw}

            new_positions, n_pos_dropped = _filter_positions(positions_json, primary)
            actions_list = actions_json.get("actions")
            if isinstance(actions_list, list):
                new_actions, n_act_dropped = _filter_actions(actions_list, primary)
            else:
                new_actions = []
                n_act_dropped = 0

            if n_pos_dropped == 0 and n_act_dropped == 0:
                n_no_change += 1
                if not quiet:
                    console.print(
                        f"  [dim][noop][/dim] rally {rally_id}: already clean"
                    )
                continue

            total_positions_dropped += n_pos_dropped
            total_actions_dropped += n_act_dropped

            if dry_run:
                if not quiet:
                    console.print(
                        f"  [cyan][DRY][/cyan]   rally {rally_id}: "
                        f"would drop {n_pos_dropped} positions, "
                        f"{n_act_dropped} actions"
                    )
                continue

            new_actions_json = dict(actions_json)
            if isinstance(actions_list, list):
                new_actions_json["actions"] = new_actions

            try:
                with conn.cursor() as wcur:
                    wcur.execute(
                        "UPDATE player_tracks "
                        "SET positions_json = %s, actions_json = %s "
                        "WHERE rally_id = %s",
                        [
                            json.dumps(new_positions),
                            json.dumps(new_actions_json),
                            rally_id,
                        ],
                    )
                conn.commit()
                n_updated += 1
                if not quiet:
                    console.print(
                        f"  [green][fix][/green]  rally {rally_id}: "
                        f"dropped {n_pos_dropped} positions, "
                        f"{n_act_dropped} actions"
                    )
            except Exception as exc:
                conn.rollback()
                if not quiet:
                    console.print(
                        f"  [red][err][/red]  rally {rally_id}: "
                        f"write failed: {exc}"
                    )

    summary_label = "Dry-run" if dry_run else "Summary"
    console.print(
        f"\n[bold]{summary_label}:[/bold] "
        f"{n_updated} updated · {n_no_change} no-change · {n_skipped} skipped · "
        f"{total_positions_dropped} positions · {total_actions_dropped} actions "
        f"{'(would be) ' if dry_run else ''}dropped"
    )
