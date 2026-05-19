"""Refresh rally_action_ground_truth.snapshot_team from current teamAssignments.

`snapshot_team` is auto-captured from the THEN-CURRENT
`player_tracks.actions_json.teamAssignments` value at GT-save time
(see `api/src/services/actionGroundTruthService.ts:367-368`). It is
never user-asserted — the labeler picks a player, and the system
records the team value that happened to be in teamAssignments.

If teamAssignments was wrong at save time (e.g., the
`redetect_all_actions.py` legacy-fallback bug fixed
2026-05-19), snapshot_team is also wrong — frozen at the bad value.

This script refreshes snapshot_team for every GT row by re-reading the
current positional teamAssignments. Safe because snapshot_team has no
user-assertion semantic. Preserves snapshot_team when the current
lookup misses (no teamAssignments entry for resolved_track_id) so we
never wipe a valid old snapshot with NULL.

Use cases:
- After a fleet-wide fix to teamAssignments (the 2026-05-19 callsite fix)
- After a court-calibration restoration that changes which players are near/far
- After any matcher-side change that re-anchors PIDs

Usage:
    cd analysis
    uv run python scripts/refresh_snapshot_team_from_teamassignments.py --dry-run
    uv run python scripts/refresh_snapshot_team_from_teamassignments.py --apply
    uv run python scripts/refresh_snapshot_team_from_teamassignments.py --apply --video <id>

See:
    side_switch_kuku_koko_diagnostic_2026_05_19 memory entry.
"""

from __future__ import annotations

import argparse
import sys

from rich.console import Console

from rallycut.evaluation.db import get_connection

console = Console()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Refresh snapshot_team from current teamAssignments"
    )
    parser.add_argument(
        "--apply", action="store_true",
        help="Write changes to DB (default: dry run)",
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Restrict to a single video ID",
    )
    args = parser.parse_args()

    where_clauses = ["rg.resolved_track_id IS NOT NULL"]
    params: list[str] = []
    if args.video:
        where_clauses.append("r.video_id = %s")
        params.append(args.video)
    where_sql = " AND ".join(where_clauses)

    diff_query = f"""
        SELECT
          rg.id,
          rg.rally_id,
          v.name AS vname,
          rg.resolved_track_id::int AS pid,
          rg.snapshot_team::text AS old_snap,
          CASE
            WHEN pt.actions_json->'teamAssignments'->>(rg.resolved_track_id::text) = 'A' THEN 'A'
            WHEN pt.actions_json->'teamAssignments'->>(rg.resolved_track_id::text) = 'B' THEN 'B'
            ELSE NULL
          END AS new_snap
        FROM rally_action_ground_truth rg
        JOIN rallies r ON r.id = rg.rally_id
        JOIN videos v ON v.id = r.video_id
        LEFT JOIN player_tracks pt ON pt.rally_id = rg.rally_id
        WHERE {where_sql}
    """

    update_sql = f"""
        UPDATE rally_action_ground_truth rg
        SET snapshot_team = refresh.new_snap::"ServingTeam"
        FROM (
          SELECT
            rg.id,
            CASE
              WHEN pt.actions_json->'teamAssignments'->>(rg.resolved_track_id::text) = 'A' THEN 'A'
              WHEN pt.actions_json->'teamAssignments'->>(rg.resolved_track_id::text) = 'B' THEN 'B'
              ELSE NULL
            END AS new_snap
          FROM rally_action_ground_truth rg
          JOIN rallies r ON r.id = rg.rally_id
          LEFT JOIN player_tracks pt ON pt.rally_id = rg.rally_id
          WHERE {where_sql}
        ) AS refresh
        WHERE rg.id = refresh.id
          AND refresh.new_snap IS NOT NULL
          AND rg.snapshot_team IS DISTINCT FROM refresh.new_snap::"ServingTeam"
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(diff_query, params)
            rows = cur.fetchall()

            total = len(rows)
            unchanged = sum(1 for r in rows if r[4] == r[5])
            would_flip = sum(
                1 for r in rows
                if r[4] is not None and r[5] is not None and r[4] != r[5]
            )
            would_populate = sum(
                1 for r in rows
                if r[4] is None and r[5] is not None
            )
            would_null_skipped = sum(
                1 for r in rows
                if r[4] is not None and r[5] is None
            )
            would_change = would_flip + would_populate

            console.print(f"[bold]Scanned {total} GT rows[/bold]"
                          f"{f' for video {args.video}' if args.video else ' fleet-wide'}")
            console.print(f"  Unchanged (already aligned): {unchanged}")
            console.print(f"  Would flip A↔B:             {would_flip}")
            console.print(f"  Would populate (was null):   {would_populate}")
            console.print(f"  Skipped (current lookup null): {would_null_skipped}")
            console.print(f"  [bold]Total changes:             {would_change}[/bold]")

            # Per-video breakdown of flips
            from collections import Counter
            flip_by_video: Counter[str] = Counter()
            for r in rows:
                if r[4] is not None and r[5] is not None and r[4] != r[5]:
                    flip_by_video[str(r[2])] += 1
            if flip_by_video:
                console.print("\n[bold]Flips by video (top 10):[/bold]")
                for vname, n in flip_by_video.most_common(10):
                    console.print(f"  {vname}: {n}")

            if not args.apply:
                console.print("\n[yellow]Dry run — use --apply to write changes to DB[/yellow]")
                return

            if would_change == 0:
                console.print("\n[green]Nothing to do.[/green]")
                return

            cur.execute(update_sql, params)
            affected = cur.rowcount
            conn.commit()
            console.print(f"\n[green]Applied {affected} updates.[/green]")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[red]Interrupted[/red]")
        sys.exit(1)
