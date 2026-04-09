"""One-time backfill: set `canonicalLocked: true` on every rally that has
GT action labels, preserving its current `match_analysis_json.trackToPlayer`
across future retracks.

Why: the literal-compare path in `production_eval.py` and in downstream
consumers (web editor, scoreboard, match stats) uses the integer
`player_track_id` stored with GT contacts. Those integers were frozen
against whatever `trackToPlayer` was current when the GT was labeled.
Any future retrack of the video silently reshuffles canonical IDs and
breaks the GT↔pred alignment — which is exactly the drift that motivated
`memory/diagnosis_2026-04-10.md`.

This script locks in the current alignment by setting a `canonicalLocked`
flag on each GT rally inside `videos.match_analysis_json`. The modified
`match_players` CLI respects the flag and preserves the stored
`trackToPlayer` / `assignmentConfidence` / `serverPlayerId` for locked
rallies during any future retrack.

Non-GT rallies are untouched — retracks will continue to recompute their
canonical assignments normally.

Videos with user reference crops already have a stronger anchor (frozen
profiles in `match_tracker.MatchPlayerTracker`); this lock is complementary
and is mainly useful for the 54 eval videos WITHOUT reference crops.

Read-only dry-run by default. Pass --apply to write the flag.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import typer  # noqa: E402
from rich.console import Console  # noqa: E402
from rich.table import Table  # noqa: E402

from rallycut.evaluation.tracking.db import get_connection  # noqa: E402

console = Console()


def _gt_rally_ids() -> set[str]:
    """Rallies that have action ground truth labels."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT r.id
              FROM rallies r
              JOIN player_tracks pt ON pt.rally_id = r.id
             WHERE pt.action_ground_truth_json IS NOT NULL
            """
        )
        return {str(row[0]) for row in cur.fetchall()}


def _videos_with_match_analysis() -> list[tuple[str, str, dict]]:
    """Videos that have match_analysis_json populated."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, filename, match_analysis_json
              FROM videos
             WHERE match_analysis_json IS NOT NULL
            """
        )
        out: list[tuple[str, str, dict]] = []
        for vid, fn, ma in cur.fetchall():
            if isinstance(ma, dict):
                out.append((str(vid), str(fn or ""), ma))
        return out


def main(
    apply: bool = typer.Option(
        False, "--apply", help="Write changes to the DB (default: dry-run)"
    ),
) -> None:
    gt_rally_ids = _gt_rally_ids()
    console.print(f"GT rallies to lock: [bold]{len(gt_rally_ids)}[/bold]")

    videos = _videos_with_match_analysis()
    console.print(f"Videos with match_analysis_json: [bold]{len(videos)}[/bold]")

    table = Table(title="Lock plan", show_lines=False)
    table.add_column("video", style="dim")
    table.add_column("total rallies", justify="right")
    table.add_column("GT rallies", justify="right")
    table.add_column("already locked", justify="right")
    table.add_column("new locks", justify="right", style="green")

    total_new_locks = 0
    updates: list[tuple[str, dict]] = []
    for video_id, filename, ma in videos:
        rallies = ma.get("rallies", [])
        if not isinstance(rallies, list):
            continue

        gt_in_video = 0
        already_locked = 0
        new_locks = 0
        new_rallies: list[dict] = []
        for entry in rallies:
            if not isinstance(entry, dict):
                new_rallies.append(entry)
                continue
            rid = entry.get("rallyId") or entry.get("rally_id", "")
            out = dict(entry)
            if rid in gt_rally_ids:
                gt_in_video += 1
                if entry.get("canonicalLocked") is True:
                    already_locked += 1
                else:
                    # Only lock rallies that have a non-empty trackToPlayer;
                    # locking an empty mapping accomplishes nothing and would
                    # prevent a future successful run from populating it.
                    t2p = entry.get("trackToPlayer") or entry.get("track_to_player")
                    if t2p:
                        out["canonicalLocked"] = True
                        new_locks += 1
            new_rallies.append(out)

        if gt_in_video == 0:
            continue

        if new_locks > 0:
            new_ma = dict(ma)
            new_ma["rallies"] = new_rallies
            updates.append((video_id, new_ma))

        total_new_locks += new_locks
        table.add_row(
            filename or video_id[:8],
            str(len(rallies)),
            str(gt_in_video),
            str(already_locked),
            str(new_locks),
        )

    console.print(table)
    console.print(f"\n[bold]Total new locks: {total_new_locks}[/bold]")

    if not apply:
        console.print("[yellow]DRY RUN[/yellow] — re-run with --apply to write.")
        return

    if not updates:
        console.print("[green]Nothing to write.[/green]")
        return

    with get_connection() as conn:
        with conn.cursor() as cur:
            for video_id, new_ma in updates:
                cur.execute(
                    "UPDATE videos SET match_analysis_json = %s WHERE id = %s",
                    [json.dumps(new_ma), video_id],
                )
        conn.commit()
    console.print(f"[green]Wrote canonical locks to {len(updates)} video(s).[/green]")


if __name__ == "__main__":
    typer.run(main)
