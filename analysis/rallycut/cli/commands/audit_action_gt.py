"""CLI: rallycut audit-action-gt [--video-id <id> | --all]

Report per-source counts for rally_action_ground_truth rows. Useful for
post-retrack health checks: a video whose UNRESOLVED count is climbing
indicates the resolver isn't finding the labeled players in the new
tracking.
"""
from __future__ import annotations

from collections import Counter
from typing import cast

import psycopg
import typer
from rich.console import Console
from rich.table import Table

from rallycut.evaluation.db import get_connection

console = Console()


def audit_video(conn: psycopg.Connection, video_id: str) -> dict[str, object]:
    """Return {video_id, counts, unresolved_rallies} for one video."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT gt.rally_id, gt.resolved_source
              FROM rally_action_ground_truth gt
              JOIN rallies r ON r.id = gt.rally_id
             WHERE r.video_id = %s
            """,
            (video_id,),
        )
        rows = cur.fetchall()

    counts: Counter[str] = Counter()
    per_rally: Counter[str] = Counter()
    for rally_id, source in rows:
        source_str = str(source)
        counts[source_str] += 1
        if source_str == "UNRESOLVED":
            per_rally[str(rally_id)] += 1

    # Initialize all enum values so output is stable even when some are zero.
    for k in ("SNAPSHOT_EXACT", "IOU_MATCH", "REID_MATCH", "NEAREST_CENTER", "MANUAL", "UNRESOLVED"):
        counts.setdefault(k, 0)

    return {
        "video_id": video_id,
        "counts": dict(counts),
        "unresolved_rallies": [
            {"rally_id": rid, "count": n} for rid, n in per_rally.most_common()
        ],
    }


def audit_action_gt_cmd(
    video_id: str | None = typer.Option(None, "--video-id", help="Single video UUID"),
    all_videos: bool = typer.Option(False, "--all", help="Audit every video with action GT"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Per-source counts for rally_action_ground_truth rows."""
    if not video_id and not all_videos:
        raise typer.BadParameter("Specify --video-id or --all")

    with get_connection() as conn:
        if all_videos:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT r.video_id
                      FROM rally_action_ground_truth gt
                      JOIN rallies r ON r.id = gt.rally_id
                     ORDER BY r.video_id
                    """,
                )
                video_ids = [str(row[0]) for row in cur.fetchall()]
        else:
            video_ids = [cast(str, video_id)]

        totals: Counter[str] = Counter()

        if not quiet:
            table = Table(title=f"action-GT audit ({len(video_ids)} video{'s' if len(video_ids) != 1 else ''})")
            table.add_column("video_id", style="dim")
            table.add_column("SNAPSHOT_EXACT", justify="right")
            table.add_column("IOU_MATCH", justify="right")
            table.add_column("REID_MATCH", justify="right")
            table.add_column("NEAREST_CENTER", justify="right")
            table.add_column("MANUAL", justify="right")
            table.add_column("UNRESOLVED", justify="right", style="yellow")
            table.add_column("rallies_w_unresolved", justify="right")

        for vid in video_ids:
            r = audit_video(conn, vid)
            counts = cast(dict[str, int], r["counts"])
            unresolved_rallies = cast(list[dict[str, object]], r["unresolved_rallies"])
            for k, v in counts.items():
                totals[k] += v
            if not quiet:
                table.add_row(
                    vid[:8] + "…",
                    str(counts.get("SNAPSHOT_EXACT", 0)),
                    str(counts.get("IOU_MATCH", 0)),
                    str(counts.get("REID_MATCH", 0)),
                    str(counts.get("NEAREST_CENTER", 0)),
                    str(counts.get("MANUAL", 0)),
                    str(counts.get("UNRESOLVED", 0)),
                    str(len(unresolved_rallies)),
                )

        if not quiet:
            console.print(table)

        total_str = ", ".join(f"{k}={v}" for k, v in totals.most_common())
        console.print(f"[bold]totals:[/bold] {total_str}")

        # Exit non-zero if any UNRESOLVED found AND --quiet is set — this lets a
        # CI gate use the CLI to fail builds on regression. Without --quiet,
        # always exit 0 so interactive runs don't surprise the user.
        if quiet and totals["UNRESOLVED"] > 0:
            raise typer.Exit(code=1)
