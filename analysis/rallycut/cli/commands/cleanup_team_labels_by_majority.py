"""CLI: rallycut cleanup-team-labels-by-majority <video-id> [--dry-run]

One-shot cleanup that rewrites scrambled team partitions per rally using
cross-rally per-PID majority. Closes I-8 violations on legacy data without
re-tracking.

Conservative by design:
  - Only commits a fix when the candidate is exactly 2A+2B.
  - Only commits a fix when the candidate matches a partition that already
    exists as a valid 2v2 in another rally of the same video.
  - Skips PIDs whose A/B count is tied (no decisive majority) — typical for
    side-switched videos.

These gates make silent corruption impossible. Worst case: leaves a rally
unchanged.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, cast

import typer
from rich.console import Console

from rallycut.evaluation.tracking.db import get_connection

console = Console()


def _partition_for_primary(
    team_assignments: dict[str, str], primary: list[int]
) -> dict[int, str] | None:
    """Project teamAssignments onto the primary set; None if any label invalid."""
    out: dict[int, str] = {}
    for tid in primary:
        label = team_assignments.get(str(tid))
        if label not in ("A", "B"):
            return None
        out[tid] = label
    return out


def _is_2v2(partition: dict[int, str]) -> bool:
    a = sum(1 for v in partition.values() if v == "A")
    b = sum(1 for v in partition.values() if v == "B")
    return a == 2 and b == 2


def _per_pid_majority(
    rallies: list[tuple[list[int], dict[str, str]]],
) -> dict[int, str | None]:
    """Per-PID majority across all rallies. None for ties or no observations."""
    pid_counts: dict[int, Counter[str]] = {}
    for primary, ta in rallies:
        for tid in primary:
            label = ta.get(str(tid))
            if label not in ("A", "B"):
                continue
            pid_counts.setdefault(tid, Counter())[label] += 1
    result: dict[int, str | None] = {}
    for tid, counts in pid_counts.items():
        a, b = counts.get("A", 0), counts.get("B", 0)
        if a > b:
            result[tid] = "A"
        elif b > a:
            result[tid] = "B"
        else:
            result[tid] = None  # tie
    return result


def _candidate_from_majority(
    primary: list[int], majority: dict[int, str | None]
) -> dict[int, str] | None:
    """Build candidate by replacing each primary's label with its majority.
    Returns None if any primary's majority is a tie (None) or absent.
    """
    out: dict[int, str] = {}
    for tid in primary:
        m = majority.get(tid)
        if m not in ("A", "B"):
            return None
        out[tid] = m
    return out


def cleanup_team_labels_by_majority_cmd(
    video_id: str = typer.Argument(..., help="Video UUID"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview changes without writing to DB"
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Suppress per-rally info"
    ),
) -> None:
    """Rewrite scrambled team partitions to per-PID majority (closes I-8 on legacy data)."""
    prefix = "[DRY RUN] " if dry_run else ""
    if not quiet:
        console.print(
            f"[dim]{prefix}Cleaning up team labels for video {video_id}…[/dim]"
        )

    rally_query = """
        SELECT
            r.id AS rally_id,
            pt.primary_track_ids,
            pt.actions_json
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE r.video_id = %s
        ORDER BY r.start_ms
    """

    n_fixed = 0
    n_noop = 0
    n_skipped = 0
    n_ambiguous = 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(rally_query, [video_id])
            rows = cur.fetchall()

        # First pass: collect rally-level data + valid 2v2 partitions seen.
        per_rally: list[
            tuple[str, list[int], dict[str, Any], dict[str, str]]
        ] = []
        valid_partitions: set[frozenset[tuple[int, str]]] = set()
        rallies_for_majority: list[tuple[list[int], dict[str, str]]] = []
        for row in rows:
            rally_id = cast(str, row[0])
            primary_raw = row[1]
            actions_json = row[2]
            if (
                not isinstance(primary_raw, list)
                or len(primary_raw) != 4
                or not isinstance(actions_json, dict)
            ):
                continue
            ta = actions_json.get("teamAssignments")
            if not isinstance(ta, dict):
                continue
            primary = [int(t) for t in primary_raw]
            partition = _partition_for_primary(ta, primary)
            if partition is None:
                continue  # I-6 territory
            per_rally.append((rally_id, primary, actions_json, ta))
            rallies_for_majority.append((primary, ta))
            if _is_2v2(partition):
                valid_partitions.add(frozenset(partition.items()))

        majority = _per_pid_majority(rallies_for_majority)

        # Second pass: per-rally fix attempt.
        for rally_id, primary, actions_json, ta in per_rally:
            partition = _partition_for_primary(ta, primary)
            if partition is None or _is_2v2(partition):
                n_noop += 1
                if not quiet:
                    console.print(
                        f"  [dim][noop][/dim] rally {rally_id}: already 2v2 or unscored"
                    )
                continue

            candidate = _candidate_from_majority(primary, majority)
            if candidate is None:
                n_ambiguous += 1
                if not quiet:
                    console.print(
                        f"  [yellow][skip-tie][/yellow] rally {rally_id}: "
                        f"some primary PID has tied majority (likely side-switched)"
                    )
                continue
            if not _is_2v2(candidate):
                n_ambiguous += 1
                if not quiet:
                    console.print(
                        f"  [yellow][ambiguous][/yellow] rally {rally_id}: "
                        f"per-PID majority does not produce 2v2"
                    )
                continue
            if frozenset(candidate.items()) not in valid_partitions:
                n_ambiguous += 1
                if not quiet:
                    console.print(
                        f"  [yellow][ambiguous][/yellow] rally {rally_id}: "
                        f"candidate {dict(sorted(candidate.items()))} "
                        f"matches no existing valid partition in this video"
                    )
                continue

            # Build new actions_json with the corrected teamAssignments.
            new_ta: dict[str, str] = dict(ta)
            for tid, label in candidate.items():
                new_ta[str(tid)] = label
            new_actions_json = dict(actions_json)
            new_actions_json["teamAssignments"] = new_ta

            old_view = {str(t): ta.get(str(t)) for t in sorted(primary)}
            new_view = {str(t): new_ta[str(t)] for t in sorted(primary)}

            if dry_run:
                if not quiet:
                    console.print(
                        f"  [cyan][DRY][/cyan]   rally {rally_id}: "
                        f"would fix {old_view} → {new_view}"
                    )
                n_fixed += 1
                continue

            try:
                with conn.cursor() as wcur:
                    wcur.execute(
                        "UPDATE player_tracks SET actions_json = %s "
                        "WHERE rally_id = %s",
                        [json.dumps(new_actions_json), rally_id],
                    )
                conn.commit()
                n_fixed += 1
                if not quiet:
                    console.print(
                        f"  [green][fix][/green]  rally {rally_id}: "
                        f"{old_view} → {new_view}"
                    )
            except Exception as exc:
                conn.rollback()
                if not quiet:
                    console.print(
                        f"  [red][err][/red]  rally {rally_id}: "
                        f"write failed: {exc}"
                    )

        n_skipped = len(rows) - len(per_rally)

    summary_label = "Dry-run" if dry_run else "Summary"
    console.print(
        f"\n[bold]{summary_label}:[/bold] "
        f"{n_fixed} fixed · {n_noop} no-op · "
        f"{n_ambiguous} ambiguous · {n_skipped} skipped"
    )
