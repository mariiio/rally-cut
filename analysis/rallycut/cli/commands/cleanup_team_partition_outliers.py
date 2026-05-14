"""CLI: rallycut cleanup-team-partition-outliers <video-id> [--dry-run]

Cross-rally team partition consensus repair (Stage 1 of upstream team-fix).

The per-rally `classify_teams` output is sometimes corrupted by a within-rally
ID switch — producing a valid 2v2 partition that *disagrees* with the rest of
the same video's rallies. The existing `cleanup-team-labels-by-majority` does
NOT catch this case because its trigger gate skips rallies that are already
2v2 (the failure mode is partition shape, not label scrambling).

This command:
  1. Reads every rally's `teamAssignments` for the given video.
  2. Computes the canonical partition signature per rally (frozen-set of frozen
     sets of track IDs, ignoring A/B label permutation).
  3. Picks the *most common* partition as the consensus.
  4. For rallies whose partition disagrees with the consensus, rewrites the
     team labels to match a nearby consistent rally's labeling (preserving the
     A/B side convention of that era — handles side switches mid-video).

Conservative gates:
  - Only fires when at least 3 rallies agree on the consensus partition.
  - Only fires on rallies whose canonical partition is in `{2v2 candidates}`
    AND differs from consensus.
  - The replacement label map is taken from the closest-in-order non-outlier
    rally so side-switch boundaries are handled.

After applying, re-run:
  - rallycut reattribute-actions <video-id>
to refresh action attributions using the corrected team_assignments.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, cast

import typer
from rich.console import Console

from rallycut.evaluation.tracking.db import get_connection

console = Console()


def _canonical_partition(team_assignments: dict[str, Any]) -> frozenset[frozenset[int]]:
    """Group track IDs by A/B label and return a label-agnostic frozen set."""
    groups: dict[str, list[int]] = {}
    for tid_str, letter in team_assignments.items():
        if letter not in ("A", "B"):
            continue
        try:
            tid = int(tid_str)
        except (ValueError, TypeError):
            continue
        groups.setdefault(letter, []).append(tid)
    return frozenset(frozenset(g) for g in groups.values() if g)


def _format_partition(partition: frozenset[frozenset[int]]) -> str:
    return "{" + "} {".join(
        "".join(str(p) for p in sorted(g)) for g in sorted(partition, key=lambda g: min(g))
    ) + "}"


def _candidate_labels_from_template(
    outlier_ta: dict[str, str],
    template_ta: dict[str, str],
    primary: list[int],
) -> dict[str, str] | None:
    """Build a new team_assignments for the outlier rally by copying labels
    from a template rally (which has the consensus partition). Returns None
    if any primary track has no label in the template."""
    new_ta: dict[str, str] = dict(outlier_ta)
    for tid in primary:
        key = str(tid)
        if key not in template_ta:
            return None
        new_ta[key] = template_ta[key]
    return new_ta


def cleanup_team_partition_outliers_cmd(
    video_id: str = typer.Argument(..., help="Video UUID"),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Preview changes without writing to DB"
    ),
    min_consensus_rallies: int = typer.Option(
        3, help="Minimum number of rallies agreeing on the consensus partition"
    ),
    quiet: bool = typer.Option(
        False, "--quiet", "-q", help="Suppress per-rally info"
    ),
) -> None:
    """Rewrite outlier-rally team partitions to match cross-rally consensus."""
    prefix = "[DRY RUN] " if dry_run else ""
    if not quiet:
        console.print(
            f"[dim]{prefix}Detecting outlier-partition rallies for video {video_id}…[/dim]"
        )

    rally_query = """
        SELECT
            r.id AS rally_id,
            r."order" AS rally_order,
            pt.primary_track_ids,
            pt.actions_json
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE r.video_id = %s AND r.status = 'CONFIRMED'
        ORDER BY r."order"
    """

    n_fixed = 0
    n_noop = 0
    n_unfixable = 0

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(rally_query, [video_id])
            rows = cur.fetchall()

        # Collect rally-level data.
        per_rally: list[
            tuple[str, int, list[int], dict[str, Any], dict[str, str], frozenset[frozenset[int]]]
        ] = []
        for row in rows:
            rally_id = cast(str, row[0])
            rally_order = cast(int, row[1])
            primary_raw = row[2]
            actions_json = row[3]
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
            partition = _canonical_partition(ta)
            if not partition or any(len(g) == 0 for g in partition):
                continue
            per_rally.append(
                (rally_id, rally_order, primary, actions_json, ta, partition)
            )

        if not per_rally:
            console.print("[yellow]No usable rallies for this video.[/yellow]")
            return

        # Identify consensus partition.
        partition_counts = Counter(p[5] for p in per_rally)
        consensus_partition, consensus_count = partition_counts.most_common(1)[0]
        if consensus_count < min_consensus_rallies:
            console.print(
                f"[yellow]No partition has >= {min_consensus_rallies} rallies agreeing "
                f"(top: {consensus_count}). Refusing to apply.[/yellow]"
            )
            return

        consensus_str = _format_partition(consensus_partition)
        console.print(
            f"  Consensus partition: [cyan]{consensus_str}[/cyan] "
            f"({consensus_count}/{len(per_rally)} rallies)"
        )

        outliers = [
            (rid, order, primary, aj, ta, p)
            for (rid, order, primary, aj, ta, p) in per_rally
            if p != consensus_partition
        ]
        if not outliers:
            console.print("[green]No outliers. Nothing to fix.[/green]")
            return

        # For each outlier, find the closest-in-order non-outlier rally and use
        # its label map as the template.
        non_outliers_by_order: dict[int, tuple[str, dict[str, str]]] = {
            order: (rid, ta)
            for (rid, order, _, _, ta, p) in per_rally
            if p == consensus_partition
        }
        sorted_non_outlier_orders = sorted(non_outliers_by_order.keys())

        def _closest_template(target_order: int) -> tuple[int, str, dict[str, str]] | None:
            if not sorted_non_outlier_orders:
                return None
            # Pick the order with minimal absolute distance.
            best = min(
                sorted_non_outlier_orders, key=lambda o: abs(o - target_order)
            )
            rid, ta = non_outliers_by_order[best]
            return best, rid, ta

        for rid, order, primary, aj, ta, p in outliers:
            outlier_str = _format_partition(p)
            template = _closest_template(order)
            if template is None:
                n_unfixable += 1
                console.print(
                    f"  [red][unfixable][/red] rally r{order+1} (id={rid[:8]}): "
                    f"no template rally available"
                )
                continue
            t_order, t_rid, t_ta = template

            new_ta = _candidate_labels_from_template(ta, t_ta, primary)
            if new_ta is None:
                n_unfixable += 1
                console.print(
                    f"  [red][unfixable][/red] rally r{order+1}: template r{t_order+1} "
                    f"has missing labels for some primary tracks"
                )
                continue
            new_partition = _canonical_partition(new_ta)
            if new_partition != consensus_partition:
                n_unfixable += 1
                console.print(
                    f"  [yellow][skip][/yellow] rally r{order+1}: applying template "
                    f"would produce {_format_partition(new_partition)} != consensus"
                )
                continue
            if new_ta == ta:
                n_noop += 1
                continue

            old_view = {str(t): ta.get(str(t)) for t in sorted(primary)}
            new_view = {str(t): new_ta[str(t)] for t in sorted(primary)}

            if dry_run:
                console.print(
                    f"  [cyan][DRY][/cyan]    rally r{order+1} (id={rid[:8]}): "
                    f"partition {outlier_str} → {consensus_str}"
                )
                console.print(
                    f"           labels {old_view} → {new_view} "
                    f"(template from r{t_order+1})"
                )
                n_fixed += 1
                continue

            # Apply the fix.
            new_actions_json = dict(aj)
            new_actions_json["teamAssignments"] = new_ta
            # Also overwrite each action's "team" field to reflect the new
            # team_assignments. The team field is derived from team_assignments
            # at write time but the JSON stores a snapshot.
            new_actions = []
            for a in new_actions_json.get("actions", []):
                if not isinstance(a, dict):
                    new_actions.append(a)
                    continue
                tid = a.get("playerTrackId")
                new_a = dict(a)
                if tid is not None:
                    label = new_ta.get(str(tid))
                    if label in ("A", "B"):
                        new_a["team"] = label
                new_actions.append(new_a)
            new_actions_json["actions"] = new_actions

            try:
                with conn.cursor() as wcur:
                    wcur.execute(
                        "UPDATE player_tracks SET actions_json = %s "
                        "WHERE rally_id = %s",
                        [json.dumps(new_actions_json), rid],
                    )
                conn.commit()
                n_fixed += 1
                console.print(
                    f"  [green][fix][/green]    rally r{order+1} (id={rid[:8]}): "
                    f"partition {outlier_str} → {consensus_str}"
                )
                console.print(
                    f"           labels {old_view} → {new_view} "
                    f"(template from r{t_order+1})"
                )
            except Exception as exc:
                conn.rollback()
                console.print(
                    f"  [red][err][/red] rally r{order+1}: write failed: {exc}"
                )

    summary_label = "Dry-run" if dry_run else "Summary"
    console.print(
        f"\n[bold]{summary_label}:[/bold] "
        f"{n_fixed} fixed · {n_noop} no-op · {n_unfixable} unfixable"
    )
    if not dry_run and n_fixed > 0:
        console.print(
            f"\n[bold yellow]Next step:[/bold yellow] re-run\n"
            f"  rallycut reattribute-actions {video_id}\n"
            f"to refresh action attributions using the corrected team_assignments."
        )
