"""Measure how many rallies have team labels flipped by verify_team_assignments.

The production code path (track_player.py:1016, analyze.py:113) always runs
`verify_team_assignments` on loaded team labels, but several eval scripts
(diagnose_serve_side via eval_sequence_enriched, etc.) skip it. This script
quantifies the discrepancy: for each GT rally, build team assignments with
rally_positions=None (unverified) and with rally_positions=real (verified,
matching production), then count how many rally dicts actually changed.

If the number is near zero, the eval and production code paths agree and the
78% serve-side baseline is already honest — pivot. If it's material, the
eval-script fix is worth landing.

Run with no arguments.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_action_detection import load_rallies_with_action_gt  # noqa: E402
from rallycut.evaluation.db import get_connection  # noqa: E402
from rallycut.tracking.match_tracker import (  # noqa: E402
    build_match_team_assignments,
)
from rallycut.tracking.player_tracker import PlayerPosition  # noqa: E402

console = Console()


def _load_match_analyses(video_ids: set[str]) -> dict[str, dict]:
    if not video_ids:
        return {}
    placeholders = ", ".join(["%s"] * len(video_ids))
    query = f"""
        SELECT id, match_analysis_json
        FROM videos
        WHERE id IN ({placeholders})
          AND match_analysis_json IS NOT NULL
    """
    out: dict[str, dict] = {}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, list(video_ids))
            for vid, ma_json in cur.fetchall():
                if isinstance(ma_json, dict):
                    out[vid] = ma_json
    return out


def _parse_positions(positions_json: list[dict]) -> list[PlayerPosition]:
    return [
        PlayerPosition(
            frame_number=pp["frameNumber"], track_id=pp["trackId"],
            x=pp["x"], y=pp["y"],
            width=pp.get("width", 0.05), height=pp.get("height", 0.10),
            confidence=pp.get("confidence", 1.0),
        )
        for pp in positions_json
    ]


def main() -> None:
    rallies = load_rallies_with_action_gt()
    console.print(f"[bold]Loaded {len(rallies)} GT rallies[/bold]")

    video_ids = {r.video_id for r in rallies}
    match_analyses = _load_match_analyses(video_ids)

    # Build rally_positions lookup.
    rally_pos_lookup: dict[str, list[PlayerPosition]] = {}
    for r in rallies:
        if r.positions_json:
            rally_pos_lookup[r.rally_id] = _parse_positions(r.positions_json)

    unverified: dict[str, dict[int, int]] = {}
    verified: dict[str, dict[int, int]] = {}
    for vid, ma in match_analyses.items():
        unverified.update(
            build_match_team_assignments(ma, min_confidence=0.70)
        )
        verified.update(
            build_match_team_assignments(
                ma, min_confidence=0.70, rally_positions=rally_pos_lookup,
            )
        )

    # Restrict to GT rallies
    gt_ids = {r.rally_id for r in rallies}
    unverified = {rid: t for rid, t in unverified.items() if rid in gt_ids}
    verified = {rid: t for rid, t in verified.items() if rid in gt_ids}

    total = len(unverified)
    flipped_rallies = 0
    tracks_changed_per_rally: Counter[int] = Counter()
    for rid, u_teams in unverified.items():
        v_teams = verified.get(rid)
        if v_teams is None:
            continue
        n_diff = sum(
            1 for tid, team in u_teams.items()
            if v_teams.get(tid) != team
        )
        tracks_changed_per_rally[n_diff] += 1
        if n_diff > 0:
            flipped_rallies += 1

    table = Table(title="verify_team_assignments effect on GT rallies")
    table.add_column("Tracks changed per rally", justify="right")
    table.add_column("Rally count", justify="right")
    table.add_column("% of rallies with match_teams", justify="right")

    for n in sorted(tracks_changed_per_rally):
        c = tracks_changed_per_rally[n]
        pct = (c / total * 100) if total else 0.0
        row_style = "bold red" if n > 0 else ""
        table.add_row(str(n), str(c), f"{pct:.1f}%", style=row_style)

    console.print(table)

    pct_flipped = (flipped_rallies / total * 100) if total else 0.0
    console.print(
        f"\n[bold]{flipped_rallies}/{total} rallies "
        f"({pct_flipped:.1f}%) changed by verify_team_assignments[/bold]"
    )
    console.print(
        f"  GT rallies total: {len(rallies)}; "
        f"rallies with match_teams (conf>=0.70): {total}"
    )

    console.print("\n[bold]Interpretation:[/bold]")
    if pct_flipped < 2:
        console.print(
            "  [green]<2% — verifier is a no-op at this sample size. "
            "Eval/production already agree; unverified-eval numbers are honest. "
            "Pivot — the bottleneck is elsewhere.[/green]"
        )
    elif pct_flipped < 10:
        console.print(
            "  [yellow]2-10% — meaningful but moderate. Wiring eval through "
            "`rally_positions` will shift numbers but not transformatively.[/yellow]"
        )
    else:
        console.print(
            "  [red]>=10% — verifier materially moves labels. Eval scripts "
            "that skip it are reporting artifacts. Fix the eval wiring.[/red]"
        )


if __name__ == "__main__":
    main()
