"""Diagnose localize_team_near failure modes.

Measures:
  1. Return None rate (narrow-angle cameras)
  2. When non-None, accuracy vs GT-derived ground truth team_near
  3. Whether restricting to serve window (frames 0-120) improves accuracy

The key question: is team_near errors the bottleneck preventing production
score_accuracy from reaching the formation accuracy ceiling?

Usage:
    cd analysis
    uv run python scripts/diagnose_team_near.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
from rich.console import Console
from rich.table import Table

from eval_action_detection import _load_track_to_player_maps  # noqa: E402
from eval_score_tracking import load_score_gt  # noqa: E402
from production_eval import _load_team_templates_by_video  # noqa: E402
from rallycut.tracking.action_classifier import (  # noqa: E402
    _find_serving_side_by_formation,
)
from rallycut.tracking.player_tracker import PlayerPosition  # noqa: E402
from rallycut.tracking.team_identity import (  # noqa: E402
    TeamTemplate,
    localize_team_near,
)

console = Console()


def _parse_positions(pos_json: list[dict]) -> list[PlayerPosition]:
    return [
        PlayerPosition(
            track_id=p.get("trackId", p.get("track_id", -1)),
            frame_number=p.get("frameNumber", p.get("frame_number", 0)),
            x=p["x"], y=p["y"],
            width=p["width"], height=p["height"],
            confidence=p.get("confidence", 1.0),
        )
        for p in pos_json
    ]


def _load_team_templates(
    video_ids: set[str],
) -> dict[str, tuple[TeamTemplate, TeamTemplate]]:
    """Load per-video team templates from match_analysis_json or player_matching_gt_json."""
    templates_by_vid: dict[str, tuple[TeamTemplate, TeamTemplate]] = {}

    if not video_ids:
        return templates_by_vid

    placeholders = ", ".join(["%s"] * len(video_ids))
    query = f"""
        SELECT id, player_matching_gt_json, match_analysis_json
        FROM videos
        WHERE id IN ({placeholders})
    """

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(query, list(video_ids))
        for vid, pm_json, ma_json in cur.fetchall():
            team_data = None
            if pm_json and isinstance(pm_json, dict):
                team_data = pm_json.get("teams")
            if team_data is None and ma_json and isinstance(ma_json, dict):
                team_data = ma_json.get("teams")
            if team_data is None or len(team_data) < 2:
                continue
            try:
                t0 = TeamTemplate(
                    team_label=str(team_data[0].get("label", "0")),
                    player_ids=tuple(team_data[0].get("playerIds", [])),
                )
                t1 = TeamTemplate(
                    team_label=str(team_data[1].get("label", "1")),
                    player_ids=tuple(team_data[1].get("playerIds", [])),
                )
                templates_by_vid[vid] = (t0, t1)
            except Exception:
                pass

    return templates_by_vid


def main() -> int:
    console.print("[bold]Loading score GT...[/bold]")
    video_rallies = load_score_gt()
    video_ids = set(video_rallies.keys())
    templates_by_vid = _load_team_templates_by_video(video_ids)
    t2p_by_rally = _load_track_to_player_maps(video_ids)
    console.print(f"  {sum(len(r) for r in video_rallies.values())} rallies, "
                  f"{len(templates_by_vid)} videos with templates, "
                  f"{len(t2p_by_rally)} rallies with track_to_player\n")

    # Per-video stats
    vid_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {
            "total": 0, "has_template": 0, "has_t2p": 0,
            "full_returns": 0, "full_none": 0, "full_correct": 0, "full_wrong": 0,
            "early_returns": 0, "early_none": 0, "early_correct": 0, "early_wrong": 0,
        }
    )

    # For each GT-labeled rally, compute:
    # 1. Full-window team_near (default)
    # 2. Early-window team_near (frames 0-120 only)
    # 3. GT team_near: the team label whose gt_serving_team = formation_side-matches

    total_rallies_with_gt = 0
    skipped_no_template = 0
    skipped_no_t2p = 0

    for vid, rallies in sorted(video_rallies.items()):
        stats = vid_stats[vid[:10]]
        templates = templates_by_vid.get(vid)
        if templates is None:
            skipped_no_template += len(rallies)
            continue
        stats["has_template"] = len(rallies)

        for rally in rallies:
            if rally.gt_serving_team is None:
                continue
            total_rallies_with_gt += 1
            stats["total"] += 1

            positions = _parse_positions(rally.positions)
            t2p = t2p_by_rally.get(rally.rally_id)
            if not t2p:
                skipped_no_t2p += 1
                continue
            stats["has_t2p"] += 1

            # Full-rally team_near
            full_tn = localize_team_near(positions, t2p, templates)

            # Early-window team_near (frames 0-120 = serve formation)
            early_positions = [p for p in positions if p.frame_number < 120]
            early_tn = localize_team_near(early_positions, t2p, templates)

            # Ground truth for team_near: derive from gt_serving_team and formation_side
            # If formation says "near" served and gt says team X served, then team X is near.
            # We use formation as the gating: only evaluate team_near when formation is correct.
            net_y = rally.court_split_y if rally.court_split_y else 0.5
            formation_side, _ = _find_serving_side_by_formation(
                positions, net_y=net_y, start_frame=0,
            )
            if formation_side is None:
                continue

            # GT-derived physical side (from team + flips)
            # Compute per-rally near_is_a
            # Use side_flipped directly: if flipped, gt_team maps to opposite side
            # We need a reference: assume initial_near_is_a=True, then invert per side_flipped
            # Simpler: use majority vote over all GT rallies in this video
            # But we don't have that here — use side_flipped flag from rally
            # Actually, what we want is: GT says team X served. If formation says "near",
            # then team X is the near team (correct team_near prediction = team X's label).

            t0, t1 = templates
            # Map gt_serving_team ("A"/"B") to template label
            # We need another source for this — use per-video majority-vote calibration
            # For simplicity, assume team_label "0" = A and "1" = B, adjusting if wrong
            # Actually, we don't need to know that — we just need to know which template
            # label matches the expected near team given formation_side and gt_team.

            # Skip: we need per-video label_a to do this properly.
            # Alternative: measure consistency — does full agree with early?
            if full_tn is not None:
                stats["full_returns"] += 1
            else:
                stats["full_none"] += 1

            if early_tn is not None:
                stats["early_returns"] += 1
            else:
                stats["early_none"] += 1

    # Second pass: use per-video calibration to measure accuracy
    # label_a = which template label corresponds to GT "A"
    # Then for a GT-labeled rally where formation is correct,
    # expected team_near = label_a if (gt=A and formation="near") or (gt=B and formation="far")

    vid_label_a: dict[str, str] = {}
    for vid, rallies in sorted(video_rallies.items()):
        templates = templates_by_vid.get(vid)
        if templates is None:
            continue
        t0, t1 = templates
        votes: dict[str, int] = {t0.team_label: 0, t1.team_label: 0}

        for rally in rallies:
            if rally.gt_serving_team is None:
                continue
            positions = _parse_positions(rally.positions)
            t2p = t2p_by_rally.get(rally.rally_id)
            if not t2p:
                continue
            net_y = rally.court_split_y if rally.court_split_y else 0.5
            formation_side, _ = _find_serving_side_by_formation(
                positions, net_y=net_y, start_frame=0,
            )
            if formation_side is None:
                continue
            tn = localize_team_near(positions, t2p, templates)
            if tn is None:
                continue
            # If formation="near", serving team's label = tn; if "far", opposite
            if formation_side == "near":
                serving_label = tn
            else:
                serving_label = t0.team_label if tn == t1.team_label else t1.team_label
            # Vote for which label = A
            if rally.gt_serving_team == "A":
                votes[serving_label] = votes.get(serving_label, 0) + 1
            else:
                votes[serving_label] = votes.get(serving_label, 0) - 1

        if votes:
            vid_label_a[vid] = max(votes, key=lambda k: votes[k])

    # Third pass: measure team_near accuracy using label_a
    for vid, rallies in sorted(video_rallies.items()):
        templates = templates_by_vid.get(vid)
        if templates is None or vid not in vid_label_a:
            continue
        stats = vid_stats[vid[:10]]
        label_a = vid_label_a[vid]
        t0, t1 = templates
        label_b = t0.team_label if label_a == t1.team_label else t1.team_label

        for rally in rallies:
            if rally.gt_serving_team is None:
                continue
            positions = _parse_positions(rally.positions)
            t2p = t2p_by_rally.get(rally.rally_id)
            if not t2p:
                continue
            net_y = rally.court_split_y if rally.court_split_y else 0.5
            formation_side, _ = _find_serving_side_by_formation(
                positions, net_y=net_y, start_frame=0,
            )
            if formation_side is None:
                continue

            # Expected team_near: the team that's near based on formation + gt
            # If formation="near" and gt="A" → near team has label_a
            # If formation="near" and gt="B" → near team has label_b
            # If formation="far" and gt="A" → near team has label_b (A is far)
            # If formation="far" and gt="B" → near team has label_a (B is far)
            if formation_side == "near":
                expected_tn = label_a if rally.gt_serving_team == "A" else label_b
            else:
                expected_tn = label_b if rally.gt_serving_team == "A" else label_a

            full_tn = localize_team_near(positions, t2p, templates)
            early_positions = [p for p in positions if p.frame_number < 120]
            early_tn = localize_team_near(early_positions, t2p, templates)

            if full_tn is not None:
                if full_tn == expected_tn:
                    stats["full_correct"] += 1
                else:
                    stats["full_wrong"] += 1

            if early_tn is not None:
                if early_tn == expected_tn:
                    stats["early_correct"] += 1
                else:
                    stats["early_wrong"] += 1

    # Print summary
    console.print(f"Total GT rallies: {total_rallies_with_gt}")
    console.print(f"  Skipped (no template): {skipped_no_template}")
    console.print(f"  Skipped (no t2p): {skipped_no_t2p}")
    console.print()

    table = Table(title="localize_team_near: Full vs Early window (frames 0-120)")
    table.add_column("Video", style="cyan")
    table.add_column("N", justify="right")
    table.add_column("Full: ret%", justify="right")
    table.add_column("Full: acc", justify="right")
    table.add_column("Early: ret%", justify="right")
    table.add_column("Early: acc", justify="right")
    table.add_column("Δ acc", justify="right")

    total_full_correct = total_full_wrong = 0
    total_early_correct = total_early_wrong = 0
    total_full_ret = total_early_ret = 0
    total_n = 0

    for vid in sorted(vid_stats.keys()):
        s = vid_stats[vid]
        n = s["total"]
        if n == 0:
            continue
        full_n = s["full_correct"] + s["full_wrong"]
        early_n = s["early_correct"] + s["early_wrong"]
        full_ret = full_n / n if n else 0
        full_acc = s["full_correct"] / full_n if full_n else 0
        early_ret = early_n / n if n else 0
        early_acc = s["early_correct"] / early_n if early_n else 0
        delta = early_acc - full_acc
        delta_str = f"{delta:+.0%}" if abs(delta) > 0.005 else "0%"
        style = "red" if full_acc < 0.9 else ""
        table.add_row(
            vid, str(n),
            f"{full_ret:.0%}", f"{full_acc:.0%}",
            f"{early_ret:.0%}", f"{early_acc:.0%}",
            delta_str, style=style,
        )
        total_n += n
        total_full_correct += s["full_correct"]
        total_full_wrong += s["full_wrong"]
        total_early_correct += s["early_correct"]
        total_early_wrong += s["early_wrong"]
        total_full_ret += full_n
        total_early_ret += early_n

    full_acc_tot = total_full_correct / max(1, total_full_correct + total_full_wrong)
    early_acc_tot = total_early_correct / max(1, total_early_correct + total_early_wrong)
    table.add_row(
        "TOTAL", str(total_n),
        f"{total_full_ret/max(1,total_n):.1%}",
        f"{full_acc_tot:.1%}",
        f"{total_early_ret/max(1,total_n):.1%}",
        f"{early_acc_tot:.1%}",
        f"{early_acc_tot - full_acc_tot:+.1%}",
        style="bold",
    )
    console.print(table)

    console.print(f"\n[bold]Summary[/bold]")
    console.print(f"  Full-window accuracy:  {full_acc_tot:.1%} "
                  f"({total_full_correct}/{total_full_correct + total_full_wrong})")
    console.print(f"  Early-window accuracy: {early_acc_tot:.1%} "
                  f"({total_early_correct}/{total_early_correct + total_early_wrong})")
    console.print(f"  Delta: {(early_acc_tot - full_acc_tot):+.1%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
