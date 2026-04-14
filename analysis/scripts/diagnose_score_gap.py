"""Diagnose the gap between formation_side accuracy and score_accuracy.

Compares per-video:
  1. Formation physical side accuracy (near/far vs GT)
  2. Final serving_team accuracy (A/B vs GT) after convention mapping + Viterbi

Shows where the convention/Viterbi pipeline loses accuracy.

Usage:
    cd analysis
    uv run python scripts/diagnose_score_gap.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rich.console import Console
from rich.table import Table

from eval_score_tracking import load_score_gt  # noqa: E402
from rallycut.tracking.action_classifier import (  # noqa: E402
    _find_serving_side_by_formation,
)
from rallycut.tracking.player_tracker import PlayerPosition  # noqa: E402

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


def main() -> int:
    console.print("[bold]Loading score GT...[/bold]")
    video_rallies = load_score_gt()
    console.print(f"Loaded {sum(len(r) for r in video_rallies.values())} rallies "
                  f"across {len(video_rallies)} videos\n")

    # Per-video stats
    video_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "formation_correct": 0, "formation_wrong": 0,
                 "formation_abstain": 0, "team_correct": 0, "team_wrong": 0}
    )

    total_formation_correct = 0
    total_formation_wrong = 0
    total_team_correct = 0
    total_team_wrong = 0
    total_scored = 0

    error_rallies: list[dict] = []

    for vid, rallies in sorted(video_rallies.items()):
        # Determine per-video convention: does near=A at start?
        # Use GT majority vote (same as eval)
        near_a_votes = 0
        near_b_votes = 0

        rally_results: list[dict] = []

        for rally in rallies:
            positions = _parse_positions(rally.positions)
            net_y = rally.court_split_y if rally.court_split_y else 0.5

            formation_side, formation_conf = _find_serving_side_by_formation(
                positions, net_y=net_y, start_frame=0,
            )

            gt_team = rally.gt_serving_team
            if gt_team is None:
                continue

            # Determine GT physical side from team label + convention
            # We'll compute this after convention is determined
            rally_results.append({
                "rally_id": rally.rally_id,
                "formation_side": formation_side,
                "formation_conf": formation_conf,
                "gt_team": gt_team,
                "side_flipped": rally.side_flipped,
            })

            # Vote for convention
            if formation_side is not None:
                if rally.side_flipped:
                    # After a side switch, near/far mapping is inverted
                    if formation_side == "near" and gt_team == "B":
                        near_a_votes += 1
                    elif formation_side == "near" and gt_team == "A":
                        near_b_votes += 1
                    elif formation_side == "far" and gt_team == "A":
                        near_a_votes += 1
                    elif formation_side == "far" and gt_team == "B":
                        near_b_votes += 1
                else:
                    if formation_side == "near" and gt_team == "A":
                        near_a_votes += 1
                    elif formation_side == "near" and gt_team == "B":
                        near_b_votes += 1
                    elif formation_side == "far" and gt_team == "B":
                        near_a_votes += 1
                    elif formation_side == "far" and gt_team == "A":
                        near_b_votes += 1

        near_is_a = near_a_votes >= near_b_votes
        stats = video_stats[vid[:10]]

        for r in rally_results:
            stats["total"] += 1
            total_scored += 1

            # GT physical side
            flipped = r["side_flipped"]
            gt_team = r["gt_team"]
            if near_is_a:
                gt_side = "near" if gt_team == "A" else "far"
            else:
                gt_side = "near" if gt_team == "B" else "far"
            if flipped:
                gt_side = "far" if gt_side == "near" else "near"

            # Formation accuracy
            if r["formation_side"] is None:
                stats["formation_abstain"] += 1
            elif r["formation_side"] == gt_side:
                stats["formation_correct"] += 1
                total_formation_correct += 1
            else:
                stats["formation_wrong"] += 1
                total_formation_wrong += 1

            # Team prediction (simple: formation_side + convention)
            if r["formation_side"] is not None:
                if flipped:
                    if near_is_a:
                        pred_team = "B" if r["formation_side"] == "near" else "A"
                    else:
                        pred_team = "A" if r["formation_side"] == "near" else "B"
                else:
                    if near_is_a:
                        pred_team = "A" if r["formation_side"] == "near" else "B"
                    else:
                        pred_team = "B" if r["formation_side"] == "near" else "A"

                if pred_team == gt_team:
                    stats["team_correct"] += 1
                    total_team_correct += 1
                else:
                    stats["team_wrong"] += 1
                    total_team_wrong += 1
                    error_rallies.append({
                        "video": vid[:10],
                        "rally": r["rally_id"][:8],
                        "formation": r["formation_side"],
                        "gt_side": gt_side,
                        "pred_team": pred_team,
                        "gt_team": gt_team,
                        "flipped": flipped,
                        "near_is_a": near_is_a,
                        "formation_correct": r["formation_side"] == gt_side,
                    })

    # Print table
    table = Table(title="Score Gap: Formation vs Team Accuracy")
    table.add_column("Video", style="cyan")
    table.add_column("Rallies", justify="right")
    table.add_column("Formation", justify="right")
    table.add_column("Team", justify="right")
    table.add_column("Gap", justify="right")
    table.add_column("Near=A")

    for vid in sorted(video_stats.keys()):
        s = video_stats[vid]
        if s["total"] == 0:
            continue
        f_acc = s["formation_correct"] / max(1, s["formation_correct"] + s["formation_wrong"])
        t_acc = s["team_correct"] / max(1, s["team_correct"] + s["team_wrong"])
        gap = t_acc - f_acc
        gap_str = f"{gap:+.0%}" if abs(gap) > 0.001 else "0%"
        style = "red" if gap < -0.05 else ""
        table.add_row(
            vid, str(s["total"]),
            f"{f_acc:.0%}", f"{t_acc:.0%}",
            gap_str, "",
            style=style,
        )

    f_total = total_formation_correct / max(1, total_formation_correct + total_formation_wrong)
    t_total = total_team_correct / max(1, total_team_correct + total_team_wrong)
    table.add_row(
        "TOTAL", str(total_scored),
        f"{f_total:.1%}", f"{t_total:.1%}",
        f"{t_total - f_total:+.1%}", "",
        style="bold",
    )

    console.print(table)

    # Error analysis
    console.print(f"\n[bold]Team errors where formation was CORRECT:[/bold]")
    convention_errors = [e for e in error_rallies if e["formation_correct"]]
    console.print(f"  {len(convention_errors)} / {len(error_rallies)} team errors "
                  f"are pure convention/mapping errors")

    if convention_errors:
        console.print("\n  Convention errors by video:")
        by_vid: dict[str, int] = defaultdict(int)
        for e in convention_errors:
            by_vid[e["video"]] += 1
        for vid, cnt in sorted(by_vid.items(), key=lambda x: -x[1]):
            console.print(f"    {vid}: {cnt} errors")

    formation_errors = [e for e in error_rallies if not e["formation_correct"]]
    console.print(f"\n[bold]Team errors where formation was WRONG:[/bold]")
    console.print(f"  {len(formation_errors)} / {len(error_rallies)} team errors "
                  f"are formation prediction errors")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
