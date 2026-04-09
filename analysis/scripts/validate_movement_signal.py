"""Test player movement-toward-ball as attribution signal.

The contacting player should be moving toward the ball in the frames
before contact. This is perspective-independent because it uses velocity
direction, not absolute position.

Signals tested:
1. Movement alignment: cos(player_velocity, player→ball vector)
2. Distance reduction: how much closer did the player get in last N frames?
3. Combined: distance × (1 - alignment_boost)

Usage:
    cd analysis
    uv run python scripts/validate_movement_signal.py
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict

from rich.console import Console
from rich.table import Table

from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.tracking.db import load_court_calibration
from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.contact_detector import (
    Contact,
    ContactDetectionConfig,
    detect_contacts,
)
from rallycut.tracking.player_filter import classify_teams
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos

from scripts.eval_action_detection import (
    load_rallies_with_action_gt,
    match_contacts,
)

console = Console()


def compute_movement_scores(
    contact_frame: int,
    ball_x: float,
    ball_y: float,
    player_positions: list[PlayerPos],
    lookback: int = 10,
    search_frames: int = 15,
) -> dict[int, dict[str, float]]:
    """Compute movement-toward-ball scores for each player.

    Returns: track_id -> {alignment, distance_reduction, current_dist, combined}
    """
    # Get player positions at contact frame and lookback frame
    pos_at_contact: dict[int, tuple[float, float]] = {}
    pos_at_prior: dict[int, tuple[float, float]] = {}

    for p in player_positions:
        if abs(p.frame_number - contact_frame) <= 2:
            px = p.x
            py = p.y - p.height * 0.25  # upper quarter
            if p.track_id not in pos_at_contact:
                pos_at_contact[p.track_id] = (px, py)
        elif abs(p.frame_number - (contact_frame - lookback)) <= 2:
            px = p.x
            py = p.y - p.height * 0.25
            if p.track_id not in pos_at_prior:
                pos_at_prior[p.track_id] = (px, py)

    results: dict[int, dict[str, float]] = {}

    for tid, (cx, cy) in pos_at_contact.items():
        current_dist = math.sqrt((ball_x - cx) ** 2 + (ball_y - cy) ** 2)

        if tid in pos_at_prior:
            px, py = pos_at_prior[tid]

            # Movement vector
            mv_x = cx - px
            mv_y = cy - py
            mv_mag = math.sqrt(mv_x ** 2 + mv_y ** 2)

            # Player→ball vector at prior position
            pb_x = ball_x - px
            pb_y = ball_y - py
            pb_mag = math.sqrt(pb_x ** 2 + pb_y ** 2)

            # Alignment (cosine similarity)
            if mv_mag > 0.001 and pb_mag > 0.001:
                alignment = (mv_x * pb_x + mv_y * pb_y) / (mv_mag * pb_mag)
            else:
                alignment = 0.0

            # Distance reduction
            prior_dist = math.sqrt((ball_x - px) ** 2 + (ball_y - py) ** 2)
            dist_reduction = prior_dist - current_dist

            results[tid] = {
                "alignment": alignment,
                "distance_reduction": dist_reduction,
                "current_dist": current_dist,
                "movement_mag": mv_mag,
            }
        else:
            results[tid] = {
                "alignment": 0.0,
                "distance_reduction": 0.0,
                "current_dist": current_dist,
                "movement_mag": 0.0,
            }

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rally", type=str)
    parser.add_argument("--tolerance-ms", type=int, default=167)
    args = parser.parse_args()

    rallies = load_rallies_with_action_gt(rally_id=args.rally)
    if not rallies:
        console.print("[red]No rallies with action GT.[/red]")
        return

    calibrators: dict[str, CourtCalibrator | None] = {}
    video_ids = {r.video_id for r in rallies}
    for vid in video_ids:
        corners = load_court_calibration(vid)
        if corners and len(corners) == 4:
            cal = CourtCalibrator()
            cal.calibrate([(c["x"], c["y"]) for c in corners])
            calibrators[vid] = cal
        else:
            calibrators[vid] = None

    console.print(f"\n[bold]Testing movement signals across {len(rallies)} rallies[/bold]\n")

    total = 0
    cur_correct = 0
    # Ranking by different signals
    by_alignment = 0
    by_dist_reduction = 0
    by_current_dist = 0
    # Combined signals: distance * (2 - alignment) — alignment close to 1 reduces score
    combined_scores: dict[str, int] = defaultdict(int)
    # Analysis: when current is wrong, does alignment pick the GT?
    fixes_alignment = 0
    regressions_alignment = 0
    # Per-action
    per_action: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    lookbacks = [5, 10, 15]
    by_alignment_lb: dict[int, int] = {lb: 0 for lb in lookbacks}

    for i, rally in enumerate(rallies):
        if not rally.ball_positions_json or not rally.positions_json:
            continue

        cal = calibrators.get(rally.video_id)

        positions = [
            PlayerPos(
                frame_number=pp["frameNumber"],
                track_id=pp["trackId"],
                x=pp["x"], y=pp["y"],
                width=pp["width"], height=pp["height"],
                confidence=pp.get("confidence", 1.0),
            )
            for pp in rally.positions_json
        ]
        ball_positions = [
            BallPos(
                frame_number=bp["frameNumber"],
                x=bp["x"], y=bp["y"],
                confidence=bp.get("confidence", 1.0),
            )
            for bp in rally.ball_positions_json
            if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
        ]

        net_y_val = rally.court_split_y or 0.5
        per_rally_teams = classify_teams(positions, net_y_val)

        contact_seq = detect_contacts(
            ball_positions=ball_positions,
            player_positions=positions,
            config=ContactDetectionConfig(),
            net_y=rally.court_split_y,
            frame_count=rally.frame_count or None,
            court_calibrator=cal,
        )
        contacts = contact_seq.contacts
        contact_by_frame = {c.frame: c for c in contacts}

        rally_actions = classify_rally_actions(contact_seq, rally.rally_id)
        pred_actions_list = [a.to_dict() for a in rally_actions.actions]
        real_pred = [a for a in pred_actions_list if not a.get("isSynthetic")]

        fps = rally.fps or 30.0
        tol = max(1, round(fps * args.tolerance_ms / 1000))
        avail_tids = {pp["trackId"] for pp in rally.positions_json}

        matches, _ = match_contacts(
            rally.gt_labels, real_pred,
            tolerance=tol, available_track_ids=avail_tids,
        )

        for m in matches:
            if m.pred_frame is None:
                continue
            gt_tid = next(
                (gt.player_track_id for gt in rally.gt_labels if gt.frame == m.gt_frame),
                -1,
            )
            if gt_tid < 0 or gt_tid not in avail_tids or gt_tid not in per_rally_teams:
                continue

            contact = contact_by_frame.get(m.pred_frame)
            if contact is None:
                continue

            total += 1
            pred_tid = next(
                (a.get("playerTrackId", -1) for a in real_pred if a.get("frame") == m.pred_frame),
                -1,
            )
            cur_ok = pred_tid == gt_tid
            if cur_ok:
                cur_correct += 1

            # Compute movement scores with default lookback=10
            scores = compute_movement_scores(
                contact.frame, contact.ball_x, contact.ball_y,
                positions, lookback=10,
            )

            if not scores:
                continue

            # Rank by alignment
            by_align = sorted(scores.items(), key=lambda x: -x[1]["alignment"])
            if by_align[0][0] == gt_tid:
                by_alignment += 1

            # Rank by distance reduction
            by_dr = sorted(scores.items(), key=lambda x: -x[1]["distance_reduction"])
            if by_dr[0][0] == gt_tid:
                by_dist_reduction += 1

            # Rank by current distance
            by_cd = sorted(scores.items(), key=lambda x: x[1]["current_dist"])
            if by_cd[0][0] == gt_tid:
                by_current_dist += 1

            # Combined: current_dist * (2 - alignment)
            # alignment ranges from -1 to 1, so (2-alignment) ranges from 1 to 3
            # Players moving toward ball get lower combined score
            combined = sorted(
                scores.items(),
                key=lambda x: x[1]["current_dist"] * (2.0 - x[1]["alignment"]),
            )
            if combined[0][0] == gt_tid:
                combined_scores["dist*(2-align)"] += 1

            # Combined: current_dist * (1.5 - 0.5*alignment)
            combined2 = sorted(
                scores.items(),
                key=lambda x: x[1]["current_dist"] * (1.5 - 0.5 * x[1]["alignment"]),
            )
            if combined2[0][0] == gt_tid:
                combined_scores["dist*(1.5-0.5a)"] += 1

            # Fix/regression analysis for alignment
            align_ok = by_align[0][0] == gt_tid if by_align else False
            if align_ok and not cur_ok:
                fixes_alignment += 1
            elif not align_ok and cur_ok:
                regressions_alignment += 1

            # Lookback sweep for alignment
            for lb in lookbacks:
                lb_scores = compute_movement_scores(
                    contact.frame, contact.ball_x, contact.ball_y,
                    positions, lookback=lb,
                )
                if lb_scores:
                    lb_ranked = sorted(lb_scores.items(), key=lambda x: -x[1]["alignment"])
                    if lb_ranked[0][0] == gt_tid:
                        by_alignment_lb[lb] += 1

            # Per-action
            per_action[m.gt_action]["total"] += 1
            if cur_ok:
                per_action[m.gt_action]["cur"] += 1
            if combined[0][0] == gt_tid:
                per_action[m.gt_action]["combined"] += 1

        if (i + 1) % 20 == 0:
            console.print(f"  [{i + 1}/{len(rallies)}] {total} contacts")

    # Report
    console.print(f"\n[bold]Results: {total} evaluable contacts[/bold]\n")
    console.print(f"  Current (nearest): {cur_correct}/{total} = {cur_correct / max(1, total):.1%}")
    console.print(f"  By alignment: {by_alignment}/{total} = {by_alignment / max(1, total):.1%}")
    console.print(f"  By distance reduction: {by_dist_reduction}/{total} = {by_dist_reduction / max(1, total):.1%}")
    console.print(f"  By current distance: {by_current_dist}/{total} = {by_current_dist / max(1, total):.1%}")

    console.print(f"\n[bold cyan]Combined signals[/bold cyan]")
    for name, count in combined_scores.items():
        console.print(f"  {name}: {count}/{total} = {count / max(1, total):.1%}")

    console.print(f"\n  Alignment fixes: {fixes_alignment}, regressions: {regressions_alignment}")

    console.print(f"\n[bold cyan]Alignment by lookback[/bold cyan]")
    for lb in lookbacks:
        c = by_alignment_lb[lb]
        console.print(f"  Lookback {lb} frames: {c}/{total} = {c / max(1, total):.1%}")

    # Per-action for best combined
    console.print(f"\n[bold cyan]Per-action: current vs combined[/bold cyan]")
    act_table = Table()
    act_table.add_column("Action")
    act_table.add_column("Current", justify="right")
    act_table.add_column("Combined", justify="right")
    act_table.add_column("Total", justify="right")
    act_table.add_column("Delta", justify="right")

    for action in ["serve", "receive", "set", "attack", "dig"]:
        cc = per_action[action].get("cur", 0)
        cb = per_action[action].get("combined", 0)
        t = per_action[action].get("total", 0)
        if t > 0:
            act_table.add_row(
                action, f"{cc} ({cc/t:.0%})", f"{cb} ({cb/t:.0%})",
                str(t), f"{cb - cc:+d}",
            )
    console.print(act_table)


if __name__ == "__main__":
    main()
