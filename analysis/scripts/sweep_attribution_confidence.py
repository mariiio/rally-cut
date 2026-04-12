"""Sweep attribution confidence thresholds to find optimal values.

Tests different pose_attribution_min_confidence and temporal_attribution_min_confidence
values and measures oracle attribution accuracy for each.

Usage:
    cd analysis
    uv run python scripts/sweep_attribution_confidence.py
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
from rich.console import Console
from rich.table import Table
from scipy.optimize import linear_sum_assignment

from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.tracking.db import load_court_calibration
from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.contact_detector import (
    ContactDetectionConfig,
    detect_contacts,
)
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos

from scripts.eval_action_detection import (
    _load_match_team_assignments,
    load_rallies_with_action_gt,
    match_contacts,
)

console = Console()


def oracle_accuracy(
    rallies: list,
    team_map: dict,
    calibrators: dict,
    cfg: ContactDetectionConfig,
    tolerance_ms: int = 167,
) -> tuple[int, int]:
    """Run detection + action classification with given config and measure oracle accuracy."""
    total_correct = 0
    total_evaluable = 0

    for rally in rallies:
        if not rally.ball_positions_json or not rally.positions_json:
            continue

        ball_positions = [
            BallPos(
                frame_number=bp["frameNumber"],
                x=bp["x"], y=bp["y"],
                confidence=bp.get("confidence", 1.0),
            )
            for bp in rally.ball_positions_json
            if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
        ]
        player_positions = [
            PlayerPos(
                frame_number=pp["frameNumber"],
                track_id=pp["trackId"],
                x=pp["x"], y=pp["y"],
                width=pp["width"], height=pp["height"],
                confidence=pp.get("confidence", 1.0),
            )
            for pp in rally.positions_json
        ]

        cal = calibrators.get(rally.video_id)
        match_teams = team_map.get(rally.rally_id)

        contact_seq = detect_contacts(
            ball_positions=ball_positions,
            player_positions=player_positions,
            config=cfg,
            net_y=rally.court_split_y,
            frame_count=rally.frame_count or None,
            court_calibrator=cal,
        )

        rally_actions = classify_rally_actions(
            contact_seq, rally.rally_id,
            match_team_assignments=match_teams,
        )
        pred_actions = [a.to_dict() for a in rally_actions.actions]
        real_pred = [a for a in pred_actions if not a.get("isSynthetic")]

        avail_tids = {pp["trackId"] for pp in rally.positions_json}
        fps = rally.fps or 30.0
        tol = max(1, round(fps * tolerance_ms / 1000))

        matches, _ = match_contacts(
            rally.gt_labels, real_pred,
            tolerance=tol, available_track_ids=avail_tids,
        )

        # Oracle matching (same as production_eval)
        gt_by_frame_action = {
            (gl.frame, gl.action): gl.player_track_id for gl in rally.gt_labels
        }
        pred_by_frame_action = {
            (a.get("frame", -1), a.get("action", "")): a.get("playerTrackId", -1)
            for a in real_pred
        }

        tid_pairs: list[tuple[int, int]] = []
        evaluable = [
            m for m in matches
            if m.player_evaluable and m.pred_frame is not None and m.gt_action != "block"
        ]

        for m in evaluable:
            gt_tid = gt_by_frame_action.get((m.gt_frame, m.gt_action))
            pred_tid = pred_by_frame_action.get((m.pred_frame, m.pred_action or ""))
            if gt_tid is not None and gt_tid >= 0 and pred_tid is not None and pred_tid >= 0:
                tid_pairs.append((gt_tid, pred_tid))

        if not tid_pairs:
            continue

        # Hungarian assignment
        gt_ids: list[int] = []
        pred_ids: list[int] = []
        for gt_tid, pred_tid in tid_pairs:
            if gt_tid not in gt_ids:
                gt_ids.append(gt_tid)
            if pred_tid not in pred_ids:
                pred_ids.append(pred_tid)

        size = max(len(gt_ids), len(pred_ids))
        cost = np.zeros((size, size), dtype=np.float64)
        for gt_tid, pred_tid in tid_pairs:
            g_idx = gt_ids.index(gt_tid)
            p_idx = pred_ids.index(pred_tid)
            cost[g_idx, p_idx] -= 1.0

        row_ind, col_ind = linear_sum_assignment(cost)
        mapping: dict[int, int] = {}
        for r, c in zip(row_ind, col_ind):
            if r < len(gt_ids) and c < len(pred_ids):
                mapping[gt_ids[r]] = pred_ids[c]

        correct = sum(
            1 for gt_tid, pred_tid in tid_pairs
            if mapping.get(gt_tid) == pred_tid
        )
        total_correct += correct
        total_evaluable += len(tid_pairs)

    return total_correct, total_evaluable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Reduced sweep grid")
    args = parser.parse_args()

    rallies = load_rallies_with_action_gt()
    if not rallies:
        console.print("[red]No rallies found.[/red]")
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

    team_map = _load_match_team_assignments(video_ids)

    # Sweep grid
    if args.quick:
        pose_thresholds = [0.5, 0.6, 0.7, 0.8]
        temporal_thresholds = [0.6, 0.7, 0.8]
    else:
        pose_thresholds = [0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0]
        temporal_thresholds = [0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.9, 1.0]

    # Also test disabling each model
    special_configs = [
        ("baseline (proximity only)", ContactDetectionConfig(
            use_pose_attribution=False, use_temporal_attribution=False)),
        ("pose only (no temporal)", ContactDetectionConfig(
            use_temporal_attribution=False)),
        ("temporal only (no pose)", ContactDetectionConfig(
            use_pose_attribution=False)),
    ]

    console.print(f"\n[bold]Attribution Confidence Threshold Sweep — {len(rallies)} rallies[/bold]\n")

    # Run special configs first
    console.print("[bold cyan]Special configurations:[/bold cyan]")
    for name, cfg in special_configs:
        correct, evaluable = oracle_accuracy(rallies, team_map, calibrators, cfg)
        acc = correct / max(1, evaluable)
        console.print(f"  {name}: {acc:.1%} ({correct}/{evaluable})")

    # Run grid sweep
    console.print(f"\n[bold cyan]Grid sweep: pose_min_conf × temporal_min_conf[/bold cyan]")
    results: list[tuple[float, float, float, int, int]] = []

    total_combos = len(pose_thresholds) * len(temporal_thresholds)
    i = 0
    for pose_t in pose_thresholds:
        for temp_t in temporal_thresholds:
            i += 1
            cfg = ContactDetectionConfig(
                pose_attribution_min_confidence=pose_t,
                temporal_attribution_min_confidence=temp_t,
            )
            correct, evaluable = oracle_accuracy(rallies, team_map, calibrators, cfg)
            acc = correct / max(1, evaluable)
            results.append((pose_t, temp_t, acc, correct, evaluable))

            if i % 5 == 0 or i == total_combos:
                console.print(f"  [{i}/{total_combos}] pose={pose_t:.2f} temp={temp_t:.2f} → {acc:.1%}")

    # Sort by accuracy
    results.sort(key=lambda x: -x[2])

    console.print(f"\n[bold green]Top 10 configurations:[/bold green]")
    top_table = Table()
    top_table.add_column("Pose Min Conf", justify="right")
    top_table.add_column("Temp Min Conf", justify="right")
    top_table.add_column("Oracle Acc", justify="right")
    top_table.add_column("Correct", justify="right")
    top_table.add_column("Evaluable", justify="right")

    for pose_t, temp_t, acc, correct, evaluable in results[:10]:
        style = "bold" if (pose_t, temp_t) == (0.5, 0.6) else ""
        top_table.add_row(
            f"{pose_t:.2f}", f"{temp_t:.2f}", f"{acc:.1%}",
            str(correct), str(evaluable),
            style=style,
        )
    console.print(top_table)

    # Current default for comparison
    default = next(
        (r for r in results if r[0] == 0.5 and r[1] == 0.6), None
    )
    best = results[0]
    if default:
        console.print(f"\n  Current default (pose=0.50, temp=0.60): {default[2]:.1%}")
    console.print(f"  Best found (pose={best[0]:.2f}, temp={best[1]:.2f}): {best[2]:.1%}")
    if default:
        delta = best[2] - default[2]
        console.print(f"  Delta: {delta:+.1%}")


if __name__ == "__main__":
    main()
