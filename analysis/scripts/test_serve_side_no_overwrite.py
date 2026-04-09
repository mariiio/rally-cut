"""Test serve side accuracy with vs without team overwrite.

Runs classify_rally_actions twice per rally:
  1. With match_team_assignments (current behavior — team overwrites position)
  2. Without match_team_assignments (position detection only, no overwrite)

Also tests lower separation thresholds for position detection.

Usage:
    cd analysis
    uv run python scripts/test_serve_side_no_overwrite.py
"""

from __future__ import annotations

import logging
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import median

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.table import Table

from eval_sequence_enriched import (
    RallyBundle,
    get_sequence_probs,
    prepare_rallies,
    train_fold_gbm,
    train_mstcn,
)
from rallycut.actions.trajectory_features import ACTION_TYPES
from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.tracking.db import load_court_calibration
from rallycut.tracking.action_classifier import (
    ActionClassifier,
    ActionClassifierConfig,
    ActionType,
    classify_rally_actions,
)
from rallycut.tracking.contact_classifier import ContactClassifier
from rallycut.tracking.contact_detector import ContactDetectionConfig, detect_contacts

logging.getLogger("rallycut").setLevel(logging.WARNING)
console = Console()


def get_serve_side(
    bundle: RallyBundle,
    model: torch.nn.Module,
    gbm: ContactClassifier,
    device: torch.device,
    calibrator: CourtCalibrator | None,
    use_teams: bool,
) -> tuple[str, int, bool]:
    """Run pipeline and return (serve_side, server_tid, is_synthetic)."""
    probs = get_sequence_probs(model, bundle, device)

    contact_seq = detect_contacts(
        ball_positions=bundle.ball_positions,
        player_positions=bundle.player_positions,
        config=ContactDetectionConfig(),
        frame_count=bundle.rally.frame_count,
        classifier=gbm,
        sequence_probs=probs,
    )

    teams = bundle.match_teams if use_teams else None
    rally_actions = classify_rally_actions(
        contact_seq,
        match_team_assignments=teams,
        calibrator=calibrator,
    )

    # MS-TCN++ override (exempt serve)
    for action in rally_actions.actions:
        if action.is_synthetic or action.action_type == ActionType.SERVE:
            continue
        frame = action.frame
        if 0 <= frame < probs.shape[1]:
            cls = int(np.argmax(probs[1:, frame]))
            action.action_type = ActionType(ACTION_TYPES[cls])

    serves = [a for a in rally_actions.actions if a.action_type == ActionType.SERVE]
    if not serves:
        return "", -1, False
    s = serves[0]
    return s.court_side, s.player_track_id, s.is_synthetic


def get_gt_side(bundle: RallyBundle) -> tuple[str, int]:
    """Get GT serve side from position (Y-based, independent of teams)."""
    gt_serves = [gt for gt in bundle.gt_labels if gt.action == "serve"]
    if not gt_serves:
        return "", -1
    gt_s = gt_serves[0]
    if gt_s.player_track_id < 0 or bundle.rally.court_split_y is None:
        return "", -1

    ys = [p.y for p in bundle.player_positions
          if p.track_id == gt_s.player_track_id]
    if not ys:
        return "", -1

    med_y = median(ys)
    side = "near" if med_y > bundle.rally.court_split_y else "far"
    return side, gt_s.player_track_id


def main() -> None:
    import argparse
    import random
    import time

    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--tmse-weight", type=float, default=0.15)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--label-spread", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str,
        default="mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()
    random.seed(args.seed)

    bundles = prepare_rallies(label_spread=args.label_spread)
    if not bundles:
        console.print("[red]No rallies.[/red]")
        return

    videos: dict[str, list[RallyBundle]] = defaultdict(list)
    for b in bundles:
        videos[b.rally.video_id].append(b)
    video_ids = sorted(videos.keys())

    n_folds = args.folds if args.folds > 0 else len(video_ids)
    n_folds = min(n_folds, len(video_ids))
    fold_map = {vid: i % n_folds for i, vid in enumerate(video_ids)}

    calibrators: dict[str, CourtCalibrator] = {}
    for vid in video_ids:
        corners = load_court_calibration(vid)
        if corners and len(corners) == 4:
            cal = CourtCalibrator()
            cal.calibrate([(c["x"], c["y"]) for c in corners])
            if cal.is_calibrated:
                calibrators[vid] = cal

    console.print(f"[bold]Serve Side: Team Overwrite vs No Overwrite[/bold]")
    console.print(f"  {n_folds}-fold CV, {len(video_ids)} videos, {len(bundles)} rallies")
    console.print(f"  Calibrations: {len(calibrators)}/{len(video_ids)}")

    device = torch.device(args.device)

    @dataclass
    class Result:
        rally_id: str
        video_id: str
        gt_side: str
        with_teams_side: str
        no_teams_side: str
        with_teams_tid: int
        no_teams_tid: int

    results: list[Result] = []
    idx = 0

    for fold in range(n_folds):
        train_bundles = [b for b in bundles if fold_map[b.rally.video_id] != fold]
        test_bundles = [b for b in bundles if fold_map[b.rally.video_id] == fold]
        if not test_bundles or not train_bundles:
            continue

        t0 = time.time()
        model = train_mstcn(train_bundles, test_bundles, args)
        gbm = train_fold_gbm(train_bundles, model, device, use_sequence_features=True)

        for bundle in test_bundles:
            gt_side, gt_tid = get_gt_side(bundle)
            if not gt_side:
                continue

            cal = calibrators.get(bundle.rally.video_id)

            with_side, with_tid, _ = get_serve_side(
                bundle, model, gbm, device, cal, use_teams=True,
            )
            no_side, no_tid, _ = get_serve_side(
                bundle, model, gbm, device, cal, use_teams=False,
            )

            results.append(Result(
                rally_id=bundle.rally.rally_id,
                video_id=bundle.rally.video_id,
                gt_side=gt_side,
                with_teams_side=with_side,
                no_teams_side=no_side,
                with_teams_tid=with_tid,
                no_teams_tid=no_tid,
            ))
            idx += 1

            w_ok = "OK" if with_side == gt_side else "WRONG"
            n_ok = "OK" if no_side == gt_side else "WRONG"
            console.print(
                f"  [{idx}] {bundle.rally.rally_id[:8]} "
                f"with_team={with_side or '-':4s}({w_ok:5s}) "
                f"no_team={no_side or '-':4s}({n_ok:5s}) "
                f"gt={gt_side}"
            )

        console.print(f"  Fold [{fold+1}/{n_folds}] done ({time.time()-t0:.0f}s)")

    # Summary
    n = len(results)
    with_correct = sum(1 for r in results if r.with_teams_side == r.gt_side)
    no_correct = sum(1 for r in results if r.no_teams_side == r.gt_side)
    both_correct = sum(1 for r in results
                       if r.with_teams_side == r.gt_side and r.no_teams_side == r.gt_side)
    with_only = sum(1 for r in results
                    if r.with_teams_side == r.gt_side and r.no_teams_side != r.gt_side)
    no_only = sum(1 for r in results
                  if r.no_teams_side == r.gt_side and r.with_teams_side != r.gt_side)
    both_wrong = n - both_correct - with_only - no_only

    console.print(f"\n[bold]Results ({n} rallies)[/bold]")
    console.print(f"  With team overwrite:    {with_correct}/{n} ({with_correct/n:.1%})")
    console.print(f"  Without team overwrite: {no_correct}/{n} ({no_correct/n:.1%})")
    delta = no_correct - with_correct
    if delta > 0:
        console.print(f"  [green]Delta: {delta:+d} ({delta/n:+.1%})[/green]")
    elif delta < 0:
        console.print(f"  [red]Delta: {delta:+d} ({delta/n:+.1%})[/red]")
    else:
        console.print(f"  Delta: 0")

    console.print(f"\n  Both correct: {both_correct}")
    console.print(f"  Team overwrite helps (team right, no-team wrong): {with_only}")
    console.print(f"  Team overwrite hurts (no-team right, team wrong): {no_only}")
    console.print(f"  Both wrong: {both_wrong}")

    # Best-of oracle
    best_of = both_correct + with_only + no_only
    console.print(f"\n  Best-of oracle: {best_of}/{n} ({best_of/n:.1%})")

    # Breakdown: when they disagree
    disagree = [r for r in results
                if r.with_teams_side != r.no_teams_side
                and r.with_teams_side and r.no_teams_side]
    if disagree:
        console.print(f"\n  [bold]Disagreements ({len(disagree)} rallies)[/bold]")
        team_wins = sum(1 for r in disagree if r.with_teams_side == r.gt_side)
        pos_wins = sum(1 for r in disagree if r.no_teams_side == r.gt_side)
        neither = len(disagree) - team_wins - pos_wins
        console.print(f"    Team overwrite right: {team_wins}")
        console.print(f"    Position right:       {pos_wins}")
        console.print(f"    Neither right:        {neither}")


if __name__ == "__main__":
    main()
