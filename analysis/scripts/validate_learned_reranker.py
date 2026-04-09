"""Train a learned candidate re-ranker for player attribution.

For each contact, re-ranks the 4 player candidates using features that
capture perspective, position, and team context. Uses leave-one-rally-out
CV for honest evaluation.

Features per candidate:
1. Image-space distance to ball
2. Perspective-corrected distance (from candidates list)
3. Player bbox height (perspective proxy)
4. Player Y position
5. Ball Y - player Y (vertical offset)
6. |Ball X - player X| (horizontal alignment)
7. Player Y - net_y (side-of-net proxy)
8. Distance rank (0-3)
9. Ball Y - net_y (which side ball appears on)

Usage:
    cd analysis
    uv run python scripts/validate_learned_reranker.py
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict

import numpy as np
from rich.console import Console
from rich.table import Table
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.tracking.db import load_court_calibration
from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.contact_detector import (
    Contact,
    ContactDetectionConfig,
    detect_contacts,
    _depth_scale_at_y,
)
from rallycut.tracking.player_filter import classify_teams
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos

from scripts.eval_action_detection import (
    load_rallies_with_action_gt,
    match_contacts,
)

console = Console()


def extract_candidate_features(
    contact: Contact,
    player_positions: list[PlayerPos],
    per_rally_teams: dict[int, int],
    net_y: float,
    court_calibrator: CourtCalibrator | None,
    search_frames: int = 15,
) -> list[tuple[int, np.ndarray]]:
    """Extract features for each candidate player at a contact.

    Returns: list of (track_id, feature_vector) for up to 4 candidates.
    """
    ball_x = contact.ball_x
    ball_y = contact.ball_y

    # Find players in search window
    best_per_track: dict[int, PlayerPos] = {}
    for p in player_positions:
        if abs(p.frame_number - contact.frame) > search_frames:
            continue
        if p.track_id not in best_per_track:
            best_per_track[p.track_id] = p
        else:
            # Keep closest frame
            curr = best_per_track[p.track_id]
            if abs(p.frame_number - contact.frame) < abs(curr.frame_number - contact.frame):
                best_per_track[p.track_id] = p

    if not best_per_track:
        return []

    # Compute features
    candidates_features: list[tuple[int, np.ndarray, float]] = []

    max_height = max(p.height for p in best_per_track.values())

    for tid, p in best_per_track.items():
        player_x = p.x
        player_y = p.y - p.height * 0.25  # upper quarter contact point

        # Distances
        dx = ball_x - player_x
        dy = ball_y - player_y
        img_dist = math.sqrt(dx ** 2 + dy ** 2)

        # Perspective-corrected distance
        scale = _depth_scale_at_y(player_y, court_calibrator)
        corrected_dist = img_dist * scale * scale

        # Team info
        team = per_rally_teams.get(tid, -1)

        features = np.array([
            img_dist,                          # 0: image distance
            corrected_dist,                    # 1: perspective-corrected distance
            p.height,                          # 2: bbox height
            p.height / max(max_height, 0.01),  # 3: relative height (1=tallest)
            player_y,                          # 4: player Y
            ball_y - player_y,                 # 5: vertical offset (+ = ball below player in image)
            abs(ball_x - player_x),            # 6: horizontal alignment
            player_y - net_y,                  # 7: player Y vs net (+ = near court)
            ball_y - net_y,                    # 8: ball Y vs net
            float(team),                       # 9: team (0 or 1)
            p.width,                           # 10: bbox width
            p.width * p.height,                # 11: bbox area
        ])

        candidates_features.append((tid, features, corrected_dist))

    # Sort by corrected distance and add rank feature
    candidates_features.sort(key=lambda x: x[2])
    result = []
    for rank, (tid, feat, _) in enumerate(candidates_features[:4]):
        feat_with_rank = np.append(feat, float(rank))  # 12: distance rank
        result.append((tid, feat_with_rank))

    return result


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

    console.print(f"\n[bold]Building candidate re-ranker dataset from {len(rallies)} rallies[/bold]\n")

    # Collect data per rally for LOO-CV
    rally_data: list[dict] = []

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
        net_y = contact_seq.net_y
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

        rally_examples: list[tuple[np.ndarray, int, int]] = []  # (features, label, gt_tid)

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

            cand_features = extract_candidate_features(
                contact, positions, per_rally_teams, net_y, cal,
            )

            for tid, feat in cand_features:
                label = 1 if tid == gt_tid else 0
                rally_examples.append((feat, label, gt_tid))

        if rally_examples:
            rally_data.append({
                "rally_id": rally.rally_id,
                "examples": rally_examples,
            })

        if (i + 1) % 20 == 0:
            console.print(f"  [{i + 1}/{len(rallies)}] {len(rally_data)} rallies with data")

    console.print(f"\n  Total rallies with data: {len(rally_data)}")
    total_examples = sum(len(rd["examples"]) for rd in rally_data)
    total_positives = sum(sum(1 for _, l, _ in rd["examples"] if l == 1) for rd in rally_data)
    console.print(f"  Total examples: {total_examples}, positives: {total_positives}")

    # LOO-CV (leave-one-rally-out)
    console.print(f"\n[bold]Running leave-one-rally-out CV[/bold]\n")

    models = {
        "LogReg": lambda: LogisticRegression(max_iter=1000, C=1.0),
        "GBM-10": lambda: GradientBoostingClassifier(
            n_estimators=10, max_depth=3, learning_rate=0.1,
        ),
        "GBM-50": lambda: GradientBoostingClassifier(
            n_estimators=50, max_depth=3, learning_rate=0.1,
        ),
    }

    results_by_model: dict[str, dict[str, int]] = {}

    for model_name, model_factory in models.items():
        correct = 0
        total = 0
        correct_by_action: dict[str, int] = defaultdict(int)
        total_by_action: dict[str, int] = defaultdict(int)

        for test_idx in range(len(rally_data)):
            # Train on all except test rally
            train_X = []
            train_y = []
            for idx, rd in enumerate(rally_data):
                if idx == test_idx:
                    continue
                for feat, label, _ in rd["examples"]:
                    train_X.append(feat)
                    train_y.append(label)

            if not train_X or sum(train_y) == 0:
                continue

            train_X = np.array(train_X)
            train_y = np.array(train_y)

            # Scale features
            scaler = StandardScaler()
            train_X_scaled = scaler.fit_transform(train_X)

            # Train
            model = model_factory()
            model.fit(train_X_scaled, train_y)

            # Test on held-out rally
            test_rd = rally_data[test_idx]
            # Group examples by contact (consecutive groups of ~4)
            contact_groups: list[list[tuple[int, np.ndarray, int]]] = []
            current_gt = None
            current_group: list[tuple[int, np.ndarray, int]] = []

            for feat, label, gt_tid in test_rd["examples"]:
                if gt_tid != current_gt:
                    if current_group:
                        contact_groups.append(current_group)
                    current_group = []
                    current_gt = gt_tid
                # Extract track_id from rank (not stored — use label to find GT)
                current_group.append((label, feat, gt_tid))
            if current_group:
                contact_groups.append(current_group)

            for group in contact_groups:
                if not group:
                    continue
                feats = np.array([f for _, f, _ in group])
                labels = [l for l, _, _ in group]
                feats_scaled = scaler.transform(feats)

                # Predict probabilities and pick highest
                probs = model.predict_proba(feats_scaled)[:, 1]
                best_idx = int(np.argmax(probs))

                if labels[best_idx] == 1:
                    correct += 1
                total += 1

        results_by_model[model_name] = {"correct": correct, "total": total}
        console.print(f"  {model_name}: {correct}/{total} = {correct / max(1, total):.1%}")

    # Baseline comparison
    console.print(f"\n[bold]Summary[/bold]")
    # Count baseline accuracy from the data
    baseline_correct = 0
    baseline_total = 0
    for rd in rally_data:
        contact_groups: list[list[tuple[int, int]]] = []
        current_gt = None
        current_group: list[tuple[int, int]] = []
        for feat, label, gt_tid in rd["examples"]:
            if gt_tid != current_gt:
                if current_group:
                    contact_groups.append(current_group)
                current_group = []
                current_gt = gt_tid
            rank = int(feat[12])  # distance rank feature
            current_group.append((rank, label))
        if current_group:
            contact_groups.append(current_group)

        for group in contact_groups:
            baseline_total += 1
            # Rank 0 = current system's pick
            if any(label == 1 for rank, label in group if rank == 0):
                baseline_correct += 1

    console.print(f"  Baseline (rank 0): {baseline_correct}/{baseline_total} = {baseline_correct / max(1, baseline_total):.1%}")
    for name, res in results_by_model.items():
        delta = res["correct"] - baseline_correct
        console.print(f"  {name}: {res['correct']}/{res['total']} = {res['correct'] / max(1, res['total']):.1%} ({delta:+d})")

    # Feature importance from GBM
    console.print(f"\n[bold cyan]Feature importance (GBM-50, trained on all data)[/bold cyan]")
    all_X = []
    all_y = []
    for rd in rally_data:
        for feat, label, _ in rd["examples"]:
            all_X.append(feat)
            all_y.append(label)
    all_X = np.array(all_X)
    all_y = np.array(all_y)

    scaler = StandardScaler()
    all_X_scaled = scaler.fit_transform(all_X)
    gbm = GradientBoostingClassifier(n_estimators=50, max_depth=3, learning_rate=0.1)
    gbm.fit(all_X_scaled, all_y)

    feature_names = [
        "img_dist", "corrected_dist", "bbox_height", "rel_height",
        "player_y", "vert_offset", "horiz_align", "player_vs_net",
        "ball_vs_net", "team", "bbox_width", "bbox_area", "rank",
    ]
    importances = gbm.feature_importances_
    for name, imp in sorted(zip(feature_names, importances), key=lambda x: -x[1]):
        bar = "█" * int(imp * 50)
        console.print(f"  {name:15s} {imp:.3f} {bar}")


if __name__ == "__main__":
    main()
