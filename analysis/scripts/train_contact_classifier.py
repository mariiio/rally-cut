"""Train a contact classifier on labeled rally data.

Extracts features from all candidates (not just validated ones), matches them
against ground truth contacts, and trains a GradientBoostingClassifier.

Reports both train-on-all and leave-one-rally-out CV metrics.

Usage:
    cd analysis
    uv run python scripts/train_contact_classifier.py
    uv run python scripts/train_contact_classifier.py --threshold 0.4
    uv run python scripts/train_contact_classifier.py --output weights/contact_classifier/contact_classifier.pkl
"""

from __future__ import annotations

import argparse

import numpy as np
from rich.console import Console
from rich.table import Table

from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.contact_classifier import (
    CandidateFeatures,
    ContactClassifier,
)
from rallycut.tracking.contact_detector import (
    ContactDetectionConfig,
    _compute_direction_change,
    _compute_velocities,
    _filter_noise_spikes,
    _find_inflection_candidates,
    _find_nearest_player,
    _find_parabolic_breakpoints,
    _find_velocity_reversal_candidates,
    _merge_candidates,
    _smooth_signal,
    estimate_net_position,
)
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos
from scripts.eval_action_detection import (
    GtLabel,
    RallyData,
    load_rallies_with_action_gt,
)

console = Console()

_CONFIDENCE_THRESHOLD = 0.3


def extract_candidate_features(
    rally: RallyData,
    config: ContactDetectionConfig | None = None,
) -> tuple[list[CandidateFeatures], list[int]]:
    """Extract features for ALL candidates in a rally (before validation gate).

    Returns:
        Tuple of (features_list, candidate_frames).
    """
    from scipy.signal import find_peaks

    cfg = config or ContactDetectionConfig()

    if not rally.ball_positions_json:
        return [], []

    ball_positions = [
        BallPos(
            frame_number=bp["frameNumber"],
            x=bp["x"],
            y=bp["y"],
            confidence=bp.get("confidence", 1.0),
        )
        for bp in rally.ball_positions_json
        if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
    ]

    if not ball_positions:
        return [], []

    # Pre-filter noise spikes
    if cfg.enable_noise_filter:
        ball_positions = _filter_noise_spikes(ball_positions, cfg.noise_spike_max_jump)

    # Net position
    estimated_net_y = (
        rally.court_split_y
        if rally.court_split_y is not None
        else estimate_net_position(ball_positions)
    )

    # Velocities
    velocities = _compute_velocities(ball_positions)
    if not velocities:
        return [], []

    frames = sorted(velocities.keys())
    if len(frames) < 3:
        return [], []

    speeds = [velocities[f][0] for f in frames]
    smoothed = _smooth_signal(speeds, cfg.smoothing_window)

    # Velocity peaks
    peak_indices, _ = find_peaks(
        smoothed,
        height=cfg.min_peak_velocity,
        prominence=cfg.min_peak_prominence,
        distance=cfg.min_peak_distance_frames,
    )
    velocity_peak_frames = [frames[idx] for idx in peak_indices]

    # Ball by frame
    ball_by_frame = {
        bp.frame_number: bp
        for bp in ball_positions
        if bp.confidence >= _CONFIDENCE_THRESHOLD
    }
    first_frame = frames[0]
    confident_frames = sorted(ball_by_frame.keys())

    # Inflection candidates
    inflection_frames = _find_inflection_candidates(
        ball_by_frame, confident_frames,
        min_angle_deg=cfg.min_inflection_angle_deg,
        check_frames=cfg.inflection_check_frames,
        min_distance_frames=cfg.min_peak_distance_frames,
    )

    # Reversal candidates
    reversal_frames = _find_velocity_reversal_candidates(
        velocities, frames, cfg.min_peak_distance_frames
    )

    # Parabolic breakpoints
    parabolic_frames, residual_by_frame = _find_parabolic_breakpoints(
        ball_by_frame, confident_frames,
        window_frames=cfg.parabolic_window_frames,
        stride=cfg.parabolic_stride,
        min_residual=cfg.parabolic_min_residual,
        min_prominence=cfg.parabolic_min_prominence,
        min_distance_frames=cfg.min_peak_distance_frames,
    )

    # Merge all candidates
    inflection_and_reversal = _merge_candidates(
        inflection_frames, reversal_frames, cfg.min_peak_distance_frames
    )
    traditional = _merge_candidates(
        velocity_peak_frames, inflection_and_reversal, cfg.min_peak_distance_frames
    )
    candidate_frames = _merge_candidates(
        traditional, parabolic_frames, cfg.min_peak_distance_frames
    )

    # Build source sets
    velocity_peak_set = set(velocity_peak_frames)
    inflection_set = set(inflection_frames)
    parabolic_set = set(parabolic_frames)

    # Build velocity lookup
    velocity_lookup = dict(zip(frames, smoothed))

    # Player positions
    player_positions: list[PlayerPos] = []
    if rally.positions_json:
        player_positions = [
            PlayerPos(
                frame_number=pp["frameNumber"],
                track_id=pp["trackId"],
                x=pp["x"],
                y=pp["y"],
                width=pp["width"],
                height=pp["height"],
                confidence=pp.get("confidence", 1.0),
            )
            for pp in rally.positions_json
        ]

    features_list: list[CandidateFeatures] = []
    valid_frames: list[int] = []
    prev_frame = 0

    for frame in candidate_frames:
        if frame - first_frame < cfg.warmup_skip_frames:
            continue

        if rally.frame_count and rally.frame_count > 0 and frame > rally.frame_count:
            continue

        velocity = velocity_lookup.get(frame, 0.0)
        if velocity < cfg.min_candidate_velocity:
            continue

        # Ball position
        ball = ball_by_frame.get(frame)
        if ball is None:
            for offset in [-1, 1, -2, 2]:
                ball = ball_by_frame.get(frame + offset)
                if ball is not None:
                    break
        if ball is None:
            continue

        # Direction change
        direction_change = _compute_direction_change(
            ball_by_frame, frame, cfg.direction_check_frames
        )

        # Player
        if player_positions:
            track_id, player_dist, _ = _find_nearest_player(
                frame, ball.x, ball.y, player_positions,
                search_frames=cfg.player_search_frames,
            )
        else:
            player_dist = float("inf")

        has_player = player_dist <= cfg.player_contact_radius
        net_zone = 0.08
        is_at_net = abs(ball.y - estimated_net_y) < net_zone
        arc_residual = residual_by_frame.get(frame, 0.0)

        frames_since_last = frame - prev_frame if prev_frame > 0 else 0

        features = CandidateFeatures(
            frame=frame,
            velocity=velocity,
            direction_change_deg=direction_change,
            arc_fit_residual=arc_residual,
            player_distance=player_dist,
            has_player=has_player,
            ball_x=ball.x,
            ball_y=ball.y,
            ball_y_relative_net=ball.y - estimated_net_y,
            is_at_net=is_at_net,
            frames_since_last=frames_since_last,
            is_velocity_peak=frame in velocity_peak_set,
            is_inflection=frame in inflection_set,
            is_parabolic=frame in parabolic_set,
        )

        features_list.append(features)
        valid_frames.append(frame)
        prev_frame = frame

    return features_list, valid_frames


def label_candidates(
    candidate_frames: list[int],
    gt_labels: list[GtLabel],
    tolerance: int = 5,
) -> list[int]:
    """Label candidates as 1 (matches GT) or 0 (no match).

    Uses greedy matching: each GT label matches at most one candidate.
    """
    labels = [0] * len(candidate_frames)
    used_gt: set[int] = set()

    for i, frame in enumerate(candidate_frames):
        for j, gt in enumerate(gt_labels):
            if j in used_gt:
                continue
            if abs(frame - gt.frame) <= tolerance:
                labels[i] = 1
                used_gt.add(j)
                break

    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Train contact classifier")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output", type=str, default="weights/contact_classifier/contact_classifier.pkl")
    parser.add_argument("--tolerance", type=int, default=5, help="Frame tolerance for GT matching")
    args = parser.parse_args()

    rallies = load_rallies_with_action_gt()
    if not rallies:
        console.print("[red]No rallies with action GT found.[/red]")
        return

    console.print(f"\n[bold]Extracting features from {len(rallies)} rallies[/bold]\n")

    all_features: list[np.ndarray] = []
    all_y: list[int] = []
    all_rally_ids: list[str] = []

    per_rally_table = Table(title="Per-Rally Feature Extraction")
    per_rally_table.add_column("Rally", max_width=10)
    per_rally_table.add_column("Candidates", justify="right")
    per_rally_table.add_column("Positive", justify="right")
    per_rally_table.add_column("Negative", justify="right")
    per_rally_table.add_column("GT Labels", justify="right")

    for rally in rallies:
        features_list, candidate_frames = extract_candidate_features(rally)

        if not features_list:
            per_rally_table.add_row(rally.rally_id[:8], "0", "0", "0", str(len(rally.gt_labels)))
            continue

        labels = label_candidates(candidate_frames, rally.gt_labels, tolerance=args.tolerance)

        for feat, label in zip(features_list, labels):
            all_features.append(feat.to_array())
            all_y.append(label)
            all_rally_ids.append(rally.rally_id)

        n_pos = sum(labels)
        n_neg = len(labels) - n_pos
        per_rally_table.add_row(
            rally.rally_id[:8],
            str(len(features_list)),
            str(n_pos),
            str(n_neg),
            str(len(rally.gt_labels)),
        )

    console.print(per_rally_table)

    if not all_features:
        console.print("[red]No candidates extracted.[/red]")
        return

    x_mat = np.array(all_features)
    y = np.array(all_y)
    rally_ids = np.array(all_rally_ids)

    console.print(f"\n[bold]Dataset: {len(y)} candidates, {int(y.sum())} positive, {int((1-y).sum())} negative[/bold]")

    # Train classifier
    classifier = ContactClassifier(threshold=args.threshold)

    # Train on all data
    train_metrics = classifier.train(x_mat, y)
    console.print("\n[bold]Train-on-all:[/bold]")
    console.print(f"  F1: {train_metrics['train_f1']:.1%}")
    console.print(f"  Precision: {train_metrics['train_precision']:.1%}")
    console.print(f"  Recall: {train_metrics['train_recall']:.1%}")
    console.print(f"  TP: {train_metrics['train_tp']}, FP: {train_metrics['train_fp']}, FN: {train_metrics['train_fn']}")

    # LOO CV
    loo_metrics = classifier.loo_cv(x_mat, y, rally_ids)
    console.print(f"\n[bold]Leave-One-Rally-Out CV ({loo_metrics['n_rallies']} folds):[/bold]")
    console.print(f"  F1: {loo_metrics['loo_f1']:.1%}")
    console.print(f"  Precision: {loo_metrics['loo_precision']:.1%}")
    console.print(f"  Recall: {loo_metrics['loo_recall']:.1%}")
    console.print(f"  TP: {loo_metrics['loo_tp']}, FP: {loo_metrics['loo_fp']}, FN: {loo_metrics['loo_fn']}")

    # Feature importance
    importance = classifier.feature_importance()
    if importance:
        imp_table = Table(title="\nFeature Importance")
        imp_table.add_column("Feature")
        imp_table.add_column("Importance", justify="right")

        for name, imp in sorted(importance.items(), key=lambda x: -x[1]):
            imp_table.add_row(name, f"{imp:.3f}")

        console.print(imp_table)

    # Threshold sweep
    console.print("\n[bold]Threshold sweep (LOO CV):[/bold]")
    sweep_table = Table()
    sweep_table.add_column("Threshold", justify="right")
    sweep_table.add_column("F1", justify="right")
    sweep_table.add_column("Precision", justify="right")
    sweep_table.add_column("Recall", justify="right")

    best_f1 = 0.0
    best_threshold = args.threshold

    for t in [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]:
        sweep_clf = ContactClassifier(threshold=t)
        sweep_metrics = sweep_clf.loo_cv(x_mat, y, rally_ids)
        f1 = sweep_metrics["loo_f1"]
        sweep_table.add_row(
            f"{t:.2f}",
            f"{f1:.1%}",
            f"{sweep_metrics['loo_precision']:.1%}",
            f"{sweep_metrics['loo_recall']:.1%}",
        )
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    console.print(sweep_table)
    console.print(f"\n[bold]Best threshold: {best_threshold:.2f} (LOO F1: {best_f1:.1%})[/bold]")

    # Retrain with best threshold and save
    classifier = ContactClassifier(threshold=best_threshold)
    classifier.train(x_mat, y)
    classifier.save(args.output)
    console.print(f"\n[green]Saved classifier to {args.output}[/green]")


if __name__ == "__main__":
    main()
