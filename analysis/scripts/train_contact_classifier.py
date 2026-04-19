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
    _check_net_crossing,
    _compute_acceleration,
    _compute_candidate_bbox_motion,
    _compute_trajectory_curvature,
    _compute_velocities,
    _compute_velocity_ratio,
    _count_consecutive_detections,
    _filter_noise_spikes,
    _find_deceleration_candidates,
    _find_direction_change_candidates,
    _find_inflection_candidates,
    _find_nearest_player,
    _find_nearest_players,
    _find_net_crossing_candidates,
    _find_parabolic_breakpoints,
    _find_player_motion_candidates,
    _find_proximity_frame,
    _find_velocity_reversal_candidates,
    _merge_candidates,
    _refine_candidates_to_trajectory_peak,
    _smooth_signal,
    compute_direction_change,
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
    sequence_probs: np.ndarray | None = None,
    gt_frames: list[int] | None = None,
) -> tuple[list[CandidateFeatures], list[int]]:
    """Extract features for ALL candidates in a rally (before validation gate).

    Args:
        rally: Rally data with ball/player positions.
        config: Contact detection configuration.
        sequence_probs: No-op, kept for backward compat.
        gt_frames: Optional GT contact frames. When provided, frames_since_last
            is computed from the last GT-matched candidate (approximating
            inference semantics where it's measured from last accepted contact).
            When None, frames_since_last is measured from the last candidate.

    Returns:
        Tuple of (features_list, candidate_frames).
    """
    from scipy.signal import find_peaks

    from rallycut.tracking.pose_attribution.features import (
        extract_contact_pose_features_for_nearest,
    )

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

    # Deceleration candidates
    deceleration_frames: list[int] = []
    if cfg.enable_deceleration_detection:
        deceleration_frames = _find_deceleration_candidates(
            velocities, frames, smoothed,
            cfg.min_peak_distance_frames,
            min_speed_before=cfg.deceleration_min_speed_before,
            min_speed_drop_ratio=cfg.deceleration_min_drop_ratio,
            window=cfg.deceleration_window,
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

    # Net-crossing candidates
    net_crossing_frames = _find_net_crossing_candidates(
        ball_by_frame, confident_frames, estimated_net_y, cfg.min_peak_distance_frames
    )

    # Merge all candidates (must match detect_contacts merge order)
    inflection_and_reversal = _merge_candidates(
        inflection_frames, reversal_frames, cfg.min_peak_distance_frames
    )
    traditional = _merge_candidates(
        velocity_peak_frames, inflection_and_reversal, cfg.min_peak_distance_frames
    )
    with_deceleration = _merge_candidates(
        traditional, deceleration_frames, cfg.min_peak_distance_frames
    )
    with_parabolic = _merge_candidates(
        with_deceleration, parabolic_frames, cfg.min_peak_distance_frames
    )

    # Direction-change peak candidates (must match detect_contacts merge order)
    direction_change_frames: list[int] = []
    if cfg.enable_direction_change_candidates:
        direction_change_frames = _find_direction_change_candidates(
            ball_by_frame, confident_frames,
            min_angle_deg=cfg.direction_change_candidate_min_deg,
            check_frames=cfg.direction_check_frames,
            min_distance_frames=cfg.min_peak_distance_frames,
            prominence=cfg.direction_change_candidate_prominence,
        )

    with_net_crossing = _merge_candidates(
        with_parabolic, net_crossing_frames, cfg.min_peak_distance_frames
    )
    # Direction-change peaks get highest priority (closer to actual contact)
    candidate_frames = _merge_candidates(
        direction_change_frames, with_net_crossing, cfg.min_peak_distance_frames
    ) if direction_change_frames else with_net_crossing

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
                keypoints=pp.get("keypoints"),
            )
            for pp in rally.positions_json
        ]

    # Player-motion candidates (must match detect_contacts logic)
    if player_positions and cfg.enable_player_motion_candidates:
        player_motion_frames = _find_player_motion_candidates(
            player_positions, ball_by_frame, candidate_frames,
            min_distance_frames=cfg.min_peak_distance_frames,
            min_d_y=cfg.player_motion_min_d_y,
            min_d_height=cfg.player_motion_min_d_height,
            max_ball_distance=cfg.player_motion_max_ball_distance,
        )
        if player_motion_frames:
            candidate_frames = _merge_candidates(
                candidate_frames, player_motion_frames, cfg.min_peak_distance_frames
            )

    # Trajectory-peak refinement (must match detect_contacts logic: before proximity)
    if cfg.enable_trajectory_refinement:
        candidate_frames = _refine_candidates_to_trajectory_peak(
            candidate_frames, ball_by_frame,
            direction_check_frames=cfg.direction_check_frames,
            search_window=cfg.trajectory_refinement_window,
            first_frame=first_frame,
            serve_window_frames=cfg.serve_window_frames,
        )

    # Generate player-proximity candidates (must match detect_contacts logic)
    if player_positions and cfg.enable_proximity_candidates:
        candidate_set = set(candidate_frames)
        for frame in list(candidate_frames):
            prox = _find_proximity_frame(
                frame, ball_by_frame, player_positions,
                search_window=cfg.proximity_search_window,
                player_search_frames=cfg.player_search_frames,
                max_distance=cfg.player_contact_radius,
            )
            if prox is not None and prox != frame and prox not in candidate_set:
                candidate_set.add(prox)
        candidate_frames = sorted(candidate_set)

    # Pre-compute GT-matched candidate frames for frames_since_last.
    # At inference, frames_since_last is measured from the last ACCEPTED contact.
    # During training, we approximate this by measuring from the last GT-matched
    # candidate (tolerance ±5 frames). This prevents the classifier from learning
    # that close candidates are always noise — real contacts CAN be 3-5 frames
    # apart (attack→dig, attack→block).
    gt_candidate_set: set[int] = set()
    if gt_frames:
        for cf in candidate_frames:
            for gf in gt_frames:
                if abs(cf - gf) <= 5:
                    gt_candidate_set.add(cf)
                    break

    features_list: list[CandidateFeatures] = []
    valid_frames: list[int] = []
    prev_accepted_frame = 0

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
            for offset in [-1, 1, -2, 2, -3, 3]:
                ball = ball_by_frame.get(frame + offset)
                if ball is not None:
                    break
        if ball is None:
            continue

        # Direction change
        direction_change = compute_direction_change(
            ball_by_frame, frame, cfg.direction_check_frames
        )

        # Player
        bbox_motion: dict[int, tuple[float, float]] = {}
        if player_positions:
            track_id, player_dist, _ = _find_nearest_player(
                frame, ball.x, ball.y, player_positions,
                search_frames=cfg.player_search_frames,
            )
            candidates_ranked = _find_nearest_players(
                frame, ball.x, ball.y, player_positions,
                search_frames=cfg.player_candidate_search_frames,
            )
            if candidates_ranked:
                bbox_motion = _compute_candidate_bbox_motion(
                    player_positions, frame,
                    [tid for tid, _, _ in candidates_ranked],
                )
        else:
            track_id = -1
            player_dist = float("inf")

        # Aggregate bbox motion features
        best_d_y = max((dy for dy, _ in bbox_motion.values()), default=0.0)
        best_d_h = max((dh for _, dh in bbox_motion.values()), default=0.0)
        nearest_d_y = bbox_motion.get(track_id, (0.0, 0.0))[0] if track_id >= 0 else 0.0
        nearest_d_h = bbox_motion.get(track_id, (0.0, 0.0))[1] if track_id >= 0 else 0.0

        arc_residual = residual_by_frame.get(frame, 0.0)

        # New features
        acceleration = _compute_acceleration(velocities, frame, window=3)
        curvature = _compute_trajectory_curvature(ball_by_frame, frame, window=5)
        is_net_cross = _check_net_crossing(
            ball_by_frame, frame, estimated_net_y, window=5
        )

        frames_since_last = (
            frame - prev_accepted_frame if prev_accepted_frame > 0 else 0
        )

        # Ball detection density: fraction of ±10 frames with confident ball
        density_window = 10
        n_with_ball = sum(
            1 for f in range(frame - density_window, frame + density_window + 1)
            if f in ball_by_frame
        )
        ball_detection_density = n_with_ball / (2 * density_window + 1)

        # Vertical velocity component
        vel_y = velocities[frame][2] if frame in velocities else 0.0

        # Velocity ratio: speed after / speed before
        vel_ratio = _compute_velocity_ratio(velocities, frame, window=5)

        # Consecutive ball detections around frame
        consec = _count_consecutive_detections(ball_by_frame, frame)

        # Sequence model context: max non-background probability within ±5 frames.
        # Provides temporal context that single-frame features lack.
        seq_max_nonbg = 0.0
        if sequence_probs is not None and sequence_probs.ndim == 2 and sequence_probs.shape[0] >= 2:
            t_seq = sequence_probs.shape[1]
            window = 5
            lo = max(0, frame - window)
            hi = min(t_seq - 1, frame + window)
            if hi >= lo:
                seq_max_nonbg = float(sequence_probs[1:, lo:hi + 1].max())

        # Pose features for nearest player (0.0 when keypoints unavailable)
        (
            pose_wrist_vel_max,
            pose_hand_ball_dist_min,
            pose_arm_ext_change,
            pose_conf_mean,
            pose_both_arms_raised,
        ) = extract_contact_pose_features_for_nearest(
            contact_frame=frame,
            nearest_track_id=track_id,
            player_positions=player_positions,
            ball_at_contact=(ball.x, ball.y),
            ball_by_frame=ball_by_frame,
        )

        features = CandidateFeatures(
            frame=frame,
            velocity=velocity,
            direction_change_deg=direction_change,
            arc_fit_residual=arc_residual,
            acceleration=acceleration,
            trajectory_curvature=curvature,
            velocity_y=vel_y,
            velocity_ratio=vel_ratio,
            player_distance=player_dist,
            best_player_max_d_y=best_d_y,
            best_player_max_d_height=best_d_h,
            nearest_player_max_d_y=nearest_d_y,
            nearest_player_max_d_height=nearest_d_h,
            ball_x=ball.x,
            ball_y=ball.y,
            ball_y_relative_net=ball.y - estimated_net_y,
            is_net_crossing=is_net_cross,
            frames_since_last=frames_since_last,
            ball_detection_density=ball_detection_density,
            consecutive_detections=consec,
            frames_since_rally_start=frame - first_frame,
            nearest_active_wrist_velocity_max=pose_wrist_vel_max,
            nearest_hand_ball_dist_min=pose_hand_ball_dist_min,
            nearest_active_arm_extension_change=pose_arm_ext_change,
            nearest_pose_confidence_mean=pose_conf_mean,
            nearest_both_arms_raised=pose_both_arms_raised,
            seq_max_nonbg=seq_max_nonbg,
        )

        features_list.append(features)
        valid_frames.append(frame)
        # Update prev_accepted_frame only for GT-matched candidates (or all if no GT)
        if not gt_frames or frame in gt_candidate_set:
            prev_accepted_frame = frame

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
    parser.add_argument("--threshold", type=float, default=0.35, help="Classifier threshold (default: 0.35, end-to-end optimal)")
    parser.add_argument("--output", type=str, default="weights/contact_classifier/contact_classifier.pkl")
    parser.add_argument("--tolerance", type=int, default=5, help="Frame tolerance for GT matching")
    parser.add_argument("--positive-weight", type=float, default=1.0, help="Weight multiplier for positive samples (recall bias)")
    parser.add_argument("--config", type=str, help="JSON config overrides for ContactDetectionConfig")
    args = parser.parse_args()

    # Build ContactDetectionConfig from overrides
    contact_config: ContactDetectionConfig | None = None
    if args.config:
        import json
        try:
            overrides = json.loads(args.config)
        except json.JSONDecodeError as e:
            console.print(f"[red]Invalid JSON in --config: {e}[/red]")
            return
        contact_config = ContactDetectionConfig(**overrides)
        console.print(f"[bold]Config overrides:[/bold] {overrides}")

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

    # Pre-compute sequence model probs for all rallies (temporal context feature)
    from rallycut.tracking.sequence_action_runtime import get_sequence_probs
    console.print("[dim]Computing sequence model probs for temporal context...[/dim]")
    seq_probs_cache: dict[str, np.ndarray | None] = {}
    for rally in rallies:
        bps = [
            BallPos(frame_number=bp["frameNumber"], x=bp["x"], y=bp["y"],
                    confidence=bp.get("confidence", 1.0))
            for bp in (rally.ball_positions_json or [])
            if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
        ]
        _pps = [
            PlayerPos(frame_number=pp["frameNumber"], track_id=pp["trackId"],
                x=pp["x"], y=pp["y"], width=pp["width"], height=pp["height"],
                confidence=pp.get("confidence", 1.0), keypoints=pp.get("keypoints"))
            for pp in (rally.positions_json or [])
        ]
        seq_probs_cache[rally.rally_id] = get_sequence_probs(
            bps, _pps, rally.court_split_y, rally.frame_count or 0, None,
        )
    n_with_seq = sum(1 for v in seq_probs_cache.values() if v is not None)
    console.print(f"[dim]  {n_with_seq}/{len(rallies)} rallies have sequence probs[/dim]")

    for rally in rallies:
        gt_frames = [gt.frame for gt in rally.gt_labels]
        features_list, candidate_frames = extract_candidate_features(
            rally, config=contact_config, gt_frames=gt_frames,
            sequence_probs=seq_probs_cache.get(rally.rally_id),
        )

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
    train_metrics = classifier.train(x_mat, y, positive_weight=args.positive_weight)
    console.print("\n[bold]Train-on-all:[/bold]")
    console.print(f"  F1: {train_metrics['train_f1']:.1%}")
    console.print(f"  Precision: {train_metrics['train_precision']:.1%}")
    console.print(f"  Recall: {train_metrics['train_recall']:.1%}")
    console.print(f"  TP: {train_metrics['train_tp']}, FP: {train_metrics['train_fp']}, FN: {train_metrics['train_fn']}")

    # LOO CV — compute probabilities once, then sweep thresholds on cached probas
    from sklearn.ensemble import GradientBoostingClassifier as _Gbc

    unique_rallies = np.unique(rally_ids)
    loo_probas = np.zeros(len(y))

    console.print(f"\n[bold]LOO CV: training {len(unique_rallies)} folds...[/bold]")
    for i, rally in enumerate(unique_rallies):
        test_mask = rally_ids == rally
        train_mask = ~test_mask

        if np.sum(train_mask) < 10 or np.sum(y[train_mask]) < 3:
            loo_probas[test_mask] = 0.5
            continue

        model = _Gbc(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=5, subsample=0.8, random_state=42,
        )
        train_weights = np.where(y[train_mask] == 1, args.positive_weight, 1.0)
        model.fit(x_mat[train_mask], y[train_mask], sample_weight=train_weights)
        loo_probas[test_mask] = model.predict_proba(x_mat[test_mask])[:, 1]

        if (i + 1) % 50 == 0:
            console.print(f"  [{i + 1}/{len(unique_rallies)}] folds complete")

    # Report LOO CV at requested threshold
    loo_preds = (loo_probas >= args.threshold).astype(int)
    tp = int(np.sum((loo_preds == 1) & (y == 1)))
    fp = int(np.sum((loo_preds == 1) & (y == 0)))
    fn = int(np.sum((loo_preds == 0) & (y == 1)))
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 2 * prec * rec / max(1e-9, prec + rec)

    console.print(f"\n[bold]Leave-One-Rally-Out CV ({len(unique_rallies)} folds):[/bold]")
    console.print(f"  F1: {f1:.1%}")
    console.print(f"  Precision: {prec:.1%}")
    console.print(f"  Recall: {rec:.1%}")
    console.print(f"  TP: {tp}, FP: {fp}, FN: {fn}")

    # Feature importance
    importance = classifier.feature_importance()
    if importance:
        imp_table = Table(title="\nFeature Importance")
        imp_table.add_column("Feature")
        imp_table.add_column("Importance", justify="right")

        for name, imp in sorted(importance.items(), key=lambda x: -x[1]):
            imp_table.add_row(name, f"{imp:.3f}")

        console.print(imp_table)

    # Threshold sweep on cached LOO probabilities (instant — no retraining)
    console.print("\n[bold]Threshold sweep (LOO CV, cached probas):[/bold]")
    sweep_table = Table()
    sweep_table.add_column("Threshold", justify="right")
    sweep_table.add_column("F1", justify="right")
    sweep_table.add_column("Precision", justify="right")
    sweep_table.add_column("Recall", justify="right")

    best_f1 = 0.0
    best_threshold = args.threshold

    for t in [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]:
        preds_t = (loo_probas >= t).astype(int)
        tp_t = int(np.sum((preds_t == 1) & (y == 1)))
        fp_t = int(np.sum((preds_t == 1) & (y == 0)))
        fn_t = int(np.sum((preds_t == 0) & (y == 1)))
        prec_t = tp_t / max(1, tp_t + fp_t)
        rec_t = tp_t / max(1, tp_t + fn_t)
        f1_t = 2 * prec_t * rec_t / max(1e-9, prec_t + rec_t)
        sweep_table.add_row(
            f"{t:.2f}",
            f"{f1_t:.1%}",
            f"{prec_t:.1%}",
            f"{rec_t:.1%}",
        )
        if f1_t > best_f1:
            best_f1 = f1_t
            best_threshold = t

    console.print(sweep_table)
    console.print(f"\n[bold]Best LOO threshold: {best_threshold:.2f} (LOO F1: {best_f1:.1%})[/bold]")

    # Use end-to-end optimal threshold (0.35) rather than LOO optimal.
    # LOO measures classifier accuracy on candidates; end-to-end measures
    # full pipeline F1 including candidate generation + action classification.
    # Run eval_action_detection.py --sweep-thresholds to re-validate.
    save_threshold = args.threshold
    console.print(f"  Saving with threshold={save_threshold:.2f} (end-to-end optimal; use --threshold to override)")

    classifier = ContactClassifier(threshold=save_threshold)
    classifier.train(x_mat, y, positive_weight=args.positive_weight)
    classifier.save(args.output)
    console.print(f"\n[green]Saved classifier to {args.output}[/green]")


if __name__ == "__main__":
    main()
