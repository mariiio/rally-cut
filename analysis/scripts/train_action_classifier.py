"""Train an action type classifier on labeled rally data.

Loads rallies with action ground truth, re-runs contact detection to get
validated contacts, matches them against GT, extracts ActionFeatures,
and trains a multiclass GradientBoostingClassifier.

Reports both train-on-all and leave-one-rally-out CV metrics.

Usage:
    cd analysis
    uv run python scripts/train_action_classifier.py
    uv run python scripts/train_action_classifier.py --output weights/action_classifier/action_classifier.pkl
"""

from __future__ import annotations

import argparse
import copy
from collections import defaultdict
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

from rallycut.evaluation.split import add_split_argument, apply_split
from rallycut.tracking.action_type_classifier import (
    ActionTypeClassifier,
    extract_action_features,
    set_prev_action_context,
)
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.contact_detector import (
    ContactDetectionConfig,
    detect_contacts,
)
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos
from scripts.eval_action_detection import (
    RallyData,
    _load_match_team_assignments,
    load_rallies_with_action_gt,
    match_contacts,
)

console = Console()


def extract_features_for_rally(
    rally: RallyData,
    config: ContactDetectionConfig | None = None,
    tolerance: int = 5,
    team_assignments: dict[int, int] | None = None,
    inject_pose: bool = False,
    calibrator: Any = None,
    camera_height: float = 0.0,
) -> tuple[list[np.ndarray], list[str], list[str]]:
    """Extract action features for matched contacts in a rally.

    Args:
        inject_pose: If True, inject keypoints from pose cache into
            PlayerPosition objects so pose features are populated.
        calibrator: Optional CourtCalibrator for court-space projections.
        camera_height: Camera height in metres (0.0 = unknown).

    Returns:
        Tuple of (feature_arrays, action_labels, rally_ids).
    """
    if not rally.ball_positions_json:
        return [], [], []

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
        return [], [], []

    # Build pose keypoint lookup from cache if requested
    pose_kps: dict[tuple[int, int], list[list[float]]] = {}
    if inject_pose:
        from rallycut.tracking.pose_attribution.pose_cache import load_pose_cache

        pose_data = load_pose_cache(rally.rally_id)
        if pose_data is not None and len(pose_data["frames"]) > 0:
            for i in range(len(pose_data["frames"])):
                key = (int(pose_data["frames"][i]), int(pose_data["track_ids"][i]))
                pose_kps[key] = pose_data["keypoints"][i].tolist()

    player_positions: list[PlayerPos] = []
    if rally.positions_json:
        for pp in rally.positions_json:
            kps = pose_kps.get((pp["frameNumber"], pp["trackId"])) if pose_kps else None
            player_positions.append(PlayerPos(
                frame_number=pp["frameNumber"],
                track_id=pp["trackId"],
                x=pp["x"],
                y=pp["y"],
                width=pp["width"],
                height=pp["height"],
                confidence=pp.get("confidence", 1.0),
                keypoints=kps,
            ))

    # Re-run contact detection
    contact_seq = detect_contacts(
        ball_positions=ball_positions,
        player_positions=player_positions,
        config=config,
        net_y=rally.court_split_y,
        frame_count=rally.frame_count or None,
    )

    if not contact_seq.contacts:
        return [], [], []

    # Match detected contacts to GT
    pred_actions = [
        {"frame": c.frame, "action": "unknown", "playerTrackId": c.player_track_id}
        for c in contact_seq.contacts
    ]
    matches, _ = match_contacts(rally.gt_labels, pred_actions, tolerance=tolerance)

    # Build contact index lookup: frame -> index in contacts list
    frame_to_idx: dict[int, int] = {}
    for idx, c in enumerate(contact_seq.contacts):
        frame_to_idx[c.frame] = idx

    features_list: list[np.ndarray] = []
    labels: list[str] = []
    rally_ids: list[str] = []

    # Build ordered list of matched GT for prev-action context
    matched_gt: list[tuple[str, int, str]] = []  # (gt_action, pred_frame, court_side)
    for m in matches:
        if m.pred_frame is None:
            continue
        if m.gt_action not in ("serve", "receive", "set", "attack", "dig", "block"):
            continue
        ci = frame_to_idx.get(m.pred_frame)
        cs = contact_seq.contacts[ci].court_side if ci is not None else ""
        matched_gt.append((m.gt_action, m.pred_frame, cs))

    prev_action_str = "unknown"
    prev_court_side = ""
    for gt_action, pred_frame, court_side in matched_gt:
        if gt_action == "block":
            prev_action_str = gt_action
            prev_court_side = court_side
            continue

        contact_idx = frame_to_idx.get(pred_frame)
        if contact_idx is None:
            prev_action_str = gt_action
            prev_court_side = court_side
            continue

        contact = contact_seq.contacts[contact_idx]
        feat = extract_action_features(
            contact=contact,
            index=contact_idx,
            all_contacts=contact_seq.contacts,
            ball_positions=contact_seq.ball_positions or None,
            net_y=contact_seq.net_y,
            rally_start_frame=contact_seq.rally_start_frame,
            team_assignments=team_assignments,
            player_positions=player_positions or None,
            calibrator=calibrator,
            camera_height=camera_height,
        )

        # Sample 1: no prev-action context (simulates first pass)
        features_list.append(feat.to_array())
        labels.append(gt_action)
        rally_ids.append(rally.rally_id)

        # Sample 2: with GT prev-action context (simulates second pass)
        feat2 = copy.copy(feat)
        same_side: bool | None = None
        if prev_court_side and court_side in ("near", "far"):
            same_side = prev_court_side == court_side
        set_prev_action_context(feat2, prev_action_str, 0.7, same_side)
        features_list.append(feat2.to_array())
        labels.append(gt_action)
        rally_ids.append(rally.rally_id)

        prev_action_str = gt_action
        prev_court_side = court_side

    return features_list, labels, rally_ids


def _build_calibrators_and_heights(
    video_ids: set[str],
) -> tuple[dict[str, Any], dict[str, float]]:
    """Build per-video CourtCalibrator and camera height from DB calibration.

    Returns (calibrators, camera_heights) dicts keyed by video_id.
    """
    from rallycut.court.calibration import CourtCalibrator  # noqa: PLC0415
    from rallycut.court.camera_model import calibrate_camera  # noqa: PLC0415
    from rallycut.evaluation.tracking.db import (  # noqa: PLC0415
        get_connection,
        load_court_calibration,
    )

    calibrators: dict[str, Any] = {}
    heights: dict[str, float] = {}

    # Query video resolutions for camera model
    resolutions: dict[str, tuple[int, int]] = {}
    if video_ids:
        placeholders = ", ".join(["%s"] * len(video_ids))
        with get_connection() as conn, conn.cursor() as cur:
            cur.execute(
                f"SELECT id, width, height FROM videos WHERE id IN ({placeholders})",
                list(video_ids),
            )
            for vid, w, h in cur.fetchall():
                if w and h:
                    resolutions[vid] = (int(w), int(h))

    for vid in video_ids:
        corners = load_court_calibration(vid)
        if not corners or len(corners) != 4:
            continue
        image_corners = [(c["x"], c["y"]) for c in corners]
        cal = CourtCalibrator()
        cal.calibrate(image_corners)
        if not cal.is_calibrated or cal.homography is None:
            continue
        calibrators[vid] = cal
        # Camera height via pinhole model
        res = resolutions.get(vid)
        if res is not None:
            cam = calibrate_camera(
                image_corners,
                cal.homography.court_corners,
                res[0], res[1],
            )
            if cam is not None and cam.is_valid:
                heights[vid] = float(cam.camera_position[2])

    return calibrators, heights


def main() -> None:
    parser = argparse.ArgumentParser(description="Train action type classifier")
    parser.add_argument(
        "--output",
        type=str,
        default="weights/action_classifier/action_classifier.pkl",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=5,
        help="Frame tolerance for GT matching",
    )
    parser.add_argument(
        "--exclude-videos",
        type=str,
        help="Comma-separated video ID prefixes to exclude from training (held-out test set)",
    )
    parser.add_argument(
        "--pose",
        action="store_true",
        help="Inject YOLO-Pose keypoints from pose cache for pose-aware features",
    )
    add_split_argument(parser)
    args = parser.parse_args()

    rallies = load_rallies_with_action_gt()
    rallies = apply_split(rallies, args)
    if args.exclude_videos:
        prefixes = [p.strip() for p in args.exclude_videos.split(",")]
        before = len(rallies)
        rallies = [r for r in rallies if not any(r.video_id.startswith(p) for p in prefixes)]
        console.print(f"  Excluded {before - len(rallies)} rallies from {len(prefixes)} video(s)")
    if not rallies:
        console.print("[red]No rallies with action GT found.[/red]")
        return

    # Load match-level team assignments for team-based features
    video_ids = {r.video_id for r in rallies}
    rally_pos_lookup: dict[str, list] = {}
    for r in rallies:
        if r.positions_json:
            rally_pos_lookup[r.rally_id] = [
                PlayerPos(
                    frame_number=pp["frameNumber"], track_id=pp["trackId"],
                    x=pp["x"], y=pp["y"],
                    width=pp["width"], height=pp["height"],
                    confidence=pp.get("confidence", 1.0),
                )
                for pp in r.positions_json
            ]
    match_teams_by_rally = _load_match_team_assignments(
        video_ids, min_confidence=0.70, rally_positions=rally_pos_lookup,
    )
    n_with_teams = sum(1 for r in rallies if r.rally_id in match_teams_by_rally)
    console.print(f"  Match teams: {n_with_teams}/{len(rallies)} rallies")

    # Build court calibrators + camera heights per video
    calibrators, camera_heights = _build_calibrators_and_heights(video_ids)
    console.print(
        f"  Court calibration: {len(calibrators)}/{len(video_ids)} videos, "
        f"camera height: {sum(1 for h in camera_heights.values() if h > 0)}/{len(video_ids)} videos"
    )

    console.print(
        f"\n[bold]Extracting action features from {len(rallies)} rallies[/bold]\n"
    )

    all_features: list[np.ndarray] = []
    all_labels: list[str] = []
    all_rally_ids: list[str] = []

    per_rally_table = Table(title="Per-Rally Feature Extraction")
    per_rally_table.add_column("Rally", max_width=10)
    per_rally_table.add_column("Matched", justify="right")
    per_rally_table.add_column("GT Labels", justify="right")
    per_rally_table.add_column("Actions", justify="right")

    use_pose = getattr(args, "pose", False)
    if use_pose:
        console.print("  [bold]Pose features enabled[/bold] (injecting keypoints from cache)")

    for rally in rallies:
        features, labels, rids = extract_features_for_rally(
            rally, tolerance=args.tolerance,
            team_assignments=match_teams_by_rally.get(rally.rally_id),
            inject_pose=use_pose,
            calibrator=calibrators.get(rally.video_id),
            camera_height=camera_heights.get(rally.video_id, 0.0),
        )

        all_features.extend(features)
        all_labels.extend(labels)
        all_rally_ids.extend(rids)

        # Count actions for display
        action_counts: dict[str, int] = defaultdict(int)
        for lbl in labels:
            action_counts[lbl] += 1
        action_str = ", ".join(
            f"{k}:{v}" for k, v in sorted(action_counts.items())
        )

        per_rally_table.add_row(
            rally.rally_id[:8],
            str(len(features)),
            str(len(rally.gt_labels)),
            action_str or "-",
        )

    console.print(per_rally_table)

    if not all_features:
        console.print("[red]No matched contacts extracted.[/red]")
        return

    x_mat = np.array(all_features)
    y = np.array(all_labels)
    rally_ids = np.array(all_rally_ids)

    # Class distribution
    console.print(f"\n[bold]Dataset: {len(y)} samples[/bold]")
    for cls in sorted(set(y)):
        console.print(f"  {cls}: {int(np.sum(y == cls))}")

    # Train on all data
    classifier = ActionTypeClassifier()
    train_metrics = classifier.train(x_mat, y)

    console.print(f"\n[bold]Train-on-all accuracy: {train_metrics['train_accuracy']:.1%}[/bold]")
    for cls, info in sorted(train_metrics["per_class"].items()):
        console.print(f"  {cls}: {info['accuracy']:.1%} ({info['count']} samples)")

    # Save model (before LOO-CV which can be slow)
    classifier.save(args.output)
    console.print(f"\n[green]Saved action type classifier to {args.output}[/green]")

    # LOO CV
    loo_metrics = classifier.loo_cv(x_mat, y, rally_ids)
    console.print(
        f"\n[bold]Leave-One-Rally-Out CV "
        f"({loo_metrics['n_rallies']} folds): "
        f"{loo_metrics['loo_accuracy']:.1%}[/bold]"
    )

    # Per-class LOO table
    loo_table = Table(title="LOO CV Per-Class Metrics")
    loo_table.add_column("Action", style="bold")
    loo_table.add_column("TP", justify="right")
    loo_table.add_column("FP", justify="right")
    loo_table.add_column("FN", justify="right")
    loo_table.add_column("P", justify="right")
    loo_table.add_column("R", justify="right")
    loo_table.add_column("F1", justify="right")
    loo_table.add_column("N", justify="right")

    for cls, info in sorted(loo_metrics["per_class"].items()):
        if info["count"] > 0:
            loo_table.add_row(
                cls,
                str(info["tp"]),
                str(info["fp"]),
                str(info["fn"]),
                f"{info['precision']:.1%}",
                f"{info['recall']:.1%}",
                f"{info['f1']:.1%}",
                str(info["count"]),
            )

    console.print(loo_table)

    # Confusion matrix
    confusion = loo_metrics.get("confusion", {})
    if confusion:
        classes = sorted(
            set(list(confusion.keys()) + [
                p for row in confusion.values() for p in row.keys()
            ])
        )
        cm_table = Table(title="\nLOO CV Confusion Matrix (rows=GT, cols=Predicted)")
        cm_table.add_column("GT \\ Pred", style="bold")
        for cls in classes:
            cm_table.add_column(cls[:5], justify="right")

        for gt_cls in classes:
            if gt_cls in confusion:
                cells = [str(confusion[gt_cls].get(p, 0)) for p in classes]
                cm_table.add_row(gt_cls, *cells)

        console.print(cm_table)

    # Feature importance
    importance = classifier.feature_importance()
    if importance:
        imp_table = Table(title="\nFeature Importance")
        imp_table.add_column("Feature")
        imp_table.add_column("Importance", justify="right")

        for name, imp in sorted(importance.items(), key=lambda x: -x[1]):
            imp_table.add_row(name, f"{imp:.3f}")

        console.print(imp_table)


if __name__ == "__main__":
    main()
