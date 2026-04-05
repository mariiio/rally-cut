"""Sequence-level action classifier using MS-TCN++ on trajectory features.

Bypasses contact detection entirely: feeds raw ball + player trajectories
as a per-frame time series into MS-TCN++ and predicts action frames
directly. The model implicitly learns possession structure from ball
trajectory without explicit contact detection or side counting.

Features: 19-dim per frame (ball position/velocity/confidence,
4 sorted players, ball-player distances, ball-relative-to-net).

Evaluation: leave-one-video-out CV to match existing GBM methodology.
Baseline to beat: 83.1% LOO-CV action accuracy (per-contact GBM).

Usage:
    cd analysis
    uv run python scripts/train_sequence_action.py
    uv run python scripts/train_sequence_action.py --sanity-check
    uv run python scripts/train_sequence_action.py --hidden-dim 128 --num-layers 10
    uv run python scripts/train_sequence_action.py --overfit-test
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from torch.utils.data import DataLoader

# Add scripts dir for imports from eval_action_detection
sys.path.insert(0, str(Path(__file__).parent))

from eval_action_detection import (  # noqa: E402
    GtLabel,
    MatchResult,
    compute_metrics,
    load_rallies_with_action_gt,
    match_contacts,
)

from rallycut.actions.sequence_dataset import (
    SequenceActionDataset,
    collate_rally_sequences,
)
from rallycut.actions.trajectory_features import (
    ACTION_TO_IDX,
    ACTION_TYPES,
    FEATURE_DIM,
    NUM_CLASSES,
    build_frame_labels,
    extract_trajectory_features,
)
from rallycut.temporal.ms_tcn.model import MSTCN, MSTCNConfig
from rallycut.temporal.temporal_maxer.training import FocalLoss, compute_tmse_loss
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.player_tracker import PlayerPosition

console = Console()

IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _parse_ball(raw: list[dict]) -> list[BallPosition]:
    return [
        BallPosition(
            frame_number=bp["frameNumber"],
            x=bp["x"],
            y=bp["y"],
            confidence=bp.get("confidence", 1.0),
        )
        for bp in raw
        if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
    ]


def _parse_players(raw: list[dict]) -> list[PlayerPosition]:
    return [
        PlayerPosition(
            frame_number=pp["frameNumber"],
            track_id=pp["trackId"],
            x=pp["x"],
            y=pp["y"],
            width=pp.get("width", 0.05),
            height=pp.get("height", 0.10),
            confidence=pp.get("confidence", 1.0),
        )
        for pp in raw
    ]


@dataclass
class RallySequenceData:
    """A rally with extracted trajectory features and labels."""

    rally_id: str
    video_id: str
    features: np.ndarray  # (num_frames, FEATURE_DIM)
    labels: np.ndarray  # (num_frames,) int64
    gt_labels: list[GtLabel]
    frame_count: int


def _load_team_assignments_and_calibrations(
    rallies: list,
) -> tuple[dict[str, dict[int, int]], dict[str, np.ndarray]]:
    """Load match-level team assignments and court homographies for all rallies.

    Returns:
        team_assignments_by_rally: rally_id → {track_id: team (0/1)}
        homographies_by_video: video_id → 3×3 homography matrix
    """
    from rallycut.court.calibration import CourtCalibrator
    from rallycut.evaluation.tracking.db import load_court_calibration
    from rallycut.tracking.match_tracker import build_match_team_assignments
    from rallycut.tracking.player_tracker import PlayerPosition

    video_ids = {r.video_id for r in rallies}

    # --- Team assignments ---
    rally_pos_lookup: dict[str, list[PlayerPosition]] = {}
    for r in rallies:
        if r.positions_json:
            rally_pos_lookup[r.rally_id] = [
                PlayerPosition(
                    frame_number=pp["frameNumber"], track_id=pp["trackId"],
                    x=pp["x"], y=pp["y"],
                    width=pp.get("width", 0.05), height=pp.get("height", 0.10),
                    confidence=pp.get("confidence", 1.0),
                )
                for pp in r.positions_json
            ]

    # Load from DB
    from rallycut.evaluation.db import get_connection
    team_assignments_by_rally: dict[str, dict[int, int]] = {}
    placeholders = ", ".join(["%s"] * len(video_ids))
    query = f"""
        SELECT id, match_analysis_json
        FROM videos
        WHERE id IN ({placeholders})
          AND match_analysis_json IS NOT NULL
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, list(video_ids))
            rows = cur.fetchall()
    for _vid, ma_json in rows:
        if isinstance(ma_json, dict):
            team_assignments_by_rally.update(
                build_match_team_assignments(
                    ma_json, min_confidence=0.70,
                    rally_positions=rally_pos_lookup,
                )
            )

    # --- Court homographies ---
    homographies_by_video: dict[str, np.ndarray] = {}
    for vid in video_ids:
        corners = load_court_calibration(vid)
        if corners and len(corners) == 4:
            cal = CourtCalibrator()
            cal.calibrate([(c["x"], c["y"]) for c in corners])
            if cal.is_calibrated and cal.homography is not None:
                homographies_by_video[vid] = cal.homography.homography

    return team_assignments_by_rally, homographies_by_video


def load_sequences(
    label_spread: int = 2,
) -> list[RallySequenceData]:
    """Load rallies from DB and extract trajectory features."""
    console.print("[bold]Loading rallies with action GT...[/bold]")
    rallies = load_rallies_with_action_gt()
    console.print(f"  {len(rallies)} rallies loaded")

    # Load team assignments and court calibrations for enhanced features
    team_by_rally, homographies = _load_team_assignments_and_calibrations(rallies)
    n_teams = sum(1 for r in rallies if r.rally_id in team_by_rally)
    n_courts = sum(1 for v in {r.video_id for r in rallies} if v in homographies)
    console.print(
        f"  Enhanced features: {n_teams}/{len(rallies)} rallies with teams, "
        f"{n_courts}/{len({r.video_id for r in rallies})} videos with court calibration"
    )

    sequences: list[RallySequenceData] = []
    skipped = 0

    for i, rally in enumerate(rallies):
        if not rally.ball_positions_json or not rally.positions_json:
            skipped += 1
            continue

        frame_count = rally.frame_count
        if not frame_count or frame_count < 10:
            skipped += 1
            continue

        ball_positions = _parse_ball(rally.ball_positions_json)
        player_positions = _parse_players(rally.positions_json)

        if not ball_positions:
            skipped += 1
            continue

        features = extract_trajectory_features(
            ball_positions=ball_positions,
            player_positions=player_positions,
            net_y=rally.court_split_y,
            frame_count=frame_count,
            team_assignments=team_by_rally.get(rally.rally_id),
            homography=homographies.get(rally.video_id),
        )

        # Build labels from GT
        gt_dicts = [
            {"frame": gt.frame, "action": gt.action}
            for gt in rally.gt_labels
        ]
        labels = build_frame_labels(gt_dicts, frame_count, label_spread=label_spread)

        sequences.append(RallySequenceData(
            rally_id=rally.rally_id,
            video_id=rally.video_id,
            features=features,
            labels=labels,
            gt_labels=rally.gt_labels,
            frame_count=frame_count,
        ))

        if (i + 1) % 50 == 0 or i == len(rallies) - 1:
            console.print(f"  [{i + 1}/{len(rallies)}] {len(sequences)} sequences loaded")

    if skipped:
        console.print(f"  [yellow]Skipped {skipped} rallies (missing data)[/yellow]")

    # Class distribution
    all_labels = np.concatenate([s.labels for s in sequences])
    counts = np.bincount(all_labels, minlength=NUM_CLASSES)
    dist = {
        "bg": int(counts[0]),
        **{ACTION_TYPES[i]: int(counts[i + 1]) for i in range(len(ACTION_TYPES))},
    }
    console.print(f"  Label distribution: {dist}")
    console.print(
        f"  Sequence lengths: min={min(s.frame_count for s in sequences)}, "
        f"max={max(s.frame_count for s in sequences)}, "
        f"mean={np.mean([s.frame_count for s in sequences]):.0f}"
    )

    return sequences


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_fold(
    train_seqs: list[RallySequenceData],
    val_seqs: list[RallySequenceData],
    args: argparse.Namespace,
) -> tuple[MSTCN, dict]:
    """Train MS-TCN++ on one fold and return best model + training info."""
    device = torch.device(args.device)

    config = MSTCNConfig(
        feature_dim=FEATURE_DIM,
        hidden_dim=args.hidden_dim,
        num_classes=NUM_CLASSES,
        num_stages=args.num_stages,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model = MSTCN(config).to(device)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Class weights from training labels
    all_labels = np.concatenate([s.labels for s in train_seqs])
    class_counts = np.bincount(all_labels, minlength=NUM_CLASSES).astype(float)
    class_counts = np.maximum(class_counts, 1.0)
    max_count = class_counts.max()
    weights = torch.tensor(
        [max_count / c for c in class_counts], dtype=torch.float32, device=device,
    )
    # Cap background weight to prevent it dominating
    weights[0] = weights[0].clamp(max=1.0)

    criterion = FocalLoss(weight=weights, gamma=args.focal_gamma, reduction="none")

    train_dataset = SequenceActionDataset(
        [s.features for s in train_seqs],
        [s.labels for s in train_seqs],
        augment=True,
        noise_std=args.noise_std,
    )
    val_dataset = SequenceActionDataset(
        [s.features for s in val_seqs],
        [s.labels for s in val_seqs],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_rally_sequences,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_rally_sequences,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_f1 = 0.0
    best_model_state = None
    best_epoch = 0
    patience_counter = 0

    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        train_loss = 0.0
        num_batches = 0

        for features, labels, mask in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()

            # Multi-stage loss
            stage_outputs = model.forward_all_stages(features, mask)
            total_loss = torch.tensor(0.0, device=device)

            for logits in stage_outputs:
                ce_loss = criterion(logits, labels)
                mask_sq = mask.squeeze(1)
                ce_loss = (ce_loss * mask_sq).sum() / mask_sq.sum()
                tmse_loss = compute_tmse_loss(logits, mask)
                total_loss = total_loss + ce_loss + args.tmse_weight * tmse_loss

            if torch.isnan(total_loss) or torch.isinf(total_loss):
                continue

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_loss += total_loss.item()
            num_batches += 1

        scheduler.step()

        # --- Validate ---
        model.eval()
        all_preds: list[int] = []
        all_targets: list[int] = []

        with torch.no_grad():
            for features, labels, mask in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                mask = mask.to(device)

                logits = model(features, mask)
                preds = logits.argmax(dim=1)  # (batch, T)

                # Collect non-background predictions and targets
                mask_flat = mask.squeeze(1).bool()
                for b in range(features.shape[0]):
                    valid = mask_flat[b]
                    p = preds[b][valid].cpu().numpy()
                    t = labels[b][valid].cpu().numpy()
                    # Only consider action frames (non-background)
                    action_mask = t > 0
                    if action_mask.any():
                        all_preds.extend(p[action_mask].tolist())
                        all_targets.extend(t[action_mask].tolist())

        # Action accuracy on GT action frames
        if all_targets:
            correct = sum(1 for p, t in zip(all_preds, all_targets) if p == t)
            val_acc = correct / len(all_targets)
            # Simple F1: fraction of action frames correctly classified
            val_f1 = val_acc  # Use accuracy as selection metric
        else:
            val_acc = 0.0
            val_f1 = 0.0

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            break

    # Restore best model
    if best_model_state:
        model.load_state_dict(best_model_state)
        model = model.to(device)

    info = {
        "best_epoch": best_epoch,
        "best_val_f1": best_val_f1,
        "num_params": num_params,
        "train_size": len(train_seqs),
        "val_size": len(val_seqs),
    }
    return model, info


# ---------------------------------------------------------------------------
# Post-processing: frame probabilities → discrete action predictions
# ---------------------------------------------------------------------------


def predict_actions(
    model: MSTCN,
    sequence: RallySequenceData,
    device: torch.device,
    peak_threshold: float = 0.3,
    min_peak_distance: int = 12,
    smoothing_sigma: float = 2.0,
) -> list[dict]:
    """Run model inference and extract discrete action predictions.

    Returns list of dicts compatible with match_contacts():
        [{"frame": int, "action": str, "playerTrackId": -1}, ...]
    """
    model.eval()
    features = torch.from_numpy(sequence.features).float().unsqueeze(0)  # (1, T, F)
    features = features.transpose(1, 2).to(device)  # (1, F, T)
    mask = torch.ones(1, 1, sequence.frame_count, device=device)

    with torch.no_grad():
        logits = model(features, mask)  # (1, C, T)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()  # (C, T)

    predictions: list[dict] = []

    # For each action class, find peaks in probability
    for cls_idx in range(1, NUM_CLASSES):  # skip background
        cls_probs = probs[cls_idx]

        # Smooth probabilities
        if smoothing_sigma > 0:
            cls_probs = gaussian_filter1d(cls_probs, sigma=smoothing_sigma)

        # Find peaks
        peaks, _ = find_peaks(
            cls_probs,
            height=peak_threshold,
            distance=min_peak_distance,
        )

        for peak_frame in peaks:
            predictions.append({
                "frame": int(peak_frame),
                "action": IDX_TO_ACTION[cls_idx],
                "playerTrackId": -1,
                "confidence": float(cls_probs[peak_frame]),
            })

    # Sort by frame
    predictions.sort(key=lambda p: p["frame"])

    # NMS: if two predictions are within min_peak_distance, keep higher confidence
    if predictions:
        filtered: list[dict] = [predictions[0]]
        for pred in predictions[1:]:
            if pred["frame"] - filtered[-1]["frame"] < min_peak_distance:
                if pred["confidence"] > filtered[-1]["confidence"]:
                    filtered[-1] = pred
            else:
                filtered.append(pred)
        predictions = filtered

    return predictions


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_fold(
    model: MSTCN,
    test_seqs: list[RallySequenceData],
    device: torch.device,
    peak_threshold: float = 0.3,
    tolerance: int = 5,
) -> tuple[list[MatchResult], list[dict]]:
    """Evaluate model on test sequences, returning matches and unmatched preds."""
    all_matches: list[MatchResult] = []
    all_unmatched: list[dict] = []

    for seq in test_seqs:
        preds = predict_actions(model, seq, device, peak_threshold=peak_threshold)
        matches, unmatched = match_contacts(seq.gt_labels, preds, tolerance=tolerance)
        all_matches.extend(matches)
        all_unmatched.extend(unmatched)

    return all_matches, all_unmatched


# ---------------------------------------------------------------------------
# Leave-one-video-out CV
# ---------------------------------------------------------------------------


def run_loo_cv(
    sequences: list[RallySequenceData],
    args: argparse.Namespace,
) -> None:
    """Run leave-one-video-out cross-validation."""
    # Group by video
    videos: dict[str, list[RallySequenceData]] = defaultdict(list)
    for seq in sequences:
        videos[seq.video_id].append(seq)

    video_ids = sorted(videos.keys())
    console.print(f"\n[bold]Leave-one-video-out CV: {len(video_ids)} videos[/bold]")

    all_matches: list[MatchResult] = []
    all_unmatched: list[dict] = []
    fold_results: list[dict] = []

    device = torch.device(args.device)
    start_time = time.time()

    for fold_idx, held_out_vid in enumerate(video_ids):
        test_seqs = videos[held_out_vid]
        train_seqs = [s for vid, seqs in videos.items() if vid != held_out_vid for s in seqs]

        if not test_seqs or not train_seqs:
            continue

        fold_start = time.time()
        model, info = train_fold(train_seqs, test_seqs, args)

        fold_matches, fold_unmatched = evaluate_fold(
            model, test_seqs, device, peak_threshold=args.peak_threshold,
        )

        # Per-fold metrics
        fold_metrics = compute_metrics(fold_matches, fold_unmatched)
        fold_time = time.time() - fold_start

        n_gt = len(fold_matches)
        n_tp = fold_metrics["tp"]
        action_acc = fold_metrics["action_accuracy"]

        console.print(
            f"  [{fold_idx + 1}/{len(video_ids)}] video={held_out_vid[:8]}… "
            f"rallies={len(test_seqs)} GT={n_gt} TP={n_tp} "
            f"action_acc={action_acc:.1%} "
            f"epoch={info['best_epoch']} "
            f"time={fold_time:.0f}s"
        )

        all_matches.extend(fold_matches)
        all_unmatched.extend(fold_unmatched)
        fold_results.append({
            "video_id": held_out_vid,
            "metrics": fold_metrics,
            "info": info,
        })

    total_time = time.time() - start_time

    # --- Aggregate results ---
    console.print(f"\n[bold]Aggregate LOO-CV Results ({total_time:.0f}s total)[/bold]")
    overall = compute_metrics(all_matches, all_unmatched)

    # Summary table
    summary = Table(title="Contact Detection + Action Classification")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")

    summary.add_row("GT contacts", str(overall["total_gt"]))
    summary.add_row("Predicted", str(overall["total_pred"]))
    summary.add_row("TP", str(overall["tp"]))
    summary.add_row("FP", str(overall["fp"]))
    summary.add_row("FN", str(overall["fn"]))
    summary.add_row("Contact Recall", f"{overall['recall']:.1%}")
    summary.add_row("Contact Precision", f"{overall['precision']:.1%}")
    summary.add_row("Contact F1", f"{overall['f1']:.1%}")
    summary.add_row("[bold]Action Accuracy[/bold]", f"[bold]{overall['action_accuracy']:.1%}[/bold]")

    console.print(summary)

    # Per-class table
    if "per_class" in overall:
        cls_table = Table(title="Per-Class Metrics (LOO-CV)")
        cls_table.add_column("Action")
        cls_table.add_column("TP", justify="right")
        cls_table.add_column("FP", justify="right")
        cls_table.add_column("FN", justify="right")
        cls_table.add_column("Precision", justify="right")
        cls_table.add_column("Recall", justify="right")
        cls_table.add_column("F1", justify="right")

        for action in ACTION_TYPES:
            if action in overall["per_class"]:
                c = overall["per_class"][action]
                cls_table.add_row(
                    action,
                    str(c["tp"]),
                    str(c["fp"]),
                    str(c["fn"]),
                    f"{c['precision']:.1%}",
                    f"{c['recall']:.1%}",
                    f"{c['f1']:.1%}",
                )

        console.print(cls_table)

    # Comparison with baseline
    console.print("\n[bold]Comparison with GBM Baseline[/bold]")
    baseline_table = Table()
    baseline_table.add_column("Model")
    baseline_table.add_column("Action Acc", justify="right")
    baseline_table.add_column("Contact F1", justify="right")
    baseline_table.add_row(
        "GBM (per-contact)",
        "83.1%",
        "90.0%",
    )
    baseline_table.add_row(
        f"Seq MS-TCN++ (h={args.hidden_dim}, s={args.num_stages}, L={args.num_layers})",
        f"{overall['action_accuracy']:.1%}",
        f"{overall['f1']:.1%}",
    )
    console.print(baseline_table)

    # Confusion matrix
    print_confusion_matrix(all_matches)



def print_confusion_matrix(matches: list[MatchResult]) -> None:
    """Print confusion matrix from matched predictions."""
    matched = [m for m in matches if m.pred_frame is not None]
    if not matched:
        return

    action_list = ACTION_TYPES
    cm: dict[str, dict[str, int]] = {a: {b: 0 for b in action_list} for a in action_list}

    for m in matched:
        gt_a = m.gt_action
        pred_a = m.pred_action or ""
        if gt_a in cm and pred_a in cm[gt_a]:
            cm[gt_a][pred_a] += 1

    table = Table(title="Confusion Matrix (GT rows × Pred cols)")
    table.add_column("GT \\ Pred")
    for a in action_list:
        table.add_column(a[:3], justify="right")
    table.add_column("N", justify="right")

    for gt_action in action_list:
        row_total = sum(cm[gt_action].values())
        cells = []
        for pred_action in action_list:
            count = cm[gt_action][pred_action]
            if count > 0 and gt_action == pred_action:
                cells.append(f"[green]{count}[/green]")
            elif count > 0:
                cells.append(f"[red]{count}[/red]")
            else:
                cells.append("·")
        table.add_row(gt_action, *cells, str(row_total))

    console.print(table)


# ---------------------------------------------------------------------------
# Sanity check mode
# ---------------------------------------------------------------------------


def run_sanity_check(sequences: list[RallySequenceData]) -> None:
    """Quick sanity check on a few rallies."""
    console.print("[bold]Sanity check: feature shapes and labels[/bold]")

    for seq in sequences[:5]:
        n_actions = (seq.labels > 0).sum()
        n_frames = seq.frame_count
        ball_coverage = (seq.features[:, 2] > 0).mean()  # ball confidence > 0

        console.print(
            f"  rally={seq.rally_id[:8]}… frames={n_frames} "
            f"actions={n_actions} ball_coverage={ball_coverage:.1%} "
            f"features_shape={seq.features.shape}"
        )

        # Per-class counts
        counts = np.bincount(seq.labels, minlength=NUM_CLASSES)
        label_str = ", ".join(
            f"{ACTION_TYPES[i]}={counts[i+1]}" for i in range(len(ACTION_TYPES)) if counts[i+1] > 0
        )
        console.print(f"    labels: bg={counts[0]}, {label_str}")

    # Feature stats
    all_features = np.concatenate([s.features for s in sequences[:5]])
    console.print("\n  Feature stats (first 5 rallies):")
    feature_names = [
        "ball_x", "ball_y", "ball_conf", "ball_dx", "ball_dy", "ball_speed",
        "p0_x", "p0_y", "p1_x", "p1_y", "p2_x", "p2_y", "p3_x", "p3_y",
        "dist_p0", "dist_p1", "dist_p2", "dist_p3", "ball_y_rel_net",
        "court_ball_x", "court_ball_y",
        "ball_det_density",
        "p0_team", "p1_team", "p2_team", "p3_team",
    ]
    for i, name in enumerate(feature_names):
        vals = all_features[:, i]
        console.print(
            f"    [{i:2d}] {name:15s}  "
            f"mean={vals.mean():.4f}  std={vals.std():.4f}  "
            f"min={vals.min():.4f}  max={vals.max():.4f}"
        )


# ---------------------------------------------------------------------------
# Overfit test mode
# ---------------------------------------------------------------------------


def run_overfit_test(sequences: list[RallySequenceData], args: argparse.Namespace) -> None:
    """Train on all data and test on same data — verify model can memorize."""
    console.print("[bold]Overfit test: training on all data[/bold]")

    model, info = train_fold(sequences, sequences, args)
    device = torch.device(args.device)

    matches, unmatched = evaluate_fold(
        model, sequences, device, peak_threshold=args.peak_threshold,
    )
    metrics = compute_metrics(matches, unmatched)

    console.print(f"  Best epoch: {info['best_epoch']}")
    console.print(f"  Params: {info['num_params']:,}")
    console.print(f"  Contact F1: {metrics['f1']:.1%}")
    console.print(f"  Action accuracy: {metrics['action_accuracy']:.1%}")
    console.print("  (Should be >90% if model can memorize)")

    print_confusion_matrix(matches)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequence action classifier (MS-TCN++)")

    # Model
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)

    # Training
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--tmse-weight", type=float, default=0.15)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--label-spread", type=int, default=2)

    # Post-processing
    parser.add_argument("--peak-threshold", type=float, default=0.3)

    # Modes
    parser.add_argument("--sanity-check", action="store_true", help="Quick feature sanity check")
    parser.add_argument("--overfit-test", action="store_true", help="Train+test on all data")
    parser.add_argument(
        "--device", type=str,
        default="mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load data
    sequences = load_sequences(label_spread=args.label_spread)

    if not sequences:
        console.print("[red]No sequences loaded. Check database connection.[/red]")
        return

    if args.sanity_check:
        run_sanity_check(sequences)
        return

    if args.overfit_test:
        run_overfit_test(sequences, args)
        return

    run_loo_cv(sequences, args)


if __name__ == "__main__":
    main()
