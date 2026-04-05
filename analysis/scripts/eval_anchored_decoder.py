"""Evaluate the contact-anchored sequence decoder.

Tests the full pipeline: MS-TCN++ sequence model + trajectory contact
proposals → merged decoder with constrained Viterbi → optional
team-based attribution.

Compares three modes:
  1. Trajectory-only: contacts from detect_contacts() (baseline)
  2. Sequence-only: peaks from MS-TCN++ contact score
  3. Anchored: merged proposals (trajectory + sequence)

Evaluation: leave-one-video-out CV.

Usage:
    cd analysis
    uv run python scripts/eval_anchored_decoder.py
    uv run python scripts/eval_anchored_decoder.py --no-viterbi
    uv run python scripts/eval_anchored_decoder.py --attribution
"""

from __future__ import annotations

import argparse
import logging
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from eval_action_detection import (  # noqa: E402
    GtLabel,
    MatchResult,
    RallyData,
    _load_match_team_assignments,
    compute_metrics,
    load_rallies_with_action_gt,
    match_contacts,
)

from rallycut.actions.contact_decoder import (
    DecoderConfig,
    attribute_with_team_hint,
    decode_actions,
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
from rallycut.tracking.contact_detector import (
    ContactDetectionConfig,
    ContactSequence,
    detect_contacts,
)
from rallycut.tracking.player_tracker import PlayerPosition

logging.getLogger("rallycut.tracking.contact_detector").setLevel(logging.WARNING)
logging.getLogger("rallycut.tracking.action_classifier").setLevel(logging.WARNING)

console = Console()

IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}


# ---------------------------------------------------------------------------
# Data loading
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
class RallyBundle:
    """All data needed to evaluate one rally."""

    rally: RallyData
    ball_positions: list[BallPosition]
    player_positions: list[PlayerPosition]
    contact_sequence: ContactSequence
    trajectory_features: np.ndarray  # (num_frames, FEATURE_DIM)
    trajectory_labels: np.ndarray  # (num_frames,) for training
    match_teams: dict[int, int] | None
    gt_labels: list[GtLabel]


def prepare_rallies(
    label_spread: int = 2,
) -> list[RallyBundle]:
    """Load rallies, run contact detection, extract trajectory features."""
    console.print("[bold]Loading rallies with action GT...[/bold]")
    rallies = load_rallies_with_action_gt()
    console.print(f"  {len(rallies)} rallies loaded")

    # Load team assignments and court calibrations
    video_ids = {r.video_id for r in rallies}
    rally_team_map = _load_match_team_assignments(video_ids)

    from rallycut.court.calibration import CourtCalibrator
    from rallycut.evaluation.tracking.db import load_court_calibration

    homographies: dict[str, np.ndarray] = {}
    for vid in video_ids:
        corners = load_court_calibration(vid)
        if corners and len(corners) == 4:
            cal = CourtCalibrator()
            cal.calibrate([(c["x"], c["y"]) for c in corners])
            if cal.is_calibrated and cal.homography is not None:
                homographies[vid] = cal.homography.homography

    bundles: list[RallyBundle] = []
    skipped = 0

    for i, rally in enumerate(rallies):
        if not rally.ball_positions_json or not rally.positions_json:
            skipped += 1
            continue
        if not rally.frame_count or rally.frame_count < 10:
            skipped += 1
            continue

        ball_positions = _parse_ball(rally.ball_positions_json)
        player_positions = _parse_players(rally.positions_json)
        if not ball_positions:
            skipped += 1
            continue

        # Run contact detection for trajectory proposals
        contact_seq = detect_contacts(
            ball_positions=ball_positions,
            player_positions=player_positions,
            config=ContactDetectionConfig(),
            net_y=rally.court_split_y,
            frame_count=rally.frame_count,
        )

        match_teams = rally_team_map.get(rally.rally_id)

        # Extract enhanced trajectory features
        features = extract_trajectory_features(
            ball_positions=ball_positions,
            player_positions=player_positions,
            net_y=rally.court_split_y,
            frame_count=rally.frame_count,
            team_assignments=match_teams,
            homography=homographies.get(rally.video_id),
        )

        gt_dicts = [{"frame": gt.frame, "action": gt.action} for gt in rally.gt_labels]
        labels = build_frame_labels(gt_dicts, rally.frame_count, label_spread=label_spread)

        bundles.append(RallyBundle(
            rally=rally,
            ball_positions=ball_positions,
            player_positions=player_positions,
            contact_sequence=contact_seq,
            trajectory_features=features,
            trajectory_labels=labels,
            match_teams=match_teams,
            gt_labels=rally.gt_labels,
        ))

        if (i + 1) % 50 == 0 or i == len(rallies) - 1:
            console.print(f"  [{i + 1}/{len(rallies)}] {len(bundles)} bundles prepared")

    if skipped:
        console.print(f"  [yellow]Skipped {skipped} rallies (missing data)[/yellow]")

    n_teams = sum(1 for b in bundles if b.match_teams)
    console.print(
        f"  Total: {len(bundles)} rallies, {n_teams} with team assignments"
    )
    return bundles


# ---------------------------------------------------------------------------
# Training (reused from train_sequence_action.py)
# ---------------------------------------------------------------------------


def train_sequence_model(
    train_bundles: list[RallyBundle],
    val_bundles: list[RallyBundle],
    args: argparse.Namespace,
) -> MSTCN:
    """Train MS-TCN++ sequence model on trajectory features."""
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

    # Class weights
    all_labels = np.concatenate([b.trajectory_labels for b in train_bundles])
    class_counts = np.bincount(all_labels, minlength=NUM_CLASSES).astype(float)
    class_counts = np.maximum(class_counts, 1.0)
    max_count = class_counts.max()
    weights = torch.tensor(
        [max_count / c for c in class_counts], dtype=torch.float32, device=device,
    )
    weights[0] = weights[0].clamp(max=1.0)

    criterion = FocalLoss(weight=weights, gamma=args.focal_gamma, reduction="none")

    train_dataset = SequenceActionDataset(
        [b.trajectory_features for b in train_bundles],
        [b.trajectory_labels for b in train_bundles],
        augment=True,
        noise_std=args.noise_std,
    )
    val_dataset = SequenceActionDataset(
        [b.trajectory_features for b in val_bundles],
        [b.trajectory_labels for b in val_bundles],
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_rally_sequences,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        collate_fn=collate_rally_sequences,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        for features, labels, mask in train_loader:
            features = features.to(device)
            labels = labels.to(device)
            mask = mask.to(device)

            optimizer.zero_grad()
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

        scheduler.step()

        # Validate: action accuracy on GT frames
        model.eval()
        all_preds: list[int] = []
        all_targets: list[int] = []
        with torch.no_grad():
            for features, labels, mask in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                mask = mask.to(device)
                logits = model(features, mask)
                preds = logits.argmax(dim=1)
                mask_flat = mask.squeeze(1).bool()
                for b in range(features.shape[0]):
                    valid = mask_flat[b]
                    p = preds[b][valid].cpu().numpy()
                    t = labels[b][valid].cpu().numpy()
                    action_mask = t > 0
                    if action_mask.any():
                        all_preds.extend(p[action_mask].tolist())
                        all_targets.extend(t[action_mask].tolist())

        val_acc = 0.0
        if all_targets:
            val_acc = sum(1 for p, t in zip(all_preds, all_targets) if p == t) / len(all_targets)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            break

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)

    return model


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------


def get_frame_probs(
    model: MSTCN,
    bundle: RallyBundle,
    device: torch.device,
) -> np.ndarray:
    """Run MS-TCN++ and return per-frame probabilities (NUM_CLASSES, T)."""
    model.eval()
    features = torch.from_numpy(bundle.trajectory_features).float().unsqueeze(0)
    features = features.transpose(1, 2).to(device)  # (1, F, T)
    mask = torch.ones(1, 1, bundle.rally.frame_count, device=device)

    with torch.no_grad():
        logits = model(features, mask)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()  # (C, T)

    return probs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_mode(
    bundles: list[RallyBundle],
    model: MSTCN,
    device: torch.device,
    mode: str,
    args: argparse.Namespace,
) -> tuple[list[MatchResult], list[dict], int]:
    """Evaluate one decoding mode on a list of rallies.

    Args:
        mode: "trajectory" | "sequence" | "anchored"

    Returns:
        (matches, unmatched_preds, attribution_correct_count)
    """
    all_matches: list[MatchResult] = []
    all_unmatched: list[dict] = []
    attr_correct = 0
    attr_total = 0

    decoder_config = DecoderConfig()

    for bundle in bundles:
        frame_probs = get_frame_probs(model, bundle, device)

        if mode == "trajectory":
            # Baseline: contacts from trajectory only, action from sequence model
            decoded = decode_actions(
                frame_probs,
                contact_sequence=bundle.contact_sequence,
                config=DecoderConfig(
                    sequence_peak_threshold=1.1,  # Effectively disable sequence proposals
                ),
                net_y=bundle.rally.court_split_y or 0.5,
            )
        elif mode == "sequence":
            # Sequence-only: no trajectory anchors
            decoded = decode_actions(
                frame_probs,
                contact_sequence=None,
                config=decoder_config,
                net_y=bundle.rally.court_split_y or 0.5,
            )
        else:  # anchored
            decoded = decode_actions(
                frame_probs,
                contact_sequence=bundle.contact_sequence,
                config=decoder_config,
                net_y=bundle.rally.court_split_y or 0.5,
            )

        # Apply team-based attribution if requested
        if args.attribution and bundle.match_teams:
            decoded = attribute_with_team_hint(
                decoded,
                team_assignments=bundle.match_teams,
                net_y=bundle.rally.court_split_y or 0.5,
            )

        # Convert to match format
        preds = [
            {
                "frame": d.frame,
                "action": d.action,
                "playerTrackId": d.player_track_id,
            }
            for d in decoded
        ]

        matches, unmatched = match_contacts(bundle.gt_labels, preds, tolerance=5)
        all_matches.extend(matches)
        all_unmatched.extend(unmatched)

        # Attribution accuracy: check if matched predictions have correct player
        if args.attribution:
            for m in matches:
                if m.pred_frame is not None and m.player_evaluable:
                    attr_total += 1
                    if m.player_correct:
                        attr_correct += 1

    return all_matches, all_unmatched, attr_correct


# ---------------------------------------------------------------------------
# LOO-CV
# ---------------------------------------------------------------------------


def run_cv(
    bundles: list[RallyBundle],
    args: argparse.Namespace,
) -> None:
    """Run k-fold cross-validation (default 5-fold, or LOO with --folds 0)."""
    videos: dict[str, list[RallyBundle]] = defaultdict(list)
    for b in bundles:
        videos[b.rally.video_id].append(b)

    video_ids = sorted(videos.keys())
    n_folds = args.folds if args.folds > 0 else len(video_ids)

    # Assign videos to folds (round-robin)
    fold_for_video: dict[str, int] = {}
    for i, vid in enumerate(video_ids):
        fold_for_video[vid] = i % n_folds

    cv_label = "LOO-CV" if n_folds == len(video_ids) else f"{n_folds}-fold CV"
    console.print(f"\n[bold]{cv_label}: {len(video_ids)} videos, {len(bundles)} rallies[/bold]")

    modes = ["trajectory", "sequence", "anchored"]
    results_by_mode: dict[str, tuple[list[MatchResult], list[dict], int]] = {
        m: ([], [], 0) for m in modes
    }

    device = torch.device(args.device)
    start_time = time.time()

    for fold_idx in range(n_folds):
        test_bundles = [b for b in bundles if fold_for_video[b.rally.video_id] == fold_idx]
        train_bundles = [b for b in bundles if fold_for_video[b.rally.video_id] != fold_idx]
        if not test_bundles or not train_bundles:
            continue

        fold_start = time.time()
        model = train_sequence_model(train_bundles, test_bundles, args)

        for mode in modes:
            matches, unmatched, attr_ok = evaluate_mode(
                test_bundles, model, device, mode, args,
            )
            old_matches, old_unmatched, old_attr = results_by_mode[mode]
            results_by_mode[mode] = (
                old_matches + matches,
                old_unmatched + unmatched,
                old_attr + attr_ok,
            )

        fold_time = time.time() - fold_start

        # Print anchored mode per-fold summary
        anch_matches = results_by_mode["anchored"][0]
        recent = [m for m in anch_matches[-len(test_bundles) * 20:]]
        anch_metrics = compute_metrics(recent, [])
        n_test_vids = len({b.rally.video_id for b in test_bundles})
        console.print(
            f"  [{fold_idx + 1}/{n_folds}] videos={n_test_vids} "
            f"rallies={len(test_bundles)} "
            f"action_acc={anch_metrics['action_accuracy']:.1%} "
            f"time={fold_time:.0f}s"
        )

    total_time = time.time() - start_time
    console.print(f"\n[bold]Results ({total_time:.0f}s total)[/bold]")

    # Summary comparison table
    summary = Table(title=f"Decoder Mode Comparison ({cv_label})")
    summary.add_column("Mode")
    summary.add_column("Contact F1", justify="right")
    summary.add_column("Precision", justify="right")
    summary.add_column("Recall", justify="right")
    summary.add_column("Action Acc", justify="right")
    if args.attribution:
        summary.add_column("Attrib Acc", justify="right")

    for mode in modes:
        matches, unmatched, attr_ok = results_by_mode[mode]
        metrics = compute_metrics(matches, unmatched)

        row = [
            mode,
            f"{metrics['f1']:.1%}",
            f"{metrics['precision']:.1%}",
            f"{metrics['recall']:.1%}",
            f"{metrics['action_accuracy']:.1%}",
        ]
        if args.attribution:
            attr_total = sum(
                1 for m in matches
                if m.pred_frame is not None and m.player_evaluable
            )
            attr_acc = attr_ok / attr_total if attr_total > 0 else 0.0
            row.append(f"{attr_acc:.1%}")
        summary.add_row(*row)

    console.print(summary)

    # Per-class for anchored mode
    anch_matches, anch_unmatched, _ = results_by_mode["anchored"]
    overall = compute_metrics(anch_matches, anch_unmatched)

    if "per_class" in overall:
        cls_table = Table(title=f"Anchored Decoder Per-Class ({cv_label})")
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
                    str(c["tp"]), str(c["fp"]), str(c["fn"]),
                    f"{c['precision']:.1%}",
                    f"{c['recall']:.1%}",
                    f"{c['f1']:.1%}",
                )
        console.print(cls_table)

    # Confusion matrix
    _print_confusion_matrix(anch_matches)


def _print_confusion_matrix(matches: list[MatchResult]) -> None:
    """Print confusion matrix from matched predictions."""
    matched = [m for m in matches if m.pred_frame is not None]
    if not matched:
        return

    cm: dict[str, dict[str, int]] = {a: {b: 0 for b in ACTION_TYPES} for a in ACTION_TYPES}
    for m in matched:
        gt_a = m.gt_action
        pred_a = m.pred_action or ""
        if gt_a in cm and pred_a in cm[gt_a]:
            cm[gt_a][pred_a] += 1

    table = Table(title="Confusion Matrix (GT rows x Pred cols)")
    table.add_column("GT \\ Pred")
    for a in ACTION_TYPES:
        table.add_column(a[:3], justify="right")
    table.add_column("N", justify="right")

    for gt_action in ACTION_TYPES:
        row_total = sum(cm[gt_action].values())
        cells = []
        for pred_action in ACTION_TYPES:
            count = cm[gt_action][pred_action]
            if count > 0 and gt_action == pred_action:
                cells.append(f"[green]{count}[/green]")
            elif count > 0:
                cells.append(f"[red]{count}[/red]")
            else:
                cells.append(".")
        table.add_row(gt_action, *cells, str(row_total))
    console.print(table)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate contact-anchored sequence decoder",
    )

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

    # Evaluation
    parser.add_argument("--attribution", action="store_true",
                        help="Evaluate team-based player attribution")
    parser.add_argument("--no-viterbi", action="store_true",
                        help="Disable Viterbi decoding (test raw decoder)")
    parser.add_argument("--folds", type=int, default=5,
                        help="Number of CV folds (0 = LOO-CV, default 5)")

    parser.add_argument(
        "--device", type=str,
        default="mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu",
    )
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    bundles = prepare_rallies(label_spread=args.label_spread)
    if not bundles:
        console.print("[red]No rallies loaded.[/red]")
        return

    run_cv(bundles, args)


if __name__ == "__main__":
    main()
