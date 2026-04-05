"""Hybrid action classifier: contact detector + sequence model.

Uses the existing contact detector for *when* contacts happen (90% F1),
then the MS-TCN++ sequence model for *what* action each contact is (93.6%
action accuracy). Replaces the GBM classifier, Viterbi, and repair rules.

Pipeline per rally:
  1. detect_contacts() → contact frames (existing pipeline)
  2. MS-TCN++ on full trajectory → per-frame action probabilities
  3. At each contact frame, read model's argmax prediction → action type
  4. Build ClassifiedAction objects
  5. reattribute_players() (kept — orthogonal to action classification)

Evaluation: leave-one-video-out CV.
Target: ~90% contact F1 (from detector) + ~93% action accuracy (from model).

Usage:
    cd analysis
    uv run python scripts/eval_hybrid_action.py
    uv run python scripts/eval_hybrid_action.py --hidden-dim 128
    uv run python scripts/eval_hybrid_action.py --skip-reattribute
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
    MatchResult,
    RallyData,
    _load_match_team_assignments,
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
from rallycut.tracking.action_classifier import (
    ActionType,
    ClassifiedAction,
    _find_server_by_position,
    assign_court_side_from_teams,
    reattribute_players,
)
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_detector import (
    ContactDetectionConfig,
    ContactSequence,
    detect_contacts,
)
from rallycut.tracking.player_tracker import PlayerPosition

# Suppress verbose logging from contact detection
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


def prepare_rallies(
    label_spread: int = 2,
) -> list[RallyBundle]:
    """Load rallies, run contact detection, extract trajectory features."""
    console.print("[bold]Loading rallies with action GT...[/bold]")
    rallies = load_rallies_with_action_gt()
    console.print(f"  {len(rallies)} rallies loaded")

    # Load team assignments
    video_ids = {r.video_id for r in rallies}
    rally_team_map = _load_match_team_assignments(video_ids)

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

        # Run contact detection (same as production pipeline)
        contact_seq = detect_contacts(
            ball_positions=ball_positions,
            player_positions=player_positions,
            config=ContactDetectionConfig(),
            net_y=rally.court_split_y,
            frame_count=rally.frame_count,
        )

        # Extract trajectory features for sequence model
        features = extract_trajectory_features(
            ball_positions=ball_positions,
            player_positions=player_positions,
            net_y=rally.court_split_y,
            frame_count=rally.frame_count,
        )

        gt_dicts = [{"frame": gt.frame, "action": gt.action} for gt in rally.gt_labels]
        labels = build_frame_labels(gt_dicts, rally.frame_count, label_spread=label_spread)

        match_teams = rally_team_map.get(rally.rally_id)

        bundles.append(RallyBundle(
            rally=rally,
            ball_positions=ball_positions,
            player_positions=player_positions,
            contact_sequence=contact_seq,
            trajectory_features=features,
            trajectory_labels=labels,
            match_teams=match_teams,
        ))

        if (i + 1) % 50 == 0 or i == len(rallies) - 1:
            console.print(f"  [{i + 1}/{len(rallies)}] {len(bundles)} bundles prepared")

    if skipped:
        console.print(f"  [yellow]Skipped {skipped} rallies (missing data)[/yellow]")

    console.print(f"  Total: {len(bundles)} rallies with contacts + features")
    return bundles


# ---------------------------------------------------------------------------
# Training (same as train_sequence_action.py)
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

    # Class weights from training labels
    all_labels = np.concatenate([b.trajectory_labels for b in train_bundles])
    class_counts = np.bincount(all_labels, minlength=NUM_CLASSES).astype(float)
    class_counts = np.maximum(class_counts, 1.0)
    max_count = class_counts.max()
    weights = torch.tensor(
        [max_count / c for c in class_counts], dtype=torch.float32, device=device,
    )
    weights[0] = weights[0].clamp(max=1.0)  # Cap background weight

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
    best_model_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        for features, labels, mask in train_loader:
            features, labels, mask = features.to(device), labels.to(device), mask.to(device)
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

        # Validate: action accuracy on GT action frames
        model.eval()
        all_preds: list[int] = []
        all_targets: list[int] = []
        with torch.no_grad():
            for features, labels, mask in val_loader:
                features, labels, mask = features.to(device), labels.to(device), mask.to(device)
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
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            break

    if best_model_state:
        model.load_state_dict(best_model_state)
        model = model.to(device)

    return model


# ---------------------------------------------------------------------------
# Hybrid classification: contacts from detector + actions from sequence model
# ---------------------------------------------------------------------------


def classify_hybrid(
    bundle: RallyBundle,
    model: MSTCN,
    device: torch.device,
    do_reattribute: bool = True,
) -> list[dict]:
    """Classify contacts using the hybrid approach.

    Pipeline:
      1. Run MS-TCN++ on full trajectory → per-frame action probabilities.
      2. For each detected contact, look up the model's prediction.
      3. If no serve detected among contacts, inject a synthetic serve
         using the existing position-based server detection heuristic.
      4. Run reattribute_players() (orthogonal to action classification).

    Returns list of predicted action dicts for eval matching.
    """
    contacts = bundle.contact_sequence.contacts

    # Step 1: Run sequence model on full rally
    model.eval()
    num_frames = bundle.trajectory_features.shape[0]
    features = torch.from_numpy(bundle.trajectory_features).float().unsqueeze(0)
    features = features.transpose(1, 2).to(device)  # (1, F, T)
    mask = torch.ones(1, 1, num_frames, device=device)

    with torch.no_grad():
        logits = model(features, mask)  # (1, C, T)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()  # (C, T)

    # Step 2: Classify each contact using model prediction at that frame
    actions: list[ClassifiedAction] = []

    for contact in contacts:
        frame = contact.frame
        if frame < 0 or frame >= num_frames:
            continue

        # Find best action class at this frame (±2 window, center-weighted)
        best_cls = 0
        best_prob = 0.0
        for offset in range(-2, 3):
            f = frame + offset
            if 0 <= f < num_frames:
                weight = 1.0 if offset == 0 else 0.6
                for cls in range(1, NUM_CLASSES):
                    p = float(probs[cls, f]) * weight
                    if p > best_prob:
                        best_prob = p
                        best_cls = cls

        if best_cls > 0:
            action_name = IDX_TO_ACTION[best_cls]
            confidence = float(probs[best_cls, frame])
        else:
            # No signal — use positional fallback
            action_name = "attack"
            confidence = 0.3

        actions.append(ClassifiedAction(
            action_type=ActionType(action_name),
            frame=contact.frame,
            ball_x=contact.ball_x,
            ball_y=contact.ball_y,
            velocity=contact.velocity,
            player_track_id=contact.player_track_id,
            court_side=contact.court_side,
            confidence=confidence,
        ))

    # Step 3: Model-guided synthetic serve injection.
    # Every rally has exactly one serve. If no contact was classified as
    # serve, find the peak serve probability in the model's output within
    # the first ~90 frames and insert a synthetic serve there.
    net_y = bundle.contact_sequence.net_y
    start_frame = bundle.contact_sequence.rally_start_frame
    serve_search_frames = 90  # ~3s at 30fps

    has_serve = any(a.action_type == ActionType.SERVE for a in actions)

    if not has_serve:
        # Find peak serve probability (class 1) in first 90 frames
        serve_cls = ACTION_TO_IDX["serve"]  # 1
        search_end = min(serve_search_frames, num_frames)
        serve_probs = probs[serve_cls, :search_end]

        if len(serve_probs) > 0 and serve_probs.max() > 0.05:
            serve_frame = int(np.argmax(serve_probs))
            serve_conf = float(serve_probs[serve_frame])

            # Determine serve side from ball position at serve frame
            ball_y_at_serve = bundle.trajectory_features[serve_frame, 1]  # ball_y
            ball_conf_at_serve = bundle.trajectory_features[serve_frame, 2]
            if ball_conf_at_serve > 0 and net_y > 0:
                serve_side = "near" if ball_y_at_serve > net_y else "far"
            else:
                serve_side = "near"

            # Find server track ID from nearest player at serve frame
            server_tid = -1
            if bundle.contact_sequence.player_positions:
                server_tid, server_pos_side, _ = _find_server_by_position(
                    bundle.contact_sequence.player_positions,
                    start_frame, net_y,
                )
                if server_pos_side:
                    serve_side = server_pos_side

            synth = ClassifiedAction(
                action_type=ActionType.SERVE,
                frame=serve_frame,
                ball_x=float(bundle.trajectory_features[serve_frame, 0]),
                ball_y=float(ball_y_at_serve),
                velocity=0.0,
                player_track_id=server_tid,
                court_side=serve_side,
                confidence=serve_conf,
                is_synthetic=True,
            )

            # Insert before first action that comes after the serve frame
            insert_idx = next(
                (i for i, a in enumerate(actions) if a.frame > serve_frame),
                len(actions),
            )
            actions.insert(insert_idx, synth)

    # Step 4: Court side + player re-attribution
    if bundle.match_teams:
        assign_court_side_from_teams(actions, bundle.match_teams)

    if do_reattribute and bundle.match_teams:
        actions = reattribute_players(
            actions, contacts, bundle.match_teams,
            max_distance_ratio=1.5,
        )

    # Convert to prediction dicts for matching
    pred_actions: list[dict] = []
    for a in actions:
        pred_actions.append({
            "frame": a.frame,
            "action": a.action_type.value,
            "playerTrackId": a.player_track_id,
        })

    return pred_actions


# ---------------------------------------------------------------------------
# LOO-video CV
# ---------------------------------------------------------------------------


def run_loo_cv(bundles: list[RallyBundle], args: argparse.Namespace) -> None:
    """Leave-one-video-out cross-validation of hybrid approach."""
    videos: dict[str, list[RallyBundle]] = defaultdict(list)
    for b in bundles:
        videos[b.rally.video_id].append(b)

    video_ids = sorted(videos.keys())
    console.print(f"\n[bold]Hybrid LOO-CV: {len(video_ids)} videos[/bold]")

    all_matches: list[MatchResult] = []
    all_unmatched: list[dict] = []
    device = torch.device(args.device)
    start_time = time.time()

    for fold_idx, held_out_vid in enumerate(video_ids):
        test_bundles = videos[held_out_vid]
        train_bundles = [
            b for vid, bs in videos.items() if vid != held_out_vid for b in bs
        ]

        if not test_bundles or not train_bundles:
            continue

        fold_start = time.time()

        # Train sequence model
        model = train_sequence_model(train_bundles, test_bundles, args)

        # Evaluate each test rally
        fold_matches: list[MatchResult] = []
        fold_unmatched: list[dict] = []

        for bundle in test_bundles:
            preds = classify_hybrid(
                bundle, model, device,
                do_reattribute=not args.skip_reattribute,
            )
            matches, unmatched = match_contacts(
                bundle.rally.gt_labels, preds, tolerance=5,
            )
            fold_matches.extend(matches)
            fold_unmatched.extend(unmatched)

        fold_metrics = compute_metrics(fold_matches, fold_unmatched)
        fold_time = time.time() - fold_start

        n_gt = len(fold_matches)
        n_tp = fold_metrics["tp"]
        action_acc = fold_metrics["action_accuracy"]
        f1 = fold_metrics["f1"]

        console.print(
            f"  [{fold_idx + 1}/{len(video_ids)}] video={held_out_vid[:8]}… "
            f"rallies={len(test_bundles)} GT={n_gt} TP={n_tp} "
            f"F1={f1:.1%} action_acc={action_acc:.1%} "
            f"time={fold_time:.0f}s"
        )

        all_matches.extend(fold_matches)
        all_unmatched.extend(fold_unmatched)

    total_time = time.time() - start_time

    # --- Aggregate results ---
    console.print(f"\n[bold]Aggregate Hybrid LOO-CV Results ({total_time:.0f}s total)[/bold]")
    overall = compute_metrics(all_matches, all_unmatched)

    summary = Table(title="Hybrid: Contact Detector + Sequence Model")
    summary.add_column("Metric")
    summary.add_column("Value", justify="right")
    summary.add_row("GT contacts", str(overall["total_gt"]))
    summary.add_row("Predicted", str(overall["total_pred"]))
    summary.add_row("TP", str(overall["tp"]))
    summary.add_row("FP", str(overall["fp"]))
    summary.add_row("FN", str(overall["fn"]))
    summary.add_row("Contact Recall", f"{overall['recall']:.1%}")
    summary.add_row("Contact Precision", f"{overall['precision']:.1%}")
    summary.add_row("[bold]Contact F1[/bold]", f"[bold]{overall['f1']:.1%}[/bold]")
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
                    str(c["tp"]), str(c["fp"]), str(c["fn"]),
                    f"{c['precision']:.1%}", f"{c['recall']:.1%}", f"{c['f1']:.1%}",
                )
        console.print(cls_table)

    # Comparison
    console.print("\n[bold]Comparison[/bold]")
    comp = Table()
    comp.add_column("Model")
    comp.add_column("Contact F1", justify="right")
    comp.add_column("Action Acc", justify="right")
    comp.add_row("GBM pipeline (baseline)", "90.0%", "83.1%")
    comp.add_row("Seq-only (t=0.3)", "68.9%", "93.6%")
    comp.add_row(
        f"Hybrid (h={args.hidden_dim}, s={args.num_stages}, L={args.num_layers})",
        f"{overall['f1']:.1%}",
        f"{overall['action_accuracy']:.1%}",
    )
    console.print(comp)

    # Confusion matrix
    _print_confusion_matrix(all_matches)


def _print_confusion_matrix(matches: list[MatchResult]) -> None:
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

    table = Table(title="Confusion Matrix (GT rows x Pred cols)")
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
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid action classifier evaluation")

    # Model
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.3)

    # Training
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--tmse-weight", type=float, default=0.15)
    parser.add_argument("--noise-std", type=float, default=0.01)
    parser.add_argument("--label-spread", type=int, default=2)

    # Pipeline
    parser.add_argument("--skip-reattribute", action="store_true",
                        help="Skip player re-attribution stage")

    # Runtime
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

    run_loo_cv(bundles, args)


if __name__ == "__main__":
    main()
