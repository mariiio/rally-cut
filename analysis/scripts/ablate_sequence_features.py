"""Feature importance analysis for the enhanced 26-dim sequence model.

Two-phase approach:
  Phase A: Train ONE model on all data with all 31 features (~30 min).
  Phase B: Permutation importance — for each feature group, shuffle those
           columns N times and measure accuracy drop on held-out folds.

This is ~10× faster than training separate models per feature config
while giving the same ranking of which features help/hurt/are dead.

Also runs a 3-fold additive ablation for the most important groups
to confirm the permutation results with actual retraining.

Usage:
    cd analysis
    uv run python scripts/ablate_sequence_features.py
    uv run python scripts/ablate_sequence_features.py --confirm-top 3
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

sys.path.insert(0, str(Path(__file__).parent))

from eval_action_detection import (  # noqa: E402
    GtLabel,
    MatchResult,
    compute_metrics,
    load_rallies_with_action_gt,
    match_contacts,
)
from train_sequence_action import (
    _load_team_assignments_and_calibrations,
    _parse_ball,
    _parse_players,
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

console = Console()
IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}


# Feature groups: name → list of column indices (26-dim layout)
FEATURE_GROUPS: dict[str, list[int]] = {
    "court_ball": [19, 20],                   # Court-space position
    "det_density": [21],                      # Ball detection density
    "team_indicators": [22, 23, 24, 25],     # Player team labels
}

# Also test individual baseline sub-groups for completeness
BASELINE_SUBGROUPS: dict[str, list[int]] = {
    "ball_pos": [0, 1, 2],                   # ball x, y, confidence
    "ball_vel": [3, 4, 5],                   # ball dx, dy, speed
    "player_pos": list(range(6, 14)),         # 4 players × (x, y)
    "ball_player_dist": [14, 15, 16, 17],    # ball-player distances
    "ball_rel_net": [18],                     # ball y relative to net
}


@dataclass
class AblationSequence:
    rally_id: str
    video_id: str
    features: np.ndarray  # (T, 26)
    labels: np.ndarray    # (T,) int64
    gt_labels: list[GtLabel]
    frame_count: int


def load_all_sequences(label_spread: int = 2) -> list[AblationSequence]:
    """Load rallies and extract full 26-dim features."""
    console.print("[bold]Loading rallies...[/bold]")
    rallies = load_rallies_with_action_gt()
    team_by_rally, homographies = _load_team_assignments_and_calibrations(rallies)

    sequences: list[AblationSequence] = []
    for rally in rallies:
        if not rally.ball_positions_json or not rally.positions_json:
            continue
        if not rally.frame_count or rally.frame_count < 10:
            continue
        ball_positions = _parse_ball(rally.ball_positions_json)
        player_positions = _parse_players(rally.positions_json)
        if not ball_positions:
            continue

        features = extract_trajectory_features(
            ball_positions=ball_positions,
            player_positions=player_positions,
            net_y=rally.court_split_y,
            frame_count=rally.frame_count,
            team_assignments=team_by_rally.get(rally.rally_id),
            homography=homographies.get(rally.video_id),
        )
        gt_dicts = [{"frame": gt.frame, "action": gt.action} for gt in rally.gt_labels]
        labels = build_frame_labels(gt_dicts, rally.frame_count, label_spread=label_spread)

        sequences.append(AblationSequence(
            rally_id=rally.rally_id,
            video_id=rally.video_id,
            features=features,
            labels=labels,
            gt_labels=rally.gt_labels,
            frame_count=rally.frame_count,
        ))

    console.print(f"  {len(sequences)} sequences loaded")
    return sequences


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    train_seqs: list[AblationSequence],
    val_seqs: list[AblationSequence],
    args: argparse.Namespace,
    feature_dim: int = FEATURE_DIM,
) -> tuple[MSTCN, int]:
    """Train MS-TCN++ and return best model + best epoch."""
    device = torch.device(args.device)

    config = MSTCNConfig(
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        num_classes=NUM_CLASSES,
        num_stages=args.num_stages,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    model = MSTCN(config).to(device)

    all_labels = np.concatenate([s.labels for s in train_seqs])
    class_counts = np.bincount(all_labels, minlength=NUM_CLASSES).astype(float)
    class_counts = np.maximum(class_counts, 1.0)
    max_count = class_counts.max()
    weights = torch.tensor(
        [max_count / c for c in class_counts], dtype=torch.float32, device=device,
    )
    weights[0] = weights[0].clamp(max=1.0)
    criterion = FocalLoss(weight=weights, gamma=args.focal_gamma, reduction="none")

    train_dataset = SequenceActionDataset(
        [s.features for s in train_seqs], [s.labels for s in train_seqs],
        augment=True, noise_std=args.noise_std,
    )
    val_dataset = SequenceActionDataset(
        [s.features for s in val_seqs], [s.labels for s in val_seqs],
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
    best_epoch = 0
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        for features, labels, mask in train_loader:
            features, labels, mask = features.to(device), labels.to(device), mask.to(device)
            optimizer.zero_grad()
            stage_outputs = model.forward_all_stages(features, mask)
            total_loss = torch.tensor(0.0, device=device)
            for logits in stage_outputs:
                ce = criterion(logits, labels)
                m_sq = mask.squeeze(1)
                ce = (ce * m_sq).sum() / m_sq.sum()
                tmse = compute_tmse_loss(logits, mask)
                total_loss = total_loss + ce + args.tmse_weight * tmse
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                continue
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        preds_all: list[int] = []
        targets_all: list[int] = []
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
                    am = t > 0
                    if am.any():
                        preds_all.extend(p[am].tolist())
                        targets_all.extend(t[am].tolist())

        val_acc = sum(1 for p, t in zip(preds_all, targets_all) if p == t) / len(targets_all) if targets_all else 0.0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        if patience_counter >= args.patience:
            break

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)

    return model, best_epoch


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


def eval_model_on_sequences(
    model: MSTCN,
    sequences: list[AblationSequence],
    device: torch.device,
    features_override: list[np.ndarray] | None = None,
) -> dict:
    """Evaluate model: action accuracy on GT frames + contact F1 via peak detection."""
    model.eval()
    all_matches: list[MatchResult] = []
    all_unmatched: list[dict] = []
    frame_correct = 0
    frame_total = 0

    for i, seq in enumerate(sequences):
        feat = features_override[i] if features_override else seq.features
        feat_t = torch.from_numpy(feat).float().unsqueeze(0).transpose(1, 2).to(device)
        mask = torch.ones(1, 1, seq.frame_count, device=device)

        with torch.no_grad():
            logits = model(feat_t, mask)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()  # (C, T)
            preds = logits.argmax(dim=1)[0].cpu().numpy()  # (T,)

        # Frame-level action accuracy on GT action frames
        for gt in seq.gt_labels:
            if 0 <= gt.frame < seq.frame_count:
                frame_total += 1
                gt_cls = ACTION_TO_IDX.get(gt.action, 0)
                if preds[gt.frame] == gt_cls:
                    frame_correct += 1

        # Contact-level: peak detection
        pred_list: list[dict] = []
        for cls_idx in range(1, NUM_CLASSES):
            cls_probs = gaussian_filter1d(probs[cls_idx], sigma=2.0)
            peaks, _ = find_peaks(cls_probs, height=0.3, distance=12)
            for peak in peaks:
                pred_list.append({
                    "frame": int(peak),
                    "action": ACTION_TYPES[cls_idx - 1],
                    "playerTrackId": -1,
                    "confidence": float(cls_probs[peak]),
                })
        pred_list.sort(key=lambda p: p["frame"])
        if pred_list:
            filtered: list[dict] = [pred_list[0]]
            for pred in pred_list[1:]:
                if pred["frame"] - filtered[-1]["frame"] < 12:
                    if pred["confidence"] > filtered[-1]["confidence"]:
                        filtered[-1] = pred
                else:
                    filtered.append(pred)
            pred_list = filtered

        matches, unmatched = match_contacts(seq.gt_labels, pred_list, tolerance=5)
        all_matches.extend(matches)
        all_unmatched.extend(unmatched)

    metrics = compute_metrics(all_matches, all_unmatched)
    metrics["frame_action_accuracy"] = frame_correct / frame_total if frame_total else 0.0
    return metrics


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------


def permutation_importance(
    model: MSTCN,
    val_seqs: list[AblationSequence],
    device: torch.device,
    n_repeats: int = 5,
) -> dict[str, dict]:
    """Compute permutation importance for each feature group.

    For each group, shuffle those columns across all val sequences and
    measure the drop in frame-level action accuracy (most stable metric).
    Repeat n_repeats times and report mean ± std of the drop.
    """
    # Baseline accuracy (no shuffling)
    baseline = eval_model_on_sequences(model, val_seqs, device)
    base_acc = baseline["frame_action_accuracy"]
    base_f1 = baseline["f1"]
    console.print(
        f"  Baseline: frame_action_acc={base_acc:.1%}, "
        f"contact_F1={base_f1:.1%}"
    )

    all_groups = {**FEATURE_GROUPS, **BASELINE_SUBGROUPS}
    results: dict[str, dict] = {}

    for group_name, col_indices in all_groups.items():
        drops_acc: list[float] = []
        drops_f1: list[float] = []

        for repeat in range(n_repeats):
            rng = np.random.default_rng(seed=repeat)
            # Create shuffled features
            shuffled_features: list[np.ndarray] = []
            for seq in val_seqs:
                feat = seq.features.copy()
                for col in col_indices:
                    if col < feat.shape[1]:
                        rng.shuffle(feat[:, col])  # shuffle within rally
                shuffled_features.append(feat)

            metrics = eval_model_on_sequences(
                model, val_seqs, device, features_override=shuffled_features,
            )
            drops_acc.append(base_acc - metrics["frame_action_accuracy"])
            drops_f1.append(base_f1 - metrics["f1"])

        results[group_name] = {
            "cols": col_indices,
            "n_dims": len(col_indices),
            "acc_drop_mean": np.mean(drops_acc),
            "acc_drop_std": np.std(drops_acc),
            "f1_drop_mean": np.mean(drops_f1),
            "f1_drop_std": np.std(drops_f1),
        }
        direction = "important" if np.mean(drops_acc) > 0.005 else (
            "HARMFUL" if np.mean(drops_acc) < -0.005 else "negligible"
        )
        console.print(
            f"  {group_name:20s} ({len(col_indices)} dims): "
            f"acc_drop={np.mean(drops_acc):+.2%} ± {np.std(drops_acc):.2%}  "
            f"F1_drop={np.mean(drops_f1):+.2%} ± {np.std(drops_f1):.2%}  "
            f"[{'green' if direction == 'important' else 'red' if direction == 'HARMFUL' else 'yellow'}]"
            f"{direction}[/]"
        )

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    sequences = load_all_sequences(label_spread=args.label_spread)
    if not sequences:
        console.print("[red]No sequences loaded.[/red]")
        return

    # Split into train/val: use ~20% for val (round-robin by video)
    videos: dict[str, list[AblationSequence]] = defaultdict(list)
    for s in sequences:
        videos[s.video_id].append(s)
    video_ids = sorted(videos.keys())

    # 3-fold: train on 2 folds, val on 1 fold, repeat and aggregate
    n_folds = 3
    fold_map: dict[str, int] = {}
    for i, vid in enumerate(video_ids):
        fold_map[vid] = i % n_folds

    console.print(f"\n[bold]Permutation importance ({n_folds}-fold, {len(sequences)} rallies)[/bold]")

    all_results: list[dict[str, dict]] = []
    device = torch.device(args.device)

    for fold in range(n_folds):
        train_seqs = [s for s in sequences if fold_map[s.video_id] != fold]
        val_seqs = [s for s in sequences if fold_map[s.video_id] == fold]
        if not val_seqs or not train_seqs:
            continue

        console.print(f"\n[bold]Fold {fold + 1}/{n_folds}[/bold] "
                       f"(train={len(train_seqs)}, val={len(val_seqs)})")

        fold_start = time.time()
        model, best_epoch = train_model(train_seqs, val_seqs, args)
        train_time = time.time() - fold_start
        console.print(f"  Trained in {train_time:.0f}s (best epoch {best_epoch})")

        perm_start = time.time()
        fold_results = permutation_importance(
            model, val_seqs, device, n_repeats=args.n_repeats,
        )
        perm_time = time.time() - perm_start
        console.print(f"  Permutation test: {perm_time:.0f}s")

        all_results.append(fold_results)

    # Aggregate across folds
    console.print(f"\n[bold]{'='*70}[/bold]")
    console.print("[bold]Aggregated Feature Importance (across all folds)[/bold]")

    all_groups = list({**FEATURE_GROUPS, **BASELINE_SUBGROUPS}.keys())

    table = Table(title="Feature Group Importance (permutation test)")
    table.add_column("Feature Group")
    table.add_column("Dims", justify="right")
    table.add_column("Acc Drop", justify="right")
    table.add_column("F1 Drop", justify="right")
    table.add_column("Verdict")

    # Sort by importance (acc drop descending)
    group_importance: list[tuple[str, float, float, int]] = []
    for group_name in all_groups:
        acc_drops = [r[group_name]["acc_drop_mean"] for r in all_results if group_name in r]
        f1_drops = [r[group_name]["f1_drop_mean"] for r in all_results if group_name in r]
        n_dims = all_results[0][group_name]["n_dims"] if all_results and group_name in all_results[0] else 0
        if acc_drops:
            group_importance.append((
                group_name, np.mean(acc_drops), np.mean(f1_drops), n_dims,
            ))

    group_importance.sort(key=lambda x: x[1], reverse=True)

    for group_name, acc_drop, f1_drop, n_dims in group_importance:
        if acc_drop > 0.01:
            verdict = "[green bold]IMPORTANT[/green bold]"
        elif acc_drop > 0.003:
            verdict = "[green]helpful[/green]"
        elif acc_drop < -0.005:
            verdict = "[red bold]HARMFUL — remove[/red bold]"
        elif acc_drop < -0.002:
            verdict = "[red]possibly harmful[/red]"
        else:
            verdict = "[yellow]negligible[/yellow]"

        is_new = group_name in FEATURE_GROUPS
        name_display = f"{'★ ' if is_new else '  '}{group_name}"

        table.add_row(
            name_display,
            str(n_dims),
            f"{acc_drop:+.2%}",
            f"{f1_drop:+.2%}",
            verdict,
        )

    console.print(table)
    console.print("\n★ = new feature group (Phase 1)")
    console.print("Positive drop = feature helps (removing it hurts accuracy)")
    console.print("Negative drop = feature hurts (removing it improves accuracy)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Feature importance for sequence model")

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

    parser.add_argument("--n-repeats", type=int, default=5,
                        help="Permutation repeats per feature group")

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

    run(args)


if __name__ == "__main__":
    main()
