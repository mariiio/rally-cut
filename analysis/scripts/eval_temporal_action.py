"""Feasibility experiment: TemporalMaxer for per-frame action detection.

Repurposes the TemporalMaxer temporal action segmentation model (currently
used for rally detection at 95% F1) for joint contact detection + action
classification. Trains on rally-level slices of cached VideoMAE features
with 7 output classes: NO_CONTACT, SERVE, RECEIVE, SET, ATTACK, DIG, BLOCK.

Uses the existing train/held-out video split for honest evaluation.

Usage:
    cd analysis
    uv run python scripts/eval_temporal_action.py
    uv run python scripts/eval_temporal_action.py --epochs 120 --label-spread 1
    uv run python scripts/eval_temporal_action.py --lr 1e-4 --hidden-dim 256
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from rich.table import Table
from sklearn.metrics import f1_score
from torch.nn import functional as nnf
from torch.utils.data import DataLoader

from rallycut.evaluation.db import get_connection
from rallycut.evaluation.split import video_split
from rallycut.temporal.features import FeatureCache
from rallycut.temporal.temporal_maxer.model import TemporalMaxer, TemporalMaxerConfig
from rallycut.temporal.temporal_maxer.training import (
    FocalLoss,
    VideoSequenceDataset,
    collate_video_sequences,
    compute_tmse_loss,
)

# Import from sibling script
sys.path.insert(0, str(Path(__file__).parent))
from eval_action_detection import (  # noqa: E402
    GtLabel,
    MatchResult,
    RallyData,
    compute_metrics,
    load_rallies_with_action_gt,
    match_contacts,
)

console = Console()

ACTION_TYPES = ["serve", "receive", "set", "attack", "dig", "block"]
ACTION_TO_IDX = {a: i + 1 for i, a in enumerate(ACTION_TYPES)}
IDX_TO_ACTION = {v: k for k, v in ACTION_TO_IDX.items()}
NUM_CLASSES = 7  # NO_CONTACT + 6 action types
STRIDE = 12
WINDOW_SIZE = 16


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_video_info(video_ids: set[str]) -> dict[str, tuple[str, float]]:
    """Query content_hash and fps for each video_id."""
    if not video_ids:
        return {}
    placeholders = ", ".join(["%s"] * len(video_ids))
    query = f"""
        SELECT id, content_hash, fps
        FROM videos
        WHERE id IN ({placeholders})
          AND content_hash IS NOT NULL
          AND deleted_at IS NULL
    """
    result: dict[str, tuple[str, float]] = {}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, list(video_ids))
            for row in cur.fetchall():
                result[str(row[0])] = (str(row[1]), float(row[2] or 30.0))
    return result


# ---------------------------------------------------------------------------
# Label generation
# ---------------------------------------------------------------------------


@dataclass
class RallySequence:
    """A rally-level training sequence."""

    rally_id: str
    video_id: str
    features: np.ndarray  # (num_windows, feature_dim)
    labels: np.ndarray  # (num_windows,) int64
    gt_labels: list[GtLabel]
    fps: float
    first_window: int  # absolute window index in full video
    rally_start_frame: int


def build_sequences(
    rallies: list[RallyData],
    video_info: dict[str, tuple[str, float]],
    cache: FeatureCache,
    stride: int,
    label_spread: int = 0,
) -> list[RallySequence]:
    """Build rally-level sequences from GT and cached features."""
    # Group rallies by video_id
    rallies_by_video: dict[str, list[RallyData]] = {}
    for r in rallies:
        rallies_by_video.setdefault(r.video_id, []).append(r)

    sequences: list[RallySequence] = []
    skipped_no_features = 0
    skipped_no_info = 0

    for video_id, video_rallies in rallies_by_video.items():
        info = video_info.get(video_id)
        if not info:
            skipped_no_info += len(video_rallies)
            continue

        content_hash, video_fps = info
        cached = cache.get(content_hash, stride)
        if cached is None:
            skipped_no_features += len(video_rallies)
            continue

        video_features, _metadata = cached
        num_video_windows = video_features.shape[0]

        for rally in video_rallies:
            fps = rally.fps if rally.fps > 0 else video_fps
            rally_start_frame = round(rally.start_ms / 1000.0 * fps)
            rally_end_frame = rally_start_frame + rally.frame_count

            first_window = rally_start_frame // stride
            last_window = min((rally_end_frame - 1) // stride, num_video_windows - 1)
            if first_window > last_window or first_window >= num_video_windows:
                continue

            rally_len = last_window - first_window + 1
            rally_features = video_features[first_window : last_window + 1]
            rally_labels = np.zeros(rally_len, dtype=np.int64)

            for gt in rally.gt_labels:
                abs_frame = rally_start_frame + gt.frame
                widx = abs_frame // stride - first_window
                cls = ACTION_TO_IDX.get(gt.action, 0)
                if cls == 0:
                    continue

                # Label the target window and optional neighbors
                for offset in range(-label_spread, label_spread + 1):
                    idx = widx + offset
                    if 0 <= idx < rally_len:
                        # Center window gets priority; neighbors only if still NO_CONTACT
                        if offset == 0 or rally_labels[idx] == 0:
                            rally_labels[idx] = cls

            sequences.append(
                RallySequence(
                    rally_id=rally.rally_id,
                    video_id=rally.video_id,
                    features=rally_features,
                    labels=rally_labels,
                    gt_labels=rally.gt_labels,
                    fps=fps,
                    first_window=first_window,
                    rally_start_frame=rally_start_frame,
                )
            )

    if skipped_no_info:
        console.print(f"  [yellow]Skipped {skipped_no_info} rallies (no video info)[/]")
    if skipped_no_features:
        console.print(
            f"  [yellow]Skipped {skipped_no_features} rallies (no cached features)[/]"
        )

    return sequences


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def train_model(
    train_seqs: list[RallySequence],
    val_seqs: list[RallySequence],
    args: argparse.Namespace,
) -> tuple[TemporalMaxer, dict]:
    """Train TemporalMaxer for action detection."""
    device = torch.device(args.device)
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    feature_dim = train_seqs[0].features.shape[1]

    config = TemporalMaxerConfig(
        feature_dim=feature_dim,
        hidden_dim=args.hidden_dim,
        num_classes=NUM_CLASSES,
        num_layers=args.num_layers,
        dropout=0.3,
    )
    model = TemporalMaxer(config).to(device)

    # Datasets
    train_features = [s.features for s in train_seqs]
    train_labels = [s.labels for s in train_seqs]
    val_features = [s.features for s in val_seqs]
    val_labels = [s.labels for s in val_seqs]

    train_dataset = VideoSequenceDataset(train_features, train_labels)
    val_dataset = VideoSequenceDataset(val_features, val_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_video_sequences,
        generator=torch.Generator().manual_seed(seed),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_video_sequences,
    )

    # Class weights: inverse frequency
    all_labels = np.concatenate(train_labels)
    class_counts = np.bincount(all_labels, minlength=NUM_CLASSES).astype(float)
    class_counts = np.maximum(class_counts, 1.0)
    max_count = class_counts.max()
    weights = torch.tensor(
        [max_count / c for c in class_counts], dtype=torch.float32, device=device
    )
    # Cap NO_CONTACT weight to avoid dominating loss
    weights[0] = min(weights[0], 1.0)

    console.print(f"  Class counts: {dict(enumerate(class_counts.astype(int).tolist()))}")
    console.print(f"  Class weights: {[f'{w:.1f}' for w in weights.tolist()]}")

    criterion: nn.Module = FocalLoss(
        weight=weights, gamma=args.focal_gamma, reduction="none"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.01
    )
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
            logits = model(features, mask)

            ce_loss = criterion(logits, labels)
            mask_sq = mask.squeeze(1)
            ce_loss = (ce_loss * mask_sq).sum() / mask_sq.sum()

            tmse_loss = compute_tmse_loss(logits, mask)
            loss = ce_loss + args.tmse_weight * tmse_loss

            if torch.isnan(loss) or torch.isinf(loss):
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_loss = train_loss / max(num_batches, 1)

        # --- Validate ---
        model.eval()
        all_preds: list[int] = []
        all_true: list[int] = []

        with torch.no_grad():
            for features, labels, mask in val_loader:
                features = features.to(device)
                labels = labels.to(device)
                mask = mask.to(device)

                logits = model(features, mask)
                preds = logits.argmax(dim=1)

                mask_bool = mask.squeeze(1).bool()
                for b in range(features.shape[0]):
                    valid = mask_bool[b]
                    all_preds.extend(preds[b][valid].cpu().tolist())
                    all_true.extend(labels[b][valid].tolist())

        y_true = np.array(all_true)
        y_pred = np.array(all_preds)

        # Contact-level F1: any non-zero prediction is a "detection"
        binary_true = (y_true > 0).astype(int)
        binary_pred = (y_pred > 0).astype(int)
        contact_f1 = f1_score(binary_true, binary_pred, zero_division=0)

        # Macro F1 over action classes (excluding NO_CONTACT)
        action_mask = y_true > 0
        if action_mask.sum() > 0:
            action_f1 = f1_score(
                y_true[action_mask],
                y_pred[action_mask],
                average="macro",
                zero_division=0,
            )
        else:
            action_f1 = 0.0

        # Use contact F1 for early stopping (primary metric)
        val_f1 = contact_f1

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0 or patience_counter == 0:
            console.print(
                f"  Epoch {epoch + 1:3d}/{args.epochs}: loss={avg_loss:.4f}  "
                f"contact_F1={contact_f1:.3f}  action_F1={action_f1:.3f}  "
                f"{'*' if patience_counter == 0 else ''}"
            )

        if patience_counter >= args.patience:
            console.print(f"  Early stopping at epoch {epoch + 1}")
            break

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model = model.to(device)

    stats = {
        "best_val_f1": best_val_f1,
        "best_epoch": best_epoch + 1,
        "total_epochs": epoch + 1,
    }
    return model, stats


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def predict_contacts(
    model: TemporalMaxer,
    seq: RallySequence,
    stride: int,
    device: torch.device,
    nms: bool = True,
) -> list[dict]:
    """Run inference on a rally and extract predicted contacts."""
    model.eval()
    features = torch.from_numpy(seq.features).float().unsqueeze(0)  # (1, T, dim)
    features = features.permute(0, 2, 1).to(device)  # (1, dim, T)

    with torch.no_grad():
        logits = model(features)  # (1, 7, T)
        preds = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # (T,)

    contacts: list[dict] = []
    i = 0
    while i < len(preds):
        cls = int(preds[i])
        if cls == 0:
            i += 1
            continue

        if nms:
            # Merge consecutive same-class predictions
            j = i + 1
            while j < len(preds) and preds[j] == cls:
                j += 1
            center = (i + j - 1) // 2
        else:
            center = i
            j = i + 1

        # Convert window index to rally-relative frame
        abs_frame = (seq.first_window + center) * stride + stride // 2
        rally_frame = abs_frame - seq.rally_start_frame

        contacts.append({
            "frame": rally_frame,
            "action": IDX_TO_ACTION[cls],
        })
        i = j

    return contacts


def evaluate_held_out(
    model: TemporalMaxer,
    held_out_seqs: list[RallySequence],
    stride: int,
    tolerance_frames: int,
    device: torch.device,
    nms: bool = True,
) -> dict:
    """Evaluate on held-out rallies, return aggregate metrics."""
    all_matches: list[MatchResult] = []
    all_unmatched: list[dict] = []

    for seq in held_out_seqs:
        pred_contacts = predict_contacts(model, seq, stride, device, nms=nms)
        matches, unmatched = match_contacts(
            seq.gt_labels, pred_contacts, tolerance=tolerance_frames
        )
        all_matches.extend(matches)
        all_unmatched.extend(unmatched)

    return compute_metrics(all_matches, all_unmatched)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="TemporalMaxer action detection experiment")
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--tolerance-ms", type=int, default=167)
    parser.add_argument("--label-spread", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--tmse-weight", type=float, default=0.05)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--no-nms", action="store_true")
    parser.add_argument(
        "--feature-dir",
        type=str,
        default="training_data/features",
        help="Directory with cached VideoMAE features (default: training_data/features)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps" if torch.backends.mps.is_available() else "cpu",
    )
    args = parser.parse_args()

    console.print("[bold]TemporalMaxer Action Detection Experiment[/]\n")

    # 1. Load data
    console.print("[bold]1. Loading data...[/]")
    rallies = load_rallies_with_action_gt()
    console.print(f"  {len(rallies)} rallies with action GT")

    video_ids = {r.video_id for r in rallies}
    video_info = load_video_info(video_ids)
    console.print(f"  {len(video_info)} videos with content_hash")

    cache = FeatureCache(cache_dir=Path(args.feature_dir))
    sequences = build_sequences(rallies, video_info, cache, args.stride, args.label_spread)
    console.print(f"  {len(sequences)} rally sequences built")

    if not sequences:
        console.print("[red]No sequences available. Ensure features are cached.[/]")
        return

    # 2. Split
    console.print("\n[bold]2. Train/held-out split...[/]")
    train_seqs = [s for s in sequences if video_split(s.video_id) == "train"]
    held_out_seqs = [s for s in sequences if video_split(s.video_id) == "held_out"]

    train_videos = len({s.video_id for s in train_seqs})
    held_out_videos = len({s.video_id for s in held_out_seqs})
    console.print(
        f"  Train: {len(train_seqs)} rallies ({train_videos} videos)  "
        f"Held-out: {len(held_out_seqs)} rallies ({held_out_videos} videos)"
    )

    # Label distribution
    all_train_labels = np.concatenate([s.labels for s in train_seqs])
    all_held_labels = np.concatenate([s.labels for s in held_out_seqs])
    train_contacts = int((all_train_labels > 0).sum())
    held_contacts = int((all_held_labels > 0).sum())
    console.print(
        f"  Train: {train_contacts} contact windows / {len(all_train_labels)} total  "
        f"Held-out: {held_contacts} / {len(all_held_labels)}"
    )

    if not train_seqs or not held_out_seqs:
        console.print("[red]Need both train and held-out sequences.[/]")
        return

    # 3. Train
    console.print(f"\n[bold]3. Training (epochs={args.epochs}, lr={args.lr}, "
                   f"hidden={args.hidden_dim}, layers={args.num_layers})...[/]")
    t0 = time.time()
    model, train_stats = train_model(train_seqs, held_out_seqs, args)
    train_time = time.time() - t0
    console.print(
        f"  Done in {train_time:.0f}s. Best epoch: {train_stats['best_epoch']}, "
        f"val contact F1: {train_stats['best_val_f1']:.3f}"
    )

    # 4. Evaluate
    console.print("\n[bold]4. Evaluating on held-out set...[/]")
    device = torch.device(args.device)
    fps = held_out_seqs[0].fps
    tolerance_frames = round(fps * args.tolerance_ms / 1000.0)
    console.print(f"  Tolerance: {args.tolerance_ms}ms = {tolerance_frames} frames @ {fps:.0f}fps")

    metrics = evaluate_held_out(
        model, held_out_seqs, args.stride, tolerance_frames, device, nms=not args.no_nms
    )

    # 5. Print results
    console.print("\n[bold]5. Results[/]\n")

    # Contact detection table
    t1 = Table(title="Contact Detection")
    t1.add_column("Metric")
    t1.add_column("TemporalMaxer", justify="right")
    t1.add_column("Current Pipeline", justify="right", style="dim")
    t1.add_row("Precision", f"{metrics['precision']:.1%}", "~90%")
    t1.add_row("Recall", f"{metrics['recall']:.1%}", "~90%")
    t1.add_row("F1", f"{metrics['f1']:.1%}", "~90%")
    t1.add_row("TP / FP / FN", f"{metrics['tp']} / {metrics['fp']} / {metrics['fn']}", "")
    console.print(t1)

    # Action accuracy table
    t2 = Table(title="\nAction Classification (matched contacts)")
    t2.add_column("Metric")
    t2.add_column("TemporalMaxer", justify="right")
    t2.add_column("Current Pipeline", justify="right", style="dim")
    t2.add_row("Accuracy", f"{metrics['action_accuracy']:.1%}", "~82%")
    console.print(t2)

    # Per-class F1 table
    t3 = Table(title="\nPer-Class F1")
    t3.add_column("Action")
    t3.add_column("P", justify="right")
    t3.add_column("R", justify="right")
    t3.add_column("F1", justify="right")
    t3.add_column("TP/FP/FN", justify="right")
    for action in ACTION_TYPES:
        cls = metrics.get("per_class", {}).get(action, {})
        if cls:
            t3.add_row(
                action,
                f"{cls['precision']:.0%}",
                f"{cls['recall']:.0%}",
                f"{cls['f1']:.0%}",
                f"{cls['tp']}/{cls['fp']}/{cls['fn']}",
            )
    console.print(t3)

    # Config summary
    t4 = Table(title="\nConfiguration")
    t4.add_column("Parameter")
    t4.add_column("Value", justify="right")
    t4.add_row("Stride", str(args.stride))
    t4.add_row("Label spread", f"±{args.label_spread}")
    t4.add_row("Hidden dim", str(args.hidden_dim))
    t4.add_row("Num layers", str(args.num_layers))
    t4.add_row("LR", str(args.lr))
    t4.add_row("TMSE weight", str(args.tmse_weight))
    t4.add_row("Focal gamma", str(args.focal_gamma))
    t4.add_row("Epochs (trained)", f"{train_stats['total_epochs']}")
    t4.add_row("Best epoch", f"{train_stats['best_epoch']}")
    t4.add_row("Training time", f"{train_time:.0f}s")
    t4.add_row("Train rallies", str(len(train_seqs)))
    t4.add_row("Held-out rallies", str(len(held_out_seqs)))
    t4.add_row("NMS", "off" if args.no_nms else "on")
    console.print(t4)


if __name__ == "__main__":
    main()
