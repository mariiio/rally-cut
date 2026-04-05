"""Train production MS-TCN++ and sequence-enriched contact GBM on all data.

Trains both models on ALL labeled data (no held-out fold — this is for
production deployment, not evaluation). Saves weights to:
  - weights/sequence_action/ms_tcn_production.pt  (MS-TCN++ checkpoint)
  - weights/contact_classifier/contact_classifier.pkl  (27-dim GBM)

Usage:
    cd analysis
    uv run python scripts/train_production_sequence.py
    uv run python scripts/train_production_sequence.py --epochs 80 --hidden-dim 64
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from rich.console import Console
from rich.table import Table
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent))

from eval_sequence_enriched import (  # noqa: E402
    RallyBundle,
    get_sequence_probs,
    prepare_rallies,
)
from train_contact_classifier import (  # noqa: E402
    extract_candidate_features,
    label_candidates,
)

from rallycut.actions.sequence_dataset import (  # noqa: E402
    SequenceActionDataset,
    collate_rally_sequences,
)
from rallycut.actions.trajectory_features import (  # noqa: E402
    FEATURE_DIM,
    NUM_CLASSES,
)
from rallycut.temporal.ms_tcn.model import MSTCN, MSTCNConfig
from rallycut.temporal.temporal_maxer.training import FocalLoss, compute_tmse_loss
from rallycut.tracking.contact_classifier import ContactClassifier

console = Console()

MSTCN_WEIGHTS_DIR = Path("weights/sequence_action")
MSTCN_WEIGHTS_PATH = MSTCN_WEIGHTS_DIR / "ms_tcn_production.pt"
GBM_WEIGHTS_PATH = Path("weights/contact_classifier/contact_classifier.pkl")


def train_mstcn_production(
    bundles: list[RallyBundle],
    args: argparse.Namespace,
) -> tuple[MSTCN, MSTCNConfig]:
    """Train MS-TCN++ on all data with 10% random val split for early stopping."""
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

    # 90/10 split for early stopping (NOT evaluation — all data contributes)
    random.shuffle(bundles)
    split = max(1, int(len(bundles) * 0.1))
    val_bundles = bundles[:split]
    train_bundles = bundles[split:]

    console.print(f"  MS-TCN++ train: {len(train_bundles)} rallies, val: {len(val_bundles)} rallies")

    # Class weights
    all_labels = np.concatenate([b.trajectory_labels for b in train_bundles])
    class_counts = np.bincount(all_labels, minlength=NUM_CLASSES).astype(float)
    class_counts = np.maximum(class_counts, 1.0)
    weights = torch.tensor(
        [class_counts.max() / c for c in class_counts], dtype=torch.float32, device=device,
    )
    weights[0] = weights[0].clamp(max=1.0)
    criterion = FocalLoss(weight=weights, gamma=args.focal_gamma, reduction="none")

    train_ds = SequenceActionDataset(
        [b.trajectory_features for b in train_bundles],
        [b.trajectory_labels for b in train_bundles],
        augment=True, noise_std=args.noise_std,
    )
    val_ds = SequenceActionDataset(
        [b.trajectory_features for b in val_bundles],
        [b.trajectory_labels for b in val_bundles],
    )
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          collate_fn=collate_rally_sequences)
    val_dl = DataLoader(val_ds, batch_size=1, shuffle=False,
                        collate_fn=collate_rally_sequences)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc, best_state, patience_ctr = 0.0, None, 0
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for feat, lbl, msk in train_dl:
            feat, lbl, msk = feat.to(device), lbl.to(device), msk.to(device)
            optimizer.zero_grad()
            total_loss = torch.tensor(0.0, device=device)
            for logits in model.forward_all_stages(feat, msk):
                ce = criterion(logits, lbl)
                m = msk.squeeze(1)
                ce = (ce * m).sum() / m.sum()
                total_loss = total_loss + ce + args.tmse_weight * compute_tmse_loss(logits, msk)
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                continue
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_loss += total_loss.item()
            n_batches += 1
        scheduler.step()

        # Validation accuracy (action frames only)
        model.eval()
        preds_all, targets_all = [], []
        with torch.no_grad():
            for feat, lbl, msk in val_dl:
                feat, lbl, msk = feat.to(device), lbl.to(device), msk.to(device)
                p = model(feat, msk).argmax(dim=1)
                m = msk.squeeze(1).bool()
                for b in range(feat.shape[0]):
                    v = m[b]
                    pi, ti = p[b][v].cpu().numpy(), lbl[b][v].cpu().numpy()
                    am = ti > 0
                    if am.any():
                        preds_all.extend(pi[am].tolist())
                        targets_all.extend(ti[am].tolist())

        acc = sum(1 for p, t in zip(preds_all, targets_all) if p == t) / len(targets_all) if targets_all else 0.0
        avg_loss = epoch_loss / max(1, n_batches)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            console.print(f"  Epoch {epoch + 1:3d}: loss={avg_loss:.4f}  val_acc={acc:.1%}")

        if acc > best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
        if patience_ctr >= args.patience:
            console.print(f"  Early stopping at epoch {epoch + 1} (patience={args.patience})")
            break

    if best_state:
        model.load_state_dict(best_state)
        model = model.to(device)

    console.print(f"  Best validation accuracy: {best_acc:.1%}")
    return model, config


def train_contact_gbm(
    bundles: list[RallyBundle],
    model: MSTCN,
    device: torch.device,
) -> ContactClassifier:
    """Train 27-dim contact GBM on all data with MS-TCN++ sequence probs."""
    all_features: list[np.ndarray] = []
    all_labels: list[int] = []

    console.print(f"\n[bold]Training contact GBM on {len(bundles)} rallies...[/bold]")

    for i, bundle in enumerate(bundles):
        seq_probs = get_sequence_probs(model, bundle, device)
        features, candidate_frames = extract_candidate_features(
            bundle.rally, sequence_probs=seq_probs,
        )
        if not features:
            continue

        labels = label_candidates(candidate_frames, bundle.gt_labels, tolerance=5)

        for feat, lbl in zip(features, labels):
            all_features.append(feat.to_array())
            all_labels.append(lbl)

        if (i + 1) % 50 == 0:
            console.print(f"  [{i + 1}/{len(bundles)}] candidates extracted")

    if not all_features:
        console.print("[red]No candidates extracted — cannot train GBM.[/red]")
        return ContactClassifier(threshold=0.40)

    x_mat = np.array(all_features)
    y = np.array(all_labels)

    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    console.print(f"  {len(y)} candidates: {n_pos} positive, {n_neg} negative")
    console.print(f"  Feature dimension: {x_mat.shape[1]}")

    clf = ContactClassifier(threshold=0.40)
    train_metrics = clf.train(x_mat, y)

    console.print(f"  Train F1: {train_metrics['train_f1']:.1%}")
    console.print(f"  Train Precision: {train_metrics['train_precision']:.1%}")
    console.print(f"  Train Recall: {train_metrics['train_recall']:.1%}")

    return clf


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train production MS-TCN++ and sequence-enriched contact GBM"
    )
    # MS-TCN++ hyperparams (matching eval_sequence_enriched.py defaults)
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
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    t0 = time.time()

    # Step 1: Load all rallies
    bundles = prepare_rallies(label_spread=2)
    if not bundles:
        console.print("[red]No rallies loaded.[/red]")
        return

    # Step 2: Train MS-TCN++
    console.print(f"\n[bold]Training MS-TCN++ on {len(bundles)} rallies...[/bold]")
    model, config = train_mstcn_production(bundles, args)

    # Save MS-TCN++ weights
    MSTCN_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
    }
    torch.save(checkpoint, MSTCN_WEIGHTS_PATH)
    console.print(f"\n[green]Saved MS-TCN++ to {MSTCN_WEIGHTS_PATH}[/green]")

    # Step 3: Train contact GBM with sequence probs
    device = torch.device(args.device)
    clf = train_contact_gbm(bundles, model, device)

    # Save contact GBM
    GBM_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    clf.save(str(GBM_WEIGHTS_PATH))
    console.print(f"[green]Saved contact GBM to {GBM_WEIGHTS_PATH}[/green]")

    elapsed = time.time() - t0
    console.print(f"\n[bold]Done in {elapsed:.0f}s[/bold]")

    # Summary
    summary = Table(title="Production Models Summary")
    summary.add_column("Model")
    summary.add_column("Path")
    summary.add_column("Details")
    summary.add_row(
        "MS-TCN++",
        str(MSTCN_WEIGHTS_PATH),
        f"feature_dim={config.feature_dim}, hidden={config.hidden_dim}, "
        f"stages={config.num_stages}, layers={config.num_layers}",
    )
    summary.add_row(
        "Contact GBM",
        str(GBM_WEIGHTS_PATH),
        "27-dim features (20 trajectory + 7 sequence probs), threshold=0.40",
    )
    console.print(summary)


if __name__ == "__main__":
    main()
