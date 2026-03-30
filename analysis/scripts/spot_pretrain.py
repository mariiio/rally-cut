"""Pretrain E2E-Spot on VNL-STES indoor volleyball dataset.

Downloads/loads VNL-STES data (1,028 rallies, ~5,900 events after
filtering 'score' class), trains the model, and saves a checkpoint
for finetuning on beach volleyball.

Usage:
    cd analysis
    uv run python scripts/spot_pretrain.py                         # Default 150 epochs
    uv run python scripts/spot_pretrain.py --epochs 50 --device cuda
    uv run python scripts/spot_pretrain.py --data-dir /path/to/vnl_stes
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from rich.console import Console

from rallycut.spotting.config import E2ESpotConfig
from rallycut.spotting.data.vnl_stes import VNL_STES_DIR, load_vnl_rallies, vnl_to_rally_infos
from rallycut.spotting.training.trainer import WEIGHTS_DIR, train

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pretrain E2E-Spot on VNL-STES")

    # Data
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help=f"VNL-STES data directory (default: {VNL_STES_DIR})",
    )
    parser.add_argument(
        "--splits",
        type=str,
        nargs="+",
        default=["train", "val"],
        help="VNL-STES splits to use for training (default: train val)",
    )
    parser.add_argument(
        "--val-split",
        type=str,
        default="test",
        help="VNL-STES split for validation (default: test)",
    )

    # Model
    parser.add_argument("--clip-length", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)

    # Training
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=4)

    # Device
    parser.add_argument("--device", type=str, default=None)

    return parser.parse_args()


def detect_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    device = detect_device(args.device)
    console.print(f"[bold]E2E-Spot VNL-STES Pretraining[/]  device={device}")

    data_dir = Path(args.data_dir) if args.data_dir else None

    # Load VNL-STES data
    console.print("\n[bold]Loading VNL-STES training data[/]")
    train_vnl = load_vnl_rallies(splits=args.splits, data_dir=data_dir)
    if not train_vnl:
        console.print("[red]No VNL-STES training data found.[/]")
        console.print(f"  Expected at: {data_dir or VNL_STES_DIR}")
        console.print("  Download from: https://hoangqnguyen.github.io/stes/")
        return

    console.print("\n[bold]Loading VNL-STES validation data[/]")
    val_vnl = load_vnl_rallies(splits=[args.val_split], data_dir=data_dir)
    if not val_vnl:
        console.print("[yellow]No VNL-STES validation data, using last 10% of train.[/]")
        split_idx = int(len(train_vnl) * 0.9)
        val_vnl = train_vnl[split_idx:]
        train_vnl = train_vnl[:split_idx]

    # Convert to RallyInfo format
    train_infos = vnl_to_rally_infos(train_vnl)
    val_infos = vnl_to_rally_infos(val_vnl)

    # Print class distribution
    from collections import Counter

    action_counts: Counter[str] = Counter()
    for r in train_vnl:
        for ev in r.events:
            action_counts[ev.mapped_label] += 1
    console.print(f"  Train class distribution: {dict(action_counts.most_common())}")
    console.print("  Note: 'dig' class has 0 instances (beach-only)")

    # Build config
    config = E2ESpotConfig()
    config.temporal.hidden_dim = args.hidden_dim
    config.temporal.num_layers = args.num_layers
    config.training.clip_length = args.clip_length
    config.training.epochs = args.epochs
    config.training.lr = args.lr
    config.training.batch_size = args.batch_size
    config.training.patience = args.patience
    config.training.num_workers = args.num_workers

    # Train
    console.print("\n[bold]Pretraining[/]")
    start_time = time.time()
    model, stats = train(train_infos, val_infos, config, device)

    # Save pretrained checkpoint
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    save_path = WEIGHTS_DIR / "pretrained.pt"
    torch.save(model.state_dict(), save_path)
    console.print(f"\n  Saved pretrained model to {save_path}")

    elapsed = time.time() - start_time
    console.print(
        f"\n  Best epoch: {stats['best_epoch']}  "
        f"Val contact F1: {stats['best_val_f1']:.3f}  "
        f"Time: {elapsed:.0f}s"
    )
    console.print(
        "\n  Next: finetune on beach data:\n"
        f"  uv run python scripts/spot_train.py "
        f"--pretrained {save_path} --freeze-backbone-epochs 10 "
        f"--lr 1e-4 --epochs 80"
    )


if __name__ == "__main__":
    main()
