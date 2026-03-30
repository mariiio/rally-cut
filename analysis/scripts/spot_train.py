"""Train E2E-Spot action spotting model on beach volleyball data.

Usage:
    cd analysis

    # Frame extraction (one-time, local)
    uv run python scripts/spot_train.py --extract-frames

    # Modal GPU training (recommended)
    uv run python scripts/spot_train.py --export-modal        # Upload frames + metadata
    modal run rallycut/spotting/training/modal_train.py        # Train on T4 GPU
    uv run python scripts/spot_train.py --download-modal       # Download trained model

    # Local training (slow on CPU/MPS)
    uv run python scripts/spot_train.py --epochs 150 --device mps

    # Evaluation
    uv run python scripts/spot_train.py --eval-only
    uv run python scripts/spot_train.py --eval-only --checkpoint weights/spotting/best.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from rich.console import Console

from rallycut.spotting.config import E2ESpotConfig
from rallycut.spotting.data.beach import FRAME_DIR, load_beach_rallies
from rallycut.spotting.evaluation.bridge import evaluate_model, print_metrics
from rallycut.spotting.model.e2e_spot import E2ESpot
from rallycut.spotting.training.trainer import WEIGHTS_DIR, train

console = Console()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train E2E-Spot for beach volleyball")

    # Data
    parser.add_argument(
        "--extract-frames",
        action="store_true",
        help="Pre-extract frames to disk before training",
    )
    parser.add_argument(
        "--frame-height",
        type=int,
        default=224,
        help="Target frame height for extraction (default: 224)",
    )

    # Modal
    parser.add_argument(
        "--export-modal",
        action="store_true",
        help="Export frames + metadata to Modal volume for GPU training",
    )
    parser.add_argument(
        "--download-modal",
        action="store_true",
        help="Download trained model from Modal volume",
    )

    # Model
    parser.add_argument("--clip-length", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.3)

    # Training
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--offset-weight", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=4)

    # Finetuning
    parser.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Path to pretrained checkpoint for finetuning",
    )
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=0,
        help="Freeze backbone for N epochs (0 = no freeze)",
    )
    parser.add_argument(
        "--backbone-lr-scale",
        type=float,
        default=0.1,
        help="LR multiplier for backbone (default: 0.1)",
    )

    # Eval
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training, only evaluate",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint path for eval-only mode",
    )
    parser.add_argument(
        "--tolerance-ms",
        type=int,
        default=167,
        help="Frame matching tolerance in ms (default: 167 = 5 frames @ 30fps)",
    )

    # Device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: cuda, mps, or cpu (auto-detected if omitted)",
    )

    return parser.parse_args()


def detect_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Modal data management
# ---------------------------------------------------------------------------


def export_to_modal() -> None:
    """Export frame data and metadata to Modal volume for GPU training."""
    import subprocess

    console.print("[bold]Exporting data to Modal volume[/]")

    # Load all rallies with GT (need DB access, local only)
    all_rallies = load_beach_rallies(split="all", extract_frames=False)
    if not all_rallies:
        console.print("[red]No rallies found. Run --extract-frames first.[/]")
        return

    console.print(f"  {len(all_rallies)} rallies with extracted frames")

    # Serialize metadata (labels, GT, split info) to JSON
    from rallycut.evaluation.split import video_split

    metadata: dict[str, list] = {"rallies": []}
    for rally in all_rallies:
        gt_labels = []
        for gt in rally.gt_labels:
            gt_labels.append({
                "frame": gt.frame,
                "action": gt.action,
                "player_track_id": gt.player_track_id,
                "ball_x": gt.ball_x,
                "ball_y": gt.ball_y,
            })

        metadata["rallies"].append({
            "rally_id": rally.rally_id,
            "video_id": rally.video_id,
            "frame_count": rally.frame_count,
            "fps": rally.fps,
            "start_ms": rally.start_ms,
            "labels": rally.labels.tolist(),
            "offsets": rally.offsets.tolist(),
            "gt_labels": gt_labels,
            "split": video_split(rally.video_id),
        })

    meta_path = FRAME_DIR.parent / "spotting_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f)
    console.print(f"  Saved metadata: {meta_path}")

    train_count = sum(1 for r in metadata["rallies"] if r["split"] == "train")
    held_count = sum(1 for r in metadata["rallies"] if r["split"] == "held_out")
    console.print(f"  Train: {train_count}, Held-out: {held_count}")

    # Upload to Modal volume
    console.print("\n[bold]Uploading frames to Modal volume[/]")
    console.print(f"  Source: {FRAME_DIR}")
    console.print("  This may take several minutes for ~2.8GB of frames...")

    # Upload metadata
    subprocess.run(
        ["modal", "volume", "put", "rallycut-spotting", str(meta_path), "spotting/metadata.json"],
        check=True,
    )
    console.print("  Uploaded metadata.json")

    # Upload frames directory
    subprocess.run(
        ["modal", "volume", "put", "-f", "rallycut-spotting",
         str(FRAME_DIR) + "/", "spotting/frames/"],
        check=True,
    )
    console.print("  Uploaded frames/")

    console.print("\n[green]Done! Now run:[/]")
    console.print("  modal run rallycut/spotting/training/modal_train.py")


def download_from_modal() -> None:
    """Download trained model from Modal volume."""
    import subprocess

    console.print("[bold]Downloading model from Modal volume[/]")

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = str(WEIGHTS_DIR / "best.pt")

    subprocess.run(
        ["modal", "volume", "get", "rallycut-spotting",
         "spotting/checkpoints/best.pt", dest],
        check=True,
    )
    console.print(f"  Saved to {dest}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    # Modal data management (doesn't need device)
    if args.export_modal:
        export_to_modal()
        return
    if args.download_modal:
        download_from_modal()
        return

    device = detect_device(args.device)
    console.print(f"[bold]E2E-Spot Action Spotting[/]  device={device}")

    # Build config from args
    config = E2ESpotConfig()
    config.temporal.hidden_dim = args.hidden_dim
    config.temporal.num_layers = args.num_layers
    config.temporal.dropout = args.dropout
    config.training.clip_length = args.clip_length
    config.training.epochs = args.epochs
    config.training.lr = args.lr
    config.training.batch_size = args.batch_size
    config.training.patience = args.patience
    config.training.focal_gamma = args.focal_gamma
    config.training.offset_weight = args.offset_weight
    config.training.num_workers = args.num_workers
    config.training.freeze_backbone_epochs = args.freeze_backbone_epochs
    config.training.backbone_lr_scale = args.backbone_lr_scale

    start_time = time.time()

    if args.eval_only:
        # Eval-only mode
        checkpoint = args.checkpoint or str(WEIGHTS_DIR / "best.pt")
        console.print(f"\n[bold]Loading model from {checkpoint}[/]")

        model = E2ESpot(config).to(device)
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state)

        console.print("\n[bold]Loading held-out rallies[/]")
        val_rallies = load_beach_rallies(
            split="held_out",
            extract_frames=args.extract_frames,
            target_height=args.frame_height,
        )

        console.print(f"\n[bold]Evaluating on {len(val_rallies)} held-out rallies[/]")
        metrics = evaluate_model(model, val_rallies, device, config, args.tolerance_ms)
        print_metrics(metrics)
    else:
        # Training mode
        console.print("\n[bold]Loading training rallies[/]")
        train_rallies = load_beach_rallies(
            split="train",
            extract_frames=args.extract_frames,
            target_height=args.frame_height,
        )

        console.print("\n[bold]Loading held-out rallies[/]")
        val_rallies = load_beach_rallies(
            split="held_out",
            extract_frames=args.extract_frames,
            target_height=args.frame_height,
        )

        if not train_rallies:
            console.print("[red]No training rallies found. Run with --extract-frames first.[/]")
            return
        if not val_rallies:
            console.print("[red]No validation rallies found.[/]")
            return

        console.print("\n[bold]Training[/]")
        pretrained = Path(args.pretrained) if args.pretrained else None
        model, stats = train(train_rallies, val_rallies, config, device, pretrained)

        console.print(f"\n  Best epoch: {stats['best_epoch']}  "
                      f"Val contact F1: {stats['best_val_f1']:.3f}")

        # Full evaluation on held-out
        console.print(f"\n[bold]Final evaluation on {len(val_rallies)} held-out rallies[/]")
        metrics = evaluate_model(model, val_rallies, device, config, args.tolerance_ms)
        print_metrics(metrics)

    elapsed = time.time() - start_time
    console.print(f"\n  Total time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
