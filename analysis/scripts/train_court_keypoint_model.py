#!/usr/bin/env python3
"""Train YOLO-pose model for court keypoint detection.

Fine-tunes a YOLO-pose model (COCO person-pose pretrained) on beach volleyball
court keypoints. Predicts 6 keypoints:
  0: near-left, 1: near-right, 2: far-right, 3: far-left (corners)
  4: center-left, 5: center-right (net-sideline intersections)

Requires dataset exported by export_court_keypoint_dataset.py.

Usage:
    uv run python scripts/train_court_keypoint_model.py
    uv run python scripts/train_court_keypoint_model.py --data datasets/court_keypoints_v3/court_keypoints.yaml
    uv run python scripts/train_court_keypoint_model.py --epochs 200
    uv run python scripts/train_court_keypoint_model.py --no-freeze  # Train full model
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

WEIGHTS_DIR = Path("weights/court_keypoint")
DEFAULT_DATA = Path("datasets/court_keypoints/court_keypoints.yaml")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train YOLO11s-pose for court keypoint detection"
    )
    parser.add_argument(
        "--data", type=Path, default=DEFAULT_DATA,
        help="Path to dataset YAML",
    )
    parser.add_argument(
        "--epochs", type=int, default=200,
        help="Training epochs (default: 200)",
    )
    parser.add_argument(
        "--batch", type=int, default=8,
        help="Batch size (default: 8)",
    )
    parser.add_argument(
        "--imgsz", type=int, default=640,
        help="Image size (default: 640)",
    )
    parser.add_argument(
        "--model", type=str, default="yolo11s-pose.pt",
        help="Pretrained model to fine-tune (default: yolo11s-pose.pt)",
    )
    parser.add_argument(
        "--no-freeze", action="store_true",
        help="Train full model (don't freeze backbone)",
    )
    parser.add_argument(
        "--name", type=str, default="train",
        help="Run name for output directory (default: train)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint",
    )
    args = parser.parse_args()

    if not args.data.exists():
        print(f"Dataset YAML not found: {args.data}")
        print("Run export_court_keypoint_dataset.py first to create the dataset.")
        return

    import torch
    from ultralytics import YOLO

    # Auto-detect device: MPS (Apple Silicon) > CUDA > CPU
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "0"
    else:
        device = "cpu"

    # Resume from last checkpoint if available, otherwise start fresh
    last_pt = Path(f"runs/court_keypoint/{args.name}/weights/last.pt").resolve()
    if args.resume and last_pt.exists():
        model = YOLO(str(last_pt))
        print(f"Resuming from {last_pt}")
        print(f"Device: {device}")

        results = model.train(
            resume=True,
            device=device,
        )
    else:
        model = YOLO(args.model)
        print(f"Loaded {args.model} (pretrained → fine-tune to 6 court kpts)")
        print(f"Dataset: {args.data}")
        print(f"Epochs: {args.epochs}, Batch: {args.batch}, Image size: {args.imgsz}")
        print(f"Freeze backbone: {not args.no_freeze}, Device: {device}")

        # Train with augmentation tuned for single-court images
        results = model.train(
            data=str(args.data),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=device,
            # Freeze backbone layers (small dataset, avoid overfitting)
            freeze=0 if args.no_freeze else 10,
            # Early stopping
            patience=30,
            # Augmentation: no mosaic (single court per image), conservative transforms
            mosaic=0.0,
            fliplr=0.5,
            flipud=0.0,
            degrees=5.0,
            translate=0.1,
            scale=0.2,
            hsv_h=0.015,
            hsv_s=0.5,
            hsv_v=0.3,
            # Flip index: swap left/right corners on horizontal flip
            # near-left(0) <-> near-right(1), far-left(3) <-> far-right(2)
            # (defined in dataset YAML as flip_idx)
            # Other settings
            workers=4,
            project=str(Path("runs/court_keypoint").resolve()),
            name=args.name,
            exist_ok=True,
            # Checkpoint every 10 epochs for sleep/crash resilience
            save_period=10,
            verbose=True,
        )

    # Copy best model to weights directory
    best_path = Path(f"runs/court_keypoint/{args.name}/weights/best.pt").resolve()
    if best_path.exists():
        WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
        dest = WEIGHTS_DIR / "court_keypoint_best.pt"
        shutil.copy2(best_path, dest)
        print(f"\nBest model saved to {dest}")
        print(f"  Metrics: {results.results_dict if hasattr(results, 'results_dict') else 'see runs/'}")
    else:
        print(f"\nWarning: best.pt not found at {best_path}")
        print("Check runs/court_keypoint/train/ for training output.")


if __name__ == "__main__":
    main()
