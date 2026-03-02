#!/usr/bin/env python3
"""Train YOLO11s-pose model for court keypoint detection.

Fine-tunes yolo11s-pose (COCO person-pose pretrained) on beach volleyball
court corners. Predicts 4 keypoints: near-left, near-right, far-right, far-left.

Requires dataset exported by export_court_keypoint_dataset.py.

Usage:
    uv run python scripts/train_court_keypoint_model.py
    uv run python scripts/train_court_keypoint_model.py --data datasets/court_keypoints/court_keypoints.yaml
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
        "--no-freeze", action="store_true",
        help="Train full model (don't freeze backbone)",
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

    # Load pretrained pose model
    model = YOLO("yolo11s-pose.pt")
    print("Loaded yolo11s-pose.pt (COCO pretrained, 17 kpts → fine-tune to 4 kpts)")
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
        name="train",
        exist_ok=True,
        resume=args.resume,
        verbose=True,
    )

    # Copy best model to weights directory
    best_path = Path("runs/court_keypoint/train/weights/best.pt").resolve()
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
