"""Phase 3: ResNet single-frame formation classifier.

Fine-tune ResNet-18 (last layer) on the first frame of each rally to classify
serve side as "near" or "far". LOO-video CV.

The server stands behind the baseline, creating a distinctive formation.
Even if the server is off-screen (near) or occluded (far), the receiver
formation differs from the server's partner position.

Usage:
  uv run python scripts/eval_resnet_serve_side.py
  uv run python scripts/eval_resnet_serve_side.py --offset 0.5   # 0.5s after start
  uv run python scripts/eval_resnet_serve_side.py --sweep         # sweep offsets + multi-frame
  uv run python scripts/eval_resnet_serve_side.py --epochs 20     # more training
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from eval_score_tracking import (  # noqa: E402
    RallyData,
    evaluate_simple,
    load_score_gt,
    print_result,
)
from rallycut.evaluation.tracking.db import get_connection  # noqa: E402
from rallycut.evaluation.video_resolver import VideoResolver  # noqa: E402

DEVICE = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"


# ── Frame extraction ─────────────────────────────────────────────────────


def _resolve_videos(video_ids: set[str]) -> dict[str, Path]:
    resolver = VideoResolver()
    paths: dict[str, Path] = {}
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT id, s3_key, content_hash FROM videos WHERE id = ANY(%s)",
                    [list(video_ids)])
        for vid, s3_key, content_hash in cur.fetchall():
            if not s3_key or not content_hash:
                continue
            try:
                paths[vid] = resolver.resolve(s3_key, content_hash)
            except Exception as e:
                print(f"  WARN: {vid[:8]}: {e}")
    return paths


def _extract_frame_np(video_path: Path, time_ms: int) -> np.ndarray | None:
    """Extract a single frame as BGR numpy array."""
    cap = cv2.VideoCapture(str(video_path))
    try:
        cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
        ret, frame = cap.read()
        if not ret:
            return None
        return frame
    finally:
        cap.release()


def extract_all_frames(
    video_rallies: dict[str, list[RallyData]],
    video_paths: dict[str, Path],
    offsets_s: list[float],
) -> dict[str, list[np.ndarray]]:
    """Extract frames for all rallies at given offsets.

    Returns: rally_id -> list of BGR frames (one per offset).
    """
    frames: dict[str, list[np.ndarray]] = {}
    total = sum(len(rs) for rs in video_rallies.values())
    done = 0

    for vid, rallies in sorted(video_rallies.items()):
        vpath = video_paths.get(vid)
        if vpath is None:
            for r in rallies:
                done += 1
            continue

        # Open video once per video
        cap = cv2.VideoCapture(str(vpath))
        if not cap.isOpened():
            for r in rallies:
                done += 1
            continue

        for r in rallies:
            done += 1
            rally_frames = []
            for off in offsets_s:
                time_ms = int(r.start_ms + off * 1000)
                cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
                ret, frame = cap.read()
                if ret:
                    rally_frames.append(frame)
            if rally_frames:
                frames[r.rally_id] = rally_frames
            if done % 50 == 0:
                print(f"  [{done}/{total}] extracted frames")

        cap.release()

    print(f"  Extracted frames for {len(frames)}/{total} rallies")
    return frames


# ── Dataset ──────────────────────────────────────────────────────────────


class ServeFrameDataset(Dataset):
    def __init__(
        self,
        frames: list[np.ndarray],
        labels: list[int],
        transform: transforms.Compose | None = None,
        augment: bool = False,
    ):
        self.frames = frames
        self.labels = labels
        self.transform = transform
        self.augment = augment

    def __len__(self) -> int:
        return len(self.frames)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        frame = self.frames[idx]
        label = self.labels[idx]

        # BGR -> RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Resize to 224x224
        frame = cv2.resize(frame, (224, 224))

        if self.augment and np.random.random() > 0.5:
            # Horizontal flip with label flip
            frame = frame[:, ::-1, :].copy()
            label = 1 - label  # near <-> far

        # Normalize
        frame = frame.astype(np.float32) / 255.0
        if self.transform:
            tensor = self.transform(torch.from_numpy(frame).permute(2, 0, 1))
        else:
            tensor = torch.from_numpy(frame).permute(2, 0, 1)

        return tensor, label


# ── Training ─────────────────────────────────────────────────────────────


def _make_model() -> nn.Module:
    """Create ResNet-18 with frozen backbone and new binary head."""
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    # Freeze all layers except the final FC
    for param in model.parameters():
        param.requires_grad = False
    # Replace FC
    model.fc = nn.Linear(model.fc.in_features, 2)
    return model


NORMALIZE = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225],
)


def train_and_predict(
    train_frames: list[np.ndarray],
    train_labels: list[int],
    test_frames: list[np.ndarray],
    epochs: int = 10,
    lr: float = 1e-3,
    batch_size: int = 16,
) -> list[float]:
    """Train ResNet-18 and return predicted probabilities for near-side serve."""
    model = _make_model().to(DEVICE)

    train_ds = ServeFrameDataset(train_frames, train_labels, transform=NORMALIZE, augment=True)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

    # Predict on test set
    model.eval()
    test_ds = ServeFrameDataset(test_frames, [0] * len(test_frames), transform=NORMALIZE)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    all_probs = []
    with torch.no_grad():
        for batch_x, _ in test_loader:
            batch_x = batch_x.to(DEVICE)
            logits = model(batch_x)
            probs = torch.softmax(logits, dim=1)
            # prob of class 0 (near_serves)
            all_probs.extend(probs[:, 0].cpu().tolist())

    return all_probs


# ── LOO-Video CV ─────────────────────────────────────────────────────────


def run_loo_cv(
    video_rallies: dict[str, list[RallyData]],
    rally_frames: dict[str, list[np.ndarray]],
    frame_idx: int = 0,
    epochs: int = 10,
) -> dict[str, str | None]:
    """Run LOO-video CV. Returns rally_id -> predicted team."""
    video_ids = sorted(video_rallies.keys())
    predictions: dict[str, str | None] = {}

    for fold, held_out_vid in enumerate(video_ids):
        # Prepare train data (all other videos)
        train_frames_list = []
        train_labels = []
        for vid in video_ids:
            if vid == held_out_vid:
                continue
            for r in video_rallies[vid]:
                frames = rally_frames.get(r.rally_id)
                if frames is None or frame_idx >= len(frames):
                    continue
                # Label: 0 = near serves, 1 = far serves
                # Determine which is the GT server side
                if r.side_flipped:
                    label = 1 if r.gt_serving_team == "A" else 0  # flipped: A=far, B=near
                else:
                    label = 0 if r.gt_serving_team == "A" else 1  # normal: A=near, B=far
                train_frames_list.append(frames[frame_idx])
                train_labels.append(label)

        # Prepare test data
        test_rallies = []
        test_frames_list = []
        for r in video_rallies[held_out_vid]:
            frames = rally_frames.get(r.rally_id)
            if frames is None or frame_idx >= len(frames):
                predictions[r.rally_id] = None
                continue
            test_rallies.append(r)
            test_frames_list.append(frames[frame_idx])

        if not test_frames_list or not train_frames_list:
            for r in video_rallies[held_out_vid]:
                predictions[r.rally_id] = None
            continue

        # Train and predict
        probs = train_and_predict(train_frames_list, train_labels, test_frames_list, epochs=epochs)

        for r, prob_near in zip(test_rallies, probs):
            side = "near" if prob_near > 0.5 else "far"
            base = "A" if side == "near" else "B"
            if r.side_flipped:
                base = "B" if base == "A" else "A"
            predictions[r.rally_id] = base

        n_test = len(test_rallies)
        n_correct = sum(
            1 for r in test_rallies
            if predictions.get(r.rally_id) == r.gt_serving_team
        )
        print(f"  Fold {fold+1}/{len(video_ids)} ({held_out_vid[:8]}): "
              f"{n_correct}/{n_test} = {n_correct/max(1,n_test)*100:.1f}%  "
              f"(train={len(train_frames_list)})")

    return predictions


# ── Main ─────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description="ResNet serve-side classifier")
    parser.add_argument("--offset", type=float, default=0.0,
                        help="Seconds after rally start (default: 0)")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep offsets: -1.0, -0.5, 0, 0.5, 1.0, 2.0")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--multi-frame", action="store_true",
                        help="Concatenate 3 frames into 6-channel input")
    args = parser.parse_args()

    print("Loading score GT...")
    video_rallies = load_score_gt()
    total = sum(len(v) for v in video_rallies.values())
    print(f"Loaded {total} rallies across {len(video_rallies)} videos")

    print("\nResolving videos...")
    video_paths = _resolve_videos(set(video_rallies.keys()))
    print(f"Resolved {len(video_paths)}/{len(video_rallies)} videos")

    if args.sweep:
        offsets = [-1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
        summary = []
        for offset in offsets:
            print(f"\n{'='*60}")
            print(f"Offset: {offset}s, epochs={args.epochs}")
            print(f"{'='*60}")
            print("Extracting frames...")
            rally_frames = extract_all_frames(video_rallies, video_paths, [offset])
            predictions = run_loo_cv(video_rallies, rally_frames, frame_idx=0, epochs=args.epochs)

            def predict(rally: RallyData) -> str | None:
                return predictions.get(rally.rally_id)

            result = evaluate_simple(f"resnet18_offset={offset}s", predict, video_rallies)
            print_result(result)
            summary.append((offset, result.accuracy, result.coverage))

        print(f"\n{'offset':>8s}  {'acc':>7s}  {'coverage':>8s}")
        print("-" * 30)
        for off, acc, cov in summary:
            print(f"{off:>7.1f}s  {acc*100:6.1f}%  {cov*100:7.1f}%")
    else:
        offset = args.offset
        print(f"\nExtracting frames at offset={offset}s...")
        rally_frames = extract_all_frames(video_rallies, video_paths, [offset])

        print(f"\nRunning LOO-video CV (epochs={args.epochs})...")
        predictions = run_loo_cv(video_rallies, rally_frames, frame_idx=0, epochs=args.epochs)

        def predict(rally: RallyData) -> str | None:
            return predictions.get(rally.rally_id)

        result = evaluate_simple(f"resnet18_offset={offset}s", predict, video_rallies)
        print_result(result)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
