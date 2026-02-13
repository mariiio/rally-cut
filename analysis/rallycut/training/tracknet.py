"""TrackNetV2 ball detection model and training.

Architecture: VGG16-encoder + UNet-decoder for ball heatmap prediction.
Input: 3 consecutive RGB frames (9 channels) at 288x512.
Output: 3 heatmap channels (one per frame) at 288x512.

Based on: https://github.com/ChgygLin/TrackNetV2-pytorch
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)

# TrackNet standard dimensions
IMG_HEIGHT = 288
IMG_WIDTH = 512
HEATMAP_RADIUS = 2.5
NUM_INPUT_FRAMES = 3


class TrackNet(nn.Module):
    """TrackNetV2 ball detection network.

    VGG16-like encoder with UNet-style decoder and skip connections.
    """

    def __init__(self, input_channels: int = 9, output_channels: int = 3) -> None:
        super().__init__()
        # Encoder (VGG16-like blocks)
        self.enc1 = self._conv_block([input_channels, 64, 64])
        self.enc2 = self._conv_block([64, 128, 128])
        self.enc3 = self._conv_block([128, 256, 256, 256])
        self.enc4 = self._conv_block([256, 512, 512, 512])
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder (UNet-style with skip connections)
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.dec3 = self._conv_block([512 + 256, 256, 256, 256])
        self.dec2 = self._conv_block([256 + 128, 128, 128])
        self.dec1 = self._conv_block([128 + 64, 64, 64])
        self.final = nn.Sequential(nn.Conv2d(64, output_channels, 1), nn.Sigmoid())

    @staticmethod
    def _conv_block(channels: list[int]) -> nn.Sequential:
        layers: list[nn.Module] = []
        for i in range(len(channels) - 1):
            layers.extend([
                nn.Conv2d(channels[i], channels[i + 1], 3, padding="same"),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(channels[i + 1]),
            ])
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up(d2), e1], dim=1))
        result: torch.Tensor = self.final(d1)
        return result


def wbce_loss(y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    """Weighted Binary Cross-Entropy with focal-like weighting.

    Harder examples get higher weight:
    - Positives: weight = (1 - y_pred)^2
    - Negatives: weight = y_pred^2
    """
    return -(
        ((1 - y_pred) ** 2) * y_true * torch.log(torch.clamp(y_pred, 1e-7, 1))
        + (y_pred ** 2) * (1 - y_true) * torch.log(torch.clamp(1 - y_pred, 1e-7, 1))
    ).sum()


def gen_heatmap(
    x_norm: float,
    y_norm: float,
    width: int = IMG_WIDTH,
    height: int = IMG_HEIGHT,
    radius: float = HEATMAP_RADIUS,
) -> np.ndarray:
    """Generate circular binary heatmap at normalized coordinates."""
    hm = np.zeros((height, width), dtype=np.float32)
    cx = max(0, min(int(x_norm * width), width - 1))
    cy = max(0, min(int(y_norm * height), height - 1))
    y_grid, x_grid = np.ogrid[-cy : height - cy, -cx : width - cx]
    mask = x_grid * x_grid + y_grid * y_grid <= radius * radius
    hm[mask] = 1.0
    return hm


class TrackNetDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset for TrackNet training from pseudo-label CSVs and frame images.

    Expected layout:
        data_dir/{rally_id}.csv          - Labels (frame_num, visible, x, y)
        data_dir/images/{rally_id}/0.jpg - Frame images at 512x288
    """

    def __init__(self, data_dir: Path, rally_ids: list[str]) -> None:
        self.data_dir = data_dir
        # Cache labels in memory: rally_id -> (N, 3) array [visible, x, y]
        self._labels: dict[str, np.ndarray] = {}
        self.samples: list[tuple[str, int]] = []

        for rally_id in rally_ids:
            csv_path = data_dir / f"{rally_id}.csv"
            img_dir = data_dir / "images" / rally_id
            if not csv_path.exists() or not img_dir.exists():
                logger.warning(f"Missing data for rally {rally_id[:8]}, skipping")
                continue

            with open(csv_path) as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                rows = [[int(r[1]), float(r[2]), float(r[3])] for r in reader]

            if len(rows) < NUM_INPUT_FRAMES:
                continue

            labels = np.array(rows, dtype=np.float32)
            self._labels[rally_id] = labels

            # 3-frame sliding windows
            for i in range(len(labels) - NUM_INPUT_FRAMES + 1):
                self.samples.append((rally_id, i))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rally_id, start = self.samples[idx]
        labels = self._labels[rally_id]

        # Load 3 consecutive frames and concatenate channels → (9, H, W)
        frames: list[np.ndarray] = []
        for i in range(NUM_INPUT_FRAMES):
            img_path = self.data_dir / "images" / rally_id / f"{start + i}.jpg"
            img = cv2.imread(str(img_path))
            if img is None:
                img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
            frames.append(img.transpose(2, 0, 1))  # HWC → CHW

        x = torch.from_numpy(np.concatenate(frames, axis=0)).float() / 255.0

        # Generate heatmaps for each frame → (3, H, W)
        heatmaps: list[np.ndarray] = []
        for i in range(NUM_INPUT_FRAMES):
            vis, lx, ly = labels[start + i]
            if vis >= 1:
                heatmaps.append(gen_heatmap(float(lx), float(ly)))
            else:
                heatmaps.append(np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32))

        y = torch.from_numpy(np.stack(heatmaps))
        return x, y


@dataclass
class TrackNetConfig:
    """Training configuration."""

    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 0.99  # Adadelta default from original paper
    lr_gamma: float = 0.9  # ExponentialLR per-epoch decay
    num_workers: int = 4
    save_every: int = 3  # Save weights every N epochs
    checkpoint_every: int = 5  # Full checkpoint every N epochs


@dataclass
class TrackNetResult:
    """Training result."""

    best_val_loss: float
    best_epoch: int
    model_path: str
    precision: float
    recall: float
    f1: float


def _compute_detection_metrics(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[int, int, int]:
    """Compute frame-level detection TP, FP, FN.

    Returns (tp, fp, fn) counts across all frames in the batch.
    """
    tp = fp = fn = 0
    batch_size = y_true.size(0)
    for b in range(batch_size):
        for f in range(NUM_INPUT_FRAMES):
            gt_has_ball = y_true[b, f].max().item() > threshold
            pred_has_ball = y_pred[b, f].max().item() > threshold
            if gt_has_ball and pred_has_ball:
                tp += 1
            elif pred_has_ball and not gt_has_ball:
                fp += 1
            elif gt_has_ball and not pred_has_ball:
                fn += 1
    return tp, fp, fn


def train_tracknet(
    data_dir: Path,
    output_dir: Path,
    config: TrackNetConfig,
    train_rally_ids: list[str],
    val_rally_ids: list[str],
    resume_checkpoint: Path | None = None,
) -> TrackNetResult:
    """Train TrackNet model on pseudo-labeled ball tracking data.

    Args:
        data_dir: Directory with {rally_id}.csv and images/{rally_id}/.
        output_dir: Directory to save model weights and checkpoints.
        config: Training configuration.
        train_rally_ids: Rally IDs for training.
        val_rally_ids: Rally IDs for validation.
        resume_checkpoint: Path to checkpoint for resuming interrupted training.

    Returns:
        TrackNetResult with best metrics and model path.
    """
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create datasets
    train_dataset = TrackNetDataset(data_dir, train_rally_ids)
    val_dataset = TrackNetDataset(data_dir, val_rally_ids) if val_rally_ids else None

    if len(train_dataset) == 0:
        raise ValueError(
            f"No training samples found in {data_dir}. "
            "Check that CSV files and images/ directory exist for the given rally IDs."
        )

    print(f"Train: {len(train_dataset)} windows from {len(train_rally_ids)} rallies")
    if val_dataset:
        print(f"Val: {len(val_dataset)} windows from {len(val_rally_ids)} rallies")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = None
    if val_dataset and len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

    # Model, optimizer, scheduler
    model = TrackNet().to(device)
    optimizer = torch.optim.Adadelta(model.parameters(), lr=config.learning_rate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.lr_gamma)

    start_epoch = 0
    best_val_loss = float("inf")
    best_f1 = 0.0

    # Resume from checkpoint
    if resume_checkpoint and resume_checkpoint.exists():
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["net"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["lr_scheduler"])
        start_epoch = ckpt["epoch"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        best_f1 = ckpt.get("best_f1", 0.0)
        print(f"Resumed from epoch {start_epoch}")

    # Output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    best_metrics: dict[str, float] = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    best_epoch = start_epoch
    avg_val_loss = float("inf")
    val_metrics: dict[str, float] = {}

    for epoch in range(start_epoch, config.epochs):
        # ── Train ──
        model.train()
        train_loss = 0.0
        train_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.epochs}")
        for x_batch, y_batch in pbar:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = wbce_loss(y_batch, y_pred)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.0f}")

        scheduler.step()
        avg_train_loss = train_loss / max(train_batches, 1)

        # ── Validate ──
        avg_val_loss = float("inf")
        val_metrics = {}
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            total_tp = total_fp = total_fn = 0

            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    y_pred = model(x_batch)
                    val_loss += wbce_loss(y_batch, y_pred).item()
                    val_batches += 1
                    tp, fp, fn = _compute_detection_metrics(y_batch, y_pred)
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn

            avg_val_loss = val_loss / max(val_batches, 1)
            precision = total_tp / max(total_tp + total_fp, 1)
            recall = total_tp / max(total_tp + total_fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-7)
            val_metrics = {"precision": precision, "recall": recall, "f1": f1}

        lr = scheduler.get_last_lr()[0]
        msg = (
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"Train: {avg_train_loss:.0f} | "
            f"Val: {avg_val_loss:.0f} | "
            f"LR: {lr:.6f}"
        )
        if val_metrics:
            msg += (
                f" | P: {val_metrics['precision']:.3f}"
                f" R: {val_metrics['recall']:.3f}"
                f" F1: {val_metrics['f1']:.3f}"
            )
        print(msg)

        # ── Save weights ──
        if (epoch + 1) % config.save_every == 0:
            torch.save(model.state_dict(), output_dir / "last.pt")
            current_f1 = val_metrics.get("f1", 0.0)
            if current_f1 > best_f1 or (current_f1 == best_f1 and avg_val_loss < best_val_loss):
                best_f1 = current_f1
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                best_metrics = val_metrics or best_metrics
                torch.save(model.state_dict(), output_dir / "best.pt")
                print(f"  -> New best model (F1: {best_f1:.3f}, val_loss: {best_val_loss:.0f})")

        # ── Full checkpoint for resume ──
        if (epoch + 1) % config.checkpoint_every == 0:
            ckpt_data = {
                "net": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "best_f1": best_f1,
            }
            torch.save(ckpt_data, ckpt_dir / f"ckpt_{epoch + 1}.pt")
            torch.save(ckpt_data, ckpt_dir / "ckpt_latest.pt")

    # Save final weights
    torch.save(model.state_dict(), output_dir / "last.pt")
    final_f1 = val_metrics.get("f1", 0.0) if val_metrics else 0.0
    if final_f1 > best_f1:
        best_f1 = final_f1
        best_val_loss = avg_val_loss
        best_epoch = config.epochs
        best_metrics = val_metrics or best_metrics
        torch.save(model.state_dict(), output_dir / "best.pt")

    best_path = output_dir / "best.pt"
    if not best_path.exists():
        # No validation or never improved — save current as best
        torch.save(model.state_dict(), best_path)

    return TrackNetResult(
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        model_path=str(best_path),
        precision=best_metrics.get("precision", 0.0),
        recall=best_metrics.get("recall", 0.0),
        f1=best_metrics.get("f1", 0.0),
    )
