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


def wbce_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """Weighted Binary Cross-Entropy with focal-like weighting.

    Harder examples get higher weight:
    - Positives: weight = (1 - y_pred)^2
    - Negatives: weight = y_pred^2

    Args:
        y_true: Ground truth heatmaps (B, 3, H, W).
        y_pred: Predicted heatmaps (B, 3, H, W).
        sample_weights: Per-frame source weights (B, 3). Scales loss per frame
            based on label source quality (gold=1.0, filtered=0.7, none=0.3).
            If None, all frames weighted equally.
    """
    per_pixel = -(
        ((1 - y_pred) ** 2) * y_true * torch.log(torch.clamp(y_pred, 1e-7, 1))
        + (y_pred ** 2) * (1 - y_true) * torch.log(torch.clamp(1 - y_pred, 1e-7, 1))
    )
    if sample_weights is not None:
        # sample_weights: (B, 3) -> (B, 3, 1, 1) to broadcast over (H, W)
        per_pixel = per_pixel * sample_weights.unsqueeze(-1).unsqueeze(-1)
    return per_pixel.sum()


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


# Source-based loss weights: how much to trust each label source
# Gold GT (source=2): fully trusted, weight=1.0
# Filtered (source=1): high-quality pseudo-label from BallFilter pipeline
# None (source=0): no detection — still informative (ball not here)
SOURCE_WEIGHTS = {0: 0.3, 1: 0.7, 2: 1.0}


class TrackNetDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    """Dataset for TrackNet training from pseudo-label CSVs and frame images.

    Expected layout:
        data_dir/{rally_id}.csv          - Labels (frame_num, visible, x, y[, source])
        data_dir/images/{rally_id}/0.jpg - Frame images at 512x288

    CSV source column (optional): 0=none, 1=filtered, 2=gold.
    If missing, all labels are treated as source=1 (filtered).
    """

    def __init__(self, data_dir: Path, rally_ids: list[str], augment: bool = False) -> None:
        self.data_dir = data_dir
        self.augment = augment
        # Cache labels: rally_id -> (N, 3) array [visible, x, y]
        self._labels: dict[str, np.ndarray] = {}
        # Cache source weights: rally_id -> (N,) array of loss weights
        self._weights: dict[str, np.ndarray] = {}
        self.samples: list[tuple[str, int]] = []

        for rally_id in rally_ids:
            csv_path = data_dir / f"{rally_id}.csv"
            img_dir = data_dir / "images" / rally_id
            if not csv_path.exists() or not img_dir.exists():
                logger.warning(f"Missing data for rally {rally_id[:8]}, skipping")
                continue

            with open(csv_path) as f:
                reader = csv.reader(f)
                header = next(reader)
                has_source = len(header) >= 5  # frame_num, visible, x, y, source
                rows = []
                sources = []
                for r in reader:
                    rows.append([int(r[1]), float(r[2]), float(r[3])])
                    source = int(r[4]) if has_source and len(r) >= 5 else 1
                    sources.append(SOURCE_WEIGHTS.get(source, 0.7))

            if len(rows) < NUM_INPUT_FRAMES:
                continue

            self._labels[rally_id] = np.array(rows, dtype=np.float32)
            self._weights[rally_id] = np.array(sources, dtype=np.float32)

            # 3-frame sliding windows
            for i in range(len(rows) - NUM_INPUT_FRAMES + 1):
                self.samples.append((rally_id, i))

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _augment(
        frames: list[np.ndarray],
        labels: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Apply data augmentation to HWC uint8 frames and labels.

        Augmentations (applied consistently across all 3 frames in triplet):
        - Brightness: uniform ±15%
        - Contrast: uniform ±10%
        - Gamma correction: log-uniform [0.7, 1.4] (simulates exposure variation)
        - Color temperature: shift B/R channels ±10% (simulates warm/cool lighting)
        - Gaussian noise: sigma 0-8 (simulates camera sensor noise)
        - Horizontal flip: 50% chance (flips x labels)

        Args:
            frames: List of HWC uint8 frame arrays.
            labels: (N, 3) array of [visible, x, y] for each frame.
            rng: Numpy random generator for thread safety.

        Returns:
            Augmented (frames, labels) tuple. Labels array is copied if modified.
        """
        # Sample all augmentation parameters once (consistent across triplet)
        brightness = rng.uniform(0.85, 1.15)
        contrast = rng.uniform(0.90, 1.10)
        gamma = float(np.exp(rng.uniform(np.log(0.7), np.log(1.4))))
        # Color temperature: shift blue (cool) vs red (warm)
        temp_shift = rng.uniform(-0.10, 0.10)
        noise_sigma = rng.uniform(0, 8)

        # Precompute gamma LUT for efficiency (uint8 → uint8)
        inv_gamma = 1.0 / gamma
        gamma_lut = ((np.arange(256, dtype=np.float32) / 255.0) ** inv_gamma) * 255.0

        aug_frames = []
        for frame in frames:
            f = frame.astype(np.float32)
            # Brightness
            f = f * brightness
            # Contrast
            mean = f.mean()
            f = (f - mean) * contrast + mean
            f = np.clip(f, 0, 255)
            # Gamma correction via LUT
            f = gamma_lut[f.astype(np.uint8)]
            # Color temperature: BGR format — shift B down/up and R up/down
            f[:, :, 0] = f[:, :, 0] * (1 - temp_shift)  # Blue
            f[:, :, 2] = f[:, :, 2] * (1 + temp_shift)  # Red
            # Gaussian noise
            if noise_sigma > 0.5:
                noise = rng.normal(0, noise_sigma, f.shape).astype(np.float32)
                f = f + noise
            aug_frames.append(np.clip(f, 0, 255).astype(np.uint8))

        # Horizontal flip: 50% chance
        aug_labels = labels
        if rng.random() < 0.5:
            aug_frames = [frame[:, ::-1, :].copy() for frame in aug_frames]
            aug_labels = labels.copy()
            # Flip x coordinate: x → 1 - x (only for visible frames)
            visible_mask = aug_labels[:, 0] >= 1
            aug_labels[visible_mask, 1] = 1.0 - aug_labels[visible_mask, 1]

        return aug_frames, aug_labels

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rally_id, start = self.samples[idx]
        labels = self._labels[rally_id]
        weights = self._weights[rally_id]

        # Load 3 consecutive frames as HWC uint8
        raw_frames: list[np.ndarray] = []
        for i in range(NUM_INPUT_FRAMES):
            img_path = self.data_dir / "images" / rally_id / f"{start + i}.jpg"
            img = cv2.imread(str(img_path))
            if img is None:
                img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
            raw_frames.append(img)

        # Get labels and weights for this window
        window_labels = labels[start : start + NUM_INPUT_FRAMES]
        window_weights = weights[start : start + NUM_INPUT_FRAMES]

        # Apply augmentation (before CHW transpose)
        if self.augment:
            rng = np.random.default_rng()
            raw_frames, window_labels = self._augment(raw_frames, window_labels, rng)

        # HWC → CHW and concatenate → (9, H, W)
        chw_frames = [f.transpose(2, 0, 1) for f in raw_frames]
        x = torch.from_numpy(np.concatenate(chw_frames, axis=0)).float() / 255.0

        # Generate heatmaps for each frame → (3, H, W)
        heatmaps: list[np.ndarray] = []
        for i in range(NUM_INPUT_FRAMES):
            vis, lx, ly = window_labels[i]
            if vis >= 1:
                heatmaps.append(gen_heatmap(float(lx), float(ly)))
            else:
                heatmaps.append(np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32))

        y = torch.from_numpy(np.stack(heatmaps))
        w = torch.from_numpy(window_weights)  # (3,) per-frame source weights
        return x, y, w


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

    # Create datasets (augment training data only)
    train_dataset = TrackNetDataset(data_dir, train_rally_ids, augment=True)
    val_dataset = TrackNetDataset(data_dir, val_rally_ids, augment=False) if val_rally_ids else None

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

    # Mixed precision for faster training on Ampere+ GPUs (A10G, A100)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("Mixed precision (fp16) enabled")

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
        for x_batch, y_batch, w_batch in pbar:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            w_batch = w_batch.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                y_pred = model(x_batch)
                loss = wbce_loss(y_batch, y_pred, w_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
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
                for x_batch, y_batch, w_batch in val_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    w_batch = w_batch.to(device)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        y_pred = model(x_batch)
                        val_loss += wbce_loss(y_batch, y_pred, w_batch).item()
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
