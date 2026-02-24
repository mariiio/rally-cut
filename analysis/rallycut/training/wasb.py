"""WASB HRNet ball detection fine-tuning.

Architecture: HRNet multi-resolution backbone for ball heatmap prediction.
Input: 3 consecutive RGB frames (9 channels) at 288x512, ImageNet-normalized.
Output: 3 heatmap channels (one per frame) at 288x512.

Fine-tunes pretrained WASB volleyball weights on pseudo-labels
to improve ball detection on beach volleyball specifically.

Based on: https://github.com/nttcom/WASB-SBDT (BMVC 2023)
"""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader, Dataset

from rallycut.tracking.wasb_model import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_HEIGHT,
    IMG_WIDTH,
    NUM_INPUT_FRAMES,
    WASB_CONFIG,
    HRNet,
)

logger = logging.getLogger(__name__)

HEATMAP_RADIUS = 2.5


@dataclass
class WASBConfig:
    """Training configuration for WASB fine-tuning."""

    epochs: int = 30
    batch_size: int = 8
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    lr_milestones: list[int] = field(default_factory=lambda: [10, 20])
    lr_gamma: float = 0.1
    heatmap_radius: float = 2.5
    num_workers: int = 4
    save_every: int = 1
    checkpoint_every: int = 3
    # Early stopping: stop if val F1 stays at 0 for this many epochs after
    # initially being > 0 (detects val collapse like TrackNet R1)
    early_stop_patience: int = 5
    # Augmentation
    aug_color_jitter: float = 0.1
    aug_hflip_prob: float = 0.5
    aug_crop_prob: float = 0.5
    aug_crop_max_rescale: float = 0.125


@dataclass
class WASBResult:
    """Training result."""

    best_val_loss: float
    best_epoch: int
    model_path: str
    precision: float
    recall: float
    f1: float
    accuracy: float


def gen_wasb_heatmap(
    x_norm: float,
    y_norm: float,
    width: int = IMG_WIDTH,
    height: int = IMG_HEIGHT,
    radius: float = HEATMAP_RADIUS,
) -> np.ndarray:
    """Generate binary disk heatmap at normalized coordinates.

    Matches WASB original: pixels within `radius` of center = 1.0, else 0.0.
    """
    hm = np.zeros((height, width), dtype=np.float32)
    cx = max(0, min(int(x_norm * width), width - 1))
    cy = max(0, min(int(y_norm * height), height - 1))
    y_grid, x_grid = np.ogrid[-cy : height - cy, -cx : width - cx]
    mask = x_grid * x_grid + y_grid * y_grid <= radius * radius
    hm[mask] = 1.0
    return hm


def wbce_loss(
    y_pred_logits: torch.Tensor,
    y_true: torch.Tensor,
) -> torch.Tensor:
    """Weighted Binary Cross-Entropy with focal-like weighting (WASB formula).

    Formula: mean(-(1-p)^2 * y * log(p) - p^2 * (1-y) * log(1-p))

    Takes raw logits (WASB model has NO sigmoid in forward), applies sigmoid
    internally using logsigmoid for numerical stability.
    Uses .mean() reduction (NOT .sum() — sum caused NaN in TrackNet R3).

    Args:
        y_pred_logits: Raw logits from model (B, 3, H, W).
        y_true: Ground truth heatmaps (B, 3, H, W).
    """
    # Numerically stable computation using logsigmoid
    log_p = functional.logsigmoid(y_pred_logits)
    log_1_minus_p = functional.logsigmoid(-y_pred_logits)
    p = torch.sigmoid(y_pred_logits)

    pos_weight = (1 - p) ** 2
    neg_weight = p ** 2

    per_pixel = -(pos_weight * y_true * log_p + neg_weight * (1 - y_true) * log_1_minus_p)
    return per_pixel.mean()


class WASBDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Dataset for WASB training from label CSVs and frame images.

    Expected layout:
        data_dir/{rally_id}.csv          - Labels (frame_num, visible, x, y)
        data_dir/images/{rally_id}/0.jpg - Frame images at 512x288
    """

    def __init__(
        self,
        data_dir: Path,
        rally_ids: list[str],
        augment: bool = False,
        config: WASBConfig | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.augment = augment
        self.config = config or WASBConfig()
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
                next(reader)  # skip header
                rows = []
                for r in reader:
                    rows.append([int(r[1]), float(r[2]), float(r[3])])

            if len(rows) < NUM_INPUT_FRAMES:
                continue

            self._labels[rally_id] = np.array(rows, dtype=np.float32)

            # 3-frame sliding windows
            for i in range(len(rows) - NUM_INPUT_FRAMES + 1):
                self.samples.append((rally_id, i))

    def __len__(self) -> int:
        return len(self.samples)

    def _augment_frames(
        self,
        frames: list[np.ndarray],
        labels: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[list[np.ndarray], np.ndarray]:
        """Apply augmentations consistently across all 3 frames."""
        cfg = self.config
        aug_labels = labels.copy()

        # Color jitter (brightness, contrast, saturation)
        if cfg.aug_color_jitter > 0:
            jitter = cfg.aug_color_jitter
            brightness = rng.uniform(1 - jitter, 1 + jitter)
            contrast = rng.uniform(1 - jitter, 1 + jitter)
            aug_frames = []
            for frame in frames:
                f = frame.astype(np.float32)
                f = f * brightness
                mean = f.mean()
                f = (f - mean) * contrast + mean
                aug_frames.append(np.clip(f, 0, 255).astype(np.uint8))
            frames = aug_frames

        # Horizontal flip
        if rng.random() < cfg.aug_hflip_prob:
            frames = [frame[:, ::-1, :].copy() for frame in frames]
            visible_mask = aug_labels[:, 0] >= 1
            aug_labels[visible_mask, 1] = 1.0 - aug_labels[visible_mask, 1]

        # Random crop
        if rng.random() < cfg.aug_crop_prob:
            h, w = frames[0].shape[:2]
            max_rescale = cfg.aug_crop_max_rescale
            scale = rng.uniform(1 - max_rescale, 1.0)
            crop_h = int(h * scale)
            crop_w = int(w * scale)
            y0 = rng.integers(0, h - crop_h + 1)
            x0 = rng.integers(0, w - crop_w + 1)

            cropped_frames = []
            for frame in frames:
                crop = frame[y0 : y0 + crop_h, x0 : x0 + crop_w]
                resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
                cropped_frames.append(resized)
            frames = cropped_frames

            # Adjust coordinates
            for i in range(len(aug_labels)):
                if aug_labels[i, 0] >= 1:
                    # Transform normalized coords to crop space
                    new_x = (aug_labels[i, 1] * w - x0) / crop_w
                    new_y = (aug_labels[i, 2] * h - y0) / crop_h
                    if 0 <= new_x <= 1 and 0 <= new_y <= 1:
                        aug_labels[i, 1] = new_x
                        aug_labels[i, 2] = new_y
                    else:
                        # Ball outside crop — mark invisible
                        aug_labels[i, 0] = 0

        return frames, aug_labels

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rally_id, start = self.samples[idx]
        labels = self._labels[rally_id]

        # Load 3 consecutive frames as HWC uint8 (BGR from OpenCV)
        raw_frames: list[np.ndarray] = []
        for i in range(NUM_INPUT_FRAMES):
            img_path = self.data_dir / "images" / rally_id / f"{start + i}.jpg"
            img = cv2.imread(str(img_path))
            if img is None:
                img = np.zeros((IMG_HEIGHT, IMG_WIDTH, 3), dtype=np.uint8)
            raw_frames.append(img)

        # Get labels for this window
        window_labels = labels[start : start + NUM_INPUT_FRAMES]

        # Apply augmentation
        if self.augment:
            rng = np.random.default_rng()
            raw_frames, window_labels = self._augment_frames(raw_frames, window_labels, rng)

        # BGR → RGB, normalize with ImageNet stats, HWC → CHW, concatenate → (9, H, W)
        chw_frames = []
        for frame in raw_frames:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD
            chw_frames.append(rgb.transpose(2, 0, 1))

        x = torch.from_numpy(np.concatenate(chw_frames, axis=0)).float()

        # Generate heatmaps for each frame → (3, H, W)
        heatmaps: list[np.ndarray] = []
        for i in range(NUM_INPUT_FRAMES):
            vis, lx, ly = window_labels[i]
            if vis >= 1:
                heatmaps.append(
                    gen_wasb_heatmap(float(lx), float(ly), radius=self.config.heatmap_radius)
                )
            else:
                heatmaps.append(np.zeros((IMG_HEIGHT, IMG_WIDTH), dtype=np.float32))

        y = torch.from_numpy(np.stack(heatmaps))
        return x, y


def _compute_detection_metrics(
    y_true: torch.Tensor,
    y_pred_logits: torch.Tensor,
    threshold: float = 0.5,
) -> tuple[int, int, int, int]:
    """Compute frame-level detection TP, FP, FN, TN.

    Returns (tp, fp, fn, tn) counts across all frames in the batch.
    """
    y_pred = torch.sigmoid(y_pred_logits)
    tp = fp = fn = tn = 0
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
            else:
                tn += 1
    return tp, fp, fn, tn


def train_wasb(
    data_dir: Path,
    output_dir: Path,
    config: WASBConfig,
    train_rally_ids: list[str],
    val_rally_ids: list[str],
    resume_checkpoint: Path | None = None,
    pretrained_weights: Path | None = None,
) -> WASBResult:
    """Train WASB HRNet on pseudo-labeled ball tracking data.

    Args:
        data_dir: Directory with {rally_id}.csv and images/{rally_id}/.
        output_dir: Directory to save model weights and checkpoints.
        config: Training configuration.
        train_rally_ids: Rally IDs for training.
        val_rally_ids: Rally IDs for validation.
        resume_checkpoint: Path to checkpoint for resuming interrupted training.
        pretrained_weights: Path to pretrained .pth.tar weights for initialization.

    Returns:
        WASBResult with best metrics and model path.
    """
    from tqdm import tqdm

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create datasets
    train_dataset = WASBDataset(data_dir, train_rally_ids, augment=True, config=config)
    val_dataset = WASBDataset(data_dir, val_rally_ids, augment=False, config=config)

    if len(train_dataset) == 0:
        raise ValueError(
            f"No training samples found in {data_dir}. "
            "Check that CSV files and images/ directory exist for the given rally IDs."
        )

    print(f"Train: {len(train_dataset)} windows from {len(train_rally_ids)} rallies")
    print(f"Val: {len(val_dataset)} windows from {len(val_rally_ids)} rallies")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
    )
    val_loader = None
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=True,
        )

    # Model
    model = HRNet(WASB_CONFIG)

    # Load pretrained weights
    if pretrained_weights and pretrained_weights.exists() and not resume_checkpoint:
        checkpoint = torch.load(pretrained_weights, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        # Handle DataParallel 'module.' prefix
        cleaned = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "", 1) if key.startswith("module.") else key
            cleaned[new_key] = value
        model.load_state_dict(cleaned)
        total = sum(p.numel() for p in model.parameters())
        print(f"Loaded pretrained weights: {total:,} params")

    model = model.to(device)

    # Optimizer and scheduler (matching WASB original)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=config.lr_milestones,
        gamma=config.lr_gamma,
    )

    start_epoch = 0
    best_score = 0.0  # f1 + accuracy
    best_f1 = 0.0
    best_val_loss = float("inf")

    # Resume from checkpoint
    if resume_checkpoint and resume_checkpoint.exists():
        ckpt = torch.load(resume_checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["net"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["lr_scheduler"])
        start_epoch = ckpt["epoch"]
        best_score = ckpt.get("best_score", 0.0)
        best_f1 = ckpt.get("best_f1", 0.0)
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}")

    # Output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = output_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # Mixed precision for faster training on Ampere+ GPUs
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if use_amp:
        print("Mixed precision (fp16) enabled")

    best_metrics: dict[str, float] = {
        "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0,
    }
    best_epoch = start_epoch
    avg_val_loss = float("inf")
    val_metrics: dict[str, float] = {}
    # Early stopping state: detect val F1 collapse
    val_ever_positive = False
    val_zero_streak = 0

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
            with torch.amp.autocast("cuda", enabled=use_amp):
                output = model(x_batch)
                y_pred_logits = output[0]  # dict[int, Tensor] → scale 0
                loss = wbce_loss(y_pred_logits, y_batch)

            loss_val = loss.item()
            if not np.isfinite(loss_val):
                print(f"\n  FATAL: NaN/Inf loss at epoch {epoch + 1}, batch {train_batches + 1}")
                print("  Stopping training. Check data quality and learning rate.")
                # Save current state for debugging
                torch.save(model.state_dict(), output_dir / "crashed.pt")
                raise RuntimeError(
                    f"Training diverged: loss={loss_val} at epoch {epoch + 1}. "
                    "Saved crashed.pt for debugging."
                )

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss_val
            train_batches += 1
            pbar.set_postfix(loss=f"{loss_val:.4f}")

        scheduler.step()
        avg_train_loss = train_loss / max(train_batches, 1)

        # ── Validate ──
        avg_val_loss = float("inf")
        val_metrics = {}
        if val_loader:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            total_tp = total_fp = total_fn = total_tn = 0

            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch = x_batch.to(device)
                    y_batch = y_batch.to(device)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        output = model(x_batch)
                        y_pred_logits = output[0]
                        val_loss += wbce_loss(y_pred_logits, y_batch).item()
                    val_batches += 1
                    tp, fp, fn, tn = _compute_detection_metrics(y_batch, y_pred_logits)
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
                    total_tn += tn

            avg_val_loss = val_loss / max(val_batches, 1)
            precision = total_tp / max(total_tp + total_fp, 1)
            recall = total_tp / max(total_tp + total_fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-7)
            accuracy = (total_tp + total_tn) / max(
                total_tp + total_fp + total_fn + total_tn, 1
            )
            val_metrics = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "accuracy": accuracy,
            }

            # Early stopping: detect val F1 collapse
            if f1 > 0.01:
                val_ever_positive = True
                val_zero_streak = 0
            elif val_ever_positive:
                val_zero_streak += 1
                if val_zero_streak >= config.early_stop_patience:
                    print(
                        f"\n  EARLY STOP: Val F1 collapsed to 0 for "
                        f"{val_zero_streak} consecutive epochs after being positive."
                    )
                    print("  This indicates overfitting — stopping to save compute.")
                    break

        lr = scheduler.get_last_lr()[0]
        msg = (
            f"Epoch {epoch + 1}/{config.epochs} | "
            f"Train: {avg_train_loss:.4f} | "
            f"Val: {avg_val_loss:.4f} | "
            f"LR: {lr:.6f}"
        )
        if val_metrics:
            msg += (
                f" | P: {val_metrics['precision']:.3f}"
                f" R: {val_metrics['recall']:.3f}"
                f" F1: {val_metrics['f1']:.3f}"
                f" Acc: {val_metrics['accuracy']:.3f}"
            )
        print(msg)

        # ── Save weights ──
        if (epoch + 1) % config.save_every == 0:
            torch.save(model.state_dict(), output_dir / "last.pt")

            # Model selection: best = argmax(f1 + accuracy) matching WASB paper
            current_score = val_metrics.get("f1", 0.0) + val_metrics.get("accuracy", 0.0)
            if current_score > best_score:
                best_score = current_score
                best_f1 = val_metrics.get("f1", 0.0)
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                best_metrics = val_metrics or best_metrics
                torch.save(model.state_dict(), output_dir / "best.pt")
                print(
                    f"  -> New best model (F1: {best_f1:.3f}, "
                    f"Acc: {val_metrics.get('accuracy', 0.0):.3f})"
                )

        # ── Full checkpoint for resume ──
        if (epoch + 1) % config.checkpoint_every == 0:
            ckpt_data = {
                "net": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "lr_scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
                "best_val_loss": best_val_loss,
                "best_f1": best_f1,
                "best_score": best_score,
            }
            torch.save(ckpt_data, ckpt_dir / f"ckpt_{epoch + 1}.pt")
            torch.save(ckpt_data, ckpt_dir / "ckpt_latest.pt")

    # Save final weights
    torch.save(model.state_dict(), output_dir / "last.pt")
    final_score = val_metrics.get("f1", 0.0) + val_metrics.get("accuracy", 0.0)
    if final_score > best_score:
        best_score = final_score
        best_f1 = val_metrics.get("f1", 0.0)
        best_val_loss = avg_val_loss
        best_epoch = config.epochs
        best_metrics = val_metrics or best_metrics
        torch.save(model.state_dict(), output_dir / "best.pt")

    best_path = output_dir / "best.pt"
    if not best_path.exists():
        torch.save(model.state_dict(), best_path)

    return WASBResult(
        best_val_loss=best_val_loss,
        best_epoch=best_epoch,
        model_path=str(best_path),
        precision=best_metrics.get("precision", 0.0),
        recall=best_metrics.get("recall", 0.0),
        f1=best_metrics.get("f1", 0.0),
        accuracy=best_metrics.get("accuracy", 0.0),
    )
