"""Training loop for temporal contact attribution model.

Supports leave-one-video-out cross-validation and full training with
early stopping, class-weighted CE loss, and data augmentation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F  # noqa: N812
from torch.utils.data import DataLoader, TensorDataset

from rallycut.tracking.temporal_attribution.model import (
    TemporalAttributionConfig,
    TemporalAttributionModel,
)

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    model_config: TemporalAttributionConfig = field(
        default_factory=TemporalAttributionConfig
    )
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 100
    batch_size: int = 32
    patience: int = 20
    seed: int = 42
    noise_std: float = 0.005  # Gaussian noise augmentation on positions
    temporal_jitter: bool = True  # Shift window ±1 frame


@dataclass
class TrainingResult:
    """Result of a training run."""

    best_val_accuracy: float
    best_epoch: int
    train_accuracy: float
    num_train: int
    num_val: int


def train_model(
    train_windows: np.ndarray,
    train_labels: np.ndarray,
    val_windows: np.ndarray,
    val_labels: np.ndarray,
    config: TrainingConfig,
    output_path: Path | None = None,
) -> TrainingResult:
    """Train a temporal attribution model.

    Args:
        train_windows: (N_train, window_size, n_features) feature arrays.
        train_labels: (N_train,) int labels (canonical slot 0-3).
        val_windows: (N_val, window_size, n_features) feature arrays.
        val_labels: (N_val,) int labels.
        config: Training hyperparameters.
        output_path: If provided, saves best checkpoint here.

    Returns:
        TrainingResult with best validation accuracy.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    device = "cpu"  # Model is tiny (~8K params), CPU is fast enough
    mc = config.model_config

    # Transpose to channels-first: (N, window, features) → (N, features, window)
    train_x = torch.from_numpy(train_windows).float().permute(0, 2, 1)
    train_y = torch.from_numpy(train_labels).long()
    val_x = torch.from_numpy(val_windows).float().permute(0, 2, 1)
    val_y = torch.from_numpy(val_labels).long()

    model = TemporalAttributionModel(mc).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )
    criterion = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )

    best_val_acc = 0.0
    best_epoch = 0
    best_state = None
    patience_counter = 0
    best_train_acc = 0.0

    for epoch in range(config.epochs):
        # --- Train ---
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Data augmentation: Gaussian noise
            if config.noise_std > 0:
                noise = torch.randn_like(batch_x) * config.noise_std
                batch_x = batch_x + noise

            # Data augmentation: temporal jitter (shift ±1)
            if config.temporal_jitter and torch.rand(1).item() > 0.5:
                shift = 1 if torch.rand(1).item() > 0.5 else -1
                batch_x = torch.roll(batch_x, shift, dims=2)
                if shift > 0:
                    batch_x[:, :, :shift] = 0
                else:
                    batch_x[:, :, shift:] = 0

            logits = model(batch_x)
            loss = criterion(logits, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item() * len(batch_y)
            correct += (logits.argmax(dim=1) == batch_y).sum().item()
            total += len(batch_y)

        scheduler.step()
        train_acc = correct / max(1, total)

        # --- Validate ---
        model.eval()
        with torch.no_grad():
            val_logits = model(val_x.to(device))
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_y.to(device)).float().mean().item()

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_train_acc = train_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= config.patience:
            break

    # Save best model
    if output_path is not None and best_state is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": best_state,
                "config": mc.to_dict(),
            },
            output_path,
        )
        logger.info(
            f"Saved best model (epoch {best_epoch}, "
            f"val_acc={best_val_acc:.3f}) to {output_path}"
        )

    return TrainingResult(
        best_val_accuracy=best_val_acc,
        best_epoch=best_epoch,
        train_accuracy=best_train_acc,
        num_train=len(train_labels),
        num_val=len(val_labels),
    )


def predict_batch(
    model: TemporalAttributionModel,
    windows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run batch prediction.

    Args:
        model: Trained model in eval mode.
        windows: (N, window_size, n_features) feature arrays.

    Returns:
        (predictions, confidences) — both (N,) arrays.
    """
    model.eval()
    x = torch.from_numpy(windows).float().permute(0, 2, 1)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1).numpy()
        confs = probs.max(dim=1).values.numpy()
    return preds, confs
