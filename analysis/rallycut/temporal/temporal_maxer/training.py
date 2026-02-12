"""Training loop for TemporalMaxer.

Sequence-level training with cross-entropy + TMSE smoothing loss.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as nnf
from torch.utils.data import DataLoader, Dataset

from rallycut.temporal.temporal_maxer.model import TemporalMaxer, TemporalMaxerConfig

logger = logging.getLogger(__name__)


@dataclass
class TemporalMaxerTrainingConfig:
    """Configuration for TemporalMaxer training."""

    # Model
    model_config: TemporalMaxerConfig = field(default_factory=TemporalMaxerConfig)

    # Training
    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    epochs: int = 50
    batch_size: int = 4  # Sequence batches (one video per sample)
    patience: int = 10

    # Loss
    tmse_weight: float = 0.15  # Smoothing loss weight

    # Device
    device: str = "cpu"


@dataclass
class TemporalMaxerTrainingResult:
    """Result from TemporalMaxer training."""

    best_val_f1: float = 0.0
    best_val_precision: float = 0.0
    best_val_recall: float = 0.0
    best_epoch: int = 0

    train_losses: list[float] = field(default_factory=list)
    val_f1s: list[float] = field(default_factory=list)

    training_time_seconds: float = 0.0


class VideoSequenceDataset(Dataset):
    """Dataset for full-video sequence training."""

    def __init__(
        self,
        video_features: list[np.ndarray],
        video_labels: list[np.ndarray],
    ) -> None:
        self.features = [torch.from_numpy(f).float() for f in video_features]
        self.labels = [torch.from_numpy(lbl).long() for lbl in video_labels]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def collate_video_sequences(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-length sequences with padding.

    Returns:
        Tuple of (padded_features, padded_labels, mask).
        Features: (batch, feature_dim, max_len) — transposed for Conv1d.
        Labels: (batch, max_len).
        Mask: (batch, 1, max_len) — binary mask for valid positions.
    """
    if not batch:
        return (
            torch.zeros(0, 768, 0),
            torch.zeros(0, 0, dtype=torch.long),
            torch.zeros(0, 1, 0),
        )

    features, labels = zip(*batch)
    lengths = [len(f) for f in features]
    max_len = max(lengths)
    feature_dim = features[0].shape[-1]

    # Pad and transpose features: (batch, seq_len, dim) → (batch, dim, seq_len)
    padded_features = torch.zeros(len(features), feature_dim, max_len)
    padded_labels = torch.zeros(len(features), max_len, dtype=torch.long)
    mask = torch.zeros(len(features), 1, max_len)

    for i, (feat, lbl) in enumerate(zip(features, labels)):
        seq_len = len(feat)
        padded_features[i, :, :seq_len] = feat.T  # Transpose (seq, dim) → (dim, seq)
        padded_labels[i, :seq_len] = lbl
        mask[i, 0, :seq_len] = 1.0

    return padded_features, padded_labels, mask


def compute_tmse_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Temporal Mean Squared Error smoothing loss.

    Penalizes sharp class transitions — standard in TAS.
    TMSE = mean((log_prob[t] - log_prob[t+1])^2)

    Args:
        logits: (batch, num_classes, T) predicted logits.
        mask: (batch, 1, T) binary mask.

    Returns:
        Scalar TMSE loss.
    """
    log_probs = nnf.log_softmax(logits, dim=1)

    # Temporal difference
    diff = log_probs[:, :, 1:] - log_probs[:, :, :-1]

    # Mask: both positions must be valid
    valid = mask[:, :, 1:] * mask[:, :, :-1]

    if valid.sum() == 0:
        return torch.tensor(0.0, device=logits.device)

    tmse = (diff ** 2 * valid).sum() / valid.sum()
    return tmse


class TemporalMaxerTrainer:
    """Trainer for TemporalMaxer model."""

    def __init__(self, config: TemporalMaxerTrainingConfig | None = None) -> None:
        self.config = config or TemporalMaxerTrainingConfig()

    def train(
        self,
        train_features: list[np.ndarray],
        train_labels: list[np.ndarray],
        val_features: list[np.ndarray],
        val_labels: list[np.ndarray],
        output_dir: Path,
    ) -> TemporalMaxerTrainingResult:
        """Train TemporalMaxer model.

        Args:
            train_features: List of (seq_len, 768) feature arrays per video.
            train_labels: List of (seq_len,) binary label arrays per video.
            val_features: List of (seq_len, 768) feature arrays per video.
            val_labels: List of (seq_len,) binary label arrays per video.
            output_dir: Directory to save model and results.

        Returns:
            Training result with metrics.
        """
        start_time = time.time()
        device = torch.device(self.config.device)
        cfg = self.config

        # Create datasets
        train_dataset = VideoSequenceDataset(train_features, train_labels)
        val_dataset = VideoSequenceDataset(val_features, val_labels)

        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_video_sequences,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_video_sequences,
        )

        # Create model
        model = TemporalMaxer(cfg.model_config).to(device)

        # Compute class weights for imbalanced data
        all_labels = np.concatenate(train_labels)
        num_pos = float(all_labels.sum())
        num_neg = float(len(all_labels) - num_pos)
        if num_pos > 0 and num_neg > 0:
            class_weights = torch.tensor(
                [1.0, num_neg / num_pos], dtype=torch.float32, device=device
            )
        else:
            class_weights = torch.ones(2, dtype=torch.float32, device=device)

        criterion = nn.CrossEntropyLoss(weight=class_weights, reduction="none")

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)

        result = TemporalMaxerTrainingResult()
        best_model_state = None
        patience_counter = 0

        output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(cfg.epochs):
            # Training
            model.train()
            train_loss = 0.0
            num_batches = 0

            for features, labels, mask in train_loader:
                features = features.to(device)
                labels = labels.to(device)
                mask = mask.to(device)

                optimizer.zero_grad()

                logits = model(features, mask)  # (batch, 2, T)

                # Cross-entropy loss (masked)
                ce_loss = criterion(logits, labels)  # (batch, T)
                mask_squeezed = mask.squeeze(1)  # (batch, T)
                ce_loss = (ce_loss * mask_squeezed).sum() / mask_squeezed.sum()

                # TMSE smoothing loss
                tmse_loss = compute_tmse_loss(logits, mask)

                loss = ce_loss + cfg.tmse_weight * tmse_loss

                if torch.isnan(loss) or torch.isinf(loss):
                    logger.warning("NaN/Inf loss at epoch %d, skipping batch", epoch + 1)
                    continue

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

            scheduler.step()
            avg_train_loss = train_loss / max(num_batches, 1)
            result.train_losses.append(avg_train_loss)

            # Validation
            model.eval()
            all_preds: list[int] = []
            all_labels_flat: list[int] = []

            with torch.no_grad():
                for features, labels, mask in val_loader:
                    features = features.to(device)
                    mask = mask.to(device)

                    logits = model(features, mask)
                    preds = logits.argmax(dim=1)  # (batch, T)

                    # Extract valid predictions
                    mask_bool = mask.squeeze(1).bool()
                    for b in range(features.shape[0]):
                        valid_mask = mask_bool[b]
                        all_preds.extend(preds[b][valid_mask].cpu().tolist())
                        all_labels_flat.extend(labels[b][valid_mask].tolist())

            # Compute F1
            from sklearn.metrics import f1_score, precision_score, recall_score

            y_true = np.array(all_labels_flat)
            y_pred = np.array(all_preds)

            f1 = f1_score(y_true, y_pred, zero_division=0)
            precision = precision_score(y_true, y_pred, zero_division=0)
            recall = recall_score(y_true, y_pred, zero_division=0)
            result.val_f1s.append(f1)

            # Check for improvement
            if f1 > result.best_val_f1:
                result.best_val_f1 = f1
                result.best_val_precision = precision
                result.best_val_recall = recall
                result.best_epoch = epoch
                patience_counter = 0
                best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                patience_counter += 1

            logger.info(
                "Epoch %d/%d: loss=%.4f, F1=%.4f, P=%.4f, R=%.4f (best F1=%.4f @ epoch %d)",
                epoch + 1,
                cfg.epochs,
                avg_train_loss,
                f1,
                precision,
                recall,
                result.best_val_f1,
                result.best_epoch + 1,
            )

            if patience_counter >= cfg.patience:
                logger.info("Early stopping at epoch %d", epoch + 1)
                break

        # Save best model
        if best_model_state is not None:
            save_data = {
                "model_state_dict": best_model_state,
                "config": asdict(cfg.model_config),
            }
            torch.save(save_data, output_dir / "best_temporal_maxer.pt")

        result.training_time_seconds = time.time() - start_time

        with open(output_dir / "temporal_maxer_result.json", "w") as f:
            json.dump(asdict(result), f, indent=2)

        return result
