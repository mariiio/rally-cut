"""Training loop for MS-TCN++.

Reuses VideoSequenceDataset and collation from TemporalMaxer.
Adds multi-stage loss (sum CE+TMSE across all stages with
exponential decay weighting for earlier stages).
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from rallycut.temporal.ms_tcn.model import MSTCN, MSTCNConfig
from rallycut.temporal.temporal_maxer.training import (
    AugmentationConfig,
    VideoSequenceDataset,
    collate_video_sequences,
    compute_tmse_loss,
)

logger = logging.getLogger(__name__)


@dataclass
class MSTCNTrainingConfig:
    """Configuration for MS-TCN++ training."""

    model_config: MSTCNConfig = field(default_factory=MSTCNConfig)

    learning_rate: float = 5e-4
    weight_decay: float = 0.01
    epochs: int = 50
    batch_size: int = 4
    patience: int = 15

    tmse_weight: float = 0.15
    label_smoothing: float = 0.0

    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    device: str = "cpu"
    seed: int = 42


@dataclass
class MSTCNTrainingResult:
    """Result from MS-TCN++ training."""

    best_val_f1: float = 0.0
    best_val_precision: float = 0.0
    best_val_recall: float = 0.0
    best_epoch: int = 0

    train_losses: list[float] = field(default_factory=list)
    val_f1s: list[float] = field(default_factory=list)

    training_time_seconds: float = 0.0


class MSTCNTrainer:
    """Trainer for MS-TCN++ model."""

    def __init__(self, config: MSTCNTrainingConfig | None = None) -> None:
        self.config = config or MSTCNTrainingConfig()

    def train(
        self,
        train_features: list[np.ndarray],
        train_labels: list[np.ndarray],
        val_features: list[np.ndarray],
        val_labels: list[np.ndarray],
        output_dir: Path,
        train_ball_features: list[np.ndarray] | None = None,
        val_ball_features: list[np.ndarray] | None = None,
    ) -> MSTCNTrainingResult:
        """Train MS-TCN++ model.

        Args:
            train_features: List of (seq_len, feature_dim) arrays per video.
            train_labels: List of (seq_len,) binary label arrays per video.
            val_features: Validation feature arrays.
            val_labels: Validation label arrays.
            output_dir: Directory to save model and results.
            train_ball_features: Optional list of (seq_len, ball_dim) arrays.
            val_ball_features: Optional list of (seq_len, ball_dim) arrays.

        Returns:
            Training result with metrics.
        """
        start_time = time.time()
        device = torch.device(self.config.device)
        cfg = self.config

        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(cfg.seed)

        train_dataset = VideoSequenceDataset(
            train_features, train_labels, augmentation=cfg.augmentation,
            ball_features=train_ball_features,
        )
        val_dataset = VideoSequenceDataset(
            val_features, val_labels, ball_features=val_ball_features,
        )

        train_generator = torch.Generator().manual_seed(cfg.seed)
        train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            collate_fn=collate_video_sequences,
            generator=train_generator,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_video_sequences,
        )

        model = MSTCN(cfg.model_config).to(device)

        # Class weights for imbalanced data
        all_labels = np.concatenate(train_labels)
        num_pos = float(all_labels.sum())
        num_neg = float(len(all_labels) - num_pos)
        if num_pos > 0 and num_neg > 0:
            class_weights = torch.tensor(
                [1.0, num_neg / num_pos], dtype=torch.float32, device=device,
            )
        else:
            class_weights = torch.ones(2, dtype=torch.float32, device=device)

        criterion = nn.CrossEntropyLoss(
            weight=class_weights, reduction="none",
            label_smoothing=cfg.label_smoothing,
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=cfg.epochs,
        )

        result = MSTCNTrainingResult()
        best_model_state = None
        patience_counter = 0

        output_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(cfg.epochs):
            # Training
            model.train()
            train_loss = 0.0
            num_batches = 0

            for features, labels, mask, ball_feat in train_loader:
                features = features.to(device)
                labels = labels.to(device)
                mask = mask.to(device)
                bf = ball_feat.to(device) if ball_feat is not None else None

                optimizer.zero_grad()

                # Get all stage outputs for multi-stage loss
                stage_outputs = model.forward_all_stages(features, mask, ball_features=bf)

                # Multi-stage loss: sum CE+TMSE for each stage
                total_loss = torch.tensor(0.0, device=device)
                mask_squeezed = mask.squeeze(1)
                for logits in stage_outputs:
                    ce_loss = criterion(logits, labels)
                    ce_loss = (ce_loss * mask_squeezed).sum() / mask_squeezed.sum()
                    tmse_loss = compute_tmse_loss(logits, mask)
                    total_loss = total_loss + ce_loss + cfg.tmse_weight * tmse_loss

                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    logger.warning("NaN/Inf loss at epoch %d, skipping batch", epoch + 1)
                    continue

                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                train_loss += total_loss.item()
                num_batches += 1

            scheduler.step()
            avg_train_loss = train_loss / max(num_batches, 1)
            result.train_losses.append(avg_train_loss)

            # Validation
            if len(val_dataset) > 0:
                model.eval()
                all_preds: list[int] = []
                all_labels_flat: list[int] = []

                with torch.no_grad():
                    for features, labels, mask, ball_feat in val_loader:
                        features = features.to(device)
                        labels = labels.to(device)
                        mask = mask.to(device)
                        bf = ball_feat.to(device) if ball_feat is not None else None

                        logits = model(features, mask, ball_features=bf)
                        preds = logits.argmax(dim=1)

                        mask_bool = mask.squeeze(1).bool()
                        for b in range(features.shape[0]):
                            valid_mask = mask_bool[b]
                            all_preds.extend(preds[b][valid_mask].cpu().tolist())
                            all_labels_flat.extend(labels[b][valid_mask].tolist())

                from sklearn.metrics import f1_score, precision_score, recall_score

                y_true = np.array(all_labels_flat)
                y_pred = np.array(all_preds)

                f1 = f1_score(y_true, y_pred, zero_division=0)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                result.val_f1s.append(f1)

                if f1 > result.best_val_f1:
                    result.best_val_f1 = f1
                    result.best_val_precision = precision
                    result.best_val_recall = recall
                    result.best_epoch = epoch
                    patience_counter = 0
                    best_model_state = {
                        k: v.cpu().clone() for k, v in model.state_dict().items()
                    }
                else:
                    patience_counter += 1

                logger.info(
                    "Epoch %d/%d: loss=%.4f, F1=%.4f, P=%.4f, R=%.4f "
                    "(best F1=%.4f @ epoch %d)",
                    epoch + 1, cfg.epochs, avg_train_loss,
                    f1, precision, recall,
                    result.best_val_f1, result.best_epoch + 1,
                )

                if patience_counter >= cfg.patience:
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break
            else:
                best_model_state = {
                    k: v.cpu().clone() for k, v in model.state_dict().items()
                }
                result.best_epoch = epoch
                logger.info(
                    "Epoch %d/%d: loss=%.4f (no validation set)",
                    epoch + 1, cfg.epochs, avg_train_loss,
                )

        # Save best model
        if best_model_state is not None:
            save_data = {
                "model_state_dict": best_model_state,
                "config": asdict(cfg.model_config),
                "head_type": "mstcn",
            }
            torch.save(save_data, output_dir / "best_temporal_maxer.pt")

        result.training_time_seconds = time.time() - start_time

        with open(output_dir / "temporal_maxer_result.json", "w") as f:
            json.dump(asdict(result), f, indent=2)

        return result
