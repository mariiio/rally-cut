"""Training pipeline for temporal models.

Provides training functions for temporal models (v1, v2, v3) on
pre-extracted VideoMAE features with sequence labels.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from rallycut.evaluation.ground_truth import EvaluationVideo

logger = logging.getLogger(__name__)


@dataclass
class TemporalTrainingConfig:
    """Configuration for temporal model training."""

    # Model selection
    model_version: str = "v1"

    # Architecture (v2/v3 only)
    hidden_dim: int = 128
    num_layers: int = 3
    kernel_size: int = 5
    dropout: float = 0.4

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    epochs: int = 50
    batch_size: int = 1  # Per-video training
    patience: int = 10

    # Labeling
    labeling_mode: str = "center"
    overlap_threshold: float = 0.5
    stride: int = 48

    # Augmentation
    use_augmentation: bool = True
    shift_range: int = 2
    keep_ratio_min: float = 0.8

    # Chunking for long videos
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Device
    device: str = "cpu"


@dataclass
class TemporalTrainingResult:
    """Results from temporal model training."""

    model_version: str
    best_val_f1: float
    best_epoch: int
    train_losses: list[float] = field(default_factory=list)
    val_f1_scores: list[float] = field(default_factory=list)
    training_time_seconds: float = 0.0


class SequenceDataset(Dataset):
    """Dataset for temporal sequence training.

    Each item is a full video's features and labels, suitable for
    sequence-to-sequence training.
    """

    def __init__(
        self,
        video_features: dict[str, np.ndarray],
        video_labels: dict[str, np.ndarray],
        augment: bool = False,
        shift_range: int = 2,
        keep_ratio_min: float = 0.8,
    ):
        """Initialize dataset.

        Args:
            video_features: Dict mapping video_id to features (seq_len, 768).
            video_labels: Dict mapping video_id to labels (seq_len,).
            augment: Whether to apply augmentation.
            shift_range: Augmentation shift range.
            keep_ratio_min: Minimum keep ratio for augmentation.
        """
        self.video_ids = list(video_features.keys())
        self.video_features = video_features
        self.video_labels = video_labels
        self.augment = augment
        self.shift_range = shift_range
        self.keep_ratio_min = keep_ratio_min

    def __len__(self) -> int:
        return len(self.video_ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        video_id = self.video_ids[idx]
        features = self.video_features[video_id].copy()
        labels = self.video_labels[video_id].copy()

        if self.augment:
            from rallycut.training.sampler import augment_sequence

            features, labels = augment_sequence(
                features,
                labels,
                shift_range=self.shift_range,
                keep_ratio_range=(self.keep_ratio_min, 1.0),
            )

        return (
            torch.from_numpy(features).float(),
            torch.from_numpy(labels).long(),
            video_id,
        )


def _collate_sequences(
    batch: list[tuple[torch.Tensor, torch.Tensor, str]],
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[str]]:
    """Custom collate for variable-length sequences.

    Since sequences have different lengths, we return lists instead of
    stacked tensors.
    """
    features_list = [item[0] for item in batch]
    labels_list = [item[1] for item in batch]
    video_ids = [item[2] for item in batch]
    return features_list, labels_list, video_ids


def prepare_training_data(
    videos: list[EvaluationVideo],
    feature_cache_dir: Path,
    stride: int = 48,
    labeling_mode: str = "center",
    overlap_threshold: float = 0.5,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Prepare training data from videos with ground truth.

    Loads cached features and generates sequence labels for each video.

    Args:
        videos: List of evaluation videos with ground truth.
        feature_cache_dir: Directory containing cached features.
        stride: Stride used for feature extraction.
        labeling_mode: "center" or "overlap" for label assignment.
        overlap_threshold: Threshold for overlap mode.

    Returns:
        Tuple of (features_dict, labels_dict) mapping video_id to arrays.
    """
    from rallycut.temporal.features import FeatureCache
    from rallycut.training.sampler import generate_sequence_labels

    cache = FeatureCache(cache_dir=feature_cache_dir)
    video_features: dict[str, np.ndarray] = {}
    video_labels: dict[str, np.ndarray] = {}

    for video in videos:
        # Load cached features
        cached = cache.get(video.content_hash, stride)
        if cached is None:
            logger.warning("No cached features for %s, skipping", video.filename)
            continue

        features, metadata = cached

        # Generate labels
        fps = video.fps or 30.0
        labels = generate_sequence_labels(
            rallies=video.ground_truth_rallies,
            video_duration_ms=video.duration_ms or 0,
            fps=fps,
            stride=stride,
            labeling_mode=labeling_mode,
            overlap_threshold=overlap_threshold,
        )

        # Ensure labels match feature length
        if len(labels) != len(features):
            diff = abs(len(labels) - len(features))
            min_len = min(len(labels), len(features))
            # Warn if mismatch is significant (> 5% or > 10 windows)
            if diff > max(10, min_len * 0.05):
                logger.warning(
                    "Large length mismatch for %s: labels=%d, features=%d (diff=%d)",
                    video.filename,
                    len(labels),
                    len(features),
                    diff,
                )
            labels = labels[:min_len]
            features = features[:min_len]

        # Validate features for NaN/Inf
        if np.any(~np.isfinite(features)):
            logger.warning("NaN/Inf values in features for %s, skipping", video.filename)
            continue

        video_features[video.id] = features
        video_labels[video.id] = np.array(labels)

    return video_features, video_labels


def compute_sequence_f1(
    predictions: list[torch.Tensor],
    labels: list[torch.Tensor],
) -> float:
    """Compute F1 score over all sequences.

    Args:
        predictions: List of prediction tensors.
        labels: List of label tensors.

    Returns:
        Macro F1 score.
    """
    all_preds = torch.cat(predictions).cpu().numpy()
    all_labels = torch.cat(labels).cpu().numpy()

    # Compute TP, FP, FN for class 1 (RALLY)
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return float(f1)


def compute_pos_weight(labels: dict[str, np.ndarray]) -> float:
    """Compute pos_weight for BCE loss from label distribution.

    This computes num_negatives / num_positives to balance the loss
    for imbalanced binary classification.

    Args:
        labels: Dict mapping video_id to label arrays.

    Returns:
        pos_weight value (>1 means positive class is minority).
    """
    all_labels = np.concatenate(list(labels.values()))
    num_positive = (all_labels == 1).sum()
    num_negative = (all_labels == 0).sum()

    if num_positive == 0:
        return 1.0  # Fallback to no weighting

    return float(num_negative / num_positive)


def chunk_sequence(
    features: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int,
    overlap: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Split long sequence into overlapping chunks.

    Args:
        features: Feature tensor (seq_len, feature_dim).
        labels: Label tensor (seq_len,).
        chunk_size: Maximum chunk size.
        overlap: Overlap between chunks.

    Returns:
        List of (features_chunk, labels_chunk) tuples.
    """
    seq_len = len(features)
    if seq_len <= chunk_size:
        return [(features, labels)]

    chunks = []
    start = 0
    while start < seq_len:
        end = min(start + chunk_size, seq_len)
        chunks.append((features[start:end], labels[start:end]))
        start = end - overlap
        if start >= seq_len - overlap:
            break

    return chunks


def train_temporal_model(
    model: nn.Module,
    train_features: dict[str, np.ndarray],
    train_labels: dict[str, np.ndarray],
    val_features: dict[str, np.ndarray],
    val_labels: dict[str, np.ndarray],
    config: TemporalTrainingConfig,
    output_dir: Path,
) -> TemporalTrainingResult:
    """Train a temporal model.

    Args:
        model: Temporal model (LearnedSmoothing, ConvCRF, or BiLSTMCRF).
        train_features: Training features dict.
        train_labels: Training labels dict.
        val_features: Validation features dict.
        val_labels: Validation labels dict.
        config: Training configuration.
        output_dir: Directory to save model and results.

    Returns:
        Training result with metrics.
    """
    import time

    start_time = time.time()
    device = torch.device(config.device)
    model = model.to(device)

    # Create datasets
    train_dataset = SequenceDataset(
        train_features,
        train_labels,
        augment=config.use_augmentation,
        shift_range=config.shift_range,
        keep_ratio_min=config.keep_ratio_min,
    )
    val_dataset = SequenceDataset(val_features, val_labels, augment=False)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=_collate_sequences,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=_collate_sequences,
    )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    # Training state
    result = TemporalTrainingResult(
        model_version=config.model_version,
        best_val_f1=0.0,
        best_epoch=0,
    )
    patience_counter = 0
    best_model_state = None

    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        # Training
        model.train()
        train_loss = 0.0
        num_batches = 0

        for features_list, labels_list, _ in train_loader:
            for features, labels in zip(features_list, labels_list):
                features = features.to(device)
                labels = labels.to(device)

                # Chunk long sequences
                chunks = chunk_sequence(
                    features, labels, config.chunk_size, config.chunk_overlap
                )

                for chunk_features, chunk_labels in chunks:
                    # Add batch dimension
                    chunk_features = chunk_features.unsqueeze(0)
                    chunk_labels = chunk_labels.unsqueeze(0)

                    optimizer.zero_grad()

                    # All models take features directly
                    output = model(chunk_features, labels=chunk_labels)

                    loss = output["loss"]

                    # Check for NaN/Inf loss (indicates data corruption or numerical issues)
                    if torch.isnan(loss) or torch.isinf(loss):
                        raise ValueError(
                            f"NaN/Inf loss detected at epoch {epoch + 1}. "
                            "Check input features for invalid values."
                        )

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
        val_predictions = []
        val_labels_all = []

        with torch.no_grad():
            for features_list, labels_list, _ in val_loader:
                for features, labels in zip(features_list, labels_list):
                    features = features.to(device).unsqueeze(0)
                    labels = labels.to(device)

                    output = model(features)

                    preds = output["predictions"].squeeze(0)
                    val_predictions.append(preds)
                    val_labels_all.append(labels)

        val_f1 = compute_sequence_f1(val_predictions, val_labels_all)
        result.val_f1_scores.append(val_f1)

        # Early stopping check
        if val_f1 > result.best_val_f1:
            result.best_val_f1 = val_f1
            result.best_epoch = epoch
            patience_counter = 0
            best_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        logger.info(
            "Epoch %d/%d: loss=%.4f, val_f1=%.4f (best=%.4f @ epoch %d)",
            epoch + 1,
            config.epochs,
            avg_train_loss,
            val_f1,
            result.best_val_f1,
            result.best_epoch + 1,
        )

        if patience_counter >= config.patience:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    # Save best model with metadata
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        # Add model_version to state dict for auto-detection during loading
        best_model_state["_model_version"] = config.model_version
        torch.save(best_model_state, output_dir / "best_temporal_model.pt")

    # Save training config and results
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    result.training_time_seconds = time.time() - start_time

    with open(output_dir / "training_result.json", "w") as f:
        json.dump(asdict(result), f, indent=2)

    return result


def video_level_split(
    videos: list[EvaluationVideo],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[EvaluationVideo], list[EvaluationVideo]]:
    """Split videos into train and validation sets.

    Uses video-level split to prevent data leakage between train and val.

    Args:
        videos: List of evaluation videos.
        train_ratio: Fraction of videos for training.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_videos, val_videos).
    """
    rng = random.Random(seed)
    shuffled = list(videos)
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]
