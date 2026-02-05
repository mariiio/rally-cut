"""Binary classification head for rally detection.

Trains a simple binary classifier on frozen VideoMAE encoder features
to test if the features contain discriminative signal for rally detection.

This is Phase 1 of the emissions-first training approach:
1. Freeze encoder, train binary head on cached features
2. Report ROC-AUC, PR-AUC, F1 (window-level)
3. Only proceed to temporal smoothing if emissions are good
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from rallycut.evaluation.ground_truth import EvaluationVideo, GroundTruthRally

logger = logging.getLogger(__name__)


@dataclass
class BinaryHeadConfig:
    """Configuration for binary head training."""

    # Architecture
    hidden_dim: int = 128
    dropout: float = 0.3

    # Training
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    epochs: int = 50
    batch_size: int = 64
    patience: int = 10

    # Labeling
    overlap_threshold: float = 0.5  # Window needs >50% overlap with rally to be positive
    stride: int = 16  # Fine stride for training

    # Device
    device: str = "cpu"

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {self.hidden_dim}")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if not 0.0 < self.overlap_threshold <= 1.0:
            raise ValueError(f"overlap_threshold must be in (0, 1], got {self.overlap_threshold}")


@dataclass
class BinaryHeadResult:
    """Results from binary head training."""

    # Window-level metrics (primary)
    best_val_roc_auc: float = 0.0
    best_val_pr_auc: float = 0.0
    best_val_f1: float = 0.0
    best_val_precision: float = 0.0
    best_val_recall: float = 0.0
    best_threshold: float = 0.5
    best_epoch: int = 0

    # Training history
    train_losses: list[float] = field(default_factory=list)
    val_roc_aucs: list[float] = field(default_factory=list)
    val_f1s: list[float] = field(default_factory=list)

    # Data stats
    train_samples: int = 0
    val_samples: int = 0
    train_positive_rate: float = 0.0
    val_positive_rate: float = 0.0

    training_time_seconds: float = 0.0


class BinaryHead(nn.Module):
    """Simple binary classifier head for rally detection.

    Architecture:
        Linear(768, hidden_dim) → ReLU → Dropout → Linear(hidden_dim, 1) → Sigmoid

    Input: (batch, 768) encoder features
    Output: (batch,) rally probabilities
    """

    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 128,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            features: (batch, 768) encoder features

        Returns:
            (batch,) logits (before sigmoid)
        """
        return self.classifier(features).squeeze(-1)

    def predict_proba(self, features: torch.Tensor) -> torch.Tensor:
        """Get rally probabilities.

        Args:
            features: (batch, 768) encoder features

        Returns:
            (batch,) rally probabilities in [0, 1]
        """
        with torch.no_grad():
            logits = self.forward(features)
            return torch.sigmoid(logits)


def generate_overlap_labels(
    rallies: list[GroundTruthRally],
    video_duration_ms: int,
    fps: float,
    stride: int,
    window_size: int = 16,
    overlap_threshold: float = 0.5,
) -> list[int]:
    """Generate window labels based on overlap with rallies.

    A window is labeled as RALLY (1) if it overlaps with any ground truth
    rally by more than overlap_threshold.

    Args:
        rallies: List of ground truth rallies.
        video_duration_ms: Video duration in milliseconds.
        fps: Video frames per second.
        stride: Frame stride between windows.
        window_size: Number of frames per window.
        overlap_threshold: Minimum overlap ratio to label as RALLY.

    Returns:
        List of binary labels (0 or 1) for each window.
    """
    video_duration_s = video_duration_ms / 1000.0
    window_duration_s = window_size / fps
    num_windows = int((video_duration_s * fps - window_size) / stride) + 1

    labels = []
    for i in range(num_windows):
        # Window time range
        window_start_s = i * stride / fps
        window_end_s = window_start_s + window_duration_s

        # Check overlap with each rally
        is_rally = False
        for rally in rallies:
            rally_start_s = rally.start_seconds
            rally_end_s = rally.end_seconds

            # Calculate overlap
            overlap_start = max(window_start_s, rally_start_s)
            overlap_end = min(window_end_s, rally_end_s)
            overlap_duration = max(0, overlap_end - overlap_start)

            overlap_ratio = overlap_duration / window_duration_s
            if overlap_ratio >= overlap_threshold:
                is_rally = True
                break

        labels.append(1 if is_rally else 0)

    return labels


class WindowDataset(Dataset):
    """Dataset for window-level binary classification."""

    def __init__(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ):
        """Initialize dataset.

        Args:
            features: (N, 768) feature array
            labels: (N,) binary label array
        """
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def prepare_window_data(
    videos: list[EvaluationVideo],
    feature_cache_dir: Path,
    stride: int = 16,
    overlap_threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Prepare window-level features and labels.

    Args:
        videos: List of evaluation videos with ground truth.
        feature_cache_dir: Directory containing cached features.
        stride: Stride for label generation (should match feature extraction).
        overlap_threshold: Overlap threshold for labeling.

    Returns:
        Tuple of (features, labels, video_ids) where:
        - features: (N, 768) array of all window features
        - labels: (N,) array of binary labels
        - video_ids: Empty list (deprecated, kept for API compatibility)
    """
    from rallycut.temporal.features import FeatureCache

    cache = FeatureCache(cache_dir=feature_cache_dir)

    all_features = []
    all_labels = []

    for video in videos:
        # Load cached features
        cached = cache.get(video.content_hash, stride)
        if cached is None:
            logger.warning(
                "No cached features for %s at stride=%d, skipping", video.filename, stride
            )
            continue

        features, metadata = cached

        # Generate labels
        fps = video.fps or 30.0
        labels = generate_overlap_labels(
            rallies=video.ground_truth_rallies,
            video_duration_ms=video.duration_ms or 0,
            fps=fps,
            stride=stride,
            overlap_threshold=overlap_threshold,
        )

        # Align lengths
        min_len = min(len(features), len(labels))
        if abs(len(features) - len(labels)) > 5:
            logger.warning(
                "Length mismatch for %s: features=%d, labels=%d",
                video.filename,
                len(features),
                len(labels),
            )

        features = features[:min_len]
        labels = labels[:min_len]

        # Validate features
        if np.any(~np.isfinite(features)):
            logger.warning("NaN/Inf in features for %s, skipping", video.filename)
            continue

        all_features.append(features)
        all_labels.extend(labels)

    if not all_features:
        raise ValueError(
            f"No features loaded for {len(videos)} videos. "
            "Run 'rallycut train extract-features' first."
        )

    features_array = np.concatenate(all_features, axis=0)
    labels_array = np.array(all_labels)

    # Return empty list for video_ids (no longer tracked)
    return features_array, labels_array, []


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute classification metrics.

    Args:
        y_true: Ground truth binary labels.
        y_prob: Predicted probabilities.
        threshold: Classification threshold.

    Returns:
        Dict with roc_auc, pr_auc, f1, precision, recall, threshold.
    """
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    # ROC-AUC and PR-AUC (threshold-independent)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.0  # Only one class present

    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except ValueError:
        pr_auc = 0.0

    # Threshold-based metrics
    y_pred = (y_prob >= threshold).astype(int)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "threshold": threshold,
    }


def find_best_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> tuple[float, float]:
    """Find threshold that maximizes the given metric.

    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        metric: Metric to optimize ("f1", "precision", "recall").

    Returns:
        Tuple of (best_threshold, best_metric_value).
    """
    best_threshold = 0.5
    best_value = 0.0

    for threshold in np.arange(0.1, 0.9, 0.05):
        metrics = compute_metrics(y_true, y_prob, threshold)
        if metrics[metric] > best_value:
            best_value = metrics[metric]
            best_threshold = threshold

    return best_threshold, best_value


def train_binary_head(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    config: BinaryHeadConfig,
    output_dir: Path,
) -> BinaryHeadResult:
    """Train the binary classification head.

    Args:
        train_features: (N_train, 768) training features.
        train_labels: (N_train,) training labels.
        val_features: (N_val, 768) validation features.
        val_labels: (N_val,) validation labels.
        config: Training configuration.
        output_dir: Directory to save model and results.

    Returns:
        Training result with metrics.
    """
    start_time = time.time()
    device = torch.device(config.device)

    # Create datasets and loaders
    train_dataset = WindowDataset(train_features, train_labels)
    val_dataset = WindowDataset(val_features, val_labels)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
    )

    # Create model
    model = BinaryHead(
        feature_dim=768,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
    ).to(device)

    # Compute pos_weight for class imbalance
    num_pos = float(train_labels.sum())
    num_neg = float(len(train_labels) - num_pos)
    if num_pos > 0:
        pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32)
    else:
        pos_weight = torch.tensor([1.0], dtype=torch.float32)
    pos_weight = pos_weight.to(device)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    # Initialize result
    result = BinaryHeadResult(
        train_samples=len(train_labels),
        val_samples=len(val_labels),
        train_positive_rate=float(train_labels.mean()),
        val_positive_rate=float(val_labels.mean()),
    )

    best_model_state = None
    patience_counter = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        # Training
        model.train()
        train_loss = 0.0
        num_batches = 0

        for features, labels in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)

            if torch.isnan(loss) or torch.isinf(loss):
                raise ValueError(f"NaN/Inf loss at epoch {epoch + 1}")

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
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device)
                probs = torch.sigmoid(model(features)).cpu().numpy()
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())

        y_true = np.array(all_labels)
        y_prob = np.array(all_probs)

        # Find best threshold and compute metrics
        best_thresh, _ = find_best_threshold(y_true, y_prob, "f1")
        metrics = compute_metrics(y_true, y_prob, best_thresh)

        result.val_roc_aucs.append(metrics["roc_auc"])
        result.val_f1s.append(metrics["f1"])

        # Check for improvement (use ROC-AUC as primary metric)
        if metrics["roc_auc"] > result.best_val_roc_auc:
            result.best_val_roc_auc = metrics["roc_auc"]
            result.best_val_pr_auc = metrics["pr_auc"]
            result.best_val_f1 = metrics["f1"]
            result.best_val_precision = metrics["precision"]
            result.best_val_recall = metrics["recall"]
            result.best_threshold = best_thresh
            result.best_epoch = epoch
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        logger.info(
            "Epoch %d/%d: loss=%.4f, ROC-AUC=%.4f, PR-AUC=%.4f, F1=%.4f @ thresh=%.2f "
            "(best ROC-AUC=%.4f @ epoch %d)",
            epoch + 1,
            config.epochs,
            avg_train_loss,
            metrics["roc_auc"],
            metrics["pr_auc"],
            metrics["f1"],
            best_thresh,
            result.best_val_roc_auc,
            result.best_epoch + 1,
        )

        if patience_counter >= config.patience:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    # Save best model
    if best_model_state is not None:
        torch.save(best_model_state, output_dir / "best_binary_head.pt")

    # Save config and results
    with open(output_dir / "binary_head_config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    result.training_time_seconds = time.time() - start_time

    with open(output_dir / "binary_head_result.json", "w") as f:
        json.dump(asdict(result), f, indent=2)

    return result


# =============================================================================
# Phase 2: Binary Head with Temporal Smoothing
# =============================================================================


class BinaryHeadWithSmoothing(nn.Module):
    """Binary head with learned temporal smoothing.

    Architecture:
        features (batch, seq, 768)
        → BinaryHead (per-window logits)
        → 1D Conv smoothing
        → learned threshold
        → predictions

    This combines the binary classification head with temporal smoothing
    to produce coherent rally segments.
    """

    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        kernel_size: int = 7,
    ):
        super().__init__()

        # Binary classification head
        self.head = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        # Temporal smoothing (1D conv on logits)
        self.smoother = nn.Conv1d(
            in_channels=1,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        # Initialize smoother to approximate mean filter
        self._init_smoother(kernel_size)

        # Learned threshold
        self.threshold = nn.Parameter(torch.tensor(0.0))  # In logit space

    def _init_smoother(self, kernel_size: int) -> None:
        """Initialize smoother to uniform average."""
        with torch.no_grad():
            self.smoother.weight.fill_(1.0 / kernel_size)
            self.smoother.bias.fill_(0.0)

    def forward(
        self,
        features: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            features: (batch, seq_len, feature_dim) encoder features
            labels: (batch, seq_len) binary labels (optional, for training)

        Returns:
            Dict with predictions, probs, logits, and optionally loss.
        """
        _, seq_len, _ = features.shape

        # Get per-window logits
        logits = self.head(features).squeeze(-1)  # (batch, seq_len)

        # Apply temporal smoothing
        # Reshape for conv: (batch, seq_len) -> (batch, 1, seq_len)
        logits_conv = logits.unsqueeze(1)
        smoothed_logits = self.smoother(logits_conv).squeeze(1)  # (batch, seq_len)

        # Get probabilities and predictions
        probs = torch.sigmoid(smoothed_logits)
        predictions = (smoothed_logits > self.threshold).long()

        result: dict[str, torch.Tensor] = {
            "predictions": predictions,
            "probs": probs,
            "logits": logits,
            "smoothed_logits": smoothed_logits,
        }

        if labels is not None:
            # BCE loss on smoothed logits
            loss = nn.functional.binary_cross_entropy_with_logits(
                smoothed_logits, labels.float(), reduction="mean"
            )
            result["loss"] = loss

        return result

    def load_pretrained_head(self, binary_head: BinaryHead) -> None:
        """Load weights from a pretrained BinaryHead.

        Args:
            binary_head: Trained BinaryHead model.
        """
        # Copy classifier weights
        with torch.no_grad():
            for (name1, param1), (name2, param2) in zip(
                self.head.named_parameters(),
                binary_head.classifier.named_parameters(),
            ):
                param1.copy_(param2)


@dataclass
class SmoothingConfig:
    """Configuration for Phase 2 training (with smoothing)."""

    # Architecture
    hidden_dim: int = 128
    dropout: float = 0.3
    kernel_size: int = 7  # Smoothing kernel (~0.37s at stride=16)

    # Training
    learning_rate: float = 1e-4  # Lower LR for fine-tuning
    weight_decay: float = 0.01
    epochs: int = 30
    batch_size: int = 8  # Sequence batches (smaller)
    patience: int = 10

    # Freezing
    freeze_head: bool = True  # Freeze binary head, only train smoother

    # Labeling
    overlap_threshold: float = 0.5
    stride: int = 16

    # Device
    device: str = "cpu"


@dataclass
class SmoothingResult:
    """Results from Phase 2 training."""

    # Window-level metrics
    best_val_roc_auc: float = 0.0
    best_val_f1: float = 0.0
    best_epoch: int = 0

    # Segment-level metrics (what we really care about)
    best_segment_f1: float = 0.0
    best_segment_precision: float = 0.0
    best_segment_recall: float = 0.0

    # Training history
    train_losses: list[float] = field(default_factory=list)
    val_f1s: list[float] = field(default_factory=list)
    segment_f1s: list[float] = field(default_factory=list)

    training_time_seconds: float = 0.0


class SequenceDataset(Dataset):
    """Dataset for sequence-level training (one video = one sample)."""

    def __init__(
        self,
        video_features: list[np.ndarray],
        video_labels: list[np.ndarray],
    ):
        """Initialize dataset.

        Args:
            video_features: List of (seq_len, 768) feature arrays per video.
            video_labels: List of (seq_len,) label arrays per video.
        """
        self.features = [torch.from_numpy(f).float() for f in video_features]
        self.labels = [torch.from_numpy(lbl).float() for lbl in video_labels]

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def collate_sequences(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Collate variable-length sequences with padding.

    Returns:
        Tuple of (padded_features, padded_labels, lengths).
    """
    features, labels = zip(*batch)
    lengths = torch.tensor([len(f) for f in features])

    # Pad to max length
    max_len = max(lengths)
    padded_features = torch.zeros(len(features), max_len, features[0].shape[-1])
    padded_labels = torch.zeros(len(features), max_len)

    for i, (feat, lbl) in enumerate(zip(features, labels)):
        padded_features[i, : len(feat)] = feat
        padded_labels[i, : len(lbl)] = lbl

    return padded_features, padded_labels, lengths


def predictions_to_segments(
    predictions: np.ndarray,
    fps: float,
    stride: int,
    window_size: int = 16,
    min_duration: float = 0.5,
) -> list[tuple[float, float]]:
    """Convert window predictions to time segments.

    Args:
        predictions: (seq_len,) binary predictions.
        fps: Video FPS.
        stride: Frame stride.
        window_size: Frames per window.
        min_duration: Minimum segment duration in seconds.

    Returns:
        List of (start_time, end_time) tuples.
    """
    window_duration = stride / fps
    segments = []

    in_rally = False
    rally_start = 0.0

    for i, pred in enumerate(predictions):
        window_time = i * window_duration

        if pred == 1 and not in_rally:
            in_rally = True
            rally_start = window_time
        elif pred == 0 and in_rally:
            in_rally = False
            end_time = window_time + window_size / fps
            if end_time - rally_start >= min_duration:
                segments.append((rally_start, end_time))

    if in_rally:
        end_time = len(predictions) * window_duration
        if end_time - rally_start >= min_duration:
            segments.append((rally_start, end_time))

    return segments


def train_with_smoothing(
    train_videos: list[EvaluationVideo],
    val_videos: list[EvaluationVideo],
    feature_cache_dir: Path,
    pretrained_head_path: Path,
    config: SmoothingConfig,
    output_dir: Path,
) -> SmoothingResult:
    """Train Phase 2: binary head with temporal smoothing.

    Args:
        train_videos: Training videos.
        val_videos: Validation videos.
        feature_cache_dir: Directory with cached features.
        pretrained_head_path: Path to pretrained binary head.
        config: Training configuration.
        output_dir: Output directory.

    Returns:
        Training result.
    """
    from sklearn.metrics import f1_score, roc_auc_score

    from rallycut.temporal.deterministic_decoder import compute_segment_metrics
    from rallycut.temporal.features import FeatureCache

    start_time = time.time()
    device = torch.device(config.device)

    # Load pretrained binary head
    pretrained_head = BinaryHead(
        feature_dim=768, hidden_dim=config.hidden_dim, dropout=config.dropout
    )
    pretrained_head.load_state_dict(torch.load(pretrained_head_path, weights_only=True))

    # Create model with smoothing
    model = BinaryHeadWithSmoothing(
        feature_dim=768,
        hidden_dim=config.hidden_dim,
        dropout=config.dropout,
        kernel_size=config.kernel_size,
    )
    model.load_pretrained_head(pretrained_head)
    model.to(device)

    # Optionally freeze the head
    if config.freeze_head:
        for param in model.head.parameters():
            param.requires_grad = False

    # Prepare data (per-video sequences)
    cache = FeatureCache(cache_dir=feature_cache_dir)

    def load_video_data(
        videos: list[EvaluationVideo],
    ) -> tuple[list[np.ndarray], list[np.ndarray], list[EvaluationVideo]]:
        features_list = []
        labels_list = []
        valid_videos = []
        for video in videos:
            cached = cache.get(video.content_hash, config.stride)
            if cached is None:
                continue
            features, metadata = cached
            labels = generate_overlap_labels(
                rallies=video.ground_truth_rallies,
                video_duration_ms=video.duration_ms or 0,
                fps=video.fps or 30.0,
                stride=config.stride,
                overlap_threshold=config.overlap_threshold,
            )
            min_len = min(len(features), len(labels))
            features_list.append(features[:min_len])
            labels_list.append(np.array(labels[:min_len]))
            valid_videos.append(video)
        return features_list, labels_list, valid_videos

    train_features, train_labels, train_valid = load_video_data(train_videos)
    val_features, val_labels, val_valid = load_video_data(val_videos)

    train_dataset = SequenceDataset(train_features, train_labels)
    val_dataset = SequenceDataset(val_features, val_labels)

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_sequences
    )
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_sequences)

    # Optimizer (only trainable params)
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    result = SmoothingResult()
    best_model_state = None
    patience_counter = 0

    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        # Training
        model.train()
        train_loss = 0.0
        num_batches = 0

        for features, labels, lengths in train_loader:
            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            output = model(features, labels)
            loss = output["loss"]

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_train_loss = train_loss / max(num_batches, 1)
        result.train_losses.append(avg_train_loss)

        # Validation (window-level)
        model.eval()
        all_probs = []
        all_labels = []

        with torch.no_grad():
            for features, labels, lengths in val_loader:
                features = features.to(device)
                output = model(features)
                probs = output["probs"].squeeze(0).cpu().numpy()
                labs = labels.squeeze(0).numpy()
                seq_len = int(lengths[0])
                all_probs.extend(probs[:seq_len])
                all_labels.extend(labs[:seq_len])

        y_true = np.array(all_labels)
        y_prob = np.array(all_probs)
        y_pred = (y_prob >= 0.5).astype(int)

        try:
            roc_auc = roc_auc_score(y_true, y_prob)
        except ValueError:
            roc_auc = 0.0
        f1 = f1_score(y_true, y_pred, zero_division=0)

        result.val_f1s.append(f1)

        # Validation (segment-level)
        total_seg_metrics = {"precision": 0.0, "recall": 0.0, "f1": 0.0}

        with torch.no_grad():
            for video, features_np, labels_np in zip(val_valid, val_features, val_labels):
                features_t = torch.from_numpy(features_np).float().unsqueeze(0).to(device)
                output = model(features_t)
                preds = output["predictions"].squeeze(0).cpu().numpy()

                pred_segments = predictions_to_segments(preds, video.fps or 30.0, config.stride)
                gt_segments = [(r.start_seconds, r.end_seconds) for r in video.ground_truth_rallies]

                seg_metrics = compute_segment_metrics(gt_segments, pred_segments)
                total_seg_metrics["precision"] += seg_metrics["precision"]
                total_seg_metrics["recall"] += seg_metrics["recall"]
                total_seg_metrics["f1"] += seg_metrics["f1"]

        n_val = len(val_valid)
        avg_seg_f1 = total_seg_metrics["f1"] / n_val if n_val > 0 else 0
        result.segment_f1s.append(avg_seg_f1)

        # Check for improvement (use segment F1 as primary metric)
        if avg_seg_f1 > result.best_segment_f1:
            result.best_segment_f1 = avg_seg_f1
            result.best_segment_precision = (
                total_seg_metrics["precision"] / n_val if n_val > 0 else 0
            )
            result.best_segment_recall = total_seg_metrics["recall"] / n_val if n_val > 0 else 0
            result.best_val_roc_auc = roc_auc
            result.best_val_f1 = f1
            result.best_epoch = epoch
            patience_counter = 0
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1

        logger.info(
            "Epoch %d/%d: loss=%.4f, win_F1=%.3f, seg_F1=%.3f (best seg_F1=%.3f @ epoch %d)",
            epoch + 1,
            config.epochs,
            avg_train_loss,
            f1,
            avg_seg_f1,
            result.best_segment_f1,
            result.best_epoch + 1,
        )

        if patience_counter >= config.patience:
            logger.info("Early stopping at epoch %d", epoch + 1)
            break

    # Save best model
    if best_model_state is not None:
        torch.save(best_model_state, output_dir / "best_binary_head_smoothed.pt")

    result.training_time_seconds = time.time() - start_time

    with open(output_dir / "smoothing_result.json", "w") as f:
        json.dump(asdict(result), f, indent=2)

    return result
