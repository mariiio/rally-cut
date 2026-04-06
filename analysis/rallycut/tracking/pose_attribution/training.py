"""Training for per-candidate pose attribution using gradient-boosted trees.

Unlike the canonical-slot model, this trains a binary classifier that scores
each candidate independently (P(touched) per candidate), then picks the
highest scorer at inference time.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from rallycut.tracking.pose_attribution.features import FEATURE_NAMES

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training hyperparameters."""

    max_iter: int = 200
    max_depth: int = 4
    learning_rate: float = 0.1
    min_samples_leaf: int = 5
    max_leaf_nodes: int = 15
    positive_weight: float = 3.0  # Weight for positive (touching) samples
    seed: int = 42


@dataclass
class TrainingResult:
    """Result of a training run."""

    train_auc: float
    num_positive: int
    num_negative: int
    feature_importances: dict[str, float] | None = None


def train_model(
    features: np.ndarray,
    labels: np.ndarray,
    config: TrainingConfig | None = None,
    output_path: Path | None = None,
) -> tuple[HistGradientBoostingClassifier, TrainingResult]:
    """Train a per-candidate binary classifier.

    Args:
        features: (N_samples, NUM_FEATURES) feature arrays.
            Each sample is one candidate at one contact.
        labels: (N_samples,) binary labels (1=touching, 0=not touching).
        config: Training hyperparameters.
        output_path: If provided, saves model here.

    Returns:
        (trained_model, TrainingResult).
    """
    if config is None:
        config = TrainingConfig()

    # Sample weights: upweight positive class
    sample_weight = np.ones(len(labels), dtype=np.float32)
    sample_weight[labels == 1] = config.positive_weight

    clf = HistGradientBoostingClassifier(
        max_iter=config.max_iter,
        max_depth=config.max_depth,
        learning_rate=config.learning_rate,
        min_samples_leaf=config.min_samples_leaf,
        max_leaf_nodes=config.max_leaf_nodes,
        random_state=config.seed,
        early_stopping=True,
        n_iter_no_change=20,
        validation_fraction=0.15,
    )

    clf.fit(features, labels, sample_weight=sample_weight)

    # Compute training AUC
    train_probs = clf.predict_proba(features)[:, 1]
    try:
        train_auc = roc_auc_score(labels, train_probs)
    except ValueError:
        train_auc = 0.0

    # Feature importances
    importances = None
    if hasattr(clf, "feature_importances_"):
        n_feats = min(len(FEATURE_NAMES), len(clf.feature_importances_))
        importances = {
            FEATURE_NAMES[i]: float(clf.feature_importances_[i])
            for i in range(n_feats)
        }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, output_path)
        logger.info(f"Saved pose attribution model to {output_path}")

    return clf, TrainingResult(
        train_auc=train_auc,
        num_positive=int(labels.sum()),
        num_negative=int((labels == 0).sum()),
        feature_importances=importances,
    )
