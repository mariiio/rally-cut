"""Training for temporal contact attribution using gradient-boosted trees.

Uses sklearn HistGradientBoostingClassifier which handles small datasets
well and supports native missing value handling.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier

from rallycut.tracking.temporal_attribution.features import FEATURE_NAMES

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training hyperparameters for gradient-boosted tree model."""

    max_iter: int = 200
    max_depth: int = 4
    learning_rate: float = 0.1
    min_samples_leaf: int = 5
    max_leaf_nodes: int = 15
    seed: int = 42


@dataclass
class TrainingResult:
    """Result of a training run."""

    best_val_accuracy: float
    num_train: int
    num_val: int
    feature_importances: dict[str, float] | None = None


def train_model(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray,
    val_labels: np.ndarray,
    config: TrainingConfig,
    output_path: Path | None = None,
) -> TrainingResult:
    """Train a gradient-boosted tree model for contact attribution.

    Args:
        train_features: (N_train, NUM_FEATURES) feature arrays.
        train_labels: (N_train,) int labels (canonical slot 0-3).
        val_features: (N_val, NUM_FEATURES) feature arrays.
        val_labels: (N_val,) int labels.
        config: Training hyperparameters.
        output_path: If provided, saves model here.

    Returns:
        TrainingResult with validation accuracy.
    """
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

    clf.fit(train_features, train_labels)

    val_preds = clf.predict(val_features)
    val_acc = float((val_preds == val_labels).mean())

    # Feature importances
    importances = None
    if hasattr(clf, "feature_importances_"):
        importances = {
            name: float(imp)
            for name, imp in zip(FEATURE_NAMES, clf.feature_importances_)
        }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(clf, output_path)
        logger.info(
            f"Saved model (val_acc={val_acc:.3f}) to {output_path}"
        )

    return TrainingResult(
        best_val_accuracy=val_acc,
        num_train=len(train_labels),
        num_val=len(val_labels),
        feature_importances=importances,
    )
