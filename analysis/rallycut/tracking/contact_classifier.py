"""Learned contact classifier for ball contact detection.

Replaces hand-tuned 3-tier validation gates with a trained binary classifier
that can find non-linear decision boundaries in the joint feature space.

Individual trajectory features (velocity, direction change, player distance)
overlap between TP and FP in marginal distributions, but a learned model can
exploit correlations between features to separate them.

Features per candidate:
- Ball velocity magnitude, direction change angle
- Arc fit residual (from parabolic detection — key discriminator)
- Player distance to nearest player
- Ball (x, y) position, ball Y relative to net
- Time since last candidate (frames)
- Court side indicator

Model: scikit-learn GradientBoostingClassifier. With ~48 TP and ~72 FP labeled
examples, a simple ensemble model is appropriate and resistant to overfitting.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CandidateFeatures:
    """Feature vector for a single contact candidate."""

    frame: int
    # Trajectory features
    velocity: float  # Smoothed ball velocity magnitude
    direction_change_deg: float  # Trajectory direction change in degrees
    arc_fit_residual: float  # Parabolic arc fit residual (high = breaks parabola)
    # Player features
    player_distance: float  # Distance to nearest player (inf if no player)
    has_player: bool  # Whether a player is within contact radius
    # Position features
    ball_x: float  # Ball X position (0-1)
    ball_y: float  # Ball Y position (0-1)
    ball_y_relative_net: float  # Ball Y minus net_y (negative = far side)
    is_at_net: bool  # Whether ball is in net zone
    # Temporal features
    frames_since_last: int  # Frames since previous candidate (0 if first)
    # Source flags
    is_velocity_peak: bool
    is_inflection: bool
    is_parabolic: bool

    def to_array(self) -> np.ndarray:
        """Convert to numpy feature array for classifier input."""
        player_dist = self.player_distance if math.isfinite(self.player_distance) else 1.0
        return np.array([
            self.velocity,
            self.direction_change_deg,
            self.arc_fit_residual,
            player_dist,
            float(self.has_player),
            self.ball_x,
            self.ball_y,
            self.ball_y_relative_net,
            float(self.is_at_net),
            self.frames_since_last,
            float(self.is_velocity_peak),
            float(self.is_inflection),
            float(self.is_parabolic),
        ], dtype=np.float64)

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "velocity",
            "direction_change_deg",
            "arc_fit_residual",
            "player_distance",
            "has_player",
            "ball_x",
            "ball_y",
            "ball_y_relative_net",
            "is_at_net",
            "frames_since_last",
            "is_velocity_peak",
            "is_inflection",
            "is_parabolic",
        ]


class ContactClassifier:
    """Binary classifier for contact candidate validation.

    Wraps scikit-learn GradientBoostingClassifier. Can be trained from
    labeled contact data (GT matches) and used to replace the hand-tuned
    3-tier validation gates in detect_contacts().
    """

    def __init__(self, model: Any = None, threshold: float = 0.5):
        self.model = model
        self.threshold = threshold
        self._feature_names = CandidateFeatures.feature_names()

    @property
    def is_trained(self) -> bool:
        return self.model is not None

    def predict(self, features: list[CandidateFeatures]) -> list[tuple[bool, float]]:
        """Predict whether each candidate is a real contact.

        Args:
            features: List of candidate feature vectors.

        Returns:
            List of (is_contact, confidence) tuples.
        """
        if not self.is_trained or not features:
            return [(False, 0.0)] * len(features)

        x_mat = np.array([f.to_array() for f in features])
        probas = self.model.predict_proba(x_mat)[:, 1]

        return [
            (float(p) >= self.threshold, float(p))
            for p in probas
        ]

    def train(
        self,
        x_train: np.ndarray,
        y: np.ndarray,
        rally_ids: np.ndarray | None = None,
    ) -> dict[str, float]:
        """Train the classifier on labeled data.

        Args:
            x_train: Feature matrix (n_samples, n_features).
            y: Binary labels (1=contact, 0=not contact).
            rally_ids: Optional rally IDs for LOO CV reporting.

        Returns:
            Dictionary of training metrics.
        """
        from sklearn.ensemble import GradientBoostingClassifier

        self.model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42,
        )
        self.model.fit(x_train, y)

        # Training metrics
        train_proba = self.model.predict_proba(x_train)[:, 1]
        train_pred = (train_proba >= self.threshold).astype(int)

        tp = int(np.sum((train_pred == 1) & (y == 1)))
        fp = int(np.sum((train_pred == 1) & (y == 0)))
        fn = int(np.sum((train_pred == 0) & (y == 1)))

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)

        return {
            "train_f1": f1,
            "train_precision": precision,
            "train_recall": recall,
            "train_tp": tp,
            "train_fp": fp,
            "train_fn": fn,
            "n_samples": len(y),
            "n_positive": int(np.sum(y)),
            "n_negative": int(np.sum(1 - y)),
        }

    def loo_cv(
        self,
        x_all: np.ndarray,
        y: np.ndarray,
        rally_ids: np.ndarray,
    ) -> dict[str, Any]:
        """Leave-one-rally-out cross-validation.

        Args:
            x_all: Feature matrix.
            y: Binary labels.
            rally_ids: Rally ID per sample (for grouping folds).

        Returns:
            Dictionary with LOO CV metrics.
        """
        from sklearn.ensemble import GradientBoostingClassifier

        unique_rallies = np.unique(rally_ids)
        all_preds = np.zeros(len(y))
        all_probas = np.zeros(len(y))

        for rally in unique_rallies:
            test_mask = rally_ids == rally
            train_mask = ~test_mask

            if np.sum(train_mask) < 10 or np.sum(y[train_mask]) < 3:
                # Not enough training data, predict default
                all_probas[test_mask] = 0.5
                all_preds[test_mask] = 1.0
                continue

            model = GradientBoostingClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=42,
            )
            model.fit(x_all[train_mask], y[train_mask])
            probas = model.predict_proba(x_all[test_mask])[:, 1]
            all_probas[test_mask] = probas
            all_preds[test_mask] = (probas >= self.threshold).astype(float)

        tp = int(np.sum((all_preds == 1) & (y == 1)))
        fp = int(np.sum((all_preds == 1) & (y == 0)))
        fn = int(np.sum((all_preds == 0) & (y == 1)))

        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)

        return {
            "loo_f1": f1,
            "loo_precision": precision,
            "loo_recall": recall,
            "loo_tp": tp,
            "loo_fp": fp,
            "loo_fn": fn,
            "n_rallies": len(unique_rallies),
        }

    def feature_importance(self) -> dict[str, float]:
        """Get feature importance from trained model."""
        if not self.is_trained:
            return {}
        importances = self.model.feature_importances_
        return dict(zip(self._feature_names, [float(v) for v in importances]))

    def save(self, path: str | Path) -> None:
        """Save trained model to disk."""
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": self.model,
            "threshold": self.threshold,
            "feature_names": self._feature_names,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

        logger.info(f"Saved contact classifier to {path}")

    @classmethod
    def load(cls, path: str | Path) -> ContactClassifier:
        """Load trained model from disk."""
        import pickle

        with open(path, "rb") as f:
            data = pickle.load(f)  # noqa: S301

        classifier = cls(
            model=data["model"],
            threshold=data.get("threshold", 0.5),
        )
        classifier._feature_names = data.get(
            "feature_names", CandidateFeatures.feature_names()
        )
        return classifier


# Default model path
DEFAULT_MODEL_PATH = Path("weights/contact_classifier/contact_classifier.pkl")


def load_contact_classifier(
    model_path: str | Path | None = None,
) -> ContactClassifier | None:
    """Load contact classifier from default or specified path.

    Returns None if no trained model exists.
    """
    path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
    if path.exists():
        try:
            return ContactClassifier.load(path)
        except Exception as e:
            logger.warning(f"Failed to load contact classifier from {path}: {e}")
    return None
