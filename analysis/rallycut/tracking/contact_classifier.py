"""Learned contact classifier for ball contact detection.

Replaces hand-tuned 3-tier validation gates with a trained binary classifier
that can find non-linear decision boundaries in the joint feature space.

Individual trajectory features (velocity, direction change, player distance)
overlap between TP and FP in marginal distributions, but a learned model can
exploit correlations between features to separate them.

Features per candidate (25 total):
- Ball velocity magnitude, direction change angle, vertical velocity, speed ratio
- Arc fit residual, acceleration, trajectory curvature
- Player distance to nearest player
- Player bbox motion: best/nearest player max delta-Y and delta-height (4 features)
- Ball (x, y) position, ball Y relative to net, net crossing flag
- Time since last candidate (frames), frames since rally start
- Ball detection density, consecutive detections
- Nearest-player pose features (5): active wrist velocity max, hand-ball distance min,
  arm extension change, pose confidence mean, both-arms-raised fraction. Computed
  from YOLO-Pose keypoints via
  pose_attribution.features.extract_contact_pose_features_for_nearest and default
  to 0.0 when keypoints are unavailable.

Note 2026-04-07: 7 MS-TCN++ `seq_p_*` features were removed after the
contact_classifier_audit found their GBM importance was exactly 0.0000 (the
trainer was always passing zero-filled sequence probs, so the trees never
split on them). The MS-TCN++ signal still reaches action classification via
`sequence_action_runtime.apply_sequence_override` at stage 14.

Model: scikit-learn GradientBoostingClassifier. A simple ensemble model is
appropriate for this dataset size and resistant to overfitting.
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
    acceleration: float  # Velocity change (second derivative) near candidate
    trajectory_curvature: float  # Curvature of ball path near candidate
    velocity_y: float  # Vertical velocity component (positive = downward)
    velocity_ratio: float  # Speed ratio after/before candidate (>1 = accelerating)
    # Player features
    player_distance: float  # Distance to nearest player (inf if no player)
    # Position features
    ball_x: float  # Ball X position (0-1)
    ball_y: float  # Ball Y position (0-1)
    ball_y_relative_net: float  # Ball Y minus net_y (negative = far side)
    is_net_crossing: bool  # Ball crosses net_y within ±5 frames
    # Temporal features
    frames_since_last: int  # Frames since previous candidate (0 if first)
    # Player bbox motion (peak frame-to-frame deltas in ±5 frames around candidate)
    best_player_max_d_y: float = 0.0  # Max d_y across all nearby players
    best_player_max_d_height: float = 0.0  # Max d_height across all nearby players
    nearest_player_max_d_y: float = 0.0  # d_y of nearest player
    nearest_player_max_d_height: float = 0.0  # d_height of nearest player
    # Detection quality
    ball_detection_density: float = 1.0  # fraction of frames with ball in ±10 window
    consecutive_detections: int = 0  # consecutive ball detections around candidate
    frames_since_rally_start: int = 0  # frames from rally start (early = serve)
    # Pose features for nearest player (0.0 when keypoints unavailable)
    nearest_active_wrist_velocity_max: float = 0.0
    nearest_hand_ball_dist_min: float = 0.0
    nearest_active_arm_extension_change: float = 0.0
    nearest_pose_confidence_mean: float = 0.0
    nearest_both_arms_raised: float = 0.0

    def to_array(self) -> np.ndarray:
        """Convert to numpy feature array for classifier input."""
        player_dist = self.player_distance if math.isfinite(self.player_distance) else 1.0
        return np.array([
            self.velocity,
            self.direction_change_deg,
            self.arc_fit_residual,
            self.acceleration,
            self.trajectory_curvature,
            self.velocity_y,
            self.velocity_ratio,
            player_dist,
            self.best_player_max_d_y,
            self.best_player_max_d_height,
            self.nearest_player_max_d_y,
            self.nearest_player_max_d_height,
            self.ball_x,
            self.ball_y,
            self.ball_y_relative_net,
            float(self.is_net_crossing),
            self.frames_since_last,
            self.ball_detection_density,
            self.consecutive_detections,
            self.frames_since_rally_start,
            self.nearest_active_wrist_velocity_max,
            self.nearest_hand_ball_dist_min,
            self.nearest_active_arm_extension_change,
            self.nearest_pose_confidence_mean,
            self.nearest_both_arms_raised,
        ], dtype=np.float64)

    @staticmethod
    def feature_names() -> list[str]:
        return [
            "velocity",
            "direction_change_deg",
            "arc_fit_residual",
            "acceleration",
            "trajectory_curvature",
            "velocity_y",
            "velocity_ratio",
            "player_distance",
            "best_player_max_d_y",
            "best_player_max_d_height",
            "nearest_player_max_d_y",
            "nearest_player_max_d_height",
            "ball_x",
            "ball_y",
            "ball_y_relative_net",
            "is_net_crossing",
            "frames_since_last",
            "ball_detection_density",
            "consecutive_detections",
            "frames_since_rally_start",
            "nearest_active_wrist_velocity_max",
            "nearest_hand_ball_dist_min",
            "nearest_active_arm_extension_change",
            "nearest_pose_confidence_mean",
            "nearest_both_arms_raised",
        ]


class ContactClassifier:
    """Binary classifier for contact candidate validation.

    Wraps scikit-learn GradientBoostingClassifier. Can be trained from
    labeled contact data (GT matches) and used to replace the hand-tuned
    3-tier validation gates in detect_contacts().
    """

    def __init__(self, model: Any = None, threshold: float = 0.40):
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

        # Handle models trained with different feature counts (backward compat).
        expected = self.model.n_features_in_
        if x_mat.shape[1] > expected:
            x_mat = x_mat[:, :expected]
        elif x_mat.shape[1] < expected:
            # Pad with zeros for features removed from code but expected by model
            pad = np.zeros((x_mat.shape[0], expected - x_mat.shape[1]))
            x_mat = np.hstack([x_mat, pad])

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
        positive_weight: float = 1.0,
    ) -> dict[str, float]:
        """Train the classifier on labeled data.

        Args:
            x_train: Feature matrix (n_samples, n_features).
            y: Binary labels (1=contact, 0=not contact).
            rally_ids: Optional rally IDs for LOO CV reporting.
            positive_weight: Weight multiplier for positive samples.
                Values > 1.0 penalize missed contacts more than false positives,
                improving recall at the cost of precision.

        Returns:
            Dictionary of training metrics.
        """
        from sklearn.ensemble import GradientBoostingClassifier

        self.model = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            min_samples_leaf=5,
            subsample=0.8,
            random_state=42,
        )
        sample_weights = np.where(y == 1, positive_weight, 1.0)
        self.model.fit(x_train, y, sample_weight=sample_weights)

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
        positive_weight: float = 1.0,
    ) -> dict[str, Any]:
        """Leave-one-rally-out cross-validation.

        Args:
            x_all: Feature matrix.
            y: Binary labels.
            rally_ids: Rally ID per sample (for grouping folds).
            positive_weight: Weight multiplier for positive samples.

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
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                min_samples_leaf=5,
                subsample=0.8,
                random_state=42,
            )
            train_weights = np.where(y[train_mask] == 1, positive_weight, 1.0)
            model.fit(x_all[train_mask], y[train_mask], sample_weight=train_weights)
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
            threshold=data.get("threshold", 0.40),
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
