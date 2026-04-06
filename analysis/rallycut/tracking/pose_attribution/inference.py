"""Inference wrapper for per-candidate pose attribution model."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import joblib
import numpy as np

logger = logging.getLogger(__name__)


class PoseAttributionInference:
    """Score each candidate independently and pick the highest."""

    def __init__(self, model_path: Path) -> None:
        self.model = joblib.load(model_path)

        # Load metadata to determine expected feature count/set
        meta_path = model_path.parent / "metadata.json"
        self.n_features = self.model.n_features_in_
        self.feature_set = "combined"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            self.feature_set = meta.get("feature_set", "combined")

        logger.info(
            f"Loaded pose attribution model from {model_path} "
            f"({self.feature_set}, {self.n_features} features)"
        )

    def predict(
        self,
        candidate_features: list[np.ndarray],
        candidate_track_ids: list[int],
    ) -> tuple[int, float]:
        """Predict which candidate touched the ball.

        Args:
            candidate_features: List of feature arrays, one per candidate.
                May have more features than the model expects — truncated
                automatically based on the model's expected feature count.
            candidate_track_ids: Corresponding track IDs.

        Returns:
            (predicted_track_id, confidence) where confidence is the
            highest P(touching) among candidates.
        """
        if not candidate_features or not candidate_track_ids:
            return -1, 0.0

        x = np.stack(candidate_features)  # (N_candidates, NUM_FEATURES)

        # Truncate to model's expected feature count (handles pose-only
        # models receiving full 30-feature vectors from the pipeline)
        if x.shape[1] > self.n_features:
            x = x[:, :self.n_features]

        probs = self.model.predict_proba(x)  # (N_candidates, 2)

        # P(touching) is class 1
        class_idx = list(self.model.classes_).index(1) if 1 in self.model.classes_ else -1
        if class_idx < 0:
            return candidate_track_ids[0], 0.5

        touch_probs = probs[:, class_idx]
        best_idx = int(touch_probs.argmax())

        return candidate_track_ids[best_idx], float(touch_probs[best_idx])
