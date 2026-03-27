"""Inference wrapper for temporal contact attribution model."""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np

logger = logging.getLogger(__name__)


class TemporalAttributionInference:
    """Load a trained temporal attribution model and run inference."""

    def __init__(self, model_path: Path) -> None:
        self.model = joblib.load(model_path)
        logger.info(f"Loaded temporal attribution model from {model_path}")

    def predict(
        self,
        features: np.ndarray,
        canonical_track_ids: list[int],
    ) -> tuple[int, float]:
        """Predict which player touched the ball.

        Args:
            features: (NUM_FEATURES,) feature array from extract_attribution_features.
            canonical_track_ids: 4 track IDs in canonical slot order.

        Returns:
            (predicted_track_id, confidence). track_id is from canonical_track_ids.
        """
        x = features.reshape(1, -1)
        probs = self.model.predict_proba(x)[0]
        # Map probability index back to actual class label (slot 0-3).
        # classes_ may not be [0,1,2,3] if a slot was absent from training data.
        slot_idx = int(self.model.classes_[probs.argmax()])
        confidence = float(probs.max())

        track_id = (
            canonical_track_ids[slot_idx]
            if slot_idx < len(canonical_track_ids)
            else -1
        )
        return track_id, confidence
