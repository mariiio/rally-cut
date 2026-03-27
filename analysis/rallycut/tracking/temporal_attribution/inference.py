"""Inference wrapper for temporal contact attribution model."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F  # noqa: N812

from rallycut.tracking.temporal_attribution.model import (
    TemporalAttributionConfig,
    TemporalAttributionModel,
)

logger = logging.getLogger(__name__)


class TemporalAttributionInference:
    """Load a trained temporal attribution model and run inference."""

    def __init__(self, model_path: Path, device: str = "cpu") -> None:
        checkpoint = torch.load(model_path, map_location=device, weights_only=True)
        config = TemporalAttributionConfig.from_dict(checkpoint["config"])
        self.model = TemporalAttributionModel(config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(device).eval()
        self.device = device
        logger.info(f"Loaded temporal attribution model from {model_path}")

    def predict(
        self,
        window: np.ndarray,
        canonical_track_ids: list[int],
    ) -> tuple[int, float]:
        """Predict which player touched the ball.

        Args:
            window: (window_size, 14) trajectory feature array.
            canonical_track_ids: 4 track IDs in canonical slot order.

        Returns:
            (predicted_track_id, confidence). track_id is from canonical_track_ids.
        """
        # (window_size, n_features) → (1, n_features, window_size) channels-first
        x = torch.from_numpy(window).float().T.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(x)
            probs = F.softmax(logits, dim=1)
            slot_idx: int = int(probs.argmax(dim=1).item())
            confidence = float(probs[0, slot_idx].item())

        track_id = (
            canonical_track_ids[slot_idx]
            if slot_idx < len(canonical_track_ids)
            else -1
        )
        return track_id, confidence
