"""Inference with trained MS-TCN++ model.

Reuses the same post-processing pipeline as TemporalMaxer inference.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch

from rallycut.temporal.ms_tcn.model import MSTCN, MSTCNConfig
from rallycut.temporal.temporal_maxer.inference import TemporalMaxerInference, TemporalMaxerResult

logger = logging.getLogger(__name__)


class MSTCNInference(TemporalMaxerInference):
    """Run inference with a trained MS-TCN++ model.

    Inherits all post-processing logic from TemporalMaxerInference.
    Only overrides model loading and forward pass.
    """

    def _load_model(self, model_path: Path) -> MSTCN:  # type: ignore[override]
        """Load trained MS-TCN++ model."""
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)

        config_dict = checkpoint.get("config", {})
        config = MSTCNConfig(**config_dict)

        model = MSTCN(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()

        return model

    def predict(
        self,
        features: np.ndarray,
        fps: float,
        stride: int,
        window_size: int = 16,
        min_segment_duration: float = 1.0,
        max_gap_duration: float = 3.0,
        max_segment_duration: float = 60.0,
        min_segment_confidence: float = 0.6,
        valley_threshold: float = 0.5,
        min_valley_duration: float = 2.0,
        tta_shifts: int = 0,
        rescue_min_avg_prob: float = 0.30,
        rescue_min_windows: int = 5,
        rescue_max_duration: float = 10.0,
        ball_features: np.ndarray | None = None,
    ) -> TemporalMaxerResult:
        """Run inference with MS-TCN++ and extract segments.

        Same interface as TemporalMaxerInference.predict().
        """
        if len(features) == 0:
            return TemporalMaxerResult()

        expected_dim = self.model.config.feature_dim
        assert features.shape[1] == expected_dim, (
            f"Expected {expected_dim}-dim features, got {features.shape[1]}"
        )

        features_t = torch.from_numpy(features).float().T.unsqueeze(0).to(self.device)

        # Prepare ball features
        ball_t: torch.Tensor | None = None
        if ball_features is not None and self.model.config.ball_feature_dim > 0:
            min_len = min(features_t.shape[2], len(ball_features))
            bf = ball_features[:min_len]
            ball_t = torch.from_numpy(bf).float().T.unsqueeze(0).to(self.device)
            features_t = features_t[:, :, :min_len]

        with torch.no_grad():
            # MS-TCN++ returns final stage logits via forward()
            logits = self.model(features_t, ball_features=ball_t)
            probs = torch.softmax(logits, dim=1)
            rally_probs = probs[0, 1].cpu().numpy()
            predictions = (rally_probs > 0.5).astype(np.int64)

        # Reuse TemporalMaxer post-processing
        window_duration = stride / fps
        segments = self._predictions_to_segments(
            predictions,
            rally_probs,
            window_duration,
            window_size / fps,
            min_segment_duration,
            max_gap_duration,
            max_segment_duration,
            min_segment_confidence,
            valley_threshold=valley_threshold,
            min_valley_duration=min_valley_duration,
        )

        if rescue_min_avg_prob > 0:
            rescued = self._rescue_short_rallies(
                segments,
                rally_probs,
                window_duration,
                rescue_min_avg_prob=rescue_min_avg_prob,
                rescue_min_windows=rescue_min_windows,
                rescue_min_duration=min_segment_duration,
                rescue_max_duration=rescue_max_duration,
            )
            if rescued:
                segments = sorted(segments + rescued)
                logger.debug("Rescued %d short rallies", len(rescued))

        return TemporalMaxerResult(
            segments=segments,
            window_probs=rally_probs,
            window_predictions=predictions,
        )
