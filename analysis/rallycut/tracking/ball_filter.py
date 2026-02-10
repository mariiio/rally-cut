"""
Temporal filtering for ball tracking using Kalman filter.

Reduces lag from temporal model context and smooths flickering/jumps.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rallycut.tracking.ball_tracker import BallPosition

logger = logging.getLogger(__name__)


@dataclass
class BallFilterConfig:
    """Configuration for ball temporal filtering."""

    # Kalman filter parameters
    process_noise_position: float = 0.001  # Low: ball position is predictable
    process_noise_velocity: float = 0.01  # Higher: velocity changes on hits
    measurement_noise: float = 0.005  # Trust measurements reasonably

    # Confidence thresholds
    min_confidence_for_update: float = 0.3  # Below this, use prediction only

    # Lag compensation for VballNet model bias
    # The model tends to output positions slightly behind the actual ball
    # Extrapolate forward using estimated velocity to compensate
    enable_lag_compensation: bool = True
    lag_frames: int = 12  # Frames to extrapolate forward (TEST: very aggressive)

    # Jump detection (rejects impossible movements)
    max_velocity: float = 0.3  # 30% of screen per frame is max plausible

    # Occlusion handling
    max_occlusion_frames: int = 30  # ~1s at 30fps before losing track


class BallTemporalFilter:
    """
    Kalman filter for ball tracking with lag compensation and occlusion handling.

    State vector: [x, y, vx, vy] (position and velocity)
    Observation: [x, y] (position from detector)

    Features:
    - Constant velocity motion model
    - Confidence-weighted measurement updates
    - Velocity-based position extrapolation for lag compensation
    - Jump rejection for impossible movements
    - Prediction-only mode during occlusion
    """

    def __init__(self, config: BallFilterConfig | None = None):
        self.config = config or BallFilterConfig()
        self._state: np.ndarray | None = None  # [x, y, vx, vy]
        self._covariance: np.ndarray | None = None  # 4x4 covariance matrix
        self._frames_since_confident: int = 0
        self._initialized: bool = False

        # State transition matrix (constant velocity model)
        # x' = x + vx, y' = y + vy, vx' = vx, vy' = vy
        self._F = np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=np.float64)

        # Observation matrix (we observe position only)
        self._H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Process noise covariance
        self._Q = np.diag([
            self.config.process_noise_position,
            self.config.process_noise_position,
            self.config.process_noise_velocity,
            self.config.process_noise_velocity,
        ])

        # Measurement noise covariance
        self._R = np.eye(2) * self.config.measurement_noise

    def reset(self) -> None:
        """Reset filter state for a new video."""
        self._state = None
        self._covariance = None
        self._frames_since_confident = 0
        self._initialized = False

    def _initialize(self, x: float, y: float) -> None:
        """Initialize filter state from first valid measurement."""
        self._state = np.array([x, y, 0.0, 0.0], dtype=np.float64)
        # High initial covariance for position, higher for velocity (unknown)
        self._covariance = np.diag([0.01, 0.01, 0.1, 0.1])
        self._initialized = True
        self._frames_since_confident = 0

    def _predict(self) -> tuple[np.ndarray, np.ndarray]:
        """Predict next state using motion model."""
        assert self._state is not None and self._covariance is not None

        # State prediction: x' = F @ x
        predicted_state = self._F @ self._state

        # Covariance prediction: P' = F @ P @ F.T + Q
        predicted_cov = self._F @ self._covariance @ self._F.T + self._Q

        return predicted_state, predicted_cov

    def _is_valid_jump(
        self,
        predicted_state: np.ndarray,
        new_x: float,
        new_y: float,
        confidence: float,
    ) -> bool:
        """Check if position change is physically plausible."""
        pred_x, pred_y = predicted_state[0], predicted_state[1]
        distance = np.sqrt((new_x - pred_x) ** 2 + (new_y - pred_y) ** 2)

        # Allow larger jumps for higher confidence detections
        # At confidence=1.0, use max_velocity; at confidence=0.5, use 1.5x max_velocity
        confidence_factor = 1.0 + (1.0 - confidence)
        max_jump = self.config.max_velocity * confidence_factor

        return bool(distance <= max_jump)

    def _update(
        self,
        predicted_state: np.ndarray,
        predicted_cov: np.ndarray,
        z: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Kalman filter update step."""
        # Innovation (measurement residual)
        y = z - self._H @ predicted_state

        # Innovation covariance
        innov_cov = self._H @ predicted_cov @ self._H.T + self._R

        # Kalman gain
        kalman_gain = predicted_cov @ self._H.T @ np.linalg.inv(innov_cov)

        # State update
        updated_state = predicted_state + kalman_gain @ y

        # Covariance update (Joseph form for numerical stability)
        i_kh = np.eye(4) - kalman_gain @ self._H
        updated_cov = i_kh @ predicted_cov @ i_kh.T + kalman_gain @ self._R @ kalman_gain.T

        return updated_state, updated_cov

    def update(self, position: "BallPosition") -> "BallPosition":
        """
        Process a single ball position through the filter.

        Args:
            position: Raw ball position from detector

        Returns:
            Filtered ball position with smoothed coordinates
        """
        x, y, confidence = position.x, position.y, position.confidence

        # Initialize on first confident detection
        if not self._initialized:
            if confidence >= self.config.min_confidence_for_update:
                self._initialize(x, y)
                return self._create_output(position)
            else:
                # Not enough confidence to initialize, return as-is
                return position

        # Predict step
        predicted_state, predicted_cov = self._predict()

        # Determine if we should update with this measurement
        is_confident = confidence >= self.config.min_confidence_for_update
        is_valid = is_confident and self._is_valid_jump(predicted_state, x, y, confidence)

        if is_valid:
            # Measurement update
            z = np.array([x, y])
            self._state, self._covariance = self._update(predicted_state, predicted_cov, z)
            self._frames_since_confident = 0
        else:
            # Prediction only (occlusion or invalid jump)
            self._state = predicted_state
            self._covariance = predicted_cov
            self._frames_since_confident += 1

            if not is_confident:
                logger.debug(
                    f"Frame {position.frame_number}: Low confidence ({confidence:.2f}), "
                    f"using prediction only"
                )
            elif not self._is_valid_jump(predicted_state, x, y, confidence):
                logger.debug(
                    f"Frame {position.frame_number}: Jump rejected "
                    f"(pred={predicted_state[0]:.3f},{predicted_state[1]:.3f} "
                    f"meas={x:.3f},{y:.3f})"
                )

        return self._create_output(position)

    def _create_output(self, original: "BallPosition") -> "BallPosition":
        """Create output position with lag compensation and confidence decay."""
        from rallycut.tracking.ball_tracker import BallPosition

        assert self._state is not None

        x, y = self._state[0], self._state[1]
        vx, vy = self._state[2], self._state[3]

        # Apply lag compensation by extrapolating forward
        if self.config.enable_lag_compensation:
            x += vx * self.config.lag_frames
            y += vy * self.config.lag_frames

        # Clamp to valid range
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))

        # Decay confidence during occlusion
        confidence = original.confidence
        if self._frames_since_confident > 0:
            decay = 1.0 - (self._frames_since_confident / self.config.max_occlusion_frames)
            confidence = max(0.0, confidence * decay)

            # If we've been occluded too long, mark as lost
            if self._frames_since_confident >= self.config.max_occlusion_frames:
                confidence = 0.0

        return BallPosition(
            frame_number=original.frame_number,
            x=float(x),
            y=float(y),
            confidence=confidence,
        )

    def filter_batch(
        self,
        positions: list["BallPosition"],
    ) -> list["BallPosition"]:
        """
        Filter a complete list of ball positions.

        Args:
            positions: List of raw ball positions from detector

        Returns:
            List of filtered ball positions
        """
        if not positions:
            return []

        self.reset()

        # Sort by frame number to ensure temporal order
        sorted_positions = sorted(positions, key=lambda p: p.frame_number)

        # Forward pass
        filtered = []
        for pos in sorted_positions:
            filtered_pos = self.update(pos)
            filtered.append(filtered_pos)

        # Log filtering summary
        if filtered:
            raw_confidences = [p.confidence for p in sorted_positions]
            filtered_confidences = [p.confidence for p in filtered]

            # Calculate position shift from lag compensation
            if len(filtered) > 10:
                mid = len(filtered) // 2
                raw_x = sorted_positions[mid].x
                filt_x = filtered[mid].x
                shift_frames = (filt_x - raw_x) / max(0.001, abs(filt_x - filtered[mid-1].x)) if mid > 0 else 0
                logger.info(
                    f"Ball filter: {len(filtered)} positions, "
                    f"lag_comp={'ON' if self.config.enable_lag_compensation else 'OFF'}, "
                    f"lag_frames={self.config.lag_frames}, "
                    f"mid_shift={filt_x - raw_x:+.4f}"
                )
            else:
                logger.info(
                    f"Ball filter: {len(filtered)} positions, "
                    f"lag_comp={'ON' if self.config.enable_lag_compensation else 'OFF'}"
                )

        return filtered
