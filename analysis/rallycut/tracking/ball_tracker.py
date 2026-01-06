"""Ball tracking with Kalman filter for trajectory smoothing and prediction."""

import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from filterpy.kalman import KalmanFilter  # type: ignore[import-untyped]

from rallycut.core.config import get_config
from rallycut.core.models import BallPosition
from rallycut.core.video import Video

logger = logging.getLogger(__name__)


@dataclass
class TrackingResult:
    """Result of ball tracking on a video segment."""

    positions: list[BallPosition]
    detection_rate: float
    total_frames: int
    detected_frames: int


class BallTracker:
    """
    Tracks volleyball using YOLO detection + Kalman filter.

    The Kalman filter predicts ball position in frames where detection fails,
    and smooths the trajectory to reduce jitter.

    Features:
    - Multi-candidate detection with temporal validation
    - Confidence-weighted Kalman measurement noise
    - Stride-aware time steps
    """

    def __init__(
        self,
        model_path: Path | None = None,
        device: str | None = None,
        confidence_threshold: float | None = None,
        max_missing_frames: int | None = None,
        use_predictions: bool = False,
    ):
        """
        Initialize ball tracker.

        Args:
            model_path: Path to YOLO ball detector weights
            device: Device for inference (cpu/cuda/mps)
            confidence_threshold: Minimum detection confidence
            max_missing_frames: Max frames to predict without detection before losing track
            use_predictions: Whether to use Kalman predictions for missing frames
        """
        config = get_config()
        self.model_path = model_path or config.ball_detector_path
        self.device = device or config.device
        self.confidence_threshold = (
            confidence_threshold or config.ball_tracking.confidence_threshold
        )
        self.max_missing_frames = max_missing_frames or config.ball_tracking.max_missing_frames
        self.use_predictions = use_predictions
        self._detector: Any = None
        self._kalman: KalmanFilter | None = None

    def _get_detector(self) -> Any:
        """Lazy load the YOLO detector."""
        if self._detector is None:
            from lib.volleyball_ml.yolo_detector import BallDetector

            config = get_config()
            self._detector = BallDetector(
                model_path=self.model_path,
                device=self.device,
                confidence_threshold=self.confidence_threshold,
                max_candidates=config.ball_tracking.max_candidates,
                min_aspect_ratio=config.ball_tracking.min_aspect_ratio,
                max_aspect_ratio=config.ball_tracking.max_aspect_ratio,
            )
        return self._detector

    def _init_kalman(self, initial_pos: tuple[float, float], dt: float = 1.0) -> KalmanFilter:
        """
        Initialize Kalman filter for ball tracking.

        State: [x, y, vx, vy] (position and velocity)
        Measurement: [x, y] (detected position)

        Args:
            initial_pos: Initial (x, y) position
            dt: Time step (typically = stride, for proper velocity estimation)
        """
        config = get_config()
        kf = KalmanFilter(dim_x=4, dim_z=2)

        # State transition matrix (constant velocity model)
        # x' = x + vx*dt, y' = y + vy*dt
        kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ])

        # Measurement matrix (we only observe position)
        kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ])

        # Measurement noise (from config, will be adapted per detection)
        r = config.ball_tracking.kalman_measurement_noise
        kf.R = np.array([
            [r, 0],
            [0, r],
        ])

        # Process noise (acceleration uncertainty, from config)
        # Scale with dt^2 for proper physics
        q = config.ball_tracking.kalman_process_noise
        dt2 = dt * dt
        kf.Q = np.array([
            [q * dt2, 0, 0, 0],
            [0, q * dt2, 0, 0],
            [0, 0, q * 2, 0],
            [0, 0, 0, q * 2],
        ])

        # Initial state covariance (lower = faster convergence)
        kf.P *= 25  # Reduced from 100 for faster convergence

        # Initial state
        kf.x = np.array([initial_pos[0], initial_pos[1], 0, 0])

        return kf

    def _update_kalman_with_confidence(
        self,
        kalman: KalmanFilter,
        detection: BallPosition,
    ) -> None:
        """
        Update Kalman filter with confidence-weighted measurement noise.

        High confidence = lower noise = trust detection more.
        Low confidence = higher noise = rely more on prediction.
        """
        config = get_config()
        base_noise = config.ball_tracking.kalman_measurement_noise

        # Scale noise inversely with confidence
        # confidence 1.0 -> noise = base_noise
        # confidence 0.5 -> noise = base_noise * 2
        # confidence 0.3 -> noise = base_noise * 3.3
        noise_scale = 1.0 / max(detection.confidence, 0.3)
        r = base_noise * noise_scale

        kalman.R = np.array([
            [r, 0],
            [0, r],
        ])

        kalman.update(np.array([detection.x, detection.y]))

    def _validate_detection(
        self,
        candidates: list[BallPosition],
        kalman: KalmanFilter | None,
        max_velocity: float,
        force_reset_threshold: float = 0.4,
    ) -> tuple[BallPosition | None, bool]:
        """
        Select best candidate using temporal validation.

        Combines detection confidence with proximity to Kalman prediction
        to reduce false positives. Can signal track reset for high-confidence
        detections that are far from current track.

        Args:
            candidates: List of detection candidates (sorted by confidence)
            kalman: Current Kalman filter state (None if tracking not initialized)
            max_velocity: Maximum allowed movement per frame (pixels)
            force_reset_threshold: Confidence above which to force track reset

        Returns:
            Tuple of (best detection or None, should_reset_kalman)
        """
        if not candidates:
            return None, False

        if kalman is None:
            # No prior tracking - return highest confidence candidate
            return candidates[0], False

        # Get predicted position from Kalman
        predicted_x, predicted_y = kalman.x[0], kalman.x[1]

        # Score each candidate: combine confidence with proximity to prediction
        scored = []
        best_distant_candidate = None  # Track high-confidence but distant candidates

        for cand in candidates:
            distance = np.sqrt((cand.x - predicted_x) ** 2 + (cand.y - predicted_y) ** 2)

            if distance <= max_velocity:
                # Within velocity limit - score normally
                proximity_score = 1.0 / (1.0 + distance / 30.0)
                combined_score = 0.4 * cand.confidence + 0.6 * proximity_score
                scored.append((cand, combined_score))
            elif cand.confidence >= force_reset_threshold:
                # High confidence but far away - candidate for track reset
                # This handles when tracking gets locked onto a false positive
                if best_distant_candidate is None or cand.confidence > best_distant_candidate.confidence:
                    best_distant_candidate = cand

        if scored:
            # Return best scored candidate
            scored.sort(key=lambda x: x[1], reverse=True)
            best_near = scored[0][0]

            # Check if there's a much better distant candidate that we should reset to
            # This handles the case where we're stuck tracking a false positive
            if best_distant_candidate is not None:
                conf_advantage = best_distant_candidate.confidence - best_near.confidence
                if conf_advantage > 0.1:  # Distant candidate is significantly more confident
                    return best_distant_candidate, True

            return best_near, False

        if best_distant_candidate is not None:
            # No valid candidates near prediction, but have high-confidence far candidate
            # Signal to reset tracking to this new target
            return best_distant_candidate, True

        return None, False

    def track_video(
        self,
        video: Video,
        stride: int = 1,
        start_frame: int = 0,
        end_frame: int | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> TrackingResult:
        """
        Track ball through video segment.

        Args:
            video: Video to track
            stride: Frame stride (1 = every frame, 2 = every other frame)
            start_frame: Starting frame index
            end_frame: Ending frame index (None = end of video)
            progress_callback: Callback for progress updates

        Returns:
            TrackingResult with positions and stats
        """
        config = get_config()
        edge_margin = config.ball_tracking.edge_margin
        max_velocity = config.ball_tracking.max_velocity_pixels * stride
        detector = self._get_detector()

        if end_frame is None:
            end_frame = video.info.frame_count

        positions: list[BallPosition] = []
        frames_processed = 0
        frames_detected = 0
        missing_count = 0
        kalman: KalmanFilter | None = None

        frame_indices = list(range(start_frame, end_frame, stride))
        total_frames = len(frame_indices)

        for i, frame_idx in enumerate(frame_indices):
            frame = video.read_frame(frame_idx)
            if frame is None:
                continue

            frames_processed += 1

            # Get all detection candidates
            candidates = detector.detect_frame_candidates(frame, frame_idx)

            # Apply temporal validation to select best candidate
            detection, should_reset = self._validate_detection(candidates, kalman, max_velocity)

            # Reset Kalman if we detected a high-confidence ball far from current track
            if should_reset:
                kalman = None
                missing_count = 0

            if detection is not None:
                # Check if detection is too close to frame edge
                near_edge = (
                    detection.x < edge_margin
                    or detection.x > video.info.width - edge_margin
                    or detection.y < edge_margin
                    or detection.y > video.info.height - edge_margin
                )

                if near_edge:
                    # Treat as no detection - ball is leaving/entering frame
                    if kalman is not None:
                        missing_count += 1
                        if missing_count >= self.max_missing_frames:
                            kalman = None
                            missing_count = 0
                    continue

                # Ball detected in valid area
                frames_detected += 1
                missing_count = 0

                if kalman is None:
                    # Initialize Kalman filter with stride-aware dt
                    kalman = self._init_kalman((detection.x, detection.y), dt=float(stride))
                else:
                    # Predict then update with confidence-weighted noise
                    kalman.predict()
                    self._update_kalman_with_confidence(kalman, detection)

                # Use filtered position
                filtered_x, filtered_y = kalman.x[0], kalman.x[1]

                positions.append(
                    BallPosition(
                        frame_idx=frame_idx,
                        x=filtered_x,
                        y=filtered_y,
                        confidence=detection.confidence,
                        is_predicted=False,
                        bbox_width=detection.bbox_width,
                        bbox_height=detection.bbox_height,
                    )
                )

            elif kalman is not None and missing_count < self.max_missing_frames:
                # No detection - optionally use Kalman prediction
                missing_count += 1
                kalman.predict()

                if self.use_predictions:
                    pred_x, pred_y = kalman.x[0], kalman.x[1]
                    vel_x, vel_y = kalman.x[2], kalman.x[3]

                    # Check if ball is heading out of frame or already outside
                    in_frame = (
                        edge_margin < pred_x < video.info.width - edge_margin
                        and edge_margin < pred_y < video.info.height - edge_margin
                    )

                    # Check if velocity is taking ball further out
                    heading_out = (
                        (pred_x < edge_margin and vel_x < 0)
                        or (pred_x > video.info.width - edge_margin and vel_x > 0)
                        or (pred_y < edge_margin and vel_y < 0)
                        or (pred_y > video.info.height - edge_margin and vel_y > 0)
                    )

                    if in_frame and not heading_out:
                        positions.append(
                            BallPosition(
                                frame_idx=frame_idx,
                                x=pred_x,
                                y=pred_y,
                                confidence=0.0,
                                is_predicted=True,
                            )
                        )
                    else:
                        # Ball went out of frame - reset tracking
                        kalman = None
                        missing_count = 0

            elif missing_count >= self.max_missing_frames:
                # Lost track - reset Kalman
                kalman = None
                missing_count = 0

            # Progress callback
            if progress_callback and i % 10 == 0:
                progress = (i + 1) / total_frames
                progress_callback(progress, f"Frame {frame_idx}/{end_frame}")

        detection_rate = frames_detected / frames_processed if frames_processed > 0 else 0

        return TrackingResult(
            positions=positions,
            detection_rate=detection_rate,
            total_frames=frames_processed,
            detected_frames=frames_detected,
        )

    def track_segment(
        self,
        video: Video,
        start_time: float,
        end_time: float,
        stride: int = 1,
        progress_callback: Callable[[float, str], None] | None = None,
    ) -> TrackingResult:
        """
        Track ball in a time segment.

        Args:
            video: Video to track
            start_time: Start time in seconds
            end_time: End time in seconds
            stride: Frame stride
            progress_callback: Callback for progress updates

        Returns:
            TrackingResult with positions
        """
        fps = video.info.fps
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        return self.track_video(
            video,
            stride=stride,
            start_frame=start_frame,
            end_frame=end_frame,
            progress_callback=progress_callback,
        )
