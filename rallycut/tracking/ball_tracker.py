"""Ball tracking with Kalman filter for trajectory smoothing and prediction."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from filterpy.kalman import KalmanFilter

from rallycut.core.config import get_config
from rallycut.core.models import BallPosition
from rallycut.core.video import Video


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
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        device: Optional[str] = None,
        confidence_threshold: float = 0.3,
        max_missing_frames: int = 10,
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
        self.confidence_threshold = confidence_threshold
        self.max_missing_frames = max_missing_frames
        self.use_predictions = use_predictions
        self._detector = None
        self._kalman = None

    def _get_detector(self):
        """Lazy load the YOLO detector."""
        if self._detector is None:
            from lib.volleyball_ml.yolo_detector import BallDetector

            self._detector = BallDetector(
                model_path=self.model_path,
                device=self.device,
                confidence_threshold=self.confidence_threshold,
            )
        return self._detector

    def _init_kalman(self, initial_pos: tuple[float, float]) -> KalmanFilter:
        """
        Initialize Kalman filter for ball tracking.

        State: [x, y, vx, vy] (position and velocity)
        Measurement: [x, y] (detected position)
        """
        kf = KalmanFilter(dim_x=4, dim_z=2)

        # State transition matrix (constant velocity model)
        # x' = x + vx*dt, y' = y + vy*dt
        dt = 1.0  # Frame interval (normalized)
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

        # Measurement noise
        kf.R = np.array([
            [10, 0],
            [0, 10],
        ])

        # Process noise (acceleration uncertainty)
        q = 5.0  # Process noise scale
        kf.Q = np.array([
            [q, 0, 0, 0],
            [0, q, 0, 0],
            [0, 0, q*2, 0],
            [0, 0, 0, q*2],
        ])

        # Initial state covariance
        kf.P *= 100

        # Initial state
        kf.x = np.array([initial_pos[0], initial_pos[1], 0, 0])

        return kf

    def track_video(
        self,
        video: Video,
        stride: int = 1,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
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
        detector = self._get_detector()

        if end_frame is None:
            end_frame = video.info.frame_count

        positions: list[BallPosition] = []
        frames_processed = 0
        frames_detected = 0
        missing_count = 0
        kalman: Optional[KalmanFilter] = None

        frame_indices = list(range(start_frame, end_frame, stride))
        total_frames = len(frame_indices)

        for i, frame_idx in enumerate(frame_indices):
            frame = video.read_frame(frame_idx)
            if frame is None:
                continue

            frames_processed += 1

            # Detect ball
            detection = detector.detect_frame(frame, frame_idx)

            if detection is not None:
                # Check if detection is too close to frame edge
                edge_margin = 80
                near_edge = (
                    detection.x < edge_margin or
                    detection.x > video.info.width - edge_margin or
                    detection.y < edge_margin or
                    detection.y > video.info.height - edge_margin
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
                    # Initialize Kalman filter
                    kalman = self._init_kalman((detection.x, detection.y))
                else:
                    # Update Kalman with measurement
                    kalman.predict()
                    kalman.update(np.array([detection.x, detection.y]))

                # Use filtered position
                filtered_x, filtered_y = kalman.x[0], kalman.x[1]

                positions.append(BallPosition(
                    frame_idx=frame_idx,
                    x=filtered_x,
                    y=filtered_y,
                    confidence=detection.confidence,
                    is_predicted=False,
                ))

            elif kalman is not None and missing_count < self.max_missing_frames:
                # No detection - optionally use Kalman prediction
                missing_count += 1
                kalman.predict()

                if self.use_predictions:
                    pred_x, pred_y = kalman.x[0], kalman.x[1]
                    vel_x, vel_y = kalman.x[2], kalman.x[3]

                    # Check if ball is heading out of frame or already outside
                    margin = 80  # pixels from edge
                    in_frame = (
                        margin < pred_x < video.info.width - margin and
                        margin < pred_y < video.info.height - margin
                    )

                    # Check if velocity is taking ball further out
                    heading_out = (
                        (pred_x < margin and vel_x < 0) or
                        (pred_x > video.info.width - margin and vel_x > 0) or
                        (pred_y < margin and vel_y < 0) or
                        (pred_y > video.info.height - margin and vel_y > 0)
                    )

                    if in_frame and not heading_out:
                        positions.append(BallPosition(
                            frame_idx=frame_idx,
                            x=pred_x,
                            y=pred_y,
                            confidence=0.0,
                            is_predicted=True,
                        ))
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
    ) -> TrackingResult:
        """
        Track ball in a time segment.

        Args:
            video: Video to track
            start_time: Start time in seconds
            end_time: End time in seconds
            stride: Frame stride

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
        )
