"""Fast motion-based game state detection.

Uses frame differencing to detect activity - much faster than ML models.
"""

from typing import Callable, Optional

import cv2
import numpy as np

from rallycut.core.models import GameState, GameStateResult
from rallycut.core.video import Video


class MotionDetector:
    """
    Fast motion-based activity detection.

    Uses frame differencing and motion intensity to detect rallies.
    Volleyball rallies have sustained, high-intensity motion.
    """

    ANALYSIS_SIZE = (320, 180)  # Small size for fast processing

    def __init__(
        self,
        high_motion_threshold: float = 0.08,  # Strong motion indicator
        low_motion_threshold: float = 0.04,  # Some motion
        window_size: int = 5,  # Frames to average over
    ):
        """
        Args:
            high_motion_threshold: Threshold for high motion (rally likely)
            low_motion_threshold: Threshold for low motion (dead time likely)
            window_size: Number of frames to smooth over
        """
        self.high_motion_threshold = high_motion_threshold
        self.low_motion_threshold = low_motion_threshold
        self.window_size = window_size

    def analyze_video(
        self,
        video: Video,
        stride: int = 8,
        progress_callback: Optional[Callable[[float, str], None]] = None,
        limit_seconds: Optional[float] = None,
    ) -> list[GameStateResult]:
        """
        Analyze video for motion/activity using sliding window smoothing.

        Uses sequential frame reading (iter_frames) instead of seeking for
        10-100x faster performance on long videos.

        Args:
            video: Video to analyze
            stride: Frames to skip between analyses
            progress_callback: Callback for progress updates
            limit_seconds: Only analyze first N seconds (for testing)

        Returns:
            List of GameStateResult
        """
        results = []
        total_frames = video.info.frame_count

        # Apply limit if specified
        if limit_seconds is not None:
            max_frames = min(total_frames, int(limit_seconds * video.info.fps))
        else:
            max_frames = total_frames

        prev_gray = None
        motion_history = []  # Store recent motion values

        total_positions = (max_frames + stride - 1) // stride
        processed = 0

        # Use sequential reading with stride - MUCH faster than seeking
        for frame_idx, frame in video.iter_frames(end_frame=max_frames, step=stride):
            # Resize for speed
            small = cv2.resize(frame, self.ANALYSIS_SIZE, interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            motion_ratio = 0.0
            if prev_gray is not None:
                # Calculate frame difference
                diff = cv2.absdiff(prev_gray, gray)
                _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
                motion_ratio = np.count_nonzero(thresh) / thresh.size

            prev_gray = gray

            # Add to history
            motion_history.append(motion_ratio)
            if len(motion_history) > self.window_size:
                motion_history.pop(0)

            # Calculate smoothed motion (average of recent frames)
            avg_motion = sum(motion_history) / len(motion_history)

            # Determine state based on motion intensity
            # High sustained motion = PLAY
            # Low motion = NO_PLAY
            if avg_motion >= self.high_motion_threshold:
                state = GameState.PLAY
                confidence = min(0.95, 0.7 + avg_motion * 5)
            elif avg_motion >= self.low_motion_threshold:
                # Ambiguous - could be transition
                state = GameState.PLAY
                confidence = 0.6
            else:
                state = GameState.NO_PLAY
                confidence = min(0.9, 0.7 + (self.low_motion_threshold - avg_motion) * 10)

            results.append(
                GameStateResult(
                    state=state,
                    confidence=confidence,
                    start_frame=frame_idx,
                    end_frame=frame_idx + stride - 1,
                )
            )

            processed += 1
            if progress_callback and processed % 10 == 0:
                progress = processed / total_positions
                progress_callback(progress, f"Frame {frame_idx}/{max_frames}")

        return results
