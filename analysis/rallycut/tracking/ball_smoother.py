"""
Post-processing trajectory smoothing for ball tracking.

Applies optional smoothing filters after Kalman filtering to reduce
residual noise in ball trajectories for visualization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rallycut.tracking.ball_tracker import BallPosition

logger = logging.getLogger(__name__)


@dataclass
class TrajectorySmoothingConfig:
    """Configuration for post-processing trajectory smoothing."""

    # Savitzky-Golay filter for smooth trajectory
    enable_savgol: bool = True
    savgol_window: int = 7  # Window size in frames (must be odd)
    savgol_order: int = 2  # Polynomial order (2 = parabolic fit)

    # Outlier removal based on velocity
    enable_outlier_removal: bool = True
    max_velocity_threshold: float = 0.4  # Max screen/frame (40% is very fast)

    # Gap interpolation
    enable_interpolation: bool = False  # Disabled by default - can introduce artifacts
    max_gap_frames: int = 5  # Maximum gap size to interpolate

    # Minimum confidence to include in smoothing
    min_confidence: float = 0.3

    # Preserve original confidence values (if False, set to 1.0 after smoothing)
    preserve_confidence: bool = True


class TrajectoryPostProcessor:
    """Post-processing for ball trajectories.

    Applies optional smoothing and outlier removal after Kalman filtering.
    Designed for visualization quality, not real-time tracking.
    """

    def __init__(self, config: TrajectorySmoothingConfig | None = None):
        self.config = config or TrajectorySmoothingConfig()

    def process(self, positions: list[BallPosition]) -> list[BallPosition]:
        """Apply all enabled post-processing steps.

        Processing order:
        1. Outlier removal (marks impossible movements as low confidence)
        2. Savitzky-Golay smoothing (smooths trajectory)
        3. Gap interpolation (fills small gaps - optional, disabled by default)

        Args:
            positions: List of ball positions (after Kalman filtering).

        Returns:
            Post-processed ball positions.
        """
        if not positions:
            return positions

        result = list(positions)

        if self.config.enable_outlier_removal:
            result = self.remove_outliers(result)

        if self.config.enable_savgol:
            result = self.smooth_savgol(result)

        if self.config.enable_interpolation:
            result = self.interpolate_gaps(result, self.config.max_gap_frames)

        return result

    def remove_outliers(self, positions: list[BallPosition]) -> list[BallPosition]:
        """Remove outlier positions based on velocity.

        Marks positions as low-confidence if they represent impossible movements
        (velocity exceeds threshold).

        Args:
            positions: List of ball positions.

        Returns:
            Positions with outliers marked as low confidence.
        """
        from rallycut.tracking.ball_tracker import BallPosition

        if len(positions) < 2:
            return positions

        # Sort by frame number
        sorted_positions = sorted(positions, key=lambda p: p.frame_number)

        result = [sorted_positions[0]]

        for i in range(1, len(sorted_positions)):
            curr = sorted_positions[i]
            prev = result[-1] if result else sorted_positions[i - 1]

            # Skip if frames are not consecutive
            frame_gap = curr.frame_number - prev.frame_number
            if frame_gap <= 0:
                continue

            # Calculate velocity (normalized units per frame)
            dx = curr.x - prev.x
            dy = curr.y - prev.y
            distance = np.sqrt(dx * dx + dy * dy)
            velocity = distance / frame_gap  # Normalize by frame gap

            if velocity > self.config.max_velocity_threshold:
                # Mark as outlier by setting low confidence
                logger.debug(
                    f"Frame {curr.frame_number}: velocity {velocity:.3f} exceeds "
                    f"threshold {self.config.max_velocity_threshold:.3f}, marking as outlier"
                )
                result.append(BallPosition(
                    frame_number=curr.frame_number,
                    x=curr.x,
                    y=curr.y,
                    confidence=0.0,  # Mark as outlier
                ))
            else:
                result.append(curr)

        return result

    def smooth_savgol(self, positions: list[BallPosition]) -> list[BallPosition]:
        """Apply Savitzky-Golay filter to smooth trajectory.

        Savitzky-Golay fits a polynomial to local windows, providing smooth
        curves while preserving sharp features better than moving average.

        Args:
            positions: List of ball positions.

        Returns:
            Smoothed ball positions.
        """
        from scipy.signal import savgol_filter

        from rallycut.tracking.ball_tracker import BallPosition

        # Filter to confident positions only
        confident = [
            p for p in positions
            if p.confidence >= self.config.min_confidence
        ]

        if len(confident) < self.config.savgol_window:
            logger.debug(
                f"Not enough confident positions ({len(confident)}) for "
                f"Savitzky-Golay window ({self.config.savgol_window}), skipping smoothing"
            )
            return positions

        # Sort by frame number
        confident = sorted(confident, key=lambda p: p.frame_number)

        # Extract x and y arrays
        x_values = np.array([p.x for p in confident])
        y_values = np.array([p.y for p in confident])
        frames = [p.frame_number for p in confident]
        confidences = [p.confidence for p in confident]

        # Ensure window size is odd and <= data length
        window = min(self.config.savgol_window, len(confident))
        if window % 2 == 0:
            window -= 1
        if window < 3:
            return positions

        # Ensure polynomial order is less than window size
        order = min(self.config.savgol_order, window - 1)

        try:
            # Apply Savitzky-Golay filter
            x_smooth = savgol_filter(x_values, window, order)
            y_smooth = savgol_filter(y_values, window, order)
        except ValueError as e:
            logger.warning(f"Savitzky-Golay filter failed: {e}")
            return positions

        # Clamp to valid range
        x_smooth = np.clip(x_smooth, 0.0, 1.0)
        y_smooth = np.clip(y_smooth, 0.0, 1.0)

        # Create lookup of smoothed positions by frame
        smoothed_by_frame = {
            frames[i]: (float(x_smooth[i]), float(y_smooth[i]), confidences[i])
            for i in range(len(frames))
        }

        # Apply smoothing to original positions
        result = []
        for pos in positions:
            if pos.frame_number in smoothed_by_frame:
                sx, sy, conf = smoothed_by_frame[pos.frame_number]
                result.append(BallPosition(
                    frame_number=pos.frame_number,
                    x=sx,
                    y=sy,
                    confidence=conf if self.config.preserve_confidence else 1.0,
                ))
            else:
                # Keep original if not in confident set
                result.append(pos)

        return result

    def interpolate_gaps(
        self,
        positions: list[BallPosition],
        max_gap_frames: int = 5,
    ) -> list[BallPosition]:
        """Interpolate small gaps in the trajectory.

        Fills in missing frames with linear interpolation when gaps are small.

        Args:
            positions: List of ball positions.
            max_gap_frames: Maximum gap size to interpolate.

        Returns:
            Positions with small gaps filled.
        """
        from rallycut.tracking.ball_tracker import BallPosition

        if len(positions) < 2:
            return positions

        # Filter to confident positions and sort
        confident = sorted(
            [p for p in positions if p.confidence >= self.config.min_confidence],
            key=lambda p: p.frame_number,
        )

        if len(confident) < 2:
            return positions

        # Create result with original positions
        result_by_frame = {p.frame_number: p for p in positions}

        # Interpolate gaps
        for i in range(1, len(confident)):
            prev = confident[i - 1]
            curr = confident[i]
            gap = curr.frame_number - prev.frame_number

            if 1 < gap <= max_gap_frames:
                # Linear interpolation
                for j in range(1, gap):
                    t = j / gap
                    interp_frame = prev.frame_number + j
                    interp_x = prev.x + t * (curr.x - prev.x)
                    interp_y = prev.y + t * (curr.y - prev.y)
                    # Interpolated confidence is average of endpoints, scaled by distance from gap center
                    interp_conf = min(prev.confidence, curr.confidence) * 0.8

                    if interp_frame not in result_by_frame:
                        result_by_frame[interp_frame] = BallPosition(
                            frame_number=interp_frame,
                            x=float(interp_x),
                            y=float(interp_y),
                            confidence=float(interp_conf),
                        )

        # Return sorted by frame
        return sorted(result_by_frame.values(), key=lambda p: p.frame_number)
