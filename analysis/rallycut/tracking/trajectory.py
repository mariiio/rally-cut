"""Trajectory processing for ball tracking visualization."""

import logging
from dataclasses import dataclass

import numpy as np
from scipy.interpolate import interp1d  # type: ignore[import-untyped]
from scipy.ndimage import gaussian_filter1d  # type: ignore[import-untyped]
from scipy.signal import savgol_filter  # type: ignore[import-untyped]

from rallycut.core.config import get_config
from rallycut.core.models import BallPosition

logger = logging.getLogger(__name__)


@dataclass
class TrajectorySegment:
    """A continuous segment of ball trajectory."""

    positions: list[BallPosition]
    start_frame: int
    end_frame: int

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame

    def get_points(self) -> list[tuple[float, float]]:
        """Get (x, y) points for drawing."""
        return [(p.x, p.y) for p in self.positions]


class TrajectoryProcessor:
    """
    Processes ball positions into smooth, drawable trajectories.

    Features:
    - Splits positions into continuous segments
    - Interpolates missing frames (linear or parabolic)
    - Smooths trajectories (Gaussian or Savitzky-Golay filter)
    - Provides trail data for visualization
    """

    def __init__(
        self,
        max_gap_frames: int | None = None,
        smooth_sigma: float | None = None,
        trail_length: int | None = None,
        interpolation_method: str | None = None,
        adaptive_smoothing: bool | None = None,
    ):
        """
        Initialize trajectory processor.

        Args:
            max_gap_frames: Maximum gap between positions to consider continuous
            smooth_sigma: Gaussian smoothing sigma (higher = smoother)
            trail_length: Number of frames to show in ball trail
            interpolation_method: "linear", "parabolic", or "spline"
            adaptive_smoothing: Use Savitzky-Golay filter instead of Gaussian
        """
        config = get_config()
        self.max_gap_frames = max_gap_frames or config.trajectory.max_gap_frames
        self.smooth_sigma = smooth_sigma or config.trajectory.smooth_sigma
        self.trail_length = trail_length or config.trajectory.trail_length
        self.interpolation_method = interpolation_method or config.trajectory.interpolation_method
        self.adaptive_smoothing = (
            adaptive_smoothing if adaptive_smoothing is not None else config.trajectory.adaptive_smoothing
        )

    def split_into_segments(
        self,
        positions: list[BallPosition],
    ) -> list[TrajectorySegment]:
        """
        Split positions into continuous trajectory segments.

        Args:
            positions: List of ball positions (may have gaps)

        Returns:
            List of continuous TrajectorySegments
        """
        if not positions:
            return []

        # Sort by frame index
        sorted_positions = sorted(positions, key=lambda p: p.frame_idx)

        segments = []
        current_segment = [sorted_positions[0]]

        for pos in sorted_positions[1:]:
            gap = pos.frame_idx - current_segment[-1].frame_idx

            if gap <= self.max_gap_frames:
                # Continue current segment
                current_segment.append(pos)
            else:
                # Start new segment
                if len(current_segment) >= 2:
                    segments.append(TrajectorySegment(
                        positions=current_segment,
                        start_frame=current_segment[0].frame_idx,
                        end_frame=current_segment[-1].frame_idx,
                    ))
                current_segment = [pos]

        # Add final segment
        if len(current_segment) >= 2:
            segments.append(TrajectorySegment(
                positions=current_segment,
                start_frame=current_segment[0].frame_idx,
                end_frame=current_segment[-1].frame_idx,
            ))

        return segments

    def interpolate_segment(
        self,
        segment: TrajectorySegment,
    ) -> TrajectorySegment:
        """
        Interpolate missing frames within a segment.

        Uses the configured interpolation method:
        - "linear": Simple linear interpolation
        - "parabolic": Quadratic fit for Y (accounts for gravity)
        - "spline": Smooth cubic spline

        Args:
            segment: Trajectory segment with potential gaps

        Returns:
            New segment with interpolated positions
        """
        if len(segment.positions) < 2:
            return segment

        # Extract frame indices and coordinates
        frames = np.array([p.frame_idx for p in segment.positions])
        x_coords = np.array([p.x for p in segment.positions])
        y_coords = np.array([p.y for p in segment.positions])

        # Generate positions for all frames in range
        all_frames = np.arange(segment.start_frame, segment.end_frame + 1)

        if self.interpolation_method == "parabolic" and len(segment.positions) >= 3:
            # Parabolic interpolation for Y (accounts for gravity in volleyball)
            # X remains linear (horizontal motion is roughly constant velocity)
            interp_x = interp1d(frames, x_coords, kind="linear", fill_value="extrapolate")
            interp_x_vals = interp_x(all_frames)

            # Quadratic fit for Y trajectory (parabola due to gravity)
            try:
                # Fit parabola: y = a*t^2 + b*t + c
                coeffs = np.polyfit(frames, y_coords, 2)
                interp_y_vals = np.polyval(coeffs, all_frames)
            except (np.linalg.LinAlgError, ValueError):
                # Fall back to linear if fit fails
                interp_y = interp1d(frames, y_coords, kind="linear", fill_value="extrapolate")
                interp_y_vals = interp_y(all_frames)

        elif self.interpolation_method == "spline" and len(segment.positions) >= 4:
            # Cubic spline for smooth curves
            try:
                from scipy.interpolate import UnivariateSpline

                # Smoothing factor based on number of points
                s = len(frames) * 0.5

                spline_x = UnivariateSpline(frames, x_coords, s=s)
                spline_y = UnivariateSpline(frames, y_coords, s=s)

                interp_x_vals = spline_x(all_frames)
                interp_y_vals = spline_y(all_frames)
            except (ValueError, TypeError):
                # Fall back to linear
                interp_x = interp1d(frames, x_coords, kind="linear", fill_value="extrapolate")
                interp_y = interp1d(frames, y_coords, kind="linear", fill_value="extrapolate")
                interp_x_vals = interp_x(all_frames)
                interp_y_vals = interp_y(all_frames)
        else:
            # Linear interpolation (default fallback)
            interp_x = interp1d(frames, x_coords, kind="linear", fill_value="extrapolate")
            interp_y = interp1d(frames, y_coords, kind="linear", fill_value="extrapolate")
            interp_x_vals = interp_x(all_frames)
            interp_y_vals = interp_y(all_frames)

        # Create new positions
        new_positions = []
        original_frames = set(frames)

        for i, frame_idx in enumerate(all_frames):
            is_original = int(frame_idx) in original_frames
            new_positions.append(BallPosition(
                frame_idx=int(frame_idx),
                x=float(interp_x_vals[i]),
                y=float(interp_y_vals[i]),
                confidence=1.0 if is_original else 0.5,
                is_predicted=not is_original,
            ))

        return TrajectorySegment(
            positions=new_positions,
            start_frame=segment.start_frame,
            end_frame=segment.end_frame,
        )

    def smooth_segment(
        self,
        segment: TrajectorySegment,
    ) -> TrajectorySegment:
        """
        Apply smoothing to trajectory.

        Uses Savitzky-Golay filter if adaptive_smoothing is enabled,
        otherwise uses Gaussian smoothing.

        Savitzky-Golay preserves trajectory shape better than Gaussian
        and doesn't blur sharp changes in direction.

        Args:
            segment: Trajectory segment

        Returns:
            New segment with smoothed positions
        """
        if len(segment.positions) < 3:
            return segment

        x_coords = np.array([p.x for p in segment.positions])
        y_coords = np.array([p.y for p in segment.positions])
        confidences = np.array([p.confidence for p in segment.positions])

        if self.adaptive_smoothing:
            # Savitzky-Golay filter - preserves trajectory shape
            # Window size must be odd and >= polyorder + 2
            window = min(11, len(segment.positions))
            if window % 2 == 0:
                window -= 1  # Make odd
            if window < 5:
                window = 5

            polyorder = min(3, window - 2)

            if len(segment.positions) >= window:
                try:
                    smooth_x = savgol_filter(x_coords, window, polyorder)
                    smooth_y = savgol_filter(y_coords, window, polyorder)
                except ValueError:
                    # Fall back to Gaussian
                    smooth_x = gaussian_filter1d(x_coords, sigma=self.smooth_sigma)
                    smooth_y = gaussian_filter1d(y_coords, sigma=self.smooth_sigma)
            else:
                # Too few points for Savitzky-Golay
                smooth_x = gaussian_filter1d(x_coords, sigma=self.smooth_sigma)
                smooth_y = gaussian_filter1d(y_coords, sigma=self.smooth_sigma)

            # Confidence-weighted blending: trust high-confidence points more
            final_x = np.zeros_like(x_coords)
            final_y = np.zeros_like(y_coords)

            for i in range(len(segment.positions)):
                # Higher confidence = more original, less smoothed
                w = min(confidences[i], 1.0)  # Clamp to [0, 1]
                final_x[i] = w * x_coords[i] + (1 - w) * smooth_x[i]
                final_y[i] = w * y_coords[i] + (1 - w) * smooth_y[i]

            smooth_x = final_x
            smooth_y = final_y
        else:
            # Original Gaussian smoothing
            smooth_x = gaussian_filter1d(x_coords, sigma=self.smooth_sigma)
            smooth_y = gaussian_filter1d(y_coords, sigma=self.smooth_sigma)

        # Create smoothed positions
        smoothed = []
        for i, pos in enumerate(segment.positions):
            smoothed.append(BallPosition(
                frame_idx=pos.frame_idx,
                x=float(smooth_x[i]),
                y=float(smooth_y[i]),
                confidence=pos.confidence,
                is_predicted=pos.is_predicted,
                bbox_width=pos.bbox_width,
                bbox_height=pos.bbox_height,
            ))

        return TrajectorySegment(
            positions=smoothed,
            start_frame=segment.start_frame,
            end_frame=segment.end_frame,
        )

    def process(
        self,
        positions: list[BallPosition],
        interpolate: bool = True,
        smooth: bool = True,
    ) -> list[TrajectorySegment]:
        """
        Full processing pipeline for ball positions.

        Args:
            positions: Raw ball positions
            interpolate: Whether to interpolate gaps
            smooth: Whether to smooth trajectories

        Returns:
            List of processed TrajectorySegments
        """
        segments = self.split_into_segments(positions)

        processed = []
        for segment in segments:
            if interpolate:
                segment = self.interpolate_segment(segment)
            if smooth:
                segment = self.smooth_segment(segment)
            processed.append(segment)

        return processed

    def get_trail_at_frame(
        self,
        segments: list[TrajectorySegment],
        frame_idx: int,
    ) -> list[tuple[float, float, float]]:
        """
        Get ball trail points for a specific frame.

        Args:
            segments: Processed trajectory segments
            frame_idx: Current frame index

        Returns:
            List of (x, y, alpha) tuples for trail rendering
            Alpha ranges from 0.0 (oldest) to 1.0 (newest)
        """
        # Find segment containing this frame
        for segment in segments:
            if segment.start_frame <= frame_idx <= segment.end_frame:
                trail = []

                for pos in segment.positions:
                    if pos.frame_idx > frame_idx:
                        break
                    if pos.frame_idx >= frame_idx - self.trail_length:
                        # Calculate alpha based on age
                        age = frame_idx - pos.frame_idx
                        alpha = 1.0 - (age / self.trail_length)
                        trail.append((pos.x, pos.y, alpha))

                return trail

        return []

    def get_position_at_frame(
        self,
        segments: list[TrajectorySegment],
        frame_idx: int,
    ) -> BallPosition | None:
        """
        Get ball position at a specific frame.

        Args:
            segments: Processed trajectory segments
            frame_idx: Frame index

        Returns:
            BallPosition if found, None otherwise
        """
        for segment in segments:
            for pos in segment.positions:
                if pos.frame_idx == frame_idx:
                    return pos
        return None
