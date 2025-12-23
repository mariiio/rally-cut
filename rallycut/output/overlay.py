"""Video overlay rendering for ball trajectory visualization."""

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

from rallycut.core.video import Video
from rallycut.tracking.trajectory import TrajectoryProcessor, TrajectorySegment


@dataclass
class OverlayStyle:
    """Configuration for trajectory overlay appearance."""

    # Ball marker
    ball_color: tuple[int, int, int] = (0, 255, 255)  # Yellow (BGR)
    ball_radius: int = 12
    ball_thickness: int = -1  # Filled

    # Trail
    trail_color: tuple[int, int, int] = (0, 200, 255)  # Orange (BGR)
    trail_max_radius: int = 8
    trail_min_radius: int = 2

    # Predicted positions
    predicted_color: tuple[int, int, int] = (100, 100, 255)  # Light red (BGR)

    # Trail line
    draw_trail_line: bool = True
    trail_line_color: tuple[int, int, int] = (0, 150, 200)
    trail_line_thickness: int = 2


class OverlayRenderer:
    """
    Renders ball trajectory overlay on video frames.

    Features:
    - Ball position marker
    - Fading trail showing recent positions
    - Different styling for predicted vs detected positions
    - Smooth trail line connecting positions
    """

    def __init__(
        self,
        style: Optional[OverlayStyle] = None,
        trail_length: int = 15,
    ):
        """
        Initialize overlay renderer.

        Args:
            style: Visual style configuration
            trail_length: Number of frames to show in trail
        """
        self.style = style or OverlayStyle()
        self.trail_length = trail_length

    def draw_ball(
        self,
        frame: np.ndarray,
        x: float,
        y: float,
        is_predicted: bool = False,
    ) -> np.ndarray:
        """
        Draw ball marker at position.

        Args:
            frame: Video frame (modified in place)
            x: Ball x position
            y: Ball y position
            is_predicted: Whether position is predicted (different color)

        Returns:
            Frame with ball drawn
        """
        color = self.style.predicted_color if is_predicted else self.style.ball_color
        center = (int(x), int(y))

        cv2.circle(
            frame,
            center,
            self.style.ball_radius,
            color,
            self.style.ball_thickness,
        )

        # Draw outline for visibility
        cv2.circle(
            frame,
            center,
            self.style.ball_radius,
            (0, 0, 0),
            2,
        )

        return frame

    def draw_trail(
        self,
        frame: np.ndarray,
        trail_points: list[tuple[float, float, float]],
    ) -> np.ndarray:
        """
        Draw fading trail of ball positions.

        Args:
            frame: Video frame (modified in place)
            trail_points: List of (x, y, alpha) tuples

        Returns:
            Frame with trail drawn
        """
        if not trail_points:
            return frame

        # Draw trail line first (behind circles)
        if self.style.draw_trail_line and len(trail_points) >= 2:
            points = [(int(x), int(y)) for x, y, _ in trail_points]
            for i in range(len(points) - 1):
                alpha = trail_points[i + 1][2]
                # Fade the line color
                color = tuple(int(c * alpha) for c in self.style.trail_line_color)
                cv2.line(
                    frame,
                    points[i],
                    points[i + 1],
                    color,
                    self.style.trail_line_thickness,
                )

        # Draw trail circles (oldest first, so newest on top)
        for x, y, alpha in trail_points[:-1]:  # Exclude current position
            # Scale radius and color based on alpha
            radius = int(
                self.style.trail_min_radius +
                (self.style.trail_max_radius - self.style.trail_min_radius) * alpha
            )
            color = tuple(int(c * alpha) for c in self.style.trail_color)

            cv2.circle(
                frame,
                (int(x), int(y)),
                radius,
                color,
                -1,
            )

        return frame

    def render_frame(
        self,
        frame: np.ndarray,
        segments: list[TrajectorySegment],
        frame_idx: int,
        processor: TrajectoryProcessor,
    ) -> np.ndarray:
        """
        Render overlay for a single frame.

        Args:
            frame: Video frame
            segments: Processed trajectory segments
            frame_idx: Current frame index
            processor: Trajectory processor for trail calculation

        Returns:
            Frame with overlay rendered
        """
        # Get trail points
        trail = processor.get_trail_at_frame(segments, frame_idx)

        if trail:
            # Draw trail
            frame = self.draw_trail(frame, trail)

            # Draw current ball position
            current_pos = processor.get_position_at_frame(segments, frame_idx)
            if current_pos:
                frame = self.draw_ball(
                    frame,
                    current_pos.x,
                    current_pos.y,
                    is_predicted=current_pos.is_predicted,
                )

        return frame

    def render_video(
        self,
        video: Video,
        segments: list[TrajectorySegment],
        processor: TrajectoryProcessor,
        output_path: Path,
        start_frame: int = 0,
        end_frame: Optional[int] = None,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Path:
        """
        Render overlay on video and save to file.

        Args:
            video: Source video
            segments: Processed trajectory segments
            processor: Trajectory processor
            output_path: Path for output video
            start_frame: Starting frame
            end_frame: Ending frame (None = end of video)
            progress_callback: Progress callback

        Returns:
            Path to rendered video
        """
        if end_frame is None:
            end_frame = video.info.frame_count

        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            video.info.fps,
            (video.info.width, video.info.height),
        )

        total_frames = end_frame - start_frame

        try:
            for i, frame_idx in enumerate(range(start_frame, end_frame)):
                frame = video.read_frame(frame_idx)
                if frame is None:
                    continue

                # Render overlay
                frame = self.render_frame(frame, segments, frame_idx, processor)

                writer.write(frame)

                # Progress
                if progress_callback and i % 30 == 0:
                    progress = (i + 1) / total_frames
                    progress_callback(progress, f"Frame {frame_idx}/{end_frame}")

        finally:
            writer.release()

        return output_path

    def render_segment(
        self,
        video: Video,
        segments: list[TrajectorySegment],
        processor: TrajectoryProcessor,
        output_path: Path,
        start_time: float,
        end_time: float,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ) -> Path:
        """
        Render overlay for a time segment.

        Args:
            video: Source video
            segments: Processed trajectory segments
            processor: Trajectory processor
            output_path: Output path
            start_time: Start time in seconds
            end_time: End time in seconds
            progress_callback: Progress callback

        Returns:
            Path to rendered video
        """
        fps = video.info.fps
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        return self.render_video(
            video,
            segments,
            processor,
            output_path,
            start_frame=start_frame,
            end_frame=end_frame,
            progress_callback=progress_callback,
        )
