"""
Ball feature extraction for rally segment validation.

Extracts aggregate ball visibility and trajectory features from tracking results
to validate detected rally segments.
"""

from dataclasses import dataclass

import numpy as np

from rallycut.tracking.ball_tracker import BallPosition


@dataclass
class SegmentBallFeatures:
    """Aggregate ball features for a video segment."""

    detection_rate: float  # % frames with ball detected (conf > threshold)
    mean_confidence: float  # Average detection confidence
    trajectory_variance: float  # Ball position variance (high = active play)
    velocity_mean: float  # Mean ball velocity (normalized units/frame)
    velocity_variance: float  # Velocity variance

    def is_valid_rally(
        self,
        min_detection_rate: float = 0.2,
        min_trajectory_variance: float = 0.03,
    ) -> bool:
        """
        Check if features indicate valid rally.

        Rally segments have:
        - High detection rate (ball visible most of the time)
        - High trajectory variance (ball moving around court)

        Dead time has:
        - Low/no detection (ball not visible or held)
        - Low variance (ball stationary or held by player)

        Args:
            min_detection_rate: Minimum % of frames with detected ball.
                Beach uses lower threshold (0.2) due to outdoor conditions.
            min_trajectory_variance: Minimum position variance threshold.

        Returns:
            True if features indicate valid rally.
        """
        return (
            self.detection_rate >= min_detection_rate
            and self.trajectory_variance >= min_trajectory_variance
        )


def compute_ball_features(
    positions: list[BallPosition],
    total_frames: int,
    confidence_threshold: float = 0.5,
) -> SegmentBallFeatures:
    """
    Compute aggregate ball features from tracking positions.

    Args:
        positions: List of ball positions from tracking.
        total_frames: Total number of frames in segment.
        confidence_threshold: Minimum confidence for detection to count.

    Returns:
        SegmentBallFeatures with aggregate statistics.
    """
    if not positions or total_frames == 0:
        return SegmentBallFeatures(
            detection_rate=0.0,
            mean_confidence=0.0,
            trajectory_variance=0.0,
            velocity_mean=0.0,
            velocity_variance=0.0,
        )

    # Filter to confident detections
    confident = [p for p in positions if p.confidence >= confidence_threshold]

    # Detection rate
    detection_rate = len(confident) / total_frames if total_frames > 0 else 0.0

    # Mean confidence (over all positions, not just confident ones)
    mean_confidence = np.mean([p.confidence for p in positions]) if positions else 0.0

    if len(confident) < 2:
        # Not enough detections for trajectory analysis
        return SegmentBallFeatures(
            detection_rate=detection_rate,
            mean_confidence=float(mean_confidence),
            trajectory_variance=0.0,
            velocity_mean=0.0,
            velocity_variance=0.0,
        )

    # Extract positions as arrays
    xs = np.array([p.x for p in confident])
    ys = np.array([p.y for p in confident])

    # Trajectory variance (combined x,y variance)
    trajectory_variance = float(np.var(xs) + np.var(ys))

    # Compute velocities between consecutive frames
    # Note: positions may not be consecutive frames, so velocity is approximate
    velocities = []
    for i in range(1, len(confident)):
        dx = confident[i].x - confident[i - 1].x
        dy = confident[i].y - confident[i - 1].y
        # Frame gap for normalization
        frame_gap = max(1, confident[i].frame_number - confident[i - 1].frame_number)
        velocity = np.sqrt(dx**2 + dy**2) / frame_gap
        velocities.append(velocity)

    if velocities:
        velocity_mean = float(np.mean(velocities))
        velocity_variance = float(np.var(velocities))
    else:
        velocity_mean = 0.0
        velocity_variance = 0.0

    return SegmentBallFeatures(
        detection_rate=detection_rate,
        mean_confidence=float(mean_confidence),
        trajectory_variance=trajectory_variance,
        velocity_mean=velocity_mean,
        velocity_variance=velocity_variance,
    )


def compute_ball_features_subsampled(
    positions: list[BallPosition],
    sample_rate: int = 3,
    confidence_threshold: float = 0.5,
) -> SegmentBallFeatures:
    """
    Compute features from subsampled positions (for faster validation).

    Args:
        positions: List of ball positions from tracking.
        sample_rate: Only use every Nth position.
        confidence_threshold: Minimum confidence for detection to count.

    Returns:
        SegmentBallFeatures with aggregate statistics.
    """
    if sample_rate <= 1:
        return compute_ball_features(positions, len(positions), confidence_threshold)

    # Subsample positions
    sampled = positions[::sample_rate]

    # Estimate total frames from subsampling
    estimated_frames = len(sampled) if sampled else 0

    return compute_ball_features(sampled, estimated_frames, confidence_threshold)
