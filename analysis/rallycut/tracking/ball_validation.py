"""
Ball-based segment validation for false positive filtering.

Uses ball visibility and trajectory to validate detected rally segments,
filtering out false positives (dead time incorrectly detected as rallies).
"""

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2

from rallycut.tracking.ball_features import SegmentBallFeatures, compute_ball_features
from rallycut.tracking.ball_tracker import BallPosition

logger = logging.getLogger(__name__)


@dataclass
class BallValidationConfig:
    """Configuration for ball-based segment validation.

    TUNED FOR BEACH VOLLEYBALL (conservative approach):
    - Only reject segments with very low ball detection (<5%)
    - High threshold for rejection to avoid false negatives
    - Ball visibility varies due to outdoor conditions, sun glare, camera angles
    """

    # Validation thresholds - CONSERVATIVE to avoid rejecting valid rallies
    # Analysis showed 0.2 was too aggressive, rejected 4 valid rallies
    min_detection_rate: float = 0.05  # Very low - only reject clear non-rallies
    min_trajectory_variance: float = 0.01  # Lowered - ball may appear stationary in some frames

    # Performance optimizations
    sample_rate: int = 3  # Process every Nth frame (3x speedup)
    early_termination_threshold_high: float = 0.6  # Lowered - accept earlier
    early_termination_threshold_low: float = 0.02  # Very low - only reject obvious cases
    max_frames_to_sample: int = 150  # Cap at 5s @ 30fps
    early_check_interval: int = 30  # Check for early termination every N frames

    # Confidence thresholds for conditional execution
    # Only validate truly uncertain segments to minimize false rejections
    high_confidence_skip: float = 0.7  # Lowered - trust decoder more
    low_confidence_skip: float = 0.3  # Raised - only validate middle range

    # Ball detection confidence
    ball_confidence_threshold: float = 0.4  # Lowered for outdoor conditions


@dataclass
class ValidationResult:
    """Result of ball-based segment validation."""

    is_valid: bool
    confidence_adjustment: float  # Amount to adjust segment confidence
    features: SegmentBallFeatures | None = None
    early_terminated: bool = False  # Whether validation exited early
    frames_processed: int = 0


def should_validate_segment(
    decoder_confidence: float,
    config: BallValidationConfig,
) -> bool:
    """
    Check if a segment should be validated with ball tracking.

    High-confidence segments skip validation (already confident).
    Low-confidence segments skip validation (already rejected).
    Only "uncertain" segments benefit from ball validation.

    Args:
        decoder_confidence: Confidence from the decoder (0-1).
        config: Validation configuration.

    Returns:
        True if segment should be validated.
    """
    # Skip high confidence (already confident, no need to validate)
    if decoder_confidence >= config.high_confidence_skip:
        return False

    # Skip low confidence (already rejected, no need to validate)
    if decoder_confidence < config.low_confidence_skip:
        return False

    return True


def validate_segment_with_ball(
    video_path: Path,
    start_time: float,
    end_time: float,
    ball_tracker: Any,
    config: BallValidationConfig | None = None,
) -> ValidationResult:
    """
    Validate a rally segment using ball tracking.

    Uses subsampling, early termination, and capping for performance.

    Args:
        video_path: Path to video file (preferably proxy).
        start_time: Segment start time in seconds.
        end_time: Segment end time in seconds.
        ball_tracker: Ball tracker instance (will use existing session).
        config: Validation configuration.

    Returns:
        ValidationResult indicating if segment is valid.
    """
    config = config or BallValidationConfig()

    duration = end_time - start_time
    if duration < 1.0:
        # Segment too short for meaningful ball validation
        logger.debug(f"Segment too short ({duration:.1f}s), skipping ball validation")
        return ValidationResult(
            is_valid=True,
            confidence_adjustment=0.0,
            features=None,
            early_terminated=True,
            frames_processed=0,
        )

    # Track ball in segment
    start_ms = int(start_time * 1000)
    end_ms = int(end_time * 1000)

    try:
        result = ball_tracker.track_video(video_path, start_ms=start_ms, end_ms=end_ms)
    except Exception as e:
        logger.warning(f"Ball tracking failed for segment {start_time:.1f}-{end_time:.1f}s: {e}")
        # Fail gracefully - don't reject segment on tracking failure
        return ValidationResult(
            is_valid=True,
            confidence_adjustment=0.0,
            features=None,
            early_terminated=False,
            frames_processed=0,
        )

    # Compute features
    features = compute_ball_features(
        result.positions,
        result.frame_count,
        config.ball_confidence_threshold,
    )

    # Check validity
    is_valid = features.is_valid_rally(
        min_detection_rate=config.min_detection_rate,
        min_trajectory_variance=config.min_trajectory_variance,
    )

    # Calculate confidence adjustment
    # High detection + high variance = boost confidence
    # Low detection = reduce confidence
    if is_valid:
        confidence_adjustment = min(0.1, features.detection_rate * 0.15)
    else:
        confidence_adjustment = -0.2 if features.detection_rate < 0.1 else -0.1

    return ValidationResult(
        is_valid=is_valid,
        confidence_adjustment=confidence_adjustment,
        features=features,
        early_terminated=False,
        frames_processed=result.frame_count,
    )


def validate_segment_with_early_exit(
    video_path: Path,
    start_time: float,
    end_time: float,
    ball_tracker: Any,
    config: BallValidationConfig | None = None,
) -> ValidationResult:
    """
    Validate segment with early exit optimization.

    Periodically checks detection rate and exits early if the result is clear.
    This can save ~50% processing time on average.

    Args:
        video_path: Path to video file (preferably proxy).
        start_time: Segment start time in seconds.
        end_time: Segment end time in seconds.
        ball_tracker: Ball tracker instance.
        config: Validation configuration.

    Returns:
        ValidationResult indicating if segment is valid.
    """
    config = config or BallValidationConfig()

    duration = end_time - start_time
    if duration < 1.0:
        return ValidationResult(
            is_valid=True,
            confidence_adjustment=0.0,
            features=None,
            early_terminated=True,
            frames_processed=0,
        )

    # Open video for streaming frame processing
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Could not open video: {video_path}")
        return ValidationResult(
            is_valid=True,
            confidence_adjustment=0.0,
            features=None,
            early_terminated=False,
            frames_processed=0,
        )

    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)

        # Cap frames to process
        max_frames = min(
            (end_frame - start_frame) // config.sample_rate,
            config.max_frames_to_sample,
        )

        # Seek to start
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        positions: list[BallPosition] = []
        frames_processed = 0
        frame_idx = start_frame

        # Use streaming tracking
        def frame_generator() -> Iterator[Any]:
            nonlocal frame_idx
            while frame_idx < end_frame and len(positions) < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break

                # Only yield every sample_rate frames
                if (frame_idx - start_frame) % config.sample_rate == 0:
                    yield frame

                frame_idx += 1

        # Process frames with early termination checks
        for pos in ball_tracker.track_frames(
            frame_generator(),
            video_fps=fps,
            video_width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            video_height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        ):
            positions.append(pos)
            frames_processed += 1

            # Early termination check
            if len(positions) >= config.early_check_interval:
                if len(positions) % config.early_check_interval == 0:
                    # Check current detection rate
                    confident_count = sum(
                        1 for p in positions if p.confidence >= config.ball_confidence_threshold
                    )
                    current_rate = confident_count / len(positions)

                    # Clearly valid
                    if current_rate >= config.early_termination_threshold_high:
                        logger.debug(
                            f"Early exit (valid): detection_rate={current_rate:.2f} "
                            f"after {len(positions)} frames"
                        )
                        return ValidationResult(
                            is_valid=True,
                            confidence_adjustment=0.1,
                            features=None,  # Skip full feature computation
                            early_terminated=True,
                            frames_processed=frames_processed,
                        )

                    # Clearly invalid
                    if current_rate <= config.early_termination_threshold_low:
                        logger.debug(
                            f"Early exit (invalid): detection_rate={current_rate:.2f} "
                            f"after {len(positions)} frames"
                        )
                        return ValidationResult(
                            is_valid=False,
                            confidence_adjustment=-0.2,
                            features=None,
                            early_terminated=True,
                            frames_processed=frames_processed,
                        )

        # Full evaluation for uncertain cases
        features = compute_ball_features(
            positions,
            len(positions),
            config.ball_confidence_threshold,
        )

        is_valid = features.is_valid_rally(
            min_detection_rate=config.min_detection_rate,
            min_trajectory_variance=config.min_trajectory_variance,
        )

        confidence_adjustment = 0.1 if is_valid else -0.1

        return ValidationResult(
            is_valid=is_valid,
            confidence_adjustment=confidence_adjustment,
            features=features,
            early_terminated=False,
            frames_processed=frames_processed,
        )

    finally:
        cap.release()


@dataclass
class SegmentValidationBatch:
    """Batch validation of multiple segments."""

    segments: list[tuple[float, float]]  # (start_time, end_time) pairs
    confidences: list[float]  # Decoder confidences for each segment
    results: list[ValidationResult] = field(default_factory=list)


def validate_segments_batch(
    video_path: Path,
    segments: list[tuple[float, float]],
    confidences: list[float],
    ball_tracker: Any,
    config: BallValidationConfig | None = None,
    use_early_exit: bool = True,
) -> list[ValidationResult]:
    """
    Validate multiple segments in batch.

    Applies conditional execution to skip segments that don't need validation.

    Args:
        video_path: Path to video file.
        segments: List of (start_time, end_time) tuples.
        confidences: Decoder confidence for each segment.
        ball_tracker: Ball tracker instance.
        config: Validation configuration.
        use_early_exit: Use early termination optimization.

    Returns:
        List of ValidationResult for each segment.
    """
    config = config or BallValidationConfig()
    results: list[ValidationResult] = []

    for (start, end), conf in zip(segments, confidences):
        # Conditional execution: skip if confidence is clear
        if not should_validate_segment(conf, config):
            # High confidence: keep segment
            if conf >= config.high_confidence_skip:
                results.append(ValidationResult(
                    is_valid=True,
                    confidence_adjustment=0.0,
                    features=None,
                    early_terminated=True,
                    frames_processed=0,
                ))
            # Low confidence: reject segment
            else:
                results.append(ValidationResult(
                    is_valid=False,
                    confidence_adjustment=0.0,
                    features=None,
                    early_terminated=True,
                    frames_processed=0,
                ))
            continue

        # Validate with ball tracking
        if use_early_exit:
            result = validate_segment_with_early_exit(
                video_path, start, end, ball_tracker, config
            )
        else:
            result = validate_segment_with_ball(
                video_path, start, end, ball_tracker, config
            )

        results.append(result)

    return results
