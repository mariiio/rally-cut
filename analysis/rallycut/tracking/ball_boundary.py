"""
Ball-based boundary refinement for rally segments.

Uses ball visibility and trajectory to refine rally start/end boundaries
detected by the decoder.
"""

import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rallycut.tracking.ball_tracker import BallPosition, BallTrackingResult

logger = logging.getLogger(__name__)


@dataclass
class BallBoundaryConfig:
    """Configuration for ball-based boundary refinement.

    TUNED FOR BEACH VOLLEYBALL (conservative approach):
    - Smaller search window to limit maximum adjustment
    - Only adjust when we have clear ball activity signals
    - Prefer keeping original boundary over uncertain adjustments

    START BOUNDARY STRATEGY (velocity-based):
    - Look for serve velocity spike (distinctive ball acceleration)
    - Fall back to sustained ball activity or first detection
    - Works well: ~80% confidence on velocity spike detection

    END BOUNDARY STRATEGY (detection-rate based):
    - ~40% of rallies have high ball detection AFTER rally ends (ball on ground,
      player holding it, next serve starting)
    - ~70% have a usable signal: detection drops below 60% and stays low
    - Only refine when detection was good before (>50%) then dropped
    - When signal is ambiguous, keep original decoder boundary
    """

    # Search window around coarse boundary
    # ±1.5s balances finding the true boundary vs avoiding large adjustments
    search_window_seconds: float = 1.5

    # Ball detection confidence threshold (tuned for outdoor beach conditions)
    min_confidence: float = 0.35

    # START BOUNDARY: Velocity-based detection
    # Serve creates distinctive velocity spike when ball is hit
    serve_velocity_spike: float = 0.04  # Normalized velocity threshold for serve
    velocity_threshold: float = 0.015  # Threshold for sustained activity detection

    # END BOUNDARY: Detection-rate based detection
    # Rally end = ball disappears (detection rate drops and stays low)
    # Tuned via grid search: more aggressive params (higher coverage) give better overall MAE
    end_sustained_low_threshold: float = 0.60  # Detection must drop below 60%
    end_sustained_low_duration: float = 0.25  # For at least 0.25 seconds (2 buckets)
    end_min_detection_before: float = 0.50  # Must have had >50% detection before

    # Confidence thresholds for applying refinements
    min_refinement_confidence: float = 0.5  # Don't adjust if confidence < this

    # Which boundaries to refine (both enabled by default)
    refine_start: bool = True
    refine_end: bool = True

    # Performance: thread pool size for parallel boundary processing
    max_workers: int = 4


@dataclass
class BoundaryRefinement:
    """Result of boundary refinement."""

    original_time: float  # Original boundary time
    refined_time: float  # Refined boundary time
    adjustment: float  # Difference (refined - original)
    confidence: float  # Confidence in refinement (0-1)
    reason: str  # Description of why boundary was adjusted


@dataclass
class SegmentRefinement:
    """Refined segment boundaries."""

    original_start: float
    original_end: float
    refined_start: float
    refined_end: float
    start_refinement: BoundaryRefinement | None
    end_refinement: BoundaryRefinement | None


def _find_activity_start(
    positions: list[BallPosition],
    fps: float,
    config: BallBoundaryConfig,
) -> tuple[float | None, str]:
    """
    Find the start of ball activity (serve or first active play).

    Looks for:
    1. First frame where ball is confidently detected
    2. Velocity spike indicating serve
    3. Ball position transition (toss detection)

    Args:
        positions: Ball positions in the boundary window.
        fps: Video frame rate.
        config: Boundary configuration.

    Returns:
        Tuple of (time_offset, reason) or (None, "") if not found.
    """
    if not positions:
        return None, ""

    confident = [p for p in positions if p.confidence >= config.min_confidence]
    if len(confident) < 3:
        return None, ""

    # Compute velocities
    velocities = []
    for i in range(1, len(confident)):
        dx = confident[i].x - confident[i - 1].x
        dy = confident[i].y - confident[i - 1].y
        frame_gap = max(1, confident[i].frame_number - confident[i - 1].frame_number)
        v = np.sqrt(dx**2 + dy**2) / frame_gap
        velocities.append((confident[i].frame_number, v, confident[i].y))

    if not velocities:
        return None, ""

    # Look for velocity spike (serve)
    for frame_num, vel, y in velocities:
        if vel >= config.serve_velocity_spike:
            time_offset = frame_num / fps
            return time_offset, "velocity_spike"

    # Look for first sustained activity (multiple frames above threshold)
    activity_start = None
    consecutive_active = 0
    for frame_num, vel, y in velocities:
        if vel >= config.velocity_threshold:
            consecutive_active += 1
            if consecutive_active >= 3 and activity_start is None:
                activity_start = frame_num
        else:
            consecutive_active = 0

    if activity_start is not None:
        time_offset = activity_start / fps
        return time_offset, "sustained_activity"

    # Fall back to first confident detection
    first_frame = confident[0].frame_number
    time_offset = first_frame / fps
    return time_offset, "first_detection"


def _compute_detection_rate_buckets(
    positions: list[BallPosition],
    fps: float,
    bucket_size: float,
    conf_threshold: float,
) -> dict[float, float]:
    """
    Compute detection rate in time buckets.

    Returns dict mapping bucket_time -> detection_rate (0-1).
    """
    buckets: dict[float, list[bool]] = defaultdict(list)
    for p in positions:
        abs_time = p.frame_number / fps
        bucket = round(abs_time / bucket_size) * bucket_size
        buckets[bucket].append(p.confidence >= conf_threshold)

    return {b: sum(v) / len(v) if v else 0.0 for b, v in buckets.items()}


def _find_activity_end(
    positions: list[BallPosition],
    fps: float,
    config: BallBoundaryConfig,
) -> tuple[float | None, str]:
    """
    Find the end of ball activity using DETECTION-RATE based approach.

    Strategy (based on ground truth analysis):
    - ~40% of rallies have high ball detection AFTER rally ends (ball on ground,
      player holding it, next serve starting)
    - Only ~30% have a CLEAR signal: detection drops significantly and stays low
    - Only refine when we have high confidence in the signal

    Looks for:
    1. Sustained low detection period (detection < 50% for 0.5s+)
    2. Only triggers if signal is clear (avoid false positives)

    Args:
        positions: Ball positions in the boundary window.
        fps: Video frame rate.
        config: Boundary configuration.

    Returns:
        Tuple of (time_offset, reason) or (None, "") if not found.
        Returns (None, "") when signal is ambiguous (keeps original boundary).
    """
    if not positions:
        return None, ""

    # Compute detection rates in 0.25s buckets
    bucket_size = 0.25
    rates = _compute_detection_rate_buckets(
        positions, fps, bucket_size, config.min_confidence
    )

    if not rates:
        return None, ""

    sorted_times = sorted(rates.keys())

    # Get thresholds from config
    sustained_low_thresh = config.end_sustained_low_threshold
    sustained_duration = config.end_sustained_low_duration
    min_det_before = config.end_min_detection_before
    required_buckets = max(2, int(sustained_duration / bucket_size))

    # Find first sustained low detection period
    # (where detection rate stays below threshold for required duration)
    for i in range(len(sorted_times) - required_buckets + 1):
        window = sorted_times[i : i + required_buckets]
        if all(rates[t] < sustained_low_thresh for t in window):
            # Found sustained low period
            low_start_time = window[0]

            # Verify this is a CLEAR signal (not just noise)
            # Check detection rate before this point
            before_times = [t for t in sorted_times if t < low_start_time - 0.25]
            det_before = np.mean([rates[t] for t in before_times[-3:]]) if before_times else 1.0

            # Only return if this is a clear transition (detection was good before)
            if det_before >= min_det_before:
                return low_start_time, "sustained_low_detection"

    # Look for detection gap (ball disappeared for >0.5s)
    for i in range(len(positions) - 1):
        frame_gap = positions[i + 1].frame_number - positions[i].frame_number
        if frame_gap > fps * 0.5:  # >0.5s gap
            time_offset = positions[i].frame_number / fps
            return time_offset, "detection_gap"

    # No clear signal - return None to keep original boundary
    # This is INTENTIONAL: better to keep decoder boundary than make a bad adjustment
    return None, ""


def find_rally_start_from_ball(
    tracking_result: BallTrackingResult,
    coarse_start: float,
    config: BallBoundaryConfig,
) -> BoundaryRefinement:
    """
    Refine rally start boundary using ball tracking.

    Args:
        tracking_result: Ball tracking result for boundary window.
        coarse_start: Original coarse start time (seconds).
        config: Boundary configuration.

    Returns:
        BoundaryRefinement with refined start time.
    """
    time_offset, reason = _find_activity_start(
        tracking_result.positions,
        tracking_result.video_fps,
        config,
    )

    if time_offset is None:
        return BoundaryRefinement(
            original_time=coarse_start,
            refined_time=coarse_start,
            adjustment=0.0,
            confidence=0.0,
            reason="no_refinement",
        )

    # time_offset is actually ABSOLUTE time in the video (frame_number / fps)
    # since BallPosition.frame_number is absolute, not relative to tracking window
    refined_start = time_offset

    # Clamp to reasonable bounds (within search window of original)
    min_bound = max(0.0, coarse_start - config.search_window_seconds)
    max_bound = coarse_start + config.search_window_seconds
    refined_start = max(min_bound, min(max_bound, refined_start))

    adjustment = refined_start - coarse_start

    # Confidence based on how clear the activity start was
    confidence = 0.8 if reason == "velocity_spike" else 0.5 if reason == "sustained_activity" else 0.3

    # CONSERVATIVE: Only apply adjustment if confidence is high enough
    # Otherwise keep the original boundary (decoder is usually reasonable)
    if confidence < config.min_refinement_confidence:
        return BoundaryRefinement(
            original_time=coarse_start,
            refined_time=coarse_start,  # Keep original
            adjustment=0.0,
            confidence=confidence,
            reason=f"{reason}_low_confidence",
        )

    return BoundaryRefinement(
        original_time=coarse_start,
        refined_time=refined_start,
        adjustment=adjustment,
        confidence=confidence,
        reason=reason,
    )


def find_rally_end_from_ball(
    tracking_result: BallTrackingResult,
    coarse_end: float,
    config: BallBoundaryConfig,
) -> BoundaryRefinement:
    """
    Refine rally end boundary using ball tracking.

    Args:
        tracking_result: Ball tracking result for boundary window.
        coarse_end: Original coarse end time (seconds).
        config: Boundary configuration.

    Returns:
        BoundaryRefinement with refined end time.
    """
    time_offset, reason = _find_activity_end(
        tracking_result.positions,
        tracking_result.video_fps,
        config,
    )

    if time_offset is None:
        return BoundaryRefinement(
            original_time=coarse_end,
            refined_time=coarse_end,
            adjustment=0.0,
            confidence=0.0,
            reason="no_refinement",
        )

    # time_offset is actually ABSOLUTE time in the video (frame_number / fps)
    # since BallPosition.frame_number is absolute, not relative to tracking window
    refined_end = time_offset

    # Clamp to reasonable bounds (within search window of original)
    min_bound = coarse_end - config.search_window_seconds
    max_bound = coarse_end + config.search_window_seconds
    refined_end = max(min_bound, min(max_bound, refined_end))

    adjustment = refined_end - coarse_end

    # Confidence based on how clear the activity end was
    # "sustained_low_detection" = clear signal (detection dropped and stayed low)
    # "detection_gap" = ball disappeared (also a clear signal)
    confidence = (
        0.8 if reason == "sustained_low_detection"
        else 0.7 if reason == "detection_gap"
        else 0.2  # Unknown reason - low confidence
    )

    # CONSERVATIVE: Only apply adjustment if confidence is high enough
    if confidence < config.min_refinement_confidence:
        return BoundaryRefinement(
            original_time=coarse_end,
            refined_time=coarse_end,  # Keep original
            adjustment=0.0,
            confidence=confidence,
            reason=f"{reason}_low_confidence",
        )

    return BoundaryRefinement(
        original_time=coarse_end,
        refined_time=refined_end,
        adjustment=adjustment,
        confidence=confidence,
        reason=reason,
    )


def _track_boundary_window(
    video_path: Path,
    boundary_time: float,
    search_window: float,
    ball_tracker: Any,
) -> BallTrackingResult:
    """
    Track ball in boundary window.

    Args:
        video_path: Path to video.
        boundary_time: Center of search window.
        search_window: Window size in seconds.
        ball_tracker: Ball tracker instance (must have track_video method).

    Returns:
        BallTrackingResult for the window.
    """
    start_ms = max(0, int((boundary_time - search_window) * 1000))
    end_ms = int((boundary_time + search_window) * 1000)

    result: BallTrackingResult = ball_tracker.track_video(video_path, start_ms=start_ms, end_ms=end_ms)
    return result


def refine_segment_boundaries(
    video_path: Path,
    start_time: float,
    end_time: float,
    ball_tracker: Any,
    config: BallBoundaryConfig | None = None,
) -> SegmentRefinement:
    """
    Refine both boundaries of a segment using ball tracking.

    Args:
        video_path: Path to video.
        start_time: Original start time.
        end_time: Original end time.
        ball_tracker: Ball tracker instance (must have track_video method).
        config: Boundary configuration.

    Returns:
        SegmentRefinement with refined boundaries.
    """
    config = config or BallBoundaryConfig()

    # Track start boundary window (only if enabled)
    if config.refine_start:
        try:
            start_result = _track_boundary_window(
                video_path,
                start_time,
                config.search_window_seconds,
                ball_tracker,
            )
            start_refinement = find_rally_start_from_ball(start_result, start_time, config)
        except Exception as e:
            logger.warning(f"Start boundary tracking failed: {e}")
            start_refinement = BoundaryRefinement(
                original_time=start_time,
                refined_time=start_time,
                adjustment=0.0,
                confidence=0.0,
                reason=f"error: {e}",
            )
    else:
        start_refinement = BoundaryRefinement(
            original_time=start_time,
            refined_time=start_time,
            adjustment=0.0,
            confidence=0.0,
            reason="disabled",
        )

    # Track end boundary window (only if enabled)
    if config.refine_end:
        try:
            end_result = _track_boundary_window(
                video_path,
                end_time,
                config.search_window_seconds,
                ball_tracker,
            )
            end_refinement = find_rally_end_from_ball(end_result, end_time, config)
        except Exception as e:
            logger.warning(f"End boundary tracking failed: {e}")
            end_refinement = BoundaryRefinement(
                original_time=end_time,
                refined_time=end_time,
                adjustment=0.0,
                confidence=0.0,
                reason=f"error: {e}",
            )
    else:
        end_refinement = BoundaryRefinement(
            original_time=end_time,
            refined_time=end_time,
            adjustment=0.0,
            confidence=0.0,
            reason="disabled",
        )

    return SegmentRefinement(
        original_start=start_time,
        original_end=end_time,
        refined_start=start_refinement.refined_time,
        refined_end=end_refinement.refined_time,
        start_refinement=start_refinement,
        end_refinement=end_refinement,
    )


def refine_boundaries_parallel(
    video_path: Path,
    segments: list[tuple[float, float]],
    ball_tracker: Any,
    config: BallBoundaryConfig | None = None,
) -> list[SegmentRefinement]:
    """
    Refine multiple segment boundaries in parallel.

    Uses thread pool for parallel I/O to maximize throughput.

    Args:
        video_path: Path to video.
        segments: List of (start_time, end_time) tuples.
        ball_tracker: Ball tracker instance (must have track_video method).
        config: Boundary configuration.

    Returns:
        List of SegmentRefinement for each segment.
    """
    config = config or BallBoundaryConfig()

    if not segments:
        return []

    # For small batches, sequential is fine
    if len(segments) <= 2:
        return [
            refine_segment_boundaries(video_path, start, end, ball_tracker, config)
            for start, end in segments
        ]

    # Parallel processing for larger batches
    results: list[SegmentRefinement] = [None] * len(segments)  # type: ignore

    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        future_to_idx = {
            executor.submit(
                refine_segment_boundaries,
                video_path,
                start,
                end,
                ball_tracker,
                config,
            ): i
            for i, (start, end) in enumerate(segments)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                results[idx] = future.result()
            except Exception as e:
                logger.warning(f"Boundary refinement failed for segment {idx}: {e}")
                # Fall back to original boundaries
                start, end = segments[idx]
                results[idx] = SegmentRefinement(
                    original_start=start,
                    original_end=end,
                    refined_start=start,
                    refined_end=end,
                    start_refinement=None,
                    end_refinement=None,
                )

    return results
