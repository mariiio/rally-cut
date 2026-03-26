"""
Player tracking using YOLO + BoT-SORT for volleyball videos.

Uses YOLO for person detection with BoT-SORT for temporal tracking.
BoT-SORT adds camera motion compensation on top of ByteTrack, reducing ID switches.
Default model is yolo11s (small) at 1280px for best accuracy/speed tradeoff
(92.5% F1, 91.3% HOTA, 6.1 FPS, 96.3% far-court recall).
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from rallycut.court.calibration import CourtCalibrator
    from rallycut.court.detector import CourtDetectionInsights
    from rallycut.tracking.appearance_descriptor import AppearanceDescriptorStore
    from rallycut.tracking.ball_tracker import BallPosition
    from rallycut.tracking.color_repair import ColorHistogramStore
    from rallycut.tracking.player_filter import PlayerFilterConfig
    from rallycut.tracking.quality_report import TrackingQualityReport

logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "yolo11s.pt"  # YOLO11 small - best accuracy/speed tradeoff (92.5% F1, 6.1 FPS)
PERSON_CLASS_ID = 0  # COCO class ID for person
DEFAULT_CONFIDENCE = 0.15  # Lower threshold for detection (tuned via grid search)
DEFAULT_IOU = 0.45  # NMS IoU threshold
DEFAULT_IMGSZ = 1280  # Inference resolution (1280 = +8pp far-court recall vs 640, 2x slower)

# Default court ROI polygon (conservative rectangle for behind-baseline cameras)
# Excludes edges where wall drawings, spectators, equipment may appear
# Polygon as list of (x, y) normalized 0-1 coordinates, clockwise
DEFAULT_COURT_ROI: list[tuple[float, float]] = [
    (0.05, 0.10),  # top-left
    (0.95, 0.10),  # top-right
    (0.95, 0.95),  # bottom-right
    (0.05, 0.95),  # bottom-left
]

# Maximum detections per frame passed to post-tracking pipeline
# More than 4 (court players) to maintain tracker continuity, but caps noise.
# Set to 8 (was 6) to avoid discarding real far-court players in crowded scenes
# where spectators within the ROI have higher confidence.
MAX_DETECTIONS_PER_FRAME = 8

# Adaptive ROI parameters
_ADAPTIVE_ROI_MIN_POINTS = 20  # Minimum confident ball detections for adaptive ROI
_ADAPTIVE_ROI_MIN_CONFIDENCE = 0.30  # Minimum ball confidence to include
_ADAPTIVE_ROI_MIN_AREA = 0.04  # Reject ROI covering less than 4% of frame (bad tracking)


def compute_court_roi_from_calibration(
    calibrator: CourtCalibrator,
    x_margin: float = 0.05,
    near_margin: float = 0.10,
    far_margin: float = 0.50,
) -> tuple[list[tuple[float, float]] | None, str]:
    """Compute court ROI rectangle from calibration homography.

    Projects the 4 court corners to image space to determine the court's
    vertical extent, then builds a rectangle with image-space margins.

    Uses a rectangle (not trapezoid) because:
    - Perspective compresses the far side, making court-space margins
      useless for Y expansion (4m moves 0.02 in image Y)
    - The trapezoid's slanted edges clip far-court players who are
      well outside the narrow far-court width in image space
    - Near-side corners already extend beyond frame edges for most
      beach volleyball camera angles

    The calibration's value is knowing the court's Y range in the image
    (vs ball trajectory which is an unreliable proxy).

    Args:
        calibrator: Calibrated CourtCalibrator with homography.
        x_margin: Margin on left/right as fraction of frame (default 5%).
        near_margin: Margin below near baseline as fraction of frame (default 10%).
        far_margin: Margin above far baseline as fraction of frame (default 50%).
            Large because low-angle cameras show far-court players well above
            the court line in the image.

    Returns:
        Tuple of (roi_polygon, quality_message):
        - roi_polygon: List of 4 (x, y) points (clockwise rectangle), or None
          if calibrator is not calibrated or homography is degenerate.
        - quality_message: Human-readable quality assessment string.
    """
    if not calibrator.is_calibrated:
        return None, "Calibrator not calibrated"

    from rallycut.court.calibration import COURT_LENGTH, COURT_WIDTH

    # Project 4 court corners (no margins) to image space.
    # Camera is behind near baseline, so court (0,0) = near-left (high Y
    # in image, bottom of frame), court (8,16) = far-right (low Y, top).
    court_corners = [
        (0.0, 0.0),                        # near-left
        (COURT_WIDTH, 0.0),                 # near-right
        (COURT_WIDTH, COURT_LENGTH),        # far-right
        (0.0, COURT_LENGTH),                # far-left
    ]

    image_pts: list[tuple[float, float]] = []
    for cx, cy in court_corners:
        try:
            ix, iy = calibrator.court_to_image((cx, cy), 1, 1)
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            return None, "Failed to project court corners (degenerate homography)"
        image_pts.append((ix, iy))

    # Identify near side (high Y) and far side (low Y)
    near_avg_y = (image_pts[0][1] + image_pts[1][1]) / 2
    far_avg_y = (image_pts[2][1] + image_pts[3][1]) / 2

    # Sanity check: near side should have higher Y than far side in image
    if near_avg_y < far_avg_y:
        return None, (
            f"Court orientation unexpected: near baseline Y={near_avg_y:.2f} "
            f"< far baseline Y={far_avg_y:.2f}. "
            "Check calibration corner order."
        )

    # Build bounding rectangle with image-space margins.
    # far_margin is intentionally large (50%) because low-angle cameras
    # show far-court players well above the court line in the image.
    # For typical calibrations (far_avg_y ≈ 0.55), this gives roi_y_min ≈ 0.05,
    # covering nearly the full frame height. The ROI's main filtering value
    # is on the X axis and below-court (near side).
    all_x = [p[0] for p in image_pts]
    roi_x_min = max(0.0, min(all_x) - x_margin)
    roi_x_max = min(1.0, max(all_x) + x_margin)
    roi_y_min = max(0.0, far_avg_y - far_margin)
    roi_y_max = min(1.0, near_avg_y + near_margin)

    # Enforce minimum Y extent — near-side players' feet can be at y=0.90+,
    # so the ROI must extend far enough down. Without this, inaccurate
    # near-corner extrapolation clips valid players.
    roi_y_max = max(roi_y_max, 0.95)

    # Reject ROIs that cover too much of the frame — they provide no
    # filtering value and indicate a bad calibration (e.g. off-screen
    # near corners producing near-full-frame ROI).
    roi_area = (roi_x_max - roi_x_min) * (roi_y_max - roi_y_min)
    if roi_area > 0.90:
        return None, (
            f"Calibration ROI covers {roi_area:.0%} of frame "
            "(near corners may be off-screen)"
        )

    roi_points: list[tuple[float, float]] = [
        (roi_x_min, roi_y_min),  # top-left
        (roi_x_max, roi_y_min),  # top-right
        (roi_x_max, roi_y_max),  # bottom-right
        (roi_x_min, roi_y_max),  # bottom-left
    ]

    quality_msg = ""
    court_height = near_avg_y - far_avg_y
    if court_height < 0.15:
        quality_msg = (
            f"Court covers only {court_height * 100:.0f}% of frame height. "
            "Far-court players may be too small for reliable detection."
        )

    return roi_points, quality_msg


def auto_detect_court(
    video_path: Path | str,
    confidence_threshold: float = 0.4,
    keypoint_model_path: str | Path | None = None,
) -> tuple[CourtCalibrator | None, Any]:
    """Auto-detect court corners from video and create calibrator if confident enough.

    Safe to call on any video — returns (None, result) on failure without raising.

    Args:
        video_path: Path to the video file.
        confidence_threshold: Minimum confidence to accept detection.
        keypoint_model_path: Optional path to YOLO-pose keypoint model weights.

    Returns:
        Tuple of (calibrator, result):
        - calibrator: CourtCalibrator if detection succeeded, None otherwise.
        - result: CourtDetectionResult with detection details (or error info).
    """
    from rallycut.court.calibration import CourtCalibrator
    from rallycut.court.detector import CourtDetectionResult, CourtDetector

    try:
        detector = CourtDetector(keypoint_model_path=keypoint_model_path)
        result = detector.detect(video_path)
    except (FileNotFoundError, RuntimeError) as e:
        logger.warning(f"Court auto-detection failed: {e}")
        return None, CourtDetectionResult(
            corners=[], confidence=0.0, warnings=[str(e)],
        )

    if result.confidence >= confidence_threshold and len(result.corners) == 4:
        calibrator = CourtCalibrator()
        corners = [(c["x"], c["y"]) for c in result.corners]
        calibrator.calibrate(corners)

        # Validate the homography isn't degenerate by checking that near side
        # projects below far side in image space (higher Y = lower in frame).
        # A degenerate homography (e.g., from elevated camera VP issues) would
        # silently cause off-court filtering to reject all player detections.
        try:
            from rallycut.court.calibration import COURT_LENGTH, COURT_WIDTH

            near_pt = calibrator.court_to_image((COURT_WIDTH / 2, 0), 1, 1)
            far_pt = calibrator.court_to_image((COURT_WIDTH / 2, COURT_LENGTH), 1, 1)
            if near_pt[1] < far_pt[1]:
                logger.warning(
                    f"Court homography is inverted (near Y={near_pt[1]:.1f} "
                    f"< far Y={far_pt[1]:.1f}), discarding calibration"
                )
                return None, result
        except (RuntimeError, ValueError, np.linalg.LinAlgError):
            logger.warning("Court homography validation failed, discarding calibration")
            return None, result

        return calibrator, result

    return None, result


def refine_court_with_players(
    initial_result: Any,
    player_positions: list[PlayerPosition],
    team_assignments: dict[int, int],
    confidence_threshold: float = 0.4,
) -> tuple[CourtCalibrator | None, Any]:
    """Refine court detection using tracked player positions.

    Called after first-pass player tracking to improve court calibration,
    especially near corners which are often off-screen and poorly estimated
    by line detection alone.

    Args:
        initial_result: CourtDetectionResult from line detection.
        player_positions: Player tracking results.
        team_assignments: Map of track_id → team (0=near, 1=far).
        confidence_threshold: Minimum confidence for line detection to refine.

    Returns:
        Tuple of (calibrator, result). Returns improved calibration, or the
        original if refinement didn't help.
    """
    from rallycut.court.calibration import CourtCalibrator
    from rallycut.court.detector import CourtDetectionResult
    from rallycut.court.player_constrained import (
        PlayerConstrainedOptimizer,
        extract_player_feet,
    )

    initial_corners = getattr(initial_result, "corners", [])
    initial_confidence = getattr(initial_result, "confidence", 0.0)

    # Extract player foot positions with team assignments
    feet = extract_player_feet(player_positions, team_assignments)
    if len(feet) < 20:
        logger.info(
            f"Court refinement: insufficient player feet ({len(feet)}), skipping"
        )
        if initial_corners and initial_confidence >= confidence_threshold:
            calibrator = CourtCalibrator()
            calibrator.calibrate([(c["x"], c["y"]) for c in initial_corners])
            return calibrator, initial_result
        return None, initial_result

    optimizer = PlayerConstrainedOptimizer()

    # Case 1: Line detection succeeded
    if len(initial_corners) == 4 and initial_confidence >= confidence_threshold:
        fitting_method = getattr(initial_result, "fitting_method", "legacy")
        unreliable_near = fitting_method in ("aspect_ratio", "harmonic_conjugate")

        # Case 1b: Near corners were extrapolated — refine them using players
        # while keeping the reliable far corners fixed. This is safe because
        # far corners from line detection are ~2.5x more accurate.
        # The 0.65 threshold is separate from confidence_threshold (0.4) —
        # it targets the confidence range where near corners are unreliable
        # but far corners and line detection are still useful.
        if unreliable_near and initial_confidence < 0.65:
            refined = optimizer.refine_corners(
                initial_corners, feet, fix_far_corners=True,
            )
            if refined is not None:
                calibrator = CourtCalibrator()
                calibrator.calibrate([(c["x"], c["y"]) for c in refined])
                result = CourtDetectionResult(
                    corners=refined,
                    # Small boost since player refinement improves near corners;
                    # stays below 0.7 (auto-save threshold) to avoid persisting
                    # player-refined results as if they were high-confidence detections
                    confidence=min(initial_confidence + 0.05, 0.69),
                    detected_lines=getattr(initial_result, "detected_lines", []),
                    warnings=["Near corners refined using player positions"],
                    fitting_method="player_refined",
                )
                logger.info(
                    f"Court refinement: near corners refined from {fitting_method} "
                    f"using player positions"
                )
                return calibrator, result

        # Case 1a: Good detection or refinement didn't help — use as-is
        calibrator = CourtCalibrator()
        calibrator.calibrate([(c["x"], c["y"]) for c in initial_corners])
        return calibrator, initial_result

    # Case 2: Line detection failed — estimate from players
    # Get net_y from player positions
    near_feet_y = [f.y for f in feet if f.team == 0]
    far_feet_y = [f.y for f in feet if f.team == 1]
    if near_feet_y and far_feet_y:
        net_y = (min(near_feet_y) + max(far_feet_y)) / 2.0
    else:
        logger.info("Court refinement: cannot estimate net_y, skipping")
        return None, initial_result

    estimated = optimizer.estimate_from_players(feet, net_y)
    if estimated is not None:
        calibrator = CourtCalibrator()
        calibrator.calibrate([(c["x"], c["y"]) for c in estimated])

        result = CourtDetectionResult(
            corners=estimated,
            confidence=0.35,  # lower confidence for player-only estimate
            detected_lines=[],
            warnings=["Court estimated from player positions (no line detection)"],
            fitting_method="player_only",
        )
        logger.info("Court refinement: estimated from player positions only")
        return calibrator, result

    return None, initial_result


def compute_court_roi_from_ball(
    ball_positions: list[BallPosition],
    x_margin: float = 0.08,
    y_margin_top: float = 0.05,
    y_margin_bottom: float = 0.20,
    percentile_low: float = 3.0,
    percentile_high: float = 97.0,
    max_roi_area: float = 0.85,
    min_roi_width: float = 0.80,
    min_roi_height: float = 0.85,
) -> tuple[list[tuple[float, float]] | None, str]:
    """Compute a tight court ROI polygon from ball trajectory positions.

    The ball stays on or near the court during play, so its trajectory
    naturally defines the playing area. Uses percentile-based bounds
    (not min/max) to be robust to outlier ball detections, then expands
    asymmetrically to cover player positions:
    - Horizontally: players stand at/outside sidelines, ball often central
    - Top (far court): far-side players slightly above ball trajectory
    - Bottom (near court): near-side players well below ball trajectory

    A minimum ROI width/height is enforced to prevent clipping the court
    when the ball trajectory covers only part of the playing area (e.g.,
    one-sided rallies where the ball stays on the left half).

    Args:
        ball_positions: Ball tracking results (normalized 0-1 coordinates).
        x_margin: Absolute margin to add on left/right (default 8%).
        y_margin_top: Absolute margin to add above (default 5%).
        y_margin_bottom: Absolute margin to add below (default 20%).
        percentile_low: Lower percentile for bounds (default 3rd).
        percentile_high: Upper percentile for bounds (default 97th).
        max_roi_area: Maximum ROI area as fraction of frame. If exceeded,
            returns None (ball trajectory too spread for useful ROI).
        min_roi_width: Minimum ROI width as fraction of frame (default 80%).
            Prevents one-sided ball trajectories from clipping the court.
        min_roi_height: Minimum ROI height as fraction of frame (default 65%).
            Ensures far- and near-court players are included.

    Returns:
        Tuple of (roi_polygon, quality_message):
        - roi_polygon: List of 4 (x, y) points (clockwise rectangle), or None
          if insufficient ball data or ROI too large/small.
        - quality_message: Human-readable quality assessment string.
          Empty string if ROI is good. Contains warning if quality is marginal.
    """
    # Filter to confident detections with actual positions
    points = [
        (bp.x, bp.y)
        for bp in ball_positions
        if bp.confidence >= _ADAPTIVE_ROI_MIN_CONFIDENCE
        and not (bp.x == 0.0 and bp.y == 0.0)
    ]

    if len(points) < _ADAPTIVE_ROI_MIN_POINTS:
        return None, (
            f"Only {len(points)} confident ball detections "
            f"(need {_ADAPTIVE_ROI_MIN_POINTS}). "
            "Use --court-roi to specify the court area manually."
        )

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]

    # Use percentile-based bounds instead of min/max to exclude outlier
    # ball detections (false positives at frame edges, etc.)
    xs_arr = np.array(xs)
    ys_arr = np.array(ys)
    ball_x_min = float(np.percentile(xs_arr, percentile_low))
    ball_x_max = float(np.percentile(xs_arr, percentile_high))
    ball_y_min = float(np.percentile(ys_arr, percentile_low))
    ball_y_max = float(np.percentile(ys_arr, percentile_high))

    # Expand bounding box to cover player positions
    roi_x_min = max(0.0, ball_x_min - x_margin)
    roi_x_max = min(1.0, ball_x_max + x_margin)
    roi_y_min = max(0.0, ball_y_min - y_margin_top)
    roi_y_max = min(1.0, ball_y_max + y_margin_bottom)

    # Quality check: raw ball trajectory too small BEFORE min-dimension expansion.
    # If the ball is stuck in one spot, the trajectory is useless even if we
    # expand the ROI to minimum dimensions (it would be centered on the wrong spot).
    ball_coverage = (ball_x_max - ball_x_min) * (ball_y_max - ball_y_min)
    if ball_coverage < _ADAPTIVE_ROI_MIN_AREA:
        return None, (
            f"Ball trajectory covers only {ball_coverage * 100:.1f}% of frame "
            "(likely poor ball tracking). "
            "Use --court-roi to specify the court area manually."
        )

    # Enforce minimum ROI dimensions by expanding symmetrically around
    # the ball trajectory center. This prevents one-sided ball trajectories
    # from clipping players on the other side of the court.
    roi_width = roi_x_max - roi_x_min
    if roi_width < min_roi_width:
        center_x = (roi_x_min + roi_x_max) / 2
        half_w = min_roi_width / 2
        roi_x_min = max(0.0, center_x - half_w)
        roi_x_max = min(1.0, roi_x_min + min_roi_width)
        # Re-adjust if clamped at right edge
        if roi_x_max - roi_x_min < min_roi_width:
            roi_x_min = max(0.0, roi_x_max - min_roi_width)

    roi_height = roi_y_max - roi_y_min
    if roi_height < min_roi_height:
        # Asymmetric expansion: bias toward bottom (near-side players are
        # well below the ball trajectory in beach volleyball)
        deficit = min_roi_height - roi_height
        expand_top = deficit * 0.3
        expand_bottom = deficit * 0.7
        roi_y_min = max(0.0, roi_y_min - expand_top)
        roi_y_max = min(1.0, roi_y_max + expand_bottom)
        # If clamped at bottom, shift remaining expansion to top
        if roi_y_max - roi_y_min < min_roi_height:
            roi_y_min = max(0.0, roi_y_max - min_roi_height)
        # If clamped at top, shift remaining expansion to bottom
        if roi_y_max - roi_y_min < min_roi_height:
            roi_y_max = min(1.0, roi_y_min + min_roi_height)

    # Quality check: ROI too large (ball spread across full frame)
    roi_area = (roi_x_max - roi_x_min) * (roi_y_max - roi_y_min)
    if roi_area > max_roi_area:
        return None, (
            f"Ball trajectory spans {roi_area * 100:.0f}% of frame "
            "(too spread for useful court masking). "
            "Use --court-roi to specify the court area manually, "
            "or use --calibration for precise court boundaries."
        )

    roi = [
        (roi_x_min, roi_y_min),  # top-left
        (roi_x_max, roi_y_min),  # top-right
        (roi_x_max, roi_y_max),  # bottom-right
        (roi_x_min, roi_y_max),  # bottom-left
    ]

    # Quality assessment (ball_coverage computed before min-dimension expansion)
    quality_msg = ""
    if ball_coverage < 0.02:
        quality_msg = (
            f"Ball trajectory is narrow ({ball_coverage * 100:.1f}% of frame). "
            "Adaptive ROI may be inaccurate. "
            "Consider using --court-roi for better results."
        )
    elif len(points) < 50:
        quality_msg = (
            f"Limited ball detections ({len(points)} points). "
            "Adaptive ROI is approximate. "
            "Consider using --court-roi for precise court masking."
        )

    return roi, quality_msg

# Available YOLO model sizes (larger = more accurate but slower)
# Benchmark on beach volleyball (10 labeled rallies, imgsz=1280):
#   yolo11s:  6.1 FPS, 92.5% F1, 91.3% HOTA (default - best accuracy/speed)
#   yolov8n:  7.7 FPS, 79.4% F1, 80.3% HOTA (faster, lower far-court recall)
#   yolo11m:  2.4 FPS, 77.0% F1, 82.4% HOTA (mid-recall regression)
#   yolov8m:  7.0 FPS, 89.2% F1 (imgsz=640 benchmark)
YOLO_MODELS = {
    "yolov8n": "yolov8n.pt",  # Nano: 3.2M params, fastest
    "yolov8s": "yolov8s.pt",  # Small: 11.2M params
    "yolov8m": "yolov8m.pt",  # Medium: 25.9M params
    "yolov8l": "yolov8l.pt",  # Large: 43.7M params
    "yolo11n": "yolo11n.pt",  # YOLO11 Nano: 2.6M params
    "yolo11s": "yolo11s.pt",  # YOLO11 Small: 9.4M params (default)
    "yolo11s-pose": "yolo11s-pose.pt",  # YOLO11 Small Pose: keypoints + detection
    "yolo11m": "yolo11m.pt",  # YOLO11 Medium: 20.1M params
    "yolo11l": "yolo11l.pt",  # YOLO11 Large: 25.3M params
}
DEFAULT_YOLO_MODEL = "yolo11s"

# Tracker configs for better tracking stability
BYTETRACK_CONFIG = Path(__file__).parent / "bytetrack_volleyball.yaml"
BOTSORT_CONFIG = Path(__file__).parent / "botsort_volleyball.yaml"

# Available trackers
TRACKER_BYTETRACK = "bytetrack"
TRACKER_BOTSORT = "botsort"
TRACKER_BOXMOT_BOTSORT = "boxmot-botsort"  # BoxMOT BoT-SORT with custom OSNet ReID
DEFAULT_TRACKER = TRACKER_BOXMOT_BOTSORT  # BoxMOT + OSNet ReID: +2.8pp F1, -59% FP vs ultralytics

# Preprocessing options
PREPROCESSING_NONE = "none"
PREPROCESSING_CLAHE = "clahe"  # Contrast Limited Adaptive Histogram Equalization


class _IdentityCMC:
    """No-op camera motion compensation for fixed tripod cameras.

    BoxMOT's BotSort requires a CMC object but doesn't support "none".
    This returns an identity affine matrix (no warp), avoiding the small
    estimation errors that SOF/ECC introduce on static cameras which
    perturb Kalman predictions and cause track fragmentation.
    """

    def apply(
        self, img: np.ndarray, dets: np.ndarray | None = None,
    ) -> np.ndarray:
        return np.eye(2, 3, dtype=np.float32)


def apply_clahe_preprocessing(frame: np.ndarray) -> np.ndarray:
    """Apply CLAHE preprocessing to improve detection on sand backgrounds.

    Based on KTH paper finding that CLAHE on grayscale helps with sand background
    contrast issues in beach volleyball. CLAHE enhances local contrast without
    over-amplifying noise.

    Args:
        frame: BGR image (OpenCV format).

    Returns:
        Preprocessed BGR image with enhanced contrast.
    """
    # Convert to grayscale for CLAHE (luminance channel)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply CLAHE with volleyball-tuned parameters
    # clipLimit=2.0 prevents over-amplification
    # tileGridSize=(16,16) provides good local adaptation for court-sized regions
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    enhanced_gray = clahe.apply(gray)

    # Convert back to BGR by replacing luminance in LAB color space
    # This preserves color information while enhancing contrast
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = enhanced_gray  # Replace L channel with CLAHE-enhanced
    enhanced_bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    return enhanced_bgr


def _get_model_cache_dir() -> Path:
    """Get the cache directory for player tracking models."""
    from platformdirs import user_cache_dir

    cache_dir = Path(user_cache_dir("rallycut")) / "models" / "player_tracking"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


@dataclass
class PlayerPosition:
    """Single player detection result."""

    frame_number: int
    track_id: int  # ByteTrack assigned ID (-1 if no tracking)
    x: float  # Normalized 0-1 (bbox center, relative to video width)
    y: float  # Normalized 0-1 (bbox center, relative to video height)
    width: float  # Normalized bbox width
    height: float  # Normalized bbox height
    confidence: float  # Detection confidence 0-1

    def to_dict(self) -> dict:
        return {
            "frameNumber": self.frame_number,
            "trackId": self.track_id,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
            "confidence": self.confidence,
        }


@dataclass
class PlayerTrackingResult:
    """Complete player tracking result for a video segment."""

    positions: list[PlayerPosition] = field(default_factory=list)
    frame_count: int = 0
    video_fps: float = 30.0
    video_width: int = 0
    video_height: int = 0
    processing_time_ms: float = 0.0
    model_version: str = MODEL_NAME

    # Court split Y for two-team filtering (debug overlay)
    # Camera is always behind baseline, so teams split by horizontal line
    court_split_y: float | None = None  # 0-1 normalized Y coordinate
    primary_track_ids: list[int] = field(default_factory=list)  # Stable track IDs
    filter_method: str | None = None  # Filter method used (e.g., "bbox_size+track_stability+two_team")

    # Ball positions for trajectory overlay
    ball_positions: list[BallPosition] = field(default_factory=list)

    # Raw positions before filtering (for parameter tuning)
    raw_positions: list[PlayerPosition] = field(default_factory=list)

    # Team classification (track_id → team 0=near/1=far)
    team_assignments: dict[int, int] = field(default_factory=dict)

    # Quality report (set when filter_enabled=True and court_roi is set)
    quality_report: TrackingQualityReport | None = None

    # Color histogram store (for post-hoc analysis/diagnostics)
    color_store: ColorHistogramStore | None = None

    # Multi-region appearance descriptor store (for post-hoc analysis/diagnostics)
    appearance_store: AppearanceDescriptorStore | None = None

    @property
    def avg_players_per_frame(self) -> float:
        """Average number of players detected per frame."""
        if self.frame_count == 0:
            return 0.0
        # Count positions by frame
        frame_counts: dict[int, int] = {}
        for p in self.positions:
            frame_counts[p.frame_number] = frame_counts.get(p.frame_number, 0) + 1
        if not frame_counts:
            return 0.0
        return sum(frame_counts.values()) / len(frame_counts)

    @property
    def unique_track_count(self) -> int:
        """Number of unique track IDs assigned."""
        return len({p.track_id for p in self.positions if p.track_id >= 0})

    @property
    def detection_rate(self) -> float:
        """Percentage of frames with at least one player detected."""
        if self.frame_count == 0:
            return 0.0
        frames_with_players = len({p.frame_number for p in self.positions})
        return frames_with_players / self.frame_count

    def to_dict(self) -> dict:
        result = {
            "positions": [p.to_dict() for p in self.positions],
            "frameCount": self.frame_count,
            "videoFps": self.video_fps,
            "videoWidth": self.video_width,
            "videoHeight": self.video_height,
            "avgPlayersPerFrame": self.avg_players_per_frame,
            "uniqueTrackCount": self.unique_track_count,
            "detectionRate": self.detection_rate,
            "processingTimeMs": self.processing_time_ms,
            "modelVersion": self.model_version,
        }
        # Include court split Y for debug overlay (horizontal line)
        if self.court_split_y is not None:
            result["courtSplitY"] = self.court_split_y
        if self.primary_track_ids:
            result["primaryTrackIds"] = self.primary_track_ids
        if self.filter_method:
            result["filterMethod"] = self.filter_method

        # Ball positions for trajectory overlay
        if self.ball_positions:
            result["ballPositions"] = [
                {
                    "frameNumber": bp.frame_number,
                    "x": bp.x,
                    "y": bp.y,
                    "confidence": bp.confidence,
                }
                for bp in self.ball_positions
            ]

        # Raw positions before filtering (for parameter tuning)
        if self.raw_positions:
            result["rawPositions"] = [p.to_dict() for p in self.raw_positions]

        # Quality report
        if self.quality_report is not None:
            result["qualityReport"] = self.quality_report.to_dict()

        # Team assignments
        if self.team_assignments:
            result["teamAssignments"] = {
                str(k): v for k, v in self.team_assignments.items()
            }

        return result

    def to_json(
        self,
        path: Path,
        extra_data: dict[str, Any] | None = None,
    ) -> None:
        """Write result to JSON file.

        Args:
            path: Output file path.
            extra_data: Optional extra data to merge into the output dict.
        """
        data = self.to_dict()
        if extra_data:
            data.update(extra_data)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> PlayerTrackingResult:
        """Load result from JSON file."""
        with open(path) as f:
            data = json.load(f)

        positions = [
            PlayerPosition(
                frame_number=p["frameNumber"],
                track_id=p["trackId"],
                x=p["x"],
                y=p["y"],
                width=p["width"],
                height=p["height"],
                confidence=p["confidence"],
            )
            for p in data.get("positions", [])
        ]

        # Parse raw positions (for parameter tuning)
        raw_positions = [
            PlayerPosition(
                frame_number=p["frameNumber"],
                track_id=p["trackId"],
                x=p["x"],
                y=p["y"],
                width=p["width"],
                height=p["height"],
                confidence=p["confidence"],
            )
            for p in data.get("rawPositions", [])
        ]

        # Deserialize team assignments (JSON keys are strings)
        raw_teams = data.get("teamAssignments", {})
        team_assignments = {int(k): v for k, v in raw_teams.items()}

        return cls(
            positions=positions,
            frame_count=data.get("frameCount", 0),
            video_fps=data.get("videoFps", 30.0),
            video_width=data.get("videoWidth", 0),
            video_height=data.get("videoHeight", 0),
            processing_time_ms=data.get("processingTimeMs", 0.0),
            model_version=data.get("modelVersion", MODEL_NAME),
            court_split_y=data.get("courtSplitY"),
            primary_track_ids=data.get("primaryTrackIds", []),
            raw_positions=raw_positions,
            team_assignments=team_assignments,
        )

    def to_api_format(self) -> dict:
        """Convert to format expected by API/UI.

        Groups positions by frame for efficient frontend rendering.
        """
        # Group by frame
        frames: dict[int, list] = {}
        for pos in self.positions:
            if pos.frame_number not in frames:
                frames[pos.frame_number] = []
            frames[pos.frame_number].append({
                "trackId": pos.track_id,
                "x": pos.x,
                "y": pos.y,
                "width": pos.width,
                "height": pos.height,
                "confidence": pos.confidence,
            })

        # Average confidence across all positions
        avg_confidence = (
            sum(p.confidence for p in self.positions) / len(self.positions)
            if self.positions
            else 0.0
        )

        return {
            "trackingData": [
                {"frameNumber": fn, "players": players}
                for fn, players in sorted(frames.items())
            ],
            "frameCount": self.frame_count,
            "detectionRate": self.detection_rate,
            "avgConfidence": avg_confidence,
            "avgPlayerCount": self.avg_players_per_frame,
            "uniqueTrackCount": self.unique_track_count,
        }


class PlayerTracker:
    """
    Player tracker using YOLO + BoT-SORT.

    Uses YOLO for person detection with BoT-SORT for temporal tracking
    across frames. Supports YOLOv8 and YOLO11 model families.
    """

    def __init__(
        self,
        model_path: Path | None = None,
        confidence: float = DEFAULT_CONFIDENCE,
        iou: float = DEFAULT_IOU,
        preprocessing: str = PREPROCESSING_NONE,
        tracker: str = DEFAULT_TRACKER,
        yolo_model: str = DEFAULT_YOLO_MODEL,
        with_reid: bool = True,
        appearance_thresh: float | None = None,
        reid_model: str | None = None,
        imgsz: int = DEFAULT_IMGSZ,
        court_roi: list[tuple[float, float]] | None = None,
        reid_weights_path: Path | None = None,
    ):
        """
        Initialize player tracker.

        Args:
            model_path: Optional path to YOLOv8 model. If not provided,
                       downloads to cache on first use.
            confidence: Detection confidence threshold.
            iou: NMS IoU threshold.
            preprocessing: Preprocessing method for frames. Options:
                          - "none": No preprocessing (default)
                          - "clahe": CLAHE contrast enhancement for sand backgrounds
            tracker: Tracking algorithm. Options:
                    - "bytetrack": ByteTrack (motion-based)
                    - "botsort": BoT-SORT (adds camera motion compensation, default)
                    - "boxmot-botsort": BoxMOT BoT-SORT with custom OSNet ReID
            yolo_model: YOLO model size (default: yolo11s). Options:
                       yolov8n/s/m/l and yolo11n/s/m/l.
            with_reid: Enable BoT-SORT ReID for appearance-based re-identification.
                      Enabled by default (reduces ID switches by ~40-60%).
            appearance_thresh: Override BoT-SORT appearance threshold for ReID.
                              If None, uses value from config YAML.
            reid_model: Override ReID model for BoT-SORT. Options:
                       - None: use config YAML default ("auto" = YOLO backbone features)
                       - "auto": use YOLO backbone features (no separate model)
                       - "yolo11n-cls.pt": YOLO11 nano classification model
                       - "yolo11s-cls.pt": YOLO11 small classification model
                       - Any YOLO-compatible model path for ReID embeddings
            imgsz: Inference resolution. Higher values improve small/far object
                  detection at the cost of speed. 1280 (default, best tradeoff),
                  640 (faster, lower far-court recall), 1920 (native resolution).
            court_roi: Court ROI polygon as list of (x, y) normalized 0-1
                      coordinates. Regions outside this polygon are masked
                      before detection, preventing tracks from forming on
                      background objects (wall drawings, spectators, etc.).
                      Use DEFAULT_COURT_ROI for a conservative rectangle.
                      None disables ROI masking.
            reid_weights_path: Path to custom ReID weights for boxmot-botsort.
                              If None and tracker is boxmot-botsort, uses the
                              default GeneralReIDModel weights.
        """
        self.model_path = model_path
        self.confidence = confidence
        self.iou = iou
        self.preprocessing = preprocessing
        self.tracker = tracker
        self.yolo_model = yolo_model
        self.with_reid = with_reid
        self.appearance_thresh = appearance_thresh
        self.reid_model = reid_model
        self.imgsz = imgsz
        self.court_roi = court_roi
        self.reid_weights_path = reid_weights_path
        self._model: Any = None
        self._custom_tracker_config: Path | None = None
        self._boxmot_tracker: Any = None
        self._reid_model: Any = None

    def _get_tracker_config(self) -> Path:
        """Get the tracker config file path based on selected tracker.

        The YAML config has with_reid and model set by default. Only creates
        a modified config if appearance_thresh, with_reid=False, or reid_model
        is explicitly set.
        """
        if self._custom_tracker_config is not None:
            return self._custom_tracker_config

        base_config = BOTSORT_CONFIG if self.tracker == TRACKER_BOTSORT else BYTETRACK_CONFIG

        # Only override if appearance_thresh, reid_model is explicitly set, or ReID is disabled
        needs_override = (
            self.tracker == TRACKER_BOTSORT
            and (
                self.appearance_thresh is not None
                or not self.with_reid
                or self.reid_model is not None
            )
        )
        if needs_override:
            import tempfile

            import yaml

            with open(base_config) as f:
                config = yaml.safe_load(f)

            if not self.with_reid:
                config["with_reid"] = False
                logger.info("Disabling BoT-SORT ReID")
            if self.appearance_thresh is not None:
                config["appearance_thresh"] = self.appearance_thresh
                logger.info(f"Overriding appearance_thresh to {self.appearance_thresh}")
            if self.reid_model is not None:
                config["model"] = self.reid_model
                logger.info(f"Overriding ReID model to {self.reid_model}")

            # Write to temp file (persists for tracker lifetime)
            tmp = tempfile.NamedTemporaryFile(
                mode="w", suffix=".yaml", prefix="botsort_", delete=False
            )
            yaml.dump(config, tmp)
            tmp.close()
            self._custom_tracker_config = Path(tmp.name)
            return self._custom_tracker_config

        return base_config

    def _init_boxmot_tracker(self, video_fps: float = 30.0) -> Any:
        """Initialize BoxMOT BotSort tracker with custom ReID embeddings.

        Creates a BotSort tracker configured with the same parameters as
        botsort_volleyball.yaml but using externally-provided ReID embeddings
        (via the embs parameter in update()) instead of BoxMOT's built-in ReID.

        This allows using our fine-tuned OSNet-x1.0 model which produces
        128-dim embeddings trained with SupCon loss on volleyball data,
        rather than BoxMOT's generic ReID or ultralytics' YOLO backbone features.

        Args:
            video_fps: Video frame rate. BoT-SORT uses this to scale
                track_buffer into real time (buffer_size = fps/30 * track_buffer).
                Passing the actual FPS ensures consistent ~1.5s track retention
                regardless of whether the video is 30fps or 60fps.
        """
        from boxmot.trackers.botsort.basetrack import BaseTrack
        from boxmot.trackers.botsort.botsort import BotSort

        BaseTrack.clear_count()

        appearance_thresh = self.appearance_thresh if self.appearance_thresh is not None else 0.30

        import torch

        # Round FPS to nearest int for BoT-SORT (expects int frame_rate)
        frame_rate = max(int(round(video_fps)), 1)

        # Create tracker with with_reid=False to skip loading BoxMOT's
        # internal ReID model. We'll set with_reid=True after construction
        # and pass embeddings via the embs parameter in update().
        tracker = BotSort(
            reid_weights=Path("unused"),  # Not loaded when with_reid=False
            device=torch.device("cpu"),   # Not used for ReID
            half=False,
            track_high_thresh=0.25,
            track_low_thresh=0.08,
            new_track_thresh=0.35,
            track_buffer=45,
            match_thresh=0.90,
            proximity_thresh=0.5,
            appearance_thresh=appearance_thresh,
            cmc_method="sof",  # Overridden with _IdentityCMC below
            frame_rate=frame_rate,
            fuse_first_associate=False,
            with_reid=False,  # Skip model loading
            min_hits=1,  # Show tracks after 1 hit (same as ultralytics default)
        )

        # Enable appearance matching in update() without loading a model.
        # Safety: BotSort.update() checks `self.with_reid and embs is None`
        # before accessing self.model (botsort.py:124). Since we always pass
        # embs, self.model is never touched. We MUST always pass embs when
        # calling update() on this tracker.
        tracker.with_reid = True

        # Disable camera motion compensation for fixed tripod cameras.
        # BoxMOT doesn't support cmc_method="none", so we replace the CMC
        # with a no-op that returns identity. SOF (sparse optical flow) on a
        # static camera introduces small affine estimation errors that perturb
        # Kalman predictions and cause severe track fragmentation.
        tracker.cmc = _IdentityCMC()

        # BoT-SORT computes: buffer_size = int(frame_rate/30 * track_buffer)
        # At 30fps: buffer_size=45 (1.5s). At 60fps: buffer_size=90 (1.5s).
        logger.info(
            "BoxMOT BotSort initialized (appearance_thresh=%.2f, "
            "track_buffer=45, frame_rate=%d, buffer_size=%d, match_thresh=0.90)",
            appearance_thresh,
            frame_rate,
            tracker.buffer_size,
        )
        return tracker

    def _load_reid_model(self) -> Any:
        """Load the GeneralReIDModel for extracting ReID embeddings.

        Returns the model ready for inference. Uses reid_weights_path if
        provided, otherwise falls back to the default WEIGHTS_PATH.
        """
        from rallycut.tracking.reid_general import WEIGHTS_PATH, GeneralReIDModel

        weights = self.reid_weights_path or WEIGHTS_PATH
        if not weights.exists():
            raise FileNotFoundError(
                f"ReID weights not found at {weights}. "
                "Train with: uv run python -m rallycut.tracking.reid_general"
            )

        t0 = time.monotonic()
        model = GeneralReIDModel(weights_path=weights)
        t_load = time.monotonic() - t0
        print(f"Loaded OSNet ReID model ({t_load:.1f}s)", flush=True)
        return model

    def _extract_reid_embeddings(
        self,
        frame: np.ndarray,
        dets: np.ndarray,
    ) -> np.ndarray:
        """Extract ReID embeddings for detected bounding boxes.

        Args:
            frame: BGR image (H, W, 3).
            dets: Detection array (N, 6) with [x1, y1, x2, y2, conf, cls].

        Returns:
            (N, 128) L2-normalized embeddings from GeneralReIDModel.
        """
        if self._reid_model is None:
            raise RuntimeError("ReID model not loaded — call _load_reid_model() first")

        if len(dets) == 0:
            return np.empty((0, 128), dtype=np.float32)

        h, w = frame.shape[:2]
        crops: list[np.ndarray] = []
        for det in dets:
            x1, y1, x2, y2 = det[:4].astype(int)
            # Clamp to frame bounds
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(w, x2)
            y2 = min(h, y2)
            if x2 <= x1 or y2 <= y1:
                # Degenerate box — use a tiny black crop
                crops.append(np.zeros((10, 10, 3), dtype=np.uint8))
                continue
            crops.append(frame[y1:y2, x1:x2].copy())

        embeddings: np.ndarray = self._reid_model.extract_embeddings(crops)
        return embeddings

    def _decode_boxmot_results(
        self,
        tracks: np.ndarray,
        frame_number: int,
        video_width: int,
        video_height: int,
    ) -> list[PlayerPosition]:
        """Decode BoxMOT tracker output to PlayerPosition list.

        Args:
            tracks: BoxMOT output (M, 8) with
                    [x1, y1, x2, y2, track_id, conf, cls, det_idx].
            frame_number: Current frame index.
            video_width: Video frame width.
            video_height: Video frame height.

        Returns:
            List of PlayerPosition for this frame.
        """
        positions: list[PlayerPosition] = []

        if tracks is None or len(tracks) == 0:
            return positions

        for row in tracks:
            x1, y1, x2, y2 = row[0], row[1], row[2], row[3]
            track_id, conf, cls = row[4], row[5], row[6]

            if int(cls) != PERSON_CLASS_ID:
                continue

            cx = (x1 + x2) / 2 / video_width
            cy = (y1 + y2) / 2 / video_height
            w = (x2 - x1) / video_width
            h = (y2 - y1) / video_height

            positions.append(PlayerPosition(
                frame_number=frame_number,
                track_id=int(track_id),
                x=float(cx),
                y=float(cy),
                width=float(w),
                height=float(h),
                confidence=float(conf),
            ))

        return positions

    def _reset_boxmot_tracker(self, video_fps: float = 30.0) -> None:
        """Reset BoxMOT tracker state between rallies.

        Re-creates the tracker to ensure all state is clean — including
        CMC (sparse optical flow) internal state which cannot be partially
        reset. The ReID model is NOT reloaded (kept in self._reid_model).
        """
        self._boxmot_tracker = self._init_boxmot_tracker(video_fps)

    def _get_model_filename(self) -> str:
        """Get the model filename for the selected YOLO model size."""
        if self.yolo_model in YOLO_MODELS:
            return YOLO_MODELS[self.yolo_model]
        # Assume it's already a filename like "yolov8s.pt"
        return self.yolo_model if self.yolo_model.endswith(".pt") else f"{self.yolo_model}.pt"

    def _ensure_model(self) -> Path:
        """Ensure model is available, downloading if necessary."""
        if self.model_path and self.model_path.exists():
            return self.model_path

        cache_dir = _get_model_cache_dir()
        model_filename = self._get_model_filename()
        cached_path = cache_dir / model_filename

        # YOLOv8 downloads models automatically on first use
        # We just return the expected cache path
        self.model_path = cached_path
        return cached_path

    def _load_model(self) -> Any:
        """Load YOLOv8 model with ByteTrack."""
        if self._model is not None:
            return self._model

        # Disable ultralytics network calls that can hang in subprocess contexts
        # (update checks, telemetry, hub sync). Must be set before import.
        os.environ.setdefault("YOLO_AUTOCHECK", "False")
        os.environ.setdefault("ULTRALYTICS_SYNC", "False")

        t0 = time.monotonic()
        # print() not logger: must be visible through Node.js subprocess pipe
        # where the default logging level is WARNING.
        print("Loading YOLO: importing ultralytics...", flush=True)

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package is required for player tracking. "
                "Install with: pip install ultralytics>=8.2.0"
            )

        t_import = time.monotonic() - t0
        print(f"Loading YOLO: ultralytics imported ({t_import:.1f}s)", flush=True)

        # Suppress ultralytics' verbose output (download progress, telemetry)
        logging.getLogger("ultralytics").setLevel(logging.WARNING)

        # Load model - ultralytics handles download automatically
        model_path = self._ensure_model()
        model_filename = self._get_model_filename()

        t1 = time.monotonic()
        # Try to load from cache first, fallback to auto-download
        if model_path.exists():
            print(f"Loading YOLO: {model_filename} from cache...", flush=True)
            self._model = YOLO(str(model_path))
        else:
            # Download model automatically
            print(f"Loading YOLO: downloading {model_filename}...", flush=True)
            self._model = YOLO(model_filename)

        t_load = time.monotonic() - t1
        print(
            f"Loading YOLO: {model_filename} ready ({t_load:.1f}s, "
            f"total {time.monotonic() - t0:.1f}s)",
            flush=True,
        )

        return self._model

    def _decode_results(
        self,
        results: Any,
        frame_number: int,
        video_width: int,
        video_height: int,
    ) -> list[PlayerPosition]:
        """
        Decode YOLO results to PlayerPosition list.

        Args:
            results: YOLO inference results.
            frame_number: Current frame index.
            video_width: Video frame width.
            video_height: Video frame height.

        Returns:
            List of PlayerPosition for this frame.
        """
        positions: list[PlayerPosition] = []

        if results is None or len(results) == 0:
            return positions

        result = results[0]  # Single image result

        # Get boxes - format depends on whether tracking is enabled
        if hasattr(result, "boxes") and result.boxes is not None:
            boxes = result.boxes

            # Get number of detections
            n_detections = len(boxes.xyxy) if hasattr(boxes, "xyxy") and boxes.xyxy is not None else 0

            for i in range(n_detections):
                try:
                    # Get class ID - only process person class
                    cls_val = boxes.cls[i] if hasattr(boxes, "cls") and boxes.cls is not None and i < len(boxes.cls) else 0
                    cls = int(cls_val.item() if hasattr(cls_val, "item") else cls_val)
                    if cls != PERSON_CLASS_ID:
                        continue

                    # Get confidence
                    conf_val = boxes.conf[i] if hasattr(boxes, "conf") and boxes.conf is not None and i < len(boxes.conf) else 1.0
                    conf = float(conf_val.item() if hasattr(conf_val, "item") else conf_val)

                    # Get track ID if available (may have fewer elements than boxes)
                    track_id = -1
                    if hasattr(boxes, "id") and boxes.id is not None and i < len(boxes.id):
                        id_val = boxes.id[i]
                        track_id = int(id_val.item() if hasattr(id_val, "item") else id_val)

                    # Get bounding box (xyxy format)
                    xyxy_raw = boxes.xyxy[i]
                    xyxy = xyxy_raw.cpu().numpy() if hasattr(xyxy_raw, "cpu") else np.asarray(xyxy_raw)
                    x1, y1, x2, y2 = xyxy

                    # Convert to normalized center coordinates
                    cx = (x1 + x2) / 2 / video_width
                    cy = (y1 + y2) / 2 / video_height
                    w = (x2 - x1) / video_width
                    h = (y2 - y1) / video_height

                    positions.append(PlayerPosition(
                        frame_number=frame_number,
                        track_id=track_id,
                        x=float(cx),
                        y=float(cy),
                        width=float(w),
                        height=float(h),
                        confidence=conf,
                    ))
                except (IndexError, RuntimeError) as e:
                    # Skip this detection if there's an indexing issue
                    logger.debug(f"Skipping detection {i} due to error: {e}")
                    continue

        return positions

    def _apply_roi_mask(
        self,
        frame: np.ndarray,
        roi: list[tuple[float, float]],
    ) -> np.ndarray:
        """Apply ROI mask to frame, blacking out regions outside the polygon.

        This prevents YOLO from detecting persons outside the court area,
        which prevents BoT-SORT from forming tracks on background objects
        (wall drawings, spectators, etc.).

        Args:
            frame: BGR image.
            roi: Polygon as list of (x, y) normalized 0-1 coordinates.

        Returns:
            Masked frame with regions outside ROI blacked out.
        """
        h, w = frame.shape[:2]
        pts = np.array(
            [(int(x * w), int(y * h)) for x, y in roi],
            dtype=np.int32,
        )
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [pts], (255,))
        return cv2.bitwise_and(frame, frame, mask=mask)

    @staticmethod
    def _filter_detections(
        positions: list[PlayerPosition],
        max_per_frame: int = MAX_DETECTIONS_PER_FRAME,
    ) -> list[PlayerPosition]:
        """Apply sport-aware detection filters to reduce noise.

        Filters:
        1. Aspect ratio: persons are taller than wide (height > width)
        2. Zone-dependent minimum size: far-court (low Y) players are smaller,
           near-court (high Y) players must be bigger
        3. Detection count cap: keep top detections by confidence

        Args:
            positions: Decoded detections for a single frame.
            max_per_frame: Maximum detections to keep per frame.

        Returns:
            Filtered positions.
        """
        filtered = []
        for p in positions:
            # Aspect ratio: reject clearly non-person shapes (very wide)
            # Lenient threshold: allows crouching/diving players (width up to 1.5x height)
            if p.width > 1.5 * p.height:
                continue

            # Zone-dependent minimum area:
            # Far-court players (low Y) are smaller in the frame
            # Linear scaling: min_area = 0.0005 at top to 0.003 at bottom
            min_area = 0.0005 + 0.0025 * p.y
            if p.width * p.height < min_area:
                continue

            filtered.append(p)

        # Cap detections per frame to reduce tracker noise
        if len(filtered) > max_per_frame:
            filtered.sort(key=lambda p: p.confidence, reverse=True)
            filtered = filtered[:max_per_frame]

        return filtered

    def track_video(
        self,
        video_path: Path | str,
        start_ms: int | None = None,
        end_ms: int | None = None,
        stride: int = 1,
        progress_callback: Callable[[float], None] | None = None,
        ball_positions: list[BallPosition] | None = None,
        filter_enabled: bool = False,
        filter_config: PlayerFilterConfig | None = None,
        court_calibrator: CourtCalibrator | None = None,
        court_detection_insights: CourtDetectionInsights | None = None,
        skip_global_identity: bool = False,
    ) -> PlayerTrackingResult:
        """
        Track players in a video segment.

        Args:
            video_path: Path to video file.
            start_ms: Start time in milliseconds (optional).
            end_ms: End time in milliseconds (optional).
            stride: Process every Nth frame (1=all, 3=every 3rd for faster processing).
            progress_callback: Optional callback(progress: float) for progress updates.
            ball_positions: Ball tracking results for court player filtering.
            filter_enabled: If True, filter to court players only.
            filter_config: Configuration for player filtering (court type, thresholds).
            court_calibrator: Optional calibrated court calibrator. Used for
                            court-space team classification and post-processing
                            court presence scoring.

        Returns:
            PlayerTrackingResult with all detected positions.
        """
        start_time = time.time()
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Load YOLO model
        model = self._load_model()
        use_boxmot = self.tracker == TRACKER_BOXMOT_BOTSORT

        if use_boxmot:
            # BoxMOT path: load ReID model (FPS-independent).
            # Tracker creation is deferred until after video FPS is known,
            # so BoT-SORT's Kalman filter and track buffer are correctly
            # scaled for the video's actual frame rate.
            if self._reid_model is None:
                self._reid_model = self._load_reid_model()
            # Clear ultralytics predictor state (no tracking, detection only)
            if hasattr(model, "predictor") and model.predictor is not None:
                if hasattr(model.predictor, "trackers"):
                    del model.predictor.trackers
        else:
            # Reset tracker state from any previous track_video call.
            # ultralytics caches trackers, feature hooks, and ReID state on the
            # predictor when persist=True. When ReID is enabled (with_reid),
            # the feature extraction hook state can't be cleanly reset without
            # a full model reload. For non-ReID, just clearing trackers suffices.
            if hasattr(model, "predictor") and model.predictor is not None:
                if self.with_reid:
                    # Full reload needed for clean ReID hook state
                    self._model = None
                    model = self._load_model()
                elif hasattr(model.predictor, "trackers"):
                    del model.predictor.trackers

        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        try:
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Initialize/reset BoxMOT tracker now that FPS is known.
            # BoT-SORT scales track_buffer by fps/30 internally, so passing
            # the actual FPS ensures consistent ~1.5s track retention.
            if use_boxmot:
                self._boxmot_tracker = self._init_boxmot_tracker(fps)

            # Scale frame-count thresholds for videos that aren't 30fps.
            # All defaults are tuned for 30fps; at 60fps the same real-world
            # durations need 2x more frames.
            if filter_config is not None:
                filter_config = filter_config.scaled_for_fps(fps)

            # Calculate frame range
            start_frame = 0
            end_frame = total_frames

            if start_ms is not None:
                start_frame = int(start_ms / 1000 * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

            if end_ms is not None:
                end_frame = min(int(end_ms / 1000 * fps), total_frames)

            total_frames_in_range = end_frame - start_frame
            # With stride, we process fewer frames
            frames_to_process = (total_frames_in_range + stride - 1) // stride
            print(
                f"YOLO tracking: {total_frames_in_range} frames "
                f"(stride={stride}, {frames_to_process} to process, {fps:.1f} fps)",
                flush=True,
            )

            positions: list[PlayerPosition] = []
            frame_idx = start_frame
            frames_processed = 0

            # Color histogram store for post-hoc track repair
            color_store = None
            appearance_store = None
            histogram_stride = 3  # Extract every 3rd processed frame
            if filter_enabled:
                from rallycut.tracking.appearance_descriptor import (
                    AppearanceDescriptorStore,
                    extract_multi_region_descriptor,
                )
                from rallycut.tracking.color_repair import (
                    ColorHistogramStore,
                    extract_shorts_histogram,
                )

                color_store = ColorHistogramStore()
                appearance_store = AppearanceDescriptorStore()

            while frame_idx < end_frame:
                # For strided processing, grab() skips frame decoding
                # (~2x faster than read() for non-processed frames)
                is_process_frame = (frame_idx - start_frame) % stride == 0
                if not is_process_frame:
                    if not cap.grab():
                        break
                    frame_idx += 1
                    continue

                ret, frame = cap.read()
                if not ret:
                    logger.warning(
                        f"Frame read failed at index {frame_idx} "
                        f"(expected {total_frames_in_range} frames)"
                    )
                    break

                # Apply preprocessing if enabled (e.g., CLAHE for sand backgrounds)
                processed_frame = frame
                if self.preprocessing == PREPROCESSING_CLAHE:
                    processed_frame = apply_clahe_preprocessing(frame)

                # Apply ROI mask to prevent tracks on background objects
                frame_to_track = processed_frame
                if self.court_roi is not None:
                    frame_to_track = self._apply_roi_mask(
                        processed_frame, self.court_roi
                    )

                # Run detection + tracking
                try:
                    if use_boxmot:
                        # BoxMOT path: YOLO detect → extract ReID → BoxMOT track
                        results = model.predict(
                            frame_to_track,
                            conf=self.confidence,
                            iou=self.iou,
                            imgsz=self.imgsz,
                            classes=[PERSON_CLASS_ID],
                            verbose=False,
                        )

                        # Convert YOLO detections to BoxMOT format (N, 6)
                        dets_array = np.empty((0, 6), dtype=np.float32)
                        if results and len(results) > 0 and results[0].boxes is not None:
                            boxes = results[0].boxes
                            if len(boxes.xyxy) > 0:
                                xyxy = boxes.xyxy.cpu().numpy()
                                conf = boxes.conf.cpu().numpy().reshape(-1, 1)
                                cls = boxes.cls.cpu().numpy().reshape(-1, 1)
                                dets_array = np.hstack([xyxy, conf, cls]).astype(np.float32)

                        # Extract ReID embeddings from our fine-tuned OSNet
                        embs = self._extract_reid_embeddings(frame_to_track, dets_array)

                        # Update BoxMOT tracker
                        tracks = self._boxmot_tracker.update(dets_array, frame_to_track, embs)

                        # Decode BoxMOT output to PlayerPosition
                        frame_positions = self._decode_boxmot_results(
                            tracks, frame_idx, video_width, video_height
                        )
                    else:
                        # Ultralytics path: YOLO detect + track in one call
                        results = model.track(
                            frame_to_track,
                            persist=True,
                            tracker=str(self._get_tracker_config()),
                            conf=self.confidence,
                            iou=self.iou,
                            imgsz=self.imgsz,
                            classes=[PERSON_CLASS_ID],
                            verbose=False,
                        )

                        # Decode results
                        frame_positions = self._decode_results(
                            results, frame_idx, video_width, video_height
                        )

                    # Sport-aware detection filters
                    if filter_enabled:
                        frame_positions = self._filter_detections(frame_positions)

                    positions.extend(frame_positions)

                    # Extract color histograms for post-hoc repair
                    if (
                        color_store is not None
                        and frames_processed % histogram_stride == 0
                    ):
                        for p in frame_positions:
                            if p.track_id >= 0:
                                hist = extract_shorts_histogram(
                                    frame,  # Original BGR, NOT masked
                                    (p.x, p.y, p.width, p.height),
                                    video_width,
                                    video_height,
                                )
                                if hist is not None:
                                    color_store.add(
                                        p.track_id, p.frame_number, hist
                                    )

                    # Extract multi-region appearance descriptors
                    if (
                        appearance_store is not None
                        and frames_processed % histogram_stride == 0
                    ):
                        for p in frame_positions:
                            if p.track_id >= 0:
                                desc = extract_multi_region_descriptor(
                                    frame,
                                    (p.x, p.y, p.width, p.height),
                                    video_width,
                                    video_height,
                                )
                                if desc is not None:
                                    appearance_store.add(
                                        p.track_id, p.frame_number, desc
                                    )
                except (IndexError, RuntimeError, ValueError) as e:
                    # Handle any errors from YOLO/ByteTrack internals
                    logger.debug(f"Frame {frame_idx} tracking failed: {e}")

                frames_processed += 1

                # Progress callback (for Rich progress bar)
                if progress_callback and frames_processed % 30 == 0:
                    pct = frames_processed / frames_to_process
                    progress_callback(pct)

                # Periodic progress to stdout (visible through pipes where
                # Rich live display is buffered)
                if frames_processed % 100 == 0 or frames_processed == 1:
                    elapsed = time.time() - start_time
                    fps_actual = frames_processed / max(elapsed, 0.001)
                    print(
                        f"YOLO tracking: {frames_processed}/{frames_to_process} "
                        f"frames ({fps_actual:.1f} FPS)",
                        flush=True,
                    )

                frame_idx += 1

            # Final progress
            elapsed = time.time() - start_time
            print(
                f"YOLO tracking: done ({frames_to_process} frames in "
                f"{elapsed:.1f}s, {frames_to_process / max(elapsed, 0.001):.1f} FPS)",
                flush=True,
            )
            if progress_callback:
                progress_callback(1.0)

            # Court split Y for debug overlay (horizontal line)
            court_split_y: float | None = None
            primary_track_ids: list[int] = []
            filter_method: str | None = None

            # Save raw positions before filtering (for parameter tuning)
            raw_positions = [
                PlayerPosition(
                    frame_number=p.frame_number,
                    track_id=p.track_id,
                    x=p.x,
                    y=p.y,
                    width=p.width,
                    height=p.height,
                    confidence=p.confidence,
                )
                for p in positions
            ]

            # Apply court player filtering if enabled (per-frame with track stability)
            num_jump_splits = 0
            num_height_swaps = 0
            num_color_splits = 0
            num_appearance_links = 0

            # Team classification (populated during filtering)
            team_assignments: dict[int, int] = {}

            if filter_enabled:
                from rallycut.tracking.player_filter import (
                    PlayerFilter,
                    PlayerFilterConfig,
                    classify_teams,
                    compute_court_split,
                    remove_stationary_background_tracks,
                    stabilize_track_ids,
                )
                from rallycut.tracking.spatial_consistency import (
                    enforce_spatial_consistency,
                )

                # Get config (or create default)
                config = filter_config or PlayerFilterConfig()

                # Pre-step: Remove stationary background tracks before any
                # post-processing to prevent them from interfering with
                # tracklet linking, court identity, etc.
                positions, removed_bg_tracks = remove_stationary_background_tracks(
                    positions, config,
                    total_frames=frames_to_process,
                )

                # Step 0: Spatial consistency — split tracks at instantaneous
                # jumps only. Drift detection is deferred to step 4e (after
                # identity optimization) so the identity modules can work
                # with full continuous tracks.
                positions, consistency_result = enforce_spatial_consistency(
                    positions,
                    color_store=color_store,
                    appearance_store=appearance_store,
                    video_fps=fps,
                    drift_detection=False,
                )
                num_jump_splits = consistency_result.jump_splits

                # Step 0a: Height-based swap correction
                from rallycut.tracking.height_consistency import fix_height_swaps

                positions, height_swap_result = fix_height_swaps(
                    positions,
                    color_store=color_store,
                    appearance_store=appearance_store,
                )
                num_height_swaps = height_swap_result.swaps

                # Step 0b: Color-based track splitting
                if color_store is not None and color_store.has_data():
                    from rallycut.tracking.color_repair import (
                        split_tracks_by_color,
                    )

                    positions, num_color_splits = split_tracks_by_color(
                        positions, color_store
                    )

                    # Step 0b2: Spatial re-link — reconnect color-split
                    # fragments that are trivially the same player (tiny gap,
                    # near-identical position). Runs before appearance-based
                    # linking to prevent greedy merges from stealing these
                    # obvious spatial matches.
                    from rallycut.tracking.tracklet_link import (
                        link_tracklets_by_appearance,
                        relink_spatial_splits,
                    )

                    positions, num_spatial_relinks = relink_spatial_splits(
                        positions, color_store,
                        appearance_store=appearance_store,
                    )

                    # Step 0c: Appearance-based tracklet linking (GTA-Link inspired)
                    # Reconnects fragments using color histogram similarity.
                    # Runs unconstrained (no team_assignments yet) -- team
                    # classification happens once on clean post-filter data.
                    positions, num_appearance_links = link_tracklets_by_appearance(
                        positions, color_store,
                        appearance_store=appearance_store,
                    )

                # Step 1: Stabilize track IDs before filtering
                # This merges tracks that represent the same player
                positions, id_mapping = stabilize_track_ids(
                    positions, config,
                )
                if id_mapping:
                    if color_store is not None:
                        color_store.remap_ids(id_mapping)
                    if appearance_store is not None:
                        appearance_store.remap_ids(id_mapping)

                num_global_segments = 0
                num_global_remapped = 0
                num_convergence_swaps = 0

                player_filter = PlayerFilter(
                    ball_positions=ball_positions,
                    total_frames=frames_to_process,
                    config=config,
                    court_calibrator=court_calibrator,
                )

                # Step 2: Analyze all positions to identify stable tracks
                # This must be done before per-frame filtering
                player_filter.analyze_tracks(positions)

                # Capture court split Y for debug overlay
                court_split_y = player_filter.court_split_y
                primary_track_ids = sorted(player_filter.primary_tracks)

                # Step 2b: Recover players lost during intermediate pipeline steps
                # If <max_players primary tracks found but raw detections show
                # more concurrent people, a valid player was lost during
                # track splitting/merging. Recover from raw positions.
                num_recovered = 0
                if len(primary_track_ids) < config.max_players and raw_positions:
                    from rallycut.tracking.player_filter import recover_missing_players

                    positions, recovered_primary_set, num_recovered = (
                        recover_missing_players(
                            pipeline_positions=positions,
                            raw_positions=raw_positions,
                            primary_track_ids=player_filter.primary_tracks,
                            total_frames=total_frames_in_range,
                            ball_positions=ball_positions,
                            config=config,
                        )
                    )
                    if num_recovered > 0:
                        player_filter.primary_tracks = recovered_primary_set
                        primary_track_ids = sorted(recovered_primary_set)
                        logger.info(
                            f"Recovered {num_recovered} players from raw: "
                            f"primary tracks now {primary_track_ids}"
                        )

                # Step 3: Group positions by frame
                frames: dict[int, list[PlayerPosition]] = {}
                for p in positions:
                    if p.frame_number not in frames:
                        frames[p.frame_number] = []
                    frames[p.frame_number].append(p)

                # Step 4: Filter each frame separately (uses track stability)
                original_count = len(positions)
                filtered_positions: list[PlayerPosition] = []
                for frame_num in sorted(frames.keys()):
                    frame_players = frames[frame_num]
                    filtered_frame = player_filter.filter(frame_players)
                    filtered_positions.extend(filtered_frame)

                positions = filtered_positions
                filter_method = player_filter.filter_method
                logger.info(
                    f"Filtered {original_count} -> {len(positions)} detections "
                    f"using {filter_method}"
                )

                # Step 4b: Single team classification on clean filtered data
                # All team-dependent operations (global identity, court
                # identity) run after this single classify_teams call.
                split_result = compute_court_split(
                    ball_positions or [], config,
                    player_positions=positions,
                    court_calibrator=court_calibrator,
                )
                split_y = split_result[0] if split_result else None
                split_confidence = split_result[1] if split_result else None
                precomputed_teams = split_result[2] if split_result else None

                if split_y is not None and split_confidence == "high":
                    team_assignments = classify_teams(
                        positions, split_y,
                        precomputed_assignments=precomputed_teams,
                    )
                elif precomputed_teams and len(set(precomputed_teams.values())) >= 2:
                    # Bbox-size ranking produced a valid 2-team split even
                    # though the Y-based split_y is unreliable.  Use the
                    # size-based assignments directly (they don't depend
                    # on split_y).
                    team_assignments = dict(precomputed_teams)
                    logger.info(
                        "Using bbox-size team assignments despite low "
                        "split confidence (%d tracks)",
                        len(team_assignments),
                    )

                # Step 4c: Global identity optimization
                # Splits tracks at interaction boundaries and reassigns
                # segments to canonical players via greedy cost minimization
                if (
                    not skip_global_identity
                    and color_store is not None
                    and color_store.has_data()
                    and team_assignments
                ):
                    from rallycut.tracking.global_identity import (
                        optimize_global_identity,
                    )

                    # Snapshot track IDs to sync color_store after
                    pre_global = {
                        (p.frame_number, id(p)): p.track_id
                        for p in positions
                    }

                    positions, global_result = optimize_global_identity(
                        positions,
                        team_assignments,
                        color_store,
                        court_split_y=split_y,
                        appearance_store=appearance_store,
                    )
                    num_global_segments = global_result.num_segments
                    num_global_remapped = global_result.num_remapped

                    # Sync color_store and appearance_store with any ID changes
                    if global_result.num_remapped > 0:
                        remap_keys: dict[tuple[int, int], int] = {}
                        for p in positions:
                            key = (p.frame_number, id(p))
                            old_tid = pre_global.get(key)
                            if old_tid is not None and old_tid != p.track_id:
                                remap_keys[(old_tid, p.frame_number)] = p.track_id
                        if remap_keys:
                            if color_store is not None:
                                color_store.remap_per_frame(remap_keys)
                            if appearance_store is not None:
                                appearance_store.remap_per_frame(remap_keys)
                    if not global_result.skipped:
                        logger.info(
                            f"Global identity: {global_result.num_segments} "
                            f"segments, {global_result.num_remapped} remapped, "
                            f"{global_result.num_interactions} interactions"
                        )
                    else:
                        logger.debug(
                            f"Global identity skipped: "
                            f"{global_result.skip_reason}"
                        )

                # Step 4d: Convergence-anchored swap detection
                # Checks each net interaction for cross-team ID swaps
                # using court-side, bbox size, and appearance signals.
                if len(primary_track_ids) >= 4:
                    from rallycut.tracking.convergence_swap import (
                        detect_convergence_swaps,
                    )

                    positions, num_convergence_swaps = (
                        detect_convergence_swaps(
                            positions,
                            primary_track_ids,
                            color_store=color_store,
                            upstream_split_y=split_y,
                            upstream_teams=team_assignments,
                        )
                    )

            # Step 4e: Post-identity spatial consistency.
            # Runs both jump detection and drift detection. Drift is
            # deferred to here (not step 0) so identity modules work with
            # full continuous tracks. Jump splits catch residual Kalman
            # artifacts from identity remapping. Drift splits catch smooth
            # ID swaps that no identity module resolved.
            if filter_enabled:
                positions, final_consistency = enforce_spatial_consistency(
                    positions,
                    color_store=color_store,
                    appearance_store=appearance_store,
                    video_fps=fps,
                )
                final_total = (
                    final_consistency.jump_splits + final_consistency.drift_splits
                )
                if final_total > 0:
                    num_jump_splits += final_total
                    logger.info(
                        f"Post-identity spatial consistency: "
                        f"{final_consistency.jump_splits} jump(s), "
                        f"{final_consistency.drift_splits} drift(s)"
                    )

            # Step 4f: Spatial re-link after drift detection.
            # Drift detection (4e) can split a fast-moving player's track
            # into two fragments at nearly the same position. Re-link them
            # if the gap is tiny and endpoints are near-identical.
            if filter_enabled and color_store is not None:
                from rallycut.tracking.tracklet_link import relink_spatial_splits

                positions, num_post_relinks = relink_spatial_splits(
                    positions, color_store,
                    appearance_store=appearance_store,
                )
                if num_post_relinks > 0:
                    logger.info(
                        f"Post-drift spatial re-link: {num_post_relinks} re-link(s)"
                    )

            # Step 5: Interpolate detection gaps for primary tracks
            num_interpolated = 0
            if filter_enabled and primary_track_ids:
                from rallycut.tracking.player_filter import interpolate_player_gaps

                positions, num_interpolated = interpolate_player_gaps(
                    positions,
                    primary_track_ids,
                    config=filter_config,
                )

            # Step 6: Compute quality report
            quality_report = None
            if filter_enabled:
                from rallycut.tracking.quality_report import compute_quality_report

                ball_det_rate = 0.0
                ball_xy: list[tuple[float, float]] | None = None
                if ball_positions:
                    confident_balls = [
                        bp for bp in ball_positions
                        if bp.confidence >= 0.3
                        and not (bp.x == 0.0 and bp.y == 0.0)
                    ]
                    ball_det_rate = min(
                        len(confident_balls) / max(total_frames_in_range, 1), 1.0
                    )
                    ball_xy = [(bp.x, bp.y) for bp in confident_balls]

                quality_report = compute_quality_report(
                    positions=positions,
                    raw_positions=raw_positions,
                    frame_count=total_frames_in_range,
                    video_fps=fps,
                    primary_track_ids=primary_track_ids,
                    ball_detection_rate=ball_det_rate,
                    ball_positions_xy=ball_xy,
                    id_switch_count=num_jump_splits,
                    color_split_count=num_color_splits,
                    height_swap_count=num_height_swaps,
                    appearance_link_count=num_appearance_links,
                    has_court_calibration=court_calibrator is not None,
                    court_detection_insights=court_detection_insights,
                    stationary_bg_removed_count=len(removed_bg_tracks),
                    global_identity_segments=num_global_segments,
                    global_identity_remapped=num_global_remapped,
                    convergence_swaps_fixed=num_convergence_swaps,
                    team_classification_skipped=split_confidence != "high",
                    interpolated_position_count=num_interpolated,
                )

            processing_time_ms = (time.time() - start_time) * 1000
            effective_fps = frames_processed / (processing_time_ms / 1000) if processing_time_ms > 0 else 0
            logger.info(
                f"Completed tracking {frames_processed} frames in "
                f"{processing_time_ms/1000:.1f}s ({effective_fps:.1f} FPS)"
            )

            # Normalize frame numbers to rally-relative (0-based).
            # When tracking a segment of the full video (start_ms provided),
            # frame_idx is absolute. Frontend and match_tracker expect 0-based.
            if start_frame > 0:
                for p in positions:
                    p.frame_number -= start_frame
                if raw_positions:
                    for p in raw_positions:
                        p.frame_number -= start_frame
                if ball_positions:
                    for bp in ball_positions:
                        bp.frame_number -= start_frame
                if color_store is not None and color_store.has_data():
                    color_store.shift_frames(-start_frame)
                if appearance_store is not None and appearance_store.has_data():
                    appearance_store.shift_frames(-start_frame)

            return PlayerTrackingResult(
                positions=positions,
                frame_count=total_frames_in_range,  # Total frames in video range (for time mapping)
                video_fps=fps,
                video_width=video_width,
                video_height=video_height,
                processing_time_ms=processing_time_ms,
                model_version=MODEL_NAME,
                court_split_y=court_split_y,
                primary_track_ids=primary_track_ids,
                filter_method=filter_method,
                raw_positions=raw_positions,
                quality_report=quality_report,
                team_assignments=team_assignments,
                color_store=color_store,
                appearance_store=appearance_store,
            )

        finally:
            cap.release()
            # Reset tracker state for next video
            if use_boxmot:
                self._reset_boxmot_tracker()
            elif hasattr(model, "predictor") and model.predictor is not None:
                model.predictor.trackers = []

    def track_frames(
        self,
        frames: Iterator[np.ndarray],
        video_fps: float,
        video_width: int,
        video_height: int,
    ) -> Iterator[list[PlayerPosition]]:
        """
        Track players in a stream of frames.

        Args:
            frames: Iterator of BGR frames.
            video_fps: Video frame rate.
            video_width: Frame width.
            video_height: Frame height.

        Yields:
            List of PlayerPosition for each frame.
        """
        model = self._load_model()
        use_boxmot = self.tracker == TRACKER_BOXMOT_BOTSORT
        frame_idx = 0

        if use_boxmot:
            if self._reid_model is None:
                self._reid_model = self._load_reid_model()
            self._boxmot_tracker = self._init_boxmot_tracker(video_fps)

        for frame in frames:
            try:
                # Apply preprocessing if enabled (e.g., CLAHE for sand backgrounds)
                processed_frame = frame
                if self.preprocessing == PREPROCESSING_CLAHE:
                    processed_frame = apply_clahe_preprocessing(frame)

                # Apply ROI mask to prevent tracks on background objects
                frame_to_track = processed_frame
                if self.court_roi is not None:
                    frame_to_track = self._apply_roi_mask(
                        processed_frame, self.court_roi
                    )

                if use_boxmot:
                    # BoxMOT path: YOLO detect → extract ReID → BoxMOT track
                    results = model.predict(
                        frame_to_track,
                        conf=self.confidence,
                        iou=self.iou,
                        imgsz=self.imgsz,
                        classes=[PERSON_CLASS_ID],
                        verbose=False,
                    )
                    dets_array = np.empty((0, 6), dtype=np.float32)
                    if results and len(results) > 0 and results[0].boxes is not None:
                        boxes = results[0].boxes
                        if len(boxes.xyxy) > 0:
                            xyxy = boxes.xyxy.cpu().numpy()
                            conf = boxes.conf.cpu().numpy().reshape(-1, 1)
                            cls = boxes.cls.cpu().numpy().reshape(-1, 1)
                            dets_array = np.hstack([xyxy, conf, cls]).astype(np.float32)
                    embs = self._extract_reid_embeddings(frame_to_track, dets_array)
                    tracks = self._boxmot_tracker.update(dets_array, frame_to_track, embs)
                    frame_positions = self._decode_boxmot_results(
                        tracks, frame_idx, video_width, video_height
                    )
                else:
                    # Ultralytics path: YOLO detect + track in one call
                    results = model.track(
                        frame_to_track,
                        persist=True,
                        tracker=str(self._get_tracker_config()),
                        conf=self.confidence,
                        iou=self.iou,
                        imgsz=self.imgsz,
                        classes=[PERSON_CLASS_ID],
                        verbose=False,
                    )
                    frame_positions = self._decode_results(
                        results, frame_idx, video_width, video_height
                    )

                # Sport-aware detection filters (only when ROI masking is active)
                if self.court_roi is not None:
                    frame_positions = self._filter_detections(frame_positions)
            except (IndexError, RuntimeError, ValueError) as e:
                # Handle any errors from YOLO/ByteTrack internals
                logger.debug(f"Frame {frame_idx} tracking failed: {e}")
                frame_positions = []

            yield frame_positions
            frame_idx += 1

        # Reset tracker state
        if use_boxmot:
            self._reset_boxmot_tracker()
        elif hasattr(model, "predictor") and model.predictor is not None:
            model.predictor.trackers = []
