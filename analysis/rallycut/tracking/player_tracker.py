"""
Player tracking using YOLOv8 + BoT-SORT for volleyball videos.

Uses YOLOv8 for person detection with BoT-SORT for temporal tracking.
BoT-SORT adds camera motion compensation on top of ByteTrack, reducing ID switches.
Default model is yolov8n (nano) for best speed/accuracy tradeoff (~23 FPS, 88% F1).
Use --yolo-model yolov8m for best accuracy at 3x slower speed.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

if TYPE_CHECKING:
    from rallycut.court.calibration import CourtCalibrator
    from rallycut.tracking.ball_tracker import BallPosition
    from rallycut.tracking.court_identity import SwapDecision
    from rallycut.tracking.player_filter import PlayerFilterConfig
    from rallycut.tracking.quality_report import TrackingQualityReport

logger = logging.getLogger(__name__)

# Model configuration
MODEL_NAME = "yolov8n.pt"  # YOLOv8 nano - fastest with good accuracy (88% F1)
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
) -> tuple[CourtCalibrator | None, Any]:
    """Auto-detect court corners from video and create calibrator if confident enough.

    Safe to call on any video — returns (None, result) on failure without raising.

    Args:
        video_path: Path to the video file.
        confidence_threshold: Minimum confidence to accept detection.

    Returns:
        Tuple of (calibrator, result):
        - calibrator: CourtCalibrator if detection succeeded, None otherwise.
        - result: CourtDetectionResult with detection details (or error info).
    """
    from rallycut.court.calibration import CourtCalibrator
    from rallycut.court.detector import CourtDetectionResult, CourtDetector

    try:
        detector = CourtDetector()
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
        return calibrator, result

    return None, result


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
# Benchmark on beach volleyball (8.8s rally):
#   yolov8n: 23 FPS, 88.0% F1, 82.4% recall (default - best speed/accuracy)
#   yolov8s: 15 FPS, 83.8% F1, 78.7% recall
#   yolov8m:  7 FPS, 89.2% F1, 86.4% recall (best accuracy, 3x slower)
#   yolov8l:  5 FPS, 88.4% F1, 85.5% recall (no benefit over medium)
YOLO_MODELS = {
    "yolov8n": "yolov8n.pt",  # Nano: 3.2M params, fastest (default)
    "yolov8s": "yolov8s.pt",  # Small: 11.2M params
    "yolov8m": "yolov8m.pt",  # Medium: 25.9M params, best accuracy
    "yolov8l": "yolov8l.pt",  # Large: 43.7M params
    "yolo11n": "yolo11n.pt",  # YOLO11 Nano: 2.6M params
    "yolo11s": "yolo11s.pt",  # YOLO11 Small: 9.4M params
    "yolo11m": "yolo11m.pt",  # YOLO11 Medium: 20.1M params
    "yolo11l": "yolo11l.pt",  # YOLO11 Large: 25.3M params
}
DEFAULT_YOLO_MODEL = "yolov8n"

# Tracker configs for better tracking stability
BYTETRACK_CONFIG = Path(__file__).parent / "bytetrack_volleyball.yaml"
BOTSORT_CONFIG = Path(__file__).parent / "botsort_volleyball.yaml"

# Available trackers
TRACKER_BYTETRACK = "bytetrack"
TRACKER_BOTSORT = "botsort"
DEFAULT_TRACKER = TRACKER_BOTSORT  # BoT-SORT reduces ID switches by 64%

# Preprocessing options
PREPROCESSING_NONE = "none"
PREPROCESSING_CLAHE = "clahe"  # Contrast Limited Adaptive Histogram Equalization


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

    # Uncertain identity windows from court-plane resolution
    # Each tuple: (start_frame, end_frame, set of affected track IDs)
    uncertain_identity_windows: list[tuple[int, int, set[int]]] = field(
        default_factory=list
    )

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
            yolo_model: YOLO model size (default: yolov8n). Options:
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
        self._model: Any = None
        self._custom_tracker_config: Path | None = None

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

        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package is required for player tracking. "
                "Install with: pip install ultralytics>=8.2.0"
            )

        # Load model - ultralytics handles download automatically
        model_path = self._ensure_model()
        model_filename = self._get_model_filename()

        # Try to load from cache first, fallback to auto-download
        if model_path.exists():
            self._model = YOLO(str(model_path))
        else:
            # Download model automatically
            logger.info(f"Downloading {model_filename} model...")
            self._model = YOLO(model_filename)

        # Configure for CPU/GPU
        # ultralytics auto-detects available hardware
        logger.info(f"Loaded YOLO model: {model_filename}")

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

    def _filter_off_court(
        self,
        positions: list[PlayerPosition],
        court_calibrator: CourtCalibrator,
        sideline_margin: float = 2.0,
        baseline_margin: float = 4.0,
    ) -> list[PlayerPosition]:
        """Filter out detections whose feet project outside court bounds.

        Uses court calibration to project each detection's feet position
        (bbox bottom-center) to court coordinates and checks bounds.
        This removes spectators, cameramen, and other off-court people
        before they can confuse the tracker or post-processing.

        Args:
            positions: Decoded player positions for a single frame.
            court_calibrator: Calibrated court calibrator with homography.
            sideline_margin: Court sideline margin in meters.
            baseline_margin: Court baseline margin in meters (larger for serves).

        Returns:
            Filtered positions (only on-court detections).
        """
        if not positions:
            return positions

        filtered: list[PlayerPosition] = []
        for p in positions:
            # Use feet position (bbox bottom-center) for court projection
            feet_x = p.x
            feet_y = p.y + p.height / 2

            try:
                # (1, 1) for image dimensions since coordinates are already normalized 0-1
                court_point = court_calibrator.image_to_court(
                    (feet_x, feet_y), 1, 1
                )
                if court_calibrator.is_point_in_court_with_margin(
                    court_point,
                    sideline_margin=sideline_margin,
                    baseline_margin=baseline_margin,
                ):
                    filtered.append(p)
            except (RuntimeError, ValueError):
                # Keep detection if projection fails (edge case)
                filtered.append(p)

        if len(filtered) < len(positions):
            logger.debug(
                f"Court filter: {len(positions)} -> {len(filtered)} "
                f"(removed {len(positions) - len(filtered)} off-court detections)"
            )

        return filtered

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
            court_calibrator: Optional calibrated court calibrator. When provided,
                            detections outside court bounds are filtered immediately
                            after YOLO inference, preventing off-court tracks.

        Returns:
            PlayerTrackingResult with all detected positions.
        """
        import time

        start_time = time.time()
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        # Load YOLO model
        model = self._load_model()

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
            logger.info(
                f"Processing frames {start_frame}-{end_frame} "
                f"({total_frames_in_range} frames, stride={stride}, processing {frames_to_process} frames, {fps:.1f} fps)"
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
                ret, frame = cap.read()
                if not ret:
                    logger.warning(
                        f"Frame read failed at index {frame_idx} "
                        f"(expected {total_frames_in_range} frames)"
                    )
                    break

                # Only process every Nth frame (stride)
                if (frame_idx - start_frame) % stride == 0:
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

                    # Run YOLO with ByteTrack
                    # persist=True enables tracking across frames
                    try:
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

                        # Filter off-court detections if calibrator available
                        if court_calibrator is not None and court_calibrator.is_calibrated:
                            frame_positions = self._filter_off_court(
                                frame_positions, court_calibrator
                            )

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

                    # Progress callback
                    if progress_callback and frames_processed % 30 == 0:
                        progress = frames_processed / frames_to_process
                        progress_callback(progress)

                frame_idx += 1

            # Final progress
            if progress_callback:
                progress_callback(1.0)

            # Court split Y for debug overlay (horizontal line)
            court_split_y: float | None = None
            primary_track_ids: list[int] = []
            filter_method: str | None = None
            uncertain_windows: list[tuple[int, int, set[int]]] = []
            court_decisions: list[SwapDecision] = []

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
            num_color_splits = 0
            num_swap_fixes = 0
            num_appearance_links = 0
            num_court_swaps = 0

            # Team classification (populated during filtering)
            team_assignments: dict[int, int] = {}

            if filter_enabled:
                from rallycut.tracking.player_filter import (
                    PlayerFilter,
                    PlayerFilterConfig,
                    classify_teams,
                    compute_court_split,
                    split_tracks_at_jumps,
                    stabilize_track_ids,
                )

                # Get config (or create default)
                config = filter_config or PlayerFilterConfig()

                # Step 0: Split tracks at large position jumps (detects ID switches)
                split_info: list[tuple[int, int, int]] = []
                positions, num_jump_splits = split_tracks_at_jumps(
                    positions, split_info_out=split_info
                )
                # Rekey stores for jump splits
                if color_store is not None:
                    for old_id, new_id, split_frame in split_info:
                        color_store.rekey(old_id, new_id, split_frame)
                if appearance_store is not None:
                    for old_id, new_id, split_frame in split_info:
                        appearance_store.rekey(old_id, new_id, split_frame)

                # Step 0b: Color-based track splitting
                if color_store is not None and color_store.has_data():
                    from rallycut.tracking.color_repair import (
                        detect_and_fix_swaps,
                        split_tracks_by_color,
                    )

                    positions, num_color_splits = split_tracks_by_color(
                        positions, color_store
                    )

                    # Classify teams using early-frame Y positions
                    preliminary_split_y = compute_court_split(
                        ball_positions or [], config, player_positions=positions
                    )
                    if preliminary_split_y is not None:
                        team_assignments = classify_teams(
                            positions, preliminary_split_y
                        )

                    # Step 0c: Court-plane identity resolution (when calibrated)
                    if (
                        court_calibrator is not None
                        and court_calibrator.is_calibrated
                        and team_assignments
                    ):
                        from rallycut.tracking.court_identity import (
                            resolve_court_identity,
                        )

                        positions, num_court_swaps, court_decisions = (
                            resolve_court_identity(
                                positions,
                                team_assignments,
                                court_calibrator,
                                video_width=video_width,
                                video_height=video_height,
                            )
                        )

                    # Collect uncertain identity windows from court decisions
                    for decision in court_decisions:
                        if not decision.confident:
                            uncertain_windows.append((
                                decision.interaction.start_frame,
                                decision.interaction.end_frame,
                                {decision.interaction.track_a,
                                 decision.interaction.track_b},
                            ))

                    # Step 0d: Color-based swap detection (fallback/complement)
                    positions, num_swap_fixes = detect_and_fix_swaps(
                        positions, color_store,
                        team_assignments=team_assignments,
                    )
                    # Note: court swaps tracked separately in num_court_swaps

                    # Step 0e: Appearance-based tracklet linking (GTA-Link inspired)
                    # Reconnects fragments using color histogram similarity
                    from rallycut.tracking.tracklet_link import (
                        link_tracklets_by_appearance,
                    )

                    positions, num_appearance_links = link_tracklets_by_appearance(
                        positions, color_store,
                        team_assignments=team_assignments,
                        appearance_store=appearance_store,
                    )

                # Step 1: Stabilize track IDs before filtering
                # This merges tracks that represent the same player
                positions, id_mapping = stabilize_track_ids(
                    positions, config, team_assignments=team_assignments
                )

                # Remap team assignments after track merging
                # id_mapping: merged_id -> canonical_id
                if team_assignments and id_mapping:
                    for merged_id in id_mapping:
                        team_assignments.pop(merged_id, None)

                player_filter = PlayerFilter(
                    ball_positions=ball_positions,
                    total_frames=total_frames_in_range,
                    config=config,
                    court_calibrator=court_calibrator,
                )

                # Step 2: Analyze all positions to identify stable tracks
                # This must be done before per-frame filtering
                player_filter.analyze_tracks(positions)

                # Capture court split Y for debug overlay
                court_split_y = player_filter.court_split_y
                primary_track_ids = sorted(player_filter.primary_tracks)

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

                # Step 4b: Post-filter court identity resolution
                # Pre-filter court identity (Step 0c) runs on noisy data with many
                # short tracks, making team separation appear smaller than it is.
                # Re-running on clean 4-player data catches swaps that were missed.
                if (
                    court_calibrator is not None
                    and court_calibrator.is_calibrated
                    and len(primary_track_ids) >= 2
                ):
                    from rallycut.tracking.court_identity import (
                        resolve_court_identity,
                    )

                    # Recompute team assignments from clean filtered positions
                    post_split_y = compute_court_split(
                        ball_positions or [], config,
                        player_positions=positions,
                    )
                    if post_split_y is not None:
                        post_team_assignments = classify_teams(
                            positions, post_split_y
                        )
                        if post_team_assignments:
                            positions, post_swaps, post_decisions = (
                                resolve_court_identity(
                                    positions,
                                    post_team_assignments,
                                    court_calibrator,
                                    video_width=video_width,
                                    video_height=video_height,
                                )
                            )
                            if post_swaps > 0:
                                num_court_swaps += post_swaps
                                team_assignments = post_team_assignments
                                logger.info(
                                    f"Post-filter court identity: "
                                    f"{post_swaps} additional swaps"
                                )
                            court_decisions.extend(post_decisions)
                            # Collect uncertain windows, deduplicating
                            # with Step 0c (same interactions may be
                            # re-detected on filtered data)
                            existing = {
                                (s, e) for s, e, _ in uncertain_windows
                            }
                            for d in post_decisions:
                                if not d.confident:
                                    key = (
                                        d.interaction.start_frame,
                                        d.interaction.end_frame,
                                    )
                                    if key not in existing:
                                        uncertain_windows.append((
                                            d.interaction.start_frame,
                                            d.interaction.end_frame,
                                            {d.interaction.track_a,
                                             d.interaction.track_b},
                                        ))

            # Step 5: Compute quality report
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
                    swap_fix_count=num_swap_fixes,
                    appearance_link_count=num_appearance_links,
                    has_court_calibration=court_calibrator is not None,
                    court_identity_interactions=len(court_decisions),
                    court_identity_swaps=num_court_swaps,
                    uncertain_identity_count=len(uncertain_windows),
                )

            processing_time_ms = (time.time() - start_time) * 1000
            effective_fps = frames_processed / (processing_time_ms / 1000) if processing_time_ms > 0 else 0
            logger.info(
                f"Completed tracking {frames_processed} frames in "
                f"{processing_time_ms/1000:.1f}s ({effective_fps:.1f} FPS)"
            )

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
                uncertain_identity_windows=uncertain_windows,
            )

        finally:
            cap.release()
            # Reset tracker state for next video
            if hasattr(model, "predictor") and model.predictor is not None:
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
        frame_idx = 0

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

                # Run YOLO with ByteTrack
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
        if hasattr(model, "predictor") and model.predictor is not None:
            model.predictor.trackers = []
