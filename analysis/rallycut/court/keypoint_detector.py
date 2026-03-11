"""Court keypoint detection using YOLO11s-pose.

Detects 6 court keypoints from video frames using a fine-tuned YOLO-pose model:
  - 4 corners: near-left, near-right, far-right, far-left
  - 2 center points: center-left, center-right (net-sideline intersections)

Handles off-screen near corners via bottom padding (same as training).
Center points are always visible in-frame and enable near-corner refinement
via sideline projection (keeping raw Y, correcting X onto the sideline).

Keypoint model is tried first by CourtDetector; falls back to classical
pipeline if confidence is too low.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rallycut.court.calibration import COURT_LENGTH, COURT_WIDTH
from rallycut.court.detector import CourtDetectionResult
from rallycut.court.line_geometry import harmonic_conjugate, line_intersection, point_line_distance

logger = logging.getLogger(__name__)

CORNER_NAMES = ["near-left", "near-right", "far-right", "far-left"]
CENTER_NAMES = ["center-left", "center-right"]


@dataclass
class CourtQualityDiagnostics:
    """Quality diagnostics for court keypoint detection.

    Helps identify problematic videos where court detection may be unreliable.
    """

    detection_rate: float  # % of sampled frames where court was detected
    per_corner_confidence: dict[str, float]  # mean keypoint confidence per corner
    per_corner_std: dict[str, float]  # std dev of position per corner (frame-to-frame)
    off_screen_corners: list[str]  # corner names where y > 1.0 (below frame)
    perspective_ratio: float  # near_width / far_width (>3 = extreme perspective)
    warnings: list[str]  # actionable quality warnings
    center_points: list[dict[str, float]] | None = None  # aggregated center-left/right
    center_confidences: dict[str, float] | None = None  # mean confidence per center point

    def to_dict(self) -> dict[str, Any]:
        return {
            "detection_rate": round(self.detection_rate, 3),
            "per_corner_confidence": {
                k: round(v, 3) for k, v in self.per_corner_confidence.items()
            },
            "per_corner_std": {
                k: round(v, 4) for k, v in self.per_corner_std.items()
            },
            "off_screen_corners": self.off_screen_corners,
            "perspective_ratio": round(self.perspective_ratio, 2),
            "warnings": self.warnings,
            "center_points": [
                {k: round(v, 6) for k, v in pt.items()} for pt in self.center_points
            ] if self.center_points else None,
            "center_confidences": {
                k: round(v, 3) for k, v in self.center_confidences.items()
            } if self.center_confidences else None,
        }

DEFAULT_MODEL_PATH = (
    Path(__file__).parent.parent.parent / "weights" / "court_keypoint" / "court_keypoint_best.pt"
)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """Compute weighted median.

    For each value, its weight determines how much it contributes.
    The weighted median is the value where cumulative weight reaches 50%.
    Falls back to unweighted median if all weights are zero.
    """
    if len(values) == 0:
        return 0.0
    total = weights.sum()
    if total <= 0:
        return float(np.median(values))

    # Sort by value
    order = np.argsort(values)
    sorted_vals = values[order]
    sorted_weights = weights[order]

    # Cumulative weight
    cumsum = np.cumsum(sorted_weights)
    cutoff = total / 2.0

    # Find first index where cumulative weight >= 50%
    idx = int(np.searchsorted(cumsum, cutoff))
    idx = min(idx, len(sorted_vals) - 1)
    return float(sorted_vals[idx])


@dataclass
class FrameKeypoints:
    """Keypoint detection result for a single frame."""

    corners: list[dict[str, float]]  # 4 corners [{x, y}] in original (unpadded) coords
    confidence: float  # bbox confidence
    kpt_confidences: list[float]  # per-corner visibility/confidence (4 values)
    center_points: list[dict[str, float]] | None = None  # 2 center points [{x, y}] (6-kpt model)
    center_confidences: list[float] | None = None  # per-center-point confidence (2 values)


class CourtKeypointDetector:
    """Court corner detection using YOLO11s-pose keypoints.

    Pads frames at the bottom (matching training) to handle off-screen
    near corners, runs YOLO inference, and un-pads Y coordinates.
    Multi-frame aggregation uses median with outlier filtering.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        pad_ratio: float = 0.3,
        conf_threshold: float = 0.3,
    ) -> None:
        self._model_path = Path(model_path) if model_path else DEFAULT_MODEL_PATH
        self._pad_ratio = pad_ratio
        self._conf_threshold = conf_threshold
        self._model: Any = None
        self._last_diagnostics: CourtQualityDiagnostics | None = None

    @property
    def model_exists(self) -> bool:
        """Check if the model weights file exists."""
        return self._model_path.exists()

    def _load_model(self) -> Any:
        """Lazy-load the YOLO model."""
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self._model_path))
            logger.info(f"Loaded court keypoint model from {self._model_path}")
        return self._model

    def detect(
        self,
        video_path: str | Path,
        n_frames: int = 30,
    ) -> CourtDetectionResult:
        """Detect court corners from a video using keypoint model.

        Samples n_frames evenly, runs per-frame inference with padding,
        and aggregates via outlier-filtered median.

        Args:
            video_path: Path to the video file.
            n_frames: Number of frames to sample.

        Returns:
            CourtDetectionResult with corners, confidence, and fitting_method="keypoint".
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        frames = self._sample_frames(video_path, n_frames)
        if not frames:
            return CourtDetectionResult(
                corners=[], confidence=0.0,
                warnings=["Could not sample frames from video"],
                fitting_method="keypoint",
            )

        # Run per-frame detection
        frame_results: list[FrameKeypoints] = []
        for frame in frames:
            result = self._detect_frame(frame)
            if result is not None:
                frame_results.append(result)

        # Keep a representative frame for sideline detection
        mid_frame = frames[len(frames) // 2] if frames else None

        n_sampled = len(frames)

        if not frame_results:
            return CourtDetectionResult(
                corners=[], confidence=0.0,
                warnings=["No court detected in any frame"],
                fitting_method="keypoint",
            )

        # Aggregate across frames
        corners, confidence, diagnostics = self._aggregate(
            frame_results, n_sampled=n_sampled,
        )

        # Refine low-confidence near corners via perspective extrapolation
        corners, vp_refined = self._refine_near_corners(
            corners, diagnostics.per_corner_confidence,
            frame=mid_frame,
            center_points=diagnostics.center_points,
            center_confidences=diagnostics.center_confidences,
        )

        # Penalize confidence for unreliable corners. The bbox confidence
        # (0.91) only means "a court exists" — it says nothing about corner
        # localization. When near corners have ~0.001 keypoint confidence,
        # the overall score must reflect that, otherwise consumers (e.g.
        # quality service auto-save at 0.7) silently store bad calibrations.
        # VP-refined corners get partial credit since the extrapolation is
        # geometrically grounded.
        confidence = self._penalize_confidence(
            confidence, diagnostics.per_corner_confidence, vp_refined,
        )

        return CourtDetectionResult(
            corners=corners,
            confidence=confidence,
            warnings=diagnostics.warnings,
            fitting_method="keypoint",
            per_corner_confidence=diagnostics.per_corner_confidence,
        )

    @property
    def last_diagnostics(self) -> CourtQualityDiagnostics | None:
        """Quality diagnostics from the most recent detect() call."""
        return self._last_diagnostics

    def detect_from_frame(self, frame: np.ndarray) -> CourtDetectionResult:
        """Detect court corners from a single frame.

        Args:
            frame: BGR image (numpy array).

        Returns:
            CourtDetectionResult from single-frame inference.
        """
        result = self._detect_frame(frame)
        if result is None:
            return CourtDetectionResult(
                corners=[], confidence=0.0,
                warnings=["No court detected in frame"],
                fitting_method="keypoint",
            )

        # Refine low-confidence near corners via perspective extrapolation
        per_corner_conf = dict(zip(CORNER_NAMES, result.kpt_confidences))
        center_pts = result.center_points
        center_confs: dict[str, float] | None = None
        if result.center_confidences is not None:
            center_confs = dict(zip(CENTER_NAMES, result.center_confidences))
        corners, vp_refined = self._refine_near_corners(
            result.corners, per_corner_conf, frame=frame,
            center_points=center_pts,
            center_confidences=center_confs,
        )
        confidence = self._penalize_confidence(result.confidence, per_corner_conf, vp_refined)

        return CourtDetectionResult(
            corners=corners,
            confidence=confidence,
            fitting_method="keypoint",
            per_corner_confidence=per_corner_conf,
        )

    def _sample_frames(
        self, video_path: Path, n_frames: int,
    ) -> list[np.ndarray]:
        """Sample evenly-spaced frames from a video."""
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []

        # Evenly-spaced indices (skip first/last 2 seconds if possible)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        skip = int(2.0 * fps)
        start = min(skip, total_frames // 4)
        end = max(start + 1, total_frames - skip)
        step = max(1, (end - start) // n_frames)

        indices = list(range(start, end, step))[:n_frames]

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        cap.release()
        return frames

    def _detect_frame(self, frame: np.ndarray) -> FrameKeypoints | None:
        """Run keypoint detection on a single frame.

        Pads bottom, runs YOLO, un-pads Y coordinates.

        Returns:
            FrameKeypoints or None if no detection.
        """
        model = self._load_model()

        # Pad bottom (same as training)
        h, w = frame.shape[:2]
        pad_h = int(h * self._pad_ratio)
        padding = np.zeros((pad_h, w, 3), dtype=frame.dtype)
        padded = np.vstack([frame, padding])

        # Run inference
        results = model.predict(
            padded,
            conf=self._conf_threshold,
            verbose=False,
            device="cpu",  # Small model, CPU is fine for inference
        )

        if not results:
            return None

        result = results[0]

        # Get best detection (highest confidence)
        if result.boxes is None or len(result.boxes) == 0:
            return None
        if result.keypoints is None or len(result.keypoints) == 0:
            return None

        # Pick detection with highest box confidence
        confs = result.boxes.conf.cpu().numpy()
        best_idx = int(confs.argmax())
        best_conf = float(confs[best_idx])

        # Extract keypoints: shape (n_kpts, 3) = (x, y, conf) in pixel coords
        kpts = result.keypoints[best_idx]
        kpt_xy = kpts.xy.cpu().numpy()[0]  # (n_kpts, 2)
        kpt_conf = kpts.conf.cpu().numpy()[0] if kpts.conf is not None else np.ones(len(kpt_xy))

        n_kpts = len(kpt_xy)
        if n_kpts not in (4, 6):
            logger.warning(f"Expected 4 or 6 keypoints, got {n_kpts}")
            return None

        padded_h, padded_w = padded.shape[:2]

        # Convert to normalized coords and un-pad Y
        # In padded image: y_padded_norm = y_pixel / padded_h
        # Original Y: y_orig_norm = y_padded_norm * (1 + pad_ratio)
        # Because during export: y_padded = y_orig / (1 + pad_ratio)
        corners = []
        kpt_confidences = []
        for i in range(4):
            x_px, y_px = float(kpt_xy[i][0]), float(kpt_xy[i][1])
            kc = float(kpt_conf[i])

            x_norm = x_px / padded_w
            y_norm = y_px / padded_h
            y_orig = y_norm * (1.0 + self._pad_ratio)

            corners.append({"x": round(x_norm, 6), "y": round(y_orig, 6)})
            kpt_confidences.append(kc)

        # Extract center points from 6-keypoint model
        center_points: list[dict[str, float]] | None = None
        center_confidences: list[float] | None = None
        if n_kpts == 6:
            center_points = []
            center_confidences = []
            for i in range(4, 6):
                x_px, y_px = float(kpt_xy[i][0]), float(kpt_xy[i][1])
                kc = float(kpt_conf[i])

                x_norm = x_px / padded_w
                y_norm = y_px / padded_h
                y_orig = y_norm * (1.0 + self._pad_ratio)

                center_points.append({"x": round(x_norm, 6), "y": round(y_orig, 6)})
                center_confidences.append(kc)

        return FrameKeypoints(
            corners=corners,
            confidence=best_conf,
            kpt_confidences=kpt_confidences,
            center_points=center_points,
            center_confidences=center_confidences,
        )

    # Max extrapolation beyond [0, 1] range for near corners.
    # Caps overshoot: near corners clamped to [-margin, 1+margin].
    # 0.20 reduces regressions from 11→5 while keeping 33% MCD improvement.
    NEAR_CORNER_MAX_MARGIN = 0.20

    # Minimum confidence to trust YOLO center points for refinement
    _CENTER_POINT_MIN_CONF = 0.3

    def _refine_near_corners(
        self,
        corners: list[dict[str, float]],
        per_corner_confidence: dict[str, float],
        conf_threshold: float = 0.5,
        frame: np.ndarray | None = None,
        center_points: list[dict[str, float]] | None = None,
        center_confidences: dict[str, float] | None = None,
    ) -> tuple[list[dict[str, float]], set[str]]:
        """Replace low-confidence near corners using best available geometry.

        Strategy priority:
        1. **YOLO center points** (6-keypoint model): Use model-predicted
           net-sideline intersections for sideline projection refinement.
           Always visible in-frame, no Hough detection needed.
        2. **Sideline detection from frame** (frame available): Detect sidelines
           via Hough lines, compute VP from actual visible lines.
        3. **VP from model corners** (fallback): Uses the model's near-corner
           guesses for VP computation. Less accurate but always works.

        Args:
            corners: 4 corners [near-left, near-right, far-right, far-left].
            per_corner_confidence: Mean confidence per corner name.
            conf_threshold: Below this, replace the near corner.
            frame: Optional BGR frame for sideline detection.
            center_points: Optional aggregated center-left/right from 6-kpt model.
            center_confidences: Optional confidence per center point name.

        Returns:
            Tuple of (refined corners, set of refined corner names).
        """
        if len(corners) != 4:
            return corners, set()

        nl_conf = per_corner_confidence.get("near-left", 1.0)
        nr_conf = per_corner_confidence.get("near-right", 1.0)

        # Skip if both near corners are confident
        if nl_conf >= conf_threshold and nr_conf >= conf_threshold:
            return corners, set()

        # Strategy 1: Use YOLO center points (6-keypoint model)
        if center_points is not None and len(center_points) == 2:
            cl_conf = (center_confidences or {}).get("center-left", 0.0)
            cr_conf = (center_confidences or {}).get("center-right", 0.0)
            min_conf = self._CENTER_POINT_MIN_CONF
            if cl_conf >= min_conf and cr_conf >= min_conf:
                result = self._refine_via_center_points(
                    corners, per_corner_confidence, conf_threshold,
                    center_points,
                )
                if result is not None:
                    return result

        # Strategy 2: Detect sidelines from the frame image
        if frame is not None:
            result = self._refine_via_sideline_detection(
                corners, per_corner_confidence, conf_threshold, frame,
            )
            if result is not None:
                return result

        # Strategy 3: Fallback — VP from model's near-corner guesses.
        # Guard against degenerate geometry (e.g. near-horizontal sidelines)
        # where VP extrapolation can push near corners far from their raw
        # positions. Keep raw if any refined corner moved > 30% of screen.
        refined, refined_names = self._refine_via_vp_fallback(
            corners, per_corner_confidence, conf_threshold, frame=frame,
        )
        if refined_names:
            max_disp = max(
                math.sqrt(
                    (refined[i]["x"] - corners[i]["x"]) ** 2
                    + (refined[i]["y"] - corners[i]["y"]) ** 2
                )
                for i in range(2)  # near-left, near-right
                if CORNER_NAMES[i] in refined_names
            )
            if max_disp > 0.30:
                logger.warning(
                    "VP fallback displaced near corner by %.2f — keeping raw",
                    max_disp,
                )
                return corners, set()
        return refined, refined_names

    def _refine_via_center_points(
        self,
        corners: list[dict[str, float]],
        per_corner_confidence: dict[str, float],
        conf_threshold: float,
        center_points: list[dict[str, float]],
    ) -> tuple[list[dict[str, float]], set[str]] | None:
        """Refine near corners using YOLO-predicted center points.

        Uses sidelines defined by high-confidence far corners and center
        points to place near corners. For each near corner:

        1. If raw Y is plausible, project onto sideline (keep Y, fix X).
        2. Otherwise, extrapolate via VP + aspect ratio (full replacement).
        """
        center_left = (center_points[0]["x"], center_points[0]["y"])
        center_right = (center_points[1]["x"], center_points[1]["y"])
        far_left = (corners[3]["x"], corners[3]["y"])
        far_right = (corners[2]["x"], corners[2]["y"])

        # Compute VP from center-point sidelines
        vp = line_intersection(far_left, center_left, far_right, center_right)
        if vp is None:
            return None

        far_mid_y = (far_left[1] + far_right[1]) / 2.0
        if vp[1] >= far_mid_y:
            logger.debug("Center-point VP below far baseline — skipping")
            return None

        # VP extrapolation as fallback for off-screen corners
        vp_corners, vp_refined = self._extrapolate_from_vp(
            corners, per_corner_confidence, conf_threshold, vp,
        )

        refined = list(corners)
        refined_names: set[str] = set()

        margin = self.NEAR_CORNER_MAX_MARGIN
        for i, name in [(0, "near-left"), (1, "near-right")]:
            if per_corner_confidence.get(name, 1.0) >= conf_threshold:
                continue

            raw_x = corners[i]["x"]
            raw_y = corners[i]["y"]
            far_pt = far_left if i == 0 else far_right
            ctr_pt = center_left if i == 0 else center_right

            # Sideline projection (keep raw Y, fix X)
            raw_y_plausible = far_pt[1] + 0.05 < raw_y < 1.0 + margin
            if raw_y_plausible:
                dy_sl = ctr_pt[1] - far_pt[1]
                if abs(dy_sl) > 1e-6:
                    t = (raw_y - far_pt[1]) / dy_sl
                    proj_x = far_pt[0] + t * (ctr_pt[0] - far_pt[0])
                    refined[i] = {"x": round(proj_x, 6), "y": round(raw_y, 6)}
                    refined_names.add(name)
                    logger.info(
                        "Sideline-projected %s: (%.3f,%.3f)→(%.3f,%.3f)",
                        name, raw_x, raw_y, proj_x, raw_y,
                    )
                else:
                    if name in vp_refined:
                        refined[i] = vp_corners[i]
                        refined_names.add(name)
            else:
                # Off-screen or implausible Y — use full VP extrapolation
                if name in vp_refined:
                    refined[i] = vp_corners[i]
                    refined_names.add(name)
                    logger.info(
                        "VP-extrapolated %s: (%.3f,%.3f)→(%.3f,%.3f)",
                        name, raw_x, raw_y,
                        vp_corners[i]["x"], vp_corners[i]["y"],
                    )

        if not refined_names:
            return corners, set()

        refined = self._clamp_near_corners(refined)
        if not self._is_convex_quad(refined):
            logger.warning("Center-point refined corners not convex — skipping")
            return None

        return refined, refined_names

    def _detect_sideline_vp(
        self,
        corners: list[dict[str, float]],
        frame: np.ndarray,
        per_corner_confidence: dict[str, float] | None = None,
        conf_threshold: float = 0.5,
    ) -> tuple[float, float] | None:
        """Detect sideline vanishing point from Hough lines in the frame.

        Looks for diagonal lines passing near the far corners, representing
        the left and right sidelines. Returns their intersection (VP).

        When one near corner is reliable, uses line(far, reliable_near) as
        the definitive sideline for that side. This is more accurate than
        Hough candidates because the YOLO keypoint gives an exact position.
        Hough is only used for the OTHER side.

        If only one sideline is detected, reflects it across the far baseline
        midpoint to synthesize the other side and compute the VP.
        """
        h, w = frame.shape[:2]
        far_left = (corners[3]["x"], corners[3]["y"])
        far_right = (corners[2]["x"], corners[2]["y"])

        fl_px = (int(far_left[0] * w), int(far_left[1] * h))
        fr_px = (int(far_right[0] * w), int(far_right[1] * h))

        far_y_px = max(fl_px[1], fr_px[1])
        roi_top = max(0, far_y_px - int(0.05 * h))

        gray = cv2.cvtColor(frame[roi_top:], cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180, threshold=30,
            minLineLength=int(0.05 * w), maxLineGap=int(0.04 * w),
        )

        # When one near corner is reliable, use line(far, near) as the
        # definitive sideline for that side — more accurate than Hough.
        nl_conf = (per_corner_confidence or {}).get("near-left", 0.0)
        nr_conf = (per_corner_confidence or {}).get("near-right", 0.0)
        reliable_left: tuple[tuple[float, float], tuple[float, float]] | None = None
        reliable_right: tuple[tuple[float, float], tuple[float, float]] | None = None

        if nl_conf >= conf_threshold:
            near_left = (corners[0]["x"], corners[0]["y"])
            reliable_left = (far_left, near_left)
        if nr_conf >= conf_threshold:
            near_right = (corners[1]["x"], corners[1]["y"])
            reliable_right = (far_right, near_right)

        # Collect Hough line candidates near far corners
        left_candidates: list[tuple[tuple[float, float], tuple[float, float]]] = []
        right_candidates: list[tuple[tuple[float, float], tuple[float, float]]] = []

        if lines is not None and len(lines) >= 1:
            far_corner_threshold = 0.05 * max(w, h)

            for line in lines:
                x1, y1, x2, y2 = line[0]
                y1 += roi_top
                y2 += roi_top

                angle = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
                if angle < 10 or angle > 80:
                    continue

                p1 = (x1 / w, y1 / h)
                p2 = (x2 / w, y2 / h)

                if point_line_distance(far_left, p1, p2) * max(w, h) < far_corner_threshold:
                    left_candidates.append((p1, p2))
                if point_line_distance(far_right, p1, p2) * max(w, h) < far_corner_threshold:
                    right_candidates.append((p1, p2))

        # Prefer reliable sideline over Hough candidates
        left_line = reliable_left or self._longest_line(left_candidates)
        right_line = reliable_right or self._longest_line(right_candidates)

        if not left_line and not right_line:
            logger.debug("Sideline detection: no lines pass near far corners")
            return None

        far_mid_x = (far_left[0] + far_right[0]) / 2.0
        far_mid_y = (far_left[1] + far_right[1]) / 2.0

        if left_line and right_line:
            # Both sidelines: VP = intersection
            vp = line_intersection(
                left_line[0], left_line[1], right_line[0], right_line[1],
            )
        elif left_line is not None or right_line is not None:
            # Single sideline: mirror across far baseline midpoint to get VP.
            detected = left_line if left_line is not None else right_line
            assert detected is not None
            mirrored = (
                (2 * far_mid_x - detected[0][0], detected[0][1]),
                (2 * far_mid_x - detected[1][0], detected[1][1]),
            )
            vp = line_intersection(
                detected[0], detected[1], mirrored[0], mirrored[1],
            )
            logger.info(
                "Sideline detection: single-sided, mirrored for VP",
            )
        else:
            return None

        if vp is None:
            return None

        if vp[1] >= far_mid_y:
            logger.debug("Sideline VP below far baseline — skipping")
            return None

        logger.info(
            "Sideline detection: VP=(%.3f, %.3f) from %d left + %d right candidates",
            vp[0], vp[1], len(left_candidates), len(right_candidates),
        )
        return vp

    def _refine_with_vp(
        self,
        corners: list[dict[str, float]],
        per_corner_confidence: dict[str, float],
        conf_threshold: float,
        vp: tuple[float, float],
        frame: np.ndarray | None = None,
    ) -> tuple[list[dict[str, float]], set[str]]:
        """Refine near corners using a VP, with optional center line improvement.

        Pipeline:
        1. Compute VP-based extrapolation (aspect-ratio formula, always works)
        2. If frame available AND both near corners unreliable, detect center line
           and try projectively correct methods (homography, harmonic conjugate)
        3. Accept improved result only if it agrees with the VP baseline

        When one near corner is reliable, we derive t from it (reliable-t).
        The VP direction may be inaccurate but t is exact. Homography/harmonic
        conjugate can make the result worse because they amplify VP errors through
        additional transformations. So we skip them in the one-reliable case.
        """
        # Step 1: VP extrapolation as baseline
        vp_result = self._extrapolate_from_vp(
            corners, per_corner_confidence, conf_threshold, vp,
        )

        # Step 2: Try center line — only when BOTH near corners are unreliable.
        # When one is reliable, the single-t derivation gives the correct court
        # depth; homography/harmonic can only regress by amplifying VP errors.
        nl_conf = per_corner_confidence.get("near-left", 1.0)
        nr_conf = per_corner_confidence.get("near-right", 1.0)
        both_unreliable = nl_conf < conf_threshold and nr_conf < conf_threshold

        if frame is not None and both_unreliable:
            far_left = (corners[3]["x"], corners[3]["y"])
            far_right = (corners[2]["x"], corners[2]["y"])
            center_line = self._detect_center_line(frame, far_left, far_right)

            if center_line is not None:
                for method_name, method in [
                    ("homography", self._refine_via_homography),
                    ("harmonic conjugate", self._extrapolate_via_harmonic_conjugate),
                ]:
                    result = method(
                        corners, per_corner_confidence, conf_threshold, vp, center_line,
                    )
                    if result is not None and self._agrees_with_baseline(
                        result[0], vp_result[0], vp_result[1],
                    ):
                        logger.info("Refined near corners via %s", method_name)
                        return result

        return vp_result

    def _refine_via_sideline_detection(
        self,
        corners: list[dict[str, float]],
        per_corner_confidence: dict[str, float],
        conf_threshold: float,
        frame: np.ndarray,
    ) -> tuple[list[dict[str, float]], set[str]] | None:
        """Refine near corners using sideline VP detected from the frame."""
        vp = self._detect_sideline_vp(
            corners, frame, per_corner_confidence, conf_threshold,
        )
        if vp is None:
            return None
        return self._refine_with_vp(
            corners, per_corner_confidence, conf_threshold, vp, frame,
        )

    def _refine_via_vp_fallback(
        self,
        corners: list[dict[str, float]],
        per_corner_confidence: dict[str, float],
        conf_threshold: float,
        frame: np.ndarray | None = None,
    ) -> tuple[list[dict[str, float]], set[str]]:
        """Refine near corners using VP from model's near-corner guesses.

        When both near corners are unreliable (conf < threshold), the raw VP
        from sideline intersection is noisy — the model can estimate convergence
        rate (VP.y) but not left-right balance (VP.x) at ~0.001 confidence.
        In this case, center VP.x at the far baseline midpoint to produce
        symmetric sidelines, which is correct for the typical centered camera.
        """
        near_left = (corners[0]["x"], corners[0]["y"])
        near_right = (corners[1]["x"], corners[1]["y"])
        far_right = (corners[2]["x"], corners[2]["y"])
        far_left = (corners[3]["x"], corners[3]["y"])

        vp = line_intersection(far_left, near_left, far_right, near_right)
        if vp is None:
            return corners, set()

        far_mid_x = (far_left[0] + far_right[0]) / 2.0
        far_mid_y = (far_left[1] + far_right[1]) / 2.0
        if vp[1] >= far_mid_y:
            return corners, set()

        # When both near corners are unreliable, center VP.x at the far
        # baseline midpoint. The model's near-corner guesses at ~0.001
        # confidence introduce random left-right VP offset.
        nl_conf = per_corner_confidence.get("near-left", 1.0)
        nr_conf = per_corner_confidence.get("near-right", 1.0)
        if nl_conf < conf_threshold and nr_conf < conf_threshold:
            vp = (far_mid_x, vp[1])

        return self._refine_with_vp(
            corners, per_corner_confidence, conf_threshold, vp, frame,
        )

    def _extrapolate_from_vp(
        self,
        corners: list[dict[str, float]],
        per_corner_confidence: dict[str, float],
        conf_threshold: float,
        vp: tuple[float, float],
    ) -> tuple[list[dict[str, float]], set[str]]:
        """Extrapolate low-confidence near corners from a vanishing point.

        Uses a single parametric `t` for both near corners to ensure the result
        is a valid perspective projection of a rectangle. Independent per-side
        `t` values produce asymmetric quads because the VP-to-corner distances
        differ, but the projective parameter must be equal for both sidelines.
        """
        nl_conf = per_corner_confidence.get("near-left", 1.0)
        nr_conf = per_corner_confidence.get("near-right", 1.0)

        far_right = (corners[2]["x"], corners[2]["y"])
        far_left = (corners[3]["x"], corners[3]["y"])

        aspect_ratio = COURT_LENGTH / COURT_WIDTH
        far_baseline_len = math.sqrt(
            (far_right[0] - far_left[0]) ** 2 + (far_right[1] - far_left[1]) ** 2
        )
        if far_baseline_len < 0.01:
            return corners, set()

        # Compute a single t (parametric distance along sideline from far
        # corner toward VP). For a perspective rectangle, both near corners
        # must lie at the same parametric t.
        #
        # When one near corner is reliable, derive t from it via projection
        # onto the sideline direction — this gives the actual court depth
        # instead of the approximation formula which overestimates it.
        # When both are unreliable, use the formula with averaged sideline
        # length for symmetry.
        t: float | None = None
        if nl_conf >= conf_threshold and nr_conf < conf_threshold:
            # Derive t from reliable near-left
            dx_l = vp[0] - far_left[0]
            dy_l = vp[1] - far_left[1]
            sl_sq = dx_l * dx_l + dy_l * dy_l
            if sl_sq > 1e-6:
                nl = (corners[0]["x"], corners[0]["y"])
                t = ((nl[0] - far_left[0]) * dx_l + (nl[1] - far_left[1]) * dy_l) / sl_sq
        elif nr_conf >= conf_threshold and nl_conf < conf_threshold:
            # Derive t from reliable near-right
            dx_r = vp[0] - far_right[0]
            dy_r = vp[1] - far_right[1]
            sl_sq = dx_r * dx_r + dy_r * dy_r
            if sl_sq > 1e-6:
                nr = (corners[1]["x"], corners[1]["y"])
                t = ((nr[0] - far_right[0]) * dx_r + (nr[1] - far_right[1]) * dy_r) / sl_sq

        if t is None:
            # Both unreliable or projection failed: use formula with avg
            sl_len_left = math.sqrt(
                (vp[0] - far_left[0]) ** 2 + (vp[1] - far_left[1]) ** 2,
            )
            sl_len_right = math.sqrt(
                (vp[0] - far_right[0]) ** 2 + (vp[1] - far_right[1]) ** 2,
            )
            avg_sl_len = (sl_len_left + sl_len_right) / 2.0
            if avg_sl_len < 0.01:
                return corners, set()
            t = -(far_baseline_len * aspect_ratio) / avg_sl_len

        refined = list(corners)
        refined_names: set[str] = set()

        if nl_conf < conf_threshold:
            dx = vp[0] - far_left[0]
            dy = vp[1] - far_left[1]
            nx, ny = far_left[0] + t * dx, far_left[1] + t * dy
            refined[0] = {"x": round(nx, 6), "y": round(ny, 6)}
            refined_names.add("near-left")
            logger.info(
                "VP refined near-left: (%.3f, %.3f) → (%.3f, %.3f)",
                corners[0]["x"], corners[0]["y"], nx, ny,
            )

        if nr_conf < conf_threshold:
            dx = vp[0] - far_right[0]
            dy = vp[1] - far_right[1]
            nx, ny = far_right[0] + t * dx, far_right[1] + t * dy
            refined[1] = {"x": round(nx, 6), "y": round(ny, 6)}
            refined_names.add("near-right")
            logger.info(
                "VP refined near-right: (%.3f, %.3f) → (%.3f, %.3f)",
                corners[1]["x"], corners[1]["y"], nx, ny,
            )

        refined = self._clamp_near_corners(refined)
        if not self._is_convex_quad(refined):
            logger.warning("VP refined corners not convex — falling back to original")
            return corners, set()

        return refined, refined_names

    def _detect_center_line(
        self,
        frame: np.ndarray,
        far_left: tuple[float, float],
        far_right: tuple[float, float],
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """Detect the net/center line as a horizontal line in the frame.

        Looks for the topmost wide horizontal line just below the far baseline.
        Limited to 25% of frame height below the far baseline to avoid
        picking up near-court lines or sand markings.

        Args:
            frame: BGR image.
            far_left: Far-left corner (x, y) in normalized coords.
            far_right: Far-right corner (x, y) in normalized coords.

        Returns:
            Two normalized points defining the center line, or None.
        """
        h, w = frame.shape[:2]

        # ROI: from just above far baseline to limited depth below it.
        # The net is always close to the far baseline in image space
        # (perspective compresses the far half of the court).
        # Limit to 25% of frame height below far baseline to avoid
        # picking up near-court lines or sand markings.
        far_y_px = int(max(far_left[1], far_right[1]) * h)
        roi_top = max(0, far_y_px - int(0.02 * h))
        roi_bottom = min(h, far_y_px + int(0.25 * h))

        if roi_bottom <= roi_top + 10:
            return None

        roi = frame[roi_top:roi_bottom]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # Hough line detection — look for horizontal lines
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180, threshold=40,
            minLineLength=int(0.15 * w), maxLineGap=int(0.03 * w),
        )

        if lines is None or len(lines) == 0:
            return None

        far_baseline_width = abs(far_right[0] - far_left[0]) * w
        min_span = 0.40 * far_baseline_width
        far_y_norm = max(far_left[1], far_right[1])

        # Collect qualifying horizontal lines, sort by Y (topmost first)
        # The net is the topmost wide horizontal line below the far baseline
        candidates: list[tuple[float, tuple[tuple[float, float], tuple[float, float]]]] = []

        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Check near-horizontal: angle < 15 degrees
            dx, dy = x2 - x1, y2 - y1
            angle = abs(math.degrees(math.atan2(dy, dx)))
            if angle > 15 and angle < 165:
                continue

            # Check span
            span = abs(x2 - x1)
            if span < min_span:
                continue

            # Convert to full-frame normalized coords
            ny1 = (y1 + roi_top) / h
            ny2 = (y2 + roi_top) / h
            mid_y = (ny1 + ny2) / 2.0

            # Must be below far baseline
            if mid_y < far_y_norm + 0.02:
                continue

            candidates.append((mid_y, ((x1 / w, ny1), (x2 / w, ny2))))

        if not candidates:
            return None

        # Pick the topmost qualifying line (closest to far baseline = net)
        candidates.sort(key=lambda c: c[0])
        best_line = candidates[0][1]

        logger.info(
            "Center line detected: (%.3f, %.3f) → (%.3f, %.3f)",
            best_line[0][0], best_line[0][1],
            best_line[1][0], best_line[1][1],
        )
        return best_line

    def _find_center_points(
        self,
        corners: list[dict[str, float]],
        per_corner_confidence: dict[str, float],
        conf_threshold: float,
        vp: tuple[float, float],
        center_line: tuple[tuple[float, float], tuple[float, float]],
    ) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        """Find where each sideline intersects the center line.

        Uses reliable near corners (if available) to define sideline directions
        instead of the VP, which may be computed from bad near-corner guesses.

        Strategy per side:
        1. If the near corner on that side is reliable, use line(far, near)
        2. Otherwise, use line(far, VP) (may be inaccurate)

        Returns:
            (center_left, center_right) — either or both may be None.
        """
        nl_conf = per_corner_confidence.get("near-left", 1.0)
        nr_conf = per_corner_confidence.get("near-right", 1.0)
        far_left = (corners[3]["x"], corners[3]["y"])
        far_right = (corners[2]["x"], corners[2]["y"])
        near_left = (corners[0]["x"], corners[0]["y"])
        near_right = (corners[1]["x"], corners[1]["y"])
        far_mid_y = (far_left[1] + far_right[1]) / 2.0

        # Left sideline → center-left
        if nl_conf >= conf_threshold:
            # Reliable near-left: use actual sideline
            center_left = line_intersection(
                far_left, near_left, center_line[0], center_line[1],
            )
        else:
            # Fallback: use VP direction
            center_left = line_intersection(
                far_left, vp, center_line[0], center_line[1],
            )

        # Right sideline → center-right
        if nr_conf >= conf_threshold:
            center_right = line_intersection(
                far_right, near_right, center_line[0], center_line[1],
            )
        else:
            center_right = line_intersection(
                far_right, vp, center_line[0], center_line[1],
            )

        # Validate: center points must be below far baseline
        if center_left is not None and center_left[1] < far_mid_y:
            logger.debug(
                "Center-left above far baseline (%.3f < %.3f)",
                center_left[1], far_mid_y,
            )
            center_left = None
        if center_right is not None and center_right[1] < far_mid_y:
            logger.debug(
                "Center-right above far baseline (%.3f < %.3f)",
                center_right[1], far_mid_y,
            )
            center_right = None

        return center_left, center_right

    def _extrapolate_via_harmonic_conjugate(
        self,
        corners: list[dict[str, float]],
        per_corner_confidence: dict[str, float],
        conf_threshold: float,
        vp: tuple[float, float],
        center_line: tuple[tuple[float, float], tuple[float, float]],
    ) -> tuple[list[dict[str, float]], set[str]] | None:
        """Extrapolate near corners using the harmonic conjugate.

        Uses center points (sideline-center line intersections) and the
        harmonic conjugate to find projectively correct near corners.
        """
        nl_conf = per_corner_confidence.get("near-left", 1.0)
        nr_conf = per_corner_confidence.get("near-right", 1.0)
        far_left = (corners[3]["x"], corners[3]["y"])
        far_right = (corners[2]["x"], corners[2]["y"])

        center_left, center_right = self._find_center_points(
            corners, per_corner_confidence, conf_threshold, vp, center_line,
        )

        refined = list(corners)
        refined_names: set[str] = set()

        if nl_conf < conf_threshold and center_left is not None:
            near_left = harmonic_conjugate(far_left, center_left, vp)
            refined[0] = {"x": round(near_left[0], 6), "y": round(near_left[1], 6)}
            refined_names.add("near-left")
            logger.info(
                "Harmonic conjugate near-left: (%.3f, %.3f) → (%.3f, %.3f)",
                corners[0]["x"], corners[0]["y"], near_left[0], near_left[1],
            )

        if nr_conf < conf_threshold and center_right is not None:
            near_right = harmonic_conjugate(far_right, center_right, vp)
            refined[1] = {"x": round(near_right[0], 6), "y": round(near_right[1], 6)}
            refined_names.add("near-right")
            logger.info(
                "Harmonic conjugate near-right: (%.3f, %.3f) → (%.3f, %.3f)",
                corners[1]["x"], corners[1]["y"], near_right[0], near_right[1],
            )

        if not refined_names:
            return None

        refined = self._clamp_near_corners(refined)
        if not self._is_convex_quad(refined):
            logger.warning("Harmonic conjugate corners not convex — skipping")
            return None

        if not self._validate_court_geometry(refined):
            logger.warning("Harmonic conjugate corners fail geometry validation — skipping")
            return None

        return refined, refined_names

    def _refine_via_homography(
        self,
        corners: list[dict[str, float]],
        per_corner_confidence: dict[str, float],
        conf_threshold: float,
        vp: tuple[float, float],
        center_line: tuple[tuple[float, float], tuple[float, float]],
    ) -> tuple[list[dict[str, float]], set[str]] | None:
        """Refine near corners via homography from correspondences.

        Builds correspondences from reliable points:
        - Far corners (always reliable from YOLO)
        - Center points (sideline-center line intersections)
        - Reliable near corners (if one side has high confidence)

        Needs >= 4 correspondences to fit a homography.
        """
        nl_conf = per_corner_confidence.get("near-left", 1.0)
        nr_conf = per_corner_confidence.get("near-right", 1.0)
        far_left = (corners[3]["x"], corners[3]["y"])
        far_right = (corners[2]["x"], corners[2]["y"])

        center_left, center_right = self._find_center_points(
            corners, per_corner_confidence, conf_threshold, vp, center_line,
        )

        # Build correspondences from reliable points
        image_pts_list: list[list[float]] = []
        court_pts_list: list[list[float]] = []

        # Far corners (always reliable)
        image_pts_list.append([far_left[0], far_left[1]])
        court_pts_list.append([0.0, COURT_LENGTH])
        image_pts_list.append([far_right[0], far_right[1]])
        court_pts_list.append([COURT_WIDTH, COURT_LENGTH])

        # Reliable near corners
        if nl_conf >= conf_threshold:
            nl = (corners[0]["x"], corners[0]["y"])
            image_pts_list.append([nl[0], nl[1]])
            court_pts_list.append([0.0, 0.0])
        if nr_conf >= conf_threshold:
            nr = (corners[1]["x"], corners[1]["y"])
            image_pts_list.append([nr[0], nr[1]])
            court_pts_list.append([COURT_WIDTH, 0.0])

        # Center points
        if center_left is not None:
            image_pts_list.append([center_left[0], center_left[1]])
            court_pts_list.append([0.0, COURT_LENGTH / 2])
        if center_right is not None:
            image_pts_list.append([center_right[0], center_right[1]])
            court_pts_list.append([COURT_WIDTH, COURT_LENGTH / 2])

        if len(image_pts_list) < 4:
            logger.debug(
                "Homography: only %d correspondences (need 4)", len(image_pts_list),
            )
            return None

        image_pts = np.array(image_pts_list, dtype=np.float64)
        court_pts = np.array(court_pts_list, dtype=np.float64)

        # Fit homography: court-space → image-space
        h_matrix, status = cv2.findHomography(court_pts, image_pts)
        if h_matrix is None:
            return None

        # Check reprojection error on the input points
        reproj = cv2.perspectiveTransform(
            court_pts.reshape(-1, 1, 2), h_matrix,
        ).reshape(-1, 2)
        reproj_error = float(np.sqrt(((reproj - image_pts) ** 2).sum(axis=1)).mean())
        if reproj_error > 0.02:
            logger.debug(
                "Homography reprojection error too high: %.4f", reproj_error,
            )
            return None

        # Project all 4 court corners
        all_court = np.array([
            [0.0, 0.0],          # near-left
            [COURT_WIDTH, 0.0],  # near-right
            [COURT_WIDTH, COURT_LENGTH],  # far-right
            [0.0, COURT_LENGTH],          # far-left
        ], dtype=np.float64)

        projected = cv2.perspectiveTransform(
            all_court.reshape(-1, 1, 2), h_matrix,
        ).reshape(-1, 2)

        refined = list(corners)
        refined_names: set[str] = set()

        if nl_conf < conf_threshold:
            refined[0] = {
                "x": round(float(projected[0, 0]), 6),
                "y": round(float(projected[0, 1]), 6),
            }
            refined_names.add("near-left")
            logger.info(
                "Homography near-left: (%.3f, %.3f) → (%.3f, %.3f)",
                corners[0]["x"], corners[0]["y"],
                projected[0, 0], projected[0, 1],
            )

        if nr_conf < conf_threshold:
            refined[1] = {
                "x": round(float(projected[1, 0]), 6),
                "y": round(float(projected[1, 1]), 6),
            }
            refined_names.add("near-right")
            logger.info(
                "Homography near-right: (%.3f, %.3f) → (%.3f, %.3f)",
                corners[1]["x"], corners[1]["y"],
                projected[1, 0], projected[1, 1],
            )

        if not refined_names:
            return None

        refined = self._clamp_near_corners(refined)
        if not self._is_convex_quad(refined):
            logger.warning("Homography refined corners not convex — skipping")
            return None

        if not self._validate_court_geometry(refined):
            logger.warning("Homography refined corners fail geometry validation — skipping")
            return None

        return refined, refined_names

    # Max distance between homography/harmonic result and VP baseline
    # for a near corner to be accepted. 0.15 = 15% of frame size.
    # Generous enough for the harmonic conjugate to correct VP errors,
    # tight enough to reject wrong center line detections.
    _BASELINE_AGREEMENT_THRESHOLD = 0.15

    @staticmethod
    def _agrees_with_baseline(
        candidate: list[dict[str, float]],
        baseline: list[dict[str, float]],
        refined_names: set[str],
    ) -> bool:
        """Check if candidate near corners agree with VP baseline result.

        Only checks corners that were refined (in refined_names).
        Returns True if all refined corners are within threshold distance.
        """
        threshold = CourtKeypointDetector._BASELINE_AGREEMENT_THRESHOLD
        for i, name in [(0, "near-left"), (1, "near-right")]:
            if name not in refined_names:
                continue
            dx = candidate[i]["x"] - baseline[i]["x"]
            dy = candidate[i]["y"] - baseline[i]["y"]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > threshold:
                logger.debug(
                    "%s: candidate (%.3f, %.3f) too far from baseline "
                    "(%.3f, %.3f), dist=%.3f > %.3f",
                    name, candidate[i]["x"], candidate[i]["y"],
                    baseline[i]["x"], baseline[i]["y"], dist, threshold,
                )
                return False
        return True

    @staticmethod
    def _validate_court_geometry(corners: list[dict[str, float]]) -> bool:
        """Validate that refined corners form a plausible perspective court.

        Checks:
        - Near baseline wider than far baseline (perspective constraint)
        - Perspective ratio between 1.0 and 8.0
        - Both sidelines converge upward (toward VP above court)
        - Near corners below far corners in Y

        Args:
            corners: 4 corners [near-left, near-right, far-right, far-left].

        Returns:
            True if geometry is plausible.
        """
        if len(corners) != 4:
            return False

        nl = (corners[0]["x"], corners[0]["y"])
        nr = (corners[1]["x"], corners[1]["y"])
        fr = (corners[2]["x"], corners[2]["y"])
        fl = (corners[3]["x"], corners[3]["y"])

        near_width = abs(nr[0] - nl[0])
        far_width = abs(fr[0] - fl[0])

        # Near baseline must be wider (perspective)
        if near_width < far_width * 0.95:
            return False

        # Perspective ratio check
        if far_width > 0.01:
            ratio = near_width / far_width
            if ratio > 8.0:
                return False

        # Near corners must be below far corners in Y
        near_mid_y = (nl[1] + nr[1]) / 2.0
        far_mid_y = (fl[1] + fr[1]) / 2.0
        if near_mid_y < far_mid_y + 0.05:
            return False

        # Sidelines should converge upward (left sideline goes left-to-right
        # from near to far, right sideline goes right-to-left)
        # Equivalently: far-left.x > near-left.x and far-right.x < near-right.x
        if fl[0] < nl[0] - 0.02:
            return False
        if fr[0] > nr[0] + 0.02:
            return False

        return True

    def _clamp_near_corners(self, corners: list[dict[str, float]]) -> list[dict[str, float]]:
        """Clamp near corners to limit extrapolation overshoot."""
        margin = self.NEAR_CORNER_MAX_MARGIN
        refined = list(corners)
        for i in (0, 1):
            c = refined[i]
            cx = max(-margin, min(1.0 + margin, c["x"]))
            cy = max(-margin, min(1.0 + margin, c["y"]))
            if cx != c["x"] or cy != c["y"]:
                refined[i] = {"x": round(cx, 6), "y": round(cy, 6)}
        return refined

    @staticmethod
    def _longest_line(
        candidates: list[tuple[tuple[float, float], tuple[float, float]]],
    ) -> tuple[tuple[float, float], tuple[float, float]] | None:
        """Pick the longest line segment from candidates."""
        if not candidates:
            return None
        return max(
            candidates,
            key=lambda c: math.hypot(c[1][0] - c[0][0], c[1][1] - c[0][1]),
        )

    # Minimum per-corner confidence to count as "reliable". Below this, the
    # corner position is essentially a guess (keypoint invisible or off-screen).
    _RELIABLE_CORNER_CONF = 0.5

    # Partial credit for VP-refined corners in confidence penalty.
    # VP extrapolation from high-confidence far corners + court aspect ratio
    # is geometrically sound, so refined corners deserve partial trust.
    _VP_REFINED_CREDIT = 0.80

    @staticmethod
    def _penalize_confidence(
        bbox_confidence: float,
        per_corner_confidence: dict[str, float],
        vp_refined_corners: set[str] | None = None,
    ) -> float:
        """Reduce overall confidence when individual corners are unreliable.

        The bbox confidence says "a court exists" but not "all corners are
        accurately localized". A detection with 2/4 reliable corners should
        not get 0.91 confidence — that causes auto-save of bad calibrations.

        VP-refined corners (where near corners were replaced by perspective
        extrapolation from accurate far corners) get partial credit since
        the extrapolation is geometrically grounded.

        Penalty: confidence * (score / 4) where each corner contributes:
        - 1.0 if keypoint confidence >= 0.5 (reliable)
        - VP_REFINED_CREDIT (0.80) if VP-refined
        - 0.0 otherwise

        Examples:
        - 4 reliable: 0.91 * 4/4 = 0.91 (no penalty)
        - 2 reliable + 2 refined: 0.91 * (2 + 1.6)/4 = 0.91 * 0.90 = 0.82
        - 2 reliable + 0 refined: 0.91 * 2/4 = 0.455 (blocks auto-save)
        """
        if not per_corner_confidence:
            return bbox_confidence
        threshold = CourtKeypointDetector._RELIABLE_CORNER_CONF
        refined = vp_refined_corners or set()
        credit = CourtKeypointDetector._VP_REFINED_CREDIT

        score = 0.0
        for name, conf in per_corner_confidence.items():
            if conf >= threshold:
                score += 1.0
            elif name in refined:
                score += credit
            # else: 0.0

        return bbox_confidence * (score / 4.0)

    @staticmethod
    def _is_convex_quad(corners: list[dict[str, float]]) -> bool:
        """Check if 4 corners form a convex quadrilateral.

        Uses cross product sign consistency around the polygon.
        """
        if len(corners) != 4:
            return False

        pts = [(c["x"], c["y"]) for c in corners]
        sign = None
        for i in range(4):
            p0 = pts[i]
            p1 = pts[(i + 1) % 4]
            p2 = pts[(i + 2) % 4]
            cross = (p1[0] - p0[0]) * (p2[1] - p1[1]) - (p1[1] - p0[1]) * (p2[0] - p1[0])
            if abs(cross) < 1e-10:
                continue  # collinear edge, skip
            s = cross > 0
            if sign is None:
                sign = s
            elif s != sign:
                return False
        return True

    def _aggregate(
        self, frame_results: list[FrameKeypoints],
        n_sampled: int | None = None,
    ) -> tuple[list[dict[str, float]], float, CourtQualityDiagnostics]:
        """Aggregate multi-frame detections via confidence-weighted median.

        For each corner, collects all predictions, removes outliers (>2σ from
        median for corners, >1.5σ for center points), and takes the weighted
        median using per-keypoint confidence.

        Returns:
            (corners, confidence, diagnostics) tuple.
        """
        n = len(frame_results)
        n_total = n_sampled if n_sampled is not None else n

        if n == 0:
            diag = self._build_diagnostics([], [], n_total, 0)
            return [], 0.0, diag

        if n == 1:
            diag = self._build_diagnostics(
                frame_results[0].corners, frame_results, n_total, n,
            )
            self._last_diagnostics = diag
            return frame_results[0].corners, frame_results[0].confidence, diag

        # Collect per-corner coordinates and per-keypoint confidences
        all_x = np.zeros((n, 4))
        all_y = np.zeros((n, 4))
        all_conf = np.zeros(n)
        all_kpt_conf = np.zeros((n, 4))

        for i, fr in enumerate(frame_results):
            all_conf[i] = fr.confidence
            for j in range(4):
                all_x[i, j] = fr.corners[j]["x"]
                all_y[i, j] = fr.corners[j]["y"]
                all_kpt_conf[i, j] = fr.kpt_confidences[j]

        # Per-corner outlier filtering + confidence-weighted median
        corners = []
        corner_stds: list[float] = []
        corner_confs: list[float] = []

        for j in range(4):
            xs = all_x[:, j]
            ys = all_y[:, j]
            weights = all_kpt_conf[:, j]

            # Track per-corner confidence (before outlier filtering)
            corner_confs.append(float(np.mean(weights)))

            # Remove outliers: distance from median > 2σ
            med_x, med_y = float(np.median(xs)), float(np.median(ys))
            dists = np.sqrt((xs - med_x) ** 2 + (ys - med_y) ** 2)
            sigma = float(np.std(dists))

            # Track per-corner std (position variance across frames)
            corner_stds.append(sigma)

            if sigma > 0:
                mask = dists <= 2.0 * sigma
                if mask.sum() >= 3:
                    xs = xs[mask]
                    ys = ys[mask]
                    weights = weights[mask]

            corners.append({
                "x": round(float(_weighted_median(xs, weights)), 6),
                "y": round(float(_weighted_median(ys, weights)), 6),
            })

        # Confidence: median of frame confidences
        confidence = float(np.median(all_conf))

        # Build diagnostics
        diagnostics = self._build_diagnostics(corners, frame_results, n_total, n)
        diagnostics.per_corner_confidence = dict(zip(CORNER_NAMES, corner_confs))
        diagnostics.per_corner_std = dict(zip(CORNER_NAMES, corner_stds))

        # Aggregate center points from 6-keypoint model
        frames_with_centers = [
            fr for fr in frame_results
            if fr.center_points is not None and fr.center_confidences is not None
        ]
        if frames_with_centers:
            nc = len(frames_with_centers)
            center_x = np.zeros((nc, 2))
            center_y = np.zeros((nc, 2))
            center_conf = np.zeros((nc, 2))
            for i, fr in enumerate(frames_with_centers):
                assert fr.center_points is not None
                assert fr.center_confidences is not None
                for j in range(2):
                    center_x[i, j] = fr.center_points[j]["x"]
                    center_y[i, j] = fr.center_points[j]["y"]
                    center_conf[i, j] = fr.center_confidences[j]

            agg_centers = []
            agg_center_confs = []
            for j in range(2):
                xs = center_x[:, j]
                ys = center_y[:, j]
                weights = center_conf[:, j]
                agg_center_confs.append(float(np.mean(weights)))

                # Tighter outlier filtering for center points (1.5σ vs 2σ for corners).
                # Center points are always visible in-frame with ~0.99 confidence,
                # so they cluster tightly — tighter gate prevents shadow/line artifacts.
                med_x, med_y = float(np.median(xs)), float(np.median(ys))
                dists = np.sqrt((xs - med_x) ** 2 + (ys - med_y) ** 2)
                sigma = float(np.std(dists))
                if sigma > 0:
                    mask = dists <= 1.5 * sigma
                    if mask.sum() >= 3:
                        xs = xs[mask]
                        ys = ys[mask]
                        weights = weights[mask]

                agg_centers.append({
                    "x": round(float(_weighted_median(xs, weights)), 6),
                    "y": round(float(_weighted_median(ys, weights)), 6),
                })

            diagnostics.center_points = agg_centers
            diagnostics.center_confidences = dict(zip(CENTER_NAMES, agg_center_confs))

        # Check per-corner confidence for weak corners
        for name, conf in zip(CORNER_NAMES, corner_confs):
            if 0 < conf < 0.5:
                diagnostics.warnings.append(
                    f"Low confidence for {name} corner ({conf:.2f}) — "
                    "position may be inaccurate"
                )

        self._last_diagnostics = diagnostics

        return corners, confidence, diagnostics

    def _build_diagnostics(
        self,
        corners: list[dict[str, float]],
        frame_results: list[FrameKeypoints],
        n_sampled: int,
        n_detected: int,
    ) -> CourtQualityDiagnostics:
        """Build quality diagnostics from aggregated results."""
        detection_rate = n_detected / max(1, n_sampled)

        # Per-corner defaults
        per_corner_conf: dict[str, float] = {name: 0.0 for name in CORNER_NAMES}
        per_corner_std: dict[str, float] = {name: 0.0 for name in CORNER_NAMES}

        # Off-screen corners (y > 1.0 means below original frame)
        off_screen: list[str] = []
        if len(corners) == 4:
            for i, name in enumerate(CORNER_NAMES):
                if corners[i]["y"] > 1.0:
                    off_screen.append(name)

        # Perspective ratio: near court width / far court width
        perspective_ratio = 0.0
        if len(corners) == 4:
            near_width = abs(corners[1]["x"] - corners[0]["x"])  # near-right - near-left
            far_width = abs(corners[2]["x"] - corners[3]["x"])  # far-right - far-left
            if far_width > 0.01:
                perspective_ratio = near_width / far_width

        # Generate warnings
        warnings: list[str] = []
        if detection_rate < 0.5:
            warnings.append(
                f"Low detection rate ({detection_rate:.0%}) — court may be "
                "partially visible or obstructed"
            )
        if off_screen:
            warnings.append(
                f"Off-screen corners: {', '.join(off_screen)} — camera is too "
                "close or at too low an angle"
            )
        if perspective_ratio > 4.0:
            warnings.append(
                f"Extreme perspective (ratio {perspective_ratio:.1f}) — camera "
                "is very low, near-court accuracy may be reduced"
            )
        elif perspective_ratio > 3.0:
            warnings.append(
                f"Strong perspective (ratio {perspective_ratio:.1f}) — consider "
                "recording from a higher vantage point"
            )

        return CourtQualityDiagnostics(
            detection_rate=detection_rate,
            per_corner_confidence=per_corner_conf,
            per_corner_std=per_corner_std,
            off_screen_corners=off_screen,
            perspective_ratio=perspective_ratio,
            warnings=warnings,
        )
