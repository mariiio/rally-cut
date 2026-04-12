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
from rallycut.court.line_geometry import line_intersection

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
            result.corners, per_corner_conf,
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

    # Below this confidence, near corner is considered off-screen and
    # homography projection is used instead of sideline projection.
    # Swept over [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50] — 0.30
    # gave lowest MCD (0.0678) and best success rate (40.9%).
    # (Coincidentally equal to _CENTER_POINT_MIN_CONF; independently determined.)
    OFF_SCREEN_CONF = 0.30

    def _refine_near_corners(
        self,
        corners: list[dict[str, float]],
        per_corner_confidence: dict[str, float],
        conf_threshold: float = 0.5,
        center_points: list[dict[str, float]] | None = None,
        center_confidences: dict[str, float] | None = None,
    ) -> tuple[list[dict[str, float]], set[str]]:
        """Replace low-confidence near corners using best available geometry.

        Strategy priority:
        1. **YOLO center points** (6-keypoint model): Use model-predicted
           net-sideline intersections for sideline projection refinement.
           Always visible in-frame (~0.99 confidence).
        2. **VP from model corners** (fallback): Uses the model's near-corner
           guesses for VP computation. Less accurate but always works.
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

        # Strategy 2: Fallback — VP from model's near-corner guesses.
        # Guard against degenerate geometry (e.g. near-horizontal sidelines)
        # where VP extrapolation can push near corners far from their raw
        # positions. Keep raw if any refined corner moved > 30% of screen.
        refined, refined_names = self._refine_via_vp_fallback(
            corners, per_corner_confidence, conf_threshold,
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
        2. Otherwise, compute via 4-point homography from in-frame keypoints.
        3. Final fallback: VP + aspect ratio extrapolation.
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

        # Pre-compute homography-based near corners from 4 in-frame keypoints
        homo_corners = self._compute_near_corners_via_homography(
            center_points, corners,
        )

        # VP extrapolation as final fallback for off-screen corners
        vp_corners, vp_refined = self._extrapolate_from_vp(
            corners, per_corner_confidence, conf_threshold, vp,
        )

        refined = list(corners)
        refined_names: set[str] = set()

        margin = self.NEAR_CORNER_MAX_MARGIN
        off_screen_conf = self.OFF_SCREEN_CONF

        for i, name in [(0, "near-left"), (1, "near-right")]:
            corner_conf = per_corner_confidence.get(name, 1.0)
            if corner_conf >= conf_threshold:
                continue

            raw_x = corners[i]["x"]
            raw_y = corners[i]["y"]
            far_pt = far_left if i == 0 else far_right
            ctr_pt = center_left if i == 0 else center_right

            # Strategy A: Sideline projection (keep raw Y, fix X)
            # Only when corner has moderate confidence — the raw Y is somewhat
            # trustworthy (corner partially visible or near frame edge).
            raw_y_plausible = far_pt[1] + 0.05 < raw_y < 1.0 + margin
            if corner_conf >= off_screen_conf and raw_y_plausible:
                dy_sl = ctr_pt[1] - far_pt[1]
                if abs(dy_sl) > 1e-6:
                    t = (raw_y - far_pt[1]) / dy_sl
                    proj_x = far_pt[0] + t * (ctr_pt[0] - far_pt[0])
                    refined[i] = {"x": round(proj_x, 6), "y": round(raw_y, 6)}
                    refined_names.add(name)
                    logger.info(
                        "Sideline-projected %s: (%.3f,%.3f)→(%.3f,%.3f) conf=%.3f",
                        name, raw_x, raw_y, proj_x, raw_y, corner_conf,
                    )
                    continue

            # Strategy B: 4-point homography (exact geometry from in-frame pts)
            # Best for off-screen corners: computes both X and Y from the 4
            # high-confidence in-frame keypoints with known court dimensions.
            if homo_corners is not None:
                hx, hy = homo_corners[i]
                refined[i] = {"x": round(hx, 6), "y": round(hy, 6)}
                refined_names.add(name)
                logger.info(
                    "Homography-projected %s: (%.3f,%.3f)→(%.3f,%.3f) conf=%.3f",
                    name, raw_x, raw_y, hx, hy, corner_conf,
                )
                continue

            # Strategy C: VP + aspect ratio fallback
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

    def _compute_near_corners_via_homography(
        self,
        center_points: list[dict[str, float]],
        corners: list[dict[str, float]],
    ) -> list[tuple[float, float]] | None:
        """Compute near corners from 4 in-frame keypoints via homography.

        Uses far-left, far-right, center-left, center-right (all high-confidence,
        always in-frame) with their known court-space coordinates to fit a
        homography, then projects near-left (0,0) and near-right (W,0).

        Returns [(near_left_x, near_left_y), (near_right_x, near_right_y)]
        or None if the homography is degenerate.
        """
        # Court-space coordinates for the 4 in-frame keypoints
        cw, cl = COURT_WIDTH, COURT_LENGTH
        court_pts = np.array([
            [0, cl],       # far-left
            [cw, cl],      # far-right
            [0, cl / 2],   # center-left
            [cw, cl / 2],  # center-right
        ], dtype=np.float32)

        # Image-space coordinates from keypoint detections
        image_pts = np.array([
            [corners[3]["x"], corners[3]["y"]],       # far-left
            [corners[2]["x"], corners[2]["y"]],       # far-right
            [center_points[0]["x"], center_points[0]["y"]],  # center-left
            [center_points[1]["x"], center_points[1]["y"]],  # center-right
        ], dtype=np.float32)

        # 4 exact correspondences → getPerspectiveTransform (no RANSAC needed)
        h_matrix = cv2.getPerspectiveTransform(court_pts, image_pts)
        if h_matrix is None:
            return None

        # Project near corners: near-left=(0,0), near-right=(cw,0)
        near_court = np.array([[0, 0], [cw, 0]], dtype=np.float32).reshape(-1, 1, 2)
        near_image = cv2.perspectiveTransform(near_court, h_matrix)

        if near_image is None:
            return None

        nl = (float(near_image[0, 0, 0]), float(near_image[0, 0, 1]))
        nr = (float(near_image[1, 0, 0]), float(near_image[1, 0, 1]))

        # Sanity checks
        far_mid_y = (corners[2]["y"] + corners[3]["y"]) / 2.0
        if nl[1] < far_mid_y or nr[1] < far_mid_y:
            logger.debug("Homography near corners above far baseline — rejecting")
            return None

        # Reject extreme X projections (e.g., very wide-angle lens)
        margin = self.NEAR_CORNER_MAX_MARGIN
        for x, y in [nl, nr]:
            if x < -margin or x > 1.0 + margin:
                logger.debug("Homography near corner X=%.2f out of bounds — rejecting", x)
                return None

        return [nl, nr]

    def _refine_via_vp_fallback(
        self,
        corners: list[dict[str, float]],
        per_corner_confidence: dict[str, float],
        conf_threshold: float,
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

        return self._extrapolate_from_vp(
            corners, per_corner_confidence, conf_threshold, vp,
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
