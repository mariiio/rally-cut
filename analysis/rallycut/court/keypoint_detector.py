"""Court keypoint detection using YOLO11s-pose.

Detects 4 court corners (near-left, near-right, far-right, far-left) from
video frames using a fine-tuned YOLO-pose model. Handles off-screen near
corners via bottom padding (same as training).

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
    kpt_confidences: list[float]  # per-keypoint visibility/confidence


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
        corners, vp_refined = self._refine_near_corners(result.corners, per_corner_conf)
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

        if len(kpt_xy) != 4:
            logger.warning(f"Expected 4 keypoints, got {len(kpt_xy)}")
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

            # Normalize to padded image
            x_norm = x_px / padded_w
            y_norm = y_px / padded_h

            # Un-pad: reverse the scale_y = 1/(1+pad_ratio) applied during export
            y_orig = y_norm * (1.0 + self._pad_ratio)

            corners.append({"x": round(x_norm, 6), "y": round(y_orig, 6)})
            kpt_confidences.append(kc)

        return FrameKeypoints(
            corners=corners,
            confidence=best_conf,
            kpt_confidences=kpt_confidences,
        )

    # Max extrapolation beyond [0, 1] range for near corners.
    # Caps overshoot: near corners clamped to [-margin, 1+margin].
    # 0.20 reduces regressions from 11→5 while keeping 33% MCD improvement.
    NEAR_CORNER_MAX_MARGIN = 0.20

    def _refine_near_corners(
        self,
        corners: list[dict[str, float]],
        per_corner_confidence: dict[str, float],
        conf_threshold: float = 0.5,
    ) -> tuple[list[dict[str, float]], set[str]]:
        """Replace low-confidence near corners with perspective-extrapolated positions.

        Uses the accurate far corners as anchors, derives the vanishing point from
        sideline directions, and extrapolates near corners using the known court
        aspect ratio (16m/8m = 2.0). Same math as classical detector Strategy 3.

        Extrapolated positions are clamped to NEAR_CORNER_MAX_MARGIN beyond the
        frame to prevent overshoot when GT corners are just slightly off-screen.

        Args:
            corners: 4 corners [near-left, near-right, far-right, far-left].
            per_corner_confidence: Mean confidence per corner name.
            conf_threshold: Below this, replace the near corner.

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

        # Extract corner positions as tuples
        near_left = (corners[0]["x"], corners[0]["y"])
        near_right = (corners[1]["x"], corners[1]["y"])
        far_right = (corners[2]["x"], corners[2]["y"])
        far_left = (corners[3]["x"], corners[3]["y"])

        # Compute vanishing point from sideline convergence
        # Left sideline: far_left → near_left, Right sideline: far_right → near_right
        vp = line_intersection(far_left, near_left, far_right, near_right)
        if vp is None:
            logger.debug("Sidelines are parallel — skipping near-corner refinement")
            return corners, set()

        # Validate VP is above far baseline (perspective convergence upward)
        far_mid_y = (far_left[1] + far_right[1]) / 2.0
        if vp[1] >= far_mid_y:
            logger.debug(
                "VP below far baseline (y=%.3f >= %.3f) — skipping refinement",
                vp[1], far_mid_y,
            )
            return corners, set()

        # Aspect ratio: court_length / court_width
        aspect_ratio = COURT_LENGTH / COURT_WIDTH  # 2.0

        # Far baseline length (in normalized image coords)
        far_baseline_len = math.sqrt(
            (far_right[0] - far_left[0]) ** 2 + (far_right[1] - far_left[1]) ** 2
        )
        if far_baseline_len < 0.01:
            return corners, set()

        refined = list(corners)  # shallow copy of dicts
        refined_names: set[str] = set()

        # Extrapolate low-confidence near-left
        if nl_conf < conf_threshold:
            dx = vp[0] - far_left[0]
            dy = vp[1] - far_left[1]
            sl_len = math.sqrt(dx ** 2 + dy ** 2)
            if sl_len > 0.01:
                t = -(far_baseline_len * aspect_ratio) / sl_len
                nx, ny = far_left[0] + t * dx, far_left[1] + t * dy
                refined[0] = {"x": round(nx, 6), "y": round(ny, 6)}
                refined_names.add("near-left")
                logger.info(
                    "Refined near-left: (%.3f, %.3f) → (%.3f, %.3f) [conf=%.3f]",
                    near_left[0], near_left[1], nx, ny, nl_conf,
                )

        # Extrapolate low-confidence near-right
        if nr_conf < conf_threshold:
            dx = vp[0] - far_right[0]
            dy = vp[1] - far_right[1]
            sl_len = math.sqrt(dx ** 2 + dy ** 2)
            if sl_len > 0.01:
                t = -(far_baseline_len * aspect_ratio) / sl_len
                nx, ny = far_right[0] + t * dx, far_right[1] + t * dy
                refined[1] = {"x": round(nx, 6), "y": round(ny, 6)}
                refined_names.add("near-right")
                logger.info(
                    "Refined near-right: (%.3f, %.3f) → (%.3f, %.3f) [conf=%.3f]",
                    near_right[0], near_right[1], nx, ny, nr_conf,
                )

        # Clamp near corners to limit extrapolation overshoot
        margin = self.NEAR_CORNER_MAX_MARGIN
        for i in (0, 1):  # near-left, near-right
            c = refined[i]
            cx = max(-margin, min(1.0 + margin, c["x"]))
            cy = max(-margin, min(1.0 + margin, c["y"]))
            if cx != c["x"] or cy != c["y"]:
                refined[i] = {"x": round(cx, 6), "y": round(cy, 6)}

        # Validate result is a convex quadrilateral; fall back if not
        if not self._is_convex_quad(refined):
            logger.warning("Refined corners not convex — falling back to original")
            return corners, set()

        return refined, refined_names

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
        median), and takes the weighted median using per-keypoint confidence.

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
