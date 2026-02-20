"""Automatic court detection for beach volleyball videos.

Detects court lines from video frames using classical CV (HSV white-line
detection on sand, Hough lines, temporal aggregation via DBSCAN, geometric
court model fitting). No training data needed.

Output: 4 normalized corners [near-left, near-right, far-right, far-left].
Values can exceed [0,1] for off-screen corners (near side).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rallycut.court.calibration import COURT_LENGTH, COURT_WIDTH
from rallycut.court.line_geometry import (
    compute_vanishing_point,
    cross2d,
    harmonic_conjugate,
    line_intersection,
    segment_angle_deg,
    segment_to_rho_theta,
    segments_to_median_line,
)

logger = logging.getLogger(__name__)


@dataclass
class CourtDetectionConfig:
    """Configuration for court detection."""

    # Stage 1: Frame sampling
    n_sample_frames: int = 30
    skip_seconds: float = 2.0
    working_width: int = 960

    # Stage 2: White line detection
    white_saturation_max: int = 60
    white_value_offset: int = 30
    white_value_min: int = 140
    morph_open_size: int = 3
    morph_close_width: int = 7
    canny_low: int = 50
    canny_high: int = 120  # grid search: 120 matches 150 (insensitive), lower is cleaner
    hough_threshold: int = 50  # grid search: 50 > 40 (+0.08 IoU, fewer false segments)
    hough_min_length: int = 50
    hough_max_gap: int = 20
    min_segment_px: int = 30
    max_angle_from_horizontal: float = 75.0
    # Blue/colored line detection (many beach courts use blue lines)
    enable_blue_detection: bool = True
    blue_hue_min: int = 90
    blue_hue_max: int = 130
    blue_saturation_min: int = 50
    blue_value_min: int = 80
    # Dark line detection (black/dark rope court boundaries)
    enable_dark_detection: bool = True
    dark_saturation_max: int = 40  # grid search: 40 > 50/60 (tighter = fewer shadow FPs)
    dark_value_offset: int = 55  # grid search: 55 > 45/35 (darker relative to sand)
    dark_value_min: int = 25  # grid search: 25 > 35/45 (include very dark ropes)

    # Stage 3: Temporal aggregation
    dbscan_eps: float = 0.03
    dbscan_min_samples: int = 5
    min_temporal_support: int = 8
    theta_scale: float = 0.3

    # Stage 4: Court line identification
    horizontal_angle_max: float = 15.0
    sideline_angle_min: float = 10.0
    sideline_angle_max: float = 75.0
    center_line_parallel_tolerance: float = 8.0

    # Stage 5: Geometric fitting
    aspect_ratio: float = COURT_LENGTH / COURT_WIDTH  # 2.0 for beach volleyball

    # Stage 6: Confidence
    confidence_auto_accept: float = 0.7
    confidence_review: float = 0.4


@dataclass
class DetectedLine:
    """A detected court line with metadata."""

    label: str  # "far_baseline", "left_sideline", "right_sideline", "center_line", "near_baseline"
    p1: tuple[float, float]
    p2: tuple[float, float]
    support: int  # Number of frames where this line was detected
    angle_deg: float


@dataclass
class CourtDetectionResult:
    """Result of automatic court detection."""

    corners: list[dict[str, float]]  # 4 corners: near-left, near-right, far-right, far-left
    confidence: float  # 0-1 quality score
    detected_lines: list[DetectedLine] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def is_valid(self) -> bool:
        return len(self.corners) == 4 and self.confidence >= 0.0

    def to_calibration_json(self) -> list[dict[str, float]]:
        """Convert to the format used by courtCalibrationJson in the DB."""
        return self.corners


class CourtDetector:
    """Automatic court detector for beach volleyball videos."""

    def __init__(self, config: CourtDetectionConfig | None = None) -> None:
        self.config = config or CourtDetectionConfig()

    def detect(
        self,
        video_path: str | Path,
        start_frame: int | None = None,
        end_frame: int | None = None,
    ) -> CourtDetectionResult:
        """Detect court corners from a video.

        Args:
            video_path: Path to the video file.
            start_frame: First frame to analyze (default: skip_seconds from start).
            end_frame: Last frame to analyze (default: skip_seconds from end).

        Returns:
            CourtDetectionResult with corners and confidence.
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise FileNotFoundError(f"Video not found: {video_path}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if total_frames <= 0 or fps <= 0:
                return CourtDetectionResult(
                    corners=[], confidence=0.0,
                    warnings=["Could not read video properties"],
                )

            skip_frames = int(self.config.skip_seconds * fps)
            if start_frame is None:
                start_frame = skip_frames
            if end_frame is None:
                end_frame = max(start_frame + 1, total_frames - skip_frames)

            # Stage 1: Sample frames
            frames = self._sample_frames(cap, start_frame, end_frame, orig_width, orig_height)
            if len(frames) < 3:
                return CourtDetectionResult(
                    corners=[], confidence=0.0,
                    warnings=[f"Only {len(frames)} usable frames sampled"],
                )

            # Stage 2: Detect white lines per frame
            all_segments = self._detect_lines_all_frames(frames)
            if not all_segments:
                return CourtDetectionResult(
                    corners=[], confidence=0.0,
                    warnings=["No white line segments detected in any frame"],
                )

            # Stage 3: Temporal aggregation
            consensus_lines = self._aggregate_lines(all_segments)
            if not consensus_lines:
                return CourtDetectionResult(
                    corners=[], confidence=0.0,
                    warnings=["No consensus lines after temporal aggregation"],
                )

            # Stage 4: Identify court lines
            identified = self._identify_court_lines(consensus_lines)
            if identified is None:
                return CourtDetectionResult(
                    corners=[], confidence=0.0,
                    detected_lines=[],
                    warnings=["Could not identify enough court lines"],
                )

            # Stage 5: Fit court model
            result = self._fit_court_model(identified)

            return result

        finally:
            cap.release()

    def detect_from_frames(
        self,
        frames: list[np.ndarray],
    ) -> CourtDetectionResult:
        """Detect court from pre-loaded frames (for testing).

        Args:
            frames: List of BGR frames at working resolution.

        Returns:
            CourtDetectionResult with corners and confidence.
        """
        if len(frames) < 1:
            return CourtDetectionResult(
                corners=[], confidence=0.0,
                warnings=["No frames provided"],
            )

        all_segments = self._detect_lines_all_frames(frames)
        if not all_segments:
            return CourtDetectionResult(
                corners=[], confidence=0.0,
                warnings=["No white line segments detected"],
            )

        consensus_lines = self._aggregate_lines(all_segments)
        if not consensus_lines:
            return CourtDetectionResult(
                corners=[], confidence=0.0,
                warnings=["No consensus lines after temporal aggregation"],
            )

        identified = self._identify_court_lines(consensus_lines)
        if identified is None:
            return CourtDetectionResult(
                corners=[], confidence=0.0,
                warnings=["Could not identify enough court lines"],
            )

        return self._fit_court_model(identified)

    def sample_frames(
        self,
        video_path: str | Path,
        start_frame: int | None = None,
        end_frame: int | None = None,
    ) -> list[np.ndarray]:
        """Sample frames from a video for court detection.

        Public API for reuse by grid search scripts that need to cache
        frames across multiple config evaluations.

        Args:
            video_path: Path to the video file.
            start_frame: First frame (default: skip_seconds from start).
            end_frame: Last frame (default: skip_seconds from end).

        Returns:
            List of BGR frames at working resolution.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return []

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if total_frames <= 0 or fps <= 0:
                return []

            skip_frames = int(self.config.skip_seconds * fps)
            if start_frame is None:
                start_frame = skip_frames
            if end_frame is None:
                end_frame = max(start_frame + 1, total_frames - skip_frames)

            return self._sample_frames(cap, start_frame, end_frame, orig_width, orig_height)
        finally:
            cap.release()

    # ── Stage 1: Frame Sampling ──────────────────────────────────────────

    def _sample_frames(
        self,
        cap: cv2.VideoCapture,
        start_frame: int,
        end_frame: int,
        orig_width: int,
        orig_height: int,
    ) -> list[np.ndarray]:
        """Sample N frames uniformly, rejecting over/underexposed ones."""
        cfg = self.config
        n = cfg.n_sample_frames
        frame_range = end_frame - start_frame
        if frame_range <= 0:
            return []

        step = max(1, frame_range // n)
        scale = cfg.working_width / orig_width
        new_h = int(orig_height * scale)

        frames: list[np.ndarray] = []
        for i in range(n):
            frame_idx = start_frame + i * step
            if frame_idx >= end_frame:
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Resize to working resolution
            frame = cv2.resize(frame, (cfg.working_width, new_h))

            # Reject over/underexposed
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            mean_brightness = float(np.mean(gray))
            if mean_brightness < 30 or mean_brightness > 240:
                logger.debug(
                    f"Frame {frame_idx}: rejected (brightness={mean_brightness:.0f})"
                )
                continue

            frames.append(frame)

        logger.info(f"Sampled {len(frames)} usable frames from {n} candidates")
        return frames

    # ── Stage 2: Per-Frame White Line Detection ──────────────────────────

    def _detect_lines_single_frame(
        self, frame: np.ndarray,
    ) -> list[tuple[float, float, float, float]]:
        """Detect court line segments in a single frame.

        Detects both white and blue/colored lines on sand background.
        Returns segments as (x1, y1, x2, y2) in normalized [0,1] coordinates.
        """
        cfg = self.config
        h, w = frame.shape[:2]

        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Adaptive sand color estimation from central strip
        strip_y1 = int(h * 0.40)
        strip_y2 = int(h * 0.70)
        strip_x1 = int(w * 0.20)
        strip_x2 = int(w * 0.80)
        central_v = hsv[strip_y1:strip_y2, strip_x1:strip_x2, 2]
        sand_median_v = float(np.median(central_v))

        # White mask: low saturation, bright relative to sand
        v_threshold = max(cfg.white_value_min, sand_median_v + cfg.white_value_offset)
        line_mask = (
            (hsv[:, :, 1] < cfg.white_saturation_max)
            & (hsv[:, :, 2] > v_threshold)
        ).astype(np.uint8) * 255

        # Blue line mask: many beach volleyball courts use blue lines
        if cfg.enable_blue_detection:
            blue_mask = (
                (hsv[:, :, 0] >= cfg.blue_hue_min)
                & (hsv[:, :, 0] <= cfg.blue_hue_max)
                & (hsv[:, :, 1] >= cfg.blue_saturation_min)
                & (hsv[:, :, 2] >= cfg.blue_value_min)
            ).astype(np.uint8) * 255
            line_mask = cv2.bitwise_or(line_mask, blue_mask)  # type: ignore[assignment]

        # Dark line mask: black/dark rope court boundaries
        # Dark ropes are desaturated (S < 50) and significantly darker than
        # sand (V < sand_V - 45) but not pure shadow (V > 35).
        if cfg.enable_dark_detection:
            dark_mask = (
                (hsv[:, :, 1] < cfg.dark_saturation_max)
                & (hsv[:, :, 2] < sand_median_v - cfg.dark_value_offset)
                & (hsv[:, :, 2] > cfg.dark_value_min)
            ).astype(np.uint8) * 255
            line_mask = cv2.bitwise_or(line_mask, dark_mask)  # type: ignore[assignment]

        # Morphological cleaning
        kernel_open = cv2.getStructuringElement(
            cv2.MORPH_RECT, (cfg.morph_open_size, cfg.morph_open_size)
        )
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_OPEN, kernel_open)  # type: ignore[assignment]

        kernel_close = cv2.getStructuringElement(
            cv2.MORPH_RECT, (cfg.morph_close_width, 1)
        )
        line_mask = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel_close)  # type: ignore[assignment]

        # Edge detection on line mask
        edges = cv2.Canny(line_mask, cfg.canny_low, cfg.canny_high)

        # Hough line detection
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180,
            threshold=cfg.hough_threshold,
            minLineLength=cfg.hough_min_length,
            maxLineGap=cfg.hough_max_gap,
        )

        if lines is None:
            return []

        segments: list[tuple[float, float, float, float]] = []
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Filter by pixel length
            px_len = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            if px_len < cfg.min_segment_px:
                continue

            # Filter near-vertical lines
            angle = segment_angle_deg(x1 / w, y1 / h, x2 / w, y2 / h)
            if angle > cfg.max_angle_from_horizontal:
                continue

            # Normalize to [0,1]
            segments.append((x1 / w, y1 / h, x2 / w, y2 / h))

        return segments

    def _detect_lines_all_frames(
        self, frames: list[np.ndarray],
    ) -> list[list[tuple[float, float, float, float]]]:
        """Detect lines in all frames. Returns list of segment lists per frame."""
        all_segments: list[list[tuple[float, float, float, float]]] = []
        for frame in frames:
            segs = self._detect_lines_single_frame(frame)
            all_segments.append(segs)
        total = sum(len(s) for s in all_segments)
        logger.info(f"Detected {total} segments across {len(frames)} frames")
        return all_segments

    # ── Stage 3: Temporal Aggregation ────────────────────────────────────

    def _aggregate_lines(
        self,
        all_segments: list[list[tuple[float, float, float, float]]],
    ) -> list[tuple[list[tuple[float, float, float, float]], int]]:
        """Cluster segments across frames using DBSCAN in (rho, theta) space.

        Returns:
            List of (segment_cluster, temporal_support) tuples.
        """
        cfg = self.config

        # Collect all segments with their (rho, theta) and frame index
        features: list[tuple[float, float]] = []
        segment_data: list[tuple[float, float, float, float]] = []
        frame_indices: list[int] = []

        for frame_idx, frame_segs in enumerate(all_segments):
            for seg in frame_segs:
                rho, theta = segment_to_rho_theta(*seg)
                features.append((rho, theta * cfg.theta_scale))
                segment_data.append(seg)
                frame_indices.append(frame_idx)

        if len(features) < cfg.dbscan_min_samples:
            return []

        # DBSCAN clustering
        from sklearn.cluster import DBSCAN

        feature_matrix = np.array(features, dtype=np.float64)
        db = DBSCAN(eps=cfg.dbscan_eps, min_samples=cfg.dbscan_min_samples).fit(feature_matrix)

        # Group segments by cluster
        clusters: dict[int, list[int]] = {}
        for idx, label in enumerate(db.labels_):
            if label == -1:
                continue
            clusters.setdefault(label, []).append(idx)

        # Compute temporal support and filter
        results: list[tuple[list[tuple[float, float, float, float]], int]] = []
        for cluster_indices in clusters.values():
            frames_seen = set(frame_indices[i] for i in cluster_indices)
            support = len(frames_seen)

            if support < cfg.min_temporal_support:
                continue

            segs = [segment_data[i] for i in cluster_indices]
            results.append((segs, support))

        # Sort by support (strongest first)
        results.sort(key=lambda x: -x[1])
        logger.info(
            f"Temporal aggregation: {len(results)} consensus lines "
            f"from {len(clusters)} clusters"
        )
        return results

    # ── Stage 4: Court Line Identification ───────────────────────────────

    def _identify_court_lines(
        self,
        consensus_lines: list[tuple[list[tuple[float, float, float, float]], int]],
    ) -> dict[str, tuple[DetectedLine, list[tuple[float, float, float, float]]]] | None:
        """Classify consensus lines as court lines.

        Returns dict mapping line label to (DetectedLine, raw_segments), or None
        if minimum requirements not met (far baseline + at least 1 sideline).
        """
        cfg = self.config

        # Compute median line for each cluster
        candidates: list[dict[str, Any]] = []
        for segs, support in consensus_lines:
            median_line = segments_to_median_line(segs)
            if median_line is None:
                continue

            p1, p2 = median_line
            angle = segment_angle_deg(p1[0], p1[1], p2[0], p2[1])
            mid_x = (p1[0] + p2[0]) / 2.0
            mid_y = (p1[1] + p2[1]) / 2.0
            # X-span: horizontal coverage of this line
            x_span = abs(p2[0] - p1[0])
            # Near endpoint: the endpoint with larger Y (closer to camera)
            near_end = p2 if p2[1] > p1[1] else p1
            far_end = p1 if p2[1] > p1[1] else p2

            candidates.append({
                "p1": p1,
                "p2": p2,
                "angle": angle,
                "mid_x": mid_x,
                "mid_y": mid_y,
                "x_span": x_span,
                "near_end": near_end,
                "far_end": far_end,
                "support": support,
                "segments": segs,
            })

        if not candidates:
            return None

        # Classify: horizontal lines vs converging (sideline) lines
        horizontal: list[dict[str, Any]] = []
        converging: list[dict[str, Any]] = []

        for c in candidates:
            if c["angle"] <= cfg.horizontal_angle_max:
                horizontal.append(c)
            elif cfg.sideline_angle_min <= c["angle"] <= cfg.sideline_angle_max:
                converging.append(c)

        logger.info(
            f"Line classification: {len(horizontal)} horizontal, "
            f"{len(converging)} converging"
        )

        # ── Net filtering ──
        # The volleyball net is a strong horizontal line (high support, wide span)
        # typically in the upper half of the frame. We detect it and exclude it.
        # Net characteristics: spans >50% of frame width, high support,
        # and is the topmost (smallest Y) wide horizontal line.
        # IMPORTANT: only flag as net if there's at least one more horizontal line
        # below it (otherwise the topmost line is likely the far baseline itself).
        net_line = None
        wide_horizontals = sorted(
            [h for h in horizontal if h["x_span"] > 0.50 and h["mid_y"] < 0.55],
            key=lambda x: x["mid_y"],
        )
        if wide_horizontals:
            candidate_net = wide_horizontals[0]
            # Check: is there another horizontal below it that could be far baseline?
            has_line_below = any(
                h["mid_y"] > candidate_net["mid_y"] + 0.03
                and h["mid_y"] < 0.75
                for h in horizontal
                if h is not candidate_net
            )
            if has_line_below:
                net_line = candidate_net

        if net_line:
            logger.info(
                f"Net detected at y={net_line['mid_y']:.3f} "
                f"(span={net_line['x_span']:.2f}, support={net_line['support']})"
            )

        # ── Far baseline selection ──
        # The far baseline is a horizontal court line in the LOWER part of the
        # court surface (but above center of frame). It should NOT be the net.
        # Strategy: find horizontal lines below the net (or in lower half if no net),
        # prefer lines in the court surface area (y=0.35-0.65).
        far_baseline = None
        net_y = net_line["mid_y"] if net_line else 0.0

        # Sort horizontals by Y (topmost first), prefer those below the net
        court_horizontals = []
        for h in horizontal:
            if net_line and h is net_line:
                continue  # Skip the net itself
            # Must be below net (or in upper 65% if no net detected)
            if h["mid_y"] < net_y + 0.02:
                continue  # Too close to or above the net
            if h["mid_y"] > 0.75:
                continue  # Too far down — likely near baseline, not far
            court_horizontals.append(h)

        # Pick the highest (smallest Y) court horizontal with good support
        court_horizontals.sort(key=lambda x: x["mid_y"])
        if court_horizontals:
            far_baseline = court_horizontals[0]
        else:
            # Fallback: just pick the strongest horizontal in upper frame
            for h in sorted(horizontal, key=lambda x: -x["support"]):
                if h is not net_line and h["mid_y"] < 0.65:
                    far_baseline = h
                    break

        if far_baseline is None:
            logger.warning("No far baseline found")
            return None

        logger.info(
            f"Far baseline at y={far_baseline['mid_y']:.3f} "
            f"(support={far_baseline['support']})"
        )

        # ── Sideline classification ──
        # Use the NEAR endpoint (bottom of the line, highest Y) to determine
        # left vs right. This is more robust than midpoint because perspective
        # can shift the midpoint across center for sidelines that run from
        # far-left-upper to near-left-lower.
        frame_center_x = 0.5
        left_sidelines = [
            c for c in converging if c["near_end"][0] < frame_center_x
        ]
        right_sidelines = [
            c for c in converging if c["near_end"][0] >= frame_center_x
        ]

        # Sort by support
        left_sidelines.sort(key=lambda x: -x["support"])
        right_sidelines.sort(key=lambda x: -x["support"])

        left_sideline = left_sidelines[0] if left_sidelines else None
        right_sideline = right_sidelines[0] if right_sidelines else None

        # Validate sidelines converge upward (toward vanishing point)
        if left_sideline and right_sideline:
            vp = line_intersection(
                left_sideline["p1"], left_sideline["p2"],
                right_sideline["p1"], right_sideline["p2"],
            )
            if vp is not None and vp[1] > far_baseline["mid_y"]:
                # Vanishing point below far baseline — wrong geometry
                logger.warning(
                    f"Sidelines vanishing point ({vp[0]:.2f}, {vp[1]:.2f}) "
                    f"is below far baseline (y={far_baseline['mid_y']:.2f})"
                )
                # Try to recover: pick the sideline with better support
                if left_sideline["support"] > right_sideline["support"]:
                    right_sideline = None
                else:
                    left_sideline = None

        if left_sideline is None and right_sideline is None:
            logger.warning("No sidelines found")
            return None

        # Center line: horizontal, below far baseline, parallel to it
        center_line = None
        far_angle = far_baseline["angle"]
        skip_lines: list[dict[str, Any]] = [far_baseline]
        if net_line:
            skip_lines.append(net_line)

        for h in sorted(horizontal, key=lambda x: -x["support"]):
            if any(h is s for s in skip_lines):
                continue
            if h["mid_y"] <= far_baseline["mid_y"]:
                continue
            if h["mid_y"] > 0.80:
                continue  # Too low to be center line
            angle_diff = abs(h["angle"] - far_angle)
            if angle_diff <= cfg.center_line_parallel_tolerance:
                center_line = h
                break

        # Near baseline: horizontal, below center line (if found)
        near_baseline = None
        ref_y = center_line["mid_y"] if center_line else far_baseline["mid_y"] + 0.15
        skip_lines_near = skip_lines + ([center_line] if center_line else [])
        for h in sorted(horizontal, key=lambda x: -x["support"]):
            if any(h is s for s in skip_lines_near):
                continue
            if h["mid_y"] <= ref_y:
                continue
            angle_diff = abs(h["angle"] - far_angle)
            if angle_diff <= cfg.center_line_parallel_tolerance:
                near_baseline = h
                break

        # Build result
        def _make_detected(label: str, c: dict[str, Any]) -> DetectedLine:
            return DetectedLine(
                label=label,
                p1=c["p1"],
                p2=c["p2"],
                support=c["support"],
                angle_deg=c["angle"],
            )

        result: dict[str, tuple[DetectedLine, list[tuple[float, float, float, float]]]] = {}
        result["far_baseline"] = (_make_detected("far_baseline", far_baseline), far_baseline["segments"])
        if left_sideline:
            result["left_sideline"] = (_make_detected("left_sideline", left_sideline), left_sideline["segments"])
        if right_sideline:
            result["right_sideline"] = (_make_detected("right_sideline", right_sideline), right_sideline["segments"])
        if center_line:
            result["center_line"] = (_make_detected("center_line", center_line), center_line["segments"])
        if near_baseline:
            result["near_baseline"] = (_make_detected("near_baseline", near_baseline), near_baseline["segments"])

        lines_found = list(result.keys())
        logger.info(f"Identified court lines: {lines_found}")

        return result

    # ── Stage 5: Geometric Court Model Fitting ───────────────────────────

    def _fit_court_model(
        self,
        identified: dict[str, tuple[DetectedLine, list[tuple[float, float, float, float]]]],
    ) -> CourtDetectionResult:
        """Fit 4-corner court model from identified lines."""
        warnings: list[str] = []
        detected_lines = [dl for dl, _ in identified.values()]

        far_bl = identified["far_baseline"][0]
        left_sl = identified.get("left_sideline", (None, None))[0]
        right_sl = identified.get("right_sideline", (None, None))[0]
        center_ln = identified.get("center_line", (None, None))[0]
        near_bl = identified.get("near_baseline", (None, None))[0]

        # ── Far corners: intersection of far baseline with sidelines ──
        far_left: tuple[float, float] | None = None
        far_right: tuple[float, float] | None = None

        if left_sl:
            far_left = line_intersection(far_bl.p1, far_bl.p2, left_sl.p1, left_sl.p2)
        if right_sl:
            far_right = line_intersection(far_bl.p1, far_bl.p2, right_sl.p1, right_sl.p2)

        # If only one sideline, mirror using far baseline midpoint
        if far_left and not far_right:
            warnings.append("Right sideline not detected, mirroring from left")
            bl_mid_x = (far_bl.p1[0] + far_bl.p2[0]) / 2.0
            bl_mid_y = (far_bl.p1[1] + far_bl.p2[1]) / 2.0
            far_right = (2 * bl_mid_x - far_left[0], 2 * bl_mid_y - far_left[1])
            # Create a synthetic right sideline for near corner computation
            if not right_sl and left_sl:
                # Mirror left sideline about far baseline midpoint
                right_sl_p1 = (2 * bl_mid_x - left_sl.p1[0], left_sl.p1[1])
                right_sl_p2 = (2 * bl_mid_x - left_sl.p2[0], left_sl.p2[1])
                right_sl = DetectedLine(
                    label="right_sideline_mirrored", p1=right_sl_p1, p2=right_sl_p2,
                    support=0, angle_deg=left_sl.angle_deg,
                )

        elif far_right and not far_left:
            warnings.append("Left sideline not detected, mirroring from right")
            bl_mid_x = (far_bl.p1[0] + far_bl.p2[0]) / 2.0
            bl_mid_y = (far_bl.p1[1] + far_bl.p2[1]) / 2.0
            far_left = (2 * bl_mid_x - far_right[0], 2 * bl_mid_y - far_right[1])
            if not left_sl and right_sl:
                left_sl_p1 = (2 * bl_mid_x - right_sl.p1[0], right_sl.p1[1])
                left_sl_p2 = (2 * bl_mid_x - right_sl.p2[0], right_sl.p2[1])
                left_sl = DetectedLine(
                    label="left_sideline_mirrored", p1=left_sl_p1, p2=left_sl_p2,
                    support=0, angle_deg=right_sl.angle_deg,
                )

        elif not far_left and not far_right:
            return CourtDetectionResult(
                corners=[], confidence=0.0,
                detected_lines=detected_lines,
                warnings=["Cannot compute far corners: no sidelines detected"],
            )

        # Unreachable: all code paths above ensure both are set or return early
        if far_left is None or far_right is None:
            return CourtDetectionResult(
                corners=[], confidence=0.0,
                detected_lines=detected_lines,
                warnings=["Cannot compute far corners"],
            )

        # ── Near corners ──
        near_left: tuple[float, float] | None = None
        near_right: tuple[float, float] | None = None
        near_method = "none"

        # Strategy 1: Near baseline detected
        if near_bl and left_sl and right_sl:
            near_left = line_intersection(near_bl.p1, near_bl.p2, left_sl.p1, left_sl.p2)
            near_right = line_intersection(near_bl.p1, near_bl.p2, right_sl.p1, right_sl.p2)
            if near_left and near_right:
                near_method = "near_baseline"

        # Strategy 2: Center line + harmonic conjugate
        if near_method == "none" and center_ln and left_sl and right_sl:
            # Find where center line intersects each sideline
            center_left = line_intersection(
                center_ln.p1, center_ln.p2, left_sl.p1, left_sl.p2
            )
            center_right = line_intersection(
                center_ln.p1, center_ln.p2, right_sl.p1, right_sl.p2
            )

            # Compute vanishing point from sidelines
            sideline_lines: list[tuple[tuple[float, float], tuple[float, float]]] = []
            if left_sl:
                sideline_lines.append((left_sl.p1, left_sl.p2))
            if right_sl:
                sideline_lines.append((right_sl.p1, right_sl.p2))

            vp = compute_vanishing_point(sideline_lines)

            if center_left and vp:
                near_left = harmonic_conjugate(far_left, center_left, vp)
            if center_right and vp:
                near_right = harmonic_conjugate(far_right, center_right, vp)

            if near_left and near_right:
                near_method = "harmonic_conjugate"

        # Strategy 3: Aspect ratio extrapolation from sideline convergence
        if near_method == "none" and left_sl and right_sl:
            vp = line_intersection(left_sl.p1, left_sl.p2, right_sl.p1, right_sl.p2)
            if vp:
                # Use aspect ratio: the court is 1:2, so near corners are
                # at distance 2x from far baseline as measured along sidelines.
                # We use the parametric position along each sideline:
                # far corner is at t=0, vanishing point at t=1.
                # For the aspect ratio to hold, we need to find t_near such that
                # the court length/width ratio is correct.

                # Along left sideline
                dx_l = vp[0] - far_left[0]
                dy_l = vp[1] - far_left[1]
                far_baseline_len = math.sqrt(
                    (far_right[0] - far_left[0]) ** 2
                    + (far_right[1] - far_left[1]) ** 2
                )

                if far_baseline_len > 0.01:
                    # The far baseline width in image corresponds to 8m.
                    # Court length is 16m. Due to perspective, we estimate
                    # near corners by extrapolating sidelines away from VP.
                    # Use negative t (opposite direction from VP).
                    # Heuristic: near corner is at t_near = -(far_baseline_len * aspect_ratio) / sideline_length
                    sl_len_l = math.sqrt(dx_l ** 2 + dy_l ** 2)
                    if sl_len_l > 0.01:
                        # Scale factor based on perspective
                        t_near_l = -(far_baseline_len * self.config.aspect_ratio) / sl_len_l
                        near_left = (far_left[0] + t_near_l * dx_l, far_left[1] + t_near_l * dy_l)

                dx_r = vp[0] - far_right[0]
                dy_r = vp[1] - far_right[1]
                sl_len_r = math.sqrt(dx_r ** 2 + dy_r ** 2)
                if sl_len_r > 0.01:
                    t_near_r = -(far_baseline_len * self.config.aspect_ratio) / sl_len_r
                    near_right = (far_right[0] + t_near_r * dx_r, far_right[1] + t_near_r * dy_r)

                if near_left and near_right:
                    near_method = "aspect_ratio"
                    warnings.append("Near corners estimated from aspect ratio (lower confidence)")

        if near_left is None or near_right is None:
            return CourtDetectionResult(
                corners=[], confidence=0.0,
                detected_lines=detected_lines,
                warnings=["Cannot determine near corners"],
            )

        # ── Validate quadrilateral ──
        valid, validation_warnings = self._validate_quadrilateral(
            far_left, far_right, near_right, near_left
        )
        warnings.extend(validation_warnings)

        # Build corners in DB format: near-left, near-right, far-right, far-left
        corners = [
            {"x": near_left[0], "y": near_left[1]},
            {"x": near_right[0], "y": near_right[1]},
            {"x": far_right[0], "y": far_right[1]},
            {"x": far_left[0], "y": far_left[1]},
        ]

        # ── Stage 6: Confidence scoring ──
        confidence = self._compute_confidence(
            identified, near_method, valid,
        )

        return CourtDetectionResult(
            corners=corners,
            confidence=confidence,
            detected_lines=detected_lines,
            warnings=warnings,
        )

    def _validate_quadrilateral(
        self,
        far_left: tuple[float, float],
        far_right: tuple[float, float],
        near_right: tuple[float, float],
        near_left: tuple[float, float],
    ) -> tuple[bool, list[str]]:
        """Validate the detected quadrilateral has reasonable properties."""
        warnings: list[str] = []

        # Check for self-intersection using cross products
        # Order: near-left, near-right, far-right, far-left
        pts = [near_left, near_right, far_right, far_left]

        # All cross products at vertices should have same sign (convex)
        n = len(pts)
        signs: list[float] = []
        for i in range(n):
            cp = cross2d(pts[i], pts[(i + 1) % n], pts[(i + 2) % n])
            signs.append(cp)

        all_positive = all(s > 0 for s in signs)
        all_negative = all(s < 0 for s in signs)
        is_convex = all_positive or all_negative

        if not is_convex:
            warnings.append("Detected quadrilateral is not convex")

        # Far side should be smaller than near side (perspective)
        far_width = math.sqrt(
            (far_right[0] - far_left[0]) ** 2 + (far_right[1] - far_left[1]) ** 2
        )
        near_width = math.sqrt(
            (near_right[0] - near_left[0]) ** 2 + (near_right[1] - near_left[1]) ** 2
        )
        if far_width > near_width * 1.2:
            warnings.append(
                f"Far side ({far_width:.3f}) wider than near side ({near_width:.3f})"
            )

        # Far side should be above near side (in image Y: smaller Y = higher)
        far_mid_y = (far_left[1] + far_right[1]) / 2
        near_mid_y = (near_left[1] + near_right[1]) / 2
        if far_mid_y > near_mid_y:
            warnings.append("Far baseline below near baseline in image")

        return (is_convex and far_mid_y <= near_mid_y, warnings)

    def _compute_confidence(
        self,
        identified: dict[str, tuple[DetectedLine, list[tuple[float, float, float, float]]]],
        near_method: str,
        geometry_valid: bool,
    ) -> float:
        """Compute confidence score 0-1."""
        # Lines detected (weight 0.30)
        expected_lines = {"far_baseline", "left_sideline", "right_sideline", "center_line"}
        lines_found = sum(1 for k in expected_lines if k in identified)
        if "near_baseline" in identified:
            lines_found = min(4, lines_found + 1)
        line_score = lines_found / 4.0

        # Temporal support (weight 0.25)
        supports = [dl.support for dl, _ in identified.values()]
        max_support = self.config.n_sample_frames
        median_support = float(np.median(supports)) if supports else 0.0
        support_score = min(1.0, median_support / max(1, max_support))

        # Geometric consistency (weight 0.25)
        geo_score = 1.0 if geometry_valid else 0.3
        if near_method == "near_baseline":
            geo_score *= 1.0
        elif near_method == "harmonic_conjugate":
            geo_score *= 0.85
        elif near_method == "aspect_ratio":
            geo_score *= 0.5

        # Far baseline coverage (weight 0.20)
        far_bl = identified.get("far_baseline")
        if far_bl:
            p1, p2 = far_bl[0].p1, far_bl[0].p2
            bl_len = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
            coverage_score = min(1.0, bl_len / 0.5)  # Expect baseline ~50% of frame width
        else:
            coverage_score = 0.0

        confidence = (
            0.30 * line_score
            + 0.25 * support_score
            + 0.25 * geo_score
            + 0.20 * coverage_score
        )

        return round(min(1.0, max(0.0, confidence)), 3)

    # ── Debug Visualization ──────────────────────────────────────────────

    def create_debug_image(
        self,
        frame: np.ndarray,
        result: CourtDetectionResult,
    ) -> np.ndarray:
        """Create debug visualization showing detected lines and court overlay.

        Args:
            frame: BGR frame at any resolution.
            result: Detection result.

        Returns:
            BGR frame with overlaid visualizations.
        """
        overlay = frame.copy()
        h, w = overlay.shape[:2]

        # Draw detected lines
        line_colors = {
            "far_baseline": (0, 0, 255),      # Red (BGR)
            "left_sideline": (255, 0, 0),      # Blue (BGR)
            "right_sideline": (0, 255, 0),     # Green (BGR)
            "center_line": (255, 255, 0),      # Cyan (BGR)
            "near_baseline": (0, 255, 255),    # Yellow (BGR)
        }
        for dl in result.detected_lines:
            color = line_colors.get(dl.label, (128, 128, 128))
            p1 = (int(dl.p1[0] * w), int(dl.p1[1] * h))
            p2 = (int(dl.p2[0] * w), int(dl.p2[1] * h))
            cv2.line(overlay, p1, p2, color, 2)

            # Label
            mid = ((p1[0] + p2[0]) // 2, (p1[1] + p2[1]) // 2)
            cv2.putText(
                overlay, f"{dl.label} (s={dl.support})",
                (mid[0] - 40, mid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )

        # Draw court corners and edges if detected
        if result.corners and len(result.corners) == 4:
            pts = [(c["x"], c["y"]) for c in result.corners]
            for i in range(4):
                cp1 = pts[i]
                cp2 = pts[(i + 1) % 4]
                x1, y1 = int(cp1[0] * w), int(cp1[1] * h)
                x2, y2 = int(cp2[0] * w), int(cp2[1] * h)
                cv2.line(overlay, (x1, y1), (x2, y2), (0, 165, 255), 3)

            # Draw corner markers
            corner_labels = ["NL", "NR", "FR", "FL"]
            for i, (cx, cy) in enumerate(pts):
                px, py = int(cx * w), int(cy * h)
                # Only draw if within expanded frame bounds
                if -w < px < 2 * w and -h < py < 2 * h:
                    cv2.circle(overlay, (px, py), 8, (0, 165, 255), -1)
                    cv2.putText(
                        overlay, corner_labels[i],
                        (px + 12, py + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2,
                    )

        # Confidence label
        cv2.putText(
            overlay,
            f"Confidence: {result.confidence:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )

        # Warnings
        for i, warning in enumerate(result.warnings[:3]):
            cv2.putText(
                overlay,
                f"! {warning}",
                (10, h - 20 - i * 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
            )

        return overlay
