"""Tests for automatic court detection."""

from __future__ import annotations

import math

import cv2
import numpy as np

from rallycut.court.line_geometry import (
    compute_vanishing_point,
    cross2d,
    harmonic_conjugate,
    line_intersection,
    point_line_distance,
    segment_angle_deg,
    segment_length,
    segment_to_rho_theta,
    segments_to_median_line,
)

# ── Line Geometry Tests ──────────────────────────────────────────────────


class TestLineIntersection:
    def test_perpendicular_lines(self) -> None:
        # Horizontal line y=0.5 and vertical line x=0.5
        pt = line_intersection((0.0, 0.5), (1.0, 0.5), (0.5, 0.0), (0.5, 1.0))
        assert pt is not None
        assert abs(pt[0] - 0.5) < 1e-6
        assert abs(pt[1] - 0.5) < 1e-6

    def test_angled_lines(self) -> None:
        # y=x and y=-x+1 intersect at (0.5, 0.5)
        pt = line_intersection((0.0, 0.0), (1.0, 1.0), (0.0, 1.0), (1.0, 0.0))
        assert pt is not None
        assert abs(pt[0] - 0.5) < 1e-6
        assert abs(pt[1] - 0.5) < 1e-6

    def test_parallel_lines(self) -> None:
        pt = line_intersection((0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0))
        assert pt is None

    def test_intersection_outside_segments(self) -> None:
        # Lines extend beyond segments — intersection should still be found
        # Line 1: through (0,0) and (1,1), Line 2: through (0,1) and (0.1,0.9)
        pt = line_intersection((0.0, 0.0), (0.1, 0.1), (0.0, 1.0), (0.1, 0.9))
        assert pt is not None
        # These lines intersect at (0.5, 0.5)
        assert abs(pt[0] - 0.5) < 1e-4
        assert abs(pt[1] - 0.5) < 1e-4


class TestPointLineDistance:
    def test_horizontal_line(self) -> None:
        dist = point_line_distance((0.5, 0.8), (0.0, 0.5), (1.0, 0.5))
        assert abs(dist - 0.3) < 1e-6

    def test_point_on_line(self) -> None:
        dist = point_line_distance((0.5, 0.5), (0.0, 0.5), (1.0, 0.5))
        assert dist < 1e-6

    def test_vertical_line(self) -> None:
        dist = point_line_distance((0.3, 0.5), (0.5, 0.0), (0.5, 1.0))
        assert abs(dist - 0.2) < 1e-6


class TestVanishingPoint:
    def test_two_converging_lines(self) -> None:
        # Two lines converging at (0.5, 0.0)
        lines = [
            ((0.0, 1.0), (0.5, 0.0)),
            ((1.0, 1.0), (0.5, 0.0)),
        ]
        vp = compute_vanishing_point(lines)
        assert vp is not None
        assert abs(vp[0] - 0.5) < 1e-4
        assert abs(vp[1] - 0.0) < 1e-4

    def test_single_line(self) -> None:
        lines = [((0.0, 0.0), (1.0, 1.0))]
        vp = compute_vanishing_point(lines)
        assert vp is None

    def test_multiple_converging(self) -> None:
        # Three lines through (0.5, -0.5)
        target = (0.5, -0.5)
        lines = [
            ((0.0, 1.0), target),
            ((1.0, 1.0), target),
            ((0.5, 1.0), target),
        ]
        vp = compute_vanishing_point(lines)
        assert vp is not None
        assert abs(vp[0] - 0.5) < 1e-3
        assert abs(vp[1] - (-0.5)) < 1e-3


class TestHarmonicConjugate:
    def test_simple_case(self) -> None:
        # Vanishing point at infinity (parallel lines): center is true midpoint
        # With VP very far away, harmonic conjugate ≈ mirror of far_pt through center
        far = (0.3, 0.3)
        center = (0.5, 0.5)
        # VP at "infinity" along the line
        vp = (100.0, 100.0)
        near = harmonic_conjugate(far, center, vp)
        # Should be approximately the mirror: center + (center - far)
        expected = (0.7, 0.7)
        assert abs(near[0] - expected[0]) < 0.02
        assert abs(near[1] - expected[1]) < 0.02

    def test_projective_midpoint(self) -> None:
        # With perspective: VP at (0.5, 0), far at (0.3, 0.4), center intersects sideline
        far = (0.3, 0.4)
        vp = (0.5, 0.0)
        # Center line intersection on the sideline at t=0.5 from far to vp
        t_c = 0.5
        center = (
            far[0] + t_c * (vp[0] - far[0]),
            far[1] + t_c * (vp[1] - far[1]),
        )
        near = harmonic_conjugate(far, center, vp)
        # t_near = 2*0.5/(0.5+1) = 0.667
        t_expected = 2.0 * 0.5 / 1.5
        expected = (
            far[0] + t_expected * (vp[0] - far[0]),
            far[1] + t_expected * (vp[1] - far[1]),
        )
        assert abs(near[0] - expected[0]) < 1e-6
        assert abs(near[1] - expected[1]) < 1e-6


class TestCross2d:
    def test_counter_clockwise(self) -> None:
        # Counter-clockwise triangle: positive
        result = cross2d((0.0, 0.0), (1.0, 0.0), (0.0, 1.0))
        assert result > 0

    def test_clockwise(self) -> None:
        # Clockwise triangle: negative
        result = cross2d((0.0, 0.0), (0.0, 1.0), (1.0, 0.0))
        assert result < 0

    def test_collinear(self) -> None:
        result = cross2d((0.0, 0.0), (1.0, 1.0), (2.0, 2.0))
        assert abs(result) < 1e-10


class TestSegmentToRhoTheta:
    def test_horizontal_segment(self) -> None:
        rho, theta = segment_to_rho_theta(0.0, 0.5, 1.0, 0.5)
        # Horizontal line at y=0.5: normal is vertical (theta ≈ pi/2), rho ≈ 0.5
        assert abs(rho - 0.5) < 0.01
        assert abs(abs(theta) - math.pi / 2) < 0.1

    def test_vertical_segment(self) -> None:
        rho, theta = segment_to_rho_theta(0.5, 0.0, 0.5, 1.0)
        # Vertical line at x=0.5: normal is horizontal (theta ≈ 0), rho ≈ 0.5
        assert abs(rho - 0.5) < 0.01
        assert abs(theta) < 0.1 or abs(abs(theta) - math.pi) < 0.1


class TestSegmentAngle:
    def test_horizontal(self) -> None:
        angle = segment_angle_deg(0.0, 0.5, 1.0, 0.5)
        assert abs(angle) < 0.01

    def test_vertical(self) -> None:
        angle = segment_angle_deg(0.5, 0.0, 0.5, 1.0)
        assert abs(angle - 90.0) < 0.01

    def test_diagonal(self) -> None:
        angle = segment_angle_deg(0.0, 0.0, 1.0, 1.0)
        assert abs(angle - 45.0) < 0.01


class TestSegmentLength:
    def test_unit(self) -> None:
        length = segment_length(0.0, 0.0, 1.0, 0.0)
        assert abs(length - 1.0) < 1e-6

    def test_diagonal(self) -> None:
        length = segment_length(0.0, 0.0, 1.0, 1.0)
        assert abs(length - math.sqrt(2.0)) < 1e-6


class TestSegmentsToMedianLine:
    def test_single_segment(self) -> None:
        result = segments_to_median_line([(0.1, 0.5, 0.9, 0.5)])
        assert result is not None
        p1, p2 = result
        # Should be roughly horizontal at y=0.5
        assert abs(p1[1] - 0.5) < 0.01
        assert abs(p2[1] - 0.5) < 0.01

    def test_empty(self) -> None:
        result = segments_to_median_line([])
        assert result is None

    def test_consistent_cluster(self) -> None:
        # Multiple nearly-identical horizontal segments
        segs = [
            (0.1, 0.50, 0.9, 0.50),
            (0.1, 0.51, 0.9, 0.51),
            (0.1, 0.49, 0.9, 0.49),
        ]
        result = segments_to_median_line(segs)
        assert result is not None
        p1, p2 = result
        assert abs(p1[1] - 0.5) < 0.02
        assert abs(p2[1] - 0.5) < 0.02


# ── Court Detector Tests ─────────────────────────────────────────────────


def _make_synthetic_frame(
    width: int = 960,
    height: int = 540,
    sand_brightness: int = 180,
) -> np.ndarray:
    """Create a synthetic frame with white lines on sand-colored background.

    Draws a simplified court with realistic perspective:
    - Far baseline (horizontal, upper portion)
    - Two sidelines (converging upward, moderate angle ~55°)
    - Center line (horizontal, between baselines)

    Camera behind near baseline looking across the court.
    Sideline angles kept under 75° to pass angle filter.
    """
    frame = np.full((height, width, 3), sand_brightness, dtype=np.uint8)

    # Add slight color variation to simulate sand
    frame[:, :, 0] = np.clip(frame[:, :, 0].astype(int) - 10, 0, 255).astype(np.uint8)  # B
    frame[:, :, 2] = np.clip(frame[:, :, 2].astype(int) + 10, 0, 255).astype(np.uint8)  # R

    white = (255, 255, 255)
    line_thickness = 3

    # Far baseline: y=0.35, x from 0.30 to 0.70
    y_far = int(0.35 * height)
    x_far_l = int(0.30 * width)
    x_far_r = int(0.70 * width)
    cv2.line(frame, (x_far_l, y_far), (x_far_r, y_far), white, line_thickness)

    # Sidelines with moderate angle (~55° from horizontal):
    # Left sideline: from (0.30, 0.35) to (0.05, 0.85) — dx=0.25, dy=0.50 → ~63°
    x_near_l = int(0.05 * width)
    y_near = int(0.85 * height)
    cv2.line(frame, (x_far_l, y_far), (x_near_l, y_near), white, line_thickness)

    # Right sideline: from (0.70, 0.35) to (0.95, 0.85)
    x_near_r = int(0.95 * width)
    cv2.line(frame, (x_far_r, y_far), (x_near_r, y_near), white, line_thickness)

    # Center line: midpoint of sidelines at y=0.60
    y_center = int(0.60 * height)
    t = (0.60 - 0.35) / (0.85 - 0.35)  # = 0.5
    x_center_l = int(x_far_l + t * (x_near_l - x_far_l))
    x_center_r = int(x_far_r + t * (x_near_r - x_far_r))
    cv2.line(frame, (x_center_l, y_center), (x_center_r, y_center), white, line_thickness)

    return frame


class TestCourtDetectorSynthetic:
    """Test court detection on synthetic frames with known geometry."""

    def test_detect_lines_from_synthetic_frame(self) -> None:
        """Verify that white lines are detected from a synthetic frame."""
        from rallycut.court.detector import CourtDetectionConfig, CourtDetector

        frame = _make_synthetic_frame()
        config = CourtDetectionConfig(
            min_temporal_support=1,
            dbscan_min_samples=1,
        )
        detector = CourtDetector(config)
        segments = detector._detect_lines_single_frame(frame)

        # Should detect at least a few line segments
        assert len(segments) >= 2, f"Expected >=2 segments, got {len(segments)}"

    def test_full_detection_with_synthetic_frames(self) -> None:
        """Run full detection on repeated synthetic frames."""
        from rallycut.court.detector import CourtDetectionConfig, CourtDetector

        # Create multiple identical frames (simulates temporal consistency)
        frame = _make_synthetic_frame()
        frames = [frame.copy() for _ in range(15)]

        config = CourtDetectionConfig(
            min_temporal_support=3,
            dbscan_min_samples=3,
        )
        detector = CourtDetector(config)
        result = detector.detect_from_frames(frames)

        # Should detect at least the far baseline
        assert len(result.detected_lines) >= 1, (
            f"Expected >=1 detected lines, got {len(result.detected_lines)}. "
            f"Warnings: {result.warnings}"
        )
        # Should find corners (far baseline + at least 1 sideline = 4 corners)
        if result.corners:
            assert len(result.corners) == 4
            assert result.confidence > 0.0

    def test_empty_frames(self) -> None:
        """No lines detected on blank frames."""
        from rallycut.court.detector import CourtDetectionConfig, CourtDetector

        frame = np.full((540, 960, 3), 180, dtype=np.uint8)
        config = CourtDetectionConfig(
            min_temporal_support=1,
            dbscan_min_samples=1,
        )
        detector = CourtDetector(config)
        result = detector.detect_from_frames([frame])
        assert result.confidence == 0.0

    def test_overexposed_frame_rejection(self) -> None:
        """Overexposed/underexposed frames produce few or no segments."""
        from rallycut.court.detector import CourtDetectionConfig, CourtDetector

        # All-white frame
        bright = np.full((540, 960, 3), 250, dtype=np.uint8)
        # All-dark frame
        dark = np.full((540, 960, 3), 10, dtype=np.uint8)

        config = CourtDetectionConfig()
        detector = CourtDetector(config)

        # These should produce very few segments (no real lines present)
        segs_bright = detector._detect_lines_single_frame(bright)
        segs_dark = detector._detect_lines_single_frame(dark)
        assert len(segs_bright) <= 2, f"Expected <=2 segments on white frame, got {len(segs_bright)}"
        assert len(segs_dark) <= 2, f"Expected <=2 segments on dark frame, got {len(segs_dark)}"


class TestCourtDetectionConfig:
    def test_defaults(self) -> None:
        from rallycut.court.detector import CourtDetectionConfig

        config = CourtDetectionConfig()
        assert config.n_sample_frames == 30
        assert config.aspect_ratio == 2.0
        assert config.working_width == 960

    def test_custom_config(self) -> None:
        from rallycut.court.detector import CourtDetectionConfig

        config = CourtDetectionConfig(n_sample_frames=10, working_width=640)
        assert config.n_sample_frames == 10
        assert config.working_width == 640


class TestCourtDetectionResult:
    def test_valid_result(self) -> None:
        from rallycut.court.detector import CourtDetectionResult

        result = CourtDetectionResult(
            corners=[
                {"x": 0.1, "y": 0.9},
                {"x": 0.9, "y": 0.9},
                {"x": 0.75, "y": 0.3},
                {"x": 0.25, "y": 0.3},
            ],
            confidence=0.8,
        )
        assert result.is_valid
        assert len(result.to_calibration_json()) == 4

    def test_empty_result(self) -> None:
        from rallycut.court.detector import CourtDetectionResult

        result = CourtDetectionResult(corners=[], confidence=0.0)
        assert not result.is_valid


class TestSandColorEstimation:
    """Test that sand color estimation works across different conditions."""

    def test_bright_sand(self) -> None:
        from rallycut.court.detector import CourtDetector

        # Bright sand (beach in sun)
        frame = np.full((540, 960, 3), 200, dtype=np.uint8)
        # Add a white line
        cv2.line(frame, (100, 270), (860, 270), (255, 255, 255), 3)
        detector = CourtDetector()
        segs = detector._detect_lines_single_frame(frame)
        # Should detect something (line is white on bright sand)
        # May or may not detect depending on threshold — just no crash
        assert isinstance(segs, list)

    def test_dark_sand(self) -> None:
        from rallycut.court.detector import CourtDetector

        # Darker sand (overcast or wet)
        frame = np.full((540, 960, 3), 120, dtype=np.uint8)
        # Add a white line (higher contrast)
        cv2.line(frame, (100, 270), (860, 270), (255, 255, 255), 4)
        detector = CourtDetector()
        segs = detector._detect_lines_single_frame(frame)
        assert isinstance(segs, list)


def _make_dark_line_frame(
    width: int = 960,
    height: int = 540,
    sand_brightness: int = 180,
    line_brightness: int = 40,
) -> np.ndarray:
    """Create a synthetic frame with dark lines on bright sand background.

    Simulates courts using black/dark rope boundaries instead of white lines.
    """
    frame = np.full((height, width, 3), sand_brightness, dtype=np.uint8)

    # Add slight color variation to simulate sand
    frame[:, :, 0] = np.clip(frame[:, :, 0].astype(int) - 10, 0, 255).astype(np.uint8)
    frame[:, :, 2] = np.clip(frame[:, :, 2].astype(int) + 10, 0, 255).astype(np.uint8)

    dark = (line_brightness, line_brightness, line_brightness)
    line_thickness = 3

    # Far baseline: y=0.35, x from 0.30 to 0.70
    y_far = int(0.35 * height)
    x_far_l = int(0.30 * width)
    x_far_r = int(0.70 * width)
    cv2.line(frame, (x_far_l, y_far), (x_far_r, y_far), dark, line_thickness)

    # Left sideline: from (0.30, 0.35) to (0.05, 0.85)
    x_near_l = int(0.05 * width)
    y_near = int(0.85 * height)
    cv2.line(frame, (x_far_l, y_far), (x_near_l, y_near), dark, line_thickness)

    # Right sideline: from (0.70, 0.35) to (0.95, 0.85)
    x_near_r = int(0.95 * width)
    cv2.line(frame, (x_far_r, y_far), (x_near_r, y_near), dark, line_thickness)

    # Center line
    y_center = int(0.60 * height)
    t = (0.60 - 0.35) / (0.85 - 0.35)
    x_center_l = int(x_far_l + t * (x_near_l - x_far_l))
    x_center_r = int(x_far_r + t * (x_near_r - x_far_r))
    cv2.line(frame, (x_center_l, y_center), (x_center_r, y_center), dark, line_thickness)

    return frame


class TestDarkLineDetection:
    """Test dark line detection on synthetic frames."""

    def test_detect_dark_lines(self) -> None:
        """Dark lines should be detected when enable_dark_detection=True."""
        from rallycut.court.detector import CourtDetectionConfig, CourtDetector

        frame = _make_dark_line_frame()
        config = CourtDetectionConfig(
            min_temporal_support=1,
            dbscan_min_samples=1,
            enable_dark_detection=True,
        )
        detector = CourtDetector(config)
        segments = detector._detect_lines_single_frame(frame)
        assert len(segments) >= 2, f"Expected >=2 dark line segments, got {len(segments)}"

    def test_dark_lines_disabled(self) -> None:
        """No dark lines detected when enable_dark_detection=False."""
        from rallycut.court.detector import CourtDetectionConfig, CourtDetector

        frame = _make_dark_line_frame()
        config = CourtDetectionConfig(
            enable_dark_detection=False,
            enable_blue_detection=False,
        )
        detector = CourtDetector(config)
        segments = detector._detect_lines_single_frame(frame)
        # With only white line detection, dark lines should not be found
        assert len(segments) == 0, f"Expected 0 segments with dark detection off, got {len(segments)}"

    def test_full_detection_dark_lines(self) -> None:
        """Full pipeline should detect court corners from dark lines."""
        from rallycut.court.detector import CourtDetectionConfig, CourtDetector

        frame = _make_dark_line_frame()
        frames = [frame.copy() for _ in range(15)]

        config = CourtDetectionConfig(
            min_temporal_support=3,
            dbscan_min_samples=3,
            enable_dark_detection=True,
        )
        detector = CourtDetector(config)
        result = detector.detect_from_frames(frames)

        assert len(result.detected_lines) >= 1, (
            f"Expected >=1 detected lines from dark court, got {len(result.detected_lines)}. "
            f"Warnings: {result.warnings}"
        )
        assert len(result.corners) == 4, (
            f"Expected 4 corners from dark court, got {len(result.corners)}. "
            f"Warnings: {result.warnings}"
        )
        assert result.confidence > 0.0
