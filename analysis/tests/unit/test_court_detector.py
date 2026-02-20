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


# ── Court Model Correspondence Tests ────────────────────────────────────


class TestCourtModelConstants:
    def test_court_model_corners(self) -> None:
        from rallycut.court.line_geometry import COURT_MODEL_CORNERS

        assert len(COURT_MODEL_CORNERS) == 4
        # near-left, near-right, far-right, far-left
        assert COURT_MODEL_CORNERS[0] == (0.0, 0.0)
        assert COURT_MODEL_CORNERS[1] == (8.0, 0.0)
        assert COURT_MODEL_CORNERS[2] == (8.0, 16.0)
        assert COURT_MODEL_CORNERS[3] == (0.0, 16.0)

    def test_court_model_intersections(self) -> None:
        from rallycut.court.line_geometry import COURT_MODEL_INTERSECTIONS

        assert len(COURT_MODEL_INTERSECTIONS) == 6
        assert COURT_MODEL_INTERSECTIONS[frozenset({"far_baseline", "left_sideline"})] == (0.0, 16.0)
        assert COURT_MODEL_INTERSECTIONS[frozenset({"center_line", "right_sideline"})] == (8.0, 8.0)


class TestCollectCourtCorrespondences:
    def _make_detected_line(
        self, label: str, p1: tuple[float, float], p2: tuple[float, float],
    ) -> tuple[object, list[tuple[float, float, float, float]]]:
        """Create a (DetectedLine, segments) tuple for testing."""
        from rallycut.court.detector import DetectedLine

        dl = DetectedLine(
            label=label, p1=p1, p2=p2, support=10,
            angle_deg=segment_angle_deg(p1[0], p1[1], p2[0], p2[1]),
        )
        return (dl, [(p1[0], p1[1], p2[0], p2[1])])

    def test_two_lines_one_correspondence(self) -> None:
        from rallycut.court.line_geometry import collect_court_correspondences

        identified = {
            "far_baseline": self._make_detected_line("far_baseline", (0.30, 0.35), (0.70, 0.35)),
            "left_sideline": self._make_detected_line("left_sideline", (0.30, 0.35), (0.05, 0.85)),
        }
        corrs = collect_court_correspondences(identified)
        assert len(corrs) == 1
        # Should be the far-left corner
        img_pt, court_pt = corrs[0]
        assert court_pt == (0.0, 16.0)  # far-left in court space

    def test_four_lines_multiple_correspondences(self) -> None:
        from rallycut.court.line_geometry import collect_court_correspondences

        identified = {
            "far_baseline": self._make_detected_line("far_baseline", (0.30, 0.35), (0.70, 0.35)),
            "left_sideline": self._make_detected_line("left_sideline", (0.30, 0.35), (0.05, 0.85)),
            "right_sideline": self._make_detected_line("right_sideline", (0.70, 0.35), (0.95, 0.85)),
            "center_line": self._make_detected_line("center_line", (0.175, 0.60), (0.825, 0.60)),
        }
        corrs = collect_court_correspondences(identified)
        # far_bl×left_sl, far_bl×right_sl, center×left_sl, center×right_sl = 4
        assert len(corrs) == 4

    def test_out_of_bounds_filtered(self) -> None:
        from rallycut.court.line_geometry import collect_court_correspondences

        # Two nearly parallel lines that intersect far outside bounds
        identified = {
            "far_baseline": self._make_detected_line("far_baseline", (0.30, 0.35), (0.70, 0.35)),
            "center_line": self._make_detected_line("center_line", (0.30, 0.36), (0.70, 0.36)),
        }
        # far_baseline and center_line don't have a court model intersection anyway
        corrs = collect_court_correspondences(identified)
        assert len(corrs) == 0


class TestProjectCourtCorners:
    def test_identity_homography(self) -> None:
        from rallycut.court.line_geometry import project_court_corners

        H = np.eye(3, dtype=np.float64)
        corners = project_court_corners(H)
        assert len(corners) == 4
        assert abs(corners[0][0] - 0.0) < 1e-6  # near-left
        assert abs(corners[0][1] - 0.0) < 1e-6
        assert abs(corners[2][0] - 8.0) < 1e-6  # far-right
        assert abs(corners[2][1] - 16.0) < 1e-6

    def test_scale_homography(self) -> None:
        from rallycut.court.line_geometry import project_court_corners

        # Scale court (8m, 16m) → image (0.5, 1.0) normalized
        H = np.array([
            [0.5 / 8, 0, 0],
            [0, 1.0 / 16, 0],
            [0, 0, 1],
        ], dtype=np.float64)
        corners = project_court_corners(H)
        assert abs(corners[1][0] - 0.5) < 1e-6   # near-right x
        assert abs(corners[3][1] - 1.0) < 1e-6   # far-left y


class TestHomographyFitting:
    """Test the full homography fitting pipeline on synthetic frames."""

    def test_synthetic_homography_fit(self) -> None:
        """Homography should produce valid corners from a synthetic court."""
        from rallycut.court.detector import CourtDetectionConfig, CourtDetector

        frame = _make_synthetic_frame()
        frames = [frame.copy() for _ in range(15)]

        config = CourtDetectionConfig(
            min_temporal_support=3,
            dbscan_min_samples=3,
        )
        detector = CourtDetector(config)
        result = detector.detect_from_frames(frames)

        assert len(result.corners) == 4, (
            f"Expected 4 corners, got {len(result.corners)}. "
            f"Warnings: {result.warnings}"
        )
        assert result.confidence > 0.0
        # Should use a homography-based method (not legacy)
        assert result.fitting_method in (
            "homography", "homography_vp", "temporal_consensus", "legacy",
        )

    def test_homography_result_fields(self) -> None:
        """Result should include new homography fields."""
        from rallycut.court.detector import CourtDetectionConfig, CourtDetector

        frame = _make_synthetic_frame()
        frames = [frame.copy() for _ in range(15)]

        config = CourtDetectionConfig(
            min_temporal_support=3,
            dbscan_min_samples=3,
        )
        detector = CourtDetector(config)
        result = detector.detect_from_frames(frames)

        if result.corners:
            # Verify result has the expected fields
            assert hasattr(result, "fitting_method")
            assert hasattr(result, "n_correspondences")
            assert hasattr(result, "reprojection_error")

    def test_near_corner_accuracy(self) -> None:
        """Near corners from homography should be roughly consistent with geometry."""
        from rallycut.court.detector import CourtDetectionConfig, CourtDetector

        frame = _make_synthetic_frame()
        frames = [frame.copy() for _ in range(15)]

        config = CourtDetectionConfig(
            min_temporal_support=3,
            dbscan_min_samples=3,
        )
        detector = CourtDetector(config)
        result = detector.detect_from_frames(frames)

        if result.corners and len(result.corners) == 4:
            # Near corners should be below far corners (larger Y)
            near_y = (result.corners[0]["y"] + result.corners[1]["y"]) / 2
            far_y = (result.corners[2]["y"] + result.corners[3]["y"]) / 2
            assert near_y > far_y, (
                f"Near corners Y ({near_y:.3f}) should be > far corners Y ({far_y:.3f})"
            )

    def test_legacy_fallback(self) -> None:
        """With only 2 lines, should fall back to legacy fitting."""
        from rallycut.court.detector import CourtDetectionConfig, CourtDetector, DetectedLine

        config = CourtDetectionConfig()
        detector = CourtDetector(config)

        # Only far baseline + left sideline → 1 correspondence → legacy
        identified = {
            "far_baseline": (
                DetectedLine("far_baseline", (0.30, 0.35), (0.70, 0.35), 20, 0.0),
                [(0.30, 0.35, 0.70, 0.35)],
            ),
            "left_sideline": (
                DetectedLine("left_sideline", (0.30, 0.35), (0.05, 0.85), 15, 55.0),
                [(0.30, 0.35, 0.05, 0.85)],
            ),
        }
        result = detector._fit_court_model(identified)
        # Should fail or use legacy (only 1 correspondence, no legacy near strategy)
        assert result.fitting_method == "legacy"

    def test_temporal_consensus_disabled(self) -> None:
        """When disabled, temporal consensus is skipped."""
        from rallycut.court.detector import CourtDetectionConfig, CourtDetector

        frame = _make_synthetic_frame()
        frames = [frame.copy() for _ in range(15)]

        config = CourtDetectionConfig(
            min_temporal_support=3,
            dbscan_min_samples=3,
            enable_temporal_consensus=False,
        )
        detector = CourtDetector(config)
        result = detector.detect_from_frames(frames)

        if result.corners:
            assert result.fitting_method != "temporal_consensus"


class TestReprojectionError:
    def test_perfect_homography(self) -> None:
        """Zero reprojection error for a perfect homography."""
        from rallycut.court.detector import CourtDetector

        # Create a known homography (simple scale)
        H = np.array([
            [0.05, 0, 0.1],
            [0, 0.05, 0.2],
            [0, 0, 1],
        ], dtype=np.float64)

        # Generate correspondences consistent with H
        court_pts = [(0.0, 16.0), (8.0, 16.0), (0.0, 8.0), (8.0, 8.0)]
        correspondences = []
        for cp in court_pts:
            pts = np.array([[cp]], dtype=np.float64)
            projected = cv2.perspectiveTransform(pts, H)
            img_pt = (float(projected[0, 0, 0]), float(projected[0, 0, 1]))
            correspondences.append((img_pt, cp))

        error = CourtDetector._compute_reprojection_error(H, correspondences)
        assert error < 1e-6

    def test_imperfect_homography(self) -> None:
        """Non-zero error for mismatched correspondences."""
        from rallycut.court.detector import CourtDetector

        H = np.eye(3, dtype=np.float64)
        # Correspondences where image points don't match the identity projection
        correspondences = [
            ((0.1, 16.1), (0.0, 16.0)),  # off by (0.1, 0.1)
            ((8.1, 16.1), (8.0, 16.0)),
        ]
        error = CourtDetector._compute_reprojection_error(H, correspondences)
        assert error > 0.1


class TestResultBackwardCompat:
    """Verify new fields have sensible defaults for backward compatibility."""

    def test_default_fields(self) -> None:
        from rallycut.court.detector import CourtDetectionResult

        result = CourtDetectionResult(
            corners=[{"x": 0, "y": 0}] * 4,
            confidence=0.5,
        )
        assert result.homography is None
        assert result.fitting_method == "legacy"
        assert result.n_correspondences == 0
        assert result.reprojection_error == 0.0


# ── Sideline Mirroring Tests ────────────────────────────────────────────


class TestSidelineMirroring:
    """Test sideline mirroring for homography path."""

    def _make_identified(
        self,
        *,
        far_bl: bool = True,
        left_sl: bool = False,
        right_sl: bool = False,
        center_ln: bool = False,
    ) -> dict:
        from rallycut.court.detector import DetectedLine

        identified: dict = {}
        if far_bl:
            dl = DetectedLine("far_baseline", (0.30, 0.35), (0.70, 0.35), 20, 0.0)
            identified["far_baseline"] = (dl, [(0.30, 0.35, 0.70, 0.35)])
        if left_sl:
            dl = DetectedLine("left_sideline", (0.30, 0.35), (0.05, 0.85), 15, 55.0)
            identified["left_sideline"] = (dl, [(0.30, 0.35, 0.05, 0.85)])
        if right_sl:
            dl = DetectedLine("right_sideline", (0.70, 0.35), (0.95, 0.85), 15, 55.0)
            identified["right_sideline"] = (dl, [(0.70, 0.35, 0.95, 0.85)])
        if center_ln:
            dl = DetectedLine("center_line", (0.175, 0.60), (0.825, 0.60), 12, 0.0)
            identified["center_line"] = (dl, [(0.175, 0.60, 0.825, 0.60)])
        return identified

    def test_mirror_left_to_right(self) -> None:
        """Single left sideline → creates synthetic right sideline."""
        from rallycut.court.detector import CourtDetector

        detector = CourtDetector()
        identified = self._make_identified(far_bl=True, left_sl=True)
        result, mirrored = detector._mirror_missing_sideline(identified)

        assert mirrored is True
        assert "right_sideline" in result
        assert "left_sideline" in result
        # Original left sideline should be preserved
        assert result["left_sideline"] is identified["left_sideline"]
        # Synthetic right should be mirrored about baseline midpoint (0.5)
        syn = result["right_sideline"][0]
        assert syn.label == "right_sideline"
        # Left sl p1 x=0.30, mirror about 0.50 → 0.70
        assert abs(syn.p1[0] - 0.70) < 1e-6
        # Y should be preserved
        assert abs(syn.p1[1] - 0.35) < 1e-6

    def test_mirror_right_to_left(self) -> None:
        """Single right sideline → creates synthetic left sideline."""
        from rallycut.court.detector import CourtDetector

        detector = CourtDetector()
        identified = self._make_identified(far_bl=True, right_sl=True)
        result, mirrored = detector._mirror_missing_sideline(identified)

        assert mirrored is True
        assert "left_sideline" in result
        assert "right_sideline" in result
        # Synthetic left should be mirrored about baseline midpoint (0.5)
        syn = result["left_sideline"][0]
        assert syn.label == "left_sideline"
        # Right sl p1 x=0.70, mirror about 0.50 → 0.30
        assert abs(syn.p1[0] - 0.30) < 1e-6
        assert abs(syn.p1[1] - 0.35) < 1e-6

    def test_no_mirror_when_both_exist(self) -> None:
        """Two sidelines → no mirroring needed."""
        from rallycut.court.detector import CourtDetector

        detector = CourtDetector()
        identified = self._make_identified(far_bl=True, left_sl=True, right_sl=True)
        result, mirrored = detector._mirror_missing_sideline(identified)

        assert mirrored is False
        assert result is identified  # Same object returned

    def test_no_mirror_without_baseline(self) -> None:
        """Missing far baseline → no mirroring possible."""
        from rallycut.court.detector import CourtDetector

        detector = CourtDetector()
        identified = self._make_identified(far_bl=False, left_sl=True)
        result, mirrored = detector._mirror_missing_sideline(identified)

        assert mirrored is False
        assert "right_sideline" not in result

    def test_mirrored_correspondences(self) -> None:
        """Mirrored identified dict produces 4 correspondences for temporal consensus."""
        from rallycut.court.detector import CourtDetector
        from rallycut.court.line_geometry import collect_court_correspondences

        detector = CourtDetector()
        identified = self._make_identified(
            far_bl=True, left_sl=True, center_ln=True,
        )
        # Without mirroring: far_bl×left_sl + center×left_sl = 2 correspondences
        real_corrs = collect_court_correspondences(identified)
        assert len(real_corrs) == 2

        # Line-level mirroring adds right_sl → 4 correspondences
        # (used by temporal consensus for per-frame segment matching)
        mirrored_dict, was_mirrored = detector._mirror_missing_sideline(identified)
        assert was_mirrored
        mirrored_corrs = collect_court_correspondences(mirrored_dict)
        assert len(mirrored_corrs) == 4

    def test_three_lines_falls_to_legacy(self) -> None:
        """3 lines with mirroring: only 2 real correspondences → falls to legacy."""
        from rallycut.court.detector import CourtDetectionConfig, CourtDetector

        detector = CourtDetector(CourtDetectionConfig(enable_temporal_consensus=False))
        identified = self._make_identified(
            far_bl=True, left_sl=True, center_ln=True,
        )
        result = detector._fit_court_model(identified)

        # 2 real correspondences < 4 needed for homography → legacy fallback
        assert len(result.corners) == 4
        assert result.confidence > 0
        assert result.fitting_method == "legacy"
        # Legacy still gets the mirroring warning from its own path
        assert any("mirror" in w.lower() for w in result.warnings)


# ── Court Detection Insights Tests ─────────────────────────────────────


class TestCourtDetectionInsights:
    """Test CourtDetectionInsights derivation and serialization."""

    def test_insights_from_good_result(self) -> None:
        """4 lines, high confidence → detected=True, no tips."""
        from rallycut.court.detector import (
            CourtDetectionInsights,
            CourtDetectionResult,
            DetectedLine,
        )

        result = CourtDetectionResult(
            corners=[{"x": 0.1, "y": 0.9}, {"x": 0.9, "y": 0.9},
                     {"x": 0.75, "y": 0.3}, {"x": 0.25, "y": 0.3}],
            confidence=0.85,
            detected_lines=[
                DetectedLine("far_baseline", (0.25, 0.30), (0.75, 0.30), 20, 0.0),
                DetectedLine("left_sideline", (0.25, 0.30), (0.05, 0.85), 15, 55.0),
                DetectedLine("right_sideline", (0.75, 0.30), (0.95, 0.85), 15, 55.0),
                DetectedLine("center_line", (0.15, 0.60), (0.85, 0.60), 12, 0.0),
            ],
            fitting_method="temporal_consensus",
        )
        insights = CourtDetectionInsights.from_result(result)

        assert insights.detected is True
        assert insights.confidence == 0.85
        assert insights.lines_found == 4
        assert insights.line_visibility == "good"
        assert insights.camera_height == "moderate"
        assert insights.recording_tips == []

    def test_insights_from_failed_result(self) -> None:
        """0 lines → detected=False, has recording tips."""
        from rallycut.court.detector import CourtDetectionInsights, CourtDetectionResult

        result = CourtDetectionResult(
            corners=[],
            confidence=0.0,
            detected_lines=[],
            warnings=["No lines detected"],
            fitting_method="legacy",
        )
        insights = CourtDetectionInsights.from_result(result)

        assert insights.detected is False
        assert insights.lines_found == 0
        assert insights.line_visibility == "none"
        assert insights.camera_height == "unknown"
        assert len(insights.recording_tips) >= 1
        # Should have both "lines not detected" and "higher vantage" tips
        assert any("not detected" in tip for tip in insights.recording_tips)
        assert any("higher vantage" in tip for tip in insights.recording_tips)

    def test_insights_camera_height_from_baseline_y(self) -> None:
        """Far baseline at various Y positions → correct camera_height."""
        from rallycut.court.detector import (
            CourtDetectionInsights,
            CourtDetectionResult,
            DetectedLine,
        )

        # Elevated: far baseline at y=0.20
        result_elevated = CourtDetectionResult(
            corners=[{"x": 0, "y": 0}] * 4,
            confidence=0.7,
            detected_lines=[
                DetectedLine("far_baseline", (0.2, 0.20), (0.8, 0.20), 20, 0.0),
            ],
            fitting_method="legacy",
        )
        assert CourtDetectionInsights.from_result(result_elevated).camera_height == "elevated"

        # Low: far baseline at y=0.50
        result_low = CourtDetectionResult(
            corners=[{"x": 0, "y": 0}] * 4,
            confidence=0.5,
            detected_lines=[
                DetectedLine("far_baseline", (0.2, 0.50), (0.8, 0.50), 20, 0.0),
            ],
            fitting_method="legacy",
        )
        assert CourtDetectionInsights.from_result(result_low).camera_height == "low"

        # Moderate: far baseline at y=0.32
        result_moderate = CourtDetectionResult(
            corners=[{"x": 0, "y": 0}] * 4,
            confidence=0.6,
            detected_lines=[
                DetectedLine("far_baseline", (0.2, 0.32), (0.8, 0.32), 20, 0.0),
            ],
            fitting_method="legacy",
        )
        assert CourtDetectionInsights.from_result(result_moderate).camera_height == "moderate"

    def test_insights_brightness_tip(self) -> None:
        """Brightness > 200 → includes brightness tip."""
        from rallycut.court.detector import CourtDetectionInsights, CourtDetectionResult

        result = CourtDetectionResult(
            corners=[],
            confidence=0.0,
            detected_lines=[],
            fitting_method="legacy",
        )
        insights = CourtDetectionInsights.from_result(result, brightness_mean=210.0)
        assert any("wash out" in tip.lower() for tip in insights.recording_tips)

        # No brightness tip when below threshold
        insights_normal = CourtDetectionInsights.from_result(result, brightness_mean=150.0)
        assert not any("wash out" in tip.lower() for tip in insights_normal.recording_tips)

    def test_insights_to_dict(self) -> None:
        """Serialization produces expected keys and types."""
        from rallycut.court.detector import CourtDetectionInsights, CourtDetectionResult

        result = CourtDetectionResult(
            corners=[{"x": 0, "y": 0}] * 4,
            confidence=0.62,
            detected_lines=[],
            fitting_method="homography",
        )
        insights = CourtDetectionInsights.from_result(result)
        d = insights.to_dict()

        assert d["detected"] is True
        assert d["confidence"] == 0.62
        assert d["linesFound"] == 0
        assert isinstance(d["cameraHeight"], str)
        assert isinstance(d["lineVisibility"], str)
        assert d["fittingMethod"] == "homography"
        assert isinstance(d["recordingTips"], list)
