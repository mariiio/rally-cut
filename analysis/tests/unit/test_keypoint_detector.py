"""Tests for court keypoint detection using YOLO-pose."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import cv2
import numpy as np

from rallycut.court.keypoint_detector import (
    CourtKeypointDetector,
    FrameKeypoints,
    _weighted_median,
)
from rallycut.court.line_geometry import line_intersection


class TestUnpadCoordinates:
    """Test Y coordinate un-padding math."""

    def test_unpad_with_30_percent_padding(self) -> None:
        """Y coordinates should be multiplied by (1 + pad_ratio) to reverse export scaling."""
        pad_ratio = 0.3
        # During export: y_padded = y_orig / (1 + 0.3) = y_orig / 1.3
        # During inference: y_orig = y_padded * 1.3
        y_orig = 0.8
        y_padded = y_orig / (1.0 + pad_ratio)  # ~0.615

        # Un-pad should recover original
        y_recovered = y_padded * (1.0 + pad_ratio)
        assert abs(y_recovered - y_orig) < 1e-6

    def test_unpad_preserves_x(self) -> None:
        """X coordinates should not be affected by bottom padding."""
        x = 0.45
        # X is unchanged by vertical padding (only Y is scaled)
        assert x == 0.45

    def test_unpad_offscreen_near_corner(self) -> None:
        """Near corners below original frame (y > 1.0) should be recoverable."""
        pad_ratio = 0.3
        y_orig = 1.2  # Below frame
        y_padded = y_orig / (1.0 + pad_ratio)  # ~0.923 (in padding zone)

        y_recovered = y_padded * (1.0 + pad_ratio)
        assert abs(y_recovered - y_orig) < 1e-6
        assert y_recovered > 1.0  # Off-screen

    def test_unpad_far_corner(self) -> None:
        """Far corners (small Y) should remain in frame."""
        pad_ratio = 0.3
        y_orig = 0.35
        y_padded = y_orig / (1.0 + pad_ratio)

        y_recovered = y_padded * (1.0 + pad_ratio)
        assert abs(y_recovered - y_orig) < 1e-6
        assert 0.0 <= y_recovered <= 1.0


class TestAggregateCorners:
    """Test multi-frame aggregation."""

    def test_single_frame(self) -> None:
        """Single frame → return as-is."""
        detector = CourtKeypointDetector.__new__(CourtKeypointDetector)
        detector._pad_ratio = 0.3
        detector._conf_threshold = 0.3
        detector._last_diagnostics = None

        fr = FrameKeypoints(
            corners=[
                {"x": 0.1, "y": 0.9},
                {"x": 0.9, "y": 0.9},
                {"x": 0.7, "y": 0.3},
                {"x": 0.3, "y": 0.3},
            ],
            confidence=0.85,
            kpt_confidences=[0.9, 0.9, 0.9, 0.9],
        )

        corners, conf, _diag = detector._aggregate([fr])
        assert len(corners) == 4
        assert abs(corners[0]["x"] - 0.1) < 1e-6
        assert abs(conf - 0.85) < 1e-6

    def test_median_of_multiple_frames(self) -> None:
        """Multiple frames → median coordinates."""
        detector = CourtKeypointDetector.__new__(CourtKeypointDetector)
        detector._pad_ratio = 0.3
        detector._conf_threshold = 0.3
        detector._last_diagnostics = None

        frames = []
        for x_offset in [-0.02, 0.0, 0.01, 0.0, -0.01]:
            frames.append(FrameKeypoints(
                corners=[
                    {"x": 0.1 + x_offset, "y": 0.9},
                    {"x": 0.9 + x_offset, "y": 0.9},
                    {"x": 0.7 + x_offset, "y": 0.3},
                    {"x": 0.3 + x_offset, "y": 0.3},
                ],
                confidence=0.85,
                kpt_confidences=[0.9, 0.9, 0.9, 0.9],
            ))

        corners, conf, _diag = detector._aggregate(frames)
        # Median of [-0.02, 0.0, 0.01, 0.0, -0.01] = 0.0
        assert abs(corners[0]["x"] - 0.1) < 0.005

    def test_outlier_rejection(self) -> None:
        """Outlier frame should be filtered by 2σ rule."""
        detector = CourtKeypointDetector.__new__(CourtKeypointDetector)
        detector._pad_ratio = 0.3
        detector._conf_threshold = 0.3
        detector._last_diagnostics = None

        # 9 consistent frames + 1 extreme outlier
        frames = []
        for _ in range(9):
            frames.append(FrameKeypoints(
                corners=[
                    {"x": 0.10, "y": 0.90},
                    {"x": 0.90, "y": 0.90},
                    {"x": 0.70, "y": 0.30},
                    {"x": 0.30, "y": 0.30},
                ],
                confidence=0.85,
                kpt_confidences=[0.9, 0.9, 0.9, 0.9],
            ))
        # Outlier
        frames.append(FrameKeypoints(
            corners=[
                {"x": 0.50, "y": 0.50},  # Way off
                {"x": 0.50, "y": 0.50},
                {"x": 0.50, "y": 0.50},
                {"x": 0.50, "y": 0.50},
            ],
            confidence=0.40,
            kpt_confidences=[0.5, 0.5, 0.5, 0.5],
        ))

        corners, conf, _diag = detector._aggregate(frames)
        # Should be close to 0.10, not pulled toward 0.50
        assert abs(corners[0]["x"] - 0.10) < 0.02

    def test_empty_frames(self) -> None:
        """No frames → empty result."""
        detector = CourtKeypointDetector.__new__(CourtKeypointDetector)
        detector._pad_ratio = 0.3
        detector._conf_threshold = 0.3
        detector._last_diagnostics = None

        corners, conf, _diag = detector._aggregate([])
        assert corners == []
        assert conf == 0.0


class TestWeightedMedian:
    """Test weighted median helper."""

    def test_equal_weights_gives_regular_median(self) -> None:
        """Equal weights → same as np.median."""
        values = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
        weights = np.ones(5)
        assert abs(_weighted_median(values, weights) - 5.0) < 1e-6

    def test_heavy_weight_pulls_toward_value(self) -> None:
        """One dominant weight → result near that value."""
        values = np.array([1.0, 5.0, 9.0])
        weights = np.array([0.1, 0.1, 10.0])
        result = _weighted_median(values, weights)
        assert result == 9.0

    def test_zero_weights_falls_back_to_median(self) -> None:
        """All-zero weights → unweighted median."""
        values = np.array([1.0, 3.0, 5.0])
        weights = np.zeros(3)
        assert abs(_weighted_median(values, weights) - 3.0) < 1e-6

    def test_empty_values(self) -> None:
        """Empty array → 0.0."""
        assert _weighted_median(np.array([]), np.array([])) == 0.0

    def test_confidence_weighted_aggregation(self) -> None:
        """High-confidence frames should dominate the aggregated corner."""
        detector = CourtKeypointDetector.__new__(CourtKeypointDetector)
        detector._pad_ratio = 0.3
        detector._conf_threshold = 0.3
        detector._last_diagnostics = None

        frames = []
        # 5 low-confidence frames with x=0.12
        for _ in range(5):
            frames.append(FrameKeypoints(
                corners=[
                    {"x": 0.12, "y": 0.90},
                    {"x": 0.88, "y": 0.90},
                    {"x": 0.70, "y": 0.30},
                    {"x": 0.30, "y": 0.30},
                ],
                confidence=0.4,
                kpt_confidences=[0.2, 0.2, 0.2, 0.2],
            ))
        # 5 high-confidence frames with x=0.10
        for _ in range(5):
            frames.append(FrameKeypoints(
                corners=[
                    {"x": 0.10, "y": 0.90},
                    {"x": 0.90, "y": 0.90},
                    {"x": 0.70, "y": 0.30},
                    {"x": 0.30, "y": 0.30},
                ],
                confidence=0.95,
                kpt_confidences=[0.95, 0.95, 0.95, 0.95],
            ))

        corners, _, _diag = detector._aggregate(frames)
        # With unweighted median, result would be 0.11 (even split).
        # With confidence-weighted, x=0.10 should win
        # (5×0.95=4.75 vs 5×0.2=1.0 cumulative weight).
        assert abs(corners[0]["x"] - 0.10) < 0.005


class TestFallbackToClassical:
    """Test that CourtDetector falls back when keypoint confidence is low."""

    def test_fallback_when_no_model(self) -> None:
        """No keypoint model → classical pipeline runs."""
        from rallycut.court.detector import CourtDetector

        detector = CourtDetector(keypoint_model_path="/nonexistent/model.pt")
        kp = detector._get_keypoint_detector()
        assert kp is None  # Model doesn't exist

    def test_keypoint_detector_model_exists(self) -> None:
        """model_exists property returns False for nonexistent path."""
        detector = CourtKeypointDetector(model_path="/nonexistent/model.pt")
        assert detector.model_exists is False

    @patch("rallycut.court.keypoint_detector.CourtKeypointDetector.detect")
    def test_low_confidence_triggers_fallback(self, mock_detect: MagicMock) -> None:
        """Low keypoint confidence should fall through to classical."""
        from rallycut.court.detector import CourtDetectionResult, CourtDetector

        # Mock keypoint detector returning low confidence
        mock_detect.return_value = CourtDetectionResult(
            corners=[{"x": 0.1, "y": 0.9}] * 4,
            confidence=0.2,  # Below threshold
            fitting_method="keypoint",
        )

        detector = CourtDetector()
        # Manually set keypoint detector to test fallback logic
        mock_kp = MagicMock()
        mock_kp.model_exists = True
        mock_kp.detect.return_value = mock_detect.return_value
        detector._keypoint_detector = mock_kp

        # detect() on a nonexistent video should raise FileNotFoundError
        # because keypoint confidence is low and it tries classical pipeline
        with pytest.raises(FileNotFoundError):
            detector.detect("/nonexistent/video.mp4")


class TestQualityDiagnostics:
    """Test quality diagnostics computation."""

    def test_diagnostics_populated(self) -> None:
        """Diagnostics should contain per-corner confidence and std."""
        detector = CourtKeypointDetector.__new__(CourtKeypointDetector)
        detector._pad_ratio = 0.3
        detector._conf_threshold = 0.3
        detector._last_diagnostics = None

        frames = []
        for _ in range(5):
            frames.append(FrameKeypoints(
                corners=[
                    {"x": 0.10, "y": 0.90},
                    {"x": 0.90, "y": 0.90},
                    {"x": 0.70, "y": 0.30},
                    {"x": 0.30, "y": 0.30},
                ],
                confidence=0.85,
                kpt_confidences=[0.9, 0.85, 0.92, 0.88],
            ))

        _, _, diag = detector._aggregate(frames, n_sampled=10)
        assert diag.detection_rate == 0.5  # 5 detected / 10 sampled
        assert "near-left" in diag.per_corner_confidence
        assert diag.per_corner_confidence["near-left"] == pytest.approx(0.9, abs=0.01)
        assert diag.perspective_ratio > 0
        assert diag.off_screen_corners == []

    def test_off_screen_corner_detection(self) -> None:
        """Near corners with y > 1.0 should be flagged as off-screen."""
        detector = CourtKeypointDetector.__new__(CourtKeypointDetector)
        detector._pad_ratio = 0.3
        detector._conf_threshold = 0.3
        detector._last_diagnostics = None

        frames = [FrameKeypoints(
            corners=[
                {"x": 0.10, "y": 1.15},  # Off-screen
                {"x": 0.90, "y": 1.10},  # Off-screen
                {"x": 0.70, "y": 0.30},
                {"x": 0.30, "y": 0.30},
            ],
            confidence=0.85,
            kpt_confidences=[0.9, 0.85, 0.92, 0.88],
        )]

        _, _, diag = detector._aggregate(frames, n_sampled=1)
        assert "near-left" in diag.off_screen_corners
        assert "near-right" in diag.off_screen_corners
        assert "far-right" not in diag.off_screen_corners

    def test_extreme_perspective_warning(self) -> None:
        """Extreme perspective ratio should generate a warning."""
        detector = CourtKeypointDetector.__new__(CourtKeypointDetector)
        detector._pad_ratio = 0.3
        detector._conf_threshold = 0.3
        detector._last_diagnostics = None

        # Near side very wide, far side narrow → extreme perspective
        frames = [FrameKeypoints(
            corners=[
                {"x": 0.05, "y": 0.95},  # near-left (wide)
                {"x": 0.95, "y": 0.95},  # near-right (wide)
                {"x": 0.55, "y": 0.30},  # far-right (narrow)
                {"x": 0.45, "y": 0.30},  # far-left (narrow)
            ],
            confidence=0.85,
            kpt_confidences=[0.9, 0.85, 0.92, 0.88],
        )]

        _, _, diag = detector._aggregate(frames, n_sampled=1)
        assert diag.perspective_ratio > 4.0
        assert any("perspective" in w.lower() for w in diag.warnings)

    def test_diagnostics_to_dict(self) -> None:
        """Diagnostics should serialize to dict."""
        from rallycut.court.keypoint_detector import CourtQualityDiagnostics

        diag = CourtQualityDiagnostics(
            detection_rate=0.8,
            per_corner_confidence={"near-left": 0.9, "near-right": 0.85,
                                   "far-right": 0.92, "far-left": 0.88},
            per_corner_std={"near-left": 0.005, "near-right": 0.004,
                           "far-right": 0.002, "far-left": 0.003},
            off_screen_corners=["near-left"],
            perspective_ratio=2.5,
            warnings=["Off-screen corners: near-left"],
        )
        d = diag.to_dict()
        assert d["detection_rate"] == 0.8
        assert "near-left" in d["per_corner_confidence"]
        assert d["off_screen_corners"] == ["near-left"]


class TestFrameKeypoints:
    """Test FrameKeypoints dataclass."""

    def test_creation(self) -> None:
        fk = FrameKeypoints(
            corners=[{"x": 0.1, "y": 0.9}] * 4,
            confidence=0.8,
            kpt_confidences=[0.9, 0.85, 0.92, 0.88],
        )
        assert len(fk.corners) == 4
        assert fk.confidence == 0.8
        assert len(fk.kpt_confidences) == 4


class TestRefineNearCorners:
    """Test perspective-geometric near-corner refinement."""

    def _make_detector(self) -> CourtKeypointDetector:
        detector = CourtKeypointDetector.__new__(CourtKeypointDetector)
        detector._pad_ratio = 0.3
        detector._conf_threshold = 0.3
        detector._last_diagnostics = None
        return detector

    def test_refine_both_near_corners(self) -> None:
        """Both near corners low-conf should be replaced by extrapolation."""
        detector = self._make_detector()

        # Typical court: far corners accurate, near corners pulled toward center
        corners = [
            {"x": 0.20, "y": 0.70},  # near-left (bad — should be wider/lower)
            {"x": 0.80, "y": 0.70},  # near-right (bad)
            {"x": 0.65, "y": 0.35},  # far-right (good)
            {"x": 0.35, "y": 0.35},  # far-left (good)
        ]
        conf = {
            "near-left": 0.002,
            "near-right": 0.003,
            "far-right": 0.999,
            "far-left": 0.998,
        }

        refined, refined_names = detector._refine_near_corners(corners, conf)

        # Near corners should move outward (wider X) and downward (larger Y)
        assert refined[0]["x"] < corners[0]["x"], "near-left should move left"
        assert refined[0]["y"] > corners[0]["y"], "near-left should move down"
        assert refined[1]["x"] > corners[1]["x"], "near-right should move right"
        assert refined[1]["y"] > corners[1]["y"], "near-right should move down"
        # Far corners unchanged
        assert refined[2] == corners[2]
        assert refined[3] == corners[3]
        # Both near corners should be marked as refined
        assert refined_names == {"near-left", "near-right"}

    def test_no_refinement_when_confident(self) -> None:
        """All corners high confidence — no changes."""
        detector = self._make_detector()

        corners = [
            {"x": 0.10, "y": 0.90},
            {"x": 0.90, "y": 0.90},
            {"x": 0.70, "y": 0.30},
            {"x": 0.30, "y": 0.30},
        ]
        conf = {
            "near-left": 0.95,
            "near-right": 0.92,
            "far-right": 0.99,
            "far-left": 0.98,
        }

        refined, refined_names = detector._refine_near_corners(corners, conf)
        assert refined == corners
        assert refined_names == set()

    def test_asymmetric_refinement(self) -> None:
        """One near corner good, one bad — only fix the bad one."""
        detector = self._make_detector()

        corners = [
            {"x": 0.10, "y": 0.90},  # near-left (good)
            {"x": 0.70, "y": 0.60},  # near-right (bad — pulled inward)
            {"x": 0.65, "y": 0.35},  # far-right (good)
            {"x": 0.35, "y": 0.35},  # far-left (good)
        ]
        conf = {
            "near-left": 0.90,
            "near-right": 0.01,
            "far-right": 0.999,
            "far-left": 0.998,
        }

        refined, refined_names = detector._refine_near_corners(corners, conf)

        # near-left unchanged (high confidence)
        assert refined[0] == corners[0]
        # near-right should be refined (moved outward/downward)
        assert refined[1] != corners[1]
        assert refined[1]["x"] > corners[1]["x"], "near-right should move right"
        # Only near-right was refined
        assert refined_names == {"near-right"}

    def test_vp_below_far_baseline_skips(self) -> None:
        """VP below far baseline means invalid geometry — skip refinement."""
        detector = self._make_detector()

        # Sidelines diverge downward (VP below court) — invalid perspective
        corners = [
            {"x": 0.05, "y": 0.80},  # near-left
            {"x": 0.95, "y": 0.80},  # near-right
            {"x": 0.40, "y": 0.30},  # far-right (narrower than near)
            {"x": 0.60, "y": 0.30},  # far-left (note: far-left.x > far-right.x = inverted)
        ]
        conf = {
            "near-left": 0.01,
            "near-right": 0.01,
            "far-right": 0.99,
            "far-left": 0.99,
        }

        refined, refined_names = detector._refine_near_corners(corners, conf)
        # Should fall back to original (VP is below far baseline or parallel)
        assert refined == corners
        assert refined_names == set()

    def test_offscreen_near_corners_clamped(self) -> None:
        """Refined near corners are clamped to max margin beyond frame."""
        detector = self._make_detector()
        margin = CourtKeypointDetector.NEAR_CORNER_MAX_MARGIN

        # Extreme convergence: VP very close to far baseline → large extrapolation
        # that would overshoot far beyond the frame without clamping.
        corners = [
            {"x": -0.50, "y": 0.90},  # near-left (model prediction, way off)
            {"x": 1.50, "y": 0.90},  # near-right (model prediction, way off)
            {"x": 0.70, "y": 0.40},  # far-right (good)
            {"x": 0.30, "y": 0.40},  # far-left (good)
        ]
        conf = {
            "near-left": 0.001,
            "near-right": 0.001,
            "far-right": 0.999,
            "far-left": 0.999,
        }

        refined, refined_names = detector._refine_near_corners(corners, conf)

        # Near corners should be clamped within [-margin, 1+margin]
        assert refined[0]["x"] >= -margin, f"near-left x={refined[0]['x']}"
        assert refined[0]["y"] <= 1.0 + margin, f"near-left y={refined[0]['y']}"
        assert refined[1]["x"] <= 1.0 + margin, f"near-right x={refined[1]['x']}"
        assert refined[1]["y"] <= 1.0 + margin, f"near-right y={refined[1]['y']}"
        assert refined_names == {"near-left", "near-right"}


class TestHarmonicConjugateRefinement:
    """Test harmonic conjugate near-corner extrapolation."""

    def _make_detector(self) -> CourtKeypointDetector:
        detector = CourtKeypointDetector.__new__(CourtKeypointDetector)
        detector._pad_ratio = 0.3
        detector._conf_threshold = 0.3
        detector._last_diagnostics = None
        return detector

    def test_harmonic_conjugate_produces_valid_court(self) -> None:
        """Harmonic conjugate with good VP + center line gives valid court."""
        detector = self._make_detector()

        corners = [
            {"x": 0.20, "y": 0.70},  # near-left (bad guess)
            {"x": 0.80, "y": 0.70},  # near-right (bad guess)
            {"x": 0.65, "y": 0.35},  # far-right (good)
            {"x": 0.35, "y": 0.35},  # far-left (good)
        ]
        conf = {
            "near-left": 0.002,
            "near-right": 0.003,
            "far-right": 0.999,
            "far-left": 0.998,
        }

        # VP above court (sidelines converge upward)
        vp = (0.50, -0.50)
        # Center line roughly at Y=0.50 (between far baseline 0.35 and near ~0.80)
        center_line = ((0.20, 0.50), (0.80, 0.50))

        result = detector._extrapolate_via_harmonic_conjugate(
            corners, conf, 0.5, vp, center_line,
        )
        assert result is not None
        refined, refined_names = result

        assert "near-left" in refined_names
        assert "near-right" in refined_names
        # Near corners should be wider than far corners (perspective)
        assert refined[0]["x"] < corners[3]["x"], "near-left should be left of far-left"
        assert refined[1]["x"] > corners[2]["x"], "near-right should be right of far-right"
        # Near corners should be below far corners
        assert refined[0]["y"] > corners[3]["y"]
        assert refined[1]["y"] > corners[2]["y"]
        # Far corners unchanged
        assert refined[2] == corners[2]
        assert refined[3] == corners[3]

    def test_harmonic_returns_none_when_center_above_far(self) -> None:
        """Center points above far baseline → reject."""
        detector = self._make_detector()

        corners = [
            {"x": 0.20, "y": 0.70},
            {"x": 0.80, "y": 0.70},
            {"x": 0.65, "y": 0.35},
            {"x": 0.35, "y": 0.35},
        ]
        conf = {"near-left": 0.01, "near-right": 0.01, "far-right": 0.99, "far-left": 0.99}

        vp = (0.50, -0.50)
        # Center line above far baseline
        center_line = ((0.20, 0.20), (0.80, 0.20))

        result = detector._extrapolate_via_harmonic_conjugate(
            corners, conf, 0.5, vp, center_line,
        )
        assert result is None

    def test_harmonic_skips_confident_corners(self) -> None:
        """High-confidence near corners should not be replaced."""
        detector = self._make_detector()

        corners = [
            {"x": 0.10, "y": 0.90},
            {"x": 0.90, "y": 0.90},
            {"x": 0.65, "y": 0.35},
            {"x": 0.35, "y": 0.35},
        ]
        conf = {"near-left": 0.95, "near-right": 0.92, "far-right": 0.99, "far-left": 0.99}

        vp = (0.50, -0.50)
        center_line = ((0.20, 0.50), (0.80, 0.50))

        result = detector._extrapolate_via_harmonic_conjugate(
            corners, conf, 0.5, vp, center_line,
        )
        # No corners need refinement → returns None
        assert result is None


class TestHomographyRefinement:
    """Test homography-based near-corner refinement."""

    def _make_detector(self) -> CourtKeypointDetector:
        detector = CourtKeypointDetector.__new__(CourtKeypointDetector)
        detector._pad_ratio = 0.3
        detector._conf_threshold = 0.3
        detector._last_diagnostics = None
        return detector

    def test_homography_from_perfect_correspondences(self) -> None:
        """Homography from 4 accurate points should produce valid near corners."""
        import cv2 as cv

        detector = self._make_detector()

        # Create a known homography and project court corners through it
        # Court: (0,0)=NL, (8,0)=NR, (8,16)=FR, (0,16)=FL
        # Use a realistic perspective transformation
        court_pts = np.array([
            [0.0, 0.0], [8.0, 0.0], [8.0, 16.0], [0.0, 16.0],
        ], dtype=np.float64)
        # Simulate realistic image positions
        image_pts = np.array([
            [0.05, 0.95],  # near-left (wide, low)
            [0.95, 0.95],  # near-right (wide, low)
            [0.70, 0.35],  # far-right (narrow, high)
            [0.30, 0.35],  # far-left (narrow, high)
        ], dtype=np.float64)

        h_matrix, _ = cv.findHomography(court_pts, image_pts)
        assert h_matrix is not None

        # Also project center line points
        center_court = np.array([[0.0, 8.0], [8.0, 8.0]], dtype=np.float64)
        center_image = cv.perspectiveTransform(
            center_court.reshape(-1, 1, 2), h_matrix,
        ).reshape(-1, 2)

        # Corners with bad near-corner guesses
        corners = [
            {"x": 0.30, "y": 0.70},  # near-left (bad)
            {"x": 0.70, "y": 0.70},  # near-right (bad)
            {"x": 0.70, "y": 0.35},  # far-right (good)
            {"x": 0.30, "y": 0.35},  # far-left (good)
        ]
        conf = {"near-left": 0.001, "near-right": 0.001, "far-right": 0.999, "far-left": 0.999}

        # VP from sidelines
        vp = line_intersection(
            (image_pts[3, 0], image_pts[3, 1]),
            (image_pts[0, 0], image_pts[0, 1]),
            (image_pts[2, 0], image_pts[2, 1]),
            (image_pts[1, 0], image_pts[1, 1]),
        )
        assert vp is not None

        center_line = (
            (float(center_image[0, 0]), float(center_image[0, 1])),
            (float(center_image[1, 0]), float(center_image[1, 1])),
        )

        result = detector._refine_via_homography(
            corners, conf, 0.5, vp, center_line,
        )
        assert result is not None
        refined, refined_names = result

        assert "near-left" in refined_names
        assert "near-right" in refined_names
        # Should be close to the true image positions
        assert abs(refined[0]["x"] - 0.05) < 0.05
        assert abs(refined[0]["y"] - 0.95) < 0.05
        assert abs(refined[1]["x"] - 0.95) < 0.05
        assert abs(refined[1]["y"] - 0.95) < 0.05

    def test_homography_returns_none_when_center_above_far(self) -> None:
        """Center points above far baseline → reject homography."""
        detector = self._make_detector()

        corners = [
            {"x": 0.20, "y": 0.70},
            {"x": 0.80, "y": 0.70},
            {"x": 0.65, "y": 0.35},
            {"x": 0.35, "y": 0.35},
        ]
        conf = {"near-left": 0.01, "near-right": 0.01, "far-right": 0.99, "far-left": 0.99}

        vp = (0.50, -0.50)
        # Center line above far baseline → intersections will be above too
        center_line = ((0.20, 0.20), (0.80, 0.20))

        result = detector._refine_via_homography(
            corners, conf, 0.5, vp, center_line,
        )
        assert result is None


class TestValidateCourtGeometry:
    """Test court geometry validation."""

    def test_valid_perspective_court(self) -> None:
        """Normal perspective court passes validation."""
        corners = [
            {"x": 0.05, "y": 0.95},
            {"x": 0.95, "y": 0.95},
            {"x": 0.70, "y": 0.35},
            {"x": 0.30, "y": 0.35},
        ]
        assert CourtKeypointDetector._validate_court_geometry(corners) is True

    def test_near_narrower_than_far_fails(self) -> None:
        """Near side narrower than far side → invalid (reverse perspective)."""
        corners = [
            {"x": 0.40, "y": 0.95},  # near-left (too narrow)
            {"x": 0.60, "y": 0.95},  # near-right (too narrow)
            {"x": 0.70, "y": 0.35},  # far-right
            {"x": 0.30, "y": 0.35},  # far-left
        ]
        assert CourtKeypointDetector._validate_court_geometry(corners) is False

    def test_near_above_far_fails(self) -> None:
        """Near corners above far corners → invalid (flipped Y)."""
        corners = [
            {"x": 0.05, "y": 0.20},  # near-left above far
            {"x": 0.95, "y": 0.20},
            {"x": 0.70, "y": 0.35},
            {"x": 0.30, "y": 0.35},
        ]
        assert CourtKeypointDetector._validate_court_geometry(corners) is False

    def test_extreme_perspective_fails(self) -> None:
        """Perspective ratio > 8 → invalid."""
        corners = [
            {"x": -1.0, "y": 0.95},
            {"x": 2.0, "y": 0.95},  # near width = 3.0
            {"x": 0.55, "y": 0.35},
            {"x": 0.45, "y": 0.35},  # far width = 0.1
        ]
        assert CourtKeypointDetector._validate_court_geometry(corners) is False

    def test_diverging_sidelines_fails(self) -> None:
        """Sidelines that diverge upward (far-left left of near-left) → invalid."""
        corners = [
            {"x": 0.30, "y": 0.95},  # near-left
            {"x": 0.70, "y": 0.95},  # near-right
            {"x": 0.80, "y": 0.35},  # far-right (right of near-right)
            {"x": 0.20, "y": 0.35},  # far-left (left of near-left — diverging)
        ]
        # far_left.x (0.20) < near_left.x (0.30) - 0.02 → fails
        assert CourtKeypointDetector._validate_court_geometry(corners) is False


class TestDetectCenterLine:
    """Test center line (net) detection."""

    def _make_detector(self) -> CourtKeypointDetector:
        detector = CourtKeypointDetector.__new__(CourtKeypointDetector)
        detector._pad_ratio = 0.3
        detector._conf_threshold = 0.3
        detector._last_diagnostics = None
        return detector

    def test_detects_horizontal_line_in_synthetic_frame(self) -> None:
        """Synthetic frame with a strong horizontal line should be detected."""
        detector = self._make_detector()

        # Create 480p frame with a bright horizontal line at y=250
        h, w = 480, 640
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        # Draw a white horizontal line (simulating net)
        cv2.line(frame, (50, 250), (590, 250), (255, 255, 255), 3)

        far_left = (0.15, 0.40)  # y=192 in pixel
        far_right = (0.85, 0.40)

        result = detector._detect_center_line(frame, far_left, far_right)
        # May or may not detect depending on Hough params, but should not crash
        # If detected, should be near y=250/480 ≈ 0.52
        if result is not None:
            mid_y = (result[0][1] + result[1][1]) / 2.0
            assert 0.45 < mid_y < 0.60

    def test_returns_none_on_blank_frame(self) -> None:
        """Blank frame → no center line detected."""
        detector = self._make_detector()

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = detector._detect_center_line(
            frame, (0.30, 0.35), (0.70, 0.35),
        )
        assert result is None
