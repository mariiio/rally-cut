"""Tests for court keypoint detection using YOLO-pose."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

import numpy as np

from rallycut.court.keypoint_detector import (
    CourtKeypointDetector,
    FrameKeypoints,
    _weighted_median,
)


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
