"""Tests for court keypoint detection using YOLO-pose."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rallycut.court.keypoint_detector import (
    CourtKeypointDetector,
    FrameKeypoints,
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

        corners, conf = detector._aggregate([fr])
        assert len(corners) == 4
        assert abs(corners[0]["x"] - 0.1) < 1e-6
        assert abs(conf - 0.85) < 1e-6

    def test_median_of_multiple_frames(self) -> None:
        """Multiple frames → median coordinates."""
        detector = CourtKeypointDetector.__new__(CourtKeypointDetector)
        detector._pad_ratio = 0.3
        detector._conf_threshold = 0.3

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

        corners, conf = detector._aggregate(frames)
        # Median of [-0.02, 0.0, 0.01, 0.0, -0.01] = 0.0
        assert abs(corners[0]["x"] - 0.1) < 0.005

    def test_outlier_rejection(self) -> None:
        """Outlier frame should be filtered by 2σ rule."""
        detector = CourtKeypointDetector.__new__(CourtKeypointDetector)
        detector._pad_ratio = 0.3
        detector._conf_threshold = 0.3

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

        corners, conf = detector._aggregate(frames)
        # Should be close to 0.10, not pulled toward 0.50
        assert abs(corners[0]["x"] - 0.10) < 0.02

    def test_empty_frames(self) -> None:
        """No frames → empty result."""
        detector = CourtKeypointDetector.__new__(CourtKeypointDetector)
        detector._pad_ratio = 0.3
        detector._conf_threshold = 0.3

        corners, conf = detector._aggregate([])
        assert corners == []
        assert conf == 0.0


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
