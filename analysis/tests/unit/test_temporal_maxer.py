"""Tests for TemporalMaxer model."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from rallycut.temporal.temporal_maxer.inference import TemporalMaxerInference
from rallycut.temporal.temporal_maxer.model import TemporalMaxer, TemporalMaxerConfig
from rallycut.temporal.temporal_maxer.training import (
    VideoSequenceDataset,
    collate_video_sequences,
    compute_tmse_loss,
)


class TestTemporalMaxerModel:
    def test_forward_shape(self) -> None:
        config = TemporalMaxerConfig(feature_dim=768, num_classes=2, num_layers=4)
        model = TemporalMaxer(config)

        # Input: (batch=1, feature_dim=768, T=100)
        x = torch.randn(1, 768, 100)
        logits = model(x)

        assert logits.shape == (1, 2, 100)

    def test_forward_with_mask(self) -> None:
        config = TemporalMaxerConfig(feature_dim=768, num_classes=2, num_layers=2)
        model = TemporalMaxer(config)

        x = torch.randn(2, 768, 50)
        mask = torch.ones(2, 1, 50)
        mask[1, :, 30:] = 0  # Second sequence shorter

        logits = model(x, mask)
        assert logits.shape == (2, 2, 50)

    def test_forward_short_sequence(self) -> None:
        config = TemporalMaxerConfig(feature_dim=768, num_classes=2, num_layers=2)
        model = TemporalMaxer(config)

        x = torch.randn(1, 768, 5)
        logits = model(x)
        assert logits.shape == (1, 2, 5)

    def test_small_feature_dim(self) -> None:
        config = TemporalMaxerConfig(feature_dim=32, hidden_dim=32, num_classes=2, num_layers=2)
        model = TemporalMaxer(config)

        x = torch.randn(1, 32, 20)
        logits = model(x)
        assert logits.shape == (1, 2, 20)

    def test_gradient_flow(self) -> None:
        """Verify gradients flow through the model."""
        config = TemporalMaxerConfig(feature_dim=32, hidden_dim=32, num_classes=2, num_layers=2)
        model = TemporalMaxer(config)

        x = torch.randn(1, 32, 20, requires_grad=True)
        logits = model(x)
        loss = logits.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.all(x.grad == 0)


class TestTMSELoss:
    def test_constant_predictions_zero_loss(self) -> None:
        """Constant predictions should have zero TMSE."""
        logits = torch.zeros(1, 2, 10)
        logits[:, 0, :] = 1.0  # All same class
        mask = torch.ones(1, 1, 10)

        loss = compute_tmse_loss(logits, mask)
        assert loss.item() < 1e-6

    def test_varying_predictions_positive_loss(self) -> None:
        """Varying predictions should have positive TMSE."""
        logits = torch.zeros(1, 2, 10)
        logits[:, 0, ::2] = 2.0  # Alternating
        logits[:, 1, 1::2] = 2.0
        mask = torch.ones(1, 1, 10)

        loss = compute_tmse_loss(logits, mask)
        assert loss.item() > 0

    def test_masked_positions_ignored(self) -> None:
        """Masked positions should not contribute to loss."""
        logits = torch.zeros(1, 2, 10)
        logits[:, 0, ::2] = 2.0
        logits[:, 1, 1::2] = 2.0
        mask = torch.zeros(1, 1, 10)  # All masked

        loss = compute_tmse_loss(logits, mask)
        assert loss.item() == 0.0


class TestCollation:
    def test_collate_equal_length(self) -> None:
        dataset = VideoSequenceDataset(
            [np.random.randn(10, 32).astype(np.float32)] * 2,
            [np.zeros(10).astype(np.int64)] * 2,
        )
        batch = [dataset[0], dataset[1]]
        features, labels, mask = collate_video_sequences(batch)

        assert features.shape == (2, 32, 10)
        assert labels.shape == (2, 10)
        assert mask.shape == (2, 1, 10)
        assert torch.all(mask == 1.0)

    def test_collate_variable_length(self) -> None:
        dataset = VideoSequenceDataset(
            [
                np.random.randn(10, 32).astype(np.float32),
                np.random.randn(5, 32).astype(np.float32),
            ],
            [
                np.zeros(10).astype(np.int64),
                np.zeros(5).astype(np.int64),
            ],
        )
        batch = [dataset[0], dataset[1]]
        features, labels, mask = collate_video_sequences(batch)

        assert features.shape == (2, 32, 10)  # Padded to max_len=10
        assert labels.shape == (2, 10)
        assert mask.shape == (2, 1, 10)
        # Second sequence: first 5 valid, rest masked
        assert torch.all(mask[1, 0, :5] == 1.0)
        assert torch.all(mask[1, 0, 5:] == 0.0)

    def test_collate_empty(self) -> None:
        features, labels, mask = collate_video_sequences([])
        assert features.shape[0] == 0


def _make_inference() -> TemporalMaxerInference:
    """Create a TemporalMaxerInference without loading a model."""
    obj = object.__new__(TemporalMaxerInference)
    obj.device = torch.device("cpu")
    obj.model = None  # type: ignore[assignment]
    return obj


class TestPredictionsToSegments:
    """Tests for _predictions_to_segments segment extraction logic."""

    def test_single_rally_end_time(self) -> None:
        """End time should be end of last rally window, not first non-rally window."""
        inf = _make_inference()
        # stride=48, fps=30 → window_duration=1.6s, window_length=16/30≈0.533s
        window_duration = 48 / 30  # 1.6s
        window_length = 16 / 30  # 0.533s

        # Windows 2,3,4 are rally (indices 2-4), rest are non-rally
        predictions = np.array([0, 0, 1, 1, 1, 0, 0, 0])
        probs = np.ones(8) * 0.9

        segments = inf._predictions_to_segments(
            predictions, probs, window_duration, window_length,
            min_duration=0.0, max_gap=0.0, max_duration=999.0,
        )

        assert len(segments) == 1
        start, end = segments[0]
        # Start = window 2 * 1.6 = 3.2s
        assert start == pytest.approx(2 * window_duration)
        # End = window 4 * 1.6 + 0.533 = 6.933s (last rally window end)
        expected_end = 4 * window_duration + window_length
        assert end == pytest.approx(expected_end)
        # NOT 5 * 1.6 + 0.533 (which would be the old buggy value)
        buggy_end = 5 * window_duration + window_length
        assert end < buggy_end

    def test_rally_at_end_of_video(self) -> None:
        """Rally extending to last window uses (n-1)*window_duration + window_length."""
        inf = _make_inference()
        window_duration = 1.6
        window_length = 0.533

        # Last 3 windows are rally
        predictions = np.array([0, 0, 1, 1, 1])
        probs = np.ones(5) * 0.9

        segments = inf._predictions_to_segments(
            predictions, probs, window_duration, window_length,
            min_duration=0.0, max_gap=0.0, max_duration=999.0,
        )

        assert len(segments) == 1
        start, end = segments[0]
        assert start == pytest.approx(2 * window_duration)
        # n=5, last window index=4: end = 4 * 1.6 + 0.533
        assert end == pytest.approx(4 * window_duration + window_length)

    def test_gap_filling(self) -> None:
        """Two rallies separated by gap <= max_gap should merge."""
        inf = _make_inference()
        window_duration = 1.0
        window_length = 0.5

        # Two rallies with 1 non-rally window between them
        predictions = np.array([1, 1, 0, 1, 1])
        probs = np.ones(5) * 0.9

        # max_gap=1.5 should merge (gap = start of second rally - end of first)
        segments = inf._predictions_to_segments(
            predictions, probs, window_duration, window_length,
            min_duration=0.0, max_gap=2.0, max_duration=999.0,
        )
        assert len(segments) == 1

        # max_gap=0.0 should keep separate
        segments = inf._predictions_to_segments(
            predictions, probs, window_duration, window_length,
            min_duration=0.0, max_gap=0.0, max_duration=999.0,
        )
        assert len(segments) == 2

    def test_short_segment_removal(self) -> None:
        """Segments shorter than min_duration should be removed."""
        inf = _make_inference()
        window_duration = 1.0
        window_length = 0.5

        # Single rally window → duration = window_length = 0.5s
        predictions = np.array([0, 1, 0, 0])
        probs = np.ones(4) * 0.9

        # min_duration=1.0 should remove it
        segments = inf._predictions_to_segments(
            predictions, probs, window_duration, window_length,
            min_duration=1.0, max_gap=0.0, max_duration=999.0,
        )
        assert len(segments) == 0

        # min_duration=0.0 should keep it
        segments = inf._predictions_to_segments(
            predictions, probs, window_duration, window_length,
            min_duration=0.0, max_gap=0.0, max_duration=999.0,
        )
        assert len(segments) == 1

    def test_empty_predictions(self) -> None:
        """Empty input should return empty segments."""
        inf = _make_inference()
        segments = inf._predictions_to_segments(
            np.array([]), np.array([]), 1.0, 0.5, 0.0, 0.0, 999.0,
        )
        assert segments == []

    def test_no_rally_windows(self) -> None:
        """All non-rally predictions should return empty segments."""
        inf = _make_inference()
        predictions = np.array([0, 0, 0, 0])
        probs = np.ones(4) * 0.1
        segments = inf._predictions_to_segments(
            predictions, probs, 1.0, 0.5, 0.0, 0.0, 999.0,
        )
        assert segments == []


class TestValleySplitting:
    """Tests for probability-valley splitting in _predictions_to_segments."""

    def test_valley_splits_merged_rallies(self) -> None:
        """A sustained low-probability valley should split a segment into two."""
        inf = _make_inference()
        # window_duration=0.8s, window_length=0.533s
        # 5 high-prob windows, 4 low-prob windows (<0.5), 5 high-prob windows
        predictions = np.ones(14, dtype=int)  # All rally
        probs = np.array([
            0.9, 0.9, 0.9, 0.9, 0.9,    # Rally 1
            0.2, 0.3, 0.2, 0.1,          # Dead time (4 windows = 3.2s > 2.0s)
            0.9, 0.9, 0.9, 0.9, 0.9,     # Rally 2
        ])
        segments = inf._predictions_to_segments(
            predictions, probs, 0.8, 0.533,
            min_duration=1.0, max_gap=5.0, max_duration=999.0,
            valley_threshold=0.5, min_valley_duration=2.0,
        )
        assert len(segments) == 2
        # First segment ends at valley start, second starts at valley end
        assert segments[0][1] == pytest.approx(5 * 0.8)  # 4.0s
        assert segments[1][0] == pytest.approx(9 * 0.8)  # 7.2s

    def test_short_valley_no_split(self) -> None:
        """A valley shorter than min_valley_duration should not cause a split."""
        inf = _make_inference()
        # 5 high, 1 low (0.8s < 2.0s min), 5 high
        predictions = np.ones(11, dtype=int)
        probs = np.array([
            0.9, 0.9, 0.9, 0.9, 0.9,
            0.2,  # Single low window = 0.8s
            0.9, 0.9, 0.9, 0.9, 0.9,
        ])
        segments = inf._predictions_to_segments(
            predictions, probs, 0.8, 0.533,
            min_duration=0.0, max_gap=5.0, max_duration=999.0,
            valley_threshold=0.5, min_valley_duration=2.0,
        )
        assert len(segments) == 1

    def test_multiple_valleys(self) -> None:
        """Two qualifying valleys should produce 3 segments."""
        inf = _make_inference()
        # Rally, valley, rally, valley, rally
        predictions = np.ones(19, dtype=int)
        probs = np.array([
            0.9, 0.9, 0.9, 0.9,           # Rally 1 (4 windows)
            0.1, 0.2, 0.1,                 # Valley 1 (3 windows = 2.4s)
            0.9, 0.9, 0.9, 0.9,            # Rally 2 (4 windows)
            0.2, 0.1, 0.2,                 # Valley 2 (3 windows = 2.4s)
            0.9, 0.9, 0.9, 0.9, 0.9,       # Rally 3 (5 windows)
        ])
        segments = inf._predictions_to_segments(
            predictions, probs, 0.8, 0.533,
            min_duration=1.0, max_gap=5.0, max_duration=999.0,
            valley_threshold=0.5, min_valley_duration=2.0,
        )
        assert len(segments) == 3

    def test_disabled_with_zero_threshold(self) -> None:
        """Valley splitting should be disabled when threshold=0."""
        inf = _make_inference()
        predictions = np.ones(14, dtype=int)
        probs = np.array([
            0.9, 0.9, 0.9, 0.9, 0.9,
            0.2, 0.3, 0.2, 0.1,  # Would be valley
            0.9, 0.9, 0.9, 0.9, 0.9,
        ])
        segments = inf._predictions_to_segments(
            predictions, probs, 0.8, 0.533,
            min_duration=0.0, max_gap=5.0, max_duration=999.0,
            valley_threshold=0.0, min_valley_duration=2.0,
        )
        assert len(segments) == 1

    def test_short_sub_segment_filtered(self) -> None:
        """Sub-segments shorter than min_duration after split should be removed."""
        inf = _make_inference()
        # 1 high window (0.533s < 1.0s min), valley, 5 high windows
        predictions = np.ones(9, dtype=int)
        probs = np.array([
            0.9,                           # Tiny rally (1 window = 0.533s)
            0.1, 0.2, 0.1,                # Valley (3 windows = 2.4s)
            0.9, 0.9, 0.9, 0.9, 0.9,      # Real rally (5 windows)
        ])
        segments = inf._predictions_to_segments(
            predictions, probs, 0.8, 0.533,
            min_duration=1.0, max_gap=5.0, max_duration=999.0,
            valley_threshold=0.5, min_valley_duration=2.0,
        )
        # First sub-segment (0.0 to 0.8) is only 0.8s < 1.0s min_duration → filtered
        assert len(segments) == 1
        assert segments[0][0] == pytest.approx(4 * 0.8)  # Starts after valley

    def test_leading_and_trailing_valley(self) -> None:
        """Valleys at the start or end of a segment should trim correctly."""
        inf = _make_inference()
        # Leading valley (3 windows), rally (5 windows), trailing valley (3 windows)
        predictions = np.ones(11, dtype=int)
        probs = np.array([
            0.1, 0.2, 0.1,                # Leading valley
            0.9, 0.9, 0.9, 0.9, 0.9,      # Real rally (5 windows)
            0.2, 0.1, 0.2,                # Trailing valley
        ])
        segments = inf._predictions_to_segments(
            predictions, probs, 0.8, 0.533,
            min_duration=1.0, max_gap=5.0, max_duration=999.0,
            valley_threshold=0.5, min_valley_duration=2.0,
        )
        assert len(segments) == 1
        # Rally starts after leading valley
        assert segments[0][0] == pytest.approx(3 * 0.8)
        # Rally ends at original segment end (trailing valley trims it)
        assert segments[0][1] < 11 * 0.8  # Shorter than full segment
