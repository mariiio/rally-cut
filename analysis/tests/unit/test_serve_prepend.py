"""Unit tests for the v1.3 serve-prepend gate.

The predicate fires when MS-TCN++ has a strong serve-class peak before
the first classified action. Five conjunctive conditions; see
`docs/superpowers/specs/2026-05-11-serve-peak-prepend-design.md`.
"""
from __future__ import annotations

import numpy as np

from rallycut.actions.trajectory_features import ACTION_TYPES
from rallycut.tracking.serve_prepend import (
    SERVE_PREPEND_FIRST_ACTION_SERVE_CEIL,
    SERVE_PREPEND_GUARD_FRAMES,
    SERVE_PREPEND_MIN_GAP,
    SERVE_PREPEND_PEAK_FLOOR,
    should_prepend_serve,
)

SERVE_IDX = ACTION_TYPES.index("serve") + 1


def _seq_with_serve_peak(peak_frame: int, peak_prob: float, length: int) -> np.ndarray:
    """Build a fake MS-TCN++ output: serve-class peak at `peak_frame`, low elsewhere."""
    n_classes = len(ACTION_TYPES) + 1  # +1 background class
    seq = np.full((n_classes, length), 0.01, dtype=np.float32)
    # Bell around peak_frame
    for f in range(max(0, peak_frame - 5), min(length, peak_frame + 6)):
        d = abs(f - peak_frame)
        seq[SERVE_IDX, f] = max(0.01, peak_prob * (1 - d / 7.0))
    seq[SERVE_IDX, peak_frame] = peak_prob
    return seq


class TestShouldPrependServe:
    def test_textbook_fire(self) -> None:
        seq = _seq_with_serve_peak(peak_frame=110, peak_prob=0.99, length=500)
        result = should_prepend_serve(
            sequence_probs=seq,
            first_action_frame=426,
            first_action_serve_prob=0.02,
            rally_start_frame=0,
        )
        assert result == 110

    def test_none_sequence_probs_returns_none(self) -> None:
        assert should_prepend_serve(
            sequence_probs=None,
            first_action_frame=200,
            first_action_serve_prob=0.05,
            rally_start_frame=0,
        ) is None

    def test_first_action_serve_prob_too_high_returns_none(self) -> None:
        """If first action itself looks like a confident serve, don't override."""
        seq = _seq_with_serve_peak(peak_frame=50, peak_prob=0.99, length=300)
        assert should_prepend_serve(
            sequence_probs=seq,
            first_action_frame=200,
            first_action_serve_prob=SERVE_PREPEND_FIRST_ACTION_SERVE_CEIL,
            rally_start_frame=0,
        ) is None

    def test_peak_below_floor_returns_none(self) -> None:
        seq = _seq_with_serve_peak(peak_frame=100, peak_prob=SERVE_PREPEND_PEAK_FLOOR - 0.01,
                                    length=400)
        assert should_prepend_serve(
            sequence_probs=seq,
            first_action_frame=400,
            first_action_serve_prob=0.05,
            rally_start_frame=0,
        ) is None

    def test_gap_below_min_returns_none(self) -> None:
        """Buildup peak just before a correctly detected serve — don't prepend."""
        seq = _seq_with_serve_peak(peak_frame=95, peak_prob=0.99, length=200)
        assert should_prepend_serve(
            sequence_probs=seq,
            first_action_frame=95 + SERVE_PREPEND_MIN_GAP - 1,
            first_action_serve_prob=0.05,
            rally_start_frame=0,
        ) is None

    def test_gap_exactly_at_min_fires(self) -> None:
        seq = _seq_with_serve_peak(peak_frame=95, peak_prob=0.99, length=300)
        result = should_prepend_serve(
            sequence_probs=seq,
            first_action_frame=95 + SERVE_PREPEND_MIN_GAP,
            first_action_serve_prob=0.05,
            rally_start_frame=0,
        )
        assert result == 95

    def test_window_too_short_returns_none(self) -> None:
        """If rally_start ≥ first_action - guard, the search window is empty."""
        seq = _seq_with_serve_peak(peak_frame=10, peak_prob=0.99, length=100)
        assert should_prepend_serve(
            sequence_probs=seq,
            first_action_frame=SERVE_PREPEND_GUARD_FRAMES + 5,
            first_action_serve_prob=0.05,
            rally_start_frame=SERVE_PREPEND_GUARD_FRAMES + 4,
        ) is None

    def test_constants_match_calibration(self) -> None:
        """Lock the constants — they came from a 338-rally fleet sweep.
        Re-tuning requires re-validation."""
        assert SERVE_PREPEND_PEAK_FLOOR == 0.95
        assert SERVE_PREPEND_MIN_GAP == 25
        assert SERVE_PREPEND_FIRST_ACTION_SERVE_CEIL == 0.50
        assert SERVE_PREPEND_GUARD_FRAMES == 15
