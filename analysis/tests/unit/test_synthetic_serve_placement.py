"""Unit tests for synthetic_serve_placement.pick_synthetic_serve_frame."""

from __future__ import annotations

import numpy as np

from rallycut.tracking.synthetic_serve_placement import (
    MAX_PRESERVE_FRAMES,
    SEARCH_GUARD,
    SERVE_SEQ_FLOOR,
    pick_synthetic_serve_frame,
)


def test_returns_seq_peak_frame() -> None:
    """Strong seq peak in window -> return that frame."""
    seq = np.zeros((7, 400))
    seq[1, 80] = 0.85  # serve-class peak at frame 80 (index 1 in seq_probs)
    result = pick_synthetic_serve_frame(
        sequence_probs=seq,
        rally_start_frame=0,
        first_contact_frame=200,
    )
    assert result == 80, result


def test_returns_none_when_seq_peak_below_floor() -> None:
    """Seq peak below SERVE_SEQ_FLOOR -> None."""
    seq = np.zeros((7, 400))
    seq[1, 80] = SERVE_SEQ_FLOOR - 0.01
    result = pick_synthetic_serve_frame(
        sequence_probs=seq,
        rally_start_frame=0,
        first_contact_frame=200,
    )
    assert result is None


def test_returns_none_when_seq_probs_empty() -> None:
    """All-zero seq probs -> argmax falls below floor -> None."""
    seq = np.zeros((7, 400))
    result = pick_synthetic_serve_frame(
        sequence_probs=seq,
        rally_start_frame=0,
        first_contact_frame=200,
    )
    assert result is None


def test_returns_none_when_search_window_collapses() -> None:
    """rally_start within SEARCH_GUARD of first_contact -> nothing to search."""
    seq = np.zeros((7, 400))
    seq[1, 50] = 0.85
    result = pick_synthetic_serve_frame(
        sequence_probs=seq,
        rally_start_frame=100,
        first_contact_frame=100 + SEARCH_GUARD - 1,
    )
    assert result is None


def test_clamps_when_picked_frame_too_early() -> None:
    """Picked frame > MAX_PRESERVE_FRAMES before first_contact -> clamp."""
    seq = np.zeros((7, 400))
    seq[1, 10] = 0.85  # very early peak
    result = pick_synthetic_serve_frame(
        sequence_probs=seq,
        rally_start_frame=0,
        first_contact_frame=300,
    )
    assert result == 300 - MAX_PRESERVE_FRAMES, result


def test_picks_only_within_search_window() -> None:
    """Seq peaks outside [rally_start, first_contact - SEARCH_GUARD] are ignored."""
    seq = np.zeros((7, 400))
    seq[1, 350] = 0.95   # outside (after first_contact)
    seq[1, 80] = 0.60    # inside, below the 350 peak globally but should still be argmax inside window
    result = pick_synthetic_serve_frame(
        sequence_probs=seq,
        rally_start_frame=0,
        first_contact_frame=200,
    )
    assert result == 80, result


def test_constants_present() -> None:
    """Smoke check: the constants are exposed for monkey-patching."""
    assert isinstance(SERVE_SEQ_FLOOR, float)
    assert isinstance(SEARCH_GUARD, int)
    assert isinstance(MAX_PRESERVE_FRAMES, int)
