"""Tests for within-track appearance segmentation (Task 2-5)."""
from __future__ import annotations

import pytest

from rallycut.tracking._subtrack import SubTrackCandidate
from rallycut.tracking.player_features import TrackAppearanceStats


def _make_stats(track_id: int) -> TrackAppearanceStats:
    return TrackAppearanceStats(track_id=track_id)


def test_subtrack_candidate_carries_parent_and_window() -> None:
    parent_stats = _make_stats(track_id=2)
    sub = SubTrackCandidate(
        parent_track_id=2,
        segment_index=0,
        f_start=0,
        f_end=110,
        appearance_stats=parent_stats,
        aggregated_argmax_pid=2,
        aggregated_margin=0.18,
    )
    assert sub.parent_track_id == 2
    assert sub.segment_index == 0
    assert sub.f_start == 0
    assert sub.f_end == 110
    assert sub.appearance_stats is parent_stats
    # Formula: -1000 * (segment_index + 1) - parent_track_id - 2
    # For parent=2, seg=0: -1000 * 1 - 2 - 2 = -1004
    assert sub.synthetic_track_id == -1004
    assert sub.aggregated_argmax_pid == 2
    assert sub.aggregated_margin == pytest.approx(0.18)


def test_subtrack_candidate_synthetic_track_id_is_unique_per_segment() -> None:
    sub_a = SubTrackCandidate(
        parent_track_id=2,
        segment_index=0,
        f_start=0,
        f_end=100,
        appearance_stats=_make_stats(2),
    )
    sub_b = SubTrackCandidate(
        parent_track_id=2,
        segment_index=1,
        f_start=100,
        f_end=200,
        appearance_stats=_make_stats(2),
    )
    assert sub_a.synthetic_track_id != sub_b.synthetic_track_id
    # Synthetic ids must not collide with real positive track_ids
    assert sub_a.synthetic_track_id < 0
    assert sub_b.synthetic_track_id < 0


def test_subtrack_candidate_overlap_detection() -> None:
    a = SubTrackCandidate(
        parent_track_id=2,
        segment_index=0,
        f_start=0,
        f_end=100,
        appearance_stats=_make_stats(2),
    )
    b = SubTrackCandidate(
        parent_track_id=3,
        segment_index=0,
        f_start=50,
        f_end=150,
        appearance_stats=_make_stats(3),
    )
    c = SubTrackCandidate(
        parent_track_id=4,
        segment_index=0,
        f_start=200,
        f_end=300,
        appearance_stats=_make_stats(4),
    )
    assert a.overlaps(b)
    assert b.overlaps(a)
    assert not a.overlaps(c)
    assert not c.overlaps(b)
