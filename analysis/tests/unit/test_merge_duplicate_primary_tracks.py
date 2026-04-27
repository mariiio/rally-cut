"""Tests for `merge_duplicate_primary_tracks` in player_filter.py.

Pins the regression caught during wawa rally 9 diagnosis: YOLO + BoT-SORT
emitted two simultaneous tracks on the same physical body (T2 and T5 with
mean IoU ≈ 1.0 across 503 overlapping frames) and `identify_primary_tracks`
accepted both. Downstream consumers (PlayerMatchingDialog, match-players)
saw the same player twice and squeezed out the legitimate fourth player.

The merge pass detects this case via simultaneous-frame IoU and drops the
shorter track, keeping the primary set at 4 distinct physical bodies.
"""

from __future__ import annotations

import random

from rallycut.tracking.player_filter import (
    TrackStats,
    merge_duplicate_primary_tracks,
)
from rallycut.tracking.player_tracker import PlayerPosition


def _pos(
    track_id: int,
    n_frames: int,
    *,
    f_start: int = 0,
    x: float = 0.5,
    y: float = 0.5,
    width: float = 0.05,
    height: float = 0.20,
    pos_jitter: float = 0.005,
    seed: int = 0,
) -> list[PlayerPosition]:
    rng = random.Random(track_id * 1000 + seed)
    return [
        PlayerPosition(
            frame_number=f_start + i,
            track_id=track_id,
            x=x + rng.uniform(-pos_jitter, pos_jitter),
            y=y + rng.uniform(-pos_jitter, pos_jitter),
            width=width,
            height=height,
            confidence=0.9,
        )
        for i in range(n_frames)
    ]


def _stub_stats(track_id: int, frames: int) -> TrackStats:
    return TrackStats(
        track_id=track_id,
        frame_count=frames,
        total_frames=frames,
        avg_bbox_area=0.05 * 0.20,
        avg_confidence=0.9,
        first_frame=0,
        last_frame=frames - 1,
        avg_x=0.5,
        avg_y=0.5,
        x_std=0.01,
        y_std=0.01,
    )


class TestMergeDuplicatePrimaryTracks:
    def test_drops_short_track_subsumed_by_long_at_same_position(self) -> None:
        """The wawa rally 9 case: T2 (835 frames) and T5 (503 frames, all
        simultaneous, IoU ≈ 1.0). T5 must drop, T2 must stay."""
        # Long track present from frame 0
        long_pos = _pos(2, 835, f_start=0, x=0.45, y=0.49)
        # Short track present from frame 332, fully simultaneous with long
        short_pos = _pos(5, 503, f_start=332, x=0.45, y=0.50)

        primaries = {2, 5}
        stats = {2: _stub_stats(2, 835), 5: _stub_stats(5, 503)}
        result = merge_duplicate_primary_tracks(
            long_pos + short_pos, primaries, stats,
        )
        assert result == {2}

    def test_keeps_both_when_iou_below_threshold(self) -> None:
        """Two simultaneous tracks at clearly separated positions stay."""
        a = _pos(1, 200, x=0.30, y=0.50)
        b = _pos(2, 200, x=0.70, y=0.50)
        primaries = {1, 2}
        stats = {1: _stub_stats(1, 200), 2: _stub_stats(2, 200)}
        result = merge_duplicate_primary_tracks(a + b, primaries, stats)
        assert result == {1, 2}

    def test_keeps_both_when_overlap_fraction_below_threshold(self) -> None:
        """Two tracks that briefly co-exist (e.g., blocker passes attacker
        at the net for ~10 frames) must NOT be merged. Even high IoU on
        those few frames isn't enough — the overlap-fraction guard catches
        this case."""
        long_track = _pos(1, 300, f_start=0, x=0.50, y=0.50)
        # Short track of 100 frames, only 20 overlap with long_track's
        # range (still 20% overlap, well below 50% min_overlap_fraction).
        short_track = _pos(
            2, 100, f_start=280, x=0.50, y=0.50,  # high IoU on overlap
        )
        primaries = {1, 2}
        stats = {1: _stub_stats(1, 300), 2: _stub_stats(2, 100)}
        result = merge_duplicate_primary_tracks(
            long_track + short_track, primaries, stats,
            min_overlap_fraction=0.5,
        )
        assert result == {1, 2}

    def test_non_overlapping_tracks_kept(self) -> None:
        """Sequential tracks (BoT-SORT ID switch) are NOT merged by this
        pass — that's a separate problem and a different fix.
        """
        first = _pos(1, 200, f_start=0, x=0.50, y=0.50)
        second = _pos(2, 200, f_start=300, x=0.50, y=0.50)  # no overlap
        primaries = {1, 2}
        stats = {1: _stub_stats(1, 200), 2: _stub_stats(2, 200)}
        result = merge_duplicate_primary_tracks(
            first + second, primaries, stats,
        )
        assert result == {1, 2}

    def test_three_tracks_one_duplicate(self) -> None:
        """4 primaries with one duplicate pair → keeps 3 distinct bodies."""
        a = _pos(1, 400, x=0.30, y=0.50)
        b_long = _pos(2, 400, x=0.50, y=0.50)
        b_dup = _pos(5, 350, f_start=25, x=0.50, y=0.50)  # duplicate of b_long
        c = _pos(3, 400, x=0.70, y=0.50)
        primaries = {1, 2, 3, 5}
        stats = {
            1: _stub_stats(1, 400),
            2: _stub_stats(2, 400),
            3: _stub_stats(3, 400),
            5: _stub_stats(5, 350),
        }
        result = merge_duplicate_primary_tracks(
            a + b_long + b_dup + c, primaries, stats,
        )
        assert result == {1, 2, 3}

    def test_singleton_primary_set_unchanged(self) -> None:
        """No work to do when there's only one primary track."""
        a = _pos(1, 100)
        result = merge_duplicate_primary_tracks(a, {1}, {1: _stub_stats(1, 100)})
        assert result == {1}

    def test_empty_primary_set_unchanged(self) -> None:
        result = merge_duplicate_primary_tracks([], set(), {})
        assert result == set()
