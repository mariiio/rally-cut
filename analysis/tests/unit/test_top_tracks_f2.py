"""Unit tests for the F2 non-player rejection / 2v2 cardinality gate
in ``MatchPlayerTracker._top_tracks_by_frames`` (MATCHER_VERSION v10).
"""
from __future__ import annotations

import pytest

from rallycut.tracking.match_tracker import (
    TOP_TRACKS_MIN_COURT_INTERIOR_FRACTION,
    TOP_TRACKS_MIN_FRAMES_FLOOR,
    MatchPlayerTracker,
)
from rallycut.tracking.player_features import (
    PlayerAppearanceFeatures,
    TrackAppearanceStats,
)
from rallycut.tracking.player_tracker import PlayerPosition


def _stats(track_id: int, num_features: int) -> TrackAppearanceStats:
    s = TrackAppearanceStats(track_id=track_id)
    for i in range(num_features):
        s.features.append(
            PlayerAppearanceFeatures(
                track_id=track_id,
                frame_number=i,
                skin_tone_hsv=(0.0, 0.0, 0.0),
                skin_pixel_count=10,
                upper_body_hist=None,
                lower_body_hist=None,
                upper_body_v_hist=None,
                lower_body_v_hist=None,
                dominant_color_hsv=(0.0, 0.0, 0.0),
            )
        )
    return s


def _pos(
    track_id: int, *, frame: int, x: float, y: float,
    width: float = 0.05, height: float = 0.15,
) -> PlayerPosition:
    return PlayerPosition(
        frame_number=frame,
        track_id=track_id,
        x=x,
        y=y,
        width=width,
        height=height,
        confidence=0.9,
    )


class _Calibrator:
    """Stand-in for CourtCalibrator that classifies foot points by a
    user-supplied predicate."""

    def __init__(self, in_court_predicate) -> None:
        self.is_calibrated = True
        self._pred = in_court_predicate

    def image_to_court(self, point, image_width: int, image_height: int):
        return point  # passthrough; the predicate uses image-space coords

    def is_point_in_court_with_margin(
        self,
        court_point,
        *,
        sideline_margin: float,
        baseline_margin: float,
    ) -> bool:
        return self._pred(court_point)


def _make_positions_for_track(
    tid: int, *, n: int, x: float, y: float,
) -> list[PlayerPosition]:
    return [_pos(tid, frame=f, x=x, y=y) for f in range(n)]


class TestLegacyPath:
    def test_returns_top_by_frames_when_fits(self) -> None:
        tracker = MatchPlayerTracker()
        ts = {1: _stats(1, 5), 2: _stats(2, 7), 3: _stats(3, 3)}
        assert tracker._top_tracks_by_frames([1, 2, 3], ts, 4) == [1, 2, 3]

    def test_returns_top_by_frames_when_no_context(self) -> None:
        tracker = MatchPlayerTracker()
        ts = {
            1: _stats(1, 10),
            2: _stats(2, 8),
            3: _stats(3, 6),
            4: _stats(4, 4),
            5: _stats(5, 2),
        }
        result = tracker._top_tracks_by_frames([1, 2, 3, 4, 5], ts, 4)
        assert result == [1, 2, 3, 4]


class TestCourtInteriorGate:
    def test_off_court_track_demoted(self) -> None:
        # Track 99 is the "photographer": always outside the court.
        # Track 1..4 are real players: always inside.
        # Track 99 has MORE frames than track 4, but should be demoted.
        calib = _Calibrator(lambda p: p[0] >= 0.1)  # in-court iff x >= 0.1
        tracker = MatchPlayerTracker(calibrator=calib)

        ts = {
            1: _stats(1, 100),
            2: _stats(2, 90),
            3: _stats(3, 80),
            4: _stats(4, 70),
            99: _stats(99, 200),  # most frames but off-court
        }
        positions = (
            _make_positions_for_track(1, n=20, x=0.5, y=0.5)
            + _make_positions_for_track(2, n=20, x=0.4, y=0.5)
            + _make_positions_for_track(3, n=20, x=0.6, y=0.4)
            + _make_positions_for_track(4, n=20, x=0.55, y=0.6)
            + _make_positions_for_track(99, n=20, x=0.02, y=0.5)  # off court
        )
        sides = {1: 0, 2: 0, 3: 1, 4: 1, 99: 0}
        result = tracker._top_tracks_by_frames(
            [1, 2, 3, 4, 99], ts, 4,
            track_court_sides=sides,
            player_positions=positions,
        )
        assert 99 not in result
        assert set(result) == {1, 2, 3, 4}

    def test_demoted_promoted_back_if_too_few_remain(self) -> None:
        # If we lose so many tracks to the court gate that fewer than n
        # remain, demoted candidates come back as fallback (better than
        # an empty slot).
        calib = _Calibrator(lambda p: False)  # everything is "off court"
        tracker = MatchPlayerTracker(calibrator=calib)
        ts = {1: _stats(1, 10), 2: _stats(2, 8), 3: _stats(3, 6)}
        positions = (
            _make_positions_for_track(1, n=20, x=0.5, y=0.5)
            + _make_positions_for_track(2, n=20, x=0.4, y=0.5)
            + _make_positions_for_track(3, n=20, x=0.6, y=0.4)
        )
        result = tracker._top_tracks_by_frames(
            [1, 2, 3], ts, 4,
            track_court_sides={1: 0, 2: 0, 3: 1},
            player_positions=positions,
        )
        # All 3 promoted back even though they all fail the court gate.
        assert sorted(result) == [1, 2, 3]


class TestCardinalityBalance:
    def test_three_one_split_rebalanced_to_two_two(self) -> None:
        tracker = MatchPlayerTracker()
        # Strict top-4 by frames is [1, 2, 3, 4]; sides are 3-near / 1-far.
        # Bench track 5 is far-side and has TOP_TRACKS_MIN_FRAMES_FLOOR + 1
        # samples — eligible to swap in.
        ts = {
            1: _stats(1, 100),
            2: _stats(2, 90),
            3: _stats(3, 80),
            4: _stats(4, 70),  # lowest near-side; gets swapped out
            5: _stats(5, 10),  # bench far-side; clears floor=6
        }
        # 1, 2, 4 near; 3 far in the strict top-4. Bench 5 is far.
        sides = {1: 0, 2: 0, 3: 1, 4: 0, 5: 1}
        result = tracker._top_tracks_by_frames(
            [1, 2, 3, 4, 5], ts, 4,
            track_court_sides=sides,
        )
        # Track 4 (lowest-frame near) is replaced with track 5 (bench far).
        # Frame ordering preserved within the chosen set; we just verify
        # composition.
        assert 5 in result
        assert 4 not in result
        # 2v2 result.
        near = [t for t in result if sides[t] == 0]
        far = [t for t in result if sides[t] == 1]
        assert len(near) == 2 and len(far) == 2

    def test_balanced_split_unchanged(self) -> None:
        tracker = MatchPlayerTracker()
        # Already 2-2: 1, 2 near; 3, 4 far.
        ts = {
            1: _stats(1, 100),
            2: _stats(2, 90),
            3: _stats(3, 80),
            4: _stats(4, 70),
            5: _stats(5, 10),  # bench (ignored when top-4 is balanced)
        }
        sides = {1: 0, 2: 0, 3: 1, 4: 1, 5: 0}
        result = tracker._top_tracks_by_frames(
            [1, 2, 3, 4, 5], ts, 4,
            track_court_sides=sides,
        )
        assert sorted(result) == [1, 2, 3, 4]

    def test_bench_below_floor_does_not_swap(self) -> None:
        tracker = MatchPlayerTracker()
        # Strict top-4 is 3-1; bench candidate exists for the under-rep
        # side but has fewer than TOP_TRACKS_MIN_FRAMES_FLOOR samples,
        # so the swap is suppressed and the 3-1 split persists.
        bench_frames = max(TOP_TRACKS_MIN_FRAMES_FLOOR - 1, 1)
        ts = {
            1: _stats(1, 100),
            2: _stats(2, 90),
            3: _stats(3, 80),
            4: _stats(4, 70),
            5: _stats(5, bench_frames),  # below floor
        }
        sides = {1: 0, 2: 0, 3: 1, 4: 0, 5: 1}
        result = tracker._top_tracks_by_frames(
            [1, 2, 3, 4, 5], ts, 4,
            track_court_sides=sides,
        )
        # Swap suppressed → strict top-4 retained.
        assert sorted(result) == [1, 2, 3, 4]

    def test_no_swap_when_top_n_not_four(self) -> None:
        tracker = MatchPlayerTracker()
        ts = {
            1: _stats(1, 100),
            2: _stats(2, 90),
            3: _stats(3, 80),
            4: _stats(4, 70),
            5: _stats(5, 10),
        }
        sides = {1: 0, 2: 0, 3: 0, 4: 0, 5: 1}
        # Asking for top-3 of 5, all near in top-3. F2's cardinality
        # rebalance is a 2v2 (n=4) heuristic; smaller n is a passthrough.
        result = tracker._top_tracks_by_frames(
            [1, 2, 3, 4, 5], ts, 3,
            track_court_sides=sides,
        )
        assert sorted(result) == [1, 2, 3]


class TestConstants:
    def test_constants_set_to_documented_defaults(self) -> None:
        assert TOP_TRACKS_MIN_COURT_INTERIOR_FRACTION == pytest.approx(0.30)
        assert TOP_TRACKS_MIN_FRAMES_FLOOR == 6
