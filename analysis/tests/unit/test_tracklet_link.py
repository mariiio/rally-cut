"""Tests for appearance-based tracklet linking."""

from __future__ import annotations

import numpy as np

from rallycut.court.calibration import CourtCalibrator
from rallycut.tracking.color_repair import ColorHistogramStore
from rallycut.tracking.player_tracker import PlayerPosition
from rallycut.tracking.tracklet_link import (
    DEFAULT_MAX_MERGE_VELOCITY,
    DEFAULT_MAX_MERGE_VELOCITY_METERS,
    DEFAULT_MERGE_VELOCITY_WINDOW,
    _would_create_velocity_anomaly,
    link_tracklets_by_appearance,
    relink_primary_fragments,
)


def _make_positions(
    track_id: int,
    frames: range,
    x: float = 0.5,
    y: float = 0.5,
) -> list[PlayerPosition]:
    """Create positions for a track at fixed location."""
    return [
        PlayerPosition(
            frame_number=f,
            track_id=track_id,
            x=x,
            y=y,
            width=0.05,
            height=0.15,
            confidence=0.9,
        )
        for f in frames
    ]


def _make_histogram(hue_peak: int = 90, sat_peak: int = 4) -> np.ndarray:
    """Create a simple 16x8 HS histogram with a dominant bin."""
    hist = np.zeros((16, 8), dtype=np.float32)
    hist[hue_peak % 16, sat_peak % 8] = 1.0
    return hist


class TestLinkTrackletsByAppearance:
    def test_empty_positions(self) -> None:
        store = ColorHistogramStore()
        result, merges = link_tracklets_by_appearance([], store)
        assert result == []
        assert merges == 0

    def test_single_track_no_merge(self) -> None:
        positions = _make_positions(1, range(0, 50))
        store = ColorHistogramStore()
        hist = _make_histogram(hue_peak=5)
        for f in range(0, 50, 3):
            store.add(1, f, hist)

        result, merges = link_tracklets_by_appearance(positions, store)
        assert merges == 0
        assert len(result) == 50

    def test_merge_similar_fragments(self) -> None:
        """Fragments with identical appearance should be merged when above target count."""
        # 5 tracks total (above target of 4): tracks 1-4 are full players,
        # track 5 is a fragment of track 1 that should merge back
        positions = (
            _make_positions(1, range(0, 50), x=0.3, y=0.4)
            + _make_positions(2, range(0, 100), x=0.7, y=0.4)
            + _make_positions(3, range(0, 100), x=0.3, y=0.7)
            + _make_positions(4, range(0, 100), x=0.7, y=0.7)
            + _make_positions(5, range(60, 110), x=0.32, y=0.42)  # Fragment of track 1
        )

        store = ColorHistogramStore()
        hist_1 = _make_histogram(hue_peak=2)
        hist_2 = _make_histogram(hue_peak=5)
        hist_3 = _make_histogram(hue_peak=8)
        hist_4 = _make_histogram(hue_peak=12)

        for f in range(0, 100, 3):
            store.add(1, f, hist_1) if f < 50 else None
            store.add(2, f, hist_2)
            store.add(3, f, hist_3)
            store.add(4, f, hist_4)
        # Track 5 has same appearance as track 1
        for f in range(60, 110, 3):
            store.add(5, f, hist_1)

        result, merges = link_tracklets_by_appearance(positions, store)
        assert merges == 1
        # Track 5 should merge into track 1 (or vice versa)
        track_ids = {p.track_id for p in result}
        assert len(track_ids) == 4

    def test_no_merge_different_appearance(self) -> None:
        """Two fragments with different appearance should not merge."""
        positions = (
            _make_positions(1, range(0, 50), x=0.3, y=0.4)
            + _make_positions(2, range(60, 110), x=0.35, y=0.42)
        )

        store = ColorHistogramStore()
        hist_a = _make_histogram(hue_peak=2)  # Red-ish
        hist_b = _make_histogram(hue_peak=10)  # Green-ish
        for f in range(0, 50, 3):
            store.add(1, f, hist_a)
        for f in range(60, 110, 3):
            store.add(2, f, hist_b)

        result, merges = link_tracklets_by_appearance(positions, store)
        assert merges == 0
        track_ids = {p.track_id for p in result}
        assert len(track_ids) == 2

    def test_no_merge_temporal_overlap(self) -> None:
        """Tracks with overlapping frames must never merge."""
        positions = (
            _make_positions(1, range(0, 50), x=0.3, y=0.4)
            + _make_positions(2, range(40, 90), x=0.35, y=0.42)
        )

        store = ColorHistogramStore()
        hist = _make_histogram(hue_peak=5)  # Same appearance
        for f in range(0, 50, 3):
            store.add(1, f, hist)
        for f in range(40, 90, 3):
            store.add(2, f, hist)

        result, merges = link_tracklets_by_appearance(positions, store)
        assert merges == 0

    def test_no_merge_large_spatial_gap(self) -> None:
        """Fragments too far apart spatially should not merge."""
        positions = (
            _make_positions(1, range(0, 50), x=0.1, y=0.2)
            + _make_positions(2, range(60, 110), x=0.9, y=0.8)
        )

        store = ColorHistogramStore()
        hist = _make_histogram(hue_peak=5)
        for f in range(0, 50, 3):
            store.add(1, f, hist)
        for f in range(60, 110, 3):
            store.add(2, f, hist)

        result, merges = link_tracklets_by_appearance(positions, store)
        assert merges == 0

    def test_target_track_count_stops_merging(self) -> None:
        """Merging stops when target track count is reached."""
        # 3 fragments, all similar appearance
        positions = (
            _make_positions(1, range(0, 30), x=0.3, y=0.4)
            + _make_positions(2, range(40, 70), x=0.32, y=0.42)
            + _make_positions(3, range(80, 110), x=0.34, y=0.44)
        )

        store = ColorHistogramStore()
        hist = _make_histogram(hue_peak=5)
        for tid in [1, 2, 3]:
            base = (tid - 1) * 40
            for f in range(base, base + 30, 3):
                store.add(tid, f, hist)

        # Target 2 tracks: should merge 2 of the 3
        result, merges = link_tracklets_by_appearance(
            positions, store, target_track_count=2
        )
        assert merges == 1
        track_ids = {p.track_id for p in result}
        assert len(track_ids) == 2

    def test_short_tracks_excluded(self) -> None:
        """Tracks shorter than min_track_frames are not linked."""
        positions = (
            _make_positions(1, range(0, 50), x=0.3, y=0.4)
            + _make_positions(2, range(60, 63), x=0.32, y=0.42)  # Only 3 frames
        )

        store = ColorHistogramStore()
        hist = _make_histogram(hue_peak=5)
        for f in range(0, 50, 3):
            store.add(1, f, hist)
        store.add(2, 60, hist)

        result, merges = link_tracklets_by_appearance(
            positions, store, min_track_frames=5
        )
        assert merges == 0

    def test_already_at_target_count(self) -> None:
        """No merging when already at target track count."""
        positions = (
            _make_positions(1, range(0, 50), x=0.3, y=0.4)
            + _make_positions(2, range(0, 50), x=0.7, y=0.6)
        )

        store = ColorHistogramStore()
        hist_a = _make_histogram(hue_peak=5)
        hist_b = _make_histogram(hue_peak=10)
        for f in range(0, 50, 3):
            store.add(1, f, hist_a)
            store.add(2, f, hist_b)

        result, merges = link_tracklets_by_appearance(
            positions, store, target_track_count=4
        )
        assert merges == 0


class TestRelinkPrimaryFragments:
    def test_merges_non_primary_into_primary(self) -> None:
        """Non-primary fragment near a primary track gets merged into it."""
        # Track 1 is primary (f0-50), track 2 is non-primary (f70-120)
        positions = (
            _make_positions(1, range(0, 50), x=0.3, y=0.4)
            + _make_positions(2, range(70, 120), x=0.34, y=0.44)
        )
        store = ColorHistogramStore()
        hist = _make_histogram(hue_peak=5)
        for f in range(0, 50, 3):
            store.add(1, f, hist)
        for f in range(70, 120, 3):
            store.add(2, f, hist)

        result, updated_ids, num = relink_primary_fragments(
            positions, {1}, store,         )
        assert num == 1
        # Track 2 merged into track 1
        assert all(p.track_id == 1 for p in result)
        assert updated_ids == [1]

    def test_no_non_primary_fragments(self) -> None:
        """When all tracks are primary, nothing to link."""
        positions = (
            _make_positions(1, range(0, 50), x=0.3, y=0.4)
            + _make_positions(2, range(0, 50), x=0.7, y=0.7)
        )
        store = ColorHistogramStore()
        hist = _make_histogram(hue_peak=5)
        for f in range(0, 50, 3):
            store.add(1, f, hist)
            store.add(2, f, hist)

        result, updated_ids, num = relink_primary_fragments(
            positions, {1, 2}, store,
        )
        assert num == 0

    def test_appearance_tiebreaker(self) -> None:
        """Non-primary fragment links to the primary with closest appearance."""
        # Two primaries (1, 2) and one non-primary (3)
        positions = (
            _make_positions(1, range(0, 50), x=0.3, y=0.4)
            + _make_positions(2, range(0, 50), x=0.7, y=0.7)
            + _make_positions(3, range(70, 120), x=0.34, y=0.44)  # near both
        )
        store = ColorHistogramStore()
        hist_a = _make_histogram(hue_peak=2)
        hist_b = _make_histogram(hue_peak=10)
        for f in range(0, 50, 3):
            store.add(1, f, hist_a)
            store.add(2, f, hist_b)
        for f in range(70, 120, 3):
            store.add(3, f, hist_a)  # same appearance as track 1

        result, updated_ids, num = relink_primary_fragments(
            positions, {1, 2}, store,         )
        assert num == 1
        # Track 3 should merge into track 1 (matching appearance)
        assert all(p.track_id != 3 for p in result)
        merged_into_1 = [p for p in result if p.track_id == 1]
        assert len(merged_into_1) == 100  # 50 original + 50 from track 3

    def test_gap_too_large(self) -> None:
        """Non-primary fragment too far temporally is not merged."""
        positions = (
            _make_positions(1, range(0, 50), x=0.3, y=0.4)
            + _make_positions(2, range(110, 160), x=0.34, y=0.44)  # gap=60
        )
        store = ColorHistogramStore()
        hist = _make_histogram(hue_peak=5)
        for f in range(0, 50, 3):
            store.add(1, f, hist)
        for f in range(110, 160, 3):
            store.add(2, f, hist)

        result, updated_ids, num = relink_primary_fragments(
            positions, {1}, store,
        )
        assert num == 0

    def test_distance_too_large(self) -> None:
        """Non-primary fragment too far spatially is not merged."""
        positions = (
            _make_positions(1, range(0, 50), x=0.1, y=0.2)
            + _make_positions(2, range(60, 100), x=0.9, y=0.8)  # dist~1.0
        )
        store = ColorHistogramStore()
        hist = _make_histogram(hue_peak=5)
        for f in range(0, 50, 3):
            store.add(1, f, hist)
        for f in range(60, 100, 3):
            store.add(2, f, hist)

        result, updated_ids, num = relink_primary_fragments(
            positions, {1}, store,
        )
        assert num == 0

    def test_overlapping_fragments(self) -> None:
        """Non-primary fragment overlapping primary must not merge."""
        positions = (
            _make_positions(1, range(0, 60), x=0.3, y=0.4)
            + _make_positions(2, range(50, 100), x=0.34, y=0.44)
        )
        store = ColorHistogramStore()
        hist = _make_histogram(hue_peak=5)
        for f in range(0, 60, 3):
            store.add(1, f, hist)
        for f in range(50, 100, 3):
            store.add(2, f, hist)

        result, updated_ids, num = relink_primary_fragments(
            positions, {1}, store,
        )
        assert num == 0

    def test_primary_set_unchanged(self) -> None:
        """Primary set stays the same — non-primary merges INTO primaries."""
        positions = (
            _make_positions(1, range(0, 50), x=0.3, y=0.4)
            + _make_positions(2, range(70, 120), x=0.34, y=0.44)
            + _make_positions(3, range(0, 120), x=0.7, y=0.7)
        )
        store = ColorHistogramStore()
        hist = _make_histogram(hue_peak=5)
        for f in range(0, 50, 3):
            store.add(1, f, hist)
        for f in range(70, 120, 3):
            store.add(2, f, hist)
        for f in range(0, 120, 3):
            store.add(3, f, hist)

        # Tracks 1 and 3 are primary; track 2 is non-primary
        result, updated_ids, num = relink_primary_fragments(
            positions, {1, 3}, store,         )
        assert num == 1
        assert set(updated_ids) == {1, 3}  # unchanged
        # Track 2 merged into track 1 (closer spatially)
        assert all(p.track_id in {1, 3} for p in result)

    def test_backward_linking_blocked(self) -> None:
        """Non-primary fragment that precedes a primary is NOT merged.

        A fragment predating a primary is likely a different player who
        was there first, not a continuation of the primary.
        """
        # Track 2 (non-primary) comes BEFORE track 1 (primary)
        positions = (
            _make_positions(1, range(70, 120), x=0.3, y=0.4)
            + _make_positions(2, range(0, 50), x=0.34, y=0.44)
        )
        store = ColorHistogramStore()
        hist = _make_histogram(hue_peak=5)
        for f in range(70, 120, 3):
            store.add(1, f, hist)
        for f in range(0, 50, 3):
            store.add(2, f, hist)

        result, updated_ids, num = relink_primary_fragments(
            positions, {1}, store,         )
        assert num == 0  # backward link blocked

    def test_appearance_gate_rejects_dissimilar(self) -> None:
        """Non-primary fragment with dissimilar appearance is rejected."""
        positions = (
            _make_positions(1, range(0, 50), x=0.3, y=0.4)
            + _make_positions(2, range(70, 120), x=0.34, y=0.44)
        )
        store = ColorHistogramStore()
        hist_a = _make_histogram(hue_peak=2)
        hist_b = _make_histogram(hue_peak=10)  # very different
        for f in range(0, 50, 3):
            store.add(1, f, hist_a)
        for f in range(70, 120, 3):
            store.add(2, f, hist_b)

        result, updated_ids, num = relink_primary_fragments(
            positions, {1}, store, max_appearance=0.20,
        )
        assert num == 0  # rejected by appearance gate

    def test_forward_merge_allowed(self) -> None:
        """Fragment after a primary's last frame gets merged (forward)."""
        # Primary T4 drops out at f100. Fragment T5 starts at f110.
        positions = (
            _make_positions(1, range(0, 150), x=0.2, y=0.3)
            + _make_positions(2, range(0, 150), x=0.4, y=0.3)
            + _make_positions(3, range(0, 150), x=0.6, y=0.7)
            + _make_positions(4, range(0, 100), x=0.8, y=0.7)
            + _make_positions(5, range(110, 150), x=0.82, y=0.72)
        )
        store = ColorHistogramStore()
        hist = _make_histogram(hue_peak=5)
        for tid in [1, 2, 3, 4]:
            for f in range(0, 150, 3):
                if f < 100 or tid != 4:
                    store.add(tid, f, hist)
        for f in range(110, 150, 3):
            store.add(5, f, hist)

        result, updated_ids, num = relink_primary_fragments(
            positions, {1, 2, 3, 4}, store,
        )
        assert num == 1  # fragment 5 continues T4


class TestOptimalPartition:
    """Tests for branch-and-bound optimal K-partition."""

    def test_optimal_beats_greedy(self) -> None:
        """Optimal partition picks globally better assignment than greedy.

        4 fragments in 2 time slots (T1/T2 concurrent, T3/T4 concurrent).
        Gaussian histograms with sigma=2: T1(c=8), T2(c=7), T3(c=9), T4(c=7.5).
        Greedy picks T1-T4 (0.09) then T2-T3 (0.34), total=0.43.
        Optimal picks T1-T3 (0.18) + T2-T4 (0.09), total=0.27.
        """

        def _gauss_hist(center: float, sigma: float = 2.0) -> np.ndarray:
            hist = np.zeros((16, 8), dtype=np.float32)
            for i in range(16):
                hist[i, 4] = np.exp(-0.5 * ((i - center) / sigma) ** 2)
            hist /= hist.sum()
            return hist

        # T1/T2 concurrent (f0-50), T3/T4 concurrent (f60-110)
        positions = (
            _make_positions(1, range(0, 50), x=0.3, y=0.3)
            + _make_positions(2, range(0, 50), x=0.7, y=0.7)
            + _make_positions(3, range(60, 110), x=0.32, y=0.32)
            + _make_positions(4, range(60, 110), x=0.72, y=0.72)
        )

        store = ColorHistogramStore()
        hist_t1 = _gauss_hist(8.0)   # Player A, time 1
        hist_t2 = _gauss_hist(7.0)   # Player B, time 1
        hist_t3 = _gauss_hist(9.0)   # Player A, time 2
        hist_t4 = _gauss_hist(7.5)   # Player B, time 2 (close to T1!)

        for f in range(0, 50, 3):
            store.add(1, f, hist_t1)
            store.add(2, f, hist_t2)
        for f in range(60, 110, 3):
            store.add(3, f, hist_t3)
            store.add(4, f, hist_t4)

        result, merges = link_tracklets_by_appearance(
            positions, store, target_track_count=2,
        )
        track_ids = {p.track_id for p in result if p.track_id >= 0}
        assert len(track_ids) == 2
        assert merges == 2

        # Verify optimal grouping: T1+T3 and T2+T4 (not T1+T4 and T2+T3)
        t1_id = next(p.track_id for p in result if p.frame_number == 25 and p.x < 0.5)
        t3_id = next(p.track_id for p in result if p.frame_number == 85 and p.x < 0.5)
        assert t1_id == t3_id, "T1 and T3 should be grouped (same player A)"

    def test_dissimilar_fragment_stays_separate(self) -> None:
        """Fragments too different from all groups stay unlinked."""
        # 4 full-length tracks + 1 short fragment with very different appearance
        positions = (
            _make_positions(1, range(0, 100), x=0.2, y=0.3)
            + _make_positions(2, range(0, 100), x=0.4, y=0.3)
            + _make_positions(3, range(0, 100), x=0.6, y=0.7)
            + _make_positions(4, range(0, 100), x=0.8, y=0.7)
            + _make_positions(5, range(110, 140), x=0.5, y=0.5)
        )

        store = ColorHistogramStore()
        for tid, peak in [(1, 2), (2, 5), (3, 8), (4, 12)]:
            hist = _make_histogram(hue_peak=peak)
            for f in range(0, 100, 3):
                store.add(tid, f, hist)
        # T5 has completely different appearance
        hist_outlier = _make_histogram(hue_peak=15, sat_peak=7)
        for f in range(110, 140, 3):
            store.add(5, f, hist_outlier)

        result, merges = link_tracklets_by_appearance(positions, store)
        # T5 should NOT be merged into any group (distance too high)
        track_ids = {p.track_id for p in result if p.track_id >= 0}
        assert 5 in track_ids or merges == 0  # T5 stays separate

    def test_swap_does_not_break_existing(self) -> None:
        """Swap optimization preserves correct greedy merges."""
        # 5 tracks, target=4. Track 5 is a fragment of track 1.
        # Greedy should merge T5→T1. Swap should not undo it.
        positions = (
            _make_positions(1, range(0, 50), x=0.3, y=0.4)
            + _make_positions(2, range(0, 100), x=0.7, y=0.4)
            + _make_positions(3, range(0, 100), x=0.3, y=0.7)
            + _make_positions(4, range(0, 100), x=0.7, y=0.7)
            + _make_positions(5, range(60, 110), x=0.32, y=0.42)
        )

        store = ColorHistogramStore()
        hist_1 = _make_histogram(hue_peak=2)
        hist_2 = _make_histogram(hue_peak=5)
        hist_3 = _make_histogram(hue_peak=8)
        hist_4 = _make_histogram(hue_peak=12)

        for f in range(0, 100, 3):
            store.add(1, f, hist_1) if f < 50 else None
            store.add(2, f, hist_2)
            store.add(3, f, hist_3)
            store.add(4, f, hist_4)
        for f in range(60, 110, 3):
            store.add(5, f, hist_1)

        result, merges = link_tracklets_by_appearance(positions, store)
        assert merges == 1
        track_ids = {p.track_id for p in result if p.track_id >= 0}
        assert len(track_ids) == 4


def _perspective_calibrator() -> CourtCalibrator:
    """Build a realistic perspective calibration.

    Trapezoid-to-rectangle homography: image top is farther (compressed,
    20m per image-x unit); image bottom is closer (expanded, 10m per
    image-x unit). This is the scale-dependence the gate is designed for.
    """
    cal = CourtCalibrator()
    cal.calibrate(
        image_corners=[
            (0.3, 0.2),  # top-left (far court)
            (0.7, 0.2),  # top-right
            (0.9, 0.8),  # bottom-right (near camera)
            (0.1, 0.8),  # bottom-left
        ],
    )
    return cal


def _velocity_positions(
    tid_a: int, tid_b: int,
    a_xy: tuple[float, float], b_xy: tuple[float, float],
    a_frames: range = range(0, 10), b_frames: range = range(20, 30),
) -> list[PlayerPosition]:
    """Two sequential tracks, each with fixed location."""
    return (
        _make_positions(tid_a, a_frames, x=a_xy[0], y=a_xy[1])
        + _make_positions(tid_b, b_frames, x=b_xy[0], y=b_xy[1])
    )


class TestVelocityGateCourtPlane:
    def test_bit_exact_when_uncalibrated(self) -> None:
        """calibrator=None preserves current image-plane behaviour."""
        # Image-plane displacement 0.30 > DEFAULT_MAX_MERGE_VELOCITY 0.20 → reject.
        positions = _velocity_positions(
            tid_a=1, tid_b=2,
            a_xy=(0.5, 0.5), b_xy=(0.5, 0.8),
        )
        assert _would_create_velocity_anomaly(positions, 1, 2) is True
        assert _would_create_velocity_anomaly(
            positions, 1, 2, calibrator=None,
        ) is True

        # Image-plane displacement 0.10 < threshold → allow.
        positions_ok = _velocity_positions(
            tid_a=1, tid_b=2,
            a_xy=(0.5, 0.5), b_xy=(0.5, 0.6),
        )
        assert _would_create_velocity_anomaly(positions_ok, 1, 2) is False

    def test_far_court_allows_modest_image_jump(self) -> None:
        """Near the baseline a 0.15 image jump is ~3m → test a smaller allowable jump.

        Use 0.08 image-x at far edge (y=0.2): 20 m/unit × 0.08 = 1.6m court.
        Image-plane gate at 0.20 allows it already. Court-plane gate at 2.5m
        allows it. The assertion is: calibrated path still allows.
        """
        cal = _perspective_calibrator()
        positions = _velocity_positions(
            tid_a=1, tid_b=2,
            a_xy=(0.42, 0.2), b_xy=(0.50, 0.2),  # top (far) edge, Δx=0.08
        )
        # Sanity: confirm court displacement is ≈ 1.6 m.
        ax, ay = cal.image_to_court((0.42, 0.2), 0, 0)
        bx, by = cal.image_to_court((0.50, 0.2), 0, 0)
        court_dist = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
        assert 1.3 < court_dist < 1.9, f"expected ~1.6m, got {court_dist:.2f}m"

        # Court-plane at 2.5m: allowed.
        assert _would_create_velocity_anomaly(
            positions, 1, 2, calibrator=cal,
        ) is False

    def test_far_court_rejects_large_real_jump(self) -> None:
        """Far-court Δx=0.15 image → ~3m court → reject at 2.5m gate.

        Same displacement that would regress the image-plane gate at 0.10
        (falsely reject) IS a real 3m jump in court space — the court-plane
        gate correctly catches it.
        """
        cal = _perspective_calibrator()
        positions = _velocity_positions(
            tid_a=1, tid_b=2,
            a_xy=(0.4, 0.2), b_xy=(0.55, 0.2),  # top edge, Δx=0.15
        )
        ax, ay = cal.image_to_court((0.4, 0.2), 0, 0)
        bx, by = cal.image_to_court((0.55, 0.2), 0, 0)
        court_dist = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
        assert 2.7 < court_dist < 3.3, f"expected ~3m, got {court_dist:.2f}m"

        assert _would_create_velocity_anomaly(
            positions, 1, 2, calibrator=cal,
        ) is True

    def test_near_camera_allows_same_image_jump(self) -> None:
        """Near-camera Δx=0.15 image → ~1.5m court → allow at 2.5m gate.

        This is the scale-dependence fix: same image displacement at the
        near edge is a much smaller real-world jump. Court-plane gate
        allows it where a tight image-plane gate (0.10) would falsely reject.
        """
        cal = _perspective_calibrator()
        positions = _velocity_positions(
            tid_a=1, tid_b=2,
            a_xy=(0.4, 0.8), b_xy=(0.55, 0.8),  # bottom edge, Δx=0.15
        )
        ax, ay = cal.image_to_court((0.4, 0.8), 0, 0)
        bx, by = cal.image_to_court((0.55, 0.8), 0, 0)
        court_dist = ((bx - ax) ** 2 + (by - ay) ** 2) ** 0.5
        assert 1.3 < court_dist < 1.8, f"expected ~1.5m, got {court_dist:.2f}m"

        # Image-plane gate at 0.10 would reject this (0.15 > 0.10) — but the
        # court-plane gate correctly allows (1.5m < 2.5m).
        assert _would_create_velocity_anomaly(
            positions, 1, 2, calibrator=cal,
        ) is False
        # Sanity: confirm a tighter image-plane gate *would* falsely reject.
        assert _would_create_velocity_anomaly(
            positions, 1, 2, max_displacement_image=0.10,
        ) is True

    def test_overlap_path_uses_court_plane(self) -> None:
        """Overlapping tracks branch also uses court-plane when calibrated."""
        cal = _perspective_calibrator()
        # Overlapping frames 5-15, track a at (0.4, 0.2), track b at (0.55, 0.2)
        # Far-edge Δx=0.15 → ~3m → reject.
        positions = (
            _make_positions(1, range(0, 15), x=0.4, y=0.2)
            + _make_positions(2, range(5, 20), x=0.55, y=0.2)
        )
        assert _would_create_velocity_anomaly(
            positions, 1, 2, calibrator=cal,
        ) is True

        # Same overlap but near-camera edge → ~1.5m → allow.
        positions_near = (
            _make_positions(1, range(0, 15), x=0.4, y=0.8)
            + _make_positions(2, range(5, 20), x=0.55, y=0.8)
        )
        assert _would_create_velocity_anomaly(
            positions_near, 1, 2, calibrator=cal,
        ) is False

    def test_uncalibrated_calibrator_falls_back_to_image_plane(self) -> None:
        """A calibrator that hasn't had .calibrate() called → treat as None."""
        cal = CourtCalibrator()
        assert not cal.is_calibrated
        positions = _velocity_positions(
            tid_a=1, tid_b=2,
            a_xy=(0.5, 0.5), b_xy=(0.5, 0.8),  # 0.30 image jump
        )
        # Should still reject via image-plane fallback, not crash.
        assert _would_create_velocity_anomaly(
            positions, 1, 2, calibrator=cal,
        ) is True

    def test_defaults_match_memory_spec(self) -> None:
        """Guard: documented defaults don't drift silently."""
        assert DEFAULT_MAX_MERGE_VELOCITY == 0.20
        assert DEFAULT_MAX_MERGE_VELOCITY_METERS == 2.5
        assert DEFAULT_MERGE_VELOCITY_WINDOW == 10

    def test_out_of_trapezoid_falls_back_to_image_plane(self) -> None:
        """Foot projection outside court+margin → image-plane threshold applies.

        Short-height detections (partial players, jumpers) have feet above
        the trapezoid and their homography projection is unreliable. The
        gate must fall back to image-plane so we don't reject legitimate
        merges in the extrapolation regime (the 740ffd88 failure mode).
        """
        cal = CourtCalibrator()
        cal.calibrate(image_corners=[
            (0.325, 0.60), (0.665, 0.60), (1.135, 0.75), (-0.125, 0.75),
        ])
        # Detections way above the trapezoid (foot_y = 0.3 < 0.6) — out of
        # calibrated region. Image displacement is small (0.05) — allow.
        positions = (
            [PlayerPosition(f, 1, 0.45, 0.30, 0.05, 0.0, 0.9) for f in range(0, 10)]
            + [PlayerPosition(f, 2, 0.50, 0.30, 0.05, 0.0, 0.9) for f in range(20, 30)]
        )
        assert _would_create_velocity_anomaly(
            positions, 1, 2, calibrator=cal,
        ) is False

        # Image displacement 0.30 (> 0.20 fallback) — reject via image-plane.
        positions_big = (
            [PlayerPosition(f, 1, 0.20, 0.30, 0.05, 0.0, 0.9) for f in range(0, 10)]
            + [PlayerPosition(f, 2, 0.60, 0.30, 0.05, 0.0, 0.9) for f in range(20, 30)]
        )
        assert _would_create_velocity_anomaly(
            positions_big, 1, 2, calibrator=cal,
        ) is True

    def test_foot_y_used_not_center_y(self) -> None:
        """Court projection must use bbox foot (y + height/2), not center.

        Real beach calibrations cover only the court floor (a narrow
        image-y band). Standing players' *centers* project above the
        trapezoid's far edge, into the homography's extrapolation regime
        where a tiny image displacement produces an arbitrarily large
        projected distance. Using the foot keeps projections inside the
        calibrated region.

        Reproduces the 9dbe457a regression (-16.08pp HOTA at every
        court-plane threshold) by using a tight trapezoid that mirrors
        real calibration: top edge at y=0.60, bottom edge at y=0.75.
        """
        cal = CourtCalibrator()
        cal.calibrate(image_corners=[
            (0.325, 0.60),  # top-left (far court)
            (0.665, 0.60),  # top-right
            (1.135, 0.75),  # bottom-right (near camera)
            (-0.125, 0.75),  # bottom-left
        ])

        # Standing players: centers slightly above trapezoid (y ≈ 0.50,
        # 0.52) — close enough to the vanishing-point regime that tiny
        # image-y differences project to wildly different court-y. Real
        # feet at y = 0.50 + 0.20 = 0.70 and 0.52 + 0.20 = 0.72 both land
        # INSIDE the trapezoid where the homography is reliable.
        pa_center = PlayerPosition(0, 1, 0.45, 0.50, 0.05, 0.0, 0.9)
        pb_center = PlayerPosition(0, 2, 0.50, 0.52, 0.05, 0.0, 0.9)
        pa_real = PlayerPosition(0, 1, 0.45, 0.50, 0.05, 0.40, 0.9)
        pb_real = PlayerPosition(0, 2, 0.50, 0.52, 0.05, 0.40, 0.9)

        from rallycut.tracking.tracklet_link import _court_displacement_meters
        center_y_dist = _court_displacement_meters(cal, pa_center, pb_center)
        foot_y_dist = _court_displacement_meters(cal, pa_real, pb_real)

        # Center-y extrapolation produces a wild distance (tens of metres
        # for a 0.05 image jump); foot-y stays on-court-sized (< 2m).
        assert center_y_dist > 10.0, (
            f"center-y should extrapolate wildly above the trapezoid, "
            f"got {center_y_dist:.2f}m"
        )
        assert foot_y_dist < 2.0, f"foot-y should stay sane, got {foot_y_dist:.2f}m"

        # And the gate uses foot-y: this legitimate slow-moving pair passes.
        positions = (
            [PlayerPosition(f, 1, 0.45, 0.50, 0.05, 0.40, 0.9) for f in range(0, 10)]
            + [PlayerPosition(f, 2, 0.50, 0.52, 0.05, 0.40, 0.9) for f in range(20, 30)]
        )
        assert _would_create_velocity_anomaly(
            positions, 1, 2, calibrator=cal,
        ) is False


class TestLinkTrackletsCalibratorThreading:
    """Calibrator flows through to the gate via link_tracklets_by_appearance."""

    def test_calibrator_accepted_as_kwarg(self) -> None:
        """Smoke: public API accepts calibrator parameter without error."""
        cal = _perspective_calibrator()
        store = ColorHistogramStore()
        result, merges = link_tracklets_by_appearance(
            [], store, calibrator=cal,
        )
        assert result == []
        assert merges == 0

    def test_calibrator_blocks_far_court_false_merge(self) -> None:
        """Two fragments with far-court large real jump should NOT merge.

        Same-appearance fragments at two different players' far-court
        positions. Without calibrator, image-plane 0.20 gate would allow
        (Δx=0.15 < 0.20). With calibrator, court-plane 2.5m gate rejects
        (court dist ~3m > 2.5m).
        """
        # 5 tracks: 4 full players (target_count=4) + 1 fragment that is
        # far-court displaced from one of them.
        positions = (
            _make_positions(1, range(0, 50), x=0.4, y=0.2)      # far-court-left
            + _make_positions(2, range(0, 100), x=0.7, y=0.4)   # mid-right
            + _make_positions(3, range(0, 100), x=0.3, y=0.7)   # near-left
            + _make_positions(4, range(0, 100), x=0.7, y=0.7)   # near-right
            + _make_positions(5, range(60, 110), x=0.55, y=0.2)  # far-court-right
        )

        store = ColorHistogramStore()
        hist_1 = _make_histogram(hue_peak=2)
        hist_2 = _make_histogram(hue_peak=5)
        hist_3 = _make_histogram(hue_peak=8)
        hist_4 = _make_histogram(hue_peak=12)

        for f in range(0, 100, 3):
            if f < 50:
                store.add(1, f, hist_1)
            store.add(2, f, hist_2)
            store.add(3, f, hist_3)
            store.add(4, f, hist_4)
        # Track 5 matches track 1 appearance — would merge on appearance alone.
        for f in range(60, 110, 3):
            store.add(5, f, hist_1)

        cal = _perspective_calibrator()
        # With calibrator: court-plane 3m > 2.5m → blocked.
        _, merges_cal = link_tracklets_by_appearance(
            list(positions), store, calibrator=cal,
        )
        # Without calibrator: image-plane 0.15 < 0.20 → allowed (control).
        _, merges_no_cal = link_tracklets_by_appearance(
            list(positions), store, calibrator=None,
        )
        assert merges_no_cal == 1
        assert merges_cal == 0


# ---------------------------------------------------------------------------
# Session 6 — learned-head merge veto tests
# ---------------------------------------------------------------------------


def _seed_learned(
    store,
    track_id: int,
    frames,
    vector,
) -> None:
    """Populate learned_store with a given (L2-normed) vector across frames."""
    v = np.asarray(vector, dtype=np.float32)
    v = v / (np.linalg.norm(v) + 1e-8)
    for f in frames:
        store.add(track_id, f, v)


class TestLearnedMergeVeto:
    def _setup_mergeable_pair(self):
        """Two same-HSV fragments that would merge by default (target 1 track)."""
        from rallycut.tracking.color_repair import LearnedEmbeddingStore
        positions = (
            _make_positions(1, range(0, 50), x=0.3, y=0.4)
            + _make_positions(2, range(60, 110), x=0.32, y=0.42)
        )
        color_store = ColorHistogramStore()
        hist = _make_histogram(hue_peak=5)
        for f in range(0, 50, 3):
            color_store.add(1, f, hist)
        for f in range(60, 110, 3):
            color_store.add(2, f, hist)
        learned = LearnedEmbeddingStore()
        return positions, color_store, learned

    def test_merge_vetoed_on_different_learned_embeddings(self) -> None:
        """Orthogonal learned embeddings at threshold 0.5 trigger veto."""
        from rallycut.tracking import tracklet_link as tl
        from rallycut.tracking.merge_veto import learned_cosine_veto
        positions, color_store, learned = self._setup_mergeable_pair()
        rng = np.random.default_rng(1)
        vec_a = rng.standard_normal(128)
        vec_b = rng.standard_normal(128)  # Orthogonal-ish — cos ≈ 0
        _seed_learned(learned, 1, range(0, 50), vec_a)
        _seed_learned(learned, 2, range(60, 110), vec_b)

        # Build track summary from positions BEFORE any mutation.
        tracks = tl._compute_track_summary(positions)
        assert learned_cosine_veto(learned, 1, tracks[1]["frames"], 2, tracks[2]["frames"], threshold=0.5) is True
        # Threshold 0 (default env) → disabled, never vetoes.
        assert learned_cosine_veto(learned, 1, tracks[1]["frames"], 2, tracks[2]["frames"], threshold=0.0) is False

    def test_merge_allowed_on_matching_learned_embeddings(self) -> None:
        """Same-player fragments (matching learned embeddings) should merge."""
        from rallycut.tracking import tracklet_link as tl
        from rallycut.tracking.merge_veto import learned_cosine_veto
        positions, color_store, learned = self._setup_mergeable_pair()
        rng = np.random.default_rng(2)
        vec = rng.standard_normal(128)
        _seed_learned(learned, 1, range(0, 50), vec)
        _seed_learned(learned, 2, range(60, 110), vec)  # same direction

        tracks = tl._compute_track_summary(positions)
        # Very high threshold (0.99): matching vectors (cos≈1) should still pass.
        assert learned_cosine_veto(learned, 1, tracks[1]["frames"], 2, tracks[2]["frames"], threshold=0.99) is False

    def test_veto_abstains_on_insufficient_embeddings(self) -> None:
        """Fewer than MIN_FRAMES embeddings on either side → abstain (no veto)."""
        from rallycut.tracking import tracklet_link as tl
        from rallycut.tracking.merge_veto import learned_cosine_veto
        positions, color_store, learned = self._setup_mergeable_pair()
        rng = np.random.default_rng(3)
        # Only 2 embeddings on track 1 (below MIN_FRAMES=5).
        _seed_learned(learned, 1, [0, 5], rng.standard_normal(128))
        _seed_learned(learned, 2, range(60, 110), rng.standard_normal(128))

        tracks = tl._compute_track_summary(positions)
        assert learned_cosine_veto(learned, 1, tracks[1]["frames"], 2, tracks[2]["frames"], threshold=0.5) is False

    def test_veto_abstains_when_store_missing_or_empty(self) -> None:
        """No learned_store or empty store → never veto."""
        from rallycut.tracking import tracklet_link as tl
        from rallycut.tracking.color_repair import LearnedEmbeddingStore
        from rallycut.tracking.merge_veto import learned_cosine_veto
        positions, _color_store, _ = self._setup_mergeable_pair()
        tracks = tl._compute_track_summary(positions)
        assert learned_cosine_veto(None, 1, tracks[1]["frames"], 2, tracks[2]["frames"], threshold=0.5) is False
        empty = LearnedEmbeddingStore()
        assert learned_cosine_veto(empty, 1, tracks[1]["frames"], 2, tracks[2]["frames"], threshold=0.5) is False

    def test_veto_end_to_end_blocks_merge(self) -> None:
        """Full link_tracklets_by_appearance call with distinct embeddings
        and a positive threshold should reject the merge.
        """
        import importlib

        import rallycut.tracking.merge_veto as mv
        import rallycut.tracking.tracklet_link as tl
        positions, color_store, learned = self._setup_mergeable_pair()
        rng = np.random.default_rng(4)
        # Orthogonal → cos ≈ 0
        _seed_learned(learned, 1, range(0, 50), rng.standard_normal(128))
        _seed_learned(learned, 2, range(60, 110), rng.standard_normal(128))

        import os
        os.environ["LEARNED_MERGE_VETO_COS"] = "0.5"
        try:
            # Reload merge_veto first so its module-level constant re-reads the env var,
            # then reload tracklet_link so its re-imported constant is up to date.
            mv = importlib.reload(mv)
            tl = importlib.reload(tl)
            from rallycut.tracking.tracklet_link import (
                link_tracklets_by_appearance as link_reloaded,
            )
            _, merges = link_reloaded(
                positions,
                color_store,
                target_track_count=1,
                learned_store=learned,
            )
            assert merges == 0, "veto should have blocked the merge"
        finally:
            os.environ.pop("LEARNED_MERGE_VETO_COS", None)
            importlib.reload(mv)  # restore default (threshold=0)
            importlib.reload(tl)
