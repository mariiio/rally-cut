"""Tests for appearance-based tracklet linking."""

from __future__ import annotations

import numpy as np

from rallycut.tracking.color_repair import ColorHistogramStore
from rallycut.tracking.player_tracker import PlayerPosition
from rallycut.tracking.tracklet_link import (
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
