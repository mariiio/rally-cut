"""Tests for color-based track repair (splitting + convergence detection)."""

from __future__ import annotations

import numpy as np

from rallycut.tracking.color_repair import (
    ColorHistogramStore,
    detect_convergence_periods,
    extract_shorts_histogram,
    split_tracks_by_color,
)
from rallycut.tracking.player_tracker import PlayerPosition

# --- Helpers ---


def _make_positions(
    track_id: int,
    frames: list[int],
    x: float = 0.5,
    y: float = 0.5,
    width: float = 0.05,
    height: float = 0.15,
) -> list[PlayerPosition]:
    """Create player positions for a single track."""
    return [
        PlayerPosition(
            frame_number=f,
            track_id=track_id,
            x=x,
            y=y,
            width=width,
            height=height,
            confidence=0.9,
        )
        for f in frames
    ]


def _make_uniform_histogram(h_peak: int, s_peak: int) -> np.ndarray:
    """Create a synthetic HS histogram with a single dominant bin.

    Args:
        h_peak: Hue bin index (0-15).
        s_peak: Saturation bin index (0-7).
    """
    hist = np.zeros((16, 8), dtype=np.float32)
    hist[h_peak, s_peak] = 0.8
    # Small noise in neighbors
    for dh in [-1, 0, 1]:
        for ds in [-1, 0, 1]:
            if dh == 0 and ds == 0:
                continue
            nh = (h_peak + dh) % 16
            ns = max(0, min(7, s_peak + ds))
            hist[nh, ns] = 0.025
    # Normalize
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def _make_store_from_tracks(
    track_histograms: dict[int, list[tuple[int, np.ndarray]]],
) -> ColorHistogramStore:
    """Build a store from {track_id: [(frame, histogram), ...]}."""
    store = ColorHistogramStore()
    for track_id, items in track_histograms.items():
        for frame_num, hist in items:
            store.add(track_id, frame_num, hist)
    return store


# --- TestColorHistogramStore ---


class TestColorHistogramStore:
    def test_add_and_get(self) -> None:
        store = ColorHistogramStore()
        hist = _make_uniform_histogram(5, 3)
        store.add(1, 10, hist)

        result = store.get(1, 10)
        assert result is not None
        np.testing.assert_array_equal(result, hist)

    def test_get_missing(self) -> None:
        store = ColorHistogramStore()
        assert store.get(1, 10) is None

    def test_get_track_histograms_sorted(self) -> None:
        store = ColorHistogramStore()
        hist_a = _make_uniform_histogram(5, 3)
        hist_b = _make_uniform_histogram(8, 2)

        store.add(1, 20, hist_a)
        store.add(1, 10, hist_b)

        items = store.get_track_histograms(1)
        assert len(items) == 2
        assert items[0][0] == 10  # Earlier frame first
        assert items[1][0] == 20

    def test_track_ids(self) -> None:
        store = ColorHistogramStore()
        store.add(1, 10, _make_uniform_histogram(5, 3))
        store.add(2, 10, _make_uniform_histogram(8, 2))

        assert store.track_ids() == {1, 2}

    def test_has_data(self) -> None:
        store = ColorHistogramStore()
        assert not store.has_data()
        store.add(1, 10, _make_uniform_histogram(5, 3))
        assert store.has_data()

    def test_rekey(self) -> None:
        store = ColorHistogramStore()
        hist_a = _make_uniform_histogram(5, 3)
        hist_b = _make_uniform_histogram(8, 2)
        hist_c = _make_uniform_histogram(10, 4)

        store.add(1, 10, hist_a)
        store.add(1, 20, hist_b)
        store.add(1, 30, hist_c)

        store.rekey(1, 99, 20)

        # Frame 10 stays with track 1
        assert store.get(1, 10) is not None
        # Frames 20, 30 moved to track 99
        assert store.get(1, 20) is None
        assert store.get(99, 20) is not None
        assert store.get(99, 30) is not None

        assert 99 in store.track_ids()
        assert 1 in store.track_ids()  # Still has frame 10

    def test_rekey_removes_old_id_when_empty(self) -> None:
        store = ColorHistogramStore()
        store.add(1, 10, _make_uniform_histogram(5, 3))

        store.rekey(1, 99, 10)

        assert 1 not in store.track_ids()
        assert 99 in store.track_ids()


# --- TestExtractShortsHistogram ---


class TestExtractShortsHistogram:
    def test_normal_bbox(self) -> None:
        """Normal-sized bbox returns a normalized histogram."""
        # Create a synthetic BGR frame
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        hist = extract_shorts_histogram(
            frame,
            bbox=(0.5, 0.5, 0.2, 0.4),  # cx, cy, w, h normalized
            frame_w=640,
            frame_h=480,
        )
        assert hist is not None
        assert hist.shape == (16, 8)
        # Should be normalized to sum ~1
        assert abs(hist.sum() - 1.0) < 1e-5

    def test_tiny_bbox_returns_none(self) -> None:
        """Too-small bbox returns None (< MIN_CROP_PIXELS)."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        hist = extract_shorts_histogram(
            frame,
            bbox=(0.5, 0.5, 0.01, 0.02),  # Very small
            frame_w=640,
            frame_h=480,
        )
        assert hist is None

    def test_edge_bbox(self) -> None:
        """Bbox at frame edge is clamped and still works."""
        frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        hist = extract_shorts_histogram(
            frame,
            bbox=(0.95, 0.95, 0.2, 0.4),  # Extends past bottom-right
            frame_w=640,
            frame_h=480,
        )
        # May or may not be None depending on clamped crop size
        if hist is not None:
            assert hist.shape == (16, 8)

    def test_brightness_invariant(self) -> None:
        """Different brightness levels produce similar histograms for same hue."""
        import cv2 as cv

        # Create HSV frames with same H/S but different V (brightness)
        # H=90 (green), S=150 (medium saturation), V varies
        hsv_bright = np.full((200, 200, 3), [90, 150, 250], dtype=np.uint8)
        hsv_dark = np.full((200, 200, 3), [90, 150, 100], dtype=np.uint8)

        # Convert to BGR (the input format for extract_shorts_histogram)
        frame_bright = cv.cvtColor(hsv_bright, cv.COLOR_HSV2BGR)
        frame_dark = cv.cvtColor(hsv_dark, cv.COLOR_HSV2BGR)

        bbox = (0.5, 0.5, 0.8, 0.8)
        hist_bright = extract_shorts_histogram(frame_bright, bbox, 200, 200)
        hist_dark = extract_shorts_histogram(frame_dark, bbox, 200, 200)

        assert hist_bright is not None
        assert hist_dark is not None

        # HS histogram should be very similar (brightness doesn't affect H/S)
        dist = cv.compareHist(
            hist_bright.astype(np.float32),
            hist_dark.astype(np.float32),
            cv.HISTCMP_BHATTACHARYYA,
        )
        # Low distance = similar
        assert dist < 0.3


# --- TestSplitTracksByColor ---


class TestSplitTracksByColor:
    def test_uniform_track_no_split(self) -> None:
        """Track with consistent color should not be split."""
        hist = _make_uniform_histogram(5, 3)
        positions = _make_positions(1, list(range(0, 100, 3)))

        store = _make_store_from_tracks({
            1: [(f, hist.copy()) for f in range(0, 100, 3)],
        })

        result_pos, num_splits = split_tracks_by_color(positions, store)
        assert num_splits == 0
        # All positions still have track_id 1
        assert all(p.track_id == 1 for p in result_pos)

    def test_abrupt_color_change_splits(self) -> None:
        """Track with abrupt color change should be split."""
        hist_a = _make_uniform_histogram(2, 3)  # Blue-ish
        hist_b = _make_uniform_histogram(10, 5)  # Red-ish

        frames = list(range(0, 90, 3))
        split_at = 45  # Color changes at frame 45
        positions = _make_positions(1, frames)

        track_hists: list[tuple[int, np.ndarray]] = []
        for f in frames:
            if f < split_at:
                track_hists.append((f, hist_a.copy()))
            else:
                track_hists.append((f, hist_b.copy()))

        store = _make_store_from_tracks({1: track_hists})

        result_pos, num_splits = split_tracks_by_color(positions, store)
        assert num_splits == 1

        # Positions before split keep track 1
        before = [p for p in result_pos if p.frame_number < split_at]
        assert all(p.track_id == 1 for p in before)

        # Positions from split point get new track ID
        after = [p for p in result_pos if p.frame_number >= split_at]
        assert len(after) > 0
        assert all(p.track_id != 1 for p in after)
        # All after positions have same new ID
        new_ids = {p.track_id for p in after}
        assert len(new_ids) == 1

    def test_single_frame_blip_no_split(self) -> None:
        """Single frame of different color should not cause a split."""
        hist_a = _make_uniform_histogram(5, 3)
        hist_b = _make_uniform_histogram(12, 6)

        frames = list(range(0, 60, 3))
        positions = _make_positions(1, frames)

        track_hists: list[tuple[int, np.ndarray]] = []
        for f in frames:
            if f == 30:  # Single frame blip
                track_hists.append((f, hist_b.copy()))
            else:
                track_hists.append((f, hist_a.copy()))

        store = _make_store_from_tracks({1: track_hists})

        result_pos, num_splits = split_tracks_by_color(
            positions, store, min_consecutive=2
        )
        assert num_splits == 0

    def test_short_track_skipped(self) -> None:
        """Track with fewer frames than min_template_frames is skipped."""
        positions = _make_positions(1, [0, 3])
        hist = _make_uniform_histogram(5, 3)
        store = _make_store_from_tracks({
            1: [(0, hist), (3, hist)],
        })

        result_pos, num_splits = split_tracks_by_color(
            positions, store, min_template_frames=5
        )
        assert num_splits == 0

    def test_empty_positions(self) -> None:
        """Empty input returns empty output."""
        store = ColorHistogramStore()
        result_pos, num_splits = split_tracks_by_color([], store)
        assert num_splits == 0
        assert len(result_pos) == 0


# --- TestDetectConvergencePeriods ---


class TestDetectConvergencePeriods:
    def test_overlapping_tracks_detected(self) -> None:
        """Two tracks with sustained overlap should produce a convergence period."""
        positions: list[PlayerPosition] = []
        # Track 1 at (0.5, 0.5), Track 2 starts at (0.3, 0.5) then moves to overlap
        for f in range(0, 30):
            positions.append(PlayerPosition(
                frame_number=f, track_id=1,
                x=0.5, y=0.5, width=0.1, height=0.2, confidence=0.9,
            ))
            if f < 10:
                # Separated
                positions.append(PlayerPosition(
                    frame_number=f, track_id=2,
                    x=0.3, y=0.5, width=0.1, height=0.2, confidence=0.9,
                ))
            else:
                # Overlapping with track 1
                positions.append(PlayerPosition(
                    frame_number=f, track_id=2,
                    x=0.5, y=0.5, width=0.1, height=0.2, confidence=0.9,
                ))

        periods = detect_convergence_periods(positions, iou_threshold=0.3, min_duration=5)
        assert len(periods) >= 1
        assert periods[0].track_a == 1 or periods[0].track_b == 1
        assert periods[0].track_a == 2 or periods[0].track_b == 2

    def test_separated_tracks_no_convergence(self) -> None:
        """Tracks that never overlap produce no convergence periods."""
        positions: list[PlayerPosition] = []
        for f in range(0, 30):
            positions.append(PlayerPosition(
                frame_number=f, track_id=1,
                x=0.2, y=0.5, width=0.1, height=0.2, confidence=0.9,
            ))
            positions.append(PlayerPosition(
                frame_number=f, track_id=2,
                x=0.8, y=0.5, width=0.1, height=0.2, confidence=0.9,
            ))

        periods = detect_convergence_periods(positions)
        assert len(periods) == 0

    def test_short_overlap_ignored(self) -> None:
        """Overlap shorter than min_duration is not reported."""
        positions: list[PlayerPosition] = []
        for f in range(0, 30):
            positions.append(PlayerPosition(
                frame_number=f, track_id=1,
                x=0.5, y=0.5, width=0.1, height=0.2, confidence=0.9,
            ))
            if 10 <= f < 13:  # Only 3 frames overlap (< min_duration=5)
                positions.append(PlayerPosition(
                    frame_number=f, track_id=2,
                    x=0.5, y=0.5, width=0.1, height=0.2, confidence=0.9,
                ))
            else:
                positions.append(PlayerPosition(
                    frame_number=f, track_id=2,
                    x=0.8, y=0.5, width=0.1, height=0.2, confidence=0.9,
                ))

        periods = detect_convergence_periods(positions, min_duration=5)
        assert len(periods) == 0


