"""Tests for multi-region appearance descriptors."""

from __future__ import annotations

import numpy as np
import pytest

from rallycut.tracking.appearance_descriptor import (
    AppearanceDescriptorStore,
    MultiRegionDescriptor,
    compute_multi_region_distance,
    compute_track_mean_descriptor,
    extract_multi_region_descriptor,
)


def _make_histogram(peak: int = 0) -> np.ndarray:
    """Create a simple 16x8 HS histogram with a peak."""
    hist = np.zeros((16, 8), dtype=np.float32)
    hist[peak % 16, 0] = 1.0
    return hist


def _make_descriptor(
    head_peak: int = 0,
    upper_peak: int = 4,
    shorts_peak: int = 8,
) -> MultiRegionDescriptor:
    return MultiRegionDescriptor(
        head=_make_histogram(head_peak),
        upper=_make_histogram(upper_peak),
        shorts=_make_histogram(shorts_peak),
    )


class TestMultiRegionDescriptor:
    def test_empty_descriptor(self) -> None:
        desc = MultiRegionDescriptor()
        assert desc.head is None
        assert desc.upper is None
        assert desc.shorts is None

    def test_descriptor_with_regions(self) -> None:
        desc = _make_descriptor()
        assert desc.head is not None
        assert desc.upper is not None
        assert desc.shorts is not None


class TestAppearanceDescriptorStore:
    def test_add_and_get(self) -> None:
        store = AppearanceDescriptorStore()
        desc = _make_descriptor()
        store.add(1, 10, desc)

        result = store.get(1, 10)
        assert result is not None
        assert result is desc

    def test_get_missing(self) -> None:
        store = AppearanceDescriptorStore()
        assert store.get(1, 10) is None

    def test_get_track_descriptors(self) -> None:
        store = AppearanceDescriptorStore()
        for f in range(0, 30, 3):
            store.add(1, f, _make_descriptor())
        store.add(2, 5, _make_descriptor(head_peak=5))

        descs = store.get_track_descriptors(1)
        assert len(descs) == 10
        # Sorted by frame number
        frames = [f for f, _ in descs]
        assert frames == sorted(frames)

    def test_has_data(self) -> None:
        store = AppearanceDescriptorStore()
        assert not store.has_data()
        store.add(1, 0, _make_descriptor())
        assert store.has_data()

    def test_rekey(self) -> None:
        store = AppearanceDescriptorStore()
        store.add(1, 5, _make_descriptor(head_peak=1))
        store.add(1, 10, _make_descriptor(head_peak=2))
        store.add(1, 15, _make_descriptor(head_peak=3))

        store.rekey(1, 99, from_frame=10)

        # Frame 5 stays with track 1
        assert store.get(1, 5) is not None
        # Frames 10+ moved to track 99
        assert store.get(99, 10) is not None
        assert store.get(99, 15) is not None
        assert store.get(1, 10) is None


class TestComputeMultiRegionDistance:
    def test_identical_descriptors(self) -> None:
        desc = _make_descriptor()
        dist = compute_multi_region_distance(desc, desc)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_different_descriptors(self) -> None:
        desc_a = _make_descriptor(head_peak=0, upper_peak=0, shorts_peak=0)
        desc_b = _make_descriptor(head_peak=8, upper_peak=8, shorts_peak=8)
        dist = compute_multi_region_distance(desc_a, desc_b)
        assert dist > 0.5  # Very different

    def test_partial_descriptors(self) -> None:
        """Distance should work with partially available regions."""
        desc_a = MultiRegionDescriptor(
            head=None,
            upper=_make_histogram(4),
            shorts=_make_histogram(8),
        )
        desc_b = MultiRegionDescriptor(
            head=None,
            upper=_make_histogram(4),
            shorts=_make_histogram(8),
        )
        dist = compute_multi_region_distance(desc_a, desc_b)
        assert dist == pytest.approx(0.0, abs=1e-6)

    def test_no_common_regions(self) -> None:
        desc_a = MultiRegionDescriptor(head=_make_histogram())
        desc_b = MultiRegionDescriptor(shorts=_make_histogram())
        dist = compute_multi_region_distance(desc_a, desc_b)
        assert dist == 1.0  # No comparable regions


class TestComputeTrackMeanDescriptor:
    def test_single_sample(self) -> None:
        store = AppearanceDescriptorStore()
        desc = _make_descriptor(head_peak=3)
        store.add(1, 0, desc)

        mean = compute_track_mean_descriptor(store, 1)
        assert mean.head is not None
        assert mean.upper is not None
        assert mean.shorts is not None

    def test_empty_track(self) -> None:
        store = AppearanceDescriptorStore()
        mean = compute_track_mean_descriptor(store, 1)
        assert mean.head is None
        assert mean.upper is None
        assert mean.shorts is None


class TestExtractMultiRegionDescriptor:
    def test_valid_bbox(self) -> None:
        """Should extract all 3 regions from a valid bbox."""
        frame = np.random.randint(0, 255, (200, 100, 3), dtype=np.uint8)
        desc = extract_multi_region_descriptor(
            frame, (0.5, 0.5, 1.0, 1.0), 100, 200
        )
        assert desc is not None
        # At least shorts should be available (largest region)
        assert desc.shorts is not None

    def test_tiny_bbox_returns_none(self) -> None:
        """Very small bbox should return None."""
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        desc = extract_multi_region_descriptor(
            frame, (0.5, 0.5, 0.01, 0.01), 100, 100
        )
        assert desc is None
