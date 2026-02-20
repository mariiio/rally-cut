"""Multi-region appearance descriptors for player identity resolution.

Extends the existing single-region shorts histogram to 3 body regions:
- Head/hair (top 15% of bbox): most discriminative — hair color/style varies
- Upper body (15-45% of bbox): captures jersey/top differences
- Shorts (60-100% of bbox): existing proven signal

Also provides calibration-based height estimation.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from rallycut.court.calibration import CourtCalibrator

logger = logging.getLogger(__name__)

# Histogram parameters (same as color_repair.py for compatibility)
HS_BINS = (16, 8)  # 128 values per region
HS_RANGES = [0, 180, 0, 256]
MIN_CROP_PIXELS = 100  # Per-region minimum

# Region weights for composite distance
WEIGHT_HEAD = 0.25
WEIGHT_UPPER = 0.30
WEIGHT_SHORTS = 0.45


@dataclass(eq=False)
class MultiRegionDescriptor:
    """Appearance descriptor with head, upper body, and shorts histograms."""

    head: np.ndarray | None = None  # 16x8 HS histogram
    upper: np.ndarray | None = None
    shorts: np.ndarray | None = None


@dataclass
class HeightEstimate:
    """Player height estimated from calibration."""

    height_meters: float = 0.0
    num_samples: int = 0
    confidence: float = 0.0


class AppearanceDescriptorStore:
    """Stores per-frame multi-region descriptors keyed by (track_id, frame)."""

    def __init__(self) -> None:
        self._descriptors: dict[tuple[int, int], MultiRegionDescriptor] = {}
        self._height_samples: dict[int, list[float]] = defaultdict(list)

    def add(
        self,
        track_id: int,
        frame_number: int,
        descriptor: MultiRegionDescriptor,
    ) -> None:
        """Add a descriptor for a track at a frame."""
        self._descriptors[(track_id, frame_number)] = descriptor

    def get(
        self,
        track_id: int,
        frame_number: int,
    ) -> MultiRegionDescriptor | None:
        """Get descriptor for a specific track and frame."""
        return self._descriptors.get((track_id, frame_number))

    def get_track_descriptors(
        self,
        track_id: int,
    ) -> list[tuple[int, MultiRegionDescriptor]]:
        """Get all descriptors for a track, sorted by frame."""
        result = [
            (fn, desc)
            for (tid, fn), desc in self._descriptors.items()
            if tid == track_id
        ]
        return sorted(result, key=lambda x: x[0])

    def has_data(self) -> bool:
        """Check if any descriptors are stored."""
        return len(self._descriptors) > 0

    def add_height_sample(self, track_id: int, height_m: float) -> None:
        """Add a height measurement for a track."""
        if 1.4 < height_m < 2.3:  # Sanity bounds
            self._height_samples[track_id].append(height_m)

    def get_height_estimate(self, track_id: int) -> HeightEstimate:
        """Get trimmed mean height estimate for a track."""
        samples = self._height_samples.get(track_id, [])
        if len(samples) < 5:
            return HeightEstimate()

        # Trimmed mean: remove top/bottom 10%
        arr = np.array(sorted(samples))
        trim = max(1, len(arr) // 10)
        trimmed = arr[trim:-trim] if trim < len(arr) // 2 else arr

        return HeightEstimate(
            height_meters=float(np.mean(trimmed)),
            num_samples=len(samples),
            confidence=min(1.0, len(samples) / 50.0),
        )

    def rekey(self, old_id: int, new_id: int, from_frame: int) -> None:
        """Rekey descriptors after track ID change (split/merge)."""
        keys_to_move = [
            (tid, fn)
            for (tid, fn) in self._descriptors
            if tid == old_id and fn >= from_frame
        ]
        for key in keys_to_move:
            desc = self._descriptors.pop(key)
            self._descriptors[(new_id, key[1])] = desc

        # Note: old_id may still have earlier descriptors under its original key


def extract_multi_region_descriptor(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
) -> MultiRegionDescriptor | None:
    """Extract multi-region HS histograms from a player bbox.

    Args:
        frame: BGR image (original, NOT ROI-masked).
        bbox: Normalized (cx, cy, w, h) bounding box.
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.

    Returns:
        MultiRegionDescriptor, or None if bbox too small.
    """
    cx, cy, w, h = bbox

    # Convert to pixel coordinates
    x1 = max(0, int((cx - w / 2) * frame_w))
    y1 = max(0, int((cy - h / 2) * frame_h))
    x2 = min(frame_w, int((cx + w / 2) * frame_w))
    y2 = min(frame_h, int((cy + h / 2) * frame_h))

    bbox_h = y2 - y1
    bbox_w = x2 - x1
    if bbox_h < 20 or bbox_w < 10:
        return None

    desc = MultiRegionDescriptor()

    # Head region: top 15%
    head_y2 = y1 + int(bbox_h * 0.15)
    desc.head = _extract_region_hist(frame, x1, y1, x2, head_y2)

    # Upper body: 15-45%
    upper_y1 = y1 + int(bbox_h * 0.15)
    upper_y2 = y1 + int(bbox_h * 0.45)
    desc.upper = _extract_region_hist(frame, x1, upper_y1, x2, upper_y2)

    # Shorts: 60-100%
    shorts_y1 = y1 + int(bbox_h * 0.60)
    desc.shorts = _extract_region_hist(frame, x1, shorts_y1, x2, y2)

    # Return None if no valid regions
    if desc.head is None and desc.upper is None and desc.shorts is None:
        return None

    return desc


def _extract_region_hist(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> np.ndarray | None:
    """Extract HS histogram from a pixel region."""
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0 or (crop.shape[0] * crop.shape[1]) < MIN_CROP_PIXELS:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, list(HS_BINS), HS_RANGES)
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist


def compute_multi_region_distance(
    desc_a: MultiRegionDescriptor,
    desc_b: MultiRegionDescriptor,
) -> float:
    """Compute weighted Bhattacharyya distance between two descriptors.

    Returns distance in [0, 1]. Lower = more similar.
    """
    total_weight = 0.0
    total_dist = 0.0

    if desc_a.head is not None and desc_b.head is not None:
        total_dist += WEIGHT_HEAD * cv2.compareHist(
            desc_a.head, desc_b.head, cv2.HISTCMP_BHATTACHARYYA
        )
        total_weight += WEIGHT_HEAD

    if desc_a.upper is not None and desc_b.upper is not None:
        total_dist += WEIGHT_UPPER * cv2.compareHist(
            desc_a.upper, desc_b.upper, cv2.HISTCMP_BHATTACHARYYA
        )
        total_weight += WEIGHT_UPPER

    if desc_a.shorts is not None and desc_b.shorts is not None:
        total_dist += WEIGHT_SHORTS * cv2.compareHist(
            desc_a.shorts, desc_b.shorts, cv2.HISTCMP_BHATTACHARYYA
        )
        total_weight += WEIGHT_SHORTS

    if total_weight == 0.0:
        return 1.0

    return total_dist / total_weight


def compute_track_mean_descriptor(
    store: AppearanceDescriptorStore,
    track_id: int,
    max_samples: int = 30,
) -> MultiRegionDescriptor:
    """Compute mean descriptor for a track from stored samples."""
    descriptors = store.get_track_descriptors(track_id)
    if not descriptors:
        return MultiRegionDescriptor()

    # Use most recent samples
    recent = descriptors[-max_samples:]

    heads = [d.head for _, d in recent if d.head is not None]
    uppers = [d.upper for _, d in recent if d.upper is not None]
    shorts = [d.shorts for _, d in recent if d.shorts is not None]

    mean_desc = MultiRegionDescriptor()
    if heads:
        mean_desc.head = _mean_histogram(heads)
    if uppers:
        mean_desc.upper = _mean_histogram(uppers)
    if shorts:
        mean_desc.shorts = _mean_histogram(shorts)

    return mean_desc


def estimate_player_height_meters(
    foot_point: tuple[float, float],
    head_point: tuple[float, float],
    calibrator: CourtCalibrator,
    video_width: int,
    video_height: int,
) -> float | None:
    """Estimate player height in meters using court calibration.

    Projects foot and head points through homography and computes
    the court-plane distance.

    Args:
        foot_point: Foot position in normalized image coordinates.
        head_point: Head position in normalized image coordinates.
        calibrator: Calibrated court calibrator.
        video_width: Video width in pixels.
        video_height: Video height in pixels.

    Returns:
        Estimated height in meters, or None if projection fails.
    """
    try:
        foot_court = calibrator.image_to_court(
            foot_point, video_width, video_height
        )
        head_court = calibrator.image_to_court(
            head_point, video_width, video_height
        )
    except (RuntimeError, ValueError) as e:
        logger.debug("Height projection failed: %s", e)
        return None

    # Euclidean distance in court space
    dx = foot_court[0] - head_court[0]
    dy = foot_court[1] - head_court[1]
    height = float((dx ** 2 + dy ** 2) ** 0.5)

    # Sanity check
    if 1.4 < height < 2.3:
        return height
    return None


def _mean_histogram(histograms: list[np.ndarray]) -> np.ndarray:
    """Compute mean of a list of histograms."""
    stacked = np.stack(histograms)
    mean: np.ndarray = np.mean(stacked, axis=0)
    # Re-normalize
    total = mean.sum()
    if total > 0:
        mean /= total
    return mean
