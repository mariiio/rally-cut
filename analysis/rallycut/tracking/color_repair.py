"""Color-based post-hoc track repair for player tracking.

Detects abrupt color changes within a track (indicates BoT-SORT incorrectly
merged two players into one track ID) and splits tracks at those points.

Uses Hue-Saturation histograms from the shorts/swimsuit region (lower 40%
of bbox) for brightness-invariant appearance comparison.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

from rallycut.tracking.player_tracker import PlayerPosition

logger = logging.getLogger(__name__)

# Histogram parameters
HS_BINS = (16, 8)  # Hue (16 bins) x Saturation (8 bins) = 128 values
HS_RANGES = [0, 180, 0, 256]  # OpenCV HSV: H=[0,180), S=[0,256)
MIN_CROP_PIXELS = 200  # Minimum pixels for reliable histogram

# Color splitting defaults
DEFAULT_BHATTACHARYYA_THRESHOLD = 0.55
DEFAULT_MIN_CONSECUTIVE = 2
DEFAULT_EMA_ALPHA = 0.05
DEFAULT_MIN_TEMPLATE_FRAMES = 5
DEFAULT_MAX_PASSES = 2

# Convergence detection defaults
DEFAULT_IOU_THRESHOLD = 0.3
DEFAULT_MIN_OVERLAP_FRAMES = 5


@dataclass
class ConvergencePeriod:
    """A period where two tracks have significant bbox overlap."""

    track_a: int
    track_b: int
    start_frame: int
    end_frame: int


class ColorHistogramStore:
    """Stores per-frame HS histograms keyed by (track_id, frame_number)."""

    def __init__(self) -> None:
        self._histograms: dict[tuple[int, int], np.ndarray] = {}
        self._track_ids: set[int] = set()

    def add(self, track_id: int, frame_number: int, histogram: np.ndarray) -> None:
        """Store a histogram for a track at a specific frame."""
        self._histograms[(track_id, frame_number)] = histogram
        self._track_ids.add(track_id)

    def get(self, track_id: int, frame_number: int) -> np.ndarray | None:
        """Retrieve a histogram, or None if not stored."""
        return self._histograms.get((track_id, frame_number))

    def get_track_histograms(
        self, track_id: int
    ) -> list[tuple[int, np.ndarray]]:
        """Get all histograms for a track, sorted by frame number."""
        items = [
            (fn, hist)
            for (tid, fn), hist in self._histograms.items()
            if tid == track_id
        ]
        items.sort(key=lambda x: x[0])
        return items

    def has_data(self) -> bool:
        """Check if any histograms are stored."""
        return len(self._histograms) > 0

    def track_ids(self) -> set[int]:
        """Get all track IDs with stored histograms."""
        return set(self._track_ids)

    def rekey(self, old_id: int, new_id: int, from_frame: int) -> None:
        """Reassign histograms from old_id to new_id starting at from_frame.

        Used after track splitting to keep histogram store in sync with
        position track IDs.
        """
        keys_to_move = [
            (tid, fn)
            for (tid, fn) in self._histograms
            if tid == old_id and fn >= from_frame
        ]
        for key in keys_to_move:
            hist = self._histograms.pop(key)
            self._histograms[(new_id, key[1])] = hist

        if keys_to_move:
            self._track_ids.add(new_id)
            # Check if old_id still has any histograms
            if not any(tid == old_id for tid, _ in self._histograms):
                self._track_ids.discard(old_id)


def extract_shorts_histogram(
    frame: np.ndarray,
    bbox: tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
) -> np.ndarray | None:
    """Extract an HS histogram from the shorts/swimsuit region of a bbox.

    Uses the lower 40% of the bounding box (shorts/swimsuit area) which is
    the most discriminative region for beach volleyball players.

    Args:
        frame: BGR image (original, NOT ROI-masked).
        bbox: Normalized (cx, cy, w, h) bounding box.
        frame_w: Frame width in pixels.
        frame_h: Frame height in pixels.

    Returns:
        Normalized HS histogram (128 bins, sum=1), or None if crop too small.
    """
    cx, cy, w, h = bbox

    # Convert to pixel coordinates
    x1 = int((cx - w / 2) * frame_w)
    y1 = int((cy - h / 2) * frame_h)
    x2 = int((cx + w / 2) * frame_w)
    y2 = int((cy + h / 2) * frame_h)

    # Clamp to frame bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(frame_w, x2)
    y2 = min(frame_h, y2)

    bbox_h = y2 - y1
    bbox_w = x2 - x1
    if bbox_h <= 0 or bbox_w <= 0:
        return None

    # Lower 40% of bbox (shorts region)
    shorts_y1 = y1 + int(bbox_h * 0.6)
    shorts_y2 = y2

    crop = frame[shorts_y1:shorts_y2, x1:x2]
    if crop.size == 0 or (crop.shape[0] * crop.shape[1]) < MIN_CROP_PIXELS:
        return None

    # Convert to HSV and compute 2D Hue-Saturation histogram
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist(
        [hsv], [0, 1], None, list(HS_BINS), HS_RANGES
    )

    # Normalize to sum=1
    cv2.normalize(hist, hist, alpha=1.0, norm_type=cv2.NORM_L1)
    return hist


def split_tracks_by_color(
    positions: list[PlayerPosition],
    color_store: ColorHistogramStore,
    threshold: float = DEFAULT_BHATTACHARYYA_THRESHOLD,
    min_consecutive: int = DEFAULT_MIN_CONSECUTIVE,
    ema_alpha: float = DEFAULT_EMA_ALPHA,
    min_template_frames: int = DEFAULT_MIN_TEMPLATE_FRAMES,
    max_passes: int = DEFAULT_MAX_PASSES,
) -> tuple[list[PlayerPosition], int]:
    """Split tracks at points where shorts color changes abruptly.

    Uses an EMA running template to detect when a track's appearance shifts
    beyond a threshold, indicating an ID assignment error (two different
    players sharing the same track ID).

    Args:
        positions: All player positions (modified in place).
        color_store: Histogram store with per-frame histograms.
        threshold: Bhattacharyya distance threshold for split (0-1).
        min_consecutive: Minimum consecutive frames above threshold.
        ema_alpha: EMA smoothing factor for running template.
        min_template_frames: Minimum histograms needed to process a track.
        max_passes: Maximum splitting passes per track.

    Returns:
        Tuple of (positions, number of splits).
    """
    if not positions:
        return positions, 0

    # Find next available track ID
    max_id = max((p.track_id for p in positions if p.track_id >= 0), default=0)
    next_id = max_id + 1
    total_splits = 0

    # Process each track
    processed_tracks: set[int] = set()

    for pass_num in range(max_passes):
        current_track_ids = color_store.track_ids() - processed_tracks
        if not current_track_ids:
            break

        splits_this_pass = 0
        for track_id in sorted(current_track_ids):
            histograms = color_store.get_track_histograms(track_id)
            if len(histograms) < min_template_frames:
                processed_tracks.add(track_id)
                continue

            # Build EMA template and detect deviation
            split_frame = _find_color_split_point(
                histograms, threshold, min_consecutive, ema_alpha
            )

            if split_frame is not None:
                new_id = next_id
                next_id += 1

                # Reassign positions from split point onward
                for p in positions:
                    if p.track_id == track_id and p.frame_number >= split_frame:
                        p.track_id = new_id

                # Rekey histogram store
                color_store.rekey(track_id, new_id, split_frame)

                logger.info(
                    f"Color split: track {track_id} at frame {split_frame} "
                    f"-> new track {new_id}"
                )

                total_splits += 1
                splits_this_pass += 1
                # Mark original as processed but new track may need checking
                processed_tracks.add(track_id)
            else:
                processed_tracks.add(track_id)

        if splits_this_pass == 0:
            break

    if total_splits:
        logger.info(f"Color-based splitting: {total_splits} tracks split")

    return positions, total_splits


def _find_color_split_point(
    histograms: list[tuple[int, np.ndarray]],
    threshold: float,
    min_consecutive: int,
    ema_alpha: float,
) -> int | None:
    """Find the frame where a track's color shifts beyond threshold.

    Uses EMA running template starting from the first histogram.

    Returns:
        Frame number at which to split, or None if no split needed.
    """
    if len(histograms) < 2:
        return None

    template = histograms[0][1].copy()
    consecutive_deviations = 0
    first_deviation_frame: int | None = None

    for i in range(1, len(histograms)):
        frame_num, hist = histograms[i]

        # Compare current frame to running template
        dist = cv2.compareHist(
            template.astype(np.float32),
            hist.astype(np.float32),
            cv2.HISTCMP_BHATTACHARYYA,
        )

        if dist > threshold:
            consecutive_deviations += 1
            if consecutive_deviations == 1:
                first_deviation_frame = frame_num
            if consecutive_deviations >= min_consecutive:
                return first_deviation_frame
        else:
            consecutive_deviations = 0
            first_deviation_frame = None
            # Update template with EMA (only when below threshold)
            template = (1 - ema_alpha) * template + ema_alpha * hist

    return None


def detect_convergence_periods(
    positions: list[PlayerPosition],
    iou_threshold: float = DEFAULT_IOU_THRESHOLD,
    min_duration: int = DEFAULT_MIN_OVERLAP_FRAMES,
) -> list[ConvergencePeriod]:
    """Detect periods where two tracks have sustained bbox overlap.

    These are the moments where BoT-SORT is most likely to swap track IDs
    (players crossing at the net).

    Args:
        positions: All player positions.
        iou_threshold: Minimum IoU to count as overlapping.
        min_duration: Minimum consecutive frames of overlap.

    Returns:
        List of convergence periods found.
    """
    if not positions:
        return []

    # Group by frame
    by_frame: dict[int, list[PlayerPosition]] = defaultdict(list)
    for p in positions:
        if p.track_id >= 0:
            by_frame[p.frame_number].append(p)

    # Track consecutive overlap per pair
    pair_streak: dict[tuple[int, int], int] = defaultdict(int)
    pair_start: dict[tuple[int, int], int] = {}
    periods: list[ConvergencePeriod] = []

    for frame_num in sorted(by_frame.keys()):
        frame_pos = by_frame[frame_num]
        overlapping_pairs: set[tuple[int, int]] = set()

        # Check all pairs in this frame
        for i in range(len(frame_pos)):
            for j in range(i + 1, len(frame_pos)):
                a, b = frame_pos[i], frame_pos[j]
                iou = _compute_iou(a, b)
                if iou > iou_threshold:
                    pair_key = (min(a.track_id, b.track_id), max(a.track_id, b.track_id))
                    overlapping_pairs.add(pair_key)

        # Update streaks
        active_pairs = set(pair_streak.keys())
        for pair in active_pairs | overlapping_pairs:
            if pair in overlapping_pairs:
                if pair not in pair_streak or pair_streak[pair] == 0:
                    pair_start[pair] = frame_num
                pair_streak[pair] += 1
            else:
                # Pair no longer overlapping — check if streak was long enough
                if pair_streak.get(pair, 0) >= min_duration:
                    periods.append(ConvergencePeriod(
                        track_a=pair[0],
                        track_b=pair[1],
                        start_frame=pair_start[pair],
                        end_frame=frame_num - 1,
                    ))
                pair_streak[pair] = 0

    # Check remaining active streaks
    for pair, streak in pair_streak.items():
        if streak >= min_duration:
            periods.append(ConvergencePeriod(
                track_a=pair[0],
                track_b=pair[1],
                start_frame=pair_start[pair],
                end_frame=max(by_frame.keys()),
            ))

    return periods


def _compute_iou(a: PlayerPosition, b: PlayerPosition) -> float:
    """Compute IoU between two normalized bounding boxes."""
    a_x1 = a.x - a.width / 2
    a_y1 = a.y - a.height / 2
    a_x2 = a.x + a.width / 2
    a_y2 = a.y + a.height / 2

    b_x1 = b.x - b.width / 2
    b_y1 = b.y - b.height / 2
    b_x2 = b.x + b.width / 2
    b_y2 = b.y + b.height / 2

    inter_x1 = max(a_x1, b_x1)
    inter_y1 = max(a_y1, b_y1)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)

    if inter_x1 >= inter_x2 or inter_y1 >= inter_y2:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    a_area = a.width * a.height
    b_area = b.width * b.height
    union_area = a_area + b_area - inter_area

    if union_area <= 0:
        return 0.0

    return inter_area / union_area


