"""Convergence-anchored swap detection for cross-team ID switches.

Detects and fixes track ID swaps that occur during net interactions
(convergence periods where cross-team players are in close proximity).

Unlike whole-rally approaches, this module checks for swaps at specific
convergence events by comparing track properties immediately before and
after each event. Multiple signals are scored:

1. Court-side flip: did both tracks switch court sides?
2. Bbox size swap: did the near-side track become small and vice versa?
3. Appearance change: did the tracks' histograms swap?

A swap is confirmed when the combined evidence exceeds a threshold.

Inserted as Step 4d after global identity and before quality report.
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np

from rallycut.tracking.color_repair import (
    ColorHistogramStore,
    ConvergencePeriod,
    detect_convergence_periods,
)
from rallycut.tracking.player_tracker import PlayerPosition

logger = logging.getLogger(__name__)

# Convergence detection
PROXIMITY_THRESHOLD = 0.10  # Centroid distance for net interactions
MIN_CONVERGENCE_FRAMES = 3  # Minimum frames of proximity

# Before/after comparison windows
COMPARE_WINDOW = 30  # Frames to sample before/after convergence
SEPARATION_GAP = 10  # Skip frames after convergence to let players separate
MIN_COMPARE_FRAMES = 5  # Minimum frames needed in a comparison window

# Signal thresholds
COURT_SIDE_MARGIN = 0.02  # Ignore Y positions within this of split
SIZE_CHANGE_THRESHOLD = 0.15  # Min relative bbox area change to count as signal

# Scoring weights (sum to 1.0)
WEIGHT_COURT_SIDE = 0.35
WEIGHT_SIZE = 0.30
WEIGHT_APPEARANCE = 0.35
MIN_SWAP_SCORE = 0.35  # Minimum combined score to apply swap


@dataclass
class _TrackWindow:
    """Summary of a track's properties in a time window."""

    median_y: float
    mean_bbox_area: float
    histogram: np.ndarray | None
    frame_count: int


@dataclass
class _SwapCandidate:
    """A potential swap detected at a convergence event."""

    track_a: int
    track_b: int
    convergence: ConvergencePeriod
    swap_frame: int
    court_side_score: float
    size_score: float
    appearance_score: float
    total_score: float


def _compute_court_split(
    positions: list[PlayerPosition],
    primary_track_ids: list[int],
    early_frames: int = 60,
) -> tuple[float, dict[int, int]] | None:
    """Compute court split from primary tracks' early-frame Y positions."""
    if len(primary_track_ids) < 4:
        return None

    track_ys: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for p in positions:
        if p.track_id in primary_track_ids:
            track_ys[p.track_id].append((p.frame_number, p.y))

    median_ys: dict[int, float] = {}
    for tid in primary_track_ids:
        entries = track_ys.get(tid, [])
        if not entries:
            continue
        entries.sort(key=lambda x: x[0])
        ys = sorted(y for _, y in entries[:early_frames])
        if ys:
            median_ys[tid] = ys[len(ys) // 2]

    if len(median_ys) < 4:
        return None

    sorted_tracks = sorted(median_ys.items(), key=lambda x: x[1])
    far_tracks = sorted_tracks[:2]
    near_tracks = sorted_tracks[2:]

    gap = near_tracks[0][1] - far_tracks[-1][1]
    if gap < 0.03:
        return None

    split_y = (far_tracks[-1][1] + near_tracks[0][1]) / 2.0
    teams: dict[int, int] = {}
    for tid, _ in near_tracks:
        teams[tid] = 0  # near
    for tid, _ in far_tracks:
        teams[tid] = 1  # far
    return split_y, teams


def _get_track_window(
    positions: list[PlayerPosition],
    track_id: int,
    start_frame: int,
    end_frame: int,
    color_store: ColorHistogramStore | None,
) -> _TrackWindow | None:
    """Get summary properties for a track in a frame range."""
    track_pos = [
        p for p in positions
        if p.track_id == track_id
        and start_frame <= p.frame_number <= end_frame
    ]
    if len(track_pos) < MIN_COMPARE_FRAMES:
        return None

    ys = sorted(p.y for p in track_pos)
    median_y = ys[len(ys) // 2]
    mean_area = sum(p.width * p.height for p in track_pos) / len(track_pos)

    histogram: np.ndarray | None = None
    if color_store is not None:
        hists: list[np.ndarray] = []
        for p in track_pos:
            h = color_store.get(track_id, p.frame_number)
            if h is not None:
                hists.append(h)
        if hists:
            mean_h: np.ndarray = np.mean(np.stack(hists), axis=0).astype(
                np.float32
            )
            total = mean_h.sum()
            if total > 0:
                mean_h /= total
            histogram = mean_h

    return _TrackWindow(
        median_y=median_y,
        mean_bbox_area=mean_area,
        histogram=histogram,
        frame_count=len(track_pos),
    )


def _score_court_side_flip(
    a_before: _TrackWindow,
    a_after: _TrackWindow,
    b_before: _TrackWindow,
    b_after: _TrackWindow,
    split_y: float,
) -> float:
    """Score court-side evidence for a swap.

    Returns 0-1. High score = both tracks flipped sides at convergence.
    """
    def _side(y: float) -> int | None:
        if abs(y - split_y) < COURT_SIDE_MARGIN:
            return None
        return 0 if y > split_y else 1  # 0=near, 1=far

    a_side_before = _side(a_before.median_y)
    a_side_after = _side(a_after.median_y)
    b_side_before = _side(b_before.median_y)
    b_side_after = _side(b_after.median_y)

    # Both tracks must have classifiable sides
    if None in (a_side_before, a_side_after, b_side_before, b_side_after):
        return 0.0

    # Both flipped: strong evidence
    a_flipped = a_side_before != a_side_after
    b_flipped = b_side_before != b_side_after
    if a_flipped and b_flipped:
        # Extra: did they flip to each other's original side?
        if a_side_after == b_side_before and b_side_after == a_side_before:
            return 1.0
        return 0.7

    # Only one flipped: moderate evidence
    if a_flipped or b_flipped:
        return 0.4

    return 0.0


def _score_size_swap(
    a_before: _TrackWindow,
    a_after: _TrackWindow,
    b_before: _TrackWindow,
    b_after: _TrackWindow,
) -> float:
    """Score bbox size evidence for a swap.

    Returns 0-1. High score = near-side track became small and vice versa.
    """
    if a_before.mean_bbox_area <= 0 or b_before.mean_bbox_area <= 0:
        return 0.0
    if a_after.mean_bbox_area <= 0 or b_after.mean_bbox_area <= 0:
        return 0.0

    # Compare size ratios: before swap, A and B have different sizes.
    # After swap, A should have B's old size and vice versa.
    a_ratio = a_after.mean_bbox_area / a_before.mean_bbox_area
    b_ratio = b_after.mean_bbox_area / b_before.mean_bbox_area

    # One got bigger, one got smaller
    if not ((a_ratio > 1.0) != (b_ratio > 1.0)):
        return 0.0

    a_change = abs(math.log(max(a_ratio, 0.01)))
    b_change = abs(math.log(max(b_ratio, 0.01)))

    if a_change < SIZE_CHANGE_THRESHOLD or b_change < SIZE_CHANGE_THRESHOLD:
        return 0.0

    # Check cross-match: A's new size should be closer to B's old size
    # than to A's old size (and vice versa)
    a_self_diff = abs(a_after.mean_bbox_area - a_before.mean_bbox_area)
    a_cross_diff = abs(a_after.mean_bbox_area - b_before.mean_bbox_area)
    b_self_diff = abs(b_after.mean_bbox_area - b_before.mean_bbox_area)
    b_cross_diff = abs(b_after.mean_bbox_area - a_before.mean_bbox_area)

    cross_better = 0
    if a_cross_diff < a_self_diff:
        cross_better += 1
    if b_cross_diff < b_self_diff:
        cross_better += 1

    if cross_better == 2:
        return 0.8
    if cross_better == 1:
        return 0.4
    return 0.0


def _score_appearance_swap(
    a_before: _TrackWindow,
    a_after: _TrackWindow,
    b_before: _TrackWindow,
    b_after: _TrackWindow,
) -> float:
    """Score appearance evidence for a swap.

    Returns 0-1. High score = tracks' histograms swapped at convergence.
    """
    h_ab = a_before.histogram
    h_aa = a_after.histogram
    h_bb = b_before.histogram
    h_ba = b_after.histogram
    if h_ab is None or h_aa is None or h_bb is None or h_ba is None:
        return 0.0

    # Compute distances
    d_a_self = cv2.compareHist(h_ab, h_aa, cv2.HISTCMP_BHATTACHARYYA)
    d_b_self = cv2.compareHist(h_bb, h_ba, cv2.HISTCMP_BHATTACHARYYA)
    d_a_cross = cv2.compareHist(h_ab, h_ba, cv2.HISTCMP_BHATTACHARYYA)
    d_b_cross = cv2.compareHist(h_bb, h_aa, cv2.HISTCMP_BHATTACHARYYA)

    # Both tracks should show appearance change (self distance high)
    # AND cross-match should be better than self-match
    cross_better = 0
    if d_a_cross < d_a_self and d_a_self > 0.15:
        cross_better += 1
    if d_b_cross < d_b_self and d_b_self > 0.15:
        cross_better += 1

    if cross_better == 2:
        return 0.8
    if cross_better == 1:
        return 0.4
    return 0.0


def detect_convergence_swaps(
    positions: list[PlayerPosition],
    primary_track_ids: list[int],
    color_store: ColorHistogramStore | None = None,
    upstream_split_y: float | None = None,
    upstream_teams: dict[int, int] | None = None,
) -> tuple[list[PlayerPosition], int]:
    """Detect and fix cross-team ID swaps at convergence points.

    For each convergence event between cross-team tracks, compares track
    properties before vs after to determine if a swap occurred. Uses
    court-side, bbox size, and appearance signals.

    Args:
        positions: Player positions (modified in place).
        primary_track_ids: The primary player track IDs.
        color_store: Optional color histogram store.
        upstream_split_y: Optional upstream court split Y.
        upstream_teams: Optional upstream team assignments.

    Returns:
        Tuple of (positions, num_repairs).
    """
    if len(primary_track_ids) < 4:
        return positions, 0

    # Compute court split (try self-computed, fall back to upstream)
    split_y: float | None = None
    teams: dict[int, int] | None = None

    split_result = _compute_court_split(positions, primary_track_ids)
    if split_result is not None:
        split_y, teams = split_result
    elif upstream_split_y is not None and upstream_teams:
        split_y = upstream_split_y
        teams = upstream_teams

    if split_y is None or not teams:
        return positions, 0

    # Detect convergence periods with proximity
    primary_set = set(primary_track_ids)
    convergences = detect_convergence_periods(
        positions,
        proximity_threshold=PROXIMITY_THRESHOLD,
        min_duration=MIN_CONVERGENCE_FRAMES,
    )

    # Filter to cross-team convergences involving primary tracks
    cross_team_convergences: list[ConvergencePeriod] = []
    for conv in convergences:
        if conv.track_a not in primary_set or conv.track_b not in primary_set:
            continue
        team_a = teams.get(conv.track_a)
        team_b = teams.get(conv.track_b)
        if team_a is not None and team_b is not None and team_a != team_b:
            cross_team_convergences.append(conv)

    if not cross_team_convergences:
        return positions, 0

    # Score each convergence event
    candidates: list[_SwapCandidate] = []

    for conv in cross_team_convergences:
        swap_frame = conv.end_frame + 1
        after_start = conv.end_frame + 1 + SEPARATION_GAP

        # Get before/after windows
        a_before = _get_track_window(
            positions, conv.track_a,
            conv.start_frame - COMPARE_WINDOW, conv.start_frame - 1,
            color_store,
        )
        a_after = _get_track_window(
            positions, conv.track_a,
            after_start, after_start + COMPARE_WINDOW,
            color_store,
        )
        b_before = _get_track_window(
            positions, conv.track_b,
            conv.start_frame - COMPARE_WINDOW, conv.start_frame - 1,
            color_store,
        )
        b_after = _get_track_window(
            positions, conv.track_b,
            after_start, after_start + COMPARE_WINDOW,
            color_store,
        )

        if any(w is None for w in (a_before, a_after, b_before, b_after)):
            continue

        assert a_before is not None and a_after is not None
        assert b_before is not None and b_after is not None

        # Score each signal
        court_score = _score_court_side_flip(
            a_before, a_after, b_before, b_after, split_y,
        )
        size_score = _score_size_swap(
            a_before, a_after, b_before, b_after,
        )
        appearance_score = _score_appearance_swap(
            a_before, a_after, b_before, b_after,
        )

        # Weighted combination
        total = (
            WEIGHT_COURT_SIDE * court_score
            + WEIGHT_SIZE * size_score
            + WEIGHT_APPEARANCE * appearance_score
        )

        if total >= MIN_SWAP_SCORE:
            candidates.append(_SwapCandidate(
                track_a=conv.track_a,
                track_b=conv.track_b,
                convergence=conv,
                swap_frame=swap_frame,
                court_side_score=court_score,
                size_score=size_score,
                appearance_score=appearance_score,
                total_score=total,
            ))

        logger.debug(
            f"Convergence swap: T{conv.track_a}<->T{conv.track_b} "
            f"frames {conv.start_frame}-{conv.end_frame}: "
            f"court={court_score:.2f} size={size_score:.2f} "
            f"appearance={appearance_score:.2f} total={total:.2f}"
            f"{' -> SWAP' if total >= MIN_SWAP_SCORE else ''}"
        )

    if not candidates:
        return positions, 0

    # Sort by score (highest first) and apply non-conflicting swaps
    candidates.sort(key=lambda c: -c.total_score)
    swapped_tracks: set[int] = set()
    num_repairs = 0

    for cand in candidates:
        if cand.track_a in swapped_tracks or cand.track_b in swapped_tracks:
            continue

        # Apply swap after convergence end
        swapped = 0
        for p in positions:
            if p.frame_number >= cand.swap_frame:
                if p.track_id == cand.track_a:
                    p.track_id = cand.track_b
                    swapped += 1
                elif p.track_id == cand.track_b:
                    p.track_id = cand.track_a
                    swapped += 1

        if color_store is not None:
            color_store.swap(cand.track_a, cand.track_b, cand.swap_frame)

        if swapped > 0:
            swapped_tracks.add(cand.track_a)
            swapped_tracks.add(cand.track_b)
            num_repairs += 1
            logger.info(
                f"Convergence swap: fixed T{cand.track_a}<->T{cand.track_b} "
                f"at frame {cand.swap_frame} ({swapped} positions, "
                f"score={cand.total_score:.2f}: court={cand.court_side_score:.2f} "
                f"size={cand.size_score:.2f} appearance={cand.appearance_score:.2f})"
            )

    return positions, num_repairs
