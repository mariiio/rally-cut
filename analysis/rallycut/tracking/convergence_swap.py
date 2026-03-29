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
MIN_SWAP_SCORE = 0.45  # Minimum combined score to apply swap

# Gap convergence detection: overlapping detection gaps between cross-team tracks
# indicate a net interaction that happened while players were undetected.
MIN_GAP_FRAMES = 15  # Minimum gap length to consider (short flickers are not interactions)
GAP_OVERLAP_MARGIN = 10  # Gaps within this many frames of each other count as overlapping


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
    team_score: float
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


@dataclass
class _TeamProfile:
    """Full-rally summary of a team's spatial properties."""

    median_y: float
    median_height: float


def _build_team_profiles(
    positions: list[PlayerPosition],
    teams: dict[int, int],
) -> dict[int, _TeamProfile]:
    """Build per-team spatial profiles from full-rally positions."""
    team_ys: dict[int, list[float]] = defaultdict(list)
    team_hs: dict[int, list[float]] = defaultdict(list)
    for p in positions:
        t = teams.get(p.track_id)
        if t is not None:
            team_ys[t].append(p.y)
            team_hs[t].append(p.height)

    profiles: dict[int, _TeamProfile] = {}
    for team_id in team_ys:
        profiles[team_id] = _TeamProfile(
            median_y=float(np.median(team_ys[team_id])),
            median_height=float(np.median(team_hs[team_id])),
        )
    return profiles


def _score_team_consistency(
    a_after: _TrackWindow,
    b_after: _TrackWindow,
    team_a: int,
    team_b: int,
    team_profiles: dict[int, _TeamProfile],
) -> float:
    """Score team-consistency evidence for a swap.

    After a convergence, checks if each track's post-convergence properties
    (Y position and bbox height) are more consistent with its OWN team's
    profile or the OTHER team's profile. Uses the same signals that team
    classification used — so it works regardless of camera angle.

    Returns 0-1. High score = tracks are on each other's team side.
    """
    if team_a not in team_profiles or team_b not in team_profiles:
        return 0.0
    if team_a == team_b:
        return 0.0

    prof_a = team_profiles[team_a]
    prof_b = team_profiles[team_b]

    # How far apart are the team profiles? If teams aren't separated,
    # this signal is uninformative.
    y_sep = abs(prof_a.median_y - prof_b.median_y)
    h_sep = abs(prof_a.median_height - prof_b.median_height)
    if y_sep < 0.02 and h_sep < 0.02:
        return 0.0

    # For each track, compute distance to own team vs other team.
    # Use whichever signal (Y or height) has better separation.
    def _team_dist(window: _TrackWindow, profile: _TeamProfile) -> float:
        y_d = abs(window.median_y - profile.median_y)
        h_d = abs(math.sqrt(window.mean_bbox_area) - profile.median_height)
        # Weight by separation quality
        if y_sep > h_sep:
            return y_d / max(y_sep, 0.01)
        return h_d / max(h_sep, 0.01)

    # Track A: distance to own team vs other team
    a_own = _team_dist(a_after, prof_a)
    a_other = _team_dist(a_after, prof_b)

    # Track B: distance to own team vs other team
    b_own = _team_dist(b_after, prof_b)
    b_other = _team_dist(b_after, prof_a)

    # Both tracks closer to the OTHER team's profile → swap evidence
    a_on_wrong_side = a_own > a_other
    b_on_wrong_side = b_own > b_other

    if a_on_wrong_side and b_on_wrong_side:
        # Both tracks are closer to the other team → strong swap evidence.
        # Score by how decisive the mismatch is.
        a_margin = (a_own - a_other) / max(a_own + a_other, 0.01)
        b_margin = (b_own - b_other) / max(b_own + b_other, 0.01)
        return min(1.0, (a_margin + b_margin) / 2 + 0.3)

    # Only one track on wrong side → not enough evidence.
    # At a net interaction, one player naturally approaches the other
    # team's side. Require BOTH to show inconsistency.
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


def _find_detection_gaps(
    positions: list[PlayerPosition],
    track_id: int,
    min_gap_frames: int = MIN_GAP_FRAMES,
) -> list[tuple[int, int]]:
    """Find detection gaps for a track.

    Returns list of (gap_start_frame, gap_end_frame) where the track
    has no detections. Only gaps >= min_gap_frames are returned.
    """
    frames = sorted(
        p.frame_number for p in positions if p.track_id == track_id
    )
    if len(frames) < 2:
        return []

    gaps: list[tuple[int, int]] = []
    for i in range(1, len(frames)):
        gap_len = frames[i] - frames[i - 1] - 1
        if gap_len >= min_gap_frames:
            gaps.append((frames[i - 1] + 1, frames[i] - 1))
    return gaps


def _detect_gap_convergences(
    positions: list[PlayerPosition],
    primary_track_ids: list[int],
    teams: dict[int, int],
    min_gap_frames: int = MIN_GAP_FRAMES,
    overlap_margin: int = GAP_OVERLAP_MARGIN,
) -> list[ConvergencePeriod]:
    """Detect "gap convergences" — overlapping detection gaps between cross-team tracks.

    When two cross-team players interact at the net and both lose detection,
    the interaction isn't visible as a normal convergence (both absent).
    This function finds these phantom interactions by checking for
    temporally overlapping gaps between cross-team primary tracks.

    Args:
        positions: Player positions.
        primary_track_ids: Primary player track IDs.
        teams: Team assignments (track_id -> 0=near, 1=far).
        min_gap_frames: Minimum gap length to consider.
        overlap_margin: Gaps within this many frames count as overlapping.

    Returns:
        List of synthetic ConvergencePeriod objects for gap-based interactions.
    """
    # Find gaps per primary track
    track_gaps: dict[int, list[tuple[int, int]]] = {}
    for tid in primary_track_ids:
        gaps = _find_detection_gaps(positions, tid, min_gap_frames)
        if gaps:
            track_gaps[tid] = gaps

    if len(track_gaps) < 2:
        return []

    # Check all cross-team pairs for overlapping gaps
    gap_convergences: list[ConvergencePeriod] = []
    checked: set[tuple[int, int]] = set()

    for tid_a, gaps_a in track_gaps.items():
        team_a = teams.get(tid_a)
        if team_a is None:
            continue
        for tid_b, gaps_b in track_gaps.items():
            if tid_b <= tid_a:
                continue
            pair = (tid_a, tid_b)
            if pair in checked:
                continue
            checked.add(pair)

            team_b = teams.get(tid_b)
            if team_b is None or team_b == team_a:
                continue  # Same team — skip

            # Check for overlapping gaps
            for ga_start, ga_end in gaps_a:
                for gb_start, gb_end in gaps_b:
                    # Gaps overlap if they're within margin of each other
                    overlap_start = max(ga_start, gb_start)
                    overlap_end = min(ga_end, gb_end)

                    if overlap_end - overlap_start >= -overlap_margin:
                        # Create synthetic convergence spanning the union
                        conv_start = min(ga_start, gb_start)
                        conv_end = max(ga_end, gb_end)
                        gap_convergences.append(ConvergencePeriod(
                            track_a=tid_a,
                            track_b=tid_b,
                            start_frame=conv_start,
                            end_frame=conv_end,
                        ))
                        logger.debug(
                            f"Gap convergence: T{tid_a}[{ga_start}-{ga_end}] "
                            f"x T{tid_b}[{gb_start}-{gb_end}] "
                            f"→ frames {conv_start}-{conv_end}"
                        )

    return gap_convergences


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

    # Also detect gap convergences: overlapping detection gaps between
    # cross-team tracks indicate a net interaction during the gap.
    gap_convergences = _detect_gap_convergences(
        positions, primary_track_ids, teams,
    )
    all_convergences = cross_team_convergences + gap_convergences

    if not all_convergences:
        return positions, 0

    # Build per-team profiles from full rally for team-consistency scoring
    team_profiles = _build_team_profiles(positions, teams)

    # Score each convergence event (both visible and gap-based)
    candidates: list[_SwapCandidate] = []

    for conv in all_convergences:
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
        team_a = teams.get(conv.track_a, -1)
        team_b = teams.get(conv.track_b, -1)
        team_score = _score_team_consistency(
            a_after, b_after, team_a, team_b, team_profiles,
        )
        size_score = _score_size_swap(
            a_before, a_after, b_before, b_after,
        )
        appearance_score = _score_appearance_swap(
            a_before, a_after, b_before, b_after,
        )

        # Weighted combination
        total = (
            WEIGHT_COURT_SIDE * team_score
            + WEIGHT_SIZE * size_score
            + WEIGHT_APPEARANCE * appearance_score
        )

        if total >= MIN_SWAP_SCORE:
            candidates.append(_SwapCandidate(
                track_a=conv.track_a,
                track_b=conv.track_b,
                convergence=conv,
                swap_frame=swap_frame,
                team_score=team_score,
                size_score=size_score,
                appearance_score=appearance_score,
                total_score=total,
            ))

        logger.debug(
            f"Convergence swap: T{conv.track_a}<->T{conv.track_b} "
            f"frames {conv.start_frame}-{conv.end_frame}: "
            f"team={team_score:.2f} size={size_score:.2f} "
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
                f"score={cand.total_score:.2f}: court={cand.team_score:.2f} "
                f"size={cand.size_score:.2f} appearance={cand.appearance_score:.2f})"
            )

    return positions, num_repairs
