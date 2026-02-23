"""Global identity optimization for player tracking.

After local post-processing (jump split, color split, court identity, swap detect,
tracklet link, stabilize), identity confusion can cascade across net interactions.
This module adds a global pass that:

1. Splits tracks at interaction boundaries into segments
2. Selects reliable anchor segments per team
3. Builds canonical player profiles from anchors
4. Assigns all segments to canonical players via greedy cost minimization

Inserted as Step 1b after stabilize_track_ids and before PlayerFilter.
"""

from __future__ import annotations

import functools
import logging
import math
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import cv2
import numpy as np

from rallycut.tracking.color_repair import (
    ColorHistogramStore,
    ConvergencePeriod,
    detect_convergence_periods,
)
from rallycut.tracking.player_tracker import PlayerPosition

if TYPE_CHECKING:
    from rallycut.tracking.appearance_descriptor import (
        AppearanceDescriptorStore,
        MultiRegionDescriptor,
    )

logger = logging.getLogger(__name__)

# Segment extraction
INTERACTION_MARGIN_FRAMES = 5  # ±frames around interaction boundary
MIN_SEGMENT_FRAMES = 5  # Shorter than track filters (20/50) to capture fine-grained splits

# Anchor selection
MIN_ANCHOR_BHATTACHARYYA = 0.20  # Minimum distance between same-team anchors

# Cost function weights (must sum to 1.0)
WEIGHT_APPEARANCE = 0.50
WEIGHT_SPATIAL = 0.25
WEIGHT_BBOX_SIZE = 0.10
WEIGHT_MULTI_REGION = 0.15
SPATIAL_NORMALIZER = 0.30  # Euclidean distance normalization

# Assignment
MAX_REASSIGNMENT_ROUNDS = 10
MAX_ASSIGNMENT_COST = 0.70  # Drop segments with best cost above this


@dataclass
class TrackSegment:
    """A contiguous segment of a track between interaction boundaries."""

    track_id: int
    start_frame: int
    end_frame: int
    team: int  # 0=near, 1=far
    positions: list[PlayerPosition] = field(default_factory=list, repr=False)

    @property
    def duration(self) -> int:
        return self.end_frame - self.start_frame + 1

    @functools.cached_property
    def centroid(self) -> tuple[float, float]:
        if not self.positions:
            return (0.5, 0.5)
        xs = [p.x for p in self.positions]
        ys = [p.y for p in self.positions]
        return (sum(xs) / len(xs), sum(ys) / len(ys))

    @functools.cached_property
    def mean_bbox_area(self) -> float:
        if not self.positions:
            return 0.0
        return sum(p.width * p.height for p in self.positions) / len(self.positions)


@dataclass
class PlayerProfile:
    """Canonical player appearance and spatial profile built from anchor."""

    player_id: int  # Canonical track ID
    team: int
    histogram: np.ndarray | None = field(default=None, repr=False)
    centroid: tuple[float, float] = (0.5, 0.5)
    mean_bbox_area: float = 0.0


@dataclass
class GlobalIdentityResult:
    """Result of the global identity optimization."""

    skipped: bool = True
    skip_reason: str = ""
    num_segments: int = 0
    num_remapped: int = 0
    num_interactions: int = 0


def optimize_global_identity(
    positions: list[PlayerPosition],
    team_assignments: dict[int, int],
    color_store: ColorHistogramStore,
    court_split_y: float | None = None,
    appearance_store: AppearanceDescriptorStore | None = None,
) -> tuple[list[PlayerPosition], GlobalIdentityResult]:
    """Run global identity optimization on tracked positions.

    Args:
        positions: Player positions after stabilize_track_ids.
        team_assignments: Track ID -> team (0=near, 1=far).
        color_store: Per-frame color histograms.
        court_split_y: Y-coordinate splitting near/far court.
        appearance_store: Optional multi-region appearance descriptors.

    Returns:
        Tuple of (positions, result). Positions are modified in place when
        optimization runs; returned unchanged when skipped.
    """
    result = GlobalIdentityResult()

    # Activation gate: need color data and team info
    if not color_store.has_data():
        result.skip_reason = "no color data"
        return positions, result

    if not team_assignments:
        result.skip_reason = "no team assignments"
        return positions, result

    # Detect cross-team convergence periods
    interactions = _detect_cross_team_interactions(
        positions, team_assignments
    )
    result.num_interactions = len(interactions)

    if not interactions:
        result.skip_reason = "no cross-team interactions"
        return positions, result

    # Check if tracks are already clean (skip gate)
    if _tracks_are_clean(positions, team_assignments):
        result.skip_reason = "tracks already clean"
        return positions, result

    # Phase A: Extract segments by splitting at interaction boundaries
    segments = _extract_segments(positions, interactions, team_assignments)

    # Check per-team segment counts
    team_segments: dict[int, list[TrackSegment]] = defaultdict(list)
    for seg in segments:
        team_segments[seg.team].append(seg)

    for team_id in (0, 1):
        if len(team_segments.get(team_id, [])) < 2:
            result.skip_reason = f"team {team_id} has <2 segments"
            return positions, result

    result.num_segments = len(segments)

    # Phase B: Select anchors and build profiles
    profiles: dict[int, list[PlayerProfile]] = {}  # team -> profiles
    for team_id in (0, 1):
        team_segs = team_segments[team_id]
        team_profiles = _select_anchors_and_build_profiles(
            team_segs, color_store, court_split_y
        )
        if len(team_profiles) < 1:
            result.skip_reason = f"team {team_id}: no valid anchors"
            return positions, result
        profiles[team_id] = team_profiles

    # Phase C: Global assignment per team
    total_remapped = 0
    for team_id in (0, 1):
        team_segs = team_segments[team_id]
        team_profiles = profiles[team_id]

        if len(team_profiles) < 2:
            # Single player on this team: assign all to that player
            canonical_id = team_profiles[0].player_id
            remapped = _assign_all_to_single(
                positions, team_segs, canonical_id
            )
            total_remapped += remapped
            continue

        remapped = _assign_segments_to_profiles(
            positions, team_segs, team_profiles,
            color_store, appearance_store,
        )
        total_remapped += remapped

    result.num_remapped = total_remapped
    if total_remapped > 0:
        result.skipped = False
    else:
        result.skip_reason = "no remapping needed"

    return positions, result


def _detect_cross_team_interactions(
    positions: list[PlayerPosition],
    team_assignments: dict[int, int],
) -> list[ConvergencePeriod]:
    """Detect convergence periods between tracks on different teams."""
    all_periods = detect_convergence_periods(positions)
    cross_team: list[ConvergencePeriod] = []
    for period in all_periods:
        team_a = team_assignments.get(period.track_a)
        team_b = team_assignments.get(period.track_b)
        if team_a is not None and team_b is not None and team_a != team_b:
            cross_team.append(period)
    return cross_team


def _tracks_are_clean(
    positions: list[PlayerPosition],
    team_assignments: dict[int, int],
    min_frames: int = 20,
) -> bool:
    """Check if tracks are already clean (no optimization needed).

    Clean = at most 2 tracks per team with >=min_frames each,
    and no same-team temporal overlap.
    """
    # Count frames per track
    track_frames: dict[int, int] = defaultdict(int)
    for p in positions:
        if p.track_id >= 0:
            track_frames[p.track_id] += 1

    # Significant tracks per team
    team_tracks: dict[int, list[int]] = defaultdict(list)
    for tid, count in track_frames.items():
        if count >= min_frames:
            team = team_assignments.get(tid)
            if team is not None:
                team_tracks[team].append(tid)

    # More than 2 significant tracks on either team = not clean
    for team_id in (0, 1):
        if len(team_tracks.get(team_id, [])) > 2:
            return False

    # Check for same-team temporal overlap
    track_ranges: dict[int, tuple[int, int]] = {}
    for p in positions:
        if p.track_id >= 0:
            if p.track_id not in track_ranges:
                track_ranges[p.track_id] = (p.frame_number, p.frame_number)
            else:
                old_min, old_max = track_ranges[p.track_id]
                track_ranges[p.track_id] = (
                    min(old_min, p.frame_number),
                    max(old_max, p.frame_number),
                )

    for team_id in (0, 1):
        tids = team_tracks.get(team_id, [])
        if len(tids) == 2:
            r0 = track_ranges.get(tids[0])
            r1 = track_ranges.get(tids[1])
            if r0 is not None and r1 is not None:
                if r0[0] <= r1[1] and r1[0] <= r0[1]:
                    return False

    return True


def _extract_segments(
    positions: list[PlayerPosition],
    interactions: list[ConvergencePeriod],
    team_assignments: dict[int, int],
) -> list[TrackSegment]:
    """Split tracks at interaction boundaries into segments.

    Each interaction boundary creates a split point with ±margin.
    Segments shorter than MIN_SEGMENT_FRAMES are dropped.
    """
    # Collect split frames per track (frame numbers where to split)
    track_split_frames: dict[int, set[int]] = defaultdict(set)
    for interaction in interactions:
        for tid in (interaction.track_a, interaction.track_b):
            track_split_frames[tid].add(
                interaction.start_frame - INTERACTION_MARGIN_FRAMES
            )
            track_split_frames[tid].add(
                interaction.end_frame + INTERACTION_MARGIN_FRAMES
            )

    # Group positions by track
    track_positions: dict[int, list[PlayerPosition]] = defaultdict(list)
    for p in positions:
        if p.track_id >= 0:
            track_positions[p.track_id].append(p)

    segments: list[TrackSegment] = []

    for tid, track_pos in track_positions.items():
        team = team_assignments.get(tid)
        if team is None:
            continue

        track_pos.sort(key=lambda p: p.frame_number)

        if tid not in track_split_frames:
            # No splits needed — entire track is one segment
            if len(track_pos) >= MIN_SEGMENT_FRAMES:
                segments.append(TrackSegment(
                    track_id=tid,
                    start_frame=track_pos[0].frame_number,
                    end_frame=track_pos[-1].frame_number,
                    team=team,
                    positions=list(track_pos),
                ))
            continue

        # Split at boundary frames using index (avoid O(n) pop(0))
        split_points = sorted(track_split_frames[tid])
        split_idx = 0
        current_segment_pos: list[PlayerPosition] = []

        for p in track_pos:
            while split_idx < len(split_points) and p.frame_number >= split_points[split_idx]:
                if len(current_segment_pos) >= MIN_SEGMENT_FRAMES:
                    segments.append(TrackSegment(
                        track_id=tid,
                        start_frame=current_segment_pos[0].frame_number,
                        end_frame=current_segment_pos[-1].frame_number,
                        team=team,
                        positions=list(current_segment_pos),
                    ))
                current_segment_pos = []
                split_idx += 1

            current_segment_pos.append(p)

        # Finalize last segment
        if len(current_segment_pos) >= MIN_SEGMENT_FRAMES:
            segments.append(TrackSegment(
                track_id=tid,
                start_frame=current_segment_pos[0].frame_number,
                end_frame=current_segment_pos[-1].frame_number,
                team=team,
                positions=list(current_segment_pos),
            ))

    return segments


def _compute_segment_reliability(
    segment: TrackSegment,
    color_store: ColorHistogramStore,
    court_split_y: float | None,
) -> float:
    """Score segment reliability for anchor selection.

    Higher = more reliable identity signal.
    """
    # Duration score (log-scaled, caps at ~200 frames)
    duration_score = min(math.log2(max(segment.duration, 1)) / 8.0, 1.0)

    # Color consistency (low intra-segment variance = stable appearance)
    histograms = []
    for p in segment.positions:
        h = color_store.get(segment.track_id, p.frame_number)
        if h is not None:
            histograms.append(h)

    color_consistency = 0.5  # default
    if len(histograms) >= 3:
        mean_hist = np.mean(np.stack(histograms), axis=0).astype(np.float32)
        total = mean_hist.sum()
        if total > 0:
            mean_hist /= total
        dists = []
        for h in histograms[::3]:  # Sample every 3rd for speed
            d = cv2.compareHist(
                mean_hist, h.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA
            )
            dists.append(d)
        avg_dist = sum(dists) / len(dists) if dists else 0.5
        # Low distance = high consistency (invert)
        color_consistency = max(0.0, 1.0 - avg_dist * 2.0)

    # Court clarity (distance from court_split_y — farther = clearer team)
    court_clarity = 0.5
    if court_split_y is not None:
        _, cy = segment.centroid
        dist_from_split = abs(cy - court_split_y)
        court_clarity = min(dist_from_split / 0.20, 1.0)

    return (
        0.40 * duration_score
        + 0.35 * color_consistency
        + 0.25 * court_clarity
    )


def _get_segment_mean_histogram(
    segment: TrackSegment,
    color_store: ColorHistogramStore,
    max_samples: int = 30,
) -> np.ndarray | None:
    """Get mean histogram for a segment from the color store."""
    histograms: list[np.ndarray] = []
    for p in segment.positions:
        h = color_store.get(segment.track_id, p.frame_number)
        if h is not None:
            histograms.append(h)

    if not histograms:
        return None

    # Use up to max_samples evenly spaced histograms
    if len(histograms) > max_samples:
        step = len(histograms) / max_samples
        histograms = [histograms[int(i * step)] for i in range(max_samples)]

    mean: np.ndarray = np.mean(np.stack(histograms), axis=0).astype(np.float32)
    total = mean.sum()
    if total > 0:
        mean /= total
    return mean


def _select_anchors_and_build_profiles(
    team_segments: list[TrackSegment],
    color_store: ColorHistogramStore,
    court_split_y: float | None,
) -> list[PlayerProfile]:
    """Select anchor segments and build player profiles for one team.

    Greedily picks the most reliable segment as anchor 1, then finds
    anchor 2 that doesn't temporally overlap and has different appearance.
    """
    if not team_segments:
        return []

    # Score all segments
    scored: list[tuple[float, int, TrackSegment]] = []
    for i, seg in enumerate(team_segments):
        rel = _compute_segment_reliability(seg, color_store, court_split_y)
        scored.append((rel, i, seg))
    scored.sort(key=lambda x: -x[0])  # Highest reliability first

    # Pre-compute histogram for each segment (used for anchor comparison)
    seg_histograms: dict[int, np.ndarray | None] = {}
    for _, i, seg in scored:
        seg_histograms[i] = _get_segment_mean_histogram(seg, color_store)

    # Pick anchor 1: highest reliability with a valid histogram
    anchor1: TrackSegment | None = None
    anchor1_idx: int = -1
    anchor1_hist: np.ndarray | None = None
    for rel, i, seg in scored:
        h = seg_histograms[i]
        if h is not None:
            anchor1 = seg
            anchor1_idx = i
            anchor1_hist = h
            break

    if anchor1 is None or anchor1_hist is None:
        return []

    # Pick anchor 2: highest reliability that:
    # - doesn't temporally overlap with anchor 1
    # - has Bhattacharyya distance > threshold from anchor 1
    # - comes from a different original track_id
    anchor2: TrackSegment | None = None
    anchor2_hist: np.ndarray | None = None
    for rel, i, seg in scored:
        if i == anchor1_idx:
            continue
        h = seg_histograms[i]
        if h is None:
            continue

        # Different track ID (same track = same player)
        if seg.track_id == anchor1.track_id:
            continue

        # No temporal overlap
        if seg.start_frame <= anchor1.end_frame and anchor1.start_frame <= seg.end_frame:
            continue

        # Sufficient appearance difference
        dist = cv2.compareHist(
            anchor1_hist, h.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA
        )
        if dist > MIN_ANCHOR_BHATTACHARYYA:
            anchor2 = seg
            anchor2_hist = h
            break

    # Build profiles
    profiles: list[PlayerProfile] = []
    profiles.append(PlayerProfile(
        player_id=anchor1.track_id,
        team=anchor1.team,
        histogram=anchor1_hist,
        centroid=anchor1.centroid,
        mean_bbox_area=anchor1.mean_bbox_area,
    ))

    if anchor2 is not None and anchor2_hist is not None:
        profiles.append(PlayerProfile(
            player_id=anchor2.track_id,
            team=anchor2.team,
            histogram=anchor2_hist,
            centroid=anchor2.centroid,
            mean_bbox_area=anchor2.mean_bbox_area,
        ))

    return profiles


def _compute_assignment_cost(
    segment: TrackSegment,
    profile: PlayerProfile,
    seg_histogram: np.ndarray | None,
    seg_multi_desc: MultiRegionDescriptor | None,
    profile_multi_desc: MultiRegionDescriptor | None,
    compute_multi_region_distance_fn: Callable[..., float] | None,
) -> float:
    """Compute cost of assigning a segment to a player profile.

    Lower cost = better match. Pre-computed histograms and multi-region
    descriptors are passed in to avoid redundant computation.

    Weights: appearance 0.50, spatial 0.25, bbox_size 0.10, multi_region 0.15.
    """
    cost = 0.0

    # 1. Appearance cost (shorts histogram Bhattacharyya)
    if seg_histogram is not None and profile.histogram is not None:
        appearance_dist = cv2.compareHist(
            profile.histogram, seg_histogram.astype(np.float32),
            cv2.HISTCMP_BHATTACHARYYA,
        )
        cost += WEIGHT_APPEARANCE * appearance_dist
    else:
        cost += WEIGHT_APPEARANCE * 0.5  # neutral penalty

    # 2. Spatial cost (segment centroid vs profile centroid)
    sx, sy = segment.centroid
    px, py = profile.centroid
    spatial_dist = math.sqrt((sx - px) ** 2 + (sy - py) ** 2)
    spatial_cost = min(spatial_dist / SPATIAL_NORMALIZER, 1.0)
    cost += WEIGHT_SPATIAL * spatial_cost

    # 3. Bbox size similarity (log ratio, saturates at ~7x difference)
    if profile.mean_bbox_area > 0 and segment.mean_bbox_area > 0:
        area_ratio = segment.mean_bbox_area / profile.mean_bbox_area
        size_cost = abs(math.log(max(area_ratio, 0.01))) / 2.0
        size_cost = min(size_cost, 1.0)
    else:
        size_cost = 0.5
    cost += WEIGHT_BBOX_SIZE * size_cost

    # 4. Multi-region appearance (when available)
    multi_cost = 0.5  # default neutral
    if (
        seg_multi_desc is not None
        and profile_multi_desc is not None
        and compute_multi_region_distance_fn is not None
    ):
        if seg_multi_desc.shorts is not None and profile_multi_desc.shorts is not None:
            multi_cost = compute_multi_region_distance_fn(
                seg_multi_desc, profile_multi_desc
            )
    cost += WEIGHT_MULTI_REGION * multi_cost

    return cost


def _assign_all_to_single(
    positions: list[PlayerPosition],
    segments: list[TrackSegment],
    canonical_id: int,
) -> int:
    """Assign all segments to a single canonical player ID."""
    # Build (track_id, frame_number) lookup for targeted remapping
    remap_keys: set[tuple[int, int]] = set()
    for seg in segments:
        if seg.track_id == canonical_id:
            continue
        for p in seg.positions:
            remap_keys.add((seg.track_id, p.frame_number))

    if not remap_keys:
        return 0

    remapped = 0
    for p in positions:
        if (p.track_id, p.frame_number) in remap_keys:
            p.track_id = canonical_id
            remapped += 1
    return remapped


def _assign_segments_to_profiles(
    positions: list[PlayerPosition],
    team_segments: list[TrackSegment],
    profiles: list[PlayerProfile],
    color_store: ColorHistogramStore,
    appearance_store: AppearanceDescriptorStore | None,
) -> int:
    """Assign each segment to its best-matching player profile.

    Uses greedy argmin per segment (not 1-to-1 Hungarian, since multiple
    segments can belong to the same player). Resolves temporal overlap
    conflicts iteratively by blocking the higher-cost assignment.
    """
    n_segs = len(team_segments)
    n_players = len(profiles)

    if n_segs == 0 or n_players == 0:
        return 0

    # Pre-compute segment histograms (avoid recomputing per profile)
    seg_histograms: list[np.ndarray | None] = [
        _get_segment_mean_histogram(seg, color_store)
        for seg in team_segments
    ]

    # Pre-compute multi-region descriptors
    seg_multi_descs: list[MultiRegionDescriptor | None] = [None] * n_segs
    profile_multi_descs: list[MultiRegionDescriptor | None] = [None] * n_players
    if appearance_store is not None and appearance_store.has_data():
        from rallycut.tracking.appearance_descriptor import (
            compute_track_mean_descriptor,
        )
        for i, seg in enumerate(team_segments):
            desc = compute_track_mean_descriptor(appearance_store, seg.track_id)
            seg_multi_descs[i] = desc if desc.shorts is not None else None
        for j, profile in enumerate(profiles):
            desc = compute_track_mean_descriptor(appearance_store, profile.player_id)
            profile_multi_descs[j] = desc if desc.shorts is not None else None

    # Hoist multi-region distance function (avoid import in hot loop)
    compute_multi_region_distance_fn = None
    if appearance_store is not None and appearance_store.has_data():
        from rallycut.tracking.appearance_descriptor import (
            compute_multi_region_distance,
        )
        compute_multi_region_distance_fn = compute_multi_region_distance

    # Build cost matrix [n_segs x n_players]
    cost_matrix = np.full((n_segs, n_players), fill_value=1.0)

    for i, seg in enumerate(team_segments):
        for j, profile in enumerate(profiles):
            cost_matrix[i, j] = _compute_assignment_cost(
                seg, profile,
                seg_histograms[i], seg_multi_descs[i], profile_multi_descs[j],
                compute_multi_region_distance_fn,
            )

    # Greedy assignment: each segment picks its lowest-cost profile
    blocked: set[tuple[int, int]] = set()  # (seg_idx, player_idx) blocked
    final_assignment: dict[int, int] = {}

    for round_num in range(MAX_REASSIGNMENT_ROUNDS):
        round_assignment: dict[int, int] = {}
        for i in range(n_segs):
            best_j = -1
            best_cost = MAX_ASSIGNMENT_COST
            for j in range(n_players):
                if (i, j) in blocked:
                    continue
                if cost_matrix[i, j] < best_cost:
                    best_cost = cost_matrix[i, j]
                    best_j = j
            if best_j >= 0:
                round_assignment[i] = best_j

        # Check for temporal overlap within same player
        conflict_found = False
        conflicted_segs: set[int] = set()
        player_segments: dict[int, list[int]] = defaultdict(list)
        for si, pi in round_assignment.items():
            player_segments[pi].append(si)

        for pi, seg_indices in player_segments.items():
            if len(seg_indices) < 2:
                continue
            for a_idx in range(len(seg_indices)):
                for b_idx in range(a_idx + 1, len(seg_indices)):
                    sa = team_segments[seg_indices[a_idx]]
                    sb = team_segments[seg_indices[b_idx]]
                    if (
                        sa.start_frame <= sb.end_frame
                        and sb.start_frame <= sa.end_frame
                    ):
                        # Block the higher-cost assignment for next round
                        cost_a = cost_matrix[seg_indices[a_idx], pi]
                        cost_b = cost_matrix[seg_indices[b_idx], pi]
                        if cost_a > cost_b:
                            blocked.add((seg_indices[a_idx], pi))
                        else:
                            blocked.add((seg_indices[b_idx], pi))
                        # Track both overlapping segments as ambiguous
                        conflicted_segs.add(seg_indices[a_idx])
                        conflicted_segs.add(seg_indices[b_idx])
                        conflict_found = True

        if not conflict_found:
            final_assignment = round_assignment
            break

        # Save non-conflicting assignments as fallback. Segments involved
        # in temporal overlap are excluded (ambiguous), but the rest are safe.
        safe_assignment = {
            si: pi for si, pi in round_assignment.items()
            if si not in conflicted_segs
        }
        if safe_assignment:
            final_assignment = safe_assignment
    else:
        if blocked:
            logger.debug(
                f"Conflict resolution exhausted {MAX_REASSIGNMENT_ROUNDS} rounds "
                f"with {len(blocked)} blocked pairs; "
                f"applying {len(final_assignment)} non-conflicting assignments"
            )

    # Apply remapping — build lookup for targeted updates
    remap_keys: dict[tuple[int, int], int] = {}  # (track_id, frame) -> new_id
    for seg_idx, player_idx in final_assignment.items():
        seg = team_segments[seg_idx]
        canonical_id = profiles[player_idx].player_id
        if seg.track_id == canonical_id:
            continue
        for p in seg.positions:
            remap_keys[(seg.track_id, p.frame_number)] = canonical_id

    if not remap_keys:
        return 0

    remapped = 0
    for p in positions:
        new_id = remap_keys.get((p.track_id, p.frame_number))
        if new_id is not None:
            p.track_id = new_id
            remapped += 1

    if remapped:
        logger.info(
            f"Global identity: remapped {remapped} positions across "
            f"{len(final_assignment)} segments "
            f"({n_players} profiles, {n_segs} segments)"
        )

    return remapped
