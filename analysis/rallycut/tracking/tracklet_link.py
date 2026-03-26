"""Appearance-based tracklet linking for player tracking.

Inspired by GTA-Link (Global Tracklet Association), this module reconnects
fragmented tracklets using appearance similarity from shorts color histograms.

After BoT-SORT tracking + splitting (jump splits, color splits), players
often get multiple track IDs across a rally. The existing stabilize_track_ids()
merges fragments using position proximity, but fails when:
- Spatial gap is too large (player moves during detection gap)
- Multiple candidate fragments exist at similar positions

This module adds appearance-based matching:
1. Compute average color histogram per track from ColorHistogramStore
2. Build pairwise Bhattacharyya distance matrix between all tracks
3. Greedily merge closest pairs (spatial + temporal + overlap constraints)
4. Stop when distance exceeds threshold

For beach volleyball (4 players), this is dramatically simpler than
general MOT linking because we know the target track count.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import cv2
import numpy as np

from rallycut.tracking.player_tracker import PlayerPosition

if TYPE_CHECKING:
    from rallycut.tracking.appearance_descriptor import AppearanceDescriptorStore
    from rallycut.tracking.color_repair import ColorHistogramStore

logger = logging.getLogger(__name__)

# Linking parameters
DEFAULT_MERGE_DISTANCE_THRESHOLD = 0.45  # Max Bhattacharyya distance to merge
DEFAULT_MAX_SPATIAL_DISPLACEMENT = 0.25  # Max normalized endpoint displacement
DEFAULT_MIN_TRACK_FRAMES = 5  # Minimum frames in a track to participate in linking
DEFAULT_MAX_TEMPORAL_GAP = 90  # Max frame gap between fragments for appearance-based merge
# Beyond this gap, appearance matching is unreliable (player may have moved,
# lighting changed, or a different same-team player is at the endpoint).
# Matches stabilize_track_ids.max_gap_frames for consistency.

# Overlap position gate: when two tracks coexist for a few frames (e.g.,
# dying Kalman ghost + new detection), they must be at nearly the same
# position to be the same player. A person can't be in two places at once.
OVERLAP_MAX_POSITION_DISTANCE = 0.08  # Max avg distance during overlap frames

# Spatial-temporal blending: weight for spatial-temporal component in merge scoring.
# When same-team players have near-identical appearance, spatial proximity breaks ties.
SPATIAL_BLEND_WEIGHT = 0.35  # 65% appearance + 35% spatial-temporal


def _compute_blended_distance(
    appearance_dist: float,
    spatial_dist: float,
    temporal_gap: int,
    max_spatial: float = DEFAULT_MAX_SPATIAL_DISPLACEMENT,
    max_gap: int = DEFAULT_MAX_TEMPORAL_GAP,
) -> float:
    """Blend appearance and spatial-temporal distances for merge ranking.

    Normalizes spatial distance and temporal gap to [0, 1], combines them
    equally, then blends with appearance distance. This ensures spatially
    closer fragments are preferred when appearance is ambiguous.
    """
    normalized_spatial = min(spatial_dist / max_spatial, 1.0) if max_spatial > 0 else 0.0
    normalized_gap = min(temporal_gap / max_gap, 1.0) if max_gap > 0 else 0.0
    spatial_temporal = 0.5 * normalized_spatial + 0.5 * normalized_gap
    return (1 - SPATIAL_BLEND_WEIGHT) * appearance_dist + SPATIAL_BLEND_WEIGHT * spatial_temporal


def _compute_track_summary(
    positions: list[PlayerPosition],
) -> dict[int, dict]:
    """Build per-track summary: frame range, start/end positions, frame set.

    Returns:
        Dict mapping track_id -> {first_frame, last_frame, first_pos, last_pos, frames, count}.
    """
    tracks: dict[int, dict] = {}
    for p in positions:
        if p.track_id < 0:
            continue
        if p.track_id not in tracks:
            tracks[p.track_id] = {
                "first_frame": p.frame_number,
                "last_frame": p.frame_number,
                "first_pos": (p.x, p.y),
                "last_pos": (p.x, p.y),
                "frames": set(),
                "count": 0,
            }
        info = tracks[p.track_id]
        if p.frame_number < info["first_frame"]:
            info["first_frame"] = p.frame_number
            info["first_pos"] = (p.x, p.y)
        if p.frame_number > info["last_frame"]:
            info["last_frame"] = p.frame_number
            info["last_pos"] = (p.x, p.y)
        info["frames"].add(p.frame_number)
        info["count"] += 1
    return tracks


def _compute_average_histogram(
    track_id: int,
    color_store: ColorHistogramStore,
) -> np.ndarray | None:
    """Compute the average color histogram for a track.

    Args:
        track_id: Track ID to compute average for.
        color_store: ColorHistogramStore with per-frame histograms.

    Returns:
        Normalized average histogram, or None if no histograms available.
    """
    histograms = color_store.get_track_histograms(track_id)
    if not histograms:
        return None

    avg_hist = np.zeros_like(histograms[0][1], dtype=np.float64)
    for _, hist in histograms:
        avg_hist += hist.astype(np.float64)

    avg_hist /= len(histograms)

    # Normalize for Bhattacharyya comparison
    total = avg_hist.sum()
    if total > 0:
        avg_hist /= total

    return avg_hist.astype(np.float32)


def _bhattacharyya_distance(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Compute Bhattacharyya distance between two histograms.

    Returns value in [0, 1] where 0 = identical, 1 = completely different.
    """
    return float(cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA))


def _compute_overlap_position_distance(
    positions: list[PlayerPosition],
    tid_a: int,
    tid_b: int,
    frames_a: set[int],
    frames_b: set[int],
    id_mapping: dict[int, int] | None = None,
) -> tuple[int, float]:
    """Compute temporal overlap and average position distance during overlap.

    A person cannot be in two places at once. If two tracks coexist and
    are far apart, they are different players regardless of appearance.

    Args:
        positions: All positions (track_ids may not yet be remapped).
        tid_a: First track ID (canonical, may have absorbed fragments).
        tid_b: Second track ID (canonical, may have absorbed fragments).
        frames_a: Frame set for track A (includes absorbed fragments).
        frames_b: Frame set for track B (includes absorbed fragments).
        id_mapping: Current merge mapping (merged_id -> canonical_id).
            Used to find positions whose track_id hasn't been remapped
            yet but logically belongs to tid_a or tid_b.

    Returns:
        (overlap_count, avg_distance): Number of overlapping frames and
        average position distance during those frames. Returns (0, 0.0)
        if no overlap.
    """
    overlap_frames = frames_a & frames_b
    if not overlap_frames:
        return 0, 0.0

    # Build set of track IDs that resolve to each canonical ID.
    # After merging T15→T20, positions still have track_id=15 but
    # frames_a (for canonical=20) includes T15's frames. We need to
    # match positions by their resolved canonical ID, not their raw ID.
    ids_a = {tid_a}
    ids_b = {tid_b}
    if id_mapping:
        for raw_id, canon_id in id_mapping.items():
            # Resolve transitively
            resolved = canon_id
            visited: set[int] = set()
            while resolved in id_mapping and resolved not in visited:
                visited.add(resolved)
                resolved = id_mapping[resolved]
            if resolved == tid_a:
                ids_a.add(raw_id)
            elif resolved == tid_b:
                ids_b.add(raw_id)

    # Build frame -> position lookup for both tracks
    pos_a: dict[int, tuple[float, float]] = {}
    pos_b: dict[int, tuple[float, float]] = {}
    for p in positions:
        if p.frame_number not in overlap_frames:
            continue
        if p.track_id in ids_a:
            pos_a[p.frame_number] = (p.x, p.y)
        elif p.track_id in ids_b:
            pos_b[p.frame_number] = (p.x, p.y)

    # Compute distance for frames where both have positions
    common_frames = set(pos_a.keys()) & set(pos_b.keys())
    if not common_frames:
        return len(overlap_frames), 0.0

    total_dist = 0.0
    for f in common_frames:
        dx = pos_a[f][0] - pos_b[f][0]
        dy = pos_a[f][1] - pos_b[f][1]
        total_dist += (dx * dx + dy * dy) ** 0.5

    avg_dist = total_dist / len(common_frames)
    return len(overlap_frames), avg_dist


def link_tracklets_by_appearance(
    positions: list[PlayerPosition],
    color_store: ColorHistogramStore,
    merge_distance_threshold: float = DEFAULT_MERGE_DISTANCE_THRESHOLD,
    max_spatial_displacement: float = DEFAULT_MAX_SPATIAL_DISPLACEMENT,
    min_track_frames: int = DEFAULT_MIN_TRACK_FRAMES,
    target_track_count: int | None = 4,
    team_assignments: dict[int, int] | None = None,
    appearance_store: AppearanceDescriptorStore | None = None,
) -> tuple[list[PlayerPosition], int]:
    """Link fragmented tracklets using appearance similarity.

    Uses color histogram Bhattacharyya distance to find tracklet pairs
    that likely belong to the same player, then merges them.

    Merge gates (all must pass):
    1. Cross-team block: fragments must be same team (if teams known).
    2. Overlap position gate: if tracks coexist, they must be at the
       same position (a person can't be in two places at once).
    3. Temporal gap: fragments must be within MAX_TEMPORAL_GAP frames.
    4. Spatial displacement: endpoint distance must be within threshold.

    Ranking uses a blended distance (appearance + spatial-temporal proximity)
    so that spatially closer fragments are preferred when appearance is
    ambiguous between same-team players.

    Args:
        positions: Player positions with track IDs (modified in place).
        color_store: ColorHistogramStore with per-frame histograms.
        merge_distance_threshold: Maximum Bhattacharyya distance to allow
            merging. Lower = stricter (fewer false merges). Range [0, 1].
        max_spatial_displacement: Maximum normalized position displacement
            between end of one fragment and start of the next.
        min_track_frames: Minimum frames in a track to participate in linking.
        target_track_count: Stop merging when this many tracks remain.
            Set to None for no target (merge until threshold exceeded).
        team_assignments: Optional team classification (track_id -> team).
            Blocks cross-team merges (a player's fragment should only
            reconnect with tracks from the same team).

    Returns:
        Tuple of (modified positions, number of merges performed).
    """
    if not positions:
        return positions, 0

    # Build track summaries
    tracks = _compute_track_summary(positions)

    # Filter to tracks with enough frames
    eligible_ids = [
        tid for tid, info in tracks.items()
        if info["count"] >= min_track_frames
    ]

    if len(eligible_ids) <= 1:
        return positions, 0

    # Already at or below target
    if target_track_count is not None and len(eligible_ids) <= target_track_count:
        logger.debug(
            f"Tracklet linking: {len(eligible_ids)} tracks already at/below "
            f"target {target_track_count}, skipping"
        )
        return positions, 0

    # Compute average histograms
    avg_hists: dict[int, np.ndarray] = {}
    for tid in eligible_ids:
        hist = _compute_average_histogram(tid, color_store)
        if hist is not None:
            avg_hists[tid] = hist

    if len(avg_hists) <= 1:
        logger.debug("Tracklet linking: insufficient histogram data, skipping")
        return positions, 0

    # Build pairwise distance matrix
    hist_ids = sorted(avg_hists.keys())
    n = len(hist_ids)
    dist_matrix = np.ones((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(i + 1, n):
            tid_i, tid_j = hist_ids[i], hist_ids[j]

            # Block cross-team merges
            if team_assignments:
                team_i = team_assignments.get(tid_i)
                team_j = team_assignments.get(tid_j)
                if team_i is not None and team_j is not None and team_i != team_j:
                    dist_matrix[i, j] = 1.0
                    dist_matrix[j, i] = 1.0
                    continue

            # Overlap position gate: if tracks coexist, check positions.
            # A person can't be in two places — if they're far apart
            # during overlapping frames, they're different players.
            overlap_count, overlap_dist = _compute_overlap_position_distance(
                positions, tid_i, tid_j,
                tracks[tid_i]["frames"], tracks[tid_j]["frames"],
            )
            if overlap_count > 0 and overlap_dist > OVERLAP_MAX_POSITION_DISTANCE:
                logger.debug(
                    f"Tracklet link: blocked {tid_i}+{tid_j}: "
                    f"overlap={overlap_count}f, dist={overlap_dist:.3f} "
                    f"(>{OVERLAP_MAX_POSITION_DISTANCE})"
                )
                dist_matrix[i, j] = 1.0
                dist_matrix[j, i] = 1.0
                continue

            # Temporal gap check: don't merge fragments separated by large
            # gaps. Over long gaps, appearance matching is unreliable —
            # same-team players with similar appearance can be confused.
            if tracks[tid_i]["last_frame"] < tracks[tid_j]["first_frame"]:
                temporal_gap = tracks[tid_j]["first_frame"] - tracks[tid_i]["last_frame"]
            elif tracks[tid_j]["last_frame"] < tracks[tid_i]["first_frame"]:
                temporal_gap = tracks[tid_i]["first_frame"] - tracks[tid_j]["last_frame"]
            else:
                temporal_gap = 0  # Overlapping

            if temporal_gap > DEFAULT_MAX_TEMPORAL_GAP:
                dist_matrix[i, j] = 1.0
                dist_matrix[j, i] = 1.0
                continue

            # Spatial displacement check between fragment endpoints
            if tracks[tid_i]["last_frame"] < tracks[tid_j]["first_frame"]:
                end_pos = tracks[tid_i]["last_pos"]
                start_pos = tracks[tid_j]["first_pos"]
            elif tracks[tid_j]["last_frame"] < tracks[tid_i]["first_frame"]:
                end_pos = tracks[tid_j]["last_pos"]
                start_pos = tracks[tid_i]["first_pos"]
            else:
                # Overlapping: use endpoints (overlap position gate above
                # already validated they're close enough)
                end_pos = tracks[tid_i]["last_pos"]
                start_pos = tracks[tid_j]["first_pos"]

            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            spatial_dist = (dx * dx + dy * dy) ** 0.5

            if spatial_dist > max_spatial_displacement:
                dist_matrix[i, j] = 1.0
                dist_matrix[j, i] = 1.0
                continue

            # Appearance distance (composite when multi-region available)
            dist = _bhattacharyya_distance(avg_hists[tid_i], avg_hists[tid_j])

            if appearance_store is not None and appearance_store.has_data():
                from rallycut.tracking.appearance_descriptor import (
                    compute_multi_region_distance,
                    compute_track_mean_descriptor,
                )

                desc_i = compute_track_mean_descriptor(appearance_store, tid_i)
                desc_j = compute_track_mean_descriptor(appearance_store, tid_j)
                if (
                    desc_i.shorts is not None
                    and desc_j.shorts is not None
                ):
                    multi_dist = compute_multi_region_distance(desc_i, desc_j)
                    # Blend: 40% shorts-only, 60% multi-region
                    dist = 0.4 * dist + 0.6 * multi_dist

            blended = _compute_blended_distance(
                dist, spatial_dist, temporal_gap, max_spatial_displacement,
            )
            dist_matrix[i, j] = blended
            dist_matrix[j, i] = blended

    # Greedy hierarchical merging
    id_mapping: dict[int, int] = {}  # merged_id -> canonical_id
    num_merges = 0
    active_mask = np.ones(n, dtype=bool)
    current_track_count = len(eligible_ids)

    while True:
        # Stop if at target count
        if target_track_count is not None and current_track_count <= target_track_count:
            break

        # Find minimum distance among active tracks
        min_dist = float("inf")
        best_i, best_j = -1, -1

        for i in range(n):
            if not active_mask[i]:
                continue
            for j in range(i + 1, n):
                if not active_mask[j]:
                    continue
                if dist_matrix[i, j] < min_dist:
                    min_dist = dist_matrix[i, j]
                    best_i, best_j = i, j

        if min_dist >= merge_distance_threshold:
            break  # No more eligible pairs

        tid_i, tid_j = hist_ids[best_i], hist_ids[best_j]

        # Resolve canonical IDs through any existing mappings
        canon_i = id_mapping.get(tid_i, tid_i)
        canon_j = id_mapping.get(tid_j, tid_j)

        # Keep the longer track as canonical
        if tracks[canon_j]["count"] > tracks[canon_i]["count"]:
            canonical, merged = canon_j, canon_i
            keep_idx, remove_idx = best_j, best_i
        else:
            canonical, merged = canon_i, canon_j
            keep_idx, remove_idx = best_i, best_j

        # Re-check overlap position gate after transitive merges.
        # Prior merges may have absorbed fragments that now overlap
        # with the merge candidate at a different position.
        overlap_count, overlap_dist = _compute_overlap_position_distance(
            positions, canonical, merged,
            tracks[canonical]["frames"], tracks[merged]["frames"],
            id_mapping=id_mapping,
        )
        if overlap_count > 0 and overlap_dist > OVERLAP_MAX_POSITION_DISTANCE:
            logger.debug(
                f"Tracklet link: blocked merge {merged} -> {canonical} "
                f"(post-transitive overlap={overlap_count}f, "
                f"dist={overlap_dist:.3f})"
            )
            dist_matrix[best_i, best_j] = 1.0
            dist_matrix[best_j, best_i] = 1.0
            continue

        # Perform merge
        id_mapping[merged] = canonical
        tracks[canonical]["frames"] |= tracks[merged]["frames"]
        tracks[canonical]["count"] += tracks[merged]["count"]

        # Update endpoint positions before frame range (needs original bounds)
        if tracks[merged]["first_frame"] < tracks[canonical]["first_frame"]:
            tracks[canonical]["first_pos"] = tracks[merged]["first_pos"]
        if tracks[merged]["last_frame"] > tracks[canonical]["last_frame"]:
            tracks[canonical]["last_pos"] = tracks[merged]["last_pos"]

        tracks[canonical]["first_frame"] = min(
            tracks[canonical]["first_frame"], tracks[merged]["first_frame"]
        )
        tracks[canonical]["last_frame"] = max(
            tracks[canonical]["last_frame"], tracks[merged]["last_frame"]
        )

        # Update average histogram for canonical track
        hist_c = avg_hists.get(canonical)
        hist_m = avg_hists.get(merged)
        if hist_c is not None and hist_m is not None:
            count_c = tracks[canonical]["count"] - tracks[merged]["count"]
            count_m = tracks[merged]["count"]
            total = count_c + count_m
            avg_hists[canonical] = (hist_c * count_c + hist_m * count_m) / total

        # Deactivate merged track
        active_mask[remove_idx] = False

        # Update distance matrix for canonical track
        for k in range(n):
            if not active_mask[k] or k == keep_idx:
                continue
            tid_k = hist_ids[k]
            canon_k = id_mapping.get(tid_k, tid_k)

            # Recheck overlap position gate with updated canonical track
            ov_count, ov_dist = _compute_overlap_position_distance(
                positions, canonical, canon_k,
                tracks[canonical]["frames"], tracks[canon_k]["frames"],
                id_mapping=id_mapping,
            )
            if ov_count > 0 and ov_dist > OVERLAP_MAX_POSITION_DISTANCE:
                dist_matrix[keep_idx, k] = 1.0
                dist_matrix[k, keep_idx] = 1.0
            elif avg_hists.get(canonical) is not None and avg_hists.get(canon_k) is not None:
                new_appearance = _bhattacharyya_distance(
                    avg_hists[canonical], avg_hists[canon_k]
                )

                # Compute spatial-temporal for blended distance
                if tracks[canonical]["last_frame"] < tracks[canon_k]["first_frame"]:
                    ep = tracks[canonical]["last_pos"]
                    sp = tracks[canon_k]["first_pos"]
                    gap = tracks[canon_k]["first_frame"] - tracks[canonical]["last_frame"]
                elif tracks[canon_k]["last_frame"] < tracks[canonical]["first_frame"]:
                    ep = tracks[canon_k]["last_pos"]
                    sp = tracks[canonical]["first_pos"]
                    gap = tracks[canonical]["first_frame"] - tracks[canon_k]["last_frame"]
                else:
                    ep = tracks[canonical]["last_pos"]
                    sp = tracks[canon_k]["first_pos"]
                    gap = 0

                dx = ep[0] - sp[0]
                dy = ep[1] - sp[1]
                sp_dist = (dx * dx + dy * dy) ** 0.5

                new_dist = _compute_blended_distance(
                    new_appearance, sp_dist, gap, max_spatial_displacement,
                )
                dist_matrix[keep_idx, k] = new_dist
                dist_matrix[k, keep_idx] = new_dist

        num_merges += 1
        current_track_count -= 1

        logger.debug(
            f"Tracklet link: merged track {merged} -> {canonical} "
            f"(dist={min_dist:.3f}, tracks remaining={current_track_count})"
        )

    if num_merges == 0:
        return positions, 0

    # Build full transitive mapping (handle chains: a -> b -> c)
    def resolve(tid: int) -> int:
        visited: set[int] = set()
        while tid in id_mapping and tid not in visited:
            visited.add(tid)
            tid = id_mapping[tid]
        return tid

    # Apply remapping to positions
    remapped = 0
    for p in positions:
        canonical = resolve(p.track_id)
        if canonical != p.track_id:
            p.track_id = canonical
            remapped += 1

    # Resolve any duplicate detections created by merging overlapping
    # tracks. Keep the higher-confidence detection per (track, frame).
    frame_best: dict[tuple[int, int], tuple[int, float]] = {}
    for i, p in enumerate(positions):
        if p.track_id < 0:
            continue
        key = (p.track_id, p.frame_number)
        if key not in frame_best or p.confidence > frame_best[key][1]:
            frame_best[key] = (i, p.confidence)

    n_dup_dropped = 0
    for i, p in enumerate(positions):
        if p.track_id < 0:
            continue
        key = (p.track_id, p.frame_number)
        if frame_best.get(key, (i, 0))[0] != i:
            positions[i].track_id = -1
            n_dup_dropped += 1

    if n_dup_dropped > 0:
        logger.info(
            f"Tracklet linking: dropped {n_dup_dropped} duplicate detections "
            f"from overlap resolution"
        )

    # Apply remapping to color_store and appearance_store
    resolved_mapping = {tid: resolve(tid) for tid in id_mapping}
    color_store.remap_ids(resolved_mapping)
    if appearance_store is not None:
        appearance_store.remap_ids(resolved_mapping)

    logger.info(
        f"Tracklet linking: {num_merges} merges, "
        f"remapped {remapped} positions, "
        f"{current_track_count} tracks remaining"
    )

    return positions, num_merges
