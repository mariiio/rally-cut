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
3. Greedily merge closest pairs (temporal non-overlap + spatial constraints)
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
    from rallycut.tracking.color_repair import ColorHistogramStore

logger = logging.getLogger(__name__)

# Linking parameters
DEFAULT_MERGE_DISTANCE_THRESHOLD = 0.45  # Max Bhattacharyya distance to merge
DEFAULT_MAX_SPATIAL_DISPLACEMENT = 0.30  # Max normalized position jump between fragments
DEFAULT_MIN_TRACK_FRAMES = 5  # Minimum frames in a track to participate in linking


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


def _tracks_overlap_temporally(
    frames_a: set[int],
    frames_b: set[int],
) -> bool:
    """Check if two tracks have any frames in common."""
    return bool(frames_a & frames_b)


def link_tracklets_by_appearance(
    positions: list[PlayerPosition],
    color_store: ColorHistogramStore,
    merge_distance_threshold: float = DEFAULT_MERGE_DISTANCE_THRESHOLD,
    max_spatial_displacement: float = DEFAULT_MAX_SPATIAL_DISPLACEMENT,
    min_track_frames: int = DEFAULT_MIN_TRACK_FRAMES,
    target_track_count: int | None = 4,
) -> tuple[list[PlayerPosition], int]:
    """Link fragmented tracklets using appearance similarity.

    Uses color histogram Bhattacharyya distance to find tracklet pairs
    that likely belong to the same player, then merges them.

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

            # Temporal overlap check: can never merge overlapping tracks
            if _tracks_overlap_temporally(tracks[tid_i]["frames"], tracks[tid_j]["frames"]):
                dist_matrix[i, j] = 1.0
                dist_matrix[j, i] = 1.0
                continue

            # Spatial displacement check between fragment endpoints
            # Determine temporal order
            if tracks[tid_i]["last_frame"] < tracks[tid_j]["first_frame"]:
                end_pos = tracks[tid_i]["last_pos"]
                start_pos = tracks[tid_j]["first_pos"]
            elif tracks[tid_j]["last_frame"] < tracks[tid_i]["first_frame"]:
                end_pos = tracks[tid_j]["last_pos"]
                start_pos = tracks[tid_i]["first_pos"]
            else:
                # Interleaved but non-overlapping (rare): skip spatial check
                end_pos = tracks[tid_i]["last_pos"]
                start_pos = tracks[tid_j]["first_pos"]

            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            spatial_dist = (dx * dx + dy * dy) ** 0.5

            if spatial_dist > max_spatial_displacement:
                dist_matrix[i, j] = 1.0
                dist_matrix[j, i] = 1.0
                continue

            # Appearance distance
            dist = _bhattacharyya_distance(avg_hists[tid_i], avg_hists[tid_j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

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

        # Verify no temporal overlap after resolving canonical IDs
        if _tracks_overlap_temporally(tracks[canonical]["frames"], tracks[merged]["frames"]):
            dist_matrix[best_i, best_j] = 1.0
            dist_matrix[best_j, best_i] = 1.0
            continue

        # Perform merge
        id_mapping[merged] = canonical
        tracks[canonical]["frames"] |= tracks[merged]["frames"]
        tracks[canonical]["count"] += tracks[merged]["count"]
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

            # Recheck temporal overlap
            if _tracks_overlap_temporally(tracks[canonical]["frames"], tracks[canon_k]["frames"]):
                dist_matrix[keep_idx, k] = 1.0
                dist_matrix[k, keep_idx] = 1.0
            elif avg_hists.get(canonical) is not None and avg_hists.get(canon_k) is not None:
                new_dist = _bhattacharyya_distance(avg_hists[canonical], avg_hists[canon_k])
                dist_matrix[keep_idx, k] = new_dist
                dist_matrix[k, keep_idx] = new_dist

        num_merges += 1
        current_track_count -= 1

        logger.debug(
            f"Tracklet link: merged track {merged} -> {canonical} "
            f"(bhatt={min_dist:.3f}, tracks remaining={current_track_count})"
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

    logger.info(
        f"Tracklet linking: {num_merges} merges, "
        f"remapped {remapped} positions, "
        f"{current_track_count} tracks remaining"
    )

    return positions, num_merges
