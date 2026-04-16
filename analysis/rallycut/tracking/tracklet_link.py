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
4. Swap-optimize: check if moving any fragment to a different group
   reduces total intra-group cost (catches greedy's suboptimal merges)
5. Stop when distance exceeds threshold

For beach volleyball (4 players), this is dramatically simpler than
general MOT linking because we know the target track count.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

import cv2
import numpy as np

from rallycut.tracking.player_tracker import PlayerPosition

if TYPE_CHECKING:
    from rallycut.court.calibration import CourtCalibrator
    from rallycut.tracking.appearance_descriptor import AppearanceDescriptorStore
    from rallycut.tracking.color_repair import ColorHistogramStore

logger = logging.getLogger(__name__)

# Linking parameters
DEFAULT_MERGE_DISTANCE_THRESHOLD = 0.45  # Max Bhattacharyya distance to merge
DEFAULT_MAX_SPATIAL_DISPLACEMENT = 0.25  # Max normalized position jump — matches enforce_spatial_consistency
DEFAULT_MIN_TRACK_FRAMES = 5  # Minimum frames in a track to participate in linking
# Velocity gate: reject merges where the junction between fragments creates
# displacement exceeding this threshold within the window. Prevents merging
# same-appearance fragments from different players at different court positions.
DEFAULT_MAX_MERGE_VELOCITY = 0.20  # Max displacement across junction (image-plane fallback)
# Court-plane threshold in metres: ~2.5m over 10 frames ≈ 7.5 m/s (Olympic
# sprint cap). Removes image-plane scale-dependence (0.10 normalized means ~1m
# at the net but >3m at the baseline — a single image-plane constant cannot
# serve both regimes). Env override `RALLYCUT_MAX_MERGE_VELOCITY_METERS` is
# supported for threshold sweeps without code changes.
DEFAULT_MAX_MERGE_VELOCITY_METERS = float(
    os.environ.get("RALLYCUT_MAX_MERGE_VELOCITY_METERS", 2.5)
)
DEFAULT_MERGE_VELOCITY_WINDOW = 10  # Frames around junction to check


# Margin (metres) outside the court lines within which we still trust the
# homography. Players chasing balls into the dead zone sit slightly outside
# the court rectangle; pure homography extrapolation is reliable within a
# couple metres. Beyond that (e.g. the far-sideline "audience" region just
# above the trapezoid's top edge) projections drift toward the vanishing
# point and a single image-y pixel can span tens of metres of projected y.
COURT_FOOT_TRUST_MARGIN_M = 2.0


def _foot_court_coord(
    calibrator: CourtCalibrator, p: PlayerPosition,
) -> tuple[float, float]:
    # Foot = bbox bottom (``y + height/2``) — the only image-y that lies on
    # the court floor for a standing player.
    fx, fy = p.x, p.y + p.height / 2
    return calibrator.image_to_court((fx, fy), 0, 0)


def _court_displacement_meters(
    calibrator: CourtCalibrator,
    pa: PlayerPosition,
    pb: PlayerPosition,
) -> float:
    """Euclidean court-plane distance in metres between two player footprints.

    Uses the bbox foot location (``y + height/2``) rather than the bbox
    center, because court calibration trapezoids cover only the court floor.
    Center-y projections for standing players fall above the trapezoid's
    far baseline and extrapolate to wildly out-of-court coordinates; foot-y
    projections land in the calibrated region where the homography is
    reliable.

    Caller is responsible for checking both points land inside the trusted
    region via ``_both_feet_in_court`` — outside that region the homography
    extrapolates and the return value is not physically meaningful.
    """
    cax, cay = _foot_court_coord(calibrator, pa)
    cbx, cby = _foot_court_coord(calibrator, pb)
    dx, dy = cbx - cax, cby - cay
    return float((dx * dx + dy * dy) ** 0.5)


def _both_feet_in_court(
    calibrator: CourtCalibrator,
    pa: PlayerPosition,
    pb: PlayerPosition,
    margin_m: float = COURT_FOOT_TRUST_MARGIN_M,
) -> bool:
    """Both foot projections land within court + margin (trusted region).

    Short-height detections (partial players, shallow bboxes) yield little
    foot-adjustment and their centers stay above the trapezoid — those
    must fall back to image-plane.
    """
    return (
        calibrator.is_point_in_court(
            _foot_court_coord(calibrator, pa), margin=margin_m,
        )
        and calibrator.is_point_in_court(
            _foot_court_coord(calibrator, pb), margin=margin_m,
        )
    )


def _would_create_velocity_anomaly(
    positions: list[PlayerPosition],
    tid_a: int,
    tid_b: int,
    max_displacement_image: float = DEFAULT_MAX_MERGE_VELOCITY,
    window: int = DEFAULT_MERGE_VELOCITY_WINDOW,
    *,
    calibrator: CourtCalibrator | None = None,
    max_displacement_meters: float = DEFAULT_MAX_MERGE_VELOCITY_METERS,
) -> bool:
    """Check if merging two tracks would create impossible velocity at the junction.

    Examines positions from both tracks near the temporal boundary where
    they meet. If any pair of positions within `window` frames has
    displacement exceeding the threshold, the merge is rejected.

    When a calibrated ``calibrator`` is passed, thresholds are applied in
    court-plane metres (``max_displacement_meters``), removing the
    scale-dependence of normalized image-plane distances. Otherwise falls
    back to image-plane Euclidean distance (``max_displacement_image``) —
    the original behaviour, preserved bit-exact.

    Args:
        positions: All positions (with original track IDs).
        tid_a: First track ID.
        tid_b: Second track ID.
        max_displacement_image: Image-plane fallback threshold (normalized
            0-1 Euclidean distance). Used when calibrator is None / uncalibrated.
        window: Number of frames around the junction to check.
        calibrator: Optional court calibrator. When calibrated, positions are
            projected to court metres and distances compared to
            ``max_displacement_meters``.
        max_displacement_meters: Court-plane threshold (metres).

    Returns:
        True if the merge would create a velocity anomaly.
    """
    # Get positions for each track sorted by frame
    pos_a = sorted(
        [p for p in positions if p.track_id == tid_a],
        key=lambda p: p.frame_number,
    )
    pos_b = sorted(
        [p for p in positions if p.track_id == tid_b],
        key=lambda p: p.frame_number,
    )

    if not pos_a or not pos_b:
        return False

    court_gate_enabled = (
        calibrator is not None
        and calibrator.is_calibrated
        and os.environ.get("RALLYCUT_DISABLE_COURT_VELOCITY_GATE", "0") != "1"
    )

    def _image_dist(pa: PlayerPosition, pb: PlayerPosition) -> float:
        dx = pb.x - pa.x
        dy = pb.y - pa.y
        return float((dx * dx + dy * dy) ** 0.5)

    additive_mode = os.environ.get("RALLYCUT_COURT_GATE_ADDITIVE", "0") == "1"

    def _exceeds_gate(pa: PlayerPosition, pb: PlayerPosition) -> bool:
        """True if pair displacement exceeds the applicable velocity gate.

        Per-pair regime: court-plane when both foot projections land in the
        trusted (court + margin) region; image-plane fallback when either
        foot lies in the homography extrapolation regime (above the
        trapezoid's far baseline, common for short-height detections).

        Additive mode (``RALLYCUT_COURT_GATE_ADDITIVE=1``): image-plane
        always active; court-plane adds rejections on top when points are
        in-court. Strictly stricter than image-plane alone.
        """
        if additive_mode and _image_dist(pa, pb) > max_displacement_image:
            return True
        if (
            court_gate_enabled
            and calibrator is not None
            and _both_feet_in_court(calibrator, pa, pb)
        ):
            return _court_displacement_meters(
                calibrator, pa, pb,
            ) > max_displacement_meters
        return _image_dist(pa, pb) > max_displacement_image

    # Determine temporal order: which track ends first?
    if pos_a[-1].frame_number <= pos_b[0].frame_number:
        earlier, later = pos_a, pos_b
    elif pos_b[-1].frame_number <= pos_a[0].frame_number:
        earlier, later = pos_b, pos_a
    else:
        # Overlapping — check positions near the overlap boundary
        overlap_start = max(pos_a[0].frame_number, pos_b[0].frame_number)
        tail = [p for p in pos_a if abs(p.frame_number - overlap_start) <= window]
        head = [p for p in pos_b if abs(p.frame_number - overlap_start) <= window]
        for pa in tail:
            for pb in head:
                frame_gap = abs(pb.frame_number - pa.frame_number)
                if frame_gap > window or frame_gap == 0:
                    continue
                if _exceeds_gate(pa, pb):
                    return True
        return False

    # Check endpoint displacement regardless of gap size.
    # Two fragments of the same player can't be >threshold apart
    # at their nearest temporal boundary.
    end_pos = earlier[-1]
    start_pos = later[0]
    if _exceeds_gate(end_pos, start_pos):
        return True

    # Also check sliding window near the junction for short-gap merges
    tail = [p for p in earlier if p.frame_number >= earlier[-1].frame_number - window]
    head = [p for p in later if p.frame_number <= later[0].frame_number + window]
    for pa in tail:
        for pb in head:
            frame_gap = abs(pb.frame_number - pa.frame_number)
            if frame_gap > window or frame_gap == 0:
                continue
            if _exceeds_gate(pa, pb):
                return True

    return False


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
    max_allowed_overlap: int = 0,
) -> bool:
    """Check if two tracks have more than max_allowed_overlap frames in common.

    Args:
        frames_a: Frame numbers for track A.
        frames_b: Frame numbers for track B.
        max_allowed_overlap: Number of overlapping frames to tolerate.
            0 = strict (any overlap blocks merge).
            >0 = allow brief handoff overlaps (e.g., Kalman ghost + new detection).
    """
    overlap = frames_a & frames_b
    return len(overlap) > max_allowed_overlap


def _greedy_merge(
    hist_ids: list[int],
    n: int,
    dist_matrix: np.ndarray,
    tracks: dict[int, dict],
    avg_hists: dict[int, np.ndarray],
    positions: list[PlayerPosition],
    merge_distance_threshold: float,
    max_overlap_frames: int,
    target_track_count: int | None,
    calibrator: CourtCalibrator | None = None,
) -> tuple[dict[int, int], int]:
    """Greedy hierarchical merging: always merge the closest valid pair.

    Returns (id_mapping, num_merges).
    """
    id_mapping: dict[int, int] = {}
    num_merges = 0
    active_mask = np.ones(n, dtype=bool)
    current_track_count = n

    while True:
        if target_track_count is not None and current_track_count <= target_track_count:
            break

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
            break

        tid_i, tid_j = hist_ids[best_i], hist_ids[best_j]
        canon_i = id_mapping.get(tid_i, tid_i)
        canon_j = id_mapping.get(tid_j, tid_j)

        if tracks[canon_j]["count"] > tracks[canon_i]["count"]:
            canonical, merged = canon_j, canon_i
            keep_idx, remove_idx = best_j, best_i
        else:
            canonical, merged = canon_i, canon_j
            keep_idx, remove_idx = best_i, best_j

        if _tracks_overlap_temporally(
            tracks[canonical]["frames"], tracks[merged]["frames"],
            max_allowed_overlap=max_overlap_frames,
        ):
            dist_matrix[best_i, best_j] = 1.0
            dist_matrix[best_j, best_i] = 1.0
            continue

        if _would_create_velocity_anomaly(
            positions, canonical, merged, calibrator=calibrator,
        ):
            logger.debug(
                f"Tracklet link: blocked merge {merged} -> {canonical} "
                f"(velocity anomaly at junction)"
            )
            dist_matrix[best_i, best_j] = 1.0
            dist_matrix[best_j, best_i] = 1.0
            continue

        id_mapping[merged] = canonical
        tracks[canonical]["frames"] |= tracks[merged]["frames"]
        tracks[canonical]["count"] += tracks[merged]["count"]
        tracks[canonical]["first_frame"] = min(
            tracks[canonical]["first_frame"], tracks[merged]["first_frame"]
        )
        tracks[canonical]["last_frame"] = max(
            tracks[canonical]["last_frame"], tracks[merged]["last_frame"]
        )

        hist_c = avg_hists.get(canonical)
        hist_m = avg_hists.get(merged)
        if hist_c is not None and hist_m is not None:
            count_c = tracks[canonical]["count"] - tracks[merged]["count"]
            count_m = tracks[merged]["count"]
            total = count_c + count_m
            avg_hists[canonical] = (hist_c * count_c + hist_m * count_m) / total

        active_mask[remove_idx] = False

        for k in range(n):
            if not active_mask[k] or k == keep_idx:
                continue
            tid_k = hist_ids[k]
            canon_k = id_mapping.get(tid_k, tid_k)

            if _tracks_overlap_temporally(
                tracks[canonical]["frames"], tracks[canon_k]["frames"],
                max_allowed_overlap=max_overlap_frames,
            ):
                dist_matrix[keep_idx, k] = 1.0
                dist_matrix[k, keep_idx] = 1.0
            elif (
                avg_hists.get(canonical) is not None
                and avg_hists.get(canon_k) is not None
            ):
                new_dist = _bhattacharyya_distance(
                    avg_hists[canonical], avg_hists[canon_k]
                )
                dist_matrix[keep_idx, k] = new_dist
                dist_matrix[k, keep_idx] = new_dist

        num_merges += 1
        current_track_count -= 1

        logger.debug(
            f"Tracklet link: merged track {merged} -> {canonical} "
            f"(bhatt={min_dist:.3f}, tracks remaining={current_track_count})"
        )

    return id_mapping, num_merges


def _swap_optimize(
    id_mapping: dict[int, int],
    hist_ids: list[int],
    n: int,
    dist_matrix: np.ndarray,
    tracks: dict[int, dict],
    positions: list[PlayerPosition],
    merge_distance_threshold: float,
    max_overlap_frames: int,
    calibrator: CourtCalibrator | None = None,
) -> dict[int, int]:
    """Improve greedy partition by reassigning fragments to better groups.

    After greedy merging, check if any non-canonical fragment has a lower
    distance to another group's canonical than to its current canonical.
    Uses the ORIGINAL (pre-greedy) distance matrix and track data to
    avoid artifacts from greedy's incremental histogram blending.

    The cost metric is distance-to-canonical (not pairwise sum), which
    correctly handles chain-connected groups where intermediate fragments
    may have high original pairwise distance to distant group members.

    Returns improved id_mapping (may be unchanged if no swap helps).
    """

    def _resolve(tid: int) -> int:
        visited: set[int] = set()
        while tid in id_mapping and tid not in visited:
            visited.add(tid)
            tid = id_mapping[tid]
        return tid

    # Build groups: canonical_tid -> set of member indices
    groups: dict[int, set[int]] = {}
    for i in range(n):
        canon = _resolve(hist_ids[i])
        groups.setdefault(canon, set()).add(i)

    # Find canonical index (longest track) in each group
    def _canon_idx(members: set[int]) -> int:
        return max(members, key=lambda i: tracks[hist_ids[i]]["count"])

    canon_indices: dict[int, int] = {
        canon_tid: _canon_idx(members)
        for canon_tid, members in groups.items()
    }

    def _can_join(frag_idx: int, members: set[int]) -> bool:
        """Check if fragment can join a group (all pairwise constraints)."""
        frag_tid = hist_ids[frag_idx]
        for m_idx in members:
            if float(dist_matrix[frag_idx, m_idx]) >= merge_distance_threshold:
                return False
            m_tid = hist_ids[m_idx]
            if _tracks_overlap_temporally(
                tracks[frag_tid]["frames"], tracks[m_tid]["frames"],
                max_allowed_overlap=max_overlap_frames,
            ):
                return False
            if _would_create_velocity_anomaly(
                positions, frag_tid, m_tid, calibrator=calibrator,
            ):
                return False
        return True

    total_swaps = 0
    max_passes = 10
    for _ in range(max_passes):
        swapped = False

        for frag_i in range(n):
            frag_tid = hist_ids[frag_i]
            src_canon_tid = _resolve(frag_tid)
            if src_canon_tid not in groups:
                continue
            src_members = groups[src_canon_tid]

            # Don't move the canonical track itself
            if frag_i == canon_indices.get(src_canon_tid):
                continue
            # Don't empty a group
            if len(src_members) <= 1:
                continue

            # Current distance to source canonical
            src_canon_i = canon_indices[src_canon_tid]
            current_dist = float(dist_matrix[frag_i, src_canon_i])

            # Find best destination group
            best_dst_tid: int | None = None
            best_dist = current_dist

            for dst_canon_tid, dst_members in groups.items():
                if dst_canon_tid == src_canon_tid:
                    continue
                dst_canon_i = canon_indices[dst_canon_tid]
                candidate_dist = float(dist_matrix[frag_i, dst_canon_i])

                if candidate_dist >= best_dist:
                    continue

                # Verify all constraints against destination members
                if not _can_join(frag_i, dst_members):
                    continue

                best_dst_tid = dst_canon_tid
                best_dist = candidate_dist

            if best_dst_tid is not None:
                # Move fragment from source to destination
                groups[src_canon_tid].discard(frag_i)
                groups[best_dst_tid].add(frag_i)
                id_mapping[frag_tid] = best_dst_tid
                total_swaps += 1
                swapped = True
                break  # Restart scan (group composition changed)

        if not swapped:
            break

    if total_swaps > 0:
        logger.info(
            f"Tracklet linking: {total_swaps} swap(s) improved partition"
        )

    return id_mapping


def _greedy_then_optimize(
    hist_ids: list[int],
    n: int,
    dist_matrix: np.ndarray,
    tracks: dict[int, dict],
    avg_hists: dict[int, np.ndarray],
    positions: list[PlayerPosition],
    merge_distance_threshold: float,
    max_overlap_frames: int,
    target_track_count: int,
    calibrator: CourtCalibrator | None = None,
) -> tuple[dict[int, int], int]:
    """Run greedy merging, then optimize with pairwise swaps.

    Greedy mutates dist_matrix, tracks, and avg_hists during merging
    (blends histograms, accumulates frame sets, blocks pairs). The swap
    optimizer needs the ORIGINAL values to make unbiased decisions, so
    we snapshot them before greedy runs.
    """
    # Snapshot originals — greedy will mutate these in place
    orig_dist = dist_matrix.copy()
    orig_tracks = {
        tid: {**info, "frames": set(info["frames"])}
        for tid, info in tracks.items()
    }

    id_mapping, num_merges = _greedy_merge(
        hist_ids, n, dist_matrix, tracks, avg_hists, positions,
        merge_distance_threshold, max_overlap_frames, target_track_count,
        calibrator=calibrator,
    )

    if num_merges > 0:
        id_mapping = _swap_optimize(
            id_mapping, hist_ids, n, orig_dist, orig_tracks,
            positions, merge_distance_threshold, max_overlap_frames,
            calibrator=calibrator,
        )

    return id_mapping, num_merges


def link_tracklets_by_appearance(
    positions: list[PlayerPosition],
    color_store: ColorHistogramStore,
    merge_distance_threshold: float = DEFAULT_MERGE_DISTANCE_THRESHOLD,
    max_spatial_displacement: float = DEFAULT_MAX_SPATIAL_DISPLACEMENT,
    min_track_frames: int = DEFAULT_MIN_TRACK_FRAMES,
    target_track_count: int | None = 4,
    team_assignments: dict[int, int] | None = None,
    appearance_store: AppearanceDescriptorStore | None = None,
    max_overlap_frames: int = 15,
    calibrator: CourtCalibrator | None = None,
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
        team_assignments: Optional team classification (track_id -> team).
            Blocks cross-team merges (a player's fragment should only
            reconnect with tracks from the same team).
        max_overlap_frames: Maximum frames of temporal overlap to allow
            between tracklets being merged. During net crossings, BoT-SORT
            may briefly have both a dying Kalman prediction and a new
            detection active. Default 15 (~0.5s at 30fps).

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

            # Temporal overlap check: block merges with large overlap
            if _tracks_overlap_temporally(
                tracks[tid_i]["frames"], tracks[tid_j]["frames"],
                max_allowed_overlap=max_overlap_frames,
            ):
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

            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    # Greedy merge + swap optimization
    if target_track_count is not None and target_track_count > 0:
        id_mapping, num_merges = _greedy_then_optimize(
            hist_ids,
            n,
            dist_matrix,
            tracks,
            avg_hists,
            positions,
            merge_distance_threshold,
            max_overlap_frames,
            target_track_count,
            calibrator=calibrator,
        )
    else:
        id_mapping, num_merges = _greedy_merge(
            hist_ids,
            n,
            dist_matrix,
            tracks,
            avg_hists,
            positions,
            merge_distance_threshold,
            max_overlap_frames,
            target_track_count,
            calibrator=calibrator,
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

    # Build resolved mapping for all merged IDs
    resolved_mapping = {tid: resolve(tid) for tid in id_mapping}

    # Apply remapping to positions
    remapped = 0
    for p in positions:
        canonical = resolve(p.track_id)
        if canonical != p.track_id:
            p.track_id = canonical
            remapped += 1

    # Resolve any overlapping frames created by merging tracks with
    # brief temporal overlap. Keep the higher-confidence detection per
    # frame, drop others by setting track_id = -1.
    frame_best: dict[tuple[int, int], tuple[int, float]] = {}  # (track_id, frame) -> (idx, conf)
    for i, p in enumerate(positions):
        if p.track_id < 0:
            continue
        key = (p.track_id, p.frame_number)
        if key not in frame_best or p.confidence > frame_best[key][1]:
            frame_best[key] = (i, p.confidence)

    # Mark duplicates
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
    color_store.remap_ids(resolved_mapping)
    if appearance_store is not None:
        appearance_store.remap_ids(resolved_mapping)

    remaining_tracks = len({p.track_id for p in positions if p.track_id >= 0})
    logger.info(
        f"Tracklet linking: {num_merges} merges, "
        f"remapped {remapped} positions, "
        f"{remaining_tracks} tracks remaining"
    )

    return positions, num_merges

# --- Spatial re-link: reconnect fragments that are trivially the same player ---

# Maximum gap (frames) for spatial-only re-linking. Fragments separated by
# more than this require appearance confirmation.
SPATIAL_RELINK_MAX_GAP = 5
# Maximum endpoint distance for spatial-only re-linking. At this distance
# the player hasn't moved — it's the same person, no appearance needed.
SPATIAL_RELINK_MAX_DISTANCE = 0.05


def relink_spatial_splits(
    positions: list[PlayerPosition],
    color_store: ColorHistogramStore,
    appearance_store: AppearanceDescriptorStore | None = None,
    max_gap: int = SPATIAL_RELINK_MAX_GAP,
    max_distance: float = SPATIAL_RELINK_MAX_DISTANCE,
) -> tuple[list[PlayerPosition], int]:
    """Reconnect fragments that are trivially the same player by position.

    Color splitting (Step 0b) can split a continuous track into fragments
    at nearly the same position. When the gap is tiny (≤5 frames) and
    endpoint distance is negligible (≤0.05), spatial continuity alone
    proves they're the same player — no appearance matching needed.

    Called after color splitting (Step 0b2): prevents greedy appearance
    merges from stealing obvious spatial matches.

    Args:
        positions: Player positions (modified in place).
        color_store: Updated with remap after merges.
        appearance_store: Updated with remap after merges (optional).
        max_gap: Maximum frame gap for spatial re-linking.
        max_distance: Maximum endpoint distance (normalized).

    Returns:
        Tuple of (modified positions, number of re-links performed).
    """
    if not positions:
        return positions, 0

    tracks = _compute_track_summary(positions)
    if len(tracks) <= 1:
        return positions, 0

    logger.info(
        f"Spatial re-link: scanning {len(tracks)} tracks "
        f"(max_gap={max_gap}, max_dist={max_distance})"
    )

    # Build sorted list of (track_id, first_frame, last_frame, first_pos, last_pos)
    track_list = sorted(tracks.items(), key=lambda x: x[1]["first_frame"])

    id_mapping: dict[int, int] = {}
    # Track canonical frame ranges to prevent overlap merges
    canonical_ranges: dict[int, tuple[int, int]] = {}

    for idx, (tid, info) in enumerate(track_list):
        if tid in id_mapping:
            continue

        # Look for the next fragment that starts shortly after this one ends
        for jdx in range(idx + 1, len(track_list)):
            next_tid, next_info = track_list[jdx]
            if next_tid in id_mapping:
                continue

            # Resolve canonical for current track
            canon = tid
            while canon in id_mapping:
                canon = id_mapping[canon]

            gap = next_info["first_frame"] - tracks[canon]["last_frame"]
            if gap < 0:
                continue  # Overlapping — skip
            if gap > max_gap:
                break  # Sorted by start frame — no more candidates

            # Endpoint distance
            end_pos = tracks[canon]["last_pos"]
            start_pos = next_info["first_pos"]
            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            dist = (dx * dx + dy * dy) ** 0.5

            if dist > max_distance:
                continue

            # Check canonical range doesn't overlap with next track
            if canon in canonical_ranges:
                c_first, c_last = canonical_ranges[canon]
                if next_info["first_frame"] <= c_last:
                    continue

            # Merge: map next_tid -> canonical
            id_mapping[next_tid] = canon

            # Update canonical's track info
            tracks[canon]["frames"] |= next_info["frames"]
            tracks[canon]["count"] += next_info["count"]
            if next_info["last_frame"] > tracks[canon]["last_frame"]:
                tracks[canon]["last_pos"] = next_info["last_pos"]
            tracks[canon]["last_frame"] = max(
                tracks[canon]["last_frame"], next_info["last_frame"]
            )

            # Update canonical range
            canonical_ranges[canon] = (
                tracks[canon]["first_frame"],
                tracks[canon]["last_frame"],
            )

            logger.debug(
                f"Spatial re-link: {next_tid} -> {canon} "
                f"(gap={gap}f, dist={dist:.4f})"
            )

    if not id_mapping:
        return positions, 0

    # Resolve transitive chains
    def resolve(tid: int) -> int:
        visited: set[int] = set()
        while tid in id_mapping and tid not in visited:
            visited.add(tid)
            tid = id_mapping[tid]
        return tid

    # Apply remapping
    remapped = 0
    for p in positions:
        canonical = resolve(p.track_id)
        if canonical != p.track_id:
            p.track_id = canonical
            remapped += 1

    resolved_mapping = {tid: resolve(tid) for tid in id_mapping}
    color_store.remap_ids(resolved_mapping)
    if appearance_store is not None:
        appearance_store.remap_ids(resolved_mapping)

    num_relinks = len(id_mapping)
    logger.info(
        f"Spatial re-link: {num_relinks} re-links, "
        f"remapped {remapped} positions"
    )

    return positions, num_relinks


# ---------------------------------------------------------------------------
# Relaxed fragment linking for primary tracks (Step 0b3)
# ---------------------------------------------------------------------------

# Thresholds for linking non-primary fragments into primary tracks.
# Much more relaxed than SPATIAL_RELINK because the primary anchor is a
# known player — no risk of merging background or spectator tracks.
PRIMARY_RELINK_MAX_GAP = 50  # ~1.7s at 30fps
PRIMARY_RELINK_MAX_DISTANCE = 0.08  # normalized, ~100px at 1280w
PRIMARY_RELINK_MAX_APPEARANCE = 0.20  # Bhattacharyya gate — reject if too dissimilar


def relink_primary_fragments(
    positions: list[PlayerPosition],
    primary_track_ids: list[int] | set[int],
    color_store: ColorHistogramStore,
    appearance_store: AppearanceDescriptorStore | None = None,
    max_gap: int = PRIMARY_RELINK_MAX_GAP,
    max_distance: float = PRIMARY_RELINK_MAX_DISTANCE,
    max_appearance: float = PRIMARY_RELINK_MAX_APPEARANCE,
) -> tuple[list[PlayerPosition], list[int], int]:
    """Link non-primary fragments into primary tracks by spatial proximity.

    After primary track selection, some fragments of a primary player may
    exist as separate non-primary tracks (too short/weak to be selected
    individually). This pass links them into the nearest primary track
    using relaxed spatial thresholds.

    Only forward merges are allowed: the fragment must start AFTER the
    primary's last frame (continuation of a dropped track). Backward
    merges are blocked because a fragment predating a primary is more
    likely a separate player than a predecessor.

    Args:
        positions: Player positions (modified in place).
        primary_track_ids: Set of track IDs identified as primary players.
        color_store: Color histogram store for appearance gating/tiebreaking.
        appearance_store: Updated with remap after merges (optional).
        max_gap: Maximum frame gap for re-linking.
        max_distance: Maximum endpoint distance (normalized).
        max_appearance: Maximum Bhattacharyya distance (0=identical, 1=different).
            Fragments with appearance distance above this are rejected.

    Returns:
        Tuple of (modified positions, updated primary_track_ids, num re-links).
    """
    primary_set = set(primary_track_ids)
    if not positions or not primary_set:
        return positions, sorted(primary_set), 0

    tracks = _compute_track_summary(positions)
    non_primary_ids = [tid for tid in tracks if tid not in primary_set]

    if not non_primary_ids:
        return positions, sorted(primary_set), 0

    logger.info(
        f"Primary re-link: {len(non_primary_ids)} non-primary fragments, "
        f"{len(primary_set)} primary tracks "
        f"(max_gap={max_gap}, max_dist={max_distance})"
    )

    # Pre-compute average histograms for appearance gating/tiebreaking
    avg_hists: dict[int, np.ndarray | None] = {}
    for tid in tracks:
        avg_hists[tid] = _compute_average_histogram(tid, color_store)

    # For each non-primary fragment, find the best primary track to merge into.
    # A fragment can link to a primary track if:
    #  1. It fills a gap (no frame overlap with the primary)
    #  2. The temporal gap at the join point is ≤ max_gap
    #  3. The spatial distance at the join point is ≤ max_distance
    #  4. Primary coverage is already full during the fragment's lifetime
    id_mapping: dict[int, int] = {}

    # Work on a mutable copy of primary track info for updating after merges
    primary_info: dict[int, dict] = {
        tid: dict(tracks[tid]) for tid in primary_set if tid in tracks
    }
    # Deep-copy frame sets so mutations don't affect originals
    for info in primary_info.values():
        info["frames"] = set(info["frames"])

    for np_tid in non_primary_ids:
        np_info = tracks[np_tid]

        best_primary: int | None = None
        best_dist: float = float("inf")
        best_appearance: float = float("inf")

        for p_tid, p_info in primary_info.items():
            # Check frame overlap
            if p_info["frames"] & np_info["frames"]:
                continue

            # Forward-only linkage: the fragment must come AFTER the
            # primary's last frame (continuation of a dropped track).
            # Backward merges (fragment precedes target) are blocked
            # because a pre-existing fragment is more likely a separate
            # player than a predecessor of a primary that starts later.
            gap = np_info["first_frame"] - p_info["last_frame"]
            if gap < 0:
                continue  # Fragment precedes or overlaps primary
            end_pos = p_info["last_pos"]
            start_pos = np_info["first_pos"]

            if gap > max_gap:
                continue

            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            dist = (dx * dx + dy * dy) ** 0.5

            if dist > max_distance:
                continue

            # Appearance gate + tiebreaker
            appearance_dist = float("inf")
            p_hist = avg_hists.get(p_tid)
            np_hist = avg_hists.get(np_tid)
            if p_hist is not None and np_hist is not None:
                appearance_dist = _bhattacharyya_distance(
                    p_hist.astype(np.float32),
                    np_hist.astype(np.float32),
                )

            # Reject if appearance is too dissimilar
            if appearance_dist > max_appearance:
                continue

            # Prefer better appearance match; use spatial distance to break
            # ties.  All candidates already passed spatial/temporal gates,
            # so appearance is the stronger discriminator.
            if appearance_dist < best_appearance or (
                appearance_dist == best_appearance and dist < best_dist
            ):
                best_primary = p_tid
                best_dist = dist
                best_appearance = appearance_dist

        if best_primary is not None:
            id_mapping[np_tid] = best_primary

            # Update primary track info to include merged fragment
            p_info = primary_info[best_primary]
            p_info["frames"] |= np_info["frames"]
            p_info["count"] += np_info["count"]
            if np_info["first_frame"] < p_info["first_frame"]:
                p_info["first_frame"] = np_info["first_frame"]
                p_info["first_pos"] = np_info["first_pos"]
            if np_info["last_frame"] > p_info["last_frame"]:
                p_info["last_frame"] = np_info["last_frame"]
                p_info["last_pos"] = np_info["last_pos"]

            logger.debug(
                f"Primary re-link: T{np_tid} -> T{best_primary} "
                f"(dist={best_dist:.4f}, appearance={best_appearance:.4f})"
            )

    if not id_mapping:
        return positions, sorted(primary_set), 0

    # Apply remapping (no transitive chains — all map directly to primaries)
    remapped = 0
    for p in positions:
        new_tid = id_mapping.get(p.track_id)
        if new_tid is not None:
            p.track_id = new_tid
            remapped += 1

    color_store.remap_ids(id_mapping)
    if appearance_store is not None:
        appearance_store.remap_ids(id_mapping)

    num_relinks = len(id_mapping)
    logger.info(
        f"Primary re-link: {num_relinks} re-links, "
        f"remapped {remapped} positions"
    )

    # Primary set unchanged — non-primary fragments merge INTO primaries
    return positions, sorted(primary_set), num_relinks
