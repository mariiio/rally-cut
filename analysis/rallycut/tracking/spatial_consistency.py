"""Unified spatial consistency enforcement for player tracks.

Three detection modes:
1. Instantaneous jump: displacement >0.25 within ≤3 frames (Kalman prediction artifacts).
2. Sliding-window drift: displacement >0.15 over a ~0.5s window (smooth BoT-SORT ID swaps
   where the Kalman filter gradually follows the wrong player over 5-15 frames).
3. Gap re-ID collision: when a track reappears after a detection gap at the position of
   another recently-active track, the re-ID matched the wrong person. Based on mutual
   exclusion — two players can't occupy the same position. Drops the post-gap detections.

Modes 1-2 split the track at the violation point and rekey appearance stores.
Mode 3 drops post-gap detections (sets track_id = -1) — irreversible.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from rallycut.tracking.appearance_descriptor import AppearanceDescriptorStore
from rallycut.tracking.color_repair import ColorHistogramStore
from rallycut.tracking.player_tracker import PlayerPosition

logger = logging.getLogger(__name__)

# Jump detection defaults (mode 1: instantaneous)
DEFAULT_JUMP_MAX_DISPLACEMENT = 0.25
DEFAULT_JUMP_MAX_FRAME_GAP = 3

# Drift detection defaults (mode 2: sliding window)
DEFAULT_DRIFT_WINDOW_SECONDS = 0.5  # Time-based window
DEFAULT_DRIFT_MAX_DISPLACEMENT = 0.15  # Max displacement over window
DEFAULT_DRIFT_MIN_WINDOW_FRAMES = 6  # Don't check windows shorter than this

# Gap re-ID collision defaults (mode 3)
DEFAULT_GAP_MIN_FRAMES = 30  # ~1s at 30fps — gaps shorter than this are just YOLO misses
DEFAULT_GAP_MIN_DISPLACEMENT = 0.04  # ~0.6m — below this the player barely moved (noise)
DEFAULT_GAP_PROXIMITY_THRESHOLD = 0.08  # ~1.3m — consistent with overlap position gate in tracklet_link
DEFAULT_GAP_LOOKBACK_FRAMES = 45  # How recently the other track must have been active


@dataclass
class SpatialConsistencyResult:
    """Result of spatial consistency enforcement."""

    jump_splits: int = 0
    drift_splits: int = 0
    gap_collisions: int = 0
    jump_details: list[tuple[int, int, int]] = field(default_factory=list)
    """(old_id, new_id, split_frame) for each jump split."""
    drift_details: list[tuple[int, int, int, float]] = field(default_factory=list)
    """(old_id, new_id, split_frame, window_displacement) for each drift split."""
    gap_collision_details: list[tuple[int, int, int, int, float]] = field(
        default_factory=list
    )
    """(track_id, other_track_id, split_frame, gap_frames, proximity) for each collision."""


def enforce_spatial_consistency(
    positions: list[PlayerPosition],
    color_store: ColorHistogramStore | None = None,
    appearance_store: AppearanceDescriptorStore | None = None,
    jump_max_displacement: float = DEFAULT_JUMP_MAX_DISPLACEMENT,
    jump_max_frame_gap: int = DEFAULT_JUMP_MAX_FRAME_GAP,
    video_fps: float = 30.0,
    drift_detection: bool = True,
    drift_max_displacement: float = DEFAULT_DRIFT_MAX_DISPLACEMENT,
    gap_collision_detection: bool = False,
    gap_min_frames: int = DEFAULT_GAP_MIN_FRAMES,
) -> tuple[list[PlayerPosition], SpatialConsistencyResult]:
    """Enforce spatial consistency on all tracks.

    Mode 1: Scans each track for large displacement jumps and splits.
    Mode 2: Scans each track for sustained anomalous velocity over a sliding
    window and splits at the fastest step (identity transition point).
    Mode 3: Gap re-ID collision detection. When a track reappears after a
    detection gap and another track was recently at that position, the re-ID
    matched the wrong person. Drops post-gap detections (track_id = -1).

    Args:
        positions: All player positions (modified in place).
        color_store: Optional histogram store (rekeyed on splits).
        appearance_store: Optional appearance store (rekeyed on splits).
        jump_max_displacement: Max displacement for jump split trigger.
        jump_max_frame_gap: Max frame gap for jump detection.
        video_fps: Video frame rate (for drift window sizing).
        drift_detection: Enable sliding-window drift detection.
        drift_max_displacement: Max displacement over the drift window.
        gap_collision_detection: Enable gap re-ID collision detection.
        gap_min_frames: Minimum gap length (frames) to check.

    Returns:
        Tuple of (positions, SpatialConsistencyResult).
    """
    if not positions:
        return positions, SpatialConsistencyResult()

    result = SpatialConsistencyResult()

    # Find next available track ID
    max_id = max((p.track_id for p in positions if p.track_id >= 0), default=0)
    next_id = max_id + 1

    # Mode 1: Instantaneous jump detection
    tracks = _group_by_track(positions)

    for track_id in sorted(tracks.keys()):
        track_pos = sorted(tracks[track_id], key=lambda p: p.frame_number)

        i = 1
        while i < len(track_pos):
            prev = track_pos[i - 1]
            curr = track_pos[i]

            frame_gap = curr.frame_number - prev.frame_number
            if frame_gap <= 0 or frame_gap > jump_max_frame_gap:
                i += 1
                continue

            dx = curr.x - prev.x
            dy = curr.y - prev.y
            dist = (dx * dx + dy * dy) ** 0.5

            if dist > jump_max_displacement:
                new_id = next_id
                next_id += 1

                split_frame = curr.frame_number
                for j in range(i, len(track_pos)):
                    track_pos[j].track_id = new_id

                if color_store is not None:
                    color_store.rekey(track_id, new_id, split_frame)
                if appearance_store is not None:
                    appearance_store.rekey(track_id, new_id, split_frame)

                result.jump_splits += 1
                result.jump_details.append((track_id, new_id, split_frame))

                logger.info(
                    f"Jump split: track {track_id} at frame {split_frame} "
                    f"(jump={dist:.3f} in {frame_gap} frames) -> track {new_id}"
                )
                break

            i += 1

    # Mode 3: Gap re-ID collision detection
    # When a track reappears after a gap at the position of another
    # recently-active track, the re-ID matched the wrong person.
    # Drop the post-gap detections — they belong to the other player.
    if gap_collision_detection:
        tracks = _group_by_track(positions)

        # Build last-known-position index for all tracks:
        # For each track, its last detection position and frame.
        track_endpoints: dict[int, tuple[int, float, float]] = {}
        for tid, tps in tracks.items():
            sorted_tps = sorted(tps, key=lambda p: p.frame_number)
            last = sorted_tps[-1]
            track_endpoints[tid] = (last.frame_number, last.x, last.y)

        for track_id in sorted(tracks.keys()):
            track_pos = sorted(tracks[track_id], key=lambda p: p.frame_number)
            if len(track_pos) < 2:
                continue

            _detect_gap_collisions(
                track_id,
                track_pos,
                tracks,
                track_endpoints,
                result,
                gap_min_frames,
                video_fps,
                color_store,
                appearance_store,
            )

    # Mode 2: Sliding-window drift detection
    if drift_detection:
        window_frames = max(
            int(round(DEFAULT_DRIFT_WINDOW_SECONDS * video_fps)),
            DEFAULT_DRIFT_MIN_WINDOW_FRAMES,
        )

        # Re-group after mode 1 may have split tracks
        tracks = _group_by_track(positions)

        for track_id in sorted(tracks.keys()):
            track_pos = sorted(tracks[track_id], key=lambda p: p.frame_number)

            if len(track_pos) < DEFAULT_DRIFT_MIN_WINDOW_FRAMES:
                continue

            split_info = _find_drift_violation(
                track_pos, window_frames, drift_max_displacement,
            )
            if split_info is None:
                continue

            split_idx, window_disp = split_info
            new_id = next_id
            next_id += 1

            split_frame = track_pos[split_idx].frame_number
            for j in range(split_idx, len(track_pos)):
                track_pos[j].track_id = new_id

            if color_store is not None:
                color_store.rekey(track_id, new_id, split_frame)
            if appearance_store is not None:
                appearance_store.rekey(track_id, new_id, split_frame)

            result.drift_splits += 1
            result.drift_details.append(
                (track_id, new_id, split_frame, window_disp)
            )

            logger.info(
                f"Drift split: track {track_id} at frame {split_frame} "
                f"(drift={window_disp:.3f} over {window_frames} frames) "
                f"-> track {new_id}"
            )

    total = result.jump_splits + result.drift_splits + result.gap_collisions
    if total:
        logger.info(
            f"Spatial consistency: {result.jump_splits} jump splits, "
            f"{result.drift_splits} drift splits, "
            f"{result.gap_collisions} gap collisions"
        )

    return positions, result


def _detect_gap_collisions(
    track_id: int,
    track_pos: list[PlayerPosition],
    all_tracks: dict[int, list[PlayerPosition]],
    track_endpoints: dict[int, tuple[int, float, float]],
    result: SpatialConsistencyResult,
    gap_min_frames: int,
    video_fps: float,
    color_store: ColorHistogramStore | None,
    appearance_store: AppearanceDescriptorStore | None,
) -> None:
    """Check one track for gap re-ID collisions and drop offending detections.

    For each detection gap in the track, checks whether the reappearance
    position collides with another recently-active track. If so, the re-ID
    was wrong — drops all post-gap detections.
    """
    for i in range(1, len(track_pos)):
        gap = track_pos[i].frame_number - track_pos[i - 1].frame_number - 1
        if gap < gap_min_frames:
            continue

        p_pre = track_pos[i - 1]  # Last position before gap
        p_post = track_pos[i]     # First position after gap

        # Check displacement — skip if negligible (just measurement noise)
        dx = p_post.x - p_pre.x
        dy = p_post.y - p_pre.y
        displacement = (dx * dx + dy * dy) ** 0.5

        if displacement < DEFAULT_GAP_MIN_DISPLACEMENT:
            continue

        # Check if another track was recently active near the reappearance
        reappear_frame = p_post.frame_number
        collision_track = _find_collision_track(
            track_id,
            reappear_frame,
            p_post.x,
            p_post.y,
            all_tracks,
            track_endpoints,
        )

        if collision_track is None:
            continue

        other_id, proximity = collision_track

        # Collision confirmed: drop all post-gap detections
        n_dropped = 0
        for j in range(i, len(track_pos)):
            track_pos[j].track_id = -1
            n_dropped += 1

        # Clean up appearance stores — remove data for dropped frames
        if color_store is not None:
            keys = [k for k in color_store._histograms
                    if k[0] == track_id and k[1] >= reappear_frame]
            for k in keys:
                del color_store._histograms[k]
            if keys and not any(
                tid == track_id for tid, _ in color_store._histograms
            ):
                color_store._track_ids.discard(track_id)
        if appearance_store is not None:
            keys = [k for k in appearance_store._descriptors
                    if k[0] == track_id and k[1] >= reappear_frame]
            for k in keys:
                del appearance_store._descriptors[k]

        result.gap_collisions += 1
        result.gap_collision_details.append(
            (track_id, other_id, reappear_frame, gap, proximity)
        )

        logger.info(
            f"Gap collision: track {track_id} reappeared at frame "
            f"{reappear_frame} near track {other_id} "
            f"(gap={gap}f/{gap / video_fps:.1f}s, "
            f"disp={displacement:.3f}, prox={proximity:.3f}) "
            f"-> dropped {n_dropped} post-gap detections"
        )
        break  # Only process first gap collision per track


def _find_collision_track(
    track_id: int,
    reappear_frame: int,
    reappear_x: float,
    reappear_y: float,
    all_tracks: dict[int, list[PlayerPosition]],
    track_endpoints: dict[int, tuple[int, float, float]],
) -> tuple[int, float] | None:
    """Find a recently-active track near the reappearance position.

    Returns (other_track_id, proximity_distance) or None if no collision.
    """
    best_id: int | None = None
    best_dist = float("inf")

    for other_id, (last_frame, last_x, last_y) in track_endpoints.items():
        if other_id == track_id:
            continue

        # Other track must have been active recently (within lookback)
        # and must have ended BEFORE or near the reappearance
        frames_since = reappear_frame - last_frame
        if frames_since < -3 or frames_since > DEFAULT_GAP_LOOKBACK_FRAMES:
            # -3: allow small overlap (track might still be active)
            continue

        # If the other track is still active well past the reappearance,
        # use its position at the reappearance frame instead of its endpoint
        other_x, other_y = last_x, last_y
        if last_frame > reappear_frame + 10:
            # Track extends past reappearance — find its position there
            other_at_frame = _position_at_frame(
                all_tracks[other_id], reappear_frame,
            )
            if other_at_frame is not None:
                other_x, other_y = other_at_frame

        dx = reappear_x - other_x
        dy = reappear_y - other_y
        dist = (dx * dx + dy * dy) ** 0.5

        if dist < DEFAULT_GAP_PROXIMITY_THRESHOLD and dist < best_dist:
            best_dist = dist
            best_id = other_id

    if best_id is not None:
        return best_id, best_dist
    return None


def _position_at_frame(
    track_pos: list[PlayerPosition],
    frame: int,
    tolerance: int = 5,
) -> tuple[float, float] | None:
    """Get a track's position at or near a specific frame."""
    best: PlayerPosition | None = None
    best_gap = tolerance + 1
    for p in track_pos:
        gap = abs(p.frame_number - frame)
        if gap < best_gap:
            best_gap = gap
            best = p
    if best is not None and best_gap <= tolerance:
        return best.x, best.y
    return None



def _find_drift_violation(
    track_pos: list[PlayerPosition],
    window_frames: int,
    max_displacement: float,
) -> tuple[int, float] | None:
    """Find the first sliding-window drift violation in a track.

    Slides a window of `window_frames` across the track. If the
    displacement between the first and last position in the window exceeds
    `max_displacement`, returns the split index (frame with maximum
    per-step displacement within the window — the identity transition
    point) and the window displacement.

    Returns:
        (split_index, window_displacement) or None if no violation.
    """
    n = len(track_pos)

    for start in range(n):
        # Find end of window by frame number
        start_frame = track_pos[start].frame_number
        end_idx = start
        while (
            end_idx + 1 < n
            and track_pos[end_idx + 1].frame_number - start_frame <= window_frames
        ):
            end_idx += 1

        if end_idx - start < DEFAULT_DRIFT_MIN_WINDOW_FRAMES - 1:
            continue

        # Displacement across the window
        dx = track_pos[end_idx].x - track_pos[start].x
        dy = track_pos[end_idx].y - track_pos[start].y
        window_disp = (dx * dx + dy * dy) ** 0.5

        if window_disp <= max_displacement:
            continue

        # Find the frame with maximum per-step displacement (transition point)
        max_step_dist = 0.0
        max_step_idx = start + 1
        for k in range(start + 1, end_idx + 1):
            sdx = track_pos[k].x - track_pos[k - 1].x
            sdy = track_pos[k].y - track_pos[k - 1].y
            step_dist = (sdx * sdx + sdy * sdy) ** 0.5
            if step_dist > max_step_dist:
                max_step_dist = step_dist
                max_step_idx = k

        return max_step_idx, window_disp

    return None


def _group_by_track(
    positions: list[PlayerPosition],
) -> dict[int, list[PlayerPosition]]:
    """Group positions by track_id, excluding negative IDs."""
    tracks: dict[int, list[PlayerPosition]] = defaultdict(list)
    for p in positions:
        if p.track_id >= 0:
            tracks[p.track_id].append(p)
    return dict(tracks)
