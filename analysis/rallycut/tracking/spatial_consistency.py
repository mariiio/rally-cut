"""Unified spatial consistency enforcement for player tracks.

Two detection modes:
1. Instantaneous jump: displacement >0.25 within ≤3 frames (Kalman prediction artifacts).
2. Sliding-window drift: displacement >0.15 over a ~0.5s window (smooth BoT-SORT ID swaps
   where the Kalman filter gradually follows the wrong player over 5-15 frames).

Both modes split the track at the violation point and rekey appearance stores.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from rallycut.tracking.appearance_descriptor import AppearanceDescriptorStore
from rallycut.tracking.color_repair import ColorHistogramStore, LearnedEmbeddingStore
from rallycut.tracking.player_tracker import PlayerPosition

logger = logging.getLogger(__name__)

# Jump detection defaults (mode 1: instantaneous)
DEFAULT_JUMP_MAX_DISPLACEMENT = 0.25
DEFAULT_JUMP_MAX_FRAME_GAP = 3

# Drift detection defaults (mode 2: sliding window)
DEFAULT_DRIFT_WINDOW_SECONDS = 0.5  # Time-based window
DEFAULT_DRIFT_MAX_DISPLACEMENT = 0.20  # Max displacement over window
DEFAULT_DRIFT_MIN_WINDOW_FRAMES = 6  # Don't check windows shorter than this


@dataclass
class SpatialConsistencyResult:
    """Result of spatial consistency enforcement."""

    jump_splits: int = 0
    drift_splits: int = 0
    jump_details: list[tuple[int, int, int]] = field(default_factory=list)
    """(old_id, new_id, split_frame) for each jump split."""
    drift_details: list[tuple[int, int, int, float]] = field(default_factory=list)
    """(old_id, new_id, split_frame, window_displacement) for each drift split."""


def enforce_spatial_consistency(
    positions: list[PlayerPosition],
    color_store: ColorHistogramStore | None = None,
    appearance_store: AppearanceDescriptorStore | None = None,
    jump_max_displacement: float = DEFAULT_JUMP_MAX_DISPLACEMENT,
    jump_max_frame_gap: int = DEFAULT_JUMP_MAX_FRAME_GAP,
    video_fps: float = 30.0,
    drift_detection: bool = True,
    drift_max_displacement: float = DEFAULT_DRIFT_MAX_DISPLACEMENT,
    learned_store: LearnedEmbeddingStore | None = None,
) -> tuple[list[PlayerPosition], SpatialConsistencyResult]:
    """Enforce spatial consistency on all tracks.

    Mode 1: Scans each track for large displacement jumps and splits.
    Mode 2: Scans each track for sustained anomalous velocity over a sliding
    window and splits at the fastest step (identity transition point).

    Args:
        positions: All player positions (modified in place).
        color_store: Optional histogram store (rekeyed on splits).
        appearance_store: Optional appearance store (rekeyed on splits).
        jump_max_displacement: Max displacement for jump split trigger.
        jump_max_frame_gap: Max frame gap for jump detection.
        video_fps: Video frame rate (for drift window sizing).
        drift_detection: Enable sliding-window drift detection.
        drift_max_displacement: Max displacement over the drift window.

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
                if learned_store is not None:
                    learned_store.rekey(track_id, new_id, split_frame)

                result.jump_splits += 1
                result.jump_details.append((track_id, new_id, split_frame))

                logger.info(
                    f"Jump split: track {track_id} at frame {split_frame} "
                    f"(jump={dist:.3f} in {frame_gap} frames) -> track {new_id}"
                )
                break

            i += 1

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
            if learned_store is not None:
                learned_store.rekey(track_id, new_id, split_frame)

            result.drift_splits += 1
            result.drift_details.append(
                (track_id, new_id, split_frame, window_disp)
            )

            logger.info(
                f"Drift split: track {track_id} at frame {split_frame} "
                f"(drift={window_disp:.3f} over {window_frames} frames) "
                f"-> track {new_id}"
            )

    total = result.jump_splits + result.drift_splits
    if total:
        logger.info(
            f"Spatial consistency: {result.jump_splits} jump splits, "
            f"{result.drift_splits} drift splits"
        )

    return positions, result


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
