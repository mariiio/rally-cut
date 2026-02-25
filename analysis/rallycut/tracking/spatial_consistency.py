"""Unified spatial consistency enforcement for player tracks.

Detects large displacement jumps (>0.25) within ≤3 frames and splits the track
at the jump point.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

from rallycut.tracking.appearance_descriptor import AppearanceDescriptorStore
from rallycut.tracking.color_repair import ColorHistogramStore
from rallycut.tracking.player_tracker import PlayerPosition

logger = logging.getLogger(__name__)

# Jump detection defaults
DEFAULT_JUMP_MAX_DISPLACEMENT = 0.25
DEFAULT_JUMP_MAX_FRAME_GAP = 3


@dataclass
class SpatialConsistencyResult:
    """Result of spatial consistency enforcement."""

    jump_splits: int = 0
    jump_details: list[tuple[int, int, int]] = field(default_factory=list)
    """(old_id, new_id, split_frame) for each jump split."""


def enforce_spatial_consistency(
    positions: list[PlayerPosition],
    color_store: ColorHistogramStore | None = None,
    appearance_store: AppearanceDescriptorStore | None = None,
    jump_max_displacement: float = DEFAULT_JUMP_MAX_DISPLACEMENT,
    jump_max_frame_gap: int = DEFAULT_JUMP_MAX_FRAME_GAP,
) -> tuple[list[PlayerPosition], SpatialConsistencyResult]:
    """Enforce spatial consistency on all tracks.

    Scans each track for large displacement jumps and splits at the jump point.
    Handles store rekeying internally.

    Args:
        positions: All player positions (modified in place).
        color_store: Optional histogram store (rekeyed on splits).
        appearance_store: Optional appearance store (rekeyed on splits).
        jump_max_displacement: Max displacement for jump split trigger.
        jump_max_frame_gap: Max frame gap for jump detection.

    Returns:
        Tuple of (positions, SpatialConsistencyResult).
    """
    if not positions:
        return positions, SpatialConsistencyResult()

    result = SpatialConsistencyResult()

    # Find next available track ID
    max_id = max((p.track_id for p in positions if p.track_id >= 0), default=0)
    next_id = max_id + 1

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
                # Reassign positions from split point onward
                for j in range(i, len(track_pos)):
                    track_pos[j].track_id = new_id

                # Rekey stores
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

    if result.jump_splits:
        logger.info(
            f"Spatial consistency: {result.jump_splits} jump splits"
        )

    return positions, result


def _group_by_track(
    positions: list[PlayerPosition],
) -> dict[int, list[PlayerPosition]]:
    """Group positions by track_id, excluding negative IDs."""
    tracks: dict[int, list[PlayerPosition]] = defaultdict(list)
    for p in positions:
        if p.track_id >= 0:
            tracks[p.track_id].append(p)
    return dict(tracks)
