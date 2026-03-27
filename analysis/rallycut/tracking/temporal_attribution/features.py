"""Feature extraction for temporal contact attribution.

Extracts fixed-size trajectory windows around contact frames, with ball
interpolation and canonical player slot ordering.
"""

from __future__ import annotations

import math

import numpy as np

from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.player_tracker import PlayerPosition

# Default window: ±10 frames around contact = 21 frames total
DEFAULT_WINDOW_HALF = 10

# Maximum gap (frames) to interpolate ball positions across
MAX_INTERP_GAP = 5

# Minimum fraction of window frames with ball data to accept
MIN_BALL_COVERAGE = 0.5


def extract_attribution_window(
    contact_frame: int,
    ball_positions: list[BallPosition],
    player_positions: list[PlayerPosition],
    window_half: int = DEFAULT_WINDOW_HALF,
) -> tuple[np.ndarray, list[int]] | None:
    """Extract a trajectory window for temporal contact attribution.

    Builds a (window_size, 14) feature array per frame:
      - Columns 0-3: ball-to-player distance per slot
      - Columns 4-7: rate of change of distance per slot
      - Column 8: ball speed
      - Column 9: ball direction change (cosine-based)
      - Columns 10-13: player bbox height per slot (depth proxy)

    Players are assigned to canonical slots 0-3 sorted by distance to ball at the
    contact frame (slot 0 = closest). This normalizes variable track IDs across
    rallies.

    Args:
        contact_frame: Rally-relative frame number of the contact.
        ball_positions: Ball tracking positions for the rally.
        player_positions: Player tracking positions for the rally.
        window_half: Half-window size (total = 2 * window_half + 1).

    Returns:
        (features, canonical_track_ids) where features is (window_size, 14) and
        canonical_track_ids lists the 4 track IDs in slot order.
        Returns None if ball data is too sparse or fewer than 2 players found.
    """
    window_size = 2 * window_half + 1
    frame_start = contact_frame - window_half
    frame_end = contact_frame + window_half

    # --- Ball positions: build frame → (x, y) lookup ---
    ball_by_frame: dict[int, tuple[float, float]] = {}
    for bp in ball_positions:
        if frame_start <= bp.frame_number <= frame_end:
            ball_by_frame[bp.frame_number] = (bp.x, bp.y)

    # Interpolate gaps up to MAX_INTERP_GAP
    ball_xy = _interpolate_ball(ball_by_frame, frame_start, frame_end)

    # Check coverage
    n_valid = sum(1 for xy in ball_xy if xy is not None)
    if n_valid < window_size * MIN_BALL_COVERAGE:
        return None

    # --- Player positions: build per-track, per-frame lookup ---
    # Store (contact_x, contact_y, bbox_height) per track per frame
    # contact_y uses upper-quarter bbox y to match _find_nearest_player semantics
    player_by_track_frame: dict[int, dict[int, tuple[float, float, float]]] = {}
    for pp in player_positions:
        if frame_start <= pp.frame_number <= frame_end and pp.track_id >= 0:
            if pp.track_id not in player_by_track_frame:
                player_by_track_frame[pp.track_id] = {}
            player_by_track_frame[pp.track_id][pp.frame_number] = (
                pp.x,
                pp.y - pp.height * 0.25,
                pp.height,
            )

    if len(player_by_track_frame) < 2:
        return None

    # --- Canonical ordering: sort tracks by distance to ball at contact frame ---
    ball_at_contact = ball_xy[window_half]  # Center of window
    if ball_at_contact is None:
        # Try nearby frames
        for offset in [-1, 1, -2, 2, -3, 3]:
            idx = window_half + offset
            if 0 <= idx < window_size and ball_xy[idx] is not None:
                ball_at_contact = ball_xy[idx]
                break
    if ball_at_contact is None:
        return None

    bx, by = ball_at_contact
    track_distances: list[tuple[int, float]] = []
    for tid, frames_dict in player_by_track_frame.items():
        # Find position closest to contact frame
        best_pos = _nearest_frame_position3(frames_dict, contact_frame, max_gap=5)
        if best_pos is not None:
            px, py, _h = best_pos
            dist = math.sqrt((bx - px) ** 2 + (by - py) ** 2)
            track_distances.append((tid, dist))

    if len(track_distances) < 2:
        return None

    # Sort by distance, take up to 4
    track_distances.sort(key=lambda td: td[1])
    canonical_tids = [tid for tid, _ in track_distances[:4]]
    num_players = len(canonical_tids)

    # --- Build feature array (distance + depth, position-invariant) ---
    # Columns 0-3: distance from ball to each player slot
    # Columns 4-7: rate of change of distance (delta_dist per frame)
    # Column 8: ball speed (magnitude of velocity)
    # Column 9: ball direction change (cosine-based, 0=same dir, 2=reversal)
    # Columns 10-13: player bbox height per slot (depth proxy: larger=closer to camera)
    n_features = 14
    features = np.zeros((window_size, n_features), dtype=np.float32)

    # First pass: compute distances and heights
    distances = np.full((window_size, 4), 0.5, dtype=np.float32)  # default 0.5
    heights = np.zeros((window_size, 4), dtype=np.float32)

    for i in range(window_size):
        frame = frame_start + i
        ball_cur = ball_xy[i]

        for slot_idx in range(min(num_players, 4)):
            tid = canonical_tids[slot_idx]
            pos = _get_player_position3(
                player_by_track_frame[tid], frame, contact_frame
            )
            if pos is not None:
                px, py, ph = pos
                heights[i, slot_idx] = ph
                if ball_cur is not None:
                    cbx, cby = ball_cur
                    distances[i, slot_idx] = math.sqrt(
                        (cbx - px) ** 2 + (cby - py) ** 2
                    )

    # Columns 0-3: distances
    features[:, :4] = distances

    # Columns 4-7: rate of change of distance (delta per frame)
    features[1:, 4:8] = distances[1:] - distances[:-1]

    # Columns 8-9: ball speed and direction change
    for i in range(1, window_size):
        ball_cur = ball_xy[i]
        ball_prev = ball_xy[i - 1]
        if ball_cur is not None and ball_prev is not None:
            dx = ball_cur[0] - ball_prev[0]
            dy = ball_cur[1] - ball_prev[1]
            features[i, 8] = math.sqrt(dx * dx + dy * dy)  # speed

            if i >= 2:
                ball_prev2 = ball_xy[i - 2]
                if ball_prev2 is not None:
                    dx0 = ball_prev[0] - ball_prev2[0]
                    dy0 = ball_prev[1] - ball_prev2[1]
                    dot = dx * dx0 + dy * dy0
                    mag0 = math.sqrt(dx0 * dx0 + dy0 * dy0)
                    mag1 = math.sqrt(dx * dx + dy * dy)
                    if mag0 > 1e-6 and mag1 > 1e-6:
                        cos_angle = max(-1.0, min(1.0, dot / (mag0 * mag1)))
                        features[i, 9] = 1.0 - cos_angle

    # Columns 10-13: player bbox height (depth proxy)
    features[:, 10:14] = heights

    # Pad canonical_tids to exactly 4 entries
    while len(canonical_tids) < 4:
        canonical_tids.append(-1)

    return features, canonical_tids


def _interpolate_ball(
    ball_by_frame: dict[int, tuple[float, float]],
    frame_start: int,
    frame_end: int,
) -> list[tuple[float, float] | None]:
    """Build per-frame ball positions with linear interpolation for short gaps."""
    window_size = frame_end - frame_start + 1
    result: list[tuple[float, float] | None] = [None] * window_size

    # Fill known positions
    for f, xy in ball_by_frame.items():
        idx = f - frame_start
        if 0 <= idx < window_size:
            result[idx] = xy

    # Interpolate gaps up to MAX_INTERP_GAP
    i = 0
    while i < window_size:
        if result[i] is not None:
            i += 1
            continue

        # Find gap boundaries
        gap_start = i
        while i < window_size and result[i] is None:
            i += 1
        gap_end = i  # exclusive

        gap_len = gap_end - gap_start
        if gap_len > MAX_INTERP_GAP:
            continue

        # Need valid positions on both sides to interpolate
        left = result[gap_start - 1] if gap_start > 0 else None
        right = result[gap_end] if gap_end < window_size else None

        if left is not None and right is not None:
            for j in range(gap_start, gap_end):
                t = (j - gap_start + 1) / (gap_len + 1)
                result[j] = (
                    left[0] + t * (right[0] - left[0]),
                    left[1] + t * (right[1] - left[1]),
                )

    return result


def _nearest_frame_position3(
    frames_dict: dict[int, tuple[float, float, float]],
    target_frame: int,
    max_gap: int = 5,
) -> tuple[float, float, float] | None:
    """Get (x, y, height) at target_frame or nearest frame within max_gap."""
    if target_frame in frames_dict:
        return frames_dict[target_frame]
    for offset in range(1, max_gap + 1):
        for sign in [1, -1]:
            f = target_frame + sign * offset
            if f in frames_dict:
                return frames_dict[f]
    return None


def _get_player_position3(
    frames_dict: dict[int, tuple[float, float, float]],
    frame: int,
    contact_frame: int,
) -> tuple[float, float, float] | None:
    """Get (x, y, height) at frame, carrying forward from nearest known frame.

    Searches outward from the target frame, biased toward the contact frame
    (where we know the player was detected for canonicalization).
    """
    if frame in frames_dict:
        return frames_dict[frame]

    for offset in range(1, 6):
        toward = frame + (1 if frame < contact_frame else -1) * offset
        away = frame - (1 if frame < contact_frame else -1) * offset
        if toward in frames_dict:
            return frames_dict[toward]
        if away in frames_dict:
            return frames_dict[away]

    return None
