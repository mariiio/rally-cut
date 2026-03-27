"""Feature extraction for temporal contact attribution.

Extracts per-contact summary features from ball + player trajectory windows.
Used by both the training script and the inference integration in contact_detector.
"""

from __future__ import annotations

import math

import numpy as np

from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.player_tracker import PlayerPosition

# Window: ±10 frames around contact = 21 frames total
WINDOW_HALF = 10

# Maximum gap (frames) to interpolate ball positions across
MAX_INTERP_GAP = 5

# Minimum fraction of window frames with ball data to accept
MIN_BALL_COVERAGE = 0.5

# Feature names in order (for tree model interpretability)
FEATURE_NAMES: list[str] = [
    # Per-slot distance features (4 slots × 4 features = 16)
    "dist_at_contact_0", "dist_at_contact_1", "dist_at_contact_2", "dist_at_contact_3",
    "dist_min_0", "dist_min_1", "dist_min_2", "dist_min_3",
    "dist_slope_pre_0", "dist_slope_pre_1", "dist_slope_pre_2", "dist_slope_pre_3",
    "dist_slope_post_0", "dist_slope_post_1", "dist_slope_post_2", "dist_slope_post_3",
    # Per-slot depth features (4 slots × 1 = 4)
    "bbox_height_0", "bbox_height_1", "bbox_height_2", "bbox_height_3",
    # Per-slot velocity/angle features (4 slots × 2 = 8)
    "ball_toward_player_0", "ball_toward_player_1",
    "ball_toward_player_2", "ball_toward_player_3",
    "player_speed_0", "player_speed_1", "player_speed_2", "player_speed_3",
    # Distance ratio features (3)
    "dist_ratio_01", "dist_ratio_02", "dist_margin_01",
    # Ball dynamics (3)
    "ball_speed", "ball_dir_change", "ball_y_at_contact",
    # Contact sequence context (2)
    "contact_index", "side_count",
]

NUM_FEATURES = len(FEATURE_NAMES)  # 36


def extract_attribution_features(
    contact_frame: int,
    ball_positions: list[BallPosition],
    player_positions: list[PlayerPosition],
    contact_index: int = 0,
    side_count: int = 1,
    window_half: int = WINDOW_HALF,
) -> tuple[np.ndarray, list[int]] | None:
    """Extract summary features for a contact for tree-based attribution.

    Args:
        contact_frame: Rally-relative frame number of the contact.
        ball_positions: Ball tracking positions for the rally.
        player_positions: Player tracking positions for the rally.
        contact_index: 0-based index of this contact in the rally.
        side_count: Number of contacts on the current court side (1-3).
        window_half: Half-window size for trajectory analysis.

    Returns:
        (features, canonical_track_ids) where features is a (NUM_FEATURES,) array
        and canonical_track_ids lists up to 4 track IDs in slot order (nearest first).
        Returns None if insufficient data.
    """
    window_size = 2 * window_half + 1
    frame_start = contact_frame - window_half
    frame_end = contact_frame + window_half

    # --- Ball positions: build frame → (x, y) lookup ---
    ball_by_frame: dict[int, tuple[float, float]] = {}
    for bp in ball_positions:
        if frame_start <= bp.frame_number <= frame_end:
            ball_by_frame[bp.frame_number] = (bp.x, bp.y)

    ball_xy = _interpolate_ball(ball_by_frame, frame_start, frame_end)

    n_valid = sum(1 for xy in ball_xy if xy is not None)
    if n_valid < window_size * MIN_BALL_COVERAGE:
        return None

    # --- Player positions: per-track, per-frame (x, y_contact, height) ---
    player_data: dict[int, dict[int, tuple[float, float, float]]] = {}
    for pp in player_positions:
        if frame_start <= pp.frame_number <= frame_end and pp.track_id >= 0:
            if pp.track_id not in player_data:
                player_data[pp.track_id] = {}
            player_data[pp.track_id][pp.frame_number] = (
                pp.x,
                pp.y - pp.height * 0.25,  # upper-quarter y
                pp.height,
            )

    if len(player_data) < 2:
        return None

    # --- Ball at contact frame ---
    ball_at_contact = ball_xy[window_half]
    if ball_at_contact is None:
        for offset in [-1, 1, -2, 2, -3, 3]:
            idx = window_half + offset
            if 0 <= idx < window_size and ball_xy[idx] is not None:
                ball_at_contact = ball_xy[idx]
                break
    if ball_at_contact is None:
        return None

    bx, by = ball_at_contact

    # --- Canonical ordering: sort tracks by distance at contact frame ---
    track_dists: list[tuple[int, float]] = []
    for tid, frames_dict in player_data.items():
        pos = _nearest_pos(frames_dict, contact_frame)
        if pos is not None:
            px, py, _h = pos
            dist = math.sqrt((bx - px) ** 2 + (by - py) ** 2)
            track_dists.append((tid, dist))

    if len(track_dists) < 2:
        return None

    track_dists.sort(key=lambda td: td[1])
    canonical_tids = [tid for tid, _ in track_dists[:4]]
    num_slots = len(canonical_tids)

    # --- Compute per-frame distances for each slot ---
    distances = np.full((window_size, 4), 0.5, dtype=np.float32)
    heights_arr = np.zeros((window_size, 4), dtype=np.float32)
    player_xy = np.zeros((window_size, 4, 2), dtype=np.float32)

    for i in range(window_size):
        frame = frame_start + i
        bc = ball_xy[i]
        for s in range(min(num_slots, 4)):
            tid = canonical_tids[s]
            pos = _get_pos(player_data[tid], frame, contact_frame)
            if pos is not None:
                px, py, ph = pos
                player_xy[i, s] = [px, py]
                heights_arr[i, s] = ph
                if bc is not None:
                    distances[i, s] = math.sqrt(
                        (bc[0] - px) ** 2 + (bc[1] - py) ** 2
                    )

    mid = window_half  # index of contact frame in window

    # --- Build feature vector ---
    features = np.zeros(NUM_FEATURES, dtype=np.float32)

    # Distance at contact (0-3)
    features[0:4] = distances[mid]

    # Minimum distance in window (4-7)
    features[4:8] = distances.min(axis=0)

    # Distance slope pre-contact: mean delta over frames [mid-5, mid] (8-11)
    pre_start = max(1, mid - 5)
    pre_deltas = distances[pre_start:mid + 1] - distances[pre_start - 1:mid]
    if len(pre_deltas) > 0:
        features[8:12] = pre_deltas.mean(axis=0)

    # Distance slope post-contact: mean delta over frames [mid, mid+5] (12-15)
    post_end = min(window_size - 1, mid + 5)
    post_deltas = distances[mid + 1:post_end + 1] - distances[mid:post_end]
    if len(post_deltas) > 0:
        features[12:16] = post_deltas.mean(axis=0)

    # Bbox height at contact frame (16-19)
    features[16:20] = heights_arr[mid]

    # Ball-toward-player: cosine similarity of ball velocity and ball→player vector (20-23)
    # Positive means ball is heading toward the player
    ball_vel = _ball_velocity_at(ball_xy, mid)
    if ball_vel is not None:
        bvx, bvy = ball_vel
        bv_mag = math.sqrt(bvx * bvx + bvy * bvy)
        if bv_mag > 1e-6:
            for s in range(min(num_slots, 4)):
                pos = _get_pos(player_data[canonical_tids[s]], contact_frame, contact_frame)
                if pos is not None and ball_at_contact is not None:
                    dx = pos[0] - ball_at_contact[0]
                    dy = pos[1] - ball_at_contact[1]
                    d_mag = math.sqrt(dx * dx + dy * dy)
                    if d_mag > 1e-6:
                        cos_sim = (bvx * dx + bvy * dy) / (bv_mag * d_mag)
                        features[20 + s] = cos_sim

    # Player speed at contact (24-27)
    for s in range(min(num_slots, 4)):
        tid = canonical_tids[s]
        speed = _player_speed_at(player_data[tid], contact_frame)
        features[24 + s] = speed

    # Distance ratios (28-30)
    d0 = max(distances[mid, 0], 1e-6)
    d1 = max(distances[mid, 1], 1e-6) if num_slots > 1 else 1.0
    d2 = max(distances[mid, 2], 1e-6) if num_slots > 2 else 1.0
    features[28] = d0 / d1  # ratio slot0/slot1 (< 1 means slot0 is closer)
    features[29] = d0 / d2  # ratio slot0/slot2
    features[30] = d1 - d0  # margin between slot1 and slot0

    # Ball dynamics (31-33)
    if ball_vel is not None:
        features[31] = math.sqrt(ball_vel[0] ** 2 + ball_vel[1] ** 2)
    features[32] = _ball_dir_change_at(ball_xy, mid)
    if ball_at_contact is not None:
        features[33] = ball_at_contact[1]  # ball y position (near vs far court)

    # Contact sequence context (34-35)
    features[34] = float(contact_index)
    features[35] = float(side_count)

    # Pad canonical_tids to 4
    while len(canonical_tids) < 4:
        canonical_tids.append(-1)

    return features, canonical_tids


# --- Helper functions ---


def _interpolate_ball(
    ball_by_frame: dict[int, tuple[float, float]],
    frame_start: int,
    frame_end: int,
) -> list[tuple[float, float] | None]:
    """Build per-frame ball positions with linear interpolation for short gaps."""
    window_size = frame_end - frame_start + 1
    result: list[tuple[float, float] | None] = [None] * window_size

    for f, xy in ball_by_frame.items():
        idx = f - frame_start
        if 0 <= idx < window_size:
            result[idx] = xy

    i = 0
    while i < window_size:
        if result[i] is not None:
            i += 1
            continue

        gap_start = i
        while i < window_size and result[i] is None:
            i += 1
        gap_end = i

        gap_len = gap_end - gap_start
        if gap_len > MAX_INTERP_GAP:
            continue

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


def _nearest_pos(
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


def _get_pos(
    frames_dict: dict[int, tuple[float, float, float]],
    frame: int,
    contact_frame: int,
    max_gap: int = 5,
) -> tuple[float, float, float] | None:
    """Get (x, y, height) at frame with directional carry-forward."""
    if frame in frames_dict:
        return frames_dict[frame]
    for offset in range(1, max_gap + 1):
        toward = frame + (1 if frame < contact_frame else -1) * offset
        away = frame - (1 if frame < contact_frame else -1) * offset
        if toward in frames_dict:
            return frames_dict[toward]
        if away in frames_dict:
            return frames_dict[away]
    return None


def _ball_velocity_at(
    ball_xy: list[tuple[float, float] | None],
    idx: int,
) -> tuple[float, float] | None:
    """Compute ball velocity (dx, dy) at window index using central difference."""
    if idx <= 0 or idx >= len(ball_xy) - 1:
        return None
    prev = ball_xy[idx - 1]
    nxt = ball_xy[idx + 1]
    if prev is not None and nxt is not None:
        return ((nxt[0] - prev[0]) / 2, (nxt[1] - prev[1]) / 2)
    # Fall back to one-sided
    cur = ball_xy[idx]
    if cur is not None and prev is not None:
        return (cur[0] - prev[0], cur[1] - prev[1])
    if cur is not None and nxt is not None:
        return (nxt[0] - cur[0], nxt[1] - cur[1])
    return None


def _ball_dir_change_at(
    ball_xy: list[tuple[float, float] | None],
    idx: int,
) -> float:
    """Incoming trajectory curvature (1 - cos_angle) using frames idx-2..idx."""
    if idx < 2 or idx >= len(ball_xy):
        return 0.0
    p0, p1, p2 = ball_xy[idx - 2], ball_xy[idx - 1], ball_xy[idx]
    if p0 is None or p1 is None or p2 is None:
        return 0.0
    dx0, dy0 = p1[0] - p0[0], p1[1] - p0[1]
    dx1, dy1 = p2[0] - p1[0], p2[1] - p1[1]
    mag0 = math.sqrt(dx0 * dx0 + dy0 * dy0)
    mag1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
    if mag0 < 1e-6 or mag1 < 1e-6:
        return 0.0
    cos_a = max(-1.0, min(1.0, (dx0 * dx1 + dy0 * dy1) / (mag0 * mag1)))
    return 1.0 - cos_a


def _player_speed_at(
    frames_dict: dict[int, tuple[float, float, float]],
    frame: int,
) -> float:
    """Compute player speed at frame using ±2 frame window."""
    before = None
    after = None
    for offset in range(1, 3):
        if before is None and (frame - offset) in frames_dict:
            before = frames_dict[frame - offset]
        if after is None and (frame + offset) in frames_dict:
            after = frames_dict[frame + offset]
    if before is not None and after is not None:
        dx = after[0] - before[0]
        dy = after[1] - before[1]
        frames_span = 4  # max span is 4 frames
        return math.sqrt(dx * dx + dy * dy) / frames_span
    return 0.0
