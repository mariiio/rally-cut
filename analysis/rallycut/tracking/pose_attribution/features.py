"""Per-candidate feature extraction for pose-based attribution.

Unlike the canonical-slot model (temporal_attribution/features.py) which
produces one feature vector per contact with slots sorted by distance,
this module produces one feature vector PER CANDIDATE, enabling a binary
classifier that scores each candidate independently.
"""

from __future__ import annotations

import math

import numpy as np

from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.player_tracker import PlayerPosition

# Reuse helpers from existing temporal attribution
from rallycut.tracking.temporal_attribution.features import (
    _ball_velocity_at,
    _get_pos,
    _interpolate_ball,
    _nearest_pos,
    _player_speed_at,
)

# COCO keypoint indices
KPT_LEFT_SHOULDER = 5
KPT_RIGHT_SHOULDER = 6
KPT_LEFT_ELBOW = 7
KPT_RIGHT_ELBOW = 8
KPT_LEFT_WRIST = 9
KPT_RIGHT_WRIST = 10
KPT_LEFT_HIP = 11
KPT_RIGHT_HIP = 12

# Window sizes
POSE_WINDOW_HALF = 5  # ±5 frames for pose features
SPATIAL_WINDOW_HALF = 10  # ±10 frames for spatial features (match existing)

MIN_BALL_COVERAGE = 0.5
MIN_KPT_CONF = 0.3

POSE_FEATURE_COUNT = 15

FEATURE_NAMES: list[str] = [
    # Pose features (15) — uses active (ball-nearest) hand
    "active_wrist_velocity_max",
    "active_wrist_velocity_at_contact",
    "active_arm_extension_max",
    "active_arm_extension_change",
    "active_wrist_elevation_max",
    "active_wrist_elevation_at_contact",
    "vertical_displacement",
    "hand_ball_dist_min",
    "hand_ball_dist_at_contact",
    "torso_lean_change",
    "pose_confidence_mean",
    "n_visible_keypoints",
    "arm_asymmetry",           # |left_elev - right_elev| — high for attacks, low for sets
    "active_hand_side",        # 0=left, 1=right (which hand is closer to ball)
    "both_arms_raised",        # fraction of frames with both wrists above shoulders
    # Spatial/motion features (9)
    "distance_at_contact",
    "distance_min",
    "distance_slope_pre",
    "distance_slope_post",
    "bbox_height",
    "ball_toward_player",
    "player_speed",
    "bbox_max_dy",
    "bbox_max_dheight",
    # Context features (6)
    "distance_rank",
    "distance_ratio_to_nearest",
    "contact_index",
    "side_count",
    "ball_speed",
    "ball_y",
]

NUM_FEATURES = len(FEATURE_NAMES)  # 30

# Indices for spatial-only mode (no pose)
SPATIAL_FEATURE_NAMES = FEATURE_NAMES[POSE_FEATURE_COUNT:]
NUM_SPATIAL_FEATURES = len(SPATIAL_FEATURE_NAMES)  # 15


def extract_candidate_features(
    contact_frame: int,
    ball_positions: list[BallPosition],
    player_positions: list[PlayerPosition],
    contact_index: int = 0,
    side_count: int = 1,
    pose_data: dict[str, np.ndarray] | None = None,
) -> list[tuple[int, np.ndarray]] | None:
    """Extract per-candidate feature vectors for a contact.

    Args:
        contact_frame: Rally-relative frame number of the contact.
        ball_positions: Ball tracking positions for the rally.
        player_positions: Player tracking positions for the rally.
        contact_index: 0-based index of this contact in the rally.
        side_count: Number of contacts on the current court side (1-3).
        pose_data: Cached YOLO-Pose keypoints dict from pose_cache.
            Keys: 'frames', 'track_ids', 'keypoints' (N, 17, 3).
            If None, pose features are set to NaN.

    Returns:
        List of (track_id, features) tuples for each candidate,
        sorted by distance (nearest first), up to 4 candidates.
        Returns None if insufficient data.
    """
    spatial_window = 2 * SPATIAL_WINDOW_HALF + 1
    frame_start = contact_frame - SPATIAL_WINDOW_HALF
    frame_end = contact_frame + SPATIAL_WINDOW_HALF

    # --- Ball positions ---
    ball_by_frame: dict[int, tuple[float, float]] = {}
    for bp in ball_positions:
        if frame_start <= bp.frame_number <= frame_end:
            ball_by_frame[bp.frame_number] = (bp.x, bp.y)

    ball_xy = _interpolate_ball(ball_by_frame, frame_start, frame_end)

    n_valid = sum(1 for xy in ball_xy if xy is not None)
    if n_valid < spatial_window * MIN_BALL_COVERAGE:
        return None

    # Ball at contact
    mid = SPATIAL_WINDOW_HALF
    ball_at_contact = ball_xy[mid]
    if ball_at_contact is None:
        for offset in [-1, 1, -2, 2, -3, 3]:
            idx = mid + offset
            if 0 <= idx < spatial_window and ball_xy[idx] is not None:
                ball_at_contact = ball_xy[idx]
                break
    if ball_at_contact is None:
        return None

    bx, by = ball_at_contact

    # --- Player positions: per-track, per-frame (x, y_contact, height) ---
    player_data: dict[int, dict[int, tuple[float, float, float]]] = {}
    # Also store raw positions for bbox motion
    raw_positions: dict[int, dict[int, tuple[float, float, float, float]]] = {}
    for pp in player_positions:
        if frame_start <= pp.frame_number <= frame_end and pp.track_id >= 0:
            if pp.track_id not in player_data:
                player_data[pp.track_id] = {}
                raw_positions[pp.track_id] = {}
            player_data[pp.track_id][pp.frame_number] = (
                pp.x,
                pp.y - pp.height * 0.25,  # upper-quarter y (match existing)
                pp.height,
            )
            raw_positions[pp.track_id][pp.frame_number] = (
                pp.x, pp.y, pp.width, pp.height,
            )

    if len(player_data) < 2:
        return None

    # --- Rank candidates by distance at contact ---
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
    candidates = track_dists[:4]
    nearest_dist = candidates[0][1]

    # --- Ball dynamics (shared) ---
    ball_vel = _ball_velocity_at(ball_xy, mid)
    ball_speed = 0.0
    if ball_vel is not None:
        ball_speed = math.sqrt(ball_vel[0] ** 2 + ball_vel[1] ** 2)

    # --- Build pose lookup: (frame, track_id) -> keypoints (17, 3) ---
    pose_lookup: dict[tuple[int, int], np.ndarray] = {}
    if pose_data is not None and len(pose_data["frames"]) > 0:
        for i in range(len(pose_data["frames"])):
            key = (int(pose_data["frames"][i]), int(pose_data["track_ids"][i]))
            pose_lookup[key] = pose_data["keypoints"][i]  # (17, 3)
    # Read inline keypoints from PlayerPosition. This is the primary source
    # in production (populated by enrich_positions_with_pose during tracking)
    # and the offline fallback for rallies enriched via inject_keypoints.py.
    # The pose_data argument above is kept for backward compatibility with
    # the .npz cache format used by older eval scripts.
    for pp in player_positions:
        if pp.keypoints is None:
            continue
        key = (pp.frame_number, pp.track_id)
        if key not in pose_lookup:
            pose_lookup[key] = np.asarray(pp.keypoints, dtype=np.float32)

    # --- Extract features per candidate ---
    result: list[tuple[int, np.ndarray]] = []

    for rank, (tid, dist_at_contact) in enumerate(candidates):
        features = np.full(NUM_FEATURES, np.nan, dtype=np.float32)

        # ===== Pose features (0-14) =====
        _fill_pose_features(
            features, tid, contact_frame, ball_xy, ball_at_contact,
            pose_lookup, mid,
        )

        # ===== Spatial/motion features (15-23) =====
        s_off = POSE_FEATURE_COUNT  # offset for spatial features
        frames_dict = player_data[tid]

        # Distance at contact
        features[s_off +0] = dist_at_contact

        # Distance min in window
        min_dist = dist_at_contact
        for i in range(spatial_window):
            frame = frame_start + i
            bc = ball_xy[i]
            if bc is None:
                continue
            pos = _get_pos(frames_dict, frame, contact_frame)
            if pos is not None:
                px, py, _h = pos
                d = math.sqrt((bc[0] - px) ** 2 + (bc[1] - py) ** 2)
                min_dist = min(min_dist, d)
        features[s_off +1] = min_dist

        # Distance slopes
        pre_dists: list[float] = []
        post_dists: list[float] = []
        for i in range(spatial_window):
            frame = frame_start + i
            bc = ball_xy[i]
            if bc is None:
                continue
            pos = _get_pos(frames_dict, frame, contact_frame)
            if pos is not None:
                px, py, _h = pos
                d = math.sqrt((bc[0] - px) ** 2 + (bc[1] - py) ** 2)
                if i < mid:
                    pre_dists.append(d)
                elif i > mid:
                    post_dists.append(d)

        if len(pre_dists) >= 2:
            deltas = [pre_dists[j + 1] - pre_dists[j] for j in range(len(pre_dists) - 1)]
            features[s_off +2] = float(np.mean(deltas))
        if len(post_dists) >= 2:
            deltas = [post_dists[j + 1] - post_dists[j] for j in range(len(post_dists) - 1)]
            features[s_off +3] = float(np.mean(deltas))

        # Bbox height
        pos_at = _nearest_pos(frames_dict, contact_frame)
        if pos_at is not None:
            features[s_off +4] = pos_at[2]

        # Ball toward player
        if ball_vel is not None and pos_at is not None:
            bvx, bvy = ball_vel
            bv_mag = math.sqrt(bvx * bvx + bvy * bvy)
            if bv_mag > 1e-6:
                dx = pos_at[0] - bx
                dy = pos_at[1] - by
                d_mag = math.sqrt(dx * dx + dy * dy)
                if d_mag > 1e-6:
                    features[s_off +5] = (bvx * dx + bvy * dy) / (bv_mag * d_mag)

        # Player speed
        features[s_off +6] = _player_speed_at(frames_dict, contact_frame)

        # Bbox motion: max dy, max dheight in ±5 frame window
        _fill_bbox_motion(features, raw_positions.get(tid, {}), contact_frame, s_off + 7)

        # ===== Context features (24-29) =====
        c_off = s_off + 9  # offset for context features
        features[c_off +0] = float(rank)
        features[c_off +1] = dist_at_contact / max(nearest_dist, 1e-6)
        features[c_off +2] = float(contact_index)
        features[c_off +3] = float(side_count)
        features[c_off +4] = ball_speed
        features[c_off +5] = by  # ball Y position

        result.append((tid, features))

    return result


def _fill_pose_features(
    features: np.ndarray,
    track_id: int,
    contact_frame: int,
    ball_xy: list[tuple[float, float] | None],
    ball_at_contact: tuple[float, float],
    pose_lookup: dict[tuple[int, int], np.ndarray],
    mid: int,
) -> None:
    """Fill pose feature slots (indices 0-14) from cached keypoints.

    Uses active-hand tracking: the wrist closest to the ball at each frame
    is the "active" (hitting) hand. This properly captures single-arm
    actions like attacks and serves.
    """
    bx, by = ball_at_contact

    # Per-frame data for the active (ball-nearest) hand
    active_wrist_velocities: list[float] = []
    active_arm_extensions: list[float] = []
    active_wrist_elevations: list[float] = []
    hip_y_values: list[float] = []
    torso_angles: list[float] = []
    all_confs: list[float] = []
    n_visible_at_contact = 0

    # Contact-frame values (tracked separately, not via mid_idx heuristic)
    wrist_velocity_at_contact: float | None = None
    wrist_elevation_at_contact: float | None = None
    hand_ball_dist_at_contact: float | None = None
    hand_ball_dist_min = float("inf")

    # For arm asymmetry: track per-side elevations separately
    left_elevations: list[float] = []
    right_elevations: list[float] = []
    both_arms_raised_count = 0
    total_frames_with_arms = 0

    # Track active hand side across frames
    active_hand_sides: list[int] = []  # 0=left, 1=right

    # Track active wrist position for velocity computation
    prev_active_wrist: tuple[float, float] | None = None

    for offset in range(-POSE_WINDOW_HALF, POSE_WINDOW_HALF + 1):
        frame = contact_frame + offset
        kps = pose_lookup.get((frame, track_id))
        if kps is None:
            prev_active_wrist = None
            continue

        # Keypoint confidence
        conf_vals = kps[:, 2]
        all_confs.extend(conf_vals.tolist())

        if offset == 0:
            n_visible_at_contact = int((conf_vals > MIN_KPT_CONF).sum())

        lw = kps[KPT_LEFT_WRIST]
        rw = kps[KPT_RIGHT_WRIST]
        ls = kps[KPT_LEFT_SHOULDER]
        rs = kps[KPT_RIGHT_SHOULDER]
        le = kps[KPT_LEFT_ELBOW]
        re = kps[KPT_RIGHT_ELBOW]
        lh = kps[KPT_LEFT_HIP]
        rh = kps[KPT_RIGHT_HIP]

        # --- Determine active hand: wrist closest to ball this frame ---
        ball_idx = mid + offset
        ball_at_frame = (
            ball_xy[ball_idx]
            if 0 <= ball_idx < len(ball_xy) and ball_xy[ball_idx] is not None
            else None
        )
        bfx, bfy = ball_at_frame if ball_at_frame is not None else ball_at_contact

        active_side = -1  # 0=left, 1=right
        active_wrist = None
        active_shoulder = None
        active_elbow = None
        min_wrist_ball_dist = float("inf")

        for side, (w, s, e) in enumerate([(lw, ls, le), (rw, rs, re)]):
            if w[2] > MIN_KPT_CONF:
                d = math.sqrt((w[0] - bfx) ** 2 + (w[1] - bfy) ** 2)
                hand_ball_dist_min = min(hand_ball_dist_min, d)
                if offset == 0:
                    if hand_ball_dist_at_contact is None or d < hand_ball_dist_at_contact:
                        hand_ball_dist_at_contact = d
                if d < min_wrist_ball_dist:
                    min_wrist_ball_dist = d
                    active_side = side
                    active_wrist = w
                    active_shoulder = s if s[2] > MIN_KPT_CONF else None
                    active_elbow = e if e[2] > MIN_KPT_CONF else None

        if active_side >= 0:
            active_hand_sides.append(active_side)

        # --- Active wrist velocity (frame-to-frame displacement of active hand) ---
        if active_wrist is not None:
            cur_pos = (active_wrist[0], active_wrist[1])
            if prev_active_wrist is not None:
                dx = cur_pos[0] - prev_active_wrist[0]
                dy = cur_pos[1] - prev_active_wrist[1]
                vel = math.sqrt(dx * dx + dy * dy)
                active_wrist_velocities.append(vel)
                if offset == 0:
                    wrist_velocity_at_contact = vel
            prev_active_wrist = cur_pos
        else:
            prev_active_wrist = None

        # --- Active arm extension (elbow angle of active arm) ---
        if active_shoulder is not None and active_elbow is not None and active_wrist is not None:
            v1 = (active_shoulder[0] - active_elbow[0], active_shoulder[1] - active_elbow[1])
            v2 = (active_wrist[0] - active_elbow[0], active_wrist[1] - active_elbow[1])
            dot = v1[0] * v2[0] + v1[1] * v2[1]
            m1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
            m2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)
            if m1 > 1e-6 and m2 > 1e-6:
                cos_a = max(-1.0, min(1.0, dot / (m1 * m2)))
                active_arm_extensions.append(math.degrees(math.acos(cos_a)))

        # --- Active wrist elevation (relative to its shoulder) ---
        if active_wrist is not None and active_shoulder is not None:
            # Positive = wrist above shoulder (image Y is inverted)
            elev = active_shoulder[1] - active_wrist[1]
            active_wrist_elevations.append(elev)
            if offset == 0:
                wrist_elevation_at_contact = elev

        # --- Per-side elevations for asymmetry ---
        left_elev = None
        right_elev = None
        if ls[2] > MIN_KPT_CONF and lw[2] > MIN_KPT_CONF:
            left_elev = ls[1] - lw[1]
            left_elevations.append(left_elev)
        if rs[2] > MIN_KPT_CONF and rw[2] > MIN_KPT_CONF:
            right_elev = rs[1] - rw[1]
            right_elevations.append(right_elev)

        # Both arms raised: both wrists above their shoulders
        if left_elev is not None and right_elev is not None:
            total_frames_with_arms += 1
            if left_elev > 0 and right_elev > 0:
                both_arms_raised_count += 1

        # --- Hip midpoint Y (for vertical displacement) ---
        if lh[2] > MIN_KPT_CONF and rh[2] > MIN_KPT_CONF:
            hip_y_values.append((lh[1] + rh[1]) / 2)

        # --- Torso lean ---
        for s, h in [(ls, lh), (rs, rh)]:
            if s[2] > MIN_KPT_CONF and h[2] > MIN_KPT_CONF:
                dx = s[0] - h[0]
                dy = s[1] - h[1]
                torso_angles.append(abs(math.degrees(math.atan2(dx, -dy))))

    # === Fill feature slots ===

    # Active wrist velocity (0-1)
    if active_wrist_velocities:
        features[0] = max(active_wrist_velocities)
    if wrist_velocity_at_contact is not None:
        features[1] = wrist_velocity_at_contact

    # Active arm extension (2-3)
    if active_arm_extensions:
        features[2] = max(active_arm_extensions)
        features[3] = max(active_arm_extensions) - min(active_arm_extensions)

    # Active wrist elevation (4-5)
    if active_wrist_elevations:
        features[4] = max(active_wrist_elevations)
    if wrist_elevation_at_contact is not None:
        features[5] = wrist_elevation_at_contact

    # Vertical displacement (6)
    if len(hip_y_values) >= 2:
        features[6] = max(hip_y_values) - min(hip_y_values)

    # Hand-ball distance (7-8): min over window, and value at contact frame
    if hand_ball_dist_min < float("inf"):
        features[7] = hand_ball_dist_min
    if hand_ball_dist_at_contact is not None:
        features[8] = hand_ball_dist_at_contact

    # Torso lean change (9)
    if torso_angles:
        features[9] = max(torso_angles) - min(torso_angles)

    # Pose confidence (10)
    if all_confs:
        features[10] = float(np.mean(all_confs))

    # Visible keypoints (11)
    features[11] = float(n_visible_at_contact)

    # Arm asymmetry (12): mean |left_elev - right_elev| across frames
    # High for single-arm actions (attack/serve), low for two-handed (set/block)
    if left_elevations and right_elevations:
        n = min(len(left_elevations), len(right_elevations))
        asymmetries = [abs(left_elevations[i] - right_elevations[i]) for i in range(n)]
        features[12] = float(np.mean(asymmetries))

    # Active hand side (13): dominant hand (0=left, 1=right)
    if active_hand_sides:
        features[13] = float(np.mean(active_hand_sides))  # 0-1 continuous

    # Both arms raised fraction (14): high for blocks/sets, low for attacks
    if total_frames_with_arms > 0:
        features[14] = both_arms_raised_count / total_frames_with_arms


def _fill_bbox_motion(
    features: np.ndarray,
    raw_pos: dict[int, tuple[float, float, float, float]],
    contact_frame: int,
    start_idx: int,
    window: int = 5,
) -> None:
    """Fill bbox motion features from raw positions."""
    max_dy = 0.0
    max_dh = 0.0

    frames = sorted(f for f in raw_pos if abs(f - contact_frame) <= window)
    for i in range(1, len(frames)):
        _, y1, _, h1 = raw_pos[frames[i - 1]]
        _, y2, _, h2 = raw_pos[frames[i]]
        max_dy = max(max_dy, abs(y2 - y1))
        max_dh = max(max_dh, abs(h2 - h1))

    features[start_idx] = max_dy
    features[start_idx + 1] = max_dh


def extract_spatial_only_features(
    contact_frame: int,
    ball_positions: list[BallPosition],
    player_positions: list[PlayerPosition],
    contact_index: int = 0,
    side_count: int = 1,
) -> list[tuple[int, np.ndarray]] | None:
    """Extract per-candidate features without pose data (spatial-only).

    Returns features with pose slots (0-14) set to NaN. This is equivalent
    to calling extract_candidate_features with pose_data=None.
    """
    return extract_candidate_features(
        contact_frame=contact_frame,
        ball_positions=ball_positions,
        player_positions=player_positions,
        contact_index=contact_index,
        side_count=side_count,
        pose_data=None,
    )
