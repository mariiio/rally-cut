"""Per-frame trajectory feature extraction for sequence action classification.

Extracts a fixed-size feature vector per frame from raw ball and player
tracking data. Features are designed so a temporal model can learn
possession structure and action types directly from trajectories,
bypassing explicit contact detection.

Feature layout (26 dims):
    [0]  ball_x            — normalized ball x position
    [1]  ball_y            — normalized ball y position
    [2]  ball_conf         — ball detection confidence (0 = missing)
    [3]  ball_dx           — frame-to-frame x velocity
    [4]  ball_dy           — frame-to-frame y velocity
    [5]  ball_speed        — magnitude of velocity vector
    [6:14] player_xy       — 4 players × (x, y), sorted by y each frame
    [14:18] ball_player_dist — distance from ball to each sorted player
    [18] ball_y_rel_net    — ball_y minus net_y
    [19:21] court_ball_xy  — homography-projected court coordinates (meters)
    [21] ball_det_density  — rolling 21-frame detection density (tracking gap signal)
    [22:26] player_team    — team indicator per sorted player (0/1/0.5=unknown)

Pruned features (empirically dead via permutation importance):
    - player_delta_y: +0.0% importance, model learns motion from position diffs
    - ball_accel: +0.0% importance, model learns from velocity diffs via convolutions
"""

from __future__ import annotations

import numpy as np

from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.player_tracker import PlayerPosition

FEATURE_DIM = 26
NUM_PLAYERS = 4

ACTION_TYPES = ["serve", "receive", "set", "attack", "dig", "block"]
ACTION_TO_IDX = {a: i + 1 for i, a in enumerate(ACTION_TYPES)}
NUM_CLASSES = 7  # background + 6 actions

# Rolling window size for ball detection density
_DENSITY_WINDOW = 21
# Court dimensions for normalizing court-space features (beach volleyball: 8m × 16m)
_COURT_WIDTH = 8.0
_COURT_LENGTH = 16.0


def extract_trajectory_features(
    ball_positions: list[BallPosition],
    player_positions: list[PlayerPosition],
    net_y: float | None,
    frame_count: int,
    *,
    team_assignments: dict[int, int] | None = None,
    homography: np.ndarray | None = None,
) -> np.ndarray:
    """Extract per-frame trajectory features for a rally.

    Args:
        ball_positions: Ball detections (frame_number, x, y, confidence).
        player_positions: Player detections (frame_number, track_id, x, y, ...).
        net_y: Estimated net y-coordinate in normalized image space.
        frame_count: Total number of frames in the rally.
        team_assignments: Optional track_id → team (0=near, 1=far) mapping.
        homography: Optional 3×3 image→court homography matrix. When provided,
            ball positions are projected to court coordinates (meters).

    Returns:
        (frame_count, 26) float32 array of trajectory features.
    """
    if net_y is None:
        net_y = 0.5

    features = np.zeros((frame_count, FEATURE_DIM), dtype=np.float32)

    # --- Ball features [0:6] ---
    ball_by_frame: dict[int, BallPosition] = {}
    for bp in ball_positions:
        if 0 <= bp.frame_number < frame_count:
            ball_by_frame[bp.frame_number] = bp

    ball_x = np.zeros(frame_count, dtype=np.float32)
    ball_y = np.zeros(frame_count, dtype=np.float32)
    ball_conf = np.zeros(frame_count, dtype=np.float32)

    for f in range(frame_count):
        ball_at_f = ball_by_frame.get(f)
        if ball_at_f is not None and ball_at_f.confidence > 0:
            ball_x[f] = ball_at_f.x
            ball_y[f] = ball_at_f.y
            ball_conf[f] = ball_at_f.confidence

    features[:, 0] = ball_x
    features[:, 1] = ball_y
    features[:, 2] = ball_conf

    # Velocity: finite difference, zero at gaps
    ball_dx = np.zeros(frame_count, dtype=np.float32)
    ball_dy = np.zeros(frame_count, dtype=np.float32)
    for f in range(1, frame_count):
        if ball_conf[f] > 0 and ball_conf[f - 1] > 0:
            ball_dx[f] = ball_x[f] - ball_x[f - 1]
            ball_dy[f] = ball_y[f] - ball_y[f - 1]

    features[:, 3] = ball_dx
    features[:, 4] = ball_dy
    features[:, 5] = np.sqrt(ball_dx ** 2 + ball_dy ** 2)

    # --- Player features [6:18] ---
    players_by_frame: dict[int, list[PlayerPosition]] = {}
    for pp in player_positions:
        if 0 <= pp.frame_number < frame_count:
            players_by_frame.setdefault(pp.frame_number, []).append(pp)

    for f in range(frame_count):
        frame_players = players_by_frame.get(f, [])
        # Sort by y-coordinate (near court = small y → far court = large y)
        frame_players.sort(key=lambda p: p.y)

        for i in range(min(NUM_PLAYERS, len(frame_players))):
            p = frame_players[i]
            features[f, 6 + i * 2] = p.x
            features[f, 6 + i * 2 + 1] = p.y

        # Ball-player distances [14:18]
        bx, by = ball_x[f], ball_y[f]
        if ball_conf[f] > 0:
            for i in range(min(NUM_PLAYERS, len(frame_players))):
                p = frame_players[i]
                dist = np.sqrt((bx - p.x) ** 2 + (by - p.y) ** 2)
                features[f, 14 + i] = dist

        # --- Player team indicators [22:26] ---
        if team_assignments:
            for i in range(min(NUM_PLAYERS, len(frame_players))):
                p = frame_players[i]
                team = team_assignments.get(p.track_id)
                features[f, 22 + i] = float(team) if team is not None else 0.5
        else:
            for i in range(min(NUM_PLAYERS, len(frame_players))):
                features[f, 22 + i] = 0.5

    # --- Court context [18] ---
    features[:, 18] = ball_y - net_y
    features[ball_conf == 0, 18] = 0.0

    # --- Court-space ball position [19:21] ---
    if homography is not None:
        for f in range(frame_count):
            if ball_conf[f] > 0:
                pt = np.array([ball_x[f], ball_y[f], 1.0], dtype=np.float64)
                result = homography @ pt
                if abs(result[2]) > 1e-8:
                    result = result / result[2]
                    cx = float(result[0]) / _COURT_WIDTH
                    cy = float(result[1]) / _COURT_LENGTH
                    if -0.5 <= cx <= 1.5 and -0.5 <= cy <= 1.5:
                        features[f, 19] = cx
                        features[f, 20] = cy

    # --- Ball detection density [21] ---
    half_w = _DENSITY_WINDOW // 2
    det_mask = (ball_conf > 0.3).astype(np.float32)
    cumsum = np.concatenate([[0.0], np.cumsum(det_mask)])
    for f in range(frame_count):
        lo = max(0, f - half_w)
        hi = min(frame_count, f + half_w + 1)
        window_size = hi - lo
        features[f, 21] = (cumsum[hi] - cumsum[lo]) / window_size

    return features


def build_frame_labels(
    gt_labels: list[dict],
    frame_count: int,
    label_spread: int = 2,
) -> np.ndarray:
    """Build per-frame action labels from GT annotations.

    Args:
        gt_labels: List of dicts with 'frame' and 'action' keys.
        frame_count: Total frames in rally.
        label_spread: Spread labels ± this many frames around GT frame.

    Returns:
        (frame_count,) int64 array. 0 = background, 1-6 = action classes.
    """
    labels = np.zeros(frame_count, dtype=np.int64)

    for gt in gt_labels:
        frame = gt["frame"] if isinstance(gt, dict) else gt.frame
        action = gt["action"] if isinstance(gt, dict) else gt.action
        cls = ACTION_TO_IDX.get(action, 0)
        if cls == 0:
            continue

        for offset in range(-label_spread, label_spread + 1):
            idx = frame + offset
            if 0 <= idx < frame_count:
                # Center frame gets priority
                if offset == 0 or labels[idx] == 0:
                    labels[idx] = cls

    return labels
