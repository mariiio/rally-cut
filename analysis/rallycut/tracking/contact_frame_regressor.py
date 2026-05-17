"""Learned contact-frame regressor — predicts per-contact frame offset.

Trained 2026-05-17 on 2480 (candidate, GT-frame) pairs from the full action-GT
corpus (74 videos). LOO CV: MAE 2.11 → 1.32; within ±5 of GT: +64 net; within
±2 of GT: +353 cases. Replaces the direction-change-MAX heuristic snap as the
final frame-refinement step in detect_contacts().

Spec & training: reports/contact_frame_regressor_2026_05_17/
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

import joblib

logger = logging.getLogger(__name__)

# Match training script constants exactly — see
# analysis/scripts/extract_contact_frame_training_data_2026_05_17.py
NEIGHBOR_STEP = 3
WINDOW = 15
KPT_LEFT_WRIST = 9
KPT_RIGHT_WRIST = 10
KPT_LEFT_ELBOW = 7
KPT_RIGHT_ELBOW = 8
WRIST_CONF_MIN = 0.30
ACTIONS = ("SERVE", "RECEIVE", "SET", "ATTACK", "DIG", "BLOCK")

# Bounded shift — the model can never move a contact more than this many
# frames in either direction. Safety bound: training data range was [-14, +15].
MAX_SHIFT_FRAMES = 12

_MODEL_PATH = (
    Path(__file__).parent.parent.parent
    / "weights" / "contact_frame_regressor" / "best_model.joblib"
)
_MODEL_CACHE: dict[str, Any] = {}


def _load_model() -> dict[str, Any] | None:
    """Lazy-load the trained regressor. Returns None if not on disk."""
    if "default" in _MODEL_CACHE:
        cached = _MODEL_CACHE["default"]
        return cached if isinstance(cached, dict) else None
    if not _MODEL_PATH.exists():
        _MODEL_CACHE["default"] = None
        return None
    bundle = joblib.load(_MODEL_PATH)
    _MODEL_CACHE["default"] = bundle
    return bundle if isinstance(bundle, dict) else None


def _dir_change(balls: dict[int, Any], frame: int) -> float:
    p = balls.get(frame - NEIGHBOR_STEP)
    c = balls.get(frame)
    n = balls.get(frame + NEIGHBOR_STEP)
    if p is None or c is None or n is None:
        return 0.0
    v1x, v1y = c.x - p.x, c.y - p.y
    v2x, v2y = n.x - c.x, n.y - c.y
    m1, m2 = math.hypot(v1x, v1y), math.hypot(v2x, v2y)
    if m1 < 1e-4 or m2 < 1e-4:
        return 0.0
    ct = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (m1 * m2)))
    return math.degrees(math.acos(ct))


def _ball_velocity(balls: dict[int, Any], frame: int) -> float:
    cur = balls.get(frame)
    if cur is None:
        return 0.0
    speeds = []
    for off in (-1, 1):
        nbr = balls.get(frame + off)
        if nbr is not None:
            speeds.append(math.hypot(cur.x - nbr.x, cur.y - nbr.y))
    return sum(speeds) / max(1, len(speeds))


def _wrist_min(positions_at_frame: list[Any], ball: Any) -> float:
    best = float("inf")
    if ball is None:
        return best
    for p in positions_at_frame:
        kps = getattr(p, "keypoints", None)
        if not kps or len(kps) < 11:
            continue
        for idx in (KPT_LEFT_WRIST, KPT_RIGHT_WRIST):
            kp = kps[idx]
            if len(kp) < 3 or kp[2] < WRIST_CONF_MIN:
                continue
            d = math.hypot(kp[0] - ball.x, kp[1] - ball.y)
            if d < best:
                best = d
    return best


def _elbow_min(positions_at_frame: list[Any], ball: Any) -> float:
    best = float("inf")
    if ball is None:
        return best
    for p in positions_at_frame:
        kps = getattr(p, "keypoints", None)
        if not kps or len(kps) < 11:
            continue
        for idx in (KPT_LEFT_ELBOW, KPT_RIGHT_ELBOW):
            kp = kps[idx]
            if len(kp) < 3 or kp[2] < WRIST_CONF_MIN:
                continue
            d = math.hypot(kp[0] - ball.x, kp[1] - ball.y)
            if d < best:
                best = d
    return best


def _bbox_dist(positions_at_frame: list[Any], ball: Any) -> float:
    best = float("inf")
    if ball is None:
        return best
    for p in positions_at_frame:
        py_uq = p.y - p.height * 0.25
        d = math.hypot(p.x - ball.x, py_uq - ball.y)
        if d < best:
            best = d
    return best


def _extract_features(
    balls: dict[int, Any],
    pos_by_frame: dict[int, list[Any]],
    cand_frame: int,
    action_type: str | None,
) -> list[float] | None:
    """Return 25-dim feature vector matching the training script schema."""
    if cand_frame not in balls:
        return None

    dc_at = _dir_change(balls, cand_frame)
    vel_at = _ball_velocity(balls, cand_frame)
    pre_vel = _ball_velocity(balls, cand_frame - 3)
    post_vel = _ball_velocity(balls, cand_frame + 3)

    window_dc, window_vel = {}, {}
    window_wrist, window_elbow, window_bbox = {}, {}, {}
    for off in range(-WINDOW, WINDOW + 1):
        ff = cand_frame + off
        b = balls.get(ff)
        if not b:
            continue
        window_dc[off] = _dir_change(balls, ff)
        window_vel[off] = _ball_velocity(balls, ff)
        ps = pos_by_frame.get(ff, [])
        window_wrist[off] = _wrist_min(ps, b)
        window_elbow[off] = _elbow_min(ps, b)
        window_bbox[off] = _bbox_dist(ps, b)

    if not window_dc:
        return None

    dc_max = max(window_dc.values())
    dc_max_off = max(window_dc, key=lambda o: window_dc[o])
    vel_max = max(window_vel.values()) if window_vel else 0.0
    vel_min = min(window_vel.values()) if window_vel else 0.0
    vel_min_off = min(window_vel, key=lambda o: window_vel[o]) if window_vel else 0

    valid_w = {o: d for o, d in window_wrist.items() if d != float("inf")}
    wrist_at = window_wrist.get(0, float("inf"))
    wrist_at = min(wrist_at, 1.0) if wrist_at != float("inf") else -1.0
    if valid_w:
        wrist_min = min(valid_w.values())
        wrist_min_off = min(valid_w, key=lambda o: valid_w[o])
    else:
        wrist_min, wrist_min_off = -1.0, 0

    valid_e = {o: d for o, d in window_elbow.items() if d != float("inf")}
    elbow_at = window_elbow.get(0, float("inf"))
    elbow_at = min(elbow_at, 1.0) if elbow_at != float("inf") else -1.0
    if valid_e:
        elbow_min = min(valid_e.values())
        elbow_min_off = min(valid_e, key=lambda o: valid_e[o])
    else:
        elbow_min, elbow_min_off = -1.0, 0

    valid_b = {o: d for o, d in window_bbox.items() if d != float("inf")}
    bbox_at = window_bbox.get(0, float("inf"))
    bbox_at = min(bbox_at, 1.0) if bbox_at != float("inf") else -1.0
    if valid_b:
        bbox_min = min(valid_b.values())
        bbox_min = min(bbox_min, 1.0)
        bbox_min_off = min(valid_b, key=lambda o: valid_b[o])
    else:
        bbox_min, bbox_min_off = -1.0, 0

    # 10 ball features + 6 pose + 3 bbox + 6 action one-hot = 25
    act_norm = (action_type or "").upper()
    act_oh = [int(act_norm == a) for a in ACTIONS]
    return [
        dc_at, dc_max, float(dc_max_off),
        vel_at, vel_max, vel_min, float(vel_min_off),
        pre_vel, post_vel, post_vel / max(1e-4, pre_vel),
        wrist_at, wrist_min, float(wrist_min_off),
        elbow_at, elbow_min, float(elbow_min_off),
        bbox_at, bbox_min, float(bbox_min_off),
        *[float(x) for x in act_oh],
    ]


def refine_contacts_with_regressor(
    contacts: list,
    ball_positions: list,
    player_positions: list | None = None,
) -> int:
    """Apply learned regressor to refine each contact's frame.

    Mutates contacts in place. Bounded by MAX_SHIFT_FRAMES per contact and
    by adjacent-contact frame bounds (no overlap). Skips silently if model
    is not on disk. Returns the number of contacts whose frame changed.

    The model was trained on the OUTPUT of the heuristic direction-change
    snap, so this function should run AFTER _snap_contacts_to_direction_change_max
    (it learns to refine the heuristic's output further, not replace it).
    """
    bundle = _load_model()
    if bundle is None:
        return 0
    model = bundle["model"]
    if not contacts or not ball_positions:
        return 0

    balls = {bp.frame_number: bp for bp in ball_positions}
    pos_by_frame: dict[int, list[Any]] = {}
    if player_positions:
        for p in player_positions:
            pos_by_frame.setdefault(int(p.frame_number), []).append(p)

    sorted_contacts = sorted(contacts, key=lambda c: c.frame)
    n_changed = 0
    total_shift = 0.0

    for i, contact in enumerate(sorted_contacts):
        orig_frame = int(contact.frame)
        action_type = getattr(contact, "action_type", None)
        action_str: str | None
        if action_type is not None and hasattr(action_type, "name"):
            action_str = str(action_type.name)
        else:
            action_str = str(action_type) if action_type else None

        feats = _extract_features(balls, pos_by_frame, orig_frame, action_str)
        if feats is None:
            continue

        # Predict offset; round to nearest frame; clamp to bounds
        offset_raw = float(model.predict([feats])[0])
        offset = max(-MAX_SHIFT_FRAMES, min(MAX_SHIFT_FRAMES, round(offset_raw)))

        # Respect adjacent-contact bounds — don't cross neighbors
        lo_bound = sorted_contacts[i - 1].frame + 1 if i > 0 else orig_frame - MAX_SHIFT_FRAMES
        hi_bound = sorted_contacts[i + 1].frame - 1 if i + 1 < len(sorted_contacts) else orig_frame + MAX_SHIFT_FRAMES
        new_frame = max(lo_bound, min(hi_bound, orig_frame + offset))

        if new_frame == orig_frame:
            continue

        new_ball = balls.get(new_frame)
        if new_ball is None:
            continue

        contact.frame = new_frame
        contact.ball_x = new_ball.x
        contact.ball_y = new_ball.y
        n_changed += 1
        total_shift += abs(new_frame - orig_frame)

    if n_changed:
        contacts.sort(key=lambda c: c.frame)
        logger.info(
            f"Learned regressor refined {n_changed}/{len(contacts)} contact frames "
            f"(avg shift {total_shift / n_changed:.2f}f)",
        )
    return n_changed
