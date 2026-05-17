#!/usr/bin/env python3
"""Train the per-action-type dynamic attribution scorer on the FULL trusted GT
corpus and save the models to disk for production use.

Usage:
    cd analysis && uv run python scripts/train_and_save_dynamic_scorer_2026_05_14.py

Outputs:
    analysis/weights/dynamic_attribution_scorer/{ACTION}_v1.joblib (one per action type)
    analysis/weights/dynamic_attribution_scorer/manifest.json (feature names, version, training corpus)

For honest measurement use train_dynamic_attribution_scorer_2026_05_14.py (LOO CV).
This script is for production: use ALL available labeled data.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import psycopg
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rallycut.tracking.match_tracker import build_match_team_assignments  # noqa: E402

# Net-crossing actions for team-chain forward propagation. Mirrors
# `_NET_CROSSING_ACTIONS` in rallycut/tracking/action_classifier.py.
_NET_CROSSING = {"SERVE", "ATTACK"}


def _compute_expected_teams_train(
    actions: list[dict], team_assignments: dict[int, int] | None,
) -> list[int | None]:
    """Mirror `_compute_expected_teams` from action_classifier.py for training.

    Returns parallel list of team-chain-derived expected team (0 or 1) per
    action; None where the chain can't be determined (no SERVE, or before
    the seeding SERVE, or chain broken by UNKNOWN / non-seed synthetic).
    """
    expected: list[int | None] = [None] * len(actions)
    if not team_assignments:
        return expected
    serve_team: int | None = None
    for a in actions:
        if (a.get("action") or "").upper() != "SERVE":
            continue
        tid = int(a.get("playerTrackId", -1))
        if tid < 0:
            continue
        st = team_assignments.get(tid)
        if st is not None:
            serve_team = int(st)
            break
    if serve_team is None:
        return expected
    current_team = serve_team
    for i, a in enumerate(actions):
        at = (a.get("action") or "").upper()
        if at == "UNKNOWN" or not at:
            continue
        if a.get("synthetic") or a.get("isSynthetic"):
            if at == "SERVE":
                expected[i] = serve_team
                current_team = 1 - serve_team
            continue
        expected[i] = current_team
        if at in _NET_CROSSING:
            current_team = 1 - current_team
    return expected

DB_DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5436/rallycut",
)
TRUSTED_CODENAMES = (
    # Original trusted-14 (player-attribution GT validated 2026-05-14)
    "titi", "toto", "lulu", "wawa", "caco", "cece", "cici", "cuco",
    "gaga", "kaka", "kiki", "juju", "yeye", "keke",
    # 7 added 2026-05-15 — trusted-21 corpus
    "gigi", "gugu", "mame", "meme", "mimi", "moma", "mumu",
    # 8 added 2026-05-17 — trusted-29 corpus
    "papa", "pepe", "pipi", "popo", "pupu", "veve", "vivi", "vovo",
)
FRAME_TOLERANCE = 5
FEATURE_NAMES = [
    "bbox_dist", "bbox_area", "bbox_aspect_ratio", "bbox_inside_frame",
    "velocity_mag", "velocity_toward_ball",
    "top_y_at_contact", "top_y_change", "height_change",
    "same_as_prev",  # 1.0 if candidate.tid == previous action's playerTrackId else 0.0
    # Pose dynamics (v2, 2026-05-15)
    "wrist_velocity_max", "wrist_to_ball_min", "body_orientation_diff",
    "arms_raised", "wrist_post_alignment", "pose_confidence_mean",
    # v2.1 — target ATTACK contest
    "wrist_y_velocity",
    # v3 (2026-05-17) — team-awareness; 0.5 = uninformative
    "team_matches_expected",
]
# COCO 17-keypoint indices
KPT_LEFT_SHOULDER = 5
KPT_RIGHT_SHOULDER = 6
KPT_LEFT_WRIST = 9
KPT_RIGHT_WRIST = 10
KPT_VIS_THRESHOLD = 0.3
POSE_WINDOW = 5
ACTION_TYPES = ["SERVE", "RECEIVE", "SET", "ATTACK", "DIG", "BLOCK"]
MODEL_VERSION = "v1"
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "weights" / "dynamic_attribution_scorer"


def _ball_dist_upper_quarter(p: dict, ball_x: float, ball_y: float) -> float:
    px = float(p.get("x", 0))
    py = float(p.get("y", 0)) - float(p.get("height", 0)) * 0.25
    return math.hypot(px - ball_x, py - ball_y)


def _find_pos(positions: list[dict], tid: int, frame: int, tolerance: int = 5) -> dict | None:
    """Note: tolerance widened from 2 to 5 on 2026-05-14 to address wawa
    regression (3 of 5 wawa regressions were cases where the GT player was
    tracked at ±3..±5 of contact but not ±2, making the scorer unable to
    pick GT). Must stay in lockstep with dynamic_attribution_scorer.py's
    _find_pos tolerance."""
    best = None
    best_delta = tolerance + 1
    for p in positions:
        if int(p.get("trackId", -1)) != tid:
            continue
        f = int(p.get("frameNumber", -1))
        delta = abs(f - frame)
        if delta < best_delta:
            best_delta = delta
            best = p
    return best


def _wrist_xy_from_dict(p: dict, which: str) -> tuple[float, float, float] | None:
    kps = p.get("keypoints")
    if not kps or len(kps) < 17:
        return None
    idx = KPT_LEFT_WRIST if which == "left" else KPT_RIGHT_WRIST
    kx, ky, kc = kps[idx]
    if kc < KPT_VIS_THRESHOLD:
        return None
    return float(kx), float(ky), float(kc)


def _shoulder_xy_from_dict(p: dict, which: str) -> tuple[float, float, float] | None:
    kps = p.get("keypoints")
    if not kps or len(kps) < 17:
        return None
    idx = KPT_LEFT_SHOULDER if which == "left" else KPT_RIGHT_SHOULDER
    kx, ky, kc = kps[idx]
    if kc < KPT_VIS_THRESHOLD:
        return None
    return float(kx), float(ky), float(kc)


def _compute_pose_features(
    positions: list[dict], tid: int, contact_frame: int,
    ball_x: float, ball_y: float,
    post_ball_x: float | None, post_ball_y: float | None,
) -> dict[str, float]:
    """Compute pose features from keypoints stored in positions_json.
    Mirrors dynamic_attribution_scorer._compute_pose_features. Keep these
    two functions in lockstep — same feature order, same logic."""
    track_positions = sorted(
        [p for p in positions if int(p.get("trackId", -1)) == tid
         and abs(int(p.get("frameNumber", -1)) - contact_frame) <= POSE_WINDOW],
        key=lambda p: int(p.get("frameNumber", -1)),
    )
    wrist_pos: dict[int, tuple[float, float]] = {}
    confs: list[float] = []
    arms_raised_at_contact = 0.0
    for p in track_positions:
        fnum = int(p.get("frameNumber", -1))
        lw = _wrist_xy_from_dict(p, "left")
        rw = _wrist_xy_from_dict(p, "right")
        best_w: tuple[float, float] | None = None
        best_d = float("inf")
        for w in (lw, rw):
            if w is None: continue
            d = math.hypot(w[0] - ball_x, w[1] - ball_y)
            if d < best_d: best_d = d; best_w = (w[0], w[1])
            confs.append(w[2])
        if best_w is not None: wrist_pos[fnum] = best_w
        ls = _shoulder_xy_from_dict(p, "left")
        rs = _shoulder_xy_from_dict(p, "right")
        for s in (ls, rs):
            if s is not None: confs.append(s[2])
        if abs(fnum - contact_frame) <= 2:
            if lw and rw and ls and rs:
                if lw[1] < ls[1] and rw[1] < rs[1]: arms_raised_at_contact = 1.0

    if not wrist_pos and not confs:
        return {
            "wrist_velocity_max": 0.0, "wrist_to_ball_min": 1.0,
            "body_orientation_diff": math.pi, "arms_raised": 0.0,
            "wrist_post_alignment": 0.0, "pose_confidence_mean": 0.0,
            "wrist_y_velocity": 0.0,
        }
    sorted_frames = sorted(wrist_pos.keys())
    wrist_velocity_max = 0.0
    wrist_y_velocity_at_contact = 0.0
    best_vel = None
    for i in range(len(sorted_frames) - 1):
        f1, f2 = sorted_frames[i], sorted_frames[i + 1]
        if f2 - f1 > 3: continue
        x1, y1 = wrist_pos[f1]; x2, y2 = wrist_pos[f2]
        gap = max(1, f2 - f1)
        d = math.hypot(x2 - x1, y2 - y1) / gap
        if d > wrist_velocity_max: wrist_velocity_max = d
        if min(abs(f1 - contact_frame), abs(f2 - contact_frame)) <= 2:
            dy_pf = (y2 - y1) / gap
            if abs(dy_pf) > abs(wrist_y_velocity_at_contact): wrist_y_velocity_at_contact = dy_pf
        # Track best velocity vector for post-alignment computation
        dx_t, dy_t = x2 - x1, y2 - y1
        d_t = math.hypot(dx_t, dy_t)
        if best_vel is None or d_t > best_vel[2]:
            best_vel = (dx_t, dy_t, d_t)

    wrist_to_ball_min = float("inf")
    for f, (wx, wy) in wrist_pos.items():
        if abs(f - contact_frame) <= 2:
            d = math.hypot(wx - ball_x, wy - ball_y)
            if d < wrist_to_ball_min: wrist_to_ball_min = d
    if not math.isfinite(wrist_to_ball_min): wrist_to_ball_min = 1.0

    body_orientation_diff = math.pi
    p_contact = next((p for p in track_positions if abs(int(p.get("frameNumber",-1)) - contact_frame) <= 1), None)
    if p_contact is not None:
        ls = _shoulder_xy_from_dict(p_contact, "left")
        rs = _shoulder_xy_from_dict(p_contact, "right")
        if ls and rs:
            sx = rs[0] - ls[0]; sy = rs[1] - ls[1]
            facing_x = -sy; facing_y = sx
            torso_x = (ls[0] + rs[0]) / 2; torso_y = (ls[1] + rs[1]) / 2
            to_ball_x = ball_x - torso_x; to_ball_y = ball_y - torso_y
            mag_f = math.hypot(facing_x, facing_y) + 1e-6
            mag_b = math.hypot(to_ball_x, to_ball_y) + 1e-6
            cos_theta = (facing_x * to_ball_x + facing_y * to_ball_y) / (mag_f * mag_b)
            cos_theta = max(-1.0, min(1.0, cos_theta))
            body_orientation_diff = math.acos(cos_theta)

    wrist_post_alignment = 0.0
    if (post_ball_x is not None and post_ball_y is not None
            and best_vel is not None and best_vel[2] > 0):
        ball_dx = post_ball_x - ball_x
        ball_dy = post_ball_y - ball_y
        ball_mag = math.hypot(ball_dx, ball_dy) + 1e-6
        wrist_post_alignment = ((best_vel[0] * ball_dx + best_vel[1] * ball_dy)
                                 / (best_vel[2] * ball_mag))

    pose_confidence_mean = (sum(confs) / len(confs)) if confs else 0.0
    return {
        "wrist_velocity_max": wrist_velocity_max,
        "wrist_to_ball_min": wrist_to_ball_min,
        "body_orientation_diff": body_orientation_diff,
        "arms_raised": arms_raised_at_contact,
        "wrist_post_alignment": wrist_post_alignment,
        "pose_confidence_mean": pose_confidence_mean,
        "wrist_y_velocity": wrist_y_velocity_at_contact,
    }


def _team_match_feature_train(
    tid: int,
    expected_team: int | None,
    team_assignments: dict[int, int] | None,
) -> float:
    """Mirrors dynamic_attribution_scorer._team_match_feature exactly."""
    if expected_team is None or team_assignments is None:
        return 0.5
    cand_team = team_assignments.get(tid)
    if cand_team is None:
        return 0.5
    return 1.0 if int(cand_team) == int(expected_team) else 0.0


def _compute_features(
    positions: list[dict], tid: int, contact_frame: int,
    ball_x: float, ball_y: float,
    prev_action_tid: int = -1,
    post_ball_x: float | None = None,
    post_ball_y: float | None = None,
    expected_team: int | None = None,
    team_assignments: dict[int, int] | None = None,
) -> list[float] | None:
    p_at = _find_pos(positions, tid, contact_frame, tolerance=5)
    if p_at is None:
        return None
    p_prev = _find_pos(positions, tid, contact_frame - 5, tolerance=5)
    p_next = _find_pos(positions, tid, contact_frame + 5, tolerance=5)
    p_pre_extend = _find_pos(positions, tid, contact_frame - 3, tolerance=5)
    p_post_extend = _find_pos(positions, tid, contact_frame + 3, tolerance=5)
    x = float(p_at.get("x", 0))
    y = float(p_at.get("y", 0))
    w = float(p_at.get("width", 0))
    h = float(p_at.get("height", 0))
    bbox_dist = _ball_dist_upper_quarter(p_at, ball_x, ball_y)
    bbox_area = w * h
    bbox_aspect_ratio = w / max(h, 1e-6)
    inside = 1.0 if (x >= 0 and y >= 0 and x + w <= 1.0 and y + h <= 1.0) else 0.0
    if p_prev and p_next:
        cx_prev = float(p_prev.get("x", 0)) + float(p_prev.get("width", 0)) / 2
        cy_prev = float(p_prev.get("y", 0)) + float(p_prev.get("height", 0)) / 2
        cx_next = float(p_next.get("x", 0)) + float(p_next.get("width", 0)) / 2
        cy_next = float(p_next.get("y", 0)) + float(p_next.get("height", 0)) / 2
        dx = cx_next - cx_prev
        dy = cy_next - cy_prev
        velocity_mag = math.hypot(dx, dy)
        cx_at = x + w / 2
        cy_at = y + h / 2
        to_ball_x = ball_x - cx_at
        to_ball_y = ball_y - cy_at
        to_ball_mag = math.hypot(to_ball_x, to_ball_y) + 1e-6
        velocity_toward_ball = (dx * to_ball_x + dy * to_ball_y) / to_ball_mag
    else:
        velocity_mag = 0.0
        velocity_toward_ball = 0.0
    top_y_change = y - float(p_prev.get("y", y)) if p_prev else 0.0
    if p_pre_extend and p_post_extend:
        height_change = float(p_post_extend.get("height", h)) - float(p_pre_extend.get("height", h))
    else:
        height_change = 0.0
    same_as_prev = 1.0 if (prev_action_tid >= 0 and tid == prev_action_tid) else 0.0
    pose = _compute_pose_features(
        positions, tid, contact_frame, ball_x, ball_y,
        post_ball_x, post_ball_y,
    )
    team_match = _team_match_feature_train(tid, expected_team, team_assignments)
    return [
        bbox_dist, bbox_area, bbox_aspect_ratio, inside,
        velocity_mag, velocity_toward_ball,
        y, top_y_change, height_change,
        same_as_prev,
        pose["wrist_velocity_max"], pose["wrist_to_ball_min"],
        pose["body_orientation_diff"], pose["arms_raised"],
        pose["wrist_post_alignment"], pose["pose_confidence_mean"],
        pose["wrist_y_velocity"],
        team_match,
    ]


@dataclass
class CandidateRow:
    action: str
    candidate_tid: int
    is_gt: bool
    features: list[float]
    # Provenance fields (added 2026-05-17 for LOO CV) — pure metadata,
    # not used in training. Allows the LOO measurement script to
    # leave-one-video-out and group by GT row identity.
    video: str = ""
    rally_id: str = ""
    gt_frame: int = -1


def build_dataset() -> list[CandidateRow]:
    """Build training dataset using PRODUCTION-MATCHED feature extraction.

    For each GT row, find the corresponding pipeline action (prefer same
    action_type within ±5 frames; else closest by frame). Use the pipeline
    action's `frame` and `ballX/ballY` as the input — NOT the GT snapshot.
    Label with GT.resolved_track_id.

    Why: at inference time the contact-detector will emit a frame + ball
    position from its own detection. Training distribution must match that
    or the model collapses on out-of-distribution serves (where the
    synth-serve placement differs significantly from the GT-labeled toss).
    """
    rows: list[CandidateRow] = []
    n_gt_seen = 0
    n_gt_matched = 0
    n_gt_skipped_no_match = 0
    with psycopg.connect(DB_DSN) as conn:
        # Build per-rally team_assignments once for the v3 team-awareness
        # feature, mirroring what redetect_all_actions does at inference time
        # (so training distribution matches inference distribution).
        match_teams_by_rally: dict[str, dict[int, int]] = {}
        vcur = conn.execute(
            "SELECT v.match_analysis_json FROM videos v "
            "WHERE v.name = ANY(%s) AND v.match_analysis_json IS NOT NULL",
            [list(TRUSTED_CODENAMES)],
        )
        for (mj_raw,) in vcur.fetchall():
            if not mj_raw:
                continue
            match_teams_by_rally.update(
                build_match_team_assignments(mj_raw, min_confidence=0.0)
            )
        cur = conn.execute(
            """
            SELECT v.name, r.id, pt.primary_track_ids, pt.positions_json,
                   pt.actions_json
            FROM videos v JOIN rallies r ON r.video_id = v.id
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE v.name = ANY(%s) AND r.status = 'CONFIRMED'
            ORDER BY v.name, r."order"
            """,
            [list(TRUSTED_CODENAMES)],
        )
        rallies = cur.fetchall()
        for video_name, rally_id, primary_raw, positions_json, actions_json in rallies:
            if positions_json is None or not isinstance(primary_raw, list) or actions_json is None:
                continue
            positions = positions_json if isinstance(positions_json, list) else []
            primary_tids = [int(t) for t in primary_raw]
            aj = json.loads(actions_json) if isinstance(actions_json, str) else actions_json
            actions = aj.get("actions") or []
            # v3: team_assignments + expected_teams chain for this rally
            team_assignments = match_teams_by_rally.get(str(rally_id))
            expected_teams = _compute_expected_teams_train(actions, team_assignments)
            # Load ball_positions_json for post-contact ball lookup (wrist_post_alignment)
            bcur = conn.execute(
                "SELECT ball_positions_json FROM player_tracks WHERE rally_id=%s",
                [rally_id],
            )
            ball_row = bcur.fetchone()
            ball_positions = (
                ball_row[0] if (ball_row and isinstance(ball_row[0], list)) else []
            )
            ball_by_frame = {
                int(b.get("frameNumber", -1)): b for b in ball_positions
            }
            gt_cur = conn.execute(
                """
                SELECT frame, action::text, resolved_track_id
                FROM rally_action_ground_truth
                WHERE rally_id = %s AND resolved_track_id IS NOT NULL
                """,
                [rally_id],
            )
            for gt_frame, gt_action, gt_tid in gt_cur.fetchall():
                n_gt_seen += 1
                # Match to pipeline action: prefer same-type within ±5, else
                # closest by frame within ±5.
                best_idx = -1
                best_delta = 6
                for i, a in enumerate(actions):
                    if a.get("action", "").upper() != gt_action.upper():
                        continue
                    delta = abs(int(a.get("frame", -10**9)) - gt_frame)
                    if delta < best_delta:
                        best_delta = delta
                        best_idx = i
                if best_idx < 0:
                    best_delta = 6
                    for i, a in enumerate(actions):
                        delta = abs(int(a.get("frame", -10**9)) - gt_frame)
                        if delta < best_delta:
                            best_delta = delta
                            best_idx = i
                if best_idx < 0:
                    # No matching pipeline action — this is a contact-FN at the
                    # training distribution. Skip it; the model wouldn't see
                    # such a case at inference (it only runs when a contact
                    # was detected).
                    n_gt_skipped_no_match += 1
                    continue
                pipe_a = actions[best_idx]
                pipe_frame = int(pipe_a.get("frame", -1))
                pipe_ball_x = pipe_a.get("ballX")
                pipe_ball_y = pipe_a.get("ballY")
                if pipe_ball_x is None or pipe_ball_y is None:
                    n_gt_skipped_no_match += 1
                    continue
                # Find the previous action's playerTrackId for the
                # same_as_prev feature. Skips UNKNOWN actions and those
                # with player_track_id < 0.
                prev_action_tid = -1
                for j in range(best_idx - 1, -1, -1):
                    pa = actions[j]
                    pa_at = (pa.get("action") or "").upper()
                    if pa_at == "UNKNOWN":
                        continue
                    pa_tid = int(pa.get("playerTrackId", -1))
                    if pa_tid >= 0:
                        prev_action_tid = pa_tid
                        break
                n_gt_matched += 1
                # Post-contact ball for wrist_post_alignment (first detection
                # in f+5..f+15 window).
                post_ball_x = post_ball_y = None
                for offset in range(5, 16):
                    b = ball_by_frame.get(pipe_frame + offset)
                    if b is not None:
                        post_ball_x = float(b.get("x") or 0)
                        post_ball_y = float(b.get("y") or 0)
                        break
                # Use PIPELINE's frame + ball position (production-matched).
                # v3.1 (2026-05-17): mask team_matches_expected for SET to
                # avoid the cascade regression (-2.1pp matched under v3).
                # Must stay in lockstep with _TEAM_FEATURE_MASKED_ACTIONS in
                # action_classifier.py::_apply_dynamic_scorer_attribution.
                if gt_action.upper() == "SET":
                    expected_team = None
                else:
                    expected_team = expected_teams[best_idx]
                for tid in primary_tids:
                    feats = _compute_features(
                        positions, tid, pipe_frame,
                        float(pipe_ball_x), float(pipe_ball_y),
                        prev_action_tid=prev_action_tid,
                        post_ball_x=post_ball_x,
                        post_ball_y=post_ball_y,
                        expected_team=expected_team,
                        team_assignments=team_assignments,
                    )
                    if feats is None:
                        continue
                    rows.append(CandidateRow(
                        action=gt_action.upper(),
                        candidate_tid=tid,
                        is_gt=(tid == gt_tid),
                        features=feats,
                        video=video_name,
                        rally_id=str(rally_id),
                        gt_frame=int(gt_frame),
                    ))
    print(f"  GT rows: {n_gt_seen} seen, {n_gt_matched} matched to pipeline action, "
          f"{n_gt_skipped_no_match} skipped (no matching pipeline action / contact-FN)",
          flush=True)
    return rows


def main() -> int:
    print(f"Output dir: {OUTPUT_DIR}", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Building feature dataset from full trusted-{len(TRUSTED_CODENAMES)} corpus…", flush=True)
    rows = build_dataset()
    by_action = defaultdict(list)
    for r in rows:
        by_action[r.action].append(r)
    total_pos = sum(1 for r in rows if r.is_gt)
    print(f"  {len(rows)} candidate rows, {total_pos} positive labels", flush=True)
    for a, action_rows in sorted(by_action.items()):
        n_pos = sum(1 for r in action_rows if r.is_gt)
        print(f"    {a:10s} {len(action_rows):>5d} rows, {n_pos:>4d} positives", flush=True)

    print(flush=True)
    print("Training per-action-type GBMs…", flush=True)
    manifest: dict[str, Any] = {
        "version": MODEL_VERSION,
        "feature_names": FEATURE_NAMES,
        "training_corpus": list(TRUSTED_CODENAMES),
        "models": {},
        "trained_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "training_notes": (
            f"Trained on FULL trusted-{len(TRUSTED_CODENAMES)} corpus (no hold-out). "
            "For honest LOO CV measurements see "
            "scripts/train_dynamic_attribution_scorer_2026_05_14.py."
        ),
    }
    for action in ACTION_TYPES:
        action_rows = by_action.get(action, [])
        if not action_rows:
            print(f"  {action:10s} NO ROWS — skipping", flush=True)
            continue
        X = np.array([r.features for r in action_rows])
        y = np.array([1 if r.is_gt else 0 for r in action_rows])
        if y.sum() == 0 or y.sum() == len(y):
            print(f"  {action:10s} DEGENERATE LABELS — skipping", flush=True)
            continue
        clf = GradientBoostingClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.05, random_state=42,
        )
        clf.fit(X, y)
        out_path = OUTPUT_DIR / f"{action}_{MODEL_VERSION}.joblib"
        joblib.dump(clf, out_path)
        manifest["models"][action] = {
            "path": out_path.name,
            "n_rows": len(action_rows),
            "n_positives": int(y.sum()),
            "feature_importances": clf.feature_importances_.tolist(),
        }
        print(f"  {action:10s} → {out_path.name} ({len(action_rows)} rows, "
              f"{int(y.sum())} positives)", flush=True)

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
