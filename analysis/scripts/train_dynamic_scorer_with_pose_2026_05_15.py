#!/usr/bin/env python3
"""Phase 2: train the dynamic scorer with bbox + pose features.

Reads:
  - Bbox features from DB (recomputed via _compute_features)
  - Pose features from analysis/reports/pose_features_2026_05_15/features.csv
  - GT labels from rally_action_ground_truth

Joins bbox + pose features per (rally_id, gt_frame, candidate_tid) and trains
per-action-type GBMs with LOO-video CV for honest measurement.

Outputs:
  analysis/reports/pose_features_2026_05_15/loo_results.md
  analysis/weights/dynamic_attribution_scorer_v2/ (if positive lift)

Usage:
    cd analysis && uv run python scripts/train_dynamic_scorer_with_pose_2026_05_15.py
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psycopg
from sklearn.ensemble import GradientBoostingClassifier

DB_DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5436/rallycut",
)
TRUSTED_CODENAMES = (
    "titi", "toto", "lulu", "wawa", "caco", "cece", "cici", "cuco",
    "gaga", "kaka", "kiki", "juju", "yeye", "keke",
    "gigi", "gugu", "mame", "meme", "mimi", "moma", "mumu",
)
POSE_CSV = Path("/Users/mario/Personal/Projects/RallyCut/analysis/reports/pose_features_2026_05_15/features.csv")
REPORT_DIR = Path("/Users/mario/Personal/Projects/RallyCut/analysis/reports/pose_features_2026_05_15")
FRAME_TOLERANCE = 5

# Bbox + dynamic features (10) — must match dynamic_attribution_scorer.py
BBOX_FEATURE_NAMES = [
    "bbox_dist", "bbox_area", "bbox_aspect_ratio", "bbox_inside_frame",
    "velocity_mag", "velocity_toward_ball",
    "top_y_at_contact", "top_y_change", "height_change",
    "same_as_prev",
]
POSE_FEATURE_NAMES = [
    "wrist_velocity_max", "wrist_to_ball_min", "body_orientation_diff",
    "arms_raised", "wrist_post_alignment", "pose_confidence_mean",
]
ALL_FEATURE_NAMES = BBOX_FEATURE_NAMES + POSE_FEATURE_NAMES


def _ball_dist_upper_quarter(p: dict, ball_x: float, ball_y: float) -> float:
    px = float(p.get("x", 0))
    py = float(p.get("y", 0)) - float(p.get("height", 0)) * 0.25
    return math.hypot(px - ball_x, py - ball_y)


def _find_pos(positions: list[dict], tid: int, frame: int, tolerance: int = 5) -> dict | None:
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


def _compute_bbox_features(
    positions: list[dict], tid: int, contact_frame: int,
    ball_x: float, ball_y: float, prev_action_tid: int,
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
    return [
        bbox_dist, bbox_area, bbox_aspect_ratio, inside,
        velocity_mag, velocity_toward_ball,
        y, top_y_change, height_change, same_as_prev,
    ]


@dataclass
class Row:
    video: str
    rally_id: str
    gt_frame: int
    action: str
    candidate_tid: int
    is_gt: bool
    features: list[float]


def load_pose_features() -> dict[tuple[str, int, int, int], list[float]]:
    """Load pose features keyed by (rally_id, gt_frame, candidate_tid)."""
    out = {}
    with open(POSE_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = (row["rally_id"], int(row["gt_frame"]), int(row["candidate_tid"]))
            out[key] = [
                float(row["wrist_velocity_max"]),
                float(row["wrist_to_ball_min"]),
                float(row["body_orientation_diff"]),
                float(row["arms_raised"]),
                float(row["wrist_post_alignment"]),
                float(row["pose_confidence_mean"]),
            ]
    return out


def build_dataset(pose_lookup: dict) -> list[Row]:
    rows: list[Row] = []
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute("""
            SELECT v.name, r.id, pt.primary_track_ids, pt.positions_json, pt.actions_json
            FROM videos v JOIN rallies r ON r.video_id = v.id
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE v.name = ANY(%s) AND r.status='CONFIRMED'
            ORDER BY v.name, r."order"
        """, [list(TRUSTED_CODENAMES)])
        rallies = cur.fetchall()
        for video_name, rally_id, primary_raw, positions_json, actions_json in rallies:
            if positions_json is None or not isinstance(primary_raw, list) or actions_json is None:
                continue
            positions = positions_json if isinstance(positions_json, list) else []
            primary_tids = [int(t) for t in primary_raw]
            aj = json.loads(actions_json) if isinstance(actions_json, str) else actions_json
            actions = aj.get("actions") or []
            gt_cur = conn.execute("""
                SELECT frame, action::text, resolved_track_id
                FROM rally_action_ground_truth
                WHERE rally_id = %s AND resolved_track_id IS NOT NULL
            """, [rally_id])
            for gt_frame, gt_action, gt_tid in gt_cur.fetchall():
                best_idx = -1; best_delta = FRAME_TOLERANCE + 1
                for i, a in enumerate(actions):
                    if a.get("action", "").upper() != gt_action.upper():
                        continue
                    d = abs(int(a.get("frame",-10**9)) - gt_frame)
                    if d < best_delta: best_delta=d; best_idx=i
                if best_idx < 0:
                    best_delta = FRAME_TOLERANCE + 1
                    for i, a in enumerate(actions):
                        d = abs(int(a.get("frame",-10**9)) - gt_frame)
                        if d < best_delta: best_delta=d; best_idx=i
                if best_idx < 0: continue
                pipe_a = actions[best_idx]
                pipe_frame = int(pipe_a.get("frame", -1))
                pipe_bx = pipe_a.get("ballX"); pipe_by = pipe_a.get("ballY")
                if pipe_bx is None or pipe_by is None: continue
                prev_action_tid = -1
                for j in range(best_idx - 1, -1, -1):
                    pa = actions[j]
                    if (pa.get("action") or "").upper() == "UNKNOWN": continue
                    pa_tid = int(pa.get("playerTrackId", -1))
                    if pa_tid >= 0: prev_action_tid = pa_tid; break
                for tid in primary_tids:
                    bbox_feats = _compute_bbox_features(
                        positions, tid, pipe_frame, float(pipe_bx), float(pipe_by),
                        prev_action_tid=prev_action_tid,
                    )
                    if bbox_feats is None: continue
                    # Look up pose features using GT frame (pose features were extracted at GT frame)
                    pose_feats = pose_lookup.get((rally_id, int(gt_frame), tid))
                    if pose_feats is None:
                        # Pose features missing — use zeros (track might not be enriched)
                        pose_feats = [0.0] * len(POSE_FEATURE_NAMES)
                    rows.append(Row(
                        video=video_name, rally_id=rally_id, gt_frame=gt_frame,
                        action=gt_action.upper(), candidate_tid=tid,
                        is_gt=(tid == gt_tid),
                        features=bbox_feats + pose_feats,
                    ))
    return rows


def evaluate_loo(rows: list[Row], use_pose: bool) -> dict:
    """Per action type, leave-one-video-out CV. If use_pose=False, mask pose
    features to 0 to compare against bbox-only baseline.
    """
    by_action: dict[str, list[Row]] = defaultdict(list)
    for r in rows:
        by_action[r.action].append(r)
    out: dict = {}
    for action, action_rows in by_action.items():
        videos = sorted({r.video for r in action_rows})
        n_gt = n_correct = 0
        importances = []
        for hold_v in videos:
            train_rows = [r for r in action_rows if r.video != hold_v]
            test_rows = [r for r in action_rows if r.video == hold_v]
            if not train_rows or not test_rows: continue
            X_train = np.array([r.features for r in train_rows])
            y_train = np.array([1 if r.is_gt else 0 for r in train_rows])
            if not use_pose:
                # mask the 6 pose feature columns
                X_train[:, -6:] = 0.0
            if y_train.sum() == 0 or y_train.sum() == len(y_train): continue
            clf = GradientBoostingClassifier(
                n_estimators=80, max_depth=3, learning_rate=0.05, random_state=42,
            )
            clf.fit(X_train, y_train)
            importances.append(clf.feature_importances_)
            grouped: dict = defaultdict(list)
            for r in test_rows:
                grouped[(r.rally_id, r.gt_frame)].append(r)
            for (_rid, _f), group in grouped.items():
                if not group or not any(r.is_gt for r in group): continue
                X_test = np.array([r.features for r in group])
                if not use_pose:
                    X_test[:, -6:] = 0.0
                probs = clf.predict_proba(X_test)[:, 1]
                pred_idx = int(np.argmax(probs))
                gt_idx = next(i for i, r in enumerate(group) if r.is_gt)
                n_gt += 1
                if pred_idx == gt_idx: n_correct += 1
        out[action] = {
            "n": n_gt, "correct": n_correct,
            "rate": (100*n_correct/max(1,n_gt)),
            "importances": (
                np.mean(importances, axis=0).tolist() if importances else None
            ),
        }
    return out


def main() -> int:
    print("Phase 2: train scorer with bbox + pose features", flush=True)
    print(f"  Pose CSV: {POSE_CSV}", flush=True)
    if not POSE_CSV.exists():
        print("ERROR: pose CSV missing — run Phase 1 first", flush=True)
        return 1

    pose_lookup = load_pose_features()
    print(f"  Loaded {len(pose_lookup)} pose-feature rows", flush=True)

    print("Building combined dataset…", flush=True)
    rows = build_dataset(pose_lookup)
    print(f"  {len(rows)} candidate rows", flush=True)
    by_action = defaultdict(int)
    for r in rows:
        by_action[r.action] += 1
    for a, n in sorted(by_action.items()):
        print(f"    {a:10s} {n}", flush=True)

    print(flush=True)
    print("LOO CV (BBOX features only, no pose)…", flush=True)
    bbox_only = evaluate_loo(rows, use_pose=False)
    print(flush=True)
    print("LOO CV (BBOX + POSE features)…", flush=True)
    with_pose = evaluate_loo(rows, use_pose=True)

    print(flush=True)
    print("=" * 90, flush=True)
    print(f"{'action':10s} {'n':>5s} {'bbox-only':>14s} {'w/ pose':>14s} {'Δ pose lift':>14s}", flush=True)
    print("=" * 90, flush=True)
    total_n = total_b = total_p = 0
    for action in ["SERVE", "RECEIVE", "SET", "ATTACK", "DIG", "BLOCK"]:
        if action not in bbox_only: continue
        b = bbox_only[action]
        p = with_pose[action]
        if b["n"] == 0: continue
        delta = p["correct"] - b["correct"]
        delta_pp = 100 * delta / b["n"]
        print(f"{action:10s} {b['n']:>5d} {b['rate']:>13.1f}% {p['rate']:>13.1f}% "
              f"{delta:>+5d} ({delta_pp:+.1f}pp)", flush=True)
        total_n += b["n"]
        total_b += b["correct"]
        total_p += p["correct"]
    print("=" * 90, flush=True)
    delta = total_p - total_b
    delta_pp = 100 * delta / max(1, total_n)
    print(f"{'TOTAL':10s} {total_n:>5d} {100*total_b/max(1,total_n):>13.1f}% "
          f"{100*total_p/max(1,total_n):>13.1f}% {delta:>+5d} ({delta_pp:+.2f}pp)", flush=True)

    print(flush=True)
    print("=" * 90, flush=True)
    print("Per-action feature importance (with pose, mean across LOO folds)", flush=True)
    print("=" * 90, flush=True)
    print(f"{'action':10s} " + " ".join(f"{f[:11]:>11s}" for f in ALL_FEATURE_NAMES), flush=True)
    for action in ["SERVE", "RECEIVE", "SET", "ATTACK", "DIG", "BLOCK"]:
        if action not in with_pose: continue
        imp = with_pose[action]["importances"]
        if imp is None: continue
        print(f"{action:10s} " + " ".join(f"{v:>11.3f}" for v in imp), flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
