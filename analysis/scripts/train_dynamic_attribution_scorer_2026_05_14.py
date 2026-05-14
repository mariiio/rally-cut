#!/usr/bin/env python3
"""Train per-action-type GBM scorer that combines static + dynamic features
for picking the correct player among candidates.

Pipeline:
  1. Extract features per (rally, GT_row, candidate) across the trusted-14
     corpus. 4 candidates per GT row × N GT rows × 8 features.
  2. For each action type, train a GBM with leave-one-video-out CV.
     - Per-fold: train on N-1 videos, test on held-out video.
     - Per GT row in held-out: score all candidates, pick argmax.
     - Compute rank-1 accuracy (matches gt_resolved_tid).
  3. Compare with bbox_dist-only baseline AND current pipeline accuracy.

Read-only on DB. Trains models in-memory; doesn't persist them.

Usage:
    cd analysis && uv run python scripts/train_dynamic_attribution_scorer_2026_05_14.py
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

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
)
FRAME_TOLERANCE = 5

# Per-candidate feature names, in fixed order.
FEATURE_NAMES = [
    "bbox_dist",            # static, current picker uses this
    "bbox_area",            # static
    "bbox_aspect_ratio",    # static
    "bbox_inside_frame",    # static
    "velocity_mag",         # dynamic
    "velocity_toward_ball", # dynamic
    "top_y_at_contact",     # dynamic
    "top_y_change",         # dynamic
    "height_change",        # dynamic
]


@dataclass
class CandidateRow:
    video: str
    rally_id: str
    gt_frame: int
    action: str
    candidate_tid: int
    is_gt: bool
    features: list[float]
    pipeline_picked: bool  # True if the actual pipeline picked this candidate


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


def _compute_features(
    positions: list[dict],
    tid: int,
    contact_frame: int,
    ball_x: float,
    ball_y: float,
) -> list[float] | None:
    p_at = _find_pos(positions, tid, contact_frame, tolerance=2)
    if p_at is None:
        return None
    p_prev = _find_pos(positions, tid, contact_frame - 5, tolerance=2)
    p_next = _find_pos(positions, tid, contact_frame + 5, tolerance=2)
    p_pre_extend = _find_pos(positions, tid, contact_frame - 3, tolerance=2)
    p_post_extend = _find_pos(positions, tid, contact_frame + 3, tolerance=2)

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

    return [
        bbox_dist, bbox_area, bbox_aspect_ratio, inside,
        velocity_mag, velocity_toward_ball,
        y,  # top_y_at_contact
        top_y_change, height_change,
    ]


def build_dataset() -> list[CandidateRow]:
    rows: list[CandidateRow] = []
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            """
            SELECT v.name, r.id, pt.primary_track_ids, pt.positions_json, pt.actions_json
            FROM videos v
            JOIN rallies r ON r.video_id = v.id
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
            gt_cur = conn.execute(
                """
                SELECT frame, action::text, resolved_track_id,
                       snapshot_ball_x, snapshot_ball_y
                FROM rally_action_ground_truth
                WHERE rally_id = %s AND resolved_track_id IS NOT NULL
                  AND snapshot_ball_x IS NOT NULL AND snapshot_ball_y IS NOT NULL
                ORDER BY frame
                """,
                [rally_id],
            )
            for gt_frame, gt_action, gt_tid, ball_x, ball_y in gt_cur.fetchall():
                # Match pipeline action: prefer same-type within tolerance,
                # otherwise closest-by-frame regardless of type.
                best_idx = -1
                best_delta = FRAME_TOLERANCE + 1
                for i, a in enumerate(actions):
                    if a.get("action", "").upper() != gt_action.upper():
                        continue
                    delta = abs(int(a.get("frame", -10**9)) - gt_frame)
                    if delta < best_delta:
                        best_delta = delta
                        best_idx = i
                if best_idx < 0:
                    best_delta = FRAME_TOLERANCE + 1
                    for i, a in enumerate(actions):
                        delta = abs(int(a.get("frame", -10**9)) - gt_frame)
                        if delta < best_delta:
                            best_delta = delta
                            best_idx = i
                pipeline_tid = -1
                if best_idx >= 0:
                    pipeline_tid = int(actions[best_idx].get("playerTrackId", -1))
                for tid in primary_tids:
                    feats = _compute_features(
                        positions, tid, gt_frame, ball_x, ball_y,
                    )
                    if feats is None:
                        continue
                    rows.append(CandidateRow(
                        video=video_name,
                        rally_id=rally_id,
                        gt_frame=gt_frame,
                        action=gt_action.upper(),
                        candidate_tid=tid,
                        is_gt=(tid == gt_tid),
                        features=feats,
                        pipeline_picked=(tid == pipeline_tid),
                    ))
    return rows


def evaluate_loo(rows: list[CandidateRow]) -> dict[str, dict[str, Any]]:
    """Per action type, leave-one-video-out CV: train on N-1 videos,
    evaluate rank-1 accuracy on held-out video.

    Returns per-action stats.
    """
    by_action: dict[str, list[CandidateRow]] = defaultdict(list)
    for r in rows:
        by_action[r.action].append(r)

    out: dict[str, dict[str, Any]] = {}
    for action, action_rows in by_action.items():
        videos = sorted({r.video for r in action_rows})
        n_gt = 0
        n_correct_scorer = 0
        n_correct_pipeline = 0
        n_correct_bbox_dist = 0
        feature_importances: list[np.ndarray] = []
        for hold_v in videos:
            train_rows = [r for r in action_rows if r.video != hold_v]
            test_rows = [r for r in action_rows if r.video == hold_v]
            if not train_rows or not test_rows:
                continue
            X_train = np.array([r.features for r in train_rows])
            y_train = np.array([1 if r.is_gt else 0 for r in train_rows])
            if y_train.sum() == 0 or y_train.sum() == len(y_train):
                # Degenerate fold (all-positive or all-negative)
                continue
            clf = GradientBoostingClassifier(
                n_estimators=80, max_depth=3, learning_rate=0.05,
                random_state=42,
            )
            clf.fit(X_train, y_train)
            feature_importances.append(clf.feature_importances_)
            # Group test rows by GT row (rally_id + gt_frame).
            grouped: dict[tuple[str, int], list[CandidateRow]] = defaultdict(list)
            for r in test_rows:
                grouped[(r.rally_id, r.gt_frame)].append(r)
            for (_rid, _f), group in grouped.items():
                if not group:
                    continue
                if not any(r.is_gt for r in group):
                    continue
                X_test = np.array([r.features for r in group])
                probs = clf.predict_proba(X_test)[:, 1]
                pred_idx = int(np.argmax(probs))
                bbox_dist_idx = int(np.argmin(X_test[:, 0]))
                pipeline_idx_list = [i for i, r in enumerate(group) if r.pipeline_picked]
                gt_idx = next(i for i, r in enumerate(group) if r.is_gt)
                n_gt += 1
                if pred_idx == gt_idx:
                    n_correct_scorer += 1
                if bbox_dist_idx == gt_idx:
                    n_correct_bbox_dist += 1
                if pipeline_idx_list and pipeline_idx_list[0] == gt_idx:
                    n_correct_pipeline += 1
        out[action] = {
            "n_gt": n_gt,
            "n_correct_scorer": n_correct_scorer,
            "n_correct_bbox_dist": n_correct_bbox_dist,
            "n_correct_pipeline": n_correct_pipeline,
            "feature_importances_mean": (
                np.mean(feature_importances, axis=0).tolist()
                if feature_importances else None
            ),
        }
    return out


def main() -> int:
    print("Building feature dataset…", flush=True)
    rows = build_dataset()
    print(f"  {len(rows)} candidate rows from {len({r.rally_id for r in rows})} rallies", flush=True)
    by_action = defaultdict(int)
    for r in rows:
        by_action[r.action] += 1
    for a, n in sorted(by_action.items()):
        print(f"    {a:10s} {n} candidate rows", flush=True)
    print(flush=True)

    print("Training per-action-type GBM with leave-one-video-out CV…", flush=True)
    results = evaluate_loo(rows)
    print(flush=True)

    print("=" * 90, flush=True)
    print(f"{'action':10s} {'n_gt':>6s} {'bbox_dist':>14s} {'pipeline':>14s} {'scorer':>14s} {'Δ vs pipe':>12s}", flush=True)
    print("=" * 90, flush=True)
    total_gt = 0
    total_scorer = 0
    total_pipe = 0
    total_bbox = 0
    for action in ["SERVE", "RECEIVE", "SET", "ATTACK", "DIG", "BLOCK"]:
        if action not in results:
            continue
        r = results[action]
        n_gt = r["n_gt"]
        if n_gt == 0:
            continue
        total_gt += n_gt
        total_scorer += r["n_correct_scorer"]
        total_pipe += r["n_correct_pipeline"]
        total_bbox += r["n_correct_bbox_dist"]
        bbox_pct = 100 * r["n_correct_bbox_dist"] / n_gt
        pipe_pct = 100 * r["n_correct_pipeline"] / n_gt
        scor_pct = 100 * r["n_correct_scorer"] / n_gt
        delta = scor_pct - pipe_pct
        print(f"{action:10s} {n_gt:>6d} {bbox_pct:>13.1f}% {pipe_pct:>13.1f}% {scor_pct:>13.1f}% {delta:>+11.2f}pp", flush=True)
    print("=" * 90, flush=True)
    if total_gt > 0:
        bbox_pct = 100 * total_bbox / total_gt
        pipe_pct = 100 * total_pipe / total_gt
        scor_pct = 100 * total_scorer / total_gt
        delta = scor_pct - pipe_pct
        print(f"{'TOTAL':10s} {total_gt:>6d} {bbox_pct:>13.1f}% {pipe_pct:>13.1f}% {scor_pct:>13.1f}% {delta:>+11.2f}pp", flush=True)
    print(flush=True)

    print("=" * 90, flush=True)
    print("Per-action feature importances (mean across LOO folds)", flush=True)
    print("=" * 90, flush=True)
    print(f"{'action':10s} " + " ".join(f"{f[:10]:>11s}" for f in FEATURE_NAMES), flush=True)
    for action in ["SERVE", "RECEIVE", "SET", "ATTACK", "DIG", "BLOCK"]:
        if action not in results:
            continue
        fi = results[action]["feature_importances_mean"]
        if fi is None:
            continue
        print(f"{action:10s} " + " ".join(f"{v:>11.3f}" for v in fi), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
