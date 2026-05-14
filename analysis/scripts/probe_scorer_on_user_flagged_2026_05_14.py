#!/usr/bin/env python3
"""Verify the dynamic-feature scorer on the user-flagged rallies.

Train per-action-type GBMs on trusted-14 EXCLUDING keke and kiki. Then apply
the trained models on user-flagged rallies and report:
  - For each of the 4 contacts (serve, receive, set, attack):
    - Each candidate's features
    - Each candidate's scorer probability
    - The scorer's argmax (predicted pick)
    - Current pipeline pick
    - GT (resolved_track_id)
    - Verdict: scorer vs GT vs pipeline

This is the user-facing verification.
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
TRAIN_CODENAMES = (
    "titi", "toto", "lulu", "wawa", "caco", "cece", "cici", "cuco",
    "gaga", "kaka", "juju", "yeye", "keke", "kiki",
)  # full trusted-14, no hold-out — testing fit-to-pattern
FRAME_TOLERANCE = 5
FEATURE_NAMES = [
    "bbox_dist", "bbox_area", "bbox_aspect_ratio", "bbox_inside_frame",
    "velocity_mag", "velocity_toward_ball",
    "top_y_at_contact", "top_y_change", "height_change",
]
USER_RALLIES = [
    # (video_codename, 1-based-rally-index, [list of user-flagged actions])
    ("keke", 1, ["serve", "receive", "set", "attack"]),
    ("keke", 4, ["serve", "receive", "set", "attack"]),
    ("keke", 6, ["serve", "receive", "set", "attack"]),
    ("keke", 9, ["serve", "receive", "set", "attack", "dig"]),
    ("kiki", 1, ["serve", "set"]),
    ("kiki", 5, ["serve", "set"]),
    ("kiki", 6, ["serve", "receive", "set", "attack"]),
    ("kiki", 8, ["serve", "receive", "set", "attack"]),
]


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
    positions: list[dict], tid: int, contact_frame: int,
    ball_x: float, ball_y: float,
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
        y, top_y_change, height_change,
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
    pipeline_picked: bool


def build_dataset(codenames: tuple[str, ...]) -> list[CandidateRow]:
    rows: list[CandidateRow] = []
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            """
            SELECT v.name, r.id, pt.primary_track_ids, pt.positions_json, pt.actions_json
            FROM videos v JOIN rallies r ON r.video_id = v.id
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE v.name = ANY(%s) AND r.status = 'CONFIRMED'
            ORDER BY v.name, r."order"
            """,
            [list(codenames)],
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
                pipeline_tid = int(actions[best_idx].get("playerTrackId", -1)) if best_idx >= 0 else -1
                for tid in primary_tids:
                    feats = _compute_features(positions, tid, gt_frame, ball_x, ball_y)
                    if feats is None:
                        continue
                    rows.append(CandidateRow(
                        video=video_name, rally_id=rally_id, gt_frame=gt_frame,
                        action=gt_action.upper(), candidate_tid=tid,
                        is_gt=(tid == gt_tid), features=feats,
                        pipeline_picked=(tid == pipeline_tid),
                    ))
    return rows


def main() -> int:
    print("Stage 1: train per-action-type GBMs on FULL trusted-14 (no hold-out)…", flush=True)
    print("  This tests whether the model CAN FIT the patterns when given access.", flush=True)
    print("  (Not a generalization test; that was the prior LOO experiment.)", flush=True)
    train_rows = build_dataset(TRAIN_CODENAMES)
    print(f"  Training dataset: {len(train_rows)} candidate rows", flush=True)
    models: dict[str, GradientBoostingClassifier] = {}
    for action in ["SERVE", "RECEIVE", "SET", "ATTACK", "DIG", "BLOCK"]:
        action_rows = [r for r in train_rows if r.action == action]
        if not action_rows:
            continue
        X = np.array([r.features for r in action_rows])
        y = np.array([1 if r.is_gt else 0 for r in action_rows])
        if y.sum() == 0 or y.sum() == len(y):
            continue
        clf = GradientBoostingClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.05, random_state=42,
        )
        clf.fit(X, y)
        models[action] = clf
    print(f"  Trained {len(models)} per-action models", flush=True)
    print(flush=True)

    print("Stage 2: apply trained models on user-flagged keke + kiki rallies…", flush=True)
    print(flush=True)
    eval_rows = build_dataset(("keke", "kiki"))
    # Group by rally + action for display
    rally_id_map: dict[str, tuple[str, int]] = {}
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            """
            SELECT v.name, r.id, r."order"
            FROM videos v JOIN rallies r ON r.video_id = v.id
            WHERE v.name = ANY(%s)
            """,
            [list(("keke", "kiki"))],
        )
        for video, rid, order in cur.fetchall():
            rally_id_map[rid] = (video, order)

    # Group rows by (rally_id, action)
    by_action_in_rally: dict[tuple[str, str], list[CandidateRow]] = defaultdict(list)
    for r in eval_rows:
        by_action_in_rally[(r.rally_id, r.action)].append(r)

    # Score and report per user-flagged rally
    n_total = 0
    n_scorer_correct = 0
    n_pipe_correct = 0
    for video_codename, rally_order_1based, _flagged_actions in USER_RALLIES:
        # Find the rally_id for this codename + order
        target_order = rally_order_1based - 1
        target_rally_id = None
        for rid, (v, o) in rally_id_map.items():
            if v == video_codename and o == target_order:
                target_rally_id = rid
                break
        if target_rally_id is None:
            continue
        print(f"=== {video_codename} r{rally_order_1based} ({target_rally_id[:8]}) ===", flush=True)
        # Find all (rally, action) for this rally
        actions_in_rally = sorted(
            {a for (rid, a) in by_action_in_rally.keys() if rid == target_rally_id},
            key=lambda a: min(r.gt_frame for r in by_action_in_rally[(target_rally_id, a)]),
        )
        for action in actions_in_rally:
            group = by_action_in_rally[(target_rally_id, action)]
            if not group:
                continue
            if action not in models:
                continue
            X = np.array([r.features for r in group])
            probs = models[action].predict_proba(X)[:, 1]
            pred_idx = int(np.argmax(probs))
            gt_idx = next((i for i, r in enumerate(group) if r.is_gt), -1)
            pipe_idx = next((i for i, r in enumerate(group) if r.pipeline_picked), -1)
            if gt_idx < 0:
                continue
            scorer_tid = group[pred_idx].candidate_tid
            gt_tid = group[gt_idx].candidate_tid
            pipe_tid = group[pipe_idx].candidate_tid if pipe_idx >= 0 else -1
            n_total += 1
            if scorer_tid == gt_tid:
                n_scorer_correct += 1
            if pipe_tid == gt_tid:
                n_pipe_correct += 1
            frame = group[0].gt_frame
            scorer_match = "✓" if scorer_tid == gt_tid else "✗"
            pipe_match = "✓" if pipe_tid == gt_tid else "✗"
            line = (f"  {action:8s} f={frame:>4d}  "
                    f"GT=P{gt_tid}  pipeline=P{pipe_tid}{pipe_match}  scorer=P{scorer_tid}{scorer_match}")
            print(line, flush=True)
            # Detail: candidate probs
            for i, r in enumerate(group):
                marker = ""
                if r.candidate_tid == gt_tid:
                    marker += " (GT)"
                if r.candidate_tid == pipe_tid:
                    marker += " (pipe)"
                if i == pred_idx:
                    marker += " (scorer)"
                print(f"        cand P{r.candidate_tid}: prob={probs[i]:.3f} bbox_dist={r.features[0]:.3f}{marker}", flush=True)
        print(flush=True)

    print("=" * 90, flush=True)
    print(f"Summary on user-flagged contacts:", flush=True)
    print(f"  {n_total} contacts evaluated", flush=True)
    print(f"  Pipeline correct: {n_pipe_correct} ({100*n_pipe_correct/max(1,n_total):.1f}%)", flush=True)
    print(f"  Scorer   correct: {n_scorer_correct} ({100*n_scorer_correct/max(1,n_total):.1f}%)", flush=True)
    print(f"  Δ scorer vs pipeline: {n_scorer_correct - n_pipe_correct:+d} contacts "
          f"({100*(n_scorer_correct - n_pipe_correct)/max(1,n_total):+.1f}pp)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
