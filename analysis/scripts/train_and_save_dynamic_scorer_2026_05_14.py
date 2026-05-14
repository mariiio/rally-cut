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

DB_DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5436/rallycut",
)
TRUSTED_CODENAMES = (
    "titi", "toto", "lulu", "wawa", "caco", "cece", "cici", "cuco",
    "gaga", "kaka", "kiki", "juju", "yeye", "keke",
)
FRAME_TOLERANCE = 5
FEATURE_NAMES = [
    "bbox_dist", "bbox_area", "bbox_aspect_ratio", "bbox_inside_frame",
    "velocity_mag", "velocity_toward_ball",
    "top_y_at_contact", "top_y_change", "height_change",
]
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


def _compute_features(
    positions: list[dict], tid: int, contact_frame: int,
    ball_x: float, ball_y: float,
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
    return [
        bbox_dist, bbox_area, bbox_aspect_ratio, inside,
        velocity_mag, velocity_toward_ball,
        y, top_y_change, height_change,
    ]


@dataclass
class CandidateRow:
    action: str
    candidate_tid: int
    is_gt: bool
    features: list[float]


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
                n_gt_matched += 1
                # Use PIPELINE's frame + ball position (production-matched).
                for tid in primary_tids:
                    feats = _compute_features(
                        positions, tid, pipe_frame,
                        float(pipe_ball_x), float(pipe_ball_y),
                    )
                    if feats is None:
                        continue
                    rows.append(CandidateRow(
                        action=gt_action.upper(),
                        candidate_tid=tid,
                        is_gt=(tid == gt_tid),
                        features=feats,
                    ))
    print(f"  GT rows: {n_gt_seen} seen, {n_gt_matched} matched to pipeline action, "
          f"{n_gt_skipped_no_match} skipped (no matching pipeline action / contact-FN)",
          flush=True)
    return rows


def main() -> int:
    print(f"Output dir: {OUTPUT_DIR}", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Building feature dataset from full trusted-14 corpus…", flush=True)
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
            "Trained on FULL trusted-14 corpus (no hold-out). "
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
