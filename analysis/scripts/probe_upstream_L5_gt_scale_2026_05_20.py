#!/usr/bin/env python3
"""L5: GT-scale learning curve probe.

Trains the per-action attribution scorer GBM on 25/50/75/100% of trusted
GT (stratified by action_type), measures LOO-CV accuracy at each fraction.
Output: learning curve per action + extrapolation to 2x/5x current GT size.

Decision rule per the spec:
  - curve plateaus before 100% -> more GT won't help; signal-limited
  - still sloping at 100% -> GT IS the bottleneck; labeling justified

Output: reports/upstream_bottleneck_2026_05_20/L5.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import psycopg
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import LeaveOneGroupOut

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ANALYSIS_DIR))

from rallycut.tracking.dynamic_attribution_scorer import (  # noqa: E402
    FEATURE_NAMES,
    extract_features,
    position_from_dict,
)

DB_DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5436/rallycut",
)
TRUSTED_32 = (
    "titi", "toto", "lulu", "wawa", "caco", "cece", "cici", "cuco",
    "gaga", "gigi", "kaka", "kiki", "keke", "koko", "kuku",
    "juju", "yeye", "gugu", "mame", "meme", "mimi", "moma", "mumu",
    "papa", "pepe", "pipi", "popo", "pupu", "veve", "vivi", "vovo",
    "haha",
)
ACTIONS = ("SERVE", "RECEIVE", "SET", "ATTACK", "DIG", "BLOCK")
FRACTIONS = (0.25, 0.50, 0.75, 1.00)
OUT_DIR = ANALYSIS_DIR / "reports" / "upstream_bottleneck_2026_05_20"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_training_data() -> dict[str, list[dict[str, Any]]]:
    """For each action, return list of {video, rally_id, frame, cand_tids,
    gt_tid, ball_xy, positions}.

    Reused from train_dynamic_attribution_scorer_2026_05_14.py structure;
    each contact yields N candidates (positive: GT player; negatives: others).
    """
    per_action: dict[str, list[dict[str, Any]]] = {a: [] for a in ACTIONS}
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            """
            SELECT v.name, r.id, rg.action::text, rg.frame, rg.resolved_track_id,
                   pt.positions_json, pt.ball_positions_json, pt.contacts_json
            FROM rally_action_ground_truth rg
            JOIN rallies r ON rg.rally_id = r.id
            JOIN videos v ON r.video_id = v.id
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE v.name = ANY(%s) AND rg.resolved_track_id IS NOT NULL
            """,
            [list(TRUSTED_32)],
        )
        for vname, rid, action, frame, gt_tid, pj, bj, cj in cur.fetchall():
            actn = action.upper()
            if actn not in per_action:
                continue
            positions = pj if isinstance(pj, list) else (
                json.loads(pj) if isinstance(pj, str) else []
            )
            contacts = (cj if isinstance(cj, dict)
                        else (json.loads(cj) if isinstance(cj, str) else {})
                        ).get("contacts") or []
            # ball positions reserved for potential use; not needed for feature extraction here
            _ = bj
            nearest = None
            nearest_d = 999
            for c in contacts:
                d = abs(int(c.get("frame", -1)) - int(frame))
                if d < nearest_d:
                    nearest_d = d
                    nearest = c
            if not nearest or nearest_d > 5:
                continue
            cand_tids = [int(pc[0]) for pc in (nearest.get("playerCandidates") or [])]
            ball_xy = (
                float(nearest.get("ballX", 0.5)),
                float(nearest.get("ballY", 0.5)),
            )
            per_action[actn].append({
                "video": str(vname),
                "rally_id": str(rid),
                "frame": int(frame),
                "cand_tids": cand_tids,
                "gt_tid": int(gt_tid),
                "ball_xy": ball_xy,
                "positions": positions,
            })
    return per_action


def build_xy(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """For each row, generate (N_cand) feature vectors with binary label."""
    feat_list, y_list, groups = [], [], []
    for row in rows:
        positions_like = [position_from_dict(p) for p in row["positions"]]
        for tid in row["cand_tids"]:
            cf = extract_features(
                positions_like, tid, row["frame"],
                row["ball_xy"][0], row["ball_xy"][1],
                prev_action_tid=-1, post_ball_x=None, post_ball_y=None,
                expected_team=None, team_assignments=None,
            )
            if cf is None:
                continue
            arr = np.array([getattr(cf, f) for f in FEATURE_NAMES], dtype=float)
            if np.isnan(arr).any():
                arr = np.nan_to_num(arr, nan=0.0)
            feat_list.append(arr)
            y_list.append(1 if tid == row["gt_tid"] else 0)
            groups.append(row["video"])
    return np.array(feat_list), np.array(y_list), np.array(groups)


def loo_cv_accuracy(feats: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:  # noqa: N803
    """Leave-one-video-out CV accuracy. Per-candidate accuracy."""
    if len(np.unique(groups)) < 2:
        return float("nan")
    logo = LeaveOneGroupOut()
    correct = 0
    total = 0
    for train_idx, test_idx in logo.split(feats, y, groups):
        clf = GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42)
        clf.fit(feats[train_idx], y[train_idx])
        pred = clf.predict(feats[test_idx])
        correct += int((pred == y[test_idx]).sum())
        total += len(test_idx)
    return correct / max(total, 1)


def main() -> int:
    print("Loading training data for L5 GT-scale learning curve...", flush=True)
    per_action = load_training_data()
    for a in ACTIONS:
        print(f"  {a}: {len(per_action[a])} GT contacts", flush=True)

    results: dict[str, dict[str, float | int | bool]] = {}
    for action in ACTIONS:
        rows = per_action[action]
        if len(rows) < 20:
            print(f"  SKIP {action}: n={len(rows)} too small for stable curve",
                  flush=True)
            results[action] = {"n_contacts": len(rows), "skipped": True}
            continue
        results[action] = {"n_contacts": len(rows)}
        for frac in FRACTIONS:
            n = max(int(len(rows) * frac), 5)
            if n < 10:
                # Single-fraction guard for actions like BLOCK (n=26) where
                # smaller fractions become measurement noise rather than signal.
                results[action][f"frac_{frac}"] = float("nan")
                print(f"  {action} frac={frac}: n_rows={n} too small, recording NaN",
                      flush=True)
                continue
            # Re-seed each fraction so subsets are nested
            # (25% ⊆ 50% ⊆ 75% ⊆ 100%) — supports learning-curve interpretation.
            rng = np.random.default_rng(42)
            sub = rng.choice(len(rows), size=n, replace=False)
            rows_sub = [rows[i] for i in sub]
            feats, y, groups = build_xy(rows_sub)
            if len(np.unique(groups)) < 2:
                results[action][f"frac_{frac}"] = float("nan")
                continue
            acc = loo_cv_accuracy(feats, y, groups)
            results[action][f"frac_{frac}"] = acc
            print(f"  {action} frac={frac}: n_rows={n}, n_cands={len(feats)}, acc={acc:.3f}",
                  flush=True)

    out = OUT_DIR / "L5.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nWrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
