#!/usr/bin/env python3
"""Leave-one-video-out CV for the v3.1 scorer on trusted-29.

Honest measurement: for each video v in trusted-29:
  - Train per-action GBMs on the OTHER 28 videos
  - Score candidates for v's GT rows
  - Compute rank-1 accuracy on v
Aggregate across all 29 folds → honest per-action accuracy estimate.

Why: the production trainer trains on the full corpus and we measure on
the full corpus → leaky. If full-corpus accuracy (89.6% matched on
trusted-29) is within ~2pp of LOO accuracy, the lift is real and the
model isn't memorizing.

Reuses `build_dataset()` from train_and_save_dynamic_scorer_2026_05_14.py
to guarantee feature parity with the production model. The only
difference vs the production trainer is the train/test split.

Usage:
    cd analysis && uv run python scripts/measure_loo_v31_2026_05_17.py
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_and_save_dynamic_scorer_2026_05_14 import (  # noqa: E402
    ACTION_TYPES, FEATURE_NAMES, TRUSTED_CODENAMES, build_dataset,
)


def main() -> int:
    print(f"Building feature dataset from full trusted-{len(TRUSTED_CODENAMES)} corpus…",
          flush=True)
    rows = build_dataset()
    n_rallies = len({r.rally_id for r in rows})
    n_videos = len({r.video for r in rows})
    print(f"  {len(rows)} candidate rows from {n_rallies} rallies, {n_videos} videos",
          flush=True)
    print(f"  feature count: {len(FEATURE_NAMES)} (must match production)", flush=True)
    print(flush=True)

    by_action: dict[str, list] = defaultdict(list)
    for r in rows:
        by_action[r.action].append(r)

    # Aggregate stats per action + per fold
    print("Running per-action LOO CV (1 fold per held-out video)…", flush=True)
    print(flush=True)

    per_action_summary: dict[str, dict[str, Any]] = {}
    per_video_summary: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"n": 0, "correct": 0})
    )
    for action in ACTION_TYPES:
        action_rows = by_action.get(action, [])
        if not action_rows:
            continue
        videos = sorted({r.video for r in action_rows})
        n_gt_total = 0
        n_correct_scorer = 0
        n_correct_bbox = 0
        feature_importances: list[np.ndarray] = []
        for hold_v in videos:
            train_rows = [r for r in action_rows if r.video != hold_v]
            test_rows = [r for r in action_rows if r.video == hold_v]
            if not train_rows or not test_rows:
                continue
            X_train = np.array([r.features for r in train_rows])
            y_train = np.array([1 if r.is_gt else 0 for r in train_rows])
            if y_train.sum() == 0 or y_train.sum() == len(y_train):
                continue
            clf = GradientBoostingClassifier(
                n_estimators=80, max_depth=3, learning_rate=0.05,
                random_state=42,
            )
            clf.fit(X_train, y_train)
            feature_importances.append(clf.feature_importances_)
            # Group test rows by GT row identity for rank-1 evaluation
            grouped: dict[tuple[str, int], list] = defaultdict(list)
            for r in test_rows:
                grouped[(r.rally_id, r.gt_frame)].append(r)
            for (_rid, _f), group in grouped.items():
                if not any(r.is_gt for r in group):
                    continue
                X_test = np.array([r.features for r in group])
                probs = clf.predict_proba(X_test)[:, 1]
                pred_idx = int(np.argmax(probs))
                bbox_idx = int(np.argmin(X_test[:, 0]))  # feature 0 = bbox_dist
                gt_idx = next(i for i, r in enumerate(group) if r.is_gt)
                n_gt_total += 1
                if pred_idx == gt_idx:
                    n_correct_scorer += 1
                if bbox_idx == gt_idx:
                    n_correct_bbox += 1
                per_video_summary[hold_v][action]["n"] += 1
                if pred_idx == gt_idx:
                    per_video_summary[hold_v][action]["correct"] += 1
        per_action_summary[action] = {
            "n_gt": n_gt_total,
            "n_correct_scorer": n_correct_scorer,
            "n_correct_bbox": n_correct_bbox,
            "feature_importances_mean": (
                np.mean(feature_importances, axis=0).tolist()
                if feature_importances else None
            ),
        }
        scor_pct = 100 * n_correct_scorer / max(1, n_gt_total)
        bbox_pct = 100 * n_correct_bbox / max(1, n_gt_total)
        print(f"  {action:8s} n={n_gt_total:4d}  bbox-dist {bbox_pct:5.1f}%  "
              f"scorer (LOO) {scor_pct:5.1f}%  Δ {scor_pct - bbox_pct:+5.1f}pp",
              flush=True)

    print(flush=True)
    print("=" * 90, flush=True)
    print(f"{'action':10s} {'n':>5s} {'bbox-dist':>12s} {'LOO scorer':>12s} {'Δ':>10s}", flush=True)
    print("=" * 90, flush=True)
    total_gt = total_scor = total_bbox = 0
    for action in ACTION_TYPES:
        r = per_action_summary.get(action)
        if not r or r["n_gt"] == 0:
            continue
        total_gt += r["n_gt"]
        total_scor += r["n_correct_scorer"]
        total_bbox += r["n_correct_bbox"]
        bbox_pct = 100 * r["n_correct_bbox"] / r["n_gt"]
        scor_pct = 100 * r["n_correct_scorer"] / r["n_gt"]
        print(f"{action:10s} {r['n_gt']:>5d} {bbox_pct:>11.1f}% {scor_pct:>11.1f}% {scor_pct - bbox_pct:>+9.2f}pp",
              flush=True)
    print("=" * 90, flush=True)
    if total_gt:
        bbox_pct = 100 * total_bbox / total_gt
        scor_pct = 100 * total_scor / total_gt
        print(f"{'TOTAL':10s} {total_gt:>5d} {bbox_pct:>11.1f}% {scor_pct:>11.1f}% {scor_pct - bbox_pct:>+9.2f}pp",
              flush=True)
    print(flush=True)

    print("Per-video LOO scorer accuracy:", flush=True)
    for v in sorted(per_video_summary.keys()):
        per_action = per_video_summary[v]
        n_v = sum(d["n"] for d in per_action.values())
        c_v = sum(d["correct"] for d in per_action.values())
        if n_v:
            print(f"  {v:6s} {c_v}/{n_v} ({c_v / n_v * 100:.1f}%)", flush=True)

    print(flush=True)
    print("Per-action feature importances (mean across LOO folds):", flush=True)
    print(f"  {'action':10s} " + " ".join(f"{f[:11]:>12s}" for f in FEATURE_NAMES), flush=True)
    for action in ACTION_TYPES:
        r = per_action_summary.get(action)
        if not r or not r.get("feature_importances_mean"):
            continue
        fi = r["feature_importances_mean"]
        print(f"  {action:10s} " + " ".join(f"{v:>12.3f}" for v in fi), flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
