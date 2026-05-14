"""Pose v2 probe — does rich pose lift per-action-type learned attribution? (2026-05-14)

Reuses today's 30-feature per-candidate baseline and adds 10 rich pose features
(per the 'What to extract' spec):
  1.  left_wrist_x_norm, left_wrist_y_norm, left_wrist_conf
  2.  right_wrist_x_norm, right_wrist_y_norm, right_wrist_conf
  3.  best_wrist_to_ball_dist
  4.  arm_extension_angle_max
  5.  wrist_above_shoulder (binary)
  6.  wrist_above_head (binary, vs nose y)
  7.  body_velocity_pre_contact (bbox-center motion magnitude, normalized)
  8.  body_velocity_post_contact
  9.  jump_height_proxy (bbox-bottom delta vs rest [-15,-10])
  10. wrist_pre_contact_velocity

Pose data is loaded from `training_data/pose_cache/<rally_id>.npz`
(already keyed by canonical pid 1..4). For each contact + candidate we read
keypoints across the contact frame for slots 1-6 and the [-3..-1] window
for the velocity features (slots 7, 10).

Per-action-type LightGBM-equivalent (HistGradientBoosting) classifier in
leave-one-video-out CV across the same 12 panel videos used today.

Decision criteria:
  - >=5pp lift on >=2 action types vs TODAY's GBM (not just baseline)
  - No type regresses by >2pp
  - New pose features in top-5 importance for >=1 type

Usage:
    cd analysis
    uv run python scripts/probe_pose_v2_attribution_2026_05_14.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402

# Reuse today's experiment's data hydration + base feature extraction
from scripts.probe_learned_attribution_per_type_2026_05_14 import (  # noqa: E402
    ACTION_TYPES_TO_TRAIN,
    FEATURE_NAMES as TODAY_FEATURE_NAMES,
    GT_FRAME_TOL,
    VIDEO_NAMES,
    ContactCtx,
    DatasetRow,
    _match_gt_to_ctx,
    compute_features_for_candidate,
    fetch_data,
)

POSE_CACHE_DIR = HERE.parent / "training_data" / "pose_cache"
REPORT_DIR = HERE.parent / "reports" / "pose_v2_probe_2026_05_14"
JSON_PATH = REPORT_DIR / "results.json"
MD_PATH = REPORT_DIR / "results.md"

# COCO-17 keypoint indices
KPT_NOSE = 0
KPT_LEFT_SHOULDER = 5
KPT_RIGHT_SHOULDER = 6
KPT_LEFT_ELBOW = 7
KPT_RIGHT_ELBOW = 8
KPT_LEFT_WRIST = 9
KPT_RIGHT_WRIST = 10

MIN_KPT_CONF = 0.3

POSE_V2_FEATURE_NAMES = [
    "pv2_left_wrist_x_norm",
    "pv2_left_wrist_y_norm",
    "pv2_left_wrist_conf",
    "pv2_right_wrist_x_norm",
    "pv2_right_wrist_y_norm",
    "pv2_right_wrist_conf",
    "pv2_best_wrist_to_ball_dist",
    "pv2_arm_extension_angle_max",
    "pv2_wrist_above_shoulder",
    "pv2_wrist_above_head",
    "pv2_body_velocity_pre_contact",
    "pv2_body_velocity_post_contact",
    "pv2_jump_height_proxy",
    "pv2_wrist_pre_contact_velocity",
]
# Note: spec says 10 features but breaks into sub-fields → 14 dimensions total.

POSE_FEATURE_NAMES = TODAY_FEATURE_NAMES + POSE_V2_FEATURE_NAMES


# ---------------------------------------------------------------------------
# Pose cache lookup
# ---------------------------------------------------------------------------

@dataclass
class RallyPoseLookup:
    """Per-rally pose data keyed by (frame, track_id) → keypoints (17, 3)."""
    rally_id: str
    kp_lookup: dict[tuple[int, int], np.ndarray]
    bbox_lookup: dict[tuple[int, int], np.ndarray]  # xyxy normalized


def _load_pose_lookup(rally_id: str) -> RallyPoseLookup | None:
    path = POSE_CACHE_DIR / f"{rally_id}.npz"
    if not path.exists():
        return None
    try:
        data = np.load(path)
    except Exception:
        return None
    frames = data["frames"]
    tids = data["track_ids"]
    kps = data["keypoints"]
    bboxes = data["bboxes"]
    kp_lookup: dict[tuple[int, int], np.ndarray] = {}
    bb_lookup: dict[tuple[int, int], np.ndarray] = {}
    for i in range(len(frames)):
        key = (int(frames[i]), int(tids[i]))
        kp_lookup[key] = kps[i]      # (17, 3) normalized x,y in [0,1]
        bb_lookup[key] = bboxes[i]   # (4,) normalized xyxy
    return RallyPoseLookup(rally_id=rally_id, kp_lookup=kp_lookup, bbox_lookup=bb_lookup)


# ---------------------------------------------------------------------------
# Pose v2 feature extraction
# ---------------------------------------------------------------------------

def _arm_angle(shoulder: np.ndarray, elbow: np.ndarray, wrist: np.ndarray) -> float | None:
    """Shoulder→Elbow→Wrist angle in degrees. 180° = fully extended."""
    if shoulder[2] < MIN_KPT_CONF or elbow[2] < MIN_KPT_CONF or wrist[2] < MIN_KPT_CONF:
        return None
    v1 = (shoulder[0] - elbow[0], shoulder[1] - elbow[1])
    v2 = (wrist[0] - elbow[0], wrist[1] - elbow[1])
    m1 = math.hypot(*v1)
    m2 = math.hypot(*v2)
    if m1 < 1e-6 or m2 < 1e-6:
        return None
    cos_a = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (m1 * m2)))
    return math.degrees(math.acos(cos_a))


def compute_pose_v2_features(
    ctx: ContactCtx,
    cand_tid: int,
    pose: RallyPoseLookup | None,
) -> list[float]:
    """Return the 14 Pose v2 features for (contact, candidate).

    NaN-filled when pose data is missing — HistGradientBoosting handles NaN natively.
    """
    out = [float("nan")] * len(POSE_V2_FEATURE_NAMES)

    if pose is None:
        return out

    f = ctx.frame
    ball = ctx.ball_at_contact  # normalized? In contacts JSON, ballX/Y are in normalized [0,1]
    # NOTE: cand_bbox positions are stored in normalized [0,1] in positions_json.
    # Pose keypoints are in normalized [0,1] (from extract_pose_cache).

    # ---- Contact-frame keypoints ----
    kps_at = pose.kp_lookup.get((f, cand_tid))
    if kps_at is not None:
        lw = kps_at[KPT_LEFT_WRIST]
        rw = kps_at[KPT_RIGHT_WRIST]
        ls = kps_at[KPT_LEFT_SHOULDER]
        rs = kps_at[KPT_RIGHT_SHOULDER]
        le = kps_at[KPT_LEFT_ELBOW]
        re = kps_at[KPT_RIGHT_ELBOW]
        nose = kps_at[KPT_NOSE]

        # 1. left wrist x, y, conf
        out[0] = float(lw[0])
        out[1] = float(lw[1])
        out[2] = float(lw[2])

        # 2. right wrist x, y, conf
        out[3] = float(rw[0])
        out[4] = float(rw[1])
        out[5] = float(rw[2])

        # 3. best_wrist_to_ball_dist (over visible wrists)
        if ball is not None:
            bx, by = ball
            wrist_dists: list[float] = []
            if lw[2] >= MIN_KPT_CONF:
                wrist_dists.append(math.hypot(lw[0] - bx, lw[1] - by))
            if rw[2] >= MIN_KPT_CONF:
                wrist_dists.append(math.hypot(rw[0] - bx, rw[1] - by))
            if wrist_dists:
                out[6] = float(min(wrist_dists))

        # 4. arm_extension_angle_max
        angles: list[float] = []
        la = _arm_angle(ls, le, lw)
        ra = _arm_angle(rs, re, rw)
        if la is not None:
            angles.append(la)
        if ra is not None:
            angles.append(ra)
        if angles:
            out[7] = float(max(angles))

        # 5. wrist_above_shoulder: any visible wrist y < shoulder y (image y inc downward)
        wrist_above_shoulder = 0.0
        if lw[2] >= MIN_KPT_CONF and ls[2] >= MIN_KPT_CONF and lw[1] < ls[1]:
            wrist_above_shoulder = 1.0
        if rw[2] >= MIN_KPT_CONF and rs[2] >= MIN_KPT_CONF and rw[1] < rs[1]:
            wrist_above_shoulder = 1.0
        out[8] = wrist_above_shoulder

        # 6. wrist_above_head: any visible wrist y < nose y
        wrist_above_head = 0.0
        if nose[2] >= MIN_KPT_CONF:
            if lw[2] >= MIN_KPT_CONF and lw[1] < nose[1]:
                wrist_above_head = 1.0
            if rw[2] >= MIN_KPT_CONF and rw[1] < nose[1]:
                wrist_above_head = 1.0
        out[9] = wrist_above_head

    # ---- Velocity features ----
    # 7. body_velocity_pre_contact: mean bbox-center motion over [f-3, f-1]
    pre_centers: list[tuple[float, float]] = []
    for fp in range(f - 3, f):
        bb = ctx.cand_bbox_pre.get(cand_tid, {}).get(fp)
        if bb is not None:
            pre_centers.append(bb)
    bb_at = ctx.cand_bbox_at.get(cand_tid)
    if bb_at is not None:
        pre_centers.append(bb_at)
    if len(pre_centers) >= 2:
        # Magnitudes between successive frames
        mags = [
            math.hypot(pre_centers[i + 1][0] - pre_centers[i][0],
                       pre_centers[i + 1][1] - pre_centers[i][1])
            for i in range(len(pre_centers) - 1)
        ]
        out[10] = float(np.mean(mags))

    # 8. body_velocity_post_contact: over [f+1, f+3]
    post_centers: list[tuple[float, float]] = []
    if bb_at is not None:
        post_centers.append(bb_at)
    for fp in range(f + 1, f + 4):
        bb = ctx.cand_bbox_post.get(cand_tid, {}).get(fp)
        if bb is not None:
            post_centers.append(bb)
    if len(post_centers) >= 2:
        mags = [
            math.hypot(post_centers[i + 1][0] - post_centers[i][0],
                       post_centers[i + 1][1] - post_centers[i][1])
            for i in range(len(post_centers) - 1)
        ]
        out[11] = float(np.mean(mags))

    # 9. jump_height_proxy: bbox-bottom y at contact - mean over [f-15, f-10]
    # Positions data is normalized: (x, y) is the bbox CENTER. Bottom = y + h/2.
    # We don't have height stored in cand_bbox_* (it's just (cx, cy)).
    # Use pose-cache bbox (xyxy normalized) where available, or fall back to
    # the center-y proxy. The bbox cache only stores frames in the contact
    # window (±10), so [-15..-10] is partly out of range; we use what we have.
    rest_bottoms: list[float] = []
    for fp in range(f - 15, f - 9):
        bb_xyxy = pose.bbox_lookup.get((fp, cand_tid)) if pose else None
        if bb_xyxy is not None:
            rest_bottoms.append(float(bb_xyxy[3]))  # y2
    contact_bottom: float | None = None
    bb_xyxy_at = pose.bbox_lookup.get((f, cand_tid)) if pose else None
    if bb_xyxy_at is not None:
        contact_bottom = float(bb_xyxy_at[3])
    if rest_bottoms and contact_bottom is not None:
        rest_mean = float(np.mean(rest_bottoms))
        # Image y increases downward, so jump = rest_bottom - contact_bottom > 0 means jumped
        out[12] = rest_mean - contact_bottom

    # 10. wrist_pre_contact_velocity: best-wrist motion magnitude pre-contact
    # Pick best wrist (highest mean confidence over [-3,-1, 0]) and compute mean velocity
    if pose is not None:
        # Gather wrist positions for both sides over [f-3..f]
        def _wrist_seq(side_idx: int) -> list[tuple[int, float, float, float]]:
            seq: list[tuple[int, float, float, float]] = []
            for fp in range(f - 3, f + 1):
                k = pose.kp_lookup.get((fp, cand_tid))
                if k is None:
                    continue
                w = k[side_idx]
                seq.append((fp, float(w[0]), float(w[1]), float(w[2])))
            return seq
        l_seq = _wrist_seq(KPT_LEFT_WRIST)
        r_seq = _wrist_seq(KPT_RIGHT_WRIST)
        # Score each: mean conf
        l_conf = np.mean([s[3] for s in l_seq]) if l_seq else 0.0
        r_conf = np.mean([s[3] for s in r_seq]) if r_seq else 0.0
        chosen = l_seq if l_conf >= r_conf else r_seq
        if len(chosen) >= 2:
            mags = []
            for i in range(len(chosen) - 1):
                if chosen[i][3] < MIN_KPT_CONF or chosen[i + 1][3] < MIN_KPT_CONF:
                    continue
                mags.append(math.hypot(
                    chosen[i + 1][1] - chosen[i][1],
                    chosen[i + 1][2] - chosen[i][2],
                ))
            if mags:
                out[13] = float(np.mean(mags))

    return out


# ---------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------

def build_dataset_pose_v2(
    rally_contacts: dict[str, list[ContactCtx]],
    gt_rows: list[Any],
) -> list[DatasetRow]:
    """Build per-(contact, candidate) dataset with 30+14 features."""
    gt_by_rally: dict[str, list[Any]] = defaultdict(list)
    for g in gt_rows:
        gt_by_rally[g.rally_id].append(g)

    pose_cache: dict[str, RallyPoseLookup | None] = {}

    rows: list[DatasetRow] = []
    n_total_gt = 0
    n_matched = 0
    n_unresolved = 0
    n_skipped_no_team = 0
    n_pose_missing = 0
    for rally_id, ctxs in rally_contacts.items():
        gts = gt_by_rally.get(rally_id, [])
        if not gts:
            continue
        if rally_id not in pose_cache:
            pose_cache[rally_id] = _load_pose_lookup(rally_id)
        pose = pose_cache[rally_id]
        pairs = _match_gt_to_ctx(ctxs, gts)
        for g, c in pairs:
            n_total_gt += 1
            if g.resolved_tid is None:
                n_unresolved += 1
                continue
            if c is None:
                continue
            n_matched += 1
            cand_pool = sorted(c.team_assignments.keys())
            if len(cand_pool) != 4:
                n_skipped_no_team += 1
                continue
            if pose is None:
                n_pose_missing += 1
            for cand_tid in cand_pool:
                base = compute_features_for_candidate(c, cand_tid)
                pose_v2 = compute_pose_v2_features(c, cand_tid, pose)
                rows.append(DatasetRow(
                    video=c.video, rally_id=c.rally_id, frame=c.frame,
                    action_uc=g.action.upper(), cand_tid=cand_tid,
                    is_gt=int(cand_tid == g.resolved_tid),
                    is_pipeline_pick=int(c.pl_tid is not None and cand_tid == c.pl_tid),
                    features=base + pose_v2,
                    gt_tid=g.resolved_tid,
                    pl_tid=c.pl_tid,
                ))
    print(
        f"Dataset build: GT={n_total_gt}, matched_to_ctx={n_matched}, "
        f"unresolved={n_unresolved}, skipped_no_4_team={n_skipped_no_team}, "
        f"pose_missing_rallies={n_pose_missing}",
    )
    return rows


# ---------------------------------------------------------------------------
# Per-type LOVO CV
# ---------------------------------------------------------------------------

@dataclass
class TypeResult:
    type_name: str
    n_gt: int
    n_predicted: int
    baseline_correct: int
    today_correct: int
    pose_v2_correct: int
    baseline_precision: float
    today_precision: float
    pose_v2_precision: float
    delta_vs_today_pp: float
    delta_vs_baseline_pp: float
    top_features: list[tuple[str, float]]


def _train_loo_with_features(
    type_rows: list[DatasetRow],
    video_names: list[str],
    n_features: int,
    label: str,
) -> tuple[int, int, np.ndarray, int]:
    """LOVO-CV with the first n_features columns. Returns (correct, n_predicted, sum_importances, n_models)."""
    correct = 0
    n_predicted = 0
    importances_sum = np.zeros(n_features, dtype=np.float64)
    n_models = 0
    for hold_out in video_names:
        train_rows = [r for r in type_rows if r.video != hold_out]
        test_rows = [r for r in type_rows if r.video == hold_out]
        if not train_rows or not test_rows:
            continue
        X_train = np.array(
            [r.features[:n_features] for r in train_rows], dtype=np.float32,
        )
        y_train = np.array([r.is_gt for r in train_rows], dtype=np.int32)
        if y_train.sum() == 0 or y_train.sum() == len(y_train):
            continue
        model = HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=10, l2_regularization=1.0, random_state=42,
        )
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"  [{label}] hold={hold_out} fit failed: {e}")
            continue

        # Permutation importance on training subsample
        try:
            sub = np.random.default_rng(42).choice(
                len(X_train), size=min(800, len(X_train)), replace=False,
            )
            pi = permutation_importance(
                model, X_train[sub], y_train[sub], n_repeats=3, random_state=42, n_jobs=1,
            )
            importances_sum += pi.importances_mean
            n_models += 1
        except Exception as e:
            print(f"  [{label}] hold={hold_out} perm-importance failed: {e}")

        # Predict
        test_groups: dict[tuple[str, str, int], list[DatasetRow]] = defaultdict(list)
        for r in test_rows:
            test_groups[(r.video, r.rally_id, r.frame)].append(r)
        for _key, rs in test_groups.items():
            X_test = np.array([r.features[:n_features] for r in rs], dtype=np.float32)
            probs = model.predict_proba(X_test)[:, 1]
            best_idx = int(np.argmax(probs))
            pick = rs[best_idx]
            n_predicted += 1
            if pick.is_gt == 1:
                correct += 1
        print(
            f"  [{label}] hold={hold_out:>4}: train={len(train_rows):>4}, "
            f"test={len(test_rows):>3}, running={correct}/{n_predicted}",
        )
    return correct, n_predicted, importances_sum, n_models


def evaluate_type(type_name: str, all_rows: list[DatasetRow], video_names: list[str]) -> TypeResult:
    type_rows = [r for r in all_rows if r.action_uc == type_name]
    # Group by contact for baseline
    contact_groups: dict[tuple[str, str, int], list[DatasetRow]] = defaultdict(list)
    for r in type_rows:
        contact_groups[(r.video, r.rally_id, r.frame)].append(r)
    n_gt = len(contact_groups)

    # Baseline (pipeline pick == GT)
    baseline_correct = 0
    for _key, rs in contact_groups.items():
        pl_picks = [r for r in rs if r.is_pipeline_pick]
        if pl_picks and any(r.is_gt == 1 for r in pl_picks):
            baseline_correct += 1
    baseline_precision = baseline_correct / n_gt if n_gt > 0 else 0.0

    n_today = len(TODAY_FEATURE_NAMES)
    n_all = len(POSE_FEATURE_NAMES)

    print(f"--- {type_name} TODAY's-GBM (30 features) ---")
    today_correct, today_predicted, _imp_today, _nm_today = _train_loo_with_features(
        type_rows, video_names, n_today, f"{type_name}-today",
    )

    print(f"--- {type_name} POSE-V2-GBM ({n_all} features) ---")
    pose_v2_correct, pose_v2_predicted, imp_pose_v2, nm_pose_v2 = _train_loo_with_features(
        type_rows, video_names, n_all, f"{type_name}-pose2",
    )

    today_precision = today_correct / today_predicted if today_predicted > 0 else 0.0
    pose_v2_precision = pose_v2_correct / pose_v2_predicted if pose_v2_predicted > 0 else 0.0
    delta_vs_today_pp = (pose_v2_precision - today_precision) * 100.0
    delta_vs_baseline_pp = (pose_v2_precision - baseline_precision) * 100.0

    if nm_pose_v2 > 0:
        imp_avg = imp_pose_v2 / nm_pose_v2
    else:
        imp_avg = imp_pose_v2
    top_features = sorted(
        zip(POSE_FEATURE_NAMES, imp_avg.tolist()),
        key=lambda x: x[1], reverse=True,
    )[:5]

    return TypeResult(
        type_name=type_name,
        n_gt=n_gt,
        n_predicted=pose_v2_predicted,
        baseline_correct=baseline_correct,
        today_correct=today_correct,
        pose_v2_correct=pose_v2_correct,
        baseline_precision=baseline_precision,
        today_precision=today_precision,
        pose_v2_precision=pose_v2_precision,
        delta_vs_today_pp=delta_vs_today_pp,
        delta_vs_baseline_pp=delta_vs_baseline_pp,
        top_features=top_features,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_report(results: list[TypeResult], pose_coverage: dict[str, int]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Aggregate stats
    n_gt_tot = sum(r.n_gt for r in results)
    base_tot = sum(r.baseline_correct for r in results)
    today_tot = sum(r.today_correct for r in results)
    pose_v2_tot = sum(r.pose_v2_correct for r in results)
    n_pred_tot = sum(r.n_predicted for r in results)
    base_agg = base_tot / n_gt_tot if n_gt_tot else 0.0
    today_agg = today_tot / n_pred_tot if n_pred_tot else 0.0
    pose_v2_agg = pose_v2_tot / n_pred_tot if n_pred_tot else 0.0

    # Decision criteria
    types_lift_5pp = [r for r in results if r.delta_vs_today_pp >= 5.0]
    types_regress_2pp = [r for r in results if r.delta_vs_today_pp <= -2.0]

    pose_in_top5 = []
    for r in results:
        for name, _imp in r.top_features:
            if name.startswith("pv2_"):
                pose_in_top5.append(r.type_name)
                break

    ship = (
        len(types_lift_5pp) >= 2
        and len(types_regress_2pp) == 0
        and len(pose_in_top5) >= 1
    )
    if ship:
        verdict = "SHIP-POSE-V2"
    elif len(types_lift_5pp) >= 1 and len(types_regress_2pp) == 0:
        verdict = "MIXED (single-type lift; pose useful but partial)"
    elif len(types_regress_2pp) > 0:
        verdict = "FLAT (regression on >=1 type)"
    else:
        verdict = "FLAT"

    # JSON
    out = {
        "results": [
            {
                "type": r.type_name,
                "n_gt": r.n_gt,
                "n_predicted": r.n_predicted,
                "baseline_correct": r.baseline_correct,
                "today_correct": r.today_correct,
                "pose_v2_correct": r.pose_v2_correct,
                "baseline_precision": r.baseline_precision,
                "today_precision": r.today_precision,
                "pose_v2_precision": r.pose_v2_precision,
                "delta_vs_today_pp": r.delta_vs_today_pp,
                "delta_vs_baseline_pp": r.delta_vs_baseline_pp,
                "top_features": r.top_features,
            }
            for r in results
        ],
        "aggregate": {
            "baseline_pct": base_agg * 100.0,
            "today_pct": today_agg * 100.0,
            "pose_v2_pct": pose_v2_agg * 100.0,
        },
        "verdict": verdict,
        "videos": VIDEO_NAMES,
        "pose_coverage": pose_coverage,
        "feature_names": POSE_FEATURE_NAMES,
        "today_feature_count": len(TODAY_FEATURE_NAMES),
        "pose_v2_feature_count": len(POSE_V2_FEATURE_NAMES),
    }
    JSON_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote JSON: {JSON_PATH}")

    # Markdown
    lines: list[str] = []
    lines.append("# Pose v2 probe — does rich pose lift learned attribution? (2026-05-14)")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- 12 panel videos: {', '.join(VIDEO_NAMES)}")
    lines.append(f"- {len(TODAY_FEATURE_NAMES)} today's features + {len(POSE_V2_FEATURE_NAMES)} new pose features = {len(POSE_FEATURE_NAMES)} total")
    lines.append("- Per-action-type HistGradientBoostingClassifier, leave-one-video-out CV")
    lines.append("- Pose features extracted from yolo11s-pose cached keypoints (training_data/pose_cache)")
    lines.append(f"- Pose coverage: {pose_coverage['rallies_with_pose']} / {pose_coverage['rallies_total']} rallies have cached pose data")
    lines.append("")
    lines.append("## Per-type precision")
    lines.append("")
    lines.append("| Type | N GT | Baseline (S0) | Today's GBM | Pose v2 GBM | Δ vs today | Δ vs S0 |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r.type_name} | {r.n_gt} | "
            f"{r.baseline_correct}/{r.n_gt} = {r.baseline_precision * 100:.1f}% | "
            f"{r.today_correct}/{r.n_predicted} = {r.today_precision * 100:.1f}% | "
            f"{r.pose_v2_correct}/{r.n_predicted} = {r.pose_v2_precision * 100:.1f}% | "
            f"{r.delta_vs_today_pp:+.1f}pp | "
            f"{r.delta_vs_baseline_pp:+.1f}pp |"
        )
    lines.append("")
    lines.append(
        f"**Aggregate**: baseline {base_tot}/{n_gt_tot} = {base_agg * 100:.1f}% → "
        f"today {today_tot}/{n_pred_tot} = {today_agg * 100:.1f}% → "
        f"pose-v2 {pose_v2_tot}/{n_pred_tot} = {pose_v2_agg * 100:.1f}%"
    )
    lines.append("")
    lines.append("## Per-type top-5 features (Pose v2)")
    lines.append("")
    for r in results:
        lines.append(f"### {r.type_name}")
        has_pose = any(name.startswith("pv2_") for name, _ in r.top_features)
        flag = " (POSE)" if has_pose else ""
        for name, imp in r.top_features:
            marker = " **[pose]**" if name.startswith("pv2_") else ""
            lines.append(f"- `{name}`: {imp:.4f}{marker}")
        lines.append(f"_pose-in-top5: {has_pose}_{flag}")
        lines.append("")

    lines.append("## Decision criteria")
    lines.append("")
    lines.append("Ship gate vs today's GBM (not just baseline):")
    lines.append("- ≥5pp lift on ≥2 action types")
    lines.append("- No type regresses by >2pp")
    lines.append("- ≥1 type has a pose feature in top-5")
    lines.append("")
    lines.append(
        f"- Types with ≥5pp lift vs today: {len(types_lift_5pp)} "
        f"({', '.join(r.type_name for r in types_lift_5pp) or 'none'})",
    )
    lines.append(
        f"- Types regressing by ≥2pp vs today: {len(types_regress_2pp)} "
        f"({', '.join(r.type_name for r in types_regress_2pp) or 'none'})",
    )
    lines.append(
        f"- Types with pose feature in top-5: {len(pose_in_top5)} "
        f"({', '.join(pose_in_top5) or 'none'})",
    )
    lines.append("")
    lines.append(f"## Verdict: **{verdict}**")
    lines.append("")

    MD_PATH.write_text("\n".join(lines))
    print(f"Wrote markdown: {MD_PATH}")
    return verdict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=== Pose v2 attribution probe (2026-05-14) ===")
    print(f"Videos: {VIDEO_NAMES}")
    print(f"GT_FRAME_TOL={GT_FRAME_TOL}")
    print()

    print("Step 1/5: Fetching DB data (same loader as today's experiment)...")
    rally_contacts, gt_rows = fetch_data(VIDEO_NAMES)
    print(f"  Rallies with contacts: {len(rally_contacts)}, GT rows: {len(gt_rows)}")
    by_type = Counter(g.action for g in gt_rows)
    print(f"  GT per type: {dict(by_type)}")
    print()

    print("Step 2/5: Loading pose cache for each rally...")
    pose_coverage = {"rallies_total": 0, "rallies_with_pose": 0}
    rallies_with_gt = set()
    for g in gt_rows:
        rallies_with_gt.add(g.rally_id)
    for rid in rallies_with_gt:
        pose_coverage["rallies_total"] += 1
        if (POSE_CACHE_DIR / f"{rid}.npz").exists():
            pose_coverage["rallies_with_pose"] += 1
    print(f"  Pose cache: {pose_coverage['rallies_with_pose']}/{pose_coverage['rallies_total']} rallies")
    print()

    print("Step 3/5: Building per-candidate dataset (30 + 14 features)...")
    rows = build_dataset_pose_v2(rally_contacts, gt_rows)
    by_type_rows = Counter(r.action_uc for r in rows)
    print(f"  Total dataset rows: {len(rows)} ({len(rows) // 4} contacts × 4 candidates)")
    print(f"  Per type (rows): {dict(by_type_rows)}")
    print()

    print("Step 4/5: Per-type LOVO CV (today's GBM + pose-v2 GBM)...")
    results: list[TypeResult] = []
    for atype in ACTION_TYPES_TO_TRAIN:
        print(f"=== {atype} ===")
        r = evaluate_type(atype, rows, VIDEO_NAMES)
        print(
            f"  RESULT [{atype}]: N={r.n_gt} | "
            f"baseline {r.baseline_correct}/{r.n_gt}={r.baseline_precision * 100:.1f}% | "
            f"today {r.today_correct}/{r.n_predicted}={r.today_precision * 100:.1f}% | "
            f"pose-v2 {r.pose_v2_correct}/{r.n_predicted}={r.pose_v2_precision * 100:.1f}% | "
            f"Δ-today {r.delta_vs_today_pp:+.1f}pp | Δ-S0 {r.delta_vs_baseline_pp:+.1f}pp",
        )
        results.append(r)
    print()

    print("Step 5/5: Writing report...")
    write_report(results, pose_coverage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
