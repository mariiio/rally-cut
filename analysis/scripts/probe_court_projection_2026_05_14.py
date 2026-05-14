"""3D court projection probe — 2026-05-14.

Tests whether projecting candidate-foot + ball positions to court-coordinate
space (via the per-video homography) provides a useful signal that the
existing image-coord features miss — specifically for cross-team occluded
attacks where the visible "near" player is in front of the actual toucher.

Setup:
  - Same 12 trusted videos, same 641 GT rows, same LOVO CV scaffold as
    `probe_learned_attribution_per_type_2026_05_14.py`.
  - Adds 5 court-projection features to the existing 30:
      * court_dist_to_ball             (m)
      * court_dist_to_ball_rank        (0..3)
      * court_x, court_y               (m)
      * court_dist_minus_pixel_dist    (m)
      * court_pixel_rank_disagree      (0/1)
  - Pure-numpy projection (homography 3x3 matrix). NO model inference.

Outputs:
  - analysis/reports/court_projection_probe_2026_05_14/results.{md,json}

Usage:
    cd analysis
    uv run python scripts/probe_court_projection_2026_05_14.py
"""
from __future__ import annotations

import json
import math
import sys
import time
import tracemalloc
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

# Reuse the existing data hydration + feature extraction so we don't
# accidentally drift from the baseline experiment.
from scripts.probe_learned_attribution_per_type_2026_05_14 import (  # noqa: E402
    ACTION_TYPES_TO_TRAIN,
    FEATURE_NAMES as BASE_FEATURE_NAMES,
    GT_FRAME_TOL,
    VIDEO_NAMES,
    ContactCtx,
    GTRow,
    _match_gt_to_ctx,
    compute_features_for_candidate,
    fetch_data,
)

from rallycut.evaluation.tracking.db import get_connection  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

REPORT_DIR = HERE.parent / "reports" / "court_projection_probe_2026_05_14"
JSON_PATH = REPORT_DIR / "results.json"
MD_PATH = REPORT_DIR / "results.md"

# New 3D-projection features appended to BASE_FEATURE_NAMES.
COURT_FEATURE_NAMES = [
    "court_dist_to_ball",
    "court_dist_to_ball_rank",
    "court_x",
    "court_y",
    "court_dist_minus_pixel_dist",
    "court_pixel_rank_disagree",
]
ALL_FEATURE_NAMES = list(BASE_FEATURE_NAMES) + COURT_FEATURE_NAMES


# ---------------------------------------------------------------------------
# Court homography hydration
# ---------------------------------------------------------------------------

# Beach VB court dims (must match rallycut.court.calibration)
COURT_WIDTH = 8.0
COURT_LENGTH = 16.0


def _build_homography(corners_normalized: list[dict[str, float]]) -> np.ndarray | None:
    """Compute 3x3 homography from 4 normalized image corners to court meters.

    Mirrors `CourtCalibrator.calibrate` (uses OpenCV's findHomography), but
    keeps the dependency surface here so the probe doesn't pull in the
    class wrapper. Returns None if calibration fails.
    """
    if not corners_normalized or len(corners_normalized) != 4:
        return None
    try:
        src_pts = np.array(
            [[float(c["x"]), float(c["y"])] for c in corners_normalized],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError):
        return None
    dst_pts = np.array(
        [
            [0.0, 0.0],
            [COURT_WIDTH, 0.0],
            [COURT_WIDTH, COURT_LENGTH],
            [0.0, COURT_LENGTH],
        ],
        dtype=np.float64,
    )
    try:
        import cv2
        h, _ = cv2.findHomography(src_pts, dst_pts)
        if h is None:
            return None
        return np.array(h, dtype=np.float64)
    except Exception:
        return None


def _project(h: np.ndarray, x: float, y: float) -> tuple[float, float] | None:
    """Apply 3x3 homography to a (normalized) image point. Returns court (m)."""
    pt = np.array([x, y, 1.0], dtype=np.float64)
    out = h @ pt
    if out[2] == 0 or not np.isfinite(out[2]):
        return None
    cx = out[0] / out[2]
    cy = out[1] / out[2]
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return None
    return float(cx), float(cy)


def fetch_homographies(video_names: list[str]) -> dict[str, np.ndarray]:
    """Map video_name -> 3x3 homography matrix. Missing videos absent."""
    out: dict[str, np.ndarray] = {}
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT name, court_calibration_json FROM videos WHERE name = ANY(%s)",
            [video_names],
        )
        # Multiple rows possible (e.g. yeye has 2 video rows). Build per
        # name; if more than one is valid, last-write-wins (matches the
        # base experiment which doesn't distinguish either).
        for name, cal in cur.fetchall():
            if not cal or not isinstance(cal, list) or len(cal) != 4:
                continue
            h = _build_homography(cal)
            if h is not None:
                out[str(name)] = h
    return out


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


def _foot_from_bbox(bbox_cx_cy: tuple[float, float], height: float | None) -> tuple[float, float]:
    """Bbox center (cx, cy) -> foot point (cx, cy + height/2).

    height is in normalized coords; if None, use cy alone (caller already
    fell back). The base script tracks only (cx, cy); to keep parity we
    use the bbox center as the projection anchor (no per-frame height
    available without re-fetching). The court projection is monotonic
    enough on cy that the *rank* signal we care about isn't sensitive to
    the foot-offset choice — and the base 30 features already use bbox
    center for everything else.
    """
    return bbox_cx_cy


def compute_court_features(
    ctx: ContactCtx,
    cand_tid: int,
    pixel_rank: int,
    homography: np.ndarray | None,
    all_cand_court_dists: dict[int, float] | None,
) -> tuple[list[float], bool]:
    """Returns (court_feature_vec, projected_ok).

    If homography is missing or projection fails for ball/candidate, returns
    NaN-padded features and projected_ok=False so the row can be excluded
    from the F3-shape lift measurement (but still trained/predicted on as
    NaN — HistGradientBoosting handles NaN natively).
    """
    if homography is None or ctx.ball_at_contact is None:
        return [float("nan")] * len(COURT_FEATURE_NAMES), False

    bbox_at = ctx.cand_bbox_at.get(cand_tid)
    if bbox_at is None:
        return [float("nan")] * len(COURT_FEATURE_NAMES), False

    # Project candidate position (use bbox center — see _foot_from_bbox note).
    cand_proj = _project(homography, bbox_at[0], bbox_at[1])
    ball_proj = _project(homography, ctx.ball_at_contact[0], ctx.ball_at_contact[1])
    if cand_proj is None or ball_proj is None:
        return [float("nan")] * len(COURT_FEATURE_NAMES), False

    court_dist = math.hypot(cand_proj[0] - ball_proj[0], cand_proj[1] - ball_proj[1])

    # Rank among the 4 candidates for this contact (court-distance ordering).
    court_rank: float
    if all_cand_court_dists is None or len(all_cand_court_dists) < 2:
        court_rank = -1.0
    else:
        sorted_tids = sorted(all_cand_court_dists.keys(), key=lambda t: all_cand_court_dists[t])
        try:
            court_rank = float(sorted_tids.index(cand_tid))
        except ValueError:
            court_rank = float(len(sorted_tids))

    # Pixel rank vs court rank disagreement.
    rank_disagree = 1.0 if (pixel_rank >= 0 and court_rank >= 0 and pixel_rank != court_rank) else 0.0

    # court_dist_minus_pixel_dist — units are different (meters vs normalized
    # image), so this is a heuristic signal not a calibrated delta. The
    # GBM is free to learn whatever scale relationship exists per-action-type.
    bbox_at_for_pix = bbox_at
    pixel_dist = math.hypot(
        bbox_at_for_pix[0] - ctx.ball_at_contact[0],
        bbox_at_for_pix[1] - ctx.ball_at_contact[1],
    )
    court_minus_pixel = court_dist - pixel_dist

    return [
        court_dist,
        court_rank,
        cand_proj[0],
        cand_proj[1],
        court_minus_pixel,
        rank_disagree,
    ], True


def _precompute_court_dists_for_contact(
    ctx: ContactCtx, cand_pool: list[int], homography: np.ndarray | None,
) -> dict[int, float] | None:
    """Return tid -> court_dist_to_ball for all 4 candidates at this contact."""
    if homography is None or ctx.ball_at_contact is None:
        return None
    ball_proj = _project(homography, ctx.ball_at_contact[0], ctx.ball_at_contact[1])
    if ball_proj is None:
        return None
    out: dict[int, float] = {}
    for tid in cand_pool:
        bbox = ctx.cand_bbox_at.get(tid)
        if bbox is None:
            continue
        cand_proj = _project(homography, bbox[0], bbox[1])
        if cand_proj is None:
            continue
        out[tid] = math.hypot(cand_proj[0] - ball_proj[0], cand_proj[1] - ball_proj[1])
    return out if out else None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class DatasetRow:
    video: str
    rally_id: str
    frame: int
    action_uc: str
    cand_tid: int
    is_gt: int
    is_pipeline_pick: int
    features: list[float]                # length len(ALL_FEATURE_NAMES)
    has_court_projection: int            # 1 if 3D features valid for THIS candidate
    prev_team_cross: int                 # 1 if this contact is in F3-shape subset
    pl_tid: int | None


def build_dataset(
    rally_contacts: dict[str, list[ContactCtx]],
    gt_rows: list[GTRow],
    homographies: dict[str, np.ndarray],
) -> tuple[list[DatasetRow], dict[str, int]]:
    """Returns (rows, stats).

    stats keys:
      - total_gt
      - matched_to_ctx
      - unresolved
      - skipped_no_4_team
      - contacts_no_homography (per-contact, all 4 cands NaN)
      - contacts_court_projection_ok
    """
    gt_by_rally: dict[str, list[GTRow]] = defaultdict(list)
    for g in gt_rows:
        gt_by_rally[g.rally_id].append(g)

    rows: list[DatasetRow] = []
    stats: dict[str, int] = {
        "total_gt": 0, "matched_to_ctx": 0, "unresolved": 0,
        "skipped_no_4_team": 0,
        "contacts_no_homography": 0, "contacts_court_projection_ok": 0,
    }

    for rally_id, ctxs in rally_contacts.items():
        gts = gt_by_rally.get(rally_id, [])
        if not gts:
            continue
        pairs = _match_gt_to_ctx(ctxs, gts)
        for g, c in pairs:
            stats["total_gt"] += 1
            if g.resolved_tid is None:
                stats["unresolved"] += 1
                continue
            if c is None:
                continue
            stats["matched_to_ctx"] += 1
            cand_pool = sorted(c.team_assignments.keys())
            if len(cand_pool) != 4:
                stats["skipped_no_4_team"] += 1
                continue

            homography = homographies.get(c.video)
            court_dists = _precompute_court_dists_for_contact(c, cand_pool, homography)
            if court_dists is None or len(court_dists) < 2:
                stats["contacts_no_homography"] += 1
            else:
                stats["contacts_court_projection_ok"] += 1

            # F3-shape subset flag (uses ctx.prev_team vs ctx.expected_team).
            prev_team_cross = int(
                c.prev_team is not None
                and c.expected_team is not None
                and c.prev_team != c.expected_team
            )

            for cand_tid in cand_pool:
                base_feats = compute_features_for_candidate(c, cand_tid)
                # pixel rank: find this tid in c.candidates ordering
                pixel_rank = -1
                for r, (tid, _d) in enumerate(c.candidates):
                    if tid == cand_tid:
                        pixel_rank = r
                        break
                if pixel_rank < 0:
                    pixel_rank = len(c.candidates)
                court_feats, _ok = compute_court_features(
                    c, cand_tid, pixel_rank, homography, court_dists,
                )
                rows.append(DatasetRow(
                    video=c.video, rally_id=c.rally_id, frame=c.frame,
                    action_uc=g.action.upper(), cand_tid=cand_tid,
                    is_gt=int(cand_tid == g.resolved_tid),
                    is_pipeline_pick=int(c.pl_tid is not None and cand_tid == c.pl_tid),
                    features=base_feats + court_feats,
                    has_court_projection=int(court_dists is not None),
                    prev_team_cross=prev_team_cross,
                    pl_tid=c.pl_tid,
                ))

    print(
        f"Dataset build: GT={stats['total_gt']} "
        f"matched_to_ctx={stats['matched_to_ctx']} "
        f"unresolved={stats['unresolved']} "
        f"skipped_no_4_team={stats['skipped_no_4_team']} "
        f"contacts_no_homography={stats['contacts_no_homography']} "
        f"contacts_with_court_proj={stats['contacts_court_projection_ok']}"
    )
    return rows, stats


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------

@dataclass
class TypeResult:
    type_name: str
    n_gt: int
    n_predicted: int
    baseline_correct: int
    today_gbm_correct: int                # baseline-GBM with 30 features
    learned_3d_correct: int               # GBM with 35 features
    baseline_precision: float
    today_gbm_precision: float
    learned_3d_precision: float
    delta_pp_vs_today_gbm: float
    delta_pp_vs_pipeline: float
    n_baseline_no_pick: int
    top_features: list[tuple[str, float]]
    # F3-shape subset
    f3_n: int
    f3_baseline_correct: int
    f3_today_gbm_correct: int
    f3_learned_3d_correct: int


def _train_predict_per_type(
    type_name: str,
    type_rows: list[DatasetRow],
    feature_slice: slice,
    feature_names: list[str],
    do_perm_importance: bool,
    video_names: list[str],
) -> tuple[int, int, dict[tuple[str, str, int], int], np.ndarray, int]:
    """LOVO CV trainer/predictor for one feature config.

    Returns (n_predicted, n_correct, picks_by_contact_key, feature_importances_avg, n_perm_models).
    picks_by_contact_key: contact_key -> 1 if pick is GT, else 0.
    """
    contact_groups: dict[tuple[str, str, int], list[DatasetRow]] = defaultdict(list)
    for r in type_rows:
        contact_groups[(r.video, r.rally_id, r.frame)].append(r)

    picks: dict[tuple[str, str, int], int] = {}
    feat_imp_sum = np.zeros(len(feature_names), dtype=np.float64)
    n_models = 0
    n_predicted = 0
    n_correct = 0

    for hold_out in video_names:
        train_rows = [r for r in type_rows if r.video != hold_out]
        test_rows = [r for r in type_rows if r.video == hold_out]
        if not train_rows or not test_rows:
            continue
        X_train = np.array([r.features[feature_slice] for r in train_rows], dtype=np.float32)
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
            print(f"  [{type_name}/{feature_names[-1]}] hold={hold_out} fit failed: {e}")
            continue

        if do_perm_importance:
            try:
                from sklearn.inspection import permutation_importance
                sub = np.random.default_rng(42).choice(
                    len(X_train), size=min(800, len(X_train)), replace=False,
                )
                pi = permutation_importance(
                    model, X_train[sub], y_train[sub],
                    n_repeats=3, random_state=42, n_jobs=1,
                )
                feat_imp_sum += pi.importances_mean
                n_models += 1
            except Exception as e:
                print(f"  [{type_name}] perm-importance failed on hold={hold_out}: {e}")

        test_groups: dict[tuple[str, str, int], list[DatasetRow]] = defaultdict(list)
        for r in test_rows:
            test_groups[(r.video, r.rally_id, r.frame)].append(r)
        for key, rs in test_groups.items():
            X_test = np.array([r.features[feature_slice] for r in rs], dtype=np.float32)
            probs = model.predict_proba(X_test)[:, 1]
            best_idx = int(np.argmax(probs))
            pick = rs[best_idx]
            n_predicted += 1
            is_correct = int(pick.is_gt == 1)
            if is_correct:
                n_correct += 1
            picks[key] = is_correct
        print(
            f"  [{type_name}] hold={hold_out:>4}: train={len(train_rows):>4} "
            f"test={len(test_rows):>3} cum={n_correct}/{n_predicted}"
        )

    feat_imp_avg = feat_imp_sum / n_models if n_models else feat_imp_sum
    return n_predicted, n_correct, picks, feat_imp_avg, n_models


def evaluate_type(
    type_name: str, all_rows: list[DatasetRow], video_names: list[str],
) -> TypeResult:
    type_rows = [r for r in all_rows if r.action_uc == type_name]
    contact_groups: dict[tuple[str, str, int], list[DatasetRow]] = defaultdict(list)
    for r in type_rows:
        contact_groups[(r.video, r.rally_id, r.frame)].append(r)

    n_gt = len(contact_groups)

    # ---- Baseline: pipeline pick ----
    baseline_correct = 0
    baseline_no_pick = 0
    baseline_picks: dict[tuple[str, str, int], int] = {}
    for key, rs in contact_groups.items():
        pl_picks = [r for r in rs if r.is_pipeline_pick]
        if not pl_picks:
            baseline_no_pick += 1
            baseline_picks[key] = 0
            continue
        is_correct = int(any(r.is_gt == 1 for r in pl_picks))
        if is_correct:
            baseline_correct += 1
        baseline_picks[key] = is_correct

    baseline_precision = baseline_correct / n_gt if n_gt > 0 else 0.0

    base_slice = slice(0, len(BASE_FEATURE_NAMES))
    full_slice = slice(0, len(ALL_FEATURE_NAMES))

    # ---- Today's GBM (30 features) ----
    print(f"--- {type_name} :: today's GBM (30 features) ---")
    n_pred_base, n_corr_base, picks_base, _fi_base, _nm_base = _train_predict_per_type(
        type_name, type_rows, base_slice, list(BASE_FEATURE_NAMES),
        do_perm_importance=False, video_names=video_names,
    )

    # ---- Learned-3D GBM (35 features) ----
    print(f"--- {type_name} :: + 3D features (35 features) ---")
    n_pred_3d, n_corr_3d, picks_3d, fi_3d, _nm_3d = _train_predict_per_type(
        type_name, type_rows, full_slice, list(ALL_FEATURE_NAMES),
        do_perm_importance=True, video_names=video_names,
    )

    today_gbm_precision = n_corr_base / n_pred_base if n_pred_base else 0.0
    learned_3d_precision = n_corr_3d / n_pred_3d if n_pred_3d else 0.0

    # ---- F3-shape subset (cross-team flow) ----
    f3_keys = {
        (r.video, r.rally_id, r.frame)
        for r in type_rows
        if r.prev_team_cross == 1
    }
    f3_n = len(f3_keys)
    f3_baseline_correct = sum(1 for k in f3_keys if baseline_picks.get(k, 0) == 1)
    f3_today_gbm_correct = sum(1 for k in f3_keys if picks_base.get(k, 0) == 1)
    f3_learned_3d_correct = sum(1 for k in f3_keys if picks_3d.get(k, 0) == 1)

    fi_ranked = sorted(
        zip(ALL_FEATURE_NAMES, fi_3d.tolist()),
        key=lambda x: x[1], reverse=True,
    )[:5]

    return TypeResult(
        type_name=type_name, n_gt=n_gt, n_predicted=n_pred_3d,
        baseline_correct=baseline_correct,
        today_gbm_correct=n_corr_base,
        learned_3d_correct=n_corr_3d,
        baseline_precision=baseline_precision,
        today_gbm_precision=today_gbm_precision,
        learned_3d_precision=learned_3d_precision,
        delta_pp_vs_today_gbm=(learned_3d_precision - today_gbm_precision) * 100.0,
        delta_pp_vs_pipeline=(learned_3d_precision - baseline_precision) * 100.0,
        n_baseline_no_pick=baseline_no_pick,
        top_features=fi_ranked,
        f3_n=f3_n,
        f3_baseline_correct=f3_baseline_correct,
        f3_today_gbm_correct=f3_today_gbm_correct,
        f3_learned_3d_correct=f3_learned_3d_correct,
    )


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_report(
    results: list[TypeResult], block_n: int, video_names: list[str],
    stats: dict[str, int], runtime_s: float, max_mem_mb: float,
) -> str:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Aggregates
    n_gt_total = sum(r.n_gt for r in results)
    n_pred_total = sum(r.n_predicted for r in results)
    baseline_total = sum(r.baseline_correct for r in results)
    today_gbm_total = sum(r.today_gbm_correct for r in results)
    learned_3d_total = sum(r.learned_3d_correct for r in results)

    agg_baseline = baseline_total / n_gt_total if n_gt_total else 0.0
    agg_today_gbm = today_gbm_total / n_pred_total if n_pred_total else 0.0
    agg_learned_3d = learned_3d_total / n_pred_total if n_pred_total else 0.0

    # F3-shape aggregates (ATTACK subset is primary; aggregate all types too)
    f3_n_total = sum(r.f3_n for r in results)
    f3_base_total = sum(r.f3_baseline_correct for r in results)
    f3_today_total = sum(r.f3_today_gbm_correct for r in results)
    f3_3d_total = sum(r.f3_learned_3d_correct for r in results)

    # ATTACK-only F3 numbers (the hypothesis-of-record)
    attack_res = next((r for r in results if r.type_name == "ATTACK"), None)

    # Verdict
    types_lift_5pp = [r for r in results if r.delta_pp_vs_today_gbm >= 5.0]
    types_regress_2pp = [r for r in results if r.delta_pp_vs_today_gbm <= -2.0]
    # F3 lift (ATTACK only): is the 3D model >= 10pp better than today's GBM on F3 ATTACK?
    if attack_res and attack_res.f3_n > 0:
        f3_attack_today = attack_res.f3_today_gbm_correct / attack_res.f3_n
        f3_attack_3d = attack_res.f3_learned_3d_correct / attack_res.f3_n
        f3_attack_lift_pp = (f3_attack_3d - f3_attack_today) * 100.0
    else:
        f3_attack_today = float("nan")
        f3_attack_3d = float("nan")
        f3_attack_lift_pp = 0.0

    ship_3d = len(types_lift_5pp) >= 2 and len(types_regress_2pp) == 0
    targeted_3d = f3_attack_lift_pp >= 10.0 and attack_res is not None and attack_res.f3_n >= 5

    if ship_3d:
        verdict = "SHIP-3D-FEATURES"
        next_probe = (
            "Design proper 3D-feature engineering workstream: cache court projections at "
            "producer time + retrain the pipeline classifier with the 5 new features."
        )
    elif targeted_3d:
        verdict = "TARGETED-3D"
        next_probe = (
            "Ship a narrow rule: for F3-shape cross-team attacks, override pipeline pick "
            "with the candidate that minimizes court_dist_to_ball. Validate on the F3 "
            "subset against a frozen baseline."
        )
    else:
        verdict = "SKIP-3D"
        next_probe = (
            "Try scaled-down Pose v2 next: ~100 contacts, ~5 min, pure-numpy pose-feature "
            "extraction over existing position snapshots (no model inference)."
        )

    lines: list[str] = []
    lines.append("# 3D court projection probe — results 2026-05-14")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- {n_gt_total} GT rows (matched-to-pipeline-contact), 12 trusted videos, leave-one-video-out CV.")
    lines.append(f"- Baseline (today's experiment): {len(BASE_FEATURE_NAMES)} features. Aggregate precision: {agg_today_gbm*100:.1f}%.")
    lines.append(f"- New: {len(BASE_FEATURE_NAMES)} + {len(COURT_FEATURE_NAMES)} court-projection features = {len(ALL_FEATURE_NAMES)} features. Aggregate precision: {agg_learned_3d*100:.1f}%.")
    lines.append(f"- Pipeline-baseline (current production pick): {agg_baseline*100:.1f}%.")
    lines.append("")
    if stats["contacts_no_homography"] > 0:
        lines.append(
            f"- Contacts with no usable homography (excluded from 3D feature contribution but kept "
            f"in baseline + today's GBM eval): **{stats['contacts_no_homography']}**. "
            f"Court-projection-ok contacts: {stats['contacts_court_projection_ok']}."
        )
        lines.append("")
    else:
        lines.append(
            f"- All {stats['contacts_court_projection_ok']} matched-to-ctx contacts had a valid "
            f"per-video homography (no court-projection-unavailable exclusions)."
        )
        lines.append("")

    lines.append("## Per-type precision lift")
    lines.append("")
    lines.append(f"| Type | N GT | Baseline (today's GBM, {len(BASE_FEATURE_NAMES)}f) | + 3D features ({len(ALL_FEATURE_NAMES)}f) | Δ vs today's GBM | Δ vs pipeline baseline |")
    lines.append("|---|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r.type_name} | {r.n_gt} | "
            f"{r.today_gbm_correct}/{r.n_predicted} = {r.today_gbm_precision*100:.1f}% | "
            f"{r.learned_3d_correct}/{r.n_predicted} = {r.learned_3d_precision*100:.1f}% | "
            f"{r.delta_pp_vs_today_gbm:+.1f}pp | "
            f"{r.delta_pp_vs_pipeline:+.1f}pp |"
        )
    if block_n > 0:
        lines.append(f"| BLOCK | {block_n} | insufficient | insufficient | - | - |")
    lines.append("")
    lines.append(
        f"**Aggregate** (excluding BLOCK): pipeline {baseline_total}/{n_gt_total}={agg_baseline*100:.1f}% → "
        f"today's GBM {today_gbm_total}/{n_pred_total}={agg_today_gbm*100:.1f}% → "
        f"+3D {learned_3d_total}/{n_pred_total}={agg_learned_3d*100:.1f}% "
        f"(Δ vs today's GBM {((agg_learned_3d-agg_today_gbm)*100):+.1f}pp, "
        f"Δ vs pipeline {((agg_learned_3d-agg_baseline)*100):+.1f}pp)"
    )
    lines.append("")

    lines.append(f"## Top features per type ({len(ALL_FEATURE_NAMES)}-feature model, by permutation importance)")
    lines.append("")
    for r in results:
        lines.append(f"### {r.type_name}")
        for name, imp in r.top_features:
            tag = "  ⭐3D" if name in COURT_FEATURE_NAMES else ""
            lines.append(f"- `{name}`: {imp:.4f}{tag}")
        lines.append("")

    lines.append("## Per-error subset: F3-shape (cross-team-flow contacts)")
    lines.append("")
    lines.append(
        "Definition: contacts where the prior action's player was on the *other* team "
        "from the expected-team in the rally chain — the occlusion-flip cases."
    )
    lines.append("")
    lines.append("### F3-shape, ATTACK-only (hypothesis-of-record)")
    lines.append("")
    if attack_res and attack_res.f3_n > 0:
        lines.append(f"- N = {attack_res.f3_n}")
        lines.append(
            f"- Pipeline baseline: {attack_res.f3_baseline_correct}/{attack_res.f3_n} = "
            f"{attack_res.f3_baseline_correct/attack_res.f3_n*100:.1f}%"
        )
        lines.append(
            f"- Today's GBM (30f): {attack_res.f3_today_gbm_correct}/{attack_res.f3_n} = "
            f"{f3_attack_today*100:.1f}%"
        )
        lines.append(
            f"- +3D (35f): {attack_res.f3_learned_3d_correct}/{attack_res.f3_n} = "
            f"{f3_attack_3d*100:.1f}%"
        )
        lines.append(f"- Δ (35f vs 30f): {f3_attack_lift_pp:+.1f}pp")
    else:
        lines.append("- N = 0 (no F3-shape ATTACK contacts in dataset)")
    lines.append("")
    lines.append("### F3-shape, all types pooled")
    lines.append("")
    if f3_n_total > 0:
        lines.append(f"- N = {f3_n_total}")
        lines.append(f"- Pipeline baseline: {f3_base_total}/{f3_n_total} = {f3_base_total/f3_n_total*100:.1f}%")
        lines.append(f"- Today's GBM (30f): {f3_today_total}/{f3_n_total} = {f3_today_total/f3_n_total*100:.1f}%")
        lines.append(f"- +3D (35f): {f3_3d_total}/{f3_n_total} = {f3_3d_total/f3_n_total*100:.1f}%")
        lines.append(f"- Δ (35f vs 30f): {(f3_3d_total/f3_n_total - f3_today_total/f3_n_total)*100:+.1f}pp")
    else:
        lines.append("- N = 0")
    lines.append("")

    lines.append("## Decision criteria")
    lines.append("")
    lines.append("- **SHIP-3D-FEATURES** if ≥5pp lift on ≥2 action types AND no type regresses by >2pp (vs today's GBM).")
    lines.append("- **TARGETED-3D** if F3-shape ATTACK subset lifts ≥10pp even without aggregate lift.")
    lines.append("- **SKIP-3D** if flat across the board.")
    lines.append("")
    lines.append(
        f"- Types with ≥5pp lift (vs today's GBM): {len(types_lift_5pp)} "
        f"({', '.join(r.type_name for r in types_lift_5pp) or 'none'})"
    )
    lines.append(
        f"- Types regressing by ≥2pp (vs today's GBM): {len(types_regress_2pp)} "
        f"({', '.join(r.type_name for r in types_regress_2pp) or 'none'})"
    )
    if attack_res and attack_res.f3_n > 0:
        lines.append(f"- F3-shape ATTACK lift: {f3_attack_lift_pp:+.1f}pp (threshold +10pp)")
    lines.append("")
    lines.append(f"## Verdict: **{verdict}**")
    lines.append("")
    lines.append(f"## Recommended next probe")
    lines.append("")
    lines.append(next_probe)
    lines.append("")
    lines.append("## Runtime")
    lines.append("")
    lines.append(f"- Total: {runtime_s:.1f}s")
    lines.append(f"- Peak Python heap (tracemalloc): {max_mem_mb:.1f} MB")
    lines.append("")

    MD_PATH.write_text("\n".join(lines))

    out_json = {
        "results": [
            {
                "type": r.type_name,
                "n_gt": r.n_gt,
                "n_predicted": r.n_predicted,
                "baseline_correct": r.baseline_correct,
                "today_gbm_correct": r.today_gbm_correct,
                "learned_3d_correct": r.learned_3d_correct,
                "baseline_precision": r.baseline_precision,
                "today_gbm_precision": r.today_gbm_precision,
                "learned_3d_precision": r.learned_3d_precision,
                "delta_pp_vs_today_gbm": r.delta_pp_vs_today_gbm,
                "delta_pp_vs_pipeline": r.delta_pp_vs_pipeline,
                "f3_n": r.f3_n,
                "f3_baseline_correct": r.f3_baseline_correct,
                "f3_today_gbm_correct": r.f3_today_gbm_correct,
                "f3_learned_3d_correct": r.f3_learned_3d_correct,
                "top_features": r.top_features,
            }
            for r in results
        ],
        "block_n_gt": block_n,
        "videos": video_names,
        "feature_names": ALL_FEATURE_NAMES,
        "court_feature_names": COURT_FEATURE_NAMES,
        "stats": stats,
        "verdict": verdict,
        "runtime_seconds": runtime_s,
        "peak_memory_mb": max_mem_mb,
    }
    JSON_PATH.write_text(json.dumps(out_json, indent=2, default=str))
    print(f"Wrote: {MD_PATH}")
    print(f"Wrote: {JSON_PATH}")
    return verdict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    tracemalloc.start()
    t0 = time.time()
    print("=== 3D court projection probe (2026-05-14) ===")
    print(f"Videos: {VIDEO_NAMES}")
    print()

    print("Step 1/4: Fetching DB data + per-video homographies...")
    rally_contacts, gt_rows = fetch_data(VIDEO_NAMES)
    homographies = fetch_homographies(VIDEO_NAMES)
    print(
        f"  Rallies with contacts: {len(rally_contacts)}; "
        f"total GT rows: {len(gt_rows)}; "
        f"videos with homography: {len(homographies)}/{len(VIDEO_NAMES)}"
    )
    by_type = Counter(g.action for g in gt_rows)
    print(f"  GT per type: {dict(by_type)}")
    print()

    print("Step 2/4: Building per-candidate feature dataset (35-dim)...")
    rows, stats = build_dataset(rally_contacts, gt_rows, homographies)
    by_type_rows = Counter(r.action_uc for r in rows)
    print(f"  Total dataset rows: {len(rows)} ({len(rows)//4} contacts x 4 candidates)")
    print(f"  Per type (rows): {dict(by_type_rows)}")
    print()

    print("Step 3/4: Per-type LOVO CV (baseline-30f vs +3D-35f)...")
    results: list[TypeResult] = []
    for atype in ACTION_TYPES_TO_TRAIN:
        print(f"\n=== Training {atype} ===")
        r = evaluate_type(atype, rows, VIDEO_NAMES)
        print(
            f"RESULT [{atype}]: N={r.n_gt} | "
            f"pipeline={r.baseline_correct}/{r.n_gt}={r.baseline_precision*100:.1f}% | "
            f"today's GBM={r.today_gbm_correct}/{r.n_predicted}={r.today_gbm_precision*100:.1f}% | "
            f"+3D={r.learned_3d_correct}/{r.n_predicted}={r.learned_3d_precision*100:.1f}% | "
            f"Δ vs today {r.delta_pp_vs_today_gbm:+.1f}pp | "
            f"F3 N={r.f3_n} ({r.f3_today_gbm_correct}->{r.f3_learned_3d_correct})"
        )
        results.append(r)

    block_rows = [r for r in rows if r.action_uc == "BLOCK"]
    block_contacts = len({(r.video, r.rally_id, r.frame) for r in block_rows})
    print(f"\nBLOCK contacts in dataset (held out): {block_contacts}")

    runtime_s = time.time() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_mb = peak / 1024 / 1024

    print(f"\nStep 4/4: Writing report (runtime={runtime_s:.1f}s, peak_mem={peak_mb:.1f}MB)...")
    write_report(results, block_contacts, VIDEO_NAMES, stats, runtime_s, peak_mb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
