"""SAFE-MODE Pose v2 probe (scaled down) — 2026-05-14.

Crash-prevention design:
  - Uses ONLY pre-cached pose data from training_data/pose_cache/. NO new YOLO
    inference is performed by this script. Zero memory pressure from pose model.
  - Caps total contacts at 40 (stratified ~7 per action type).
  - Strict serial. Single rally → single contact at a time.
  - gc.collect() after every contact.
  - Writes intermediate JSONL after every 10 contacts.
  - Hard 15-minute wall-clock budget; exits cleanly if approached.

What it measures:
  - Per-action-type precision LOVO-CV on the 40-contact subset, comparing:
      (a) Baseline pipeline pick precision
      (b) Today's 30-feature GBM (TODAY_FEATURE_NAMES)
      (c) Pose-augmented (30 + 14 pose) GBM
  - Pose feature importance ranking.
  - Saves per-contact pose feature vectors to JSONL for reuse.

Usage:
    cd analysis
    uv run python -u scripts/probe_pose_v2_safemode_2026_05_14.py

Reports:
    analysis/reports/pose_v2_probe_2026_05_14/results.md
    analysis/reports/pose_v2_probe_2026_05_14/results.json
    analysis/reports/pose_v2_probe_2026_05_14/pose_features.jsonl
    /tmp/pose_v2_intermediate.jsonl  (live progress mirror)
"""
from __future__ import annotations

import gc
import json
import math
import sys
import time
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
    FEATURE_NAMES as TODAY_FEATURE_NAMES,
    GT_FRAME_TOL,
    VIDEO_NAMES,
    ContactCtx,
    DatasetRow,
    GTRow,
    _match_gt_to_ctx,
    compute_features_for_candidate,
    fetch_data,
)

# Reuse the pose-v2 feature extractor from the cached probe
from scripts.probe_pose_v2_attribution_2026_05_14 import (  # noqa: E402
    POSE_V2_FEATURE_NAMES,
    _load_pose_lookup,
    compute_pose_v2_features,
)

POSE_FEATURE_NAMES = TODAY_FEATURE_NAMES + POSE_V2_FEATURE_NAMES
POSE_CACHE_DIR = HERE.parent / "training_data" / "pose_cache"
REPORT_DIR = HERE.parent / "reports" / "pose_v2_probe_2026_05_14"
JSON_PATH = REPORT_DIR / "results.json"
MD_PATH = REPORT_DIR / "results.md"
FEATURES_JSONL_PATH = REPORT_DIR / "pose_features.jsonl"
INTERMEDIATE_JSONL = Path("/tmp/pose_v2_intermediate.jsonl")

# Hard caps
HARD_TIMEOUT_SECONDS = 15 * 60
MAX_CONTACTS = 40
MAX_PER_VIDEO = 4
PER_TYPE_TARGET = {"SERVE": 7, "RECEIVE": 7, "SET": 7, "ATTACK": 7, "DIG": 7, "BLOCK": 5}
ACTION_TYPES = ["SERVE", "RECEIVE", "SET", "ATTACK", "DIG", "BLOCK"]


def _now() -> float:
    return time.time()


# ---------------------------------------------------------------------------
# Stratified sampler
# ---------------------------------------------------------------------------

def _select_stratified_contacts(
    rally_contacts: dict[str, list[ContactCtx]],
    gt_rows: list[GTRow],
) -> list[tuple[GTRow, ContactCtx]]:
    """Pick up to MAX_CONTACTS contacts, stratified per action type, capped MAX_PER_VIDEO/video.

    Requirements per pick:
      - Pose cache file exists for the rally
      - Pipeline contact present within ±GT_FRAME_TOL of GT frame
      - GT resolved_tid is set
      - team_assignments has 4 canonical pids
    """
    # Index GT by rally
    gt_by_rally: dict[str, list[GTRow]] = defaultdict(list)
    for g in gt_rows:
        gt_by_rally[g.rally_id].append(g)

    # Build candidate pool: (type, gt, ctx, video, rally_id)
    pool: dict[str, list[tuple[GTRow, ContactCtx, str, str]]] = defaultdict(list)
    rallies_skipped_no_pose = 0
    rallies_with_pose = 0
    for rally_id, ctxs in rally_contacts.items():
        cache_path = POSE_CACHE_DIR / f"{rally_id}.npz"
        if not cache_path.exists():
            rallies_skipped_no_pose += 1
            continue
        rallies_with_pose += 1
        gts = gt_by_rally.get(rally_id, [])
        if not gts:
            continue
        pairs = _match_gt_to_ctx(ctxs, gts)
        for g, c in pairs:
            if c is None or g.resolved_tid is None:
                continue
            if len(c.team_assignments) != 4:
                continue
            type_uc = g.action.upper()
            if type_uc not in PER_TYPE_TARGET:
                continue
            pool[type_uc].append((g, c, c.video, rally_id))

    print(f"  Pool sizes: { {t: len(v) for t, v in pool.items()} }")
    print(f"  Rallies w/ pose cache: {rallies_with_pose} | skipped (no pose): {rallies_skipped_no_pose}")

    # Greedy pick: respect per-type target + per-video cap, prefer video diversity
    rng = np.random.default_rng(20260514)
    selected: list[tuple[GTRow, ContactCtx]] = []
    per_video_count: Counter[str] = Counter()
    for type_uc in ACTION_TYPES:
        candidates = pool.get(type_uc, [])
        rng.shuffle(candidates)
        target = PER_TYPE_TARGET[type_uc]
        added = 0
        # First pass: respect per-video cap
        for g, c, vid, _rid in candidates:
            if added >= target:
                break
            if len(selected) >= MAX_CONTACTS:
                break
            if per_video_count[vid] >= MAX_PER_VIDEO:
                continue
            selected.append((g, c))
            per_video_count[vid] += 1
            added += 1
        # Second pass (relaxed): if we couldn't fill the type-target, allow exceeding per-video cap
        # by 1 (still very small). Only matters for tiny pools.
        if added < target:
            for g, c, vid, _rid in candidates:
                if added >= target:
                    break
                if len(selected) >= MAX_CONTACTS:
                    break
                if (g, c) in selected:
                    continue
                if per_video_count[vid] >= MAX_PER_VIDEO + 1:
                    continue
                # avoid duplicates by checking identity
                if any(g is sg and c is sc for sg, sc in selected):
                    continue
                selected.append((g, c))
                per_video_count[vid] += 1
                added += 1

    return selected


# ---------------------------------------------------------------------------
# Feature build per contact (serial, with cleanup)
# ---------------------------------------------------------------------------

def _build_dataset_for_selected(
    selected: list[tuple[GTRow, ContactCtx]],
    deadline: float,
) -> tuple[list[DatasetRow], dict[str, Any]]:
    rows: list[DatasetRow] = []
    pose_cache: dict[str, Any] = {}
    n_done = 0
    n_pose_present = 0
    n_pose_empty = 0  # cache loaded but contact frame has no detections
    intermediate: list[dict[str, Any]] = []
    INTERMEDIATE_JSONL.write_text("")  # truncate
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    for idx, (g, c) in enumerate(selected):
        if _now() > deadline:
            print(f"[abort] hit hard timeout, stopping at contact {idx}/{len(selected)}")
            break

        # Single-rally pose load (cached across contacts on same rally)
        if c.rally_id not in pose_cache:
            pose_cache[c.rally_id] = _load_pose_lookup(c.rally_id)
        pose = pose_cache[c.rally_id]
        if pose is not None:
            n_pose_present += 1

        cand_pool = sorted(c.team_assignments.keys())  # 4 canonical pids

        contact_rows: list[DatasetRow] = []
        per_contact_pose_features: dict[int, list[float]] = {}
        for cand_tid in cand_pool:
            base = compute_features_for_candidate(c, cand_tid)
            pose_v2 = compute_pose_v2_features(c, cand_tid, pose)
            per_contact_pose_features[cand_tid] = pose_v2
            row = DatasetRow(
                video=c.video, rally_id=c.rally_id, frame=c.frame,
                action_uc=g.action.upper(), cand_tid=cand_tid,
                is_gt=int(cand_tid == g.resolved_tid),
                is_pipeline_pick=int(c.pl_tid is not None and cand_tid == c.pl_tid),
                features=base + pose_v2,
                gt_tid=g.resolved_tid if g.resolved_tid is not None else -1,
                pl_tid=c.pl_tid,
            )
            contact_rows.append(row)

        # Track if at least one candidate had non-NaN pose data
        any_pose = any(
            any(not math.isnan(v) for v in feats) for feats in per_contact_pose_features.values()
        )
        if not any_pose:
            n_pose_empty += 1

        rows.extend(contact_rows)

        intermediate.append({
            "idx": idx,
            "video": c.video,
            "rally_id": c.rally_id,
            "frame": c.frame,
            "action": g.action.upper(),
            "gt_tid": g.resolved_tid,
            "pl_tid": c.pl_tid,
            "cand_pose_features": {
                int(tid): per_contact_pose_features[tid] for tid in cand_pool
            },
            "any_pose_present": any_pose,
        })

        n_done += 1
        print(
            f"[{n_done}/{len(selected)}] {g.action.upper():7s} {c.video:6s} "
            f"rally={c.rally_id[:8]} f={c.frame:>4d} gt={g.resolved_tid} pl={c.pl_tid} pose={any_pose}",
            flush=True,
        )

        # Hard cap on contacts
        if n_done >= MAX_CONTACTS:
            print(f"[cap] hit MAX_CONTACTS={MAX_CONTACTS}, stopping")
            break

        # Cleanup every contact
        del contact_rows
        del per_contact_pose_features
        gc.collect()

        # Intermediate flush every 10 contacts
        if n_done % 10 == 0:
            _flush_intermediate(intermediate)
            print(f"  [intermediate] flushed {len(intermediate)} entries to {INTERMEDIATE_JSONL}", flush=True)

    # Final intermediate flush
    _flush_intermediate(intermediate)

    # Write definitive pose-features JSONL
    with FEATURES_JSONL_PATH.open("w") as fh:
        for entry in intermediate:
            fh.write(json.dumps(entry, default=str) + "\n")

    stats = {
        "n_contacts_done": n_done,
        "n_pose_present_rallies": n_pose_present,
        "n_pose_empty_contacts": n_pose_empty,
    }
    return rows, stats


def _flush_intermediate(entries: list[dict[str, Any]]) -> None:
    with INTERMEDIATE_JSONL.open("w") as fh:
        for e in entries:
            fh.write(json.dumps(e, default=str) + "\n")


# ---------------------------------------------------------------------------
# LOVO training (per type) — copied / simplified
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


def _train_loo(
    type_rows: list[DatasetRow],
    video_names: list[str],
    n_features: int,
    label: str,
) -> tuple[int, int, np.ndarray, int]:
    correct = 0
    n_predicted = 0
    importances_sum = np.zeros(n_features, dtype=np.float64)
    n_models = 0
    for hold_out in video_names:
        train_rows = [r for r in type_rows if r.video != hold_out]
        test_rows = [r for r in type_rows if r.video == hold_out]
        if not train_rows or not test_rows:
            continue
        X_train = np.array([r.features[:n_features] for r in train_rows], dtype=np.float32)
        y_train = np.array([r.is_gt for r in train_rows], dtype=np.int32)
        if y_train.sum() == 0 or y_train.sum() == len(y_train):
            continue
        model = HistGradientBoostingClassifier(
            max_iter=200, max_depth=4, learning_rate=0.05,
            min_samples_leaf=5, l2_regularization=1.0, random_state=42,
        )
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"  [{label}] hold={hold_out} fit failed: {e}", flush=True)
            continue
        try:
            sub_size = min(400, len(X_train))
            sub = np.random.default_rng(42).choice(len(X_train), size=sub_size, replace=False)
            pi = permutation_importance(
                model, X_train[sub], y_train[sub], n_repeats=3, random_state=42, n_jobs=1,
            )
            importances_sum += pi.importances_mean
            n_models += 1
        except Exception as e:
            print(f"  [{label}] hold={hold_out} perm-importance failed: {e}", flush=True)

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
        del model, X_train, y_train
        gc.collect()
    return correct, n_predicted, importances_sum, n_models


def evaluate_type(type_name: str, all_rows: list[DatasetRow], video_names: list[str]) -> TypeResult:
    type_rows = [r for r in all_rows if r.action_uc == type_name]
    contact_groups: dict[tuple[str, str, int], list[DatasetRow]] = defaultdict(list)
    for r in type_rows:
        contact_groups[(r.video, r.rally_id, r.frame)].append(r)
    n_gt = len(contact_groups)

    baseline_correct = 0
    for _key, rs in contact_groups.items():
        pl_picks = [r for r in rs if r.is_pipeline_pick]
        if pl_picks and any(r.is_gt == 1 for r in pl_picks):
            baseline_correct += 1
    baseline_precision = baseline_correct / n_gt if n_gt > 0 else 0.0

    n_today = len(TODAY_FEATURE_NAMES)
    n_all = len(POSE_FEATURE_NAMES)

    print(f"--- {type_name} today's GBM ({n_today} features) ---", flush=True)
    today_correct, today_predicted, _imp_today, _nm_today = _train_loo(
        type_rows, video_names, n_today, f"{type_name}-today",
    )

    print(f"--- {type_name} pose-v2 GBM ({n_all} features) ---", flush=True)
    pose_v2_correct, pose_v2_predicted, imp_pose_v2, nm_pose_v2 = _train_loo(
        type_rows, video_names, n_all, f"{type_name}-pose2",
    )

    today_precision = today_correct / today_predicted if today_predicted > 0 else 0.0
    pose_v2_precision = pose_v2_correct / pose_v2_predicted if pose_v2_predicted > 0 else 0.0
    delta_vs_today_pp = (pose_v2_precision - today_precision) * 100.0
    delta_vs_baseline_pp = (pose_v2_precision - baseline_precision) * 100.0

    imp_avg = imp_pose_v2 / nm_pose_v2 if nm_pose_v2 > 0 else imp_pose_v2
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

def write_report(
    results: list[TypeResult],
    setup: dict[str, Any],
    build_stats: dict[str, Any],
) -> str:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    n_gt_tot = sum(r.n_gt for r in results)
    base_tot = sum(r.baseline_correct for r in results)
    today_tot = sum(r.today_correct for r in results)
    pose_v2_tot = sum(r.pose_v2_correct for r in results)
    n_pred_tot = sum(r.n_predicted for r in results)
    base_agg = base_tot / n_gt_tot if n_gt_tot else 0.0
    today_agg = today_tot / n_pred_tot if n_pred_tot else 0.0
    pose_v2_agg = pose_v2_tot / n_pred_tot if n_pred_tot else 0.0

    # Decision: small-sample lift signal
    types_lift_3pp = [r for r in results if r.delta_vs_today_pp >= 3.0 and r.n_gt >= 5]
    types_regress_3pp = [r for r in results if r.delta_vs_today_pp <= -3.0 and r.n_gt >= 5]
    pose_in_top5 = []
    for r in results:
        for name, _imp in r.top_features:
            if name.startswith("pv2_"):
                pose_in_top5.append(r.type_name)
                break

    if len(types_lift_3pp) >= 2 and len(types_regress_3pp) == 0 and len(pose_in_top5) >= 1:
        verdict = "LIFTS — proceed to scaled extraction over full 641 corpus"
    elif len(types_lift_3pp) >= 1 and len(types_regress_3pp) == 0:
        verdict = "MIXED — single-type lift, partial signal; consider larger pose corpus before commitment"
    elif len(types_regress_3pp) > 0 and not types_lift_3pp:
        verdict = "FLAT — pose adds noise, no lift"
    else:
        verdict = "FLAT — no measurable lift over today's GBM at this scale"

    out = {
        "setup": setup,
        "build_stats": build_stats,
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
        "feature_names": POSE_FEATURE_NAMES,
    }
    JSON_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote JSON: {JSON_PATH}", flush=True)

    lines: list[str] = []
    lines.append("# Pose v2 probe (scaled-down) — 2026-05-14")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- {build_stats['n_contacts_done']} stratified GT contacts × ~4 canonical-pid candidates each")
    lines.append(f"- Pose model: cached `yolo11s-pose.pt` keypoints (NO re-inference)")
    lines.append(f"- Total inferences: 0 (uses pre-existing pose cache)")
    lines.append(f"- Runtime: {setup['runtime_seconds']:.1f}s ({setup['runtime_seconds']/60:.1f} min)")
    lines.append(f"- Hard caps respected: MAX_CONTACTS={MAX_CONTACTS}, MAX_PER_VIDEO={MAX_PER_VIDEO}, HARD_TIMEOUT={HARD_TIMEOUT_SECONDS}s")
    lines.append(f"- Strict serial execution, gc.collect() every contact, intermediate dump every 10")
    lines.append(f"- Pose data coverage: {build_stats['n_pose_present_rallies']} rallies with pose, {build_stats['n_pose_empty_contacts']} contacts had no pose detections at contact frame")
    lines.append("")
    lines.append("## Per-type precision (on 40-contact subset)")
    lines.append("")
    lines.append("| Type | N | Baseline pipeline | Today's GBM | + Pose v2 | Δ vs today | Δ vs baseline |")
    lines.append("|---|---|---|---|---|---|---|")
    for r in results:
        if r.n_predicted == 0:
            lines.append(
                f"| {r.type_name} | {r.n_gt} | {r.baseline_correct}/{r.n_gt} = "
                f"{r.baseline_precision*100:.1f}% | n/a (insufficient LOO splits) | n/a | n/a | n/a |"
            )
            continue
        lines.append(
            f"| {r.type_name} | {r.n_gt} | "
            f"{r.baseline_correct}/{r.n_gt} = {r.baseline_precision*100:.1f}% | "
            f"{r.today_correct}/{r.n_predicted} = {r.today_precision*100:.1f}% | "
            f"{r.pose_v2_correct}/{r.n_predicted} = {r.pose_v2_precision*100:.1f}% | "
            f"{r.delta_vs_today_pp:+.1f}pp | "
            f"{r.delta_vs_baseline_pp:+.1f}pp |"
        )
    lines.append("")
    lines.append(
        f"**Aggregate**: baseline {base_tot}/{n_gt_tot} = {base_agg*100:.1f}% → "
        f"today {today_tot}/{n_pred_tot} = {today_agg*100:.1f}% → "
        f"pose-v2 {pose_v2_tot}/{n_pred_tot} = {pose_v2_agg*100:.1f}%"
    )
    lines.append("")
    lines.append("## Per-type top-5 features (by permutation importance)")
    lines.append("")
    for r in results:
        if not r.top_features:
            continue
        lines.append(f"### {r.type_name}")
        has_pose = any(name.startswith("pv2_") for name, _ in r.top_features)
        for name, imp in r.top_features:
            marker = " **[pose]**" if name.startswith("pv2_") else ""
            lines.append(f"- `{name}`: {imp:.4f}{marker}")
        lines.append(f"_pose-feature-in-top5: {has_pose}_")
        lines.append("")
    lines.append("## Decision criteria (small-sample friendly)")
    lines.append("")
    lines.append("Scaled-down criteria (40-contact subset is smaller than the 641-row experiment so")
    lines.append("thresholds are looser than the canonical 5pp/2pp gates):")
    lines.append("- ≥3pp lift on ≥2 action types (where n≥5)")
    lines.append("- No type regresses by ≥3pp (where n≥5)")
    lines.append("- ≥1 type has a pose feature in top-5")
    lines.append("")
    lines.append(f"- Types lifting ≥3pp (n≥5): {len(types_lift_3pp)} ({', '.join(r.type_name for r in types_lift_3pp) or 'none'})")
    lines.append(f"- Types regressing ≥3pp (n≥5): {len(types_regress_3pp)} ({', '.join(r.type_name for r in types_regress_3pp) or 'none'})")
    lines.append(f"- Types with pose feature in top-5: {len(pose_in_top5)} ({', '.join(pose_in_top5) or 'none'})")
    lines.append("")
    lines.append(f"## Verdict: **{verdict}**")
    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"- Pose feature vectors per contact: `{FEATURES_JSONL_PATH}`")
    lines.append(f"- Intermediate progress mirror: `{INTERMEDIATE_JSONL}`")
    lines.append(f"- Machine-readable results: `{JSON_PATH}`")

    MD_PATH.write_text("\n".join(lines))
    print(f"Wrote markdown: {MD_PATH}", flush=True)
    return verdict


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    t_start = _now()
    deadline = t_start + HARD_TIMEOUT_SECONDS
    print(f"=== Pose v2 SAFE-MODE probe (2026-05-14) ===", flush=True)
    print(f"  HARD_TIMEOUT={HARD_TIMEOUT_SECONDS}s, MAX_CONTACTS={MAX_CONTACTS}, MAX_PER_VIDEO={MAX_PER_VIDEO}", flush=True)
    print(f"  Strict serial, NO new YOLO inference (cache only), gc.collect() per contact", flush=True)
    print(f"  Videos: {VIDEO_NAMES}", flush=True)
    print(flush=True)

    print("Step 1/4: Fetching DB data (same loader as today's experiment)...", flush=True)
    rally_contacts, gt_rows = fetch_data(VIDEO_NAMES)
    print(f"  Rallies with contacts: {len(rally_contacts)}, GT rows: {len(gt_rows)}", flush=True)
    print(f"  GT per action: {dict(Counter(g.action.upper() for g in gt_rows))}", flush=True)
    print(flush=True)

    print("Step 2/4: Stratified selection of contacts...", flush=True)
    selected = _select_stratified_contacts(rally_contacts, gt_rows)
    print(f"  Selected {len(selected)} contacts (target {MAX_CONTACTS}):", flush=True)
    by_type = Counter(g.action.upper() for g, _ in selected)
    print(f"  By type: {dict(by_type)}", flush=True)
    print(flush=True)

    if not selected:
        print("[fatal] no contacts selected — pose cache may be empty for the panel", flush=True)
        return 1

    print("Step 3/4: Per-contact feature build (serial, cache only)...", flush=True)
    rows, build_stats = _build_dataset_for_selected(selected, deadline)
    print(f"  Built {len(rows)} candidate rows from {build_stats['n_contacts_done']} contacts", flush=True)
    print(flush=True)

    print("Step 4/4: Per-type LOVO CV...", flush=True)
    results: list[TypeResult] = []
    for atype in ACTION_TYPES:
        if _now() > deadline:
            print(f"  [abort] hit hard timeout in eval phase at {atype}, stopping early", flush=True)
            break
        type_rows = [r for r in rows if r.action_uc == atype]
        n_contacts = len({(r.video, r.rally_id, r.frame) for r in type_rows})
        if n_contacts == 0:
            print(f"=== {atype}: no rows, skipping ===", flush=True)
            continue
        print(f"=== {atype} (n={n_contacts}) ===", flush=True)
        r = evaluate_type(atype, rows, VIDEO_NAMES)
        print(
            f"  RESULT [{atype}]: N={r.n_gt} | "
            f"baseline {r.baseline_correct}/{r.n_gt}={r.baseline_precision*100:.1f}% | "
            f"today {r.today_correct}/{r.n_predicted}={r.today_precision*100:.1f}% | "
            f"pose-v2 {r.pose_v2_correct}/{r.n_predicted}={r.pose_v2_precision*100:.1f}% | "
            f"Δ-today {r.delta_vs_today_pp:+.1f}pp",
            flush=True,
        )
        results.append(r)
        gc.collect()

    runtime = _now() - t_start
    setup = {
        "videos": VIDEO_NAMES,
        "max_contacts": MAX_CONTACTS,
        "max_per_video": MAX_PER_VIDEO,
        "per_type_target": PER_TYPE_TARGET,
        "hard_timeout_seconds": HARD_TIMEOUT_SECONDS,
        "runtime_seconds": runtime,
        "feature_count_today": len(TODAY_FEATURE_NAMES),
        "feature_count_pose_v2": len(POSE_V2_FEATURE_NAMES),
        "feature_count_total": len(POSE_FEATURE_NAMES),
    }
    verdict = write_report(results, setup, build_stats)
    print(flush=True)
    print(f"=== DONE in {runtime:.1f}s ({runtime/60:.1f} min) ===", flush=True)
    print(f"Verdict: {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
