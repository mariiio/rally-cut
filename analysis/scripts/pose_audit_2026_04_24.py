"""Pose-feature discrimination audit on classifier-rejected FNs vs TPs.

Diagnostic for the question: do the 5 GBM pose features separate true-positive
contacts from the 128 non-block, classifier-rejected false-negative contacts?

Pass 1 — compute pose features at every GT contact frame using
`extract_contact_pose_features_for_nearest`. The pose nearest-track logic
mirrors the inference path so values match what GBM saw at decision time.

Pass 2 — label each row TP/FN via `match_contacts` against `contacts_json`
(7-frame tolerance, matching `train_contact_classifier.py`). Cross-reference
FNs against `corpus_annotated.jsonl` to keep only `rejected_by_classifier`
non-block cases (the 128 bucket). Sample TPs per class for the AUC compare.

Outputs: per-row JSONL + per-class TP/FN summary tables (mean/p25/p75) and
per-feature ROC-AUC. Exits to stdout in compact form for the early-exit gate.

Usage:
    cd analysis
    uv run python scripts/pose_audit_2026_04_24.py
    uv run python scripts/pose_audit_2026_04_24.py --tp-cap 200
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rallycut.tracking.ball_tracker import BallPosition  # noqa: E402
from rallycut.tracking.pose_attribution.features import (  # noqa: E402
    extract_contact_pose_features_for_nearest,
)
from scripts.eval_action_detection import (  # noqa: E402
    _build_player_positions,
    load_rallies_with_action_gt,
    match_contacts,
)

POSE_FEATURE_NAMES = (
    "wrist_velocity_max",
    "hand_ball_dist_min",
    "arm_extension_change",
    "pose_confidence_mean",
    "both_arms_raised",
)

NON_BLOCK_CLASSES = ("serve", "receive", "set", "attack", "dig")
DEFAULT_TOLERANCE = 7  # matches train_contact_classifier.py default
DEFAULT_CORPUS = (
    REPO_ROOT / "outputs" / "action_errors" / "corpus_annotated.jsonl"
)
DEFAULT_OUT_DIR = REPO_ROOT / "reports" / "pose_audit"


def load_corpus_fn_index(corpus_path: Path) -> dict[tuple[str, int], dict]:
    """Return {(rally_id, gt_frame): record} for rejected_by_classifier rows."""
    idx: dict[tuple[str, int], dict] = {}
    with corpus_path.open() as f:
        for line in f:
            d = json.loads(line)
            if d.get("error_class") != "FN_contact":
                continue
            if d.get("fn_subcategory") != "rejected_by_classifier":
                continue
            idx[(d["rally_id"], int(d["gt_frame"]))] = d
    return idx


def find_nearest_track_at_frame(
    player_positions: list, ball_x: float, ball_y: float, frame: int,
    search_window: int = 0,
) -> int:
    """Return the track_id closest to (ball_x, ball_y) at the given frame.

    Mirrors the candidate inference code (contact_detector._find_nearest_player)
    closely enough for pose attribution: nearest-by-Euclidean among tracks
    that have a position at the frame (or within ±search_window).
    """
    best_tid = -1
    best_dist = float("inf")
    for pp in player_positions:
        if abs(pp.frame_number - frame) > search_window:
            continue
        if pp.track_id < 0:
            continue
        dx = pp.x - ball_x
        dy = pp.y - ball_y
        d = math.sqrt(dx * dx + dy * dy)
        if d < best_dist:
            best_dist = d
            best_tid = pp.track_id
    return best_tid


def get_ball_at_frame(
    ball_positions: list[BallPosition], frame: int
) -> tuple[float, float] | None:
    """Return (x, y) at frame, else nearest within ±5 frames, else None."""
    by_frame = {bp.frame_number: bp for bp in ball_positions}
    if frame in by_frame:
        return (by_frame[frame].x, by_frame[frame].y)
    for offset in [-1, 1, -2, 2, -3, 3, -4, 4, -5, 5]:
        f = frame + offset
        if f in by_frame:
            return (by_frame[f].x, by_frame[f].y)
    return None


def auc_roc(positive_scores: list[float], negative_scores: list[float]) -> float:
    """Compute ROC-AUC where higher score = positive class.

    AUC = P(score(+) > score(-)) using the rank-sum (Mann-Whitney) identity.
    Returns 0.5 when either class is empty.
    """
    if not positive_scores or not negative_scores:
        return 0.5
    pos = np.asarray(positive_scores, dtype=np.float64)
    neg = np.asarray(negative_scores, dtype=np.float64)
    combined = np.concatenate([pos, neg])
    # Average rank for ties
    order = np.argsort(combined, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(combined) + 1)
    # Tie-break: average ranks of equal values
    _, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    sums = np.zeros_like(counts, dtype=np.float64)
    np.add.at(sums, inverse, ranks)
    avg_ranks = sums / counts
    ranks = avg_ranks[inverse]
    pos_ranks = ranks[: len(pos)]
    rank_sum = pos_ranks.sum()
    n_pos = len(pos)
    n_neg = len(neg)
    auc = (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    # Higher feature value should be the discriminator; if AUC < 0.5 the
    # feature discriminates in the OPPOSITE direction. Report magnitude
    # via max(auc, 1-auc) since either direction is usable.
    return float(max(auc, 1.0 - auc))


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    return float(np.percentile(np.asarray(values), p))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerance", type=int, default=DEFAULT_TOLERANCE)
    parser.add_argument("--tp-cap", type=int, default=300,
                        help="Max TPs sampled per class (random.seed=42)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--limit-rallies", type=int, default=0,
                        help="Smoke test on N rallies (0 = all)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.out_dir / "pose_audit_rows.jsonl"
    summary_path = args.out_dir / "pose_audit_summary.md"

    # --- Load FN corpus index ---
    print(f"[pose-audit] Loading corpus: {args.corpus}", flush=True)
    fn_index = load_corpus_fn_index(args.corpus)
    print(f"[pose-audit] FN corpus rows (rejected_by_classifier): {len(fn_index)}",
          flush=True)

    # --- Load rallies ---
    print("[pose-audit] Loading rallies with action GT...", flush=True)
    t0 = time.time()
    rallies = load_rallies_with_action_gt()
    print(f"[pose-audit] Loaded {len(rallies)} rallies in {time.time() - t0:.1f}s",
          flush=True)

    if args.limit_rallies:
        rallies = rallies[: args.limit_rallies]

    rows: list[dict] = []
    pose_blind_count = 0
    pose_have_kpts = 0

    for rally_idx, rally in enumerate(rallies, 1):
        if not rally.gt_labels or not rally.positions_json or not rally.ball_positions_json:
            continue

        player_positions = _build_player_positions(
            rally.positions_json, rally_id=rally.rally_id, inject_pose=True,
        )
        ball_positions = [
            BallPosition(
                frame_number=bp["frameNumber"],
                x=bp["x"],
                y=bp["y"],
                confidence=bp.get("confidence", 1.0),
            )
            for bp in rally.ball_positions_json
        ]
        ball_by_frame = {bp.frame_number: bp for bp in ball_positions}

        # Get predicted contacts for TP/FN labeling (matched within tolerance)
        contacts_data = rally.contacts_json or {}
        pred_actions = []
        for c in contacts_data.get("contacts", []) if isinstance(contacts_data, dict) else []:
            f = c.get("frame", c.get("frameNumber"))
            if f is None:
                continue
            pred_actions.append({"frame": int(f),
                                 "action": c.get("action") or c.get("actionType") or ""})

        matches, _unmatched = match_contacts(
            rally.gt_labels, pred_actions, tolerance=args.tolerance,
        )
        # Map gt_frame -> pred_frame (None if FN)
        match_by_gt_frame: dict[int, int | None] = {}
        for m in matches:
            match_by_gt_frame[m.gt_frame] = m.pred_frame

        for gt in rally.gt_labels:
            if gt.action not in NON_BLOCK_CLASSES and gt.action != "block":
                continue

            ball_xy = get_ball_at_frame(ball_positions, gt.frame)
            if ball_xy is None:
                # Skip — can't extract pose features without ball anchor.
                continue
            bx, by = ball_xy

            # Find nearest track at the GT frame (search ±2 frames for sparse tracking)
            tid = find_nearest_track_at_frame(
                player_positions, bx, by, gt.frame, search_window=2,
            )
            if tid < 0:
                continue

            pose_vals = extract_contact_pose_features_for_nearest(
                contact_frame=gt.frame,
                nearest_track_id=tid,
                player_positions=player_positions,
                ball_at_contact=(bx, by),
                ball_by_frame=ball_by_frame,
            )
            # had_keypoints: pose_confidence_mean > 0 means at least one frame
            # in ±5 had keypoints. The fallback fills 0.0 across the board.
            had_kpts = bool(pose_vals[3] > 0.0)
            if had_kpts:
                pose_have_kpts += 1
            else:
                pose_blind_count += 1

            # Determine TP/FN status against contacts_json
            pred_frame = match_by_gt_frame.get(gt.frame)
            is_tp = pred_frame is not None

            # Cross-ref against corpus FN index for the 128 bucket
            in_corpus_fn = (rally.rally_id, gt.frame) in fn_index
            corpus_record = fn_index.get((rally.rally_id, gt.frame))

            row = {
                "rally_id": rally.rally_id,
                "video_id": rally.video_id,
                "gt_frame": gt.frame,
                "gt_action": gt.action,
                "track_id": tid,
                "ball_x": bx,
                "ball_y": by,
                "wrist_velocity_max": float(pose_vals[0]),
                "hand_ball_dist_min": float(pose_vals[1]),
                "arm_extension_change": float(pose_vals[2]),
                "pose_confidence_mean": float(pose_vals[3]),
                "both_arms_raised": float(pose_vals[4]),
                "had_keypoints": had_kpts,
                "is_tp": is_tp,
                "pred_frame": pred_frame,
                "in_corpus_fn": in_corpus_fn,
                "corpus_gbm_conf": corpus_record["classifier_conf"] if corpus_record else None,
                "corpus_seq_peak": corpus_record.get("seq_peak_nonbg_within_5f") if corpus_record else None,
            }
            rows.append(row)

        if rally_idx % 50 == 0 or rally_idx == len(rallies):
            print(f"[pose-audit] [{rally_idx}/{len(rallies)}] rallies processed, "
                  f"{len(rows)} GT contacts, {pose_have_kpts} with keypoints, "
                  f"{pose_blind_count} pose-blind", flush=True)

    # --- Persist rows ---
    with rows_path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"[pose-audit] Wrote {len(rows)} rows to {rows_path}", flush=True)

    # --- Pass 1 sanity: pose-blind on the 128 bucket ---
    fn_rows_in_bucket = [r for r in rows if r["in_corpus_fn"]
                         and r["gt_action"] != "block"]
    fn_block_rows = [r for r in rows if r["in_corpus_fn"]
                     and r["gt_action"] == "block"]
    fn_with_kpts = sum(1 for r in fn_rows_in_bucket if r["had_keypoints"])
    fn_blind = sum(1 for r in fn_rows_in_bucket if not r["had_keypoints"])
    print(f"[pose-audit] FN corpus 128-bucket recovered: {len(fn_rows_in_bucket)} "
          f"(expected ~128); pose-coverage {fn_with_kpts}/{len(fn_rows_in_bucket)} "
          f"({fn_blind} pose-blind)", flush=True)
    print(f"[pose-audit] FN corpus block-bucket recovered: {len(fn_block_rows)}",
          flush=True)

    # Per-class FN bucket counts
    fn_by_class: dict[str, int] = defaultdict(int)
    fn_by_class_kpts: dict[str, int] = defaultdict(int)
    for r in fn_rows_in_bucket:
        fn_by_class[r["gt_action"]] += 1
        if r["had_keypoints"]:
            fn_by_class_kpts[r["gt_action"]] += 1
    print("[pose-audit] FN by class (with-kpts/total):", flush=True)
    for cls in NON_BLOCK_CLASSES:
        print(f"    {cls}: {fn_by_class_kpts[cls]}/{fn_by_class[cls]}",
              flush=True)

    # --- Build TP pool per class (with keypoints) ---
    rng = np.random.default_rng(42)
    tp_pool: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if not r["is_tp"]:
            continue
        if not r["had_keypoints"]:
            continue
        if r["gt_action"] not in NON_BLOCK_CLASSES:
            continue
        tp_pool[r["gt_action"]].append(r)

    tp_sample: dict[str, list[dict]] = {}
    for cls, lst in tp_pool.items():
        if len(lst) > args.tp_cap:
            idx = rng.choice(len(lst), args.tp_cap, replace=False)
            tp_sample[cls] = [lst[i] for i in idx]
        else:
            tp_sample[cls] = list(lst)
        print(f"[pose-audit] TP pool {cls}: {len(tp_sample[cls])} "
              f"(of {len(lst)} available)", flush=True)

    # --- Per-class summary + AUC ---
    summary_lines = ["# Pose-feature discrimination audit — TP vs FN (rejected-by-classifier)"]
    summary_lines.append("")
    summary_lines.append(f"Tolerance for TP/FN matching: ±{args.tolerance} frames")
    summary_lines.append(f"Total rows emitted: {len(rows)}")
    summary_lines.append(f"FN bucket (non-block, rejected_by_classifier) "
                         f"recovered: {len(fn_rows_in_bucket)}")
    summary_lines.append(f"FN with keypoints: {fn_with_kpts} / "
                         f"{len(fn_rows_in_bucket)} "
                         f"(target ~90/128 per pre-validated cov)")
    summary_lines.append("")

    summary_lines.append("## Per-class FN coverage (in 128 bucket)")
    summary_lines.append("")
    summary_lines.append("| Class | FN total | FN with kpts | TP sampled |")
    summary_lines.append("|---|---:|---:|---:|")
    for cls in NON_BLOCK_CLASSES:
        summary_lines.append(
            f"| {cls} | {fn_by_class[cls]} | {fn_by_class_kpts[cls]} | "
            f"{len(tp_sample.get(cls, []))} |"
        )
    summary_lines.append("")

    # AUC per class per feature (TP=positive, FN=negative; AUC reports magnitude)
    auc_table_rows: list[tuple[str, dict[str, float]]] = []
    summary_lines.append("## Per-feature ROC-AUC (TP vs FN, both with keypoints)")
    summary_lines.append("")
    header_cells = ["Class", "n_TP", "n_FN"] + list(POSE_FEATURE_NAMES) + ["best_AUC"]
    summary_lines.append("| " + " | ".join(header_cells) + " |")
    summary_lines.append("|" + "|".join(["---:"] * len(header_cells)) + "|")
    overall_best = 0.0
    overall_best_cls_feature: tuple[str, str] | None = None
    for cls in NON_BLOCK_CLASSES:
        fn_rows = [r for r in fn_rows_in_bucket
                   if r["gt_action"] == cls and r["had_keypoints"]]
        tp_rows = tp_sample.get(cls, [])
        cell_parts = [cls, str(len(tp_rows)), str(len(fn_rows))]
        per_feat: dict[str, float] = {}
        for feat in POSE_FEATURE_NAMES:
            tp_vals = [r[feat] for r in tp_rows]
            fn_vals = [r[feat] for r in fn_rows]
            auc = auc_roc(tp_vals, fn_vals)
            per_feat[feat] = auc
            cell_parts.append(f"{auc:.3f}")
            if auc > overall_best:
                overall_best = auc
                overall_best_cls_feature = (cls, feat)
        best_in_cls = max(per_feat.values()) if per_feat else 0.5
        cell_parts.append(f"{best_in_cls:.3f}")
        summary_lines.append("| " + " | ".join(cell_parts) + " |")
        auc_table_rows.append((cls, per_feat))

    summary_lines.append("")
    if overall_best_cls_feature is not None:
        summary_lines.append(
            f"**Overall best feature/class:** "
            f"{overall_best_cls_feature[1]} on {overall_best_cls_feature[0]} "
            f"(AUC = {overall_best:.3f})"
        )
    summary_lines.append("")

    # Per-class TP vs FN distributions (median + p25/p75)
    summary_lines.append("## TP vs FN distributions (median [p25, p75])")
    summary_lines.append("")
    for cls in NON_BLOCK_CLASSES:
        fn_rows = [r for r in fn_rows_in_bucket
                   if r["gt_action"] == cls and r["had_keypoints"]]
        tp_rows = tp_sample.get(cls, [])
        if not fn_rows or not tp_rows:
            continue
        summary_lines.append(f"### {cls}  (TP n={len(tp_rows)}, FN n={len(fn_rows)})")
        summary_lines.append("")
        summary_lines.append("| Feature | TP median [p25,p75] | FN median [p25,p75] | TP/FN ratio |")
        summary_lines.append("|---|---|---|---:|")
        for feat in POSE_FEATURE_NAMES:
            tp_vals = [r[feat] for r in tp_rows]
            fn_vals = [r[feat] for r in fn_rows]
            tp_m = statistics.median(tp_vals)
            fn_m = statistics.median(fn_vals)
            tp_p25 = percentile(tp_vals, 25)
            tp_p75 = percentile(tp_vals, 75)
            fn_p25 = percentile(fn_vals, 25)
            fn_p75 = percentile(fn_vals, 75)
            ratio = (tp_m / fn_m) if fn_m > 1e-9 else float("nan")
            summary_lines.append(
                f"| {feat} | {tp_m:.4f} [{tp_p25:.4f}, {tp_p75:.4f}] | "
                f"{fn_m:.4f} [{fn_p25:.4f}, {fn_p75:.4f}] | {ratio:.2f} |"
            )
        summary_lines.append("")

    summary = "\n".join(summary_lines)
    summary_path.write_text(summary)
    print(f"[pose-audit] Wrote summary to {summary_path}", flush=True)

    # --- Compact stdout for early-exit gate ---
    print()
    print("=" * 70)
    print("EARLY-EXIT GATE INPUT")
    print("=" * 70)
    print(f"Best AUC across all (class, feature) pairs: {overall_best:.3f}")
    if overall_best_cls_feature:
        print(f"  → {overall_best_cls_feature[1]} on {overall_best_cls_feature[0]}")
    print()
    print(f"{'Class':<10} {'n_TP':>5} {'n_FN':>5}  " + "  ".join(
        f"{n[:14]:>14}" for n in POSE_FEATURE_NAMES))
    for cls in NON_BLOCK_CLASSES:
        fn_rows = [r for r in fn_rows_in_bucket
                   if r["gt_action"] == cls and r["had_keypoints"]]
        tp_rows = tp_sample.get(cls, [])
        cells = [f"{cls:<10}", f"{len(tp_rows):>5}", f"{len(fn_rows):>5}", " "]
        for feat in POSE_FEATURE_NAMES:
            tp_vals = [r[feat] for r in tp_rows]
            fn_vals = [r[feat] for r in fn_rows]
            auc = auc_roc(tp_vals, fn_vals)
            cells.append(f"{auc:>14.3f}")
        print("  ".join(cells))
    print()
    if overall_best <= 0.55:
        print("[GATE] best AUC ≤ 0.55 → NO-GO. Pose features do not discriminate.")
    elif overall_best <= 0.65:
        print("[GATE] best AUC ∈ (0.55, 0.65] → AMBIGUOUS. Run Pass 3 retrain.")
    else:
        print("[GATE] best AUC ≥ 0.65 → PROCEED to Pass 3.")


if __name__ == "__main__":
    main()
