"""Pass 3 — focused-subpopulation retrain probe.

Re-train a GBM with the same hyperparams as production, but with positives
restricted to the 128 non-block, classifier-rejected FN cases (the bucket
where pose features are hypothesised to live) and negatives drawn only from
the same rallies' non-GT candidates. This isolates whether the pose-feature
importance dilution observed in production (≤1%) is intrinsic to the feature
or a consequence of being drowned out by easier positives.

If pose importance rises meaningfully (>5% sum) under the focused training,
pose features ARE useful and the production GBM under-uses them — feature
weighting / curriculum / class-balanced training becomes the lever.

If pose importance stays low even when forced to discriminate the FN bucket
from same-rally noise, pose features themselves are not the lever for the
128-case bucket and we should look elsewhere.

Comparison baseline: re-train on the SAME rallies but with NORMAL labels
(all GT contacts as positives). Compare feature importances. Production's
weights (39.9% seq, 39.2% frames_since_last, 5.4% player_distance, ≤2%
each pose feature) are the global reference.

Usage:
    cd analysis
    uv run python scripts/pose_audit_pass3_2026_04_24.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rallycut.tracking.ball_tracker import BallPosition as BallPos  # noqa: E402
from rallycut.tracking.contact_classifier import (  # noqa: E402
    CandidateFeatures,
    ContactClassifier,
)
from rallycut.tracking.contact_detector import ContactDetectionConfig  # noqa: E402
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos  # noqa: E402
from scripts.eval_action_detection import load_rallies_with_action_gt  # noqa: E402
from scripts.train_contact_classifier import (  # noqa: E402
    extract_candidate_features,
    label_candidates,
)

POSE_FEATURE_NAMES = (
    "nearest_active_wrist_velocity_max",
    "nearest_hand_ball_dist_min",
    "nearest_active_arm_extension_change",
    "nearest_pose_confidence_mean",
    "nearest_both_arms_raised",
)
DEFAULT_CORPUS = REPO_ROOT / "outputs" / "action_errors" / "corpus_annotated.jsonl"
DEFAULT_OUT_DIR = REPO_ROOT / "reports" / "pose_audit"
TOLERANCE = 7  # match production training tolerance
MATCH_TOLERANCE = 5  # for "is this candidate near a GT?"


def load_fn_bucket(corpus_path: Path) -> dict[str, set[int]]:
    """Return {rally_id: set(gt_frames)} for non-block, rejected_by_classifier."""
    by_rally: dict[str, set[int]] = {}
    with corpus_path.open() as f:
        for line in f:
            d = json.loads(line)
            if d.get("error_class") != "FN_contact":
                continue
            if d.get("fn_subcategory") != "rejected_by_classifier":
                continue
            if d.get("gt_action") == "block":
                continue
            by_rally.setdefault(d["rally_id"], set()).add(int(d["gt_frame"]))
    return by_rally


def extract_features_for_rally(  # type: ignore[no-untyped-def]
    rally,
    sequence_probs: np.ndarray | None,
    gt_frames: list[int] | None,
    contact_config: ContactDetectionConfig,
) -> tuple[list, list[int]]:
    """Wrap extract_candidate_features with a stable interface."""
    return extract_candidate_features(
        rally, config=contact_config, gt_frames=gt_frames,
        sequence_probs=sequence_probs,
    )


def matches_any_frame(candidate_frame: int, target_frames: set[int],
                      tolerance: int = MATCH_TOLERANCE) -> bool:
    return any(abs(candidate_frame - tf) <= tolerance for tf in target_frames)


def train_and_report(
    label: str, x_mat: np.ndarray, y: np.ndarray, feat_names: list[str],
) -> dict[str, float]:
    """Train GBM with production hyperparams; return feature importances."""
    classifier = ContactClassifier(threshold=0.40)
    metrics = classifier.train(x_mat, y, positive_weight=1.0)
    importances = classifier.feature_importance()
    sorted_imps = sorted(importances.items(), key=lambda kv: -kv[1])
    print(f"\n=== {label} (n={len(y)}, +{int(y.sum())}/-{int((1-y).sum())}) ===")
    print(f"  F1={metrics['train_f1']:.3f}  P={metrics['train_precision']:.3f}  "
          f"R={metrics['train_recall']:.3f}")
    print("  Top features by importance:")
    for name, imp in sorted_imps[:15]:
        marker = " ◀ POSE" if name in POSE_FEATURE_NAMES else ""
        print(f"    {name:<40} {imp:.4f}{marker}")
    pose_total = sum(importances.get(n, 0.0) for n in POSE_FEATURE_NAMES)
    print(f"  POSE total importance: {pose_total:.4f}")
    return importances


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    fn_bucket = load_fn_bucket(args.corpus)
    print(f"[pass3] FN bucket: {sum(len(v) for v in fn_bucket.values())} "
          f"non-block cases across {len(fn_bucket)} rallies", flush=True)

    print("[pass3] Loading rallies with action GT...", flush=True)
    t0 = time.time()
    all_rallies = load_rallies_with_action_gt()
    print(f"[pass3] Loaded {len(all_rallies)} rallies in "
          f"{time.time() - t0:.1f}s", flush=True)

    target_rallies = [r for r in all_rallies if r.rally_id in fn_bucket]
    print(f"[pass3] Restricting to {len(target_rallies)} rallies that contain "
          "an FN-bucket case", flush=True)

    # --- Compute sequence probs once per rally ---
    from rallycut.tracking.sequence_action_runtime import get_sequence_probs
    seq_cache: dict[str, np.ndarray | None] = {}
    print("[pass3] Computing sequence probs per rally (this is the slow step)...",
          flush=True)
    t0 = time.time()
    for i, rally in enumerate(target_rallies, 1):
        bps = [
            BallPos(frame_number=bp["frameNumber"], x=bp["x"], y=bp["y"],
                    confidence=bp.get("confidence", 1.0))
            for bp in (rally.ball_positions_json or [])
            if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
        ]
        _pps = [
            PlayerPos(frame_number=pp["frameNumber"], track_id=pp["trackId"],
                      x=pp["x"], y=pp["y"], width=pp["width"],
                      height=pp["height"], confidence=pp.get("confidence", 1.0),
                      keypoints=pp.get("keypoints"))
            for pp in (rally.positions_json or [])
        ]
        seq_cache[rally.rally_id] = get_sequence_probs(
            bps, _pps, rally.court_split_y, rally.frame_count or 0, None,
        )
        if i % 10 == 0 or i == len(target_rallies):
            print(f"[pass3] [{i}/{len(target_rallies)}] seq probs in "
                  f"{time.time() - t0:.0f}s", flush=True)
    n_seq = sum(1 for v in seq_cache.values() if v is not None)
    print(f"[pass3] Seq probs ready for {n_seq}/{len(target_rallies)} rallies",
          flush=True)

    contact_config = ContactDetectionConfig()

    # --- Extract features per rally ---
    print("[pass3] Extracting candidate features per rally...", flush=True)
    t0 = time.time()
    all_features: list[np.ndarray] = []
    all_labels_standard: list[int] = []  # 1 if matches any GT
    all_labels_focused: list[int] = []   # 1 if matches FN-bucket frame, 0 if no GT match, -1 to skip
    all_rally_ids: list[str] = []

    for i, rally in enumerate(target_rallies, 1):
        gt_frames = [gt.frame for gt in rally.gt_labels]
        features_list, candidate_frames = extract_features_for_rally(
            rally, seq_cache.get(rally.rally_id), gt_frames, contact_config,
        )
        if not features_list:
            continue
        labels_standard = label_candidates(
            candidate_frames, rally.gt_labels, tolerance=TOLERANCE,
        )
        fn_frames = fn_bucket.get(rally.rally_id, set())
        for feat, frame, std_lbl in zip(
            features_list, candidate_frames, labels_standard,
        ):
            all_features.append(feat.to_array())
            all_labels_standard.append(std_lbl)
            # Focused labels:
            #   1 if candidate matches an FN-bucket frame (the hard positive)
            #   0 if candidate matches NO GT (true noise)
            #  -1 if candidate matches a non-FN GT (skip, label-leakage)
            if matches_any_frame(frame, fn_frames):
                all_labels_focused.append(1)
            elif std_lbl == 0:
                all_labels_focused.append(0)
            else:
                # Matches a GT but not an FN-bucket frame — it's an "easy"
                # production positive; exclude from focused training to keep
                # the comparison clean.
                all_labels_focused.append(-1)
            all_rally_ids.append(rally.rally_id)

        if i % 10 == 0 or i == len(target_rallies):
            print(f"[pass3] [{i}/{len(target_rallies)}] features in "
                  f"{time.time() - t0:.0f}s, "
                  f"{len(all_features)} candidates accumulated", flush=True)

    x_all = np.array(all_features)
    y_std = np.array(all_labels_standard)
    y_foc = np.array(all_labels_focused)
    feat_names = CandidateFeatures.feature_names()

    print(f"\n[pass3] Total candidates: {len(y_std)}")
    print(f"[pass3] Standard pos: {int(y_std.sum())}, neg: "
          f"{int((1 - y_std).sum())}")
    print(f"[pass3] Focused: pos {int((y_foc == 1).sum())}, "
          f"neg {int((y_foc == 0).sum())}, skipped (other-GT) "
          f"{int((y_foc == -1).sum())}")

    # --- Baseline: train on standard labels (all GT positives) ---
    imp_std = train_and_report(
        "STANDARD (all GT contacts as positives, same rallies)",
        x_all, y_std.astype(np.int64), feat_names,
    )

    # --- Focused: train on FN-bucket positives only ---
    mask = y_foc != -1
    if mask.sum() < 10:
        print("[pass3] Insufficient focused-training data")
        return
    imp_foc = train_and_report(
        "FOCUSED (FN-bucket positives + same-rally noise negatives)",
        x_all[mask], y_foc[mask].astype(np.int64), feat_names,
    )

    # --- Comparison report ---
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE COMPARISON: STANDARD vs FOCUSED")
    print("=" * 70)
    print(f"{'Feature':<42} {'STD':>8} {'FOC':>8} {'Δ':>8}")
    sorted_by_focused = sorted(
        feat_names, key=lambda n: -imp_foc.get(n, 0.0),
    )
    pose_std_total = 0.0
    pose_foc_total = 0.0
    for name in sorted_by_focused[:18]:
        s = imp_std.get(name, 0.0)
        f = imp_foc.get(name, 0.0)
        d = f - s
        marker = " ◀ POSE" if name in POSE_FEATURE_NAMES else ""
        print(f"{name:<42} {s:>8.4f} {f:>8.4f} {d:>+8.4f}{marker}")
    for name in POSE_FEATURE_NAMES:
        pose_std_total += imp_std.get(name, 0.0)
        pose_foc_total += imp_foc.get(name, 0.0)
    print()
    print(f"  POSE total: standard={pose_std_total:.4f}  focused={pose_foc_total:.4f}  "
          f"Δ={pose_foc_total - pose_std_total:+.4f}")
    print()
    print(">5% rise in pose importance under focused training would indicate "
          "pose features ARE useful and production GBM dilutes them.")
    if pose_foc_total - pose_std_total > 0.05:
        print(f"[VERDICT] Pose importance rose by "
              f"{(pose_foc_total - pose_std_total) * 100:.1f}pp → "
              f"pose features ARE underused in production.")
    elif pose_foc_total > 0.10:
        print(f"[VERDICT] Pose importance under focused = "
              f"{pose_foc_total * 100:.1f}% (still meaningful, "
              f"but rise vs standard small).")
    else:
        print(f"[VERDICT] Pose importance stays low ({pose_foc_total * 100:.1f}%) "
              f"even on FN bucket — pose is not the lever.")

    # Persist for later analysis
    out_path = args.out_dir / "pose_audit_pass3.json"
    out_path.write_text(json.dumps({
        "n_candidates": int(len(y_std)),
        "n_focused_pos": int((y_foc == 1).sum()),
        "n_focused_neg": int((y_foc == 0).sum()),
        "importance_standard": imp_std,
        "importance_focused": imp_foc,
        "pose_total_standard": pose_std_total,
        "pose_total_focused": pose_foc_total,
    }, indent=2))
    print(f"\n[pass3] Wrote {out_path}")


if __name__ == "__main__":
    main()
