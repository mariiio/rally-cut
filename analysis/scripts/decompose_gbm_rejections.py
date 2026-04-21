"""Per-feature GBM log-odds attribution on Cat 2 FNs.

For each `2-kin_underreports` FN, re-extract the 26-feature vector at the
nearest candidate frame, then decompose the GBM's log-odds output feature-by-
feature via a leave-one-out perturbation: substitute each feature with its
training-set median and measure the change in predicted probability. The
feature with the biggest NEGATIVE impact when restored (i.e., whose real value
drops the probability most below the median-substituted value) is the most
responsible for the rejection.

Tells us whether Fix F's premise holds:
  - if `direction_change_deg` is the dominant negative feature in >70% of
    Cat 2 FNs → Fix F (narrower window) is viable
  - if `ball_detection_density` or pose features dominate → Fix F doesn't help,
    consider Fix G or defer

Usage:
    cd analysis && uv run python scripts/decompose_gbm_rejections.py
"""
from __future__ import annotations

import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rallycut.tracking.contact_classifier import CandidateFeatures
from rallycut.tracking.contact_detector import ContactDetectionConfig
from scripts.eval_action_detection import load_rallies_with_action_gt
from scripts.eval_loo_video import (
    _precompute,
    _train_fold,
)
from scripts.train_contact_classifier import extract_candidate_features

logging.getLogger("rallycut.tracking.action_classifier").setLevel(logging.ERROR)

REPO = Path(__file__).resolve().parent.parent
CAT_PATH = REPO / "outputs" / "phase4_category_assignments.jsonl"
OUT_PATH = REPO / "outputs" / "gbm_decomposition_2026_04_21.jsonl"


def main() -> None:
    # Load Cat 2 FNs
    cat2 = []
    for line in CAT_PATH.open():
        a = json.loads(line)
        if a["primary_category"] == "2-kin_underreports":
            cat2.append(a)
    print(f"Loaded {len(cat2)} Cat 2 FNs.")

    rallies_of_interest = defaultdict(list)
    for fn in cat2:
        rallies_of_interest[fn["rally_id"]].append(fn)
    print(f"Spanning {len(rallies_of_interest)} rallies. Loading + precomputing...")

    cfg = ContactDetectionConfig()
    all_rallies = load_rallies_with_action_gt()
    target_ids = set(rallies_of_interest.keys())

    # Precompute all (for training)
    all_precomputed = []
    for r in all_rallies:
        pre = _precompute(r, cfg)
        if pre is not None:
            all_precomputed.append(pre)
    print(f"Precomputed {len(all_precomputed)}. Target: {len(target_ids)}")

    by_video = defaultdict(list)
    for p in all_precomputed:
        by_video[p.rally.video_id].append(p)

    # Compute training-set medians for each feature (use all_precomputed training data)
    # We use per-rally candidates → features, not just matched ones. Medians will be
    # approximate but sufficient for per-case attribution.
    print("Computing training-set feature medians...")
    all_feat_matrices = []
    for pre in all_precomputed[:100]:  # Sample for speed
        feats, _ = extract_candidate_features(
            pre.rally, config=cfg, sequence_probs=pre.sequence_probs,
            gt_frames=[gt.frame for gt in pre.rally.gt_labels],
        )
        if feats:
            all_feat_matrices.append(np.array([f.to_array() for f in feats]))
    if all_feat_matrices:
        big_matrix = np.vstack(all_feat_matrices)
        medians = np.median(big_matrix, axis=0)
    else:
        medians = np.zeros(26)
    feature_names = CandidateFeatures.feature_names()
    print(f"Medians computed over {sum(m.shape[0] for m in all_feat_matrices)} candidates.")
    for i, (n, m) in enumerate(zip(feature_names, medians)):
        print(f"  [{i:2d}] {n:40s} median={m:.4f}")

    # Process each target rally
    target_precomputed = {p.rally.rally_id: p for p in all_precomputed
                          if p.rally.rally_id in target_ids}
    video_ids = sorted({p.rally.video_id for p in target_precomputed.values()})

    results = []
    for v_idx, vid in enumerate(video_ids, 1):
        held = [p for p in target_precomputed.values() if p.rally.video_id == vid]
        train = [p for v, rs in by_video.items() if v != vid for p in rs]
        contact_clf, _ = _train_fold(train, threshold=0.30)
        print(f"[{v_idx}/{len(video_ids)}] video {vid[:8]} ({len(held)} target rallies)")

        if not contact_clf.is_trained:
            continue

        # For each rally, extract features and locate the candidate nearest to each FN
        for pre in held:
            rally = pre.rally
            gt_frames = [gt.frame for gt in rally.gt_labels]
            feats, cand_frames = extract_candidate_features(
                rally, config=cfg, sequence_probs=pre.sequence_probs,
                gt_frames=gt_frames,
            )
            if not feats:
                continue

            for fn in rallies_of_interest.get(rally.rally_id, []):
                gt_frame = fn["gt_frame"]
                # Find closest candidate
                best_idx = -1
                best_dist = 1e9
                for i, cf in enumerate(cand_frames):
                    d = abs(cf - gt_frame)
                    if d < best_dist:
                        best_dist = d
                        best_idx = i
                if best_idx < 0 or best_dist > 5:
                    continue
                feat = feats[best_idx]
                x = feat.to_array()

                # Baseline probability at real features
                expected = contact_clf.model.n_features_in_
                x_padded = x[:expected] if len(x) > expected else np.concatenate([x, np.zeros(expected - len(x))])
                base_prob = float(contact_clf.model.predict_proba(
                    x_padded.reshape(1, -1))[:, 1][0])

                # Leave-one-out: substitute each feature with median, see prob change
                deltas = []
                for i in range(min(expected, len(x))):
                    x_perturb = x_padded.copy()
                    x_perturb[i] = medians[i]
                    perturb_prob = float(contact_clf.model.predict_proba(
                        x_perturb.reshape(1, -1))[:, 1][0])
                    # Delta = perturb - base; positive means "substituting to median
                    # would INCREASE prob" → current value is pushing rejection
                    deltas.append({
                        "feature": feature_names[i],
                        "value": float(x[i]),
                        "median": float(medians[i]),
                        "base_prob": base_prob,
                        "perturb_prob": perturb_prob,
                        "delta": perturb_prob - base_prob,
                    })
                deltas.sort(key=lambda d: -d["delta"])
                top3_blockers = [d for d in deltas[:3] if d["delta"] > 0.001]

                results.append({
                    "rally_id": rally.rally_id,
                    "gt_frame": gt_frame,
                    "gt_action": fn["gt_action"],
                    "candidate_frame": cand_frames[best_idx],
                    "candidate_distance_from_gt": best_dist,
                    "base_prob": base_prob,
                    "top_blockers": top3_blockers,
                })

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")

    # Summary: histogram of TOP-1 blocker feature across all Cat 2 FNs
    top1_counter: Counter = Counter()
    top_any_counter: Counter = Counter()  # count feature appearing in any top-3 blocker
    for r in results:
        if r["top_blockers"]:
            top1_counter[r["top_blockers"][0]["feature"]] += 1
            for b in r["top_blockers"]:
                top_any_counter[b["feature"]] += 1
        else:
            top1_counter["<none_substantial>"] += 1

    total = len(results)
    print(f"\n=== Cat 2 GBM feature attribution (n={total}) ===")
    print("\nTop-1 blocker feature (ranked by freq):")
    for f, n in top1_counter.most_common(10):
        pct = 100 * n / total if total else 0
        print(f"  {f:40s} {n:3d} ({pct:.1f}%)")
    print("\nTop-3 blocker coverage (feature appearing in top-3):")
    for f, n in top_any_counter.most_common(10):
        pct = 100 * n / total if total else 0
        print(f"  {f:40s} {n:3d} ({pct:.1f}%)")

    print(f"\nWrote {OUT_PATH.relative_to(REPO)}")


if __name__ == "__main__":
    main()
