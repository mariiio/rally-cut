#!/usr/bin/env python3
"""Diagnose rally detection on specific videos with detailed probability analysis.

Runs TemporalMaxer on target videos, matches predictions vs GT, and categorizes
each error by root cause (model-level, threshold, over-split, over-merge, spurious).

Usage:
    uv run python scripts/diagnose_new_videos.py
    uv run python scripts/diagnose_new_videos.py --video-ids b097dd2a fb83f876
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from rallycut.evaluation.ground_truth import EvaluationVideo, load_evaluation_videos
from rallycut.evaluation.matching import compute_iou, match_rallies
from rallycut.temporal.ball_features import (
    BALL_FEATURE_DIM,
    combine_features,
    extract_ball_features,
    load_ball_density,
)
from rallycut.temporal.features import FeatureCache
from rallycut.temporal.temporal_maxer.inference import TemporalMaxerInference

STRIDE = 12
DEFAULT_VIDEO_IDS = ["b097dd2a", "fb83f876"]


def load_and_prepare_features(
    inference: TemporalMaxerInference,
    feature_cache: FeatureCache,
    video: EvaluationVideo,
) -> tuple[np.ndarray, float] | None:
    """Load features with ball feature handling. Returns (features, fps) or None."""
    cached_data = feature_cache.get(video.content_hash, STRIDE)
    if cached_data is None:
        return None

    features, metadata = cached_data
    expected_dim = inference.model.config.feature_dim

    if expected_dim > features.shape[1]:
        ball_dim = expected_dim - features.shape[1]
        if ball_dim == BALL_FEATURE_DIM:
            ball_data = load_ball_density(video.id, Path("training_data/ball_density"))
            if ball_data is not None:
                confs, ball_fps = ball_data
                ball_feats = extract_ball_features(
                    confs, ball_fps, feature_fps=metadata.fps, stride=STRIDE,
                )
                features = combine_features(features, ball_feats)

        if features.shape[1] < expected_dim:
            pad_width = expected_dim - features.shape[1]
            padding = np.zeros((features.shape[0], pad_width), dtype=features.dtype)
            features = np.concatenate([features, padding], axis=1)

    return features, metadata.fps


def classify_fn(
    gt_start: float,
    gt_end: float,
    window_probs: np.ndarray,
    window_duration: float,
    predictions: list[tuple[float, float]],
) -> dict:
    """Classify a false negative by root cause."""
    # Get probabilities in GT region
    start_idx = int(gt_start / window_duration)
    end_idx = min(int(gt_end / window_duration) + 1, len(window_probs))
    if end_idx <= start_idx:
        end_idx = start_idx + 1
    region_probs = window_probs[start_idx:min(end_idx, len(window_probs))]

    avg_prob = float(np.mean(region_probs)) if len(region_probs) > 0 else 0.0
    max_prob = float(np.max(region_probs)) if len(region_probs) > 0 else 0.0
    min_prob = float(np.min(region_probs)) if len(region_probs) > 0 else 0.0

    # Check if any predictions partially overlap this GT
    partial_overlaps = []
    for pred_s, pred_e in predictions:
        iou = compute_iou(gt_start, gt_end, pred_s, pred_e)
        if iou > 0:
            partial_overlaps.append((pred_s, pred_e, iou))

    if max_prob < 0.3:
        cause = "model-level"
        explanation = f"Model gives near-zero confidence (max={max_prob:.3f})"
    elif max_prob < 0.5:
        cause = "threshold-low"
        explanation = f"Moderate confidence (max={max_prob:.3f}), below default threshold"
    elif partial_overlaps:
        # There are overlapping predictions but IoU too low for match
        best_overlap = max(partial_overlaps, key=lambda x: x[2])
        if len(partial_overlaps) > 1:
            cause = "over-split"
            explanation = (
                f"Rally split into {len(partial_overlaps)} pieces "
                f"(best IoU={best_overlap[2]:.2f})"
            )
        else:
            cause = "boundary-error"
            explanation = (
                f"Partial overlap IoU={best_overlap[2]:.2f} "
                f"pred={best_overlap[0]:.1f}-{best_overlap[1]:.1f}s"
            )
    elif avg_prob >= 0.5:
        cause = "threshold-tunable"
        explanation = f"Good confidence (avg={avg_prob:.3f}) but not segmented"
    else:
        cause = "model-level"
        explanation = f"Low avg confidence ({avg_prob:.3f})"

    return {
        "cause": cause,
        "explanation": explanation,
        "avg_prob": avg_prob,
        "max_prob": max_prob,
        "min_prob": min_prob,
        "region_probs": region_probs.tolist() if len(region_probs) > 0 else [],
        "partial_overlaps": partial_overlaps,
    }


def classify_fp(
    pred_start: float,
    pred_end: float,
    window_probs: np.ndarray,
    window_duration: float,
    ground_truth: list[tuple[float, float]],
) -> dict:
    """Classify a false positive by root cause."""
    start_idx = int(pred_start / window_duration)
    end_idx = min(int(pred_end / window_duration) + 1, len(window_probs))
    region_probs = window_probs[start_idx:min(end_idx, len(window_probs))]
    avg_prob = float(np.mean(region_probs)) if len(region_probs) > 0 else 0.0

    # Check for adjacent GT rallies that might be over-merged
    adjacent_gts = []
    for gt_s, gt_e in ground_truth:
        iou = compute_iou(pred_start, pred_end, gt_s, gt_e)
        if iou > 0:
            adjacent_gts.append((gt_s, gt_e, iou))

    if len(adjacent_gts) >= 2:
        cause = "over-merge"
        explanation = f"Merged {len(adjacent_gts)} GT rallies into one segment"
    elif len(adjacent_gts) == 1:
        cause = "boundary-extension"
        explanation = f"Extends beyond GT rally (IoU={adjacent_gts[0][2]:.2f})"
    else:
        cause = "spurious"
        nearest = min(
            (min(abs(pred_start - gt_e), abs(gt_s - pred_end))
             for gt_s, gt_e in ground_truth),
            default=float("inf"),
        )
        explanation = f"No GT overlap, nearest GT {nearest:.1f}s away"

    return {
        "cause": cause,
        "explanation": explanation,
        "avg_prob": avg_prob,
        "adjacent_gts": adjacent_gts,
    }


def print_probability_timeline(
    video: EvaluationVideo,
    window_probs: np.ndarray,
    window_duration: float,
    predictions: list[tuple[float, float]],
) -> None:
    """Print per-window probability with GT overlay."""
    ground_truth = [(r.start_seconds, r.end_seconds) for r in video.ground_truth_rallies]
    total_windows = len(window_probs)

    print(f"\n  Probability timeline ({total_windows} windows, {window_duration:.2f}s each):")
    print(f"  {'Time':>7} {'Prob':>5} {'Bar':<40} GT  Pred")
    print(f"  {'-' * 70}")

    for i in range(total_windows):
        t = i * window_duration
        prob = window_probs[i]
        bar_len = int(prob * 40)
        bar = "█" * bar_len + "░" * (40 - bar_len)

        # Check if in GT region
        in_gt = any(gt_s <= t <= gt_e for gt_s, gt_e in ground_truth)
        # Check if in prediction region
        in_pred = any(p_s <= t <= p_e for p_s, p_e in predictions)

        gt_marker = " GT" if in_gt else "   "
        pred_marker = " PR" if in_pred else "   "

        # Only print every Nth window for long videos, or transitions
        if total_windows > 100:
            # Always print transitions and every 5th window
            prev_prob = window_probs[i - 1] if i > 0 else 0
            is_transition = abs(prob - prev_prob) > 0.2
            if not is_transition and i % 5 != 0 and i != total_windows - 1:
                continue

        print(f"  {t:6.1f}s {prob:5.3f} {bar}{gt_marker}{pred_marker}")


def diagnose_video(
    inference: TemporalMaxerInference,
    feature_cache: FeatureCache,
    video: EvaluationVideo,
    iou_threshold: float = 0.4,
) -> dict | None:
    """Run full diagnosis on a single video."""
    print(f"\n{'=' * 70}")
    print(f"VIDEO: {video.filename} (id: {video.id[:8]})")
    print(f"{'=' * 70}")

    prepared = load_and_prepare_features(inference, feature_cache, video)
    if prepared is None:
        print("  SKIP: No cached features")
        return None

    features, fps = prepared
    print(f"  Features: {features.shape}, FPS: {fps:.1f}")

    ground_truth = [(r.start_seconds, r.end_seconds) for r in video.ground_truth_rallies]
    print(f"  GT rallies: {len(ground_truth)}")
    for i, (s, e) in enumerate(ground_truth):
        print(f"    [{i}] {s:.1f}-{e:.1f}s ({e - s:.1f}s)")

    # Run inference
    result = inference.predict(
        features=features, fps=fps, stride=STRIDE,
        valley_threshold=0.5, min_valley_duration=2.0,
    )
    predictions = result.segments
    window_probs = result.window_probs
    window_duration = STRIDE / fps

    print(f"\n  Predictions: {len(predictions)}")
    for i, (s, e) in enumerate(predictions):
        avg_p = float(np.mean(window_probs[int(s / window_duration):int(e / window_duration) + 1]))
        print(f"    [{i}] {s:.1f}-{e:.1f}s ({e - s:.1f}s, avg_prob={avg_p:.3f})")

    # Match
    matching = match_rallies(ground_truth, predictions, iou_threshold)
    tp = matching.true_positives
    fp = matching.false_positives
    fn = matching.false_negatives
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    print(f"\n  Results: TP={tp} FP={fp} FN={fn} P={p:.0%} R={r:.0%} F1={f1:.0%}")

    # TP details
    if matching.matches:
        print(f"\n  --- True Positives ({tp}) ---")
        for m in matching.matches:
            gt_s, gt_e = ground_truth[m.ground_truth_idx]
            pred_s, pred_e = predictions[m.predicted_idx]
            print(
                f"    GT[{m.ground_truth_idx}] {gt_s:.1f}-{gt_e:.1f}s → "
                f"Pred[{m.predicted_idx}] {pred_s:.1f}-{pred_e:.1f}s  "
                f"IoU={m.iou:.2f} start={m.start_error_ms:+d}ms end={m.end_error_ms:+d}ms"
            )

    # FN details with classification
    fn_details = []
    if matching.unmatched_ground_truth:
        print(f"\n  --- False Negatives ({fn}) ---")
        for gt_idx in matching.unmatched_ground_truth:
            gt_s, gt_e = ground_truth[gt_idx]
            classification = classify_fn(
                gt_s, gt_e, window_probs, window_duration, predictions,
            )
            fn_details.append({
                "gt_idx": gt_idx, "gt_start": gt_s, "gt_end": gt_e,
                **classification,
            })
            print(
                f"    GT[{gt_idx}] {gt_s:.1f}-{gt_e:.1f}s ({gt_e - gt_s:.1f}s) "
                f"→ {classification['cause'].upper()}: {classification['explanation']}"
            )

    # FP details with classification
    fp_details = []
    if matching.unmatched_predictions:
        print(f"\n  --- False Positives ({fp}) ---")
        for pred_idx in matching.unmatched_predictions:
            pred_s, pred_e = predictions[pred_idx]
            classification = classify_fp(
                pred_s, pred_e, window_probs, window_duration, ground_truth,
            )
            fp_details.append({
                "pred_idx": pred_idx, "pred_start": pred_s, "pred_end": pred_e,
                **classification,
            })
            print(
                f"    Pred[{pred_idx}] {pred_s:.1f}-{pred_e:.1f}s ({pred_e - pred_s:.1f}s) "
                f"→ {classification['cause'].upper()}: {classification['explanation']}"
            )

    # Probability timeline
    print_probability_timeline(video, window_probs, window_duration, predictions)

    return {
        "video_id": video.id,
        "filename": video.filename,
        "tp": tp, "fp": fp, "fn": fn, "f1": f1,
        "fn_details": fn_details,
        "fp_details": fp_details,
    }


def print_summary(results: list[dict]) -> None:
    """Print cross-video summary and root cause breakdown."""
    print(f"\n{'=' * 70}")
    print("CROSS-VIDEO SUMMARY")
    print(f"{'=' * 70}")

    total_tp = sum(r["tp"] for r in results)
    total_fp = sum(r["fp"] for r in results)
    total_fn = sum(r["fn"] for r in results)
    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    print(f"  Total: TP={total_tp} FP={total_fp} FN={total_fn} P={p:.0%} R={r:.0%} F1={f1:.0%}")

    # Root cause breakdown for FNs
    all_fn = [fn for r in results for fn in r["fn_details"]]
    if all_fn:
        print(f"\n  FN Root Causes ({len(all_fn)} total):")
        causes = {}
        for fn in all_fn:
            cause = fn["cause"]
            causes[cause] = causes.get(cause, 0) + 1
        for cause, count in sorted(causes.items(), key=lambda x: -x[1]):
            print(f"    {cause:<20} {count}")

    # Root cause breakdown for FPs
    all_fp = [fp for r in results for fp in r["fp_details"]]
    if all_fp:
        print(f"\n  FP Root Causes ({len(all_fp)} total):")
        causes = {}
        for fp in all_fp:
            cause = fp["cause"]
            causes[cause] = causes.get(cause, 0) + 1
        for cause, count in sorted(causes.items(), key=lambda x: -x[1]):
            print(f"    {cause:<20} {count}")

    # Recommendation
    print(f"\n  RECOMMENDATION:")
    model_level_fns = sum(1 for fn in all_fn if fn["cause"] == "model-level")
    threshold_fns = sum(1 for fn in all_fn if fn["cause"].startswith("threshold"))
    if model_level_fns > len(all_fn) / 2:
        print("    → Retrain TemporalMaxer with these videos included")
        print("      Run: uv run rallycut train temporal-maxer --epochs 50")
    elif threshold_fns > 0:
        print("    → Tune post-processing thresholds (rescue pass, confidence)")
    else:
        print("    → Investigate specific error patterns above")


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose rally detection on specific videos")
    parser.add_argument(
        "--video-ids", nargs="*", default=DEFAULT_VIDEO_IDS,
        help="Video ID prefixes to diagnose (default: b097dd2a fb83f876)",
    )
    parser.add_argument("--iou", type=float, default=0.4, help="IoU threshold")
    args = parser.parse_args()

    model_path = Path("weights/temporal_maxer/best_temporal_maxer.pt")
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    inference = TemporalMaxerInference(model_path, device="cpu")
    feature_cache = FeatureCache(cache_dir=Path("training_data/features"))

    print("Loading evaluation videos...")
    all_videos = load_evaluation_videos()

    # Filter to target videos by ID prefix
    videos = []
    for v in all_videos:
        for prefix in args.video_ids:
            if v.id.startswith(prefix):
                videos.append(v)
                break

    if not videos:
        print(f"ERROR: No videos found matching {args.video_ids}")
        print(f"Available: {[v.id[:8] for v in all_videos]}")
        sys.exit(1)

    print(f"Diagnosing {len(videos)} videos...")
    results = []
    for video in videos:
        result = diagnose_video(inference, feature_cache, video, args.iou)
        if result:
            results.append(result)

    if results:
        print_summary(results)


if __name__ == "__main__":
    main()
