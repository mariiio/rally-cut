#!/usr/bin/env python3
"""Diagnose rally detection failures on weak videos.

For each weak video (LOO-trained model), classifies every FN and FP:
- Zero-detection FN: model outputs near-zero probability in GT range
- Low-confidence FN: model detects weakly but below threshold
- Boundary FN: detected but IoU < threshold (alignment issue)
- FP: false alarm with confidence, duration, distance to nearest GT

Usage:
    uv run python scripts/diagnose_rally_weak_videos.py
    uv run python scripts/diagnose_rally_weak_videos.py --videos muchi matahtach machi mechi
    uv run python scripts/diagnose_rally_weak_videos.py --all
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from rallycut.evaluation.ground_truth import EvaluationVideo, load_evaluation_videos
from rallycut.evaluation.matching import compute_iou, match_rallies
from rallycut.temporal.features import FeatureCache, generate_overlap_labels
from rallycut.temporal.temporal_maxer.inference import TemporalMaxerInference
from rallycut.temporal.temporal_maxer.training import (
    TemporalMaxerTrainer,
    TemporalMaxerTrainingConfig,
)

logging.basicConfig(level=logging.WARNING)

DEFAULT_STRIDE = 24
FEATURE_DIR = Path("training_data/features")
WEAK_VIDEOS = ["muchi", "matahtach", "machi", "mechi"]


def auto_detect_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_video_data(
    video: EvaluationVideo,
    cache: FeatureCache,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    cached_data = cache.get(video.content_hash, stride=stride)
    if cached_data is None:
        return None
    features, metadata = cached_data
    duration_ms = int(metadata.duration_seconds * 1000)
    labels = generate_overlap_labels(
        rallies=video.ground_truth_rallies,
        video_duration_ms=duration_ms,
        fps=metadata.fps,
        stride=stride,
    )
    min_len = min(len(features), len(labels))
    features = features[:min_len]
    labels_arr = np.array(labels[:min_len], dtype=np.float32)
    return features, labels_arr, metadata.fps, metadata.duration_seconds


def classify_fn(
    gt_start: float,
    gt_end: float,
    window_probs: np.ndarray,
    window_duration: float,
    pred_segments: list[tuple[float, float]],
    iou_threshold: float,
) -> dict:
    """Classify a false negative into failure mode."""
    s_idx = max(0, int(gt_start / window_duration))
    e_idx = min(int(gt_end / window_duration) + 1, len(window_probs))
    gt_probs = window_probs[s_idx:e_idx] if e_idx > s_idx else np.array([0.0])

    max_prob = float(np.max(gt_probs))
    mean_prob = float(np.mean(gt_probs))
    frac_above_05 = float(np.mean(gt_probs > 0.5))

    # Check if any prediction partially overlaps (boundary issue)
    best_iou = 0.0
    best_pred = None
    for ps, pe in pred_segments:
        iou = compute_iou(gt_start, gt_end, ps, pe)
        if iou > best_iou:
            best_iou = iou
            best_pred = (ps, pe)

    if best_iou > 0 and best_iou < iou_threshold:
        mode = "boundary"
    elif max_prob < 0.3:
        mode = "zero_detection"
    elif max_prob < 0.6:
        mode = "low_confidence"
    else:
        mode = "missed_postprocess"

    return {
        "mode": mode,
        "gt_range": f"{gt_start:.1f}-{gt_end:.1f}s",
        "gt_duration": gt_end - gt_start,
        "max_prob": max_prob,
        "mean_prob": mean_prob,
        "frac_above_05": frac_above_05,
        "best_iou": best_iou,
        "best_pred": best_pred,
        "prob_profile": gt_probs,
    }


def classify_fp(
    pred_start: float,
    pred_end: float,
    window_probs: np.ndarray,
    window_duration: float,
    gt_segments: list[tuple[float, float]],
) -> dict:
    """Classify a false positive."""
    s_idx = max(0, int(pred_start / window_duration))
    e_idx = min(int(pred_end / window_duration) + 1, len(window_probs))
    seg_probs = window_probs[s_idx:e_idx] if e_idx > s_idx else np.array([0.0])

    nearest_gap = float("inf")
    nearest_gt = None
    for gs, ge in gt_segments:
        if pred_start > ge:
            gap = pred_start - ge
        elif gs > pred_end:
            gap = gs - pred_end
        else:
            gap = 0.0
        if gap < nearest_gap:
            nearest_gap = gap
            nearest_gt = (gs, ge)

    return {
        "pred_range": f"{pred_start:.1f}-{pred_end:.1f}s",
        "duration": pred_end - pred_start,
        "avg_prob": float(np.mean(seg_probs)),
        "max_prob": float(np.max(seg_probs)),
        "nearest_gt_gap": nearest_gap if nearest_gap < float("inf") else None,
        "nearest_gt": nearest_gt,
    }


def diagnose_video(
    video: EvaluationVideo,
    all_videos: list[EvaluationVideo],
    cache: FeatureCache,
    device: str,
    stride: int,
    iou_threshold: float = 0.4,
) -> None:
    """Run LOO diagnosis on a single video."""
    train_videos = [v for v in all_videos if v.id != video.id]

    # Load data
    held_data = load_video_data(video, cache, stride)
    if held_data is None:
        print(f"  SKIP: no cached features")
        return

    ho_features, ho_labels, ho_fps, ho_duration = held_data

    train_features = []
    train_labels = []
    for v in train_videos:
        data = load_video_data(v, cache, stride)
        if data is not None:
            train_features.append(data[0])
            train_labels.append(data[1])

    # Train LOO model
    config = TemporalMaxerTrainingConfig(
        learning_rate=5e-4, epochs=50, batch_size=4, patience=15,
        device=device, seed=42,
    )
    trainer = TemporalMaxerTrainer(config=config)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        t0 = time.time()
        trainer.train(train_features, train_labels, [], [], output_dir)
        train_time = time.time() - t0

        model_path = output_dir / "best_temporal_maxer.pt"
        inference = TemporalMaxerInference(model_path, device="cpu")
        result = inference.predict(
            features=ho_features, fps=ho_fps, stride=stride,
            min_segment_confidence=0.6,
            valley_threshold=0.5, min_valley_duration=2.0,
        )

    gt_segments = [(r.start_seconds, r.end_seconds) for r in video.ground_truth_rallies]
    pred_segments = result.segments
    window_duration = stride / ho_fps

    matching = match_rallies(gt_segments, pred_segments, iou_threshold)

    # Print header
    gt_durations = [e - s for s, e in gt_segments]
    print(f"\n{'='*70}")
    print(f"VIDEO: {video.filename}")
    print(f"{'='*70}")
    print(f"  GT rallies: {len(gt_segments)}, Predictions: {len(pred_segments)}")
    print(f"  TP={matching.true_positives} FP={matching.false_positives} FN={matching.false_negatives}")
    print(f"  Train time: {train_time:.1f}s")
    print(f"  GT durations: {', '.join(f'{d:.1f}s' for d in gt_durations)}")
    print(f"  Duration: {ho_duration:.1f}s, FPS: {ho_fps:.1f}")

    # Diagnose FNs
    if matching.unmatched_ground_truth:
        print(f"\n  FALSE NEGATIVES ({len(matching.unmatched_ground_truth)}):")
        print(f"  {'Mode':<18} {'GT Range':<16} {'Dur':>5} {'MaxP':>5} {'MeanP':>6} {'%>0.5':>6} {'BestIoU':>8} {'BestPred':<18}")
        print(f"  {'-'*90}")

        fn_modes: dict[str, int] = {}
        for gt_idx in matching.unmatched_ground_truth:
            gs, ge = gt_segments[gt_idx]
            info = classify_fn(gs, ge, result.window_probs, window_duration, pred_segments, iou_threshold)
            fn_modes[info["mode"]] = fn_modes.get(info["mode"], 0) + 1
            bp = f"{info['best_pred'][0]:.1f}-{info['best_pred'][1]:.1f}s" if info["best_pred"] else "-"
            print(
                f"  {info['mode']:<18} {info['gt_range']:<16} {info['gt_duration']:>4.1f}s "
                f"{info['max_prob']:>5.2f} {info['mean_prob']:>5.2f} {info['frac_above_05']:>5.1%} "
                f"{info['best_iou']:>7.3f} {bp:<18}"
            )

            # Print probability profile for zero/low-confidence FNs
            if info["mode"] in ("zero_detection", "low_confidence"):
                probs = info["prob_profile"]
                if len(probs) <= 20:
                    prob_str = " ".join(f"{p:.2f}" for p in probs)
                else:
                    prob_str = " ".join(f"{p:.2f}" for p in probs[:10]) + " ... " + " ".join(f"{p:.2f}" for p in probs[-5:])
                print(f"    probs: [{prob_str}]")

        print(f"\n  FN mode summary: {fn_modes}")

    # Diagnose FPs
    if matching.unmatched_predictions:
        print(f"\n  FALSE POSITIVES ({len(matching.unmatched_predictions)}):")
        print(f"  {'Pred Range':<18} {'Dur':>5} {'AvgP':>5} {'MaxP':>5} {'NearGT':>8} {'NearGT Range':<18}")
        print(f"  {'-'*70}")

        for pred_idx in matching.unmatched_predictions:
            ps, pe = pred_segments[pred_idx]
            info = classify_fp(ps, pe, result.window_probs, window_duration, gt_segments)
            ng = f"{info['nearest_gt_gap']:.1f}s" if info["nearest_gt_gap"] is not None else "-"
            ngr = f"{info['nearest_gt'][0]:.1f}-{info['nearest_gt'][1]:.1f}s" if info["nearest_gt"] else "-"
            print(
                f"  {info['pred_range']:<18} {info['duration']:>4.1f}s "
                f"{info['avg_prob']:>5.2f} {info['max_prob']:>5.2f} {ng:>8} {ngr:<18}"
            )

    # Print TPs with boundary errors
    if matching.matches:
        print(f"\n  TRUE POSITIVES ({len(matching.matches)}):")
        print(f"  {'GT Range':<16} {'Pred Range':<16} {'IoU':>5} {'StartErr':>9} {'EndErr':>9}")
        print(f"  {'-'*60}")
        for m in matching.matches:
            gs, ge = gt_segments[m.ground_truth_idx]
            ps, pe = pred_segments[m.predicted_idx]
            print(
                f"  {gs:.1f}-{ge:.1f}s{'':<6} {ps:.1f}-{pe:.1f}s{'':<6} "
                f"{m.iou:>5.2f} {m.start_error_ms:>+8}ms {m.end_error_ms:>+8}ms"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose rally detection weak videos")
    parser.add_argument(
        "--videos", nargs="*", default=None,
        help=f"Video filename substrings to diagnose (default: {WEAK_VIDEOS})",
    )
    parser.add_argument("--all", action="store_true", help="Diagnose all videos")
    parser.add_argument("--stride", type=int, default=DEFAULT_STRIDE)
    parser.add_argument("--iou", type=float, default=0.4, help="IoU threshold")
    args = parser.parse_args()

    device = auto_detect_device()
    print(f"Device: {device}")

    videos = load_evaluation_videos()
    cache = FeatureCache(cache_dir=FEATURE_DIR)
    videos = [v for v in videos if cache.has(v.content_hash, stride=args.stride)]
    print(f"Found {len(videos)} videos with cached features")

    if args.all:
        targets = videos
    else:
        patterns = args.videos or WEAK_VIDEOS
        targets = [
            v for v in videos
            if any(p.lower() in v.filename.lower() for p in patterns)
        ]

    if not targets:
        print("No matching videos found!")
        sys.exit(1)

    print(f"Diagnosing {len(targets)} videos: {', '.join(v.filename for v in targets)}")

    for video in targets:
        diagnose_video(video, videos, cache, device, args.stride, args.iou)

    print(f"\n{'='*70}")
    print("DONE")


if __name__ == "__main__":
    main()
