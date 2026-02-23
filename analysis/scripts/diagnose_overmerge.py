#!/usr/bin/env python3
"""Diagnose over-merged rally segments by dumping per-window probabilities.

Shows per-window rally probabilities for a video in a time range, helping
verify that dead-time regions have low probabilities suitable for valley
splitting.

Usage:
    uv run python scripts/diagnose_overmerge.py --video <content_hash> --start 150 --end 200
    uv run python scripts/diagnose_overmerge.py --video beb70f61 --start 150 --end 200
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from rallycut.evaluation.ground_truth import load_evaluation_videos
from rallycut.temporal.features import FeatureCache
from rallycut.temporal.temporal_maxer.inference import TemporalMaxerInference

STRIDE = 24


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose over-merged rally segments")
    parser.add_argument(
        "--video", required=True,
        help="Video content hash (or prefix) to analyze",
    )
    parser.add_argument("--start", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--end", type=float, default=0.0, help="End time in seconds (0 = all)")
    parser.add_argument(
        "--valley-threshold", type=float, default=0.5,
        help="Valley threshold to highlight (default: 0.5)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    model_path = Path("weights/temporal_maxer/best_temporal_maxer.pt")
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    # Find matching video
    videos = load_evaluation_videos()
    matches = [v for v in videos if v.content_hash.startswith(args.video)]
    if not matches:
        print(f"ERROR: No video found matching '{args.video}'")
        print("Available videos:")
        for v in videos:
            print(f"  {v.content_hash[:8]} {v.filename}")
        sys.exit(1)
    video = matches[0]
    print(f"Video: {video.filename} ({video.content_hash[:8]})")

    # Load features
    cache = FeatureCache(cache_dir=Path("training_data/features"))
    cached_data = cache.get(video.content_hash, STRIDE)
    if cached_data is None:
        print(f"ERROR: No cached features for {video.filename}")
        sys.exit(1)

    features, metadata = cached_data
    print(f"FPS: {metadata.fps}, Duration: {metadata.duration_seconds:.1f}s")
    print(f"Windows: {len(features)}, Stride: {STRIDE}")

    # Run inference
    inference = TemporalMaxerInference(model_path, device="cpu")
    result = inference.predict(
        features=features, fps=metadata.fps, stride=STRIDE,
        valley_threshold=0.0,  # Disable valley splitting for raw view
    )

    window_duration = STRIDE / metadata.fps
    probs = result.window_probs

    # Print GT rallies in range
    gt_rallies = [
        (r.start_seconds, r.end_seconds) for r in video.ground_truth_rallies
    ]
    start_t = args.start
    end_t = args.end if args.end > 0 else metadata.duration_seconds

    gt_in_range = [(s, e) for s, e in gt_rallies if e >= start_t and s <= end_t]
    if gt_in_range:
        print(f"\nGT rallies in [{start_t:.1f}s, {end_t:.1f}s]:")
        for s, e in gt_in_range:
            print(f"  {s:.1f}s - {e:.1f}s ({e - s:.1f}s)")

    # Print predicted segments in range
    pred_in_range = [(s, e) for s, e in result.segments if e >= start_t and s <= end_t]
    if pred_in_range:
        print("\nPredicted segments (no valley split):")
        for s, e in pred_in_range:
            print(f"  {s:.1f}s - {e:.1f}s ({e - s:.1f}s)")

    # Also show with valley splitting enabled
    result_split = inference.predict(
        features=features, fps=metadata.fps, stride=STRIDE,
        valley_threshold=args.valley_threshold,
    )
    split_in_range = [(s, e) for s, e in result_split.segments if e >= start_t and s <= end_t]
    if split_in_range:
        print(f"\nPredicted segments (valley_threshold={args.valley_threshold}):")
        for s, e in split_in_range:
            print(f"  {s:.1f}s - {e:.1f}s ({e - s:.1f}s)")

    # Per-window probability dump
    start_idx = max(0, int(start_t / window_duration))
    end_idx = min(len(probs), int(end_t / window_duration) + 1)

    print(f"\nPer-window probabilities [{start_t:.1f}s - {end_t:.1f}s]:")
    print(f"{'Window':>6} {'Time':>8} {'Prob':>6} {'Pred':>5} {'Bar'}")
    print("-" * 60)

    for i in range(start_idx, end_idx):
        t = i * window_duration
        p = probs[i]
        pred = result.window_predictions[i]
        bar_len = int(p * 40)
        bar = "#" * bar_len + "." * (40 - bar_len)

        # Highlight valleys
        marker = " "
        if p < args.valley_threshold:
            marker = "V"

        # Check if in any GT rally
        in_gt = any(s <= t <= e for s, e in gt_rallies)
        gt_marker = "GT" if in_gt else "  "

        print(f"{i:>6} {t:>7.1f}s {p:>5.3f} {'R' if pred == 1 else '-':>5} "
              f"|{bar}| {marker} {gt_marker}")

    # Summary stats for the range
    range_probs = probs[start_idx:end_idx]
    if len(range_probs) == 0:
        print("\nNo windows in specified range.")
    else:
        print(f"\nRange stats: mean={np.mean(range_probs):.3f}, "
              f"median={np.median(range_probs):.3f}, "
              f"min={np.min(range_probs):.3f}, max={np.max(range_probs):.3f}")

        valley_windows = int(np.sum(range_probs < args.valley_threshold))
        print(f"Windows below {args.valley_threshold}: {valley_windows}/{len(range_probs)} "
              f"({valley_windows / len(range_probs):.1%})")


if __name__ == "__main__":
    main()
