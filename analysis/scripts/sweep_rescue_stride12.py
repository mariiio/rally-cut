#!/usr/bin/env python3
"""Sweep rescue pass parameters for stride=12 using production model.

Quick sweep (seconds per config) to find optimal rescue params at stride=12.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from rallycut.evaluation.ground_truth import load_evaluation_videos
from rallycut.evaluation.matching import match_rallies
from rallycut.temporal.ball_features import (
    BALL_FEATURE_DIM,
    combine_features,
    extract_ball_features,
    load_ball_density,
)
from rallycut.temporal.features import FeatureCache
from rallycut.temporal.temporal_maxer.inference import TemporalMaxerInference

STRIDE = 12
MODEL_PATH = Path("weights/temporal_maxer/best_temporal_maxer.pt")
FEATURE_DIR = Path("training_data/features")
IOU_THRESHOLD = 0.4


def compute_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def run_sweep() -> None:
    videos = load_evaluation_videos()
    cache = FeatureCache(FEATURE_DIR)
    inference = TemporalMaxerInference(MODEL_PATH, device="mps")

    # Pre-load all features with ball feature handling
    expected_dim = inference.model.config.feature_dim
    ball_density_dir = Path("training_data/ball_density")
    video_data: list[tuple] = []
    for v in videos:
        cached = cache.get(v.content_hash, STRIDE)
        if cached is None:
            continue
        features, metadata = cached

        if expected_dim > features.shape[1]:
            ball_dim = expected_dim - features.shape[1]
            if ball_dim == BALL_FEATURE_DIM and ball_density_dir.exists():
                ball_data = load_ball_density(v.id, ball_density_dir)
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

        video_data.append((v, features, metadata))

    print(f"Loaded {len(video_data)} videos, {sum(len(v.ground_truth_rallies) for v, _, _ in video_data)} GT rallies\n")

    # Sweep configs
    configs = [
        # (rescue_min_avg_prob, rescue_min_windows, rescue_max_duration, label)
        (0, 3, 10.0, "no rescue"),
        (0.45, 3, 10.0, "current defaults (stride=24 tuned)"),
        (0.45, 5, 10.0, "prob=0.45, win=5"),
        (0.45, 6, 10.0, "prob=0.45, win=6"),
        (0.45, 8, 10.0, "prob=0.45, win=8"),
        (0.50, 4, 10.0, "prob=0.50, win=4"),
        (0.50, 5, 10.0, "prob=0.50, win=5"),
        (0.50, 6, 10.0, "prob=0.50, win=6"),
        (0.50, 8, 10.0, "prob=0.50, win=8"),
        (0.55, 4, 10.0, "prob=0.55, win=4"),
        (0.55, 5, 10.0, "prob=0.55, win=5"),
        (0.55, 6, 10.0, "prob=0.55, win=6"),
        (0.55, 8, 10.0, "prob=0.55, win=8"),
        (0.60, 4, 10.0, "prob=0.60, win=4"),
        (0.60, 5, 10.0, "prob=0.60, win=5"),
        (0.60, 6, 10.0, "prob=0.60, win=6"),
        (0.40, 5, 10.0, "prob=0.40, win=5"),
        (0.40, 6, 10.0, "prob=0.40, win=6"),
        (0.40, 8, 10.0, "prob=0.40, win=8"),
        (0.45, 6, 8.0, "prob=0.45, win=6, max=8s"),
        (0.50, 6, 8.0, "prob=0.50, win=6, max=8s"),
        (0.55, 5, 8.0, "prob=0.55, win=5, max=8s"),
    ]

    header = f"{'Config':<40} {'TP':>4} {'FP':>3} {'FN':>3} {'P':>6} {'R':>6} {'F1':>6}"
    print(header)
    print("-" * len(header))

    best_f1 = 0.0
    best_label = ""

    for rescue_prob, rescue_win, rescue_max, label in configs:
        total_tp = total_fp = total_fn = 0

        for v, features, metadata in video_data:
            result = inference.predict(
                features=features,
                fps=metadata.fps,
                stride=STRIDE,
                rescue_min_avg_prob=rescue_prob,
                rescue_min_windows=rescue_win,
                rescue_max_duration=rescue_max,
            )

            gt_segs = [(r.start_ms / 1000, r.end_ms / 1000) for r in v.ground_truth_rallies]
            mr = match_rallies(
                result.segments, gt_segs, iou_threshold=IOU_THRESHOLD,
            )
            total_tp += len(mr.matches)
            total_fp += len(mr.unmatched_predictions)
            total_fn += len(mr.unmatched_ground_truth)

        p, r, f1 = compute_f1(total_tp, total_fp, total_fn)
        marker = " <-- BEST" if f1 > best_f1 else ""
        if f1 > best_f1:
            best_f1 = f1
            best_label = label
        print(f"{label:<40} {total_tp:>4} {total_fp:>3} {total_fn:>3} {p:>5.1%} {r:>5.1%} {f1:>5.1%}{marker}")

    print(f"\nBest: {best_label} (F1={best_f1:.1%})")


if __name__ == "__main__":
    run_sweep()
