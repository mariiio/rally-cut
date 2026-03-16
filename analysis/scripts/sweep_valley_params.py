#!/usr/bin/env python3
"""Sweep valley splitting parameters on videos with over-merged FNs.

Only runs LOO folds for the ~11 videos that had FNs in the diagnostic run,
but trains on all other videos. Tests combinations of valley_threshold,
min_valley_duration, and max_gap_duration to find the best anti-overmerge config.

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/sweep_valley_params.py
"""

from __future__ import annotations

import logging
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from rallycut.evaluation.ground_truth import EvaluationVideo, load_evaluation_videos
from rallycut.evaluation.matching import MatchingResult, match_rallies
from rallycut.temporal.ball_features import (
    BALL_FEATURE_DIM,
    DEFAULT_BALL_DENSITY_DIR,
    combine_features,
    extract_ball_features,
    load_ball_density,
)
from rallycut.temporal.features import FeatureCache, generate_overlap_labels
from rallycut.temporal.temporal_maxer.inference import TemporalMaxerInference
from rallycut.temporal.temporal_maxer.model import TemporalMaxerConfig
from rallycut.temporal.temporal_maxer.training import (
    AugmentationConfig,
    TemporalMaxerTrainer,
    TemporalMaxerTrainingConfig,
)

logging.basicConfig(level=logging.WARNING)

STRIDE = 12
FEATURE_DIR = Path("training_data/features")

# Videos that had FNs in the diagnostic run
FN_VIDEO_PREFIXES = [
    "meme", "vuvu", "IMG_1820", "IMG_1887", "IMG_2313",
    "vovo", "muchi", "mame", "matchope", "mech", "macho",
]

# Parameter grid
VALLEY_THRESHOLDS = [0.4, 0.5, 0.6]
MIN_VALLEY_DURATIONS = [1.0, 1.5, 2.0]
MAX_GAP_DURATIONS = [2.0, 3.0]


@dataclass
class ParamConfig:
    valley_threshold: float
    min_valley_duration: float
    max_gap_duration: float


@dataclass
class SweepResult:
    config: ParamConfig
    tp: int
    fp: int
    fn: int

    @property
    def f1(self) -> float:
        p = self.tp / (self.tp + self.fp) if (self.tp + self.fp) > 0 else 0.0
        r = self.tp / (self.tp + self.fn) if (self.tp + self.fn) > 0 else 0.0
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


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
    ball_density_dir: Path,
    feature_dim: int = 773,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Load features and labels for a single video."""
    cached_data = cache.get(video.content_hash, stride=STRIDE)
    if cached_data is None:
        return None

    features, metadata = cached_data

    ball_data = load_ball_density(video.id, ball_density_dir)
    if ball_data is not None:
        confs, ball_fps = ball_data
        ball_feats = extract_ball_features(
            confs, ball_fps,
            feature_fps=metadata.fps, stride=STRIDE,
        )
        features = combine_features(features, ball_feats)

    # Zero-pad if ball density not available (modality dropout handles this)
    if features.shape[1] < feature_dim:
        pad_width = feature_dim - features.shape[1]
        padding = np.zeros((features.shape[0], pad_width), dtype=features.dtype)
        features = np.concatenate([features, padding], axis=1)

    duration_ms = int(metadata.duration_seconds * 1000)
    labels = generate_overlap_labels(
        rallies=video.ground_truth_rallies,
        video_duration_ms=duration_ms,
        fps=metadata.fps,
        stride=STRIDE,
    )
    min_len = min(len(features), len(labels))
    features = features[:min_len]
    labels_arr = np.array(labels[:min_len], dtype=np.float32)
    return features, labels_arr, metadata.fps


def main() -> None:
    total_start = time.time()

    print("=" * 80)
    print("Valley Parameter Sweep (FN videos only)")
    print("=" * 80)
    print()

    device = auto_detect_device()
    ball_density_dir = DEFAULT_BALL_DENSITY_DIR
    feature_dim = 768 + BALL_FEATURE_DIM

    videos = load_evaluation_videos()
    cache = FeatureCache(cache_dir=FEATURE_DIR)
    videos = [v for v in videos if cache.has(v.content_hash, stride=STRIDE)]
    n = len(videos)

    # Identify FN videos
    fn_indices: list[int] = []
    for i, v in enumerate(videos):
        for prefix in FN_VIDEO_PREFIXES:
            if prefix in v.filename:
                fn_indices.append(i)
                break

    print(f"Total videos: {n}")
    print(f"FN videos to evaluate: {len(fn_indices)}")
    print(f"Parameter configs: {len(VALLEY_THRESHOLDS) * len(MIN_VALLEY_DURATIONS) * len(MAX_GAP_DURATIONS)}")
    print()

    aug_config = AugmentationConfig(
        enabled=True,
        ball_feature_dropout=0.5,
        ball_feature_dim=BALL_FEATURE_DIM,
    )

    # Train models for each FN video fold (train once, evaluate with multiple param configs)
    # Store: fold_idx -> (inference, features, fps, gt_segments)
    fold_data: dict[int, tuple[TemporalMaxerInference, np.ndarray, float, list[tuple[float, float]]]] = {}

    print(f"Training {len(fn_indices)} fold models...")
    print("-" * 80)

    for fold_num, i in enumerate(fn_indices):
        held_out = videos[i]
        train_videos = [v for j, v in enumerate(videos) if j != i]

        print(
            f"[{fold_num + 1}/{len(fn_indices)}] {held_out.filename}...",
            end=" ",
            flush=True,
        )

        # Load training data
        train_features: list[np.ndarray] = []
        train_labels: list[np.ndarray] = []
        for video in train_videos:
            data = load_video_data(video, cache, ball_density_dir, feature_dim)
            if data is not None:
                train_features.append(data[0])
                train_labels.append(data[1])

        # Load held-out data
        held_out_data = load_video_data(held_out, cache, ball_density_dir, feature_dim)
        if held_out_data is None:
            print("SKIP")
            continue

        ho_features, ho_labels, ho_fps = held_out_data
        gt_segments = [
            (r.start_seconds, r.end_seconds) for r in held_out.ground_truth_rallies
        ]

        # Train
        with tempfile.TemporaryDirectory() as tmpdir:
            config = TemporalMaxerTrainingConfig(
                model_config=TemporalMaxerConfig(feature_dim=feature_dim),
                learning_rate=5e-4, epochs=50, batch_size=4, patience=15,
                device=device, seed=42,
                augmentation=aug_config,
            )
            trainer = TemporalMaxerTrainer(config=config)
            output_dir = Path(tmpdir)
            trainer.train(train_features, train_labels, [], [], output_dir)

            model_path = output_dir / "best_temporal_maxer.pt"
            # Load to CPU so we can reuse across param configs
            inference = TemporalMaxerInference(model_path, device="cpu")

        fold_data[i] = (inference, ho_features, ho_fps, gt_segments)
        print(f"done (GT={len(gt_segments)})")

    train_time = time.time() - total_start
    print(f"\nTraining complete in {train_time:.0f}s")
    print()

    # Now sweep parameters (fast — just re-runs inference post-processing)
    print("=" * 80)
    print("Sweeping parameters...")
    print("=" * 80)
    print()

    configs: list[ParamConfig] = []
    for vt in VALLEY_THRESHOLDS:
        for mvd in MIN_VALLEY_DURATIONS:
            for mgd in MAX_GAP_DURATIONS:
                configs.append(ParamConfig(vt, mvd, mgd))

    results: list[SweepResult] = []

    for cfg in configs:
        total_tp = 0
        total_fp = 0
        total_fn = 0

        for i, (inference, ho_features, ho_fps, gt_segments) in fold_data.items():
            result = inference.predict(
                features=ho_features,
                fps=ho_fps,
                stride=STRIDE,
                min_segment_confidence=0.6,
                valley_threshold=cfg.valley_threshold,
                min_valley_duration=cfg.min_valley_duration,
                max_gap_duration=cfg.max_gap_duration,
                rescue_min_avg_prob=0.50,
            )

            match_result: MatchingResult = match_rallies(
                gt_segments, result.segments, iou_threshold=0.4,
            )
            total_tp += match_result.true_positives
            total_fp += match_result.false_positives
            total_fn += match_result.false_negatives

        sr = SweepResult(cfg, total_tp, total_fp, total_fn)
        results.append(sr)

    # Sort by F1
    results.sort(key=lambda r: r.f1, reverse=True)

    # Print results table
    header = (
        f"{'#':>3} {'valley_th':>10} {'min_valley':>11} {'max_gap':>8} "
        f"{'TP':>4} {'FP':>4} {'FN':>4} {'F1':>7}"
    )
    print(header)
    print("-" * len(header))

    # Current production config for reference
    prod = ParamConfig(0.5, 2.0, 3.0)

    for idx, sr in enumerate(results, 1):
        is_prod = (
            sr.config.valley_threshold == prod.valley_threshold
            and sr.config.min_valley_duration == prod.min_valley_duration
            and sr.config.max_gap_duration == prod.max_gap_duration
        )
        marker = " <-- current" if is_prod else ""
        print(
            f"{idx:>3} {sr.config.valley_threshold:>10.1f} "
            f"{sr.config.min_valley_duration:>11.1f} {sr.config.max_gap_duration:>8.1f} "
            f"{sr.tp:>4} {sr.fp:>4} {sr.fn:>4} {sr.f1:>6.1%}{marker}"
        )

    # Best vs current
    print()
    best = results[0]
    current = next(
        (r for r in results
         if r.config.valley_threshold == prod.valley_threshold
         and r.config.min_valley_duration == prod.min_valley_duration
         and r.config.max_gap_duration == prod.max_gap_duration),
        None,
    )
    if current:
        print(f"Current: valley={prod.valley_threshold} min_valley={prod.min_valley_duration} "
              f"max_gap={prod.max_gap_duration} → TP={current.tp} FP={current.fp} FN={current.fn} "
              f"F1={current.f1:.1%}")
    print(f"Best:    valley={best.config.valley_threshold} "
          f"min_valley={best.config.min_valley_duration} "
          f"max_gap={best.config.max_gap_duration} → TP={best.tp} FP={best.fp} FN={best.fn} "
          f"F1={best.f1:.1%}")

    if current and best.f1 > current.f1:
        delta_tp = best.tp - current.tp
        delta_fp = best.fp - current.fp
        delta_fn = best.fn - current.fn
        print(f"Delta:   TP{delta_tp:+d} FP{delta_fp:+d} FN{delta_fn:+d} "
              f"F1{best.f1 - current.f1:+.1%}")

    elapsed = time.time() - total_start
    print(f"\nTotal time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
