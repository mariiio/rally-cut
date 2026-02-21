#!/usr/bin/env python3
"""Leave-one-video-out cross-validation for TemporalMaxer rally detection.

For each of the N videos with GT and cached features, trains a TemporalMaxer
on the remaining N-1 videos, then evaluates on the held-out video. Aggregates
TP/FP/FN across all folds and reports P/R/F1 at IoU=0.4 and IoU=0.5.
"""

from __future__ import annotations

import logging
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

# Ensure analysis package is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from rallycut.evaluation.ground_truth import EvaluationVideo, load_evaluation_videos
from rallycut.evaluation.matching import MatchingResult, match_rallies
from rallycut.temporal.binary_head import generate_overlap_labels
from rallycut.temporal.features import FeatureCache
from rallycut.temporal.temporal_maxer.inference import TemporalMaxerInference
from rallycut.temporal.temporal_maxer.training import (
    TemporalMaxerTrainer,
    TemporalMaxerTrainingConfig,
)

# Suppress noisy training/library logs
logging.basicConfig(level=logging.WARNING)

DEFAULT_STRIDE = 24
FEATURE_DIR = Path("training_data/features")


@dataclass
class FPDetail:
    """Detail for a single false positive segment."""

    video: str
    start: float
    end: float
    duration: float
    avg_prob: float
    max_prob: float
    num_windows: int


@dataclass
class FoldResult:
    """Result for a single LOO fold."""

    video_id: str
    filename: str
    gt_count: int
    pred_count: int
    # IoU=0.4
    tp_04: int = 0
    fp_04: int = 0
    fn_04: int = 0
    # IoU=0.5
    tp_05: int = 0
    fp_05: int = 0
    fn_05: int = 0
    # Boundary errors (from IoU=0.4 matches)
    start_errors_ms: list[int] = field(default_factory=list)
    end_errors_ms: list[int] = field(default_factory=list)
    training_time_s: float = 0.0
    # FP details (from IoU=0.4)
    fp_details: list[FPDetail] = field(default_factory=list)


def auto_detect_device() -> str:
    """Auto-detect best available device: cuda > mps > cpu."""
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
    """Load features and generate labels for a single video.

    Returns (features, labels, fps, duration_seconds) or None if no cache.
    """
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


def run_fold(
    fold_idx: int,
    total_folds: int,
    held_out: EvaluationVideo,
    train_videos: list[EvaluationVideo],
    cache: FeatureCache,
    device: str,
    stride: int = DEFAULT_STRIDE,
    min_segment_confidence: float = 0.0,
) -> FoldResult:
    """Run a single LOO fold: train on N-1, evaluate on held-out."""
    print(
        f"Fold {fold_idx}/{total_folds}: training on {len(train_videos)} videos, "
        f"evaluating on {held_out.filename}..."
    )

    # Load training data
    train_features: list[np.ndarray] = []
    train_labels: list[np.ndarray] = []
    for video in train_videos:
        data = load_video_data(video, cache, stride)
        if data is not None:
            train_features.append(data[0])
            train_labels.append(data[1])

    # Load held-out data
    held_out_data = load_video_data(held_out, cache, stride)
    if held_out_data is None:
        print(f"  WARNING: No cached features for held-out video {held_out.filename}")
        return FoldResult(
            video_id=held_out.id,
            filename=held_out.filename,
            gt_count=len(held_out.ground_truth_rallies),
            pred_count=0,
        )

    ho_features, ho_labels, ho_fps, ho_duration = held_out_data

    # Train model in a temp directory
    config = TemporalMaxerTrainingConfig(
        learning_rate=5e-4,
        epochs=50,
        batch_size=4,
        patience=10,
        device=device,
    )

    trainer = TemporalMaxerTrainer(config=config)

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir)
        t0 = time.time()
        # Train with no validation set (LOO uses held-out for evaluation, not early stopping)
        # Pass empty val lists so trainer trains for all epochs
        trainer.train(
            train_features=train_features,
            train_labels=train_labels,
            val_features=[],
            val_labels=[],
            output_dir=output_dir,
        )
        train_time = time.time() - t0

        # Run inference on held-out video
        model_path = output_dir / "best_temporal_maxer.pt"
        inference = TemporalMaxerInference(model_path, device="cpu")
        result = inference.predict(
            features=ho_features,
            fps=ho_fps,
            stride=stride,
            min_segment_confidence=min_segment_confidence,
        )

    # Match predictions against GT
    gt_segments = [
        (r.start_seconds, r.end_seconds) for r in held_out.ground_truth_rallies
    ]
    pred_segments = result.segments
    window_duration = stride / ho_fps

    # Match at IoU=0.4
    match_04: MatchingResult = match_rallies(gt_segments, pred_segments, iou_threshold=0.4)
    # Match at IoU=0.5
    match_05: MatchingResult = match_rallies(gt_segments, pred_segments, iou_threshold=0.5)

    # Collect FP details with confidence
    fp_details: list[FPDetail] = []
    for pred_idx in match_04.unmatched_predictions:
        pred_s, pred_e = pred_segments[pred_idx]
        start_idx = int(pred_s / window_duration)
        end_idx = min(int(pred_e / window_duration) + 1, len(result.window_probs))
        if end_idx <= start_idx:
            end_idx = start_idx + 1
        seg_probs = result.window_probs[start_idx:min(end_idx, len(result.window_probs))]
        fp_details.append(FPDetail(
            video=held_out.filename,
            start=pred_s,
            end=pred_e,
            duration=pred_e - pred_s,
            avg_prob=float(np.mean(seg_probs)) if len(seg_probs) > 0 else 0.0,
            max_prob=float(np.max(seg_probs)) if len(seg_probs) > 0 else 0.0,
            num_windows=len(seg_probs),
        ))

    fold = FoldResult(
        video_id=held_out.id,
        filename=held_out.filename,
        gt_count=len(gt_segments),
        pred_count=len(pred_segments),
        tp_04=match_04.true_positives,
        fp_04=match_04.false_positives,
        fn_04=match_04.false_negatives,
        tp_05=match_05.true_positives,
        fp_05=match_05.false_positives,
        fn_05=match_05.false_negatives,
        start_errors_ms=[m.start_error_ms for m in match_04.matches],
        end_errors_ms=[m.end_error_ms for m in match_04.matches],
        training_time_s=train_time,
        fp_details=fp_details,
    )

    # Print fold summary
    f1_04 = (
        2 * fold.tp_04 / (2 * fold.tp_04 + fold.fp_04 + fold.fn_04)
        if (2 * fold.tp_04 + fold.fp_04 + fold.fn_04) > 0
        else 0.0
    )
    f1_05 = (
        2 * fold.tp_05 / (2 * fold.tp_05 + fold.fp_05 + fold.fn_05)
        if (2 * fold.tp_05 + fold.fp_05 + fold.fn_05) > 0
        else 0.0
    )
    print(
        f"  GT={fold.gt_count} Pred={fold.pred_count} | "
        f"IoU=0.4: TP={fold.tp_04} FP={fold.fp_04} FN={fold.fn_04} F1={f1_04:.1%} | "
        f"IoU=0.5: TP={fold.tp_05} FP={fold.fp_05} FN={fold.fn_05} F1={f1_05:.1%} | "
        f"train={train_time:.1f}s"
    )

    return fold


def compute_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Compute precision, recall, F1 from counts."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def main() -> None:
    """Run LOO cross-validation."""
    import argparse

    parser = argparse.ArgumentParser(description="LOO CV for TemporalMaxer")
    parser.add_argument(
        "--stride", type=int, default=DEFAULT_STRIDE,
        help=f"Feature stride (default: {DEFAULT_STRIDE})",
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.0,
        help="Min avg probability to keep a segment (0 = disabled)",
    )
    args = parser.parse_args()
    stride = args.stride
    min_confidence = args.min_confidence

    total_start = time.time()

    print("=" * 80)
    print("TemporalMaxer Leave-One-Video-Out Cross-Validation")
    print("=" * 80)
    print()

    device = auto_detect_device()
    print(f"Device: {device}")
    print(f"Feature stride: {stride}")
    if min_confidence > 0:
        print(f"Min segment confidence: {min_confidence}")
    print(f"Feature directory: {FEATURE_DIR}")
    print()

    # Load evaluation videos
    print("Loading evaluation videos from database...")
    videos = load_evaluation_videos()
    print(f"Found {len(videos)} videos with ground truth")

    # Filter to those with cached features
    cache = FeatureCache(cache_dir=FEATURE_DIR)
    videos_with_features = [v for v in videos if cache.has(v.content_hash, stride=stride)]
    print(
        f"Videos with cached features at stride={stride}: "
        f"{len(videos_with_features)}/{len(videos)}"
    )

    if len(videos_with_features) < 2:
        print("ERROR: Need at least 2 videos with cached features for LOO CV")
        sys.exit(1)

    videos = videos_with_features
    n = len(videos)
    total_gt = sum(len(v.ground_truth_rallies) for v in videos)
    print(f"Total ground truth rallies: {total_gt}")
    print()

    # Run LOO
    print(f"Running {n}-fold LOO cross-validation...")
    print("-" * 80)

    fold_results: list[FoldResult] = []
    for i, held_out in enumerate(videos):
        train_videos = [v for j, v in enumerate(videos) if j != i]
        fold = run_fold(
            i + 1, n, held_out, train_videos, cache, device, stride, min_confidence,
        )
        fold_results.append(fold)

    print("-" * 80)
    print()

    # Per-video table
    print("=" * 80)
    print("PER-VIDEO RESULTS")
    print("=" * 80)
    print()

    # IoU=0.4 table
    print("IoU=0.4:")
    header = f"{'Video':<30} {'GT':>4} {'Pred':>4} {'TP':>4} {'FP':>4} {'FN':>4} {'P':>6} {'R':>6} {'F1':>6}"
    print(header)
    print("-" * len(header))
    for fold in fold_results:
        p, r, f1 = compute_f1(fold.tp_04, fold.fp_04, fold.fn_04)
        name = fold.filename[:29]
        print(
            f"{name:<30} {fold.gt_count:>4} {fold.pred_count:>4} "
            f"{fold.tp_04:>4} {fold.fp_04:>4} {fold.fn_04:>4} "
            f"{p:>5.1%} {r:>5.1%} {f1:>5.1%}"
        )
    print()

    # IoU=0.5 table
    print("IoU=0.5:")
    print(header)
    print("-" * len(header))
    for fold in fold_results:
        p, r, f1 = compute_f1(fold.tp_05, fold.fp_05, fold.fn_05)
        name = fold.filename[:29]
        print(
            f"{name:<30} {fold.gt_count:>4} {fold.pred_count:>4} "
            f"{fold.tp_05:>4} {fold.fp_05:>4} {fold.fn_05:>4} "
            f"{p:>5.1%} {r:>5.1%} {f1:>5.1%}"
        )
    print()

    # Aggregate metrics
    print("=" * 80)
    print("AGGREGATE RESULTS")
    print("=" * 80)
    print()

    total_gt_count = sum(f.gt_count for f in fold_results)
    total_pred_count = sum(f.pred_count for f in fold_results)

    for iou_label, tp_attr, fp_attr, fn_attr in [
        ("IoU=0.4", "tp_04", "fp_04", "fn_04"),
        ("IoU=0.5", "tp_05", "fp_05", "fn_05"),
    ]:
        total_tp = sum(getattr(f, tp_attr) for f in fold_results)
        total_fp = sum(getattr(f, fp_attr) for f in fold_results)
        total_fn = sum(getattr(f, fn_attr) for f in fold_results)
        p, r, f1 = compute_f1(total_tp, total_fp, total_fn)

        print(f"{iou_label}:")
        print(f"  Total GT:  {total_gt_count}")
        print(f"  Total Pred: {total_pred_count}")
        print(f"  TP={total_tp}  FP={total_fp}  FN={total_fn}")
        print(f"  Precision: {p:.1%}")
        print(f"  Recall:    {r:.1%}")
        print(f"  F1:        {f1:.1%}")
        print()

    # FP detail report
    all_fp_details: list[FPDetail] = []
    for fold in fold_results:
        all_fp_details.extend(fold.fp_details)

    if all_fp_details:
        print("=" * 80)
        print(f"FP DETAILS ({len(all_fp_details)} false positives, IoU=0.4)")
        print("=" * 80)
        print()

        fp_avg_probs = np.array([fp.avg_prob for fp in all_fp_details])
        fp_durations = np.array([fp.duration for fp in all_fp_details])
        print(f"  avg_prob: mean={fp_avg_probs.mean():.3f}  median={np.median(fp_avg_probs):.3f}  "
              f"min={fp_avg_probs.min():.3f}  max={fp_avg_probs.max():.3f}")
        print(f"  duration: mean={fp_durations.mean():.1f}s  median={np.median(fp_durations):.1f}s  "
              f"min={fp_durations.min():.1f}s  max={fp_durations.max():.1f}s")
        print()

        header = f"  {'Video':<30} {'Range':>14} {'Dur':>5} {'AvgP':>5} {'MaxP':>5} {'Win':>4}"
        print(header)
        print("  " + "-" * (len(header) - 2))
        for fp in sorted(all_fp_details, key=lambda x: x.avg_prob):
            name = fp.video[:29]
            rng = f"{fp.start:.1f}-{fp.end:.1f}s"
            print(
                f"  {name:<30} {rng:>14} {fp.duration:>4.1f}s "
                f"{fp.avg_prob:>5.3f} {fp.max_prob:>5.3f} {fp.num_windows:>4}"
            )
        print()

    # Boundary error statistics (from IoU=0.4 matches)
    all_start_errors: list[int] = []
    all_end_errors: list[int] = []
    for fold in fold_results:
        all_start_errors.extend(fold.start_errors_ms)
        all_end_errors.extend(fold.end_errors_ms)

    if all_start_errors:
        print("=" * 80)
        print("BOUNDARY ERROR STATISTICS (from IoU=0.4 matches)")
        print("=" * 80)
        print()

        start_arr = np.array(all_start_errors)
        end_arr = np.array(all_end_errors)

        print(f"  N matches: {len(start_arr)}")
        print()
        print("  Start error (pred - GT, positive = late start):")
        print(f"    Median:  {np.median(start_arr):+.0f} ms")
        print(f"    Mean:    {np.mean(start_arr):+.0f} ms")
        print(f"    Abs median: {np.median(np.abs(start_arr)):.0f} ms")
        print(f"    Abs mean:   {np.mean(np.abs(start_arr)):.0f} ms")
        print(f"    P25/P75: {np.percentile(start_arr, 25):+.0f} / {np.percentile(start_arr, 75):+.0f} ms")
        print()
        print("  End error (pred - GT, positive = late end):")
        print(f"    Median:  {np.median(end_arr):+.0f} ms")
        print(f"    Mean:    {np.mean(end_arr):+.0f} ms")
        print(f"    Abs median: {np.median(np.abs(end_arr)):.0f} ms")
        print(f"    Abs mean:   {np.mean(np.abs(end_arr)):.0f} ms")
        print(f"    P25/P75: {np.percentile(end_arr, 25):+.0f} / {np.percentile(end_arr, 75):+.0f} ms")
        print()

    total_time = time.time() - total_start
    total_train_time = sum(f.training_time_s for f in fold_results)
    print(f"Total time: {total_time:.1f}s (training: {total_train_time:.1f}s)")


if __name__ == "__main__":
    main()
