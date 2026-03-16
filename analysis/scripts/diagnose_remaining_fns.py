#!/usr/bin/env python3
"""Diagnose false negatives from TemporalMaxer LOO CV with ball fusion.

Runs a full 41-fold LOO CV with ball features + 50% dropout (production config),
then for each FN prints detailed diagnostics: GT rally times, duration, model
probabilities, ball density. Groups FNs into categories and assesses recoverability.

Usage:
    PYTHONUNBUFFERED=1 uv run python scripts/diagnose_remaining_fns.py
"""

from __future__ import annotations

import logging
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from rallycut.evaluation.ground_truth import EvaluationVideo, load_evaluation_videos
from rallycut.evaluation.matching import MatchingResult, match_rallies
from rallycut.temporal.ball_features import (
    BALL_FEATURE_DIM,
    DEFAULT_BALL_DENSITY_DIR,
    WASB_CONF_THRESHOLD,
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


@dataclass
class FNDetail:
    """Detailed info about a single false negative."""

    video_filename: str
    video_id: str
    gt_start: float
    gt_end: float
    duration: float
    max_prob: float  # max model probability within GT segment
    avg_prob: float  # mean model probability within GT segment
    ball_density: float  # fraction of frames with ball detected in GT segment
    ball_available: bool  # whether ball density data exists for this video
    best_iou: float  # best IoU with any predicted segment
    best_pred_start: float | None  # start of closest prediction
    best_pred_end: float | None  # end of closest prediction
    category: str = ""


@dataclass
class FoldFNResult:
    """FN results from a single LOO fold."""

    video_id: str
    filename: str
    gt_count: int
    tp: int
    fp: int
    fn: int
    fn_details: list[FNDetail] = field(default_factory=list)


def auto_detect_device() -> str:
    """Auto-detect best available device."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_video_data(
    video: EvaluationVideo,
    cache: FeatureCache,
    ball_density_dir: Path | None = None,
    feature_dim: int = 768,
) -> tuple[np.ndarray, np.ndarray, float, float] | None:
    """Load features and labels for a single video. Returns (features, labels, fps, duration)."""
    cached_data = cache.get(video.content_hash, stride=STRIDE)
    if cached_data is None:
        return None

    features, metadata = cached_data

    if ball_density_dir is not None:
        from rallycut.temporal.ball_features import (
            combine_features,
            extract_ball_features,
        )

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
    return features, labels_arr, metadata.fps, metadata.duration_seconds


def compute_ball_density_for_segment(
    video_id: str,
    gt_start: float,
    gt_end: float,
    ball_density_dir: Path,
) -> tuple[float, bool]:
    """Compute fraction of frames with ball detected within a GT segment.

    Returns (density, data_available).
    """
    ball_data = load_ball_density(video_id, ball_density_dir)
    if ball_data is None:
        return 0.0, False

    confs, ball_fps = ball_data
    start_frame = int(gt_start * ball_fps)
    end_frame = min(int(gt_end * ball_fps), len(confs))

    if end_frame <= start_frame:
        return 0.0, True

    segment_confs = confs[start_frame:end_frame]
    detected = segment_confs >= WASB_CONF_THRESHOLD
    density = float(detected.mean())
    return density, True


def compute_best_iou_with_preds(
    gt_start: float,
    gt_end: float,
    pred_segments: list[tuple[float, float]],
) -> tuple[float, float | None, float | None]:
    """Find the best IoU between a GT segment and any prediction.

    Returns (best_iou, best_pred_start, best_pred_end).
    """
    from rallycut.evaluation.matching import compute_iou

    best_iou = 0.0
    best_ps: float | None = None
    best_pe: float | None = None

    for ps, pe in pred_segments:
        iou = compute_iou(gt_start, gt_end, ps, pe)
        if iou > best_iou:
            best_iou = iou
            best_ps = ps
            best_pe = pe

    # Also check overlap even if IoU is 0 (find nearest prediction)
    if best_iou == 0.0 and pred_segments:
        gt_mid = (gt_start + gt_end) / 2.0
        nearest_dist = float("inf")
        for ps, pe in pred_segments:
            pred_mid = (ps + pe) / 2.0
            dist = abs(pred_mid - gt_mid)
            if dist < nearest_dist:
                nearest_dist = dist
                best_ps = ps
                best_pe = pe

    return best_iou, best_ps, best_pe


def categorize_fn(fn: FNDetail) -> str:
    """Assign a category to a false negative."""
    if not fn.ball_available:
        if fn.duration < 5.0:
            return "short_no_ball_data"
        return "no_ball_data"

    if fn.ball_density < 0.10:
        return "ace_no_ball"
    elif fn.duration < 5.0:
        return "short_rally"
    elif fn.duration < 10.0:
        return "medium_rally"
    else:
        return "long_rally"


def main() -> None:
    """Run full LOO CV and diagnose all false negatives."""
    total_start = time.time()

    print("=" * 90)
    print("TemporalMaxer FN Diagnosis (LOO CV, ball features + 50% dropout)")
    print("=" * 90)
    print()

    device = auto_detect_device()
    print(f"Device: {device}")
    print(f"Stride: {STRIDE}")
    print()

    # Ball density setup
    ball_density_dir = DEFAULT_BALL_DENSITY_DIR
    if not ball_density_dir.exists():
        print(f"ERROR: Ball density dir not found at {ball_density_dir}")
        print("Run extract_ball_density.py first.")
        sys.exit(1)
    n_cached = len(list(ball_density_dir.glob("*.npz")))
    feature_dim = 768 + BALL_FEATURE_DIM
    print(f"Ball density files: {n_cached}")
    print(f"Feature dim: {feature_dim}")
    print()

    # Load evaluation videos
    print("Loading evaluation videos...")
    videos = load_evaluation_videos()
    cache = FeatureCache(cache_dir=FEATURE_DIR)
    videos = [v for v in videos if cache.has(v.content_hash, stride=STRIDE)]
    n = len(videos)
    total_gt = sum(len(v.ground_truth_rallies) for v in videos)
    print(f"Videos with features: {n}")
    print(f"Total GT rallies: {total_gt}")
    print()

    if n < 2:
        print("ERROR: Need at least 2 videos")
        sys.exit(1)

    # Augmentation config: ball feature dropout = 0.5 (production)
    aug_config = AugmentationConfig(
        enabled=True,
        ball_feature_dropout=0.5,
        ball_feature_dim=BALL_FEATURE_DIM,
    )

    # Run all 41 folds but only collect FN details
    print(f"Running {n}-fold LOO CV...")
    print("-" * 90)

    all_fold_results: list[FoldFNResult] = []
    all_fns: list[FNDetail] = []
    total_tp = 0
    total_fp = 0
    total_fn = 0

    for i, held_out in enumerate(videos):
        train_videos = [v for j, v in enumerate(videos) if j != i]

        print(
            f"[{i + 1}/{n}] {held_out.filename} "
            f"(GT={len(held_out.ground_truth_rallies)})...",
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
            print("SKIP (no features)")
            all_fold_results.append(FoldFNResult(
                video_id=held_out.id,
                filename=held_out.filename,
                gt_count=len(held_out.ground_truth_rallies),
                tp=0, fp=0, fn=len(held_out.ground_truth_rallies),
            ))
            total_fn += len(held_out.ground_truth_rallies)
            continue

        ho_features, ho_labels, ho_fps, ho_duration = held_out_data

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
            inference = TemporalMaxerInference(model_path, device="cpu")
            result = inference.predict(
                features=ho_features,
                fps=ho_fps,
                stride=STRIDE,
                min_segment_confidence=0.6,
                valley_threshold=0.5,
                min_valley_duration=2.0,
                rescue_min_avg_prob=0.50,
            )

        pred_segments = result.segments
        window_probs = result.window_probs
        window_duration = STRIDE / ho_fps

        # Match at IoU=0.4
        gt_segments = [
            (r.start_seconds, r.end_seconds) for r in held_out.ground_truth_rallies
        ]
        match_result: MatchingResult = match_rallies(gt_segments, pred_segments, iou_threshold=0.4)

        fold_tp = match_result.true_positives
        fold_fp = match_result.false_positives
        fold_fn = match_result.false_negatives
        total_tp += fold_tp
        total_fp += fold_fp
        total_fn += fold_fn

        # Collect FN details
        fold_fn_details: list[FNDetail] = []
        for gt_idx in match_result.unmatched_ground_truth:
            gt_s, gt_e = gt_segments[gt_idx]
            duration = gt_e - gt_s

            # Model probabilities within GT segment
            s_idx = max(0, int(gt_s / window_duration))
            e_idx = min(int(gt_e / window_duration) + 1, len(window_probs))
            if e_idx <= s_idx:
                e_idx = s_idx + 1
            seg_probs = window_probs[s_idx:min(e_idx, len(window_probs))]
            max_prob = float(np.max(seg_probs)) if len(seg_probs) > 0 else 0.0
            avg_prob = float(np.mean(seg_probs)) if len(seg_probs) > 0 else 0.0

            # Ball density
            ball_density, ball_available = compute_ball_density_for_segment(
                held_out.id, gt_s, gt_e, ball_density_dir,
            )

            # Best IoU with predictions
            best_iou, best_ps, best_pe = compute_best_iou_with_preds(
                gt_s, gt_e, pred_segments,
            )

            fn_detail = FNDetail(
                video_filename=held_out.filename,
                video_id=held_out.id,
                gt_start=gt_s,
                gt_end=gt_e,
                duration=duration,
                max_prob=max_prob,
                avg_prob=avg_prob,
                ball_density=ball_density,
                ball_available=ball_available,
                best_iou=best_iou,
                best_pred_start=best_ps,
                best_pred_end=best_pe,
            )
            fn_detail.category = categorize_fn(fn_detail)
            fold_fn_details.append(fn_detail)
            all_fns.append(fn_detail)

        fold_result = FoldFNResult(
            video_id=held_out.id,
            filename=held_out.filename,
            gt_count=len(gt_segments),
            tp=fold_tp,
            fp=fold_fp,
            fn=fold_fn,
            fn_details=fold_fn_details,
        )
        all_fold_results.append(fold_result)

        fn_str = f"FN={fold_fn}" if fold_fn > 0 else "FN=0"
        print(f"TP={fold_tp} FP={fold_fp} {fn_str}")

    print("-" * 90)
    print()

    # Aggregate summary
    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    print(f"Aggregate (IoU=0.4): TP={total_tp} FP={total_fp} FN={total_fn}")
    print(f"  Precision={p:.1%}  Recall={r:.1%}  F1={f1:.1%}")
    print()

    # Per-video FN counts (only videos with FNs)
    print("=" * 90)
    print("PER-VIDEO FN COUNTS")
    print("=" * 90)
    print()
    header = f"{'Video':<35} {'GT':>4} {'TP':>4} {'FP':>4} {'FN':>4}"
    print(header)
    print("-" * len(header))
    for fold in sorted(all_fold_results, key=lambda f: f.fn, reverse=True):
        if fold.fn > 0:
            print(
                f"{fold.filename[:34]:<35} {fold.gt_count:>4} "
                f"{fold.tp:>4} {fold.fp:>4} {fold.fn:>4}"
            )
    videos_with_fn = sum(1 for f in all_fold_results if f.fn > 0)
    print(f"\nVideos with FN: {videos_with_fn}/{n}")
    print()

    # Detailed FN list
    print("=" * 90)
    print(f"ALL FALSE NEGATIVES ({len(all_fns)} total)")
    print("=" * 90)
    print()
    header = (
        f"{'#':>3} {'Video':<25} {'GT Range':>14} {'Dur':>5} "
        f"{'AvgP':>5} {'MaxP':>5} {'BallD':>5} {'BestIoU':>7} {'Category':<20}"
    )
    print(header)
    print("-" * len(header))

    for idx, fn in enumerate(
        sorted(all_fns, key=lambda x: x.max_prob, reverse=True), 1
    ):
        rng = f"{fn.gt_start:.1f}-{fn.gt_end:.1f}s"
        ball_str = f"{fn.ball_density:.2f}" if fn.ball_available else "N/A"
        print(
            f"{idx:>3} {fn.video_filename[:24]:<25} {rng:>14} {fn.duration:>4.1f}s "
            f"{fn.avg_prob:>5.3f} {fn.max_prob:>5.3f} {ball_str:>5} "
            f"{fn.best_iou:>7.3f} {fn.category:<20}"
        )

        # If there's a nearby prediction, show it
        if fn.best_pred_start is not None and fn.best_iou > 0:
            pred_rng = f"{fn.best_pred_start:.1f}-{fn.best_pred_end:.1f}s"
            print(f"    ^-- nearest pred: {pred_rng} (IoU={fn.best_iou:.3f})")

    print()

    # Category summary
    print("=" * 90)
    print("FN CATEGORY SUMMARY")
    print("=" * 90)
    print()

    categories: dict[str, list[FNDetail]] = {}
    for fn in all_fns:
        categories.setdefault(fn.category, []).append(fn)

    cat_descriptions = {
        "ace_no_ball": "Ace / no ball visible (density < 0.10)",
        "short_rally": "Short rally, <5s",
        "medium_rally": "Medium rally, 5-10s",
        "long_rally": "Long rally, >10s",
        "short_no_ball_data": "Short, no ball density data",
        "no_ball_data": "No ball density data available",
    }

    for cat in ["ace_no_ball", "short_rally", "medium_rally", "long_rally",
                "short_no_ball_data", "no_ball_data"]:
        fns = categories.get(cat, [])
        if not fns:
            continue
        desc = cat_descriptions.get(cat, cat)
        avg_max_p = np.mean([fn.max_prob for fn in fns])
        avg_avg_p = np.mean([fn.avg_prob for fn in fns])
        avg_dur = np.mean([fn.duration for fn in fns])
        print(f"  {desc}: {len(fns)} FNs")
        print(f"    avg duration: {avg_dur:.1f}s")
        print(f"    avg max_prob: {avg_max_p:.3f}, avg avg_prob: {avg_avg_p:.3f}")
        for fn in fns:
            ball_str = f"ball={fn.ball_density:.2f}" if fn.ball_available else "no_ball_data"
            print(
                f"      {fn.video_filename[:24]:<25} "
                f"{fn.gt_start:.1f}-{fn.gt_end:.1f}s ({fn.duration:.1f}s) "
                f"max_p={fn.max_prob:.3f} {ball_str}"
            )
        print()

    # Recoverability analysis
    print("=" * 90)
    print("RECOVERABILITY ANALYSIS")
    print("=" * 90)
    print()

    # Group by recoverability
    recoverable: list[FNDetail] = []
    marginal: list[FNDetail] = []
    hard: list[FNDetail] = []

    for fn in all_fns:
        if fn.max_prob >= 0.50:
            # Model sees something -- potentially recoverable with threshold tuning
            recoverable.append(fn)
        elif fn.max_prob >= 0.30:
            # Weak signal -- might recover with architecture changes
            marginal.append(fn)
        else:
            # Model sees nothing -- hard cases
            hard.append(fn)

    print(f"  Recoverable (max_prob >= 0.50): {len(recoverable)} FNs")
    print("    These have model signal above rescue threshold.")
    print("    Potential fix: lower rescue_min_avg_prob or min_segment_confidence.")
    for fn in recoverable:
        print(
            f"      {fn.video_filename[:24]:<25} "
            f"{fn.duration:.1f}s max_p={fn.max_prob:.3f} avg_p={fn.avg_prob:.3f} "
            f"cat={fn.category}"
        )
    print()

    print(f"  Marginal (0.30 <= max_prob < 0.50): {len(marginal)} FNs")
    print("    Weak signal present but below current thresholds.")
    print("    Potential fix: more training data, feature engineering, lower thresholds (risk FP).")
    for fn in marginal:
        print(
            f"      {fn.video_filename[:24]:<25} "
            f"{fn.duration:.1f}s max_p={fn.max_prob:.3f} avg_p={fn.avg_prob:.3f} "
            f"cat={fn.category}"
        )
    print()

    print(f"  Hard (max_prob < 0.30): {len(hard)} FNs")
    print("    Model produces near-zero probability. Likely fundamentally different from training.")
    print("    Potential fix: label more similar videos, augmentation, or accept as detection limit.")
    for fn in hard:
        ball_str = f"ball={fn.ball_density:.2f}" if fn.ball_available else "no_ball_data"
        print(
            f"      {fn.video_filename[:24]:<25} "
            f"{fn.duration:.1f}s max_p={fn.max_prob:.3f} {ball_str} "
            f"cat={fn.category}"
        )
    print()

    # Final summary
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print()
    print(f"  Total FNs: {len(all_fns)}")
    print(f"  Recoverable (threshold tuning): {len(recoverable)}")
    print(f"  Marginal (needs model improvement): {len(marginal)}")
    print(f"  Hard (fundamental limit): {len(hard)}")
    ace_count = len(categories.get("ace_no_ball", []))
    if ace_count > 0:
        print(f"  Aces / no-ball: {ace_count} ({ace_count / len(all_fns) * 100:.0f}% of FNs)")
    print()
    theoretical_best_f1 = (
        2 * (total_tp + len(recoverable))
        / (2 * (total_tp + len(recoverable)) + total_fp + len(marginal) + len(hard))
    )
    print(f"  Current F1: {f1:.1%}")
    print(f"  If all recoverable FNs fixed: F1~{theoretical_best_f1:.1%}")
    print()

    elapsed = time.time() - total_start
    print(f"Total time: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
