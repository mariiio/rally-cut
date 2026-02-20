#!/usr/bin/env python3
"""Analyze rally detection quality with detailed boundary error analysis.

Runs TemporalMaxer on all videos with rally GT labels and produces a detailed
report covering overall metrics, boundary error percentiles, per-video breakdown,
and individual TP/FP/FN details.

Usage:
    uv run python scripts/analyze_rally_detection.py
    uv run python scripts/analyze_rally_detection.py --iou 0.5
    uv run python scripts/analyze_rally_detection.py -o results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from rallycut.evaluation.ground_truth import load_evaluation_videos
from rallycut.evaluation.matching import RallyMatch, compute_iou, match_rallies
from rallycut.temporal.features import FeatureCache
from rallycut.temporal.temporal_maxer.inference import TemporalMaxerInference

if TYPE_CHECKING:
    from rallycut.evaluation.ground_truth import EvaluationVideo

logger = logging.getLogger(__name__)

STRIDE = 48


def _compute_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Compute precision, recall, F1 from counts."""
    p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1


def _edge_to_edge_gap(
    pred_start: float, pred_end: float, gt_start: float, gt_end: float
) -> float:
    """Compute edge-to-edge gap between two intervals (0 if overlapping)."""
    if pred_start > gt_end:
        return pred_start - gt_end
    if gt_start > pred_end:
        return gt_start - pred_end
    return 0.0


def run_inference_for_video(
    inference: TemporalMaxerInference,
    feature_cache: FeatureCache,
    video: EvaluationVideo,
) -> list[tuple[float, float]] | None:
    """Run TemporalMaxer inference on a single video's cached features."""
    cached_data = feature_cache.get(video.content_hash, STRIDE)
    if cached_data is None:
        return None

    features, metadata = cached_data
    result = inference.predict(features=features, fps=metadata.fps, stride=STRIDE)
    return result.segments


def print_overall(
    all_matches: list[RallyMatch],
    total_fp: int,
    total_fn: int,
    iou_threshold: float,
) -> None:
    """Print overall TP/FP/FN and P/R/F1."""
    tp = len(all_matches)
    p, r, f1 = _compute_f1(tp, total_fp, total_fn)

    print(f"\n{'=' * 70}")
    print(f"OVERALL (IoU >= {iou_threshold})")
    print(f"{'=' * 70}")
    print(f"  TP: {tp}   FP: {total_fp}   FN: {total_fn}")
    print(f"  Precision: {p:.1%}   Recall: {r:.1%}   F1: {f1:.1%}")


def print_boundary_errors(all_matches: list[RallyMatch]) -> None:
    """Print boundary error statistics with percentiles."""
    if not all_matches:
        print("\nNo matches to analyze boundary errors.")
        return

    start_errors = np.array([m.start_error_ms for m in all_matches])
    end_errors = np.array([m.end_error_ms for m in all_matches])

    print(f"\n{'=' * 70}")
    print("BOUNDARY ERRORS (ms, positive = late)")
    print(f"{'=' * 70}")

    for label, arr in [("Start", start_errors), ("End", end_errors)]:
        print(f"\n  {label} error:")
        print(f"    Mean:   {arr.mean():+.0f} ms")
        print(f"    Median: {np.median(arr):+.0f} ms")
        print(f"    Std:    {arr.std():.0f} ms")
        print(f"    P10:    {np.percentile(arr, 10):+.0f} ms")
        print(f"    P25:    {np.percentile(arr, 25):+.0f} ms")
        print(f"    P75:    {np.percentile(arr, 75):+.0f} ms")
        print(f"    P90:    {np.percentile(arr, 90):+.0f} ms")


def print_per_video_table(
    video_results: list[dict[str, Any]],
) -> None:
    """Print per-video summary table."""
    print(f"\n{'=' * 70}")
    print("PER-VIDEO RESULTS")
    print(f"{'=' * 70}")

    header = f"{'Video':<30} {'GT':>3} {'Pred':>4} {'TP':>3} {'FP':>3} {'FN':>3} {'F1':>6} {'MdStart':>8} {'MdEnd':>8}"
    print(header)
    print("-" * len(header))

    for vr in video_results:
        _, _, f1 = _compute_f1(vr["tp"], vr["fp"], vr["fn"])
        name = vr["filename"][:29]
        md_start = f"{vr['median_start_ms']:+.0f}" if vr["median_start_ms"] is not None else "-"
        md_end = f"{vr['median_end_ms']:+.0f}" if vr["median_end_ms"] is not None else "-"

        print(
            f"{name:<30} {vr['gt_count']:>3} {vr['pred_count']:>4} "
            f"{vr['tp']:>3} {vr['fp']:>3} {vr['fn']:>3} "
            f"{f1:>5.0%} {md_start:>8} {md_end:>8}"
        )


def print_tp_details(
    all_tp_details: list[dict[str, Any]],
) -> None:
    """Print individual TP match details."""
    if not all_tp_details:
        return

    print(f"\n{'=' * 70}")
    print(f"TP DETAILS ({len(all_tp_details)} matches)")
    print(f"{'=' * 70}")

    header = f"{'Video':<25} {'GT range':>15} {'Pred range':>15} {'StartErr':>9} {'EndErr':>9} {'IoU':>6}"
    print(header)
    print("-" * len(header))

    for d in all_tp_details:
        name = d["filename"][:24]
        gt_range = f"{d['gt_start']:.1f}-{d['gt_end']:.1f}s"
        pred_range = f"{d['pred_start']:.1f}-{d['pred_end']:.1f}s"
        print(
            f"{name:<25} {gt_range:>15} {pred_range:>15} "
            f"{d['start_error_ms']:>+8.0f} {d['end_error_ms']:>+8.0f} {d['iou']:>5.0%}"
        )


def print_fp_details(
    all_fp_details: list[dict[str, Any]],
) -> None:
    """Print individual FP details with distance to nearest GT."""
    if not all_fp_details:
        return

    print(f"\n{'=' * 70}")
    print(f"FP DETAILS ({len(all_fp_details)} false positives)")
    print(f"{'=' * 70}")

    header = f"{'Video':<25} {'Pred range':>15} {'Duration':>8} {'NearestGT gap':>14}"
    print(header)
    print("-" * len(header))

    for d in all_fp_details:
        name = d["filename"][:24]
        pred_range = f"{d['pred_start']:.1f}-{d['pred_end']:.1f}s"
        duration = f"{d['duration']:.1f}s"
        gap = f"{d['nearest_gt_gap']:.1f}s" if d["nearest_gt_gap"] is not None else "no GT"
        print(f"{name:<25} {pred_range:>15} {duration:>8} {gap:>14}")


def print_fn_details(
    all_fn_details: list[dict[str, Any]],
) -> None:
    """Print individual FN details with best IoU against any prediction."""
    if not all_fn_details:
        return

    print(f"\n{'=' * 70}")
    print(f"FN DETAILS ({len(all_fn_details)} missed rallies)")
    print(f"{'=' * 70}")

    header = f"{'Video':<25} {'GT range':>15} {'Duration':>8} {'Best IoU':>9}"
    print(header)
    print("-" * len(header))

    for d in all_fn_details:
        name = d["filename"][:24]
        gt_range = f"{d['gt_start']:.1f}-{d['gt_end']:.1f}s"
        duration = f"{d['duration']:.1f}s"
        best_iou = f"{d['best_iou']:.2f}" if d["best_iou"] > 0 else "0.00"
        print(f"{name:<25} {gt_range:>15} {duration:>8} {best_iou:>9}")


def analyze(
    iou_threshold: float = 0.4,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Run full analysis and print report."""
    # Load model
    model_path = Path("weights/temporal_maxer/best_temporal_maxer.pt")
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        sys.exit(1)

    inference = TemporalMaxerInference(model_path, device="cpu")
    feature_cache = FeatureCache(cache_dir=Path("training_data/features"))

    # Load GT videos
    print("Loading evaluation videos from database...")
    videos = load_evaluation_videos()
    if not videos:
        print("ERROR: No videos with ground truth found!")
        sys.exit(1)

    total_gt = sum(len(v.ground_truth_rallies) for v in videos)
    print(f"Found {len(videos)} videos with {total_gt} ground truth rallies")

    # Process each video
    all_matches: list[RallyMatch] = []
    total_fp = 0
    total_fn = 0
    video_results: list[dict[str, Any]] = []
    all_tp_details: list[dict[str, Any]] = []
    all_fp_details: list[dict[str, Any]] = []
    all_fn_details: list[dict[str, Any]] = []
    skipped = 0

    for video in videos:
        predictions = run_inference_for_video(inference, feature_cache, video)
        if predictions is None:
            print(f"  SKIP {video.filename}: no cached features")
            skipped += 1
            continue

        ground_truth = [(r.start_seconds, r.end_seconds) for r in video.ground_truth_rallies]
        matching = match_rallies(ground_truth, predictions, iou_threshold)

        # Collect matches
        all_matches.extend(matching.matches)
        total_fp += matching.false_positives
        total_fn += matching.false_negatives

        # Per-video summary
        start_errors = [m.start_error_ms for m in matching.matches]
        end_errors = [m.end_error_ms for m in matching.matches]
        video_results.append({
            "video_id": video.id,
            "filename": video.filename,
            "gt_count": len(ground_truth),
            "pred_count": len(predictions),
            "tp": matching.true_positives,
            "fp": matching.false_positives,
            "fn": matching.false_negatives,
            "median_start_ms": float(np.median(start_errors)) if start_errors else None,
            "median_end_ms": float(np.median(end_errors)) if end_errors else None,
        })

        # TP details
        for m in matching.matches:
            gt_s, gt_e = ground_truth[m.ground_truth_idx]
            pred_s, pred_e = predictions[m.predicted_idx]
            all_tp_details.append({
                "filename": video.filename,
                "gt_start": gt_s,
                "gt_end": gt_e,
                "pred_start": pred_s,
                "pred_end": pred_e,
                "start_error_ms": m.start_error_ms,
                "end_error_ms": m.end_error_ms,
                "iou": m.iou,
            })

        # FP details
        for pred_idx in matching.unmatched_predictions:
            pred_s, pred_e = predictions[pred_idx]
            nearest_gap: float | None = None
            for gt_s, gt_e in ground_truth:
                gap = _edge_to_edge_gap(pred_s, pred_e, gt_s, gt_e)
                if nearest_gap is None or gap < nearest_gap:
                    nearest_gap = gap
            all_fp_details.append({
                "filename": video.filename,
                "pred_start": pred_s,
                "pred_end": pred_e,
                "duration": pred_e - pred_s,
                "nearest_gt_gap": nearest_gap,
            })

        # FN details
        for gt_idx in matching.unmatched_ground_truth:
            gt_s, gt_e = ground_truth[gt_idx]
            best_iou = max(
                (compute_iou(gt_s, gt_e, pred_s, pred_e) for pred_s, pred_e in predictions),
                default=0.0,
            )
            all_fn_details.append({
                "filename": video.filename,
                "gt_start": gt_s,
                "gt_end": gt_e,
                "duration": gt_e - gt_s,
                "best_iou": best_iou,
            })

    if skipped:
        print(f"\nSkipped {skipped} videos (no cached features)")

    if not video_results:
        print("\nERROR: No videos had cached features!")
        sys.exit(1)

    # Print report
    print_overall(all_matches, total_fp, total_fn, iou_threshold)
    print_boundary_errors(all_matches)
    print_per_video_table(video_results)
    print_tp_details(all_tp_details)
    print_fp_details(all_fp_details)
    print_fn_details(all_fn_details)

    # Build export data
    export_data: dict[str, Any] = {
        "iou_threshold": iou_threshold,
        "num_videos": len(video_results),
        "tp": len(all_matches),
        "fp": total_fp,
        "fn": total_fn,
        "boundary_errors": {
            "start_mean_ms": float(np.mean([m.start_error_ms for m in all_matches])) if all_matches else None,
            "start_median_ms": float(np.median([m.start_error_ms for m in all_matches])) if all_matches else None,
            "end_mean_ms": float(np.mean([m.end_error_ms for m in all_matches])) if all_matches else None,
            "end_median_ms": float(np.median([m.end_error_ms for m in all_matches])) if all_matches else None,
        },
        "per_video": video_results,
        "tp_details": all_tp_details,
        "fp_details": all_fp_details,
        "fn_details": all_fn_details,
    }

    if output_path:
        output_path.write_text(json.dumps(export_data, indent=2))
        print(f"\nResults exported to: {output_path}")

    return export_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze rally detection quality")
    parser.add_argument("--iou", type=float, default=0.4, help="IoU threshold (default: 0.4)")
    parser.add_argument("-o", "--output", type=Path, help="Export results to JSON")
    parser.add_argument("-v", "--verbose", action="store_true", help="Enable debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    analyze(iou_threshold=args.iou, output_path=args.output)


if __name__ == "__main__":
    main()
