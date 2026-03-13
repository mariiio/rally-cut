#!/usr/bin/env python3
"""Evaluate ball trajectory fusion potential for rally detection.

Analyzes whether ball detection density can recover FN short rallies that
TemporalMaxer misses, and whether it would introduce new FPs.

Usage:
    uv run python scripts/eval_ball_fusion.py
    uv run python scripts/eval_ball_fusion.py --density-threshold 0.3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from rallycut.evaluation.ground_truth import load_evaluation_videos
from rallycut.evaluation.matching import match_rallies
from rallycut.temporal.features import FeatureCache
from rallycut.temporal.temporal_maxer.inference import TemporalMaxerInference
from scripts.extract_ball_density import compute_density_curve, load_density

STRIDE = 12
MODEL_PATH = Path("weights/temporal_maxer/best_temporal_maxer.pt")
FEATURE_DIR = Path("training_data/features")
IOU_THRESHOLD = 0.4


def fmt_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--density-threshold", type=float, default=0.3,
                        help="Min ball detection density to consider (default: 0.3)")
    parser.add_argument("--min-duration", type=float, default=2.0,
                        help="Min duration (seconds) of ball-dense region (default: 2.0)")
    parser.add_argument("--max-duration", type=float, default=12.0,
                        help="Max duration for ball-detected rally (default: 12.0)")
    parser.add_argument("--window", type=float, default=2.0,
                        help="Density computation window in seconds (default: 2.0)")
    parser.add_argument("--conf-threshold", type=float, default=0.3,
                        help="WASB confidence threshold for detection (default: 0.3)")
    args = parser.parse_args()

    videos = load_evaluation_videos()
    cache = FeatureCache(FEATURE_DIR)
    inference = TemporalMaxerInference(MODEL_PATH, device="mps")

    # Collect FN and FP analysis
    fn_with_ball: list[dict] = []
    fn_without_ball: list[dict] = []
    potential_new_fps: list[dict] = []
    videos_missing_density = []
    total_tp = total_fp = total_fn = 0
    recoverable_fn = 0

    for v in videos:
        cached = cache.get(v.content_hash, STRIDE)
        if cached is None:
            continue

        features, metadata = cached
        result = inference.predict(features=features, fps=metadata.fps, stride=STRIDE)
        gt_segs = [(r.start_ms / 1000, r.end_ms / 1000) for r in v.ground_truth_rallies]

        mr = match_rallies(result.segments, gt_segs, iou_threshold=IOU_THRESHOLD)
        total_tp += len(mr.matches)
        total_fp += len(mr.unmatched_predictions)
        total_fn += len(mr.unmatched_ground_truth)

        # Load ball density
        density_data = load_density(v.id)
        if density_data is None:
            if mr.unmatched_ground_truth:
                videos_missing_density.append((v.filename, len(mr.unmatched_ground_truth)))
            continue

        density = compute_density_curve(
            density_data["confidences"], density_data["fps"],
            window_s=args.window, conf_threshold=args.conf_threshold,
        )
        video_duration = density_data["n_frames"] / density_data["fps"]

        # Analyze each FN: does ball density cover it?
        for fn_idx in mr.unmatched_ground_truth:
            gt_start, gt_end = gt_segs[fn_idx]
            gt_dur = gt_end - gt_start

            # Get density in GT rally window
            s_idx = max(0, int(gt_start))
            e_idx = min(len(density), int(gt_end) + 1)
            if e_idx <= s_idx:
                fn_without_ball.append({
                    "video": v.filename, "start": gt_start, "end": gt_end,
                    "duration": gt_dur, "density": 0.0, "reason": "out of range",
                })
                continue

            rally_density = density[s_idx:e_idx]
            avg_density = float(rally_density.mean())
            max_density = float(rally_density.max())

            # Check if there's a contiguous dense region
            dense_seconds = int((rally_density >= args.density_threshold).sum())

            info = {
                "video": v.filename, "start": gt_start, "end": gt_end,
                "duration": gt_dur, "avg_density": avg_density,
                "max_density": max_density, "dense_seconds": dense_seconds,
            }

            if dense_seconds >= args.min_duration and gt_dur <= args.max_duration:
                fn_with_ball.append(info)
                recoverable_fn += 1
            else:
                info["reason"] = (
                    f"dense={dense_seconds}s<{args.min_duration}s"
                    if dense_seconds < args.min_duration
                    else f"dur={gt_dur:.1f}s>{args.max_duration}s"
                )
                fn_without_ball.append(info)

        # Check gaps between predictions for potential new FPs
        # (ball-dense regions in gaps that are NOT matched to any GT)
        pred_segs = sorted(result.segments)
        gap_regions: list[tuple[float, float]] = []
        prev_end = 0.0
        for s, e in pred_segs:
            if s > prev_end:
                gap_regions.append((prev_end, s))
            prev_end = max(prev_end, e)
        if prev_end < video_duration:
            gap_regions.append((prev_end, video_duration))

        for gap_s, gap_e in gap_regions:
            s_idx = max(0, int(gap_s))
            e_idx = min(len(density), int(gap_e) + 1)
            if e_idx <= s_idx:
                continue

            gap_density = density[s_idx:e_idx]

            # Find contiguous dense runs in this gap
            in_run = False
            run_start = 0
            for j in range(len(gap_density)):
                if gap_density[j] >= args.density_threshold and not in_run:
                    run_start = j
                    in_run = True
                elif gap_density[j] < args.density_threshold and in_run:
                    run_len = j - run_start
                    if run_len >= args.min_duration:
                        cand_s = s_idx + run_start
                        cand_e = s_idx + j
                        cand_dur = cand_e - cand_s
                        if cand_dur <= args.max_duration:
                            # Check if this overlaps any GT
                            overlaps_gt = any(
                                compute_overlap(cand_s, cand_e, gs, ge) > 0
                                for gs, ge in gt_segs
                            )
                            if not overlaps_gt:
                                cand_density = float(gap_density[run_start:j].mean())
                                potential_new_fps.append({
                                    "video": v.filename,
                                    "start": float(cand_s), "end": float(cand_e),
                                    "duration": cand_dur, "avg_density": cand_density,
                                })
                    in_run = False
            if in_run:
                run_len = len(gap_density) - run_start
                if run_len >= args.min_duration:
                    cand_s = s_idx + run_start
                    cand_e = s_idx + len(gap_density)
                    cand_dur = cand_e - cand_s
                    if cand_dur <= args.max_duration:
                        overlaps_gt = any(
                            compute_overlap(cand_s, cand_e, gs, ge) > 0
                            for gs, ge in gt_segs
                        )
                        if not overlaps_gt:
                            cand_density = float(gap_density[run_start:].mean())
                            potential_new_fps.append({
                                "video": v.filename,
                                "start": float(cand_s), "end": float(cand_e),
                                "duration": cand_dur, "avg_density": cand_density,
                            })

    # Report
    print("=" * 70)
    print("BALL TRAJECTORY FUSION ANALYSIS")
    print(f"  Density threshold: {args.density_threshold}, min duration: {args.min_duration}s")
    print(f"  Window: {args.window}s, WASB conf: {args.conf_threshold}")
    print("=" * 70)

    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print(f"\nBaseline (production model, no fusion):")
    print(f"  TP={total_tp} FP={total_fp} FN={total_fn} P={p:.1%} R={r:.1%} F1={f1:.1%}")

    if videos_missing_density:
        print(f"\nMissing ball density ({len(videos_missing_density)} videos):")
        for name, n_fn in videos_missing_density:
            print(f"  {name}: {n_fn} FN without density data")

    print(f"\n{'=' * 70}")
    print(f"FN RECOVERABLE BY BALL DENSITY ({len(fn_with_ball)}/{total_fn})")
    print(f"{'=' * 70}")
    if fn_with_ball:
        for fn in sorted(fn_with_ball, key=lambda x: -x["avg_density"]):
            print(
                f"  {fn['video']:<30} {fmt_time(fn['start'])}-{fmt_time(fn['end'])} "
                f"({fn['duration']:.1f}s) avg_density={fn['avg_density']:.2f} "
                f"max={fn['max_density']:.2f} dense={fn['dense_seconds']}s"
            )
    else:
        print("  None")

    print(f"\n{'=' * 70}")
    print(f"FN NOT RECOVERABLE ({len(fn_without_ball)}/{total_fn})")
    print(f"{'=' * 70}")
    for fn in sorted(fn_without_ball, key=lambda x: x["video"]):
        print(
            f"  {fn['video']:<30} {fmt_time(fn.get('start', 0))}-{fmt_time(fn.get('end', 0))} "
            f"({fn.get('duration', 0):.1f}s) "
            f"avg={fn.get('avg_density', 0):.2f} reason={fn.get('reason', 'low density')}"
        )

    print(f"\n{'=' * 70}")
    print(f"POTENTIAL NEW FPs ({len(potential_new_fps)})")
    print(f"{'=' * 70}")
    if potential_new_fps:
        for fp in sorted(potential_new_fps, key=lambda x: -x["avg_density"]):
            print(
                f"  {fp['video']:<30} {fmt_time(fp['start'])}-{fmt_time(fp['end'])} "
                f"({fp['duration']}s) avg_density={fp['avg_density']:.2f}"
            )
    else:
        print("  None")

    # Projected results
    new_tp = total_tp + recoverable_fn
    new_fn = total_fn - recoverable_fn
    new_fp = total_fp + len(potential_new_fps)
    new_p = new_tp / (new_tp + new_fp) if (new_tp + new_fp) > 0 else 0
    new_r = new_tp / (new_tp + new_fn) if (new_tp + new_fn) > 0 else 0
    new_f1 = 2 * new_p * new_r / (new_p + new_r) if (new_p + new_r) > 0 else 0

    print(f"\n{'=' * 70}")
    print("PROJECTED RESULTS WITH BALL FUSION")
    print(f"{'=' * 70}")
    print(f"  Baseline:  TP={total_tp} FP={total_fp} FN={total_fn} F1={f1:.1%}")
    print(f"  Projected: TP={new_tp} FP={new_fp} FN={new_fn} F1={new_f1:.1%}")
    print(f"  Delta: +{recoverable_fn} TP, +{len(potential_new_fps)} FP, -{recoverable_fn} FN")


def compute_overlap(s1: float, e1: float, s2: float, e2: float) -> float:
    return max(0, min(e1, e2) - max(s1, s2))


if __name__ == "__main__":
    main()
