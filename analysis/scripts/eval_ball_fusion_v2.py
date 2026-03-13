#!/usr/bin/env python3
"""Evaluate ball fusion with TemporalMaxer confidence gating.

Instead of blindly looking for ball-dense regions in ALL gaps,
only consider gaps where the TemporalMaxer has some signal (prob > gate).
This should dramatically reduce false positives.

Usage:
    uv run python scripts/eval_ball_fusion_v2.py
    uv run python scripts/eval_ball_fusion_v2.py --prob-gate 0.2 --density-threshold 0.4
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
                        help="Min ball density (default: 0.3)")
    parser.add_argument("--min-duration", type=float, default=2.0,
                        help="Min duration of candidate region (default: 2.0s)")
    parser.add_argument("--max-duration", type=float, default=12.0,
                        help="Max duration for ball-detected rally (default: 12.0)")
    parser.add_argument("--prob-gate", type=float, default=0.15,
                        help="Min TemporalMaxer avg prob in gap to consider fusion (default: 0.15)")
    parser.add_argument("--window", type=float, default=2.0,
                        help="Density computation window (default: 2.0s)")
    parser.add_argument("--conf-threshold", type=float, default=0.3,
                        help="WASB confidence threshold (default: 0.3)")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep prob-gate and density-threshold")
    args = parser.parse_args()

    videos = load_evaluation_videos()
    cache = FeatureCache(FEATURE_DIR)
    inference = TemporalMaxerInference(MODEL_PATH, device="mps")

    if args.sweep:
        run_sweep(videos, cache, inference, args)
        return

    run_eval(videos, cache, inference, args)


def run_eval(videos, cache, inference, args) -> None:
    fn_with_ball: list[dict] = []
    fn_without_ball: list[dict] = []
    potential_new_fps: list[dict] = []
    gated_out_fps: list[dict] = []
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

        # Get per-second model probabilities
        window_probs = result.window_probs
        window_duration = STRIDE / metadata.fps  # seconds per window

        def get_avg_prob(start_s: float, end_s: float) -> float:
            """Get average TemporalMaxer probability for a time range."""
            w_start = max(0, int(start_s / window_duration))
            w_end = min(len(window_probs), int(end_s / window_duration) + 1)
            if w_end <= w_start:
                return 0.0
            return float(window_probs[w_start:w_end].mean())

        def get_max_prob(start_s: float, end_s: float) -> float:
            """Get max TemporalMaxer probability for a time range."""
            w_start = max(0, int(start_s / window_duration))
            w_end = min(len(window_probs), int(end_s / window_duration) + 1)
            if w_end <= w_start:
                return 0.0
            return float(window_probs[w_start:w_end].max())

        # Analyze each FN
        for fn_idx in mr.unmatched_ground_truth:
            gt_start, gt_end = gt_segs[fn_idx]
            gt_dur = gt_end - gt_start

            s_idx = max(0, int(gt_start))
            e_idx = min(len(density), int(gt_end) + 1)
            if e_idx <= s_idx:
                fn_without_ball.append({
                    "video": v.filename, "start": gt_start, "end": gt_end,
                    "duration": gt_dur, "density": 0.0, "model_prob": 0.0,
                    "reason": "out of range",
                })
                continue

            rally_density = density[s_idx:e_idx]
            avg_density = float(rally_density.mean())
            max_density = float(rally_density.max())
            dense_seconds = int((rally_density >= args.density_threshold).sum())
            avg_prob = get_avg_prob(gt_start, gt_end)
            max_prob = get_max_prob(gt_start, gt_end)

            info = {
                "video": v.filename, "start": gt_start, "end": gt_end,
                "duration": gt_dur, "avg_density": avg_density,
                "max_density": max_density, "dense_seconds": dense_seconds,
                "avg_prob": avg_prob, "max_prob": max_prob,
            }

            passes_density = dense_seconds >= args.min_duration and gt_dur <= args.max_duration
            passes_gate = avg_prob >= args.prob_gate

            if passes_density and passes_gate:
                fn_with_ball.append(info)
                recoverable_fn += 1
            else:
                reasons = []
                if not passes_density:
                    reasons.append(f"dense={dense_seconds}s<{args.min_duration}s")
                if not passes_gate:
                    reasons.append(f"prob={avg_prob:.2f}<{args.prob_gate}")
                info["reason"] = ", ".join(reasons)
                fn_without_ball.append(info)

        # Check gaps for potential new FPs (with prob gating)
        pred_segs = sorted(result.segments)
        video_duration = density_data["n_frames"] / density_data["fps"]
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

            # Find contiguous dense runs
            in_run = False
            run_start = 0
            for j in range(len(gap_density)):
                if gap_density[j] >= args.density_threshold and not in_run:
                    run_start = j
                    in_run = True
                elif gap_density[j] < args.density_threshold and in_run:
                    _process_run(
                        run_start, j, s_idx, gap_density, gt_segs, v.filename,
                        args, get_avg_prob, get_max_prob,
                        potential_new_fps, gated_out_fps,
                    )
                    in_run = False
            if in_run:
                _process_run(
                    run_start, len(gap_density), s_idx, gap_density, gt_segs, v.filename,
                    args, get_avg_prob, get_max_prob,
                    potential_new_fps, gated_out_fps,
                )

    # Report
    print("=" * 70)
    print("BALL TRAJECTORY FUSION WITH CONFIDENCE GATING")
    print(f"  Density threshold: {args.density_threshold}, min duration: {args.min_duration}s")
    print(f"  Prob gate: {args.prob_gate}, window: {args.window}s, WASB conf: {args.conf_threshold}")
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
    print(f"FN RECOVERABLE ({len(fn_with_ball)}/{total_fn})")
    print(f"{'=' * 70}")
    if fn_with_ball:
        for fn in sorted(fn_with_ball, key=lambda x: -x["avg_density"]):
            print(
                f"  {fn['video']:<30} {fmt_time(fn['start'])}-{fmt_time(fn['end'])} "
                f"({fn['duration']:.1f}s) density={fn['avg_density']:.2f} "
                f"prob={fn['avg_prob']:.2f}/{fn['max_prob']:.2f} dense={fn['dense_seconds']}s"
            )
    else:
        print("  None")

    print(f"\n{'=' * 70}")
    print(f"FN NOT RECOVERABLE ({len(fn_without_ball)}/{total_fn})")
    print(f"{'=' * 70}")
    for fn in sorted(fn_without_ball, key=lambda x: x["video"]):
        prob_str = f"prob={fn.get('avg_prob', 0):.2f}/{fn.get('max_prob', 0):.2f}"
        print(
            f"  {fn['video']:<30} {fmt_time(fn.get('start', 0))}-{fmt_time(fn.get('end', 0))} "
            f"({fn.get('duration', 0):.1f}s) density={fn.get('avg_density', fn.get('density', 0)):.2f} "
            f"{prob_str} reason={fn.get('reason', 'low density')}"
        )

    print(f"\n{'=' * 70}")
    print(f"POTENTIAL NEW FPs AFTER GATING ({len(potential_new_fps)})")
    print(f"{'=' * 70}")
    if potential_new_fps:
        for fp in sorted(potential_new_fps, key=lambda x: -x["avg_prob"]):
            print(
                f"  {fp['video']:<30} {fmt_time(fp['start'])}-{fmt_time(fp['end'])} "
                f"({fp['duration']}s) density={fp['avg_density']:.2f} "
                f"prob={fp['avg_prob']:.2f}/{fp['max_prob']:.2f}"
            )
    else:
        print("  None")

    print(f"\n{'=' * 70}")
    print(f"FPs BLOCKED BY PROB GATE ({len(gated_out_fps)})")
    print(f"{'=' * 70}")
    if gated_out_fps:
        for fp in sorted(gated_out_fps, key=lambda x: -x["avg_density"])[:10]:
            print(
                f"  {fp['video']:<30} {fmt_time(fp['start'])}-{fmt_time(fp['end'])} "
                f"({fp['duration']}s) density={fp['avg_density']:.2f} "
                f"prob={fp['avg_prob']:.2f}/{fp['max_prob']:.2f}"
            )
        if len(gated_out_fps) > 10:
            print(f"  ... and {len(gated_out_fps) - 10} more")
    else:
        print("  None")

    # Projected
    new_tp = total_tp + recoverable_fn
    new_fn = total_fn - recoverable_fn
    new_fp = total_fp + len(potential_new_fps)
    new_p = new_tp / (new_tp + new_fp) if (new_tp + new_fp) > 0 else 0
    new_r = new_tp / (new_tp + new_fn) if (new_tp + new_fn) > 0 else 0
    new_f1 = 2 * new_p * new_r / (new_p + new_r) if (new_p + new_r) > 0 else 0

    print(f"\n{'=' * 70}")
    print("PROJECTED RESULTS")
    print(f"{'=' * 70}")
    print(f"  Baseline:  TP={total_tp} FP={total_fp} FN={total_fn} F1={f1:.1%}")
    print(f"  Projected: TP={new_tp} FP={new_fp} FN={new_fn} F1={new_f1:.1%}")
    print(f"  Delta: +{recoverable_fn} TP, +{len(potential_new_fps)} FP, -{recoverable_fn} FN")


def _process_run(
    run_start, run_end, s_idx, gap_density, gt_segs, video_name,
    args, get_avg_prob, get_max_prob,
    potential_new_fps, gated_out_fps,
):
    run_len = run_end - run_start
    if run_len < args.min_duration:
        return

    cand_s = s_idx + run_start
    cand_e = s_idx + run_end
    cand_dur = cand_e - cand_s
    if cand_dur > args.max_duration:
        return

    # Check GT overlap
    overlaps_gt = any(
        max(0, min(cand_e, ge) - max(cand_s, gs)) > 0
        for gs, ge in gt_segs
    )
    if overlaps_gt:
        return

    cand_density = float(gap_density[run_start:run_end].mean())
    avg_prob = get_avg_prob(float(cand_s), float(cand_e))
    max_prob = get_max_prob(float(cand_s), float(cand_e))

    info = {
        "video": video_name,
        "start": float(cand_s), "end": float(cand_e),
        "duration": cand_dur, "avg_density": cand_density,
        "avg_prob": avg_prob, "max_prob": max_prob,
    }

    if avg_prob >= args.prob_gate:
        potential_new_fps.append(info)
    else:
        gated_out_fps.append(info)


def run_sweep(videos, cache, inference, args):
    """Sweep prob-gate and density-threshold to find optimal operating point."""
    from rallycut.evaluation.matching import match_rallies

    # Pre-compute all video data
    video_data = []
    for v in videos:
        cached = cache.get(v.content_hash, STRIDE)
        if cached is None:
            continue
        features, metadata = cached
        result = inference.predict(features=features, fps=metadata.fps, stride=STRIDE)
        gt_segs = [(r.start_ms / 1000, r.end_ms / 1000) for r in v.ground_truth_rallies]
        mr = match_rallies(result.segments, gt_segs, iou_threshold=IOU_THRESHOLD)
        density_data = load_density(v.id)
        video_data.append({
            "v": v, "result": result, "gt_segs": gt_segs, "mr": mr,
            "density_data": density_data, "metadata": metadata,
        })

    total_tp = sum(len(d["mr"].matches) for d in video_data)
    total_fp = sum(len(d["mr"].unmatched_predictions) for d in video_data)
    total_fn = sum(len(d["mr"].unmatched_ground_truth) for d in video_data)

    p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0

    print(f"Baseline: TP={total_tp} FP={total_fp} FN={total_fn} F1={f1:.1%}")
    print()
    print(f"{'prob_gate':>10} {'density':>8} {'min_dur':>8} {'rec_FN':>7} {'new_FP':>7} "
          f"{'proj_F1':>8} {'delta':>8}")
    print("-" * 70)

    best_f1 = f1
    best_params = None

    for prob_gate in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
        for density_th in [0.3, 0.4, 0.5, 0.6]:
            for min_dur in [2.0, 3.0, 4.0]:
                rec_fn = 0
                new_fps = 0

                for d in video_data:
                    if d["density_data"] is None:
                        continue

                    density = compute_density_curve(
                        d["density_data"]["confidences"], d["density_data"]["fps"],
                        window_s=args.window, conf_threshold=args.conf_threshold,
                    )
                    window_probs = d["result"].window_probs
                    window_duration = STRIDE / d["metadata"].fps

                    def get_avg_prob(start_s, end_s):
                        w_start = max(0, int(start_s / window_duration))
                        w_end = min(len(window_probs), int(end_s / window_duration) + 1)
                        if w_end <= w_start:
                            return 0.0
                        return float(window_probs[w_start:w_end].mean())

                    # Count recoverable FNs
                    for fn_idx in d["mr"].unmatched_ground_truth:
                        gt_start, gt_end = d["gt_segs"][fn_idx]
                        gt_dur = gt_end - gt_start
                        s_idx = max(0, int(gt_start))
                        e_idx = min(len(density), int(gt_end) + 1)
                        if e_idx <= s_idx:
                            continue
                        rally_density = density[s_idx:e_idx]
                        dense_seconds = int((rally_density >= density_th).sum())
                        avg_prob = get_avg_prob(gt_start, gt_end)
                        if (dense_seconds >= min_dur and gt_dur <= args.max_duration
                                and avg_prob >= prob_gate):
                            rec_fn += 1

                    # Count new FPs in gaps
                    pred_segs = sorted(d["result"].segments)
                    video_duration = d["density_data"]["n_frames"] / d["density_data"]["fps"]
                    gaps = []
                    prev_end = 0.0
                    for s, e in pred_segs:
                        if s > prev_end:
                            gaps.append((prev_end, s))
                        prev_end = max(prev_end, e)
                    if prev_end < video_duration:
                        gaps.append((prev_end, video_duration))

                    for gap_s, gap_e in gaps:
                        si = max(0, int(gap_s))
                        ei = min(len(density), int(gap_e) + 1)
                        if ei <= si:
                            continue
                        gd = density[si:ei]
                        in_run = False
                        rs = 0
                        for j in range(len(gd)):
                            if gd[j] >= density_th and not in_run:
                                rs = j
                                in_run = True
                            elif gd[j] < density_th and in_run:
                                rl = j - rs
                                if rl >= min_dur:
                                    cs, ce = si + rs, si + j
                                    cd = ce - cs
                                    if cd <= args.max_duration:
                                        overlaps = any(
                                            max(0, min(ce, ge) - max(cs, gs)) > 0
                                            for gs, ge in d["gt_segs"]
                                        )
                                        if not overlaps:
                                            ap = get_avg_prob(float(cs), float(ce))
                                            if ap >= prob_gate:
                                                new_fps += 1
                                in_run = False
                        if in_run:
                            rl = len(gd) - rs
                            if rl >= min_dur:
                                cs, ce = si + rs, si + len(gd)
                                cd = ce - cs
                                if cd <= args.max_duration:
                                    overlaps = any(
                                        max(0, min(ce, ge) - max(cs, gs)) > 0
                                        for gs, ge in d["gt_segs"]
                                    )
                                    if not overlaps:
                                        ap = get_avg_prob(float(cs), float(ce))
                                        if ap >= prob_gate:
                                            new_fps += 1

                new_tp = total_tp + rec_fn
                new_fn_count = total_fn - rec_fn
                new_fp_count = total_fp + new_fps
                np_ = new_tp / (new_tp + new_fp_count) if (new_tp + new_fp_count) > 0 else 0
                nr = new_tp / (new_tp + new_fn_count) if (new_tp + new_fn_count) > 0 else 0
                nf1 = 2 * np_ * nr / (np_ + nr) if (np_ + nr) > 0 else 0
                delta = nf1 - f1

                marker = ""
                if nf1 > best_f1:
                    best_f1 = nf1
                    best_params = (prob_gate, density_th, min_dur)
                    marker = " ***"

                if rec_fn > 0 or new_fps > 0:
                    print(
                        f"{prob_gate:>10.2f} {density_th:>8.1f} {min_dur:>8.1f} "
                        f"{rec_fn:>7d} {new_fps:>7d} {nf1:>8.1%} {delta:>+8.2%}{marker}"
                    )

    print()
    if best_params:
        print(f"Best: prob_gate={best_params[0]}, density={best_params[1]}, "
              f"min_dur={best_params[2]}, F1={best_f1:.1%}")
    else:
        print("No improvement found over baseline")


if __name__ == "__main__":
    main()
