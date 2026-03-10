#!/usr/bin/env python3
"""Analyze whether a rescue pass can recover missed short rallies.

Tests lowering the detection threshold in gaps between detected rallies,
using frame-differencing motion energy as a secondary signal to filter
false positives from real short rallies.

Usage:
    uv run python scripts/analyze_fn_rescue.py
    uv run python scripts/analyze_fn_rescue.py --rescue-threshold 0.35
    uv run python scripts/analyze_fn_rescue.py --no-motion  # prob-only baseline
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from rallycut.evaluation.ground_truth import EvaluationVideo, load_evaluation_videos
from rallycut.evaluation.matching import compute_iou, match_rallies
from rallycut.evaluation.video_resolver import VideoResolver
from rallycut.temporal.features import FeatureCache
from rallycut.temporal.temporal_maxer.inference import TemporalMaxerInference

STRIDE = 24
MODEL_PATH = Path("weights/temporal_maxer/best_temporal_maxer.pt")


def compute_motion_energy(
    video_path: Path,
    start_sec: float,
    end_sec: float,
    sample_fps: float = 2.0,
) -> float:
    """Compute motion energy in a time range via frame differencing.

    Samples frames at sample_fps, converts to grayscale, computes
    mean absolute difference between consecutive frames.

    Returns mean motion energy (0-255 scale, typically 3-15 for real scenes).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(fps / sample_fps))
    start_frame = int(start_sec * fps)
    end_frame = int(end_sec * fps)

    # Sample frames
    frames: list[np.ndarray] = []
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    for f in range(start_frame, end_frame, frame_interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ret, frame = cap.read()
        if not ret:
            break
        # Resize for speed (160px wide)
        h, w = frame.shape[:2]
        scale = 160 / w
        small = cv2.resize(frame, (160, int(h * scale)))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        frames.append(gray)

    cap.release()

    if len(frames) < 2:
        return 0.0

    # Mean absolute difference between consecutive frames
    diffs = []
    for i in range(1, len(frames)):
        diff = cv2.absdiff(frames[i], frames[i - 1])
        diffs.append(float(diff.mean()))

    return float(np.mean(diffs)) if diffs else 0.0


def resolve_video(video: EvaluationVideo, resolver: VideoResolver) -> Path | None:
    """Resolve video path from S3/cache."""
    try:
        return resolver.resolve(video.s3_key, video.content_hash)
    except Exception:
        return None


def find_rescue_candidates(
    probs: np.ndarray,
    segments: list[tuple[float, float]],
    window_dur: float,
    rescue_thresh: float,
    min_windows: int,
    min_dur: float,
    max_dur: float,
) -> list[dict]:
    """Find contiguous runs above threshold in gaps between segments."""
    total_time = len(probs) * window_dur
    candidates: list[dict] = []

    # Build gap regions
    gaps: list[tuple[float, float]] = []
    prev_end = 0.0
    for s, e in sorted(segments):
        if s > prev_end:
            gaps.append((prev_end, s))
        prev_end = max(prev_end, e)
    if prev_end < total_time:
        gaps.append((prev_end, total_time))

    for gap_s, gap_e in gaps:
        si = max(0, int(gap_s / window_dur))
        ei = min(len(probs), int(gap_e / window_dur) + 1)
        gap_probs = probs[si:ei]

        above = gap_probs >= rescue_thresh
        in_run = False
        run_start = 0

        def _emit(start: int, end: int) -> None:
            if end - start < min_windows:
                return
            cand_s = (si + start) * window_dur
            cand_e = (si + end) * window_dur
            cand_dur = cand_e - cand_s
            if not (min_dur <= cand_dur <= max_dur):
                return
            run_probs = gap_probs[start:end]
            candidates.append({
                "start": cand_s,
                "end": cand_e,
                "duration": cand_dur,
                "avg_prob": float(run_probs.mean()),
                "max_prob": float(run_probs.max()),
            })

        for i in range(len(above)):
            if above[i] and not in_run:
                run_start = i
                in_run = True
            elif not above[i] and in_run:
                _emit(run_start, i)
                in_run = False
        if in_run:
            _emit(run_start, len(above))

    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rescue-threshold", type=float, default=0.35)
    parser.add_argument("--min-windows", type=int, default=3)
    parser.add_argument("--min-duration", type=float, default=3.0)
    parser.add_argument("--max-duration", type=float, default=10.0)
    parser.add_argument("--motion-fps", type=float, default=2.0,
                        help="Frame rate for motion energy sampling")
    parser.add_argument("--no-motion", action="store_true",
                        help="Skip motion energy (prob-only baseline)")
    args = parser.parse_args()

    videos = load_evaluation_videos()
    cache = FeatureCache(cache_dir=Path("training_data/features"))
    inference = TemporalMaxerInference(MODEL_PATH, device="cpu")
    resolver = VideoResolver()

    print(f"Rescue threshold: {args.rescue_threshold}")
    print(f"Min windows: {args.min_windows}, Duration: {args.min_duration}-{args.max_duration}s")
    print(f"Motion: {'OFF' if args.no_motion else f'{args.motion_fps} FPS'}")
    print(f"Videos: {len(videos)}\n")

    all_entries: list[dict] = []  # All candidates with labels
    total_tp = 0
    total_fp_orig = 0
    total_fn = 0

    for vi, v in enumerate(videos):
        cached_data = cache.get(v.content_hash, STRIDE)
        if cached_data is None:
            continue

        features, meta = cached_data
        result = inference.predict(features=features, fps=meta.fps, stride=STRIDE)
        probs = result.window_probs
        segments = result.segments
        window_dur = STRIDE / meta.fps

        gt = [(r.start_ms / 1000, r.end_ms / 1000) for r in v.ground_truth_rallies]

        matching = match_rallies(gt, segments, iou_threshold=0.4)
        total_tp += matching.true_positives
        total_fp_orig += matching.false_positives
        total_fn += matching.false_negatives

        # Find rescue candidates
        candidates = find_rescue_candidates(
            probs, segments, window_dur,
            args.rescue_threshold, args.min_windows,
            args.min_duration, args.max_duration,
        )

        if not candidates:
            n_cand = 0
        else:
            # Resolve video for motion energy
            video_path = None
            if not args.no_motion:
                video_path = resolve_video(v, resolver)

            n_cand = len(candidates)
            for cand in candidates:
                best_iou = max(
                    (compute_iou(cand["start"], cand["end"], gs, ge) for gs, ge in gt),
                    default=0.0,
                )
                cand["filename"] = v.filename
                cand["is_tp"] = best_iou >= 0.3
                cand["best_iou"] = best_iou

                if video_path and not args.no_motion:
                    cand["motion"] = compute_motion_energy(
                        video_path, cand["start"], cand["end"],
                        sample_fps=args.motion_fps,
                    )
                else:
                    cand["motion"] = None

                all_entries.append(cand)

        print(
            f"[{vi+1}/{len(videos)}] {v.filename[:30]:<30s} "
            f"segs={len(segments)} candidates={n_cand}"
        )

    # Separate TP rescues from FP
    rescued = [e for e in all_entries if e["is_tp"]]
    new_fp = [e for e in all_entries if not e["is_tp"]]

    print(f"\n{'='*70}")
    print(f"RESCUE PASS RESULTS (prob-only)")
    print(f"{'='*70}")
    print(f"Original: TP={total_tp} FP={total_fp_orig} FN={total_fn}")
    print(f"Candidates: {len(all_entries)} ({len(rescued)} TP, {len(new_fp)} FP)")

    _print_f1("Original", total_tp, total_fp_orig, total_fn)
    _print_f1("+ rescue (all)", total_tp + len(rescued), total_fp_orig + len(new_fp),
              total_fn - len(rescued))

    # Print motion energy distributions
    if any(e["motion"] is not None for e in all_entries):
        tp_motion = [e["motion"] for e in rescued if e["motion"] is not None]
        fp_motion = [e["motion"] for e in new_fp if e["motion"] is not None]

        print(f"\n{'='*70}")
        print(f"MOTION ENERGY DISTRIBUTIONS")
        print(f"{'='*70}")
        if tp_motion:
            arr = np.array(tp_motion)
            print(f"  TP rescues ({len(arr)}): mean={arr.mean():.2f} median={np.median(arr):.2f} "
                  f"min={arr.min():.2f} max={arr.max():.2f}")
        if fp_motion:
            arr = np.array(fp_motion)
            print(f"  FP candidates ({len(arr)}): mean={arr.mean():.2f} median={np.median(arr):.2f} "
                  f"min={arr.min():.2f} max={arr.max():.2f}")

        # Separability
        if tp_motion and fp_motion:
            tp_arr = np.array(tp_motion)
            fp_arr = np.array(fp_motion)
            gap = tp_arr.mean() - fp_arr.mean()
            print(f"\n  Gap (TP mean - FP mean): {gap:.2f}")

        # Sweep motion thresholds
        print(f"\n{'='*70}")
        print(f"MOTION THRESHOLD SWEEP (rescue_prob >= {args.rescue_threshold})")
        print(f"{'='*70}")
        print(f"  {'Motion':>8s} {'Rescued':>8s} {'NewFP':>6s} {'TP':>5s} {'FP':>5s} {'FN':>5s}  {'P':>6s} {'R':>6s} {'F1':>6s}")
        for motion_thresh in [0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]:
            r_count = sum(1 for e in rescued if e["motion"] is not None and e["motion"] >= motion_thresh)
            fp_count = sum(1 for e in new_fp if e["motion"] is not None and e["motion"] >= motion_thresh)
            _print_sweep_row(motion_thresh, r_count, fp_count, total_tp, total_fp_orig, total_fn)

        # Print all candidates sorted by motion
        print(f"\n{'='*70}")
        print(f"ALL CANDIDATES (sorted by motion energy)")
        print(f"{'='*70}")
        for e in sorted(all_entries, key=lambda x: x.get("motion") or 0, reverse=True):
            label = "TP" if e["is_tp"] else "FP"
            motion = f"{e['motion']:.1f}" if e["motion"] is not None else "N/A"
            print(
                f"  {label:2s} {e['filename'][:22]:<22s} {e['start']:6.1f}-{e['end']:6.1f}s "
                f"dur={e['duration']:.1f}s avgP={e['avg_prob']:.3f} maxP={e['max_prob']:.3f} "
                f"motion={motion:>5s}"
                + (f" IoU={e['best_iou']:.2f}" if e["is_tp"] else "")
            )


def _print_f1(label: str, tp: int, fp: int, fn: int) -> None:
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print(f"  {label}: F1={f1:.1%} (P={p:.1%} R={r:.1%}) TP={tp} FP={fp} FN={fn}")


def _print_sweep_row(
    thresh: float, r_count: int, fp_count: int,
    base_tp: int, base_fp: int, base_fn: int,
) -> None:
    tp = base_tp + r_count
    fp = base_fp + fp_count
    fn = base_fn - r_count
    p = tp / (tp + fp) if (tp + fp) > 0 else 0
    r = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    print(f"  {thresh:>8.1f} {r_count:>8d} {fp_count:>6d} {tp:>5d} {fp:>5d} {fn:>5d}  {p:>5.1%} {r:>5.1%} {f1:>5.1%}")


if __name__ == "__main__":
    main()
