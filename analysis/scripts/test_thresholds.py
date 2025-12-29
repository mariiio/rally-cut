#!/usr/bin/env python3
"""Test different classification thresholds and smoothing settings."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rallycut.analysis.game_state import GameStateAnalyzer
from rallycut.core.models import GameState, GameStateResult
from rallycut.core.video import Video


def load_ground_truth(path: Path) -> dict[str, list[dict[str, float | str]]]:
    """Load ground truth segments."""
    with open(path) as f:
        data = json.load(f)
    return {name: info.get("segments", []) for name, info in data.items()}


def get_gt_label(time: float, segments: list[dict[str, float | str]]) -> str:
    """Get ground truth label at time."""
    for seg in segments:
        if float(seg["start"]) <= time < float(seg["end"]):
            return str(seg["label"])
    return "no_play"


def evaluate_with_threshold(
    results: list[GameStateResult],
    fps: float,
    duration: float,
    gt_segments: list[dict[str, float | str]],
    active_threshold: float,
) -> dict[str, float]:
    """Evaluate using a custom active threshold.

    Instead of argmax, classify as active if play_prob + service_prob > threshold.
    """
    correct = 0
    total = 0
    active_detected = 0
    active_gt = 0

    t = 0.0
    while t < duration:
        frame_idx = int(t * fps)

        # Find prediction for this frame
        pred_active = False
        for r in results:
            if r.start_frame is not None and r.end_frame is not None:
                if r.start_frame <= frame_idx <= r.end_frame:
                    # Use threshold on active probability
                    active_prob = r.play_confidence + r.service_confidence
                    pred_active = active_prob > active_threshold
                    break

        gt_label = get_gt_label(t, gt_segments)
        gt_active = gt_label in ("service", "play")

        if pred_active == gt_active:
            correct += 1
        if pred_active:
            active_detected += 1
        if gt_active:
            active_gt += 1

        total += 1
        t += 1.0

    return {
        "accuracy": correct / total * 100,
        "active_detected": active_detected,
        "active_gt": active_gt,
        "detection_ratio": active_detected / active_gt * 100 if active_gt > 0 else 0,
    }


def analyze_probabilities(
    results: list[GameStateResult],
    fps: float,
    duration: float,
    gt_segments: list[dict[str, float | str]],
) -> None:
    """Show probability distribution for misclassified frames."""
    print("\nProbability Analysis for Misclassified Frames:")
    print("-" * 70)
    print(f"{'Time':>6} | {'GT':>3} | {'Pred':>3} | {'NOP':>5} | {'PLY':>5} | {'SRV':>5} | Active")
    print("-" * 70)

    t = 0.0
    misses = []
    while t < duration:
        frame_idx = int(t * fps)

        for r in results:
            if r.start_frame is not None and r.end_frame is not None:
                if r.start_frame <= frame_idx <= r.end_frame:
                    gt_label = get_gt_label(t, gt_segments)
                    gt_active = gt_label in ("service", "play")
                    pred_active = r.state in (GameState.PLAY, GameState.SERVICE)

                    if gt_active != pred_active:
                        active_prob = r.play_confidence + r.service_confidence
                        misses.append({
                            "time": t,
                            "gt": gt_label[:3].upper(),
                            "pred": r.state.value[:3].upper(),
                            "nop": r.no_play_confidence,
                            "ply": r.play_confidence,
                            "srv": r.service_confidence,
                            "active": active_prob,
                        })
                    break
        t += 1.0

    # Show first 20 misses
    for m in misses[:20]:
        print(f"{m['time']:6.1f} | {m['gt']:>3} | {m['pred']:>3} | {m['nop']:5.2f} | {m['ply']:5.2f} | {m['srv']:5.2f} | {m['active']:5.2f}")

    if len(misses) > 20:
        print(f"... and {len(misses) - 20} more misses")

    # Analyze threshold that would fix misses
    if misses:
        # For false negatives (GT=active, pred=no_play), check if lowering threshold helps
        fn_misses = [m for m in misses if m["gt"] in ("SRV", "PLY")]
        if fn_misses:
            active_probs = [m["active"] for m in fn_misses]
            print(f"\nFalse Negatives (missed active): {len(fn_misses)}")
            print(f"  Active prob range: {min(active_probs):.3f} - {max(active_probs):.3f}")
            print(f"  Active prob mean:  {sum(active_probs)/len(active_probs):.3f}")

            # How many would be fixed at different thresholds
            for thresh in [0.3, 0.35, 0.4, 0.45]:
                fixed = sum(1 for p in active_probs if p > thresh)
                print(f"  Fixed at >{thresh}: {fixed}/{len(fn_misses)} ({fixed/len(fn_misses)*100:.0f}%)")


def main() -> int:
    project_root = Path(__file__).parent.parent
    gt_path = project_root / "tests/fixtures/ground_truth.json"
    video_dir = project_root / "tests/fixtures"

    ground_truth = load_ground_truth(gt_path)

    # Test configurations
    configs = [
        {"name": "Baseline (smooth=ON)", "smooth": True, "threshold": 0.5},
        {"name": "No smoothing", "smooth": False, "threshold": 0.5},
        {"name": "Threshold 0.40", "smooth": False, "threshold": 0.40},
        {"name": "Threshold 0.35", "smooth": False, "threshold": 0.35},
        {"name": "Threshold 0.30", "smooth": False, "threshold": 0.30},
        {"name": "Threshold 0.25", "smooth": False, "threshold": 0.25},
    ]

    for video_name, gt_segments in ground_truth.items():
        video_path = video_dir / video_name
        if not video_path.exists():
            continue

        print(f"\n{'='*70}")
        print(f"VIDEO: {video_name}")
        print("=" * 70)

        video = Video(video_path)
        fps = video.info.fps
        duration = min(video.info.duration, 120.0)

        # Get raw predictions (no smoothing applied yet)
        analyzer = GameStateAnalyzer(enable_temporal_smoothing=False)
        raw_results = analyzer.analyze_video(video, limit_seconds=duration, return_raw=False)
        assert isinstance(raw_results, list)

        # Analyze probabilities first
        analyze_probabilities(raw_results, fps, duration, gt_segments)

        # Test each configuration
        print(f"\n{'Config':<25} | {'Accuracy':>8} | {'Detected':>8} | {'GT Active':>9} | {'Ratio':>8}")
        print("-" * 70)

        for cfg in configs:
            if cfg["smooth"]:
                results = analyzer.smooth_results(raw_results)
            else:
                results = raw_results

            metrics = evaluate_with_threshold(
                results, fps, duration, gt_segments, cfg["threshold"]
            )

            print(f"{cfg['name']:<25} | {metrics['accuracy']:7.1f}% | {metrics['active_detected']:>8} | {metrics['active_gt']:>9} | {metrics['detection_ratio']:7.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
