#!/usr/bin/env python3
"""Simulate rally continuation heuristic on test videos."""

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


def apply_rally_continuation(
    results: list[GameStateResult],
    fps: float,
    min_no_play_to_end: int,  # consecutive NO_PLAY predictions to end rally
) -> list[tuple[float, bool]]:
    """Apply rally continuation heuristic.

    Once active (PLAY/SERVICE) is detected, keep it active until
    we see min_no_play_to_end consecutive NO_PLAY predictions.

    Returns list of (frame_time, is_active) tuples.
    """
    # Build frame->prediction map
    frame_states: dict[int, GameState] = {}
    for r in results:
        if r.start_frame is not None and r.end_frame is not None:
            for f in range(r.start_frame, r.end_frame + 1):
                frame_states[f] = r.state

    if not frame_states:
        return []

    max_frame = max(frame_states.keys())

    # Apply heuristic
    in_rally = False
    consecutive_no_play = 0
    output = []

    for frame in range(max_frame + 1):
        raw_state = frame_states.get(frame, GameState.NO_PLAY)
        raw_active = raw_state in (GameState.PLAY, GameState.SERVICE)

        if raw_active:
            in_rally = True
            consecutive_no_play = 0
        else:
            consecutive_no_play += 1
            if consecutive_no_play >= min_no_play_to_end:
                in_rally = False

        time = frame / fps
        output.append((time, in_rally))

    return output


def evaluate_heuristic(
    heuristic_output: list[tuple[float, bool]],
    gt_segments: list[dict[str, float | str]],
    duration: float,
) -> dict[str, float]:
    """Evaluate heuristic output against ground truth."""
    # Sample at 1-second intervals
    correct = 0
    total = 0
    tp = 0  # true positives (active correctly detected)
    fp = 0  # false positives (inactive marked as active)
    fn = 0  # false negatives (active missed)
    tn = 0  # true negatives

    # Build time->active map from heuristic output
    time_active: dict[int, bool] = {}
    for t, active in heuristic_output:
        time_active[int(t)] = active

    for t in range(int(duration)):
        pred_active = time_active.get(t, False)
        gt_label = get_gt_label(float(t), gt_segments)
        gt_active = gt_label in ("service", "play")

        if pred_active == gt_active:
            correct += 1

        if pred_active and gt_active:
            tp += 1
        elif pred_active and not gt_active:
            fp += 1
        elif not pred_active and gt_active:
            fn += 1
        else:
            tn += 1

        total += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "accuracy": correct / total * 100,
        "precision": precision * 100,
        "recall": recall * 100,
        "f1": f1 * 100,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def main() -> int:
    project_root = Path(__file__).parent.parent
    gt_path = project_root / "tests/fixtures/ground_truth.json"
    video_dir = project_root / "tests/fixtures"

    ground_truth = load_ground_truth(gt_path)

    # Test different min_no_play_to_end values (in frames)
    # At 30fps: 30 frames = 1s, 60 = 2s, 90 = 3s
    # At 60fps: 60 frames = 1s, 120 = 2s, 180 = 3s

    thresholds_seconds = [1, 2, 3, 4, 5, 8, 10]

    print("=" * 90)
    print("RALLY CONTINUATION HEURISTIC SIMULATION")
    print("=" * 90)
    print("\nHeuristic: Once PLAY/SERVICE detected, keep active until N seconds of consecutive NO_PLAY\n")

    all_results: dict[str, list[dict[str, float]]] = {str(t): [] for t in thresholds_seconds}

    for video_name, gt_segments in ground_truth.items():
        video_path = video_dir / video_name
        if not video_path.exists():
            continue

        print(f"\n{video_name}")
        print("-" * 90)

        video = Video(video_path)
        fps = video.info.fps
        duration = min(video.info.duration, 120.0)

        # Get raw predictions
        analyzer = GameStateAnalyzer(enable_temporal_smoothing=False)
        raw_results = analyzer.analyze_video(video, limit_seconds=duration, return_raw=False)
        assert isinstance(raw_results, list)

        # Baseline (no heuristic)
        baseline_output = []
        frame_states: dict[int, GameState] = {}
        for r in raw_results:
            if r.start_frame is not None and r.end_frame is not None:
                for f in range(r.start_frame, r.end_frame + 1):
                    frame_states[f] = r.state

        if frame_states:
            max_frame = max(frame_states.keys())
            for frame in range(max_frame + 1):
                state = frame_states.get(frame, GameState.NO_PLAY)
                active = state in (GameState.PLAY, GameState.SERVICE)
                baseline_output.append((frame / fps, active))

        baseline_metrics = evaluate_heuristic(baseline_output, gt_segments, duration)

        print(f"{'Config':<20} | {'Acc':>6} | {'Prec':>6} | {'Recall':>6} | {'F1':>6} | {'TP':>4} | {'FP':>4} | {'FN':>4}")
        print("-" * 90)
        print(f"{'Baseline (raw ML)':<20} | {baseline_metrics['accuracy']:5.1f}% | {baseline_metrics['precision']:5.1f}% | {baseline_metrics['recall']:5.1f}% | {baseline_metrics['f1']:5.1f}% | {baseline_metrics['tp']:>4} | {baseline_metrics['fp']:>4} | {baseline_metrics['fn']:>4}")

        # Test each threshold
        for thresh_sec in thresholds_seconds:
            min_frames = int(thresh_sec * fps)
            heuristic_output = apply_rally_continuation(raw_results, fps, min_frames)
            metrics = evaluate_heuristic(heuristic_output, gt_segments, duration)
            all_results[str(thresh_sec)].append(metrics)

            label = f"Continue {thresh_sec}s"
            print(f"{label:<20} | {metrics['accuracy']:5.1f}% | {metrics['precision']:5.1f}% | {metrics['recall']:5.1f}% | {metrics['f1']:5.1f}% | {metrics['tp']:>4} | {metrics['fp']:>4} | {metrics['fn']:>4}")

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY ACROSS ALL VIDEOS")
    print("=" * 90)
    print(f"\n{'Config':<20} | {'Avg Acc':>8} | {'Avg Prec':>8} | {'Avg Recall':>10} | {'Avg F1':>8}")
    print("-" * 70)

    for thresh_sec in thresholds_seconds:
        results = all_results[str(thresh_sec)]
        if results:
            avg_acc = sum(r["accuracy"] for r in results) / len(results)
            avg_prec = sum(r["precision"] for r in results) / len(results)
            avg_recall = sum(r["recall"] for r in results) / len(results)
            avg_f1 = sum(r["f1"] for r in results) / len(results)
            label = f"Continue {thresh_sec}s"
            print(f"{label:<20} | {avg_acc:7.1f}% | {avg_prec:7.1f}% | {avg_recall:9.1f}% | {avg_f1:7.1f}%")

    return 0


if __name__ == "__main__":
    sys.exit(main())
