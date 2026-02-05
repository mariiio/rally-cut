#!/usr/bin/env python3
"""Comprehensive evaluation of binary head + deterministic decoder.

Computes:
- Segment precision/recall/F1
- Boundary MAE (mean absolute error)
- Overmerge rate
- Segments per minute
- Runtime
"""

import time
from pathlib import Path

import numpy as np
import torch
from rich import print as rprint
from rich.table import Table

from rallycut.evaluation.ground_truth import load_evaluation_videos
from rallycut.temporal.binary_head import BinaryHead
from rallycut.temporal.deterministic_decoder import (
    DecoderConfig,
    compute_boundary_errors,
    compute_overmerge_rate,
    compute_segment_metrics,
    decode,
)
from rallycut.temporal.features import FeatureCache

# Baseline F1 score from heuristics-only pipeline (for comparison)
HEURISTICS_BASELINE_F1 = 0.568


def main():
    """Run comprehensive evaluation."""
    rprint("\n[bold cyan]Binary Head + Deterministic Decoder Evaluation[/bold cyan]")
    rprint("=" * 60)

    # Load binary head model
    model_path = Path("weights/binary_head/best_binary_head.pt")
    if not model_path.exists():
        rprint(f"[red]Model not found: {model_path}[/red]")
        return

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    model = BinaryHead(
        feature_dim=768,
        hidden_dim=128,
        dropout=0.0,  # Inference mode
    )
    # Check if checkpoint is wrapped or just state_dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()

    # Best decoder config from grid search
    best_config = DecoderConfig(
        smooth_window=3,
        t_on=0.4,
        t_off=0.3,
        patience=1,
        min_segment_windows=3,
        max_gap_windows=1,
        max_duration_seconds=60.0,
        stride=48,
    )

    rprint("\n[bold]Decoder Configuration:[/bold]")
    rprint(f"  smooth_window: {best_config.smooth_window}")
    rprint(f"  t_on: {best_config.t_on}")
    rprint(f"  t_off: {best_config.t_off}")
    rprint(f"  patience: {best_config.patience}")
    rprint(f"  min_segment_windows: {best_config.min_segment_windows}")
    rprint(f"  max_gap_windows: {best_config.max_gap_windows}")
    rprint(f"  max_duration_seconds: {best_config.max_duration_seconds}s")

    # Load ground truth
    rprint("\n[bold]Loading ground truth...[/bold]")
    gt_videos = load_evaluation_videos(require_ground_truth=True)
    rprint(f"  Found {len(gt_videos)} videos with ground truth")

    # Load feature cache
    cache = FeatureCache(Path("training_data/features"))
    stride = 48

    # Collect results
    all_tp, all_fp, all_fn = 0, 0, 0
    all_start_errors = []
    all_end_errors = []
    all_overmerge = 0
    all_segments = 0
    total_duration_minutes = 0.0
    total_inference_time = 0.0

    per_video_results = []

    for video in gt_videos:
        # Load features
        cached = cache.get(video.content_hash, stride)
        if cached is None:
            rprint(f"  [yellow]Skipping {video.filename} (no cached features)[/yellow]")
            continue

        features, metadata = cached

        # Get ground truth segments
        gt_segments = [(r.start_seconds, r.end_seconds) for r in video.ground_truth_rallies]
        if not gt_segments:
            continue

        # Use default FPS if not available
        fps = video.fps if video.fps else 30.0

        # Inference
        start_time = time.perf_counter()

        features_t = torch.from_numpy(features).float()
        with torch.no_grad():
            logits = model(features_t)
            probs = torch.sigmoid(logits).squeeze(-1).numpy()

        # Decode
        config = DecoderConfig(
            smooth_window=best_config.smooth_window,
            t_on=best_config.t_on,
            t_off=best_config.t_off,
            patience=best_config.patience,
            min_segment_windows=best_config.min_segment_windows,
            max_gap_windows=best_config.max_gap_windows,
            max_duration_seconds=best_config.max_duration_seconds,
            fps=fps,
            stride=stride,
        )
        result = decode(probs, config)

        inference_time = time.perf_counter() - start_time
        total_inference_time += inference_time

        # Compute metrics
        seg_metrics = compute_segment_metrics(gt_segments, result.segments, iou_threshold=0.5)
        boundary = compute_boundary_errors(gt_segments, result.segments, iou_threshold=0.5)
        overmerge = compute_overmerge_rate(result.segments, max_duration=60.0)

        # Video duration
        video_duration_min = (video.duration_seconds or 0) / 60.0
        total_duration_minutes += video_duration_min

        # Accumulate
        all_tp += seg_metrics["tp"]
        all_fp += seg_metrics["fp"]
        all_fn += seg_metrics["fn"]

        if boundary["mean_start_error"] > 0:
            all_start_errors.append(boundary["mean_start_error"])
            all_end_errors.append(boundary["mean_end_error"])

        all_overmerge += overmerge["overmerge_count"]
        all_segments += len(result.segments)

        # Per-video results
        per_video_results.append({
            "filename": video.filename,
            "f1": seg_metrics["f1"],
            "precision": seg_metrics["precision"],
            "recall": seg_metrics["recall"],
            "tp": seg_metrics["tp"],
            "fp": seg_metrics["fp"],
            "fn": seg_metrics["fn"],
            "gt_count": len(video.ground_truth_rallies),
            "pred_count": len(result.segments),
            "start_mae": boundary["mean_start_error"],
            "end_mae": boundary["mean_end_error"],
            "overmerge": overmerge["overmerge_count"],
            "duration_min": video_duration_min,
            "inference_ms": inference_time * 1000,
        })

    # Aggregate metrics
    precision = all_tp / (all_tp + all_fp) if (all_tp + all_fp) > 0 else 0
    recall = all_tp / (all_tp + all_fn) if (all_tp + all_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    overmerge_rate = all_overmerge / all_segments if all_segments > 0 else 0
    segments_per_minute = all_segments / total_duration_minutes if total_duration_minutes > 0 else 0

    mean_start_mae = np.mean(all_start_errors) if all_start_errors else 0
    mean_end_mae = np.mean(all_end_errors) if all_end_errors else 0

    # Print results
    rprint("\n" + "=" * 60)
    rprint("[bold cyan]AGGREGATE METRICS[/bold cyan]")
    rprint("=" * 60)

    rprint("\n[bold]Segment Detection:[/bold]")
    rprint(f"  Precision:  {precision:.1%}")
    rprint(f"  Recall:     {recall:.1%}")
    rprint(f"  F1 Score:   {f1:.1%}")
    rprint(f"  TP: {all_tp}, FP: {all_fp}, FN: {all_fn}")

    rprint("\n[bold]Boundary Accuracy:[/bold]")
    rprint(f"  Mean Start MAE: {mean_start_mae:.2f}s")
    rprint(f"  Mean End MAE:   {mean_end_mae:.2f}s")
    rprint(f"  Combined MAE:   {(mean_start_mae + mean_end_mae) / 2:.2f}s")

    rprint("\n[bold]Overmerge:[/bold]")
    rprint(f"  Overmerged Segments: {all_overmerge} / {all_segments}")
    rprint(f"  Overmerge Rate:      {overmerge_rate:.1%}")

    rprint("\n[bold]Density:[/bold]")
    rprint(f"  Total Segments:      {all_segments}")
    rprint(f"  Total Duration:      {total_duration_minutes:.1f} min")
    rprint(f"  Segments per Minute: {segments_per_minute:.2f}")

    rprint("\n[bold]Runtime:[/bold]")
    rprint(f"  Total Inference:     {total_inference_time * 1000:.1f}ms")
    rprint(f"  Per Video Avg:       {total_inference_time / len(per_video_results) * 1000:.2f}ms")
    rprint(f"  Per Minute Video:    {total_inference_time / total_duration_minutes * 1000:.2f}ms")

    # Per-video table
    rprint("\n" + "=" * 60)
    rprint("[bold cyan]PER-VIDEO BREAKDOWN[/bold cyan]")
    rprint("=" * 60)

    table = Table(show_header=True, header_style="bold")
    table.add_column("Video", style="cyan", max_width=25)
    table.add_column("F1", justify="right")
    table.add_column("P", justify="right")
    table.add_column("R", justify="right")
    table.add_column("GT", justify="right")
    table.add_column("Pred", justify="right")
    table.add_column("Start MAE", justify="right")
    table.add_column("End MAE", justify="right")

    for r in sorted(per_video_results, key=lambda x: x["f1"], reverse=True):
        table.add_row(
            r["filename"][:25],
            f"{r['f1']:.0%}",
            f"{r['precision']:.0%}",
            f"{r['recall']:.0%}",
            str(r["gt_count"]),
            str(r["pred_count"]),
            f"{r['start_mae']:.1f}s" if r["start_mae"] > 0 else "-",
            f"{r['end_mae']:.1f}s" if r["end_mae"] > 0 else "-",
        )

    rprint(table)

    # Summary comparison with heuristics baseline
    rprint("\n" + "=" * 60)
    rprint("[bold cyan]COMPARISON WITH HEURISTICS BASELINE[/bold cyan]")
    rprint("=" * 60)
    relative_improvement = (f1 - HEURISTICS_BASELINE_F1) / HEURISTICS_BASELINE_F1
    rprint(f"""
    Metric              Heuristics    Binary+Decoder    Change
    ─────────────────────────────────────────────────────────────
    F1 Score            {HEURISTICS_BASELINE_F1:.1%}         {f1:.1%}             {f1 - HEURISTICS_BASELINE_F1:+.1%}
    Relative Improvement                                 {relative_improvement:.1%}
    """)


if __name__ == "__main__":
    main()
