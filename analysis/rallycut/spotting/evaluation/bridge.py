"""Bridge between E2E-Spot model output and existing evaluation harness.

Runs inference on held-out rallies and evaluates using match_contacts()
and compute_metrics() from eval_action_detection.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from rich.console import Console
from rich.table import Table

from rallycut.spotting.config import ACTION_TYPES, E2ESpotConfig
from rallycut.spotting.data.beach import RallyInfo
from rallycut.spotting.data.clip_dataset import _load_frames_from_disk
from rallycut.spotting.data.transforms import ClipTransform
from rallycut.spotting.model.e2e_spot import E2ESpot
from rallycut.spotting.training.postprocess import extract_events

# Import eval types from sibling script
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from eval_action_detection import (  # type: ignore[import-not-found]  # noqa: E402
    MatchResult,
    compute_metrics,
    match_contacts,
)

console = Console()


def _run_rally_inference(
    model: E2ESpot,
    rally: RallyInfo,
    device: torch.device,
    config: E2ESpotConfig,
) -> list[dict]:
    """Run inference on a single rally and return event detections.

    Processes the rally in overlapping clips and merges predictions.
    """
    model.eval()
    transform = ClipTransform(size=224, is_train=False)
    clip_length = config.training.clip_length

    # Accumulate per-frame logits and offsets across overlapping clips
    logits_accum = torch.zeros(rally.frame_count, config.head.num_classes)
    offsets_accum = torch.zeros(rally.frame_count, 1)
    count_accum = torch.zeros(rally.frame_count)

    # Slide clips with 50% overlap
    stride = clip_length // 2
    starts = list(range(0, max(1, rally.frame_count - clip_length + 1), stride))
    # Ensure we cover the end
    if rally.frame_count > clip_length and starts[-1] + clip_length < rally.frame_count:
        starts.append(rally.frame_count - clip_length)

    with torch.no_grad():
        for start in starts:
            end = min(start + clip_length, rally.frame_count)
            actual_len = end - start

            # Load frames
            frames = _load_frames_from_disk(rally.frame_dir, start, end)

            # Pad if needed
            if actual_len < clip_length:
                pad_len = clip_length - actual_len
                frames = frames + [frames[-1]] * pad_len

            clip = transform(frames).unsqueeze(0).to(device)  # (1, T, 3, H, W)

            out = model(clip)
            logits = out["logits"][0, :actual_len].cpu()   # (actual_len, C)
            offsets = out["offsets"][0, :actual_len].cpu()  # (actual_len, 1)

            logits_accum[start:end] += logits
            offsets_accum[start:end] += offsets
            count_accum[start:end] += 1.0

    # Average overlapping predictions
    count_accum = count_accum.clamp(min=1.0).unsqueeze(-1)
    avg_logits = logits_accum / count_accum
    avg_offsets = offsets_accum / count_accum

    # Extract events
    events = extract_events(avg_logits, avg_offsets, config.postprocess)
    return events


def evaluate_model(
    model: E2ESpot,
    rallies: list[RallyInfo],
    device: torch.device,
    config: E2ESpotConfig,
    tolerance_ms: int = 167,
) -> dict:
    """Evaluate E2E-Spot model on rallies using the existing eval harness.

    Args:
        model: Trained E2E-Spot model.
        rallies: Rally data to evaluate on.
        device: Inference device.
        config: Model configuration.
        tolerance_ms: Frame matching tolerance in milliseconds.

    Returns:
        Metrics dict from compute_metrics().
    """
    all_matches: list[MatchResult] = []
    all_unmatched: list[dict] = []

    for i, rally in enumerate(rallies):
        if not rally.frame_dir.exists():
            console.print(f"  [yellow]Skipping {rally.rally_id} (no frames)[/]")
            continue

        preds = _run_rally_inference(model, rally, device, config)

        tolerance_frames = round(rally.fps * tolerance_ms / 1000.0)
        matches, unmatched = match_contacts(
            rally.gt_labels, preds, tolerance=tolerance_frames
        )
        all_matches.extend(matches)
        all_unmatched.extend(unmatched)

        # Per-rally summary
        matched = [m for m in matches if m.pred_frame is not None]
        correct = sum(1 for m in matched if m.gt_action == m.pred_action)
        gt_count = len(rally.gt_labels)
        console.print(
            f"  [{i + 1}/{len(rallies)}] {rally.rally_id[:8]}: "
            f"GT={gt_count} pred={len(preds)} matched={len(matched)} "
            f"action_correct={correct}/{len(matched)}"
        )

    metrics: dict = compute_metrics(all_matches, all_unmatched)
    return metrics


def print_metrics(metrics: dict) -> None:
    """Print evaluation metrics in a formatted table."""
    console.print()
    console.print("[bold]Contact Detection[/]")
    console.print(
        f"  Precision: {metrics['precision']:.1%}  "
        f"Recall: {metrics['recall']:.1%}  "
        f"F1: {metrics['f1']:.1%}"
    )
    console.print(
        f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}  "
        f"Total GT={metrics['total_gt']}"
    )

    console.print(f"\n[bold]Action Accuracy: {metrics['action_accuracy']:.1%}[/]")

    # Per-class table
    table = Table(title="Per-Class Metrics")
    table.add_column("Action", style="cyan")
    table.add_column("TP", justify="right")
    table.add_column("FP", justify="right")
    table.add_column("FN", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")

    per_class = metrics.get("per_class", {})
    for action in ACTION_TYPES:
        if action not in per_class:
            continue
        c = per_class[action]
        table.add_row(
            action,
            str(c["tp"]),
            str(c["fp"]),
            str(c["fn"]),
            f"{c['precision']:.1%}",
            f"{c['recall']:.1%}",
            f"{c['f1']:.1%}",
        )

    console.print(table)
