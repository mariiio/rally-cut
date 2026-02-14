"""Evaluate WASB HRNet vs VballNet on ball tracking ground truth.

WASB (Widely Applicable Strong Baseline, BMVC 2023) uses an HRNet backbone
pretrained on indoor volleyball (40 training + 16 test matches). This script
evaluates whether it outperforms VballNet on our beach volleyball GT rallies.

Setup:
    1. Download WASB volleyball weights:
       gdown 1M9y4wPJqLc0K-z-Bo5DP8Ft5XwJuLqIS -O weights/wasb/wasb_volleyball_best.pth.tar

    2. Run evaluation:
       cd analysis
       uv run python scripts/eval_wasb.py

    3. Compare filtered (with ball filter pipeline):
       uv run python scripts/eval_wasb.py --filtered

    4. Use different heatmap threshold:
       uv run python scripts/eval_wasb.py --threshold 0.3
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from rich.console import Console
from rich.table import Table

from rallycut.evaluation.tracking.ball_grid_search import (
    BallRawCache,
    apply_ball_filter_config,
)
from rallycut.evaluation.tracking.ball_metrics import (
    evaluate_ball_tracking,
    find_optimal_frame_offset,
)
from rallycut.evaluation.tracking.db import (
    get_video_path,
    load_labeled_rallies,
)
from rallycut.tracking.ball_filter import BallFilterConfig
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.wasb_model import (
    NUM_INPUT_FRAMES,
    decode_heatmap_wasb,
    load_wasb_model,
    preprocess_frame,
)

console = Console()


def run_wasb_inference(
    model: torch.nn.Module,
    video_path: Path,
    start_ms: int,
    end_ms: int,
    device: str = "cpu",
    threshold: float = 0.5,
) -> list[BallPosition]:
    """Run WASB inference on a video segment, returning BallPosition list.

    Uses 3-frame sliding window with stride 1 for maximum coverage.
    For each frame, keeps the highest-confidence detection across overlapping windows.
    """
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(start_ms / 1000 * fps)
    end_frame = min(int(end_ms / 1000 * fps), total_frames)

    # Seek to start
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Read all frames in the rally segment
    frames: list[np.ndarray] = []
    frame_numbers: list[int] = []
    for frame_idx in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)  # Keep raw BGR for preprocessing
        frame_numbers.append(frame_idx - start_frame)  # Rally-relative

    cap.release()

    if len(frames) < NUM_INPUT_FRAMES:
        return []

    # Track best confidence detection per frame (overlapping windows)
    frame_detections: dict[int, BallPosition] = {}

    with torch.no_grad():
        for i in range(len(frames) - NUM_INPUT_FRAMES + 1):
            # Preprocess 3 consecutive frames and concatenate (9 channels)
            preprocessed = [preprocess_frame(frames[i + j]) for j in range(NUM_INPUT_FRAMES)]
            triplet = np.concatenate(preprocessed, axis=0)  # (9, H, W)
            x = torch.from_numpy(triplet).float().unsqueeze(0).to(device)  # (1, 9, H, W)

            # Forward pass → {0: (1, 3, H, W)} raw logits
            output = model(x)
            heatmaps = output[0][0].cpu().numpy()  # (3, H, W) raw logits

            # Decode each of the 3 output heatmaps
            for j in range(NUM_INPUT_FRAMES):
                frame_idx = i + j
                if frame_idx >= len(frame_numbers):
                    break

                x_norm, y_norm, conf = decode_heatmap_wasb(heatmaps[j], threshold=threshold)

                if conf > 0.0:
                    bp = BallPosition(
                        frame_number=frame_numbers[frame_idx],
                        x=x_norm,
                        y=y_norm,
                        confidence=conf,
                        motion_energy=0.0,
                    )
                    # Keep highest confidence detection per frame
                    existing = frame_detections.get(frame_idx)
                    if existing is None or conf > existing.confidence:
                        frame_detections[frame_idx] = bp

    # Convert to sorted list
    return [frame_detections[k] for k in sorted(frame_detections.keys())]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate WASB HRNet vs VballNet")
    parser.add_argument(
        "--weights",
        default=None,
        help="Path to WASB weights (default: weights/wasb/wasb_volleyball_best.pth.tar)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Heatmap score threshold (WASB default: 0.5, try 0.3 for higher recall)",
    )
    parser.add_argument(
        "--filtered",
        action="store_true",
        help="Apply ball filter pipeline to WASB output (like VballNet uses)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for inference (cpu, cuda, mps)",
    )
    parser.add_argument(
        "--rally",
        default=None,
        help="Evaluate specific rally ID (prefix match)",
    )
    args = parser.parse_args()

    # Load WASB model
    try:
        model = load_wasb_model(args.weights, device=args.device)
    except FileNotFoundError as e:
        console.print(f"[red]{e}[/red]")
        console.print()
        console.print("[bold]To download WASB volleyball weights:[/bold]")
        console.print("  cd analysis")
        console.print("  mkdir -p weights/wasb")
        console.print(
            "  gdown 1M9y4wPJqLc0K-z-Bo5DP8Ft5XwJuLqIS "
            "-O weights/wasb/wasb_volleyball_best.pth.tar"
        )
        return

    param_count = sum(p.numel() for p in model.parameters())
    console.print(f"WASB HRNet: {param_count / 1e6:.1f}M parameters")
    console.print(f"Heatmap threshold: {args.threshold}")
    console.print(f"Filter pipeline: {'enabled' if args.filtered else 'disabled (raw)'}")
    console.print(f"Device: {args.device}")
    print()

    # Load rallies with ball GT
    rallies = load_labeled_rallies()
    if not rallies:
        console.print("[red]No rallies with ball GT found[/red]")
        return

    # Filter to specific rally if requested
    if args.rally:
        rallies = [r for r in rallies if r.rally_id.startswith(args.rally)]
        if not rallies:
            console.print(f"[red]No rally matching '{args.rally}'[/red]")
            return

    console.print(f"Found {len(rallies)} rallies with ball ground truth")
    print()

    # Collect results
    vballnet_results: list[tuple[str, str, dict[str, float]]] = []
    wasb_results: list[tuple[str, str, dict[str, float]]] = []

    raw_cache = BallRawCache()
    filter_config = BallFilterConfig()

    total_wasb_time = 0.0
    total_frames = 0

    for rally in rallies:
        rally_label = rally.rally_id[:8]
        console.print(f"[bold]Rally {rally_label}[/bold] (video {rally.video_id[:8]})")

        # --- VballNet baseline ---
        cached = raw_cache.get(rally.rally_id)
        if cached is not None:
            vballnet_preds = apply_ball_filter_config(cached.raw_ball_positions, filter_config)
        elif rally.predictions is not None and rally.predictions.ball_positions:
            vballnet_preds = rally.predictions.ball_positions
        else:
            console.print("  [yellow]No VballNet predictions, skipping[/yellow]")
            continue

        vball_metrics = evaluate_ball_tracking(
            ground_truth=rally.ground_truth.positions,
            predictions=vballnet_preds,
            video_width=rally.video_width,
            video_height=rally.video_height,
            video_fps=rally.video_fps,
        )
        vballnet_results.append((
            rally.rally_id,
            rally.video_id,
            {
                "detection": vball_metrics.detection_rate,
                "match": vball_metrics.match_rate,
                "mean_err": vball_metrics.mean_error_px,
                "gt_frames": vball_metrics.num_gt_frames,
            },
        ))

        # --- WASB inference ---
        video_path = get_video_path(rally.video_id)
        if video_path is None:
            console.print("  [yellow]Video not available, skipping WASB[/yellow]")
            wasb_results.append((
                rally.rally_id,
                rally.video_id,
                {"detection": 0.0, "match": 0.0, "mean_err": 0.0, "gt_frames": 0},
            ))
            continue

        t0 = time.time()
        wasb_preds_raw = run_wasb_inference(
            model, video_path, rally.start_ms, rally.end_ms,
            device=args.device, threshold=args.threshold,
        )
        inference_time = time.time() - t0
        total_wasb_time += inference_time

        rally_frames = int((rally.end_ms - rally.start_ms) / 1000 * rally.video_fps)
        total_frames += rally_frames

        # Optionally apply ball filter pipeline
        if args.filtered:
            wasb_preds = apply_ball_filter_config(wasb_preds_raw, filter_config)
        else:
            wasb_preds = wasb_preds_raw

        # Find optimal frame offset for WASB
        best_offset, _ = find_optimal_frame_offset(
            rally.ground_truth.positions, wasb_preds, rally.video_width, rally.video_height
        )

        # Apply offset if found
        if best_offset > 0:
            wasb_preds_shifted = [
                BallPosition(
                    frame_number=p.frame_number - best_offset,
                    x=p.x, y=p.y,
                    confidence=p.confidence,
                    motion_energy=p.motion_energy,
                )
                for p in wasb_preds
            ]
        else:
            wasb_preds_shifted = wasb_preds

        wasb_metrics = evaluate_ball_tracking(
            ground_truth=rally.ground_truth.positions,
            predictions=wasb_preds_shifted,
            video_width=rally.video_width,
            video_height=rally.video_height,
            video_fps=rally.video_fps,
        )
        wasb_results.append((
            rally.rally_id,
            rally.video_id,
            {
                "detection": wasb_metrics.detection_rate,
                "match": wasb_metrics.match_rate,
                "mean_err": wasb_metrics.mean_error_px,
                "gt_frames": wasb_metrics.num_gt_frames,
            },
        ))

        # Print per-rally results
        console.print(
            f"  VballNet:  det={vball_metrics.detection_rate:.1%}  "
            f"match={vball_metrics.match_rate:.1%}  "
            f"err={vball_metrics.mean_error_px:.1f}px"
        )
        console.print(
            f"  WASB:      det={wasb_metrics.detection_rate:.1%}  "
            f"match={wasb_metrics.match_rate:.1%}  "
            f"err={wasb_metrics.mean_error_px:.1f}px  "
            f"({inference_time:.1f}s, {len(wasb_preds_raw)} raw"
            + (f" → {len(wasb_preds)} filtered" if args.filtered else "")
            + f", offset={best_offset})"
        )

        # Highlight improvements/regressions
        match_diff = wasb_metrics.match_rate - vball_metrics.match_rate
        if match_diff > 0.05:
            console.print(f"  [green]  WASB +{match_diff:.1%} match rate[/green]")
        elif match_diff < -0.05:
            console.print(f"  [red]  WASB {match_diff:.1%} match rate[/red]")
        print()

    # --- Summary comparison table ---
    if not vballnet_results:
        console.print("[red]No results to compare[/red]")
        return

    console.print("[bold]== Summary Comparison ==[/bold]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Rally")
    table.add_column("Video")
    table.add_column("GT Fr", justify="right")
    table.add_column("VNet Det%", justify="right")
    table.add_column("WASB Det%", justify="right")
    table.add_column("VNet Match%", justify="right")
    table.add_column("WASB Match%", justify="right")
    table.add_column("VNet Err", justify="right")
    table.add_column("WASB Err", justify="right")
    table.add_column("Winner")

    total_v_det = total_w_det = 0.0
    total_v_match = total_w_match = 0.0
    total_v_err = total_w_err = 0.0
    count = 0
    wasb_wins = vnet_wins = ties = 0

    for (rid, vid, vr), (_, _, wr) in zip(vballnet_results, wasb_results):
        if vr["match"] > wr["match"] + 0.005:
            winner = "[blue]VballNet[/blue]"
            vnet_wins += 1
        elif wr["match"] > vr["match"] + 0.005:
            winner = "[green]WASB[/green]"
            wasb_wins += 1
        else:
            winner = "Tie"
            ties += 1

        table.add_row(
            rid[:8],
            vid[:8],
            str(vr["gt_frames"]),
            f"{vr['detection']:.1%}",
            f"{wr['detection']:.1%}",
            f"{vr['match']:.1%}",
            f"{wr['match']:.1%}",
            f"{vr['mean_err']:.1f}px",
            f"{wr['mean_err']:.1f}px" if wr["mean_err"] > 0 else "N/A",
            winner,
        )

        total_v_det += vr["detection"]
        total_w_det += wr["detection"]
        total_v_match += vr["match"]
        total_w_match += wr["match"]
        total_v_err += vr["mean_err"]
        total_w_err += wr["mean_err"]
        count += 1

    if count > 0:
        avg_v_match = total_v_match / count
        avg_w_match = total_w_match / count
        if avg_w_match > avg_v_match + 0.005:
            avg_winner = "[green]WASB[/green]"
        elif avg_v_match > avg_w_match + 0.005:
            avg_winner = "[blue]VballNet[/blue]"
        else:
            avg_winner = "Tie"

        table.add_row(
            "[bold]Average",
            "",
            "",
            f"[bold]{total_v_det / count:.1%}",
            f"[bold]{total_w_det / count:.1%}",
            f"[bold]{total_v_match / count:.1%}",
            f"[bold]{total_w_match / count:.1%}",
            f"[bold]{total_v_err / count:.1f}px",
            f"[bold]{total_w_err / count:.1f}px",
            avg_winner,
        )

    console.print(table)

    # Summary stats
    print()
    console.print(f"WASB wins: {wasb_wins}/{count}  |  VballNet wins: {vnet_wins}/{count}  |  Ties: {ties}/{count}")
    if total_frames > 0 and total_wasb_time > 0:
        console.print(f"WASB speed: {total_frames / total_wasb_time:.1f} FPS ({args.device})")
    console.print(
        f"Average match rate: VballNet={total_v_match / count:.1%}  "
        f"WASB={total_w_match / count:.1%}  "
        f"(delta={((total_w_match - total_v_match) / count):+.1%})"
    )


if __name__ == "__main__":
    main()
