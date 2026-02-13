"""Evaluate TrackNet vs VballNet on ball tracking ground truth.

Usage:
    cd analysis
    uv run python scripts/eval_tracknet.py
    uv run python scripts/eval_tracknet.py --model last  # Use last.pt instead of best.pt
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
from rallycut.evaluation.tracking.ball_metrics import evaluate_ball_tracking
from rallycut.evaluation.tracking.db import (
    get_video_path,
    load_labeled_rallies,
)
from rallycut.tracking.ball_filter import BallFilterConfig
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.training.tracknet import IMG_HEIGHT, IMG_WIDTH, TrackNet

console = Console()

WEIGHTS_DIR = Path(__file__).parent.parent / "weights" / "tracknet"


def load_tracknet(model_name: str = "best") -> TrackNet:
    """Load trained TrackNet model."""
    model_path = WEIGHTS_DIR / f"{model_name}.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"TrackNet weights not found at {model_path}")

    model = TrackNet()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    console.print(f"Loaded TrackNet from {model_path.name}")
    return model


def decode_heatmap(heatmap: np.ndarray, threshold: float = 0.3) -> tuple[float, float, float]:
    """Decode a single heatmap to (x_norm, y_norm, confidence).

    Same logic as VballNet's contour-based centroid decoding.
    """
    max_val = float(heatmap.max())
    if max_val < threshold:
        return 0.5, 0.5, 0.0

    # Binarize and find contours
    binary = (heatmap > threshold).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return 0.5, 0.5, 0.0

    largest = max(contours, key=cv2.contourArea) if len(contours) > 1 else contours[0]
    moments = cv2.moments(largest)
    if moments["m00"] == 0:
        return 0.5, 0.5, 0.0

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    x_norm = max(0.0, min(1.0, cx / heatmap.shape[1]))
    y_norm = max(0.0, min(1.0, cy / heatmap.shape[0]))

    return x_norm, y_norm, max_val


def run_tracknet_inference(
    model: TrackNet,
    video_path: Path,
    start_ms: int,
    end_ms: int,
) -> list[BallPosition]:
    """Run TrackNet inference on a video segment, returning BallPosition list."""
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
        # Resize to TrackNet input size (RGB, HxW = 288x512)
        resized = cv2.resize(frame, (IMG_WIDTH, IMG_HEIGHT))
        frames.append(resized)
        # Rally-relative frame number (0-indexed)
        frame_numbers.append(frame_idx - start_frame)

    cap.release()

    if len(frames) < 3:
        return []

    # Run inference with 3-frame sliding window (stride=1 for max coverage)
    # Track best confidence detection per frame (overlapping windows produce duplicates)
    frame_detections: dict[int, BallPosition] = {}

    with torch.no_grad():
        for i in range(len(frames) - 2):
            # Stack 3 consecutive RGB frames → (9, H, W)
            triplet = np.concatenate(
                [frames[i].transpose(2, 0, 1),
                 frames[i + 1].transpose(2, 0, 1),
                 frames[i + 2].transpose(2, 0, 1)],
                axis=0,
            )
            x = torch.from_numpy(triplet).float().unsqueeze(0) / 255.0  # (1, 9, H, W)

            # Forward pass → (1, 3, H, W)
            output = model(x)
            heatmaps = output[0].cpu().numpy()  # (3, H, W)

            # Decode each of the 3 output heatmaps
            for j in range(3):
                frame_idx = i + j
                if frame_idx >= len(frame_numbers):
                    break
                x_norm, y_norm, conf = decode_heatmap(heatmaps[j])

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
    positions = [frame_detections[k] for k in sorted(frame_detections.keys())]
    return positions


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate TrackNet vs VballNet")
    parser.add_argument("--model", default="best", help="Model name: best or last")
    args = parser.parse_args()

    # Load model
    model = load_tracknet(args.model)

    # Load rallies with ball GT
    rallies = load_labeled_rallies()
    if not rallies:
        console.print("[red]No rallies with ball GT found[/red]")
        return

    console.print(f"Found {len(rallies)} rallies with ball ground truth\n")

    # Collect results for comparison table
    vballnet_results: list[tuple[str, dict[str, float]]] = []
    tracknet_results: list[tuple[str, dict[str, float]]] = []

    raw_cache = BallRawCache()

    for rally in rallies:
        rally_label = f"{rally.rally_id[:8]}..."
        console.print(f"[bold]Rally {rally_label}[/bold] (video {rally.video_id[:8]})")

        # --- VballNet baseline ---
        cached = raw_cache.get(rally.rally_id)
        if cached is not None:
            vballnet_preds = apply_ball_filter_config(
                cached.raw_ball_positions, BallFilterConfig()
            )
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
        vballnet_results.append((rally.rally_id, {
            "detection": vball_metrics.detection_rate,
            "match": vball_metrics.match_rate,
            "mean_err": vball_metrics.mean_error_px,
        }))

        # --- TrackNet inference ---
        video_path = get_video_path(rally.video_id)
        if video_path is None:
            console.print("  [yellow]Video not available, skipping TrackNet[/yellow]")
            tracknet_results.append((rally.rally_id, {
                "detection": 0.0, "match": 0.0, "mean_err": 0.0,
            }))
            continue

        t0 = time.time()
        tracknet_preds = run_tracknet_inference(
            model, video_path, rally.start_ms, rally.end_ms
        )
        inference_time = time.time() - t0

        # Apply same ball filter pipeline
        tracknet_filtered = apply_ball_filter_config(tracknet_preds, BallFilterConfig())

        tracknet_metrics = evaluate_ball_tracking(
            ground_truth=rally.ground_truth.positions,
            predictions=tracknet_filtered,
            video_width=rally.video_width,
            video_height=rally.video_height,
            video_fps=rally.video_fps,
        )
        tracknet_results.append((rally.rally_id, {
            "detection": tracknet_metrics.detection_rate,
            "match": tracknet_metrics.match_rate,
            "mean_err": tracknet_metrics.mean_error_px,
        }))

        console.print(
            f"  VballNet:  det={vball_metrics.detection_rate:.1%}  "
            f"match={vball_metrics.match_rate:.1%}  "
            f"err={vball_metrics.mean_error_px:.1f}px"
        )
        console.print(
            f"  TrackNet:  det={tracknet_metrics.detection_rate:.1%}  "
            f"match={tracknet_metrics.match_rate:.1%}  "
            f"err={tracknet_metrics.mean_error_px:.1f}px  "
            f"({inference_time:.1f}s, {len(tracknet_preds)} raw → {len(tracknet_filtered)} filtered)"
        )
        print()

    # Summary comparison table
    console.print("\n[bold]== Summary Comparison ==[/bold]")
    table = Table(show_header=True, header_style="bold")
    table.add_column("Rally")
    table.add_column("VballNet Det%", justify="right")
    table.add_column("TrackNet Det%", justify="right")
    table.add_column("VballNet Match%", justify="right")
    table.add_column("TrackNet Match%", justify="right")
    table.add_column("VballNet Err", justify="right")
    table.add_column("TrackNet Err", justify="right")
    table.add_column("Winner")

    total_v_det = total_t_det = 0.0
    total_v_match = total_t_match = 0.0
    total_v_err = total_t_err = 0.0
    count = 0

    for (rid, vr), (_, tr) in zip(vballnet_results, tracknet_results):
        winner = ""
        if tr["match"] > vr["match"]:
            winner = "[green]TrackNet[/green]"
        elif vr["match"] > tr["match"]:
            winner = "[blue]VballNet[/blue]"
        else:
            winner = "Tie"

        table.add_row(
            rid[:8],
            f"{vr['detection']:.1%}",
            f"{tr['detection']:.1%}",
            f"{vr['match']:.1%}",
            f"{tr['match']:.1%}",
            f"{vr['mean_err']:.1f}px",
            f"{tr['mean_err']:.1f}px" if tr['mean_err'] > 0 else "N/A",
            winner,
        )
        total_v_det += vr["detection"]
        total_t_det += tr["detection"]
        total_v_match += vr["match"]
        total_t_match += tr["match"]
        total_v_err += vr["mean_err"]
        total_t_err += tr["mean_err"]
        count += 1

    if count > 0:
        avg_winner = ""
        avg_v_match = total_v_match / count
        avg_t_match = total_t_match / count
        if avg_t_match > avg_v_match:
            avg_winner = "[green]TrackNet[/green]"
        else:
            avg_winner = "[blue]VballNet[/blue]"

        table.add_row(
            "[bold]Average",
            f"[bold]{total_v_det / count:.1%}",
            f"[bold]{total_t_det / count:.1%}",
            f"[bold]{total_v_match / count:.1%}",
            f"[bold]{total_t_match / count:.1%}",
            f"[bold]{total_v_err / count:.1f}px",
            f"[bold]{total_t_err / count:.1f}px",
            avg_winner,
        )

    console.print(table)


if __name__ == "__main__":
    main()
