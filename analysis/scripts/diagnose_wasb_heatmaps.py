"""Diagnose WASB heatmap values for a rally.

Shows the raw heatmap peak (max value) per frame before any thresholding.
This reveals whether WASB "sees" the ball but at sub-threshold confidence,
or genuinely produces no signal.

Usage:
    uv run python scripts/diagnose_wasb_heatmaps.py <video_path> [--start-ms N] [--end-ms N]

Example:
    uv run python scripts/diagnose_wasb_heatmaps.py ~/Desktop/rallies/video.mp4 --start-ms 5000 --end-ms 20000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from rallycut.tracking.wasb_model import (
    NUM_INPUT_FRAMES,
    WASBBallTracker,
    preprocess_frame,
)


def diagnose(
    video_path: str,
    start_ms: int | None = None,
    end_ms: int | None = None,
    threshold: float = 0.3,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: Could not open video: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = 0
    end_frame = total_frames
    if start_ms is not None:
        start_frame = int(start_ms / 1000 * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    if end_ms is not None:
        end_frame = min(int(end_ms / 1000 * fps), total_frames)

    frames_to_process = end_frame - start_frame
    print(f"Video: {video_path}")
    print(f"  {width}x{height} @ {fps:.1f}fps, {total_frames} total frames")
    print(f"  Processing frames {start_frame}-{end_frame} ({frames_to_process} frames)")
    print(f"  Threshold: {threshold}")

    # Read frames
    print("\nReading frames...")
    raw_frames: list[np.ndarray] = []
    for _ in range(frames_to_process):
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
    cap.release()

    num_frames = len(raw_frames)
    print(f"  Read {num_frames} frames")

    if num_frames < NUM_INPUT_FRAMES:
        print("ERROR: Not enough frames")
        sys.exit(1)

    # Preprocess
    print("Preprocessing...")
    preprocessed = [preprocess_frame(f) for f in raw_frames]
    del raw_frames

    # Load WASB model
    print("Loading WASB model...")
    tracker = WASBBallTracker(threshold=threshold)
    onnx_session = None
    try:
        onnx_session = tracker._load_onnx_session()
        onnx_input_name = onnx_session.get_inputs()[0].name
        print("  Using ONNX backend")
    except Exception:
        print("  Using PyTorch backend")

    # Run inference and collect raw heatmap peaks
    print("\nRunning inference...")
    start_time = time.time()
    num_windows = num_frames - NUM_INPUT_FRAMES + 1

    # Per-frame: track max heatmap value and best detection position
    frame_max_vals: dict[int, float] = {}
    frame_best_pos: dict[int, tuple[float, float, float]] = {}  # (x, y, conf)

    for window_start in range(num_windows):
        # Build input
        inp = np.concatenate(
            [preprocessed[window_start + j] for j in range(NUM_INPUT_FRAMES)],
            axis=0,
        )
        inp = inp[np.newaxis]  # (1, 9, H, W)

        if onnx_session is not None:
            logits = onnx_session.run(None, {onnx_input_name: inp})[0]
            probs = (1.0 / (1.0 + np.exp(-logits.astype(np.float64)))).astype(np.float32)
        else:
            import torch
            model = tracker._ensure_model()
            with torch.inference_mode():
                x = torch.from_numpy(inp).float().to(tracker.device)
                output = model(x)
                probs = torch.sigmoid(output[0]).cpu().numpy()

        # Decode each of the 3 output heatmaps
        for j in range(NUM_INPUT_FRAMES):
            frame_idx = window_start + j
            heatmap = probs[0, j]  # (H, W)
            max_val = float(heatmap.max())

            # Track the maximum heatmap value across all windows for this frame
            if frame_idx not in frame_max_vals or max_val > frame_max_vals[frame_idx]:
                frame_max_vals[frame_idx] = max_val

                # Find position of max
                flat_idx = int(heatmap.argmax())
                h, w = heatmap.shape
                y_idx = flat_idx // w
                x_idx = flat_idx % w
                x_norm = x_idx / w
                y_norm = y_idx / h
                frame_best_pos[frame_idx] = (x_norm, y_norm, max_val)

        if window_start % 50 == 0:
            print(f"  Window {window_start}/{num_windows}...")

    elapsed = time.time() - start_time
    print(f"  Done in {elapsed:.1f}s")

    # Print results
    print(f"\n{'='*80}")
    print(f"Raw heatmap peak values per frame (threshold={threshold}):")
    print(f"{'='*80}")
    print(f"{'Frame':>6}  {'MaxVal':>7}  {'Status':>10}  {'Position (x, y)':>20}")
    print(f"{'─'*6}  {'─'*7}  {'─'*10}  {'─'*20}")

    above_threshold = 0
    below_threshold = 0
    near_threshold = 0  # 0.2-0.3 range

    for frame_idx in range(num_frames):
        max_val = frame_max_vals.get(frame_idx, 0.0)
        pos = frame_best_pos.get(frame_idx, (0.5, 0.5, 0.0))

        if max_val > threshold:
            status = "DETECTED"
            above_threshold += 1
        elif max_val > threshold * 0.67:  # 0.2-0.3 range
            status = "NEAR"
            near_threshold += 1
        else:
            status = "BELOW"
            below_threshold += 1

        print(
            f"{frame_idx:>6}  {max_val:>7.4f}  {status:>10}  "
            f"({pos[0]:.3f}, {pos[1]:.3f})"
        )

    print(f"\n{'='*80}")
    print("Summary:")
    print(f"  Above threshold (>{threshold}):  {above_threshold}/{num_frames} ({above_threshold/num_frames*100:.1f}%)")
    print(f"  Near threshold ({threshold*0.67:.2f}-{threshold}): {near_threshold}/{num_frames} ({near_threshold/num_frames*100:.1f}%)")
    print(f"  Below ({threshold*0.67:.2f}):              {below_threshold}/{num_frames} ({below_threshold/num_frames*100:.1f}%)")

    # Show distribution
    vals = sorted(frame_max_vals.values())
    if vals:
        print("\n  Distribution of max heatmap values:")
        print(f"    Min:    {vals[0]:.4f}")
        print(f"    P10:    {vals[int(len(vals)*0.1)]:.4f}")
        print(f"    P25:    {vals[int(len(vals)*0.25)]:.4f}")
        print(f"    Median: {vals[int(len(vals)*0.5)]:.4f}")
        print(f"    P75:    {vals[int(len(vals)*0.75)]:.4f}")
        print(f"    P90:    {vals[int(len(vals)*0.9)]:.4f}")
        print(f"    Max:    {vals[-1]:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", help="Video file path")
    parser.add_argument("--start-ms", type=int, default=None, help="Start time in ms")
    parser.add_argument("--end-ms", type=int, default=None, help="End time in ms")
    parser.add_argument("--threshold", type=float, default=0.3, help="Heatmap threshold")
    args = parser.parse_args()

    diagnose(
        video_path=args.video,
        start_ms=args.start_ms,
        end_ms=args.end_ms,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
