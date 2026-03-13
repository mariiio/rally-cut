#!/usr/bin/env python3
"""Extract ball detection density for full videos and cache results.

Runs WASB ball detection on full videos (unfiltered) and saves per-frame
detection data. Used by ball trajectory fusion for rally detection.

Usage:
    uv run python scripts/extract_ball_density.py          # All GT videos
    uv run python scripts/extract_ball_density.py --video-id abc123  # Single video
    uv run python scripts/extract_ball_density.py --list    # Show cached status
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from rallycut.evaluation.ground_truth import load_evaluation_videos
from rallycut.evaluation.tracking.db import get_video_path

CACHE_DIR = Path("training_data/ball_density")


def get_cache_path(video_id: str) -> Path:
    return CACHE_DIR / f"{video_id}.npz"


def is_cached(video_id: str) -> bool:
    return get_cache_path(video_id).exists()


def extract_ball_density(video_id: str, video_path: Path, chunk_s: float = 60.0) -> dict:
    """Run WASB on full video in chunks and return per-frame detection data.

    Processes video in chunks to avoid OOM from loading all frames at once.
    """
    import cv2

    from rallycut.tracking.ball_tracker import create_ball_tracker

    tracker = create_ball_tracker()
    t0 = time.time()

    # Get video metadata
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    duration_ms = int(n_frames / fps * 1000)
    chunk_ms = int(chunk_s * 1000)

    confidences = np.zeros(n_frames, dtype=np.float32)

    # Process in chunks to limit memory usage
    start_ms = 0
    chunk_idx = 0
    while start_ms < duration_ms:
        end_ms = min(start_ms + chunk_ms, duration_ms)
        chunk_idx += 1
        print(f"  chunk {chunk_idx} ({start_ms // 1000}-{end_ms // 1000}s)...", end=" ", flush=True)

        result = tracker.track_video(
            video_path=video_path,
            start_ms=start_ms,
            end_ms=end_ms,
            enable_filtering=False,
        )

        start_frame = int(start_ms / 1000 * fps)
        for p in result.positions:
            abs_frame = start_frame + p.frame_number
            if 0 <= abs_frame < n_frames:
                confidences[abs_frame] = max(confidences[abs_frame], p.confidence)

        print(f"{len(result.positions)} dets", flush=True)
        start_ms = end_ms

    elapsed = time.time() - t0

    return {
        "confidences": confidences,
        "fps": fps,
        "n_frames": n_frames,
        "video_width": video_width,
        "video_height": video_height,
        "elapsed_s": elapsed,
    }


def save_density(video_id: str, data: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = get_cache_path(video_id)
    np.savez_compressed(
        path,
        confidences=data["confidences"],
        fps=np.float32(data["fps"]),
        n_frames=np.int32(data["n_frames"]),
        video_width=np.int32(data["video_width"]),
        video_height=np.int32(data["video_height"]),
    )


def load_density(video_id: str) -> dict | None:
    path = get_cache_path(video_id)
    if not path.exists():
        return None
    data = np.load(path)
    return {
        "confidences": data["confidences"],
        "fps": float(data["fps"]),
        "n_frames": int(data["n_frames"]),
    }


def compute_density_curve(
    confidences: np.ndarray,
    fps: float,
    window_s: float = 2.0,
    conf_threshold: float = 0.3,
) -> np.ndarray:
    """Compute per-second ball detection density from raw frame confidences.

    Uses cumulative sum for O(n) computation. Returns array of detection
    rates (0-1), one per second of video.
    """
    window_frames = max(1, int(window_s * fps))
    half_win = window_frames // 2

    detected = (confidences >= conf_threshold).astype(np.float32)
    cumsum = np.concatenate([[0.0], np.cumsum(detected)])
    duration_s = len(confidences) / fps
    n_seconds = int(np.ceil(duration_s))

    density = np.zeros(n_seconds, dtype=np.float32)
    for s in range(n_seconds):
        center_frame = int((s + 0.5) * fps)
        start = max(0, center_frame - half_win)
        end = min(len(detected), start + window_frames)
        if end > start:
            density[s] = (cumsum[end] - cumsum[start]) / (end - start)

    return density


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract ball density for GT videos")
    parser.add_argument("--video-id", type=str, help="Single video ID to process")
    parser.add_argument("--list", action="store_true", help="Show cached status")
    parser.add_argument("--force", action="store_true", help="Overwrite existing cache")
    args = parser.parse_args()

    videos = load_evaluation_videos()
    print(f"Found {len(videos)} GT videos")

    if args.list:
        cached = sum(1 for v in videos if is_cached(v.id))
        print(f"Cached: {cached}/{len(videos)}")
        for v in videos:
            status = "cached" if is_cached(v.id) else "missing"
            print(f"  [{status}] {v.filename} ({v.id[:8]})")
        return

    to_process = videos
    if args.video_id:
        to_process = [v for v in videos if v.id.startswith(args.video_id)]
        if not to_process:
            print(f"Video {args.video_id} not found in GT videos")
            return

    skipped = 0
    processed = 0
    failed = 0

    for i, v in enumerate(to_process):
        if is_cached(v.id) and not args.force:
            skipped += 1
            continue

        video_path = get_video_path(v.id)
        if video_path is None:
            print(f"[{i+1}/{len(to_process)}] {v.filename}: video file not found, skipping")
            failed += 1
            continue

        print(f"[{i+1}/{len(to_process)}] {v.filename}...", end=" ", flush=True)
        try:
            data = extract_ball_density(v.id, video_path)
            save_density(v.id, data)

            det_rate = (data["confidences"] >= 0.3).mean()
            print(
                f"done in {data['elapsed_s']:.1f}s "
                f"({data['n_frames']} frames, {data['fps']:.1f} fps, "
                f"det_rate={det_rate:.1%})"
            )
            processed += 1
        except Exception as e:
            print(f"FAILED: {e}")
            failed += 1

    print(f"\nProcessed: {processed}, Skipped (cached): {skipped}, Failed: {failed}")


if __name__ == "__main__":
    main()
