#!/usr/bin/env python3
"""Export YOLO-pose training dataset for court keypoint detection.

Extracts frames from videos with court calibration GT in the DB and creates
a YOLO-pose format dataset with 4 keypoints (court corners).

Corner order: near-left, near-right, far-right, far-left
(matches CourtCalibrator input format)

Handles off-screen near corners by padding images at the bottom.

Usage:
    uv run python scripts/export_court_keypoint_dataset.py
    uv run python scripts/export_court_keypoint_dataset.py --output-dir datasets/court_keypoints
    uv run python scripts/export_court_keypoint_dataset.py --frames-per-video 20
    uv run python scripts/export_court_keypoint_dataset.py --pad-ratio 0.3  # 30% bottom padding
    uv run python scripts/export_court_keypoint_dataset.py --val-split 0.2  # 20% validation
    uv run python scripts/export_court_keypoint_dataset.py --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

CORNER_NAMES = ["near-left", "near-right", "far-right", "far-left"]

# Horizontal flip swaps left/right pairs
FLIP_IDX = [1, 0, 3, 2]


def load_calibrated_videos() -> list[dict[str, Any]]:
    """Load all videos with court calibration from the DB."""
    from rallycut.evaluation.db import get_connection

    query = """
        SELECT id, court_calibration_json, width, height, s3_key, content_hash
        FROM videos
        WHERE court_calibration_json IS NOT NULL
          AND deleted_at IS NULL
    """

    videos = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            for row in cur.fetchall():
                vid_id, cal_json, width, height, s3_key, content_hash = row
                if not isinstance(cal_json, list) or len(cal_json) != 4:
                    continue
                videos.append({
                    "id": str(vid_id),
                    "corners": cal_json,
                    "width": width or 1920,
                    "height": height or 1080,
                    "s3_key": s3_key,
                    "content_hash": content_hash,
                })

    return videos


def extract_frames(
    video_path: Path,
    n_frames: int,
    seed: int = 42,
) -> list[np.ndarray]:
    """Extract evenly-spaced frames from a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning(f"Cannot open video: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    # Sample evenly-spaced frames with small random jitter
    rng = random.Random(seed)
    step = max(1, total_frames // (n_frames + 1))
    frame_indices = []
    for i in range(1, n_frames + 1):
        base = i * step
        jitter = rng.randint(-step // 4, step // 4)
        idx = max(0, min(total_frames - 1, base + jitter))
        frame_indices.append(idx)

    # Remove duplicates and sort
    frame_indices = sorted(set(frame_indices))

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames


def pad_frame_bottom(frame: np.ndarray, pad_ratio: float) -> np.ndarray:
    """Add black padding to the bottom of a frame."""
    h, w = frame.shape[:2]
    pad_h = int(h * pad_ratio)
    padding = np.zeros((pad_h, w, 3), dtype=frame.dtype)
    return np.vstack([frame, padding])


def corners_to_yolo_pose(
    corners: list[dict[str, float]],
    orig_height: int,
    pad_ratio: float,
) -> str | None:
    """Convert 4 court corners to YOLO-pose annotation format.

    Args:
        corners: 4 corners [{x, y}] in normalized coords (may exceed [0,1]).
        orig_height: Original image height before padding.
        pad_ratio: Bottom padding ratio applied.

    Returns:
        YOLO-pose format string, or None if annotation is invalid.
    """
    # Rescale Y coordinates to account for padding
    # Original y was normalized to orig_height, new total is orig_height * (1 + pad_ratio)
    scale_y = 1.0 / (1.0 + pad_ratio)

    keypoints = []
    for i, corner in enumerate(corners):
        x = corner["x"]
        y = corner["y"] * scale_y  # Rescale to padded image

        # Determine visibility
        # scale_y marks boundary between original image and padding in padded coords
        # 2 = visible (in original image), 1 = occluded (in padding zone), 0 = off-canvas
        if 0 <= x <= 1 and 0 <= y <= scale_y:
            vis = 2
        elif 0 <= x <= 1 and y <= 1.0:
            # In padded area — labeled but not visible in original image
            vis = 1
        else:
            # Beyond padded canvas
            vis = 0

        # Clamp to [0, 1] for YOLO format
        x_clamped = max(0.0, min(1.0, x))
        y_clamped = max(0.0, min(1.0, y))
        keypoints.append((x_clamped, y_clamped, vis))

    # Compute bounding box from corners (in padded space)
    xs = [c["x"] for c in corners]
    ys = [c["y"] * scale_y for c in corners]

    # Clamp bbox to [0, 1]
    x_min = max(0.0, min(xs))
    x_max = min(1.0, max(xs))
    y_min = max(0.0, min(ys))
    y_max = min(1.0, max(ys))

    # Bbox center + size
    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    bw = x_max - x_min
    bh = y_max - y_min

    if bw < 0.05 or bh < 0.05:
        return None  # Degenerate bbox

    # Format: class cx cy w h kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ...
    parts = [f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"]
    for x, y, v in keypoints:
        parts.append(f"{x:.6f} {y:.6f} {v}")

    return " ".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export YOLO-pose court keypoint dataset"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("datasets/court_keypoints"),
        help="Output directory for dataset",
    )
    parser.add_argument(
        "--frames-per-video", type=int, default=15,
        help="Number of frames to extract per video",
    )
    parser.add_argument(
        "--pad-ratio", type=float, default=0.3,
        help="Bottom padding ratio (0.3 = 30%% extra height)",
    )
    parser.add_argument(
        "--val-split", type=float, default=0.2,
        help="Fraction of videos for validation split",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would be done without writing files",
    )
    parser.add_argument(
        "--video-id", type=str,
        help="Export only a specific video",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    from rallycut.evaluation.tracking.db import get_video_path

    # Load calibrated videos
    videos = load_calibrated_videos()
    if args.video_id:
        videos = [v for v in videos if v["id"] == args.video_id]

    print(f"Found {len(videos)} videos with court calibration GT")

    if not videos:
        print("No calibrated videos found. Nothing to export.")
        return

    # Analyze corner positions
    off_screen_count = 0
    for v in videos:
        corners = v["corners"]
        for c in corners:
            if c["y"] > 1.0 or c["x"] < 0 or c["x"] > 1.0:
                off_screen_count += 1
                break
    print(f"Videos with off-screen corners: {off_screen_count}/{len(videos)}")
    print(f"Padding ratio: {args.pad_ratio:.0%}")

    if args.dry_run:
        print(f"\nWould export ~{len(videos) * args.frames_per_video} frames")
        print(f"  Output: {args.output_dir}")
        print(f"  Val split: {args.val_split:.0%}")

        for v in videos:
            corners = v["corners"]
            ys = [c["y"] for c in corners]
            max_y = max(ys)
            near_status = "off-screen" if max_y > 1.0 else f"y={max_y:.2f}"
            print(f"  {v['id'][:12]}  near corners: {near_status}")
        return

    # Split videos into train/val (video-level split for honest evaluation)
    rng = random.Random(args.seed)
    video_ids = [v["id"] for v in videos]
    rng.shuffle(video_ids)
    n_val = max(1, int(len(video_ids) * args.val_split))
    val_ids = set(video_ids[:n_val])
    train_ids = set(video_ids[n_val:])

    print(f"Train: {len(train_ids)} videos, Val: {len(val_ids)} videos")

    # Create output directories
    for split in ["train", "val"]:
        (args.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    total_frames = 0
    total_skipped = 0

    for vi, video in enumerate(videos):
        vid_id = video["id"]
        corners = video["corners"]
        split = "val" if vid_id in val_ids else "train"

        # Resolve video path
        video_path = get_video_path(vid_id)
        if video_path is None:
            print(f"  [{vi+1}/{len(videos)}] {vid_id[:12]}  SKIP (video not found)")
            continue

        # Extract frames
        frames = extract_frames(video_path, args.frames_per_video, args.seed + vi)

        exported = 0
        for fi, frame in enumerate(frames):
            # Pad frame
            padded = pad_frame_bottom(frame, args.pad_ratio)

            # Create annotation
            label = corners_to_yolo_pose(corners, frame.shape[0], args.pad_ratio)
            if label is None:
                total_skipped += 1
                continue

            # Save image and label
            fname = f"{vid_id[:12]}_{fi:03d}"
            img_path = args.output_dir / "images" / split / f"{fname}.jpg"
            lbl_path = args.output_dir / "labels" / split / f"{fname}.txt"

            cv2.imwrite(str(img_path), padded, [cv2.IMWRITE_JPEG_QUALITY, 95])
            lbl_path.write_text(label + "\n")

            exported += 1
            total_frames += 1

        print(
            f"  [{vi+1}/{len(videos)}] {vid_id[:12]}  {split:5s}  "
            f"{exported} frames exported"
        )

    # Write dataset YAML
    yaml_content = f"""# Court keypoint detection dataset
# Generated by export_court_keypoint_dataset.py
# {len(videos)} videos, {total_frames} frames

path: {args.output_dir.resolve()}
train: images/train
val: images/val

kpt_shape: [4, 3]  # 4 keypoints (corners), 3 dims (x, y, visibility)
flip_idx: {FLIP_IDX}  # near-left<->near-right, far-left<->far-right

names:
  0: court
"""
    yaml_path = args.output_dir / "court_keypoints.yaml"
    yaml_path.write_text(yaml_content)

    # Write metadata
    metadata = {
        "total_frames": total_frames,
        "total_skipped": total_skipped,
        "videos": len(videos),
        "train_videos": len(train_ids),
        "val_videos": len(val_ids),
        "pad_ratio": args.pad_ratio,
        "frames_per_video": args.frames_per_video,
        "val_video_ids": sorted(val_ids),
        "train_video_ids": sorted(train_ids),
    }
    meta_path = args.output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"\nDataset exported to {args.output_dir}")
    print(f"  Total frames: {total_frames}")
    print(f"  Skipped: {total_skipped}")
    print(f"  YAML config: {yaml_path}")
    print("\nTo train:")
    print("  from ultralytics import YOLO")
    print("  model = YOLO('yolo11s-pose.pt')")
    print(f"  model.train(data='{yaml_path}', epochs=100, imgsz=640)")


if __name__ == "__main__":
    main()
