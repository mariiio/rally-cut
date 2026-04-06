"""Extract YOLO-Pose keypoints for all GT rally contact windows.

Runs yolo11s-pose on full frames at ±10 frames around each GT contact,
matches detections to player tracks via IoU, and caches results per rally.

Optimized: processes all rallies per video in a single sequential video pass,
avoiding redundant video opens/seeks.

Usage:
    cd analysis
    uv run python scripts/extract_pose_cache.py
    uv run python scripts/extract_pose_cache.py --rally <id>      # Single rally
    uv run python scripts/extract_pose_cache.py --skip-existing    # Skip already cached
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict

import cv2
import numpy as np
from rich.console import Console

from rallycut.evaluation.tracking.db import get_video_path
from rallycut.tracking.pose_attribution.pose_cache import (
    _bbox_iou,
    load_pose_cache,
    save_pose_cache,
)
from scripts.eval_action_detection import (
    RallyData,
    load_rallies_with_action_gt,
)

console = Console()


def _process_video(
    video_path: str,
    video_rallies: list[RallyData],
    pose_model: object,
    window_half: int = 10,
    iou_threshold: float = 0.3,
    imgsz: int = 1280,
    batch_size: int = 8,
) -> dict[str, dict[str, np.ndarray]]:
    """Process all rallies for one video in a single sequential pass.

    Returns {rally_id: {frames, track_ids, keypoints, bboxes}}.
    """
    # Collect all needed absolute frames and their rally mappings
    # frame_info: abs_frame -> list of (rally_id, rally_relative_frame)
    frame_info: dict[int, list[tuple[str, int]]] = defaultdict(list)

    # Player position lookup: (rally_id, rally_frame) -> [(track_id, xyxy_norm)]
    player_lookup: dict[tuple[str, int], list[tuple[int, tuple[float, float, float, float]]]] = {}

    for rally in video_rallies:
        if not rally.positions_json or not rally.gt_labels:
            continue

        abs_offset = int(rally.start_ms / 1000 * rally.fps)
        contact_frames = [gl.frame for gl in rally.gt_labels]

        # Determine needed rally-relative frames
        needed_rally_frames: set[int] = set()
        for cf in contact_frames:
            for offset in range(-window_half, window_half + 1):
                f = cf + offset
                if f >= 0:
                    needed_rally_frames.add(f)

        # Map to absolute frames
        for rf in needed_rally_frames:
            abs_f = abs_offset + rf
            frame_info[abs_f].append((rally.rally_id, rf))

        # Build player position lookup for this rally
        for pp in rally.positions_json:
            fn = pp["frameNumber"]
            if fn not in needed_rally_frames:
                continue
            tid = pp["trackId"]
            if tid < 0:
                continue
            cx, cy = pp["x"], pp["y"]
            w, h = pp["width"], pp["height"]
            xyxy = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
            key = (rally.rally_id, fn)
            player_lookup.setdefault(key, []).append((tid, xyxy))

    if not frame_info:
        return {}

    # Open video once
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        console.print(f"  [red]Cannot open video: {video_path}[/red]")
        return {}

    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Sort needed frames for sequential reading
    sorted_abs_frames = sorted(frame_info.keys())

    # Per-rally accumulation
    rally_data: dict[str, dict[str, list]] = defaultdict(
        lambda: {"frames": [], "track_ids": [], "keypoints": [], "bboxes": []}
    )

    # Read all needed frames first, then batch-predict
    first_abs = sorted_abs_frames[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_abs)
    current_abs = first_abs

    # Read frames into buffer
    frame_buffer: list[tuple[int, np.ndarray]] = []  # (abs_frame, image)
    for abs_frame in sorted_abs_frames:
        while current_abs < abs_frame:
            cap.grab()
            current_abs += 1

        ret, frame = cap.read()
        if not ret:
            current_abs += 1
            continue
        current_abs += 1
        frame_buffer.append((abs_frame, frame))

    cap.release()

    # Batch predict with YOLO-Pose
    for batch_start in range(0, len(frame_buffer), batch_size):
        batch = frame_buffer[batch_start:batch_start + batch_size]
        batch_frames = [f for _, f in batch]
        batch_abs = [a for a, _ in batch]

        results_list = pose_model.predict(batch_frames, verbose=False, imgsz=imgsz)  # type: ignore[attr-defined]

        for result, abs_frame in zip(results_list, batch_abs):
            if result.keypoints is None or result.boxes is None:
                continue

            kps_all = result.keypoints.data.cpu().numpy()
            boxes = result.boxes.xyxy.cpu().numpy()

            if len(kps_all) == 0:
                continue

            boxes_norm = boxes.copy()
            boxes_norm[:, [0, 2]] /= img_w
            boxes_norm[:, [1, 3]] /= img_h

            for rally_id, rally_frame in frame_info[abs_frame]:
                players = player_lookup.get((rally_id, rally_frame), [])
                if not players:
                    continue

                for det_idx in range(len(kps_all)):
                    det_box = tuple(boxes_norm[det_idx])

                    best_iou = 0.0
                    best_tid = -1
                    for tid, track_box in players:
                        iou = _bbox_iou(det_box, track_box)
                        if iou > best_iou:
                            best_iou = iou
                            best_tid = tid

                    if best_iou >= iou_threshold and best_tid >= 0:
                        kps = kps_all[det_idx].copy()
                        kps[:, 0] /= img_w
                        kps[:, 1] /= img_h

                        rd = rally_data[rally_id]
                        rd["frames"].append(rally_frame)
                        rd["track_ids"].append(best_tid)
                        rd["keypoints"].append(kps.astype(np.float32))
                        rd["bboxes"].append(boxes_norm[det_idx].astype(np.float32))

    # Convert to numpy arrays
    result_dict: dict[str, dict[str, np.ndarray]] = {}
    for rally_id, rd in rally_data.items():
        if rd["frames"]:
            result_dict[rally_id] = {
                "frames": np.array(rd["frames"], dtype=np.int32),
                "track_ids": np.array(rd["track_ids"], dtype=np.int32),
                "keypoints": np.stack(rd["keypoints"]),
                "bboxes": np.stack(rd["bboxes"]),
            }
        else:
            result_dict[rally_id] = {
                "frames": np.array([], dtype=np.int32),
                "track_ids": np.array([], dtype=np.int32),
                "keypoints": np.zeros((0, 17, 3), dtype=np.float32),
                "bboxes": np.zeros((0, 4), dtype=np.float32),
            }

    return result_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract YOLO-Pose keypoints for GT rallies")
    parser.add_argument("--rally", type=str, help="Process single rally ID")
    parser.add_argument("--skip-existing", action="store_true", help="Skip already cached rallies")
    parser.add_argument("--window", type=int, default=10, help="Half-window around contacts (default: 10)")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference resolution (default: 960)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for YOLO inference (default: 8)")
    args = parser.parse_args()

    rallies = load_rallies_with_action_gt(rally_id=args.rally)
    console.print(f"Loaded {len(rallies)} rallies with action GT")

    if not rallies:
        console.print("[red]No rallies found[/red]")
        sys.exit(1)

    # Group by video
    by_video: dict[str, list[RallyData]] = {}
    for r in rallies:
        by_video.setdefault(r.video_id, []).append(r)

    total_contacts = sum(len(r.gt_labels) for r in rallies)
    console.print(f"  {len(by_video)} unique videos, {total_contacts} GT contacts")

    # Filter out already-cached if requested
    if args.skip_existing:
        filtered: dict[str, list[RallyData]] = {}
        n_skip = 0
        for vid, vr in by_video.items():
            remaining = [r for r in vr if load_pose_cache(r.rally_id) is None]
            if remaining:
                filtered[vid] = remaining
            n_skip += len(vr) - len(remaining)
        by_video = filtered
        if n_skip > 0:
            console.print(f"  Skipping {n_skip} already cached rallies")

    total_rallies = sum(len(vr) for vr in by_video.values())
    if total_rallies == 0:
        console.print("[green]All rallies already cached[/green]")
        _print_visibility_stats(rallies)
        return

    # Load pose model once
    console.print("[dim]Loading yolo11s-pose model...[/dim]")
    from ultralytics import YOLO
    pose_model = YOLO("yolo11s-pose.pt")

    processed_rallies = 0
    total_detections = 0
    t0 = time.time()

    for vid_idx, (video_id, video_rallies) in enumerate(by_video.items()):
        video_path = get_video_path(video_id)
        if video_path is None:
            console.print(f"  [yellow]Skipping video {video_id[:8]}: not found[/yellow]")
            continue

        n_contacts = sum(len(r.gt_labels) for r in video_rallies)
        console.print(
            f"[{vid_idx+1}/{len(by_video)}] Video {video_id[:8]}: "
            f"{len(video_rallies)} rallies, {n_contacts} contacts"
        )

        vt0 = time.time()
        results = _process_video(
            video_path=str(video_path),
            video_rallies=video_rallies,
            pose_model=pose_model,
            window_half=args.window,
            imgsz=args.imgsz,
            batch_size=args.batch_size,
        )

        # Save per-rally caches
        for rally in video_rallies:
            data = results.get(rally.rally_id)
            if data is None:
                data = {
                    "frames": np.array([], dtype=np.int32),
                    "track_ids": np.array([], dtype=np.int32),
                    "keypoints": np.zeros((0, 17, 3), dtype=np.float32),
                    "bboxes": np.zeros((0, 4), dtype=np.float32),
                }
            n_det = len(data["frames"])
            total_detections += n_det
            save_pose_cache(rally.rally_id, data)
            processed_rallies += 1

        vt = time.time() - vt0
        console.print(f"  → {len(results)} rallies cached in {vt:.1f}s")

    elapsed = time.time() - t0
    console.print(f"\n[bold]Done.[/bold] {processed_rallies} rallies, {total_detections} detections")
    console.print(f"  Time: {elapsed:.0f}s ({elapsed/max(len(by_video), 1):.1f}s/video)")

    _print_visibility_stats(rallies)


def _print_visibility_stats(rallies: list[RallyData]) -> None:
    """Print keypoint visibility stats from cached data."""
    from rich.table import Table

    kpt_names = [
        "nose", "left_eye", "right_eye", "left_ear", "right_ear",
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist", "left_hip", "right_hip",
        "left_knee", "right_knee", "left_ankle", "right_ankle",
    ]

    all_kps: list[np.ndarray] = []
    for r in rallies:
        data = load_pose_cache(r.rally_id)
        if data is not None and len(data["keypoints"]) > 0:
            all_kps.append(data["keypoints"])

    if not all_kps:
        console.print("[yellow]No cached pose data to analyze[/yellow]")
        return

    kps = np.concatenate(all_kps)
    n_total = len(kps)

    console.print(f"\n[bold]Keypoint visibility[/bold] ({n_total} detections)")

    table = Table(show_header=True)
    table.add_column("Keypoint")
    table.add_column("Visible (>0.3)", justify="right")
    table.add_column("%", justify="right")

    for i, name in enumerate(kpt_names):
        visible = int((kps[:, i, 2] > 0.3).sum())
        pct = visible / n_total * 100
        table.add_row(name, str(visible), f"{pct:.1f}%")

    console.print(table)


if __name__ == "__main__":
    main()
