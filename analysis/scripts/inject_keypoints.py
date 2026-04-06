"""Inject YOLO-Pose keypoints into existing DB positions_json.

Runs YOLO-Pose on video frames, matches detections to stored player
positions via bbox IoU, and adds keypoints to positions_json WITHOUT
re-tracking. Track IDs are preserved — GT labels remain valid.

Usage:
    cd analysis
    uv run python scripts/inject_keypoints.py --all              # All GT rallies
    uv run python scripts/inject_keypoints.py --video 44e89f6c   # Specific video
    uv run python scripts/inject_keypoints.py --all --dry-run    # Preview only
"""

from __future__ import annotations

import argparse
import json
import sys
import time

import cv2
import numpy as np
from rich.console import Console

from rallycut.evaluation.db import get_connection
from rallycut.evaluation.tracking.db import get_video_path
from rallycut.tracking.pose_attribution.pose_cache import _bbox_iou
from scripts.eval_action_detection import RallyData, load_rallies_with_action_gt

console = Console()

IOU_THRESHOLD = 0.3


def inject_keypoints_for_rally(
    rally: RallyData,
    video_path: str,
    pose_model: object,
    imgsz: int = 960,
) -> tuple[list[dict], int, int]:
    """Run YOLO-Pose and inject keypoints into rally's positions_json.

    Returns (updated_positions, n_injected, n_total).
    """
    if not rally.positions_json:
        return [], 0, 0

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return rally.positions_json, 0, len(rally.positions_json)

    img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    abs_offset = int(rally.start_ms / 1000 * fps)

    # Group positions by frame
    pos_by_frame: dict[int, list[dict]] = {}
    for pp in rally.positions_json:
        pos_by_frame.setdefault(pp["frameNumber"], []).append(pp)

    # Get unique frames sorted
    rally_frames = sorted(pos_by_frame.keys())
    if not rally_frames:
        cap.release()
        return rally.positions_json, 0, len(rally.positions_json)

    # Batch read frames and run YOLO-Pose
    first_abs = abs_offset + rally_frames[0]
    cap.set(cv2.CAP_PROP_POS_FRAMES, first_abs)
    current_abs = first_abs

    n_injected = 0
    batch_frames: list[tuple[int, np.ndarray]] = []
    BATCH_SIZE = 8

    for rally_frame in rally_frames:
        target_abs = abs_offset + rally_frame

        while current_abs < target_abs:
            cap.grab()
            current_abs += 1

        ret, frame = cap.read()
        if not ret:
            current_abs += 1
            continue
        current_abs += 1

        batch_frames.append((rally_frame, frame))

        # Process batch
        if len(batch_frames) >= BATCH_SIZE:
            n_injected += _process_batch(
                batch_frames, pos_by_frame, pose_model, img_w, img_h, imgsz,
            )
            batch_frames = []

    # Process remaining
    if batch_frames:
        n_injected += _process_batch(
            batch_frames, pos_by_frame, pose_model, img_w, img_h, imgsz,
        )

    cap.release()
    return rally.positions_json, n_injected, len(rally.positions_json)


def _process_batch(
    batch: list[tuple[int, np.ndarray]],
    pos_by_frame: dict[int, list[dict]],
    pose_model: object,
    img_w: int,
    img_h: int,
    imgsz: int,
) -> int:
    """Run YOLO-Pose on a batch of frames and inject keypoints."""
    frames = [f for _, f in batch]
    rally_frames = [rf for rf, _ in batch]

    results = pose_model.predict(frames, verbose=False, imgsz=imgsz)  # type: ignore[attr-defined]

    n_injected = 0
    for result, rally_frame in zip(results, rally_frames):
        if result.keypoints is None or result.boxes is None:
            continue

        kps_all = result.keypoints.data.cpu().numpy()  # (N, 17, 3)
        boxes = result.boxes.xyxy.cpu().numpy()  # (N, 4)

        if len(kps_all) == 0:
            continue

        # Normalize
        boxes_norm = boxes.copy()
        boxes_norm[:, [0, 2]] /= img_w
        boxes_norm[:, [1, 3]] /= img_h

        # Match each stored position to a pose detection via IoU
        for pp in pos_by_frame.get(rally_frame, []):
            cx, cy = pp["x"], pp["y"]
            w, h = pp["width"], pp["height"]
            pp_box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)

            best_iou = 0.0
            best_idx = -1
            for det_idx in range(len(boxes_norm)):
                det_box = tuple(boxes_norm[det_idx])
                iou = _bbox_iou(pp_box, det_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = det_idx

            if best_iou >= IOU_THRESHOLD and best_idx >= 0:
                kps = kps_all[best_idx].copy()
                kps[:, 0] /= img_w
                kps[:, 1] /= img_h
                pp["keypoints"] = kps.tolist()
                n_injected += 1

    return n_injected


def main() -> None:
    parser = argparse.ArgumentParser(description="Inject YOLO-Pose keypoints into stored positions")
    parser.add_argument("--all", action="store_true", help="Process all GT rallies")
    parser.add_argument("--video", nargs="*", help="Video ID prefixes to process")
    parser.add_argument("--dry-run", action="store_true", help="Preview without updating DB")
    parser.add_argument("--imgsz", type=int, default=960, help="YOLO inference size (default: 960)")
    parser.add_argument("--skip-existing", action="store_true", help="Skip rallies that already have keypoints")
    args = parser.parse_args()

    if not args.all and not args.video:
        console.print("[red]Specify --all or --video[/red]")
        sys.exit(1)

    rallies = load_rallies_with_action_gt()
    console.print(f"Loaded {len(rallies)} rallies")

    if args.video:
        rallies = [r for r in rallies if any(r.video_id.startswith(v) for v in args.video)]
        console.print(f"Filtered to {len(rallies)} rallies")

    if args.skip_existing:
        before = len(rallies)
        rallies = [r for r in rallies
                   if not r.positions_json
                   or not any("keypoints" in p for p in r.positions_json)]
        console.print(f"Skipping {before - len(rallies)} rallies with existing keypoints")

    if not rallies:
        console.print("[yellow]No rallies to process[/yellow]")
        return

    # Group by video
    by_video: dict[str, list[RallyData]] = {}
    for r in rallies:
        by_video.setdefault(r.video_id, []).append(r)

    console.print(f"Processing {len(rallies)} rallies across {len(by_video)} videos")

    # Load pose model
    console.print("[dim]Loading yolo11s-pose...[/dim]")
    from ultralytics import YOLO
    pose_model = YOLO("yolo11s-pose.pt")

    total_injected = 0
    total_positions = 0
    total_rallies = 0
    t0 = time.time()

    for vid_idx, (video_id, video_rallies) in enumerate(by_video.items()):
        video_path = get_video_path(video_id)
        if video_path is None:
            console.print(f"  [yellow]Video {video_id[:8]} not found[/yellow]")
            continue

        n_contacts = sum(len(r.gt_labels) for r in video_rallies)
        console.print(
            f"[{vid_idx+1}/{len(by_video)}] Video {video_id[:8]}: "
            f"{len(video_rallies)} rallies, {n_contacts} contacts"
        )

        for rally in video_rallies:
            rt0 = time.time()
            positions, n_inj, n_total = inject_keypoints_for_rally(
                rally, str(video_path), pose_model, imgsz=args.imgsz,
            )
            elapsed = time.time() - rt0

            total_injected += n_inj
            total_positions += n_total
            total_rallies += 1

            pct = n_inj / n_total * 100 if n_total > 0 else 0
            console.print(
                f"  {rally.rally_id[:8]}: {n_inj}/{n_total} positions enriched "
                f"({pct:.0f}%) [{elapsed:.1f}s]"
            )

            if not args.dry_run and n_inj > 0:
                with get_connection() as conn:
                    with conn.cursor() as cur:
                        cur.execute(
                            "UPDATE player_tracks SET positions_json = %s WHERE rally_id = %s",
                            [json.dumps(positions), rally.rally_id],
                        )
                    conn.commit()

    elapsed = time.time() - t0
    pct = total_injected / total_positions * 100 if total_positions > 0 else 0
    console.print(
        f"\n[bold]Done.[/bold] {total_rallies} rallies, "
        f"{total_injected}/{total_positions} positions enriched ({pct:.0f}%)"
    )
    console.print(f"  Time: {elapsed:.0f}s ({elapsed/max(len(by_video), 1):.1f}s/video)")
    if args.dry_run:
        console.print("  [yellow]Dry run — no DB updates[/yellow]")


if __name__ == "__main__":
    main()
