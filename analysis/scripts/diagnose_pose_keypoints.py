"""Diagnostic: validate pose keypoint visibility and Y-separation during overlaps.

Runs yolo11s-pose on a rally to test the hypothesis that lower-body keypoints
(hips, knees, ankles) remain spatially separated even when bboxes overlap.
Also compares detection count/confidence of pose model vs standard yolo11s.

Usage:
    uv run python scripts/diagnose_pose_keypoints.py --rally fad29c31
    uv run python scripts/diagnose_pose_keypoints.py --rally fad29c31 --max-frames 200
"""

from __future__ import annotations

import argparse
import logging
import sys

import cv2
import numpy as np
from ultralytics import YOLO

from rallycut.evaluation.tracking.db import get_video_path, load_labeled_rallies

logger = logging.getLogger(__name__)

# COCO pose keypoint indices for lower body
LOWER_BODY_INDICES = [11, 12, 13, 14, 15, 16]  # L/R hip, knee, ankle
KP_CONF_THRESHOLD = 0.3
OVERLAP_IOU_THRESHOLD = 0.15


def bbox_iou(b1: np.ndarray, b2: np.ndarray) -> float:
    """Compute IoU between two xyxy bounding boxes."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def lower_body_centroid(
    kpt: np.ndarray, conf: np.ndarray
) -> tuple[np.ndarray | None, int]:
    """Compute centroid of visible lower-body keypoints.

    Args:
        kpt: (17, 2) keypoint xy coordinates (pixel or normalized).
        conf: (17,) confidence scores.

    Returns:
        (centroid_xy, num_visible) or (None, 0) if insufficient keypoints.
    """
    visible_pts = []
    for idx in LOWER_BODY_INDICES:
        if conf[idx] >= KP_CONF_THRESHOLD:
            visible_pts.append(kpt[idx])
    if len(visible_pts) < 2:
        return None, len(visible_pts)
    return np.mean(visible_pts, axis=0), len(visible_pts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose pose keypoints during overlaps")
    parser.add_argument("--rally", required=True, help="Rally ID (prefix match)")
    parser.add_argument(
        "--max-frames", type=int, default=0,
        help="Max frames to process (0=all)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    # Load rally (support prefix matching)
    all_rallies = load_labeled_rallies()
    rallies = [r for r in all_rallies if r.rally_id.startswith(args.rally)]
    if not rallies:
        print(f"No rally found matching '{args.rally}'")
        sys.exit(1)
    rally = rallies[0]
    print(f"Rally: {rally.rally_id[:12]} ({(rally.end_ms - rally.start_ms)/1000:.1f}s)")

    video_path = get_video_path(rally.video_id)
    if video_path is None:
        print(f"Video not available for {rally.video_id}")
        sys.exit(1)

    # Open video
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(rally.start_ms / 1000 * fps)
    end_frame = int(rally.end_ms / 1000 * fps)
    total_frames = end_frame - start_frame
    if args.max_frames > 0:
        total_frames = min(total_frames, args.max_frames)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Load both models
    print("Loading models...")
    pose_model = YOLO("yolo11s-pose.pt")
    det_model = YOLO("yolo11s.pt")

    # Stats
    overlap_frames = 0
    total_kp_visible = 0
    total_kp_possible = 0
    y_separations: list[float] = []
    det_count_pose: list[int] = []
    det_count_standard: list[int] = []
    det_conf_pose: list[float] = []
    det_conf_standard: list[float] = []

    print(f"Processing {total_frames} frames...")
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # Run both models (no tracking, just detection)
        pose_results = pose_model(frame, conf=0.15, imgsz=1280, classes=[0], verbose=False)
        det_results = det_model(frame, conf=0.15, imgsz=1280, classes=[0], verbose=False)

        # Detection parity
        n_pose = len(pose_results[0].boxes) if pose_results[0].boxes is not None else 0
        n_det = len(det_results[0].boxes) if det_results[0].boxes is not None else 0
        det_count_pose.append(n_pose)
        det_count_standard.append(n_det)

        if pose_results[0].boxes is not None and len(pose_results[0].boxes.conf) > 0:
            det_conf_pose.extend(
                float(c) for c in pose_results[0].boxes.conf.cpu().numpy()
            )
        if det_results[0].boxes is not None and len(det_results[0].boxes.conf) > 0:
            det_conf_standard.extend(
                float(c) for c in det_results[0].boxes.conf.cpu().numpy()
            )

        # Get pose data
        r = pose_results[0]
        if r.keypoints is None or r.boxes is None:
            continue
        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        kpts_xy = r.keypoints.xy.cpu().numpy()  # (N, 17, 2) pixel coords
        kpts_conf = r.keypoints.conf.cpu().numpy()  # (N, 17)
        n = len(boxes_xyxy)

        # Find overlapping pairs
        for i in range(n):
            for j in range(i + 1, n):
                iou = bbox_iou(boxes_xyxy[i], boxes_xyxy[j])
                if iou < OVERLAP_IOU_THRESHOLD:
                    continue

                overlap_frames += 1

                # Lower-body centroids
                c_i, vis_i = lower_body_centroid(kpts_xy[i], kpts_conf[i])
                c_j, vis_j = lower_body_centroid(kpts_xy[j], kpts_conf[j])

                total_kp_possible += 2
                if c_i is not None:
                    total_kp_visible += 1
                if c_j is not None:
                    total_kp_visible += 1

                if c_i is not None and c_j is not None:
                    # Y-separation in normalized coordinates
                    y_sep = abs(c_i[1] - c_j[1]) / h
                    y_separations.append(y_sep)

        if (frame_idx + 1) % 50 == 0:
            print(f"  Frame {frame_idx + 1}/{total_frames}...")

    cap.release()

    # Report
    print(f"\n{'='*60}")
    print("POSE KEYPOINT DIAGNOSTIC RESULTS")
    print(f"{'='*60}")

    print(f"\nFrames analyzed: {total_frames}")
    print(f"Overlapping bbox pairs (IoU > {OVERLAP_IOU_THRESHOLD}): {overlap_frames}")

    if total_kp_possible > 0:
        visibility = total_kp_visible / total_kp_possible * 100
        print(f"\nKeypoint visibility during overlap: {visibility:.1f}% "
              f"({total_kp_visible}/{total_kp_possible})")
    else:
        visibility = 0
        print("\nNo overlapping bbox pairs found!")

    if y_separations:
        seps = np.array(y_separations)
        print("\nY-separation (normalized) during overlap:")
        print(f"  Median: {np.median(seps):.4f}")
        print(f"  Mean:   {np.mean(seps):.4f}")
        print(f"  P25:    {np.percentile(seps, 25):.4f}")
        print(f"  P75:    {np.percentile(seps, 75):.4f}")
        print(f"  > 0.03: {np.mean(seps > 0.03)*100:.1f}%")
        print(f"  > 0.05: {np.mean(seps > 0.05)*100:.1f}%")
        median_sep = float(np.median(seps))
    else:
        median_sep = 0

    # Detection parity
    pose_arr = np.array(det_count_pose)
    det_arr = np.array(det_count_standard)
    print("\nDetection parity (pose vs standard yolo11s):")
    print(f"  Pose model:     {np.mean(pose_arr):.1f} avg detections/frame")
    print(f"  Standard model: {np.mean(det_arr):.1f} avg detections/frame")
    if det_conf_pose:
        print(f"  Pose conf:      {np.mean(det_conf_pose):.3f} avg")
    if det_conf_standard:
        print(f"  Standard conf:  {np.mean(det_conf_standard):.3f} avg")

    # Go/no-go
    print(f"\n{'='*60}")
    go = visibility > 70 and median_sep > 0.03
    if go:
        print("GO: Keypoint visibility > 70% AND median Y-separation > 0.03")
        print("Proceed with pose-based keypoint association.")
    else:
        reasons = []
        if visibility <= 70:
            reasons.append(f"visibility {visibility:.1f}% <= 70%")
        if median_sep <= 0.03:
            reasons.append(f"median Y-sep {median_sep:.4f} <= 0.03")
        print(f"NO-GO: {', '.join(reasons)}")
        print("Pose keypoint association unlikely to help.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
