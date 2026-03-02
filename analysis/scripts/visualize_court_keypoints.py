#!/usr/bin/env python3
"""Visualize court keypoint detection on video frames.

Draws detected corners + court polygon on a mid-frame and opens the result.

Usage:
    uv run python scripts/visualize_court_keypoints.py                    # All GT videos
    uv run python scripts/visualize_court_keypoints.py --video-id abc123  # Single video
    uv run python scripts/visualize_court_keypoints.py --video path.mp4   # Direct video path
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

CORNER_NAMES = ["near-left", "near-right", "far-right", "far-left"]
CORNER_COLORS = [
    (0, 0, 255),    # near-left: red
    (0, 165, 255),   # near-right: orange
    (0, 255, 0),    # far-right: green
    (255, 0, 0),    # far-left: blue
]
GT_COLOR = (255, 255, 0)  # cyan for GT


def draw_court(
    frame: np.ndarray,
    corners: list[dict[str, float]],
    color: tuple[int, ...] = (0, 255, 255),
    label: str = "",
    gt_corners: list[dict[str, float]] | None = None,
) -> np.ndarray:
    """Draw court corners and polygon on frame."""
    h, w = frame.shape[:2]
    vis = frame.copy()

    # Draw GT polygon if provided
    if gt_corners and len(gt_corners) == 4:
        gt_pts = np.array(
            [[int(c["x"] * w), int(c["y"] * h)] for c in gt_corners],
            dtype=np.int32,
        )
        cv2.polylines(vis, [gt_pts], True, GT_COLOR, 2, cv2.LINE_AA)
        for i, c in enumerate(gt_corners):
            px, py = int(c["x"] * w), int(c["y"] * h)
            cv2.circle(vis, (px, py), 6, GT_COLOR, 2, cv2.LINE_AA)
            cv2.putText(
                vis, f"GT {CORNER_NAMES[i]}", (px + 10, py - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, GT_COLOR, 1, cv2.LINE_AA,
            )

    if len(corners) != 4:
        cv2.putText(
            vis, f"{label}: NO DETECTION", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA,
        )
        return vis

    # Draw predicted polygon
    pts = np.array(
        [[int(c["x"] * w), int(c["y"] * h)] for c in corners],
        dtype=np.int32,
    )
    # Semi-transparent fill
    overlay = vis.copy()
    cv2.fillPoly(overlay, [pts], (*color[:3], 40))
    cv2.addWeighted(overlay, 0.2, vis, 0.8, 0, vis)
    cv2.polylines(vis, [pts], True, color, 3, cv2.LINE_AA)

    # Draw corners with names
    for i, c in enumerate(corners):
        px, py = int(c["x"] * w), int(c["y"] * h)
        cv2.circle(vis, (px, py), 8, CORNER_COLORS[i], -1, cv2.LINE_AA)
        cv2.circle(vis, (px, py), 8, (255, 255, 255), 2, cv2.LINE_AA)
        txt = f"{CORNER_NAMES[i]} ({c['x']:.3f}, {c['y']:.3f})"
        cv2.putText(
            vis, txt, (px + 12, py + 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, CORNER_COLORS[i], 2, cv2.LINE_AA,
        )

    if label:
        cv2.putText(
            vis, label, (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2, cv2.LINE_AA,
        )

    return vis


def get_mid_frame(video_path: Path) -> np.ndarray | None:
    """Read the middle frame from a video."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize court keypoint detection")
    parser.add_argument("--video", type=Path, help="Direct video path")
    parser.add_argument("--video-id", help="Video ID from database")
    parser.add_argument(
        "--model", type=Path,
        default=Path("weights/court_keypoint/court_keypoint_best.pt"),
        help="Keypoint model path",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("debug_court_keypoints"))
    parser.add_argument("--n-frames", type=int, default=30, help="Frames to sample")
    args = parser.parse_args()

    from rallycut.court.keypoint_detector import CourtKeypointDetector

    detector = CourtKeypointDetector(model_path=args.model)
    if not detector.model_exists:
        print(f"Model not found: {args.model}")
        return

    args.output_dir.mkdir(exist_ok=True)

    if args.video:
        # Single video file
        videos = [("direct", args.video, None)]
    elif args.video_id:
        from rallycut.evaluation.tracking.db import get_video_path
        path = get_video_path(args.video_id)
        if path is None:
            print(f"Video not found: {args.video_id}")
            return

        # Try to load GT
        gt = _load_gt(args.video_id)
        videos = [(args.video_id[:12], Path(path), gt)]
    else:
        # All GT videos from DB
        videos = _load_all_gt_videos()

    for vid_label, video_path, gt_corners in videos:
        if not video_path.exists():
            print(f"  {vid_label}: video not found at {video_path}")
            continue

        frame = get_mid_frame(video_path)
        if frame is None:
            print(f"  {vid_label}: could not read frame")
            continue

        result = detector.detect(video_path, n_frames=args.n_frames)

        label = f"Keypoint conf={result.confidence:.3f}"
        vis = draw_court(frame, result.corners, (0, 255, 255), label, gt_corners)

        out_path = args.output_dir / f"{vid_label}_keypoints.jpg"
        cv2.imwrite(str(out_path), vis)

        mcd_str = ""
        if gt_corners and len(result.corners) == 4:
            dists = [
                ((r["x"] - g["x"]) ** 2 + (r["y"] - g["y"]) ** 2) ** 0.5
                for r, g in zip(result.corners, gt_corners)
            ]
            mcd_str = f"  MCD={sum(dists)/4:.4f}"

        print(f"  {vid_label}: conf={result.confidence:.3f}{mcd_str}  → {out_path}")

    print(f"\nImages saved to {args.output_dir}/")
    # Open the output directory
    subprocess.run(["open", str(args.output_dir)], check=False)


def _load_gt(video_id: str) -> list[dict[str, float]] | None:
    """Load GT corners from database."""
    try:
        from rallycut.evaluation.tracking.db import load_court_calibration

        return load_court_calibration(video_id)
    except Exception:
        pass
    return None


def _load_all_gt_videos() -> list[tuple[str, Path, list[dict[str, float]] | None]]:
    """Load all videos with GT court calibration."""
    try:
        import json

        from rallycut.evaluation.db import get_connection
        from rallycut.evaluation.tracking.db import get_video_path

        with get_connection() as conn:
            cur = conn.cursor()
            cur.execute(
                """SELECT id, court_calibration_json FROM videos
                   WHERE court_calibration_json IS NOT NULL
                   ORDER BY id""",
            )
            rows = cur.fetchall()
            cur.close()

        videos = []
        for vid_id, cal_json in rows:
            path = get_video_path(vid_id)
            if path is None:
                continue
            gt = None
            if cal_json:
                data = json.loads(cal_json) if isinstance(cal_json, str) else cal_json
                if isinstance(data, list) and len(data) == 4:
                    gt = data
            videos.append((vid_id[:12], Path(path), gt))

        print(f"Found {len(videos)} videos with court GT\n")
        return videos
    except Exception as e:
        print(f"DB error: {e}")
        return []


if __name__ == "__main__":
    main()
