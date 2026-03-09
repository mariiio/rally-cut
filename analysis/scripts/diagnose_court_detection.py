#!/usr/bin/env python3
"""Diagnose court keypoint detection failures on specific videos.

Runs the keypoint detector and prints per-corner confidence, std, off-screen
status, perspective ratio, and warnings. Saves sample frames with detected
corners overlaid for visual inspection.

Usage:
    uv run python scripts/diagnose_court_detection.py --video-id ce4c67a1 --video-id 90266c1d
    uv run python scripts/diagnose_court_detection.py --video-id ce4c67a1 --save-frames
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


CORNER_NAMES = ["near-left", "near-right", "far-right", "far-left"]
CORNER_COLORS = [
    (0, 0, 255),    # near-left: red
    (0, 165, 255),  # near-right: orange
    (0, 255, 0),    # far-right: green
    (255, 0, 0),    # far-left: blue
]


def draw_corners(
    frame: np.ndarray,
    corners: list[dict[str, float]],
    confidences: list[float] | None = None,
    label: str = "",
) -> np.ndarray:
    """Draw court corners on a frame with labels."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    for i, corner in enumerate(corners):
        px = int(corner["x"] * w)
        py = int(corner["y"] * h)
        color = CORNER_COLORS[i]
        name = CORNER_NAMES[i]

        # Clamp to frame bounds for drawing
        draw_x = max(0, min(w - 1, px))
        draw_y = max(0, min(h - 1, py))

        cv2.circle(vis, (draw_x, draw_y), 8, color, -1)
        conf_str = f" ({confidences[i]:.3f})" if confidences else ""
        text = f"{name}{conf_str}"
        cv2.putText(vis, text, (draw_x + 12, draw_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Draw off-screen indicator
        if corner["y"] > 1.0 or corner["x"] < 0 or corner["x"] > 1.0:
            cv2.putText(vis, "OFF-SCREEN", (draw_x + 12, draw_y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Draw court polygon
    if len(corners) == 4:
        pts = np.array([
            [int(c["x"] * w), int(c["y"] * h)] for c in corners
        ], dtype=np.int32)
        cv2.polylines(vis, [pts], True, (255, 255, 0), 2)

    if label:
        cv2.putText(vis, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    return vis


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose court keypoint detection")
    parser.add_argument(
        "--video-id", type=str, action="append", required=True,
        help="Video ID(s) to diagnose (can specify multiple)",
    )
    parser.add_argument(
        "--save-frames", action="store_true",
        help="Save sample frames with detected corners",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("debug_court_detection"),
        help="Output directory for debug images",
    )
    parser.add_argument(
        "--n-frames", type=int, default=30,
        help="Number of frames to sample",
    )
    parser.add_argument(
        "--keypoint-model", type=str, default=None,
        help="Path to keypoint model weights",
    )
    args = parser.parse_args()

    from rallycut.court.keypoint_detector import CourtKeypointDetector
    from rallycut.evaluation.db import get_connection
    from rallycut.evaluation.tracking.db import get_video_path

    detector = CourtKeypointDetector(model_path=args.keypoint_model)
    if not detector.model_exists:
        print("Keypoint model not found. Train it first:")
        print("  uv run python scripts/train_court_keypoint_model.py")
        return

    if args.save_frames:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    # Check if videos have existing calibration in DB
    for vid_id in args.video_id:
        print(f"\n{'='*70}")
        print(f"Video: {vid_id}")
        print(f"{'='*70}")

        # Check DB for existing calibration
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT id, court_calibration_json, court_calibration_source, width, height "
                    "FROM videos WHERE id::text LIKE %s",
                    [vid_id + "%"],
                )
                row = cur.fetchone()

        if row is None:
            print(f"  Video not found in DB!")
            continue

        full_vid_id, cal_json, cal_source, width, height = row
        vid_id = str(full_vid_id)
        print(f"  Full ID: {vid_id}")
        print(f"  Resolution: {width}x{height}")
        if cal_json:
            print(f"  Existing calibration: source={cal_source}")
            for i, name in enumerate(CORNER_NAMES):
                c = cal_json[i]
                print(f"    {name:>12s}: ({c['x']:.4f}, {c['y']:.4f})")
        else:
            print(f"  No existing calibration in DB")

        # Resolve video path
        video_path = get_video_path(vid_id)
        if video_path is None:
            print(f"  Video file not found!")
            continue

        print(f"  Video path: {video_path}")

        # Run keypoint detection
        print(f"\n  Running keypoint detection ({args.n_frames} frames)...")
        result = detector.detect(video_path, n_frames=args.n_frames)

        # Print results
        print(f"\n  Detection result:")
        print(f"    Confidence: {result.confidence:.4f}")
        print(f"    Method: {result.fitting_method}")
        if result.corners:
            print(f"    Corners:")
            for i, name in enumerate(CORNER_NAMES):
                c = result.corners[i]
                per_conf = result.per_corner_confidence or {}
                conf = per_conf.get(name, 0.0)
                off = " (OFF-SCREEN)" if c["y"] > 1.0 else ""
                print(f"      {name:>12s}: ({c['x']:.4f}, {c['y']:.4f})  conf={conf:.4f}{off}")
        else:
            print(f"    No corners detected!")

        # Print diagnostics
        diag = detector.last_diagnostics
        if diag:
            print(f"\n  Diagnostics:")
            print(f"    Detection rate: {diag.detection_rate:.0%}")
            print(f"    Perspective ratio: {diag.perspective_ratio:.2f}")
            print(f"    Off-screen corners: {diag.off_screen_corners or 'none'}")
            print(f"    Per-corner std:")
            for name, std in diag.per_corner_std.items():
                print(f"      {name:>12s}: {std:.5f}")
            if diag.warnings:
                print(f"    Warnings:")
                for w in diag.warnings:
                    print(f"      - {w}")

        # Analyze what's blocking auto-save
        print(f"\n  Auto-save analysis:")
        print(f"    Final confidence: {result.confidence:.4f}")
        print(f"    Auto-save threshold: 0.70")
        if result.confidence >= 0.7:
            print(f"    Status: WOULD AUTO-SAVE")
        else:
            print(f"    Status: BLOCKED (confidence too low)")
            # Diagnose why
            per_conf = result.per_corner_confidence or {}
            reliable = sum(1 for c in per_conf.values() if c >= 0.5)
            print(f"    Reliable corners (conf >= 0.5): {reliable}/4")
            low_corners = [
                name for name, c in per_conf.items() if c < 0.5
            ]
            if low_corners:
                print(f"    Low-confidence corners: {', '.join(low_corners)}")
                print(f"    → Penalty formula: bbox_conf * {reliable}/4 = "
                      f"{per_conf.get('near-left', 0):.3f} (need all 4 reliable)")

        # Save debug frames
        if args.save_frames and result.corners:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                # Sample 3 frames: early, middle, late
                for label, frac in [("early", 0.2), ("mid", 0.5), ("late", 0.8)]:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, int(total * frac))
                    ret, frame = cap.read()
                    if ret:
                        per_conf = result.per_corner_confidence or {}
                        confs = [per_conf.get(n, 0.0) for n in CORNER_NAMES]
                        vis = draw_corners(
                            frame, result.corners, confs,
                            f"{vid_id[:12]} - {label} (conf={result.confidence:.3f})",
                        )
                        out_path = args.output_dir / f"{vid_id[:12]}_{label}.jpg"
                        cv2.imwrite(str(out_path), vis)
                        print(f"    Saved: {out_path}")
                cap.release()

    print()


if __name__ == "__main__":
    main()
