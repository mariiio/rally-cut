#!/usr/bin/env python3
"""Evaluate automatic court detection against manually calibrated ground truth.

Loads all videos with court_calibration_json from the DB, runs CourtDetector
on each, and computes per-video and aggregate metrics.

Metrics:
- MCD: Mean Corner Distance (normalized + pixels)
- Court Polygon IoU: rasterized overlap
- Reprojection Error: court center projected through both homographies
- Detection Success Rate: % with MCD < 5% frame diagonal

Usage:
    uv run python scripts/eval_court_detection.py
    uv run python scripts/eval_court_detection.py --debug    # Save debug images
    uv run python scripts/eval_court_detection.py -o results.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np


def point_in_polygon(
    px: float, py: float, polygon: list[tuple[float, float]],
) -> bool:
    """Ray-casting point-in-polygon test."""
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > py) != (yj > py)) and (
            px < (xj - xi) * (py - yi) / (yj - yi) + xi
        ):
            inside = not inside
        j = i
    return inside


def rasterized_iou(
    poly_a: list[tuple[float, float]],
    poly_b: list[tuple[float, float]],
    grid_size: int = 200,
) -> float:
    """Compute IoU between two polygons by rasterizing on a grid.

    Supports coordinates outside [0,1] by expanding the grid bounds.
    """
    # Find bounding box of both polygons
    all_pts = poly_a + poly_b
    min_x = min(p[0] for p in all_pts)
    max_x = max(p[0] for p in all_pts)
    min_y = min(p[1] for p in all_pts)
    max_y = max(p[1] for p in all_pts)

    # Add margin
    margin = 0.05
    min_x -= margin
    max_x += margin
    min_y -= margin
    max_y += margin

    range_x = max_x - min_x
    range_y = max_y - min_y

    if range_x <= 0 or range_y <= 0:
        return 0.0

    intersection = 0
    union = 0
    step_x = range_x / grid_size
    step_y = range_y / grid_size

    for gy in range(grid_size):
        py = min_y + (gy + 0.5) * step_y
        for gx in range(grid_size):
            px = min_x + (gx + 0.5) * step_x
            in_a = point_in_polygon(px, py, poly_a)
            in_b = point_in_polygon(px, py, poly_b)
            if in_a or in_b:
                union += 1
            if in_a and in_b:
                intersection += 1

    return intersection / union if union > 0 else 0.0


def mean_corner_distance(
    detected: list[dict[str, float]],
    gt: list[dict[str, float]],
) -> float:
    """Mean Euclidean distance between corresponding corners (normalized coords)."""
    if len(detected) != 4 or len(gt) != 4:
        return float("inf")

    total = 0.0
    for d, g in zip(detected, gt):
        dx = d["x"] - g["x"]
        dy = d["y"] - g["y"]
        total += math.sqrt(dx * dx + dy * dy)

    return total / 4.0


def reprojection_error_meters(
    detected: list[dict[str, float]],
    gt: list[dict[str, float]],
) -> float | None:
    """Reprojection error at court center through both homographies, in meters."""
    try:
        from rallycut.court.calibration import COURT_LENGTH, COURT_WIDTH, CourtCalibrator

        # Build calibrator from GT
        gt_cal = CourtCalibrator()
        gt_corners = [(c["x"], c["y"]) for c in gt]
        gt_cal.calibrate(gt_corners)

        # Build calibrator from detected
        det_cal = CourtCalibrator()
        det_corners = [(c["x"], c["y"]) for c in detected]
        det_cal.calibrate(det_corners)

        # Project court center (4, 8) through both
        court_center = (COURT_WIDTH / 2.0, COURT_LENGTH / 2.0)

        gt_img = gt_cal.court_to_image(court_center, 1920, 1080)
        det_court = det_cal.image_to_court(gt_img, 1920, 1080)

        dx = det_court[0] - court_center[0]
        dy = det_court[1] - court_center[1]
        return math.sqrt(dx * dx + dy * dy)

    except (np.linalg.LinAlgError, ValueError, AttributeError):
        logging.debug("Reprojection error computation failed", exc_info=True)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate court detection")
    parser.add_argument("--debug", action="store_true", help="Save debug images")
    parser.add_argument("-o", "--output", type=str, help="Output JSON path")
    parser.add_argument("--video-id", type=str, help="Evaluate single video")
    args = parser.parse_args()

    from rallycut.court.detector import CourtDetectionConfig, CourtDetector
    from rallycut.evaluation.db import get_connection
    from rallycut.evaluation.tracking.db import get_video_path

    # Load all videos with court calibration
    query = """
        SELECT id, court_calibration_json, width, height
        FROM videos
        WHERE court_calibration_json IS NOT NULL
    """
    params: list[str] = []
    if args.video_id:
        query += " AND id = %s"
        params.append(args.video_id)

    videos: list[dict[str, Any]] = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, params)
            for row in cur.fetchall():
                vid_id, cal_json, width, height = row
                if isinstance(cal_json, list) and len(cal_json) == 4:
                    videos.append({
                        "video_id": str(vid_id),
                        "gt_corners": cal_json,
                        "width": width or 1920,
                        "height": height or 1080,
                    })

    print(f"Found {len(videos)} video(s) with court calibration GT\n")

    if not videos:
        print("No calibrated videos found.")
        return

    # Debug output directory
    debug_dir = Path("debug_court_detection")
    if args.debug:
        debug_dir.mkdir(exist_ok=True)

    detector = CourtDetector(CourtDetectionConfig())

    results: list[dict[str, Any]] = []
    success_count = 0
    total_mcd = 0.0
    total_iou = 0.0
    evaluated = 0

    # Header
    print(f"{'Video':>12s}  {'MCD':>7s}  {'MCD px':>7s}  {'IoU':>5s}  {'Reproj':>7s}  {'Conf':>5s}  {'Lines':>5s}  Warnings")
    print("-" * 90)

    for video_info in videos:
        vid_id = video_info["video_id"]
        gt_corners = video_info["gt_corners"]
        vid_width = video_info["width"]
        vid_height = video_info["height"]

        # Resolve video path
        video_path = get_video_path(vid_id)
        if video_path is None:
            print(f"{vid_id[:12]:>12s}  {'SKIP':>7s}  (video not found)")
            continue

        # Run detection
        try:
            result = detector.detect(video_path)
        except Exception as e:
            print(f"{vid_id[:12]:>12s}  {'ERROR':>7s}  {e}")
            continue

        # Compute metrics
        detected = result.corners
        lines_found = len(result.detected_lines)

        if detected and len(detected) == 4:
            mcd = mean_corner_distance(detected, gt_corners)
            mcd_px = mcd * math.sqrt(vid_width ** 2 + vid_height ** 2)
            is_success = mcd < 0.05  # < 5% of frame diagonal (normalized)

            gt_poly = [(c["x"], c["y"]) for c in gt_corners]
            det_poly = [(c["x"], c["y"]) for c in detected]
            iou = rasterized_iou(det_poly, gt_poly)

            reproj = reprojection_error_meters(detected, gt_corners)
            reproj_str = f"{reproj:.2f}m" if reproj is not None else "N/A"

            warn_str = "; ".join(result.warnings[:2]) if result.warnings else ""

            print(
                f"{vid_id[:12]:>12s}  "
                f"{mcd:.4f}  "
                f"{mcd_px:6.1f}  "
                f"{iou:.3f}  "
                f"{reproj_str:>7s}  "
                f"{result.confidence:.3f}  "
                f"{lines_found:>5d}  "
                f"{warn_str}"
            )

            if is_success:
                success_count += 1
            total_mcd += mcd
            total_iou += iou
            evaluated += 1

            results.append({
                "video_id": vid_id,
                "mcd_norm": mcd,
                "mcd_px": mcd_px,
                "iou": iou,
                "reprojection_error_m": reproj,
                "confidence": result.confidence,
                "lines_found": lines_found,
                "success": is_success,
                "warnings": result.warnings,
                "detected_corners": detected,
                "gt_corners": gt_corners,
            })
        else:
            print(
                f"{vid_id[:12]:>12s}  {'FAIL':>7s}  "
                f"conf={result.confidence:.3f}  "
                f"lines={lines_found}  "
                f"{'; '.join(result.warnings[:2])}"
            )
            results.append({
                "video_id": vid_id,
                "mcd_norm": None,
                "iou": None,
                "confidence": result.confidence,
                "lines_found": lines_found,
                "success": False,
                "warnings": result.warnings,
            })

        # Debug images
        if args.debug and video_path:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                # Read a middle frame
                total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.set(cv2.CAP_PROP_POS_FRAMES, total_f // 2)
                ret, frame = cap.read()
                if ret and frame is not None:
                    debug_frame = detector.create_debug_image(frame, result)
                    out_path = debug_dir / f"{vid_id[:12]}_court_debug.jpg"
                    cv2.imwrite(str(out_path), debug_frame)
                cap.release()

    # Summary
    print("-" * 90)
    if evaluated > 0:
        avg_mcd = total_mcd / evaluated
        avg_iou = total_iou / evaluated
        rate = success_count / evaluated * 100
        print(f"\nAggregate ({evaluated} videos):")
        print(f"  Mean MCD:      {avg_mcd:.4f} (norm)")
        print(f"  Mean IoU:      {avg_iou:.3f}")
        print(f"  Success Rate:  {success_count}/{evaluated} ({rate:.1f}%)")
    else:
        print("\nNo videos evaluated.")

    # Save results
    if args.output:
        with open(args.output, "w") as f:
            json.dump({
                "per_video": results,
                "aggregate": {
                    "evaluated": evaluated,
                    "success_count": success_count,
                    "mean_mcd_norm": total_mcd / evaluated if evaluated else None,
                    "mean_iou": total_iou / evaluated if evaluated else None,
                    "success_rate": success_count / evaluated if evaluated else None,
                },
            }, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
