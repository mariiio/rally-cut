#!/usr/bin/env python3
"""Compare adaptive vs calibration ROI methods for player tracking.

For each labeled rally, computes:
- Adaptive ROI from ball positions in DB (no re-tracking)
- Calibration ROI from court_calibration_json (skip if not calibrated)
- GT player coverage: % of GT player positions inside each ROI
- Area: ROI area as fraction of frame
- IoU: Polygon overlap (rasterized at 200x200 grid, no shapely dependency)

Usage:
    uv run python scripts/compare_roi_methods.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add analysis root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def point_in_polygon(
    px: float, py: float, polygon: list[tuple[float, float]]
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


def polygon_area(polygon: list[tuple[float, float]]) -> float:
    """Shoelace formula for polygon area."""
    n = len(polygon)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    return abs(area) / 2.0


def rasterized_iou(
    poly_a: list[tuple[float, float]],
    poly_b: list[tuple[float, float]],
    grid_size: int = 200,
) -> float:
    """Compute IoU between two polygons by rasterizing on a grid."""
    intersection = 0
    union = 0
    step = 1.0 / grid_size

    for gy in range(grid_size):
        py = (gy + 0.5) * step
        for gx in range(grid_size):
            px = (gx + 0.5) * step
            in_a = point_in_polygon(px, py, poly_a)
            in_b = point_in_polygon(px, py, poly_b)
            if in_a or in_b:
                union += 1
            if in_a and in_b:
                intersection += 1

    return intersection / union if union > 0 else 0.0


def gt_coverage(
    roi: list[tuple[float, float]],
    gt_positions: list[tuple[float, float]],
) -> float:
    """Fraction of GT positions inside ROI polygon."""
    if not gt_positions:
        return 0.0
    inside = sum(1 for px, py in gt_positions if point_in_polygon(px, py, roi))
    return inside / len(gt_positions)


def main() -> None:
    from rallycut.court.calibration import CourtCalibrator
    from rallycut.evaluation.tracking.db import load_labeled_rallies
    from rallycut.tracking.player_tracker import (
        compute_court_roi_from_ball,
        compute_court_roi_from_calibration,
    )

    print("Loading labeled rallies from database...")
    rallies = load_labeled_rallies()
    print(f"Found {len(rallies)} rally(s) with ground truth\n")

    if not rallies:
        print("No rallies found!")
        return

    # Table header
    print(
        f"{'Rally':<10} {'Video':<10} "
        f"{'Adapt Cov':>9} {'Adapt Area':>10} "
        f"{'Cal Cov':>8} {'Cal Area':>9} "
        f"{'IoU':>5} {'Flags':>20}"
    )
    print("-" * 95)

    total_adapt_cov = 0.0
    total_cal_cov = 0.0
    total_adapt_area = 0.0
    total_cal_area = 0.0
    count_adapt = 0
    count_cal = 0
    flagged_rallies: list[str] = []

    for rally in rallies:
        rally_short = rally.rally_id[:8]
        video_short = rally.video_id[:8]

        # Extract GT player positions (center x, y)
        gt_xy: list[tuple[float, float]] = [
            (p.x, p.y) for p in rally.ground_truth.positions
        ]

        # Adaptive ROI from ball predictions
        adapt_roi = None
        adapt_cov_val = 0.0
        adapt_area_val = 0.0
        if rally.predictions and rally.predictions.ball_positions:
            adapt_roi, _msg = compute_court_roi_from_ball(
                rally.predictions.ball_positions
            )

        if adapt_roi is not None:
            adapt_cov_val = gt_coverage(adapt_roi, gt_xy)
            adapt_area_val = polygon_area(adapt_roi)
            total_adapt_cov += adapt_cov_val
            total_adapt_area += adapt_area_val
            count_adapt += 1

        # Calibration ROI
        cal_roi = None
        cal_cov_val = 0.0
        cal_area_val = 0.0
        if rally.court_calibration_json:
            calibrator = CourtCalibrator()
            image_corners = [
                (c["x"], c["y"]) for c in rally.court_calibration_json
            ]
            calibrator.calibrate(image_corners)
            cal_roi, _cal_msg = compute_court_roi_from_calibration(calibrator)

        if cal_roi is not None:
            cal_cov_val = gt_coverage(cal_roi, gt_xy)
            cal_area_val = polygon_area(cal_roi)
            total_cal_cov += cal_cov_val
            total_cal_area += cal_area_val
            count_cal += 1

        # IoU between the two ROIs
        iou_val = 0.0
        if adapt_roi is not None and cal_roi is not None:
            iou_val = rasterized_iou(adapt_roi, cal_roi)

        # Flags
        flags: list[str] = []
        if adapt_roi is not None and adapt_cov_val < 0.95:
            flags.append("adapt_clips")
        if cal_roi is not None and cal_cov_val < 0.95:
            flags.append("cal_clips")
        if adapt_roi is not None and cal_roi is not None:
            if adapt_cov_val < 0.95 and cal_cov_val >= 0.99:
                flags.append("CAL_WINS")
                flagged_rallies.append(rally_short)
        if adapt_roi is None:
            flags.append("no_adapt")
        if cal_roi is None:
            flags.append("no_cal")

        flags_str = ",".join(flags) if flags else ""

        adapt_cov_str = f"{adapt_cov_val:.1%}" if adapt_roi else "N/A"
        adapt_area_str = f"{adapt_area_val:.3f}" if adapt_roi else "N/A"
        cal_cov_str = f"{cal_cov_val:.1%}" if cal_roi else "N/A"
        cal_area_str = f"{cal_area_val:.3f}" if cal_roi else "N/A"
        iou_str = f"{iou_val:.3f}" if (adapt_roi and cal_roi) else "N/A"

        print(
            f"{rally_short:<10} {video_short:<10} "
            f"{adapt_cov_str:>9} {adapt_area_str:>10} "
            f"{cal_cov_str:>8} {cal_area_str:>9} "
            f"{iou_str:>5} {flags_str:>20}"
        )

    # Aggregate
    print("-" * 95)
    if count_adapt > 0:
        print(
            f"{'Adaptive':>20} avg coverage: {total_adapt_cov / count_adapt:.1%}, "
            f"avg area: {total_adapt_area / count_adapt:.3f} "
            f"({count_adapt} rallies)"
        )
    if count_cal > 0:
        print(
            f"{'Calibration':>20} avg coverage: {total_cal_cov / count_cal:.1%}, "
            f"avg area: {total_cal_area / count_cal:.3f} "
            f"({count_cal} rallies)"
        )

    if flagged_rallies:
        print(
            f"\nRallies where calibration ROI wins (adapt clips, cal doesn't): "
            f"{', '.join(flagged_rallies)}"
        )


if __name__ == "__main__":
    main()
