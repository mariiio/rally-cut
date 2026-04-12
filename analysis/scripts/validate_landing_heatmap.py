"""Validate landing detection accuracy on dense ball-GT rallies.

For each GT rally that has ball annotations:
1. Run find_landing() on WASB ball detections
2. Project the detected landing through the court homography
3. Compare to the GT ball position at the same frame (also projected)
4. Report median and p90 position error in metres

Gate: median < 0.5m, p90 < 1.5m
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

# Allow imports from parent package
_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from rallycut.court.calibration import CourtCalibrator  # noqa: E402
from rallycut.evaluation.tracking.db import (  # noqa: E402
    TrackingEvaluationRally,
    load_labeled_rallies,
)
from rallycut.statistics.landing_detector import find_landing  # noqa: E402
from rallycut.tracking.ball_tracker import BallPosition  # noqa: E402


def _create_calibrator(
    corners_json: list[dict[str, float]] | None,
) -> CourtCalibrator | None:
    if not corners_json or len(corners_json) != 4:
        return None
    calibrator = CourtCalibrator()
    image_corners = [(c["x"], c["y"]) for c in corners_json]
    calibrator.calibrate(image_corners)
    return calibrator


def _parse_ball_positions(
    rally: TrackingEvaluationRally,
) -> list[BallPosition]:
    """Parse WASB ball positions from the predictions."""
    if rally.predictions is None:
        return []
    # Ball positions are on the PlayerTrackingResult (added during tracking)
    ball_pos: list[BallPosition] | None = getattr(
        rally.predictions, "ball_positions", None,
    )
    if ball_pos:
        return ball_pos
    return []


def _get_gt_ball_positions(
    rally: TrackingEvaluationRally,
) -> list[tuple[int, float, float]]:
    """Extract ball GT positions as (frame, x, y)."""
    return [
        (p.frame_number, p.x, p.y)
        for p in rally.ground_truth.positions
        if p.is_ball
    ]


def _find_gt_landing(
    gt_ball: list[tuple[int, float, float]],
) -> tuple[int, float, float] | None:
    """Find the GT landing = last ball GT position where velocity is low.

    Uses the same algorithm as find_landing but on GT data.
    """
    if len(gt_ball) < 6:
        return None

    # Use find_landing by wrapping GT into BallPositions
    ball_pos = [
        BallPosition(
            frame_number=f, x=x, y=y, confidence=1.0,
        )
        for f, x, y in gt_ball
    ]
    start_frame = gt_ball[0][0]
    return find_landing(ball_pos, start_frame)


def main() -> None:
    print("Loading evaluation rallies with ball GT...")
    rallies = load_labeled_rallies()
    print(f"Loaded {len(rallies)} rallies with ground truth")

    errors_m: list[float] = []
    errors_detail: list[dict[str, Any]] = []
    skipped = 0
    no_landing = 0
    no_gt_landing = 0
    no_calibration = 0

    for i, rally in enumerate(rallies):
        # Get GT ball positions
        gt_ball = _get_gt_ball_positions(rally)
        if len(gt_ball) < 10:
            skipped += 1
            continue

        # Get WASB ball positions
        wasb_ball = _parse_ball_positions(rally)
        if len(wasb_ball) < 10:
            skipped += 1
            continue

        # Create calibrator
        calibrator = _create_calibrator(rally.court_calibration_json)
        if calibrator is None:
            no_calibration += 1
            continue

        # Find GT landing
        gt_landing = _find_gt_landing(gt_ball)
        if gt_landing is None:
            no_gt_landing += 1
            continue
        gt_frame, gt_x, gt_y = gt_landing

        # Find WASB landing (from first frame of rally)
        start_frame = wasb_ball[0].frame_number if wasb_ball else 0
        wasb_landing = find_landing(wasb_ball, start_frame)
        if wasb_landing is None:
            no_landing += 1
            continue
        det_frame, det_x, det_y = wasb_landing

        # Project both through homography
        try:
            gt_court = calibrator.image_to_court(
                (gt_x, gt_y), rally.video_width, rally.video_height,
            )
            det_court = calibrator.image_to_court(
                (det_x, det_y), rally.video_width, rally.video_height,
            )
        except (RuntimeError, ValueError):
            skipped += 1
            continue

        # Compute court-plane error
        err_m = math.sqrt(
            (gt_court[0] - det_court[0]) ** 2
            + (gt_court[1] - det_court[1]) ** 2
        )
        errors_m.append(err_m)
        errors_detail.append({
            "rally_id": rally.rally_id,
            "error_m": round(err_m, 3),
            "gt_court": (round(gt_court[0], 2), round(gt_court[1], 2)),
            "det_court": (round(det_court[0], 2), round(det_court[1], 2)),
            "gt_frame": gt_frame,
            "det_frame": det_frame,
        })

        status = "OK" if err_m < 1.0 else "HIGH"
        print(
            f"[{i+1}/{len(rallies)}] {rally.rally_id[:8]}: "
            f"err={err_m:.2f}m  gt=({gt_court[0]:.1f},{gt_court[1]:.1f})  "
            f"det=({det_court[0]:.1f},{det_court[1]:.1f})  [{status}]"
        )

    # Summary
    print("\n" + "=" * 60)
    print(f"Rallies with ball GT: {len(rallies)}")
    print(f"Skipped (too few points / projection error): {skipped}")
    print(f"No calibration: {no_calibration}")
    print(f"No GT landing detected: {no_gt_landing}")
    print(f"No WASB landing detected: {no_landing}")
    print(f"Evaluated: {len(errors_m)}")

    if errors_m:
        arr = np.array(errors_m)
        median = float(np.median(arr))
        p90 = float(np.percentile(arr, 90))
        mean = float(np.mean(arr))
        print(f"\nMedian error: {median:.3f} m")
        print(f"Mean error:   {mean:.3f} m")
        print(f"P90 error:    {p90:.3f} m")
        print(f"Max error:    {float(arr.max()):.3f} m")

        # Gate check
        median_pass = median < 0.5
        p90_pass = p90 < 1.5
        print(f"\nGate: median < 0.5m → {'PASS' if median_pass else 'FAIL'} ({median:.3f})")
        print(f"Gate: p90 < 1.5m   → {'PASS' if p90_pass else 'FAIL'} ({p90:.3f})")

        if median_pass and p90_pass:
            print("\n>>> ALL GATES PASS <<<")
        else:
            print("\n>>> GATES FAILED — investigate per-rally errors above <<<")
    else:
        print("\nNo rallies evaluated — cannot compute metrics.")


if __name__ == "__main__":
    main()
