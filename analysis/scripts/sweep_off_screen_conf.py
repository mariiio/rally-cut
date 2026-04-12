#!/usr/bin/env python3
"""Sweep OFF_SCREEN_CONF threshold for near-corner refinement strategy.

Runs YOLO inference once per video, caches pre-refinement data, then
replays refinement with different thresholds (instant, no GPU needed).
"""
from __future__ import annotations

import gc
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

CORNER_NAMES = ["near-left", "near-right", "far-right", "far-left"]
THRESHOLDS = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
PROBLEM_PREFIXES = ["4f2bd66a", "b026dc6c", "950fbe5d"]


def compute_mcd(
    gt: list[dict[str, float]], pred: list[dict[str, float]],
) -> tuple[float, dict[str, float]]:
    per_corner: dict[str, float] = {}
    total = 0.0
    for i, name in enumerate(CORNER_NAMES):
        d = math.sqrt((gt[i]["x"] - pred[i]["x"]) ** 2 + (gt[i]["y"] - pred[i]["y"]) ** 2)
        per_corner[name] = d
        total += d
    return total / 4, per_corner


def main() -> None:
    from rallycut.court.keypoint_detector import CourtKeypointDetector
    from rallycut.evaluation.db import get_connection
    from rallycut.evaluation.tracking.db import get_video_path

    # Load GT
    videos: list[dict[str, Any]] = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, court_calibration_json, width, height "
                "FROM videos WHERE court_calibration_json IS NOT NULL"
            )
            for row in cur.fetchall():
                vid_id, cal_json, w, h = row
                if isinstance(cal_json, list) and len(cal_json) == 4:
                    videos.append({
                        "id": str(vid_id), "gt": cal_json,
                        "w": w or 1920, "h": h or 1080,
                    })

    print(f"Found {len(videos)} videos with GT\n")

    # Phase 1: Run inference once, cache pre-refinement data
    print("Phase 1: Running YOLO inference (one-time)...")
    kp_detector = CourtKeypointDetector()
    if not kp_detector.model_exists:
        print("Keypoint model not found!")
        return

    # We need the pre-refinement corners. The detect() method calls
    # _aggregate() then _refine_near_corners(). We'll monkey-patch
    # _refine_near_corners to capture and skip refinement.
    cached_data: list[dict[str, Any]] = []
    original_refine = CourtKeypointDetector._refine_near_corners

    def capture_refine(
        self: Any, corners: Any, per_corner_confidence: Any, **kwargs: Any,
    ) -> Any:
        """Intercept refinement to cache pre-refinement data."""
        # Store the inputs for later replay
        self._captured = {
            "pre_corners": [dict(c) for c in corners],
            "per_corner_confidence": dict(per_corner_confidence),
            "center_points": kwargs.get("center_points"),
            "center_confidences": kwargs.get("center_confidences"),
        }
        # Return unrefined corners (no-op refinement)
        return corners, set()

    CourtKeypointDetector._refine_near_corners = capture_refine  # type: ignore[assignment]

    for vi, v in enumerate(videos):
        path = get_video_path(v["id"])
        if path is None:
            continue

        result = kp_detector.detect(str(path))
        captured = getattr(kp_detector, '_captured', None)
        if captured is None or len(result.corners) != 4:
            continue

        cached_data.append({
            "id": v["id"],
            "gt": v["gt"],
            **captured,
        })
        print(f"  [{vi+1}/{len(videos)}] {v['id'][:12]}")
        gc.collect()

    # Restore original method
    CourtKeypointDetector._refine_near_corners = original_refine  # type: ignore[assignment]

    print(f"\nCached {len(cached_data)} videos. Unloading model...")
    del kp_detector
    gc.collect()

    # Phase 2: Replay refinement with different thresholds (instant)
    print(f"\nPhase 2: Sweeping {len(THRESHOLDS)} thresholds...\n")
    print(f"{'Thresh':>7s}  {'MCD':>7s}  {'Near':>7s}  {'Far':>7s}  "
          f"{'<0.05':>6s}  {'4f2b':>7s}  {'b026':>7s}  {'950f':>7s}")
    print("-" * 72)

    detector = CourtKeypointDetector.__new__(CourtKeypointDetector)
    # Initialize just enough for refinement methods to work
    detector.NEAR_CORNER_MAX_MARGIN = 0.20
    detector._CENTER_POINT_MIN_CONF = 0.3

    for thresh in THRESHOLDS:
        detector.OFF_SCREEN_CONF = thresh

        mcds, nears, fars = [], [], []
        problem: dict[str, float] = {}

        for item in cached_data:
            # Replay refinement with this threshold
            corners, _ = original_refine(
                detector,
                [dict(c) for c in item["pre_corners"]],
                dict(item["per_corner_confidence"]),
                center_points=item["center_points"],
                center_confidences=item["center_confidences"],
            )

            mcd, pc = compute_mcd(item["gt"], corners)
            mcds.append(mcd)
            nears.append((pc["near-left"] + pc["near-right"]) / 2)
            fars.append((pc["far-left"] + pc["far-right"]) / 2)

            for pfx in PROBLEM_PREFIXES:
                if item["id"].startswith(pfx):
                    problem[pfx] = mcd

        n = len(mcds)
        mean_mcd = sum(mcds) / n
        near = sum(nears) / n
        far = sum(fars) / n
        success = sum(1 for m in mcds if m < 0.05) / n * 100

        print(f"{thresh:7.2f}  {mean_mcd:7.4f}  {near:7.4f}  {far:7.4f}  "
              f"{success:5.1f}%  "
              f"{problem.get('4f2bd66a', -1):7.4f}  "
              f"{problem.get('b026dc6c', -1):7.4f}  "
              f"{problem.get('950fbe5d', -1):7.4f}")


if __name__ == "__main__":
    main()
