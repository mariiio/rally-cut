#!/usr/bin/env python3
"""Audit whether the Hough-based sideline detection (strategy 2) in _refine_near_corners
ever fires and whether it helps or hurts court detection accuracy.

Runs the keypoint detector twice per video:
  - Baseline: unmodified detector (all 3 strategies enabled)
  - No-Hough: strategy 2 (_refine_via_sideline_detection) monkeypatched to return None

Compares per-corner MCD and overall MCD for both runs.

Usage:
    cd analysis
    uv run python scripts/audit_hough_fallback.py
"""

from __future__ import annotations

import logging
import math
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

# Suppress noisy YOLO/ultralytics output
logging.basicConfig(level=logging.WARNING)

CORNER_NAMES = ["near-left", "near-right", "far-right", "far-left"]


def corner_distance(c1: dict[str, float], c2: dict[str, float]) -> float:
    """Euclidean distance between two normalised corners."""
    return math.sqrt((c1["x"] - c2["x"]) ** 2 + (c1["y"] - c2["y"]) ** 2)


def compute_mcd(
    pred: list[dict[str, float]],
    gt: list[dict[str, float]],
) -> tuple[float, dict[str, float]]:
    """Return mean corner distance and per-corner distances."""
    per = {}
    for name, p, g in zip(CORNER_NAMES, pred, gt):
        per[name] = corner_distance(p, g)
    return sum(per.values()) / len(per), per


# ---------------------------------------------------------------------------
# Strategy tracking: monkeypatch _refine_near_corners to record which
# strategy fires without modifying production code.
# ---------------------------------------------------------------------------

_strategy_log: dict[str, str] = {}  # video_id -> strategy used

def _make_tracking_refine(original_refine, video_id: str) -> Any:
    """Return a patched _refine_near_corners that records which strategy fired."""
    from rallycut.court.keypoint_detector import CourtKeypointDetector

    def patched_refine(
        self,
        corners,
        per_corner_confidence,
        conf_threshold=0.5,
        frame=None,
        center_points=None,
        center_confidences=None,
    ):
        nl_conf = per_corner_confidence.get("near-left", 1.0)
        nr_conf = per_corner_confidence.get("near-right", 1.0)

        # If both confident, no strategy needed
        if nl_conf >= conf_threshold and nr_conf >= conf_threshold:
            _strategy_log[video_id] = "none (both confident)"
            return original_refine(
                self, corners, per_corner_confidence,
                conf_threshold, frame, center_points, center_confidences,
            )

        # Check if strategy 1 would succeed (center points available + confident)
        if center_points is not None and len(center_points) == 2:
            cl_conf = (center_confidences or {}).get("center-left", 0.0)
            cr_conf = (center_confidences or {}).get("center-right", 0.0)
            min_conf = CourtKeypointDetector._CENTER_POINT_MIN_CONF
            if cl_conf >= min_conf and cr_conf >= min_conf:
                result = self._refine_via_center_points(
                    corners, per_corner_confidence, conf_threshold, center_points,
                )
                if result is not None:
                    _strategy_log[video_id] = "strategy_1_center_points"
                    return result

        # Check if strategy 2 (Hough) would succeed
        if frame is not None:
            result = self._refine_via_sideline_detection(
                corners, per_corner_confidence, conf_threshold, frame,
            )
            if result is not None:
                _strategy_log[video_id] = "strategy_2_hough"
                return result

        # Strategy 3 fallback
        _strategy_log[video_id] = "strategy_3_vp_fallback"
        return original_refine(
            self, corners, per_corner_confidence,
            conf_threshold, frame, center_points, center_confidences,
        )

    return patched_refine


def run_detection(detector_class, video_path: str, disable_hough: bool = False) -> Any:
    """Run CourtKeypointDetector.detect(), optionally with strategy 2 disabled."""
    from rallycut.court.keypoint_detector import CourtKeypointDetector

    detector = CourtKeypointDetector()

    if disable_hough:
        # Monkeypatch _refine_via_sideline_detection to always return None
        original = CourtKeypointDetector._refine_via_sideline_detection
        CourtKeypointDetector._refine_via_sideline_detection = lambda self, *args, **kwargs: None  # type: ignore[method-assign]
        try:
            result = detector.detect(video_path)
        finally:
            CourtKeypointDetector._refine_via_sideline_detection = original  # type: ignore[method-assign]
    else:
        result = detector.detect(video_path)

    return result


def main() -> None:
    from rallycut.court.keypoint_detector import CourtKeypointDetector
    from rallycut.evaluation.db import get_connection
    from rallycut.evaluation.tracking.db import get_video_path

    # Load GT calibrations
    query = """
        SELECT id, court_calibration_json, width, height
        FROM videos
        WHERE court_calibration_json IS NOT NULL
    """
    videos: list[dict[str, Any]] = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            for row in cur.fetchall():
                vid_id, cal_json, width, height = row
                if isinstance(cal_json, list) and len(cal_json) == 4:
                    videos.append({
                        "video_id": str(vid_id),
                        "gt_corners": cal_json,
                        "width": width or 1920,
                        "height": height or 1080,
                    })

    print(f"Found {len(videos)} videos with GT court calibration")

    # Check model exists
    detector_check = CourtKeypointDetector()
    if not detector_check.model_exists:
        print("ERROR: Keypoint model not found at", detector_check._model_path)
        print("Train it first with: uv run python scripts/train_court_keypoint_model.py")
        sys.exit(1)

    print(f"Model: {detector_check._model_path}\n")

    # Per-video results
    results: list[dict[str, Any]] = []

    for idx, vid in enumerate(videos):
        vid_id = vid["video_id"]
        gt_corners = vid["gt_corners"]

        video_path = get_video_path(vid_id)
        if video_path is None:
            print(f"[{idx+1}/{len(videos)}] {vid_id[:12]}  SKIP  (video not found)")
            results.append({"video_id": vid_id, "skipped": True})
            continue

        print(f"[{idx+1}/{len(videos)}] {vid_id[:12]}  ...", end=" ", flush=True)

        # ---- Baseline run (all strategies) ----
        # Patch to record which strategy fires
        original_refine = CourtKeypointDetector._refine_near_corners

        _strategy_log[vid_id] = "none (both confident)"  # default
        tracking_fn = _make_tracking_refine(original_refine, vid_id)
        CourtKeypointDetector._refine_near_corners = tracking_fn  # type: ignore[method-assign]
        try:
            baseline_result = CourtKeypointDetector().detect(str(video_path))
        finally:
            CourtKeypointDetector._refine_near_corners = original_refine  # type: ignore[method-assign]

        strategy_used = _strategy_log.get(vid_id, "unknown")

        if len(baseline_result.corners) != 4:
            print(f"FAIL  (no corners detected)")
            results.append({
                "video_id": vid_id, "skipped": False, "detection_failed": True,
                "strategy": strategy_used,
            })
            continue

        base_mcd, base_per = compute_mcd(baseline_result.corners, gt_corners)

        # ---- No-Hough run (strategy 2 disabled) ----
        nohough_result = run_detection(CourtKeypointDetector, str(video_path), disable_hough=True)

        if len(nohough_result.corners) != 4:
            nohough_mcd = None
            nohough_per: dict[str, float] | None = None
        else:
            nohough_mcd, nohough_per = compute_mcd(nohough_result.corners, gt_corners)

        # Delta: positive = baseline is better, negative = no-hough is better
        delta = None
        if nohough_mcd is not None:
            delta = nohough_mcd - base_mcd  # >0 means baseline (with Hough) is better

        tag = ""
        if strategy_used == "strategy_2_hough":
            tag = " *** HOUGH FIRED ***"

        nohough_str = f"{nohough_mcd:.4f}" if nohough_mcd is not None else "     N/A"
        delta_str = f"{delta:+.4f}" if delta is not None else "     N/A"
        print(
            f"strategy={strategy_used:<35s}  "
            f"MCD_base={base_mcd:.4f}  "
            f"MCD_nohough={nohough_str:>8s}  "
            f"delta={delta_str:>8s}"
            f"{tag}"
        )

        results.append({
            "video_id": vid_id,
            "skipped": False,
            "detection_failed": False,
            "strategy": strategy_used,
            "baseline_mcd": base_mcd,
            "baseline_per_corner": base_per,
            "nohough_mcd": nohough_mcd,
            "nohough_per_corner": nohough_per,
            "delta_mcd": delta,  # positive = baseline better, negative = no-hough better
        })

    # ---- Summary ----
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    valid = [r for r in results if not r.get("skipped") and not r.get("detection_failed")]
    skipped = [r for r in results if r.get("skipped")]
    failed = [r for r in results if r.get("detection_failed")]

    print(f"Total videos: {len(results)}  |  Evaluated: {len(valid)}  |  Skipped: {len(skipped)}  |  Failed: {len(failed)}")

    # Strategy breakdown
    from collections import Counter
    strat_counts = Counter(r["strategy"] for r in valid)
    print("\nStrategy usage:")
    for strat, count in sorted(strat_counts.items(), key=lambda x: -x[1]):
        print(f"  {strat:<40s}: {count}")

    # Videos where Hough fired
    hough_videos = [r for r in valid if r["strategy"] == "strategy_2_hough"]
    print(f"\nVideos where strategy 2 (Hough) fired: {len(hough_videos)}")
    if hough_videos:
        print(f"  {'video_id':<15}  {'MCD_baseline':>12}  {'MCD_nohough':>12}  {'delta':>8}  {'verdict'}")
        print(f"  {'-'*14}  {'-'*12}  {'-'*12}  {'-'*8}  {'-'*20}")
        for r in hough_videos:
            base = r["baseline_mcd"]
            noh = r["nohough_mcd"]
            d = r["delta_mcd"]
            if d is None:
                verdict = "no-hough failed"
            elif d > 0.001:
                verdict = "HOUGH HELPS"
            elif d < -0.001:
                verdict = "HOUGH HURTS"
            else:
                verdict = "neutral"
            print(f"  {r['video_id'][:14]:<15}  {base:>12.4f}  {noh if noh is not None else float('nan'):>12.4f}  {d if d is not None else float('nan'):>+8.4f}  {verdict}")

        # Per-corner breakdown for Hough videos
        print("\n  Per-corner MCD (baseline vs no-hough) for Hough-fired videos:")
        for r in hough_videos:
            print(f"\n  {r['video_id'][:14]}:")
            for cn in CORNER_NAMES:
                base_c = r["baseline_per_corner"].get(cn, float("nan"))
                noh_c = (r["nohough_per_corner"] or {}).get(cn, float("nan"))
                delta_c = noh_c - base_c if r["nohough_per_corner"] else float("nan")
                print(f"    {cn:<12}  base={base_c:.4f}  nohough={noh_c:.4f}  delta={delta_c:+.4f}")
    else:
        print("  -> Strategy 2 (Hough) never fires on any of the GT-calibrated videos.")
        print("     It is dead code for this dataset and can be safely removed.")

    # Overall MCD comparison
    has_both = [r for r in valid if r["baseline_mcd"] is not None and r["nohough_mcd"] is not None]
    if has_both:
        avg_base = sum(r["baseline_mcd"] for r in has_both) / len(has_both)
        avg_noh = sum(r["nohough_mcd"] for r in has_both) / len(has_both)
        print(f"\nOverall mean MCD (all {len(has_both)} videos with both runs):")
        print(f"  Baseline (with Hough): {avg_base:.4f}")
        print(f"  No-Hough:              {avg_noh:.4f}")
        print(f"  Delta:                 {avg_noh - avg_base:+.4f}  (positive = no-hough is worse)")
        if abs(avg_noh - avg_base) < 0.0001:
            print("  -> No measurable difference. Strategy 2 has zero net effect.")
        elif avg_noh > avg_base:
            print("  -> Hough provides marginal improvement across all videos.")
        else:
            print("  -> Removing Hough would IMPROVE overall MCD.")


if __name__ == "__main__":
    main()
