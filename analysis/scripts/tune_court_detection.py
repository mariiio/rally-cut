#!/usr/bin/env python3
"""Grid search for optimal court detection parameters.

Loads all calibrated videos from DB, runs CourtDetector with each config
in the grid, and reports best config by mean IoU and detection rate.

Frame sampling is cached across configs (main bottleneck), so only
line detection + aggregation rerun per config.

Usage:
    uv run python scripts/tune_court_detection.py --grid quick
    uv run python scripts/tune_court_detection.py --grid detection -o results.json
    uv run python scripts/tune_court_detection.py --grid dark --debug
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from scripts.eval_court_detection import mean_corner_distance, rasterized_iou

GRIDS: dict[str, dict[str, list[Any]]] = {
    "quick": {
        "dbscan_eps": [0.02, 0.03, 0.05],
        "min_temporal_support": [4, 6, 8],
        "hough_threshold": [30, 40],
    },
    "detection": {
        "white_saturation_max": [50, 60, 80],
        "white_value_offset": [20, 30, 40],
        "canny_low": [30, 50],
        "canny_high": [120, 150],
        "hough_threshold": [30, 40, 50],
    },
    "temporal": {
        "dbscan_eps": [0.02, 0.03, 0.04, 0.05],
        "dbscan_min_samples": [3, 5, 7],
        "min_temporal_support": [4, 6, 8, 10],
        "theta_scale": [0.2, 0.3, 0.4],
    },
    "dark": {
        "dark_saturation_max": [40, 50, 60],
        "dark_value_offset": [35, 45, 55],
        "dark_value_min": [25, 35, 45],
        "enable_dark_detection": [True, False],
    },
    "full": {
        "dbscan_eps": [0.02, 0.03, 0.05],
        "min_temporal_support": [4, 6, 8],
        "hough_threshold": [30, 40],
        "white_saturation_max": [50, 60, 80],
        "white_value_offset": [20, 30, 40],
        "enable_dark_detection": [True, False],
    },
}


def load_calibrated_videos() -> list[dict[str, Any]]:
    """Load all videos with court calibration GT from the database."""
    from rallycut.evaluation.db import get_connection
    from rallycut.evaluation.tracking.db import get_video_path

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
                    video_path = get_video_path(str(vid_id))
                    if video_path is not None:
                        videos.append({
                            "video_id": str(vid_id),
                            "gt_corners": cal_json,
                            "width": width or 1920,
                            "height": height or 1080,
                            "path": video_path,
                        })
    return videos


def evaluate_config(
    config_overrides: dict[str, Any],
    video_frames: dict[str, list[np.ndarray]],
    videos: list[dict[str, Any]],
) -> dict[str, Any]:
    """Run detection with a specific config on all cached frames."""
    from rallycut.court.detector import CourtDetectionConfig, CourtDetector

    config = CourtDetectionConfig(**config_overrides)
    detector = CourtDetector(config)

    total_mcd = 0.0
    total_iou = 0.0
    successes = 0
    detected = 0
    n_videos = 0

    per_video: list[dict[str, Any]] = []

    for video_info in videos:
        vid_id = video_info["video_id"]
        frames = video_frames.get(vid_id, [])
        if not frames:
            continue

        n_videos += 1
        gt_corners = video_info["gt_corners"]

        result = detector.detect_from_frames(frames)

        if result.corners and len(result.corners) == 4:
            mcd = mean_corner_distance(result.corners, gt_corners)
            gt_poly = [(c["x"], c["y"]) for c in gt_corners]
            det_poly = [(c["x"], c["y"]) for c in result.corners]
            iou = rasterized_iou(det_poly, gt_poly)

            detected += 1
            total_mcd += mcd
            total_iou += iou
            if mcd < 0.05:
                successes += 1

            per_video.append({
                "video_id": vid_id,
                "mcd": mcd,
                "iou": iou,
                "confidence": result.confidence,
            })
        else:
            per_video.append({
                "video_id": vid_id,
                "mcd": None,
                "iou": None,
                "confidence": result.confidence,
            })

    mean_iou = total_iou / detected if detected > 0 else 0.0
    mean_mcd = total_mcd / detected if detected > 0 else float("inf")
    detection_rate = detected / n_videos if n_videos > 0 else 0.0

    return {
        "config": config_overrides,
        "mean_iou": mean_iou,
        "mean_mcd": mean_mcd,
        "detection_rate": detection_rate,
        "detected": detected,
        "successes": successes,
        "n_videos": n_videos,
        "per_video": per_video,
    }


def generate_configs(grid_name: str) -> list[dict[str, Any]]:
    """Generate all parameter combinations for a grid."""
    grid = GRIDS[grid_name]

    # Validate all grid keys are valid CourtDetectionConfig fields
    from rallycut.court.detector import CourtDetectionConfig

    valid_fields = {f.name for f in dataclasses.fields(CourtDetectionConfig)}
    for key in grid:
        if key not in valid_fields:
            raise ValueError(
                f"Grid '{grid_name}' has invalid parameter '{key}'. "
                f"Valid fields: {sorted(valid_fields)}"
            )

    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    configs: list[dict[str, Any]] = []
    for combo in itertools.product(*values):
        configs.append(dict(zip(keys, combo)))
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid search for court detection")
    parser.add_argument(
        "--grid", required=True, choices=list(GRIDS.keys()),
        help="Grid to search",
    )
    parser.add_argument("-o", "--output", type=str, help="Output JSON path")
    parser.add_argument("--debug", action="store_true", help="Save debug images for best config")
    args = parser.parse_args()

    from rallycut.court.detector import CourtDetector

    # Load videos
    print("Loading calibrated videos from DB...")
    videos = load_calibrated_videos()
    print(f"Found {len(videos)} videos with calibration GT and local paths\n")

    if not videos:
        print("No calibrated videos found.")
        return

    # Cache frames for all videos (one-time cost).
    # Uses CourtDetector.sample_frames() to stay in sync with the detection pipeline.
    print("Sampling frames from all videos (cached across configs)...")
    sampler = CourtDetector()
    video_frames: dict[str, list[np.ndarray]] = {}
    for video_info in videos:
        vid_id = video_info["video_id"]
        frames = sampler.sample_frames(video_info["path"])
        video_frames[vid_id] = frames
        print(f"  {vid_id[:12]}: {len(frames)} frames")
    print()

    # Generate configs (validates parameter names)
    configs = generate_configs(args.grid)
    print(f"Grid '{args.grid}': {len(configs)} configurations\n")

    # Run grid search
    best_result: dict[str, Any] | None = None
    all_results: list[dict[str, Any]] = []

    t0 = time.time()
    for i, config_overrides in enumerate(configs):
        result = evaluate_config(config_overrides, video_frames, videos)
        all_results.append(result)

        # Track best by mean IoU (primary), detection rate (secondary)
        if best_result is None or (
            result["mean_iou"] > best_result["mean_iou"]
            or (
                abs(result["mean_iou"] - best_result["mean_iou"]) < 0.001
                and result["detection_rate"] > best_result["detection_rate"]
            )
        ):
            best_result = result

        # Progress
        elapsed = time.time() - t0
        rate = (i + 1) / elapsed
        remaining = (len(configs) - i - 1) / rate if rate > 0 else 0
        config_str = ", ".join(f"{k}={v}" for k, v in config_overrides.items())
        print(
            f"  [{i + 1}/{len(configs)}] "
            f"IoU={result['mean_iou']:.3f} det={result['detected']}/{result['n_videos']}  "
            f"best IoU={best_result['mean_iou']:.3f}  "
            f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)  "
            f"{config_str}"
        )

    elapsed = time.time() - t0
    print(f"\nGrid search completed in {elapsed:.1f}s\n")

    # Report best
    if best_result:
        print("=" * 70)
        print("BEST CONFIGURATION:")
        print("=" * 70)
        for k, v in best_result["config"].items():
            print(f"  {k}: {v}")
        print()
        print(f"  Mean IoU:       {best_result['mean_iou']:.3f}")
        print(f"  Mean MCD:       {best_result['mean_mcd']:.4f}")
        print(f"  Detection rate: {best_result['detected']}/{best_result['n_videos']} "
              f"({best_result['detection_rate'] * 100:.1f}%)")
        print(f"  Successes:      {best_result['successes']}/{best_result['n_videos']} "
              f"(MCD < 5%)")
        print()

        # Show top 5
        all_results.sort(key=lambda r: (-r["mean_iou"], -r["detection_rate"]))
        print("Top 5 configurations:")
        print(f"{'#':>3s}  {'IoU':>6s}  {'MCD':>7s}  {'Det':>5s}  Config")
        print("-" * 70)
        for i, r in enumerate(all_results[:5]):
            config_str = ", ".join(f"{k}={v}" for k, v in r["config"].items())
            print(
                f"{i + 1:>3d}  "
                f"{r['mean_iou']:.3f}  "
                f"{r['mean_mcd']:.4f}  "
                f"{r['detected']:>2d}/{r['n_videos']:<2d}  "
                f"{config_str}"
            )
        print()

        # Per-video breakdown for best config
        print("Per-video results (best config):")
        print(f"{'Video':>12s}  {'MCD':>7s}  {'IoU':>5s}  {'Conf':>5s}")
        print("-" * 40)
        for pv in best_result["per_video"]:
            if pv["mcd"] is not None:
                print(
                    f"{pv['video_id'][:12]:>12s}  "
                    f"{pv['mcd']:.4f}  "
                    f"{pv['iou']:.3f}  "
                    f"{pv['confidence']:.3f}"
                )
            else:
                print(f"{pv['video_id'][:12]:>12s}  {'FAIL':>7s}")

    # Save results
    if args.output:
        # Strip per_video from all results to keep file small
        compact_results = []
        for r in all_results:
            compact = {k: v for k, v in r.items() if k != "per_video"}
            compact_results.append(compact)

        output_data = {
            "grid": args.grid,
            "n_configs": len(configs),
            "n_videos": len(videos),
            "elapsed_seconds": round(elapsed, 1),
            "best": best_result,
            "all_results": compact_results,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")

    # Debug images for best config
    if args.debug and best_result:
        from rallycut.court.detector import CourtDetectionConfig
        from rallycut.court.detector import CourtDetector as DebugDetector

        debug_dir = Path("debug_court_tune")
        debug_dir.mkdir(exist_ok=True)
        config = CourtDetectionConfig(**best_result["config"])
        detector = DebugDetector(config)

        for video_info in videos:
            vid_id = video_info["video_id"]
            frames = video_frames.get(vid_id, [])
            if not frames:
                continue

            detect_result = detector.detect_from_frames(frames)
            debug_frame = detector.create_debug_image(frames[len(frames) // 2], detect_result)
            out_path = debug_dir / f"{vid_id[:12]}_tune_debug.jpg"
            cv2.imwrite(str(out_path), debug_frame)

        print(f"Debug images saved to {debug_dir}/")


if __name__ == "__main__":
    main()
