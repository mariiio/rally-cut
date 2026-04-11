"""Stratify ball 3D eval metrics by camera height tier.

Loads the latest eval_ball_3d output and re-calibrates cameras to get
per-video camera heights, then groups serve speeds by tier to answer:
does raising the minimum camera height threshold improve metrics?

Usage:
    cd analysis
    uv run python scripts/analyze_ball_3d_by_camera_height.py
    uv run python scripts/analyze_ball_3d_by_camera_height.py <eval.json>
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from rallycut.court.camera_model import calibrate_camera, calibrate_camera_with_net  # noqa: E402
from rallycut.court.calibration import CourtCalibrator  # noqa: E402
from eval_ball_3d import (  # noqa: E402
    COURT_CORNERS,
    GATE_SERVE_SPEED,
    load_calibrated_videos,
    load_rallies_for_videos,
    _build_contact_sequence,
    _parse_ball_positions,
    _parse_player_positions,
)


TIERS: list[tuple[str, float, float]] = [
    ("<1.5m",   0.0,  1.5),
    ("1.5-2m",  1.5,  2.0),
    ("2-2.5m",  2.0,  2.5),
    ("≥2.5m",   2.5, 10.0),
]


def _estimate_net_ys(videos: dict[str, Any], rallies_by_video: dict[str, list[Any]]) -> dict[str, float]:
    """Estimate per-video net_y from first 15 rallies (same logic as eval_ball_3d)."""
    net_y_per_video: dict[str, float] = {}
    for vid_id, vid_rallies in rallies_by_video.items():
        if vid_id not in videos:
            continue
        vcal = videos[vid_id]
        calibrator = CourtCalibrator()
        calibrator.calibrate(vcal.image_corners)

        net_ys: list[float] = []
        for rally in vid_rallies[:15]:
            bp = _parse_ball_positions(rally.ball_positions_json)
            pp = _parse_player_positions(rally.positions_json)
            if len(bp) < 20:
                continue
            cs = _build_contact_sequence(bp, pp, calibrator)
            if 0.1 < cs.net_y < 0.9:
                net_ys.append(cs.net_y)
        if net_ys:
            net_y_per_video[vid_id] = float(np.median(net_ys))
    return net_y_per_video


def _calibrate_all(videos: dict[str, Any], net_y_per_video: dict[str, float]) -> dict[str, float]:
    """Return camera height in metres per video_id."""
    heights: dict[str, float] = {}
    for vid_id, vcal in videos.items():
        cam = None
        if vid_id in net_y_per_video:
            cam = calibrate_camera_with_net(
                vcal.image_corners, COURT_CORNERS,
                vcal.width, vcal.height,
                net_y_image=net_y_per_video[vid_id],
            )
        if cam is None or not cam.is_valid:
            cam = calibrate_camera(
                vcal.image_corners, COURT_CORNERS, vcal.width, vcal.height,
            )
        if cam is not None and cam.is_valid:
            heights[vid_id] = float(cam.camera_position[2])
    return heights


def _tier_for_height(h: float) -> str:
    for label, lo, hi in TIERS:
        if lo <= h < hi:
            return label
    return "?"


def _summarise_speeds(speeds: list[float]) -> dict[str, float]:
    if not speeds:
        return {"n": 0}
    arr = np.array(speeds)
    in_range = int(((arr >= 10) & (arr <= 35)).sum())
    return {
        "n":        int(len(arr)),
        "in_range": in_range,
        "pct":      float(in_range / len(arr)),
        "mean":     float(arr.mean()),
        "median":   float(np.median(arr)),
        "std":      float(arr.std()),
        "min":      float(arr.min()),
        "max":      float(arr.max()),
    }


def main() -> None:
    # Find the latest eval file if not specified.
    if len(sys.argv) > 1:
        eval_path = Path(sys.argv[1])
    else:
        outs = sorted(Path("outputs/ball_3d").glob("eval_*.json"))
        if not outs:
            print("No eval_*.json files found in outputs/ball_3d/")
            sys.exit(1)
        eval_path = outs[-1]
    print(f"Using eval: {eval_path}")

    with eval_path.open() as f:
        eval_data = json.load(f)
    per_video = eval_data["per_video"]
    print(f"  {len(per_video)} videos in eval output")

    print("\nLoading calibrated videos from DB...")
    videos = load_calibrated_videos(None)
    print(f"  {len(videos)} calibrated videos")

    print("Loading rallies (for net_y estimation)...")
    rallies = load_rallies_for_videos(set(videos.keys()))
    rallies_by_video: dict[str, list[Any]] = defaultdict(list)
    for r in rallies:
        rallies_by_video[r.video_id].append(r)

    print("Estimating net_y per video...")
    net_y_per_video = _estimate_net_ys(videos, rallies_by_video)
    print(f"  net_y estimated for {len(net_y_per_video)}/{len(videos)} videos")

    print("Re-calibrating cameras to get heights...")
    heights = _calibrate_all(videos, net_y_per_video)
    print(f"  calibrated {len(heights)}/{len(videos)} cameras")

    # Stratify videos by tier.
    videos_by_tier: dict[str, list[str]] = defaultdict(list)
    for vid_id, h in heights.items():
        videos_by_tier[_tier_for_height(h)].append(vid_id)

    print("\n=== CAMERA HEIGHT DISTRIBUTION ===")
    for label, lo, hi in TIERS:
        n = len(videos_by_tier.get(label, []))
        print(f"  {label:<8s} ({lo:.1f}-{hi:.1f}m): {n} videos")

    # Aggregate serve speeds per tier.
    speeds_per_tier: dict[str, list[float]] = defaultdict(list)
    arcs_per_tier: dict[str, int] = defaultdict(int)
    valid_per_tier: dict[str, int] = defaultdict(int)

    for vid_id, stats in per_video.items():
        if vid_id not in heights:
            continue
        tier = _tier_for_height(heights[vid_id])
        speeds_per_tier[tier].extend(stats.get("serve_speeds", []))
        arcs_per_tier[tier]  += int(stats.get("arcs", 0))
        valid_per_tier[tier] += int(stats.get("valid", 0))

    print("\n=== SERVE SPEED BY CAMERA HEIGHT TIER ===")
    print(f"  Gate: {GATE_SERVE_SPEED:.0%} of arcs in [10, 35] m/s")
    print()
    header = f"  {'Tier':<8s} {'Videos':>7s} {'Arcs':>6s} {'Valid':>6s} {'Serves':>7s} {'InRange':>9s} {'Mean':>6s} {'Median':>7s} {'Std':>6s}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for label, _, _ in TIERS:
        summary = _summarise_speeds(speeds_per_tier.get(label, []))
        n_vids = len(videos_by_tier.get(label, []))
        if summary["n"] == 0:
            print(f"  {label:<8s} {n_vids:>7d} {arcs_per_tier[label]:>6d} {valid_per_tier[label]:>6d} {0:>7d} {'':>9s} {'':>6s} {'':>7s} {'':>6s}")
            continue
        print(
            f"  {label:<8s} {n_vids:>7d} {arcs_per_tier[label]:>6d} {valid_per_tier[label]:>6d} "
            f"{summary['n']:>7d} {summary['pct']:>8.0%} ({summary['in_range']}) "
            f"{summary['mean']:>5.1f} {summary['median']:>6.1f} {summary['std']:>5.1f}"
        )

    print("\n=== CUMULATIVE (≥threshold) ===")
    header2 = f"  {'Min height':<12s} {'Videos':>7s} {'Arcs':>6s} {'Serves':>7s} {'InRange':>9s} {'Median':>7s}"
    print(header2)
    print("  " + "-" * (len(header2) - 2))
    for label, lo, _ in TIERS:
        cum_speeds: list[float] = []
        cum_vids = 0
        cum_arcs = 0
        for vid_id, h in heights.items():
            if h >= lo and vid_id in per_video:
                cum_vids += 1
                cum_arcs += int(per_video[vid_id].get("arcs", 0))
                cum_speeds.extend(per_video[vid_id].get("serve_speeds", []))
        s = _summarise_speeds(cum_speeds)
        in_range_str = f"{s['pct']:.0%} ({s['in_range']})" if s["n"] > 0 else ""
        median_str = f"{s['median']:.1f}" if s["n"] > 0 else ""
        print(
            f"  ≥{lo:.1f}m       {cum_vids:>7d} {cum_arcs:>6d} {s['n']:>7d} {in_range_str:>9s} {median_str:>7s}"
        )


if __name__ == "__main__":
    main()
