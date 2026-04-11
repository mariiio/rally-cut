"""Compare ball 3D metrics on session 'short' (target deployment) vs full dataset.

Loads the latest eval output, filters to videos in session short (the typical
1.5-3m tripod behind-baseline deployment), and reports stratified metrics.
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
from rallycut.evaluation.db import get_connection  # noqa: E402
from eval_ball_3d import (  # noqa: E402
    COURT_CORNERS,
    load_calibrated_videos,
    load_rallies_for_videos,
    _build_contact_sequence,
    _parse_ball_positions,
    _parse_player_positions,
)


SESSION_SHORT_ID = "41e1f30d-d5bb-4386-9908-fa37216eb535"


def _load_session_video_ids(session_id: str) -> set[str]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT video_id FROM session_videos WHERE session_id = %s",
                (session_id,),
            )
            return {str(row[0]) for row in cur.fetchall()}


def _estimate_net_ys(videos: dict[str, Any], rallies_by_video: dict[str, list[Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
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
            out[vid_id] = float(np.median(net_ys))
    return out


def _calibrate_heights(videos: dict[str, Any], net_y: dict[str, float]) -> dict[str, float]:
    heights: dict[str, float] = {}
    for vid_id, vcal in videos.items():
        cam = None
        if vid_id in net_y:
            cam = calibrate_camera_with_net(
                vcal.image_corners, COURT_CORNERS, vcal.width, vcal.height,
                net_y_image=net_y[vid_id],
            )
        if cam is None or not cam.is_valid:
            cam = calibrate_camera(
                vcal.image_corners, COURT_CORNERS, vcal.width, vcal.height,
            )
        if cam is not None and cam.is_valid:
            heights[vid_id] = float(cam.camera_position[2])
    return heights


def _summarise(speeds: list[float]) -> dict[str, Any]:
    if not speeds:
        return {"n": 0, "pct_in_range": None}
    arr = np.array(speeds)
    in_range = int(((arr >= 10) & (arr <= 35)).sum())
    return {
        "n": len(arr),
        "in_range": in_range,
        "pct_in_range": in_range / len(arr),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
    }


def main() -> None:
    if len(sys.argv) > 1:
        eval_path = Path(sys.argv[1])
    else:
        outs = sorted(Path("outputs/ball_3d").glob("eval_*.json"))
        eval_path = outs[-1]
    print(f"Using eval: {eval_path}")

    with eval_path.open() as f:
        eval_data = json.load(f)
    per_video = eval_data["per_video"]

    print("\nLoading session short videos from DB...")
    session_vids = _load_session_video_ids(SESSION_SHORT_ID)
    print(f"  {len(session_vids)} videos in session short")

    print("Loading all calibrated videos...")
    videos = load_calibrated_videos(None)
    rallies = load_rallies_for_videos(set(videos.keys()))
    rallies_by_video: dict[str, list[Any]] = defaultdict(list)
    for r in rallies:
        rallies_by_video[r.video_id].append(r)

    print("Estimating net_y per video...")
    net_ys = _estimate_net_ys(videos, rallies_by_video)

    print("Re-calibrating cameras to get heights...")
    heights = _calibrate_heights(videos, net_ys)

    short_heights = {v: h for v, h in heights.items() if v in session_vids}
    other_heights = {v: h for v, h in heights.items() if v not in session_vids}

    print(f"\n=== CAMERA HEIGHT DISTRIBUTIONS ===")
    for label, hs in [("session short", short_heights), ("other", other_heights)]:
        if not hs:
            print(f"  {label:<15s}: no heights")
            continue
        vals = list(hs.values())
        arr = np.array(vals)
        print(f"  {label:<15s}: n={len(vals)}, "
              f"min={arr.min():.2f}, median={np.median(arr):.2f}, max={arr.max():.2f}, "
              f"mean={arr.mean():.2f} m")

    print("\n  Session short per-video heights:")
    for vid, h in sorted(short_heights.items(), key=lambda x: x[1]):
        in_eval = vid in per_video
        n_arcs = per_video.get(vid, {}).get("arcs", 0)
        n_serves = len(per_video.get(vid, {}).get("serve_speeds", []))
        print(f"    {vid[:8]}  {h:.2f}m  arcs={n_arcs:>3d}  serves={n_serves:>2d}  {'(in eval)' if in_eval else '(NOT in eval)'}")

    # Aggregate metrics for session short.
    print("\n=== SESSION SHORT AGGREGATE METRICS ===")
    short_speeds: list[float] = []
    short_arcs = 0
    short_valid = 0
    for vid in session_vids:
        if vid in per_video:
            stats = per_video[vid]
            short_speeds.extend(stats.get("serve_speeds", []))
            short_arcs += int(stats.get("arcs", 0))
            short_valid += int(stats.get("valid", 0))

    other_speeds: list[float] = []
    other_arcs = 0
    other_valid = 0
    for vid, stats in per_video.items():
        if vid not in session_vids:
            other_speeds.extend(stats.get("serve_speeds", []))
            other_arcs += int(stats.get("arcs", 0))
            other_valid += int(stats.get("valid", 0))

    def fmt_summary(label: str, arcs: int, valid: int, speeds: list[float]) -> None:
        s = _summarise(speeds)
        pct = f"{s['pct_in_range']:.0%}" if s["n"] > 0 else "n/a"
        median = f"{s['median']:.1f}" if s["n"] > 0 else "n/a"
        mean = f"{s['mean']:.1f}" if s["n"] > 0 else "n/a"
        valid_pct = f"{valid / arcs:.0%}" if arcs > 0 else "n/a"
        print(f"  {label:<18s} arcs={arcs:>4d}  valid={valid:>3d} ({valid_pct:>4s})  "
              f"serves={s['n']:>3d}  in[10,35]={pct:>4s}  median={median:>5s}  mean={mean:>5s}")

    fmt_summary("Session short",  short_arcs, short_valid, short_speeds)
    fmt_summary("Other videos",   other_arcs, other_valid, other_speeds)
    fmt_summary("Full dataset",   short_arcs + other_arcs, short_valid + other_valid, short_speeds + other_speeds)

    # Speed histogram comparison.
    print("\n=== SERVE SPEED HISTOGRAM (session short vs other) ===")
    bins = [0, 5, 10, 15, 20, 25, 30, 35, 50]
    print(f"  {'Range':<12s}  {'Short':>10s}  {'Other':>10s}")
    for i in range(len(bins) - 1):
        lo, hi = bins[i], bins[i + 1]
        n_s = sum(1 for s in short_speeds if lo <= s < hi)
        n_o = sum(1 for s in other_speeds if lo <= s < hi)
        pct_s = f"{n_s/len(short_speeds):.0%}" if short_speeds else "n/a"
        pct_o = f"{n_o/len(other_speeds):.0%}" if other_speeds else "n/a"
        print(f"  [{lo:>2d},{hi:>2d})     {n_s:>4d} {pct_s:>4s}  {n_o:>4d} {pct_o:>4s}")


if __name__ == "__main__":
    main()
