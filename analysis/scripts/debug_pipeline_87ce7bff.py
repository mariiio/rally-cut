"""Run player tracking pipeline step-by-step for rally 87ce7bff with full logging."""

import sys
sys.path.insert(0, ".")

import logging
import json
import numpy as np
from collections import Counter

# Enable DEBUG logging for all tracking modules
logging.basicConfig(level=logging.DEBUG, format="%(name)s:%(levelname)s: %(message)s")
for name in [
    "rallycut.tracking.player_tracker",
    "rallycut.tracking.player_filter",
    "rallycut.tracking.color_repair",
    "rallycut.tracking.spatial_consistency",
    "rallycut.tracking.global_identity",
    "rallycut.tracking.tracklet_link",
]:
    logging.getLogger(name).setLevel(logging.DEBUG)

from rallycut.evaluation.tracking.db import (
    load_labeled_rallies,
    get_video_path,
    load_court_calibration,
)
from rallycut.tracking.player_tracker import PlayerTracker
from rallycut.tracking.player_filter import PlayerFilterConfig, compute_court_split, _find_net_from_bbox_clustering
from rallycut.tracking.ball_tracker import create_ball_tracker
from rallycut.court.calibration import CourtCalibrator


RALLY_ID = "87ce7bff-2dd3-434e-829c-365e0c53cfcb"


def print_track_stats(positions, label):
    """Print track statistics."""
    track_data = {}
    for p in positions:
        tid = p.track_id
        if tid not in track_data:
            track_data[tid] = {"sizes": [], "ys": [], "xs": [], "frames": set()}
        track_data[tid]["sizes"].append(p.width * p.height)
        track_data[tid]["ys"].append(p.y)
        track_data[tid]["xs"].append(p.x)
        track_data[tid]["frames"].add(p.frame_number)

    if not track_data:
        print(f"  [{label}] No positions!")
        return

    all_frames = set()
    for d in track_data.values():
        all_frames |= d["frames"]
    total_frames = max(all_frames) - min(all_frames) + 1 if all_frames else 1

    print(f"\n  [{label}] {len(positions)} positions, {len(track_data)} tracks, {total_frames} frames")
    print(f"  {'TID':>5} {'AvgSize':>10} {'AvgY':>8} {'AvgX':>8} {'Frames':>8} {'Pres%':>8} {'YRange':>15}")

    sorted_tids = sorted(track_data.keys(), key=lambda t: len(track_data[t]["frames"]), reverse=True)
    for tid in sorted_tids:
        d = track_data[tid]
        n = len(d["frames"])
        # Check for duplicate frames
        all_pos_count = len(d["sizes"])
        dups = all_pos_count - n
        dup_str = f" ({dups} dups!)" if dups > 0 else ""
        print(
            f"  {tid:>5} {np.mean(d['sizes']):>10.5f} {np.mean(d['ys']):>8.3f} "
            f"{np.mean(d['xs']):>8.3f} {n:>8} {n/total_frames*100:>7.1f}% "
            f"{min(d['ys']):.2f}-{max(d['ys']):.2f}{dup_str}"
        )

    # Check court_split_y
    split_result = _find_net_from_bbox_clustering(positions)
    print(f"  bbox_clustering split_result = {split_result}")


def main():
    print(f"Loading rally {RALLY_ID[:8]}...")
    rallies = load_labeled_rallies(rally_id=RALLY_ID)
    rally = rallies[0]
    video_path = get_video_path(rally.video_id)
    print(f"Video: {video_path}")
    print(f"Time: {rally.start_ms}ms - {rally.end_ms}ms ({(rally.end_ms - rally.start_ms)/1000:.1f}s)")

    # Production auto-loads calibration from DB
    cal_json = rally.court_calibration_json
    calibrator = None
    court_roi = None
    if cal_json and len(cal_json) == 4:
        calibrator = CourtCalibrator()
        image_corners = [(c["x"], c["y"]) for c in cal_json]
        calibrator.calibrate(image_corners)
        print(f"Court calibration: YES (calibrated={calibrator.is_calibrated})")
        print(f"  Corners: {image_corners}")

        # Compute calibration ROI (same as production CLI)
        from rallycut.tracking.player_tracker import compute_court_roi_from_calibration
        court_roi, cal_msg = compute_court_roi_from_calibration(calibrator)
        if court_roi:
            xs = [p[0] for p in court_roi]
            ys = [p[1] for p in court_roi]
            print(f"  ROI: x={min(xs):.2f}-{max(xs):.2f}, y={min(ys):.2f}-{max(ys):.2f}")
        else:
            print(f"  ROI failed: {cal_msg}")
    else:
        print("Court calibration: NO")

    # Run ball tracking (same as production)
    print("\n--- Ball Tracking ---")
    ball_tracker = create_ball_tracker()
    ball_result = ball_tracker.track_video(
        video_path,
        start_ms=rally.start_ms,
        end_ms=rally.end_ms,
    )
    ball_positions = ball_result.positions
    print(f"Ball positions: {len(ball_positions)}, detection rate: {ball_result.detection_rate*100:.1f}%")

    # Create player tracker (same as production)
    print("\n--- Player Tracking ---")
    player_tracker = PlayerTracker(court_roi=court_roi)

    # Run full pipeline with debug logging
    result = player_tracker.track_video(
        video_path,
        start_ms=rally.start_ms,
        end_ms=rally.end_ms,
        stride=1,
        ball_positions=ball_positions,
        filter_enabled=True,
        filter_config=PlayerFilterConfig(),
        court_calibrator=calibrator,
    )

    print(f"\n=== FINAL RESULT ===")
    print(f"Tracks: {len(set(p.track_id for p in result.positions))}")
    print(f"Primary tracks: {result.primary_track_ids}")
    print(f"Court split Y: {result.court_split_y}")
    print(f"Positions: {len(result.positions)}")
    print(f"Raw positions: {len(result.raw_positions)}")
    print(f"Avg players/frame: {result.avg_players_per_frame:.2f}")

    print_track_stats(result.positions, "Filtered")
    print_track_stats(result.raw_positions, "Raw")


if __name__ == "__main__":
    main()
