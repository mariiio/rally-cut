"""Debug court_presence filtering for FFmpeg segment of rally 87ce7bff."""
import sys
sys.path.insert(0, ".")

import logging
import numpy as np

# Enable targeted logging
logging.basicConfig(level=logging.WARNING)
for name in [
    "rallycut.tracking.player_filter",
]:
    logging.getLogger(name).setLevel(logging.DEBUG)

from rallycut.court.calibration import CourtCalibrator
from rallycut.tracking.player_tracker import PlayerTracker, compute_court_roi_from_calibration
from rallycut.tracking.player_filter import PlayerFilterConfig
from rallycut.tracking.ball_tracker import create_ball_tracker

VIDEO = "/tmp/rally_87ce7bff_segment.mp4"

CAL_JSON = [
    {"x": -0.3267, "y": 0.8492},
    {"x": 1.2133, "y": 0.7781},
    {"x": 0.6883, "y": 0.4936},
    {"x": 0.3400, "y": 0.4981},
]


def main():
    # Setup calibration
    calibrator = CourtCalibrator()
    image_corners = [(c["x"], c["y"]) for c in CAL_JSON]
    calibrator.calibrate(image_corners)
    print(f"Calibrated: {calibrator.is_calibrated}")

    court_roi, _ = compute_court_roi_from_calibration(calibrator)
    if court_roi:
        xs = [p[0] for p in court_roi]
        ys = [p[1] for p in court_roi]
        print(f"ROI: x={min(xs):.2f}-{max(xs):.2f}, y={min(ys):.2f}-{max(ys):.2f}")

    # Ball tracking
    print("\nBall tracking...")
    ball_tracker = create_ball_tracker()
    ball_result = ball_tracker.track_video(VIDEO)
    ball_positions = ball_result.positions
    print(f"Ball positions: {len(ball_positions)}")

    # Player tracking
    print("\nPlayer tracking...")
    player_tracker = PlayerTracker(court_roi=court_roi)

    result = player_tracker.track_video(
        VIDEO,
        stride=1,
        ball_positions=ball_positions,
        filter_enabled=True,
        filter_config=PlayerFilterConfig(),
        court_calibrator=calibrator,
    )

    print(f"\n=== RESULT ===")
    print(f"Primary tracks: {result.primary_track_ids}")
    print(f"Court split Y: {result.court_split_y}")
    print(f"Unique tracks: {len(set(p.track_id for p in result.positions))}")
    print(f"Avg players: {result.avg_players_per_frame:.2f}")

    # Analyze tracks in detail
    track_data = {}
    for p in result.raw_positions:
        tid = p.track_id
        if tid not in track_data:
            track_data[tid] = {"sizes": [], "ys": [], "xs": [], "frames": set()}
        track_data[tid]["sizes"].append(p.width * p.height)
        track_data[tid]["ys"].append(p.y)
        track_data[tid]["xs"].append(p.x)
        track_data[tid]["frames"].add(p.frame_number)

    print(f"\nRaw tracks ({len(track_data)}):")
    for tid in sorted(track_data.keys(), key=lambda t: len(track_data[t]["frames"]), reverse=True):
        d = track_data[tid]
        n = len(d["frames"])
        print(f"  Track {tid}: {n} frames, avg_y={np.mean(d['ys']):.3f}, avg_x={np.mean(d['xs']):.3f}, avg_size={np.mean(d['sizes']):.5f}")

    # Check filtered tracks
    filtered_data = {}
    for p in result.positions:
        tid = p.track_id
        if tid not in filtered_data:
            filtered_data[tid] = {"frames": set()}
        filtered_data[tid]["frames"].add(p.frame_number)

    print(f"\nFiltered tracks ({len(filtered_data)}):")
    for tid in sorted(filtered_data.keys(), key=lambda t: len(filtered_data[t]["frames"]), reverse=True):
        n = len(filtered_data[tid]["frames"])
        print(f"  Track {tid}: {n} frames")


if __name__ == "__main__":
    main()
