"""Count unique tracks at each pipeline step for FFmpeg segment of rally 87ce7bff."""
import sys
sys.path.insert(0, ".")

import logging
logging.basicConfig(level=logging.WARNING)

from rallycut.court.calibration import CourtCalibrator
from rallycut.tracking.player_tracker import PlayerTracker, compute_court_roi_from_calibration
from rallycut.tracking.player_filter import (
    PlayerFilterConfig,
    PlayerFilter,
    classify_teams,
    compute_court_split,
    remove_stationary_background_tracks,
    stabilize_track_ids,
)
from rallycut.tracking.spatial_consistency import enforce_spatial_consistency
from rallycut.tracking.ball_tracker import create_ball_tracker

VIDEO = "/tmp/rally_87ce7bff_segment.mp4"
CAL_JSON = [
    {"x": -0.3267, "y": 0.8492},
    {"x": 1.2133, "y": 0.7781},
    {"x": 0.6883, "y": 0.4936},
    {"x": 0.3400, "y": 0.4981},
]


def count_tracks(positions):
    tids = set(p.track_id for p in positions if p.track_id >= 0)
    return len(tids), sorted(tids)


def main():
    # Calibration
    calibrator = CourtCalibrator()
    calibrator.calibrate([(c["x"], c["y"]) for c in CAL_JSON])
    court_roi, _ = compute_court_roi_from_calibration(calibrator)

    # Ball tracking
    print("Ball tracking...")
    ball_tracker = create_ball_tracker()
    ball_result = ball_tracker.track_video(VIDEO)
    ball_positions = ball_result.positions

    # Player tracking (raw YOLO+BoTSORT)
    print("Player tracking (raw)...")
    player_tracker = PlayerTracker(court_roi=court_roi)

    # We need to manually step through the pipeline
    # First, get raw positions using track_video WITHOUT filter
    raw_result = player_tracker.track_video(VIDEO, stride=1, filter_enabled=False, court_calibrator=calibrator)
    positions = raw_result.positions
    n, tids = count_tracks(positions)
    print(f"\n[RAW] {len(positions)} positions, {n} tracks: {tids}")

    # Check color store
    color_store = player_tracker._last_color_store if hasattr(player_tracker, '_last_color_store') else None
    print(f"[RAW] Color store: {color_store is not None and color_store.has_data() if color_store else 'N/A'}")

    # Now manually run the filter pipeline
    config = PlayerFilterConfig()
    total_frames = raw_result.frame_count

    # Pre-step: stationary background
    positions, removed_bg = remove_stationary_background_tracks(positions, config, total_frames=total_frames)
    n, tids = count_tracks(positions)
    print(f"\n[STATIONARY_BG] {len(positions)} positions, {n} tracks, removed: {removed_bg}")

    # Step 0: spatial consistency
    positions, consistency = enforce_spatial_consistency(positions)
    n, tids = count_tracks(positions)
    print(f"[SPATIAL_CONSIST] {len(positions)} positions, {n} tracks, jumps: {consistency.jump_splits}")

    # Step 0b: color repair (WOULD BE SKIPPED if no color store)
    # For debugging, check what would happen
    print(f"\n[COLOR_STORE] Would color repair run? color_store exists: {color_store is not None}")

    # Step 1: stabilize
    positions, id_mapping = stabilize_track_ids(positions, config)
    n, tids = count_tracks(positions)
    print(f"[STABILIZE] {len(positions)} positions, {n} tracks: {tids}")
    if id_mapping:
        print(f"  Mappings: {id_mapping}")

    # Step 2: analyze_tracks
    player_filter = PlayerFilter(
        ball_positions=ball_positions,
        total_frames=total_frames,
        config=config,
        court_calibrator=calibrator,
    )
    player_filter.analyze_tracks(positions)
    print(f"\n[ANALYZE] track_stats has {len(player_filter.track_stats)} tracks")
    print(f"[ANALYZE] primary: {sorted(player_filter.primary_tracks)}")
    print(f"[ANALYZE] court_split_y: {player_filter.court_split_y}")

    for tid in sorted(player_filter.track_stats.keys()):
        s = player_filter.track_stats[tid]
        court_str = f", court={s.court_presence_ratio:.2f}" if s.has_court_stats else ""
        print(f"  Track {tid}: presence={s.presence_rate:.2f}, spread={s.position_spread:.4f}, bbox={s.avg_bbox_area:.4f}, ball={s.ball_proximity_score:.2f}{court_str}")


if __name__ == "__main__":
    main()
