"""Trace track counts at each pipeline step with color store for FFmpeg segment."""
import sys
sys.path.insert(0, ".")

import logging
logging.basicConfig(level=logging.WARNING)

# Monkey-patch key functions to trace track counts
import rallycut.tracking.player_tracker as pt
import rallycut.tracking.player_filter as pf

_orig_track_video = pt.PlayerTracker.track_video

def _patched_track_video(self, *args, **kwargs):
    # Patch the internal pipeline to add logging
    _orig_stabilize = pf.stabilize_track_ids

    def _patched_stabilize(positions, config, **kw):
        tids = set(p.track_id for p in positions if p.track_id >= 0)
        print(f"\n  [PRE-STABILIZE] {len(positions)} positions, {len(tids)} tracks")
        team_assignments = kw.get('team_assignments', {})
        if team_assignments:
            for tid in sorted(tids):
                team = team_assignments.get(tid, '?')
                count = sum(1 for p in positions if p.track_id == tid)
                print(f"    Track {tid}: team={team}, {count} pos")
        result = _orig_stabilize(positions, config, **kw)
        tids2 = set(p.track_id for p in result[0] if p.track_id >= 0)
        print(f"  [POST-STABILIZE] {len(result[0])} positions, {len(tids2)} tracks: {sorted(tids2)}")
        return result
    pf.stabilize_track_ids = _patched_stabilize

    from rallycut.tracking import tracklet_link as tl
    _orig_link = tl.link_tracklets_by_appearance

    def _patched_link(positions, color_store, **kw):
        tids = set(p.track_id for p in positions if p.track_id >= 0)
        print(f"\n  [PRE-TRACKLET-LINK] {len(positions)} positions, {len(tids)} tracks")
        result = _orig_link(positions, color_store, **kw)
        tids2 = set(p.track_id for p in result[0] if p.track_id >= 0)
        print(f"  [POST-TRACKLET-LINK] {len(result[0])} positions, {len(tids2)} tracks: {sorted(tids2)}")
        return result
    tl.link_tracklets_by_appearance = _patched_link

    # Also patch color repair
    from rallycut.tracking import color_repair as cr
    _orig_split = cr.split_tracks_by_color

    def _patched_split(positions, color_store, **kw):
        tids = set(p.track_id for p in positions if p.track_id >= 0)
        print(f"\n  [PRE-COLOR-SPLIT] {len(positions)} positions, {len(tids)} tracks")
        result = _orig_split(positions, color_store, **kw)
        tids2 = set(p.track_id for p in result[0] if p.track_id >= 0)
        print(f"  [POST-COLOR-SPLIT] {len(result[0])} positions, {len(tids2)} tracks (splits: {result[1]})")
        return result
    cr.split_tracks_by_color = _patched_split

    # Patch global identity
    from rallycut.tracking import global_identity as gi
    _orig_global = gi.optimize_global_identity

    def _patched_global(positions, team_assignments, color_store, **kw):
        tids = set(p.track_id for p in positions if p.track_id >= 0)
        print(f"\n  [PRE-GLOBAL-IDENTITY] {len(positions)} positions, {len(tids)} tracks: {sorted(tids)}")
        result = _orig_global(positions, team_assignments, color_store, **kw)
        tids2 = set(p.track_id for p in result[0] if p.track_id >= 0)
        print(f"  [POST-GLOBAL-IDENTITY] {len(result[0])} positions, {len(tids2)} tracks: {sorted(tids2)}")
        gr = result[1]
        if gr.skipped:
            print(f"    Skipped: {gr.skip_reason}")
        else:
            print(f"    Segments: {gr.num_segments}, Remapped: {gr.num_remapped}, Interactions: {gr.num_interactions}")
        return result
    gi.optimize_global_identity = _patched_global

    # Patch analyze_tracks
    _orig_analyze = pf.PlayerFilter.analyze_tracks

    def _patched_analyze(self_filter, all_positions):
        tids = set(p.track_id for p in all_positions if p.track_id >= 0)
        print(f"\n  [ANALYZE_TRACKS INPUT] {len(all_positions)} positions, {len(tids)} tracks: {sorted(tids)}")
        return _orig_analyze(self_filter, all_positions)
    pf.PlayerFilter.analyze_tracks = _patched_analyze

    result = _orig_track_video(self, *args, **kwargs)

    # Restore
    pf.stabilize_track_ids = _orig_stabilize
    tl.link_tracklets_by_appearance = _orig_link
    cr.split_tracks_by_color = _orig_split
    pf.PlayerFilter.analyze_tracks = _orig_analyze

    return result

pt.PlayerTracker.track_video = _patched_track_video

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
    calibrator = CourtCalibrator()
    calibrator.calibrate([(c["x"], c["y"]) for c in CAL_JSON])
    court_roi, _ = compute_court_roi_from_calibration(calibrator)

    print("Ball tracking...")
    ball_tracker = create_ball_tracker()
    ball_result = ball_tracker.track_video(VIDEO)

    print("\nPlayer tracking (with filter + color store)...")
    player_tracker = PlayerTracker(court_roi=court_roi)
    result = player_tracker.track_video(
        VIDEO,
        stride=1,
        ball_positions=ball_result.positions,
        filter_enabled=True,
        filter_config=PlayerFilterConfig(),
        court_calibrator=calibrator,
    )

    print(f"\n=== FINAL ===")
    print(f"Primary tracks: {result.primary_track_ids}")
    print(f"Court split Y: {result.court_split_y}")
    print(f"Unique filtered tracks: {len(set(p.track_id for p in result.positions))}")
    print(f"Avg players: {result.avg_players_per_frame:.2f}")


if __name__ == "__main__":
    main()
