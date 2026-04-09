"""Diagnose fragment linking for a specific rally.

Traces each pipeline step to show how raw BoT-SORT fragments get linked/filtered.
"""

import json
import logging
import sys

logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")

from rallycut.tracking.player_tracker import PlayerPosition
from rallycut.tracking.tracklet_link import (
    _compute_track_summary,
    _compute_average_histogram,
    _compute_blended_distance,
    _bhattacharyya_distance,
    DEFAULT_MAX_SPATIAL_DISPLACEMENT,
    DEFAULT_MAX_TEMPORAL_GAP,
    DEFAULT_MIN_TRACK_FRAMES,
)


def load_positions(json_path: str, raw: bool = False) -> list[PlayerPosition]:
    with open(json_path) as f:
        data = json.load(f)
    key = "rawPositions" if raw and "rawPositions" in data else "positions"
    positions = []
    for p in data[key]:
        positions.append(PlayerPosition(
            frame_number=p["frameNumber"],
            track_id=p["trackId"],
            x=p["x"],
            y=p["y"],
            width=p["width"],
            height=p["height"],
            confidence=p["confidence"],
        ))
    return positions


def main():
    json_path = sys.argv[1] if len(sys.argv) > 1 else None
    if not json_path:
        print("Usage: python diagnose_fragment_linking.py <tracking_output.json>")
        sys.exit(1)

    positions = load_positions(json_path, raw=True)
    tracks = _compute_track_summary(positions)

    print(f"\n{'='*70}")
    print(f"TRACK SUMMARY ({len(tracks)} tracks)")
    print(f"{'='*70}")
    print(f"{'TID':>5} {'Frames':>8} {'First':>6} {'Last':>6} {'FirstPos':>16} {'LastPos':>16}")
    for tid in sorted(tracks.keys()):
        info = tracks[tid]
        fp = info["first_pos"]
        lp = info["last_pos"]
        eligible = "✓" if info["count"] >= DEFAULT_MIN_TRACK_FRAMES else "✗"
        print(
            f"{tid:>5} {info['count']:>8} {info['first_frame']:>6} "
            f"{info['last_frame']:>6} ({fp[0]:.3f},{fp[1]:.3f}) "
            f"({lp[0]:.3f},{lp[1]:.3f})  {eligible}"
        )

    # Check which tracks have histogram data
    # We can't easily get the color store from JSON, but we can check
    # spatial-temporal distances between consecutive fragments
    print(f"\n{'='*70}")
    print("PAIRWISE SPATIAL-TEMPORAL DISTANCES (sequential fragments)")
    print(f"{'='*70}")

    sorted_tracks = sorted(tracks.items(), key=lambda x: x[1]["first_frame"])
    for i in range(len(sorted_tracks)):
        tid_i, info_i = sorted_tracks[i]
        if info_i["count"] < DEFAULT_MIN_TRACK_FRAMES:
            continue
        for j in range(i + 1, len(sorted_tracks)):
            tid_j, info_j = sorted_tracks[j]
            if info_j["count"] < DEFAULT_MIN_TRACK_FRAMES:
                continue

            # Temporal gap
            if info_i["last_frame"] < info_j["first_frame"]:
                gap = info_j["first_frame"] - info_i["last_frame"]
                end_pos = info_i["last_pos"]
                start_pos = info_j["first_pos"]
            elif info_j["last_frame"] < info_i["first_frame"]:
                gap = info_i["first_frame"] - info_j["last_frame"]
                end_pos = info_j["last_pos"]
                start_pos = info_i["first_pos"]
            else:
                # Overlapping
                continue

            if gap > DEFAULT_MAX_TEMPORAL_GAP:
                continue

            dx = end_pos[0] - start_pos[0]
            dy = end_pos[1] - start_pos[1]
            spatial = (dx*dx + dy*dy) ** 0.5

            if spatial > DEFAULT_MAX_SPATIAL_DISPLACEMENT:
                continue

            print(
                f"  T{tid_i} -> T{tid_j}: gap={gap:>3}f, "
                f"spatial={spatial:.4f}, "
                f"end=({end_pos[0]:.3f},{end_pos[1]:.3f}) "
                f"start=({start_pos[0]:.3f},{start_pos[1]:.3f})"
            )


if __name__ == "__main__":
    main()
