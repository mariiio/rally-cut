"""Debug bbox clustering for rally 87ce7bff to understand court_split_y=0.367."""

import sys
sys.path.insert(0, ".")

import numpy as np
from rallycut.evaluation.tracking.db import load_labeled_rallies

RALLY_ID = "87ce7bff-2dd3-434e-829c-365e0c53cfcb"


def main():
    print(f"Loading stored tracking for rally {RALLY_ID[:8]}...")
    rallies = load_labeled_rallies(rally_id=RALLY_ID)
    if not rallies:
        print("Rally not found!")
        return
    rally = rallies[0]
    if rally.predictions is None:
        print("No predictions!")
        return
    positions = rally.predictions.positions
    ball_positions = rally.predictions.ball_positions or []

    print(f"Total positions: {len(positions)}")
    print(f"Ball positions: {len(ball_positions)}")

    # Reproduce _find_net_from_bbox_clustering
    track_sizes: dict[int, list[float]] = {}
    track_y_values: dict[int, list[float]] = {}
    track_frames: dict[int, list[int]] = {}

    for p in positions:
        if p.track_id < 0:
            continue
        if p.track_id not in track_sizes:
            track_sizes[p.track_id] = []
            track_y_values[p.track_id] = []
            track_frames[p.track_id] = []
        track_sizes[p.track_id].append(p.width * p.height)
        track_y_values[p.track_id].append(p.y)
        track_frames[p.track_id].append(p.frame_number)

    print(f"\nUnique tracks: {len(track_sizes)}")

    # Compute averages
    track_avg_size = {tid: float(np.mean(sizes)) for tid, sizes in track_sizes.items()}
    track_avg_y = {tid: float(np.mean(ys)) for tid, ys in track_y_values.items()}
    track_presence = {tid: len(frames) for tid, frames in track_frames.items()}

    total_frames = max(f for frames in track_frames.values() for f in frames) + 1

    # Sort by size (descending) - same as _find_net_from_bbox_clustering
    sorted_tracks = sorted(track_avg_size.keys(), key=lambda t: track_avg_size[t], reverse=True)

    print(f"\nAll tracks sorted by bbox size (descending):")
    print(f"{'TID':>5} {'AvgSize':>10} {'AvgY':>8} {'Frames':>8} {'Presence':>10} {'YRange':>15}")
    for tid in sorted_tracks:
        ys = track_y_values[tid]
        y_range = f"{min(ys):.2f}-{max(ys):.2f}"
        presence = f"{track_presence[tid]/total_frames*100:.1f}%"
        print(
            f"{tid:>5} {track_avg_size[tid]:>10.5f} {track_avg_y[tid]:>8.3f} "
            f"{track_presence[tid]:>8} {presence:>10} {y_range:>15}"
        )

    # Simulate bbox clustering with top 4
    num_players = 4
    top_tracks = sorted_tracks[:num_players]
    near_tracks = top_tracks[:2]  # Largest 2
    far_tracks = top_tracks[2:]   # Smaller 2

    print(f"\n--- BBOX CLUSTERING ---")
    print(f"Top 4 by size: {top_tracks}")
    print(f"Near team (largest 2):  {near_tracks} -> ys={[f'{track_avg_y[t]:.3f}' for t in near_tracks]}")
    print(f"Far team (smallest 2):  {far_tracks} -> ys={[f'{track_avg_y[t]:.3f}' for t in far_tracks]}")

    near_ys = [track_avg_y[t] for t in near_tracks]
    far_ys = [track_avg_y[t] for t in far_tracks]

    max_far_y = max(far_ys)
    min_near_y = min(near_ys)

    print(f"\nmax(far_ys)={max_far_y:.3f}, min(near_ys)={min_near_y:.3f}")

    if max_far_y >= min_near_y:
        all_ys = near_ys + far_ys
        split_y = float(np.median(all_ys))
        print(f"OVERLAP! Teams overlap -> using median: {split_y:.3f}")
    else:
        split_y = (max_far_y + min_near_y) / 2
        print(f"Split at midpoint: {split_y:.3f}")

    # Show what select_two_teams would do with this split
    print(f"\n--- TEAM SELECTION with split_y={split_y:.3f} ---")
    # Count positions per track above/below split
    for tid in top_tracks:
        ys = track_y_values[tid]
        above = sum(1 for y in ys if y > split_y)
        below = sum(1 for y in ys if y <= split_y)
        team = "NEAR" if track_avg_y[tid] > split_y else "FAR"
        print(f"  Track {tid}: avg_y={track_avg_y[tid]:.3f} -> {team} (above:{above} below:{below})")

    # Also check ball-based methods
    if ball_positions:
        from rallycut.tracking.player_filter import (
            _find_net_from_ball_crossings,
        )
        ball_split = _find_net_from_ball_crossings(ball_positions)
        print(f"\nBall-crossing split: {ball_split}")

        confident_ys = [p.y for p in ball_positions if p.confidence >= 0.4]
        if confident_ys:
            ball_center = (max(confident_ys) + min(confident_ys)) / 2
            print(f"Ball trajectory center: {ball_center:.3f}")
            print(f"Ball Y range: {min(confident_ys):.3f} - {max(confident_ys):.3f}")


if __name__ == "__main__":
    main()
