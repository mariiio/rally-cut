#!/usr/bin/env python3
"""Dump MatchSolver internals for a specific rally.

Loads a video, runs Pass-1 extraction (process_rally), then invokes
MatchSolver and prints the appearance / position / total cost matrices
plus the solver's final assignment for the requested rally index.

Used to debug specific kill-gate failures (e.g., wawa rally 10 P1↔P3).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--video-id", required=True)
    ap.add_argument("--rally-index", type=int, required=True)
    args = ap.parse_args()

    from rallycut.evaluation.db import get_connection
    from rallycut.evaluation.tracking.db import (
        get_video_path,
        load_rallies_for_video,
    )
    from rallycut.tracking.match_solver import (
        MatchSolver,
        _high_confidence_sides_for_team_pair,
    )
    from rallycut.tracking.match_tracker import (
        MatchPlayerTracker,
        extract_rally_appearances,
    )

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM videos WHERE id::text LIKE %s",
            [f"{args.video_id}%"],
        )
        rows = cur.fetchall()
    if not rows:
        raise SystemExit(f"No video matching {args.video_id!r}")
    full_id = str(rows[0][0])

    rallies = load_rallies_for_video(full_id)
    video_path = get_video_path(full_id)
    if not rallies or not video_path:
        raise SystemExit("missing rallies or video file")

    print(f"video {full_id}  rallies={len(rallies)}")

    tracker = MatchPlayerTracker()
    for i, rally in enumerate(rallies):
        track_stats = extract_rally_appearances(
            video_path=video_path,
            positions=rally.positions,
            primary_track_ids=rally.primary_track_ids,
            start_ms=rally.start_ms,
            end_ms=rally.end_ms,
            num_samples=12,
        )
        tracker.process_rally(
            track_stats=track_stats,
            player_positions=rally.positions,
            ball_positions=rally.ball_positions,
            court_split_y=rally.court_split_y,
            team_assignments=rally.team_assignments,
            start_ms=rally.start_ms,
            end_ms=rally.end_ms,
        )

    solver = MatchSolver()
    solved = solver.solve(tracker.stored_rally_data)

    idx = args.rally_index
    if not 0 <= idx < len(tracker.stored_rally_data):
        raise SystemExit(f"rally index {idx} out of range")

    stored = tracker.stored_rally_data[idx]
    top = list(stored.top_tracks)
    print(f"\n=== rally {idx + 1} (idx={idx}) top_tracks={top} ===")
    print(f"track_court_sides = {stored.track_court_sides}")
    print(f"sides_by_bbox     = {stored.sides_by_bbox}")
    print(f"early_positions   = {stored.early_positions}")
    print(f"late_positions    = {stored.late_positions}")
    print(f"player_side_assignment (snapshot) = {stored.player_side_assignment}")

    # Recompute cluster members from the final solver output (so we can
    # show what the cost matrix would look like at convergence).
    members_by_cluster: dict[int, list[tuple[int, int]]] = {
        c: [] for c in range(1, 5)
    }
    for ri, a in enumerate(solved):
        for tid, cid in a.items():
            members_by_cluster.setdefault(cid, []).append((ri, tid))

    cluster_ids = sorted(members_by_cluster.keys())
    n_clusters = len(cluster_ids)

    appearance = solver._build_appearance_cost(  # noqa: SLF001
        top=top,
        stored=stored,
        cluster_ids=cluster_ids,
        members_by_cluster=members_by_cluster,
        all_rallies=tracker.stored_rally_data,
        rally_idx=idx,
    )
    position = solver._build_position_cost(  # noqa: SLF001
        top=top,
        stored=stored,
        cluster_ids=cluster_ids,
        members_by_cluster=members_by_cluster,
        all_rallies=tracker.stored_rally_data,
        rally_idx=idx,
    )

    POSITION_WEIGHT = 0.30  # noqa: N806 — local mirror
    total = appearance * (1 - POSITION_WEIGHT) + position * POSITION_WEIGHT

    def _print_matrix(name: str, mat: np.ndarray) -> None:
        print(f"\n  {name} cost (rows=top_tracks, cols=clusters {cluster_ids}):")
        header = "  " + " " * 6 + "  ".join(f"  c{c}  " for c in cluster_ids)
        print(header)
        for ti, tid in enumerate(top):
            cells = "  ".join(f"{mat[ti, ci]:6.3f}" for ci in range(n_clusters))
            print(f"    t{tid:>2}  {cells}")

    _print_matrix("APPEARANCE", appearance)
    _print_matrix("POSITION  ", position)
    _print_matrix("TOTAL     ", total)

    print(f"\n  high_confidence_sides_for_team_pair = "
          f"{_high_confidence_sides_for_team_pair(stored.track_court_sides, stored.sides_by_bbox)}")

    print(f"\n  solver assignment for rally {idx}: {solved[idx]}")

    # Cluster late positions in the rally immediately preceding `idx`
    # (and any prior rally) — useful for explaining why position
    # continuity scored a particular way.
    print("\n  cluster latest-prior-late positions used by position cost:")
    for cid in cluster_ids:
        members = members_by_cluster.get(cid, [])
        best_idx = -1
        best_pos = None
        for (mr, mt) in members:
            if mr >= idx:
                continue
            late = tracker.stored_rally_data[mr].late_positions.get(mt)
            if late is None:
                continue
            if mr > best_idx:
                best_idx = mr
                best_pos = late
        print(f"    cluster {cid}: prev_rally={best_idx} late_pos={best_pos}")


if __name__ == "__main__":
    main()
