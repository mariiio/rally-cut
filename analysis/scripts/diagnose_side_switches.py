#!/usr/bin/env python3
"""Diagnose side switch detection signal using profile-free pairwise comparison.

For each pair of rallies, compares team-level track appearances directly
(no accumulated profiles) to determine if they have the same or opposite
team orientation. A side switch should show as a clear partition: pre-switch
rallies agree with each other, post-switch rallies agree with each other,
but cross-boundary pairs disagree.

Usage:
    cd analysis
    uv run python scripts/diagnose_side_switches.py
    uv run python scripts/diagnose_side_switches.py --video-id 1efa35cf
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose side switch detection signal"
    )
    parser.add_argument("--video-id", type=str, default=None)
    args = parser.parse_args()

    from rallycut.evaluation.db import get_connection
    from rallycut.evaluation.tracking.db import (
        get_video_path,
        load_rallies_for_video,
    )
    from rallycut.tracking.match_tracker import (
        MatchPlayerTracker,
        extract_rally_appearances,
    )
    from rallycut.tracking.player_features import compute_track_similarity

    # Load GT entries
    gt_entries: list[tuple[str, list[int]]] = []
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, player_matching_gt_json FROM videos "
                "WHERE player_matching_gt_json IS NOT NULL ORDER BY name"
            )
            for row in cur.fetchall():
                vid, gt_json = row[0], row[1]
                if not gt_json:
                    continue
                if args.video_id and not vid.startswith(args.video_id):
                    continue
                switches = gt_json.get(
                    "sideSwitches", gt_json.get("side_switches", [])
                )
                gt_entries.append((vid, switches))

    if not gt_entries:
        print("No videos found")
        return

    for video_id, gt_switches in gt_entries:
        rallies = load_rallies_for_video(video_id)
        video_path = get_video_path(video_id)
        if not rallies or not video_path:
            continue

        n_sw = len(gt_switches)
        label = f"({n_sw} switch{'es' if n_sw != 1 else ''})" if n_sw else "(no switches)"
        print(f"\n{'='*70}")
        print(f"Video: {video_id[:8]}  {label}  ({len(rallies)} rallies)")
        if gt_switches:
            print(f"GT switch indices: {gt_switches}")
        print(f"{'='*70}")

        # Run match-players to get track stats and team sides
        tracker = MatchPlayerTracker()
        all_results = []

        for rally in rallies:
            track_stats = extract_rally_appearances(
                video_path=video_path,
                positions=rally.positions,
                primary_track_ids=rally.primary_track_ids,
                start_ms=rally.start_ms,
                end_ms=rally.end_ms,
                num_samples=12,
            )
            result = tracker.process_rally(
                track_stats=track_stats,
                player_positions=rally.positions,
                ball_positions=rally.ball_positions,
                court_split_y=rally.court_split_y,
                team_assignments=rally.team_assignments,
            )
            all_results.append(result)

        # Collect per-rally team track stats
        # For each rally: team 0 tracks (near), team 1 tracks (far)
        rally_teams: list[
            tuple[list[int], list[int], dict]
        ] = []  # (team0_tids, team1_tids, track_stats)

        for i, data in enumerate(tracker.stored_rally_data):
            t0 = [
                tid for tid in data.top_tracks
                if data.track_court_sides.get(tid) == 0
                and tid in data.track_stats
            ]
            t1 = [
                tid for tid in data.top_tracks
                if data.track_court_sides.get(tid) == 1
                and tid in data.track_stats
            ]
            rally_teams.append((t0, t1, data.track_stats))

        # Compute pairwise "same orientation" preference
        # For rallies i and j:
        #   same_cost: best matching of i's team0 ↔ j's team0 AND i's team1 ↔ j's team1
        #   cross_cost: best matching of i's team0 ↔ j's team1 AND i's team1 ↔ j's team0
        #   preference > 0 means same orientation preferred
        n = len(rallies)
        preference = np.zeros((n, n))

        def team_match_cost(
            tids_a: list[int], stats_a: dict,
            tids_b: list[int], stats_b: dict,
        ) -> float:
            """Best-matching cost between two sets of tracks."""
            if not tids_a or not tids_b:
                return 1.0
            # Try all pairings and pick the cheapest
            if len(tids_a) == 1 and len(tids_b) == 1:
                return compute_track_similarity(
                    stats_a[tids_a[0]], stats_b[tids_b[0]]
                )
            if len(tids_a) >= 2 and len(tids_b) >= 2:
                # 2v2: try both pairings
                cost_ab = (
                    compute_track_similarity(stats_a[tids_a[0]], stats_b[tids_b[0]])
                    + compute_track_similarity(stats_a[tids_a[1]], stats_b[tids_b[1]])
                )
                cost_ba = (
                    compute_track_similarity(stats_a[tids_a[0]], stats_b[tids_b[1]])
                    + compute_track_similarity(stats_a[tids_a[1]], stats_b[tids_b[0]])
                )
                return min(cost_ab, cost_ba) / 2.0
            # Uneven: match first tracks
            return compute_track_similarity(
                stats_a[tids_a[0]], stats_b[tids_b[0]]
            )

        for i in range(n):
            t0_i, t1_i, stats_i = rally_teams[i]
            if not t0_i or not t1_i:
                continue
            for j in range(i + 1, n):
                t0_j, t1_j, stats_j = rally_teams[j]
                if not t0_j or not t1_j:
                    continue

                # Same orientation: team0↔team0, team1↔team1
                same_cost = (
                    team_match_cost(t0_i, stats_i, t0_j, stats_j)
                    + team_match_cost(t1_i, stats_i, t1_j, stats_j)
                )
                # Cross orientation: team0↔team1, team1↔team0
                cross_cost = (
                    team_match_cost(t0_i, stats_i, t1_j, stats_j)
                    + team_match_cost(t1_i, stats_i, t0_j, stats_j)
                )

                # Positive = same orientation preferred
                pref = cross_cost - same_cost
                preference[i, j] = pref
                preference[j, i] = pref

        # Iterative binary labeling (same as within-team voting)
        # label=0: same orientation as rally 0
        # label=1: opposite orientation (side switch happened)
        labels = np.zeros(n, dtype=int)
        for _iteration in range(10):
            changed = False
            for k in range(1, n):
                score = 0.0
                for j in range(n):
                    if j == k:
                        continue
                    p = preference[k, j]
                    if labels[j] == 1:
                        p = -p
                    score += p
                new_label = 0 if score >= 0 else 1
                if new_label != labels[k]:
                    labels[k] = new_label
                    changed = True
            if not changed:
                break

        # Display results
        print(f"\n{'Rally':<4} {'RallyID':<10} "
              f"{'AvgSame':>8} {'AvgCross':>9} {'Label':>6} {'GT':>6}")
        print("-" * 50)

        for i in range(n):
            rid = rallies[i].rally_id[:8]
            # Average same-orientation and cross-orientation preferences
            same_prefs = []
            cross_prefs = []
            for j in range(n):
                if j == i:
                    continue
                if labels[j] == labels[i]:
                    same_prefs.append(preference[i, j])
                else:
                    cross_prefs.append(preference[i, j])

            avg_same = float(np.mean(same_prefs)) if same_prefs else 0.0
            avg_cross = float(np.mean(cross_prefs)) if cross_prefs else 0.0

            gt_mark = ""
            if i in gt_switches:
                gt_mark = "<<SW"

            lbl = "SWAP" if labels[i] == 1 else "same"
            print(f"{i:<4} {rid:<10} "
                  f"{avg_same:>+8.3f} {avg_cross:>+9.3f} {lbl:>6} {gt_mark:>6}")

        # Detect switch points (where label changes)
        detected_switches = []
        for i in range(1, n):
            if labels[i] != labels[i - 1]:
                detected_switches.append(i)

        print(f"\n--- Switch detection ---")
        print(f"Detected switch points: {detected_switches}")
        if gt_switches:
            print(f"GT switch points:      {gt_switches}")
            # Match detected to GT
            for gs in gt_switches:
                closest = min(detected_switches, key=lambda d: abs(d - gs)) if detected_switches else -1
                dist = abs(closest - gs) if closest >= 0 else -1
                status = "HIT" if dist <= 1 else ("NEAR" if dist <= 2 else "MISS")
                print(f"  GT switch at {gs}: closest detected at {closest} "
                      f"(dist={dist}) → {status}")

        # Show preference matrix summary
        if n <= 20:
            print(f"\nPreference matrix (positive = same orientation):")
            print("     ", end="")
            for j in range(n):
                print(f" {j:>5}", end="")
            print()
            for i in range(n):
                sw = "*" if i in gt_switches else " "
                print(f"{sw}{i:>3}: ", end="")
                for j in range(n):
                    if i == j:
                        print("    - ", end="")
                    else:
                        p = preference[i, j]
                        print(f"{p:>+6.2f}", end="")
                print()

        # === Ball trajectory serve direction ===
        print(f"\n{'─'*50}")
        print("Ball trajectory serve direction (first N frames):")

        serve_dirs: list[tuple[int, str, float]] = []
        for i, rally in enumerate(rallies):
            ball = rally.ball_positions
            if not ball:
                serve_dirs.append((i, "?", 0.0))
                continue

            # Get early ball positions with actual detections
            valid_ball = sorted(
                [b for b in ball if b.confidence >= 0.3
                 and not (b.x == 0.0 and b.y == 0.0)],
                key=lambda b: b.frame_number,
            )
            if len(valid_ball) < 5:
                serve_dirs.append((i, "?", 0.0))
                continue

            # Use first 30 frames of ball data
            early_ball = [b for b in valid_ball if b.frame_number <= valid_ball[0].frame_number + 45]
            if len(early_ball) < 5:
                serve_dirs.append((i, "?", 0.0))
                continue

            # Compute net Y displacement (linear regression for robustness)
            frames = np.array([b.frame_number for b in early_ball], dtype=float)
            ys = np.array([b.y for b in early_ball])
            # Simple: first half vs second half mean
            mid = len(early_ball) // 2
            y_start = float(np.mean(ys[:mid]))
            y_end = float(np.mean(ys[mid:]))
            dy = y_end - y_start

            # Positive dy = ball moving down = from far to near = far team served
            # Negative dy = ball moving up = from near to far = near team served
            if abs(dy) < 0.01:
                serve_dirs.append((i, "?", dy))
            elif dy < 0:
                serve_dirs.append((i, "near", dy))
            else:
                serve_dirs.append((i, "far", dy))

        # Display and detect alternation breaks
        print(f"\n{'Rally':<6} {'Side':<6} {'dY':>8} {'Pattern':>10} {'GT':>6}")
        print("-" * 50)

        serve_switch_points: list[int] = []
        prev_side = None
        consecutive_same = 0
        for i, side, dy in serve_dirs:
            gt_mark = "<<SW" if i in gt_switches else ""
            pattern = ""
            if side != "?" and prev_side is not None and prev_side != "?":
                if side == prev_side:
                    consecutive_same += 1
                    pattern = f"SAME x{consecutive_same}"
                    if consecutive_same == 1:
                        serve_switch_points.append(i)
                else:
                    consecutive_same = 0
                    pattern = "alt"
            else:
                consecutive_same = 0

            print(f"{i:<6} {side:<6} {dy:>+8.4f} {pattern:>10} {gt_mark:>6}")
            if side != "?":
                prev_side = side

        # Also detect runs of ≥2 consecutive same-side
        run_switch_points: list[int] = []
        prev_side2 = None
        run_len = 0
        for i, side, dy in serve_dirs:
            if side != "?" and prev_side2 is not None and prev_side2 != "?":
                if side == prev_side2:
                    run_len += 1
                    if run_len == 2:  # Second consecutive = likely switch
                        run_switch_points.append(i - 1)
                else:
                    run_len = 0
            else:
                run_len = 0
            if side != "?":
                prev_side2 = side

        print(f"\nSwitch candidates (≥1 same): {serve_switch_points}")
        print(f"Switch candidates (≥2 same): {run_switch_points}")
        if gt_switches:
            print(f"GT switch points:            {gt_switches}")
            for label, cands in [
                ("≥1 same", serve_switch_points),
                ("≥2 same", run_switch_points),
            ]:
                for gs in gt_switches:
                    closest = min(
                        cands, key=lambda d: abs(d - gs)
                    ) if cands else -1
                    dist = abs(closest - gs) if closest >= 0 else -1
                    status = "HIT" if dist <= 1 else (
                        "NEAR" if dist <= 2 else "MISS"
                    )
                    print(f"  [{label}] GT@{gs}: closest={closest} "
                          f"(dist={dist}) → {status}")

    print()


if __name__ == "__main__":
    main()
