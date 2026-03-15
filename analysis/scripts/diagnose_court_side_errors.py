#!/usr/bin/env python3
"""Diagnose court-side misclassification in cross-rally matching.

For each cross-team error, determines:
1. Which classification source was used (team_assignments, court_split_y, median)
2. Whether team_assignments are available and correct
3. Whether court_split_y agrees with GT
4. Track Y positions vs GT team

Usage:
    cd analysis
    uv run python scripts/diagnose_court_side_errors.py
"""

from __future__ import annotations

import itertools
import logging
import sys
from collections import defaultdict
from typing import Any, cast

import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(message)s")


def _find_best_permutation(
    gt_rallies: dict[str, dict[str, int]],
    pred_rallies: dict[str, dict[str, int]],
) -> tuple[dict[int, int], int, int]:
    player_ids = [1, 2, 3, 4]
    best_perm: dict[int, int] = {}
    best_correct = -1
    best_total = 0
    for perm in itertools.permutations(player_ids):
        pred_to_gt = {pid: gpid for pid, gpid in zip(player_ids, perm)}
        correct = 0
        total = 0
        for rid in gt_rallies:
            if rid not in pred_rallies:
                continue
            gt = gt_rallies[rid]
            pred = pred_rallies[rid]
            for tid_str in gt:
                if tid_str not in pred:
                    continue
                total += 1
                if pred_to_gt.get(pred[tid_str]) == gt[tid_str]:
                    correct += 1
        if correct > best_correct:
            best_correct = correct
            best_total = total
            best_perm = pred_to_gt
    return best_perm, best_correct, best_total


def classify_side_source(
    track_ids: list[int],
    track_avg_y: dict[int, float],
    court_split_y: float | None,
    team_assignments: dict[int, int] | None,
) -> str:
    """Reproduce _classify_track_sides logic to determine which source was used."""
    if team_assignments:
        covered = [t for t in track_avg_y if t in team_assignments]
        if len(covered) >= len(track_avg_y) * 0.75:
            return "team_assignments"

    if court_split_y is not None:
        near = [t for t in track_avg_y if track_avg_y[t] > court_split_y]
        far = [t for t in track_avg_y if track_avg_y[t] <= court_split_y]
        if near and far:
            return "court_split_y"

        # Fallback to team_assignments with any coverage
        if team_assignments:
            covered = [t for t in track_avg_y if t in team_assignments]
            if covered:
                sides = set()
                for t in track_avg_y:
                    if t in team_assignments:
                        sides.add(team_assignments[t])
                if len(sides) >= 2:
                    return "team_assignments_fallback"

    return "median_y"


def main() -> None:
    from rallycut.evaluation.db import get_connection
    from rallycut.evaluation.tracking.db import get_video_path, load_rallies_for_video
    from rallycut.tracking.match_tracker import (
        MatchPlayerTracker,
        extract_rally_appearances,
    )

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, player_matching_gt_json
                FROM videos WHERE player_matching_gt_json IS NOT NULL
                ORDER BY name
            """)
            rows = cur.fetchall()

    if not rows:
        print("No videos with GT found")
        sys.exit(1)

    # Counters
    source_counts: dict[str, int] = defaultdict(int)  # which source classified the error
    source_error_counts: dict[str, int] = defaultdict(int)  # errors per source
    team_assign_available_correct = 0
    team_assign_available_wrong = 0
    team_assign_unavailable = 0
    court_split_y_correct = 0
    court_split_y_wrong = 0
    court_split_y_unavailable = 0

    # Per-video details for worst cases
    video_details: list[tuple[str, int, list[str]]] = []

    total_cross_errors = 0

    for row_idx, row in enumerate(rows):
        vid = str(row[0])
        gt_data = cast(dict[str, Any], row[1])
        gt_rallies_raw = gt_data.get("rallies", {})
        gt_rallies: dict[str, dict[str, int]] = {
            rid: {str(k): int(v) for k, v in mapping.items()}
            for rid, mapping in gt_rallies_raw.items()
        }

        rallies = load_rallies_for_video(vid)
        if not rallies:
            continue

        video_path = get_video_path(vid)
        if not video_path:
            continue

        print(f"[{row_idx + 1}/{len(rows)}] {vid[:8]}...", end="", flush=True)

        # Run match-players
        tracker = MatchPlayerTracker(collect_diagnostics=True)
        initial_results = []

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
            initial_results.append(result)

        refined_results = tracker.refine_assignments(initial_results)

        # Build predictions
        pred_rallies: dict[str, dict[str, int]] = {}
        for rally, rr in zip(rallies, refined_results):
            pred_rallies[rally.rally_id] = {
                str(k): v for k, v in rr.track_to_player.items()
            }

        best_perm, correct, total = _find_best_permutation(gt_rallies, pred_rallies)

        video_errors: list[str] = []
        video_cross_count = 0

        for i, (rally, rr) in enumerate(zip(rallies, refined_results)):
            rid = rally.rally_id
            if rid not in gt_rallies:
                continue

            gt = gt_rallies[rid]
            pred = pred_rallies.get(rid, {})
            data = tracker.stored_rally_data[i] if i < len(tracker.stored_rally_data) else None

            # Compute track avg Y for this rally
            track_y_values: dict[int, list[float]] = defaultdict(list)
            for p in rally.positions:
                if p.track_id >= 0:
                    track_y_values[p.track_id].append(p.y)
            track_avg_y = {tid: float(np.mean(ys)) for tid, ys in track_y_values.items()}

            # Determine which source was used
            source = classify_side_source(
                list(track_avg_y.keys()),
                track_avg_y,
                rally.court_split_y,
                rally.team_assignments,
            )

            for tid_str in gt:
                if tid_str not in pred:
                    continue
                gt_pid = gt[tid_str]
                raw_pred_pid = pred[tid_str]
                mapped_pid = best_perm.get(raw_pred_pid)

                if mapped_pid == gt_pid:
                    continue

                gt_team = 0 if gt_pid <= 2 else 1
                mapped_team = 0 if (mapped_pid or 0) <= 2 else 1
                if gt_team == mapped_team:
                    continue  # within-team, skip

                total_cross_errors += 1
                video_cross_count += 1
                tid = int(tid_str)

                # What side was this track classified as?
                classified_side = data.track_court_sides.get(tid) if data else None

                source_counts[source] += 1
                if classified_side is not None and classified_side != gt_team:
                    source_error_counts[source] += 1

                # Check if team_assignments would have been correct
                if rally.team_assignments and tid in rally.team_assignments:
                    ta_side = rally.team_assignments[tid]
                    if ta_side == gt_team:
                        team_assign_available_correct += 1
                    else:
                        team_assign_available_wrong += 1
                else:
                    team_assign_unavailable += 1

                # Check if court_split_y would have been correct
                if rally.court_split_y is not None and tid in track_avg_y:
                    csy_side = 0 if track_avg_y[tid] > rally.court_split_y else 1
                    if csy_side == gt_team:
                        court_split_y_correct += 1
                    else:
                        court_split_y_wrong += 1
                else:
                    court_split_y_unavailable += 1

                # Detail string
                avg_y = track_avg_y.get(tid, 0)
                ta_val = rally.team_assignments.get(tid, "N/A") if rally.team_assignments else "N/A"
                csy_val = f"{rally.court_split_y:.3f}" if rally.court_split_y is not None else "N/A"
                classified = classified_side if classified_side is not None else "?"
                video_errors.append(
                    f"  R{i} T{tid}: GT_team={gt_team} classified={classified} "
                    f"src={source} avgY={avg_y:.3f} csy={csy_val} "
                    f"ta={ta_val} | P{mapped_pid}→P{gt_pid}"
                )

        print(f" {video_cross_count} cross-team errors")
        if video_cross_count > 0:
            video_details.append((vid, video_cross_count, video_errors))

    # Summary
    print(f"\n{'=' * 70}")
    print("COURT-SIDE MISCLASSIFICATION ROOT CAUSE")
    print(f"{'=' * 70}")
    print(f"\nTotal cross-team errors: {total_cross_errors}")

    print("\n--- Classification source used for error tracks ---")
    for source, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        misclass = source_error_counts.get(source, 0)
        print(
            f"  {source:<30} {count:>4} errors "
            f"({misclass} with wrong side = {misclass / count * 100:.0f}%)"
        )

    print("\n--- Would team_assignments have been correct? ---")
    ta_total = team_assign_available_correct + team_assign_available_wrong
    print(f"  Available & correct: {team_assign_available_correct}")
    print(f"  Available & wrong:   {team_assign_available_wrong}")
    print(f"  Unavailable:         {team_assign_unavailable}")
    if ta_total > 0:
        print(
            f"  Accuracy when available: "
            f"{team_assign_available_correct / ta_total * 100:.1f}%"
        )

    print("\n--- Would court_split_y have been correct? ---")
    csy_total = court_split_y_correct + court_split_y_wrong
    print(f"  Correct: {court_split_y_correct}")
    print(f"  Wrong:   {court_split_y_wrong}")
    print(f"  N/A:     {court_split_y_unavailable}")
    if csy_total > 0:
        print(
            f"  Accuracy when available: "
            f"{court_split_y_correct / csy_total * 100:.1f}%"
        )

    # Show worst videos
    print("\n--- Per-video cross-team error details ---")
    video_details.sort(key=lambda x: -x[1])
    for vid, count, errors in video_details[:8]:
        print(f"\n  {vid[:8]}: {count} cross-team errors")
        for e in errors:
            print(f"    {e}")


if __name__ == "__main__":
    main()
