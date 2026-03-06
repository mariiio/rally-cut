#!/usr/bin/env python3
"""Diagnose cross-rally player matching errors.

Goes beyond accuracy to show error taxonomy, appearance discriminability,
cost matrices, and assignment margins.

Usage:
    uv run python scripts/diagnose_match_players.py
    uv run python scripts/diagnose_match_players.py --video-id <id>
    uv run python scripts/diagnose_match_players.py --rerun  # re-run match-players with diagnostics
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from pathlib import Path
from typing import Any, cast

import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)


def _find_best_permutation(
    gt_rallies: dict[str, dict[str, int]],
    pred_rallies: dict[str, dict[str, int]],
) -> tuple[dict[int, int], int, int]:
    """Find optimal global permutation mapping pred player IDs to GT player IDs."""
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


def _classify_error(
    gt_pid: int, mapped_pid: int | None,
) -> str:
    """Classify an error as within-team or cross-team swap."""
    if mapped_pid is None:
        return "unmapped"
    gt_team = 0 if gt_pid <= 2 else 1
    mapped_team = 0 if mapped_pid <= 2 else 1
    if gt_team == mapped_team:
        return "within-team"
    return "cross-team"


def diagnose_from_db(video_id_filter: str | None = None) -> None:
    """Diagnose matching errors using GT and predictions from DB."""
    from rallycut.evaluation.db import get_connection

    with get_connection() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT id, match_analysis_json, player_matching_gt_json
                FROM videos WHERE player_matching_gt_json IS NOT NULL
            """
            params: list[str] = []
            if video_id_filter:
                query += " AND id LIKE %s"
                params.append(f"{video_id_filter}%")
            cur.execute(query, params)
            rows = cur.fetchall()

    if not rows:
        print("No videos with GT found")
        sys.exit(1)

    total_within = 0
    total_cross = 0
    total_correct = 0
    total_assignments = 0

    for row in rows:
        vid = str(row[0])
        gt_data = cast(dict[str, Any], row[2])
        match_data = cast(dict[str, Any], row[1]) if row[1] else {}

        gt_rallies_raw = gt_data.get("rallies", {})
        gt_rallies: dict[str, dict[str, int]] = {
            rid: {str(k): int(v) for k, v in mapping.items()}
            for rid, mapping in gt_rallies_raw.items()
        }

        # Build predictions
        pred_rallies: dict[str, dict[str, int]] = {}
        pred_rallies_list = match_data.get("rallies", [])
        for entry in pred_rallies_list:
            rid = entry.get("rallyId", entry.get("rally_id", ""))
            ttp = entry.get("trackToPlayer", entry.get("track_to_player", {}))
            if rid and ttp:
                pred_rallies[rid] = {str(k): int(v) for k, v in ttp.items()}

        best_perm, correct, total = _find_best_permutation(gt_rallies, pred_rallies)
        if total == 0:
            continue

        accuracy = correct / total * 100
        total_correct += correct
        total_assignments += total

        print(f"\n{'='*70}")
        print(f"Video: {vid[:8]}  Accuracy: {correct}/{total} = {accuracy:.1f}%")
        print(f"Best permutation: {best_perm}")
        print(f"{'='*70}")

        # Per-rally analysis with error taxonomy
        within_team = 0
        cross_team = 0

        rally_order = [e.get("rallyId", "") for e in pred_rallies_list]
        gt_rally_ids = list(gt_rallies.keys())

        for i, rid in enumerate(gt_rally_ids):
            if rid not in pred_rallies:
                print(f"  [{i}] {rid[:8]}: NOT IN PREDICTIONS")
                continue

            gt = gt_rallies[rid]
            pred = pred_rallies[rid]

            errors: list[str] = []
            r_correct = 0
            r_total = 0

            for tid_str in sorted(gt.keys()):
                if tid_str not in pred:
                    continue
                r_total += 1
                gt_pid = gt[tid_str]
                pred_pid = pred[tid_str]
                mapped = best_perm.get(pred_pid)
                if mapped == gt_pid:
                    r_correct += 1
                else:
                    err_type = _classify_error(gt_pid, mapped)
                    if err_type == "within-team":
                        within_team += 1
                    else:
                        cross_team += 1
                    errors.append(
                        f"T{tid_str}:P{mapped}→P{gt_pid}({err_type[:5]})"
                    )

            # Get confidence from predictions
            conf = 0.0
            for entry in pred_rallies_list:
                if entry.get("rallyId", "") == rid:
                    conf = entry.get("assignmentConfidence", 0)
                    break

            status = "OK" if r_correct == r_total else "ERR"
            err_str = " ".join(errors) if errors else ""
            print(
                f"  [{i:2d}] {rid[:8]} {r_correct}/{r_total} "
                f"conf={conf:.2f} {status} {err_str}"
            )

        total_errors = within_team + cross_team
        total_within += within_team
        total_cross += cross_team
        if total_errors > 0:
            print(
                f"\n  Error taxonomy: within-team={within_team} "
                f"({within_team/total_errors*100:.0f}%), "
                f"cross-team={cross_team} "
                f"({cross_team/total_errors*100:.0f}%)"
            )

    # Aggregate
    all_errors = total_within + total_cross
    agg_acc = total_correct / total_assignments * 100 if total_assignments > 0 else 0
    print(f"\n{'='*70}")
    print(f"AGGREGATE: {total_correct}/{total_assignments} = {agg_acc:.1f}% accuracy")
    if all_errors > 0:
        print(
            f"Errors: {all_errors} total — "
            f"within-team={total_within} ({total_within/all_errors*100:.0f}%), "
            f"cross-team={total_cross} ({total_cross/all_errors*100:.0f}%)"
        )


def diagnose_with_rerun(video_id_filter: str | None = None) -> None:
    """Re-run match-players with diagnostics collection and analyze."""
    from rallycut.evaluation.db import get_connection
    from rallycut.evaluation.tracking.db import get_video_path, load_rallies_for_video
    from rallycut.tracking.match_tracker import (
        RallyAssignmentDiagnostics,
        match_players_across_rallies,
    )

    with get_connection() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT id, player_matching_gt_json
                FROM videos WHERE player_matching_gt_json IS NOT NULL
            """
            params: list[str] = []
            if video_id_filter:
                query += " AND id LIKE %s"
                params.append(f"{video_id_filter}%")
            cur.execute(query, params)
            rows = cur.fetchall()

    if not rows:
        print("No videos with GT found")
        sys.exit(1)

    total_correct = 0
    total_assignments = 0

    for row in rows:
        vid = str(row[0])
        gt_data = cast(dict[str, Any], row[1])
        gt_rallies_raw = gt_data.get("rallies", {})
        gt_rallies: dict[str, dict[str, int]] = {
            rid: {str(k): int(v) for k, v in mapping.items()}
            for rid, mapping in gt_rallies_raw.items()
        }

        # Load rallies and get video path
        rallies = load_rallies_for_video(vid)
        if not rallies:
            print(f"Video {vid[:8]}: no rallies found")
            continue

        video_path = get_video_path(vid)
        if not video_path:
            print(f"Video {vid[:8]}: video file not found")
            continue

        print(f"\nRunning match-players for {vid[:8]} ({len(rallies)} rallies)...")
        result = match_players_across_rallies(
            video_path=video_path,
            rallies=rallies,
            collect_diagnostics=True,
        )

        # Build predictions
        pred_rallies: dict[str, dict[str, int]] = {}
        rally_id_order: list[str] = []
        for rally_data, rally_result in zip(rallies, result.rally_results):
            rid = rally_data.rally_id
            rally_id_order.append(rid)
            pred_rallies[rid] = {
                str(k): v for k, v in rally_result.track_to_player.items()
            }

        best_perm, correct, total = _find_best_permutation(gt_rallies, pred_rallies)
        accuracy = correct / total * 100 if total > 0 else 0
        total_correct += correct
        total_assignments += total

        print(f"\n{'='*70}")
        print(f"Video: {vid[:8]}  Accuracy: {correct}/{total} = {accuracy:.1f}%")
        print(f"Best permutation: {best_perm}")
        print(f"{'='*70}")

        # Show per-rally with cost matrices
        for i, (rally_data, rally_result) in enumerate(
            zip(rallies, result.rally_results)
        ):
            rid = rally_data.rally_id
            if rid not in gt_rallies:
                continue

            gt = gt_rallies[rid]
            pred = pred_rallies.get(rid, {})
            conf = rally_result.assignment_confidence
            switch = "↔" if rally_result.side_switch_detected else " "

            errors: list[str] = []
            r_correct = 0
            r_total = 0
            for tid_str in sorted(gt.keys()):
                if tid_str not in pred:
                    continue
                r_total += 1
                gt_pid = gt[tid_str]
                mapped = best_perm.get(pred[tid_str])
                if mapped == gt_pid:
                    r_correct += 1
                else:
                    err_type = _classify_error(gt_pid, mapped)
                    errors.append(
                        f"T{tid_str}:P{mapped}→P{gt_pid}({err_type[:5]})"
                    )

            status = "OK" if r_correct == r_total else "ERR"
            err_str = " ".join(errors) if errors else ""
            print(
                f"  [{i:2d}]{switch} {rid[:8]} {r_correct}/{r_total} "
                f"conf={conf:.2f} {status} {err_str}"
            )

            # Show cost matrix for this rally from diagnostics
            diag = _find_diagnostic(result.diagnostics, i)
            if diag is not None and r_total > 0:
                _print_cost_matrix(diag, best_perm, gt.get, indent=8)

        # Show appearance discriminability between player profiles
        print(f"\n  Appearance discriminability (Bhattacharyya distance):")
        _print_profile_discriminability(result.player_profiles, indent=4)

    if total_assignments > 0:
        agg = total_correct / total_assignments * 100
        print(f"\n{'='*70}")
        print(f"AGGREGATE: {total_correct}/{total_assignments} = {agg:.1f}%")


def _find_diagnostic(
    diagnostics: list[Any], rally_index: int,
) -> Any | None:
    """Find diagnostic for a rally index."""
    for d in diagnostics:
        if d.rally_index == rally_index:
            return d
    return None


def _print_cost_matrix(
    diag: Any,
    perm: dict[int, int],
    gt_lookup: Any,
    indent: int = 4,
) -> None:
    """Print cost matrix with margins."""
    prefix = " " * indent
    cm = diag.cost_matrix
    tids = diag.track_ids
    pids = diag.player_ids

    # Header
    header = f"{prefix}{'':>6}"
    for pid in pids:
        gt_label = f"P{perm.get(pid, pid)}"
        header += f"  {gt_label:>6}"
    header += "  margin"
    print(header)

    # Rows
    for i, tid in enumerate(tids):
        if i >= cm.shape[0]:
            break
        row = f"{prefix}T{tid:>4}:"
        assigned_pid = diag.assignment.get(tid)
        for j, pid in enumerate(pids):
            if j >= cm.shape[1]:
                break
            cost = cm[i, j]
            marker = " *" if pid == assigned_pid else "  "
            row += f" {cost:.3f}{marker}"
        # Show margin for assigned player
        if assigned_pid in diag.assignment_margins:
            row += f"  {diag.assignment_margins[assigned_pid]:.3f}"
        print(row)


def _print_profile_discriminability(
    profiles: dict[int, Any],
    indent: int = 4,
) -> None:
    """Print pairwise similarity between player profiles."""
    from rallycut.tracking.player_features import (
        PlayerAppearanceProfile,
        TrackAppearanceStats,
        compute_appearance_similarity,
    )

    prefix = " " * indent
    pids = sorted(profiles.keys())

    # Header
    header = f"{prefix}{'':>4}"
    for pid in pids:
        header += f"  P{pid:>3}"
    print(header)

    for i, pid_a in enumerate(pids):
        row = f"{prefix}P{pid_a:>2}:"
        for j, pid_b in enumerate(pids):
            if i == j:
                row += f"  {'---':>4}"
            else:
                # Create a fake TrackAppearanceStats from profile B
                profile_b = profiles[pid_b]
                stats_b = TrackAppearanceStats(track_id=pid_b)
                stats_b.avg_skin_tone_hsv = profile_b.avg_skin_tone_hsv
                stats_b.avg_upper_hist = profile_b.avg_upper_hist
                stats_b.avg_lower_hist = profile_b.avg_lower_hist
                stats_b.avg_lower_v_hist = profile_b.avg_lower_v_hist
                stats_b.avg_upper_v_hist = profile_b.avg_upper_v_hist
                stats_b.avg_dominant_color_hsv = profile_b.avg_dominant_color_hsv
                cost = compute_appearance_similarity(profiles[pid_a], stats_b)
                row += f"  {cost:.2f}"
            pass
        print(row)

    # Highlight same-team pairs
    teams = {1: "near", 2: "near", 3: "far", 4: "far"}
    for team_name, team_pids in [("near", [1, 2]), ("far", [3, 4])]:
        if all(p in profiles for p in team_pids):
            pa, pb = team_pids
            profile_b = profiles[pb]
            stats_b = TrackAppearanceStats(track_id=pb)
            stats_b.avg_skin_tone_hsv = profile_b.avg_skin_tone_hsv
            stats_b.avg_upper_hist = profile_b.avg_upper_hist
            stats_b.avg_lower_hist = profile_b.avg_lower_hist
            stats_b.avg_lower_v_hist = profile_b.avg_lower_v_hist
            stats_b.avg_upper_v_hist = profile_b.avg_upper_v_hist
            stats_b.avg_dominant_color_hsv = profile_b.avg_dominant_color_hsv
            cost = compute_appearance_similarity(profiles[pa], stats_b)
            quality = "GOOD" if cost > 0.15 else "POOR" if cost > 0.05 else "BAD"
            print(
                f"{prefix}  {team_name} team P{pa}↔P{pb}: "
                f"cost={cost:.3f} ({quality} discriminability)"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose cross-rally player matching errors"
    )
    parser.add_argument("--video-id", type=str, help="Filter to specific video")
    parser.add_argument(
        "--rerun", action="store_true",
        help="Re-run match-players with diagnostics collection",
    )
    args = parser.parse_args()

    if args.rerun:
        diagnose_with_rerun(args.video_id)
    else:
        diagnose_from_db(args.video_id)


if __name__ == "__main__":
    main()
