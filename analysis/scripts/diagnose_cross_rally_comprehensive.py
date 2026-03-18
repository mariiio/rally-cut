#!/usr/bin/env python3
"""Comprehensive cross-rally player matching diagnosis.

Re-runs match-players on all GT videos with diagnostics collection,
then produces a structured report with 8 analysis sections:
  A: Confidence vs Accuracy
  B: Side Switch Cascade Impact
  C: Per-Video Deep Dive (worst 3)
  D: Profile Quality Over Time
  E: Feature Contribution Analysis
  F: Optimization Summary
  G: GT Quality Audit
  H: Delta Comparison (vs previous baseline)

Usage:
    cd analysis
    uv run python scripts/diagnose_cross_rally_comprehensive.py
    uv run python scripts/diagnose_cross_rally_comprehensive.py --video-id a5866029
    uv run python scripts/diagnose_cross_rally_comprehensive.py --sections a,b,c
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from dataclasses import dataclass
from typing import Any, cast

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class AssignmentRecord:
    """One track-to-player assignment with metadata."""

    video_id: str
    rally_id: str
    rally_index: int
    track_id: int
    gt_player_id: int
    pred_player_id: int  # after global permutation mapping
    raw_pred_player_id: int  # before mapping
    correct: bool
    confidence: float
    margin: float  # assignment margin (2nd best - best cost)
    error_type: str  # "correct", "within-team", "cross-team"
    side_switch_detected: bool


@dataclass
class VideoResult:
    """Aggregated result for one video."""

    video_id: str
    n_rallies: int
    correct: int
    total: int
    accuracy: float
    best_perm: dict[int, int]
    assignments: list[AssignmentRecord]
    # From tracker
    stored_rally_data: list[Any]
    diagnostics: list[Any]
    player_profiles: dict[int, Any]
    rally_results: list[Any]
    gt_rallies: dict[str, dict[str, int]]
    pred_rallies: dict[str, dict[str, int]]
    rally_id_order: list[str]
    gt_switches: list[int]
    # Raw GT metadata for quality audit
    all_pred_track_ids: dict[str, set[int]]  # rally_id -> set of predicted track IDs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


def _classify_error(gt_pid: int, mapped_pid: int | None) -> str:
    if mapped_pid is None:
        return "unmapped"
    if mapped_pid == gt_pid:
        return "correct"
    gt_team = 0 if gt_pid <= 2 else 1
    mapped_team = 0 if mapped_pid <= 2 else 1
    if gt_team == mapped_team:
        return "within-team"
    return "cross-team"


# ---------------------------------------------------------------------------
# Data collection: re-run match-players on all GT videos
# ---------------------------------------------------------------------------

def collect_data(video_id_filter: str | None = None) -> list[VideoResult]:
    """Re-run match-players on all GT videos, collecting full diagnostics."""
    from rallycut.evaluation.db import get_connection
    from rallycut.evaluation.tracking.db import get_video_path, load_rallies_for_video
    from rallycut.tracking.match_tracker import (
        MatchPlayerTracker,
        extract_rally_appearances,
    )

    # Load GT videos
    with get_connection() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT id, player_matching_gt_json
                FROM videos WHERE player_matching_gt_json IS NOT NULL
                ORDER BY name
            """
            params: list[str] = []
            if video_id_filter:
                query = """
                    SELECT id, player_matching_gt_json
                    FROM videos WHERE player_matching_gt_json IS NOT NULL
                    AND id LIKE %s ORDER BY name
                """
                params.append(f"{video_id_filter}%")
            cur.execute(query, params)
            rows = cur.fetchall()

    if not rows:
        print("No videos with GT found")
        sys.exit(1)

    results: list[VideoResult] = []

    for row_idx, row in enumerate(rows):
        vid = str(row[0])
        gt_data = cast(dict[str, Any], row[1])
        gt_rallies_raw = gt_data.get("rallies", {})
        gt_rallies: dict[str, dict[str, int]] = {
            rid: {str(k): int(v) for k, v in mapping.items()}
            for rid, mapping in gt_rallies_raw.items()
        }
        gt_switches = gt_data.get(
            "sideSwitches", gt_data.get("side_switches", [])
        )

        rallies = load_rallies_for_video(vid)
        if not rallies:
            print(f"  {vid[:8]}: no rallies, skipping")
            continue

        video_path = get_video_path(vid)
        if not video_path:
            print(f"  {vid[:8]}: video not found, skipping")
            continue

        print(f"[{row_idx + 1}/{len(rows)}] {vid[:8]}: {len(rallies)} rallies...", end="", flush=True)

        # Run match-players with diagnostics
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

        # Pass 2: refine assignments
        refined_results = tracker.refine_assignments(initial_results)

        # Build predictions
        pred_rallies: dict[str, dict[str, int]] = {}
        rally_id_order: list[str] = []
        all_pred_track_ids: dict[str, set[int]] = {}
        for rally, rr in zip(rallies, refined_results):
            rid = rally.rally_id
            rally_id_order.append(rid)
            pred_rallies[rid] = {str(k): v for k, v in rr.track_to_player.items()}
            # Collect all track IDs present in predictions for this rally
            all_pred_track_ids[rid] = set(rr.track_to_player.keys())

        best_perm, correct, total = _find_best_permutation(gt_rallies, pred_rallies)
        accuracy = correct / total * 100 if total > 0 else 0

        # Build per-assignment records
        assignments: list[AssignmentRecord] = []
        for i, (rally, rr) in enumerate(zip(rallies, refined_results)):
            rid = rally.rally_id
            if rid not in gt_rallies:
                continue
            gt = gt_rallies[rid]
            pred = pred_rallies.get(rid, {})

            # Find diagnostic for this rally
            diag = None
            for d in tracker.diagnostics:
                if d.rally_index == i:
                    diag = d
                    break

            for tid_str in gt:
                if tid_str not in pred:
                    continue
                gt_pid = gt[tid_str]
                raw_pred_pid = pred[tid_str]
                mapped_pid = best_perm.get(raw_pred_pid)
                is_correct = mapped_pid == gt_pid
                err_type = _classify_error(gt_pid, mapped_pid)

                # Get margin from diagnostics
                margin = 0.0
                if diag is not None and raw_pred_pid in diag.assignment_margins:
                    margin = diag.assignment_margins[raw_pred_pid]

                assignments.append(AssignmentRecord(
                    video_id=vid,
                    rally_id=rid,
                    rally_index=i,
                    track_id=int(tid_str),
                    gt_player_id=gt_pid,
                    pred_player_id=mapped_pid if mapped_pid is not None else -1,
                    raw_pred_player_id=raw_pred_pid,
                    correct=is_correct,
                    confidence=rr.assignment_confidence,
                    margin=margin,
                    error_type=err_type,
                    side_switch_detected=rr.side_switch_detected,
                ))

        print(f" {accuracy:.1f}%")

        results.append(VideoResult(
            video_id=vid,
            n_rallies=len(rallies),
            correct=correct,
            total=total,
            accuracy=accuracy,
            best_perm=best_perm,
            assignments=assignments,
            stored_rally_data=tracker.stored_rally_data,
            diagnostics=tracker.diagnostics,
            player_profiles=dict(tracker.state.players),
            rally_results=refined_results,
            gt_rallies=gt_rallies,
            pred_rallies=pred_rallies,
            rally_id_order=rally_id_order,
            gt_switches=gt_switches,
            all_pred_track_ids=all_pred_track_ids,
        ))

    return results


# ---------------------------------------------------------------------------
# Section A: Confidence vs Accuracy
# ---------------------------------------------------------------------------

def section_a(all_results: list[VideoResult]) -> None:
    print("\n" + "=" * 70)
    print("SECTION A: Confidence vs Accuracy")
    print("=" * 70)

    all_assignments: list[AssignmentRecord] = []
    for vr in all_results:
        all_assignments.extend(vr.assignments)

    if not all_assignments:
        print("  No assignments found")
        return

    total_correct = sum(1 for a in all_assignments if a.correct)
    total = len(all_assignments)
    print(f"\nBaseline: {total_correct}/{total} = {total_correct / total * 100:.1f}%\n")

    # Confidence buckets
    buckets = [
        (0.0, 0.5), (0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01),
    ]

    print(f"{'Confidence':<14} {'Count':>6} {'Correct':>8} {'Accuracy':>9} {'Errors':>7}")
    print("-" * 50)

    for lo, hi in buckets:
        bucket = [a for a in all_assignments if lo <= a.confidence < hi]
        if not bucket:
            continue
        n_correct = sum(1 for a in bucket if a.correct)
        n_total = len(bucket)
        acc = n_correct / n_total * 100 if n_total > 0 else 0
        errors = n_total - n_correct
        label = f"[{lo:.1f}, {hi:.1f})"
        print(f"{label:<14} {n_total:>6} {n_correct:>8} {acc:>8.1f}% {errors:>7}")

    # Assignment margin analysis
    print(f"\n{'Margin':<14} {'Count':>6} {'Correct':>8} {'Accuracy':>9}")
    print("-" * 44)

    margin_buckets = [
        (0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.0),
    ]

    for lo, hi in margin_buckets:
        bucket = [a for a in all_assignments if lo <= a.margin < hi]
        if not bucket:
            continue
        n_correct = sum(1 for a in bucket if a.correct)
        n_total = len(bucket)
        acc = n_correct / n_total * 100 if n_total > 0 else 0
        label = f"[{lo:.2f}, {hi:.2f})"
        print(f"{label:<14} {n_total:>6} {n_correct:>8} {acc:>8.1f}%")

    # Actionable thresholds
    print("\nActionable confidence gates:")
    for threshold in [0.60, 0.65, 0.70, 0.75, 0.80]:
        above = [a for a in all_assignments if a.confidence >= threshold]
        below = [a for a in all_assignments if a.confidence < threshold]
        if not above:
            continue
        acc_above = sum(1 for a in above if a.correct) / len(above) * 100
        errs_above = sum(1 for a in above if not a.correct)
        errs_below = sum(1 for a in below if not a.correct)
        print(
            f"  conf >= {threshold:.2f}: {len(above)} assignments, "
            f"{acc_above:.1f}% accuracy, {errs_above} errors "
            f"({errs_below} errors dropped)"
        )


# ---------------------------------------------------------------------------
# Section B: Side Switch Cascade Impact
# ---------------------------------------------------------------------------

def section_b(all_results: list[VideoResult]) -> None:
    print("\n" + "=" * 70)
    print("SECTION B: Side Switch Cascade Impact")
    print("=" * 70)

    from rallycut.tracking.match_tracker import (
        MatchPlayerTracker,
    )

    total_errors_detected = 0
    total_errors_perfect = 0
    total_assignments = 0

    videos_with_switches = 0

    for vr in all_results:
        # Count errors with detected switches (from the main run)
        errors_detected = sum(1 for a in vr.assignments if not a.correct)
        n_assignments = len(vr.assignments)
        total_errors_detected += errors_detected
        total_assignments += n_assignments

        # Check if this video has GT switches
        if not vr.gt_switches:
            total_errors_perfect += errors_detected  # no switches = same result
            continue

        videos_with_switches += 1

        # Re-run Pass 2 with perfect switches using cached stored_rally_data
        # Create a new tracker and transplant the stored data + profiles
        perfect_tracker = MatchPlayerTracker(collect_diagnostics=False)
        perfect_tracker.stored_rally_data = vr.stored_rally_data
        perfect_tracker.rally_count = len(vr.stored_rally_data)
        # Copy the state from the original tracker profiles
        for pid, profile in vr.player_profiles.items():
            perfect_tracker.state.players[pid] = profile

        # Override _detect_side_switches_combinatorial to return GT switches
        gt_sw = list(vr.gt_switches)

        class PerfectSwitchTracker(MatchPlayerTracker):
            def _detect_side_switches_combinatorial(self) -> list[int]:
                return gt_sw

        # Rebless the tracker
        perfect_tracker.__class__ = PerfectSwitchTracker

        # Run refine_assignments with perfect switches
        # Need initial results as input
        initial_for_refine = list(vr.rally_results)
        # Reset side assignments to Pass 1 state for proper refinement
        for i, data in enumerate(perfect_tracker.stored_rally_data):
            # Restore original player_side_assignment (undo any switch from main run)
            data.player_side_assignment = {
                1: 0, 2: 0, 3: 1, 4: 1,
            }

        perfect_results = perfect_tracker.refine_assignments(initial_for_refine)

        # Build predictions with perfect switches
        pred_perfect: dict[str, dict[str, int]] = {}
        for rid, rr in zip(vr.rally_id_order, perfect_results):
            pred_perfect[rid] = {str(k): v for k, v in rr.track_to_player.items()}

        _, correct_perfect, total_perfect = _find_best_permutation(
            vr.gt_rallies, pred_perfect
        )

        errors_perfect = total_perfect - correct_perfect
        total_errors_perfect += errors_perfect

        print(
            f"\n  {vr.video_id[:8]}: GT switches at {vr.gt_switches}"
        )
        print(
            f"    Detected switches: {errors_detected} errors / {n_assignments} assignments"
        )
        print(
            f"    Perfect switches:  {errors_perfect} errors / {total_perfect} assignments"
        )
        delta = errors_detected - errors_perfect
        if delta > 0:
            print(f"    -> {delta} errors attributable to switch detection")
        elif delta < 0:
            print(f"    -> Perfect switches WORSE by {-delta} (switch detection helped)")
        else:
            print("    -> No difference")

    print("\n--- Summary ---")
    print(f"Videos with GT switches: {videos_with_switches}")
    print(f"Total errors (detected switches): {total_errors_detected}")
    print(f"Total errors (perfect switches):  {total_errors_perfect}")
    delta = total_errors_detected - total_errors_perfect
    if total_assignments > 0:
        acc_detected = (total_assignments - total_errors_detected) / total_assignments * 100
        acc_perfect = (total_assignments - total_errors_perfect) / total_assignments * 100
        print(f"Accuracy (detected): {acc_detected:.1f}%")
        print(f"Accuracy (perfect):  {acc_perfect:.1f}% (+{acc_perfect - acc_detected:.1f}pp)")
    if delta > 0:
        print(f"Upper bound from perfect switch detection: {delta} fewer errors")


# ---------------------------------------------------------------------------
# Section C: Per-Video Deep Dive (worst 3)
# ---------------------------------------------------------------------------

def section_c(all_results: list[VideoResult]) -> None:
    print("\n" + "=" * 70)
    print("SECTION C: Per-Video Deep Dive (worst 3)")
    print("=" * 70)


    # Sort by accuracy, take worst 3
    sorted_videos = sorted(all_results, key=lambda v: v.accuracy)
    worst = sorted_videos[:3]

    for vr in worst:
        print(f"\n{'─' * 70}")
        print(f"Video: {vr.video_id[:8]}  Accuracy: {vr.correct}/{vr.total} = {vr.accuracy:.1f}%")
        print(f"Permutation: {vr.best_perm}")
        if vr.gt_switches:
            print(f"GT side switches: {vr.gt_switches}")
        print(f"{'─' * 70}")

        # Rally-by-rally timeline
        print(f"\n{'Idx':<4} {'RallyID':<10} {'Result':>8} {'Conf':>6} {'Switch':>7} {'Errors'}")
        print("-" * 65)

        for i, rid in enumerate(vr.rally_id_order):
            if rid not in vr.gt_rallies:
                continue

            rally_assignments = [
                a for a in vr.assignments
                if a.rally_id == rid
            ]

            r_correct = sum(1 for a in rally_assignments if a.correct)
            r_total = len(rally_assignments)
            conf = rally_assignments[0].confidence if rally_assignments else 0.0
            switch = "YES" if any(a.side_switch_detected for a in rally_assignments) else ""
            gt_sw = " <<GT_SW" if i in vr.gt_switches else ""

            errors_str = ""
            for a in rally_assignments:
                if not a.correct:
                    errors_str += f" T{a.track_id}:P{a.pred_player_id}→P{a.gt_player_id}({a.error_type[:5]})"

            status = f"{r_correct}/{r_total}"
            print(f"{i:<4} {rid[:8]:<10} {status:>8} {conf:>6.2f} {switch:>7} {errors_str}{gt_sw}")

            # Show cost matrix for wrong rallies
            if r_correct < r_total:
                diag = None
                for d in vr.diagnostics:
                    if d.rally_index == i:
                        diag = d
                        break

                if diag is not None:
                    cm = diag.cost_matrix
                    tids = diag.track_ids
                    pids = diag.player_ids

                    # Header
                    header = "        "
                    for pid in pids:
                        gt_label = f"P{vr.best_perm.get(pid, pid)}"
                        header += f"  {gt_label:>6}"
                    header += "  margin"
                    print(header)

                    for ti, tid in enumerate(tids):
                        if ti >= cm.shape[0]:
                            break
                        row = f"    T{tid:>3}:"
                        assigned_pid = diag.assignment.get(tid)
                        for pj, pid in enumerate(pids):
                            if pj >= cm.shape[1]:
                                break
                            cost = cm[ti, pj]
                            marker = " *" if pid == assigned_pid else "  "
                            row += f" {cost:.3f}{marker}"
                        if assigned_pid in diag.assignment_margins:
                            row += f"  {diag.assignment_margins[assigned_pid]:.3f}"
                        print(row)

                    # Root cause analysis
                    for a in rally_assignments:
                        if a.correct:
                            continue
                        # Was GT player 2nd-best or distant?
                        # Find row for this track and column for GT vs pred
                        tid_idx = tids.index(a.track_id) if a.track_id in tids else -1
                        if tid_idx < 0 or tid_idx >= cm.shape[0]:
                            continue

                        # Find inverse perm to get raw player IDs
                        inv_perm = {v: k for k, v in vr.best_perm.items()}
                        gt_raw_pid = inv_perm.get(a.gt_player_id)
                        pred_raw_pid = inv_perm.get(a.pred_player_id)

                        if gt_raw_pid is not None and pred_raw_pid is not None:
                            gt_col = pids.index(gt_raw_pid) if gt_raw_pid in pids else -1
                            pred_col = pids.index(pred_raw_pid) if pred_raw_pid in pids else -1
                            if gt_col >= 0 and pred_col >= 0:
                                gt_cost = cm[tid_idx, gt_col]
                                pred_cost = cm[tid_idx, pred_col]
                                gap = gt_cost - pred_cost
                                if gap < 0.02:
                                    cause = "marginal (GT almost as good)"
                                elif a.error_type == "cross-team":
                                    cause = "cross-team confusion (side error?)"
                                else:
                                    cause = f"appearance confusion (gap={gap:.3f})"
                                print(
                                    f"    -> T{a.track_id}: GT P{a.gt_player_id} "
                                    f"cost={gt_cost:.3f}, got P{a.pred_player_id} "
                                    f"cost={pred_cost:.3f} => {cause}"
                                )


# ---------------------------------------------------------------------------
# Section D: Profile Quality Over Time
# ---------------------------------------------------------------------------

def section_d(all_results: list[VideoResult]) -> None:
    print("\n" + "=" * 70)
    print("SECTION D: Profile Quality Over Time (rally index buckets)")
    print("=" * 70)

    buckets = [(0, 4), (5, 9), (10, 14), (15, 999)]
    bucket_labels = ["0-4", "5-9", "10-14", "15+"]

    print(f"\n{'Bucket':<10} {'Count':>6} {'Correct':>8} {'Accuracy':>9}")
    print("-" * 38)

    for (lo, hi), label in zip(buckets, bucket_labels):
        assignments = [
            a for vr in all_results
            for a in vr.assignments
            if lo <= a.rally_index <= hi
        ]
        if not assignments:
            continue
        n_correct = sum(1 for a in assignments if a.correct)
        n_total = len(assignments)
        acc = n_correct / n_total * 100
        print(f"{label:<10} {n_total:>6} {n_correct:>8} {acc:>8.1f}%")

    # Error type breakdown by bucket
    print(f"\n{'Bucket':<10} {'Within':>8} {'Cross':>8} {'Total Err':>10}")
    print("-" * 40)

    for (lo, hi), label in zip(buckets, bucket_labels):
        assignments = [
            a for vr in all_results
            for a in vr.assignments
            if lo <= a.rally_index <= hi
        ]
        within = sum(1 for a in assignments if a.error_type == "within-team")
        cross = sum(1 for a in assignments if a.error_type == "cross-team")
        total_err = within + cross
        if total_err == 0:
            continue
        print(f"{label:<10} {within:>8} {cross:>8} {total_err:>10}")


# ---------------------------------------------------------------------------
# Section E: Feature Contribution Analysis
# ---------------------------------------------------------------------------

def section_e(all_results: list[VideoResult]) -> None:
    print("\n" + "=" * 70)
    print("SECTION E: Feature Contribution Analysis")
    print("=" * 70)

    from rallycut.tracking.player_features import (
        _WEIGHT_DOMINANT_COLOR,
        _WEIGHT_LOWER_HIST,
        _WEIGHT_LOWER_V_HIST,
        _WEIGHT_SKIN,
        _WEIGHT_UPPER_HIST,
        _WEIGHT_UPPER_V_HIST,
        _histogram_similarity,
        _hsv_similarity,
    )

    feature_names = [
        ("lower_hs", "Lower HS hist", _WEIGHT_LOWER_HIST),
        ("lower_v", "Lower V hist", _WEIGHT_LOWER_V_HIST),
        ("upper_hs", "Upper HS hist", _WEIGHT_UPPER_HIST),
        ("upper_v", "Upper V hist", _WEIGHT_UPPER_V_HIST),
        ("skin", "Skin tone", _WEIGHT_SKIN),
        ("dominant", "Dominant color", _WEIGHT_DOMINANT_COLOR),
    ]

    # For each wrong assignment, decompose which features favored incorrect vs correct
    feature_correct_rank = {name: 0 for name, _, _ in feature_names}
    feature_total = {name: 0 for name, _, _ in feature_names}
    feature_favored_wrong = {name: 0 for name, _, _ in feature_names}

    n_analyzed = 0

    for vr in all_results:
        inv_perm = {v: k for k, v in vr.best_perm.items()}

        for a in vr.assignments:
            if a.correct:
                continue

            # Find the stored rally data for this rally
            rally_idx = a.rally_index
            if rally_idx >= len(vr.stored_rally_data):
                continue

            data = vr.stored_rally_data[rally_idx]
            if a.track_id not in data.track_stats:
                continue

            track_stats = data.track_stats[a.track_id]

            # Get raw player IDs for GT and pred
            gt_raw_pid = inv_perm.get(a.gt_player_id)
            pred_raw_pid = inv_perm.get(a.pred_player_id)
            if gt_raw_pid is None or pred_raw_pid is None:
                continue
            if gt_raw_pid not in vr.player_profiles or pred_raw_pid not in vr.player_profiles:
                continue

            gt_profile = vr.player_profiles[gt_raw_pid]
            pred_profile = vr.player_profiles[pred_raw_pid]

            n_analyzed += 1

            # Compute per-feature similarity for GT and pred profiles
            def get_feature_sims(profile: Any) -> dict[str, float | None]:
                sims: dict[str, float | None] = {}
                sims["lower_hs"] = _histogram_similarity(
                    profile.avg_lower_hist, track_stats.avg_lower_hist
                )
                sims["lower_v"] = _histogram_similarity(
                    profile.avg_lower_v_hist, track_stats.avg_lower_v_hist
                )
                sims["upper_hs"] = _histogram_similarity(
                    profile.avg_upper_hist, track_stats.avg_upper_hist
                )
                sims["upper_v"] = _histogram_similarity(
                    profile.avg_upper_v_hist, track_stats.avg_upper_v_hist
                )
                if profile.avg_skin_tone_hsv is not None and track_stats.avg_skin_tone_hsv is not None:
                    sims["skin"] = _hsv_similarity(
                        profile.avg_skin_tone_hsv, track_stats.avg_skin_tone_hsv
                    )
                else:
                    sims["skin"] = None
                if profile.avg_dominant_color_hsv is not None and track_stats.avg_dominant_color_hsv is not None:
                    sims["dominant"] = _hsv_similarity(
                        profile.avg_dominant_color_hsv, track_stats.avg_dominant_color_hsv
                    )
                else:
                    sims["dominant"] = None
                return sims

            gt_sims = get_feature_sims(gt_profile)
            pred_sims = get_feature_sims(pred_profile)

            for name, _, _ in feature_names:
                gt_s = gt_sims.get(name)
                pred_s = pred_sims.get(name)
                if gt_s is not None and pred_s is not None:
                    feature_total[name] += 1
                    if gt_s > pred_s:
                        feature_correct_rank[name] += 1
                    elif pred_s > gt_s:
                        feature_favored_wrong[name] += 1
                    # Equal: neither counted

    print(f"\nAnalyzed {n_analyzed} wrong assignments\n")
    print(f"{'Feature':<18} {'Weight':>7} {'Avail':>6} {'Correct':>8} {'Wrong':>7} {'Discrim':>8}")
    print("-" * 60)

    for name, label, weight in feature_names:
        total = feature_total[name]
        correct = feature_correct_rank[name]
        wrong = feature_favored_wrong[name]
        discrim = correct / total * 100 if total > 0 else 0
        print(
            f"{label:<18} {weight:>6.0%} {total:>6} {correct:>8} {wrong:>7} {discrim:>7.1f}%"
        )

    print(
        "\nDiscrim% = how often this feature ranks GT player above the "
        "incorrect player (higher = more useful)"
    )


# ---------------------------------------------------------------------------
# Section F: Optimization Summary
# ---------------------------------------------------------------------------

def section_f(all_results: list[VideoResult]) -> None:
    print("\n" + "=" * 70)
    print("SECTION F: Optimization Summary")
    print("=" * 70)

    all_assignments: list[AssignmentRecord] = []
    for vr in all_results:
        all_assignments.extend(vr.assignments)

    total = len(all_assignments)
    total_correct = sum(1 for a in all_assignments if a.correct)
    total_errors = total - total_correct
    within_errors = sum(1 for a in all_assignments if a.error_type == "within-team")
    cross_errors = sum(1 for a in all_assignments if a.error_type == "cross-team")

    print(f"\nCurrent: {total_correct}/{total} = {total_correct / total * 100:.1f}% accuracy")
    print(f"Errors: {total_errors} ({within_errors} within-team, {cross_errors} cross-team)")

    # Recommendation 1: Confidence gating
    print("\n1. CONFIDENCE GATING")
    for threshold in [0.65, 0.70, 0.75]:
        above = [a for a in all_assignments if a.confidence >= threshold]
        if not above:
            continue
        errs = sum(1 for a in above if not a.correct)
        dropped = sum(1 for a in all_assignments if a.confidence < threshold)
        dropped_errs = sum(1 for a in all_assignments if a.confidence < threshold and not a.correct)
        acc = sum(1 for a in above if a.correct) / len(above) * 100
        print(
            f"   conf >= {threshold:.2f}: {acc:.1f}% accuracy on {len(above)} assignments, "
            f"drops {dropped} assignments ({dropped_errs} of which are errors)"
        )

    # Recommendation 2: Error concentration
    print("\n2. ERROR CONCENTRATION")
    sorted_videos = sorted(all_results, key=lambda v: v.accuracy)
    worst_3 = sorted_videos[:3]
    worst_3_errors = sum(
        sum(1 for a in vr.assignments if not a.correct)
        for vr in worst_3
    )
    print(
        f"   Worst 3 videos contain {worst_3_errors}/{total_errors} errors "
        f"({worst_3_errors / total_errors * 100:.0f}%):"
    )
    for vr in worst_3:
        errs = sum(1 for a in vr.assignments if not a.correct)
        print(f"     {vr.video_id[:8]}: {vr.accuracy:.1f}% ({errs} errors)")

    # Recommendation 3: Rally index
    print("\n3. EARLY vs LATE RALLIES")
    early = [a for a in all_assignments if a.rally_index <= 4]
    late = [a for a in all_assignments if a.rally_index > 4]
    if early and late:
        early_acc = sum(1 for a in early if a.correct) / len(early) * 100
        late_acc = sum(1 for a in late if a.correct) / len(late) * 100
        print(f"   Rallies 0-4: {early_acc:.1f}% accuracy ({len(early)} assignments)")
        print(f"   Rallies 5+:  {late_acc:.1f}% accuracy ({len(late)} assignments)")
        if early_acc < late_acc:
            print(f"   -> Profiles improve over time (+{late_acc - early_acc:.1f}pp)")
        else:
            print(f"   -> Profile drift detected (-{early_acc - late_acc:.1f}pp)")

    # Recommendation 4: Error type focus
    print("\n4. ERROR TYPE FOCUS")
    if cross_errors > within_errors:
        print(
            f"   Cross-team errors dominate ({cross_errors} vs {within_errors}). "
            f"Focus on side switch detection and team classification."
        )
    else:
        print(
            f"   Within-team errors dominate ({within_errors} vs {cross_errors}). "
            f"Focus on within-team discrimination (position, fine-grained appearance)."
        )

    # Recommendation 5: Margin analysis
    print("\n5. MARGIN ANALYSIS")
    low_margin_errors = [
        a for a in all_assignments if not a.correct and a.margin < 0.05
    ]
    high_margin_errors = [
        a for a in all_assignments if not a.correct and a.margin >= 0.05
    ]
    print(
        f"   Low-margin errors (margin < 0.05): {len(low_margin_errors)} "
        f"(ambiguous, may benefit from better features)"
    )
    print(
        f"   High-margin errors (margin >= 0.05): {len(high_margin_errors)} "
        f"(confident but wrong, likely cascade/side errors)"
    )


# ---------------------------------------------------------------------------
# Section G: GT Quality Audit
# ---------------------------------------------------------------------------

def section_g(all_results: list[VideoResult]) -> None:
    print("\n" + "=" * 70)
    print("SECTION G: GT Quality Audit")
    print("=" * 70)

    print(f"\n{'Video':<10} {'GTRly':>6} {'<4asgn':>7} {'DupID':>6} {'Stale':>6} "
          f"{'Orphan':>7} {'SwOK':>5} {'Acc':>7} {'Quality'}")
    print("-" * 78)

    total_issues = 0
    suspect_videos: list[str] = []

    for vr in all_results:
        issues: list[str] = []
        n_gt_rallies = len(vr.gt_rallies)

        # 1. Rallies with < 4 player assignments (incomplete labeling)
        incomplete = 0
        for rid, mapping in vr.gt_rallies.items():
            player_ids_used = set(mapping.values())
            if len(player_ids_used) < 4:
                incomplete += 1
        if incomplete > 0:
            issues.append(f"{incomplete} incomplete")

        # 2. Duplicate player IDs in a single rally
        duplicates = 0
        for rid, mapping in vr.gt_rallies.items():
            pid_counts: dict[int, int] = {}
            for pid in mapping.values():
                pid_counts[pid] = pid_counts.get(pid, 0) + 1
            if any(c > 1 for c in pid_counts.values()):
                duplicates += 1
        if duplicates > 0:
            issues.append(f"{duplicates} dup-IDs")

        # 3. Stale track IDs (GT references tracks not in current predictions)
        stale = 0
        for rid, mapping in vr.gt_rallies.items():
            if rid not in vr.all_pred_track_ids:
                continue
            pred_tids = vr.all_pred_track_ids[rid]
            for tid_str in mapping:
                if int(tid_str) not in pred_tids:
                    stale += 1
        if stale > 0:
            issues.append(f"{stale} stale-trk")

        # 4. Orphaned GT rallies (not in rally_id_order)
        orphaned = 0
        rally_set = set(vr.rally_id_order)
        for rid in vr.gt_rallies:
            if rid not in rally_set:
                orphaned += 1
        if orphaned > 0:
            issues.append(f"{orphaned} orphaned")

        # 5. Side switch validation
        sw_ok = True
        if vr.gt_switches:
            max_idx = len(vr.rally_id_order) - 1
            for sw in vr.gt_switches:
                if sw < 0 or sw > max_idx:
                    sw_ok = False
                    break
            # Check spacing >= 4
            sorted_sw = sorted(vr.gt_switches)
            for i in range(1, len(sorted_sw)):
                if sorted_sw[i] - sorted_sw[i - 1] < 4:
                    sw_ok = False
                    break
        if not sw_ok:
            issues.append("bad-switch")

        # 6. Low accuracy = suspect GT
        is_suspect = vr.accuracy < 60.0 and vr.total >= 4

        quality = "OK" if not issues else "; ".join(issues)
        if is_suspect:
            quality = "SUSPECT " + quality
            suspect_videos.append(vr.video_id[:8])

        total_issues += len(issues)

        print(
            f"{vr.video_id[:8]:<10} {n_gt_rallies:>6} {incomplete:>7} {duplicates:>6} "
            f"{stale:>6} {orphaned:>7} {'Y' if sw_ok else 'N':>5} "
            f"{vr.accuracy:>6.1f}% {quality}"
        )

    print(f"\nTotal videos: {len(all_results)}, videos with issues: "
          f"{sum(1 for vr in all_results if vr.accuracy < 60.0 or any(True for rid, m in vr.gt_rallies.items() if len(set(m.values())) < 4))}")
    if suspect_videos:
        print(f"Suspect GT (accuracy < 60%): {', '.join(suspect_videos)}")
    else:
        print("No suspect GT videos (all >= 60% accuracy)")


# ---------------------------------------------------------------------------
# Section H: Delta Comparison (vs previous baseline)
# ---------------------------------------------------------------------------

# Previous baseline (hardcoded from prior evaluation)
_BASELINE_VIDEOS = 28
_BASELINE_ASSIGNMENTS = 838
_BASELINE_ACCURACY = 85.8
_BASELINE_CROSS_TEAM_ERRORS = 80
_BASELINE_WITHIN_TEAM_ERRORS = 39


def section_h(all_results: list[VideoResult]) -> None:
    print("\n" + "=" * 70)
    print("SECTION H: Delta Comparison vs Previous Baseline")
    print("=" * 70)

    all_assignments: list[AssignmentRecord] = []
    for vr in all_results:
        all_assignments.extend(vr.assignments)

    total = len(all_assignments)
    total_correct = sum(1 for a in all_assignments if a.correct)
    total_errors = total - total_correct
    cross_errors = sum(1 for a in all_assignments if a.error_type == "cross-team")
    within_errors = sum(1 for a in all_assignments if a.error_type == "within-team")
    accuracy = total_correct / total * 100 if total > 0 else 0

    baseline_errors = _BASELINE_CROSS_TEAM_ERRORS + _BASELINE_WITHIN_TEAM_ERRORS

    # Comparison table
    print(f"\n{'Metric':<25} {'Baseline':>12} {'Current':>12} {'Delta':>12}")
    print("-" * 65)
    print(f"{'Videos':<25} {_BASELINE_VIDEOS:>12} {len(all_results):>12} "
          f"{len(all_results) - _BASELINE_VIDEOS:>+12}")
    print(f"{'Assignments':<25} {_BASELINE_ASSIGNMENTS:>12} {total:>12} "
          f"{total - _BASELINE_ASSIGNMENTS:>+12}")
    print(f"{'Accuracy':<25} {_BASELINE_ACCURACY:>11.1f}% {accuracy:>11.1f}% "
          f"{accuracy - _BASELINE_ACCURACY:>+11.1f}%")
    print(f"{'Total errors':<25} {baseline_errors:>12} {total_errors:>12} "
          f"{total_errors - baseline_errors:>+12}")
    print(f"{'Cross-team errors':<25} {_BASELINE_CROSS_TEAM_ERRORS:>12} {cross_errors:>12} "
          f"{cross_errors - _BASELINE_CROSS_TEAM_ERRORS:>+12}")
    print(f"{'Within-team errors':<25} {_BASELINE_WITHIN_TEAM_ERRORS:>12} {within_errors:>12} "
          f"{within_errors - _BASELINE_WITHIN_TEAM_ERRORS:>+12}")

    # Try to identify which videos are "new" vs "original corpus"
    # We can't know exactly which 28 videos were in the baseline, but we can
    # show per-video accuracy sorted to help identify patterns
    print(f"\n--- Per-Video Accuracy (sorted) ---\n")
    print(f"{'Video':<10} {'Correct':>8} {'Total':>6} {'Accuracy':>9} {'Cross':>6} {'Within':>7}")
    print("-" * 52)

    sorted_videos = sorted(all_results, key=lambda v: v.accuracy)
    for vr in sorted_videos:
        v_cross = sum(1 for a in vr.assignments if a.error_type == "cross-team")
        v_within = sum(1 for a in vr.assignments if a.error_type == "within-team")
        print(
            f"{vr.video_id[:8]:<10} {vr.correct:>8} {vr.total:>6} "
            f"{vr.accuracy:>8.1f}% {v_cross:>6} {v_within:>7}"
        )

    # Summary interpretation
    print(f"\n--- Interpretation ---")
    if accuracy > _BASELINE_ACCURACY:
        print(f"Accuracy improved by {accuracy - _BASELINE_ACCURACY:+.1f}pp")
    elif accuracy < _BASELINE_ACCURACY:
        print(f"Accuracy dropped by {accuracy - _BASELINE_ACCURACY:+.1f}pp")
    else:
        print("Accuracy unchanged")

    if len(all_results) > _BASELINE_VIDEOS:
        n_new = len(all_results) - _BASELINE_VIDEOS
        print(f"{n_new} new video(s) added to GT corpus")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Comprehensive cross-rally player matching diagnosis"
    )
    parser.add_argument("--video-id", type=str, help="Analyze single video")
    parser.add_argument(
        "--sections", type=str, default="a,b,c,d,e,f,g,h",
        help="Comma-separated sections to run (default: a,b,c,d,e,f,g,h)",
    )
    args = parser.parse_args()

    sections = set(args.sections.lower().split(","))

    print("Collecting data (re-running match-players with diagnostics)...")
    all_results = collect_data(args.video_id)

    if not all_results:
        print("No results collected")
        sys.exit(1)

    # Print aggregate
    total_correct = sum(vr.correct for vr in all_results)
    total_assignments = sum(vr.total for vr in all_results)
    agg_acc = total_correct / total_assignments * 100 if total_assignments > 0 else 0
    print(f"\nAggregate: {total_correct}/{total_assignments} = {agg_acc:.1f}%")

    if "g" in sections:
        section_g(all_results)
    if "a" in sections:
        section_a(all_results)
    if "b" in sections:
        section_b(all_results)
    if "c" in sections:
        section_c(all_results)
    if "d" in sections:
        section_d(all_results)
    if "e" in sections:
        section_e(all_results)
    if "f" in sections:
        section_f(all_results)
    if "h" in sections:
        section_h(all_results)


if __name__ == "__main__":
    main()
