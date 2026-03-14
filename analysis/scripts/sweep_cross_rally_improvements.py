#!/usr/bin/env python3
"""Sweep cross-rally player matching improvements.

Three analyses using pre-extracted track features (single video-read pass):
  1. Cross-team error root cause (court-side misclassification vs appearance)
  2. Feature weight grid search
  3. Bootstrap delay (seed profiles from rally N, then re-score all)

Usage:
    cd analysis
    uv run python scripts/sweep_cross_rally_improvements.py
    uv run python scripts/sweep_cross_rally_improvements.py --video-id a5866029
    uv run python scripts/sweep_cross_rally_improvements.py --analysis 1,2,3
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np

logging.basicConfig(level=logging.WARNING, format="%(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class VideoData:
    """Pre-extracted data for one video."""

    video_id: str
    gt_rallies: dict[str, dict[str, int]]
    gt_switches: list[int]
    # Per-rally extracted track stats (from video frames)
    rally_track_stats: list[dict[int, Any]]  # TrackAppearanceStats per rally
    # Per-rally metadata
    rally_ids: list[str]
    rally_positions: list[list[Any]]  # PlayerPosition lists
    rally_ball_positions: list[list[Any] | None]
    rally_court_split_y: list[float | None]
    rally_team_assignments: list[dict[int, int] | None]
    rally_primary_track_ids: list[list[int]]


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


# ---------------------------------------------------------------------------
# Data collection (single video-read pass)
# ---------------------------------------------------------------------------

def collect_data(video_id_filter: str | None = None) -> list[VideoData]:
    """Extract track features for all GT videos (one-time video read)."""
    from rallycut.evaluation.db import get_connection
    from rallycut.evaluation.tracking.db import get_video_path, load_rallies_for_video
    from rallycut.tracking.match_tracker import extract_rally_appearances

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

    results: list[VideoData] = []

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
            continue

        video_path = get_video_path(vid)
        if not video_path:
            continue

        print(
            f"[{row_idx + 1}/{len(rows)}] {vid[:8]}: extracting features "
            f"for {len(rallies)} rallies...",
            end="", flush=True,
        )

        rally_track_stats = []
        rally_ids = []
        rally_positions = []
        rally_ball_positions: list[list[Any] | None] = []
        rally_court_split_y: list[float | None] = []
        rally_team_assignments: list[dict[int, int] | None] = []
        rally_primary_track_ids: list[list[int]] = []

        for rally in rallies:
            track_stats = extract_rally_appearances(
                video_path=video_path,
                positions=rally.positions,
                primary_track_ids=rally.primary_track_ids,
                start_ms=rally.start_ms,
                end_ms=rally.end_ms,
                num_samples=12,
            )
            rally_track_stats.append(track_stats)
            rally_ids.append(rally.rally_id)
            rally_positions.append(rally.positions)
            rally_ball_positions.append(rally.ball_positions)
            rally_court_split_y.append(rally.court_split_y)
            rally_team_assignments.append(rally.team_assignments)
            rally_primary_track_ids.append(rally.primary_track_ids)

        print(" done")

        results.append(VideoData(
            video_id=vid,
            gt_rallies=gt_rallies,
            gt_switches=gt_switches,
            rally_track_stats=rally_track_stats,
            rally_ids=rally_ids,
            rally_positions=rally_positions,
            rally_ball_positions=rally_ball_positions,
            rally_court_split_y=rally_court_split_y,
            rally_team_assignments=rally_team_assignments,
            rally_primary_track_ids=rally_primary_track_ids,
        ))

    return results


def run_matching(
    vd: VideoData,
    weights: dict[str, float] | None = None,
    bootstrap_from: int = 0,
) -> tuple[dict[str, dict[str, int]], list[Any], list[Any], dict[int, Any]]:
    """Run match-players pipeline on pre-extracted data.

    Args:
        vd: Pre-extracted video data.
        weights: Optional weight overrides for feature similarity.
        bootstrap_from: If >0, use first N rallies to build profiles,
            then re-score all rallies with frozen profiles.

    Returns:
        (pred_rallies, rally_results, stored_rally_data, player_profiles)
    """
    import rallycut.tracking.player_features as pf
    from rallycut.tracking.match_tracker import MatchPlayerTracker

    # Monkey-patch weights if provided
    original_weights = {}
    if weights:
        weight_map = {
            "lower_hs": "_WEIGHT_LOWER_HIST",
            "lower_v": "_WEIGHT_LOWER_V_HIST",
            "upper_hs": "_WEIGHT_UPPER_HIST",
            "upper_v": "_WEIGHT_UPPER_V_HIST",
            "skin": "_WEIGHT_SKIN",
            "dominant": "_WEIGHT_DOMINANT_COLOR",
        }
        for key, attr in weight_map.items():
            if key in weights:
                original_weights[attr] = getattr(pf, attr)
                setattr(pf, attr, weights[key])

    try:
        tracker = MatchPlayerTracker(collect_diagnostics=True)
        initial_results = []

        for i in range(len(vd.rally_ids)):
            result = tracker.process_rally(
                track_stats=vd.rally_track_stats[i],
                player_positions=vd.rally_positions[i],
                ball_positions=vd.rally_ball_positions[i],
                court_split_y=vd.rally_court_split_y[i],
                team_assignments=vd.rally_team_assignments[i],
            )
            initial_results.append(result)

        refined_results = tracker.refine_assignments(initial_results)

        # Build predictions
        pred_rallies: dict[str, dict[str, int]] = {}
        for rid, rr in zip(vd.rally_ids, refined_results):
            pred_rallies[rid] = {str(k): v for k, v in rr.track_to_player.items()}

        return (
            pred_rallies,
            refined_results,
            tracker.stored_rally_data,
            dict(tracker.state.players),
        )
    finally:
        # Restore original weights
        for attr, val in original_weights.items():
            setattr(pf, attr, val)


def evaluate(
    vd: VideoData,
    pred_rallies: dict[str, dict[str, int]],
) -> tuple[int, int, dict[int, int]]:
    """Evaluate predictions against GT. Returns (correct, total, best_perm)."""
    best_perm, correct, total = _find_best_permutation(vd.gt_rallies, pred_rallies)
    return correct, total, best_perm


# ---------------------------------------------------------------------------
# Analysis 1: Cross-team error root cause
# ---------------------------------------------------------------------------

def analysis_1(all_data: list[VideoData]) -> None:
    print("\n" + "=" * 70)
    print("ANALYSIS 1: Cross-Team Error Root Cause")
    print("=" * 70)

    court_side_correct = 0  # track_court_sides matches GT team
    court_side_wrong = 0  # track_court_sides disagrees with GT team
    total_cross_errors = 0

    for vd in all_data:
        pred_rallies, rally_results, stored_data, profiles = run_matching(vd)
        correct, total, best_perm = evaluate(vd, pred_rallies)

        for i, (rid, rr) in enumerate(zip(vd.rally_ids, rally_results)):
            if rid not in vd.gt_rallies:
                continue

            gt = vd.gt_rallies[rid]
            pred = pred_rallies.get(rid, {})
            data = stored_data[i] if i < len(stored_data) else None

            for tid_str in gt:
                if tid_str not in pred:
                    continue
                gt_pid = gt[tid_str]
                raw_pred_pid = pred[tid_str]
                mapped_pid = best_perm.get(raw_pred_pid)

                if mapped_pid == gt_pid:
                    continue

                # Classify error
                gt_team = 0 if gt_pid <= 2 else 1
                mapped_team = 0 if (mapped_pid or 0) <= 2 else 1
                if gt_team == mapped_team:
                    continue  # within-team, skip

                total_cross_errors += 1

                # Check court side classification
                if data is not None:
                    track_side = data.track_court_sides.get(int(tid_str))
                    if track_side is not None:
                        if track_side == gt_team:
                            court_side_correct += 1
                        else:
                            court_side_wrong += 1

        print(f"  {vd.video_id[:8]}: processed")

    print(f"\n--- Cross-Team Error Root Cause ---")
    print(f"Total cross-team errors: {total_cross_errors}")
    print(f"  Court side CORRECT (appearance confusion): {court_side_correct}")
    print(f"  Court side WRONG (side misclassification): {court_side_wrong}")
    if total_cross_errors > 0:
        pct_side = court_side_wrong / total_cross_errors * 100
        pct_app = court_side_correct / total_cross_errors * 100
        print(
            f"\n  {pct_side:.0f}% caused by court-side misclassification"
        )
        print(
            f"  {pct_app:.0f}% caused by appearance confusion "
            f"(correct side, wrong team by appearance)"
        )
    if court_side_wrong > court_side_correct:
        print(
            "\n  >> Recommendation: Improve court-side classification "
            "(team_assignments, court_split_y)"
        )
    else:
        print(
            "\n  >> Recommendation: Improve cross-team appearance "
            "discrimination or increase side penalty"
        )


# ---------------------------------------------------------------------------
# Analysis 2: Feature weight grid search
# ---------------------------------------------------------------------------

def analysis_2(all_data: list[VideoData]) -> None:
    print("\n" + "=" * 70)
    print("ANALYSIS 2: Feature Weight Grid Search")
    print("=" * 70)

    # Current weights (baseline)
    baseline_weights = {
        "lower_hs": 0.35, "lower_v": 0.15, "upper_hs": 0.15,
        "upper_v": 0.10, "skin": 0.10, "dominant": 0.15,
    }

    # First, compute baseline
    print("\nComputing baseline...")
    baseline_correct = 0
    baseline_total = 0
    for vd in all_data:
        pred_rallies, _, _, _ = run_matching(vd, weights=baseline_weights)
        c, t, _ = evaluate(vd, pred_rallies)
        baseline_correct += c
        baseline_total += t
    baseline_acc = baseline_correct / baseline_total * 100
    print(f"Baseline: {baseline_correct}/{baseline_total} = {baseline_acc:.1f}%")

    # Weight configurations to try
    # Strategy: since all features have <36% discrimination on errors,
    # try rebalancing, dropping features, and emphasizing spatial
    configs: list[tuple[str, dict[str, float]]] = [
        # Rebalancing
        ("equal_weights", {
            "lower_hs": 1/6, "lower_v": 1/6, "upper_hs": 1/6,
            "upper_v": 1/6, "skin": 1/6, "dominant": 1/6,
        }),
        # Emphasize upper (slightly better discrimination)
        ("upper_heavy", {
            "lower_hs": 0.15, "lower_v": 0.10, "upper_hs": 0.30,
            "upper_v": 0.20, "skin": 0.10, "dominant": 0.15,
        }),
        # Emphasize V histograms + dominant (best discriminators)
        ("v_dominant_heavy", {
            "lower_hs": 0.15, "lower_v": 0.20, "upper_hs": 0.10,
            "upper_v": 0.20, "skin": 0.10, "dominant": 0.25,
        }),
        # Drop skin (29.9% discrim)
        ("no_skin", {
            "lower_hs": 0.35, "lower_v": 0.20, "upper_hs": 0.15,
            "upper_v": 0.10, "skin": 0.00, "dominant": 0.20,
        }),
        # Drop lower_hs (worst discriminator at 22.8%)
        ("no_lower_hs", {
            "lower_hs": 0.00, "lower_v": 0.25, "upper_hs": 0.25,
            "upper_v": 0.15, "skin": 0.10, "dominant": 0.25,
        }),
        # Lower HS reduced
        ("lower_hs_reduced", {
            "lower_hs": 0.15, "lower_v": 0.20, "upper_hs": 0.20,
            "upper_v": 0.15, "skin": 0.10, "dominant": 0.20,
        }),
        # Only histograms (drop skin + dominant color)
        ("hists_only", {
            "lower_hs": 0.35, "lower_v": 0.20, "upper_hs": 0.25,
            "upper_v": 0.20, "skin": 0.00, "dominant": 0.00,
        }),
        # Only V + dominant (best discriminators)
        ("best_discriminators", {
            "lower_hs": 0.00, "lower_v": 0.25, "upper_hs": 0.00,
            "upper_v": 0.30, "skin": 0.10, "dominant": 0.35,
        }),
        # Dominant color heavy (33.1% discrim, second best)
        ("dominant_heavy", {
            "lower_hs": 0.20, "lower_v": 0.10, "upper_hs": 0.15,
            "upper_v": 0.10, "skin": 0.10, "dominant": 0.35,
        }),
        # Upper V heavy (35.9% discrim, best)
        ("upper_v_heavy", {
            "lower_hs": 0.20, "lower_v": 0.10, "upper_hs": 0.10,
            "upper_v": 0.35, "skin": 0.10, "dominant": 0.15,
        }),
    ]

    results: list[tuple[str, int, int, float]] = []
    results.append(("baseline", baseline_correct, baseline_total, baseline_acc))

    for name, weights in configs:
        total_correct = 0
        total_total = 0
        for vd in all_data:
            pred_rallies, _, _, _ = run_matching(vd, weights=weights)
            c, t, _ = evaluate(vd, pred_rallies)
            total_correct += c
            total_total += t
        acc = total_correct / total_total * 100
        results.append((name, total_correct, total_total, acc))
        delta = acc - baseline_acc
        marker = ">>>" if delta > 0.5 else "   "
        print(f"  {marker} {name:<25} {total_correct}/{total_total} = {acc:.1f}% ({delta:+.1f}pp)")

    # Sort by accuracy
    results.sort(key=lambda x: -x[3])
    print(f"\n--- Ranking ---")
    print(f"{'Config':<25} {'Correct':>8} {'Total':>6} {'Acc':>7} {'Delta':>7}")
    print("-" * 58)
    for name, correct, total, acc in results:
        delta = acc - baseline_acc
        print(f"{name:<25} {correct:>8} {total:>6} {acc:>6.1f}% {delta:>+6.1f}pp")


# ---------------------------------------------------------------------------
# Analysis 3: Bootstrap delay
# ---------------------------------------------------------------------------

def analysis_3(all_data: list[VideoData]) -> None:
    print("\n" + "=" * 70)
    print("ANALYSIS 3: Bootstrap Delay (Profile Seeding)")
    print("=" * 70)

    # Strategy: Run Pass 1 normally to build profiles, then in Pass 2,
    # re-score ALL rallies (including early ones) with the final profiles.
    # This is actually what the current code already does in stage 1 of
    # refine_assignments. But let's test: what if we skip profile updates
    # for the first N rallies, then build profiles only from rally N onward?

    from rallycut.tracking.match_tracker import MatchPlayerTracker

    # Test different MIN_PROFILE_UPDATE_CONFIDENCE thresholds
    import rallycut.tracking.match_tracker as mt

    original_min_conf = mt.MIN_PROFILE_UPDATE_CONFIDENCE

    # First compute baseline
    print("\nComputing baseline (MIN_PROFILE_UPDATE_CONFIDENCE=0.55)...")
    baseline_correct = 0
    baseline_total = 0
    for vd in all_data:
        pred_rallies, _, _, _ = run_matching(vd)
        c, t, _ = evaluate(vd, pred_rallies)
        baseline_correct += c
        baseline_total += t
    baseline_acc = baseline_correct / baseline_total * 100
    print(f"Baseline: {baseline_correct}/{baseline_total} = {baseline_acc:.1f}%")

    # Test different confidence gates for profile updates
    gate_values = [0.0, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]

    print(f"\n{'Gate':>6} {'Correct':>8} {'Total':>6} {'Acc':>7} {'Delta':>7}")
    print("-" * 40)

    for gate in gate_values:
        mt.MIN_PROFILE_UPDATE_CONFIDENCE = gate
        total_correct = 0
        total_total = 0
        for vd in all_data:
            pred_rallies, _, _, _ = run_matching(vd)
            c, t, _ = evaluate(vd, pred_rallies)
            total_correct += c
            total_total += t
        acc = total_correct / total_total * 100
        delta = acc - baseline_acc
        marker = ">>>" if delta > 0.5 else "   "
        print(f"{marker}{gate:>5.2f} {total_correct:>8} {total_total:>6} {acc:>6.1f}% {delta:>+6.1f}pp")

    mt.MIN_PROFILE_UPDATE_CONFIDENCE = original_min_conf

    # Test SIDE_PENALTY sensitivity
    print(f"\nSide penalty sensitivity:")
    original_side_penalty = mt.SIDE_PENALTY

    penalty_values = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]

    print(f"{'Penalty':>8} {'Correct':>8} {'Total':>6} {'Acc':>7} {'Delta':>7}")
    print("-" * 42)

    for penalty in penalty_values:
        mt.SIDE_PENALTY = penalty
        total_correct = 0
        total_total = 0
        for vd in all_data:
            pred_rallies, _, _, _ = run_matching(vd)
            c, t, _ = evaluate(vd, pred_rallies)
            total_correct += c
            total_total += t
        acc = total_correct / total_total * 100
        delta = acc - baseline_acc
        marker = ">>>" if delta > 0.5 else "   "
        print(f"{marker}{penalty:>7.2f} {total_correct:>8} {total_total:>6} {acc:>6.1f}% {delta:>+6.1f}pp")

    mt.SIDE_PENALTY = original_side_penalty

    # Test POSITION_WEIGHT sensitivity
    print(f"\nPosition weight sensitivity:")
    original_pos_weight = mt.POSITION_WEIGHT

    pos_values = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60]

    print(f"{'PosW':>6} {'Correct':>8} {'Total':>6} {'Acc':>7} {'Delta':>7}")
    print("-" * 40)

    for pw in pos_values:
        mt.POSITION_WEIGHT = pw
        total_correct = 0
        total_total = 0
        for vd in all_data:
            pred_rallies, _, _, _ = run_matching(vd)
            c, t, _ = evaluate(vd, pred_rallies)
            total_correct += c
            total_total += t
        acc = total_correct / total_total * 100
        delta = acc - baseline_acc
        marker = ">>>" if delta > 0.5 else "   "
        print(f"{marker}{pw:>5.2f} {total_correct:>8} {total_total:>6} {acc:>6.1f}% {delta:>+6.1f}pp")

    mt.POSITION_WEIGHT = original_pos_weight

    # Test switch penalty sensitivity
    print(f"\nSwitch penalty sensitivity:")

    switch_values = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]

    print(f"{'SwPen':>6} {'Correct':>8} {'Total':>6} {'Acc':>7} {'Delta':>7}")
    print("-" * 40)

    # Import the actual module to set its globals (globals().get() reads
    # from match_tracker's own namespace, not our reference)
    import rallycut.tracking.match_tracker as mt_mod

    for sp in switch_values:
        mt_mod._SWITCH_PENALTY_OVERRIDE = sp  # type: ignore[attr-defined]
        total_correct = 0
        total_total = 0
        for vd in all_data:
            pred_rallies, _, _, _ = run_matching(vd)
            c, t, _ = evaluate(vd, pred_rallies)
            total_correct += c
            total_total += t
        acc = total_correct / total_total * 100
        delta = acc - baseline_acc
        marker = ">>>" if delta > 0.5 else "   "
        print(f"{marker}{sp:>5.1f} {total_correct:>8} {total_total:>6} {acc:>6.1f}% {delta:>+6.1f}pp")

    # Clean up
    if hasattr(mt_mod, "_SWITCH_PENALTY_OVERRIDE"):
        delattr(mt_mod, "_SWITCH_PENALTY_OVERRIDE")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep cross-rally player matching improvements"
    )
    parser.add_argument("--video-id", type=str, help="Analyze single video")
    parser.add_argument(
        "--analysis", type=str, default="1,2,3",
        help="Comma-separated analyses to run (default: 1,2,3)",
    )
    args = parser.parse_args()

    analyses = set(args.analysis.split(","))

    print("Extracting track features from video frames (one-time)...")
    all_data = collect_data(args.video_id)

    if not all_data:
        print("No data collected")
        sys.exit(1)

    print(f"\nCollected data for {len(all_data)} videos")

    if "1" in analyses:
        analysis_1(all_data)
    if "2" in analyses:
        analysis_2(all_data)
    if "3" in analyses:
        analysis_3(all_data)


if __name__ == "__main__":
    main()
