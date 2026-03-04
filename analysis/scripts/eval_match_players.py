#!/usr/bin/env python3
"""Evaluate cross-rally player matching against ground truth.

Loads GT JSON files from evaluation/match_gt/, compares with match_analysis_json
from the database. Finds the optimal global permutation of predicted player IDs
to GT player IDs and reports accuracy.

Usage:
    uv run python scripts/eval_match_players.py
    uv run python scripts/eval_match_players.py --gt-dir path/to/gt
    uv run python scripts/eval_match_players.py --video-id <id>
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
from pathlib import Path
from typing import Any, cast

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def find_best_permutation(
    gt_rallies: dict[str, dict[str, int]],
    pred_rallies: dict[str, dict[str, int]],
) -> tuple[dict[int, int], int, int]:
    """Find the optimal permutation mapping predicted player IDs to GT player IDs.

    Brute-force tries all 24 permutations of {1,2,3,4} to find the one
    that maximizes agreement between predicted and GT assignments.

    Args:
        gt_rallies: rally_id -> {track_id_str: gt_player_id}
        pred_rallies: rally_id -> {track_id_str: pred_player_id}

    Returns:
        Tuple of (best_permutation, correct_count, total_count) where
        best_permutation maps predicted_pid -> gt_pid.
    """
    player_ids = [1, 2, 3, 4]
    best_perm: dict[int, int] = {}
    best_correct = -1
    best_total = 0

    for perm in itertools.permutations(player_ids):
        # perm[i] = GT player ID that predicted player (i+1) maps to
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
                gt_pid = gt[tid_str]
                pred_pid = pred[tid_str]
                mapped_pid = pred_to_gt.get(pred_pid)
                if mapped_pid == gt_pid:
                    correct += 1

        if correct > best_correct:
            best_correct = correct
            best_total = total
            best_perm = pred_to_gt

    return best_perm, best_correct, best_total


def evaluate_side_switches(
    gt_switches: list[int],
    pred_rallies_list: list[dict[str, Any]],
) -> tuple[int, int, int]:
    """Evaluate side switch detection.

    Args:
        gt_switches: List of rally indices with side switches (from GT).
        pred_rallies_list: List of rally entries from match_analysis_json.

    Returns:
        Tuple of (TP, FP, FN) for side switch detection.
    """
    pred_switches = set()
    for i, entry in enumerate(pred_rallies_list):
        if entry.get("sideSwitchDetected", entry.get("side_switch_detected", False)):
            pred_switches.add(i)

    gt_set = set(gt_switches)
    tp = len(gt_set & pred_switches)
    fp = len(pred_switches - gt_set)
    fn = len(gt_set - pred_switches)
    return tp, fp, fn


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate cross-rally player matching"
    )
    parser.add_argument(
        "--gt-dir", type=Path,
        default=Path(__file__).parent.parent / "evaluation" / "match_gt",
        help="Directory containing GT JSON files",
    )
    parser.add_argument(
        "--video-id", type=str, default=None,
        help="Evaluate only this video ID",
    )
    parser.add_argument(
        "--from-db", action="store_true",
        help="Read GT from player_matching_gt_json in DB instead of JSON files",
    )
    args = parser.parse_args()

    from rallycut.evaluation.db import get_connection

    # Load GT entries: list of (video_id, gt_rallies, gt_switches, notes)
    gt_entries: list[tuple[str, dict[str, dict[str, int]], list[int], str]] = []

    if args.from_db:
        # Load GT from player_matching_gt_json in DB
        with get_connection() as conn:
            with conn.cursor() as cur:
                query = "SELECT id, player_matching_gt_json FROM videos WHERE player_matching_gt_json IS NOT NULL"
                params: list[str] = []
                if args.video_id:
                    query += " AND id LIKE %s"
                    params.append(f"{args.video_id}%")
                cur.execute(query, params)
                rows = cur.fetchall()

        if not rows:
            logger.error("No videos with player_matching_gt_json found in DB.")
            sys.exit(1)

        for vid, gt_json in rows:
            video_id = str(vid)
            gt_data = cast(dict[str, Any], gt_json)
            gt_rallies_raw = gt_data.get("rallies", {})
            gt_rallies: dict[str, dict[str, int]] = {
                rid: {str(k): int(v) for k, v in mapping.items()}
                for rid, mapping in gt_rallies_raw.items()
            }
            gt_switches: list[int] = gt_data.get(
                "sideSwitches", gt_data.get("side_switches", [])
            )
            gt_entries.append((video_id, gt_rallies, gt_switches, "from-db"))
    else:
        # Load GT from JSON files
        gt_dir: Path = args.gt_dir
        if not gt_dir.exists():
            logger.error("GT directory not found: %s", gt_dir)
            logger.error("Run scripts/label_match_players.py first to create GT.")
            sys.exit(1)

        gt_files = sorted(gt_dir.glob("*.json"))
        if not gt_files:
            logger.error("No GT JSON files found in %s", gt_dir)
            sys.exit(1)

        for gt_file in gt_files:
            with open(gt_file) as f:
                gt_data = json.load(f)

            video_id = gt_data["video_id"]
            if args.video_id and not video_id.startswith(args.video_id):
                continue

            gt_rallies = gt_data.get("rallies", {})
            gt_switches = gt_data.get("side_switches", [])
            notes = gt_data.get("notes", "")
            gt_entries.append((video_id, gt_rallies, gt_switches, notes))

    if not gt_entries:
        logger.error("No GT entries to evaluate.")
        sys.exit(1)

    total_correct = 0
    total_assignments = 0
    total_switch_tp = 0
    total_switch_fp = 0
    total_switch_fn = 0
    video_results: list[dict[str, Any]] = []

    for video_id, gt_rallies, gt_switches, notes in gt_entries:

        # Load match analysis from DB
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT match_analysis_json FROM videos WHERE id = %s",
                    [video_id],
                )
                row = cur.fetchone()
                if not row or not row[0]:
                    logger.warning(
                        "No match_analysis_json for video %s, skipping", video_id[:8]
                    )
                    continue

                match_analysis = cast(dict[str, Any], row[0])

        pred_rallies_list = match_analysis.get("rallies", [])

        # Build pred_rallies dict: rally_id -> {track_id_str: player_id}
        pred_rallies: dict[str, dict[str, int]] = {}
        for entry in pred_rallies_list:
            rid = entry.get("rallyId", entry.get("rally_id", ""))
            ttp = entry.get("trackToPlayer", entry.get("track_to_player", {}))
            if rid and ttp:
                pred_rallies[rid] = {str(k): int(v) for k, v in ttp.items()}

        # Find best permutation
        best_perm, correct, total = find_best_permutation(gt_rallies, pred_rallies)

        if total == 0:
            logger.warning("No overlapping assignments for video %s", video_id[:8])
            continue

        accuracy = correct / total * 100

        # Per-rally breakdown
        rally_details: list[dict[str, Any]] = []
        for rid in gt_rallies:
            if rid not in pred_rallies:
                rally_details.append({
                    "rally_id": rid, "correct": 0, "total": 0, "accuracy": 0,
                    "errors": ["not in predictions"],
                })
                continue

            gt = gt_rallies[rid]
            pred = pred_rallies[rid]
            r_correct = 0
            r_total = 0
            errors: list[str] = []

            for tid_str in gt:
                if tid_str not in pred:
                    continue
                r_total += 1
                gt_pid = gt[tid_str]
                pred_pid = pred[tid_str]
                mapped = best_perm.get(pred_pid)
                if mapped == gt_pid:
                    r_correct += 1
                else:
                    errors.append(
                        f"T{tid_str}: pred P{pred_pid}→P{mapped} vs GT P{gt_pid}"
                    )

            rally_details.append({
                "rally_id": rid,
                "correct": r_correct,
                "total": r_total,
                "accuracy": r_correct / r_total * 100 if r_total > 0 else 0,
                "errors": errors,
            })

        # Side switch eval
        sw_tp, sw_fp, sw_fn = evaluate_side_switches(gt_switches, pred_rallies_list)

        total_correct += correct
        total_assignments += total
        total_switch_tp += sw_tp
        total_switch_fp += sw_fp
        total_switch_fn += sw_fn

        video_results.append({
            "video_id": video_id,
            "notes": notes,
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            "best_permutation": {str(k): v for k, v in best_perm.items()},
            "rally_details": rally_details,
            "side_switches": {"tp": sw_tp, "fp": sw_fp, "fn": sw_fn},
        })

    if not video_results:
        logger.error("No videos evaluated.")
        sys.exit(1)

    # Print results
    logger.info("=" * 70)
    logger.info("Cross-Rally Player Matching Evaluation")
    logger.info("=" * 70)

    for vr in video_results:
        logger.info(
            "\nVideo: %s  (%s)", vr["video_id"][:8], vr["notes"]
        )
        logger.info(
            "  Accuracy: %d/%d = %.1f%%",
            vr["correct"], vr["total"], vr["accuracy"],
        )
        logger.info(
            "  Best permutation: %s",
            ", ".join(
                f"pred P{k}→GT P{v}"
                for k, v in sorted(vr["best_permutation"].items())
            ),
        )

        sw = vr["side_switches"]
        if sw["tp"] + sw["fp"] + sw["fn"] > 0:
            logger.info(
                "  Side switches: TP=%d, FP=%d, FN=%d",
                sw["tp"], sw["fp"], sw["fn"],
            )

        # Per-rally table
        wrong_rallies = [
            rd for rd in vr["rally_details"]
            if rd["accuracy"] < 100 and rd["total"] > 0
        ]
        if wrong_rallies:
            logger.info("  Wrong rallies:")
            for rd in wrong_rallies:
                logger.info(
                    "    %s: %d/%d (%.0f%%) - %s",
                    rd["rally_id"][:8],
                    rd["correct"], rd["total"], rd["accuracy"],
                    "; ".join(rd["errors"]),
                )
        else:
            logger.info("  All rallies correct!")

    # Aggregate
    if len(video_results) > 1:
        logger.info("\n" + "=" * 70)

    agg_acc = total_correct / total_assignments * 100 if total_assignments > 0 else 0
    logger.info(
        "\nAggregate: %d/%d = %.1f%% accuracy",
        total_correct, total_assignments, agg_acc,
    )

    if total_switch_tp + total_switch_fp + total_switch_fn > 0:
        logger.info(
            "Side switches: TP=%d, FP=%d, FN=%d",
            total_switch_tp, total_switch_fp, total_switch_fn,
        )

    logger.info("Videos evaluated: %d", len(video_results))


if __name__ == "__main__":
    main()
