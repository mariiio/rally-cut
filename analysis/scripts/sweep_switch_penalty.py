#!/usr/bin/env python3
"""Sweep switch_penalty for side switch detection.

Extracts appearances ONCE per video (~8 min total), then replays the
assignment + side-switch logic for each penalty value (~1s each).

Usage:
    uv run python scripts/sweep_switch_penalty.py
"""

from __future__ import annotations

import itertools
import logging
import sys
from pathlib import Path
from typing import Any, cast

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.WARNING, format="%(message)s")

from rallycut.evaluation.db import get_connection
from rallycut.evaluation.tracking.db import get_video_path, load_rallies_for_video
from rallycut.tracking.match_tracker import (
    MatchPlayerTracker,
    MatchPlayersResult,
    RallyTrackingResult,
    extract_rally_appearances,
)
import rallycut.tracking.match_tracker as mt_module


def find_best_permutation(
    gt_rallies: dict[str, dict[str, int]],
    pred_rallies: dict[str, dict[str, int]],
) -> tuple[dict[int, int], int, int]:
    player_ids = [1, 2, 3, 4]
    best_perm: dict[int, int] = {}
    best_correct = -1
    best_total = 0
    for perm in itertools.permutations(player_ids):
        pred_to_gt = dict(zip(player_ids, perm))
        correct = total = 0
        for rid in gt_rallies:
            if rid not in pred_rallies:
                continue
            for tid_str in gt_rallies[rid]:
                if tid_str not in pred_rallies[rid]:
                    continue
                total += 1
                if pred_to_gt.get(pred_rallies[rid][tid_str]) == gt_rallies[rid][tid_str]:
                    correct += 1
        if correct > best_correct:
            best_correct = correct
            best_total = total
            best_perm = pred_to_gt
    return best_perm, best_correct, best_total


def evaluate_switches(
    gt_switches: list[int],
    pred_list: list[dict[str, Any]],
) -> tuple[int, int, int]:
    pred = {i for i, e in enumerate(pred_list) if e.get("sideSwitchDetected", False)}
    gt = set(gt_switches)
    return len(gt & pred), len(pred - gt), len(gt - pred)


def load_gt_entries() -> list[tuple[str, dict[str, dict[str, int]], list[int]]]:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, player_matching_gt_json FROM videos "
                "WHERE player_matching_gt_json IS NOT NULL"
            )
            rows = cur.fetchall()
    entries = []
    for vid, gt_json in rows:
        gt_data = cast(dict[str, Any], gt_json)
        gt_rallies = {
            rid: {str(k): int(v) for k, v in mapping.items()}
            for rid, mapping in gt_data.get("rallies", {}).items()
        }
        gt_switches = gt_data.get("sideSwitches", gt_data.get("side_switches", []))
        entries.append((str(vid), gt_rallies, gt_switches))
    return entries


# Cached data per video: rally appearances + rally metadata
CachedVideo = list[dict[str, Any]]  # list of per-rally dicts


def extract_video_cache(video_id: str) -> CachedVideo | None:
    """Extract and cache all appearance data for a video's rallies."""
    rallies = load_rallies_for_video(video_id)
    video_path = get_video_path(video_id)
    if not rallies or not video_path:
        return None

    cached: CachedVideo = []
    for rally in rallies:
        track_stats = extract_rally_appearances(
            video_path=video_path,
            positions=rally.positions,
            primary_track_ids=rally.primary_track_ids,
            start_ms=rally.start_ms,
            end_ms=rally.end_ms,
            num_samples=12,
        )
        cached.append({
            "rally_id": rally.rally_id,
            "track_stats": track_stats,
            "positions": rally.positions,
            "ball_positions": rally.ball_positions,
            "court_split_y": rally.court_split_y,
            "team_assignments": rally.team_assignments,
        })
    return cached


def run_matching_with_cache(
    cache: CachedVideo,
    penalty: float | None = None,
) -> list[dict[str, Any]]:
    """Run matching pipeline using cached appearances.

    If penalty is set, overrides switch_penalty via module global.
    """
    if penalty is not None:
        mt_module._SWITCH_PENALTY_OVERRIDE = penalty  # type: ignore[attr-defined]

    tracker = MatchPlayerTracker()
    results: list[RallyTrackingResult] = []

    for entry in cache:
        result = tracker.process_rally(
            track_stats=entry["track_stats"],
            player_positions=entry["positions"],
            ball_positions=entry["ball_positions"],
            court_split_y=entry["court_split_y"],
            team_assignments=entry["team_assignments"],
        )
        results.append(result)

    results = tracker.refine_assignments(results)

    if penalty is not None:
        mt_module._SWITCH_PENALTY_OVERRIDE = None  # type: ignore[attr-defined]

    pred_list: list[dict[str, Any]] = []
    pred_rallies: dict[str, dict[str, int]] = {}
    for entry, rr in zip(cache, results):
        ttp = {str(k): v for k, v in rr.track_to_player.items()}
        pred_rallies[entry["rally_id"]] = ttp
        pred_list.append({
            "rallyId": entry["rally_id"],
            "trackToPlayer": ttp,
            "sideSwitchDetected": rr.side_switch_detected,
        })

    return pred_list


def main() -> None:
    print("Loading GT entries from DB...", flush=True)
    gt_entries = load_gt_entries()
    print(f"Found {len(gt_entries)} videos with GT\n", flush=True)

    # Phase 1: Extract appearances once
    print("Phase 1: Extracting appearances (one-time, ~8 min)...", flush=True)
    video_caches: dict[str, CachedVideo] = {}

    for i, (video_id, _, _) in enumerate(gt_entries):
        try:
            cache = extract_video_cache(video_id)
        except Exception as e:
            print(f"  [{i+1}/{len(gt_entries)}] {video_id[:8]} ERROR: {e}", flush=True)
            continue
        if cache is None:
            print(f"  [{i+1}/{len(gt_entries)}] {video_id[:8]} SKIP", flush=True)
            continue
        video_caches[video_id] = cache
        print(f"  [{i+1}/{len(gt_entries)}] {video_id[:8]} ({len(cache)} rallies)", flush=True)

    print(f"\nCached {len(video_caches)} videos\n", flush=True)

    # Phase 2: Sweep penalties (fast — no video I/O)
    print("Phase 2: Sweeping penalties...", flush=True)

    penalties = [0.8, 1.0, 1.2, 1.3, 1.4, 1.5, 1.8, 2.0, 2.5]
    results: list[dict[str, Any]] = []

    for pi, penalty in enumerate(penalties):
        total_correct = total_total = 0
        total_sw_tp = total_sw_fp = total_sw_fn = 0
        per_video: list[dict[str, Any]] = []

        for video_id, gt_rallies, gt_switches in gt_entries:
            if video_id not in video_caches:
                continue

            pred_list = run_matching_with_cache(video_caches[video_id], penalty)
            pred_rallies = {
                e["rallyId"]: e["trackToPlayer"] for e in pred_list
            }

            best_perm, correct, total = find_best_permutation(gt_rallies, pred_rallies)
            sw_tp, sw_fp, sw_fn = evaluate_switches(gt_switches, pred_list)

            total_correct += correct
            total_total += total
            total_sw_tp += sw_tp
            total_sw_fp += sw_fp
            total_sw_fn += sw_fn

            if total > 0:
                per_video.append({
                    "video_id": video_id[:8],
                    "accuracy": correct / total * 100,
                })

        acc = total_correct / total_total * 100 if total_total > 0 else 0
        sw_p = total_sw_tp / (total_sw_tp + total_sw_fp) if (total_sw_tp + total_sw_fp) else 0
        sw_r = total_sw_tp / (total_sw_tp + total_sw_fn) if (total_sw_tp + total_sw_fn) else 0
        sw_f1 = 2 * sw_p * sw_r / (sw_p + sw_r) if (sw_p + sw_r) else 0

        r = {
            "penalty": penalty, "accuracy": acc,
            "correct": total_correct, "total": total_total,
            "sw_tp": total_sw_tp, "sw_fp": total_sw_fp, "sw_fn": total_sw_fn,
            "sw_f1": sw_f1 * 100, "per_video": per_video,
        }
        results.append(r)
        print(
            f"  [{pi+1}/{len(penalties)}] penalty={penalty:.1f}: "
            f"Acc={acc:.1f}% ({total_correct}/{total_total})  "
            f"Switch: TP={total_sw_tp} FP={total_sw_fp} FN={total_sw_fn} F1={r['sw_f1']:.1f}%",
            flush=True,
        )

    # Summary
    print(f"\n{'='*80}", flush=True)
    print(f"SWITCH PENALTY SWEEP ({len(video_caches)} videos)", flush=True)
    print(f"{'='*80}", flush=True)
    print(f"{'Penalty':>8s} {'Accuracy':>9s} {'Correct':>8s} "
          f"{'SW_TP':>6s} {'SW_FP':>6s} {'SW_FN':>6s} {'SW_F1':>7s}", flush=True)
    print("-" * 60, flush=True)

    best_acc = max(r["accuracy"] for r in results)
    best_f1 = max(r["sw_f1"] for r in results)

    for r in results:
        marks = ""
        if r["accuracy"] == best_acc:
            marks += " [best acc]"
        if r["sw_f1"] == best_f1:
            marks += " [best F1]"
        print(
            f"{r['penalty']:>8.1f} {r['accuracy']:>8.1f}% {r['correct']:>8d} "
            f"{r['sw_tp']:>6d} {r['sw_fp']:>6d} {r['sw_fn']:>6d} "
            f"{r['sw_f1']:>6.1f}%{marks}",
            flush=True,
        )

    # Per-video diff: show only videos that change across penalties
    print(f"\n{'='*80}", flush=True)
    print("PER-VIDEO ACCURACY (videos that vary)", flush=True)
    print(f"{'='*80}", flush=True)
    header = f"{'Video':>10s}"
    for r in results:
        header += f" {r['penalty']:>5.1f}"
    print(header, flush=True)

    all_vids = sorted({v["video_id"] for r in results for v in r["per_video"]})
    for vid in all_vids:
        accs = []
        row = f"{vid:>10s}"
        for r in results:
            v_data = next((v for v in r["per_video"] if v["video_id"] == vid), None)
            if v_data:
                accs.append(v_data["accuracy"])
                row += f" {v_data['accuracy']:>4.1f}%"
            else:
                row += "   N/A"
        # Only show if accuracy varies
        if accs and max(accs) - min(accs) > 0.1:
            print(row, flush=True)


if __name__ == "__main__":
    main()
