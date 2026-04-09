#!/usr/bin/env python3
"""Session 7b-cross-team pre-work — switch-locality of cross-team errors.

7a verdict: cross-team errors dominate (65% of all errors). All three
candidate 7b approaches (serve-order CRF, score-counting switch detection,
endpoint IoU) attack the *side-switch* failure mode. They only help if
cross-team errors actually cluster near side-switch boundaries.

This script measures that. For each cross-team error in the 7a
decomposition, classify the host rally as one of:

  * SWITCH_REGION — within ±2 rallies of a missed GT switch
                     OR within ±2 of a hallucinated predicted switch
  * STRUCTURAL    — far from any switch boundary; the error is a
                     wrong-from-rally-0 (or persistent) inversion that no
                     switch-detection lever can fix

Go/no-go (per playbook 40% rule):
  * SWITCH_REGION ≥ 40% of cross-team errors → Session 7b-cross-team
    is the right session; pick the cheapest approach (endpoint IoU first)
  * SWITCH_REGION < 40%                       → cross-team errors are
    structural; build a different lever (e.g. serve-anchored team
    assignment at rally 0, or oracle-team-bootstrap from reference crops)

Re-runs OSNet baseline (~30-40min on 51 videos). Read-only.

Usage:
    cd analysis
    uv run python scripts/diagnose_cross_team_switch_locality.py
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

SWITCH_WINDOW = 2  # ±N rallies counts as "near" a switch boundary


def _team(pid: int) -> int:
    return 0 if pid <= 2 else 1


def _classify_error(gt_pid: int, mapped_pid: int | None) -> str:
    if mapped_pid is None:
        return "track-quality"
    if mapped_pid == gt_pid:
        return "correct"
    if _team(mapped_pid) == _team(gt_pid):
        return "within-team"
    return "cross-team"


def _find_best_permutation(
    gt_rallies: dict[str, dict[str, int]],
    pred_rallies: dict[str, dict[str, int]],
) -> tuple[dict[int, int], int, int]:
    player_ids = [1, 2, 3, 4]
    best_perm: dict[int, int] = {pid: pid for pid in player_ids}
    best_c = -1
    best_t = 0
    for perm in itertools.permutations(player_ids):
        pm = {pid: gt for pid, gt in zip(player_ids, perm)}
        c = 0
        t = 0
        for rid, gt_map in gt_rallies.items():
            if rid not in pred_rallies:
                continue
            pred_map = pred_rallies[rid]
            for tid in gt_map:
                if tid in pred_map:
                    t += 1
                    if pm.get(pred_map[tid]) == gt_map[tid]:
                        c += 1
        if c > best_c:
            best_c = c
            best_t = t
            best_perm = pm
    return best_perm, max(best_c, 0), best_t


def run_match_with_switches(
    video_path: Any,
    rallies: list[Any],
    reid_model: Any,
) -> tuple[dict[str, dict[str, int]], list[int]]:
    """Run OSNet baseline; return (predictions, predicted_switch_indices)."""
    from rallycut.tracking.match_tracker import match_players_across_rallies

    result = match_players_across_rallies(
        video_path=video_path,
        rallies=rallies,
        reid_model=reid_model,
    )
    preds = {
        rallies[i].rally_id: {str(k): v for k, v in r.track_to_player.items()}
        for i, r in enumerate(result.rally_results)
    }
    pred_switches = [
        i for i, r in enumerate(result.rally_results) if r.side_switch_detected
    ]
    return preds, pred_switches


def classify_video(
    rally_id_order: list[str],
    gt_rallies: dict[str, dict[str, int]],
    pred_rallies: dict[str, dict[str, int]],
    gt_switches: list[int],
    pred_switches: list[int],
) -> dict[str, Any]:
    """Per-video error decomposition + switch-locality classification."""
    perm, _, _ = _find_best_permutation(gt_rallies, pred_rallies)

    # Symmetric difference of switches: missed GT switches + hallucinated preds.
    gt_set = set(gt_switches)
    pred_set = set(pred_switches)
    suspect_indices = (gt_set - pred_set) | (pred_set - gt_set)
    # Build a fast "near a suspect switch" lookup.
    near_suspect = set()
    for s in suspect_indices:
        for d in range(-SWITCH_WINDOW, SWITCH_WINDOW + 1):
            near_suspect.add(s + d)

    counts = Counter(
        {
            "correct": 0,
            "cross-team": 0,
            "within-team": 0,
            "track-quality": 0,
        }
    )
    cross_team_switch_region = 0
    cross_team_structural = 0
    cross_team_rally_indices: list[int] = []
    total = 0

    for rally_idx, rid in enumerate(rally_id_order):
        gt_map = gt_rallies.get(rid, {})
        if not gt_map:
            continue
        pred_map = pred_rallies.get(rid, {})

        # Merged-track detection (same as 7a)
        mapped_pids: list[tuple[str, int | None]] = []
        for tid in gt_map:
            raw = pred_map.get(tid)
            mapped = perm.get(raw) if raw is not None else None
            mapped_pids.append((tid, mapped))
        pid_to_tids: dict[int, list[str]] = {}
        for tid, mapped in mapped_pids:
            if mapped is not None:
                pid_to_tids.setdefault(mapped, []).append(tid)

        for tid, mapped in mapped_pids:
            total += 1
            gt_pid = gt_map[tid]
            if mapped is None:
                counts["track-quality"] += 1
                continue
            siblings = pid_to_tids[mapped]
            if len(siblings) > 1 and mapped != gt_pid:
                counts["track-quality"] += 1
                continue
            kind = _classify_error(gt_pid, mapped)
            counts[kind] += 1
            if kind == "cross-team":
                cross_team_rally_indices.append(rally_idx)
                if rally_idx in near_suspect:
                    cross_team_switch_region += 1
                else:
                    cross_team_structural += 1

    return {
        "total": total,
        "correct": counts["correct"],
        "cross_team": counts["cross-team"],
        "within_team": counts["within-team"],
        "track_quality": counts["track-quality"],
        "cross_team_switch_region": cross_team_switch_region,
        "cross_team_structural": cross_team_structural,
        "cross_team_rally_indices": sorted(set(cross_team_rally_indices)),
        "gt_switches": sorted(gt_set),
        "pred_switches": sorted(pred_set),
        "missed_switches": sorted(gt_set - pred_set),
        "false_switches": sorted(pred_set - gt_set),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", type=str, default=None)
    args = parser.parse_args()

    from rallycut.evaluation.db import get_connection
    from rallycut.evaluation.gt_loader import load_all_from_db
    from rallycut.evaluation.tracking.db import get_video_path, load_rallies_for_video
    from rallycut.tracking.reid_general import (
        WEIGHTS_PATH as REID_WEIGHTS_PATH,
    )
    from rallycut.tracking.reid_general import (
        GeneralReIDModel,
    )

    if not REID_WEIGHTS_PATH.exists():
        logger.error("OSNet weights not found at %s", REID_WEIGHTS_PATH)
        sys.exit(1)
    reid_model = GeneralReIDModel(weights_path=REID_WEIGHTS_PATH)
    logger.info("Loaded general ReID (OSNet SupCon)")

    with get_connection() as conn:
        with conn.cursor() as cur:
            gt_rows = load_all_from_db(cur, video_id_prefix=args.video_id)

    if not gt_rows:
        logger.error("No GT videos found.")
        sys.exit(1)

    logger.info("Found %d GT videos\n", len(gt_rows))

    per_video: list[dict[str, Any]] = []
    started = time.time()

    for idx, row in enumerate(gt_rows, start=1):
        video_id = row.video_id
        short = video_id[:8]
        gt_rallies = row.gt.rallies
        gt_switches = list(row.gt.side_switches or [])
        if not gt_rallies:
            logger.warning("[%d/%d %s] empty GT, skipping", idx, len(gt_rows), short)
            continue
        rallies = load_rallies_for_video(video_id)
        video_path = get_video_path(video_id)
        if not rallies or not video_path:
            logger.warning("[%d/%d %s] cannot load rallies/video", idx, len(gt_rows), short)
            continue

        rally_id_order = [r.rally_id for r in rallies]
        n_labels = sum(len(v) for v in gt_rallies.values())
        logger.info(
            "[%d/%d %s] %d rallies, %d GT labels, gt_switches=%s — running OSNet...",
            idx, len(gt_rows), short, len(rallies), n_labels, gt_switches,
        )

        try:
            preds, pred_switches = run_match_with_switches(
                video_path, rallies, reid_model=reid_model,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%d/%d %s] match failed: %s", idx, len(gt_rows), short, exc)
            continue

        d = classify_video(
            rally_id_order, gt_rallies, preds, gt_switches, pred_switches,
        )
        d["video_id"] = video_id
        d["short"] = short
        d["accuracy"] = d["correct"] / d["total"] * 100 if d["total"] else 0.0
        per_video.append(d)
        logger.info(
            "    cross=%d (switch_region=%d, structural=%d)  within=%d  track=%d  "
            "pred_switches=%s  missed=%s  false=%s",
            d["cross_team"], d["cross_team_switch_region"], d["cross_team_structural"],
            d["within_team"], d["track_quality"],
            d["pred_switches"], d["missed_switches"], d["false_switches"],
        )

    elapsed = time.time() - started
    logger.info("\nFinished %d videos in %.1fs\n", len(per_video), elapsed)

    if not per_video:
        logger.error("No per-video results.")
        sys.exit(1)

    # ---- Per-video table (cross-team focused) ----
    logger.info("=" * 90)
    logger.info("CROSS-TEAM ERROR SWITCH-LOCALITY")
    logger.info("=" * 90)
    logger.info("%-10s %6s %6s %8s %8s  %s",
                "Video", "Cross", "Within", "SwReg", "Struct", "GTsw / PredSw")
    logger.info("-" * 90)
    for v in sorted(per_video, key=lambda x: -x["cross_team"]):
        if v["cross_team"] == 0 and v["within_team"] == 0:
            continue
        logger.info(
            "%-10s %6d %6d %8d %8d  %s / %s",
            v["short"], v["cross_team"], v["within_team"],
            v["cross_team_switch_region"], v["cross_team_structural"],
            v["gt_switches"], v["pred_switches"],
        )

    # ---- Aggregate ----
    agg_cross = sum(v["cross_team"] for v in per_video)
    agg_within = sum(v["within_team"] for v in per_video)
    agg_track = sum(v["track_quality"] for v in per_video)
    agg_sw_region = sum(v["cross_team_switch_region"] for v in per_video)
    agg_structural = sum(v["cross_team_structural"] for v in per_video)
    agg_total = sum(v["total"] for v in per_video)
    agg_correct = sum(v["correct"] for v in per_video)

    logger.info("-" * 90)
    logger.info("AGGREGATE (%d videos)", len(per_video))
    logger.info("  total assignments     : %d", agg_total)
    logger.info("  correct               : %d (%.1f%%)",
                agg_correct, agg_correct / agg_total * 100 if agg_total else 0)
    logger.info("  cross-team errors     : %d", agg_cross)
    if agg_cross > 0:
        sr_pct = agg_sw_region / agg_cross * 100
        st_pct = agg_structural / agg_cross * 100
        logger.info("    switch-region (±%d)  : %d  (%.1f%% of cross-team)",
                    SWITCH_WINDOW, agg_sw_region, sr_pct)
        logger.info("    structural          : %d  (%.1f%% of cross-team)",
                    agg_structural, st_pct)
    else:
        sr_pct = st_pct = 0.0
    logger.info("  within-team errors    : %d", agg_within)
    logger.info("  track-quality errors  : %d", agg_track)

    # ---- Verdict ----
    logger.info("\n" + "=" * 90)
    if sr_pct >= 40:
        verdict = (
            f"SWITCH-REGION dominates: {sr_pct:.0f}% of cross-team errors near switches"
        )
        next_step = (
            "→ Session 7b-cross-team is the right session. Build endpoint-IoU "
            "switch detection first (cheapest); fall back to score-counting, "
            "then serve-order CRF if neither clears the dashboard gate."
        )
    else:
        verdict = (
            f"STRUCTURAL dominates: only {sr_pct:.0f}% of cross-team errors near "
            "switches; {st:.0f}% are from-rally-0 inversions"
        ).format(st=st_pct)
        next_step = (
            "→ Session 7b-cross-team approaches will NOT close the gap. Pivot to "
            "rally-0 anchoring (serve-anchored initial team assignment) or "
            "oracle-team-bootstrap from reference crops."
        )
    logger.info("VERDICT: %s", verdict)
    logger.info("        %s", next_step)
    logger.info("=" * 90)

    # ---- JSON dump ----
    out_dir = Path("outputs/cross_rally_errors")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d-%H%M%S")
    out_path = out_dir / f"switch_locality_{ts}.json"
    with out_path.open("w") as fh:
        json.dump(
            {
                "n_videos": len(per_video),
                "switch_window": SWITCH_WINDOW,
                "aggregate": {
                    "total": agg_total,
                    "correct": agg_correct,
                    "cross_team": agg_cross,
                    "within_team": agg_within,
                    "track_quality": agg_track,
                    "cross_team_switch_region": agg_sw_region,
                    "cross_team_structural": agg_structural,
                    "switch_region_pct": sr_pct,
                    "structural_pct": st_pct,
                },
                "verdict": verdict,
                "per_video": per_video,
            },
            fh,
            indent=2,
        )
    logger.info("\nWrote %s", out_path)


if __name__ == "__main__":
    main()
