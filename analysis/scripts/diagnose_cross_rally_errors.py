#!/usr/bin/env python3
"""Session 7a — per-video error decomposition for cross-rally matching.

Read-only diagnostic. Re-runs the OSNet baseline (same path as
``eval_dinov2_clean.py``) on the 52-video clean GT pool and decomposes
each error into one of three buckets:

  * cross-team    — predicted team != GT team (likely missed side switch)
  * within-team   — correct team, wrong teammate (appearance ceiling)
  * track-quality — GT track has no prediction or two GT tracks were
                    merged onto one predicted track (upstream tracking)

Outputs:
  * Per-video table
  * Aggregate table + dominant-slice header
  * JSON dump under analysis/outputs/cross_rally_errors/run_<ts>.json
  * Hard-video cross-reference (ce4c67a1, b03b461b, a7ee3d38, cb3b68f0)

This script does NOT change production code, tune anything, or run
experiments. It exists to gate Session 7b/7c.

Usage:
    cd analysis
    uv run python scripts/diagnose_cross_rally_errors.py
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

HARD_VIDEOS = ("ce4c67a1", "b03b461b", "a7ee3d38", "cb3b68f0")


def _team(pid: int) -> int:
    return 0 if pid <= 2 else 1


def _classify_error(gt_pid: int, mapped_pid: int | None) -> str:
    """Classify a single (gt, mapped) pair. Mirrors diagnose_cross_rally_comprehensive."""
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
    """Same global permutation search as eval_dinov2_clean.find_best_permutation,
    but returns the winning mapping so we can re-classify each error."""
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


def run_match(video_path: Any, rallies: list[Any], reid_model: Any) -> dict[str, dict[str, int]]:
    from rallycut.tracking.match_tracker import match_players_across_rallies

    result = match_players_across_rallies(
        video_path=video_path,
        rallies=rallies,
        reid_model=reid_model,
    )
    return {
        rallies[i].rally_id: {str(k): v for k, v in r.track_to_player.items()}
        for i, r in enumerate(result.rally_results)
    }


def decompose_video(
    gt_rallies: dict[str, dict[str, int]],
    pred_rallies: dict[str, dict[str, int]],
) -> dict[str, Any]:
    """Decompose one video's errors into the four buckets."""
    perm, _, _ = _find_best_permutation(gt_rallies, pred_rallies)

    counts = Counter({"correct": 0, "cross-team": 0, "within-team": 0, "track-quality": 0})
    total = 0

    for rid, gt_map in gt_rallies.items():
        pred_map = pred_rallies.get(rid, {})

        # Detect merged tracks: two GT tracks mapping to the same predicted pid.
        mapped_pids: list[tuple[str, int | None]] = []
        for tid in gt_map:
            raw = pred_map.get(tid)
            mapped = perm.get(raw) if raw is not None else None
            mapped_pids.append((tid, mapped))

        # Count how many GT tracks share each non-None mapped pid.
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
            # Merged-track case: more than one GT track mapped to the same pid.
            # We blame track-quality on the loser(s); the GT track that
            # actually equals the mapped pid is "correct".
            siblings = pid_to_tids[mapped]
            if len(siblings) > 1 and mapped != gt_pid:
                counts["track-quality"] += 1
                continue
            counts[_classify_error(gt_pid, mapped)] += 1

    return {
        "total": total,
        "correct": counts["correct"],
        "cross_team": counts["cross-team"],
        "within_team": counts["within-team"],
        "track_quality": counts["track-quality"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", type=str, default=None,
                        help="Optional video id prefix filter (debug only)")
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
        logger.error("OSNet weights not found at %s — Session 7a needs the "
                     "production OSNet baseline.", REID_WEIGHTS_PATH)
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
        if not gt_rallies:
            logger.warning("[%d/%d %s] empty GT, skipping", idx, len(gt_rows), short)
            continue
        rallies = load_rallies_for_video(video_id)
        video_path = get_video_path(video_id)
        if not rallies or not video_path:
            logger.warning("[%d/%d %s] cannot load rallies/video, skipping",
                           idx, len(gt_rows), short)
            continue

        n_labels = sum(len(v) for v in gt_rallies.values())
        logger.info("[%d/%d %s] %d rallies, %d GT labels — running OSNet...",
                    idx, len(gt_rows), short, len(rallies), n_labels)

        try:
            pred = run_match(video_path, rallies, reid_model=reid_model)
        except Exception as exc:  # noqa: BLE001 — diagnostic, surface and continue
            logger.warning("[%d/%d %s] match failed: %s", idx, len(gt_rows), short, exc)
            continue

        d = decompose_video(gt_rallies, pred)
        d["video_id"] = video_id
        d["short"] = short
        d["accuracy"] = d["correct"] / d["total"] * 100 if d["total"] else 0.0
        per_video.append(d)
        logger.info(
            "    total=%d  correct=%d (%.1f%%)  cross=%d  within=%d  track=%d",
            d["total"], d["correct"], d["accuracy"],
            d["cross_team"], d["within_team"], d["track_quality"],
        )

    elapsed = time.time() - started
    logger.info("\nFinished %d videos in %.1fs\n", len(per_video), elapsed)

    if not per_video:
        logger.error("No per-video results — aborting.")
        sys.exit(1)

    # ---- Per-video table ----
    logger.info("=" * 84)
    logger.info("PER-VIDEO ERROR DECOMPOSITION")
    logger.info("=" * 84)
    logger.info("%-10s %6s %6s %6s %6s %6s %8s",
                "Video", "N", "OK", "Cross", "Within", "Track", "Acc")
    logger.info("-" * 84)
    for v in per_video:
        logger.info(
            "%-10s %6d %6d %6d %6d %6d %7.1f%%",
            v["short"], v["total"], v["correct"],
            v["cross_team"], v["within_team"], v["track_quality"],
            v["accuracy"],
        )

    # ---- Aggregate ----
    agg_total = sum(v["total"] for v in per_video)
    agg_correct = sum(v["correct"] for v in per_video)
    agg_cross = sum(v["cross_team"] for v in per_video)
    agg_within = sum(v["within_team"] for v in per_video)
    agg_track = sum(v["track_quality"] for v in per_video)
    agg_errors = agg_cross + agg_within + agg_track

    logger.info("-" * 84)
    logger.info("AGGREGATE (%d videos)", len(per_video))
    logger.info("  total assignments : %d", agg_total)
    logger.info("  correct           : %d (%.1f%%)",
                agg_correct, agg_correct / agg_total * 100 if agg_total else 0)
    logger.info("  errors            : %d", agg_errors)
    if agg_errors > 0:
        cp = agg_cross / agg_errors * 100
        wp = agg_within / agg_errors * 100
        tp = agg_track / agg_errors * 100
        logger.info("    cross-team    : %d  (%.1f%% of errors)", agg_cross, cp)
        logger.info("    within-team   : %d  (%.1f%% of errors)", agg_within, wp)
        logger.info("    track-quality : %d  (%.1f%% of errors)", agg_track, tp)
    else:
        cp = wp = tp = 0.0

    # ---- Dominant-slice header ----
    logger.info("\n" + "=" * 84)
    if cp >= 50:
        verdict = f"cross-team dominates: {cp:.0f}%"
        next_step = "→ run Session 7b-cross-team (side switch attack)"
    elif wp >= 50:
        verdict = f"within-team dominates: {wp:.0f}%"
        next_step = "→ run Session 7b-within-team (likely defensive)"
    elif tp >= 30:
        verdict = f"track-quality non-trivial: {tp:.0f}%"
        next_step = "→ run Session 7c first (fixing tracks changes the decomposition)"
    else:
        verdict = f"roughly balanced: C/W/T = {cp:.0f}/{wp:.0f}/{tp:.0f}%"
        next_step = "→ default to 7b-cross-team (highest leverage untried lever)"
    logger.info("VERDICT: %s", verdict)
    logger.info("        %s", next_step)
    logger.info("=" * 84)

    # ---- Hard videos cross-reference ----
    logger.info("\nHard-video cross-reference:")
    for hv in HARD_VIDEOS:
        match = next((v for v in per_video if v["short"] == hv), None)
        if match is None:
            logger.info("  %s : not in pool", hv)
            continue
        errs = match["cross_team"] + match["within_team"] + match["track_quality"]
        dominant = "—"
        if errs > 0:
            slice_counts = {
                "cross": match["cross_team"],
                "within": match["within_team"],
                "track": match["track_quality"],
            }
            dominant = max(slice_counts, key=lambda k: slice_counts[k])
        logger.info(
            "  %s  acc=%.1f%%  cross=%d within=%d track=%d  → dominant=%s",
            hv, match["accuracy"], match["cross_team"],
            match["within_team"], match["track_quality"], dominant,
        )

    # ---- JSON dump ----
    out_dir = Path("outputs/cross_rally_errors")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d-%H%M%S")
    out_path = out_dir / f"run_{ts}.json"
    with out_path.open("w") as fh:
        json.dump(
            {
                "n_videos": len(per_video),
                "aggregate": {
                    "total": agg_total,
                    "correct": agg_correct,
                    "accuracy_pct": agg_correct / agg_total * 100 if agg_total else 0,
                    "cross_team": agg_cross,
                    "within_team": agg_within,
                    "track_quality": agg_track,
                    "cross_pct_of_errors": cp,
                    "within_pct_of_errors": wp,
                    "track_pct_of_errors": tp,
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
