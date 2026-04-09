#!/usr/bin/env python3
"""Evaluate DINOv2 impact on cross-rally matching with clean bbox-keyed GT.

Runs three modes per video:
  A. HSV-only — no ReID at all
  B. OSNet — general ReID model
  C. DINOv2 — user reference crops (if available)

Uses rallycut.evaluation.gt_loader.load_all_from_db to resolve bbox-keyed
GT labels to current track ids via IoU matching.

Usage:
    uv run python scripts/eval_dinov2_clean.py
    uv run python scripts/eval_dinov2_clean.py --video-id 84e66e74
    uv run python scripts/eval_dinov2_clean.py --ref-crops-only
"""

from __future__ import annotations

import argparse
import itertools
import logging
import sys
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def find_best_permutation(
    gt_rallies: dict[str, dict[str, int]],
    pred_rallies: dict[str, dict[str, int]],
) -> tuple[int, int]:
    """Return (best_correct, total) over all ID permutations."""
    best_c = 0
    best_t = 0
    for perm in itertools.permutations([1, 2, 3, 4]):
        pm = {i + 1: p for i, p in enumerate(perm)}
        c = 0
        t = 0
        for rid in gt_rallies:
            if rid not in pred_rallies:
                continue
            for tid in gt_rallies[rid]:
                if tid in pred_rallies[rid]:
                    t += 1
                    if pm.get(pred_rallies[rid][tid]) == gt_rallies[rid][tid]:
                        c += 1
        if c > best_c:
            best_c = c
            best_t = t
    return best_c, best_t


def run_match(
    video_path: Any,
    rallies: list[Any],
    *,
    reid_model: Any = None,
    reference_profiles: Any = None,
) -> dict[str, dict[str, int]]:
    from rallycut.tracking.match_tracker import match_players_across_rallies

    result = match_players_across_rallies(
        video_path=video_path,
        rallies=rallies,
        reid_model=reid_model,
        reference_profiles=reference_profiles,
    )
    return {
        rallies[i].rally_id: {str(k): v for k, v in r.track_to_player.items()}
        for i, r in enumerate(result.rally_results)
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate DINOv2 on cross-rally matching with clean GT"
    )
    parser.add_argument("--video-id", type=str, default=None,
                        help="Evaluate only this video ID (prefix)")
    parser.add_argument("--ref-crops-only", action="store_true",
                        help="Only evaluate videos that have user reference crops")
    args = parser.parse_args()

    from rallycut.evaluation.db import get_connection
    from rallycut.evaluation.gt_loader import load_all_from_db
    from rallycut.evaluation.tracking.db import (
        get_video_path,
        load_rallies_for_video,
    )
    from rallycut.tracking.reid_general import (
        WEIGHTS_PATH as REID_WEIGHTS_PATH,
    )
    from rallycut.tracking.reid_general import (
        GeneralReIDModel,
    )

    # Load general ReID model (once)
    reid_model = None
    if REID_WEIGHTS_PATH.exists():
        reid_model = GeneralReIDModel(weights_path=REID_WEIGHTS_PATH)
        logger.info("Loaded general ReID (OSNet SupCon)")

    # Load all GT via the canonical loader
    with get_connection() as conn:
        with conn.cursor() as cur:
            gt_rows = load_all_from_db(cur, video_id_prefix=args.video_id)
            # Also find videos with ref crops
            cur.execute(
                "SELECT DISTINCT video_id FROM player_reference_crops"
            )
            ref_crop_videos = {str(r[0]) for r in cur.fetchall()}

    if not gt_rows:
        logger.error("No GT videos found.")
        sys.exit(1)

    logger.info("Found %d GT videos, %d with ref crops\n",
                len(gt_rows), len(ref_crop_videos))

    totals = {
        "hsv": [0, 0],    # [correct, total]
        "osnet": [0, 0],
        "dinov2": [0, 0],
    }
    per_video: list[dict[str, Any]] = []

    for row in gt_rows:
        video_id = row.video_id
        short = video_id[:8]
        has_ref = video_id in ref_crop_videos

        if args.ref_crops_only and not has_ref:
            continue

        gt_rallies = row.gt.rallies
        if not gt_rallies:
            logger.warning("[%s] empty GT (all labels dropped or no rallies)",
                           short)
            continue

        rallies = load_rallies_for_video(video_id)
        video_path = get_video_path(video_id)
        if not rallies or not video_path:
            logger.warning("[%s] cannot load rallies/video, skipping", short)
            continue

        n_labels = sum(len(v) for v in gt_rallies.values())
        logger.info("[%s] %d rallies, %d GT labels%s",
                    short, len(rallies), n_labels,
                    " (has ref crops)" if has_ref else "")

        video_result: dict[str, Any] = {
            "id": short, "n_rallies": len(rallies), "has_ref": has_ref,
        }

        # A: HSV-only
        logger.info("  [A] HSV-only...")
        pred = run_match(video_path, rallies)
        c, t = find_best_permutation(gt_rallies, pred)
        totals["hsv"][0] += c
        totals["hsv"][1] += t
        video_result["hsv"] = (c, t)
        logger.info("      %d/%d = %.1f%%",
                    c, t, c / t * 100 if t else 0)

        # B: OSNet
        if reid_model is not None:
            logger.info("  [B] OSNet...")
            pred = run_match(video_path, rallies, reid_model=reid_model)
            c, t = find_best_permutation(gt_rallies, pred)
            totals["osnet"][0] += c
            totals["osnet"][1] += t
            video_result["osnet"] = (c, t)
            logger.info("      %d/%d = %.1f%%",
                        c, t, c / t * 100 if t else 0)

        # C: DINOv2 (user reference crops)
        if has_ref:
            logger.info("  [C] DINOv2 reference crops...")
            from rallycut.cli.commands.match_players import (
                _load_db_reference_crops,
            )
            _, ref_profiles = _load_db_reference_crops(
                video_id, video_path, quiet=True,
            )
            if ref_profiles:
                pred = run_match(
                    video_path, rallies, reference_profiles=ref_profiles,
                )
                c, t = find_best_permutation(gt_rallies, pred)
                totals["dinov2"][0] += c
                totals["dinov2"][1] += t
                video_result["dinov2"] = (c, t)
                logger.info("      %d/%d = %.1f%%",
                            c, t, c / t * 100 if t else 0)

        per_video.append(video_result)
        logger.info("")

    # Summary
    logger.info("=" * 78)
    logger.info("SUMMARY")
    logger.info("=" * 78)

    header = f"{'Video':10} {'Rlys':>5} {'HSV':>8} {'OSNet':>8} {'DINOv2':>8}"
    logger.info(header)
    logger.info("-" * 78)
    for v in per_video:
        cols = [v["id"], f"{v['n_rallies']:>5}"]
        for mode in ("hsv", "osnet", "dinov2"):
            if mode in v:
                c, t = v[mode]
                acc = c / t * 100 if t else 0
                cols.append(f"{acc:>7.1f}%")
            else:
                cols.append("      —")
        logger.info("%-10s %s %s %s %s", *cols)

    logger.info("-" * 78)
    logger.info("AGGREGATE")
    for mode in ("hsv", "osnet", "dinov2"):
        c, t = totals[mode]
        if t > 0:
            logger.info("  %-7s: %d/%d = %.1f%%", mode, c, t, c / t * 100)

    if totals["osnet"][1] > 0 and totals["hsv"][1] > 0:
        d = totals["osnet"][0] / totals["osnet"][1] * 100 - \
            totals["hsv"][0] / totals["hsv"][1] * 100
        logger.info("  Delta HSV→OSNet:   %+.1fpp", d)
    if totals["dinov2"][1] > 0:
        # Compare DINOv2 vs OSNet (and HSV) on the same videos
        dinov2_videos = [v for v in per_video if "dinov2" in v]
        for baseline in ("hsv", "osnet"):
            bc = sum(v[baseline][0] for v in dinov2_videos if baseline in v)
            bt = sum(v[baseline][1] for v in dinov2_videos if baseline in v)
            dc = sum(v["dinov2"][0] for v in dinov2_videos)
            dt = sum(v["dinov2"][1] for v in dinov2_videos)
            if bt > 0 and dt > 0:
                d = dc / dt * 100 - bc / bt * 100
                logger.info(
                    "  Delta %s→DINOv2 (%d ref-crop videos): %+.1fpp",
                    baseline, len(dinov2_videos), d,
                )


if __name__ == "__main__":
    main()
