#!/usr/bin/env python3
"""Leave-one-video-out cross-validation for the general ReID model.

For each GT video: train OSNet+SupCon on the other 40 videos' crops,
then run cross-rally matching on the held-out video with the trained
model. Compare accuracy vs HSV-only baseline.

Usage:
    uv run python scripts/eval_reid_loocv.py
    uv run python scripts/eval_reid_loocv.py --video-id abc123   # single fold
    uv run python scripts/eval_reid_loocv.py --epochs 15         # fewer epochs
    uv run python scripts/eval_reid_loocv.py --skip-extract      # reuse cached crops
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import shutil
import sys
import time
from pathlib import Path
from typing import Any, cast

logging.basicConfig(level=logging.WARNING, format="%(message)s")
# Suppress verbose model loading logs during LOO-CV
logging.getLogger("rallycut.tracking.reid_general").setLevel(logging.WARNING)
logging.getLogger("rallycut.tracking.match_tracker").setLevel(logging.WARNING)
logger = logging.getLogger("eval_reid_loocv")
logger.setLevel(logging.INFO)


CROPS_DIR = Path(__file__).parent.parent / "reid_training_data"


def find_best_permutation(
    gt_rallies: dict[str, dict[str, int]],
    pred_rallies: dict[str, dict[str, int]],
) -> tuple[dict[int, int], int, int]:
    """Find optimal permutation mapping pred→GT player IDs."""
    player_ids = [1, 2, 3, 4]
    best_perm: dict[int, int] = {}
    best_correct = -1
    best_total = 0

    for perm in itertools.permutations(player_ids):
        pred_to_gt = dict(zip(player_ids, perm))
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


def extract_all_crops(video_rows: list[tuple[str, Any]]) -> None:
    """Extract training crops for all GT videos."""
    from rallycut.evaluation.db import get_connection
    from rallycut.evaluation.tracking.db import get_video_path
    from scripts.build_reid_training_data import extract_crops_for_video

    logger.info("Extracting crops for %d GT videos...", len(video_rows))

    for i, (video_id, gt_json) in enumerate(video_rows):
        video_dir = CROPS_DIR / video_id[:12]
        if video_dir.exists() and any(video_dir.rglob("*.jpg")):
            continue  # Already extracted

        video_path = get_video_path(video_id)
        if video_path is None:
            continue

        gt_data = cast(dict[str, Any], gt_json)

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT r.id, r.start_ms, pt.positions_json, pt.primary_track_ids
                       FROM rallies r
                       JOIN player_tracks pt ON pt.rally_id = r.id
                       WHERE r.video_id = %s AND pt.positions_json IS NOT NULL
                       ORDER BY r.start_ms""",
                    [video_id],
                )
                rally_rows = cur.fetchall()

        rallies_db = []
        for rid, start_ms, pos_json, ptids in rally_rows:
            ptids_list = json.loads(ptids) if isinstance(ptids, str) else (ptids or [])
            rallies_db.append({
                "rally_id": str(rid),
                "start_ms": start_ms,
                "positions_json": pos_json,
                "primary_track_ids": ptids_list,
            })

        extract_crops_for_video(
            video_id, video_path, gt_data, rallies_db, CROPS_DIR,
        )
        logger.info("  [%d/%d] %s: extracted", i + 1, len(video_rows), video_id[:8])


def run_matching_with_model(
    video_id: str,
    reid_model: Any,
) -> dict[str, dict[str, int]]:
    """Run cross-rally matching with a trained general model."""
    from rallycut.evaluation.tracking.db import get_video_path, load_rallies_for_video
    from rallycut.tracking.match_tracker import match_players_across_rallies

    rallies = load_rallies_for_video(video_id)
    video_path = get_video_path(video_id)
    if not rallies or not video_path:
        return {}

    result = match_players_across_rallies(
        video_path=video_path,
        rallies=rallies,
        reid_model=reid_model,
    )

    pred: dict[str, dict[str, int]] = {}
    for rally, rr in zip(rallies, result.rally_results):
        pred[rally.rally_id] = {str(k): v for k, v in rr.track_to_player.items()}
    return pred


def run_matching_hsv_only(video_id: str) -> dict[str, dict[str, int]]:
    """Run cross-rally matching with HSV only (baseline)."""
    from rallycut.evaluation.tracking.db import get_video_path, load_rallies_for_video
    from rallycut.tracking.match_tracker import match_players_across_rallies

    rallies = load_rallies_for_video(video_id)
    video_path = get_video_path(video_id)
    if not rallies or not video_path:
        return {}

    result = match_players_across_rallies(
        video_path=video_path,
        rallies=rallies,
    )

    pred: dict[str, dict[str, int]] = {}
    for rally, rr in zip(rallies, result.rally_results):
        pred[rally.rally_id] = {str(k): v for k, v in rr.track_to_player.items()}
    return pred


def train_model_excluding(
    exclude_video_prefix: str,
    epochs: int,
) -> Any:
    """Train a GeneralReIDModel on all crops except the held-out video."""
    from rallycut.tracking.reid_general import GeneralReIDModel

    # Create a temporary dataset dir excluding the held-out video
    tmp_dir = CROPS_DIR.parent / "reid_loocv_tmp"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)

    # Symlink all video dirs except the held-out one
    n_linked = 0
    for video_dir in sorted(CROPS_DIR.iterdir()):
        if not video_dir.is_dir():
            continue
        if video_dir.name.startswith(exclude_video_prefix):
            continue
        (tmp_dir / video_dir.name).symlink_to(video_dir.resolve())
        n_linked += 1

    model = GeneralReIDModel()
    stats = model.train_on_dataset(
        tmp_dir,
        epochs=epochs,
        verbose=False,
    )

    # Clean up symlinks
    shutil.rmtree(tmp_dir)

    logger.info(
        "  Trained on %d videos (%d crops, %d identities, loss=%.4f)",
        n_linked, stats.get("n_samples", 0),
        stats.get("n_identities", 0), stats.get("loss", 0),
    )

    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="LOO-CV for general ReID model")
    parser.add_argument("--video-id", type=str, default=None, help="Single fold only")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs per fold")
    parser.add_argument("--skip-extract", action="store_true", help="Skip crop extraction")
    args = parser.parse_args()

    from rallycut.evaluation.db import get_connection

    # Load all GT videos
    with get_connection() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT v.id, v.player_matching_gt_json
                FROM videos v
                WHERE v.player_matching_gt_json IS NOT NULL
                ORDER BY v.id
            """
            cur.execute(query)
            all_rows = cur.fetchall()

    video_rows = [
        (str(vid), gt) for vid, gt in all_rows
        if args.video_id is None or str(vid).startswith(args.video_id)
    ]

    if not video_rows:
        logger.error("No GT videos found")
        sys.exit(1)

    # Step 1: Extract crops for ALL GT videos (not just the subset)
    if not args.skip_extract:
        extract_all_crops([(str(vid), gt) for vid, gt in all_rows])

    logger.info("")
    logger.info("=" * 70)
    logger.info("LOO-CV: OSNet-x1.0 MSMT17 + SupCon (%d epochs)", args.epochs)
    logger.info("=" * 70)
    logger.info("Folds: %d", len(video_rows))
    logger.info("")

    total_hsv_correct = 0
    total_reid_correct = 0
    total_assignments = 0
    results_table: list[tuple[str, int, float, float, float]] = []

    for fold_idx, (video_id, gt_json) in enumerate(video_rows):
        gt_data = cast(dict[str, Any], gt_json)
        gt_rallies_raw = gt_data.get("rallies", {})
        gt_rallies: dict[str, dict[str, int]] = {
            rid: {str(k): int(v) for k, v in mapping.items()}
            for rid, mapping in gt_rallies_raw.items()
        }

        n_gt = sum(len(m) for m in gt_rallies.values())
        logger.info(
            "[%d/%d] %s (%d GT assignments):",
            fold_idx + 1, len(video_rows), video_id[:8], n_gt,
        )

        # HSV baseline
        t0 = time.time()
        pred_hsv = run_matching_hsv_only(video_id)
        _perm_h, c_hsv, t_hsv = find_best_permutation(gt_rallies, pred_hsv)
        hsv_time = time.time() - t0

        # Train model on other videos, run matching
        t0 = time.time()
        model = train_model_excluding(video_id[:12], epochs=args.epochs)
        pred_reid = run_matching_with_model(video_id, model)
        _perm_r, c_reid, t_reid = find_best_permutation(gt_rallies, pred_reid)
        reid_time = time.time() - t0

        acc_hsv = c_hsv / t_hsv * 100 if t_hsv > 0 else 0
        acc_reid = c_reid / t_reid * 100 if t_reid > 0 else 0
        delta = acc_reid - acc_hsv

        marker = "+" if delta > 0 else ("=" if delta == 0 else "")
        logger.info(
            "  HSV: %d/%d = %.1f%%  |  ReID: %d/%d = %.1f%%  [%s%.1fpp]  "
            "(train %.0fs, match %.0fs)",
            c_hsv, t_hsv, acc_hsv, c_reid, t_reid, acc_reid,
            marker, delta, reid_time - hsv_time, hsv_time,
        )

        total_hsv_correct += c_hsv
        total_reid_correct += c_reid
        total_assignments += t_hsv
        results_table.append((video_id[:8], t_hsv, acc_hsv, acc_reid, delta))

        # Free GPU memory between folds
        del model

    # Print summary table
    logger.info("")
    logger.info("=" * 70)
    logger.info("%-10s %6s %8s %8s %8s", "Video", "N", "HSV", "ReID", "Delta")
    logger.info("-" * 70)

    for vid, n, hsv, reid, delta in sorted(results_table, key=lambda r: -r[4]):
        marker = "+" if delta > 0 else ""
        logger.info("%-10s %6d %7.1f%% %7.1f%% %+7.1fpp", vid, n, hsv, reid, delta)

    logger.info("-" * 70)
    if total_assignments > 0:
        agg_hsv = total_hsv_correct / total_assignments * 100
        agg_reid = total_reid_correct / total_assignments * 100
        delta = agg_reid - agg_hsv
        logger.info(
            "%-10s %6d %7.1f%% %7.1f%% %+7.1fpp",
            "AGGREGATE", total_assignments, agg_hsv, agg_reid, delta,
        )
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
