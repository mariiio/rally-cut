#!/usr/bin/env python3
"""Evaluate ReID-augmented cross-rally matching on videos with reference crops.

Compares HSV-only vs HSV+ReID matching accuracy on GT videos that have
user-selected reference crops.

Usage:
    uv run python scripts/eval_reid_matching.py
    uv run python scripts/eval_reid_matching.py --video-id fb83f876
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np

from scripts.eval_match_players import compute_team_accuracy, find_best_permutation

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def load_reference_profiles(
    video_id: str,
    video_path: Path,
) -> dict[int, Any] | None:
    """Load reference crops from DB and build HSV + ReID profiles."""
    import cv2

    from rallycut.evaluation.db import get_connection
    from rallycut.tracking.player_features import (
        build_profiles_from_crops,
        extract_appearance_features,
    )
    from rallycut.tracking.reid_embeddings import extract_backbone_features

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT player_id, frame_ms, bbox_x, bbox_y, bbox_w, bbox_h
                   FROM player_reference_crops
                   WHERE video_id = %s
                   ORDER BY player_id, created_at""",
                [video_id],
            )
            crop_rows = cur.fetchall()

    if not crop_rows:
        return None

    crop_infos = [
        {
            "player_id": r[0], "frame_ms": r[1],
            "bbox_x": r[2], "bbox_y": r[3],
            "bbox_w": r[4], "bbox_h": r[5],
        }
        for r in crop_rows
    ]

    cap = cv2.VideoCapture(str(video_path))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    features_by_player: dict[int, list[Any]] = {}
    bgr_crops_by_player: dict[int, list[Any]] = {}

    for info in sorted(crop_infos, key=lambda c: int(c["frame_ms"])):
        pid = int(info["player_id"])
        frame_ms = int(info["frame_ms"])

        if fw > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(frame_ms))
            ret, frame = cap.read()
            if ret and frame is not None:
                frame_arr = np.asarray(frame)
                bx = float(info["bbox_x"])
                by = float(info["bbox_y"])
                bw = float(info["bbox_w"])
                bh = float(info["bbox_h"])
                features = extract_appearance_features(
                    frame=frame_arr, track_id=0, frame_number=0,
                    bbox=(bx, by, bw, bh), frame_width=fw, frame_height=fh,
                )
                features_by_player.setdefault(pid, []).append(features)

                x1 = max(0, int((bx - bw / 2) * fw))
                y1 = max(0, int((by - bh / 2) * fh))
                x2 = min(fw, int((bx + bw / 2) * fw))
                y2 = min(fh, int((by + bh / 2) * fh))
                if x2 > x1 and y2 > y1:
                    crop = frame_arr[y1:y2, x1:x2]
                    if crop.size > 0:
                        bgr_crops_by_player.setdefault(pid, []).append(crop)

    cap.release()

    # Extract DINOv2 embeddings
    reid_emb: dict[int, Any] = {}
    for pid, crops in bgr_crops_by_player.items():
        if not crops:
            continue
        embeddings = extract_backbone_features(crops)
        mean_emb = embeddings.mean(axis=0)
        norm = np.linalg.norm(mean_emb)
        if norm > 0:
            mean_emb /= norm
        reid_emb[pid] = mean_emb

    return build_profiles_from_crops(
        features_by_player,
        reid_embeddings_by_player=reid_emb or None,
    )


def run_matching(
    video_id: str,
    video_path: Path,
    rallies: list[Any],
    reference_profiles: dict[int, Any] | None,
    extract_reid: bool,
) -> dict[str, dict[str, int]]:
    """Run match_players_across_rallies and return pred_rallies dict."""
    from rallycut.tracking.match_tracker import match_players_across_rallies

    result = match_players_across_rallies(
        video_path=video_path,
        rallies=rallies,
        reference_profiles=reference_profiles,
        extract_reid=extract_reid,
    )

    pred: dict[str, dict[str, int]] = {}
    for rally, rr in zip(rallies, result.rally_results):
        pred[rally.rally_id] = {str(k): v for k, v in rr.track_to_player.items()}
    return pred


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ReID-augmented cross-rally matching",
    )
    parser.add_argument("--video-id", type=str, default=None)
    args = parser.parse_args()

    from rallycut.evaluation.db import get_connection
    from rallycut.evaluation.gt_loader import (
        build_positions_lookup_from_db,
        load_player_matching_gt,
    )
    from rallycut.evaluation.tracking.db import get_video_path, load_rallies_for_video

    # Find videos with reference crops + GT. Custom SELECT because we need to
    # join against player_reference_crops. Route each row's gt_json through the
    # shared loader so v1 + v2 both work.
    with get_connection() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT v.id, v.player_matching_gt_json
                FROM videos v
                WHERE v.player_matching_gt_json IS NOT NULL
                  AND v.id IN (SELECT DISTINCT video_id FROM player_reference_crops)
            """
            params: list[str] = []
            if args.video_id:
                query += " AND v.id LIKE %s"
                params.append(f"{args.video_id}%")
            cur.execute(query, params)
            raw_rows = cur.fetchall()
            positions_lookup = build_positions_lookup_from_db(cur)
            rows = [
                (r[0], load_player_matching_gt(r[1], positions_lookup=positions_lookup))
                for r in raw_rows
            ]

    if not rows:
        logger.error("No videos with both GT and reference crops found.")
        sys.exit(1)

    logger.info("=" * 70)
    logger.info("ReID Cross-Rally Matching Evaluation")
    logger.info("=" * 70)
    logger.info(f"Videos: {len(rows)}")
    logger.info("")

    total_hsv_correct = 0
    total_reid_correct = 0
    total_assignments = 0
    total_hsv_team = 0
    total_reid_team = 0

    for video_id, gt_normalized in rows:
        video_id = str(video_id)
        gt_rallies = gt_normalized.rallies

        rallies = load_rallies_for_video(video_id)
        video_path = get_video_path(video_id)
        if not rallies or not video_path:
            logger.warning("Cannot load video %s, skipping", video_id[:8])
            continue

        n_rallies = len(rallies)
        logger.info(f"Video {video_id[:8]} ({n_rallies} rallies):")

        # Run HSV-only (no reference profiles, no ReID)
        pred_hsv = run_matching(video_id, video_path, rallies, None, False)
        perm_hsv, c_hsv, t_hsv = find_best_permutation(gt_rallies, pred_hsv)
        tc_hsv, _ = compute_team_accuracy(gt_rallies, pred_hsv, perm_hsv)

        # Run HSV + ReID (reference profiles with embeddings)
        ref_profiles = load_reference_profiles(video_id, video_path)
        pred_reid = run_matching(video_id, video_path, rallies, ref_profiles, True)
        perm_reid, c_reid, t_reid = find_best_permutation(gt_rallies, pred_reid)
        tc_reid, _ = compute_team_accuracy(gt_rallies, pred_reid, perm_reid)

        acc_hsv = c_hsv / t_hsv * 100 if t_hsv > 0 else 0
        acc_reid = c_reid / t_reid * 100 if t_reid > 0 else 0
        team_hsv = tc_hsv / t_hsv * 100 if t_hsv > 0 else 0
        team_reid = tc_reid / t_reid * 100 if t_reid > 0 else 0
        delta = acc_reid - acc_hsv

        marker = "+" if delta > 0 else ("=" if delta == 0 else "-")
        logger.info(
            f"  HSV-only: {c_hsv}/{t_hsv} = {acc_hsv:.1f}% "
            f"(team {team_hsv:.1f}%)"
        )
        logger.info(
            f"  HSV+ReID: {c_reid}/{t_reid} = {acc_reid:.1f}% "
            f"(team {team_reid:.1f}%)  [{marker}{abs(delta):.1f}pp]"
        )

        total_hsv_correct += c_hsv
        total_reid_correct += c_reid
        total_assignments += t_hsv  # Same total for both
        total_hsv_team += tc_hsv
        total_reid_team += tc_reid

    logger.info("")
    logger.info("=" * 70)
    logger.info("AGGREGATE")
    logger.info("=" * 70)
    if total_assignments > 0:
        agg_hsv = total_hsv_correct / total_assignments * 100
        agg_reid = total_reid_correct / total_assignments * 100
        agg_hsv_t = total_hsv_team / total_assignments * 100
        agg_reid_t = total_reid_team / total_assignments * 100
        delta = agg_reid - agg_hsv
        logger.info(
            f"HSV-only:  {total_hsv_correct}/{total_assignments} = {agg_hsv:.1f}% "
            f"(team {agg_hsv_t:.1f}%)"
        )
        logger.info(
            f"HSV+ReID:  {total_reid_correct}/{total_assignments} = {agg_reid:.1f}% "
            f"(team {agg_reid_t:.1f}%)  [delta: {delta:+.1f}pp]"
        )
    else:
        logger.info("No assignments evaluated.")


if __name__ == "__main__":
    main()
