#!/usr/bin/env python3
"""Extract labeled player crops from GT videos for general ReID model training.

Uses player_matching_gt_json (track→player mappings per rally) + stored positions
to extract labeled player crops. Output is organized for contrastive training:

    reid_training_data/{video_id}/player_{pid}/{rally_id}_{frame}.jpg

Each video has ~4 players, each rally has ~12 frames per player. Expected total:
~33k crops from 41 GT videos.

Usage:
    uv run python scripts/build_reid_training_data.py
    uv run python scripts/build_reid_training_data.py --output-dir custom_path
    uv run python scripts/build_reid_training_data.py --video-id abc123
    uv run python scripts/build_reid_training_data.py --num-samples 8  # fewer per track
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path(__file__).parent.parent / "reid_training_data"
NUM_SAMPLES = 12  # frames per track per rally


def extract_crops_for_video(
    video_id: str,
    video_path: Path,
    gt_data: dict[str, Any],
    rallies_db: list[dict[str, Any]],
    output_dir: Path,
    num_samples: int = NUM_SAMPLES,
) -> dict[str, int]:
    """Extract labeled player crops from one video.

    Args:
        video_id: Video identifier.
        video_path: Path to video file.
        gt_data: player_matching_gt_json with {rallies: {rally_id: {track_id: player_id}}}.
        rallies_db: Rally metadata with positions_json, start_ms, primary_track_ids.
        output_dir: Root output directory.
        num_samples: Frames to sample per track per rally.

    Returns:
        {player_id_str: num_crops_extracted}
    """
    gt_rallies = gt_data.get("rallies", {})
    if not gt_rallies:
        return {}

    # Index rally data by ID
    rally_by_id = {r["rally_id"]: r for r in rallies_db}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("  Cannot open video: %s", video_path)
        return {}

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    crop_counts: dict[str, int] = {}
    total_crops = 0

    for rally_id, track_to_player in gt_rallies.items():
        rally = rally_by_id.get(rally_id)
        if not rally:
            continue

        positions_json = rally.get("positions_json")
        if not positions_json:
            continue

        start_ms = rally.get("start_ms", 0)
        start_frame = int(start_ms / 1000 * fps)
        primary_tids = set(rally.get("primary_track_ids", []))

        # Group positions by track_id
        track_positions: dict[int, list[dict]] = {}
        for p in positions_json:
            tid = p.get("trackId", p.get("track_id"))
            if tid is not None:
                track_positions.setdefault(tid, []).append(p)

        # Process each track with a GT player assignment
        for tid_str, pid in track_to_player.items():
            tid = int(tid_str)
            pid = int(pid)
            pid_str = str(pid)

            pos_list = track_positions.get(tid, [])
            if not pos_list:
                continue

            # Sort by frame and sample evenly
            pos_list.sort(key=lambda pp: pp.get("frameNumber", pp.get("frame_number", 0)))
            n = len(pos_list)
            if n <= num_samples:
                indices = list(range(n))
            else:
                indices = [int(i * (n - 1) / (num_samples - 1)) for i in range(num_samples)]

            # Create output directory for this player
            player_dir = output_dir / video_id[:12] / f"player_{pid}"
            player_dir.mkdir(parents=True, exist_ok=True)

            for idx in indices:
                p = pos_list[idx]
                fn = p.get("frameNumber", p.get("frame_number", 0))
                abs_frame = start_frame + fn

                cap.set(cv2.CAP_PROP_POS_FRAMES, abs_frame)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                bx = p.get("x", 0.0)
                by = p.get("y", 0.0)
                bw = p.get("width", 0.0)
                bh = p.get("height", 0.0)

                x1 = max(0, int((bx - bw / 2) * fw))
                y1 = max(0, int((by - bh / 2) * fh))
                x2 = min(fw, int((bx + bw / 2) * fw))
                y2 = min(fh, int((by + bh / 2) * fh))

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = frame[y1:y2, x1:x2]
                if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 8:
                    continue

                crop_path = player_dir / f"{rally_id[:8]}_{fn:05d}.jpg"
                cv2.imwrite(str(crop_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 90])
                crop_counts[pid_str] = crop_counts.get(pid_str, 0) + 1
                total_crops += 1

    cap.release()

    if total_crops > 0:
        per_player = ", ".join(f"P{k}:{v}" for k, v in sorted(crop_counts.items()))
        logger.info(f"  Extracted {total_crops} crops ({per_player})")

    return crop_counts


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build ReID training data from GT videos",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
        help="Output directory for extracted crops",
    )
    parser.add_argument("--video-id", type=str, default=None)
    parser.add_argument(
        "--num-samples", type=int, default=NUM_SAMPLES,
        help="Frames to sample per track per rally",
    )
    args = parser.parse_args()

    from rallycut.evaluation.db import get_connection
    from rallycut.evaluation.tracking.db import get_video_path

    # Find videos with GT
    with get_connection() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT v.id, v.player_matching_gt_json
                FROM videos v
                WHERE v.player_matching_gt_json IS NOT NULL
            """
            params: list[str] = []
            if args.video_id:
                query += " AND v.id LIKE %s"
                params.append(f"{args.video_id}%")
            query += " ORDER BY v.id"
            cur.execute(query, params)
            video_rows = cur.fetchall()

    if not video_rows:
        logger.error("No videos with player_matching_gt_json found.")
        sys.exit(1)

    logger.info(f"Found {len(video_rows)} GT videos")
    logger.info(f"Output: {args.output_dir}")
    logger.info("")

    # Load rally data for all videos
    total_videos = 0
    total_crops = 0
    total_players = 0

    for video_id, gt_json in video_rows:
        video_id = str(video_id)
        gt_data = cast(dict[str, Any], gt_json)

        video_path = get_video_path(video_id)
        if video_path is None:
            logger.warning(f"[{total_videos+1}/{len(video_rows)}] {video_id[:8]}: no video file")
            continue

        # Load rally metadata
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT r.id, r.start_ms, pt.positions_json, pt.primary_track_ids
                    FROM rallies r
                    JOIN player_tracks pt ON pt.rally_id = r.id
                    WHERE r.video_id = %s
                      AND pt.positions_json IS NOT NULL
                    ORDER BY r.start_ms
                    """,
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

        n_gt_rallies = len(gt_data.get("rallies", {}))
        total_videos += 1
        logger.info(
            f"[{total_videos}/{len(video_rows)}] {video_id[:8]}: "
            f"{n_gt_rallies} GT rallies, {len(rallies_db)} tracked"
        )

        counts = extract_crops_for_video(
            video_id, video_path, gt_data, rallies_db,
            args.output_dir, args.num_samples,
        )

        n_crops = sum(counts.values())
        total_crops += n_crops
        total_players += len(counts)

    logger.info("")
    logger.info("=" * 50)
    logger.info(f"Total: {total_crops} crops from {total_videos} videos, {total_players} players")
    logger.info(f"Output: {args.output_dir}")


if __name__ == "__main__":
    main()
