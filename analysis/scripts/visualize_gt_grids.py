#!/usr/bin/env python3
"""Render GT player identity grids for visual quality inspection.

For each GT video, extracts one crop per player (1-4) per rally and
renders a grid image (rows=rallies, columns=P1-P4). Helps catch
labeling errors where the wrong player appears in a column.

Output: analysis/outputs/gt_grids/{video_id}.jpg

Usage:
    uv run python scripts/visualize_gt_grids.py
    uv run python scripts/visualize_gt_grids.py --video-id abc123
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

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "gt_grids"
CROP_W = 60
CROP_H = 120
PADDING = 2
HEADER_H = 20
LABEL_H = 14


def render_grid(
    video_id: str,
    video_path: Path,
    gt_data: dict[str, Any],
    rallies_db: list[dict[str, Any]],
    output_dir: Path,
) -> bool:
    """Render a grid image for one GT video.

    Returns True if grid was successfully created.
    """
    gt_rallies = gt_data.get("rallies", {})
    if not gt_rallies:
        return False

    rally_by_id = {r["rally_id"]: r for r in rallies_db}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("  Cannot open video: %s", video_path)
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Collect one crop per player per rally
    # Structure: list of (rally_label, {player_id: crop})
    grid_rows: list[tuple[str, dict[int, np.ndarray]]] = []

    rally_ids_sorted = sorted(
        gt_rallies.keys(),
        key=lambda rid: rally_by_id.get(rid, {}).get("start_ms", 0),
    )

    for rally_id in rally_ids_sorted:
        track_to_player = gt_rallies[rally_id]
        rally = rally_by_id.get(rally_id)
        if not rally or not rally.get("positions_json"):
            continue

        positions = rally["positions_json"]
        start_ms = rally.get("start_ms", 0)
        start_frame = int(start_ms / 1000 * fps)

        # Group positions by track, pick middle frame
        track_positions: dict[int, list[dict]] = {}
        for p in positions:
            tid = p.get("trackId", p.get("track_id"))
            if tid is not None:
                track_positions.setdefault(tid, []).append(p)

        # Auto-translate GT track IDs if they don't match positions (pre-remap GT)
        gt_tids = {int(t) for t in track_to_player}
        pos_tids = set(track_positions.keys())
        if gt_tids and not gt_tids & pos_tids:
            afm = rally.get("applied_full_mapping")
            if afm:
                translated = {}
                for t, p in track_to_player.items():
                    new_t = str(afm.get(str(t), afm.get(int(t), t)))
                    translated[new_t] = p
                logger.warning("    %s: translated GT IDs via appliedFullMapping", rally_id[:8])
                track_to_player = translated

        player_crops: dict[int, np.ndarray] = {}

        for tid_str, pid in track_to_player.items():
            tid = int(tid_str)
            pid = int(pid)
            pos_list = track_positions.get(tid, [])
            if not pos_list:
                continue

            # Pick the middle frame for a representative crop
            pos_list.sort(key=lambda pp: pp.get("frameNumber", pp.get("frame_number", 0)))
            mid_pos = pos_list[len(pos_list) // 2]
            fn = mid_pos.get("frameNumber", mid_pos.get("frame_number", 0))
            abs_frame = start_frame + fn

            cap.set(cv2.CAP_PROP_POS_FRAMES, abs_frame)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            bx = mid_pos.get("x", 0.0)
            by = mid_pos.get("y", 0.0)
            bw = mid_pos.get("width", 0.0)
            bh = mid_pos.get("height", 0.0)

            x1 = max(0, int((bx - bw / 2) * fw))
            y1 = max(0, int((by - bh / 2) * fh))
            x2 = min(fw, int((bx + bw / 2) * fw))
            y2 = min(fh, int((by + bh / 2) * fh))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            crop = cv2.resize(crop, (CROP_W, CROP_H))
            player_crops[pid] = crop

        if player_crops:
            grid_rows.append((rally_id[:6], player_crops))

    cap.release()

    if not grid_rows:
        return False

    # Render the grid image
    n_rows = len(grid_rows)
    n_cols = 4  # Players 1-4
    cell_w = CROP_W + PADDING * 2
    cell_h = CROP_H + PADDING * 2
    img_w = LABEL_H + n_cols * cell_w + PADDING
    img_h = HEADER_H + n_rows * cell_h + PADDING

    grid = np.full((img_h, img_w, 3), 40, dtype=np.uint8)  # Dark background

    # Column headers (P1, P2, P3, P4)
    for col in range(n_cols):
        pid = col + 1
        x = LABEL_H + col * cell_w + cell_w // 2 - 8
        cv2.putText(
            grid, f"P{pid}", (x, HEADER_H - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1,
        )

    # Grid cells
    for row_idx, (rally_label, player_crops) in enumerate(grid_rows):
        y_base = HEADER_H + row_idx * cell_h

        # Row label (rally ID prefix)
        cv2.putText(
            grid, rally_label[:4], (1, y_base + cell_h // 2 + 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.25, (140, 140, 140), 1,
        )

        for col in range(n_cols):
            pid = col + 1
            x_base = LABEL_H + col * cell_w + PADDING
            y_pos = y_base + PADDING

            if pid in player_crops:
                crop = player_crops[pid]
                grid[y_pos:y_pos + CROP_H, x_base:x_base + CROP_W] = crop
            else:
                # Missing player — draw red X
                cv2.line(
                    grid,
                    (x_base + 5, y_pos + 5),
                    (x_base + CROP_W - 5, y_pos + CROP_H - 5),
                    (0, 0, 180), 2,
                )
                cv2.line(
                    grid,
                    (x_base + CROP_W - 5, y_pos + 5),
                    (x_base + 5, y_pos + CROP_H - 5),
                    (0, 0, 180), 2,
                )

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{video_id[:12]}.jpg"
    cv2.imwrite(str(out_path), grid, [cv2.IMWRITE_JPEG_QUALITY, 90])
    logger.info("  Saved %s (%d rallies × 4 players)", out_path.name, n_rows)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Render GT player identity grids")
    parser.add_argument("--video-id", type=str, default=None)
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Output directory for grid images",
    )
    args = parser.parse_args()

    from rallycut.evaluation.db import get_connection
    from rallycut.evaluation.gt_loader import (
        build_positions_lookup_from_db,
        load_player_matching_gt,
    )
    from rallycut.evaluation.tracking.db import get_video_path

    with get_connection() as conn:
        with conn.cursor() as cur:
            query = """
                SELECT v.id, v.player_matching_gt_json, v.match_analysis_json
                FROM videos v
                WHERE v.player_matching_gt_json IS NOT NULL
            """
            params: list[str] = []
            if args.video_id:
                query += " AND v.id LIKE %s"
                params.append(f"{args.video_id}%")
            query += " ORDER BY v.id"
            cur.execute(query, params)
            raw_rows = cur.fetchall()
            positions_lookup = build_positions_lookup_from_db(cur)
            # Normalize GT while cursor is open (v2 lookups need it). Shape it
            # as a dict with a "rallies" key so downstream code keeps working.
            video_rows = [
                (
                    r[0],
                    {
                        "rallies": load_player_matching_gt(
                            r[1], positions_lookup=positions_lookup,
                        ).rallies,
                    },
                    r[2],
                )
                for r in raw_rows
            ]

    if not video_rows:
        logger.error("No videos with GT found.")
        sys.exit(1)

    logger.info("Generating GT grids for %d videos...", len(video_rows))
    logger.info("Output: %s", args.output_dir)
    logger.info("")

    n_success = 0
    for i, (video_id, gt_json, match_analysis_json) in enumerate(video_rows):
        video_id = str(video_id)
        gt_data = cast(dict[str, Any], gt_json)
        match_analysis = cast(dict[str, Any] | None, match_analysis_json)

        video_path = get_video_path(video_id)
        if video_path is None:
            logger.warning("[%d/%d] %s: no video file", i + 1, len(video_rows), video_id[:8])
            continue

        # Load rally metadata
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

        # Build per-rally appliedFullMapping from match analysis
        remap_by_rally: dict[str, dict[str, int]] = {}
        if match_analysis:
            for entry in match_analysis.get("rallies", []):
                rid_key = entry.get("rallyId") or entry.get("rally_id", "")
                afm = entry.get("appliedFullMapping")
                if afm and entry.get("remapApplied"):
                    remap_by_rally[rid_key] = afm

        rallies_db = []
        for rid, start_ms, pos_json, ptids in rally_rows:
            ptids_list = json.loads(ptids) if isinstance(ptids, str) else (ptids or [])
            rallies_db.append({
                "rally_id": str(rid),
                "start_ms": start_ms,
                "positions_json": pos_json,
                "primary_track_ids": ptids_list,
                "applied_full_mapping": remap_by_rally.get(str(rid)),
            })

        logger.info("[%d/%d] %s:", i + 1, len(video_rows), video_id[:8])
        if render_grid(video_id, video_path, gt_data, rallies_db, args.output_dir):
            n_success += 1

    logger.info("")
    logger.info("Done: %d/%d grids generated in %s", n_success, len(video_rows), args.output_dir)


if __name__ == "__main__":
    main()
