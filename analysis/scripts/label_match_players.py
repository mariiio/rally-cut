#!/usr/bin/env python3
"""Labeling helper for cross-rally player matching ground truth.

Generates a visual grid of player crops and a pre-filled GT JSON template
from existing match-players assignments. The user corrects any wrong entries.

Usage:
    uv run python scripts/label_match_players.py <video-id>
    uv run python scripts/label_match_players.py <video-id> --output-dir /path/to/dir
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

# Player ID colors (matching editor: P1=green, P2=blue, P3=orange, P4=purple)
PLAYER_COLORS = {
    1: (0, 200, 0),      # green (BGR)
    2: (200, 100, 0),     # blue (BGR)
    3: (0, 140, 255),     # orange (BGR)
    4: (180, 0, 180),     # purple (BGR)
}

PLAYER_LABELS = {
    1: "P1 (green)",
    2: "P2 (blue)",
    3: "P3 (orange)",
    4: "P4 (purple)",
}


def extract_player_crop(
    cap: cv2.VideoCapture,
    frame_number: int,
    bbox: tuple[float, float, float, float],
    frame_width: int,
    frame_height: int,
    crop_size: tuple[int, int] = (80, 160),
) -> np.ndarray | None:
    """Extract and resize a player crop from a video frame.

    Args:
        cap: Open video capture.
        frame_number: Absolute frame number to seek to.
        bbox: (x, y, w, h) in normalized coordinates.
        frame_width: Video frame width.
        frame_height: Video frame height.
        crop_size: Output (width, height) for the crop.

    Returns:
        Resized crop as BGR numpy array, or None if frame read fails.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        return None

    x, y, w, h = bbox
    x1 = max(0, int((x - w / 2) * frame_width))
    y1 = max(0, int((y - h / 2) * frame_height))
    x2 = min(frame_width, int((x + w / 2) * frame_width))
    y2 = min(frame_height, int((y + h / 2) * frame_height))

    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, crop_size) if crop.size > 0 else None


def build_visual_grid(
    video_path: Path,
    rallies: list[dict[str, Any]],
    rally_data: list[dict[str, Any]],
    fps: float,
    frame_width: int,
    frame_height: int,
    crop_size: tuple[int, int] = (80, 160),
) -> np.ndarray:
    """Build a visual grid: rows=rallies, columns=player IDs 1-4.

    Args:
        video_path: Path to video file.
        rallies: List of rally entries from match_analysis_json.
        rally_data: List of {rally_id, start_ms, end_ms, positions_json}.
        fps: Video FPS.
        frame_width: Video width.
        frame_height: Video height.
        crop_size: Size of each player crop.

    Returns:
        BGR image of the visual grid.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    cw, ch = crop_size
    header_height = 40
    row_label_width = 120
    padding = 4

    n_rows = len(rallies)
    n_cols = 4  # Player IDs 1-4

    grid_width = row_label_width + n_cols * (cw + padding) + padding
    grid_height = header_height + n_rows * (ch + padding) + padding

    grid = np.full((grid_height, grid_width, 3), 40, dtype=np.uint8)  # dark bg

    # Draw column headers
    for pid in range(1, 5):
        col_x = row_label_width + (pid - 1) * (cw + padding) + padding
        color = PLAYER_COLORS[pid]
        cv2.putText(
            grid, PLAYER_LABELS[pid], (col_x, header_height - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )

    # Build rally ID -> positions lookup
    rally_positions: dict[str, list[dict[str, Any]]] = {}
    rally_times: dict[str, tuple[int, int]] = {}
    for rd in rally_data:
        rid = rd["rally_id"]
        rally_positions[rid] = rd.get("positions_json", [])
        rally_times[rid] = (rd["start_ms"], rd["end_ms"])

    try:
        for row_idx, rally_entry in enumerate(rallies):
            rid = rally_entry.get("rallyId", rally_entry.get("rally_id", ""))
            track_to_player = rally_entry.get("trackToPlayer", rally_entry.get("track_to_player", {}))

            row_y = header_height + row_idx * (ch + padding) + padding

            # Row label
            cv2.putText(
                grid, rid[:8], (4, row_y + ch // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180, 180, 180), 1, cv2.LINE_AA,
            )

            if rid not in rally_positions:
                continue

            positions = rally_positions[rid]
            start_ms, end_ms = rally_times.get(rid, (0, 0))
            start_frame = int(start_ms / 1000 * fps)

            # Build player_id -> mid-rally bbox
            player_to_track: dict[int, int] = {}
            for tid_str, pid in track_to_player.items():
                player_to_track[int(pid)] = int(tid_str)

            # Find mid-rally frame for each player
            mid_ms = (start_ms + end_ms) // 2
            mid_frame_rel = int((mid_ms - start_ms) / 1000 * fps)

            for pid in range(1, 5):
                col_x = row_label_width + (pid - 1) * (cw + padding) + padding
                tid = player_to_track.get(pid)
                if tid is None:
                    # No track assigned to this player
                    cv2.putText(
                        grid, "?", (col_x + cw // 2 - 5, row_y + ch // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2,
                    )
                    continue

                # Find the position closest to mid-rally for this track
                best_pos = None
                best_dist = float("inf")
                for p in positions:
                    if p["trackId"] == tid:
                        dist = abs(p["frameNumber"] - mid_frame_rel)
                        if dist < best_dist:
                            best_dist = dist
                            best_pos = p

                if best_pos is None:
                    continue

                abs_frame = start_frame + best_pos["frameNumber"]
                bbox = (best_pos["x"], best_pos["y"], best_pos["width"], best_pos["height"])
                crop = extract_player_crop(
                    cap, abs_frame, bbox, frame_width, frame_height, crop_size
                )

                if crop is not None:
                    # Draw colored border
                    color = PLAYER_COLORS[pid]
                    cv2.rectangle(crop, (0, 0), (cw - 1, ch - 1), color, 2)
                    grid[row_y:row_y + ch, col_x:col_x + cw] = crop
    finally:
        cap.release()

    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Label cross-rally player matching GT")
    parser.add_argument("video_id", help="Video ID to label")
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).parent.parent / "evaluation" / "match_gt",
        help="Output directory for GT files",
    )
    args = parser.parse_args()

    from rallycut.evaluation.db import get_connection
    from rallycut.evaluation.tracking.db import get_video_path

    video_id = args.video_id
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load match analysis from DB
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT match_analysis_json, fps, width, height, s3_key
                   FROM videos WHERE id = %s""",
                [video_id],
            )
            row = cur.fetchone()
            if not row:
                logger.error("Video %s not found", video_id)
                sys.exit(1)

            match_analysis, fps, width, height, s3_key = row
            if not match_analysis:
                logger.error(
                    "No match_analysis_json for video %s. "
                    "Run 'rallycut match-players %s' first.",
                    video_id, video_id,
                )
                sys.exit(1)

            fps = float(fps) if fps else 30.0
            width = int(width) if width else 1920
            height = int(height) if height else 1080

            # Load positions for each rally
            rallies = cast(dict[str, Any], match_analysis).get("rallies", [])
            rally_ids = [
                r.get("rallyId", r.get("rally_id", "")) for r in rallies
            ]

            rally_data: list[dict[str, Any]] = []
            for rid in rally_ids:
                cur.execute(
                    """SELECT r.start_ms, r.end_ms, pt.positions_json
                       FROM rallies r
                       JOIN player_tracks pt ON pt.rally_id = r.id
                       WHERE r.id = %s""",
                    [rid],
                )
                rrow = cur.fetchone()
                if rrow:
                    rally_data.append({
                        "rally_id": rid,
                        "start_ms": rrow[0],
                        "end_ms": rrow[1],
                        "positions_json": rrow[2] or [],
                    })

    # Resolve video path
    video_path = get_video_path(video_id)
    if video_path is None:
        logger.error("Could not resolve video file for %s", video_id)
        sys.exit(1)

    # Get video name from s3_key
    video_name = Path(s3_key).stem if s3_key else video_id[:8]

    # Build visual grid
    logger.info("Building visual grid for %d rallies...", len(rallies))
    grid = build_visual_grid(
        video_path, rallies, rally_data, fps, width, height,
    )

    # Save grid image
    grid_path = output_dir / f"{video_id[:8]}_grid.png"
    cv2.imwrite(str(grid_path), grid)
    logger.info("Visual grid saved to %s", grid_path)

    # Build pre-filled GT JSON template
    # Detect side switches from the match analysis
    side_switches: list[int] = []
    gt_rallies: dict[str, dict[str, int]] = {}

    for i, rally_entry in enumerate(rallies):
        rid = rally_entry.get("rallyId", rally_entry.get("rally_id", ""))
        track_to_player = rally_entry.get(
            "trackToPlayer", rally_entry.get("track_to_player", {})
        )

        if rally_entry.get("sideSwitchDetected", rally_entry.get("side_switch_detected", False)):
            side_switches.append(i)

        # Invert: for GT, we store track_id -> player_id
        gt_rallies[rid] = {str(k): int(v) for k, v in track_to_player.items()}

    gt_json = {
        "video_id": video_id,
        "notes": f"{video_name} - review grid image and correct wrong assignments",
        "side_switches": side_switches,
        "rallies": gt_rallies,
    }

    gt_path = output_dir / f"{video_id[:8]}.json"
    with open(gt_path, "w") as f:
        json.dump(gt_json, f, indent=2)

    logger.info("GT template saved to %s", gt_path)

    # Print rally timestamps for cross-reference
    logger.info("\nRally timestamps:")
    for i, rd in enumerate(rally_data):
        start_s = rd["start_ms"] / 1000
        end_s = rd["end_ms"] / 1000
        rid = rd["rally_id"]
        switch_marker = " [SWITCH]" if i in side_switches else ""
        logger.info(
            "  [%2d] %s  %5.1fs - %5.1fs  (%.1fs)%s",
            i, rid[:8], start_s, end_s, end_s - start_s, switch_marker,
        )

    logger.info(
        "\nNext steps:\n"
        "  1. Open %s to see player crops per rally\n"
        "  2. Edit %s to correct wrong track→player assignments\n"
        "  3. Run: uv run python scripts/eval_match_players.py",
        grid_path, gt_path,
    )


if __name__ == "__main__":
    main()
