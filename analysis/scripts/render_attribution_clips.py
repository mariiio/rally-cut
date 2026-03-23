"""Render side-by-side player clips for human attribution accuracy testing.

For the hardest within-team attribution errors, extracts 16-frame clips of
both candidate players (GT player vs proximity-attributed player) and renders
them as montage images for human review.

Output: one PNG per contact showing two rows (candidate A, candidate B) with
key frames. The GT player is NOT labeled — the reviewer must judge which
player touched the ball, then compare to the answer key.

Usage:
    cd analysis
    uv run python scripts/render_attribution_clips.py
    uv run python scripts/render_attribution_clips.py --n 30 --output-dir /tmp/clips
    uv run python scripts/render_attribution_clips.py --answers  # Show answer key
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from rich.console import Console
from rich.table import Table

from rallycut.evaluation.db import get_connection
from rallycut.tracking.match_tracker import build_match_team_assignments
from rallycut.tracking.visual_attribution import (
    build_positions_by_frame,
    get_same_side_track_ids,
)

console = Console()

# Frames to show in montage (subsample 16 → 8 for readability)
# Show frames at: F-14, F-10, F-7, F-4, F-2, F-1, F, F+1
DISPLAY_OFFSETS = [-14, -10, -7, -4, -2, -1, 0, 1]
CROP_PAD = 0.2
CROP_SIZE = 128  # per-cell size in montage
BORDER = 2


@dataclass
class WithinTeamError:
    """A contact where proximity attribution picks the wrong same-side player."""
    rally_id: str
    video_id: str
    frame: int
    action: str
    gt_track_id: int
    prox_track_id: int  # proximity-attributed (wrong) player
    positions_json: list[dict[str, Any]]
    court_split_y: float | None
    start_ms: int
    fps: float


def find_within_team_errors(
    n: int = 30,
) -> list[WithinTeamError]:
    """Find contacts where proximity attributes to wrong same-side player."""
    query = """
        SELECT
            r.id AS rally_id,
            r.video_id,
            pt.action_ground_truth_json,
            pt.actions_json,
            pt.positions_json,
            pt.court_split_y,
            r.start_ms,
            pt.fps
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE pt.action_ground_truth_json IS NOT NULL
          AND pt.actions_json IS NOT NULL
          AND pt.positions_json IS NOT NULL
        ORDER BY r.video_id, r.start_ms
    """

    # Load team assignments
    teams_query = """
        SELECT id, match_analysis_json
        FROM videos WHERE match_analysis_json IS NOT NULL
    """
    all_teams: dict[str, dict[int, int]] = {}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(teams_query)
            for _vid, ma_json in cur.fetchall():
                if isinstance(ma_json, dict):
                    all_teams.update(build_match_team_assignments(ma_json, 0.0))

    errors: list[WithinTeamError] = []

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query)
            for row in cur.fetchall():
                (rally_id, video_id, gt_json, actions_json,
                 pos_json, split_y, start_ms, fps) = row

                if not gt_json or not actions_json or not pos_json:
                    continue

                # Build predicted action lookup by frame
                pred_by_frame: dict[int, int] = {}
                for a in actions_json.get("actions", []):
                    pred_by_frame[a.get("frame", -1)] = a.get("playerTrackId", -1)

                teams = all_teams.get(rally_id)

                for label in gt_json:
                    gt_tid = label.get("playerTrackId", -1)
                    if gt_tid < 0:
                        continue

                    frame = label["frame"]
                    action = label["action"]

                    # Find closest predicted action
                    pred_tid = -1
                    for delta in range(4):
                        for f in [frame + delta, frame - delta]:
                            if f in pred_by_frame:
                                pred_tid = pred_by_frame[f]
                                break
                        if pred_tid >= 0:
                            break

                    if pred_tid < 0 or pred_tid == gt_tid:
                        continue  # Correct or no prediction

                    # Check if within-team error (same side)
                    same_side = get_same_side_track_ids(
                        pos_json, gt_tid, frame, teams, split_y,
                    )
                    if pred_tid not in same_side:
                        continue  # Cross-team error, not within-team

                    errors.append(WithinTeamError(
                        rally_id=rally_id,
                        video_id=video_id,
                        frame=frame,
                        action=action,
                        gt_track_id=gt_tid,
                        prox_track_id=pred_tid,
                        positions_json=pos_json,
                        court_split_y=split_y,
                        start_ms=start_ms or 0,
                        fps=fps or 30.0,
                    ))

    console.print(f"  Found {len(errors)} within-team attribution errors")

    # Shuffle and take n, seeded for reproducibility
    random.seed(42)
    random.shuffle(errors)
    return errors[:n]


def extract_player_strip(
    cap: cv2.VideoCapture,
    positions_by_frame: dict[int, dict[str, float]],
    contact_frame: int,
    rally_start_frame: int,
    frame_w: int,
    frame_h: int,
) -> list[np.ndarray | None]:
    """Extract crops at DISPLAY_OFFSETS frames for one player.

    Returns list of crops (or None for missing frames).
    """
    crops: list[np.ndarray | None] = []

    for offset in DISPLAY_OFFSETS:
        rel_frame = contact_frame + offset

        # Find position with tolerance
        pos = None
        for f_search in [rel_frame, rel_frame - 1, rel_frame + 1]:
            pos = positions_by_frame.get(f_search)
            if pos is not None:
                break

        if pos is None:
            # Try wider search
            for delta in range(2, 8):
                pos = positions_by_frame.get(rel_frame - delta)
                if pos is None:
                    pos = positions_by_frame.get(rel_frame + delta)
                if pos is not None:
                    break

        if pos is None:
            crops.append(None)
            continue

        abs_frame = rally_start_frame + rel_frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, abs_frame)
        ret, frame = cap.read()
        if not ret or frame is None:
            crops.append(None)
            continue

        # Crop with padding
        cx = pos["x"] * frame_w
        cy = pos["y"] * frame_h
        bw = pos["width"] * frame_w
        bh = pos["height"] * frame_h
        pad_w = bw * CROP_PAD
        pad_h = bh * CROP_PAD

        x1 = max(0, int(cx - bw / 2 - pad_w))
        y1 = max(0, int(cy - bh / 2 - pad_h))
        x2 = min(frame_w, int(cx + bw / 2 + pad_w))
        y2 = min(frame_h, int(cy + bh / 2 + pad_h))

        if x2 - x1 < 8 or y2 - y1 < 8:
            crops.append(None)
            continue

        crop = frame[y1:y2, x1:x2]
        crop_resized = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
        crops.append(crop_resized)

    return crops


def render_montage(
    strips: list[tuple[str, list[np.ndarray | None]]],
    contact_idx: int,
    action: str,
    contact_frame: int,
) -> np.ndarray:
    """Render a montage image with multiple player strips.

    Args:
        strips: List of (label, crops) per player row.
        contact_idx: Contact number for labeling.
        action: Action type for labeling.
        contact_frame: Frame number of contact.

    Returns:
        BGR montage image.
    """
    n_cols = len(DISPLAY_OFFSETS)
    n_rows = len(strips)

    header_h = 30
    label_w = 100
    cell_h = CROP_SIZE + 20  # crop + frame label
    total_w = label_w + n_cols * (CROP_SIZE + BORDER) + BORDER
    total_h = header_h + n_rows * (cell_h + BORDER) + BORDER + 25  # +footer

    montage = np.full((total_h, total_w, 3), 40, dtype=np.uint8)  # dark gray bg

    # Header: contact info
    header_text = f"Contact #{contact_idx}  |  action: {action}  |  frame: {contact_frame}"
    cv2.putText(montage, header_text, (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    for row_idx, (label, crops) in enumerate(strips):
        y_start = header_h + row_idx * (cell_h + BORDER) + BORDER

        # Row label
        cv2.putText(montage, label, (5, y_start + CROP_SIZE // 2 + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

        for col_idx, crop in enumerate(crops):
            x_start = label_w + col_idx * (CROP_SIZE + BORDER) + BORDER

            if crop is not None:
                montage[y_start:y_start + CROP_SIZE,
                        x_start:x_start + CROP_SIZE] = crop
            else:
                # Gray placeholder
                cv2.putText(montage, "?",
                            (x_start + CROP_SIZE // 2 - 5, y_start + CROP_SIZE // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

            # Frame offset label
            offset = DISPLAY_OFFSETS[col_idx]
            offset_text = f"F{offset:+d}" if offset != 0 else "F=0"
            # Highlight contact frame
            color = (0, 255, 255) if offset == 0 else (150, 150, 150)
            cv2.putText(montage, offset_text,
                        (x_start + 5, y_start + CROP_SIZE + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # Footer: instruction
    footer_y = total_h - 8
    cv2.putText(montage, "Which player (A or B) touched the ball at F=0?",
                (10, footer_y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 200, 200), 1)

    return montage


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=30,
                        help="Number of clips to render")
    parser.add_argument("--output-dir", type=str, default="outputs/attribution_clips",
                        help="Output directory for montage images")
    parser.add_argument("--answers", action="store_true",
                        help="Print answer key (which player is GT)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    console.print("[bold]Finding within-team attribution errors...[/bold]")
    errors = find_within_team_errors(n=args.n)

    if not errors:
        console.print("[red]No within-team errors found.[/red]")
        return

    console.print(f"  Rendering {len(errors)} clips to {output_dir}/")

    from rallycut.evaluation.tracking.db import get_video_path

    # Group by video for efficient access
    by_video: dict[str, list[tuple[int, WithinTeamError]]] = defaultdict(list)
    for i, err in enumerate(errors):
        by_video[err.video_id].append((i, err))

    # Randomize A/B assignment per contact so GT isn't always in same row
    random.seed(123)
    gt_is_row_a: list[bool] = [random.random() < 0.5 for _ in errors]

    answer_key: list[dict[str, Any]] = []
    rendered = 0

    for video_id, err_list in sorted(by_video.items()):
        video_path = get_video_path(video_id)
        if video_path is None:
            console.print(f"  [yellow]Skipping {video_id[:8]}: no video[/yellow]")
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        try:
            for idx, err in err_list:
                rally_start_frame = int(err.start_ms / 1000.0 * fps)

                # Build position indices for both players
                gt_pos = build_positions_by_frame(err.positions_json, err.gt_track_id)
                prox_pos = build_positions_by_frame(err.positions_json, err.prox_track_id)

                if not gt_pos or not prox_pos:
                    continue

                gt_strip = extract_player_strip(
                    cap, gt_pos, err.frame, rally_start_frame, frame_w, frame_h,
                )
                prox_strip = extract_player_strip(
                    cap, prox_pos, err.frame, rally_start_frame, frame_w, frame_h,
                )

                # Check we have at least some frames
                gt_valid = sum(1 for c in gt_strip if c is not None)
                prox_valid = sum(1 for c in prox_strip if c is not None)
                if gt_valid < 4 or prox_valid < 4:
                    continue

                # Randomize which player is row A vs B
                if gt_is_row_a[idx]:
                    strips = [("Player A", gt_strip), ("Player B", prox_strip)]
                    answer = "A"
                else:
                    strips = [("Player A", prox_strip), ("Player B", gt_strip)]
                    answer = "B"

                montage = render_montage(strips, idx + 1, err.action, err.frame)

                out_path = output_dir / f"contact_{idx + 1:03d}_{err.action}.png"
                cv2.imwrite(str(out_path), montage)

                answer_key.append({
                    "contact": idx + 1,
                    "action": err.action,
                    "answer": answer,
                    "rally_id": err.rally_id[:8],
                    "frame": err.frame,
                    "video_id": err.video_id[:8],
                })

                rendered += 1
        finally:
            cap.release()

        console.print(f"  {video_id[:8]}: rendered {len(err_list)} clips")

    # Save answer key
    key_path = output_dir / "answer_key.json"
    with open(key_path, "w") as f:
        json.dump(answer_key, f, indent=2)

    console.print(f"\n  [green]Rendered {rendered} montages to {output_dir}/[/green]")
    console.print(f"  Answer key: {key_path}")
    console.print(f"\n  Review each image and note which player (A or B) touched the ball.")
    console.print(f"  Then compare to the answer key.\n")

    if args.answers:
        table = Table(title="Answer Key")
        table.add_column("#", justify="right")
        table.add_column("Action")
        table.add_column("Answer", style="bold green")
        table.add_column("Rally")
        table.add_column("Frame")

        for entry in answer_key:
            table.add_row(
                str(entry["contact"]),
                entry["action"],
                entry["answer"],
                entry["rally_id"],
                str(entry["frame"]),
            )
        console.print(table)


if __name__ == "__main__":
    main()
