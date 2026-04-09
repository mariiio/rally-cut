#!/usr/bin/env python3
"""Render annotated frames for player attribution errors.

For each attribution error, saves an image showing:
- The video frame at contact time
- Ball position (yellow circle)
- All candidate player bboxes with track IDs and distances
- Chosen player bbox (red = wrong)
- GT player bbox (green = correct)
- Text overlay: action type, court_side, rally_id, error type

Usage:
    uv run python scripts/render_attribution_errors.py
    uv run python scripts/render_attribution_errors.py --max-errors 30
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

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "attribution_errors"


def render_error_frame(
    frame: np.ndarray,
    ball_x: float,
    ball_y: float,
    candidates: list[tuple[int, float, dict]],  # (tid, dist, pos_dict)
    chosen_tid: int,
    gt_tid: int,
    action_type: str,
    court_side: str,
    rally_id: str,
    error_type: str,
    frame_width: int,
    frame_height: int,
) -> np.ndarray:
    """Draw annotations on frame."""
    img = frame.copy()

    # Ball position
    bx_px = int(ball_x * frame_width)
    by_px = int(ball_y * frame_height)
    cv2.circle(img, (bx_px, by_px), 12, (0, 255, 255), 3)  # Yellow
    cv2.circle(img, (bx_px, by_px), 4, (0, 255, 255), -1)

    # Draw candidate bboxes
    for tid, dist, pos in candidates:
        px = pos.get("x", 0.0)
        py = pos.get("y", 0.0)
        pw = pos.get("width", 0.0)
        ph = pos.get("height", 0.0)

        x1 = int((px - pw / 2) * frame_width)
        y1 = int((py - ph / 2) * frame_height)
        x2 = int((px + pw / 2) * frame_width)
        y2 = int((py + ph / 2) * frame_height)

        if tid == gt_tid:
            color = (0, 255, 0)  # Green = GT correct player
            thickness = 3
            label = f"T{tid} GT d={dist:.3f}"
        elif tid == chosen_tid:
            color = (0, 0, 255)  # Red = wrong chosen player
            thickness = 3
            label = f"T{tid} CHOSEN d={dist:.3f}"
        else:
            color = (200, 200, 200)  # Gray = other candidate
            thickness = 1
            label = f"T{tid} d={dist:.3f}"

        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Label above bbox
        label_y = max(y1 - 8, 15)
        cv2.putText(
            img, label, (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
        )

        # Line from ball to player center
        cx = int(px * frame_width)
        cy = int(py * frame_height)
        cv2.line(img, (bx_px, by_px), (cx, cy), color, 1)

    # Text overlay
    lines = [
        f"Action: {action_type}  Side: {court_side}",
        f"Rally: {rally_id[:8]}  Error: {error_type}",
        f"Chosen: T{chosen_tid} (wrong)  GT: T{gt_tid} (correct)",
    ]
    for i, line in enumerate(lines):
        y = 25 + i * 22
        cv2.putText(
            img, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA,
        )
        cv2.putText(
            img, line, (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA,
        )

    return img


def main() -> None:
    parser = argparse.ArgumentParser(description="Render attribution error frames")
    parser.add_argument("--max-errors", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    from rallycut.evaluation.db import get_connection
    from rallycut.evaluation.tracking.db import get_video_path

    # Load GT action labels
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT r.video_id, r.id as rally_id, r.start_ms,
                       pt.action_ground_truth_json, pt.contacts_json,
                       pt.actions_json, pt.positions_json
                FROM rallies r
                JOIN player_tracks pt ON pt.rally_id = r.id
                WHERE pt.action_ground_truth_json IS NOT NULL
                  AND pt.contacts_json IS NOT NULL
                  AND pt.positions_json IS NOT NULL
            """)
            rows = cur.fetchall()

    # Load match team assignments
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id, match_analysis_json FROM videos
                WHERE match_analysis_json IS NOT NULL
            """)
            ma_rows = cur.fetchall()

    team_by_rally: dict[str, dict[int, int]] = {}
    for vid, ma in ma_rows:
        if not ma:
            continue
        for entry in ma.get("rallies", []):
            rid = entry.get("rallyId", entry.get("rally_id", ""))
            t2p = entry.get("trackToPlayer", entry.get("track_to_player", {}))
            if rid and t2p:
                team_by_rally[rid] = {int(k): int(v) for k, v in t2p.items()}

    logger.info("Loaded %d rallies with action GT", len(rows))

    errors: list[dict[str, Any]] = []

    for video_id, rally_id, start_ms, gt_json, contacts_json, actions_json, positions_json in rows:
        video_id = str(video_id)
        rally_id = str(rally_id)

        gt_actions = gt_json if isinstance(gt_json, list) else gt_json.get("actions", [])
        pred_contacts = contacts_json.get("contacts", []) if contacts_json else []
        pred_actions = actions_json.get("actions", []) if actions_json else []

        if not gt_actions or not pred_contacts:
            continue

        # Build position lookup: (frame, track_id) -> position dict
        pos_by_frame_track: dict[tuple[int, int], dict] = {}
        for p in (positions_json or []):
            fn = p.get("frameNumber", p.get("frame_number", 0))
            tid = p.get("trackId", p.get("track_id", 0))
            pos_by_frame_track[(fn, tid)] = p

        # Match GT to predictions by frame proximity (±5 frames = ±167ms)
        tolerance = 5
        for gt_act in gt_actions:
            gt_frame = gt_act.get("frame", 0)
            gt_tid = gt_act.get("playerTrackId", gt_act.get("player_track_id", -1))
            gt_action = gt_act.get("action", "unknown")

            if gt_tid < 0:
                continue

            # Find closest predicted contact
            best_pred = None
            best_dist = tolerance + 1
            for pc in pred_contacts:
                pf = pc.get("frame", 0)
                d = abs(pf - gt_frame)
                if d < best_dist:
                    best_dist = d
                    best_pred = pc

            if best_pred is None or best_dist > tolerance:
                continue

            pred_tid = best_pred.get("playerTrackId", -1)
            if pred_tid == gt_tid:
                continue  # Correct — skip

            # This is an error
            candidates_raw = best_pred.get("playerCandidates", [])
            ball_x = best_pred.get("ballX", 0.0)
            ball_y = best_pred.get("ballY", 0.0)
            court_side = best_pred.get("courtSide", "unknown")
            pred_frame = best_pred.get("frame", 0)

            # Classify error type
            t2p = team_by_rally.get(rally_id, {})
            gt_player = t2p.get(gt_tid, -1)
            pred_player = t2p.get(pred_tid, -1)

            if gt_player > 0 and pred_player > 0:
                gt_team = 0 if gt_player <= 2 else 1
                pred_team = 0 if pred_player <= 2 else 1
                error_type = "within-team" if gt_team == pred_team else "cross-team"
            else:
                error_type = "unknown-team"

            errors.append({
                "video_id": video_id,
                "rally_id": rally_id,
                "start_ms": start_ms,
                "frame": pred_frame,
                "ball_x": ball_x,
                "ball_y": ball_y,
                "chosen_tid": pred_tid,
                "gt_tid": gt_tid,
                "action_type": gt_action,
                "court_side": court_side,
                "error_type": error_type,
                "candidates": candidates_raw,
                "pos_by_frame_track": pos_by_frame_track,
            })

    logger.info("Found %d attribution errors", len(errors))

    # Sort: cross-team first, then within-team
    errors.sort(key=lambda e: (0 if e["error_type"] == "cross-team" else 1, e["video_id"]))
    errors = errors[:args.max_errors]

    # Render frames
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Group by video for efficient video reading
    by_video: dict[str, list[dict]] = {}
    for err in errors:
        by_video.setdefault(err["video_id"], []).append(err)

    n_rendered = 0
    for video_id, video_errors in by_video.items():
        video_path = get_video_path(video_id)
        if video_path is None:
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        for err in video_errors:
            rally_start_frame = int((err["start_ms"] or 0) / 1000.0 * fps)
            abs_frame = rally_start_frame + err["frame"]

            cap.set(cv2.CAP_PROP_POS_FRAMES, abs_frame)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            # Build candidate info with positions
            cand_info: list[tuple[int, float, dict]] = []
            for c in err["candidates"]:
                if c[1] is None:
                    continue
                tid = int(c[0])
                dist = float(c[1])
                # Find position near contact frame
                pos = None
                for delta in range(6):
                    for fn in [err["frame"] + delta, err["frame"] - delta]:
                        pos = err["pos_by_frame_track"].get((fn, tid))
                        if pos is not None:
                            break
                    if pos is not None:
                        break
                if pos is not None:
                    cand_info.append((tid, dist, pos))

            if not cand_info:
                continue

            annotated = render_error_frame(
                frame, err["ball_x"], err["ball_y"],
                cand_info, err["chosen_tid"], err["gt_tid"],
                err["action_type"], err["court_side"],
                err["rally_id"], err["error_type"],
                fw, fh,
            )

            filename = (
                f"{err['error_type']}_{err['video_id'][:8]}_"
                f"{err['rally_id'][:8]}_f{err['frame']:05d}.jpg"
            )
            cv2.imwrite(
                str(args.output_dir / filename), annotated,
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
            n_rendered += 1

        cap.release()

    logger.info("Rendered %d error frames to %s", n_rendered, args.output_dir)

    # Summary
    cross = sum(1 for e in errors[:n_rendered] if e["error_type"] == "cross-team")
    within = sum(1 for e in errors[:n_rendered] if e["error_type"] == "within-team")
    logger.info("  Cross-team: %d, Within-team: %d", cross, within)


if __name__ == "__main__":
    main()
