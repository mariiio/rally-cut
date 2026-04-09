#!/usr/bin/env python3
"""Render attribution errors as multi-frame strips with pose keypoints.

For each error, shows 5 frames: contact-4, contact-2, contact, contact+2, contact+4
with ball position, player bboxes, and wrist keypoints (if RTMPose available).

Usage:
    uv run python scripts/render_attribution_errors_v2.py
    uv run python scripts/render_attribution_errors_v2.py --max-errors 30
    uv run python scripts/render_attribution_errors_v2.py --no-pose  # skip pose estimation
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent.parent / "outputs" / "attribution_errors_v2"
FRAME_OFFSETS = [-4, -2, 0, 2, 4]  # Frames around contact
STRIP_FRAME_W = 640  # Width per frame in strip
STRIP_FRAME_H = 360  # Height per frame in strip
HEADER_H = 30


def try_load_pose_model() -> Any:
    """Try to load RTMPose for wrist keypoint detection."""
    try:
        from mmpose.apis import MMPoseInferencer
        inferencer = MMPoseInferencer("rtmpose-m_8xb256-420e_body8-256x192")
        logger.info("RTMPose loaded successfully")
        return inferencer
    except Exception:
        pass

    try:
        import mediapipe as mp
        pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.3,
        )
        logger.info("MediaPipe Pose loaded as fallback")
        return ("mediapipe", pose)
    except Exception:
        pass

    logger.warning("No pose model available — rendering without wrist keypoints")
    return None


def get_wrist_positions(
    pose_model: Any,
    frame: np.ndarray,
    bbox: tuple[int, int, int, int],  # x1, y1, x2, y2
) -> list[tuple[int, int]]:
    """Get wrist positions (left, right) in pixel coordinates."""
    if pose_model is None:
        return []

    x1, y1, x2, y2 = bbox
    crop = frame[max(0, y1):y2, max(0, x1):x2]
    if crop.size == 0:
        return []

    if isinstance(pose_model, tuple) and pose_model[0] == "mediapipe":
        _, pose = pose_model
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)
        if not results.pose_landmarks:
            return []
        landmarks = results.pose_landmarks.landmark
        wrists = []
        # MediaPipe indices: 15=left_wrist, 16=right_wrist
        for idx in [15, 16]:
            lm = landmarks[idx]
            if lm.visibility > 0.3:
                px = x1 + int(lm.x * (x2 - x1))
                py = y1 + int(lm.y * (y2 - y1))
                wrists.append((px, py))
        return wrists
    else:
        # MMPose inferencer
        try:
            results = pose_model(frame, bboxes=[[x1, y1, x2, y2]])
            if results and results[0].get("keypoints"):
                kpts = results[0]["keypoints"][0]
                wrists = []
                # COCO indices: 9=left_wrist, 10=right_wrist
                for idx in [9, 10]:
                    if idx < len(kpts) and kpts[idx][2] > 0.3:
                        wrists.append((int(kpts[idx][0]), int(kpts[idx][1])))
                return wrists
        except Exception:
            pass
    return []


def render_strip(
    cap: cv2.VideoCapture,
    fps: float,
    frame_width: int,
    frame_height: int,
    rally_start_frame: int,
    contact_frame: int,
    ball_x: float,
    ball_y: float,
    candidates: list[tuple[int, float, dict]],
    chosen_tid: int,
    gt_tid: int,
    action_type: str,
    court_side: str,
    rally_id: str,
    error_type: str,
    pose_model: Any,
) -> np.ndarray | None:
    """Render a multi-frame strip for one error."""

    frames_data: list[tuple[np.ndarray, int]] = []

    for offset in FRAME_OFFSETS:
        target_frame = contact_frame + offset
        abs_frame = rally_start_frame + target_frame

        cap.set(cv2.CAP_PROP_POS_FRAMES, abs_frame)
        ret, frame = cap.read()
        if not ret or frame is None:
            return None
        frames_data.append((frame.copy(), offset))

    if len(frames_data) != len(FRAME_OFFSETS):
        return None

    # Build strip
    n_frames = len(FRAME_OFFSETS)
    strip_w = n_frames * STRIP_FRAME_W
    strip_h = HEADER_H + STRIP_FRAME_H
    strip = np.zeros((strip_h, strip_w, 3), dtype=np.uint8)

    bx_px = int(ball_x * frame_width)
    by_px = int(ball_y * frame_height)

    for i, (frame, offset) in enumerate(frames_data):
        # Annotate frame
        annotated = frame.copy()

        # Ball position (yellow circle)
        cv2.circle(annotated, (bx_px, by_px), 10, (0, 255, 255), 2)
        cv2.circle(annotated, (bx_px, by_px), 3, (0, 255, 255), -1)

        # Draw candidates
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
                color = (0, 255, 0)  # Green = GT
                thickness = 3
            elif tid == chosen_tid:
                color = (0, 0, 255)  # Red = wrong
                thickness = 3
            else:
                color = (180, 180, 180)
                thickness = 1

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

            # Wrist keypoints (only at contact frame and ±2)
            if abs(offset) <= 2 and pose_model is not None:
                wrists = get_wrist_positions(
                    pose_model, frame,
                    (max(0, x1), max(0, y1), min(frame_width, x2), min(frame_height, y2)),
                )
                for wx, wy in wrists:
                    # Wrist circle
                    cv2.circle(annotated, (wx, wy), 6, color, -1)
                    cv2.circle(annotated, (wx, wy), 6, (255, 255, 255), 1)
                    # Line from wrist to ball
                    cv2.line(annotated, (wx, wy), (bx_px, by_px), color, 1)

            # Line from bbox center to ball
            cx = int(px * frame_width)
            cy = int(py * frame_height)
            cv2.line(annotated, (cx, cy), (bx_px, by_px), color, 1)

        # Frame offset label
        label = f"{'CONTACT' if offset == 0 else f'{offset:+d}'}"
        if offset == 0:
            # Highlight contact frame
            cv2.rectangle(annotated, (0, 0), (frame_width - 1, frame_height - 1), (0, 255, 255), 3)

        # Resize to strip frame size
        resized = cv2.resize(annotated, (STRIP_FRAME_W, STRIP_FRAME_H))

        # Place in strip
        x_offset = i * STRIP_FRAME_W
        strip[HEADER_H:HEADER_H + STRIP_FRAME_H, x_offset:x_offset + STRIP_FRAME_W] = resized

        # Frame label in header
        label_color = (0, 255, 255) if offset == 0 else (200, 200, 200)
        cv2.putText(
            strip, label, (x_offset + STRIP_FRAME_W // 2 - 30, HEADER_H - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 1, cv2.LINE_AA,
        )

    # Title bar
    title = (
        f"{error_type} | {action_type} | side:{court_side} | "
        f"rally:{rally_id[:8]} | chosen:T{chosen_tid}(red) GT:T{gt_tid}(green)"
    )
    cv2.putText(
        strip, title, (10, 15),
        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
    )

    return strip


def main() -> None:
    parser = argparse.ArgumentParser(description="Render attribution error strips")
    parser.add_argument("--max-errors", type=int, default=60)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--no-pose", action="store_true", help="Skip pose estimation")
    parser.add_argument("--type", choices=["cross-team", "within-team", "all"], default="all")
    args = parser.parse_args()

    from rallycut.evaluation.db import get_connection
    from rallycut.evaluation.tracking.db import get_video_path

    # Load pose model
    pose_model = None if args.no_pose else try_load_pose_model()

    # Load GT + contacts (same logic as render_attribution_errors.py)
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

    errors: list[dict[str, Any]] = []

    for video_id, rally_id, start_ms, gt_json, contacts_json, actions_json, positions_json in rows:
        video_id = str(video_id)
        rally_id = str(rally_id)

        gt_actions = gt_json if isinstance(gt_json, list) else gt_json.get("actions", [])
        pred_contacts = contacts_json.get("contacts", []) if contacts_json else []

        if not gt_actions or not pred_contacts:
            continue

        pos_by_frame_track: dict[tuple[int, int], dict] = {}
        for p in (positions_json or []):
            fn = p.get("frameNumber", p.get("frame_number", 0))
            tid = p.get("trackId", p.get("track_id", 0))
            pos_by_frame_track[(fn, tid)] = p

        tolerance = 5
        for gt_act in gt_actions:
            gt_frame = gt_act.get("frame", 0)
            gt_tid = gt_act.get("playerTrackId", gt_act.get("player_track_id", -1))
            gt_action = gt_act.get("action", "unknown")
            if gt_tid < 0:
                continue

            best_pred = None
            best_dist = tolerance + 1
            for pc in pred_contacts:
                d = abs(pc.get("frame", 0) - gt_frame)
                if d < best_dist:
                    best_dist = d
                    best_pred = pc

            if best_pred is None or best_dist > tolerance:
                continue

            pred_tid = best_pred.get("playerTrackId", -1)
            if pred_tid == gt_tid:
                continue

            candidates_raw = best_pred.get("playerCandidates", [])
            pred_frame = best_pred.get("frame", 0)
            ball_x = best_pred.get("ballX", 0.0)
            ball_y = best_pred.get("ballY", 0.0)
            court_side = best_pred.get("courtSide", "unknown")

            t2p = team_by_rally.get(rally_id, {})
            gt_player = t2p.get(gt_tid, -1)
            pred_player = t2p.get(pred_tid, -1)
            if gt_player > 0 and pred_player > 0:
                gt_team = 0 if gt_player <= 2 else 1
                pred_team = 0 if pred_player <= 2 else 1
                error_type = "within-team" if gt_team == pred_team else "cross-team"
            else:
                error_type = "unknown-team"

            if args.type != "all" and error_type != args.type:
                continue

            # Build candidate positions
            cand_info: list[tuple[int, float, dict]] = []
            for c in candidates_raw:
                if c[1] is None:
                    continue
                tid = int(c[0])
                dist = float(c[1])
                pos = None
                for delta in range(6):
                    for fn in [pred_frame + delta, pred_frame - delta]:
                        pos = pos_by_frame_track.get((fn, tid))
                        if pos is not None:
                            break
                    if pos is not None:
                        break
                if pos is not None:
                    cand_info.append((tid, dist, pos))

            if not cand_info:
                continue

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
                "candidates": cand_info,
            })

    errors.sort(key=lambda e: (0 if e["error_type"] == "cross-team" else 1, e["video_id"]))
    errors = errors[:args.max_errors]

    logger.info("Rendering %d error strips (pose: %s)", len(errors), "yes" if pose_model else "no")

    args.output_dir.mkdir(parents=True, exist_ok=True)

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

            strip = render_strip(
                cap, fps, fw, fh,
                rally_start_frame, err["frame"],
                err["ball_x"], err["ball_y"],
                err["candidates"], err["chosen_tid"], err["gt_tid"],
                err["action_type"], err["court_side"],
                err["rally_id"], err["error_type"],
                pose_model,
            )

            if strip is None:
                continue

            filename = (
                f"{err['error_type']}_{err['video_id'][:8]}_"
                f"{err['rally_id'][:8]}_f{err['frame']:05d}.jpg"
            )
            cv2.imwrite(
                str(args.output_dir / filename), strip,
                [cv2.IMWRITE_JPEG_QUALITY, 90],
            )
            n_rendered += 1

        cap.release()
        logger.info("  %s: %d strips", video_id[:8], len(video_errors))

    logger.info("Rendered %d strips to %s", n_rendered, args.output_dir)


if __name__ == "__main__":
    main()
