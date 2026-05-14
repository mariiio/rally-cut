"""Re-run contact detection + action classification for all tracked rallies.

Uses stored ball/player positions (from tracking) to re-detect contacts with
the current pipeline (classifier + threshold) and re-classify actions. Saves
updated contacts_json and actions_json back to the database.

This fixes stale stored actions from older pipeline versions.

Usage:
    cd analysis
    uv run python scripts/redetect_all_actions.py                # Dry run
    uv run python scripts/redetect_all_actions.py --apply        # Write to DB
    uv run python scripts/redetect_all_actions.py --video <id>   # Single video
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, cast

import cv2

from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.db import get_connection
from rallycut.evaluation.tracking.db import load_court_calibration
from rallycut.evaluation.video_resolver import VideoResolver
from rallycut.tracking.action_classifier import ACTION_PIPELINE_VERSION, classify_rally_actions
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.block_reclassification import estimate_net_y_image
from rallycut.tracking.contact_detector import CONTACT_PIPELINE_VERSION, detect_contacts
from rallycut.tracking.match_tracker import build_match_team_assignments
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos
from rallycut.tracking.sequence_action_runtime import get_sequence_probs

# COCO keypoint indices (left/right wrist).
KPT_LEFT_WRIST = 9
KPT_RIGHT_WRIST = 10
WRIST_CONF_FLOOR = 0.30
POSE_TRACK_IOU_THRESHOLD = 0.30


def _bbox_iou(b1: tuple[float, float, float, float], b2: tuple[float, float, float, float]) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


_POSE_MODEL_CACHE: dict[str, Any] = {}


def _get_pose_model() -> Any:
    if "model" not in _POSE_MODEL_CACHE:
        from ultralytics import YOLO
        weight_path = Path(__file__).resolve().parent.parent / "yolo11s-pose.pt"
        if weight_path.exists():
            _POSE_MODEL_CACHE["model"] = YOLO(str(weight_path))
        else:
            _POSE_MODEL_CACHE["model"] = YOLO("yolo11s-pose.pt")
    return _POSE_MODEL_CACHE["model"]


def _extract_wrist_y_for_contacts(
    video_path: Path,
    rally_start_ms: int,
    fps: float,
    contact_frames: list[int],
    player_positions: list[PlayerPos],
) -> dict[tuple[int, int], float]:
    """Run yolo11s-pose at each contact frame; return {(frame, tid): wrist_y}.

    Matches pose detections to tracked player bboxes by IoU. Picks the
    higher of left/right wrist (smaller image-y) when both pass the
    confidence floor.
    """
    if not contact_frames:
        return {}

    # Build position lookup at each contact frame.
    pos_by_frame: dict[int, list[PlayerPos]] = {}
    needed = set(contact_frames)
    for pp in player_positions:
        if pp.frame_number in needed and pp.track_id >= 0:
            pos_by_frame.setdefault(pp.frame_number, []).append(pp)
    if not pos_by_frame:
        return {}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {}
    try:
        img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        abs_offset = int(rally_start_ms / 1000 * fps)
        sorted_frames = sorted(pos_by_frame.keys())
        first_abs = abs_offset + sorted_frames[0]
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_abs)
        current_abs = first_abs

        pose_model = _get_pose_model()
        wrist_by_key: dict[tuple[int, int], float] = {}

        for rally_frame in sorted_frames:
            target_abs = abs_offset + rally_frame
            while current_abs < target_abs:
                cap.grab()
                current_abs += 1
            ret, frame_bgr = cap.read()
            current_abs += 1
            if not ret or frame_bgr is None:
                continue

            results = pose_model.predict(frame_bgr, verbose=False, imgsz=1280)
            if not results:
                continue
            res = results[0]
            if res.keypoints is None or res.boxes is None:
                continue
            kps_all = res.keypoints.data.cpu().numpy()
            boxes = res.boxes.xyxy.cpu().numpy()
            if len(kps_all) == 0:
                continue

            boxes_norm = boxes.copy()
            boxes_norm[:, [0, 2]] /= img_w
            boxes_norm[:, [1, 3]] /= img_h

            det_boxes = [
                (
                    float(boxes_norm[i, 0]), float(boxes_norm[i, 1]),
                    float(boxes_norm[i, 2]), float(boxes_norm[i, 3]),
                )
                for i in range(len(boxes_norm))
            ]

            for pp in pos_by_frame.get(rally_frame, []):
                cx, cy = pp.x, pp.y
                w, h = pp.width, pp.height
                pp_box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
                best_iou = 0.0
                best_idx = -1
                for di, db in enumerate(det_boxes):
                    iou = _bbox_iou(pp_box, db)
                    if iou >= POSE_TRACK_IOU_THRESHOLD and iou > best_iou:
                        best_iou = iou
                        best_idx = di
                if best_idx < 0:
                    continue
                kps = kps_all[best_idx]
                # Higher of left/right wrist (smaller image-y).
                best_wy: float | None = None
                for kp_idx in (KPT_LEFT_WRIST, KPT_RIGHT_WRIST):
                    wy = float(kps[kp_idx, 1]) / img_h
                    wc = float(kps[kp_idx, 2])
                    if wc < WRIST_CONF_FLOOR:
                        continue
                    if best_wy is None or wy < best_wy:
                        best_wy = wy
                if best_wy is not None:
                    wrist_by_key[(rally_frame, pp.track_id)] = best_wy
        return wrist_by_key
    finally:
        cap.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-detect contacts + actions for all rallies")
    parser.add_argument("--apply", action="store_true", help="Write changes to DB (default: dry run)")
    parser.add_argument("--video", type=str, help="Only process this video ID")
    parser.add_argument("--rally-id", type=str, help="Only process this rally ID")
    args = parser.parse_args()

    use_block_reclass = os.getenv("USE_BLOCK_RECLASSIFICATION", "0") == "1"
    if use_block_reclass:
        print("A3 BLOCK reclassification: ENABLED (USE_BLOCK_RECLASSIFICATION=1)")
    else:
        print("A3 BLOCK reclassification: disabled (USE_BLOCK_RECLASSIFICATION=0)")

    # Load match team assignments and rally data
    where_clauses = ["pt.ball_positions_json IS NOT NULL"]
    params: list[str] = []
    if args.video:
        where_clauses.append("r.video_id = %s")
        params.append(args.video)
    if args.rally_id:
        where_clauses.append("r.id = %s")
        params.append(args.rally_id)
    where_sql = " AND ".join(where_clauses)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT v.id, v.match_analysis_json FROM videos v "
                "WHERE v.match_analysis_json IS NOT NULL"
            )
            # Derive per-rally team assignments via the canonical match_tracker
            # helper. min_confidence=0.0 mirrors `reattribute-actions`'s "all_teams"
            # path, which is what stamps teamAssignments onto every rally.
            # The previous implementation here was reading `match_analysis_json.
            # team_assignments` which is a different (legacy) schema and now
            # returns 0 rallies — leaving teamAssignments null fleet-wide.
            match_teams_by_rally: dict[str, dict[int, int]] = {}
            for vid, mj_raw in cur.fetchall():
                mj = cast(dict[str, Any], mj_raw)
                if not mj:
                    continue
                match_teams_by_rally.update(
                    build_match_team_assignments(mj, min_confidence=0.0)
                )

            cur.execute(f"""
                SELECT r.id, r.video_id, pt.id as pt_id,
                       pt.ball_positions_json, pt.positions_json,
                       pt.frame_count, pt.court_split_y,
                       pt.actions_json, r.start_ms
                FROM rallies r
                JOIN player_tracks pt ON pt.rally_id = r.id
                WHERE {where_sql}
                ORDER BY r.video_id, r.start_ms
            """, params)
            rows = cur.fetchall()

    print(f"Found {len(rows)} rallies with ball positions")
    if not args.apply:
        print("  DRY RUN — use --apply to write changes to DB\n")

    # Load court calibrations
    calibrators: dict[str, CourtCalibrator | None] = {}
    net_y_by_video: dict[str, float | None] = {}
    video_meta_cache: dict[str, dict[str, Any]] = {}
    resolver: VideoResolver | None = None  # lazy-init when block reclass is on

    t_start = time.monotonic()
    updated = 0
    skipped = 0
    errors = 0

    for i, row in enumerate(rows):
        rally_id = str(row[0])
        video_id = str(row[1])
        pt_id = cast(int, row[2])
        ball_json = cast(list[dict[str, Any]], row[3])
        positions_json = cast(list[dict[str, Any]] | None, row[4])
        frame_count = cast(int | None, row[5])
        court_split_y = cast(float | None, row[6])
        existing_actions_json = cast(dict[str, Any] | None, row[7]) or {}
        rally_start_ms = int(cast(int | None, row[8]) or 0)

        # Load court calibration (cached per video)
        if video_id not in calibrators:
            corners = load_court_calibration(video_id)
            if corners and len(corners) == 4:
                cal = CourtCalibrator()
                cal.calibrate([(c["x"], c["y"]) for c in corners])
                calibrators[video_id] = cal
                # net_y_image (normalized) for the A3 block helper.
                net_y_by_video[video_id] = estimate_net_y_image(corners)
            else:
                calibrators[video_id] = None
                net_y_by_video[video_id] = None

        # Convert DB dicts to typed objects
        ball_positions = [
            BallPos(
                frame_number=bp["frameNumber"], x=bp["x"], y=bp["y"],
                confidence=bp.get("confidence", 1.0),
            )
            for bp in ball_json
            if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
        ]

        if not ball_positions:
            skipped += 1
            continue

        player_positions = []
        if positions_json:
            player_positions = [
                PlayerPos(
                    frame_number=pp["frameNumber"], track_id=pp["trackId"],
                    x=pp["x"], y=pp["y"], width=pp["width"], height=pp["height"],
                    confidence=pp.get("confidence", 1.0),
                )
                for pp in positions_json
            ]

        match_teams = match_teams_by_rally.get(rally_id)

        try:
            # Compute MS-TCN++ probs once. Required for v1.1 synthetic-serve
            # placement AND for apply_sequence_override (the post-Viterbi
            # action-type correction). Returns None if weights are missing —
            # in that case both effects degrade gracefully.
            sequence_probs = get_sequence_probs(
                ball_positions, player_positions, court_split_y,
                frame_count or 0, match_teams,
                calibrator=calibrators.get(video_id),
            )

            contacts = detect_contacts(
                ball_positions=ball_positions,
                player_positions=player_positions,
                net_y=court_split_y,
                frame_count=frame_count or None,
                court_calibrator=calibrators.get(video_id),
                team_assignments=match_teams,
                sequence_probs=sequence_probs,
            )

            # A3: pose extraction at contact frames (gated by env flag).
            pose_wrist: dict[tuple[int, int], float] | None = None
            net_y_image_norm: float | None = None
            if use_block_reclass and contacts.contacts:
                net_y_image_norm = net_y_by_video.get(video_id)
                if video_id not in video_meta_cache:
                    with get_connection() as vconn, vconn.cursor() as vcur:
                        vcur.execute(
                            "SELECT fps, s3_key, proxy_s3_key, processed_s3_key, "
                            "content_hash FROM videos WHERE id = %s",
                            (video_id,),
                        )
                        vrow = vcur.fetchone()
                    if vrow is not None:
                        video_meta_cache[video_id] = {
                            "fps": float(cast(float, vrow[0])) if vrow[0] is not None else 30.0,
                            "s3_key": vrow[1],
                            "proxy_s3_key": vrow[2],
                            "processed_s3_key": vrow[3],
                            "content_hash": vrow[4],
                        }
                vmeta = video_meta_cache.get(video_id)
                video_path: Path | None = None
                if vmeta and vmeta.get("content_hash"):
                    if resolver is None:
                        resolver = VideoResolver()
                    for label_key in ("proxy_s3_key", "s3_key", "processed_s3_key"):
                        sk = vmeta.get(label_key)
                        if not sk:
                            continue
                        try:
                            video_path = resolver.resolve(sk, vmeta["content_hash"])
                            break
                        except Exception as exc:
                            print(f"    resolve {label_key} failed: {exc}")
                if video_path is not None:
                    contact_frames = sorted({c.frame for c in contacts.contacts})
                    pose_wrist = _extract_wrist_y_for_contacts(
                        video_path=video_path,
                        rally_start_ms=rally_start_ms,
                        fps=(vmeta or {}).get("fps", 30.0),
                        contact_frames=contact_frames,
                        player_positions=player_positions,
                    )

            rally_actions = classify_rally_actions(
                contacts, rally_id,
                use_classifier=True,
                team_assignments=match_teams,
                match_team_assignments=match_teams,
                sequence_probs=sequence_probs,
                net_y_image=net_y_image_norm,
                pose_wrist_by_frame_tid=pose_wrist,
            )

            # Serialize via RallyActions.to_dict() so teamAssignments + servingTeam
            # are emitted from the current pipeline output (was previously dropped:
            # this script only wrote {"actions": [...]} which stripped both fields
            # on every deploy). Layer over the existing actions_json so any
            # downstream-stamped fields (formation hints, etc.) survive.
            new_contacts_json = contacts.to_dict()
            new_actions_json = {**existing_actions_json, **rally_actions.to_dict()}

            n_contacts = len(contacts.contacts)
            n_actions = len(rally_actions.actions)

            if args.apply:
                with get_connection() as wconn:
                    with wconn.cursor() as wcur:
                        wcur.execute(
                            "UPDATE player_tracks SET "
                            "contacts_json = %s, actions_json = %s, "
                            "contacts_pipeline_version = %s, "
                            "actions_pipeline_version = %s "
                            "WHERE id = %s",
                            (
                                json.dumps(new_contacts_json),
                                json.dumps(new_actions_json),
                                CONTACT_PIPELINE_VERSION,
                                ACTION_PIPELINE_VERSION,
                                pt_id,
                            ),
                        )
                    wconn.commit()

            updated += 1
            elapsed = time.monotonic() - t_start
            print(
                f"  [{i+1}/{len(rows)}] {rally_id[:8]}: "
                f"{n_contacts} contacts, {n_actions} actions ({elapsed:.1f}s)"
            )

        except Exception as e:
            errors += 1
            print(f"  ERROR {rally_id[:8]}: {e}")

    elapsed = time.monotonic() - t_start
    print(f"\nDone: {updated} updated, {skipped} skipped, {errors} errors ({elapsed:.1f}s)")
    if not args.apply:
        print("  DRY RUN — no changes written. Use --apply to write.")


if __name__ == "__main__":
    main()
