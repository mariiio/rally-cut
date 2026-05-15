#!/usr/bin/env python3
"""Phase 1: Standalone pose-feature extraction probe.

For each GT row across the trusted-21 corpus, runs YOLO-Pose on the contact
frame's neighborhood (±5 frames) for each primary track candidate, then
computes 6 pose-based features per candidate:

  - wrist_velocity_max: max |Δwrist| across f-3..f+3 (striking-hand signal)
  - wrist_to_ball_min: min wrist-to-ball distance across f-2..f+2
  - body_orientation_diff: angle between body-facing-direction (shoulder-vector
    perpendicular) and direction-to-ball
  - arms_raised: 1.0 if both wrists are above both shoulders at contact
  - wrist_post_alignment: max dot(wrist_velocity, post_contact_ball_velocity)
    — striking-hand sends ball where it goes
  - pose_confidence_mean: mean keypoint confidence (gates noisy poses)

Outputs:
  analysis/reports/pose_features_2026_05_15/features.csv
  analysis/reports/pose_features_2026_05_15/summary.md

No DB writes. Pure-read, pure-compute. Phase 1 of the pose-dynamics rollout.

Usage:
    cd analysis && uv run python scripts/probe_pose_features_2026_05_15.py
    cd analysis && uv run python scripts/probe_pose_features_2026_05_15.py --video keke  # single video
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import psycopg

from rallycut.tracking.pose_attribution.features import (
    KPT_LEFT_SHOULDER, KPT_RIGHT_SHOULDER, KPT_LEFT_WRIST, KPT_RIGHT_WRIST,
)
from rallycut.tracking.pose_attribution.pose_cache import (
    _get_pose_model, enrich_positions_with_pose,
)
from rallycut.tracking.player_tracker import PlayerPosition

DB_DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5436/rallycut",
)
TRUSTED_CODENAMES = (
    "titi", "toto", "lulu", "wawa", "caco", "cece", "cici", "cuco",
    "gaga", "kaka", "kiki", "juju", "yeye", "keke",
    "gigi", "gugu", "mame", "meme", "mimi", "moma", "mumu",
)
VIDEOS_DIR = Path("/tmp/rca_videos")
REPORT_DIR = Path(
    "/Users/mario/Personal/Projects/RallyCut/analysis/reports/pose_features_2026_05_15"
)
KPT_VIS_THRESHOLD = 0.3
FRAME_TOLERANCE = 5
WINDOW_HALF = 5


def positions_from_json(positions_json: list[dict]) -> list[PlayerPosition]:
    return [
        PlayerPosition(
            frame_number=int(p.get("frameNumber", -1)),
            track_id=int(p.get("trackId", -1)),
            x=float(p.get("x", 0)),
            y=float(p.get("y", 0)),
            width=float(p.get("width", 0)),
            height=float(p.get("height", 0)),
            confidence=float(p.get("confidence", 0)),
        )
        for p in positions_json
    ]


def _wrist_xy(p: PlayerPosition, which: str) -> tuple[float, float, float] | None:
    """Return (x, y, conf) of left or right wrist; None if not visible."""
    if p.keypoints is None or len(p.keypoints) < 17:
        return None
    idx = KPT_LEFT_WRIST if which == "left" else KPT_RIGHT_WRIST
    kx, ky, kc = p.keypoints[idx]
    if kc < KPT_VIS_THRESHOLD:
        return None
    return float(kx), float(ky), float(kc)


def _shoulder_xy(p: PlayerPosition, which: str) -> tuple[float, float, float] | None:
    if p.keypoints is None or len(p.keypoints) < 17:
        return None
    idx = KPT_LEFT_SHOULDER if which == "left" else KPT_RIGHT_SHOULDER
    kx, ky, kc = p.keypoints[idx]
    if kc < KPT_VIS_THRESHOLD:
        return None
    return float(kx), float(ky), float(kc)


def compute_pose_features(
    positions: list[PlayerPosition], track_id: int,
    contact_frame: int, ball_x: float, ball_y: float,
    post_ball_x: float | None, post_ball_y: float | None,
) -> dict[str, float]:
    """Compute 6 pose features for one candidate."""
    track_positions = sorted(
        [p for p in positions if p.track_id == track_id
         and abs(p.frame_number - contact_frame) <= WINDOW_HALF],
        key=lambda p: p.frame_number,
    )
    # Wrist positions per frame
    wrist_pos: dict[int, tuple[float, float]] = {}
    confs: list[float] = []
    arms_raised_at_contact = 0.0
    for p in track_positions:
        # Pick whichever wrist is closer to the ball
        lw = _wrist_xy(p, "left")
        rw = _wrist_xy(p, "right")
        best_w: tuple[float, float] | None = None
        best_d = float("inf")
        for w in (lw, rw):
            if w is None:
                continue
            d = math.hypot(w[0] - ball_x, w[1] - ball_y)
            if d < best_d:
                best_d = d
                best_w = (w[0], w[1])
            confs.append(w[2])
        if best_w is not None:
            wrist_pos[p.frame_number] = best_w
        # Shoulder confidences
        ls = _shoulder_xy(p, "left")
        rs = _shoulder_xy(p, "right")
        for s in (ls, rs):
            if s is not None:
                confs.append(s[2])
        # Arms raised at contact: both wrists above both shoulders (y < y)
        if abs(p.frame_number - contact_frame) <= 2:
            if lw and rw and ls and rs:
                if lw[1] < ls[1] and rw[1] < rs[1]:
                    arms_raised_at_contact = 1.0

    # wrist_velocity_max: max consecutive-frame wrist displacement
    sorted_frames = sorted(wrist_pos.keys())
    wrist_velocity_max = 0.0
    for i in range(len(sorted_frames) - 1):
        f1, f2 = sorted_frames[i], sorted_frames[i + 1]
        if f2 - f1 > 3:
            continue
        x1, y1 = wrist_pos[f1]
        x2, y2 = wrist_pos[f2]
        d = math.hypot(x2 - x1, y2 - y1) / max(1, f2 - f1)
        if d > wrist_velocity_max:
            wrist_velocity_max = d

    # wrist_to_ball_min: minimum wrist-to-ball over the contact window
    wrist_to_ball_min = float("inf")
    for f, (wx, wy) in wrist_pos.items():
        if abs(f - contact_frame) <= 2:
            d = math.hypot(wx - ball_x, wy - ball_y)
            if d < wrist_to_ball_min:
                wrist_to_ball_min = d
    if not math.isfinite(wrist_to_ball_min):
        wrist_to_ball_min = 1.0  # missing — treat as far

    # body_orientation_diff: angle between body-perpendicular and direction-to-ball
    # Find pose at contact frame
    p_contact = next((p for p in track_positions if abs(p.frame_number - contact_frame) <= 1), None)
    body_orientation_diff = math.pi  # default to "facing away"
    if p_contact is not None and p_contact.keypoints is not None:
        ls = _shoulder_xy(p_contact, "left")
        rs = _shoulder_xy(p_contact, "right")
        if ls is not None and rs is not None:
            # Shoulder vector (left-to-right), perpendicular = body-facing direction
            sx = rs[0] - ls[0]
            sy = rs[1] - ls[1]
            # Perpendicular in 2D (assume "facing forward" means rotated 90° from shoulder)
            facing_x = -sy
            facing_y = sx
            # Direction to ball from torso center
            torso_x = (ls[0] + rs[0]) / 2
            torso_y = (ls[1] + rs[1]) / 2
            to_ball_x = ball_x - torso_x
            to_ball_y = ball_y - torso_y
            mag_f = math.hypot(facing_x, facing_y) + 1e-6
            mag_b = math.hypot(to_ball_x, to_ball_y) + 1e-6
            cos_theta = (facing_x * to_ball_x + facing_y * to_ball_y) / (mag_f * mag_b)
            cos_theta = max(-1.0, min(1.0, cos_theta))
            body_orientation_diff = math.acos(cos_theta)

    # wrist_post_alignment: dot product of wrist velocity vector with
    # post-contact ball direction
    wrist_post_alignment = 0.0
    if (post_ball_x is not None and post_ball_y is not None
            and len(sorted_frames) >= 2):
        # Find wrist velocity vector near contact
        # use the largest velocity vector
        best_vel = None
        for i in range(len(sorted_frames) - 1):
            f1, f2 = sorted_frames[i], sorted_frames[i + 1]
            if f2 - f1 > 3:
                continue
            x1, y1 = wrist_pos[f1]
            x2, y2 = wrist_pos[f2]
            dx, dy = x2 - x1, y2 - y1
            d = math.hypot(dx, dy)
            if best_vel is None or d > best_vel[2]:
                best_vel = (dx, dy, d)
        if best_vel is not None and best_vel[2] > 0:
            ball_dx = post_ball_x - ball_x
            ball_dy = post_ball_y - ball_y
            ball_mag = math.hypot(ball_dx, ball_dy) + 1e-6
            # Normalize and dot
            wrist_post_alignment = (
                (best_vel[0] * ball_dx + best_vel[1] * ball_dy)
                / (best_vel[2] * ball_mag)
            )

    pose_confidence_mean = float(np.mean(confs)) if confs else 0.0

    return {
        "wrist_velocity_max": wrist_velocity_max,
        "wrist_to_ball_min": wrist_to_ball_min,
        "body_orientation_diff": body_orientation_diff,
        "arms_raised": arms_raised_at_contact,
        "wrist_post_alignment": wrist_post_alignment,
        "pose_confidence_mean": pose_confidence_mean,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, help="Only process this video codename")
    parser.add_argument("--limit", type=int, default=0, help="Max GT rows (debug)")
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    codenames = (args.video,) if args.video else TRUSTED_CODENAMES

    print("Loading pose model…", flush=True)
    pose_model = _get_pose_model()
    if pose_model is None:
        print("ERROR: pose model unavailable. Check weights/yolo/yolo11s-pose.pt", flush=True)
        return 1
    print(f"Pose model loaded: {type(pose_model).__name__}", flush=True)

    csv_path = REPORT_DIR / "features.csv"
    csv_f = open(csv_path, "w", newline="")
    writer = csv.writer(csv_f)
    writer.writerow([
        "video", "rally_id", "rally_order", "gt_frame", "gt_action",
        "candidate_tid", "is_gt",
        "wrist_velocity_max", "wrist_to_ball_min", "body_orientation_diff",
        "arms_raised", "wrist_post_alignment", "pose_confidence_mean",
    ])

    n_total_gt = 0
    n_extracted = 0
    t_start = time.time()
    with psycopg.connect(DB_DSN) as conn:
        for codename in codenames:
            video_path = VIDEOS_DIR / f"{codename}.mp4"
            if not video_path.exists():
                print(f"  [{codename}] video missing at {video_path} — skipping", flush=True)
                continue
            print(f"  [{codename}] processing…", flush=True)
            cur = conn.execute("""
                SELECT v.id, r.id, r."order", r.start_ms, v.fps,
                       pt.positions_json, pt.primary_track_ids, pt.ball_positions_json
                FROM videos v JOIN rallies r ON r.video_id = v.id
                JOIN player_tracks pt ON pt.rally_id = r.id
                WHERE v.name = %s AND r.status='CONFIRMED'
                ORDER BY r."order"
            """, [codename])
            rallies = cur.fetchall()
            n_rallies_processed = 0
            for vid, rid, order, start_ms, fps, positions_json, primary_raw, ball_positions_json in rallies:
                if positions_json is None or not isinstance(primary_raw, list):
                    continue
                positions = positions_from_json(
                    positions_json if isinstance(positions_json, list) else []
                )
                primary_tids = [int(t) for t in primary_raw]
                ball_positions = (
                    ball_positions_json if isinstance(ball_positions_json, list) else []
                )
                ball_by_frame = {int(b.get("frameNumber", -1)): b for b in ball_positions}
                # Fetch GT
                gt_cur = conn.execute("""
                    SELECT frame, action::text, resolved_track_id,
                           snapshot_ball_x, snapshot_ball_y
                    FROM rally_action_ground_truth
                    WHERE rally_id = %s AND resolved_track_id IS NOT NULL
                      AND snapshot_ball_x IS NOT NULL AND snapshot_ball_y IS NOT NULL
                    ORDER BY frame
                """, [rid])
                gt_rows = gt_cur.fetchall()
                if not gt_rows:
                    continue
                # Determine contact frames (use GT frame directly for the pose enrichment window).
                contact_frames = sorted({int(g[0]) for g in gt_rows})
                # Run pose enrichment on this rally's positions (mutates in place)
                n_enriched = enrich_positions_with_pose(
                    positions, str(video_path), contact_frames,
                    rally_start_ms=int(start_ms or 0),
                    window_half=WINDOW_HALF,
                    pose_model=pose_model,
                )
                n_rallies_processed += 1
                if n_enriched == 0:
                    # Pose enrichment failed — skip rally
                    continue
                # Compute features per GT row × candidate
                for gt_frame, gt_action, gt_tid, ball_x, ball_y in gt_rows:
                    n_total_gt += 1
                    if args.limit and n_total_gt > args.limit:
                        break
                    # Post-contact ball position (find ball at gt_frame+5..+15)
                    post_ball_x = post_ball_y = None
                    for offset in range(5, 16):
                        b = ball_by_frame.get(gt_frame + offset)
                        if b is not None:
                            post_ball_x = float(b.get("x") or 0)
                            post_ball_y = float(b.get("y") or 0)
                            break
                    for tid in primary_tids:
                        feats = compute_pose_features(
                            positions, tid, gt_frame, float(ball_x), float(ball_y),
                            post_ball_x, post_ball_y,
                        )
                        writer.writerow([
                            codename, rid, order, gt_frame, gt_action,
                            tid, 1 if tid == gt_tid else 0,
                            f"{feats['wrist_velocity_max']:.5f}",
                            f"{feats['wrist_to_ball_min']:.5f}",
                            f"{feats['body_orientation_diff']:.5f}",
                            f"{feats['arms_raised']:.1f}",
                            f"{feats['wrist_post_alignment']:.5f}",
                            f"{feats['pose_confidence_mean']:.3f}",
                        ])
                        n_extracted += 1
            print(f"    {n_rallies_processed} rallies processed", flush=True)

    csv_f.close()
    elapsed = time.time() - t_start
    print(flush=True)
    print(f"Extracted {n_extracted} candidate-features across {n_total_gt} GT rows", flush=True)
    print(f"Elapsed: {elapsed:.1f}s", flush=True)
    print(f"Output: {csv_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
