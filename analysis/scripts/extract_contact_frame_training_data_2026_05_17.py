#!/usr/bin/env python3
"""Phase 1: extract training data for the learned contact-frame regressor.

For each action GT row across the FULL 74-video corpus (NOT just trusted-29):
  1. Find the current pipeline contact in DB closest to GT (within ±15f).
     This is the model's input — features are computed at this frame.
  2. Target = (gt_frame - current_frame), the signed frame offset the model
     should predict.
  3. Compute features:
     - Ball trajectory: direction-change at f, max in window, max-offset
       velocity at f, max in window
     - Pose: wrist-min-dist at f, wrist-min over window, wrist-min-offset,
       wrist-confidence at min-frame
     - Player bbox: bbox-min-dist over window, bbox-min-offset
     - Action context: GT action type (one-hot)

The regressor target is just the CONTACT FRAME, not player attribution, so
we can use all ~2775 action GT rows (not just the 1252 with player-attribution
GT). This more than doubles the training data — critical for a ~20-feature
model to avoid overfitting.

If we trained ONLY on NEAR cases, the model would learn "always shift back ~10f."
By including all GT-matched cases (most placements are already correct, target=0),
the model learns WHEN to shift vs leave-as-is.

Output: parquet/CSV with one row per (gt_row, candidate) pair.
"""
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict
from pathlib import Path

import psycopg

DB_DSN = "postgresql://postgres:postgres@localhost:5436/rallycut"
NEIGHBOR_STEP = 3
WINDOW = 15  # +/- frames around current contact to compute features over
KPT_LEFT_WRIST = 9
KPT_RIGHT_WRIST = 10
KPT_LEFT_ELBOW = 7
KPT_RIGHT_ELBOW = 8
WRIST_CONF_MIN = 0.30
ACTIONS = ("SERVE", "RECEIVE", "SET", "ATTACK", "DIG", "BLOCK")

OUT_DIR = Path("reports/contact_frame_regressor_2026_05_17")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def dir_change(b, f):
    p, c, n = b.get(f - NEIGHBOR_STEP), b.get(f), b.get(f + NEIGHBOR_STEP)
    if not all((p, c, n)):
        return 0.0
    v1x, v1y = c["x"] - p["x"], c["y"] - p["y"]
    v2x, v2y = n["x"] - c["x"], n["y"] - c["y"]
    m1, m2 = math.hypot(v1x, v1y), math.hypot(v2x, v2y)
    if m1 < 1e-4 or m2 < 1e-4:
        return 0.0
    ct = max(-1.0, min(1.0, (v1x * v2x + v1y * v2y) / (m1 * m2)))
    return math.degrees(math.acos(ct))


def ball_velocity(b, f):
    """Approximate ball speed at frame f (avg over ±1)."""
    cur = b.get(f)
    if not cur:
        return 0.0
    speeds = []
    for off in (-1, 1):
        nbr = b.get(f + off)
        if nbr:
            speeds.append(math.hypot(cur["x"] - nbr["x"], cur["y"] - nbr["y"]))
    return sum(speeds) / max(1, len(speeds))


def wrist_kp_at(positions_at_frame, ball, return_full=False):
    """Min wrist-to-ball over all (player, wrist) pairs at this frame."""
    best_d = float("inf")
    best_conf = 0.0
    best_player = None
    if not ball:
        return (best_d, best_conf, None) if return_full else best_d
    for p in positions_at_frame:
        kps = p.get("keypoints")
        if not kps or len(kps) < 11:
            continue
        for idx in (KPT_LEFT_WRIST, KPT_RIGHT_WRIST):
            kp = kps[idx]
            if len(kp) < 3 or kp[2] < WRIST_CONF_MIN:
                continue
            d = math.hypot(kp[0] - ball["x"], kp[1] - ball["y"])
            if d < best_d:
                best_d = d
                best_conf = kp[2]
                best_player = p
    return (best_d, best_conf, best_player) if return_full else best_d


def elbow_kp_at(positions_at_frame, ball):
    """Min elbow-to-ball across players (for SET / overhead context)."""
    best_d = float("inf")
    if not ball:
        return best_d
    for p in positions_at_frame:
        kps = p.get("keypoints")
        if not kps or len(kps) < 11:
            continue
        for idx in (KPT_LEFT_ELBOW, KPT_RIGHT_ELBOW):
            kp = kps[idx]
            if len(kp) < 3 or kp[2] < WRIST_CONF_MIN:
                continue
            d = math.hypot(kp[0] - ball["x"], kp[1] - ball["y"])
            if d < best_d:
                best_d = d
    return best_d


def bbox_dist_at(positions_at_frame, ball):
    """Min player-bbox-center to ball distance (upper-quarter adjusted)."""
    best = float("inf")
    if not ball:
        return best
    for p in positions_at_frame:
        py_uq = p["y"] - p["height"] * 0.25
        d = math.hypot(p["x"] - ball["x"], py_uq - ball["y"])
        if d < best:
            best = d
    return best


def extract_features(balls, pos_by_frame, candidate_frame):
    """Compute the 22-dim feature vector for a candidate contact at this frame."""
    f = candidate_frame
    cur_ball = balls.get(f)
    if not cur_ball:
        return None  # no ball at candidate — can't extract features

    # Ball features at candidate
    dc_at = dir_change(balls, f)
    vel_at = ball_velocity(balls, f)
    pre_vel = ball_velocity(balls, f - 3)
    post_vel = ball_velocity(balls, f + 3)

    # Window-wide features
    window_dc = {}
    window_wrist = {}
    window_elbow = {}
    window_bbox = {}
    window_vel = {}
    for off in range(-WINDOW, WINDOW + 1):
        ff = f + off
        b = balls.get(ff)
        if not b:
            continue
        window_dc[off] = dir_change(balls, ff)
        window_vel[off] = ball_velocity(balls, ff)
        ps = pos_by_frame.get(ff, [])
        window_wrist[off] = wrist_kp_at(ps, b)
        window_elbow[off] = elbow_kp_at(ps, b)
        window_bbox[off] = bbox_dist_at(ps, b)

    if not window_dc:
        return None

    # Aggregates
    dc_max = max(window_dc.values())
    dc_max_off = max(window_dc, key=lambda o: window_dc[o]) if window_dc else 0
    vel_max = max(window_vel.values()) if window_vel else 0.0
    vel_min = min(window_vel.values()) if window_vel else 0.0
    vel_min_off = min(window_vel, key=lambda o: window_vel[o]) if window_vel else 0

    # Wrist
    valid_wrist = {o: d for o, d in window_wrist.items() if d != float("inf")}
    wrist_at_cand = window_wrist.get(0, float("inf"))
    if valid_wrist:
        wrist_min = min(valid_wrist.values())
        wrist_min_off = min(valid_wrist, key=lambda o: valid_wrist[o])
    else:
        wrist_min = -1.0  # sentinel for no pose data
        wrist_min_off = 0

    # Elbow
    valid_elbow = {o: d for o, d in window_elbow.items() if d != float("inf")}
    elbow_at_cand = window_elbow.get(0, float("inf"))
    if valid_elbow:
        elbow_min = min(valid_elbow.values())
        elbow_min_off = min(valid_elbow, key=lambda o: valid_elbow[o])
    else:
        elbow_min = -1.0
        elbow_min_off = 0

    # Bbox
    valid_bbox = {o: d for o, d in window_bbox.items() if d != float("inf")}
    bbox_at_cand = window_bbox.get(0, float("inf"))
    if valid_bbox:
        bbox_min = min(valid_bbox.values())
        bbox_min_off = min(valid_bbox, key=lambda o: valid_bbox[o])
    else:
        bbox_min = float("inf")
        bbox_min_off = 0

    return {
        "dc_at": dc_at,
        "dc_max": dc_max,
        "dc_max_off": dc_max_off,
        "vel_at": vel_at,
        "vel_max": vel_max,
        "vel_min": vel_min,
        "vel_min_off": vel_min_off,
        "pre_vel": pre_vel,
        "post_vel": post_vel,
        "vel_ratio": post_vel / max(1e-4, pre_vel),
        "wrist_at_cand": min(wrist_at_cand, 1.0) if wrist_at_cand != float("inf") else -1.0,
        "wrist_min": wrist_min,
        "wrist_min_off": wrist_min_off,
        "elbow_at_cand": min(elbow_at_cand, 1.0) if elbow_at_cand != float("inf") else -1.0,
        "elbow_min": elbow_min,
        "elbow_min_off": elbow_min_off,
        "bbox_at_cand": min(bbox_at_cand, 1.0) if bbox_at_cand != float("inf") else -1.0,
        "bbox_min": min(bbox_min, 1.0) if bbox_min != float("inf") else -1.0,
        "bbox_min_off": bbox_min_off,
    }


def main() -> int:
    print("Fetching full-corpus action GT rows + rally data...", flush=True)
    # Use ALL action GT rows (74 videos), not just trusted-29's 1252.
    # The regressor predicts contact frame offset — no player-attribution
    # needed in the target. ~2775 rows total.
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            """
            SELECT v.name, r.id, rg.action::text, rg.frame, rg.resolved_track_id,
                   pt.ball_positions_json, pt.positions_json, pt.contacts_json
            FROM rally_action_ground_truth rg
            JOIN rallies r ON rg.rally_id = r.id
            JOIN videos v ON r.video_id = v.id
            JOIN player_tracks pt ON pt.rally_id = r.id
            """,
        )
        rows = cur.fetchall()
    print(f"Got {len(rows)} GT rows across full corpus", flush=True)

    # Group by rally for efficient processing
    rally_cache: dict = {}
    n_no_candidate = 0
    n_examples = 0
    n_skipped_no_features = 0

    feature_names = None
    out_rows = []

    for video, rid, ga, gf, gtid, bj, pj, cj in rows:
        ga = ga.upper()
        if rid not in rally_cache:
            bj = bj if isinstance(bj, list) else json.loads(bj)
            pj = pj if isinstance(pj, list) else json.loads(pj)
            cj = cj if isinstance(cj, dict) else json.loads(cj)
            balls = {}
            for b in bj:
                fn = int(b.get("frameNumber", -1))
                x, y = b.get("x", 0), b.get("y", 0)
                if fn < 0 or (x == 0 and y == 0):
                    continue
                balls[fn] = {"x": x, "y": y}
            pos_by_frame = defaultdict(list)
            for p in pj:
                fn = int(p.get("frameNumber", -1))
                if fn < 0:
                    continue
                pos_by_frame[fn].append(p)
            rally_cache[rid] = {
                "balls": balls,
                "pos_by_frame": dict(pos_by_frame),
                "contacts": cj.get("contacts") or [],
            }
        rd = rally_cache[rid]

        # Find pipeline contact closest to GT within ±WINDOW
        closest = None
        closest_delta = WINDOW + 1
        for c in rd["contacts"]:
            d = abs(int(c["frame"]) - gf)
            if d < closest_delta:
                closest_delta = d
                closest = c
        if closest is None:
            n_no_candidate += 1
            continue

        cand_frame = int(closest["frame"])
        feat = extract_features(rd["balls"], rd["pos_by_frame"], cand_frame)
        if feat is None:
            n_skipped_no_features += 1
            continue

        # One-hot action type
        act_oh = {f"act_{a}": int(a == ga) for a in ACTIONS}

        row = {
            "video": video,
            "rally_id": rid,
            "gt_frame": gf,
            "gt_action": ga,
            "gt_tid": gtid,
            "cand_frame": cand_frame,
            "target_offset": gf - cand_frame,
            **feat,
            **act_oh,
        }
        out_rows.append(row)
        n_examples += 1

        if feature_names is None:
            feature_names = [k for k in row.keys() if k not in {
                "video", "rally_id", "gt_frame", "gt_action", "gt_tid",
                "cand_frame", "target_offset",
            }]

    print(f"\nExtracted {n_examples} examples")
    print(f"  No candidate within ±{WINDOW}f: {n_no_candidate}")
    print(f"  Skipped (no features): {n_skipped_no_features}")
    print(f"  Feature count: {len(feature_names) if feature_names else 0}")

    # Save CSV
    out_path = OUT_DIR / "training_data.csv"
    with open(out_path, "w") as f:
        if not out_rows:
            print("No examples — exiting")
            return 1
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"Wrote {out_path}")

    # Summary stats on target_offset
    targets = [r["target_offset"] for r in out_rows]
    print(f"\nTarget offset distribution:")
    print(f"  Range: [{min(targets)}, {max(targets)}]")
    print(f"  Mean: {sum(targets)/len(targets):.2f}, MeanAbs: {sum(abs(t) for t in targets)/len(targets):.2f}")
    print(f"  Within ±2: {sum(1 for t in targets if abs(t) <= 2)} ({sum(1 for t in targets if abs(t) <= 2)/len(targets)*100:.1f}%)")
    print(f"  Within ±5: {sum(1 for t in targets if abs(t) <= 5)} ({sum(1 for t in targets if abs(t) <= 5)/len(targets)*100:.1f}%)")
    print(f"  Outside ±5: {sum(1 for t in targets if abs(t) > 5)} ({sum(1 for t in targets if abs(t) > 5)/len(targets)*100:.1f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
