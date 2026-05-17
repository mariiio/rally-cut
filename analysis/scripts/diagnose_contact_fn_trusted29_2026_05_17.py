#!/usr/bin/env python3
"""Diagnose contact-detection FN failures on trusted-29.

For each GT row that has no pipeline contact within ±5 frames, gather:
  - GT bbox / ball-y / action_type
  - Was the ball tracked at the GT frame?
  - Was a player tracked at the GT bbox?
  - What's the nearest pipeline contact?
  - What's the ball trajectory around the GT frame (did the ball
    change direction, accelerate, etc.)?

Categorize failure mode:
  BALL_NOT_TRACKED — no ball detection within ±3 frames of GT
  NO_PLAYER_NEAR_BALL — ball tracked but no player bbox within proximity
  NEAR_PIPELINE_CONTACT — pipeline has a contact within ±15f (not ±5f)
                          — placement off, not a true FN
  NO_TRAJECTORY_CHANGE — ball + player both present but no direction
                         change / velocity spike (primitive doesn't fire)
  UPSTREAM_TRACKER — multiple primitives missing

Output: CSV with one row per FN + summary breakdown.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import psycopg

DB = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5436/rallycut",
)
TRUSTED_29 = (
    "titi", "toto", "lulu", "wawa", "caco", "cece", "cici", "cuco",
    "gaga", "kaka", "kiki", "juju", "yeye", "keke",
    "gigi", "gugu", "mame", "meme", "mimi", "moma", "mumu",
    "papa", "pepe", "pipi", "popo", "pupu", "veve", "vivi", "vovo",
)
MATCH_WINDOW = 5
NEAR_WINDOW = 15
BALL_PROXIMITY_THRESH = 0.06    # ~normalized px / fraction of frame
PLAYER_PROXIMITY_THRESH = 0.15

OUT_DIR = Path("reports/contact_fn_trusted29_2026_05_17")
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "fn_cases.csv"
SUMMARY_PATH = OUT_DIR / "summary.md"


def find_nearest_action(actions, gt_frame, gt_action):
    """Closest pipeline action by frame within NEAR_WINDOW (not action_type
    matched). Returns (delta, action) or (None, None)."""
    best, best_d = None, NEAR_WINDOW + 1
    for a in actions:
        d = abs(int(a.get("frame", -10**9)) - gt_frame)
        if d < best_d:
            best_d = d
            best = a
    return (best_d, best) if best is not None else (None, None)


def categorize(
    gt_frame, gt_action,
    actions, ball_positions, all_player_positions,
):
    """Return failure-mode label + extra diagnostics."""
    # Did pipeline find a contact within ±MATCH_WINDOW? (sanity — should be no
    # if we got here, but verify)
    matched = [
        a for a in actions
        if abs(int(a.get("frame", -10**9)) - gt_frame) <= MATCH_WINDOW
    ]
    if matched:
        return "MATCHED_WITHIN_5F", {}

    # 1. Is the ball tracked at GT frame ±3?
    nearby_balls = [
        bp for bp in ball_positions
        if abs(bp["frameNumber"] - gt_frame) <= 3
        and (bp.get("x", 0) > 0 or bp.get("y", 0) > 0)
    ]
    if not nearby_balls:
        return "BALL_NOT_TRACKED", {"nearby_ball_frames": 0}

    # 2. Is there a player tracked near the ball at GT frame?
    ball_at = min(nearby_balls, key=lambda b: abs(b["frameNumber"] - gt_frame))
    bx, by = ball_at["x"], ball_at["y"]
    players_at = [
        p for p in all_player_positions
        if abs(int(p.get("frameNumber", -1)) - gt_frame) <= 3
    ]
    if not players_at:
        return "NO_PLAYER_TRACKED", {}

    # Min distance from ball to any player upper-quarter
    def upper_q_dist(p):
        py_uq = float(p["y"]) - float(p["height"]) * 0.25
        return math.hypot(float(p["x"]) - bx, py_uq - by)

    min_dist = min(upper_q_dist(p) for p in players_at)
    if min_dist > PLAYER_PROXIMITY_THRESH:
        return "NO_PLAYER_NEAR_BALL", {
            "min_player_dist": round(min_dist, 3),
            "ball_x": round(bx, 3), "ball_y": round(by, 3),
        }

    # 3. Pipeline has a nearby (but >MATCH_WINDOW) contact?
    delta, near = find_nearest_action(actions, gt_frame, gt_action)
    if delta is not None and delta <= NEAR_WINDOW:
        return "NEAR_PIPELINE_CONTACT", {
            "delta_frames": delta,
            "near_action_type": near.get("action"),
            "near_player_tid": near.get("playerTrackId"),
        }

    # 4. Ball is moving (trajectory check)
    pre = [bp for bp in ball_positions if gt_frame - 8 <= bp["frameNumber"] < gt_frame]
    post = [bp for bp in ball_positions if gt_frame < bp["frameNumber"] <= gt_frame + 8]
    if not pre or not post:
        return "BALL_TRAJECTORY_INCOMPLETE", {
            "pre_count": len(pre), "post_count": len(post),
            "min_player_dist": round(min_dist, 3),
        }

    pre_dy = pre[-1]["y"] - pre[0]["y"] if len(pre) >= 2 else 0
    post_dy = post[-1]["y"] - post[0]["y"] if len(post) >= 2 else 0
    pre_dx = pre[-1]["x"] - pre[0]["x"] if len(pre) >= 2 else 0
    post_dx = post[-1]["x"] - post[0]["x"] if len(post) >= 2 else 0

    # Compute direction change angle
    if (pre_dx == 0 and pre_dy == 0) or (post_dx == 0 and post_dy == 0):
        return "NO_TRAJECTORY_CHANGE", {
            "min_player_dist": round(min_dist, 3),
            "reason": "stationary",
        }
    pre_mag = math.hypot(pre_dx, pre_dy) + 1e-9
    post_mag = math.hypot(post_dx, post_dy) + 1e-9
    cos_theta = (pre_dx * post_dx + pre_dy * post_dy) / (pre_mag * post_mag)
    cos_theta = max(-1.0, min(1.0, cos_theta))
    theta_deg = math.degrees(math.acos(cos_theta))

    if theta_deg < 15:
        return "NO_TRAJECTORY_CHANGE", {
            "min_player_dist": round(min_dist, 3),
            "direction_change_deg": round(theta_deg, 1),
            "pre_speed": round(pre_mag, 3),
            "post_speed": round(post_mag, 3),
        }

    # Trajectory + player both present + direction change: contact-detector
    # should have fired. Classifier rejected? Other gate?
    return "TRAJECTORY_CHANGE_BUT_NO_CONTACT", {
        "min_player_dist": round(min_dist, 3),
        "direction_change_deg": round(theta_deg, 1),
        "pre_speed": round(pre_mag, 3),
        "post_speed": round(post_mag, 3),
    }


def main() -> int:
    rows: list[dict] = []
    cat_counter: Counter = Counter()
    per_action_cat: dict[str, Counter] = defaultdict(Counter)
    per_video_cat: dict[str, Counter] = defaultdict(Counter)

    with psycopg.connect(DB) as conn:
        cur = conn.execute(
            """
            SELECT v.name, r.id, rg.frame, rg.action::text, rg.resolved_track_id,
                   rg.snapshot_bbox_x1, rg.snapshot_bbox_y1,
                   rg.snapshot_bbox_x2, rg.snapshot_bbox_y2,
                   pt.actions_json, pt.contacts_json,
                   pt.ball_positions_json, pt.positions_json
            FROM rally_action_ground_truth rg
            JOIN rallies r ON rg.rally_id = r.id
            JOIN videos v ON r.video_id = v.id
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE v.name = ANY(%s)
              AND rg.resolved_track_id IS NOT NULL
            """,
            [list(TRUSTED_29)],
        )
        gt_rows = cur.fetchall()

    print(f"Analysing {len(gt_rows)} GT rows on trusted-29")
    n_matched = 0
    for (video, rid, gt_f, gt_action, gt_tid,
         gx1, gy1, gx2, gy2,
         aj, cj, bj, pj) in gt_rows:
        aj = aj if isinstance(aj, dict) else (json.loads(aj) if aj else {})
        cj = cj if isinstance(cj, dict) else (json.loads(cj) if cj else {})
        bj = bj if isinstance(bj, list) else (json.loads(bj) if bj else [])
        pj = pj if isinstance(pj, list) else (json.loads(pj) if pj else [])
        actions = aj.get("actions") or []

        cat, extras = categorize(
            gt_f, gt_action.upper(), actions, bj, pj,
        )
        if cat == "MATCHED_WITHIN_5F":
            n_matched += 1
            continue
        cat_counter[cat] += 1
        per_action_cat[gt_action.upper()][cat] += 1
        per_video_cat[video][cat] += 1
        rows.append({
            "video": video,
            "rally_id_short": rid[:8],
            "gt_frame": gt_f,
            "gt_action": gt_action,
            "gt_tid": gt_tid,
            "gt_bbox": f"({gx1:.2f},{gy1:.2f})-({gx2:.2f},{gy2:.2f})" if gx1 else "",
            "category": cat,
            **{k: v for k, v in extras.items() if k in {
                "delta_frames", "near_action_type", "near_player_tid",
                "min_player_dist", "direction_change_deg",
                "pre_speed", "post_speed", "nearby_ball_frames",
            }},
        })

    n_fn = len(rows)
    print(f"  matched within ±{MATCH_WINDOW}f: {n_matched}")
    print(f"  FN cases:                       {n_fn}")
    print()
    print("Failure mode breakdown:")
    for cat, n in cat_counter.most_common():
        print(f"  {cat:35s} {n:4d}  ({n / max(1, n_fn) * 100:5.1f}%)")
    print()
    print("Per action type:")
    for action in ("SERVE", "RECEIVE", "SET", "ATTACK", "DIG", "BLOCK"):
        if action not in per_action_cat:
            continue
        n_total = sum(per_action_cat[action].values())
        print(f"  {action} ({n_total} FNs):")
        for cat, n in per_action_cat[action].most_common():
            print(f"    {cat:35s} {n:4d}  ({n / max(1, n_total) * 100:5.1f}%)")
    print()
    print("Per video (top regressors):")
    for v in sorted(per_video_cat.keys(), key=lambda x: -sum(per_video_cat[x].values())):
        tot = sum(per_video_cat[v].values())
        if tot < 5:
            continue
        print(f"  {v}: {tot} FNs", end=" | ")
        top3 = ", ".join(f"{c}={n}" for c, n in per_video_cat[v].most_common(3))
        print(top3)

    # CSV
    if rows:
        fieldnames = ["video", "rally_id_short", "gt_frame", "gt_action", "gt_tid",
                      "gt_bbox", "category",
                      "delta_frames", "near_action_type", "near_player_tid",
                      "min_player_dist", "direction_change_deg",
                      "pre_speed", "post_speed", "nearby_ball_frames"]
        with CSV_PATH.open("w") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nWrote {CSV_PATH}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
