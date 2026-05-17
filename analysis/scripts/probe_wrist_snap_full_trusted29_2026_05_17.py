#!/usr/bin/env python3
"""Validate wrist-snap on ALL trusted-29 contacts (not just NEARs).

Simulates: would wrist-snap regress contacts that are already correctly
placed? Critical because the NEAR-cases probe showed strong recovery (75%
within ±5) but also overshoots. If wrist-snap moves correctly-placed
contacts off-target, we'd regress more than we gain.

Tests several guarded variants:
  V1: Pure wrist-min (no constraint)
  V2: Wrist-min subject to dc>=15°
  V3: Wrist-min within ±8f of current (cap max move)
  V4: Wrist-min within ±8f AND dc>=15° AND wrist_dist<0.05

The right deployment criteria are conservative — minimize LOST while
maximizing GAINED.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict

import psycopg

DB_DSN = "postgresql://postgres:postgres@localhost:5436/rallycut"
TRUSTED_29 = (
    "titi", "toto", "lulu", "wawa", "caco", "cece", "cici", "cuco",
    "gaga", "kaka", "kiki", "juju", "yeye", "keke",
    "gigi", "gugu", "mame", "meme", "mimi", "moma", "mumu",
    "papa", "pepe", "pipi", "popo", "pupu", "veve", "vivi", "vovo",
)
MATCH_WINDOW = 5
NEIGHBOR_STEP = 3
KPT_LEFT_WRIST = 9
KPT_RIGHT_WRIST = 10
WRIST_CONF_MIN = 0.30


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


def wrist_min_dist(positions_at_frame, ball):
    """Min wrist-to-ball over all (player, wrist) pairs at this frame."""
    best = float("inf")
    for p in positions_at_frame:
        kps = p.get("keypoints")
        if not kps or len(kps) < 11:
            continue
        for idx in (KPT_LEFT_WRIST, KPT_RIGHT_WRIST):
            kp = kps[idx]
            if len(kp) < 3 or kp[2] < WRIST_CONF_MIN:
                continue
            d = math.hypot(kp[0] - ball["x"], kp[1] - ball["y"])
            if d < best:
                best = d
    return best


def simulate_wrist_snap(
    balls, pos_by_frame, f_now,
    window=12, max_move=12,
    require_dc=False, min_dc=15.0,
    max_wrist_dist=float("inf"),
):
    """Find frame in window with minimum wrist-to-ball distance (with constraints)."""
    best_f = f_now
    best_d = float("inf")
    for f in range(f_now - window, f_now + window + 1):
        if abs(f - f_now) > max_move:
            continue
        ball = balls.get(f)
        if not ball:
            continue
        ps = pos_by_frame.get(f, [])
        d = wrist_min_dist(ps, ball)
        if d >= max_wrist_dist:
            continue
        if require_dc and dir_change(balls, f) < min_dc:
            continue
        if d < best_d:
            best_d = d
            best_f = f
    return best_f, best_d


def main() -> int:
    print("Fetching trusted-29 GT + rally data...", flush=True)
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            """
            SELECT v.name, r.id, rg.action::text, rg.frame, rg.resolved_track_id,
                   pt.ball_positions_json, pt.positions_json, pt.contacts_json
            FROM rally_action_ground_truth rg
            JOIN rallies r ON rg.rally_id = r.id
            JOIN videos v ON r.video_id = v.id
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE v.name = ANY(%s) AND rg.resolved_track_id IS NOT NULL
            """,
            [list(TRUSTED_29)],
        )
        rows = cur.fetchall()
    print(f"Got {len(rows)} GT rows", flush=True)

    rally_cache: dict = {}
    for video, rid, _, _, _, bj, pj, cj in rows:
        if rid in rally_cache:
            continue
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

    variants = [
        ("V1 pure",           dict(require_dc=False,                  max_wrist_dist=float("inf"))),
        ("V2 dc≥15",          dict(require_dc=True, min_dc=15.0,      max_wrist_dist=float("inf"))),
        ("V3 cap±8",          dict(require_dc=False,                  max_wrist_dist=float("inf"), max_move=8)),
        ("V4 cap±8+dc≥15",    dict(require_dc=True, min_dc=15.0,      max_wrist_dist=float("inf"), max_move=8)),
        ("V5 wd<0.05",        dict(require_dc=False,                  max_wrist_dist=0.05)),
        ("V6 wd<0.05+dc≥15",  dict(require_dc=True, min_dc=15.0,      max_wrist_dist=0.05)),
        ("V7 wd<0.03+dc≥10",  dict(require_dc=True, min_dc=10.0,      max_wrist_dist=0.03)),
    ]

    # Pre-compute current per-GT-row "matched?"
    gt_per_rally = defaultdict(list)
    for video, rid, ga, gf, gtid, *_ in rows:
        gt_per_rally[rid].append((ga.upper(), gf, gtid))

    print(f"\n{'Variant':<22s}{'GAINED':>8s}{'LOST':>6s}{'NET':>6s}{'before':>9s}{'after':>9s}")
    for vname, kwargs in variants:
        n_now = 0
        n_after = 0
        gained = 0
        lost = 0
        for rid, gts in gt_per_rally.items():
            rd = rally_cache[rid]
            # Snap all contacts
            snapped = []
            for c in rd["contacts"]:
                f_new, _ = simulate_wrist_snap(
                    rd["balls"], rd["pos_by_frame"], int(c["frame"]),
                    window=12,
                    **kwargs,
                )
                snapped.append({"frame": f_new})
            # Check each GT
            for ga, gf, gtid in gts:
                was_m = any(abs(int(c["frame"]) - gf) <= MATCH_WINDOW for c in rd["contacts"])
                is_m = any(abs(c["frame"] - gf) <= MATCH_WINDOW for c in snapped)
                if was_m:
                    n_now += 1
                if is_m:
                    n_after += 1
                if not was_m and is_m:
                    gained += 1
                if was_m and not is_m:
                    lost += 1
        net = gained - lost
        print(f"{vname:<22s}{gained:>7d}{lost:>7d}{net:>+6d}{n_now:>9d}{n_after:>9d}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
