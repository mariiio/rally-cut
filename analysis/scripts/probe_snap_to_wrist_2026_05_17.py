#!/usr/bin/env python3
"""Probe pose-wrist-based snap on the 58 NEAR_PIPELINE_CONTACT cases.

Strategy: at each frame in the snap window, look up the GT player's wrist
keypoints (left=9, right=10). Find the frame where the closer-of-the-two
wrists is closest to the ball. That should be the strike frame.

Two variants:
  A. ORACLE — use GT player's wrist (upper bound; not deployable but tells
     us if wrist signal CAN disambiguate the strike frame).
  B. REALISTIC — use nearest-player-wrist regardless of identity (closer to
     production deployment).

If oracle is dramatically better than bbox-center snap, pose-aware snap is
worth implementing. If oracle doesn't help much, pose precision isn't the
bottleneck and we should look elsewhere.
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys
from collections import defaultdict

import psycopg

DB_DSN = "postgresql://postgres:postgres@localhost:5436/rallycut"
WINDOW = 15  # ±15 frames search window
KPT_LEFT_WRIST = 9
KPT_RIGHT_WRIST = 10
WRIST_CONF_MIN = 0.30  # min keypoint confidence to trust


def wrist_to_ball_dist(
    kps: list,
    ball: dict,
) -> float:
    """Min distance from ball to either wrist (with confidence filter)."""
    if not kps or len(kps) < 11:
        return float("inf")
    best = float("inf")
    for idx in (KPT_LEFT_WRIST, KPT_RIGHT_WRIST):
        kp = kps[idx]
        if len(kp) < 3 or kp[2] < WRIST_CONF_MIN:
            continue
        d = math.hypot(kp[0] - ball["x"], kp[1] - ball["y"])
        if d < best:
            best = d
    return best


def main() -> int:
    # Load NEAR cases
    near_cases = []
    with open("reports/contact_fn_trusted29_2026_05_17/fn_cases.csv") as f:
        for r in csv.DictReader(f):
            if r["category"] != "NEAR_PIPELINE_CONTACT":
                continue
            near_cases.append({
                "video": r["video"],
                "rally_id_short": r["rally_id_short"],
                "gt_frame": int(r["gt_frame"]),
                "gt_action": r["gt_action"].upper(),
                "delta_now": int(r["delta_frames"]),
                "gt_tid": int(r["gt_tid"]) if r.get("gt_tid") else -1,
            })
    print(f"Loaded {len(near_cases)} NEAR cases", flush=True)

    # Fetch rally data + GT info
    print("Fetching rally data with positions_json (with keypoints)...", flush=True)
    rally_cache: dict[str, dict] = {}
    with psycopg.connect(DB_DSN) as conn:
        for case in near_cases:
            cur = conn.execute(
                """SELECT r.id, pt.ball_positions_json, pt.positions_json,
                          rg.resolved_track_id
                   FROM rallies r
                   JOIN videos v ON r.video_id=v.id
                   JOIN player_tracks pt ON pt.rally_id=r.id
                   JOIN rally_action_ground_truth rg ON rg.rally_id=r.id
                   WHERE v.name=%s AND r.id LIKE %s
                     AND rg.frame=%s AND rg.action::text=%s
                   LIMIT 1""",
                [case["video"], case["rally_id_short"] + "%",
                 case["gt_frame"], case["gt_action"].upper()],
            )
            row = cur.fetchone()
            if not row:
                continue
            rid, bj, pj, gt_tid = row
            case["rally_id"] = rid
            case["gt_tid_resolved"] = gt_tid
            if rid in rally_cache:
                continue
            bj = bj if isinstance(bj, list) else json.loads(bj)
            pj = pj if isinstance(pj, list) else json.loads(pj)
            balls = {}
            for b in bj:
                fn = int(b.get("frameNumber", -1))
                x, y = b.get("x", 0), b.get("y", 0)
                if fn < 0 or (x == 0 and y == 0):
                    continue
                balls[fn] = {"x": x, "y": y}
            # Index: (frame, tid) -> position with keypoints
            pos_by_frame_tid: dict[tuple[int, int], dict] = {}
            pos_by_frame: dict[int, list[dict]] = defaultdict(list)
            for p in pj:
                fn = int(p.get("frameNumber", -1))
                tid = int(p.get("trackId", -1))
                if fn < 0 or tid < 0:
                    continue
                pos_by_frame_tid[(fn, tid)] = p
                pos_by_frame[fn].append(p)
            rally_cache[rid] = {
                "balls": balls,
                "pos_by_frame_tid": pos_by_frame_tid,
                "pos_by_frame": dict(pos_by_frame),
            }

    # For each case with a resolved rally + pose available, simulate snap
    results = {
        "current": [],   # baseline (bbox-snap, what's in DB)
        "oracle": [],    # snap to GT player's wrist-min frame
        "realistic": [], # snap to nearest-player wrist-min frame
    }
    no_pose_count = 0
    sample_lines = []

    for case in near_cases:
        rid = case.get("rally_id")
        if not rid or rid not in rally_cache:
            continue
        rd = rally_cache[rid]
        f_gt = case["gt_frame"]
        f_now = f_gt + case["delta_now"]
        gt_tid = case.get("gt_tid_resolved", -1)

        # ORACLE: snap on GT player's wrist
        best_f_oracle = f_now
        best_d_oracle = float("inf")
        gt_wrist_frames = 0
        for f in range(f_now - WINDOW, f_now + WINDOW + 1):
            ball = rd["balls"].get(f)
            if not ball:
                continue
            pos = rd["pos_by_frame_tid"].get((f, gt_tid))
            if not pos or not pos.get("keypoints"):
                continue
            gt_wrist_frames += 1
            d = wrist_to_ball_dist(pos["keypoints"], ball)
            if d < best_d_oracle:
                best_d_oracle = d
                best_f_oracle = f

        # REALISTIC: snap on any player's wrist-min (nearest wrist)
        best_f_real = f_now
        best_d_real = float("inf")
        for f in range(f_now - WINDOW, f_now + WINDOW + 1):
            ball = rd["balls"].get(f)
            if not ball:
                continue
            ps = rd["pos_by_frame"].get(f, [])
            for p in ps:
                kps = p.get("keypoints")
                if not kps:
                    continue
                d = wrist_to_ball_dist(kps, ball)
                if d < best_d_real:
                    best_d_real = d
                    best_f_real = f

        if best_d_oracle == float("inf") and best_d_real == float("inf"):
            no_pose_count += 1
            continue

        results["current"].append(
            (case["gt_action"], abs(case["delta_now"]), case["delta_now"])
        )
        if best_d_oracle != float("inf"):
            results["oracle"].append(
                (case["gt_action"], abs(best_f_oracle - f_gt), best_f_oracle - f_gt)
            )
        if best_d_real != float("inf"):
            results["realistic"].append(
                (case["gt_action"], abs(best_f_real - f_gt), best_f_real - f_gt)
            )

        if len(sample_lines) < 25:
            sample_lines.append(
                f"{case['video']:<8s}{case['gt_action']:<7s}f{f_gt}  "
                f"now:{case['delta_now']:+d}  "
                f"oracle:{best_f_oracle - f_gt:+d}(d={best_d_oracle:.3f}, kp_frms={gt_wrist_frames})  "
                f"real:{best_f_real - f_gt:+d}(d={best_d_real:.3f})"
            )

    print(f"\nCases with NO pose data anywhere in window: {no_pose_count}")
    print(f"Cases evaluated: {len(results['current'])} (current) / "
          f"{len(results['oracle'])} (oracle) / {len(results['realistic'])} (realistic)")

    def summarize(name, rs):
        if not rs:
            print(f"  {name}: no data")
            return
        deltas = [d for _, d, _ in rs]
        signed = [s for _, _, s in rs]
        mean_d = sum(deltas) / len(deltas)
        med = sorted(deltas)[len(deltas) // 2]
        w5 = sum(1 for d in deltas if d <= 5)
        w3 = sum(1 for d in deltas if d <= 3)
        w2 = sum(1 for d in deltas if d <= 2)
        over_neg3 = sum(1 for s in signed if s < -3)
        over_pos3 = sum(1 for s in signed if s > 3)
        print(f"  {name:<12s} n={len(rs):<3d} mean|d|={mean_d:5.2f}  med={med}  "
              f"within±5={w5:>2d}  within±3={w3:>2d}  within±2={w2:>2d}  "
              f"<-3:{over_neg3:>2d}  >+3:{over_pos3:>2d}")

    print(f"\n{'Strategy':<12s}{'metrics'}")
    summarize("current", results["current"])
    summarize("oracle", results["oracle"])
    summarize("realistic", results["realistic"])

    # Per-action oracle vs current
    print(f"\nPer-action (ORACLE - upper bound if we knew GT player):")
    print(f"  {'Action':<8s}{'n':>4s}{'mean|d|cur':>12s}{'mean|d|orc':>12s}{'w±5 cur':>10s}{'w±5 orc':>10s}")
    by_act = defaultdict(lambda: {"cur": [], "orc": []})
    for (act, d_now, _), (_, d_orc, _) in zip(results["current"], results["oracle"], strict=False):
        by_act[act]["cur"].append(d_now)
        by_act[act]["orc"].append(d_orc)
    for act in ("SERVE", "RECEIVE", "SET", "ATTACK", "DIG", "BLOCK"):
        d = by_act.get(act)
        if not d or not d["cur"]:
            continue
        n = len(d["cur"])
        mc = sum(d["cur"]) / n
        mo = sum(d["orc"]) / n if d["orc"] else 0
        w5c = sum(1 for x in d["cur"] if x <= 5)
        w5o = sum(1 for x in d["orc"] if x <= 5)
        print(f"  {act:<8s}{n:>4d}{mc:>11.2f}{mo:>12.2f}{w5c:>10d}{w5o:>10d}")

    print(f"\nSample cases (delta from GT in frames):")
    for line in sample_lines:
        print(line)

    return 0


if __name__ == "__main__":
    sys.exit(main())
