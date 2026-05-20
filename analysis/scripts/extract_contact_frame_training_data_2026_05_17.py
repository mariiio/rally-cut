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

import argparse
import csv
import json
import math
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


def _in_memory_contacts(
    rally_id: str, video_id: str, balls_dict: dict, pos_by_frame: dict,
    court_split_y, frame_count, calibrator, match_teams,
) -> list[dict]:
    """Run detect_contacts in-memory with current default classifier.

    Returns contacts list in the same shape `rd["contacts"]` would have
    (each entry has at least a `frame` key).
    """
    # Lazy imports to keep --no-in-memory path free of pipeline deps
    from rallycut.tracking.ball_tracker import BallPosition
    from rallycut.tracking.contact_detector import detect_contacts
    from rallycut.tracking.player_tracker import PlayerPosition
    from rallycut.tracking.sequence_action_runtime import get_sequence_probs

    ball_positions = [
        BallPosition(frame_number=fn, x=b["x"], y=b["y"], confidence=1.0)
        for fn, b in sorted(balls_dict.items())
    ]
    player_positions = []
    for fn, pos_list in sorted(pos_by_frame.items()):
        for p in pos_list:
            player_positions.append(PlayerPosition(
                frame_number=fn,
                track_id=p.get("trackId", 0),
                x=p.get("x", 0), y=p.get("y", 0),
                width=p.get("width", 0), height=p.get("height", 0),
                confidence=p.get("confidence", 0.9),
            ))
    try:
        sequence_probs = get_sequence_probs(
            ball_positions, player_positions, court_split_y,
            frame_count or 0, match_teams, calibrator=calibrator,
        )
        cs = detect_contacts(
            ball_positions=ball_positions,
            player_positions=player_positions,
            frame_count=frame_count or None,
            court_calibrator=calibrator,
            team_assignments=match_teams,
            sequence_probs=sequence_probs,
        )
        return [{"frame": c.frame} for c in cs.contacts]
    except Exception as e:
        print(f"  WARN rally {rally_id[:8]} in-memory detect failed: {e}", flush=True)
        return []


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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--in-memory", action="store_true",
        help="Run detect_contacts in-memory with current default classifier "
        "for the candidate frames, instead of reading pt.contacts_json from "
        "DB. Used to regenerate regressor training data after retraining "
        "the contact_classifier without writing to the DB.",
    )
    ap.add_argument(
        "--output-suffix", type=str, default="",
        help="Append this suffix to training_data.csv to avoid clobbering "
        "the v4 training set. E.g. --output-suffix _post_60fps_retrain",
    )
    args = ap.parse_args()

    print("Fetching full-corpus action GT rows + rally data...", flush=True)
    select_cols = (
        "v.name, v.id AS video_id, r.id, rg.action::text, rg.frame, "
        "rg.resolved_track_id, pt.ball_positions_json, pt.positions_json, "
        "pt.contacts_json, pt.court_split_y, pt.frame_count"
    )
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            f"""
            SELECT {select_cols}
            FROM rally_action_ground_truth rg
            JOIN rallies r ON rg.rally_id = r.id
            JOIN videos v ON r.video_id = v.id
            JOIN player_tracks pt ON pt.rally_id = r.id
            """,
        )
        rows = cur.fetchall()
    print(f"Got {len(rows)} GT rows across full corpus", flush=True)

    # If --in-memory, pre-load court calibrators + match teams per rally
    calibrators: dict = {}
    match_teams_by_rally: dict = {}
    if args.in_memory:
        from rallycut.court.calibration import CourtCalibrator
        from rallycut.evaluation.tracking.db import load_court_calibration
        from rallycut.tracking.match_tracker import build_match_team_assignments
        from rallycut.tracking.player_tracker import PlayerPosition

        print("Pre-loading calibrators + match team assignments...", flush=True)
        seen_videos = {r[1] for r in rows}
        for vid in seen_videos:
            corners = load_court_calibration(vid)
            if corners and len(corners) == 4:
                cal = CourtCalibrator()
                cal.calibrate([(c["x"], c["y"]) for c in corners])
                calibrators[vid] = cal
            else:
                calibrators[vid] = None

        with psycopg.connect(DB_DSN) as conn2:
            cur2 = conn2.execute(
                "SELECT rally_id, positions_json FROM player_tracks "
                "WHERE positions_json IS NOT NULL",
            )
            rally_positions: dict = {}
            for rid_raw, pos_raw in cur2.fetchall():
                rid_s = str(rid_raw)
                pos_list = pos_raw if isinstance(pos_raw, list) else []
                rally_positions[rid_s] = [
                    PlayerPosition(
                        frame_number=p.get("frameNumber", 0),
                        track_id=p.get("trackId", 0),
                        x=p.get("x", 0), y=p.get("y", 0),
                        width=p.get("width", 0), height=p.get("height", 0),
                        confidence=p.get("confidence", 0),
                    )
                    for p in pos_list if isinstance(p, dict)
                ]
            cur3 = conn2.execute(
                "SELECT id, match_analysis_json FROM videos "
                "WHERE match_analysis_json IS NOT NULL",
            )
            for _vid, mj_raw in cur3.fetchall():
                if not mj_raw:
                    continue
                match_teams_by_rally.update(
                    build_match_team_assignments(
                        mj_raw, min_confidence=0.0,
                        rally_positions=rally_positions,
                    )
                )
        print(
            f"  Calibrators: {sum(1 for c in calibrators.values() if c)}/{len(calibrators)}, "
            f"match teams: {len(match_teams_by_rally)} rallies", flush=True,
        )

    # Group by rally for efficient processing
    rally_cache: dict = {}
    n_no_candidate = 0
    n_examples = 0
    n_skipped_no_features = 0

    feature_names = None
    out_rows = []

    rally_count = 0
    for video, video_id, rid, ga, gf, gtid, bj, pj, cj, court_split_y, frame_count in rows:
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
            if args.in_memory:
                contacts = _in_memory_contacts(
                    str(rid), str(video_id), balls, dict(pos_by_frame),
                    court_split_y, frame_count,
                    calibrators.get(str(video_id)),
                    match_teams_by_rally.get(str(rid)),
                )
                rally_count += 1
                if rally_count % 50 == 0:
                    print(f"  [{rally_count} rallies processed]", flush=True)
            else:
                contacts = cj.get("contacts") or []
            rally_cache[rid] = {
                "balls": balls,
                "pos_by_frame": dict(pos_by_frame),
                "contacts": contacts,
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
    out_path = OUT_DIR / f"training_data{args.output_suffix}.csv"
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
    print("\nTarget offset distribution:")
    print(f"  Range: [{min(targets)}, {max(targets)}]")
    print(f"  Mean: {sum(targets)/len(targets):.2f}, MeanAbs: {sum(abs(t) for t in targets)/len(targets):.2f}")
    print(f"  Within ±2: {sum(1 for t in targets if abs(t) <= 2)} ({sum(1 for t in targets if abs(t) <= 2)/len(targets)*100:.1f}%)")
    print(f"  Within ±5: {sum(1 for t in targets if abs(t) <= 5)} ({sum(1 for t in targets if abs(t) <= 5)/len(targets)*100:.1f}%)")
    print(f"  Outside ±5: {sum(1 for t in targets if abs(t) > 5)} ({sum(1 for t in targets if abs(t) > 5)/len(targets)*100:.1f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
