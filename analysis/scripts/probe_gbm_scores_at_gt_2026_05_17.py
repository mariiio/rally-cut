#!/usr/bin/env python3
"""For TRAJECTORY_CHANGE_BUT_NO_CONTACT FN cases, run the contact
classifier on the candidate at the GT frame and report the GBM score.

Tells us whether these rejections are 'narrowly below threshold' (a
small threshold lowering would recover) or 'GBM strongly disagrees'
(need feature engineering or retraining).
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rallycut.tracking.ball_tracker import BallPosition  # noqa: E402
from rallycut.tracking.contact_detector import (  # noqa: E402
    ContactDetectionConfig,
    _get_default_classifier,
    _prepare_candidates,
)
from rallycut.tracking.player_tracker import PlayerPosition  # noqa: E402

DB = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5436/rallycut",
)
CSV_IN = Path("reports/contact_fn_trusted29_2026_05_17/fn_cases.csv")
OUT_DIR = Path("reports/contact_fn_trusted29_2026_05_17")


def main() -> int:
    classifier = _get_default_classifier()
    if classifier is None or not classifier.is_trained:
        print("No classifier loaded", file=sys.stderr)
        return 1
    print(f"GBM threshold: {classifier.threshold}")

    fn_rows = list(csv.DictReader(CSV_IN.open()))
    target = [r for r in fn_rows if r["category"] == "TRAJECTORY_CHANGE_BUT_NO_CONTACT"]
    print(f"Probing {len(target)} TRAJECTORY_CHANGE FN cases")

    # We need to extract CandidateFeatures at the GT frame.
    # Simplest: run _prepare_candidates, find candidate frame closest to GT,
    # then extract its features by hand (or run subset of detect_contacts).
    # That's substantial. For a quick probe, let's compute on the GT-frame's
    # nearest candidate from _prepare_candidates and report the GBM probability
    # via the same feature extraction the detector uses.

    # Mirror the inline feature-building from contact_detector.py:
    from rallycut.tracking.contact_classifier import CandidateFeatures  # noqa: E402

    # Pre-fetch rally data
    needed = {(r["video"], r["rally_id_short"]) for r in target}
    rally_data: dict[tuple[str, str], tuple] = {}
    with psycopg.connect(DB) as conn:
        for vname, rid_short in needed:
            cur = conn.execute(
                """SELECT pt.positions_json, pt.ball_positions_json, pt.court_split_y
                FROM rallies r JOIN videos v ON r.video_id=v.id
                JOIN player_tracks pt ON pt.rally_id=r.id
                WHERE v.name=%s AND r.id LIKE %s""",
                [vname, rid_short + "%"],
            )
            row = cur.fetchone()
            if row:
                pj, bj, csy = row
                rally_data[(vname, rid_short)] = (
                    pj if isinstance(pj, list) else json.loads(pj or "[]"),
                    bj if isinstance(bj, list) else json.loads(bj or "[]"),
                    csy,
                )

    bucket_counter: Counter = Counter()
    scores: list[float] = []
    rows_out: list[dict] = []

    for r in target:
        key = (r["video"], r["rally_id_short"])
        if key not in rally_data:
            continue
        positions, balls, court_split = rally_data[key]
        gt_f = int(r["gt_frame"])

        ball_positions = [
            BallPosition(
                frame_number=int(b["frameNumber"]),
                x=float(b["x"]), y=float(b["y"]),
                confidence=float(b.get("confidence", 1.0)),
            )
            for b in balls
            if (b.get("x", 0) > 0 or b.get("y", 0) > 0)
        ]
        player_positions = [
            PlayerPosition(
                frame_number=int(p["frameNumber"]),
                track_id=int(p["trackId"]),
                x=float(p["x"]), y=float(p["y"]),
                width=float(p["width"]), height=float(p["height"]),
                confidence=float(p.get("confidence", 1.0)),
            )
            for p in positions
        ]
        cfg = ContactDetectionConfig()
        try:
            prep = _prepare_candidates(ball_positions, player_positions, cfg)
        except Exception:
            continue
        # Find the candidate frame closest to GT
        near = [f for f in prep.candidate_frames if abs(f - gt_f) <= 5]
        if not near:
            continue
        cand_frame = min(near, key=lambda f: abs(f - gt_f))
        # Find ball at this frame
        ball = next((bp for bp in ball_positions if bp.frame_number == cand_frame), None)
        if ball is None:
            continue
        # Find nearest player
        cands_here = sorted(
            [(p.track_id, ((p.x - ball.x) ** 2 + (p.y - ball.y) ** 2) ** 0.5, p)
             for p in player_positions if p.frame_number == cand_frame],
            key=lambda t: t[1],
        )
        if not cands_here:
            continue
        nearest_tid, nearest_dist, nearest_p = cands_here[0]
        velocity = prep.velocity_lookup.get(cand_frame, 0.0) or 0.0
        has_direction_change = cand_frame in set(prep.direction_change_frames)
        # Build CandidateFeatures with subset of what contact_detector populates.
        # Many fields default — we just need enough for the GBM to score.
        try:
            # Use the empirical direction_change_deg the diagnostic
            # computed (from FN CSV) when available; else 0.
            try:
                dir_change_deg = float(r.get("direction_change_deg") or 0.0)
            except (TypeError, ValueError):
                dir_change_deg = 0.0
            features = CandidateFeatures(
                frame=cand_frame,
                velocity=velocity,
                direction_change_deg=dir_change_deg,
                arc_fit_residual=0.0,
                acceleration=0.0,
                trajectory_curvature=0.0,
                velocity_y=0.0,
                velocity_ratio=1.0,
                player_distance=nearest_dist,
                ball_x=ball.x,
                ball_y=ball.y,
                ball_y_relative_net=ball.y - prep.estimated_net_y,
                is_net_crossing=False,
                frames_since_last=10,
                nearest_hand_ball_dist_min=nearest_dist,
                frames_since_rally_start=cand_frame - prep.first_frame,
            )
            preds = classifier.predict([features])
            is_valid, conf = preds[0]
        except Exception as exc:  # noqa: BLE001
            print(f"  {r['video']} f{gt_f}: feature build failed: {exc}")
            continue
        scores.append(conf)
        bucket = (
            "<0.10" if conf < 0.10
            else "0.10-0.20" if conf < 0.20
            else "0.20-0.30" if conf < 0.30
            else "0.30-0.35" if conf < 0.35
            else "0.35-0.40" if conf < 0.40
            else ">=0.40"
        )
        bucket_counter[bucket] += 1
        rows_out.append({
            "video": r["video"], "rally": r["rally_id_short"],
            "gt_frame": gt_f, "gt_action": r["gt_action"],
            "cand_frame": cand_frame, "gbm_score": round(conf, 3),
            "min_player_dist": r.get("min_player_dist", ""),
            "direction_change_deg": r.get("direction_change_deg", ""),
        })

    print(f"\nScored {len(scores)} candidates (out of {len(target)} target FNs)")
    print(f"  mean: {sum(scores)/max(1,len(scores)):.3f}")
    print(f"  median: {sorted(scores)[len(scores)//2]:.3f}" if scores else "")
    print()
    print(f"Score distribution (threshold={classifier.threshold}):")
    for bucket in ("<0.10", "0.10-0.20", "0.20-0.30", "0.30-0.35", "0.35-0.40", ">=0.40"):
        n = bucket_counter.get(bucket, 0)
        print(f"  {bucket:12s} {n:4d}")

    # If threshold dropped to 0.30, how many recovered?
    n_above_030 = sum(1 for s in scores if s >= 0.30)
    n_above_025 = sum(1 for s in scores if s >= 0.25)
    n_above_020 = sum(1 for s in scores if s >= 0.20)
    print()
    print("Hypothetical recovery if threshold lowered:")
    print(f"  threshold 0.30: {n_above_030}/{len(scores)} recovered")
    print(f"  threshold 0.25: {n_above_025}/{len(scores)} recovered")
    print(f"  threshold 0.20: {n_above_020}/{len(scores)} recovered")

    out_path = OUT_DIR / "gbm_scores_at_gt.csv"
    with out_path.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        w.writeheader()
        w.writerows(rows_out)
    print(f"\nWrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
