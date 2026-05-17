#!/usr/bin/env python3
"""Probe the contact-detector to find which gate rejects the
TRAJECTORY_CHANGE_BUT_NO_CONTACT FN cases.

For each FN case:
  1. Run _prepare_candidates → list of candidate frames per generator
  2. Check whether any candidate frame is within ±5f of GT
  3. If yes → candidate was generated; the per-candidate validation/
     classifier loop rejected it
  4. If no → no generator fired; thresholds are too tight

Outputs per-case verdict: which generator (if any) caught the GT frame.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rallycut.tracking.ball_tracker import BallPosition  # noqa: E402
from rallycut.tracking.contact_detector import (  # noqa: E402
    ContactDetectionConfig,
    _prepare_candidates,
)
from rallycut.tracking.player_tracker import PlayerPosition  # noqa: E402

DB = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5436/rallycut",
)
CSV_IN = Path("reports/contact_fn_trusted29_2026_05_17/fn_cases.csv")
OUT_DIR = Path("reports/contact_fn_trusted29_2026_05_17")
OUT_PATH = OUT_DIR / "gate_diagnosis.csv"


def main() -> int:
    fn_rows = list(csv.DictReader(CSV_IN.open()))
    # Focus on the highest-signal category first
    target = [r for r in fn_rows if r["category"] == "TRAJECTORY_CHANGE_BUT_NO_CONTACT"]
    print(f"Probing {len(target)} TRAJECTORY_CHANGE FN cases")

    # Pre-fetch rally data per (video, rally_id_short)
    needed = {(r["video"], r["rally_id_short"]) for r in target}
    rally_data: dict[tuple[str, str], tuple] = {}
    with psycopg.connect(DB) as conn:
        for vname, rid_short in needed:
            cur = conn.execute(
                """SELECT pt.positions_json, pt.ball_positions_json,
                          pt.court_split_y
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

    out_rows: list[dict] = []
    verdict_counter: Counter = Counter()
    generator_hit_counter: Counter = Counter()
    for r in target:
        key = (r["video"], r["rally_id_short"])
        if key not in rally_data:
            continue
        positions, balls, court_split = rally_data[key]
        gt_f = int(r["gt_frame"])

        # Build BallPosition + PlayerPosition lists
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
        except Exception as exc:  # noqa: BLE001
            print(f"  prep failed for {r['video']} f{gt_f}: {exc}")
            continue

        # Which generators caught the GT frame within ±5?
        gen_hits = []
        for gen_name, gen_set in [
            ("velocity_peak", set(prep.velocity_peak_frames)),
            ("inflection", set(prep.inflection_frames)),
            ("deceleration", set(prep.deceleration_frames)),
            ("parabolic", set(prep.parabolic_frames)),
            ("direction_change", set(prep.direction_change_frames)),
            ("net_crossing", set(prep.net_crossing_frames)),
        ]:
            if any(abs(f - gt_f) <= 5 for f in gen_set):
                gen_hits.append(gen_name)
                generator_hit_counter[gen_name] += 1

        # Was ANY candidate frame near GT?
        candidate_near_gt = any(abs(f - gt_f) <= 5 for f in prep.candidate_frames)

        if not candidate_near_gt:
            verdict = "NO_CANDIDATE_GENERATED"
        elif gen_hits:
            verdict = f"GENERATED_BY_{gen_hits[0]}"  # report first generator
            # If candidate was generated but no contact at GT, GBM/rescue rejected
            verdict = "CANDIDATE_GENERATED_BUT_REJECTED"
        else:
            verdict = "CANDIDATE_NEAR_BUT_NO_GENERATOR_HIT"

        verdict_counter[verdict] += 1
        out_rows.append({
            "video": r["video"], "rally": r["rally_id_short"],
            "gt_frame": gt_f, "gt_action": r["gt_action"],
            "min_player_dist": r.get("min_player_dist", ""),
            "direction_change_deg": r.get("direction_change_deg", ""),
            "verdict": verdict,
            "generators_hit": ",".join(gen_hits) if gen_hits else "(none)",
            "n_candidate_frames_near_gt": sum(
                1 for f in prep.candidate_frames if abs(f - gt_f) <= 5
            ),
        })

    print()
    print("Verdict breakdown:")
    for v, n in verdict_counter.most_common():
        print(f"  {v:45s} {n:4d}  ({n/max(1,len(out_rows))*100:5.1f}%)")
    print()
    print("Generator hit counts (which generator caught the GT frame):")
    for g, n in generator_hit_counter.most_common():
        print(f"  {g:25s} {n:4d}")

    with OUT_PATH.open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nWrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
