"""Probe X-G: per-stage candidate audit for the 27 trusted-31 GT blocks.

For each GT block, runs the candidate-generation phase (`_prepare_candidates`)
and the full pipeline (`detect_contacts`) and reports:

  - what generators fire near GT block frame (and near `block_frame - 5..15`
    where the implied ATTACK should sit)
  - what candidates survive the merge chain
  - what frames the final pipeline emits within ±15 of GT block
  - what closeness band (±0..±2, ±3..±5, ±6..±10, ±11..±15, miss) each
    GT block falls into

Output: CSV per-block + 4-way distribution falsifying one of:

  H-A1: candidates ARE generated near both attack-frame and block-frame,
         merger collapses cross-side pairs at 5-7 frame spacing.
         (Look for: attack-side & block-side candidates at gap ≤ 7 in the
         per-generator output but only one survives in the final list.)

  H-A2: only one candidate is generated (the other has no signal at all
         within ±5 frames).

  H-A3: candidates survive merging but GBM rejects them.
         (Look for: candidate exists in final candidate_frames list but
         not in detected contacts.)

  H-A4: contact lands at ±8-10 from GT (right at eval ±7 tolerance bound).
         (Look for: detected contact exists but offset >7 from GT.)
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import psycopg

from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_classifier import ContactClassifier
from rallycut.tracking.contact_detector import (
    ContactDetectionConfig,
    _prepare_candidates,
    detect_contacts,
)
from rallycut.tracking.player_tracker import PlayerPosition
from rallycut.tracking.sequence_action_runtime import get_sequence_probs

DB_DSN = "postgresql://postgres:postgres@localhost:5436/rallycut"
TRUSTED_31 = (
    "titi,toto,lulu,wawa,caco,cece,cici,cuco,gaga,gigi,kaka,kiki,keke,koko,"
    "kuku,juju,yeye,gugu,mame,meme,mimi,moma,mumu,papa,pepe,pipi,popo,pupu,"
    "veve,vivi,vovo"
).split(",")

OUT_CSV = Path("reports/contact_fps_2026_05_19/probe_X_g_block_stages.csv")


def _build_positions(bp_json, pos_json):
    ball_positions = [
        BallPosition(
            frame_number=bp["frameNumber"], x=bp["x"], y=bp["y"],
            confidence=bp.get("confidence", 1.0),
        )
        for bp in bp_json
        if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
    ]
    player_positions = [
        PlayerPosition(
            frame_number=pp.get("frameNumber"),
            track_id=pp.get("trackId", -1),
            x=pp.get("x", 0.0), y=pp.get("y", 0.0),
            width=pp.get("width", 0.05), height=pp.get("height", 0.15),
            confidence=pp.get("confidence", 0.9),
        )
        for pp in pos_json
    ]
    return ball_positions, player_positions


def _frames_near(frames: list[int], target: int, window: int = 15) -> list[int]:
    return sorted([f for f in frames if abs(f - target) <= window])


def _match_teams_for_rally(rally_id: str, conn) -> dict[int, int] | None:
    cur = conn.execute(
        """
        SELECT v.match_analysis_json
        FROM rallies r
        JOIN videos v ON r.video_id = v.id
        WHERE r.id::text = %s
        """,
        (rally_id,),
    )
    row = cur.fetchone()
    if not row or not row[0]:
        return None
    ma = row[0] if isinstance(row[0], dict) else json.loads(row[0])
    for r in ma.get("rallies", []):
        if r.get("rallyId") == rally_id and r.get("assignmentConfidence", 0.0) >= 0.70:
            t2t = r.get("teamAssignments")
            if t2t:
                return {int(k): int(v) for k, v in t2t.items()}
    return None


def main() -> int:
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    classifier = ContactClassifier.load("weights/contact_classifier/contact_classifier.pkl")
    cfg = ContactDetectionConfig()

    with psycopg.connect(DB_DSN) as conn:
        placeholders = ",".join(["%s"] * len(TRUSTED_31))
        cur = conn.execute(
            f"""
            SELECT v.name, r.id::text, gt.frame AS gt_frame,
                   pt.frame_count, pt.court_split_y,
                   pt.ball_positions_json, pt.positions_json
            FROM rally_action_ground_truth gt
            JOIN rallies r ON gt.rally_id = r.id
            JOIN videos v ON r.video_id = v.id
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE v.name IN ({placeholders})
              AND gt.action = 'BLOCK'
              AND gt.resolved_track_id IS NOT NULL
            ORDER BY v.name, r.start_ms
            """,
            TRUSTED_31,
        )
        rows = cur.fetchall()
    print(f"Loaded {len(rows)} GT blocks across trusted-31\n", flush=True)

    out_rows = []
    pattern_counter = {"H-A1": 0, "H-A2": 0, "H-A3": 0, "H-A4": 0, "DETECTED_AT_GT": 0}

    with psycopg.connect(DB_DSN) as conn:
        for vname, rid, gt_frame, frame_count, court_split_y, bp_json, pos_json in rows:
            ball_positions, player_positions = _build_positions(bp_json, pos_json)
            match_teams = _match_teams_for_rally(rid, conn)

            prep = _prepare_candidates(ball_positions, player_positions, cfg)

            seq_probs = get_sequence_probs(
                ball_positions=ball_positions,
                player_positions=player_positions,
                court_split_y=court_split_y,
                frame_count=frame_count or 0,
                team_assignments=match_teams,
                calibrator=None,
            )
            seq = detect_contacts(
                ball_positions=ball_positions,
                player_positions=player_positions,
                frame_count=frame_count or None,
                classifier=classifier,
                use_classifier=True,
                team_assignments=match_teams,
                sequence_probs=seq_probs,
            )

            # Implied attack frame range: 4-15 frames before GT block, on
            # opposite court side.
            attack_lo = gt_frame - 15
            attack_hi = gt_frame - 4

            # Per-generator hits near GT block frame and the implied attack
            # frame.
            gen_at_block = {
                "vel_peak": _frames_near(prep.velocity_peak_frames, gt_frame),
                "inflect": _frames_near(prep.inflection_frames, gt_frame),
                "decel": _frames_near(prep.deceleration_frames, gt_frame),
                "parabolic": _frames_near(prep.parabolic_frames, gt_frame),
                "dir_change": _frames_near(prep.direction_change_frames, gt_frame),
                "net_cross": _frames_near(prep.net_crossing_frames, gt_frame),
            }
            gen_at_attack = {
                "vel_peak": [f for f in prep.velocity_peak_frames if attack_lo <= f <= attack_hi],
                "inflect": [f for f in prep.inflection_frames if attack_lo <= f <= attack_hi],
                "decel": [f for f in prep.deceleration_frames if attack_lo <= f <= attack_hi],
                "parabolic": [f for f in prep.parabolic_frames if attack_lo <= f <= attack_hi],
                "dir_change": [f for f in prep.direction_change_frames if attack_lo <= f <= attack_hi],
                "net_cross": [f for f in prep.net_crossing_frames if attack_lo <= f <= attack_hi],
            }

            cand_at_block = _frames_near(prep.candidate_frames, gt_frame)
            cand_at_attack = [f for f in prep.candidate_frames if attack_lo <= f <= attack_hi]

            detected_at_block = [c.frame for c in seq.contacts if abs(c.frame - gt_frame) <= 15]
            detected_at_attack = [c.frame for c in seq.contacts if attack_lo <= c.frame <= attack_hi]

            # Distance from GT to nearest detected contact
            nearest = min((abs(c.frame - gt_frame) for c in seq.contacts), default=999)

            # Pattern classification (single dominant cause per case)
            any_gen_at_block = any(v for v in gen_at_block.values())
            any(v for v in gen_at_attack.values())
            cand_present_block = bool(cand_at_block)
            bool(cand_at_attack)
            detected_at_block_arr = bool(detected_at_block)

            if nearest <= 7:
                pattern = "DETECTED_AT_GT"  # block-aligned contact exists, eval would match
            elif 8 <= nearest <= 15:
                pattern = "H-A4"            # detected but at frame ±8-15 (outside eval tolerance)
            elif not any_gen_at_block:
                pattern = "H-A2"            # no signal at GT block frame
            elif cand_present_block and not detected_at_block_arr:
                pattern = "H-A3"            # generator fired, GBM rejected
            elif any_gen_at_block and not cand_present_block:
                pattern = "H-A1"            # merger collapsed cross-side pairs
            else:
                pattern = "OTHER"
            pattern_counter[pattern] = pattern_counter.get(pattern, 0) + 1

            out_row = {
                "video": vname, "rally": rid[:8], "gt_frame": gt_frame,
                "nearest_detected": nearest,
                "pattern": pattern,
                "gen_at_block": {k: v for k, v in gen_at_block.items() if v},
                "gen_at_attack": {k: v for k, v in gen_at_attack.items() if v},
                "cand_at_block": cand_at_block,
                "cand_at_attack": cand_at_attack,
                "detected_at_block": detected_at_block,
                "detected_at_attack": detected_at_attack,
            }
            out_rows.append(out_row)
            print(
                f"{vname:<6} {rid[:8]:<10} gt={gt_frame:>4} nearest=±{nearest:<3} "
                f"pat={pattern:<14} "
                f"gen@blk={sum(len(v) for v in gen_at_block.values())} "
                f"gen@att={sum(len(v) for v in gen_at_attack.values())} "
                f"cand@blk={len(cand_at_block)} cand@att={len(cand_at_attack)} "
                f"det@blk={detected_at_block} det@att={detected_at_attack}",
                flush=True,
            )

    # Distribution summary
    print("\nPattern distribution:")
    for k, v in pattern_counter.items():
        print(f"  {k:<14} {v:>3}")

    # CSV dump
    with OUT_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        for r in out_rows:
            r2 = {k: (json.dumps(v) if not isinstance(v, (str, int, float)) else v) for k, v in r.items()}
            w.writerow(r2)
    print(f"\nWrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
