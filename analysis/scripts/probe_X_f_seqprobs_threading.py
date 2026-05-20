"""Probe X-F: does threading sequence_probs into classify_rally_actions recover stored blocks?

Hypothesis (F4 in the plan): eval_action_detection.py:1249 calls
classify_rally_actions WITHOUT passing the seq_probs it just computed.
action_classifier.py:4187-4189 only fires apply_sequence_override when
sequence_probs is not None — and apply_sequence_override (line 201-302)
can rewrite an action to ANY MS-TCN++ argmax class including 'block'.

Probe-from-stored found 6 GT-block frames already labeled 'block' in
the DB but eval re-detect produces 0 blocks. Theory: the MS-TCN++
override produced those 6 blocks at original ship time, and eval
isn't replicating it.

Test: for 3 of the 6 ALREADY_BLOCK rallies, re-run the eval-equivalent
pipeline twice — once WITHOUT seq_probs (the current eval behavior)
and once WITH seq_probs (the would-be fix). Compare action labels at
the GT block frames.
"""
from __future__ import annotations

import json
import sys

import psycopg

from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_classifier import ContactClassifier
from rallycut.tracking.contact_detector import detect_contacts
from rallycut.tracking.player_tracker import PlayerPosition
from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.sequence_action_runtime import get_sequence_probs

DB_DSN = "postgresql://postgres:postgres@localhost:5436/rallycut"

# Subset of the 6 ALREADY_BLOCK cases — 3 are enough to falsify.
CASES = [
    ("kiki", "a0aba15e", 972),
    ("toto", "67b3e1ad", 173),
    ("yeye", "2d3cb54b", 509),
]


def _build_match_teams(rally_id: str, conn) -> dict[int, int] | None:
    """Fetch high-confidence match-team assignments for the rally."""
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
    contact_classifier = ContactClassifier.load(
        "weights/contact_classifier/contact_classifier.pkl"
    )

    with psycopg.connect(DB_DSN) as conn:
        for vname, prefix, gt_frame in CASES:
            cur = conn.execute(
                """
                SELECT r.id::text, pt.frame_count, pt.court_split_y,
                       pt.ball_positions_json, pt.positions_json,
                       pt.actions_json
                FROM rallies r
                JOIN videos v ON r.video_id = v.id
                JOIN player_tracks pt ON pt.rally_id = r.id
                WHERE v.name = %s AND r.id::text LIKE %s || '%%'
                LIMIT 1
                """,
                (vname, prefix),
            )
            row = cur.fetchone()
            if not row:
                print(f"\n{vname} {prefix}: NOT FOUND", flush=True)
                continue
            rid, frame_count, court_split_y, bp_json, pos_json, aj = row

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
            match_teams = _build_match_teams(rid, conn)

            stored_actions = (aj or {}).get("actions", [])
            stored_at_gt = [a for a in stored_actions if abs(a.get("frame", -1) - gt_frame) <= 7]
            print(f"\n=== {vname} {prefix} (rally={rid[:8]}, GT block frame={gt_frame}) ===", flush=True)
            print(
                f"STORED actions within ±7 of GT: "
                f"{[(a.get('frame'), a.get('action')) for a in stored_at_gt]}",
                flush=True,
            )

            seq_probs = get_sequence_probs(
                ball_positions=ball_positions,
                player_positions=player_positions,
                court_split_y=court_split_y,
                frame_count=frame_count or 0,
                team_assignments=match_teams,
                calibrator=None,
            )

            contacts = detect_contacts(
                ball_positions=ball_positions,
                player_positions=player_positions,
                net_y=court_split_y,
                frame_count=frame_count or None,
                classifier=contact_classifier,
                use_classifier=True,
                team_assignments=match_teams,
                sequence_probs=seq_probs,
            )
            print(f"  detected contacts: {len(contacts.contacts)}", flush=True)

            # Arm A: WITHOUT sequence_probs (current eval behaviour)
            ra_a = classify_rally_actions(
                contacts, rid,
                use_classifier=True,
                match_team_assignments=match_teams,
            )
            arm_a = [(a.frame, a.action_type.value) for a in ra_a.actions
                     if abs(a.frame - gt_frame) <= 7]

            # Arm B: WITH sequence_probs (the would-be fix)
            ra_b = classify_rally_actions(
                contacts, rid,
                use_classifier=True,
                match_team_assignments=match_teams,
                sequence_probs=seq_probs,
            )
            arm_b = [(a.frame, a.action_type.value) for a in ra_b.actions
                     if abs(a.frame - gt_frame) <= 7]

            print(f"  arm A (no seq_probs): {arm_a}", flush=True)
            print(f"  arm B (seq_probs):    {arm_b}", flush=True)
            verdict = "DIFFERENT" if arm_a != arm_b else "IDENTICAL"
            print(f"  VERDICT: {verdict}", flush=True)
            if verdict == "DIFFERENT":
                a_block = any(t == "block" for _, t in arm_a)
                b_block = any(t == "block" for _, t in arm_b)
                print(
                    f"  block label arm A={a_block} arm B={b_block} "
                    f"→ seq_probs threading {'helps' if (b_block and not a_block) else 'changes-but-not-block'}",
                    flush=True,
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
