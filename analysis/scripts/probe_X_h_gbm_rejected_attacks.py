"""Probe X-H: GBM rejection probabilities for attack candidates near GT blocks.

For each of the 13 H-A3 cases identified by X-G, run detect_contacts with
threshold=0.0 (accept all) and report:
  - GBM probability of the candidate(s) in the attack window
    (gt_block - 15 .. gt_block - 4)
  - The current default threshold = 0.40
  - The seq_max_nonbg signal at that candidate (used by the seq-anchored
    rescue when GBM is below 0.10)
  - The candidate's ball_y - net_y, court_side band, frames_since_last
    (the most likely culprits for low GBM probability)

This falsifies / confirms the following sub-hypotheses for H-A3:
  H-A3a: probabilities cluster in (0.10, 0.40) — "just below threshold",
         a small threshold drop or a context-aware rescue gate would help.
  H-A3b: probabilities < 0.10 — GBM is CONFIDENTLY rejecting; the GBM
         genuinely thinks these aren't contacts. Threshold drop is unsafe.
         Need either GBM retrain with hard-negative→hard-positive flips
         or a STRUCTURAL rescue rule using the upcoming block contact.
  H-A3c: frames_since_last is small (~5-10) and dominates the feature
         contribution — this would mean the GBM is penalising attack
         candidates because a previous accepted contact (the set 5-10
         frames earlier on the same side) lowers its probability. Hint
         from contact_detector.py:2839-2845 — frames_since_last is
         measured from the LAST ACCEPTED contact.
"""
from __future__ import annotations

import json
import sys
from copy import deepcopy

import psycopg

from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_classifier import ContactClassifier
from rallycut.tracking.contact_detector import (
    ContactDetectionConfig,
    detect_contacts,
)
from rallycut.tracking.player_tracker import PlayerPosition
from rallycut.tracking.sequence_action_runtime import get_sequence_probs

DB_DSN = "postgresql://postgres:postgres@localhost:5436/rallycut"

# 13 H-A3 cases (GBM rejects). Each is (video, rally_prefix, gt_block_frame).
H_A3_CASES = [
    ("caco", "9452ee5a", 190),
    ("gigi", "72c8229b", 631),
    ("gigi", "3e07342a", 231),
    ("gigi", "b8d333ae", 234),
    ("juju", "d810943e", 390),
    ("juju", "acada27e", 241),
    ("juju", "c89b346b", 227),
    ("kiki", "a0aba15e", 972),
    ("mimi", "f3695225", 211),
    ("moma", "753a4ec7", 201),
    ("moma", "9bb60892", 205),
    ("popo", "c1052008", 195),
    ("toto", "67b3e1ad", 173),
]


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


def _match_teams_for_rally(rally_id: str, conn) -> dict[int, int] | None:
    cur = conn.execute(
        """
        SELECT v.match_analysis_json FROM rallies r
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
    base_classifier = ContactClassifier.load("weights/contact_classifier/contact_classifier.pkl")
    print(f"Default GBM threshold = {base_classifier.threshold}", flush=True)
    print("Rescue gates: GBM<0.10 AND seq_max_nonbg>=0.95 (analytical: contact_detector.py:99-100)", flush=True)
    print()

    # Permissive: accept all to read every candidate's probability
    permissive = deepcopy(base_classifier)
    permissive.threshold = 0.0

    cfg = ContactDetectionConfig()
    n_below_010 = 0
    n_in_010_040 = 0
    n_in_040_default = 0

    with psycopg.connect(DB_DSN) as conn:
        for vname, prefix, gt_frame in H_A3_CASES:
            cur = conn.execute(
                """
                SELECT r.id::text, pt.frame_count, pt.court_split_y,
                       pt.ball_positions_json, pt.positions_json
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
            rid, frame_count, court_split_y, bp_json, pos_json = row
            ball_positions, player_positions = _build_positions(bp_json, pos_json)
            match_teams = _match_teams_for_rally(rid, conn)

            seq_probs = get_sequence_probs(
                ball_positions=ball_positions,
                player_positions=player_positions,
                court_split_y=court_split_y,
                frame_count=frame_count or 0,
                team_assignments=match_teams,
                calibrator=None,
            )
            permissive_seq = detect_contacts(
                ball_positions=ball_positions,
                player_positions=player_positions,
                config=cfg,
                net_y=court_split_y,
                frame_count=frame_count or None,
                classifier=permissive,
                use_classifier=True,
                team_assignments=match_teams,
                sequence_probs=seq_probs,
            )

            # Implied attack window
            attack_lo = gt_frame - 15
            attack_hi = gt_frame - 4

            in_window = [c for c in permissive_seq.contacts if attack_lo <= c.frame <= attack_hi]
            print(
                f"\n=== {vname} {prefix} (gt block={gt_frame}, attack window=[{attack_lo},{attack_hi}]) ===",
                flush=True,
            )
            print(
                f"  total contacts (threshold=0): {len(permissive_seq.contacts)}",
                flush=True,
            )

            if not in_window:
                print("  NO candidate in attack window even at threshold=0", flush=True)
                continue
            for c in in_window:
                gbm = c.confidence
                bucket = (
                    "GBM<0.10"
                    if gbm < 0.10
                    else "GBM in [0.10, 0.40)"
                    if gbm < 0.40
                    else "GBM in [0.40, default)"
                )
                if gbm < 0.10:
                    n_below_010 += 1
                elif gbm < 0.40:
                    n_in_010_040 += 1
                else:
                    n_in_040_default += 1
                print(
                    f"  frame={c.frame} gbm={gbm:.3f} ({bucket}) "
                    f"is_at_net={c.is_at_net} court={c.court_side} "
                    f"ball.y={c.ball_y:.3f} delta_net={c.ball_y - permissive_seq.net_y:+.3f}",
                    flush=True,
                )

    print(
        f"\n=== H-A3 sub-bucket distribution ({n_below_010 + n_in_010_040 + n_in_040_default} candidates total) ===",
        flush=True,
    )
    print(f"  GBM<0.10 (confidently rejected):     {n_below_010:>3}", flush=True)
    print(f"  GBM in [0.10, 0.40) (just below):    {n_in_010_040:>3}", flush=True)
    print(f"  GBM in [0.40, default) (edge case):  {n_in_040_default:>3}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
