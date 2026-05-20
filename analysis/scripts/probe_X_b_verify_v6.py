"""Probe X-B verify: confirm v6 asymmetric is_at_net is wired into detect_contacts.

Runs detect_contacts on one of the 4 expected-to-recover NOT_AT_NET cases
(gigi b8d333ae) and prints the isAtNet flag for the contact at the GT
block frame. With v5 (symmetric 0.08) the contact had isAtNet=False;
with v6 (asymmetric -0.15/+0.08), it must be True.

This is a Phase 1 sanity check — if the eval shows no metric change but
this probe shows the flag flipped, then Phase 1 is correctly wired but
needs Phase 3 (missing-attack) to move Block F1.
"""
from __future__ import annotations

import sys

import psycopg

from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_detector import (
    CONTACT_PIPELINE_VERSION,
    detect_contacts,
)
from rallycut.tracking.player_tracker import PlayerPosition

DB_DSN = "postgresql://postgres:postgres@localhost:5436/rallycut"


def main() -> int:
    print(f"CONTACT_PIPELINE_VERSION = {CONTACT_PIPELINE_VERSION}", flush=True)

    # gigi b8d333ae GT block frame = 234, ball.y=0.248, expected net_y=0.373
    # Pre-v6: delta=-0.125, |delta|=0.125 > 0.08 → isAtNet=False
    # Post-v6: -0.15 <= -0.125 <= 0.08 → isAtNet=True
    rally_prefix = "b8d333ae"
    expected_gt_frame = 234

    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            """
            SELECT r.id::text, pt.frame_count, pt.court_split_y,
                   pt.ball_positions_json, pt.positions_json
            FROM rallies r
            JOIN videos v ON r.video_id = v.id
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE v.name = 'gigi' AND r.id::text LIKE %s || '%%'
            LIMIT 1
            """,
            (rally_prefix,),
        )
        row = cur.fetchone()

    if not row:
        print(f"Rally not found: gigi {rally_prefix}", flush=True)
        return 1
    rid, frame_count, court_split_y, bp_json, pos_json = row

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

    print(f"Rally: gigi {rid[:8]}", flush=True)
    print(f"  ball positions: {len(ball_positions)}", flush=True)
    print(f"  player positions: {len(player_positions)}", flush=True)
    print(f"  court_split_y: {court_split_y}", flush=True)

    # Skip the GBM gate for this probe — we just need to verify is_at_net
    # is computed correctly. With use_classifier=False all raw candidates
    # survive and we can read is_at_net directly.
    seq = detect_contacts(
        ball_positions=ball_positions,
        player_positions=player_positions,
        frame_count=frame_count,
        use_classifier=False,
    )
    print(f"  detected contacts: {len(seq.contacts)}", flush=True)
    print(f"  estimated_net_y: {seq.net_y:.3f}", flush=True)
    print()

    # Find contact closest to GT frame
    best = None
    best_d = 1000
    for c in seq.contacts:
        d = abs(c.frame - expected_gt_frame)
        if d < best_d:
            best_d = d
            best = c
    if best is None:
        print("No contacts at all in this rally", flush=True)
        return 2
    print(
        f"Closest contact to GT frame {expected_gt_frame}: "
        f"frame={best.frame} delta={best.frame - expected_gt_frame:+d} "
        f"ball.y={best.ball_y:.3f} "
        f"ball.y-net_y={best.ball_y - seq.net_y:+.3f} "
        f"isAtNet={best.is_at_net}",
        flush=True,
    )
    expected_at_net = -0.15 <= (best.ball_y - seq.net_y) <= 0.08
    print(f"Expected isAtNet under v6: {expected_at_net}", flush=True)
    print(
        f"Verdict: {'PASS' if best.is_at_net == expected_at_net else 'FAIL — v6 not wired'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
