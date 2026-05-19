"""Find the ACTUAL contact frame for each EMPTY-candidate Serve case
and check player visibility at that exact frame.
"""
from __future__ import annotations

import json
import sys

import psycopg

DB_DSN = "postgresql://postgres:postgres@localhost:5436/rallycut"

FAILURES = [
    ("haha", "18175bae", 203),
    ("kaka", "ca5d5f57", 169),
    ("kaka", "f33d7ac8", 213),
    ("ruru", "cc2b967b", 224),
    ("yoyo", "9d24aa93", 90),
]


def main() -> int:
    with psycopg.connect(DB_DSN) as conn:
        for vname, rally_prefix, gt_frame in FAILURES:
            cur = conn.execute(
                """
                SELECT r.id, pt.positions_json, pt.actions_json, pt.contacts_json,
                       pt.frame_count, pt.primary_track_ids,
                       pt.ball_positions_json
                FROM rallies r
                JOIN videos v ON r.video_id = v.id
                JOIN player_tracks pt ON pt.rally_id = r.id
                WHERE v.name = %s AND r.id::text LIKE %s || '%%'
                """,
                (vname, rally_prefix),
            )
            row = cur.fetchone()
            if not row:
                continue
            rid, positions, actions, contacts, frame_count, primary, balls = row
            positions = positions if isinstance(positions, list) else json.loads(positions or '[]')
            actions = actions if isinstance(actions, dict) else json.loads(actions or '{}')
            contacts = contacts if isinstance(contacts, dict) else json.loads(contacts or '{}')
            balls = balls if isinstance(balls, list) else json.loads(balls or '[]')

            # Find the SERVE action near GT frame
            serve = None
            for a in actions.get("actions", []):
                if a.get("action") == "serve" and abs(a.get("frame", -10000) - gt_frame) <= 10:
                    serve = a
                    break
            if not serve:
                # Look at ALL actions to find what's near GT frame
                near = [a for a in actions.get("actions", []) if abs(a.get("frame", -10000) - gt_frame) <= 20]
                print(f"\n=== {vname} {rally_prefix} GT_frame={gt_frame}: NO SERVE within ±10 ===")
                print(f"  Actions near GT frame: {[(a.get('frame'), a.get('action'), a.get('playerTrackId')) for a in near]}")
                continue

            actual_frame = serve["frame"]
            # Find the contact for that frame
            contact = None
            for c in contacts.get("contacts", []):
                if abs(c.get("frame", -10000) - actual_frame) <= 2:
                    contact = c
                    break

            # Count players at the actual frame and surrounding window
            counts: dict[int, int] = {}
            tids_at_exact: list[int] = []
            tids_in_30: dict[int, int] = {}
            for p in positions:
                tid = p.get("trackId")
                fn = p.get("frameNumber", -10000)
                d = abs(fn - actual_frame)
                if d == 0:
                    tids_at_exact.append(tid)
                if d <= 30:
                    tids_in_30[tid] = tids_in_30.get(tid, 0) + 1

            # Ball at frame
            ball_at = None
            for b in balls:
                if abs(b.get("frameNumber", -10000) - actual_frame) <= 2:
                    ball_at = (b.get("x"), b.get("y"), b.get("confidence"))
                    break

            print(f"\n=== {vname} {rally_prefix} ===")
            print(f"  GT_frame={gt_frame} actual_serve_frame={actual_frame} delta={actual_frame-gt_frame:+d}")
            print(f"  primary_track_ids={primary}")
            print(f"  serve.playerTrackId={serve.get('playerTrackId')}, ball=(x={serve.get('ballX'):.3f}, y={serve.get('ballY'):.3f})")
            print(f"  contact.playerCandidates={contact.get('playerCandidates') if contact else 'NO_CONTACT'}")
            print(f"  ball at frame {actual_frame}: {ball_at}")
            print(f"  Players at EXACT frame {actual_frame}: {tids_at_exact}")
            print(f"  Players in ±30f window: {dict(sorted(tids_in_30.items()))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
