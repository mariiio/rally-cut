"""Probe X-I: instrument the snap + regressor stages on the 13 H-A3 rallies.

For each of the 13 H-A3 cases, monkey-patches
`_snap_contacts_to_direction_change_max` and `refine_contacts_with_regressor`
to capture per-contact pre/post frame snapshots. Then runs the full
`detect_contacts` pipeline and prints, for any contact whose original
frame fell in the attack window (gt_block - 15 .. gt_block - 4), where
that contact ended up after each stage.

Confirms or falsifies the X-H hypothesis: that the snap or regressor
moves attack-window candidates by 6-10 frames INTO the block window,
collapsing the attack+block pair into a single contact.
"""
from __future__ import annotations

import json
import sys
from collections import Counter

import psycopg

from rallycut.tracking import contact_detector as cd_mod
from rallycut.tracking import contact_frame_regressor as cfr_mod
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_classifier import ContactClassifier
from rallycut.tracking.contact_detector import (
    ContactDetectionConfig,
    detect_contacts,
)
from rallycut.tracking.player_tracker import PlayerPosition
from rallycut.tracking.sequence_action_runtime import get_sequence_probs

DB_DSN = "postgresql://postgres:postgres@localhost:5436/rallycut"

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

# Globals updated by the monkey-patches per-call.
_snap_snapshot_pre: list[tuple[int, float, float]] = []
_snap_snapshot_post: list[tuple[int, float, float]] = []
_reg_snapshot_pre: list[tuple[int, float, float]] = []
_reg_snapshot_post: list[tuple[int, float, float]] = []


def _wrap_snap(original):
    def wrapped(contacts, ball_by_frame, *args, **kwargs):
        _snap_snapshot_pre.clear()
        _snap_snapshot_post.clear()
        for c in contacts:
            _snap_snapshot_pre.append((int(c.frame), float(c.ball_x), float(c.ball_y)))
        res = original(contacts, ball_by_frame, *args, **kwargs)
        for c in contacts:
            _snap_snapshot_post.append((int(c.frame), float(c.ball_x), float(c.ball_y)))
        return res

    return wrapped


def _wrap_regressor(original):
    def wrapped(contacts, ball_positions, *args, **kwargs):
        _reg_snapshot_pre.clear()
        _reg_snapshot_post.clear()
        for c in contacts:
            _reg_snapshot_pre.append((int(c.frame), float(c.ball_x), float(c.ball_y)))
        res = original(contacts, ball_positions, *args, **kwargs)
        for c in contacts:
            _reg_snapshot_post.append((int(c.frame), float(c.ball_x), float(c.ball_y)))
        return res

    return wrapped


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


def _match_teams_for_rally(rally_id, conn):
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
    classifier = ContactClassifier.load("weights/contact_classifier/contact_classifier.pkl")
    cfg = ContactDetectionConfig()

    # Install monkey-patches once for the entire run.
    cd_mod._snap_contacts_to_direction_change_max = _wrap_snap(
        cd_mod._snap_contacts_to_direction_change_max
    )
    cfr_mod.refine_contacts_with_regressor = _wrap_regressor(
        cfr_mod.refine_contacts_with_regressor
    )

    rally_into_block_snap = 0
    rally_into_block_reg = 0
    rally_no_attack_candidate = 0
    snap_shifts = Counter()  # (case_id) -> shift size buckets
    reg_shifts = Counter()

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

            seq = detect_contacts(
                ball_positions=ball_positions,
                player_positions=player_positions,
                config=cfg,
                net_y=court_split_y,
                frame_count=frame_count or None,
                classifier=classifier,
                use_classifier=True,
                team_assignments=match_teams,
                sequence_probs=seq_probs,
            )

            attack_lo = gt_frame - 15
            attack_hi = gt_frame - 4
            block_lo = gt_frame - 3
            block_hi = gt_frame + 7

            # Match snap-pre to snap-post by index (order preserved by snap).
            print(f"\n=== {vname} {prefix} (gt block={gt_frame}, "
                  f"attack=[{attack_lo},{attack_hi}], block=[{block_lo},{block_hi}]) ===",
                  flush=True)
            print(f"  PRE-SNAP   contacts: {[p[0] for p in _snap_snapshot_pre]}", flush=True)
            print(f"  POST-SNAP  contacts: {[p[0] for p in _snap_snapshot_post]}", flush=True)
            print(f"  PRE-REG    contacts: {[p[0] for p in _reg_snapshot_pre]}", flush=True)
            print(f"  POST-REG   contacts: {[p[0] for p in _reg_snapshot_post]}", flush=True)
            print(f"  FINAL      contacts (post-dedup): {[c.frame for c in seq.contacts]}", flush=True)

            # Find any pre-snap contact that started in attack window
            attack_pre_snap_idx = [
                i for i, (f, _x, _y) in enumerate(_snap_snapshot_pre)
                if attack_lo <= f <= attack_hi
            ]
            if not attack_pre_snap_idx:
                rally_no_attack_candidate += 1
                print("  → no candidate ENTERS snap in attack window", flush=True)
                continue

            for idx in attack_pre_snap_idx:
                pre_f = _snap_snapshot_pre[idx][0]
                post_snap_f = _snap_snapshot_post[idx][0] if idx < len(_snap_snapshot_post) else None
                post_reg_f = _reg_snapshot_post[idx][0] if idx < len(_reg_snapshot_post) else None
                snap_delta = (post_snap_f - pre_f) if post_snap_f is not None else None
                reg_delta = (post_reg_f - post_snap_f) if post_snap_f is not None and post_reg_f is not None else None
                snap_shifts[abs(snap_delta) if snap_delta is not None else 0] += 1
                reg_shifts[abs(reg_delta) if reg_delta is not None else 0] += 1
                print(
                    f"  candidate idx={idx} pre={pre_f} -> snap={post_snap_f} "
                    f"(Δsnap={snap_delta:+d}) -> reg={post_reg_f} "
                    f"(Δreg={reg_delta:+d})",
                    flush=True,
                )
                # Check if it landed in the BLOCK window after snap or after regressor
                if post_snap_f is not None and block_lo <= post_snap_f <= block_hi:
                    rally_into_block_snap += 1
                    print("    ⚠ SNAPPED INTO BLOCK WINDOW", flush=True)
                if post_reg_f is not None and block_lo <= post_reg_f <= block_hi:
                    rally_into_block_reg += 1
                    print("    ⚠ REGRESSED INTO BLOCK WINDOW", flush=True)

    print("\n=== Distribution ===", flush=True)
    print("Snap shift |delta|:", flush=True)
    for k in sorted(snap_shifts.keys()):
        print(f"  {k:>3} frames: {snap_shifts[k]}", flush=True)
    print("Regressor shift |delta|:", flush=True)
    for k in sorted(reg_shifts.keys()):
        print(f"  {k:>3} frames: {reg_shifts[k]}", flush=True)
    print()
    print(f"Attack-window candidates that ended up in block window after SNAP:      {rally_into_block_snap}", flush=True)
    print(f"Attack-window candidates that ended up in block window after REGRESSOR: {rally_into_block_reg}", flush=True)
    print(f"Rallies with NO attack-window candidate even entering snap stage:        {rally_no_attack_candidate}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
