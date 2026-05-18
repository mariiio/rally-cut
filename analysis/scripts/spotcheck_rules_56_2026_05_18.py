#!/usr/bin/env python3
"""Spot-check whether enabling repair rules 0, 5, 6 would catch additional
action-type errors on the trusted-31 attribution corpus.

For each rally:
  1. Reconstruct ContactSequence from stored contacts_json + ball + positions
  2. Run classify_rally → repair_action_sequence(varying disabled_rules) → viterbi
  3. Match repaired actions to action GT (within ±5 frames, same action type)
  4. Count action-type matches per class, per config

This isolates the EFFECT of repair rules without re-running contact detection
(which the dedicated ablation script does badly per the v4 GBM strictness
diagnosis).

We measure post-repair / post-viterbi, BEFORE apply_sequence_override (which
needs MS-TCN++ probs we don't have stored). Override could later swallow or
amplify any deltas we observe — but seeing a meaningful positive signal here
is necessary even if not sufficient.
"""
from __future__ import annotations

import json
import time
from collections import defaultdict
from typing import Any

import psycopg

from rallycut.tracking.action_classifier import (
    ActionClassifier,
    ActionClassifierConfig,
    ActionType,
    assign_court_side_from_teams,
    repair_action_sequence,
    validate_action_sequence,
    viterbi_decode_actions,
)
from rallycut.tracking.ball_tracker import BallPosition
from rallycut.tracking.contact_detector import (
    Contact,
    ContactSequence,
)
from rallycut.tracking.player_tracker import PlayerPosition

DB = "postgresql://postgres:postgres@localhost:5436/rallycut"

# Action GT is good across all 74 videos. We use the trusted-31 attribution set
# because those are the videos this session has been working with — keeps the
# corpus aligned with [[contact_fn_investigation_2026_05_17]] /
# [[cross_team_prior_no_ship_2026_05_17]] / [[trusted-attribution-corpus]]
# baselines. Action-type errors are what we're measuring, not attribution.
TRUSTED_31 = (
    "titi", "toto", "lulu", "wawa", "caco", "cece", "cici", "cuco",
    "gaga", "gigi", "kaka", "kiki", "keke", "koko", "kuku", "juju",
    "yeye", "gugu", "mame", "meme", "mimi", "moma", "mumu", "papa",
    "pepe", "pipi", "popo", "pupu", "veve", "vivi", "vovo",
)

# BASELINE_DISABLED preserved at its pre-v7-ship value so the table layout
# remains stable across runs. After the 2026-05-18 v7 ship (Rule 6 re-enabled)
# and the 2026-05-18 v8 retrain, current production = the `+rule6` row in
# the output, NOT the `baseline` row. The `baseline` row continues to show
# the rules-only pre-v7 reference point; read deltas accordingly.
# Pre-v7 production: disabled_rules={0, 2, 5, 6}  (rules enabled: 1, 3, 4, 8)
# Post-v7 production: disabled_rules={0, 2, 5}    (rules enabled: 1, 3, 4, 6, 8)
BASELINE_DISABLED = frozenset({0, 2, 5, 6})

CONFIGS: dict[str, frozenset[int]] = {
    "baseline":      BASELINE_DISABLED,                      # {0,2,5,6} disabled
    "+rule0":        frozenset({2, 5, 6}),                   # add 0 back
    "+rule5":        frozenset({0, 2, 6}),                   # add 5 back
    "+rule6":        frozenset({0, 2, 5}),                   # add 6 back
    "+rule5+6":      frozenset({0, 2}),                      # add 5 + 6 back
    "+rule0+5+6":    frozenset({2}),                         # add 0, 5, 6 back
}

MATCH_WINDOW = 5  # frames


def contact_from_dict(d: dict[str, Any]) -> Contact:
    cands_raw = d.get("playerCandidates") or []
    cands: list[tuple[int, float]] = []
    for c in cands_raw:
        if isinstance(c, list) and len(c) == 2:
            tid, dist = c
            cands.append((int(tid), float(dist) if dist is not None else float("inf")))
    return Contact(
        frame=int(d["frame"]),
        ball_x=float(d["ballX"]),
        ball_y=float(d["ballY"]),
        velocity=float(d.get("velocity", 0.0)),
        direction_change_deg=float(d.get("directionChangeDeg", 0.0)),
        player_track_id=int(d.get("playerTrackId", -1)),
        player_distance=(
            float(d["playerDistance"])
            if d.get("playerDistance") is not None
            else float("inf")
        ),
        player_candidates=cands,
        court_side=d.get("courtSide", "unknown"),
        is_at_net=bool(d.get("isAtNet", False)),
        is_validated=bool(d.get("isValidated", True)),
        confidence=float(d.get("confidence", 0.0)),
        arc_fit_residual=float(d.get("arcFitResidual", 0.0)),
    )


def load_rally(conn: psycopg.Connection, video_name: str, rally_order: int) -> dict | None:
    cur = conn.execute(
        """
        SELECT r.id, pt.contacts_json, pt.ball_positions_json, pt.positions_json,
               pt.actions_json->'teamAssignments' AS team_json,
               pt.actions_json->'matchTeamAssignments' AS match_team_json
        FROM rallies r
        JOIN videos v ON r.video_id=v.id
        JOIN player_tracks pt ON pt.rally_id=r.id
        WHERE v.name=%s AND r."order"=%s
        """,
        [video_name, rally_order],
    )
    row = cur.fetchone()
    if not row:
        return None
    rid, cj, bj, pj, team_json, match_team_json = row
    if cj is None:
        return None
    cj_obj = cj if isinstance(cj, dict) else json.loads(cj)
    bj_obj = bj if isinstance(bj, list) else (json.loads(bj) if bj else [])
    pj_obj = pj if isinstance(pj, list) else (json.loads(pj) if pj else [])
    team = team_json if isinstance(team_json, dict) else (json.loads(team_json) if team_json else {})

    contacts = [contact_from_dict(c) for c in cj_obj.get("contacts", [])]
    if not contacts:
        return None

    ball_positions = [
        BallPosition(
            frame_number=bp["frameNumber"], x=bp["x"], y=bp["y"],
            confidence=bp.get("confidence", 1.0),
        )
        for bp in bj_obj if (bp.get("x", 0) or 0) > 0 or (bp.get("y", 0) or 0) > 0
    ]
    player_positions = [
        PlayerPosition(
            frame_number=pp["frameNumber"], track_id=pp["trackId"],
            x=pp["x"], y=pp["y"],
            width=pp["width"], height=pp["height"],
            confidence=pp.get("confidence", 1.0),
        )
        for pp in pj_obj
    ]

    # Team assignments: A→0, B→1 (matches action_classifier convention)
    match_teams = {int(k): (0 if v == "A" else 1) for k, v in team.items()}

    seq = ContactSequence(
        contacts=sorted(contacts, key=lambda c: c.frame),
        net_y=cj_obj.get("netY") or 0.5,
        rally_start_frame=cj_obj.get("rallyStartFrame") or 0,
        ball_positions=ball_positions,
        player_positions=player_positions,
    )
    return {"rid": rid, "seq": seq, "match_teams": match_teams}


def run_pipeline(
    seq: ContactSequence,
    rid: str,
    match_teams: dict[int, int],
    disabled_rules: frozenset[int],
) -> tuple[list[Any], dict[int, int]]:
    ac = ActionClassifier(ActionClassifierConfig())
    result = ac.classify_rally(
        seq, rid,
        match_team_assignments=match_teams,
    )

    repaired, triggers = repair_action_sequence(
        result.actions,
        net_y=seq.net_y,
        ball_positions=seq.ball_positions,
        rally_start_frame=seq.rally_start_frame,
        disabled_rules=set(disabled_rules),
    )
    result.actions = repaired
    result.actions = viterbi_decode_actions(result.actions)
    result.actions = validate_action_sequence(result.actions, rid)
    if match_teams:
        assign_court_side_from_teams(result.actions, match_teams)
    return result.actions, triggers


def match_to_gt(actions: list[Any], gt: list[tuple[int, str]]) -> dict[str, dict[str, int]]:
    """Per-class action match: was the GT action present at the correct frame
    with the correct action type?

    Returns: { gt_action: {'total': N, 'correct': N, 'wrong_type': N, 'missing': N} }
    """
    stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0, "wrong_type": 0, "missing": 0}
    )
    real_actions = [a for a in actions if not getattr(a, "is_synthetic", False)]
    used = set()
    for gt_frame, gt_action in gt:
        stats[gt_action]["total"] += 1
        # Find closest action (any type) within window
        best, best_delta = None, MATCH_WINDOW + 1
        for i, a in enumerate(real_actions):
            if i in used:
                continue
            delta = abs(a.frame - gt_frame)
            if delta < best_delta:
                best, best_delta, best_i = a, delta, i
        if best is None or best_delta > MATCH_WINDOW:
            stats[gt_action]["missing"] += 1
            continue
        used.add(best_i)
        pred_type = best.action_type.value if isinstance(best.action_type, ActionType) else str(best.action_type)
        if pred_type == gt_action.lower():
            stats[gt_action]["correct"] += 1
        else:
            stats[gt_action]["wrong_type"] += 1
    return stats


def main() -> None:
    t0 = time.monotonic()
    print(f"Loading action GT + contacts for {len(TRUSTED_31)} videos…", flush=True)

    # Aggregated per-config per-class stats
    agg: dict[str, dict[str, dict[str, int]]] = {
        cname: defaultdict(lambda: {"total": 0, "correct": 0, "wrong_type": 0, "missing": 0})
        for cname in CONFIGS
    }
    triggers_agg: dict[str, dict[int, int]] = {cname: defaultdict(int) for cname in CONFIGS}
    rallies_evaluated = 0
    rallies_skipped = 0

    with psycopg.connect(DB) as conn:
        cur = conn.execute(
            """
            SELECT v.name, r."order", r.id,
                   array_agg(json_build_array(rg.frame::text, rg.action::text) ORDER BY rg.frame) AS gt
            FROM rally_action_ground_truth rg
            JOIN rallies r ON rg.rally_id=r.id
            JOIN videos v ON r.video_id=v.id
            WHERE v.name = ANY(%s)
            GROUP BY v.name, r."order", r.id
            ORDER BY v.name, r."order"
            """,
            [list(TRUSTED_31)],
        )
        gt_rows = cur.fetchall()

    print(f"  {len(gt_rows)} rally×GT rows in DB", flush=True)

    last_video = None
    for v_idx, (vname, rno, _rid, gt_raw) in enumerate(gt_rows):
        if vname != last_video:
            if last_video is not None:
                print("", flush=True)
            print(f"[{v_idx+1}/{len(gt_rows)}] {vname}", end=" ", flush=True)
            last_video = vname
        gt = [(int(item[0]), item[1].upper()) for item in gt_raw]

        with psycopg.connect(DB) as conn:
            rd = load_rally(conn, vname, rno)
        if rd is None:
            rallies_skipped += 1
            print(f"r{rno}=skip", end=" ", flush=True)
            continue
        rallies_evaluated += 1
        print(f"r{rno}", end=" ", flush=True)

        for cname, disabled in CONFIGS.items():
            actions, triggers = run_pipeline(
                rd["seq"], rd["rid"], rd["match_teams"], disabled,
            )
            stats = match_to_gt(actions, gt)
            for gt_action, s in stats.items():
                for k in ("total", "correct", "wrong_type", "missing"):
                    agg[cname][gt_action][k] += s[k]
            for rule_id, count in triggers.items():
                triggers_agg[cname][rule_id] += count

    print(f"\n\nEvaluated {rallies_evaluated} rallies ({rallies_skipped} skipped, no contacts)\n", flush=True)

    # Summary per config
    classes = ["SERVE", "RECEIVE", "SET", "ATTACK", "DIG"]

    print(f"{'config':<14s} {'tot.triggers':>12s} {'overall':>10s}   " + "  ".join(f"{c:>10s}" for c in classes), flush=True)
    print("-" * (14 + 12 + 1 + 10 + 3 + (10 + 2) * len(classes)), flush=True)
    for cname in CONFIGS:
        per_class = agg[cname]
        total_correct = sum(per_class[c]["correct"] for c in classes)
        total_total = sum(per_class[c]["total"] for c in classes)
        overall = total_correct / max(1, total_total)
        trigs = sum(triggers_agg[cname].values())
        per_class_acc = [
            f"{(per_class[c]['correct'] / max(1, per_class[c]['total'])):.1%} ({per_class[c]['correct']}/{per_class[c]['total']})"
            for c in classes
        ]
        print(
            f"{cname:<14s} {trigs:>12d} {overall:>10.1%}   " + "  ".join(f"{x:>10s}" for x in per_class_acc),
            flush=True,
        )

    # Trigger detail per rule per config
    print(f"\n{'Rule triggers per config':<40s}")
    print(f"{'config':<14s} " + "  ".join(f"r{r}" for r in range(9)), flush=True)
    for cname in CONFIGS:
        line = f"{cname:<14s} "
        line += "  ".join(f"{triggers_agg[cname].get(r, 0):>3d}" for r in range(9))
        print(line, flush=True)

    # Delta from baseline
    print(f"\nDeltas vs baseline (per-class correct count change):")
    print(f"{'config':<14s} {'overall_Δ':>10s}   " + "  ".join(f"{c:>10s}" for c in classes))
    base = agg["baseline"]
    base_overall = sum(base[c]["correct"] for c in classes)
    for cname in CONFIGS:
        if cname == "baseline":
            continue
        per_class = agg[cname]
        overall = sum(per_class[c]["correct"] for c in classes)
        d_overall = overall - base_overall
        deltas = [per_class[c]["correct"] - base[c]["correct"] for c in classes]
        print(
            f"{cname:<14s} {d_overall:>+10d}   "
            + "  ".join(f"{d:>+10d}" for d in deltas),
            flush=True,
        )

    print(f"\nTotal time: {time.monotonic() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
