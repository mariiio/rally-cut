#!/usr/bin/env python3
"""Smoke test: joint Viterbi over (action_type, player_id) on one rally.

Pulls a rally's contacts + positions + team assignments from DB, builds
emission probabilities by calling the existing ActionTypeClassifier and
DynamicAttributionScorer for each candidate × action_type, then runs the
joint Viterbi from `rallycut.tracking.joint_viterbi`. Prints the path
alongside the current pipeline's action sequence for visual comparison.

Phase 1B of task #72. Validates that the joint Viterbi produces a
sensible output on real data. Not yet integrated into production
classify_rally_actions.

Usage:
    cd analysis
    uv run python scripts/probe_joint_viterbi_2026_05_17.py <rally_id>
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rallycut.tracking.action_classifier import (  # noqa: E402
    ActionType,
    _get_default_action_classifier,
)
from rallycut.tracking.action_type_classifier import extract_action_features  # noqa: E402
from rallycut.tracking.ball_tracker import BallPosition  # noqa: E402
from rallycut.tracking.contact_detector import Contact  # noqa: E402
from rallycut.tracking.dynamic_attribution_scorer import (  # noqa: E402
    DynamicAttributionScorer,
    extract_features,
    position_from_dict,
)
from rallycut.tracking.joint_viterbi import StateCandidate, joint_viterbi  # noqa: E402
from rallycut.tracking.match_tracker import build_match_team_assignments  # noqa: E402

DB = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5436/rallycut",
)

_ACTION_NAME_TO_ENUM = {
    "serve": ActionType.SERVE,
    "receive": ActionType.RECEIVE,
    "set": ActionType.SET,
    "attack": ActionType.ATTACK,
    "dig": ActionType.DIG,
    "block": ActionType.BLOCK,
}


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: probe_joint_viterbi_2026_05_17.py <rally_id>", file=sys.stderr)
        return 1
    rally_id = sys.argv[1]

    with psycopg.connect(DB) as conn:
        cur = conn.execute(
            """
            SELECT v.id, v.name, r.id, pt.positions_json, pt.ball_positions_json,
                   pt.contacts_json, pt.actions_json, pt.primary_track_ids,
                   pt.court_split_y, pt.frame_count
            FROM rallies r JOIN videos v ON r.video_id=v.id
            JOIN player_tracks pt ON pt.rally_id=r.id
            WHERE r.id=%s
            """,
            [rally_id],
        )
        row = cur.fetchone()
        if not row:
            print(f"Rally {rally_id} not found", file=sys.stderr)
            return 1
        vid, vname, rid, pj, bj, cj, aj, primary, court_split, frame_count = row
        # match_analysis for team assignments
        mcur = conn.execute(
            "SELECT match_analysis_json FROM videos WHERE id=%s", [vid],
        )
        mj_raw = mcur.fetchone()[0]
        if isinstance(mj_raw, str):
            mj_raw = json.loads(mj_raw)
        teams_by_rally = build_match_team_assignments(mj_raw, min_confidence=0.0)
        team_assignments = teams_by_rally.get(rid, {})

    positions = pj if isinstance(pj, list) else (json.loads(pj) if pj else [])
    ball_positions_raw = bj if isinstance(bj, list) else (json.loads(bj) if bj else [])
    contacts_dict = cj if isinstance(cj, dict) else (json.loads(cj) if cj else {})
    actions_dict = aj if isinstance(aj, dict) else (json.loads(aj) if aj else {})

    positions_like = [position_from_dict(p) for p in positions]
    ball_positions = [
        BallPosition(
            frame_number=int(bp["frameNumber"]),
            x=float(bp["x"]),
            y=float(bp["y"]),
            confidence=float(bp.get("confidence", 1.0)),
        )
        for bp in ball_positions_raw
        if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
    ]

    contacts_raw = contacts_dict.get("contacts", [])
    contacts = []
    for c in contacts_raw:
        contacts.append(Contact(
            frame=int(c["frame"]),
            ball_x=float(c.get("ballX", 0.5)),
            ball_y=float(c.get("ballY", 0.5)),
            velocity=float(c.get("velocity", 0)),
            court_side=c.get("courtSide", "near"),
            is_at_net=bool(c.get("isAtNet", False)),
            confidence=float(c.get("confidence", 0.5)),
            is_validated=bool(c.get("isValidated", False)),
            player_track_id=int(c.get("playerTrackId", -1)),
            arc_fit_residual=float(c.get("arcFitResidual") or 0),
            player_distance=float(c.get("playerDistance") or 0),
            player_candidates=[
                (int(pc[0]), float(pc[1])) for pc in (c.get("playerCandidates") or [])
            ],
            direction_change_deg=float(c.get("directionChangeDeg") or 0),
            candidate_bbox_motion=c.get("candidateBboxMotion"),
        ))

    print(f"Rally: {vname} {rid[:8]} ({len(contacts)} contacts)")
    print(f"team_assignments: {team_assignments}")
    print()

    # Load classifier
    classifier = _get_default_action_classifier()
    if classifier is None or not classifier.is_trained:
        print("WARNING: action classifier unavailable, using uniform action priors")
    # Load scorer
    scorer = DynamicAttributionScorer()
    if not scorer.is_available:
        print("WARNING: dynamic scorer not available", file=sys.stderr)
        return 1

    # Build features per contact for the classifier (one ActionFeatures per contact)
    classifier_feats = []
    for i, contact in enumerate(contacts):
        feat = extract_action_features(
            contact=contact, index=i, all_contacts=contacts,
            ball_positions=ball_positions,
            net_y=court_split or 0.5,
            rally_start_frame=0,
            team_assignments=team_assignments,
            player_positions=positions_like,
            calibrator=None,
            camera_height=None,
        )
        classifier_feats.append(feat)
    if classifier is not None and classifier.is_trained:
        # NaN-clean feature vectors before classifier (some action features
        # may be NaN when ball-tracking data is missing; replace with 0)
        import numpy as np
        for f in classifier_feats:
            arr = f.to_array()
            if np.isnan(arr).any():
                # Overwrite in-place via dataclass field substitution
                for fname in type(f).feature_names():
                    val = getattr(f, fname)
                    if isinstance(val, float) and np.isnan(val):
                        object.__setattr__(f, fname, 0.0)
        classifier_probs_per_contact = classifier.predict_proba(classifier_feats)
    else:
        # Uniform fallback
        classifier_probs_per_contact = [
            {at.value: 1.0 / len(_ACTION_NAME_TO_ENUM) for at in ActionType
             if at != ActionType.UNKNOWN}
            for _ in contacts
        ]

    # Build emissions per contact: (action_type, player_tid) → emission prob
    emissions_per_contact: list[list[StateCandidate]] = []
    for i, contact in enumerate(contacts):
        states: list[StateCandidate] = []
        cand_tids = [int(pc[0]) for pc in contact.player_candidates]
        if not cand_tids:
            emissions_per_contact.append(states)
            continue

        # Get prev action's tid for the same_as_prev feature (use prev contact's pick if available)
        prev_action_tid = -1
        if i > 0 and contacts[i - 1].player_track_id >= 0:
            prev_action_tid = contacts[i - 1].player_track_id

        # Post-ball for wrist_post_alignment
        post_ball_x = post_ball_y = None
        for offset in range(5, 16):
            for bp in ball_positions:
                if bp.frame_number == contact.frame + offset:
                    post_ball_x, post_ball_y = bp.x, bp.y
                    break
            if post_ball_x is not None:
                break

        # Build CandidateFeatures for each candidate (team-aware: this script
        # uses a chain-naive expected_team — Phase 1C will integrate with the
        # production chain logic)
        cand_features = []
        for tid in cand_tids:
            cf = extract_features(
                positions_like, tid, contact.frame, contact.ball_x, contact.ball_y,
                prev_action_tid=prev_action_tid,
                post_ball_x=post_ball_x, post_ball_y=post_ball_y,
                expected_team=None,  # let scorer fall back to bbox features
                team_assignments=team_assignments,
            )
            if cf is not None:
                cand_features.append(cf)

        if not cand_features:
            emissions_per_contact.append(states)
            continue

        # For each action_type, query the scorer for per-candidate probs
        for at in (ActionType.SERVE, ActionType.RECEIVE, ActionType.SET,
                   ActionType.ATTACK, ActionType.DIG, ActionType.BLOCK):
            atp = classifier_probs_per_contact[i].get(at.value, 0.0)
            if atp <= 0.01:
                continue
            scorer_probs = scorer.score(at.value, cand_features)
            if scorer_probs is None:
                continue
            # Normalise so per-action-type emission sums to 1 across candidates
            ssum = sum(scorer_probs) or 1.0
            for cf, sp in zip(cand_features, scorer_probs, strict=False):
                norm_sp = sp / ssum
                states.append(StateCandidate(
                    action_type=at,
                    player_track_id=cf.track_id,
                    emission_prob=atp * norm_sp,
                ))
        emissions_per_contact.append(states)

    # Seed serve team from match_analysis if available
    serving_team_str = actions_dict.get("servingTeam")
    seed_serve_team = None
    if serving_team_str in ("A", "B"):
        # Map A/B back to 0/1. Convention isn't always consistent;
        # infer from team_assignments + any A-labeled tid we observe.
        # Simplest: assume team 0 = "A".
        seed_serve_team = 0 if serving_team_str == "A" else 1

    path = joint_viterbi(
        emissions_per_contact,
        team_assignments=team_assignments,
        seed_serve_team=seed_serve_team,
    )

    print(f"{'#':>3s} {'frame':>5s}   {'PIPELINE':<20s}   {'JOINT VITERBI':<20s}   {'DIFF':<6s}")
    pipeline_actions = actions_dict.get("actions", [])
    pipeline_by_frame = {int(a["frame"]): a for a in pipeline_actions}
    for i, (c, vstate) in enumerate(zip(contacts, path, strict=False)):
        pa = pipeline_by_frame.get(c.frame) or {}
        pa_str = (f"{pa.get('action','?'):8s} P{pa.get('playerTrackId','?')}"
                  if pa else "—")
        vstate_str = f"{vstate.action_type.value:8s} P{vstate.player_track_id}"
        diff = "★" if pa_str.strip() != vstate_str.strip() else " "
        print(f"{i:>3d} f={c.frame:>4d}   {pa_str:<20s}   {vstate_str:<20s}   {diff}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
