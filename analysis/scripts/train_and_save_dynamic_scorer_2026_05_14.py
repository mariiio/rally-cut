#!/usr/bin/env python3
"""Train the per-action-type dynamic attribution scorer on the FULL trusted GT
corpus and save the models to disk for production use.

Usage:
    cd analysis && uv run python scripts/train_and_save_dynamic_scorer_2026_05_14.py

Outputs:
    analysis/weights/dynamic_attribution_scorer/{ACTION}_v1.joblib (one per action type)
    analysis/weights/dynamic_attribution_scorer/manifest.json (feature names, version, training corpus)

For honest measurement use train_dynamic_attribution_scorer_2026_05_14.py (LOO CV).
This script is for production: use ALL available labeled data.
"""
from __future__ import annotations

import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import psycopg
from sklearn.ensemble import GradientBoostingClassifier

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rallycut.tracking.dynamic_attribution_scorer import (  # noqa: E402
    FEATURE_NAMES as _INFERENCE_FEATURE_NAMES,
)
from rallycut.tracking.dynamic_attribution_scorer import (
    extract_features,
    position_from_dict,
)
from rallycut.tracking.match_tracker import (  # noqa: E402
    build_match_team_assignments,
)

# Net-crossing actions for team-chain forward propagation. Mirrors
# `_NET_CROSSING_ACTIONS` in rallycut/tracking/action_classifier.py.
_NET_CROSSING = {"SERVE", "ATTACK"}


def _compute_expected_teams_train(
    actions: list[dict], team_assignments: dict[int, int] | None,
) -> list[int | None]:
    """Mirror `_compute_expected_teams` from action_classifier.py for training.

    Returns parallel list of team-chain-derived expected team (0 or 1) per
    action; None where the chain can't be determined (no SERVE, or before
    the seeding SERVE, or chain broken by UNKNOWN / non-seed synthetic).
    """
    expected: list[int | None] = [None] * len(actions)
    if not team_assignments:
        return expected
    serve_team: int | None = None
    for a in actions:
        if (a.get("action") or "").upper() != "SERVE":
            continue
        tid = int(a.get("playerTrackId", -1))
        if tid < 0:
            continue
        st = team_assignments.get(tid)
        if st is not None:
            serve_team = int(st)
            break
    if serve_team is None:
        return expected
    current_team = serve_team
    for i, a in enumerate(actions):
        at = (a.get("action") or "").upper()
        if at == "UNKNOWN" or not at:
            continue
        if a.get("synthetic") or a.get("isSynthetic"):
            if at == "SERVE":
                expected[i] = serve_team
                current_team = 1 - serve_team
            continue
        expected[i] = current_team
        if at in _NET_CROSSING:
            current_team = 1 - current_team
    return expected

DB_DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5436/rallycut",
)
TRUSTED_CODENAMES = (
    # Original trusted-14 (player-attribution GT validated 2026-05-14)
    "titi", "toto", "lulu", "wawa", "caco", "cece", "cici", "cuco",
    "gaga", "kaka", "kiki", "juju", "yeye", "keke",
    # 7 added 2026-05-15 — trusted-21 corpus
    "gigi", "gugu", "mame", "meme", "mimi", "moma", "mumu",
    # 8 added 2026-05-17 — trusted-29 corpus
    "papa", "pepe", "pipi", "popo", "pupu", "veve", "vivi", "vovo",
)
FRAME_TOLERANCE = 5
# Re-export the inference feature-names list as the single source of truth.
# Any FEATURE_NAMES change MUST be made in dynamic_attribution_scorer.py;
# this training script always trains with whatever feature set inference
# defines, eliminating drift risk (refactored 2026-05-17, after the
# triple-duplication risk was flagged in the post-v3.1 audit).
FEATURE_NAMES = list(_INFERENCE_FEATURE_NAMES)
ACTION_TYPES = ["SERVE", "RECEIVE", "SET", "ATTACK", "DIG", "BLOCK"]
MODEL_VERSION = "v2"  # 2026-05-17 — retrained on v4 contact frames (post-regressor)
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "weights" / "dynamic_attribution_scorer"


def _compute_features(
    positions_like: list,  # list[PlayerPositionLike]
    tid: int, contact_frame: int,
    ball_x: float, ball_y: float,
    prev_action_tid: int = -1,
    post_ball_x: float | None = None,
    post_ball_y: float | None = None,
    expected_team: int | None = None,
    team_assignments: dict[int, int] | None = None,
) -> list[float] | None:
    """Compute feature vector for one candidate at one contact.

    Refactored 2026-05-17 (post-v3.1 audit): delegates to inference's
    `extract_features` to eliminate the triple-duplication risk. The
    previous ~200-line duplicate implementation here drifted from the
    inference module silently. Now the inference module is the single
    source of truth; this script's training distribution can never
    diverge from the inference distribution.
    """
    cf = extract_features(
        positions_like, tid, contact_frame, ball_x, ball_y,
        prev_action_tid=prev_action_tid,
        post_ball_x=post_ball_x,
        post_ball_y=post_ball_y,
        expected_team=expected_team,
        team_assignments=team_assignments,
    )
    if cf is None:
        return None
    return cf.as_vector()


@dataclass
class CandidateRow:
    action: str
    candidate_tid: int
    is_gt: bool
    features: list[float]
    # Provenance fields (added 2026-05-17 for LOO CV) — pure metadata,
    # not used in training. Allows the LOO measurement script to
    # leave-one-video-out and group by GT row identity.
    video: str = ""
    rally_id: str = ""
    gt_frame: int = -1


def build_dataset() -> list[CandidateRow]:
    """Build training dataset using PRODUCTION-MATCHED feature extraction.

    For each GT row, find the corresponding pipeline action (prefer same
    action_type within ±5 frames; else closest by frame). Use the pipeline
    action's `frame` and `ballX/ballY` as the input — NOT the GT snapshot.
    Label with GT.resolved_track_id.

    Why: at inference time the contact-detector will emit a frame + ball
    position from its own detection. Training distribution must match that
    or the model collapses on out-of-distribution serves (where the
    synth-serve placement differs significantly from the GT-labeled toss).
    """
    rows: list[CandidateRow] = []
    n_gt_seen = 0
    n_gt_matched = 0
    n_gt_skipped_no_match = 0
    with psycopg.connect(DB_DSN) as conn:
        # Build per-rally team_assignments once for the v3 team-awareness
        # feature, mirroring what redetect_all_actions does at inference time
        # (so training distribution matches inference distribution).
        match_teams_by_rally: dict[str, dict[int, int]] = {}
        vcur = conn.execute(
            "SELECT v.match_analysis_json FROM videos v "
            "WHERE v.name = ANY(%s) AND v.match_analysis_json IS NOT NULL",
            [list(TRUSTED_CODENAMES)],
        )
        for (mj_raw,) in vcur.fetchall():
            if not mj_raw:
                continue
            match_teams_by_rally.update(
                build_match_team_assignments(mj_raw, min_confidence=0.0)
            )
        cur = conn.execute(
            """
            SELECT v.name, r.id, pt.primary_track_ids, pt.positions_json,
                   pt.actions_json
            FROM videos v JOIN rallies r ON r.video_id = v.id
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE v.name = ANY(%s) AND r.status = 'CONFIRMED'
            ORDER BY v.name, r."order"
            """,
            [list(TRUSTED_CODENAMES)],
        )
        rallies = cur.fetchall()
        for video_name, rally_id, primary_raw, positions_json, actions_json in rallies:
            if positions_json is None or not isinstance(primary_raw, list) or actions_json is None:
                continue
            positions = positions_json if isinstance(positions_json, list) else []
            # Convert once per rally to PlayerPositionLike (the inference
            # module's input type), so all downstream feature computations
            # share the inference module's logic.
            positions_like = [position_from_dict(p) for p in positions]
            primary_tids = [int(t) for t in primary_raw]
            aj = json.loads(actions_json) if isinstance(actions_json, str) else actions_json
            actions = aj.get("actions") or []
            # v3: team_assignments + expected_teams chain for this rally
            team_assignments = match_teams_by_rally.get(str(rally_id))
            expected_teams = _compute_expected_teams_train(actions, team_assignments)
            # Load ball_positions_json for post-contact ball lookup (wrist_post_alignment)
            bcur = conn.execute(
                "SELECT ball_positions_json FROM player_tracks WHERE rally_id=%s",
                [rally_id],
            )
            ball_row = bcur.fetchone()
            ball_positions = (
                ball_row[0] if (ball_row and isinstance(ball_row[0], list)) else []
            )
            ball_by_frame = {
                int(b.get("frameNumber", -1)): b for b in ball_positions
            }
            gt_cur = conn.execute(
                """
                SELECT frame, action::text, resolved_track_id
                FROM rally_action_ground_truth
                WHERE rally_id = %s AND resolved_track_id IS NOT NULL
                """,
                [rally_id],
            )
            for gt_frame, gt_action, gt_tid in gt_cur.fetchall():
                n_gt_seen += 1
                # Match to pipeline action: prefer same-type within ±5, else
                # closest by frame within ±5.
                best_idx = -1
                best_delta = 6
                for i, a in enumerate(actions):
                    if a.get("action", "").upper() != gt_action.upper():
                        continue
                    delta = abs(int(a.get("frame", -10**9)) - gt_frame)
                    if delta < best_delta:
                        best_delta = delta
                        best_idx = i
                if best_idx < 0:
                    best_delta = 6
                    for i, a in enumerate(actions):
                        delta = abs(int(a.get("frame", -10**9)) - gt_frame)
                        if delta < best_delta:
                            best_delta = delta
                            best_idx = i
                if best_idx < 0:
                    # No matching pipeline action — this is a contact-FN at the
                    # training distribution. Skip it; the model wouldn't see
                    # such a case at inference (it only runs when a contact
                    # was detected).
                    n_gt_skipped_no_match += 1
                    continue
                pipe_a = actions[best_idx]
                pipe_frame = int(pipe_a.get("frame", -1))
                pipe_ball_x = pipe_a.get("ballX")
                pipe_ball_y = pipe_a.get("ballY")
                if pipe_ball_x is None or pipe_ball_y is None:
                    n_gt_skipped_no_match += 1
                    continue
                # Find the previous action's playerTrackId for the
                # same_as_prev feature. Skips UNKNOWN actions and those
                # with player_track_id < 0.
                prev_action_tid = -1
                for j in range(best_idx - 1, -1, -1):
                    pa = actions[j]
                    pa_at = (pa.get("action") or "").upper()
                    if pa_at == "UNKNOWN":
                        continue
                    pa_tid = int(pa.get("playerTrackId", -1))
                    if pa_tid >= 0:
                        prev_action_tid = pa_tid
                        break
                n_gt_matched += 1
                # Post-contact ball for wrist_post_alignment (first detection
                # in f+5..f+15 window).
                post_ball_x = post_ball_y = None
                for offset in range(5, 16):
                    b = ball_by_frame.get(pipe_frame + offset)
                    if b is not None:
                        post_ball_x = float(b.get("x") or 0)
                        post_ball_y = float(b.get("y") or 0)
                        break
                # Use PIPELINE's frame + ball position (production-matched).
                # v3.2 (2026-05-17): generic candidate-team-uniformity gate.
                # When the candidate set spans only one team, team_match
                # provides no discriminating signal — pass None so feature
                # defaults to 0.5 (uninformative) for all candidates. Must
                # stay in lockstep with action_classifier.py's
                # team_aware_is_informative check in
                # _apply_dynamic_scorer_attribution.
                candidate_teams = (
                    {team_assignments.get(t) for t in primary_tids}
                    if team_assignments else set()
                )
                candidate_teams.discard(None)
                team_aware_is_informative = len(candidate_teams) > 1
                expected_team = (
                    expected_teams[best_idx] if team_aware_is_informative
                    else None
                )
                for tid in primary_tids:
                    feats = _compute_features(
                        positions_like, tid, pipe_frame,
                        float(pipe_ball_x), float(pipe_ball_y),
                        prev_action_tid=prev_action_tid,
                        post_ball_x=post_ball_x,
                        post_ball_y=post_ball_y,
                        expected_team=expected_team,
                        team_assignments=team_assignments,
                    )
                    if feats is None:
                        continue
                    rows.append(CandidateRow(
                        action=gt_action.upper(),
                        candidate_tid=tid,
                        is_gt=(tid == gt_tid),
                        features=feats,
                        video=video_name,
                        rally_id=str(rally_id),
                        gt_frame=int(gt_frame),
                    ))
    print(f"  GT rows: {n_gt_seen} seen, {n_gt_matched} matched to pipeline action, "
          f"{n_gt_skipped_no_match} skipped (no matching pipeline action / contact-FN)",
          flush=True)
    return rows


def main() -> int:
    print(f"Output dir: {OUTPUT_DIR}", flush=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Building feature dataset from full trusted-{len(TRUSTED_CODENAMES)} corpus…", flush=True)
    rows = build_dataset()
    by_action = defaultdict(list)
    for r in rows:
        by_action[r.action].append(r)
    total_pos = sum(1 for r in rows if r.is_gt)
    print(f"  {len(rows)} candidate rows, {total_pos} positive labels", flush=True)
    for a, action_rows in sorted(by_action.items()):
        n_pos = sum(1 for r in action_rows if r.is_gt)
        print(f"    {a:10s} {len(action_rows):>5d} rows, {n_pos:>4d} positives", flush=True)

    print(flush=True)
    print("Training per-action-type GBMs…", flush=True)
    manifest: dict[str, Any] = {
        "version": MODEL_VERSION,
        "feature_names": FEATURE_NAMES,
        "training_corpus": list(TRUSTED_CODENAMES),
        "models": {},
        "trained_at": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "training_notes": (
            f"Trained on FULL trusted-{len(TRUSTED_CODENAMES)} corpus (no hold-out). "
            "For honest LOO CV measurements see "
            "scripts/train_dynamic_attribution_scorer_2026_05_14.py."
        ),
    }
    for action in ACTION_TYPES:
        action_rows = by_action.get(action, [])
        if not action_rows:
            print(f"  {action:10s} NO ROWS — skipping", flush=True)
            continue
        X = np.array([r.features for r in action_rows])
        y = np.array([1 if r.is_gt else 0 for r in action_rows])
        if y.sum() == 0 or y.sum() == len(y):
            print(f"  {action:10s} DEGENERATE LABELS — skipping", flush=True)
            continue
        clf = GradientBoostingClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.05, random_state=42,
        )
        clf.fit(X, y)
        out_path = OUTPUT_DIR / f"{action}_{MODEL_VERSION}.joblib"
        joblib.dump(clf, out_path)
        manifest["models"][action] = {
            "path": out_path.name,
            "n_rows": len(action_rows),
            "n_positives": int(y.sum()),
            "feature_importances": clf.feature_importances_.tolist(),
        }
        print(f"  {action:10s} → {out_path.name} ({len(action_rows)} rows, "
              f"{int(y.sum())} positives)", flush=True)

    manifest_path = OUTPUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest: {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
