#!/usr/bin/env python3
"""Probe: do dynamic per-candidate features discriminate GT-correct from
GT-incorrect candidates better than the current static picker?

Static features (current picker uses):
  - bbox upper-quarter distance to ball
  - depth-corrected distance

Dynamic features tested here (all computable from saved DB data):
  - candidate_velocity: |dx, dy| of bbox center across f-5..f+5
  - velocity_toward_ball: radial component of player velocity toward ball
  - bbox_top_y_at_contact: smaller = higher in image (jump height proxy)
  - bbox_top_y_change: d(top_y)/df across f-5..f+1 (rising = jumping)
  - bbox_height_change: d(height)/df across f-3..f+3 (extension/contraction)
  - bbox_aspect_ratio: w/h at contact (degenerate detection signal)
  - bbox_area: w*h at contact (size — far-court players are smaller)
  - bbox_inside_frame: 1.0 if fully inside, less if extending off

For each GT row across the trusted-14 corpus, compute these features for
each primary track candidate, then measure: how often does the GT candidate
have the extremum (min or max) on each feature? Compare with current
picker's accuracy.

This tells us:
  (a) Whether dynamic features have any independent discriminative power
  (b) Which features are most informative per action type
  (c) Whether a learned scorer is likely to beat the current picker

Read-only.
"""
from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import psycopg

DB_DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5436/rallycut",
)
TRUSTED_CODENAMES = (
    "titi", "toto", "lulu", "wawa", "caco", "cece", "cici", "cuco",
    "gaga", "kaka", "kiki", "juju", "yeye", "keke",
)
FRAME_TOLERANCE = 5


@dataclass
class CandidateFeatures:
    tid: int
    # static (current picker uses some of these)
    bbox_dist: float       # bbox upper-quarter distance to ball
    bbox_area: float       # w * h at contact
    bbox_aspect_ratio: float  # w / h
    bbox_inside_frame: float  # 1.0 if fully inside
    # dynamic (untested)
    velocity_mag: float    # |dx, dy| across f-5..f+5
    velocity_toward_ball: float  # radial component
    top_y_at_contact: float    # bbox y (lower = higher in image)
    top_y_change: float        # delta y across f-5..f+1
    height_change: float       # delta height across f-3..f+3


def _ball_dist_upper_quarter(p: dict, ball_x: float, ball_y: float) -> float:
    px = float(p.get("x", 0))
    py = float(p.get("y", 0)) - float(p.get("height", 0)) * 0.25
    return math.hypot(px - ball_x, py - ball_y)


def _find_pos(positions: list[dict], tid: int, frame: int, tolerance: int = 5) -> dict | None:
    best = None
    best_delta = tolerance + 1
    for p in positions:
        if int(p.get("trackId", -1)) != tid:
            continue
        f = int(p.get("frameNumber", -1))
        delta = abs(f - frame)
        if delta < best_delta:
            best_delta = delta
            best = p
    return best


def _compute_features(
    positions: list[dict],
    tid: int,
    contact_frame: int,
    ball_x: float,
    ball_y: float,
) -> CandidateFeatures | None:
    p_at = _find_pos(positions, tid, contact_frame, tolerance=2)
    if p_at is None:
        return None
    p_prev = _find_pos(positions, tid, contact_frame - 5, tolerance=2)
    p_next = _find_pos(positions, tid, contact_frame + 5, tolerance=2)
    p_pre_extend = _find_pos(positions, tid, contact_frame - 3, tolerance=2)
    p_post_extend = _find_pos(positions, tid, contact_frame + 3, tolerance=2)

    # Static
    x = float(p_at.get("x", 0))
    y = float(p_at.get("y", 0))
    w = float(p_at.get("width", 0))
    h = float(p_at.get("height", 0))
    bbox_dist = _ball_dist_upper_quarter(p_at, ball_x, ball_y)
    bbox_area = w * h
    bbox_aspect_ratio = w / max(h, 1e-6)
    inside = 1.0
    if x < 0 or y < 0 or x + w > 1.0 or y + h > 1.0:
        inside = 0.0

    # Dynamic
    if p_prev and p_next:
        cx_prev = float(p_prev.get("x", 0)) + float(p_prev.get("width", 0)) / 2
        cy_prev = float(p_prev.get("y", 0)) + float(p_prev.get("height", 0)) / 2
        cx_next = float(p_next.get("x", 0)) + float(p_next.get("width", 0)) / 2
        cy_next = float(p_next.get("y", 0)) + float(p_next.get("height", 0)) / 2
        dx = cx_next - cx_prev
        dy = cy_next - cy_prev
        velocity_mag = math.hypot(dx, dy)
        # Velocity-toward-ball: dot product of (dx, dy) with (ball - player_center)
        cx_at = x + w / 2
        cy_at = y + h / 2
        to_ball_x = ball_x - cx_at
        to_ball_y = ball_y - cy_at
        to_ball_mag = math.hypot(to_ball_x, to_ball_y) + 1e-6
        velocity_toward_ball = (dx * to_ball_x + dy * to_ball_y) / to_ball_mag
    else:
        velocity_mag = 0.0
        velocity_toward_ball = 0.0

    top_y_at_contact = y
    if p_prev:
        top_y_change = y - float(p_prev.get("y", y))
    else:
        top_y_change = 0.0
    if p_pre_extend and p_post_extend:
        height_change = float(p_post_extend.get("height", h)) - float(p_pre_extend.get("height", h))
    else:
        height_change = 0.0

    return CandidateFeatures(
        tid=tid,
        bbox_dist=bbox_dist,
        bbox_area=bbox_area,
        bbox_aspect_ratio=bbox_aspect_ratio,
        bbox_inside_frame=inside,
        velocity_mag=velocity_mag,
        velocity_toward_ball=velocity_toward_ball,
        top_y_at_contact=top_y_at_contact,
        top_y_change=top_y_change,
        height_change=height_change,
    )


def _rank_candidates_by(
    candidates: list[CandidateFeatures], key: str, ascending: bool,
) -> list[int]:
    sorted_cands = sorted(candidates, key=lambda c: getattr(c, key), reverse=not ascending)
    return [c.tid for c in sorted_cands]


def main() -> int:
    print("Dynamic-feature attribution probe", flush=True)
    print(f"  Corpus: trusted-14", flush=True)
    print(flush=True)

    # Per-feature, per-action: count of contacts where GT track is rank-1
    feature_specs = [
        # (name, ascending = "smaller is better"?, intuition)
        ("bbox_dist", True, "smaller = closer to ball"),
        ("velocity_mag", False, "larger = more motion (attacker > blocker?)"),
        ("velocity_toward_ball", False, "larger = moving toward ball (attacker)"),
        ("top_y_at_contact", True, "smaller y = higher in image (jumping)"),
        ("top_y_change", True, "smaller = rising (jumping up)"),
        ("height_change", False, "larger = extending (reaching)"),
        ("bbox_aspect_ratio", False, "larger = more normal-shaped (non-degenerate)"),
        ("bbox_area", False, "larger = closer player"),
    ]

    stats: dict[str, dict[str, dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"correct_rank1": 0, "total": 0})
    )

    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            """
            SELECT v.name, r.id, pt.primary_track_ids, pt.positions_json
            FROM videos v
            JOIN rallies r ON r.video_id = v.id
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE v.name = ANY(%s) AND r.status = 'CONFIRMED'
            ORDER BY v.name, r."order"
            """,
            [list(TRUSTED_CODENAMES)],
        )
        rallies = cur.fetchall()
        for video_name, rally_id, primary_raw, positions_json in rallies:
            if positions_json is None or not isinstance(primary_raw, list):
                continue
            positions = positions_json if isinstance(positions_json, list) else []
            primary_tids = [int(t) for t in primary_raw]
            gt_cur = conn.execute(
                """
                SELECT frame, action::text, resolved_track_id,
                       snapshot_ball_x, snapshot_ball_y
                FROM rally_action_ground_truth
                WHERE rally_id = %s AND resolved_track_id IS NOT NULL
                  AND snapshot_ball_x IS NOT NULL AND snapshot_ball_y IS NOT NULL
                ORDER BY frame
                """,
                [rally_id],
            )
            for gt_frame, gt_action, gt_tid, ball_x, ball_y in gt_cur.fetchall():
                # Compute features for each primary candidate
                cands: list[CandidateFeatures] = []
                for tid in primary_tids:
                    feat = _compute_features(
                        positions, tid, gt_frame, ball_x, ball_y,
                    )
                    if feat is not None:
                        cands.append(feat)
                if len(cands) < 2:
                    continue
                # Only count if GT track is among the candidates
                if gt_tid not in [c.tid for c in cands]:
                    continue
                for fname, ascending, _ in feature_specs:
                    ranking = _rank_candidates_by(cands, fname, ascending)
                    correct = ranking[0] == gt_tid
                    stats[fname][gt_action]["total"] += 1
                    if correct:
                        stats[fname][gt_action]["correct_rank1"] += 1

    print("=" * 90, flush=True)
    print("Per-feature rank-1 accuracy (does GT candidate top the ranking?)", flush=True)
    print("=" * 90, flush=True)
    actions_order = ["SERVE", "RECEIVE", "SET", "ATTACK", "DIG", "BLOCK"]
    # Header
    hdr = f"{'feature':25s} " + " ".join(f"{a:>10s}" for a in actions_order) + f" {'TOTAL':>10s}"
    print(hdr, flush=True)
    for fname, ascending, intuition in feature_specs:
        row = f"{fname:25s} "
        total_correct = 0
        total = 0
        for a in actions_order:
            stat = stats[fname][a]
            n = stat["total"]
            c = stat["correct_rank1"]
            if n == 0:
                row += f"{'-':>10s} "
            else:
                row += f"{100*c/n:>9.1f}% "
                total_correct += c
                total += n
        if total > 0:
            row += f"{100*total_correct/total:>9.1f}%"
        print(row, flush=True)
        print(f"  ({intuition})", flush=True)
    print(flush=True)

    # Random baseline: 1/4 = 25%
    print(f"Random-baseline rank-1 accuracy (1/4 candidates): 25.0%", flush=True)
    print(f"Current picker accuracy (action-level, trusted-14): ~84.4%", flush=True)
    print(flush=True)
    print("A feature beats noise if rank-1 accuracy is meaningfully > 25%.", flush=True)
    print("A feature is INFORMATIVE if it beats noise AND is independent of bbox_dist.", flush=True)

    # ==========================================================================
    # Joint analysis: on contacts where bbox_dist is WRONG, do other features
    # identify the GT correctly? This measures INDEPENDENCE / complementarity.
    # ==========================================================================
    print(flush=True)
    print("=" * 90, flush=True)
    print("Joint analysis: on bbox_dist-WRONG contacts, do dynamic features rescue?", flush=True)
    print("=" * 90, flush=True)

    # Re-walk corpus to compute the per-contact ranking for each feature
    # simultaneously, then condition on bbox_dist being wrong.
    bbox_wrong: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "rescued_top_y_change": 0, "rescued_top_y_at_contact": 0,
                 "rescued_height_change": 0, "rescued_any_dynamic": 0}
    )
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            """
            SELECT v.name, r.id, pt.primary_track_ids, pt.positions_json
            FROM videos v
            JOIN rallies r ON r.video_id = v.id
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE v.name = ANY(%s) AND r.status = 'CONFIRMED'
            ORDER BY v.name, r."order"
            """,
            [list(TRUSTED_CODENAMES)],
        )
        rallies = cur.fetchall()
        for video_name, rally_id, primary_raw, positions_json in rallies:
            if positions_json is None or not isinstance(primary_raw, list):
                continue
            positions = positions_json if isinstance(positions_json, list) else []
            primary_tids = [int(t) for t in primary_raw]
            gt_cur = conn.execute(
                """
                SELECT frame, action::text, resolved_track_id,
                       snapshot_ball_x, snapshot_ball_y
                FROM rally_action_ground_truth
                WHERE rally_id = %s AND resolved_track_id IS NOT NULL
                  AND snapshot_ball_x IS NOT NULL AND snapshot_ball_y IS NOT NULL
                ORDER BY frame
                """,
                [rally_id],
            )
            for gt_frame, gt_action, gt_tid, ball_x, ball_y in gt_cur.fetchall():
                cands: list[CandidateFeatures] = []
                for tid in primary_tids:
                    feat = _compute_features(positions, tid, gt_frame, ball_x, ball_y)
                    if feat is not None:
                        cands.append(feat)
                if len(cands) < 2 or gt_tid not in [c.tid for c in cands]:
                    continue
                bbox_rank = _rank_candidates_by(cands, "bbox_dist", True)
                if bbox_rank[0] == gt_tid:
                    continue  # bbox_dist got it right; not in this analysis
                bbox_wrong[gt_action]["total"] += 1
                # Check each dynamic feature
                top_y_change_rank = _rank_candidates_by(cands, "top_y_change", True)
                top_y_at_contact_rank = _rank_candidates_by(cands, "top_y_at_contact", True)
                height_change_rank = _rank_candidates_by(cands, "height_change", False)
                rescued_any = False
                if top_y_change_rank[0] == gt_tid:
                    bbox_wrong[gt_action]["rescued_top_y_change"] += 1
                    rescued_any = True
                if top_y_at_contact_rank[0] == gt_tid:
                    bbox_wrong[gt_action]["rescued_top_y_at_contact"] += 1
                    rescued_any = True
                if height_change_rank[0] == gt_tid:
                    bbox_wrong[gt_action]["rescued_height_change"] += 1
                    rescued_any = True
                if rescued_any:
                    bbox_wrong[gt_action]["rescued_any_dynamic"] += 1

    print(f"{'action':10s} {'bbox_wrong':>12s} {'rescued by top_y_change':>26s}"
          f" {'rescued by top_y_at_contact':>30s} {'rescued by height_change':>26s}"
          f" {'rescued by ANY':>16s}", flush=True)
    total_bw = 0
    total_any = 0
    for action in actions_order:
        s = bbox_wrong[action]
        bw = s["total"]
        if bw == 0:
            continue
        total_bw += bw
        total_any += s["rescued_any_dynamic"]
        print(f"{action:10s} {bw:>12d}"
              f" {s['rescued_top_y_change']:>3d} ({100*s['rescued_top_y_change']/bw:>5.1f}%)         "
              f" {s['rescued_top_y_at_contact']:>3d} ({100*s['rescued_top_y_at_contact']/bw:>5.1f}%)             "
              f" {s['rescued_height_change']:>3d} ({100*s['rescued_height_change']/bw:>5.1f}%)        "
              f" {s['rescued_any_dynamic']:>3d} ({100*s['rescued_any_dynamic']/bw:>5.1f}%)",
              flush=True)
    if total_bw > 0:
        print(f"{'TOTAL':10s} {total_bw:>12d}"
              f"                                                                  "
              f"  {total_any:>3d} ({100*total_any/total_bw:>5.1f}%)",
              flush=True)
    print(flush=True)
    print(f"Interpretation: on the bbox_dist-wrong cases, what fraction does a", flush=True)
    print(f"dynamic feature correctly identify the GT? This is the upside ceiling", flush=True)
    print(f"for a learned model combining bbox_dist + dynamic features.", flush=True)

    # ==========================================================================
    # Full-pipeline rescue: on contacts where the actual pipeline (with pose +
    # depth correction) is wrong, do dynamic features identify the GT?
    # ==========================================================================
    print(flush=True)
    print("=" * 90, flush=True)
    print("Full-pipeline rescue: on actions.playerTrackId-WRONG contacts, do", flush=True)
    print("dynamic features rescue?", flush=True)
    print("=" * 90, flush=True)

    pipe_wrong: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "rescued_top_y_change": 0, "rescued_top_y_at_contact": 0,
                 "rescued_height_change": 0, "rescued_any_dynamic": 0}
    )
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            """
            SELECT v.name, r.id, pt.primary_track_ids, pt.positions_json, pt.actions_json
            FROM videos v
            JOIN rallies r ON r.video_id = v.id
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
            primary_tids = [int(t) for t in primary_raw]
            aj = json.loads(actions_json) if isinstance(actions_json, str) else actions_json
            actions = aj.get("actions") or []
            gt_cur = conn.execute(
                """
                SELECT frame, action::text, resolved_track_id,
                       snapshot_ball_x, snapshot_ball_y
                FROM rally_action_ground_truth
                WHERE rally_id = %s AND resolved_track_id IS NOT NULL
                  AND snapshot_ball_x IS NOT NULL AND snapshot_ball_y IS NOT NULL
                ORDER BY frame
                """,
                [rally_id],
            )
            for gt_frame, gt_action, gt_tid, ball_x, ball_y in gt_cur.fetchall():
                # Match GT to an action by frame ±5 and action type
                best_idx = -1
                best_delta = FRAME_TOLERANCE + 1
                for i, a in enumerate(actions):
                    if a.get("action", "").upper() != gt_action.upper():
                        continue
                    delta = abs(int(a.get("frame", -10**9)) - gt_frame)
                    if delta < best_delta:
                        best_delta = delta
                        best_idx = i
                if best_idx < 0:
                    for i, a in enumerate(actions):
                        delta = abs(int(a.get("frame", -10**9)) - gt_frame)
                        if delta < best_delta:
                            best_delta = delta
                            best_idx = i
                if best_idx < 0:
                    continue
                pipe_tid = int(actions[best_idx].get("playerTrackId", -1))
                if pipe_tid == gt_tid:
                    continue  # pipeline got it right
                cands: list[CandidateFeatures] = []
                for tid in primary_tids:
                    feat = _compute_features(positions, tid, gt_frame, ball_x, ball_y)
                    if feat is not None:
                        cands.append(feat)
                if len(cands) < 2 or gt_tid not in [c.tid for c in cands]:
                    continue
                pipe_wrong[gt_action]["total"] += 1
                top_y_change_rank = _rank_candidates_by(cands, "top_y_change", True)
                top_y_at_contact_rank = _rank_candidates_by(cands, "top_y_at_contact", True)
                height_change_rank = _rank_candidates_by(cands, "height_change", False)
                rescued_any = False
                if top_y_change_rank[0] == gt_tid:
                    pipe_wrong[gt_action]["rescued_top_y_change"] += 1
                    rescued_any = True
                if top_y_at_contact_rank[0] == gt_tid:
                    pipe_wrong[gt_action]["rescued_top_y_at_contact"] += 1
                    rescued_any = True
                if height_change_rank[0] == gt_tid:
                    pipe_wrong[gt_action]["rescued_height_change"] += 1
                    rescued_any = True
                if rescued_any:
                    pipe_wrong[gt_action]["rescued_any_dynamic"] += 1

    print(f"{'action':10s} {'pipe_wrong':>12s} {'rescued_top_y_change':>22s}"
          f" {'rescued_top_y_at_contact':>26s} {'rescued_height_change':>24s}"
          f" {'rescued_ANY':>14s}", flush=True)
    total_pw = 0
    total_any = 0
    for action in actions_order:
        s = pipe_wrong[action]
        pw = s["total"]
        if pw == 0:
            continue
        total_pw += pw
        total_any += s["rescued_any_dynamic"]
        print(f"{action:10s} {pw:>12d}"
              f"  {s['rescued_top_y_change']:>3d} ({100*s['rescued_top_y_change']/pw:>5.1f}%)        "
              f"  {s['rescued_top_y_at_contact']:>3d} ({100*s['rescued_top_y_at_contact']/pw:>5.1f}%)         "
              f"  {s['rescued_height_change']:>3d} ({100*s['rescued_height_change']/pw:>5.1f}%)       "
              f"  {s['rescued_any_dynamic']:>3d} ({100*s['rescued_any_dynamic']/pw:>5.1f}%)",
              flush=True)
    if total_pw > 0:
        print(f"{'TOTAL':10s} {total_pw:>12d}"
              f"                                                                              "
              f"  {total_any:>3d} ({100*total_any/total_pw:>5.1f}%)",
              flush=True)
    print(flush=True)
    if total_pw > 0:
        max_lift_pp = 100 * total_any / 610  # 610 total GT rows
        print(f"Upside ceiling if a learned model perfectly leverages dynamic features:", flush=True)
        print(f"  rescue {total_any}/{total_pw} pipeline-wrong cases = +{max_lift_pp:.1f}pp on trusted-14", flush=True)
        print(f"  (going from current ~83.4% → {83.4 + max_lift_pp:.1f}% upper bound)", flush=True)
    print(f"\nReal-world lift will be lower (learners imperfectly combine signals)", flush=True)
    print(f"but +2-4pp is plausible based on rescue rate.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
