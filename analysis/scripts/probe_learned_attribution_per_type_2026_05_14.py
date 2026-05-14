"""Per-action-type learned attribution — validation experiment (2026-05-14).

Goal: validate whether learning can lift per-action-type attribution
precision above the 84.1% baseline on the existing 641 trusted-GT rows.
Decides whether to invest in 2-3K more GT for PGM Phase B.

Approach:
  - For each GT row that has a pipeline contact within ±5 frames, enumerate
    the 4 same-team-pair candidate players (canonical pids 1..4).
  - Compute per-candidate features (proximity, motion, team match, ...).
  - Per-action-type: train gradient-boosting (HistGradientBoostingClassifier)
    in leave-one-video-out CV on per-candidate binary "is GT" target.
  - At inference: score the 4 candidates, pick argmax. Compare to GT.
  - Compare aggregate + per-type precision to baseline (pipeline pick).

Critical anti-leakage:
  - DO NOT include `is_pipeline_pick` as a feature (leaks baseline).
  - DO include `is_prev_toucher`, `team_match_expected` (legit signals).

Usage:
    cd analysis
    uv run python scripts/probe_learned_attribution_per_type_2026_05_14.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from rallycut.evaluation.tracking.db import get_connection  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402

VIDEO_NAMES = [
    "titi", "toto", "lulu", "wawa", "caco", "cece",
    "cici", "cuco", "gaga", "kaka", "juju", "yeye",
]
GT_FRAME_TOL = 5
PRE_WINDOW = 10                 # frames before contact
POST_WINDOW = 3                 # frames after contact for body velocity
ACTION_TYPES_TO_TRAIN = ["SERVE", "RECEIVE", "SET", "ATTACK", "DIG"]
ACTION_TYPES_INSUFFICIENT = ["BLOCK"]

NET_CROSSING = {"serve", "attack"}

REPORT_DIR = HERE.parent / "reports" / "learned_attribution_experiment_2026_05_14"
JSON_PATH = REPORT_DIR / "results.json"
MD_PATH = REPORT_DIR / "results.md"


# ---------------------------------------------------------------------------
# Data hydration
# ---------------------------------------------------------------------------

@dataclass
class GTRow:
    rally_id: str
    video: str
    frame: int
    action: str
    resolved_tid: int | None


@dataclass
class ContactCtx:
    """Per-contact context joined to pipeline action."""
    rally_id: str
    video: str
    frame: int
    action: str                                          # lowercase
    pl_tid: int | None
    confidence: float
    team_assignments: dict[int, int]                      # tid -> 0 (A) or 1 (B)
    ball_at_contact: tuple[float, float] | None
    direction_change_deg: float | None
    arc_fit_residual: float | None
    velocity: float | None
    is_at_net: bool | None
    candidates: list[tuple[int, float]]                   # (tid, dist) ascending
    cand_bbox_motion: dict[int, tuple[float, float]]      # tid -> (dx, dy) over contact
    cand_bbox_pre: dict[int, dict[int, tuple[float, float]]]  # tid -> {f -> (cx, cy)}
    cand_bbox_at: dict[int, tuple[float, float]]          # tid -> (cx, cy) at frame f
    cand_bbox_post: dict[int, dict[int, tuple[float, float]]]
    pre_ball: list[tuple[int, float, float]]              # (f, bx, by)
    post_ball: list[tuple[int, float, float]]
    # Adjacent action
    prev_action: str | None
    prev_tid: int | None
    prev_team: int | None       # via team_assignments
    next_action: str | None
    next_tid: int | None
    # Expected team (volleyball-rule chain)
    expected_team: int | None
    # Court side at contact (from contacts JSON)
    court_side: str | None


def _to_int_team(value: Any) -> int | None:
    """Convert team_assignments value into 0 (A) or 1 (B)."""
    if value is None:
        return None
    if isinstance(value, int):
        if value in (0, 1):
            return value
        return None
    s = str(value).strip().upper()
    if s in ("A", "0", "TEAM_A"):
        return 0
    if s in ("B", "1", "TEAM_B"):
        return 1
    return None


def _compute_expected_teams(actions_sorted: list[dict[str, Any]], team_assignments: dict[int, int]) -> list[int | None]:
    """Mirror of action_classifier._compute_expected_teams using JSON dicts.

    Net-crossing: serve & attack flip team. Synthetic actions are skipped.
    Block does not cross (defender reacts on same side); SET/RECEIVE/DIG don't either.
    """
    out: list[int | None] = [None] * len(actions_sorted)
    serve_team: int | None = None
    for a in actions_sorted:
        if str(a.get("action", "")).lower() == "serve":
            tid = a.get("playerTrackId")
            if tid is not None:
                tt = team_assignments.get(int(tid))
                if tt is not None:
                    serve_team = tt
                    break
    if serve_team is None:
        return out
    current_team = serve_team
    for i, a in enumerate(actions_sorted):
        atype = str(a.get("action", "")).lower()
        if atype in ("", "unknown"):
            continue
        if a.get("isSynthetic"):
            if atype == "serve":
                out[i] = serve_team
                current_team = 1 - serve_team
            continue
        out[i] = current_team
        if atype in NET_CROSSING:
            current_team = 1 - current_team
    return out


def _build_contacts_for_rally(
    actions: list[dict[str, Any]],
    contacts: list[dict[str, Any]],
    positions: list[dict[str, Any]],
    ball_positions: list[dict[str, Any]],
    team_assignments_raw: dict[str, Any],
    rally_id: str,
    video: str,
) -> tuple[list[ContactCtx], dict[int, int]]:
    """Build ContactCtx per pipeline action joined to contact + positions."""
    if not actions:
        return [], {}

    actions_sorted = sorted(actions, key=lambda a: int(a.get("frame", 0)))
    # team_assignments: keys may be str
    team_assignments: dict[int, int] = {}
    for k, v in (team_assignments_raw or {}).items():
        try:
            tid = int(k)
        except (TypeError, ValueError):
            continue
        tnorm = _to_int_team(v)
        if tnorm is not None:
            team_assignments[tid] = tnorm

    # Indexes
    contacts_by_frame: dict[int, dict[str, Any]] = {}
    for c in contacts:
        try:
            f = int(c.get("frame", -1))
        except (TypeError, ValueError):
            continue
        contacts_by_frame[f] = c

    ball_by_frame: dict[int, tuple[float, float]] = {}
    for bp in ball_positions:
        try:
            f = int(bp.get("frameNumber", -1))
            bx = float(bp.get("x", 0.0))
            by = float(bp.get("y", 0.0))
        except (KeyError, TypeError, ValueError):
            continue
        if bx <= 0 and by <= 0:
            continue
        ball_by_frame[f] = (bx, by)

    pos_by_tid_frame: dict[int, dict[int, tuple[float, float]]] = defaultdict(dict)
    for p in positions:
        try:
            tid = int(p.get("trackId", -1))
            fnum = int(p.get("frameNumber", -1))
            cx = float(p.get("x", 0.0))
            cy = float(p.get("y", 0.0))
        except (TypeError, ValueError):
            continue
        if tid < 0 or fnum < 0:
            continue
        pos_by_tid_frame[tid][fnum] = (cx, cy)

    expected_teams_list = _compute_expected_teams(actions_sorted, team_assignments)

    out: list[ContactCtx] = []
    for idx, a in enumerate(actions_sorted):
        try:
            f = int(a.get("frame", 0))
        except (TypeError, ValueError):
            continue
        action_lc = str(a.get("action", "")).lower()
        pl_tid_raw = a.get("playerTrackId")
        pl_tid: int | None = None
        if pl_tid_raw is not None:
            try:
                pl_tid = int(pl_tid_raw)
            except (TypeError, ValueError):
                pl_tid = None
        confidence = float(a.get("confidence", 0.0) or 0.0)

        contact = contacts_by_frame.get(f)
        candidates: list[tuple[int, float]] = []
        cand_bbox_motion: dict[int, tuple[float, float]] = {}
        is_at_net: bool | None = None
        ball_at: tuple[float, float] | None = None
        direction_change_deg: float | None = None
        arc_fit_residual: float | None = None
        velocity: float | None = None
        court_side: str | None = None

        if contact:
            for cd in contact.get("playerCandidates", []) or []:
                if not cd or len(cd) < 2:
                    continue
                try:
                    candidates.append((int(cd[0]), float(cd[1])))
                except (TypeError, ValueError):
                    continue
            candidates.sort(key=lambda x: x[1])
            motion_raw = contact.get("candidateBboxMotion") or {}
            for k, v in motion_raw.items():
                try:
                    cand_bbox_motion[int(k)] = (float(v[0]), float(v[1]))
                except (TypeError, ValueError, IndexError):
                    continue
            iatn = contact.get("isAtNet")
            if isinstance(iatn, bool):
                is_at_net = iatn
            bx_raw = contact.get("ballX")
            by_raw = contact.get("ballY")
            if bx_raw is not None and by_raw is not None:
                try:
                    ball_at = (float(bx_raw), float(by_raw))
                except (TypeError, ValueError):
                    pass
            try:
                direction_change_deg = float(contact.get("directionChangeDeg")) if contact.get("directionChangeDeg") is not None else None
            except (TypeError, ValueError):
                pass
            try:
                arc_fit_residual = float(contact.get("arcFitResidual")) if contact.get("arcFitResidual") is not None else None
            except (TypeError, ValueError):
                pass
            try:
                velocity = float(contact.get("velocity")) if contact.get("velocity") is not None else None
            except (TypeError, ValueError):
                pass
            cs = contact.get("courtSide")
            if isinstance(cs, str):
                court_side = cs

        if ball_at is None:
            ball_at = ball_by_frame.get(f)

        # Pre/post ball windows
        pre_ball: list[tuple[int, float, float]] = []
        for fp in range(f - PRE_WINDOW, f):
            if fp in ball_by_frame:
                bx, by = ball_by_frame[fp]
                pre_ball.append((fp, bx, by))
        post_ball: list[tuple[int, float, float]] = []
        for fp in range(f + 1, f + 1 + POST_WINDOW):
            if fp in ball_by_frame:
                bx, by = ball_by_frame[fp]
                post_ball.append((fp, bx, by))

        # Per-candidate bbox snapshots
        cand_tids = {tid for tid, _ in candidates}
        if pl_tid is not None:
            cand_tids.add(pl_tid)
        # Also union with team_assignments (canonical 1..4)
        for tid in team_assignments.keys():
            cand_tids.add(tid)

        cand_bbox_pre: dict[int, dict[int, tuple[float, float]]] = {}
        cand_bbox_post: dict[int, dict[int, tuple[float, float]]] = {}
        cand_bbox_at: dict[int, tuple[float, float]] = {}
        for tid in cand_tids:
            tmap = pos_by_tid_frame.get(tid, {})
            cand_bbox_pre[tid] = {fp: tmap[fp] for fp in range(f - PRE_WINDOW, f) if fp in tmap}
            cand_bbox_post[tid] = {fp: tmap[fp] for fp in range(f + 1, f + 1 + POST_WINDOW) if fp in tmap}
            if f in tmap:
                cand_bbox_at[tid] = tmap[f]

        # Adjacent action
        prev_action: str | None = None
        prev_tid: int | None = None
        prev_team: int | None = None
        next_action: str | None = None
        next_tid: int | None = None
        if idx > 0:
            pa = actions_sorted[idx - 1]
            prev_action = str(pa.get("action", "")).lower() or None
            pt = pa.get("playerTrackId")
            if pt is not None:
                try:
                    prev_tid = int(pt)
                except (TypeError, ValueError):
                    prev_tid = None
            if prev_tid is not None:
                prev_team = team_assignments.get(prev_tid)
        if idx + 1 < len(actions_sorted):
            na = actions_sorted[idx + 1]
            next_action = str(na.get("action", "")).lower() or None
            nt = na.get("playerTrackId")
            if nt is not None:
                try:
                    next_tid = int(nt)
                except (TypeError, ValueError):
                    next_tid = None

        out.append(ContactCtx(
            rally_id=rally_id, video=video, frame=f, action=action_lc,
            pl_tid=pl_tid, confidence=confidence, team_assignments=team_assignments,
            ball_at_contact=ball_at, direction_change_deg=direction_change_deg,
            arc_fit_residual=arc_fit_residual, velocity=velocity,
            is_at_net=is_at_net, candidates=candidates,
            cand_bbox_motion=cand_bbox_motion,
            cand_bbox_pre=cand_bbox_pre, cand_bbox_at=cand_bbox_at,
            cand_bbox_post=cand_bbox_post,
            pre_ball=pre_ball, post_ball=post_ball,
            prev_action=prev_action, prev_tid=prev_tid, prev_team=prev_team,
            next_action=next_action, next_tid=next_tid,
            expected_team=expected_teams_list[idx], court_side=court_side,
        ))
    return out, team_assignments


def fetch_data(
    video_names: list[str],
) -> tuple[dict[str, list[ContactCtx]], list[GTRow]]:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT v.id, v.name
            FROM videos v
            WHERE v.name = ANY(%s)
            """,
            [video_names],
        )
        vid_rows = cur.fetchall()
        video_id_to_name = {str(r[0]): str(r[1]) for r in vid_rows}
        video_ids = list(video_id_to_name.keys())

        cur.execute(
            """
            SELECT r.id, r.video_id, pt.actions_json, pt.contacts_json,
                   pt.positions_json, pt.ball_positions_json
            FROM rallies r
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE r.video_id = ANY(%s)
              AND (r.status = 'CONFIRMED' OR r.status IS NULL)
            ORDER BY r.start_ms
            """,
            [video_ids],
        )
        prows = cur.fetchall()

        cur.execute(
            """
            SELECT g.rally_id, r.video_id, g.frame, g.action, g.resolved_track_id
            FROM rally_action_ground_truth g
            JOIN rallies r ON r.id = g.rally_id
            WHERE r.video_id = ANY(%s)
            """,
            [video_ids],
        )
        gt_db_rows = cur.fetchall()

    rally_contacts: dict[str, list[ContactCtx]] = {}
    for rrow in prows:
        rally_id = str(rrow[0])
        video_id = str(rrow[1])
        aj = rrow[2] or {}
        cj = rrow[3] or {}
        positions = rrow[4] or []
        ball_positions = rrow[5] or []

        actions = aj.get("actions", []) if isinstance(aj, dict) else []
        team_ass = aj.get("teamAssignments", {}) if isinstance(aj, dict) else {}
        contacts = cj.get("contacts", []) if isinstance(cj, dict) else []
        video_name = video_id_to_name.get(video_id, video_id)

        contact_ctx, _team_int = _build_contacts_for_rally(
            actions=actions, contacts=contacts, positions=positions,
            ball_positions=ball_positions, team_assignments_raw=team_ass,
            rally_id=rally_id, video=video_name,
        )
        rally_contacts[rally_id] = contact_ctx

    gt_rows: list[GTRow] = []
    for g in gt_db_rows:
        gt_rows.append(GTRow(
            rally_id=str(g[0]),
            video=video_id_to_name.get(str(g[1]), str(g[1])),
            frame=int(g[2]),
            action=str(g[3]),
            resolved_tid=int(g[4]) if g[4] is not None else None,
        ))
    return rally_contacts, gt_rows


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    # Per-candidate features
    "proximity_to_ball",                    # dist(bbox_at, ball_at) — same as candidate distance
    "proximity_pre_contact_mean",           # mean dist over [f-10, f-1]
    "proximity_pre_contact_min",            # min dist over [f-10, f-1]
    "proximity_trajectory_endpoint",        # dist(bbox_at, ball_traj_endpoint)
    "is_prev_toucher",                      # 1 if this candidate == prev action's tid
    "is_next_toucher",                      # 1 if this candidate == next action's tid
    "team_match_expected",                  # 1 if candidate.team == expected_team
    "candidate_team",                       # 0/1/-1 if unknown
    "candidate_dist_rank",                  # 0..3 rank in playerCandidates order
    "bbox_motion_dx",                       # dx over contact window
    "bbox_motion_dy",                       # dy over contact window
    "bbox_motion_mag",                      # hypot(dx, dy)
    "body_velocity_pre",                    # bbox-center motion over [f-3, f]
    "body_velocity_post",                   # bbox-center motion over [f, f+3]
    "body_velocity_around",                 # bbox-center motion across [f-3, f+3]
    "approach_speed_to_ball",               # change in dist to ball: pre_mean - at
    # Shared (per-contact) features — duplicated across all 4 candidate rows
    "direction_change_deg",                 # in contact
    "arc_fit_residual",
    "velocity",
    "is_at_net",                            # 0/1
    "court_side_near",                      # 1 if "near"
    "action_confidence",
    "prev_action_serve", "prev_action_receive", "prev_action_set",
    "prev_action_attack", "prev_action_block", "prev_action_dig",
    "prev_is_cross_team",                   # 1 if prev_team != expected_team (cross-team rally flow)
    "n_candidates",
]


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _extrapolate_ball_endpoint(pre_ball: list[tuple[int, float, float]], target_frame: int) -> tuple[float, float] | None:
    """Linear extrapolation of ball trajectory from last 5 pre-window positions to target_frame."""
    if len(pre_ball) < 3:
        return None
    # Use last 5 frames of pre_ball
    pts = pre_ball[-5:]
    xs = np.array([p[0] for p in pts], dtype=np.float32)
    bxs = np.array([p[1] for p in pts], dtype=np.float32)
    bys = np.array([p[2] for p in pts], dtype=np.float32)
    # Linear fit
    if len(np.unique(xs)) < 2:
        return None
    coef_x = np.polyfit(xs, bxs, 1)
    coef_y = np.polyfit(xs, bys, 1)
    px = coef_x[0] * target_frame + coef_x[1]
    py = coef_y[0] * target_frame + coef_y[1]
    return (float(px), float(py))


def compute_features_for_candidate(
    ctx: ContactCtx, cand_tid: int,
) -> list[float]:
    """Compute the 28-dim feature vector for (ctx, candidate)."""
    f = ctx.frame
    ball = ctx.ball_at_contact

    # Proximity at contact
    bbox_at = ctx.cand_bbox_at.get(cand_tid)
    if bbox_at is None or ball is None:
        proximity_to_ball = float("nan")
    else:
        proximity_to_ball = _dist(bbox_at, ball)

    # Pre-contact proximity mean/min
    pre_dists: list[float] = []
    bbox_pre_map = ctx.cand_bbox_pre.get(cand_tid, {})
    for fp, bx, by in ctx.pre_ball:
        bb = bbox_pre_map.get(fp)
        if bb is None:
            continue
        pre_dists.append(_dist(bb, (bx, by)))
    proximity_pre_mean = float(np.mean(pre_dists)) if pre_dists else float("nan")
    proximity_pre_min = float(np.min(pre_dists)) if pre_dists else float("nan")

    # Trajectory endpoint proximity (linear extrapolation of ball -> contact frame)
    traj_end = _extrapolate_ball_endpoint(ctx.pre_ball, f)
    if traj_end is None or bbox_at is None:
        proximity_traj_endpoint = float("nan")
    else:
        proximity_traj_endpoint = _dist(bbox_at, traj_end)

    is_prev_toucher = 1.0 if (ctx.prev_tid is not None and cand_tid == ctx.prev_tid) else 0.0
    is_next_toucher = 1.0 if (ctx.next_tid is not None and cand_tid == ctx.next_tid) else 0.0

    cand_team = ctx.team_assignments.get(cand_tid)
    if cand_team is None:
        team_match_expected = 0.0
        candidate_team = -1.0
    else:
        candidate_team = float(cand_team)
        team_match_expected = 1.0 if (ctx.expected_team is not None and cand_team == ctx.expected_team) else 0.0

    # Distance rank in playerCandidates
    cand_rank = -1.0
    for r, (tid, _d) in enumerate(ctx.candidates):
        if tid == cand_tid:
            cand_rank = float(r)
            break
    if cand_rank < 0:
        cand_rank = float(len(ctx.candidates))  # not in candidates list -> bottom rank

    # bbox motion at contact
    mot = ctx.cand_bbox_motion.get(cand_tid)
    if mot is None:
        bbox_motion_dx = float("nan")
        bbox_motion_dy = float("nan")
        bbox_motion_mag = float("nan")
    else:
        bbox_motion_dx = float(mot[0])
        bbox_motion_dy = float(mot[1])
        bbox_motion_mag = math.hypot(mot[0], mot[1])

    # Body velocities — from positions data
    def _bbox_motion_span(start_f: int, end_f: int) -> float:
        # Get earliest and latest positions in [start_f, end_f]
        pts = []
        for fp in range(start_f, end_f + 1):
            bb = (ctx.cand_bbox_pre.get(cand_tid, {}).get(fp)
                  or ctx.cand_bbox_post.get(cand_tid, {}).get(fp)
                  or (ctx.cand_bbox_at.get(cand_tid) if fp == f else None))
            if bb is not None:
                pts.append((fp, bb))
        if len(pts) < 2:
            return float("nan")
        pts.sort()
        a = pts[0][1]
        b = pts[-1][1]
        return _dist(a, b)

    body_velocity_pre = _bbox_motion_span(f - 3, f)
    body_velocity_post = _bbox_motion_span(f, f + 3)
    body_velocity_around = _bbox_motion_span(f - 3, f + 3)

    # Approach speed: distance change between pre-mean and contact
    if not math.isnan(proximity_pre_mean) and not math.isnan(proximity_to_ball):
        approach_speed_to_ball = proximity_pre_mean - proximity_to_ball  # positive = approaching
    else:
        approach_speed_to_ball = float("nan")

    # Shared per-contact features
    dchg = ctx.direction_change_deg if ctx.direction_change_deg is not None else float("nan")
    arc = ctx.arc_fit_residual if ctx.arc_fit_residual is not None else float("nan")
    vel = ctx.velocity if ctx.velocity is not None else float("nan")
    is_at_net = 1.0 if ctx.is_at_net is True else (0.0 if ctx.is_at_net is False else float("nan"))
    cs_near = 1.0 if (ctx.court_side == "near") else (0.0 if ctx.court_side == "far" else float("nan"))
    action_conf = ctx.confidence if ctx.confidence is not None else float("nan")

    prev_serve = 1.0 if ctx.prev_action == "serve" else 0.0
    prev_receive = 1.0 if ctx.prev_action == "receive" else 0.0
    prev_set = 1.0 if ctx.prev_action == "set" else 0.0
    prev_attack = 1.0 if ctx.prev_action == "attack" else 0.0
    prev_block = 1.0 if ctx.prev_action == "block" else 0.0
    prev_dig = 1.0 if ctx.prev_action == "dig" else 0.0

    prev_is_cross_team = (
        1.0 if (ctx.prev_team is not None and ctx.expected_team is not None and ctx.prev_team != ctx.expected_team)
        else 0.0
    )

    return [
        proximity_to_ball,
        proximity_pre_mean,
        proximity_pre_min,
        proximity_traj_endpoint,
        is_prev_toucher,
        is_next_toucher,
        team_match_expected,
        candidate_team,
        cand_rank,
        bbox_motion_dx,
        bbox_motion_dy,
        bbox_motion_mag,
        body_velocity_pre,
        body_velocity_post,
        body_velocity_around,
        approach_speed_to_ball,
        dchg, arc, vel, is_at_net, cs_near, action_conf,
        prev_serve, prev_receive, prev_set, prev_attack, prev_block, prev_dig,
        prev_is_cross_team,
        float(len(ctx.candidates)),
    ]


# ---------------------------------------------------------------------------
# Pair GT to pipeline contact + build dataset
# ---------------------------------------------------------------------------

def _match_gt_to_ctx(
    ctxs: list[ContactCtx], gt_rows_for_rally: list[GTRow],
) -> list[tuple[GTRow, ContactCtx | None]]:
    out: list[tuple[GTRow, ContactCtx | None]] = []
    for g in gt_rows_for_rally:
        target = g.action.lower()
        best: ContactCtx | None = None
        best_dist = math.inf
        for c in ctxs:
            if c.action != target:
                continue
            dt = abs(c.frame - g.frame)
            if dt <= GT_FRAME_TOL and dt < best_dist:
                best = c
                best_dist = dt
        out.append((g, best))
    return out


@dataclass
class DatasetRow:
    video: str
    rally_id: str
    frame: int
    action_uc: str                   # uppercase action ("SERVE"/...)
    cand_tid: int
    is_gt: int                       # 1 if cand_tid == gt.resolved_tid
    is_pipeline_pick: int            # 1 if cand_tid == ctx.pl_tid
    features: list[float]
    gt_tid: int                      # for bookkeeping
    pl_tid: int | None               # baseline pipeline pick


def build_dataset(
    rally_contacts: dict[str, list[ContactCtx]], gt_rows: list[GTRow],
) -> list[DatasetRow]:
    gt_by_rally: dict[str, list[GTRow]] = defaultdict(list)
    for g in gt_rows:
        gt_by_rally[g.rally_id].append(g)

    rows: list[DatasetRow] = []
    n_total_gt = 0
    n_matched_to_ctx = 0
    n_unresolved = 0
    n_skipped_no_team_assignments = 0
    for rally_id, ctxs in rally_contacts.items():
        gts = gt_by_rally.get(rally_id, [])
        if not gts:
            continue
        pairs = _match_gt_to_ctx(ctxs, gts)
        for g, c in pairs:
            n_total_gt += 1
            if g.resolved_tid is None:
                n_unresolved += 1
                continue
            if c is None:
                continue
            n_matched_to_ctx += 1
            # Enumerate canonical pids 1..4 via team_assignments keys
            cand_pool = sorted(c.team_assignments.keys())
            if len(cand_pool) != 4:
                # If team_assignments isn't 4-player, skip this row
                n_skipped_no_team_assignments += 1
                continue
            for cand_tid in cand_pool:
                feats = compute_features_for_candidate(c, cand_tid)
                rows.append(DatasetRow(
                    video=c.video, rally_id=c.rally_id, frame=c.frame,
                    action_uc=g.action.upper(), cand_tid=cand_tid,
                    is_gt=int(cand_tid == g.resolved_tid),
                    is_pipeline_pick=int(c.pl_tid is not None and cand_tid == c.pl_tid),
                    features=feats, gt_tid=g.resolved_tid,
                    pl_tid=c.pl_tid,
                ))
    print(f"Dataset build: GT={n_total_gt}, matched_to_ctx={n_matched_to_ctx}, unresolved={n_unresolved}, skipped_no_4_team={n_skipped_no_team_assignments}")
    return rows


# ---------------------------------------------------------------------------
# Train + LOVO CV
# ---------------------------------------------------------------------------

@dataclass
class TypeResult:
    type_name: str
    n_gt: int
    n_predicted: int                # rows where model produced a pick
    baseline_correct: int
    learned_correct: int
    baseline_precision: float
    learned_precision: float
    delta_pp: float
    n_baseline_no_pick: int          # baseline had no pl_tid match
    top_features: list[tuple[str, float]]


def evaluate_type(
    type_name: str, all_rows: list[DatasetRow], video_names: list[str],
) -> TypeResult:
    type_rows = [r for r in all_rows if r.action_uc == type_name]
    # Group by (video, rally_id, frame) — each "contact" gets 4 candidate rows.
    contact_groups: dict[tuple[str, str, int], list[DatasetRow]] = defaultdict(list)
    for r in type_rows:
        contact_groups[(r.video, r.rally_id, r.frame)].append(r)

    n_gt = len(contact_groups)

    # Baseline precision: pipeline-pick == GT
    baseline_correct = 0
    baseline_no_pick = 0
    for key, rs in contact_groups.items():
        pl_picks = [r for r in rs if r.is_pipeline_pick]
        if not pl_picks:
            baseline_no_pick += 1
            continue
        # Among pipeline picks, if any matches GT, correct
        if any(r.is_gt == 1 for r in pl_picks):
            baseline_correct += 1

    baseline_precision = baseline_correct / n_gt if n_gt > 0 else 0.0

    # LOVO CV. For each held-out video, train on other videos and predict.
    learned_correct = 0
    n_predicted = 0
    feature_importances_sum = np.zeros(len(FEATURE_NAMES), dtype=np.float64)
    n_models = 0

    for hold_out in video_names:
        train_rows = [r for r in type_rows if r.video != hold_out]
        test_rows = [r for r in type_rows if r.video == hold_out]
        if not train_rows or not test_rows:
            continue
        X_train = np.array([r.features for r in train_rows], dtype=np.float32)
        y_train = np.array([r.is_gt for r in train_rows], dtype=np.int32)
        if y_train.sum() == 0 or y_train.sum() == len(y_train):
            continue
        model = HistGradientBoostingClassifier(
            max_iter=200,
            max_depth=4,
            learning_rate=0.05,
            min_samples_leaf=10,
            l2_regularization=1.0,
            random_state=42,
        )
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            print(f"  [{type_name}] hold={hold_out} fit failed: {e}")
            continue
        # Feature importances via permutation on training set would be costly;
        # we approximate via the model's split-based importance. HistGB exposes
        # internal feature importance via `_predictors`. Use sklearn's
        # `permutation_importance` on a small held-out slice instead.
        # Simpler: collect partial dependence-like signal via .predict_proba diffs.
        # We'll do permutation importance on the TRAIN set (cheap, K iters small).
        try:
            from sklearn.inspection import permutation_importance
            # Permute on a small subsample to keep cost low
            sub = np.random.default_rng(42).choice(len(X_train), size=min(800, len(X_train)), replace=False)
            pi = permutation_importance(model, X_train[sub], y_train[sub], n_repeats=3, random_state=42, n_jobs=1)
            feature_importances_sum += pi.importances_mean
            n_models += 1
        except Exception as e:
            print(f"  [{type_name}] hold={hold_out} perm-importance failed: {e}")

        # Predict
        test_groups: dict[tuple[str, str, int], list[DatasetRow]] = defaultdict(list)
        for r in test_rows:
            test_groups[(r.video, r.rally_id, r.frame)].append(r)
        for key, rs in test_groups.items():
            X_test = np.array([r.features for r in rs], dtype=np.float32)
            probs = model.predict_proba(X_test)[:, 1]
            best_idx = int(np.argmax(probs))
            pick = rs[best_idx]
            n_predicted += 1
            if pick.is_gt == 1:
                learned_correct += 1
        # Per-fold progress
        print(f"  [{type_name}] hold={hold_out:>4}: train={len(train_rows):>4}, test={len(test_rows):>3}, learned_running={learned_correct}/{n_predicted}")

    learned_precision = learned_correct / n_predicted if n_predicted > 0 else 0.0
    delta_pp = (learned_precision - baseline_precision) * 100.0
    # Average feature importances
    if n_models > 0:
        feature_importances_avg = feature_importances_sum / n_models
    else:
        feature_importances_avg = feature_importances_sum
    fi_ranked = sorted(
        zip(FEATURE_NAMES, feature_importances_avg.tolist()),
        key=lambda x: x[1], reverse=True,
    )[:5]

    return TypeResult(
        type_name=type_name, n_gt=n_gt, n_predicted=n_predicted,
        baseline_correct=baseline_correct, learned_correct=learned_correct,
        baseline_precision=baseline_precision, learned_precision=learned_precision,
        delta_pp=delta_pp, n_baseline_no_pick=baseline_no_pick,
        top_features=fi_ranked,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_report(
    results: list[TypeResult], block_n: int, video_names: list[str],
) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # JSON
    out = {
        "results": [
            {
                "type": r.type_name,
                "n_gt": r.n_gt,
                "n_predicted": r.n_predicted,
                "baseline_correct": r.baseline_correct,
                "learned_correct": r.learned_correct,
                "baseline_precision": r.baseline_precision,
                "learned_precision": r.learned_precision,
                "delta_pp": r.delta_pp,
                "n_baseline_no_pick": r.n_baseline_no_pick,
                "top_features": r.top_features,
            }
            for r in results
        ],
        "block_n_gt": block_n,
        "videos": video_names,
        "feature_names": FEATURE_NAMES,
    }
    JSON_PATH.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote JSON: {JSON_PATH}")

    # Markdown
    n_baseline_total = sum(r.baseline_correct for r in results)
    n_learned_total = sum(r.learned_correct for r in results)
    n_predicted_total = sum(r.n_predicted for r in results)
    n_gt_total = sum(r.n_gt for r in results)
    agg_baseline = n_baseline_total / n_gt_total if n_gt_total else 0.0
    agg_learned = n_learned_total / n_predicted_total if n_predicted_total else 0.0
    agg_delta = (agg_learned - agg_baseline) * 100.0

    types_with_lift_5pp = [r for r in results if r.delta_pp >= 5.0]
    types_with_regression_2pp = [r for r in results if r.delta_pp <= -2.0]

    # Decision criteria
    ship = len(types_with_lift_5pp) >= 2 and len(types_with_regression_2pp) == 0
    if ship:
        verdict = "SHIP-LEARNED-APPROACH (invest in 2-3K more labels)"
    elif len(types_with_lift_5pp) >= 1 and len(types_with_regression_2pp) == 0:
        verdict = "MIXED — single-type lift only; insufficient evidence to invest in 2-3K labels"
    elif agg_delta >= 3.0 and len(types_with_regression_2pp) == 0:
        verdict = "MIXED — small aggregate lift; consider before investing in 2-3K labels"
    else:
        verdict = "NO-SIGNAL-DONT-LABEL"

    lines: list[str] = []
    lines.append("# Per-action-type learned attribution — validation experiment (2026-05-14)")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- 641 GT rows across 12 trusted-GT videos: {', '.join(video_names)}")
    lines.append(f"- {len(FEATURE_NAMES)} features per candidate (mixed per-candidate + per-contact-shared signals)")
    lines.append("- HistGradientBoostingClassifier (sklearn) per-action-type — equivalent gradient boosting to LightGBM")
    lines.append("- Per-candidate binary target `is_gt`; argmax(P(is_gt=1)) across 4 candidates picks the predicted attribution")
    lines.append(f"- Leave-one-video-out CV across all {len(video_names)} videos")
    lines.append("- Anti-leakage: `is_pipeline_pick` is excluded from features")
    lines.append("")
    lines.append("## Per-type precision")
    lines.append("")
    lines.append("| Type | N GT (matched-to-contact) | Baseline pipeline | Learned | Δ (pp) |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r.type_name} | {r.n_gt} | "
            f"{r.baseline_correct}/{r.n_gt} = {r.baseline_precision*100:.1f}% | "
            f"{r.learned_correct}/{r.n_predicted} = {r.learned_precision*100:.1f}% | "
            f"{r.delta_pp:+.1f}pp |"
        )
    if block_n > 0:
        lines.append(f"| BLOCK | {block_n} | n/a | INSUFFICIENT DATA (<20 examples; held out) | - |")
    lines.append("")
    lines.append(
        f"**Aggregate** (excluding BLOCK): "
        f"baseline {n_baseline_total}/{n_gt_total} = {agg_baseline*100:.1f}% → "
        f"learned {n_learned_total}/{n_predicted_total} = {agg_learned*100:.1f}% "
        f"({agg_delta:+.1f}pp)"
    )
    lines.append("")
    lines.append("Note: 'N GT' here is GT rows that matched a pipeline contact within ±5 frames")
    lines.append("(the same denominator as the baseline-precision A-table in `precision_trusted_videos_2026_05_14`).")
    lines.append("Baseline 'no pipeline pick' (pl_tid missing) counts as wrong; learned always picks 1 of 4.")
    lines.append("")
    lines.append("## Per-type top-5 features (by permutation importance)")
    lines.append("")
    for r in results:
        lines.append(f"### {r.type_name}")
        for name, imp in r.top_features:
            lines.append(f"- `{name}`: {imp:.4f}")
        lines.append("")

    lines.append("## Decision criteria")
    lines.append("")
    lines.append("Ship gate for labeling investment:")
    lines.append("- ≥ 5pp lift over baseline on **at least 2 action types**")
    lines.append("- No action type regresses by more than 2pp")
    lines.append("")
    lines.append(f"- Types with ≥5pp lift: {len(types_with_lift_5pp)} ({', '.join(r.type_name for r in types_with_lift_5pp) or 'none'})")
    lines.append(f"- Types regressing by ≥2pp: {len(types_with_regression_2pp)} ({', '.join(r.type_name for r in types_with_regression_2pp) or 'none'})")
    lines.append("")
    lines.append(f"## Verdict: **{verdict}**")
    lines.append("")

    MD_PATH.write_text("\n".join(lines))
    print(f"Wrote markdown: {MD_PATH}")
    return verdict, results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=== Per-action-type learned attribution experiment (2026-05-14) ===")
    print(f"Videos: {VIDEO_NAMES}")
    print(f"GT_FRAME_TOL={GT_FRAME_TOL}, PRE_WINDOW={PRE_WINDOW}, POST_WINDOW={POST_WINDOW}")
    print()

    print("Step 1/4: Fetching DB data...")
    rally_contacts, gt_rows = fetch_data(VIDEO_NAMES)
    print(f"  Rallies with contacts: {len(rally_contacts)}, total GT rows: {len(gt_rows)}")
    by_type = Counter(g.action for g in gt_rows)
    print(f"  GT per type: {dict(by_type)}")
    print()

    print("Step 2/4: Building per-candidate feature dataset...")
    rows = build_dataset(rally_contacts, gt_rows)
    by_type_rows = Counter(r.action_uc for r in rows)
    print(f"  Total dataset rows: {len(rows)} ({len(rows)//4} contacts × 4 candidates)")
    print(f"  Per type (rows): {dict(by_type_rows)}")
    print()

    print("Step 3/4: Per-type LOVO CV training...")
    results: list[TypeResult] = []
    for atype in ACTION_TYPES_TO_TRAIN:
        print(f"--- Training {atype} ---")
        r = evaluate_type(atype, rows, VIDEO_NAMES)
        print(
            f"  RESULT [{atype}]: N={r.n_gt} | "
            f"baseline {r.baseline_correct}/{r.n_gt}={r.baseline_precision*100:.1f}% | "
            f"learned {r.learned_correct}/{r.n_predicted}={r.learned_precision*100:.1f}% | "
            f"Δ {r.delta_pp:+.1f}pp"
        )
        results.append(r)
    print()

    block_rows = [r for r in rows if r.action_uc == "BLOCK"]
    block_contacts = len(set((r.video, r.rally_id, r.frame) for r in block_rows))
    print(f"BLOCK contacts in dataset (held out): {block_contacts}")

    print("Step 4/4: Writing report...")
    write_report(results, block_contacts, VIDEO_NAMES)
    return 0


if __name__ == "__main__":
    sys.exit(main())
