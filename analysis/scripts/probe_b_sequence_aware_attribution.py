"""Probe B — Sequence-aware role attribution (2026-05-14).

Tests Hypotheses B.1 / B.2 / B.3 from the spec:

  B.1: The ball's pre-contact trajectory ENDPOINT (extrapolating the
       ball motion in the K=10 frames leading up to the contact, NOT
       just the ball position AT contact) better predicts the actual
       toucher than the ball-position-at-contact.

  B.2: For SET/RECEIVE/DIG actions, the actual toucher is the
       same-team player whose body-center is closest to the BALL
       TRAJECTORY (not just position) over the K=10 frames pre-contact.
       Setter / digger positions UNDER the ball.

  B.3: For consecutive same-team contacts, the second-toucher is
       NEVER the previous toucher (volleyball rule, block exception).
       Applied as a final tie-break.

For each of N hand-picked + stratified probe cases (same-team errors
where pl_tid != gt resolved_track_id AND both pl & gt sit in the contact's
playerCandidates), we compute five candidate-picking strategies and
compare each to GT:

  S0 (baseline) Pipeline pick: `actions_json[i].playerTrackId`.
  S1 (ball-pos) argmin distance(bbox_center(c@f), ball@f), c ∈ same-team cands.
  S2 (traj-end) Linear-fit ball trajectory over [f-K, f-1]; project to f;
                argmin distance(bbox_center(c@f), traj_endpoint).
  S3 (traj-int) For each cand, integrate dist(bbox_center(c@f'), ball@f')
                over f' in [f-K, f-1] (gaps skipped). Pick min integral.
  S4 (anti-self) S3 + exclude the previous toucher unless its action was
                 BLOCK.

Ship threshold:  if (S2 OR S3 OR S4) beats S1 by ≥ 3/10 cases, "worth
                 building"; else proximity-based signals are insufficient
                 even with sequence context.

Outputs (under analysis/reports/probe_b_sequence_aware/2026_05_14/):
  results.json       — full per-case + aggregate
  results.md         — human-readable
  visual_frames/*.jpg — annotated frames (ball trajectory + candidate boxes)
  visual.html        — visual page

Usage:
    cd analysis
    uv run python scripts/probe_b_sequence_aware_attribution.py
"""
from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from rallycut.evaluation.tracking.db import get_connection  # noqa: E402
from rallycut.evaluation.video_resolver import VideoResolver  # noqa: E402

REPORT_DIR = HERE.parent / "reports" / "probe_b_sequence_aware" / "2026_05_14"
FRAMES_DIR = REPORT_DIR / "visual_frames"
JSON_PATH = REPORT_DIR / "results.json"
MD_PATH = REPORT_DIR / "results.md"
HTML_PATH = REPORT_DIR / "visual.html"

K_PRE = 10                       # frames before contact for trajectory analysis
GT_FRAME_TOL = 5                 # GT frame match tolerance
MIN_BALL_PRE = 5                 # require ≥ N ball positions in pre-window


# ---------------------------------------------------------------------------
# Probe case list
# ---------------------------------------------------------------------------
#
# We hand-pick 4 mandatory cases (the 2 cascade + F3 + F5 from spec) plus
# 6 stratified across videos chosen from the live same-team-error pool.
# Per the spec:
#  - F3 keke 0144acfb f=223 has no GT yet (`rally_action_ground_truth` empty)
#    so we skip gracefully.
#  - F5 keke 99091ec6 f=184: pipeline already agrees with GT (resolved=2,
#    pipeline=2). Becomes an effective control case — pipeline is already
#    correct, S2/S3/S4 should preserve.
# Mandatory + 6 stratified additional same-team errors:

PROBE_CASES: list[dict[str, Any]] = [
    # MANDATORY 4
    dict(
        idx=1, label="cascade-f128",
        video="titi", rally_short="a0881d82", frame=128, action="SET",
        note="cascade f128 — pipeline picked p2 (closest), GT=p1 (same-team B)",
    ),
    dict(
        idx=2, label="cascade-f225",
        video="titi", rally_short="a0881d82", frame=225, action="DIG",
        note="cascade f225 — pipeline picked p2, GT=p1 (same-team B)",
    ),
    dict(
        idx=3, label="F3-keke",
        video="keke", rally_short="0144acfb", frame=223, action="ATTACK",
        note="F3 occlusion case — likely no GT yet, will skip",
    ),
    dict(
        idx=4, label="F5-keke",
        video="keke", rally_short="99091ec6", frame=184, action="ATTACK",
        note="F5 mid-rally attack — pipeline currently matches GT (control)",
    ),
    # STRATIFIED 6 — same-team errors across distinct videos with K=10 ball coverage
    dict(
        idx=5, label="stratified-jojo",
        video="jojo", rally_short="36d3aa2c", frame=143, action="RECEIVE",
        note="same-team error: pl=4 closest, gt=3",
    ),
    dict(
        idx=6, label="stratified-tutu-set",
        video="tutu", rally_short="9064ba7b", frame=254, action="SET",
        note="same-team error: pl=1 closest, gt=2 (tight gap 0.009 vs 0.025)",
    ),
    dict(
        idx=7, label="stratified-gugu-dig",
        video="gugu", rally_short="62b6c286", frame=230, action="DIG",
        note="same-team error: pl=3 closest, gt=4 (tight gap)",
    ),
    dict(
        idx=8, label="stratified-jaja",
        video="jaja", rally_short="2ae99d01", frame=146, action="RECEIVE",
        note="same-team error: pl=1, gt=2",
    ),
    dict(
        idx=9, label="stratified-natch",
        video="natch", rally_short="e5e4c0b7", frame=214, action="ATTACK",
        note="same-team error: pl=1 closest, gt=2",
    ),
    dict(
        idx=10, label="stratified-lala-attack",
        video="lala", rally_short="793625cd", frame=335, action="ATTACK",
        note="same-team error: pl=4 closest, gt=3 (tight: 0.091 vs 0.092)",
    ),
]


# ---------------------------------------------------------------------------
# Data hydration
# ---------------------------------------------------------------------------

@dataclass
class HydratedCase:
    idx: int
    label: str
    video_name: str
    video_id: str
    rally_id: str
    rally_short: str
    pl_frame: int                 # rally-relative frame the pipeline picked
    action_type: str              # uppercase
    pl_tid: int                   # pipeline pick (S0)
    gt_tid: int | None            # rally_action_ground_truth.resolved_track_id
    ball_xy_at_contact: tuple[float, float]
    same_team_cands: list[tuple[int, float]]   # (tid, bbox-dist) sorted
    cand_team: dict[int, str]                  # tid → team for ALL contact cands
    team_assignments: dict[str, str]
    rally_start_ms: int
    pre_ball_positions: list[tuple[int, float, float]]   # (frame, x, y) frames in [f-K, f-1]
    cand_bbox_pre: dict[int, dict[int, tuple[float, float]]]  # tid → frame → (cx, cy)
    cand_bbox_at_contact: dict[int, tuple[float, float, float, float]]  # tid → (cx, cy, w, h)
    prev_action: dict[str, Any] | None         # previous action in actions list
    note: str
    skipped: bool = False
    skip_reason: str = ""


def _fetch_cases() -> list[HydratedCase]:
    video_names = sorted({c["video"] for c in PROBE_CASES})

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT name, id FROM videos WHERE name = ANY(%s)",
            [video_names],
        )
        name_to_vid = {r[0]: r[1] for r in cur.fetchall()}

    out: list[HydratedCase] = []
    with get_connection() as conn, conn.cursor() as cur:
        for c in PROBE_CASES:
            vname = c["video"]
            vid = name_to_vid.get(vname)
            if not vid:
                print(f"[{c['idx']}] SKIP — video {vname} not found")
                out.append(_skipped(c, vid="", rid="", reason=f"video {vname} missing"))
                continue
            cur.execute(
                """
                SELECT r.id, r.start_ms
                FROM rallies r
                WHERE r.video_id = %s AND r.id::text LIKE %s || '%%'
                """,
                [vid, c["rally_short"]],
            )
            rrow = cur.fetchone()
            if not rrow:
                print(f"[{c['idx']}] SKIP — rally {c['rally_short']} not found")
                out.append(_skipped(c, vid=vid, rid="", reason="rally missing"))
                continue
            rid, start_ms = str(rrow[0]), int(rrow[1] or 0)

            cur.execute(
                """
                SELECT actions_json, contacts_json, positions_json, ball_positions_json
                FROM player_tracks WHERE rally_id = %s
                """,
                [rid],
            )
            prow = cur.fetchone()
            if not prow:
                out.append(_skipped(c, vid=vid, rid=rid, reason="no player_tracks row"))
                continue
            aj, cj, posj, bj = prow
            actions = (aj or {}).get("actions", []) or []
            team_ass = (aj or {}).get("teamAssignments", {}) or {}
            contacts = (cj or {}).get("contacts", []) or []
            positions = posj or []
            ball_positions = bj or []

            # ---- Find the pipeline action at the target frame (action match) ----
            target_action_type = c["action"].lower()
            target_a = None
            target_idx = -1
            for i, a in enumerate(actions):
                if abs(int(a.get("frame", -1)) - c["frame"]) <= GT_FRAME_TOL \
                        and str(a.get("action", "")).lower() == target_action_type:
                    target_a = a
                    target_idx = i
                    break
            if target_a is None:
                # Fallback: any action within tolerance
                for i, a in enumerate(actions):
                    if abs(int(a.get("frame", -1)) - c["frame"]) <= GT_FRAME_TOL:
                        target_a = a
                        target_idx = i
                        break

            if target_a is None:
                out.append(_skipped(c, vid=vid, rid=rid,
                                    reason=f"no pipeline action near f={c['frame']}"))
                continue

            pl_frame = int(target_a["frame"])
            pl_tid = int(target_a.get("playerTrackId") or -1)
            ball_xy = (float(target_a.get("ballX") or 0.0),
                       float(target_a.get("ballY") or 0.0))
            prev_action = actions[target_idx - 1] if target_idx > 0 else None

            # ---- Contact (same frame) for player_candidates ----
            cont = next((cc for cc in contacts
                         if int(cc.get("frame", -1)) == pl_frame), None)
            if not cont:
                out.append(_skipped(c, vid=vid, rid=rid,
                                    reason=f"no contact at f={pl_frame}"))
                continue
            raw_cands = cont.get("playerCandidates", []) or []
            cand_team: dict[int, str] = {}
            for cd in raw_cands:
                if not cd or len(cd) < 2:
                    continue
                try:
                    tid = int(cd[0])
                except (TypeError, ValueError):
                    continue
                t = team_ass.get(str(tid))
                if t:
                    cand_team[tid] = str(t)

            pl_team = team_ass.get(str(pl_tid))
            same_team_cands: list[tuple[int, float]] = []
            for cd in raw_cands:
                if not cd or len(cd) < 2:
                    continue
                try:
                    tid = int(cd[0]); dist = float(cd[1])
                except (TypeError, ValueError):
                    continue
                if pl_team is not None and team_ass.get(str(tid)) == pl_team:
                    same_team_cands.append((tid, dist))
            # Always sort by distance ascending
            same_team_cands.sort(key=lambda x: x[1])

            # ---- GT resolved_track_id (frame tolerance match) ----
            cur.execute(
                """
                SELECT frame, resolved_track_id
                FROM rally_action_ground_truth
                WHERE rally_id = %s
                  AND action::text = %s
                  AND resolved_track_id IS NOT NULL
                  AND ABS(frame - %s) <= %s
                ORDER BY ABS(frame - %s) ASC
                LIMIT 1
                """,
                [rid, c["action"].upper(), pl_frame, GT_FRAME_TOL, pl_frame],
            )
            gt_row = cur.fetchone()
            gt_tid: int | None = int(gt_row[1]) if gt_row and gt_row[1] is not None else None

            # ---- Ball positions in [pl_frame - K, pl_frame - 1] ----
            ball_by_frame: dict[int, tuple[float, float]] = {}
            for bp in ball_positions:
                try:
                    fnum = int(bp["frameNumber"])
                    bx = float(bp.get("x", 0.0))
                    by = float(bp.get("y", 0.0))
                except (KeyError, TypeError, ValueError):
                    continue
                if bx <= 0 and by <= 0:
                    continue
                ball_by_frame[fnum] = (bx, by)

            pre_ball: list[tuple[int, float, float]] = []
            for f in range(pl_frame - K_PRE, pl_frame):
                if f in ball_by_frame:
                    pre_ball.append((f, ball_by_frame[f][0], ball_by_frame[f][1]))

            # ---- Candidate bbox-center over [pl_frame - K, pl_frame] ----
            # Use ALL candidate tids (need them for S0/S1 too).
            all_tids = sorted({int(cd[0]) for cd in raw_cands if cd and len(cd) >= 2})
            cand_pre: dict[int, dict[int, tuple[float, float]]] = {t: {} for t in all_tids}
            cand_at: dict[int, tuple[float, float, float, float]] = {}
            for p in positions:
                try:
                    fnum = int(p.get("frameNumber", -1))
                    tid = int(p.get("trackId", -1))
                except (TypeError, ValueError):
                    continue
                if tid not in cand_pre:
                    continue
                if pl_frame - K_PRE <= fnum < pl_frame:
                    cand_pre[tid][fnum] = (float(p.get("x", 0.0)), float(p.get("y", 0.0)))
                if fnum == pl_frame:
                    cand_at[tid] = (
                        float(p.get("x", 0.0)),
                        float(p.get("y", 0.0)),
                        float(p.get("width", 0.0)),
                        float(p.get("height", 0.0)),
                    )

            out.append(HydratedCase(
                idx=c["idx"], label=c["label"],
                video_name=vname, video_id=vid,
                rally_id=rid, rally_short=c["rally_short"],
                pl_frame=pl_frame,
                action_type=c["action"].upper(),
                pl_tid=pl_tid,
                gt_tid=gt_tid,
                ball_xy_at_contact=ball_xy,
                same_team_cands=same_team_cands,
                cand_team=cand_team,
                team_assignments={str(k): str(v) for k, v in team_ass.items()},
                rally_start_ms=start_ms,
                pre_ball_positions=pre_ball,
                cand_bbox_pre=cand_pre,
                cand_bbox_at_contact=cand_at,
                prev_action=prev_action,
                note=c["note"],
            ))

    return out


def _skipped(c: dict[str, Any], vid: str, rid: str, reason: str) -> HydratedCase:
    return HydratedCase(
        idx=c["idx"], label=c["label"],
        video_name=c["video"], video_id=vid, rally_id=rid,
        rally_short=c["rally_short"],
        pl_frame=int(c["frame"]), action_type=c["action"].upper(),
        pl_tid=-1, gt_tid=None, ball_xy_at_contact=(0.0, 0.0),
        same_team_cands=[], cand_team={}, team_assignments={},
        rally_start_ms=0, pre_ball_positions=[],
        cand_bbox_pre={}, cand_bbox_at_contact={},
        prev_action=None, note=c.get("note", ""),
        skipped=True, skip_reason=reason,
    )


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    """positions_json `x`,`y` are already the bbox center (normalized)."""
    return (bbox[0], bbox[1])


def _extrapolate_endpoint(
    pre_ball: list[tuple[int, float, float]],
    target_frame: int,
) -> tuple[float, float] | None:
    """Linear least-squares fit on pre-ball (frame, x), (frame, y); project to target_frame."""
    if len(pre_ball) < 3:
        return None
    frames = np.array([p[0] for p in pre_ball], dtype=np.float64)
    xs = np.array([p[1] for p in pre_ball], dtype=np.float64)
    ys = np.array([p[2] for p in pre_ball], dtype=np.float64)
    # Linear fit: x = a*f + b
    fx_coef = np.polyfit(frames, xs, 1)
    fy_coef = np.polyfit(frames, ys, 1)
    px = float(np.polyval(fx_coef, target_frame))
    py = float(np.polyval(fy_coef, target_frame))
    return (px, py)


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


@dataclass
class CaseResult:
    idx: int
    label: str
    video: str
    rally_short: str
    pl_frame: int
    action_type: str
    gt_tid: int | None
    pl_tid: int
    s0_pick: int                              # pipeline (same as pl_tid)
    s1_pick: int | None                       # ball-pos
    s2_pick: int | None                       # traj-endpoint
    s2_endpoint: tuple[float, float] | None
    s3_pick: int | None                       # traj-integral
    s3_integrals: dict[int, float] = field(default_factory=dict)
    s4_pick: int | None = None                # anti-self tiebreak on S3
    prev_toucher_tid: int | None = None
    prev_action_type: str | None = None
    same_team_cands: list[tuple[int, float]] = field(default_factory=list)
    n_pre_ball: int = 0
    s0_correct: bool | None = None
    s1_correct: bool | None = None
    s2_correct: bool | None = None
    s3_correct: bool | None = None
    s4_correct: bool | None = None
    skipped: bool = False
    skip_reason: str = ""
    visual_path: str | None = None
    note: str = ""


def _evaluate(case: HydratedCase) -> CaseResult:
    base = CaseResult(
        idx=case.idx, label=case.label,
        video=case.video_name, rally_short=case.rally_short,
        pl_frame=case.pl_frame, action_type=case.action_type,
        gt_tid=case.gt_tid, pl_tid=case.pl_tid,
        s0_pick=case.pl_tid,
        s1_pick=None, s2_pick=None, s2_endpoint=None, s3_pick=None,
        prev_toucher_tid=(int(case.prev_action.get("playerTrackId"))
                          if case.prev_action and case.prev_action.get("playerTrackId") is not None
                          else None),
        prev_action_type=(str(case.prev_action.get("action")).upper()
                          if case.prev_action and case.prev_action.get("action") is not None
                          else None),
        same_team_cands=case.same_team_cands,
        n_pre_ball=len(case.pre_ball_positions),
        note=case.note,
    )

    if case.skipped:
        base.skipped = True
        base.skip_reason = case.skip_reason
        return base
    if not case.same_team_cands:
        base.skipped = True
        base.skip_reason = "no same-team candidates"
        return base
    if len(case.pre_ball_positions) < MIN_BALL_PRE:
        base.skipped = True
        base.skip_reason = f"only {len(case.pre_ball_positions)} ball pre-frames"
        return base

    cand_tids = [tid for tid, _ in case.same_team_cands]

    # S0
    base.s0_pick = case.pl_tid

    # S1: nearest same-team to ball@contact frame
    ball_at = case.ball_xy_at_contact
    s1_best, s1_bd = -1, math.inf
    for tid in cand_tids:
        bb = case.cand_bbox_at_contact.get(tid)
        if bb is None:
            continue
        cx, cy = _bbox_center(bb)
        d = _dist((cx, cy), ball_at)
        if d < s1_bd:
            s1_best, s1_bd = tid, d
    base.s1_pick = s1_best if s1_best >= 0 else None

    # S2: trajectory endpoint extrapolated to pl_frame
    endpoint = _extrapolate_endpoint(case.pre_ball_positions, case.pl_frame)
    base.s2_endpoint = endpoint
    s2_best, s2_bd = -1, math.inf
    if endpoint is not None:
        for tid in cand_tids:
            bb = case.cand_bbox_at_contact.get(tid)
            if bb is None:
                continue
            cx, cy = _bbox_center(bb)
            d = _dist((cx, cy), endpoint)
            if d < s2_bd:
                s2_best, s2_bd = tid, d
    base.s2_pick = s2_best if s2_best >= 0 else None

    # S3: integral of distance over pre-window
    s3_integrals: dict[int, float] = {}
    for tid in cand_tids:
        bb_map = case.cand_bbox_pre.get(tid, {})
        if not bb_map:
            continue
        total = 0.0
        count = 0
        for f, bx, by in case.pre_ball_positions:
            bb = bb_map.get(f)
            if bb is None:
                continue
            total += _dist(bb, (bx, by))
            count += 1
        if count > 0:
            # Normalize by count (mean dist over evaluated frames) to be robust
            # to mismatched coverage between candidates.
            s3_integrals[tid] = total / count
    base.s3_integrals = s3_integrals

    if s3_integrals:
        s3_best = min(s3_integrals.items(), key=lambda x: x[1])[0]
        base.s3_pick = s3_best
    else:
        base.s3_pick = None

    # S4: anti-self tiebreak on S3 — exclude prev_toucher unless prev action == BLOCK
    if s3_integrals:
        prev_tid = base.prev_toucher_tid
        prev_act = base.prev_action_type or ""
        if prev_tid is not None and prev_act != "BLOCK":
            filt = {t: v for t, v in s3_integrals.items() if t != prev_tid}
            if filt:
                base.s4_pick = min(filt.items(), key=lambda x: x[1])[0]
            else:
                base.s4_pick = base.s3_pick
        else:
            base.s4_pick = base.s3_pick
    else:
        base.s4_pick = None

    # GT correctness
    if case.gt_tid is None:
        return base
    base.s0_correct = (base.s0_pick == case.gt_tid)
    base.s1_correct = (base.s1_pick == case.gt_tid) if base.s1_pick is not None else None
    base.s2_correct = (base.s2_pick == case.gt_tid) if base.s2_pick is not None else None
    base.s3_correct = (base.s3_pick == case.gt_tid) if base.s3_pick is not None else None
    base.s4_correct = (base.s4_pick == case.gt_tid) if base.s4_pick is not None else None

    return base


# ---------------------------------------------------------------------------
# Annotation
# ---------------------------------------------------------------------------

COLOR_BALL = (0, 255, 255)
COLOR_TRAJ = (255, 200, 100)
COLOR_ENDPOINT = (255, 80, 255)
COLOR_S0 = (200, 200, 200)
COLOR_S1 = (220, 80, 80)
COLOR_S2 = (255, 200, 60)
COLOR_S3 = (60, 220, 220)
COLOR_S4 = (60, 220, 60)
COLOR_GT = (255, 60, 255)


def _annotate(
    *, case: HydratedCase, result: CaseResult,
    video_path: Path, source_frame: int, img_w: int, img_h: int,
) -> Path | None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, source_frame)
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    h, w = frame.shape[:2]
    img = frame.copy()

    # Draw candidate bboxes (all candidates at contact frame)
    for tid, bb in case.cand_bbox_at_contact.items():
        cx, cy, bw, bh = bb
        if bw <= 0 or bh <= 0:
            continue
        x1 = int((cx - bw / 2) * w); y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w); y2 = int((cy + bh / 2) * h)
        team = case.cand_team.get(tid, "?")
        color = (160, 160, 160)
        is_same_team = any(tid == c[0] for c in case.same_team_cands)
        if is_same_team:
            color = (90, 200, 90)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

        # Strategy halos (stacked)
        if is_same_team:
            offset = -3
            picks = [
                (result.s0_pick, COLOR_S0, "S0"),
                (result.s1_pick, COLOR_S1, "S1"),
                (result.s2_pick, COLOR_S2, "S2"),
                (result.s3_pick, COLOR_S3, "S3"),
                (result.s4_pick, COLOR_S4, "S4"),
                (case.gt_tid,    COLOR_GT, "GT"),
            ]
            for pid, col, _name in picks:
                if pid is None:
                    continue
                if pid != tid:
                    continue
                cv2.rectangle(img, (x1 + offset, y1 + offset),
                              (x2 - offset, y2 - offset), col, 2)
                offset -= 3

        # Label
        label_parts = [f"p{tid}({team})"]
        for pid, _col, name in [
            (result.s0_pick, None, "S0"),
            (result.s1_pick, None, "S1"),
            (result.s2_pick, None, "S2"),
            (result.s3_pick, None, "S3"),
            (result.s4_pick, None, "S4"),
            (case.gt_tid,    None, "GT"),
        ]:
            if pid == tid and pid is not None:
                label_parts.append(name)
        label = " ".join(label_parts)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        ly = max(y1 - 6, 14)
        cv2.rectangle(img, (x1 - 2, ly - th - 4), (x1 + tw + 4, ly + 2),
                      (0, 0, 0), -1)
        cv2.putText(img, label, (x1, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw pre-ball trajectory polyline
    pre_pts = [(int(x * w), int(y * h)) for _f, x, y in case.pre_ball_positions]
    for i in range(1, len(pre_pts)):
        cv2.line(img, pre_pts[i - 1], pre_pts[i], COLOR_TRAJ, 2, cv2.LINE_AA)
    for p in pre_pts:
        cv2.circle(img, p, 3, COLOR_TRAJ, -1)
    # Ball at contact
    bx_n, by_n = case.ball_xy_at_contact
    if bx_n > 0 or by_n > 0:
        bx, by = int(bx_n * w), int(by_n * h)
        cv2.circle(img, (bx, by), 14, COLOR_BALL, 3)
        cv2.circle(img, (bx, by), 2, COLOR_BALL, -1)
    # Extrapolated endpoint
    if result.s2_endpoint:
        ex, ey = result.s2_endpoint
        if 0.0 <= ex <= 1.0 and 0.0 <= ey <= 1.0:
            cv2.drawMarker(img, (int(ex * w), int(ey * h)),
                           COLOR_ENDPOINT, cv2.MARKER_CROSS, 18, 3)

    # Banner
    banner_h = 70
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.65, img, 0.35, 0, img)
    title = (f"#{case.idx} {case.label}  {case.video_name}/{case.rally_short} "
             f"{case.action_type} f={case.pl_frame}")
    sub1 = (f"GT=p{case.gt_tid}  S0=p{result.s0_pick}  S1=p{result.s1_pick}  "
            f"S2=p{result.s2_pick}  S3=p{result.s3_pick}  S4=p{result.s4_pick}")
    sub2 = f"prev={result.prev_action_type} prev_pid=p{result.prev_toucher_tid}  n_pre_ball={result.n_pre_ball}"
    cv2.putText(img, title, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(img, sub1, (10, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (240, 240, 240), 1, cv2.LINE_AA)
    cv2.putText(img, sub2, (10, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    out = FRAMES_DIR / f"{case.idx:02d}_{case.video_name}_{case.rally_short}_f{case.pl_frame}.jpg"
    cv2.imwrite(str(out), img, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    return out


def _resolve_video(resolver: VideoResolver, vm: dict[str, Any]) -> Path | None:
    candidates: list[tuple[str, str]] = []
    if vm.get("proxy_s3_key"):
        candidates.append(("proxy", vm["proxy_s3_key"]))
    if vm.get("s3_key"):
        candidates.append(("original", vm["s3_key"]))
    if vm.get("processed_s3_key"):
        candidates.append(("processed", vm["processed_s3_key"]))
    for _label, key in candidates:
        try:
            return resolver.resolve(key, vm["content_hash"])
        except Exception as e:  # noqa: BLE001
            print(f"    resolver miss {_label}: {e}")
    return None


def _fetch_video_meta(video_id: str) -> dict[str, Any]:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """SELECT id, name, filename, fps, width, height,
                      s3_key, proxy_s3_key, processed_s3_key, content_hash
               FROM videos WHERE id = %s""",
            (video_id,),
        )
        r = cur.fetchone()
    if not r:
        return {}
    return {
        "id": r[0], "name": r[1], "filename": r[2],
        "fps": float(r[3]) if r[3] is not None else 30.0,
        "width": int(r[4] or 0), "height": int(r[5] or 0),
        "s3_key": r[6], "proxy_s3_key": r[7],
        "processed_s3_key": r[8], "content_hash": r[9],
    }


# ---------------------------------------------------------------------------
# Aggregation + reporting
# ---------------------------------------------------------------------------

def _aggregate(results: list[CaseResult]) -> dict[str, Any]:
    evaluable = [r for r in results if not r.skipped and r.gt_tid is not None]
    # Within-team-only subset: GT is in same-team candidate set (probe scope).
    within_team = [
        r for r in evaluable
        if r.gt_tid is not None and any(t == r.gt_tid for t, _ in r.same_team_cands)
    ]
    n = len(evaluable)
    nw = len(within_team)

    def _count(rs: list[CaseResult], field_name: str) -> int:
        return sum(1 for r in rs if getattr(r, field_name) is True)

    s0 = _count(evaluable, "s0_correct")
    s1 = _count(evaluable, "s1_correct")
    s2 = _count(evaluable, "s2_correct")
    s3 = _count(evaluable, "s3_correct")
    s4 = _count(evaluable, "s4_correct")

    s0_w = _count(within_team, "s0_correct")
    s1_w = _count(within_team, "s1_correct")
    s2_w = _count(within_team, "s2_correct")
    s3_w = _count(within_team, "s3_correct")
    s4_w = _count(within_team, "s4_correct")

    best_traj = max(s2, s3, s4)
    lift_over_s1 = best_traj - s1
    worth = lift_over_s1 >= 3

    verdict = (
        f"SHIP-A4-design (lift={lift_over_s1}, threshold ≥ 3)"
        if worth
        else f"NO-SHIP-A4 (lift={lift_over_s1}, threshold ≥ 3)"
    )
    if n < 8:
        verdict = f"NEED-MORE-DATA (only {n} evaluable cases)"

    return {
        "n_total": len(results),
        "n_evaluable": n,
        "n_within_team": nw,
        "S0_correct": s0,
        "S1_correct": s1,
        "S2_correct": s2,
        "S3_correct": s3,
        "S4_correct": s4,
        "within_team": {
            "S0": s0_w, "S1": s1_w, "S2": s2_w, "S3": s3_w, "S4": s4_w,
        },
        "best_trajectory": best_traj,
        "lift_over_S1": lift_over_s1,
        "verdict": verdict,
    }


def _write_markdown(results: list[CaseResult], agg: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Probe B — Sequence-Aware Role Attribution (2026-05-14)")
    lines.append("")
    lines.append("Test 3 hypotheses on whether pre-contact ball-trajectory signals")
    lines.append("can disambiguate the within-team toucher better than ball-position-")
    lines.append("at-contact (which today's pipeline + A1/A2 all use).")
    lines.append("")
    lines.append("## Strategies")
    lines.append("")
    lines.append("| S | Name | Description |")
    lines.append("|---|------|-------------|")
    lines.append("| S0 | baseline | pipeline pick (current production) |")
    lines.append("| S1 | ball-pos | argmin dist(bbox_center, ball@contact) |")
    lines.append("| S2 | traj-end | extrapolate ball traj over K=10; project to contact |")
    lines.append("| S3 | traj-int | mean(dist(bbox_center, ball)) over [f-K, f-1] |")
    lines.append("| S4 | anti-self | S3 excluding prev_toucher unless prev=BLOCK |")
    lines.append("")
    lines.append("## Aggregate")
    lines.append("")
    lines.append(f"- Cases run: **{agg['n_total']}**")
    lines.append(f"- GT-evaluable: **{agg['n_evaluable']}**")
    lines.append("")
    lines.append("| Strategy | Correct (of evaluable) |")
    lines.append("|----------|------------------------|")
    lines.append(f"| S0 (baseline) | {agg['S0_correct']} / {agg['n_evaluable']} |")
    lines.append(f"| S1 (ball-pos) | {agg['S1_correct']} / {agg['n_evaluable']} |")
    lines.append(f"| S2 (traj-end) | {agg['S2_correct']} / {agg['n_evaluable']} |")
    lines.append(f"| S3 (traj-int) | {agg['S3_correct']} / {agg['n_evaluable']} |")
    lines.append(f"| S4 (anti-self) | {agg['S4_correct']} / {agg['n_evaluable']} |")
    lines.append("")
    lines.append(f"**Best trajectory strategy:** {agg['best_trajectory']}; lift over S1 = "
                 f"`{agg['lift_over_S1']}`")
    lines.append("")
    lines.append(f"### Within-team subset ({agg['n_within_team']} of {agg['n_evaluable']})")
    lines.append("")
    lines.append("Within-team subset = GT track is in the same-team candidate set,")
    lines.append("so within-team disambiguation is in-scope. Cross-team errors (e.g. #3")
    lines.append("F3-keke) are unreachable by this probe's same-team-only search.")
    lines.append("")
    lines.append("| Strategy | Correct (within-team) |")
    lines.append("|----------|------------------------|")
    lines.append(f"| S0 | {agg['within_team']['S0']} / {agg['n_within_team']} |")
    lines.append(f"| S1 | {agg['within_team']['S1']} / {agg['n_within_team']} |")
    lines.append(f"| S2 | {agg['within_team']['S2']} / {agg['n_within_team']} |")
    lines.append(f"| S3 | {agg['within_team']['S3']} / {agg['n_within_team']} |")
    lines.append(f"| S4 | {agg['within_team']['S4']} / {agg['n_within_team']} |")
    lines.append("")
    lines.append(f"### Verdict: **{agg['verdict']}**")
    lines.append("")
    lines.append("## Per-case detail")
    lines.append("")
    lines.append("| # | label | video/rally | f | type | GT | S0 | S1 | S2 | S3 | S4 | n_pre |")
    lines.append("|---|-------|-------------|---|------|----|----|----|----|----|----|-------|")
    for r in results:
        if r.skipped:
            lines.append(f"| {r.idx} | {r.label} | {r.video}/{r.rally_short} | "
                         f"{r.pl_frame} | {r.action_type} | — | — | — | — | — | — | SKIP: {r.skip_reason} |")
            continue
        def fmt(pick: int | None, correct: bool | None) -> str:
            if pick is None: return "—"
            mark = "✓" if correct else ("✗" if correct is False else "?")
            return f"p{pick}{mark}"
        lines.append(
            f"| {r.idx} | {r.label} | {r.video}/{r.rally_short} | {r.pl_frame} | "
            f"{r.action_type} | p{r.gt_tid} | "
            f"{fmt(r.s0_pick, r.s0_correct)} | {fmt(r.s1_pick, r.s1_correct)} | "
            f"{fmt(r.s2_pick, r.s2_correct)} | {fmt(r.s3_pick, r.s3_correct)} | "
            f"{fmt(r.s4_pick, r.s4_correct)} | {r.n_pre_ball} |"
        )
    lines.append("")
    lines.append("## Per-case detail (verbose)")
    lines.append("")
    for r in results:
        lines.append(f"### #{r.idx} {r.label} — {r.video}/{r.rally_short} {r.action_type} f={r.pl_frame}")
        lines.append("")
        lines.append(f"- note: {r.note}")
        if r.skipped:
            lines.append(f"- **SKIPPED**: {r.skip_reason}")
            lines.append("")
            continue
        lines.append(f"- GT resolved_track_id: `p{r.gt_tid}`")
        lines.append(f"- pipeline (S0): `p{r.s0_pick}` correct={r.s0_correct}")
        lines.append(f"- S1 ball-pos: `p{r.s1_pick}` correct={r.s1_correct}")
        lines.append(f"- S2 traj-end: `p{r.s2_pick}` correct={r.s2_correct} endpoint={r.s2_endpoint}")
        lines.append(f"- S3 traj-int: `p{r.s3_pick}` correct={r.s3_correct} integrals={r.s3_integrals}")
        lines.append(f"- S4 anti-self: `p{r.s4_pick}` correct={r.s4_correct} "
                     f"prev_toucher=p{r.prev_toucher_tid} prev_action={r.prev_action_type}")
        lines.append(f"- same_team_cands: {r.same_team_cands}")
        lines.append(f"- n_pre_ball: {r.n_pre_ball}")
        if r.visual_path:
            lines.append(f"- visual: `{r.visual_path}`")
        lines.append("")

    MD_PATH.write_text("\n".join(lines))
    print(f"Wrote markdown: {MD_PATH}")


_HTML_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><title>Probe B — 2026-05-14</title>
<style>
  body { background:#0f0f10; color:#eaeaea; font-family:-apple-system,sans-serif; margin:0; padding:24px; }
  h1 { margin:0 0 16px 0; }
  .agg { background:#1a1a1c; border:1px solid #2a2a2e; border-radius:8px; padding:16px 20px; margin-bottom:18px; }
  table { border-collapse:collapse; margin-top:10px; }
  td, th { border:1px solid #2a2a2e; padding:6px 10px; font-size:13px; }
  th { background:#222; }
  .case { background:#1a1a1c; border:1px solid #2a2a2e; border-radius:8px; padding:14px 18px; margin-bottom:14px; }
  .case h3 { margin:0 0 8px 0; }
  .case img { max-width:100%; max-height:75vh; display:block; margin:10px 0; cursor:zoom-in; background:#000; }
  .ok { color:#5fe49d; } .bad { color:#f08680; } .skip { color:#aaa; }
  pre { background:#0a0a0a; padding:8px 12px; border-radius:6px; overflow:auto; font-size:12px; }
  #zoom-overlay { display:none; position:fixed; inset:0; background:rgba(0,0,0,.94); z-index:99; align-items:center; justify-content:center; cursor:zoom-out; }
  #zoom-overlay.open { display:flex; }
  #zoom-overlay img { max-width:96vw; max-height:96vh; }
</style></head><body>
<h1>Probe B — Sequence-Aware Attribution (2026-05-14)</h1>
<div class="agg" id="agg"></div>
<div id="cases"></div>
<div id="zoom-overlay"><img id="zoom-img"></div>
<script id="data" type="application/json">__DATA_JSON__</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
function agg() {
  const a = DATA.agg;
  return `
    <h2 style="margin:0 0 10px">Aggregate</h2>
    <table>
      <tr><th>Metric</th><th>Value</th></tr>
      <tr><td>Cases</td><td>${a.n_total}</td></tr>
      <tr><td>Evaluable</td><td>${a.n_evaluable}</td></tr>
      <tr><td>S0 baseline</td><td>${a.S0_correct} / ${a.n_evaluable}</td></tr>
      <tr><td>S1 ball-pos</td><td>${a.S1_correct} / ${a.n_evaluable}</td></tr>
      <tr><td>S2 traj-end</td><td>${a.S2_correct} / ${a.n_evaluable}</td></tr>
      <tr><td>S3 traj-int</td><td>${a.S3_correct} / ${a.n_evaluable}</td></tr>
      <tr><td>S4 anti-self</td><td>${a.S4_correct} / ${a.n_evaluable}</td></tr>
      <tr><td>Best traj</td><td>${a.best_trajectory}</td></tr>
      <tr><td>Lift over S1</td><td>${a.lift_over_S1}</td></tr>
      <tr><td>Verdict</td><td><b>${a.verdict}</b></td></tr>
    </table>`;
}
function fmt(pick, correct) {
  if (pick === null || pick === undefined) return '—';
  const cls = correct === true ? 'ok' : (correct === false ? 'bad' : '');
  const m = correct === true ? '✓' : (correct === false ? '✗' : '?');
  return `<span class="${cls}">p${pick}${m}</span>`;
}
function caseHtml(r) {
  if (r.skipped) {
    return `<div class="case"><h3>#${r.idx} ${r.label} — ${r.video}/${r.rally_short} ${r.action_type} f=${r.pl_frame}</h3><div class="skip">SKIPPED: ${r.skip_reason}</div></div>`;
  }
  const img = r.visual_path ? `<img src="${r.visual_path}" onclick="zoom(this.src)">` : '';
  return `
    <div class="case">
      <h3>#${r.idx} ${r.label} — ${r.video}/${r.rally_short} ${r.action_type} f=${r.pl_frame}</h3>
      <p><b>GT:</b> p${r.gt_tid} | S0=${fmt(r.s0_pick, r.s0_correct)} | S1=${fmt(r.s1_pick, r.s1_correct)} | S2=${fmt(r.s2_pick, r.s2_correct)} | S3=${fmt(r.s3_pick, r.s3_correct)} | S4=${fmt(r.s4_pick, r.s4_correct)}</p>
      <p><i>${r.note}</i></p>
      ${img}
      <pre>same_team_cands: ${JSON.stringify(r.same_team_cands)}
s2_endpoint: ${JSON.stringify(r.s2_endpoint)}
s3_integrals: ${JSON.stringify(r.s3_integrals)}
prev_toucher: p${r.prev_toucher_tid} (${r.prev_action_type})  n_pre_ball: ${r.n_pre_ball}</pre>
    </div>`;
}
document.getElementById('agg').innerHTML = agg();
document.getElementById('cases').innerHTML = DATA.cases.map(caseHtml).join('');
function zoom(s){const o=document.getElementById('zoom-overlay');document.getElementById('zoom-img').src=s;o.classList.add('open');}
document.getElementById('zoom-overlay').addEventListener('click', () => document.getElementById('zoom-overlay').classList.remove('open'));
</script></body></html>
"""


def _write_html(results: list[CaseResult], agg: dict[str, Any]) -> None:
    payload = {
        "agg": agg,
        "cases": [asdict(r) for r in results],
    }
    html = _HTML_TEMPLATE.replace("__DATA_JSON__", json.dumps(payload, indent=2))
    HTML_PATH.write_text(html)
    print(f"Wrote HTML: {HTML_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Probe B — Sequence-Aware Attribution ===")
    print(f"Report dir: {REPORT_DIR}")
    print(f"K_PRE={K_PRE}, GT_FRAME_TOL={GT_FRAME_TOL}, MIN_BALL_PRE={MIN_BALL_PRE}")
    print()

    print(f"Hydrating {len(PROBE_CASES)} probe cases...")
    cases = _fetch_cases()
    print(f"Hydrated {len(cases)} (skipped={sum(1 for c in cases if c.skipped)})")
    print()

    resolver = VideoResolver()
    print(f"VideoResolver endpoint={resolver.s3_endpoint} bucket={resolver.bucket_name}")
    print()

    results: list[CaseResult] = []
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] #{case.idx} {case.label} {case.video_name}/"
              f"{case.rally_short} {case.action_type} f={case.pl_frame}")
        if case.skipped:
            print(f"    SKIP: {case.skip_reason}")
        r = _evaluate(case)
        if r.skipped:
            print(f"    EVAL SKIP: {r.skip_reason}")
            results.append(r)
            continue
        print(f"    GT=p{case.gt_tid} S0=p{r.s0_pick}{'✓' if r.s0_correct else '✗'} "
              f"S1=p{r.s1_pick}{'✓' if r.s1_correct else ('✗' if r.s1_correct is False else '?')} "
              f"S2=p{r.s2_pick}{'✓' if r.s2_correct else ('✗' if r.s2_correct is False else '?')} "
              f"S3=p{r.s3_pick}{'✓' if r.s3_correct else ('✗' if r.s3_correct is False else '?')} "
              f"S4=p{r.s4_pick}{'✓' if r.s4_correct else ('✗' if r.s4_correct is False else '?')}")
        print(f"    s3_integrals={r.s3_integrals}")
        print(f"    s2_endpoint={r.s2_endpoint}  n_pre={r.n_pre_ball}")

        # Try to write annotated frame
        vm = _fetch_video_meta(case.video_id)
        video_path = _resolve_video(resolver, vm) if vm else None
        if video_path:
            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                vid_fps = cap.get(cv2.CAP_PROP_FPS) or vm.get("fps", 30.0)
                img_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                img_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                rally_start_frame = int(round(case.rally_start_ms / 1000.0 * vid_fps))
                source_frame = rally_start_frame + case.pl_frame
                vpath = _annotate(case=case, result=r, video_path=video_path,
                                   source_frame=source_frame,
                                   img_w=img_w, img_h=img_h)
                if vpath:
                    r.visual_path = str(vpath.relative_to(REPORT_DIR))
                    print(f"    wrote: {vpath.name}")
        results.append(r)

    agg = _aggregate(results)
    print()
    print("=== Aggregate ===")
    print(json.dumps(agg, indent=2))

    JSON_PATH.write_text(json.dumps({
        "agg": agg,
        "cases": [asdict(r) for r in results],
    }, indent=2))
    print(f"\nWrote JSON: {JSON_PATH}")

    _write_markdown(results, agg)
    _write_html(results, agg)
    print(f"\nOpen visual: open {HTML_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
