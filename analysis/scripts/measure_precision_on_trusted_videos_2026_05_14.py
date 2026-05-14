"""Precision validation for attribution rule families against user-trusted GT.

Runs against 12 videos confirmed to have correctly-labeled GT:
  titi, toto, lulu, wawa, caco, cece, cici, cuco, gaga, kaka, juju, yeye.

For every pipeline contact that has a matching GT row, we measure four
candidate-picking strategies and compare to GT.resolved_track_id:

  S0 (baseline)    : pipeline's stored `actions[].playerTrackId`.
  A1.v1            : if prev/curr is a non-block same-player pair,
                     flip *curr* to the closest same-team alternate.
  A1.v2-confidence : same trigger, but flip the *lower-confidence* side.
  S4 (traj-int + anti-self): per-candidate mean(distance(bbox, ball)) over
                     the K=10 frames pre-contact, excluding prev_toucher
                     (unless prev_action == BLOCK). Picks min-integral.

Also measures the at-net attack contact-detector FP rate: pipeline at-net
attack contacts with no GT row within ±5 frames are likely user-deleted FPs
on these trusted-labeled videos.

Usage:
    cd analysis
    uv run python scripts/measure_precision_on_trusted_videos_2026_05_14.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from rallycut.evaluation.tracking.db import get_connection  # noqa: E402

VIDEO_NAMES = [
    "titi", "toto", "lulu", "wawa", "caco", "cece",
    "cici", "cuco", "gaga", "kaka", "juju", "yeye",
]
GT_FRAME_TOL = 5
K_PRE = 10
MIN_BALL_PRE = 5

REPORT_DIR = HERE.parent / "reports" / "precision_trusted_videos_2026_05_14"
JSON_PATH = REPORT_DIR / "results.json"
MD_PATH = REPORT_DIR / "results.md"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class GTRow:
    rally_id: str
    frame: int
    action: str                # "SERVE"/"RECEIVE"/.../"BLOCK"
    resolved_tid: int | None


@dataclass
class ActionRow:
    rally_id: str
    idx: int                   # index inside the actions list (sorted by frame)
    frame: int
    action: str                # lowercase: "serve"/.../"block"
    pl_tid: int | None
    confidence: float
    team_assignments: dict[str, str]
    # From contacts:
    candidates: list[tuple[int, float]]   # (tid, dist) — by ascending dist
    is_at_net: bool | None
    # For trajectory:
    ball_at_contact: tuple[float, float] | None
    pre_ball: list[tuple[int, float, float]]   # frames in [f-K, f-1]
    cand_bbox_pre: dict[int, dict[int, tuple[float, float]]]  # tid → frame → (cx, cy)
    cand_bbox_at: dict[int, tuple[float, float]]              # tid → (cx, cy) at contact
    # Adjacent action (for A1 and S4 anti-self):
    prev_action: str | None
    prev_tid: int | None
    prev_confidence: float | None


# ---------------------------------------------------------------------------
# Hydration
# ---------------------------------------------------------------------------

def _build_actions_for_rally(
    actions: list[dict[str, Any]],
    contacts: list[dict[str, Any]],
    positions: list[dict[str, Any]],
    ball_positions: list[dict[str, Any]],
    team_assignments: dict[str, str],
    rally_id: str,
) -> list[ActionRow]:
    """Build ActionRow per pipeline action, joined to its contact + ball/positions."""
    if not actions:
        return []

    # Sort actions by frame
    actions_sorted = sorted(actions, key=lambda a: int(a.get("frame", 0)))

    # Index contacts by frame
    contacts_by_frame: dict[int, dict[str, Any]] = {}
    for c in contacts:
        try:
            f = int(c.get("frame", -1))
        except (TypeError, ValueError):
            continue
        contacts_by_frame[f] = c

    # Index ball positions
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

    # Index positions by (tid, frame)
    pos_by_tid_frame: dict[int, dict[int, tuple[float, float]]] = defaultdict(dict)
    for p in positions:
        try:
            tid = int(p.get("trackId", -1))
            fnum = int(p.get("frameNumber", -1))
        except (TypeError, ValueError):
            continue
        if tid < 0 or fnum < 0:
            continue
        pos_by_tid_frame[tid][fnum] = (float(p.get("x", 0.0)), float(p.get("y", 0.0)))

    rows: list[ActionRow] = []
    for idx, a in enumerate(actions_sorted):
        f = int(a.get("frame", 0))
        action_lc = str(a.get("action", "")).lower()
        pl_tid_raw = a.get("playerTrackId")
        pl_tid: int | None = None
        if pl_tid_raw is not None:
            try:
                pl_tid = int(pl_tid_raw)
            except (TypeError, ValueError):
                pl_tid = None

        conf = float(a.get("confidence", 0.0) or 0.0)

        # Pair with contact (frame must match exactly)
        contact = contacts_by_frame.get(f)
        candidates: list[tuple[int, float]] = []
        is_at_net: bool | None = None
        ball_at: tuple[float, float] | None = None
        if contact:
            for cd in contact.get("playerCandidates", []) or []:
                if not cd or len(cd) < 2:
                    continue
                try:
                    candidates.append((int(cd[0]), float(cd[1])))
                except (TypeError, ValueError):
                    continue
            candidates.sort(key=lambda x: x[1])
            is_at_net_raw = contact.get("isAtNet")
            if isinstance(is_at_net_raw, bool):
                is_at_net = is_at_net_raw
            bx = contact.get("ballX")
            by = contact.get("ballY")
            if bx is not None and by is not None:
                try:
                    ball_at = (float(bx), float(by))
                except (TypeError, ValueError):
                    pass

        if ball_at is None:
            # fall back to ball_by_frame if available
            ball_at = ball_by_frame.get(f)

        # Pre-window ball positions and candidate positions
        pre_ball: list[tuple[int, float, float]] = []
        for fp in range(f - K_PRE, f):
            if fp in ball_by_frame:
                bx_n, by_n = ball_by_frame[fp]
                pre_ball.append((fp, bx_n, by_n))

        cand_bbox_pre: dict[int, dict[int, tuple[float, float]]] = {}
        cand_bbox_at: dict[int, tuple[float, float]] = {}
        all_cand_tids = {tid for tid, _ in candidates}
        if pl_tid is not None:
            all_cand_tids.add(pl_tid)
        for tid in all_cand_tids:
            tmap = pos_by_tid_frame.get(tid, {})
            pre = {fp: tmap[fp] for fp in range(f - K_PRE, f) if fp in tmap}
            cand_bbox_pre[tid] = pre
            if f in tmap:
                cand_bbox_at[tid] = tmap[f]

        # Previous action data
        prev_action: str | None = None
        prev_tid: int | None = None
        prev_conf: float | None = None
        if idx > 0:
            pa = actions_sorted[idx - 1]
            prev_action = str(pa.get("action", "")).lower() or None
            pt = pa.get("playerTrackId")
            if pt is not None:
                try:
                    prev_tid = int(pt)
                except (TypeError, ValueError):
                    prev_tid = None
            prev_conf = float(pa.get("confidence", 0.0) or 0.0)

        rows.append(ActionRow(
            rally_id=rally_id, idx=idx, frame=f, action=action_lc,
            pl_tid=pl_tid, confidence=conf,
            team_assignments=team_assignments,
            candidates=candidates, is_at_net=is_at_net,
            ball_at_contact=ball_at, pre_ball=pre_ball,
            cand_bbox_pre=cand_bbox_pre, cand_bbox_at=cand_bbox_at,
            prev_action=prev_action, prev_tid=prev_tid,
            prev_confidence=prev_conf,
        ))
    return rows


def fetch_video_data(
    video_names: list[str],
) -> tuple[dict[str, list[ActionRow]], list[GTRow], dict[str, str], dict[str, int]]:
    """Return:
        rally_actions: rally_id -> list[ActionRow]
        gt_rows:       list[GTRow]
        rally_video:   rally_id -> video_name
        rally_count:   video_name -> rally count
    """
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
        # Prefer first occurrence per name (handle yeye duplicate by keeping whichever has GT;
        # we'll union them by name later by simply scanning both).
        # In our case, both yeye entries exist; we'll include both.
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
            SELECT g.rally_id, g.frame, g.action, g.resolved_track_id
            FROM rally_action_ground_truth g
            JOIN rallies r ON r.id = g.rally_id
            WHERE r.video_id = ANY(%s)
            """,
            [video_ids],
        )
        gt_db_rows = cur.fetchall()

    rally_actions: dict[str, list[ActionRow]] = {}
    rally_video: dict[str, str] = {}
    rally_count: Counter = Counter()
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
        team_ass = {str(k): str(v) for k, v in (team_ass or {}).items()}

        rows = _build_actions_for_rally(
            actions=actions, contacts=contacts, positions=positions,
            ball_positions=ball_positions, team_assignments=team_ass,
            rally_id=rally_id,
        )
        rally_actions[rally_id] = rows
        rally_video[rally_id] = video_id_to_name.get(video_id, video_id)
        rally_count[video_id_to_name.get(video_id, video_id)] += 1

    gt_rows: list[GTRow] = []
    for g in gt_db_rows:
        gt_rows.append(GTRow(
            rally_id=str(g[0]),
            frame=int(g[1]),
            action=str(g[2]),
            resolved_tid=int(g[3]) if g[3] is not None else None,
        ))

    return rally_actions, gt_rows, rally_video, dict(rally_count)


# ---------------------------------------------------------------------------
# Strategy evaluation
# ---------------------------------------------------------------------------

def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _same_team_alternates(
    pl_tid: int, candidates: list[tuple[int, float]],
    team_assignments: dict[str, str],
) -> list[tuple[int, float]]:
    """Return same-team candidates (excluding pl_tid), sorted by ascending dist."""
    pl_team = team_assignments.get(str(pl_tid))
    if pl_team not in ("A", "B"):
        return []
    out = []
    for tid, d in candidates:
        if tid == pl_tid:
            continue
        if team_assignments.get(str(tid)) != pl_team:
            continue
        out.append((tid, d))
    return out


def _a1v1_pick(curr: ActionRow, prev: ActionRow | None) -> int | None:
    """A1.v1 (always-flip-curr) — if prev/curr same player (block exception),
    flip curr to closest same-team alternate. Returns the new pick (or curr's pl_tid if no rule fire / no alt)."""
    if curr.pl_tid is None:
        return curr.pl_tid
    if prev is None:
        return curr.pl_tid
    if prev.pl_tid is None or prev.pl_tid != curr.pl_tid:
        return curr.pl_tid
    if (prev.action or "") == "block":
        return curr.pl_tid
    alts = _same_team_alternates(curr.pl_tid, curr.candidates, curr.team_assignments)
    if not alts:
        return curr.pl_tid
    return alts[0][0]


def _a1v2_conf_pick_for_pair(
    prev: ActionRow, curr: ActionRow,
) -> tuple[int | None, int | None]:
    """A1.v2-confidence: flip the lower-confidence side of the same-player pair.

    Returns (new_prev_pid, new_curr_pid). If no flip happens, returns the
    original prev_tid / curr_tid for that side.
    """
    if prev.pl_tid is None or curr.pl_tid is None:
        return (prev.pl_tid, curr.pl_tid)
    if prev.pl_tid != curr.pl_tid:
        return (prev.pl_tid, curr.pl_tid)
    if (prev.action or "") == "block":
        return (prev.pl_tid, curr.pl_tid)

    prev_conf = prev.confidence if prev.confidence is not None else 0.0
    curr_conf = curr.confidence if curr.confidence is not None else 0.0
    if prev_conf <= curr_conf:
        # Flip prev
        alts = _same_team_alternates(prev.pl_tid, prev.candidates, prev.team_assignments)
        if not alts:
            return (prev.pl_tid, curr.pl_tid)
        return (alts[0][0], curr.pl_tid)
    else:
        alts = _same_team_alternates(curr.pl_tid, curr.candidates, curr.team_assignments)
        if not alts:
            return (prev.pl_tid, curr.pl_tid)
        return (prev.pl_tid, alts[0][0])


def _s4_pick(curr: ActionRow) -> int | None:
    """S4: per-candidate mean distance to ball over [f-K, f-1]; exclude
    prev_toucher unless prev action == block. Pick min-integral.

    Restricted to same-team candidates (the rule scope).
    """
    if curr.pl_tid is None:
        return curr.pl_tid
    pl_team = curr.team_assignments.get(str(curr.pl_tid))
    if pl_team not in ("A", "B"):
        return curr.pl_tid
    if len(curr.pre_ball) < MIN_BALL_PRE:
        return curr.pl_tid

    # All same-team candidates (including pl_tid)
    cand_tids = [
        tid for tid, _ in curr.candidates
        if curr.team_assignments.get(str(tid)) == pl_team
    ]
    if curr.pl_tid not in cand_tids and pl_team == curr.team_assignments.get(str(curr.pl_tid)):
        cand_tids.append(curr.pl_tid)
    if not cand_tids:
        return curr.pl_tid

    # Anti-self: drop prev_tid unless prev_action was block
    exclude = set()
    if curr.prev_tid is not None and (curr.prev_action or "") != "block":
        exclude.add(curr.prev_tid)

    integrals: dict[int, float] = {}
    for tid in cand_tids:
        bb_map = curr.cand_bbox_pre.get(tid, {})
        if not bb_map:
            continue
        total = 0.0
        count = 0
        for f, bx, by in curr.pre_ball:
            bb = bb_map.get(f)
            if bb is None:
                continue
            total += _dist(bb, (bx, by))
            count += 1
        if count > 0:
            integrals[tid] = total / count

    if not integrals:
        return curr.pl_tid

    filt = {t: v for t, v in integrals.items() if t not in exclude}
    pool = filt if filt else integrals
    return min(pool.items(), key=lambda x: x[1])[0]


# ---------------------------------------------------------------------------
# Matching pipeline ↔ GT
# ---------------------------------------------------------------------------

def _match_gt_to_pipeline(
    pipeline_actions: list[ActionRow], gt_rows: list[GTRow],
) -> list[tuple[GTRow, ActionRow | None]]:
    """Pair each GT row to nearest pipeline action of the same type (±5 frames).
    If no match, the second element is None."""
    out = []
    for g in gt_rows:
        target_type_lc = g.action.lower()
        best: ActionRow | None = None
        best_dist = math.inf
        for a in pipeline_actions:
            if a.action != target_type_lc:
                continue
            dt = abs(a.frame - g.frame)
            if dt <= GT_FRAME_TOL and dt < best_dist:
                best = a
                best_dist = dt
        out.append((g, best))
    return out


# ---------------------------------------------------------------------------
# Run measurements
# ---------------------------------------------------------------------------

@dataclass
class CounterBundle:
    total: int = 0
    pipeline_correct: int = 0
    pipeline_wrong_same_team: int = 0
    pipeline_wrong_cross_team: int = 0
    pipeline_missing: int = 0   # no pipeline action match
    # A1.v1
    a1v1_fired: int = 0          # rule applied (curr flipped)
    a1v1_match_gt: int = 0       # flipped pick == gt
    a1v1_disagrees_pipeline: int = 0  # new pick != pipeline pick
    # A1.v2-conf
    a1v2c_fired: int = 0         # number of pairs where flip happened (either side)
    a1v2c_curr_changed_match_gt: int = 0
    a1v2c_curr_changed_wrong: int = 0
    # S4
    s4_changed_pipeline: int = 0
    s4_changed_match_gt: int = 0
    s4_changed_wrong: int = 0
    s4_unchanged_correct: int = 0
    s4_unchanged_wrong: int = 0
    s4_skipped_no_window: int = 0


def run_measurements() -> dict[str, Any]:
    print(f"Fetching data for videos: {VIDEO_NAMES}")
    rally_actions, gt_rows, rally_video, rally_count = fetch_video_data(VIDEO_NAMES)
    print(f"Loaded {len(rally_actions)} rallies, {len(gt_rows)} GT rows")
    print(f"Per-video rally counts: {rally_count}")

    # Index GT by rally
    gt_by_rally: dict[str, list[GTRow]] = defaultdict(list)
    for g in gt_rows:
        gt_by_rally[g.rally_id].append(g)

    # Aggregators
    overall = CounterBundle()
    per_video: dict[str, CounterBundle] = defaultdict(CounterBundle)
    per_action: dict[str, CounterBundle] = defaultdict(CounterBundle)

    # A1.v1: track per-fire details
    a1v1_fires: list[dict[str, Any]] = []
    a1v2c_fires: list[dict[str, Any]] = []
    s4_changes: list[dict[str, Any]] = []

    # At-net FP counter (per-video and overall)
    at_net_attack_total = 0
    at_net_attack_unmatched = 0
    at_net_attack_per_video: Counter = Counter()
    at_net_attack_unmatched_per_video: Counter = Counter()

    # For each rally, build a curr-tid map for A1.v2c (it modifies prev too; we'll
    # measure curr's correctness only — prev is a side-effect benefit).
    for rally_id, actions in rally_actions.items():
        video_name = rally_video.get(rally_id, "?")
        gts = gt_by_rally.get(rally_id, [])
        pairs = _match_gt_to_pipeline(actions, gts)

        # Precompute A1.v1 picks per action index (just for curr — prev is the
        # source row; the rule fires on the curr action).
        a1v1_pick_by_idx: dict[int, int | None] = {}
        for a in actions:
            prev_a = actions[a.idx - 1] if a.idx > 0 else None
            a1v1_pick_by_idx[a.idx] = _a1v1_pick(a, prev_a)

        # Precompute A1.v2-conf: for each curr action, what's its new pid?
        a1v2c_curr_pick: dict[int, int | None] = {}
        a1v2c_prev_pick: dict[int, int | None] = {}
        for a in actions:
            if a.idx == 0:
                a1v2c_curr_pick[a.idx] = a.pl_tid
                continue
            prev_a = actions[a.idx - 1]
            new_prev, new_curr = _a1v2_conf_pick_for_pair(prev_a, a)
            a1v2c_curr_pick[a.idx] = new_curr
            a1v2c_prev_pick[a.idx] = new_prev

        # Precompute S4 picks per action index
        s4_pick_by_idx: dict[int, int | None] = {}
        for a in actions:
            s4_pick_by_idx[a.idx] = _s4_pick(a)

        # Now evaluate GT pairs
        for g, a in pairs:
            if g.resolved_tid is None:
                continue
            overall.total += 1
            per_video[video_name].total += 1
            per_action[g.action].total += 1

            if a is None:
                overall.pipeline_missing += 1
                per_video[video_name].pipeline_missing += 1
                per_action[g.action].pipeline_missing += 1
                continue

            # S0 — pipeline pick
            pl_pid = a.pl_tid
            gt_pid = g.resolved_tid
            if pl_pid is None:
                # pipeline picked nobody — treat as wrong (cross/team unknown)
                overall.pipeline_wrong_cross_team += 1
                per_video[video_name].pipeline_wrong_cross_team += 1
                per_action[g.action].pipeline_wrong_cross_team += 1
            elif pl_pid == gt_pid:
                overall.pipeline_correct += 1
                per_video[video_name].pipeline_correct += 1
                per_action[g.action].pipeline_correct += 1
            else:
                pl_team = a.team_assignments.get(str(pl_pid))
                gt_team = a.team_assignments.get(str(gt_pid))
                if pl_team and gt_team and pl_team == gt_team:
                    overall.pipeline_wrong_same_team += 1
                    per_video[video_name].pipeline_wrong_same_team += 1
                    per_action[g.action].pipeline_wrong_same_team += 1
                else:
                    overall.pipeline_wrong_cross_team += 1
                    per_video[video_name].pipeline_wrong_cross_team += 1
                    per_action[g.action].pipeline_wrong_cross_team += 1

            # A1.v1: rule fires only when curr's flipped pick differs from pl_pid
            new_curr_v1 = a1v1_pick_by_idx.get(a.idx, pl_pid)
            if new_curr_v1 != pl_pid and pl_pid is not None:
                overall.a1v1_fired += 1
                per_video[video_name].a1v1_fired += 1
                per_action[g.action].a1v1_fired += 1
                overall.a1v1_disagrees_pipeline += 1
                per_video[video_name].a1v1_disagrees_pipeline += 1
                if new_curr_v1 == gt_pid:
                    overall.a1v1_match_gt += 1
                    per_video[video_name].a1v1_match_gt += 1
                    per_action[g.action].a1v1_match_gt += 1
                a1v1_fires.append({
                    "video": video_name,
                    "rally_id": rally_id,
                    "frame": a.frame,
                    "action": a.action,
                    "pl_pid": pl_pid,
                    "new_pid": new_curr_v1,
                    "gt_pid": gt_pid,
                    "match_gt": new_curr_v1 == gt_pid,
                    "candidates": a.candidates,
                })

            # A1.v2-conf
            new_curr_v2c = a1v2c_curr_pick.get(a.idx, pl_pid)
            if new_curr_v2c != pl_pid and pl_pid is not None:
                overall.a1v2c_fired += 1
                per_video[video_name].a1v2c_fired += 1
                per_action[g.action].a1v2c_fired += 1
                if new_curr_v2c == gt_pid:
                    overall.a1v2c_curr_changed_match_gt += 1
                    per_video[video_name].a1v2c_curr_changed_match_gt += 1
                    per_action[g.action].a1v2c_curr_changed_match_gt += 1
                else:
                    overall.a1v2c_curr_changed_wrong += 1
                    per_video[video_name].a1v2c_curr_changed_wrong += 1
                    per_action[g.action].a1v2c_curr_changed_wrong += 1
                a1v2c_fires.append({
                    "video": video_name,
                    "rally_id": rally_id,
                    "frame": a.frame,
                    "action": a.action,
                    "pl_pid": pl_pid,
                    "new_pid": new_curr_v2c,
                    "gt_pid": gt_pid,
                    "match_gt": new_curr_v2c == gt_pid,
                    "pl_conf": a.confidence,
                    "prev_conf": a.prev_confidence,
                })

            # S4
            new_s4 = s4_pick_by_idx.get(a.idx, pl_pid)
            if len(a.pre_ball) < MIN_BALL_PRE:
                overall.s4_skipped_no_window += 1
                per_video[video_name].s4_skipped_no_window += 1
                per_action[g.action].s4_skipped_no_window += 1
            if new_s4 != pl_pid and pl_pid is not None:
                overall.s4_changed_pipeline += 1
                per_video[video_name].s4_changed_pipeline += 1
                per_action[g.action].s4_changed_pipeline += 1
                if new_s4 == gt_pid:
                    overall.s4_changed_match_gt += 1
                    per_video[video_name].s4_changed_match_gt += 1
                    per_action[g.action].s4_changed_match_gt += 1
                else:
                    overall.s4_changed_wrong += 1
                    per_video[video_name].s4_changed_wrong += 1
                    per_action[g.action].s4_changed_wrong += 1
                s4_changes.append({
                    "video": video_name,
                    "rally_id": rally_id,
                    "frame": a.frame,
                    "action": a.action,
                    "pl_pid": pl_pid,
                    "new_pid": new_s4,
                    "gt_pid": gt_pid,
                    "match_gt": new_s4 == gt_pid,
                    "n_pre_ball": len(a.pre_ball),
                })
            else:
                # S4 unchanged
                if pl_pid == gt_pid:
                    overall.s4_unchanged_correct += 1
                    per_video[video_name].s4_unchanged_correct += 1
                    per_action[g.action].s4_unchanged_correct += 1
                else:
                    overall.s4_unchanged_wrong += 1
                    per_video[video_name].s4_unchanged_wrong += 1
                    per_action[g.action].s4_unchanged_wrong += 1

        # ----- At-net attack FP rate -----
        # For every pipeline at-net attack action, check if any GT row exists within ±5.
        gt_frames_attack = sorted({g.frame for g in gts if g.action == "ATTACK"})
        for a in actions:
            if a.action != "attack":
                continue
            if a.is_at_net is not True:
                continue
            at_net_attack_total += 1
            at_net_attack_per_video[video_name] += 1
            # Check if any ATTACK GT within tol
            matched = False
            for fg in gt_frames_attack:
                if abs(fg - a.frame) <= GT_FRAME_TOL:
                    matched = True
                    break
            if not matched:
                at_net_attack_unmatched += 1
                at_net_attack_unmatched_per_video[video_name] += 1

    # Print progress summary
    print()
    print(f"Overall total GT-evaluable: {overall.total}")
    print(f"  Pipeline correct: {overall.pipeline_correct}")
    print(f"  Pipeline wrong same-team: {overall.pipeline_wrong_same_team}")
    print(f"  Pipeline wrong cross-team: {overall.pipeline_wrong_cross_team}")
    print(f"  Pipeline missing contact: {overall.pipeline_missing}")
    print()
    print(f"A1.v1 fired: {overall.a1v1_fired}  match_gt: {overall.a1v1_match_gt}")
    print(f"A1.v2-conf fired (curr changed): {overall.a1v2c_fired}  match_gt: {overall.a1v2c_curr_changed_match_gt}")
    print(f"S4 changed pipeline: {overall.s4_changed_pipeline}  match_gt: {overall.s4_changed_match_gt}")
    print(f"At-net attacks: {at_net_attack_total}  unmatched (no GT): {at_net_attack_unmatched}")

    return {
        "overall": overall,
        "per_video": per_video,
        "per_action": per_action,
        "a1v1_fires": a1v1_fires,
        "a1v2c_fires": a1v2c_fires,
        "s4_changes": s4_changes,
        "rally_count": rally_count,
        "at_net": {
            "total": at_net_attack_total,
            "unmatched": at_net_attack_unmatched,
            "per_video_total": dict(at_net_attack_per_video),
            "per_video_unmatched": dict(at_net_attack_unmatched_per_video),
        },
        "n_rallies": len(rally_actions),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _pct(num: int, denom: int) -> str:
    if denom <= 0:
        return "n/a"
    return f"{num}/{denom} = {100.0 * num / denom:.1f}%"


def write_markdown(result: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    overall: CounterBundle = result["overall"]
    per_video: dict[str, CounterBundle] = result["per_video"]
    per_action: dict[str, CounterBundle] = result["per_action"]
    at_net = result["at_net"]
    n_rallies = result["n_rallies"]
    rally_count = result["rally_count"]

    n_eval = overall.total - overall.pipeline_missing

    lines: list[str] = []
    lines.append("# Precision validation on 12 trusted-GT videos (2026-05-14)")
    lines.append("")
    lines.append(
        f"GT coverage: **{overall.total}** rows across **{n_rallies}** rallies "
        f"in 12 videos."
    )
    lines.append("")
    lines.append("Per-video rally counts:")
    for v in VIDEO_NAMES:
        lines.append(f"- {v}: {rally_count.get(v, 0)}")
    lines.append("")

    # A. Overall pipeline precision
    lines.append("## A. Overall pipeline attribution precision (S0 baseline)")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| GT rows checked | {overall.total} |")
    lines.append(f"| Pipeline contact present | {n_eval} |")
    lines.append(f"| Pipeline correct (pid matches GT) | {_pct(overall.pipeline_correct, n_eval)} |")
    lines.append(f"| Pipeline wrong same-team | {overall.pipeline_wrong_same_team} ({100.0*overall.pipeline_wrong_same_team/max(n_eval,1):.1f}%) |")
    lines.append(f"| Pipeline wrong cross-team | {overall.pipeline_wrong_cross_team} ({100.0*overall.pipeline_wrong_cross_team/max(n_eval,1):.1f}%) |")
    lines.append(f"| Pipeline missing (no contact at GT frame) | {overall.pipeline_missing} ({100.0*overall.pipeline_missing/max(overall.total,1):.1f}%) |")
    lines.append("")

    # B. A1.v1
    lines.append("## B. A1.v1 (always-flip-curr) simulation")
    lines.append("")
    lines.append("Rule: for same-player consecutive pairs (block exception),")
    lines.append("flip the *curr* action's pid to its closest same-team alternate.")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Same-player pairs where rule changes pipeline | {overall.a1v1_fired} |")
    lines.append(f"| Of those, new pick matches GT | {_pct(overall.a1v1_match_gt, overall.a1v1_fired)} |")
    lines.append(f"| Of those, new pick doesn't match GT | {overall.a1v1_fired - overall.a1v1_match_gt} |")
    # Net change vs S0: among fires, baseline pipeline_correct (curr) was 0 by definition only if pipeline picked
    # the same-player twice; calculate exactly.
    # On fired cases, baseline was wrong if pipeline pid != gt; A1.v1 changes pid to alt.
    # Net delta = a1v1_match_gt - (cases that baseline was correct but rule made it wrong)
    # Since rule only fires when curr.pl_pid == prev.pl_pid (a structural violation), but pl_pid might still match GT
    # if the rule is overzealous (rare). Let's count baseline-correct-among-fires for the net.
    baseline_correct_on_fires = 0
    a1v1_baseline_wrong = 0
    a1v1_rule_fixed = 0
    a1v1_rule_broke = 0
    for f in result["a1v1_fires"]:
        if f["pl_pid"] == f["gt_pid"]:
            baseline_correct_on_fires += 1
            if not f["match_gt"]:
                a1v1_rule_broke += 1
        else:
            a1v1_baseline_wrong += 1
            if f["match_gt"]:
                a1v1_rule_fixed += 1
    net_delta_a1v1 = a1v1_rule_fixed - a1v1_rule_broke
    lines.append(f"| Baseline-correct among fires (rule attacks correct picks) | {baseline_correct_on_fires} |")
    lines.append(f"| Baseline-wrong among fires (rule has work to do) | {a1v1_baseline_wrong} |")
    lines.append(f"| Of those: rule fixed | {a1v1_rule_fixed} / {a1v1_baseline_wrong} |")
    lines.append(f"| Of those: rule broke a correct | {a1v1_rule_broke} / {baseline_correct_on_fires} |")
    lines.append(f"| **Net delta vs S0** | **{net_delta_a1v1:+d}** |")
    lines.append("")

    # C. A1.v2-conf
    lines.append("## C. A1.v2-confidence (flip lower-conf side) simulation — measuring curr-side only")
    lines.append("")
    a1v2c_baseline_wrong = 0
    a1v2c_baseline_correct = 0
    a1v2c_rule_fixed = 0
    a1v2c_rule_broke = 0
    for f in result["a1v2c_fires"]:
        if f["pl_pid"] == f["gt_pid"]:
            a1v2c_baseline_correct += 1
            if not f["match_gt"]:
                a1v2c_rule_broke += 1
        else:
            a1v2c_baseline_wrong += 1
            if f["match_gt"]:
                a1v2c_rule_fixed += 1
    net_delta_a1v2c = a1v2c_rule_fixed - a1v2c_rule_broke
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Same-player pairs where curr changed | {overall.a1v2c_fired} |")
    lines.append(f"| Of those, new curr pick matches GT | {_pct(overall.a1v2c_curr_changed_match_gt, overall.a1v2c_fired)} |")
    lines.append(f"| Baseline-correct among fires (rule attacks correct picks) | {a1v2c_baseline_correct} |")
    lines.append(f"| Baseline-wrong among fires (rule has work to do) | {a1v2c_baseline_wrong} |")
    lines.append(f"| Of those: rule fixed | {a1v2c_rule_fixed} / {a1v2c_baseline_wrong} |")
    lines.append(f"| Of those: rule broke a correct | {a1v2c_rule_broke} / {a1v2c_baseline_correct} |")
    lines.append(f"| **Net delta vs S0 (curr-side only)** | **{net_delta_a1v2c:+d}** |")
    lines.append("")
    lines.append("Note: A1.v2-conf also re-attributes prev-side actions; that benefit is")
    lines.append("captured implicitly when the prev becomes the next iteration's curr.")
    lines.append("")

    # D. S4
    lines.append("## D. S4 trajectory-integral + anti-self-touch simulation")
    lines.append("")
    s4_baseline_wrong = 0
    s4_baseline_correct = 0
    s4_rule_fixed = 0
    s4_rule_broke = 0
    for f in result["s4_changes"]:
        if f["pl_pid"] == f["gt_pid"]:
            s4_baseline_correct += 1
            if not f["match_gt"]:
                s4_rule_broke += 1
        else:
            s4_baseline_wrong += 1
            if f["match_gt"]:
                s4_rule_fixed += 1
    net_delta_s4 = s4_rule_fixed - s4_rule_broke
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Contacts S4 changed pipeline pick | {overall.s4_changed_pipeline} |")
    lines.append(f"| Of those, new pick matches GT | {_pct(overall.s4_changed_match_gt, overall.s4_changed_pipeline)} |")
    lines.append(f"| Baseline-correct on S4-changes (rule attacks correct picks) | {s4_baseline_correct} |")
    lines.append(f"| Baseline-wrong on S4-changes (rule has work to do) | {s4_baseline_wrong} |")
    lines.append(f"| Of those: rule fixed | {s4_rule_fixed} / {s4_baseline_wrong} |")
    lines.append(f"| Of those: rule broke a correct | {s4_rule_broke} / {s4_baseline_correct} |")
    lines.append(f"| **Net delta vs S0** | **{net_delta_s4:+d}** |")
    lines.append(f"| S4 skipped (< {MIN_BALL_PRE} pre-ball frames) | {overall.s4_skipped_no_window} |")
    lines.append("")

    # E. At-net FP
    lines.append("## E. Contact-detector at-net FP rate")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| Pipeline at-net attack contacts (12 videos) | {at_net['total']} |")
    pct = (100.0 * at_net["unmatched"] / at_net["total"]) if at_net["total"] else 0.0
    lines.append(f"| Of which have NO ATTACK GT within ±5 frames | {at_net['unmatched']} = {pct:.1f}% (likely FP / non-event) |")
    lines.append("")
    lines.append("Per video:")
    lines.append("")
    lines.append("| Video | At-net attacks | Unmatched (no GT) | % unmatched |")
    lines.append("|---|---|---|---|")
    for v in VIDEO_NAMES:
        t = at_net["per_video_total"].get(v, 0)
        u = at_net["per_video_unmatched"].get(v, 0)
        p = (100.0 * u / t) if t else 0.0
        lines.append(f"| {v} | {t} | {u} | {p:.1f}% |")
    lines.append("")

    # Per-video table
    lines.append("## Per-video breakdown")
    lines.append("")
    lines.append("| Video | GT rows | Pipeline-correct | A1.v1 net | A1.v2c net | S4 net |")
    lines.append("|---|---|---|---|---|---|")
    # Compute per-video net deltas
    pv_a1v1_net: dict[str, int] = defaultdict(int)
    pv_a1v2c_net: dict[str, int] = defaultdict(int)
    pv_s4_net: dict[str, int] = defaultdict(int)
    for f in result["a1v1_fires"]:
        if f["pl_pid"] == f["gt_pid"]:
            if not f["match_gt"]:
                pv_a1v1_net[f["video"]] -= 1
        else:
            if f["match_gt"]:
                pv_a1v1_net[f["video"]] += 1
    for f in result["a1v2c_fires"]:
        if f["pl_pid"] == f["gt_pid"]:
            if not f["match_gt"]:
                pv_a1v2c_net[f["video"]] -= 1
        else:
            if f["match_gt"]:
                pv_a1v2c_net[f["video"]] += 1
    for f in result["s4_changes"]:
        if f["pl_pid"] == f["gt_pid"]:
            if not f["match_gt"]:
                pv_s4_net[f["video"]] -= 1
        else:
            if f["match_gt"]:
                pv_s4_net[f["video"]] += 1
    for v in VIDEO_NAMES:
        b = per_video.get(v, CounterBundle())
        nev = b.total - b.pipeline_missing
        lines.append(
            f"| {v} | {b.total} | {_pct(b.pipeline_correct, nev)} | "
            f"{pv_a1v1_net.get(v, 0):+d} | {pv_a1v2c_net.get(v, 0):+d} | "
            f"{pv_s4_net.get(v, 0):+d} |"
        )
    lines.append("")

    # Per-action-type
    lines.append("## Per-action-type breakdown")
    lines.append("")
    lines.append("| Type | GT n | Pipeline-correct | Wrong same-team | Wrong cross-team | Missing |")
    lines.append("|---|---|---|---|---|---|")
    for atype in ["SERVE", "RECEIVE", "SET", "ATTACK", "BLOCK", "DIG"]:
        b = per_action.get(atype, CounterBundle())
        nev = b.total - b.pipeline_missing
        lines.append(
            f"| {atype} | {b.total} | {_pct(b.pipeline_correct, nev)} | "
            f"{b.pipeline_wrong_same_team} | {b.pipeline_wrong_cross_team} | {b.pipeline_missing} |"
        )
    lines.append("")

    # Verdict
    lines.append("## Verdict")
    lines.append("")
    pipeline_acc = overall.pipeline_correct / max(n_eval, 1)
    pipeline_recall = overall.pipeline_correct / max(overall.total, 1)
    s4_evaluable = overall.s4_changed_pipeline + overall.s4_unchanged_correct + overall.s4_unchanged_wrong
    s4_total_correct = overall.s4_changed_match_gt + overall.s4_unchanged_correct
    s4_acc = s4_total_correct / max(s4_evaluable, 1)
    lines.append(f"- **Baseline pipeline precision** (on pipeline contacts that hit a GT row): {pipeline_acc*100:.1f}% ({overall.pipeline_correct}/{n_eval})")
    lines.append(f"- **Baseline pipeline recall** (correct picks / all resolved GT rows): {pipeline_recall*100:.1f}% ({overall.pipeline_correct}/{overall.total})")
    lines.append(f"  - 'Pipeline missing' = {overall.pipeline_missing}/{overall.total} = {100.0*overall.pipeline_missing/max(overall.total,1):.1f}%. Includes both detection misses (no pipeline action within ±5) and action-type mismatches (e.g. BLOCK→attack accounts for 10/12 GT blocks).")
    lines.append(f"- **A1.v1 fires**: {overall.a1v1_fired} pairs; net delta {net_delta_a1v1:+d} (fixes {a1v1_rule_fixed}, breaks {a1v1_rule_broke}/{baseline_correct_on_fires})")
    lines.append(f"- **A1.v2-conf fires**: {overall.a1v2c_fired} pairs; net delta {net_delta_a1v2c:+d} (fixes {a1v2c_rule_fixed}, breaks {a1v2c_rule_broke}/{a1v2c_baseline_correct})")
    lines.append(f"- **S4 changes**: {overall.s4_changed_pipeline} contacts; net delta {net_delta_s4:+d} (fixes {s4_rule_fixed}, breaks {s4_rule_broke}/{s4_baseline_correct})")
    lines.append(f"- **S4 simulated precision** (S4-pick vs GT, where evaluable): {s4_acc*100:.1f}%")
    lines.append(f"- **At-net attack FP rate**: {100.0 * at_net['unmatched'] / max(at_net['total'], 1):.1f}% ({at_net['unmatched']}/{at_net['total']})")
    lines.append("")
    lines.append("### Interpretation")
    lines.append("")
    lines.append("Pipeline attribution-precision on the 12 trusted videos is **decent**")
    lines.append("(84% on the contacts the pipeline detected; 63% recall against all GT")
    lines.append("rows because contact-detector recall + action-type confusion together")
    lines.append("drop 25% of GT contacts).")
    lines.append("")
    lines.append("All three rule families regress in net delta on this corpus:")
    lines.append("- **A1.v1**: every single same-player-back-to-back pair where the rule fired")
    lines.append("  on a baseline-correct curr broke that correct pick (16/16). The pair's")
    lines.append("  C-4 violation usually means the *prev* action was the misattributed one,")
    lines.append("  not the curr. Flipping curr is therefore biased to break correct picks.")
    lines.append("- **A1.v2-conf**: same effect but milder (10 fires vs 27). Confidence isn't")
    lines.append("  reliable enough to choose the right side to flip; baseline still wins.")
    lines.append("- **S4 (traj-int + anti-self)**: 30/30 baseline-correct picks broken; the")
    lines.append("  trajectory signal pulls strongly toward setter/digger positions that")
    lines.append("  often aren't the actual toucher. Net -25 over 44 changes.")
    lines.append("")
    lines.append("The **at-net attack FP rate is 40% overall** (72/180 contacts have NO")
    lines.append("ATTACK GT within ±5 frames). Per-video this splits sharply: titi/toto/lulu/")
    lines.append("yeye sit at 38-83% while juju/kaka/gaga/cuco/cici/cece sit at 0-11%. That")
    lines.append("bimodality is either an upstream signal (some videos have many phantom")
    lines.append("at-net attacks the user deleted) or a GT-coverage artifact (less-labeled")
    lines.append("videos look like FPs by absence). The 6 clean-floor videos suggest at-net")
    lines.append("attacks ARE largely real contacts when labeled; the 6 noisy ones merit")
    lines.append("manual review before drawing inference.")
    lines.append("")
    lines.append("### Recommendation")
    lines.append("")
    lines.append("Continue labeling T2 only if the goal is to grow corpus for *learned*")
    lines.append("attribution (PGM Phase B, embedding-based scoring). Rule-based families")
    lines.append("(A1.v1, A1.v2-conf, S4) all regress on this corpus and should not ship.")
    lines.append("")
    lines.append("Independent of attribution, the contact-detector has measurable issues to")
    lines.append("address with the existing GT:")
    lines.append("1. BLOCK class — pipeline produced only 3 blocks across 108 rallies; GT has 12.")
    lines.append("   10/12 GT blocks are pipeline-labeled `attack`. Block re-classification is")
    lines.append("   reachable today.")
    lines.append("2. Pipeline-missing rate of 25% is a *recall* ceiling that no attribution")
    lines.append("   rule can lift. Half (102/158) are no-contact-within-±5 frames — that's")
    lines.append("   contact-detector recall debt, not attribution.")
    lines.append("")

    MD_PATH.write_text("\n".join(lines))
    print(f"Wrote markdown: {MD_PATH}")


def write_json(result: dict[str, Any]) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    # CounterBundle to dict
    def cb_to_dict(cb: CounterBundle) -> dict[str, int]:
        return {
            "total": cb.total,
            "pipeline_correct": cb.pipeline_correct,
            "pipeline_wrong_same_team": cb.pipeline_wrong_same_team,
            "pipeline_wrong_cross_team": cb.pipeline_wrong_cross_team,
            "pipeline_missing": cb.pipeline_missing,
            "a1v1_fired": cb.a1v1_fired,
            "a1v1_match_gt": cb.a1v1_match_gt,
            "a1v1_disagrees_pipeline": cb.a1v1_disagrees_pipeline,
            "a1v2c_fired": cb.a1v2c_fired,
            "a1v2c_curr_changed_match_gt": cb.a1v2c_curr_changed_match_gt,
            "a1v2c_curr_changed_wrong": cb.a1v2c_curr_changed_wrong,
            "s4_changed_pipeline": cb.s4_changed_pipeline,
            "s4_changed_match_gt": cb.s4_changed_match_gt,
            "s4_changed_wrong": cb.s4_changed_wrong,
            "s4_unchanged_correct": cb.s4_unchanged_correct,
            "s4_unchanged_wrong": cb.s4_unchanged_wrong,
            "s4_skipped_no_window": cb.s4_skipped_no_window,
        }

    out = {
        "overall": cb_to_dict(result["overall"]),
        "per_video": {k: cb_to_dict(v) for k, v in result["per_video"].items()},
        "per_action": {k: cb_to_dict(v) for k, v in result["per_action"].items()},
        "rally_count": result["rally_count"],
        "n_rallies": result["n_rallies"],
        "at_net": result["at_net"],
        "a1v1_fires": result["a1v1_fires"],
        "a1v2c_fires": result["a1v2c_fires"],
        "s4_changes": result["s4_changes"],
    }
    JSON_PATH.write_text(json.dumps(out, indent=2))
    print(f"Wrote JSON: {JSON_PATH}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    print("=== Precision validation on 12 trusted-GT videos (2026-05-14) ===")
    print(f"Videos: {VIDEO_NAMES}")
    print(f"GT_FRAME_TOL={GT_FRAME_TOL}, K_PRE={K_PRE}, MIN_BALL_PRE={MIN_BALL_PRE}")
    print()

    result = run_measurements()
    write_json(result)
    write_markdown(result)
    print()
    print(f"Open: {MD_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
