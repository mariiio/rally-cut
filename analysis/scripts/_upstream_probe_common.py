"""Shared helpers for upstream-pipeline bottleneck probes (2026-05-20).

Consumed by probe_upstream_L1..L4, L6. L5 uses a different (training-loop)
pathway and doesn't need these helpers.

Functions:
  load_wrong_attribution_corpus()      -> list[WrongAttributionRow]
  fetch_rally_state(rally_id)          -> dict with positions, ball_positions,
                                          contacts, actions, teams
  derive_gt_team_chain(gt_contacts,    -> list[str|None]  # 'A'/'B'/None per contact
                       team_assignments)
  rescore_contact(...)                 -> int | None     # picked track_id
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psycopg

# Add rallycut to path (scripts run from analysis/, scripts/ subdir)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rallycut.tracking.dynamic_attribution_scorer import (  # noqa: E402
    DynamicAttributionScorer,
    extract_features,
    position_from_dict,
)

DB_DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5436/rallycut",
)

TRUSTED_32 = (
    "titi", "toto", "lulu", "wawa", "caco", "cece", "cici", "cuco",
    "gaga", "gigi", "kaka", "kiki", "keke", "koko", "kuku",
    "juju", "yeye", "gugu", "mame", "meme", "mimi", "moma", "mumu",
    "papa", "pepe", "pipi", "popo", "pupu", "veve", "vivi", "vovo",
    "haha",
)


@dataclass(frozen=True)
class WrongAttributionRow:
    """One contact where pipeline playerTrackId != GT resolved_track_id."""

    rally_id: str
    video: str
    action_frame: int
    action_type: str
    pipeline_pid: int
    gt_pid: int
    pipeline_match_delta: int  # |pipeline_frame - gt_frame| at matching


def load_wrong_attribution_corpus(
    videos: tuple[str, ...] = TRUSTED_32,
) -> list[WrongAttributionRow]:
    """Load all contacts on `videos` where pipeline picked the wrong player.

    Match rule: for each GT row, find the pipeline action closest within ±5
    frames, prefer same action_type. Same matching rule as
    measure_attribution_trusted_31_2026_05_20.py.
    """
    out: list[WrongAttributionRow] = []
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            """
            SELECT v.name, r.id, rg.action::text, rg.frame, rg.resolved_track_id,
                   pt.actions_json
            FROM rally_action_ground_truth rg
            JOIN rallies r ON rg.rally_id = r.id
            JOIN videos v ON r.video_id = v.id
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE v.name = ANY(%s) AND rg.resolved_track_id IS NOT NULL
            """,
            [list(videos)],
        )
        rows = cur.fetchall()

    for vname, rid, gt_action_raw, gt_frame, gt_tid, actions_json in rows:
        gt_action = gt_action_raw.upper()
        aj = actions_json if isinstance(actions_json, dict) else (
            json.loads(actions_json) if isinstance(actions_json, str) else {}
        )
        actions = aj.get("actions") or []
        # Match: same type within ±5 frames, else closest within ±5
        best = None
        best_delta = 6
        for a in actions:
            if (a.get("action") or "").upper() != gt_action:
                continue
            d = abs(int(a.get("frame", -10**9)) - int(gt_frame))
            if d < best_delta:
                best_delta = d
                best = a
        if best is None:
            best_delta = 6
            for a in actions:
                d = abs(int(a.get("frame", -10**9)) - int(gt_frame))
                if d < best_delta:
                    best_delta = d
                    best = a
        if best is None:
            continue  # unmatched (contact-detection FN) — out of scope
        pipeline_pid = int(best.get("playerTrackId", -1))
        if pipeline_pid == int(gt_tid):
            continue  # correct attribution; not in wrong-attribution corpus
        out.append(WrongAttributionRow(
            rally_id=str(rid),
            video=str(vname),
            action_frame=int(best.get("frame", 0)),
            action_type=gt_action.lower(),
            pipeline_pid=pipeline_pid,
            gt_pid=int(gt_tid),
            pipeline_match_delta=best_delta,
        ))
    return out


def fetch_rally_state(rally_id: str) -> dict[str, Any] | None:
    """Return positions/ball/contacts/actions/teams for one rally."""
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            """
            SELECT v.name, pt.actions_json, pt.contacts_json,
                   pt.positions_json, pt.ball_positions_json
            FROM rallies r
            JOIN videos v ON r.video_id = v.id
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE r.id = %s
            """,
            [rally_id],
        )
        row = cur.fetchone()
    if not row:
        return None
    vname, aj, cj, pj, bj = row
    if isinstance(aj, str):
        aj = json.loads(aj)
    if isinstance(cj, str):
        cj = json.loads(cj)
    if isinstance(pj, str):
        pj = json.loads(pj)
    if isinstance(bj, str):
        bj = json.loads(bj)
    return {
        "video": vname,
        "actions": (aj or {}).get("actions") or [],
        "teams": (aj or {}).get("teamAssignments") or {},
        "contacts": (cj or {}).get("contacts") or [],
        "positions": pj or [],
        "ball_positions": bj or [],
    }


def derive_gt_team_chain(
    gt_contacts: list[tuple[int, str, int]],  # [(frame, action, resolved_tid)]
    team_assignments: dict[str, str],
) -> list[str | None]:
    """Walk GT contacts in frame order and return team ('A'/'B') of actor per contact."""
    out: list[str | None] = []
    for _frame, _action, tid in sorted(gt_contacts, key=lambda x: x[0]):
        team = team_assignments.get(str(tid))
        out.append(team if team in ("A", "B") else None)
    return out


def rescore_contact(
    rally_state: dict[str, Any],
    contact: dict[str, Any],
    action_type: str,
    cand_tids: list[int],
    expected_team: int | None = None,
    team_assignments_int: dict[int, int] | None = None,
    contact_frame_override: int | None = None,
    ball_position_override: tuple[float, float] | None = None,
) -> int | None:
    """Re-score a contact with optional input substitutions; return picked tid.

    Used by L1/L2/L3/L4/L6 to substitute oracle inputs and observe the
    scorer's pick under different upstream-layer conditions.
    """
    scorer = DynamicAttributionScorer()
    positions_like = [position_from_dict(p) for p in rally_state["positions"]]
    frame = contact_frame_override if contact_frame_override is not None else int(contact["frame"])
    ball_x = ball_position_override[0] if ball_position_override else float(contact.get("ballX", 0.5))
    ball_y = ball_position_override[1] if ball_position_override else float(contact.get("ballY", 0.5))

    cf_list = []
    for tid in cand_tids:
        cf = extract_features(
            positions_like, tid, frame, ball_x, ball_y,
            prev_action_tid=-1,
            post_ball_x=None, post_ball_y=None,
            expected_team=expected_team,
            team_assignments=team_assignments_int,
        )
        if cf is not None:
            cf_list.append(cf)
    if not cf_list:
        return None
    probs = scorer.score(action_type, cf_list)
    if probs is None:
        return None
    best_idx = max(range(len(probs)), key=lambda i: probs[i])
    return cf_list[best_idx].track_id
