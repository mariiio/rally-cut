"""Phase 0.5 — v0 attribution visual debug viewer (reviewer-friendly).

Reads a baseline-schema JSON and emits a simple static-HTML dashboard focused on
fast visual review:

- ``reports/attribution_audit/index.html`` — fixture tiles with aggregate rates,
  click-through to per-rally.
- ``reports/attribution_audit/{fixture}/_index.html`` — per-fixture rally list.
- ``reports/attribution_audit/{fixture}/{rally_id}.html`` — per-rally viewer.

Per-rally view is **error-first**: wrongs + missings shown by default as big
cards; correct actions collapsed under a toggle. Each error card shows the GT
actor, the pipeline pick, and a plain-English reason.

Usage:
    uv run python scripts/render_attribution_viewer.py
    uv run python scripts/render_attribution_viewer.py --input reports/attribution_rebuild/phase2_confidence_gated.json
"""
from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path
from typing import Any

from rallycut.evaluation.attribution_bench import (
    WRONG_CATEGORIES,
    aggregate,
)
from rallycut.evaluation.db import get_connection

DEFAULT_INPUT = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "attribution_rebuild"
    / "baseline_2026_04_24.json"
)
DEFAULT_OUT = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "attribution_audit"
)

CATEGORY = {
    "correct": {"color": "#2f7d2f", "bg": "#e7f5e7", "icon": "✓", "label": "correct"},
    "wrong_cross_team": {
        "color": "#c23b3b", "bg": "#fdecec", "icon": "✗", "label": "WRONG • across team",
    },
    "wrong_same_team": {
        "color": "#d97a2c", "bg": "#fdf1e4", "icon": "✗", "label": "WRONG • same team",
    },
    "wrong_unknown_team": {
        "color": "#7a4da1", "bg": "#f1ecf7", "icon": "✗", "label": "WRONG • unknown team",
    },
    "abstained": {"color": "#888", "bg": "#f3f3f3", "icon": "?", "label": "abstained"},
    "missing": {"color": "#333", "bg": "#eee", "icon": "○", "label": "MISSING"},
}

TEAM_PALETTE = {
    "A": {"color": "#1f6feb", "bg": "#dbeafe"},
    "B": {"color": "#b91c1c", "bg": "#fee2e2"},
}

DEFAULT_FPS = 30.0  # fallback; real value per-video fetched via _fetch_video_fps

# Position jump greater than this (normalized distance per frame) is suspicious.
# At 30fps a sprinting player moves ~0.02/frame across a 15m court width.
SWAP_DIST_THRESHOLD = 0.05


def _audit_label_flip(
    matches: list[dict[str, Any]],
    team_assignments: dict[str, str],
) -> dict[str, Any]:
    """Detect systematic A↔B label flip.

    For rallies where the pairing is correct (positional near == teamA pids) but
    the LABELS are swapped (GT actor's team != pipeline pick's team on every
    action), pipeline picks the opposite team consistently. Heuristic: if ≥80%
    of matches where pipeline made a pick are cross-team wrong, AND ≥3 such
    picks exist, flag it.
    """
    n_picked = 0  # excludes missing + abstained
    n_cross_team = 0
    for m in matches:
        cat = m["category"]
        if cat in ("missing", "abstained"):
            continue
        n_picked += 1
        if cat == "wrong_cross_team":
            n_cross_team += 1
    if n_picked < 3:
        return {"label_flip_suspect": False, "cross_team_ratio": 0.0, "n_picked": n_picked}
    ratio = n_cross_team / n_picked
    return {
        "label_flip_suspect": ratio >= 0.8,
        "cross_team_ratio": round(ratio, 3),
        "n_picked": n_picked,
        "n_cross_team": n_cross_team,
    }


def _audit_primitives(
    tracking: dict[str, Any],
    team_assignments: dict[str, str],
) -> dict[str, Any]:
    """Compute per-rally primitive audit flags.

    Returns dict with:
      - swap_events: list[{tid, frame, dist}] where a tid jumps > SWAP_DIST_THRESHOLD
        from one consecutive frame to the next.
      - team_flip_suspect: True if positional median-y grouping disagrees with
        teamAssignments pairing.
      - positional_pairs: sets of pids by side.
      - team_pairs: sets of pids by team.
    """
    from collections import defaultdict
    from statistics import median

    per_tid: dict[int, list[dict]] = defaultdict(list)
    for p in tracking["positions"]:
        if p["t"] in (1, 2, 3, 4):
            per_tid[p["t"]].append(p)
    # Sort by frame
    for t in per_tid:
        per_tid[t].sort(key=lambda p: p["f"])

    swap_events: list[dict] = []
    for tid, entries in per_tid.items():
        for i in range(1, len(entries)):
            prev = entries[i - 1]
            curr = entries[i]
            dt_frames = curr["f"] - prev["f"]
            if dt_frames == 0 or dt_frames > 5:
                continue  # skip gaps (detection missed) — not a swap signal
            dx = curr["x"] - prev["x"]
            dy = curr["y"] - prev["y"]
            dist = (dx * dx + dy * dy) ** 0.5
            per_frame = dist / dt_frames
            if per_frame > SWAP_DIST_THRESHOLD:
                swap_events.append(
                    {
                        "tid": tid,
                        "frame": curr["f"],
                        "dist": round(per_frame, 3),
                    }
                )

    # Team-flip detector: use rally-start positions (first 15 frames) to pair
    # players by side. At rally start (serve), teams are strictly on their
    # sides by volleyball rule — no crossings yet. Median-over-rally misleads
    # on fixtures where setters/defenders cross the midline.
    med_y: dict[int, float] = {}
    for tid, entries in per_tid.items():
        early = [e["y"] for e in entries if e["f"] < 15]
        if len(early) >= 3:
            med_y[tid] = median(early)
        elif entries:
            med_y[tid] = median(e["y"] for e in entries)
    positional_near: set[int] = set()
    positional_far: set[int] = set()
    pair_confidence = 0.0
    if len(med_y) >= 4:
        sorted_tids = sorted(med_y.keys(), key=lambda t: -med_y[t])
        # pair_confidence = gap between 2nd-highest and 3rd-highest median y.
        # If small, the positional split is near the midline and unreliable.
        pair_confidence = med_y[sorted_tids[1]] - med_y[sorted_tids[2]]
        positional_near = set(sorted_tids[:2])
        positional_far = set(sorted_tids[2:])

    team_pairs: dict[str, set[int]] = {"A": set(), "B": set()}
    for pid_str, team in team_assignments.items():
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        if team in team_pairs and pid in (1, 2, 3, 4):
            team_pairs[team].add(pid)

    # Minimum gap between near/far median-y groups to trust the positional
    # split. Below this, both groups straddle the midline and positional
    # grouping is noise — do not flag.
    PAIR_CONFIDENCE_MIN = 0.05
    team_flip_suspect = False
    if (
        positional_near
        and team_pairs["A"]
        and team_pairs["B"]
        and pair_confidence >= PAIR_CONFIDENCE_MIN
    ):
        a_set, b_set = team_pairs["A"], team_pairs["B"]
        matches_a = (a_set == positional_near) or (a_set == positional_far)
        matches_b = (b_set == positional_near) or (b_set == positional_far)
        team_flip_suspect = not (matches_a and matches_b)

    return {
        "swap_events": swap_events,
        "team_flip_suspect": team_flip_suspect,
        "positional_pairs": {
            "near": sorted(positional_near),
            "far": sorted(positional_far),
        },
        "pair_confidence": round(pair_confidence, 4),
        "team_pairs": {k: sorted(v) for k, v in team_pairs.items()},
        "med_y": {t: round(y, 3) for t, y in med_y.items()},
    }

CSS = """
:root { --ink: #1c1c1c; --muted: #667; --line: #e2e2e2; --bg: #fafafa; }
* { box-sizing: border-box; }
body { font-family: -apple-system, system-ui, sans-serif; color: var(--ink);
       background: var(--bg); margin: 0; padding: 28px 40px; }
h1 { font-size: 24px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 24px 0 10px; color: var(--muted); text-transform: uppercase;
     letter-spacing: 0.08em; font-weight: 600; }
h3 { font-size: 15px; margin: 0 0 8px; }
a { color: #0a58ca; text-decoration: none; }
a:hover { text-decoration: underline; }
.back { display: inline-block; margin-bottom: 14px; color: var(--muted); font-size: 13px; }
.subtitle { color: var(--muted); font-size: 13px; margin-bottom: 18px; }

.summary-strip { display: flex; gap: 10px; margin: 8px 0 18px; flex-wrap: wrap; }
.summary-tile { padding: 12px 18px; border-radius: 8px; min-width: 110px;
                border: 1px solid var(--line); background: #fff; }
.summary-tile .count { font-size: 24px; font-weight: 700; }
.summary-tile .label { font-size: 11px; text-transform: uppercase;
                       letter-spacing: 0.06em; color: var(--muted); }

.roster { display: flex; gap: 8px; margin: 8px 0 18px; flex-wrap: wrap; }
.pid-chip { display: inline-flex; align-items: center; gap: 6px; padding: 4px 10px;
            border-radius: 16px; font-size: 13px; background: #fff;
            border: 1px solid var(--line); }
.pid-chip .pid { font-weight: 700; }
.pid-chip .team { font-size: 10px; padding: 1px 6px; border-radius: 8px;
                  text-transform: uppercase; letter-spacing: 0.05em; }

.fixture-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(270px, 1fr));
                gap: 14px; margin-top: 16px; }
.fixture-card { padding: 14px; background: #fff; border: 1px solid var(--line);
                border-radius: 10px; }
.fixture-card h3 { font-size: 18px; }
.fixture-card h3 a { color: var(--ink); }
.rate-row { display: flex; gap: 4px; margin: 10px 0 6px; align-items: baseline; }
.rate-val { font-weight: 700; font-size: 18px; }
.rate-label { font-size: 11px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
.bar { display: flex; height: 8px; border-radius: 4px; overflow: hidden;
       background: #f0f0f0; margin-top: 6px; }

.rally-table { width: 100%; border-collapse: collapse; background: #fff;
               border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }
.rally-table th { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
                  color: var(--muted); font-weight: 600; padding: 10px 14px;
                  border-bottom: 1px solid var(--line); text-align: left; background: #fbfbfb; }
.rally-table td { padding: 10px 14px; border-bottom: 1px solid var(--line); font-size: 14px; }
.rally-table tr:last-child td { border-bottom: none; }
.rally-table tr:hover td { background: #fafafa; }
.num { text-align: right; font-variant-numeric: tabular-nums; }

.err-list { display: flex; flex-direction: column; gap: 10px; }
.err-card { background: #fff; border: 1px solid var(--line); border-radius: 8px;
            padding: 14px 18px; border-left: 4px solid var(--line); display: grid;
            grid-template-columns: 110px 1fr 1fr; gap: 14px; align-items: center; }
.err-card .frame-block { font-size: 12px; color: var(--muted); }
.err-card .frame-block .frame { font-size: 20px; font-weight: 700; color: var(--ink); }
.err-card .actor-block { font-size: 13px; }
.err-card .actor-block .label { color: var(--muted); font-size: 11px;
                                text-transform: uppercase; letter-spacing: 0.05em;
                                margin-bottom: 2px; }
.err-card .actor-block .pid { font-weight: 700; font-size: 15px; }
.err-card .cat-pill { display: inline-block; padding: 3px 10px; border-radius: 12px;
                      font-size: 11px; font-weight: 700; letter-spacing: 0.04em;
                      text-transform: uppercase; }
.err-reason { grid-column: 1 / -1; color: var(--muted); font-size: 12px;
              font-style: italic; padding-top: 6px; border-top: 1px dashed var(--line); }

.toggle { display: inline-flex; gap: 6px; align-items: center; font-size: 13px;
          color: var(--muted); cursor: pointer; user-select: none; }
details > summary { cursor: pointer; font-size: 13px; color: var(--muted); padding: 6px 0; }

.nav { display: flex; justify-content: space-between; align-items: center; gap: 16px;
       margin-top: 28px; padding-top: 14px; border-top: 1px solid var(--line); }
.nav a { padding: 8px 14px; border: 1px solid var(--line); border-radius: 6px;
         background: #fff; font-size: 14px; }
.nav a.disabled { color: #bbb; pointer-events: none; background: #f5f5f5; }

.stale { color: #b45; }
code { background: #f0f0f0; padding: 1px 5px; border-radius: 3px; font-size: 12px; }

.clip-wrap { background: #000; border-radius: 10px; overflow: hidden; position: relative;
             max-width: 960px; margin: 0 auto 10px; }
.clip-wrap video { display: block; width: 100%; height: auto; }
.clip-wrap canvas { position: absolute; top: 0; left: 0; pointer-events: none; }
.clip-banner { position: absolute; left: 10px; right: 10px; top: 10px;
               background: rgba(0,0,0,0.78); color: #fff; padding: 10px 14px;
               border-radius: 8px; font-size: 14px; line-height: 1.4;
               pointer-events: none; display: none; border-left: 4px solid #fff; }
.clip-banner b { color: #ffd; }
.clip-banner .cat { display: inline-block; padding: 2px 8px; border-radius: 10px;
                    font-size: 11px; font-weight: 700; text-transform: uppercase;
                    letter-spacing: 0.04em; margin-right: 6px; }
.timeline { position: relative; height: 38px; max-width: 960px; margin: 0 auto 20px;
            background: #fff; border: 1px solid var(--line); border-radius: 6px; }
.timeline .tick { position: absolute; top: 4px; bottom: 4px; width: 3px;
                  border-radius: 2px; cursor: pointer; }
.timeline .tick:hover { width: 5px; top: 2px; bottom: 2px; }
.timeline .tick::after { content: attr(data-label); position: absolute;
                          left: 6px; top: 50%; transform: translateY(-50%);
                          font-size: 10px; white-space: nowrap;
                          background: rgba(255,255,255,0.9); padding: 1px 4px;
                          border-radius: 3px; pointer-events: none; display: none; }
.timeline .tick:hover::after { display: inline-block; z-index: 10; }
.playhead { position: absolute; top: 0; bottom: 0; width: 2px;
            background: #0a58ca; pointer-events: none; }

.err-card.clickable { cursor: pointer; }
.err-card.clickable:hover { transform: translateY(-1px);
                            box-shadow: 0 2px 8px rgba(0,0,0,0.08); }

.primitive-row { margin: 4px 0 14px; display: flex; gap: 8px; flex-wrap: wrap; }
.pbadge { padding: 4px 10px; border-radius: 6px; font-size: 12px; font-weight: 600; }
.pbadge.ok { background: #e7f5e7; color: #2f7d2f; }
.pbadge.fail { background: #fdecec; color: #c23b3b; border: 1px solid #f8c8c8; }
.pbadge.warn { background: #fdf5e4; color: #a56611; border: 1px solid #f2e2b8; }
.rally-row .prim-cell { font-size: 11px; }
"""


def _esc(s: Any) -> str:
    return html.escape(str(s))


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return "—"
    return f"{x * 100:.1f}%"


def _fmt_delta(count: int, total: int, color: str) -> str:
    pct = (count / total * 100) if total else 0.0
    return (
        f"<div class='summary-tile' style='border-left:4px solid {color};'>"
        f"<div class='count' style='color:{color};'>{count}</div>"
        f"<div class='label'>{pct:.0f}% of {total}</div></div>"
    )


def _team_chip(pid: Any, team: str | None, extra: str = "") -> str:
    pal = TEAM_PALETTE.get(team, {"color": "#888", "bg": "#f2f2f2"})
    team_lbl = team if team else "?"
    return (
        f"<span class='pid-chip'>"
        f"<span class='pid'>pid {_esc(pid)}</span>"
        f"<span class='team' style='background:{pal['bg']};color:{pal['color']};'>"
        f"team {_esc(team_lbl)}</span>"
        f"{extra}"
        f"</span>"
    )


def _reason_sentence(m: dict[str, Any], team_assignments: dict) -> str:
    cat = m["category"]
    gt_pid = m.get("gt_pid")
    pl_pid = m.get("pl_pid")
    gt_team = team_assignments.get(str(gt_pid))
    pl_team = team_assignments.get(str(pl_pid))
    if cat == "correct":
        return f"Pipeline correctly picked pid {gt_pid}."
    if cat == "missing":
        return f"No pipeline action landed within ±10 frames of GT frame {m['gt_frame']}."
    if cat == "abstained":
        return "Pipeline emitted this action but declined to pick a player."
    if cat == "wrong_same_team":
        return (
            f"Pipeline picked teammate pid {pl_pid} (team {pl_team}) "
            f"instead of the actual actor pid {gt_pid} (team {gt_team}) — "
            "within-team confusion."
        )
    if cat == "wrong_cross_team":
        return (
            f"Pipeline picked pid {pl_pid} on team {pl_team}, "
            f"but GT actor pid {gt_pid} is on team {gt_team} — "
            "opposing-team mistake."
        )
    if cat == "wrong_unknown_team":
        missing = []
        if gt_team is None:
            missing.append(f"GT pid {gt_pid}")
        if pl_team is None:
            missing.append(f"pipeline pid {pl_pid}")
        who = " and ".join(missing)
        return f"{who} not in team_assignments — stage-2 primitive gap."
    return ""


_FPS_CACHE: dict[str, float] = {}


def _fetch_video_fps(video_id: str) -> float:
    """Fetch the source fps for a video. Tracker indexes positions at this rate
    regardless of clip-encoding fps."""
    if video_id in _FPS_CACHE:
        return _FPS_CACHE[video_id]
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("SELECT fps FROM videos WHERE id = %s", (video_id,))
        row = cur.fetchone()
    fps = float(row[0]) if row and row[0] else DEFAULT_FPS
    _FPS_CACHE[video_id] = fps
    return fps


def _fetch_tracking(rally_id: str) -> dict[str, Any]:
    """Pull positions + ball for a rally, stripping keypoints to keep size down."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT positions_json, ball_positions_json FROM player_tracks "
            "WHERE rally_id = %s",
            (rally_id,),
        )
        row = cur.fetchone()
    positions, ball = (row or (None, None))
    slim_pos: list[dict] = []
    if positions:
        for p in positions:
            slim_pos.append(
                {
                    "f": p.get("frameNumber"),
                    "t": p.get("trackId"),
                    "x": p.get("x"),
                    "y": p.get("y"),
                    "w": p.get("width"),
                    "h": p.get("height"),
                }
            )
    slim_ball: list[dict] = []
    if ball:
        for b in ball:
            slim_ball.append(
                {
                    "f": b.get("frameNumber"),
                    "x": b.get("x"),
                    "y": b.get("y"),
                }
            )
    return {"positions": slim_pos, "ball": slim_ball}


def render_rally(
    rally: dict[str, Any],
    out_path: Path,
    prev_rid: str | None,
    next_rid: str | None,
    clips_dir: Path,
) -> dict[str, Any]:
    rid = rally["rally_id"]
    fx = rally["fixture"]
    totals = rally.get("rally_totals") or {}
    n = totals.get("n_gt_actions", 0)
    wrong = sum(totals.get(k, 0) for k in WRONG_CATEGORIES)
    correct = totals.get("correct", 0)
    missing = totals.get("missing", 0)
    abstained = totals.get("abstained", 0)
    team_assignments = rally.get("team_assignments") or {}
    ptids = rally.get("primary_track_ids") or []
    serving_team = rally.get("serving_team")
    start_ms = rally.get("start_ms", 0)
    end_ms = rally.get("end_ms", start_ms)
    clip_duration_s = max((end_ms - start_ms) / 1000.0, 0.1)
    clip_rel_path = f"clips/{rid}.mp4"
    clip_exists = (clips_dir / f"{rid}.mp4").exists()
    fps = _fetch_video_fps(rally["video_id"])

    ptid_stable = len(ptids) == 4 and set(ptids) == {1, 2, 3, 4}
    ptid_note = (
        "<span style='color:#2f7d2f;'>stable {1,2,3,4}</span>"
        if ptid_stable
        else f"<span class='stale'>unstable: {_esc(ptids)}</span>"
    )

    # Primitive audit badges (computed after clip_block fetches tracking)
    # Set defaults in case the clip doesn't exist.
    _primitive_badges = ""
    _rally_primitive_fail = False

    # Roster chips (pid → team)
    roster_chips = "".join(
        _team_chip(pid, team_assignments.get(str(pid)))
        for pid in sorted({p for p in ptids if p is not None})
    )

    # Error cards (wrongs + missings first, correct collapsed)
    errors: list[str] = []
    corrects: list[str] = []
    timeline_ticks: list[str] = []
    for m in rally.get("matches", []):
        cat = m["category"]
        pal = CATEGORY[cat]
        gt_team = team_assignments.get(str(m.get("gt_pid")))
        pl_team = team_assignments.get(str(m.get("pl_pid")))
        reason = _reason_sentence(m, team_assignments)
        # GT frame is rally-local (positions + ball both use rally-local
        # frameNumber starting at 0). Time in clip = frame / fps. Use the
        # video's source fps — tracker indexes positions at that rate.
        frame = m["gt_frame"]
        time_in_clip = max(0.0, frame / fps)
        pct = (time_in_clip / clip_duration_s * 100.0) if clip_duration_s else 0.0
        pct = max(0.0, min(pct, 100.0))
        tick_label = f"f{frame} {m.get('gt_action') or ''} · {pal['label']}"
        timeline_ticks.append(
            f"<div class='tick' data-seek='{time_in_clip:.3f}' "
            f"data-label='{_esc(tick_label)}' "
            f"style='left:{pct:.2f}%;background:{pal['color']};'></div>"
        )
        card = (
            f"<div class='err-card clickable' data-seek='{time_in_clip:.3f}' "
            f"style='border-left-color:{pal['color']}; background:{pal['bg']};'>"
            f"<div class='frame-block'>"
            f"<div class='frame'>f{_esc(m['gt_frame'])}</div>"
            f"<div>{_esc(m['gt_action']) if m['gt_action'] else '—'}</div>"
            f"</div>"
            f"<div class='actor-block'>"
            f"<div class='label'>Actual (GT)</div>"
            f"{_team_chip(m.get('gt_pid'), gt_team)}"
            f"</div>"
            f"<div class='actor-block'>"
            f"<div class='label'>Pipeline pick</div>"
            + (
                _team_chip(
                    m.get('pl_pid'),
                    pl_team,
                    f" <span style='color:var(--muted);font-size:11px;'>conf {_fmt_pct(m.get('pl_confidence'))}</span>"
                    if m.get('pl_pid') is not None and m.get('pl_confidence') is not None
                    else "",
                )
                if m.get('pl_pid') is not None
                else "<span style='color:var(--muted);'>—</span>"
            )
            + "</div>"
            f"<div class='err-reason'>"
            f"<span class='cat-pill' style='background:{pal['color']};color:#fff;'>"
            f"{pal['icon']} {pal['label']}</span> &nbsp; {_esc(reason)}"
            f"</div>"
            f"</div>"
        )
        if cat == "correct":
            corrects.append(card)
        else:
            errors.append(card)

    prev_link = f"<a href='{prev_rid}.html'>← prev rally</a>" if prev_rid else "<a class='disabled'>← prev rally</a>"
    next_link = f"<a href='{next_rid}.html'>next rally →</a>" if next_rid else "<a class='disabled'>next rally →</a>"

    primitive_audit: dict[str, Any] = {}
    # Label-flip check runs regardless of clip presence (uses matches only).
    label_flip = _audit_label_flip(rally.get("matches", []), team_assignments)
    if clip_exists:
        tracking = _fetch_tracking(rid)
        primitive_audit = _audit_primitives(tracking, team_assignments)
        primitive_audit["label_flip"] = label_flip
        tracking_json = json.dumps(tracking, separators=(",", ":"))
        matches_json = json.dumps(
            [
                {
                    "f": m["gt_frame"],
                    "a": m.get("gt_action"),
                    "gp": m.get("gt_pid"),
                    "pp": m.get("pl_pid"),
                    "c": m["category"],
                }
                for m in rally.get("matches", [])
            ],
            separators=(",", ":"),
        )
        team_json = json.dumps(
            {str(k): v for k, v in team_assignments.items()},
            separators=(",", ":"),
        )
        cat_colors = {k: v["color"] for k, v in CATEGORY.items()}
        cat_labels = {k: v["label"] for k, v in CATEGORY.items()}
        team_palette = {k: v["color"] for k, v in TEAM_PALETTE.items()}
        swaps_json = json.dumps(primitive_audit.get("swap_events", []), separators=(",", ":"))

        # Build primitive-audit badges
        badges: list[str] = []
        swaps = primitive_audit.get("swap_events", [])
        if swaps:
            _rally_primitive_fail = True
            swap_frames = sorted({s["frame"] for s in swaps})[:5]
            badges.append(
                f"<span class='pbadge fail'>⚡ {len(swaps)} ID swap(s) "
                f"@ f{','.join(str(f) for f in swap_frames)}"
                f"{'…' if len(swaps) > len(swap_frames) else ''}</span>"
            )
        lf = primitive_audit.get("label_flip", {})
        cross_ratio = lf.get("cross_team_ratio", 0)
        if primitive_audit.get("team_flip_suspect"):
            pp = primitive_audit["positional_pairs"]
            tp = primitive_audit["team_pairs"]
            if cross_ratio > 0.3:
                _rally_primitive_fail = True
                badges.append(
                    f"<span class='pbadge fail'>⚠ team pair mismatch — "
                    f"positional {pp}, teamAssignments {tp}</span>"
                )
            else:
                # Cosmetic — teams physically wrong but attribution isn't affected.
                badges.append(
                    f"<span class='pbadge warn'>◐ cosmetic team mislabel — "
                    f"positional {pp}, teamAssignments {tp} "
                    f"(no attribution impact)</span>"
                )
        if lf.get("label_flip_suspect"):
            _rally_primitive_fail = True
            badges.append(
                f"<span class='pbadge fail'>🔄 label flip — "
                f"{lf.get('n_cross_team', 0)}/{lf.get('n_picked', 0)} picks cross-team "
                f"({cross_ratio * 100:.0f}%)</span>"
            )
        if not badges:
            badges.append("<span class='pbadge ok'>✓ primitives OK</span>")
        _primitive_badges = (
            "<div class='primitive-row'>" + "".join(badges) + "</div>"
        )
        clip_block = (
            f"<div class='clip-wrap' id='clip-wrap'>"
            f"<video id='rally-clip' controls preload='metadata' src='{clip_rel_path}'></video>"
            f"<canvas id='overlay'></canvas>"
            f"<div class='clip-banner' id='banner'></div>"
            f"</div>"
            f"<div class='timeline' id='timeline'>"
            f"{''.join(timeline_ticks)}"
            f"<div class='playhead' id='playhead'></div>"
            f"</div>"
            f"<script id='tracking-data' type='application/json'>{tracking_json}</script>"
            f"<script id='matches-data' type='application/json'>{matches_json}</script>"
            f"<script id='teams-data' type='application/json'>{team_json}</script>"
            f"<script id='swaps-data' type='application/json'>{swaps_json}</script>"
            f"<script>window.__CAT_COLORS = {json.dumps(cat_colors)};"
            f"window.__CAT_LABELS = {json.dumps(cat_labels)};"
            f"window.__TEAM_PALETTE = {json.dumps(team_palette)};"
            f"window.__FPS = {fps};</script>"
        )
    else:
        clip_block = (
            "<div style='padding:14px;background:#fff;border:1px dashed var(--line);"
            "border-radius:8px;color:var(--muted);font-size:13px;margin-bottom:18px;'>"
            "No clip extracted. Run <code>uv run python scripts/extract_attribution_clips.py</code> "
            "to enable the embedded video."
            "</div>"
        )

    js = r"""
<script>
(function(){
  var v = document.getElementById('rally-clip');
  if (!v) return;

  var trackingEl = document.getElementById('tracking-data');
  var matchesEl = document.getElementById('matches-data');
  var teamsEl = document.getElementById('teams-data');
  var tracking = trackingEl ? JSON.parse(trackingEl.textContent) : {positions: [], ball: []};
  var matches = matchesEl ? JSON.parse(matchesEl.textContent) : [];
  var teams = teamsEl ? JSON.parse(teamsEl.textContent) : {};
  var swapsEl = document.getElementById('swaps-data');
  var swaps = swapsEl ? JSON.parse(swapsEl.textContent) : [];
  var FPS = window.__FPS || 30.0;
  var CAT_COLORS = window.__CAT_COLORS || {};
  var CAT_LABELS = window.__CAT_LABELS || {};
  var TEAM_PAL = window.__TEAM_PALETTE || {};

  // Index positions by frame
  var posByFrame = {};
  for (var i = 0; i < tracking.positions.length; i++) {
    var p = tracking.positions[i];
    if (p.f == null) continue;
    (posByFrame[p.f] = posByFrame[p.f] || []).push(p);
  }
  var ballByFrame = {};
  for (var j = 0; j < tracking.ball.length; j++) {
    var b = tracking.ball[j];
    if (b.f != null) ballByFrame[b.f] = b;
  }

  var canvas = document.getElementById('overlay');
  var banner = document.getElementById('banner');
  var ctx = canvas.getContext('2d');

  function resizeCanvas() {
    var rect = v.getBoundingClientRect();
    canvas.width = rect.width * (window.devicePixelRatio || 1);
    canvas.height = rect.height * (window.devicePixelRatio || 1);
    canvas.style.width = rect.width + 'px';
    canvas.style.height = rect.height + 'px';
    ctx.setTransform(window.devicePixelRatio || 1, 0, 0, window.devicePixelRatio || 1, 0, 0);
  }
  v.addEventListener('loadedmetadata', resizeCanvas);
  v.addEventListener('loadeddata', resizeCanvas);
  window.addEventListener('resize', resizeCanvas);

  function teamColor(pid) {
    var t = teams[String(pid)];
    return TEAM_PAL[t] || '#888';
  }

  function drawBox(p, isContact, isError) {
    var W = parseFloat(canvas.style.width);
    var H = parseFloat(canvas.style.height);
    // yolo normalized bbox: x,y = center
    var cx = p.x * W, cy = p.y * H;
    var w = p.w * W, h = p.h * H;
    var x = cx - w/2, y = cy - h/2;
    var color = teamColor(p.t);
    ctx.lineWidth = isContact ? 4 : 2;
    ctx.strokeStyle = isError ? '#ff2222' : color;
    ctx.strokeRect(x, y, w, h);
    // Label background
    var team = teams[String(p.t)] || '?';
    var label = 'pid ' + p.t + ' · ' + team;
    ctx.font = 'bold 12px -apple-system, system-ui, sans-serif';
    var tw = ctx.measureText(label).width + 10;
    ctx.fillStyle = isError ? '#ff2222' : color;
    ctx.fillRect(x, y - 18, tw, 18);
    ctx.fillStyle = '#fff';
    ctx.fillText(label, x + 5, y - 5);
  }

  function drawBall(b) {
    if (!b) return;
    var W = parseFloat(canvas.style.width);
    var H = parseFloat(canvas.style.height);
    ctx.beginPath();
    ctx.arc(b.x * W, b.y * H, 7, 0, 2 * Math.PI);
    ctx.fillStyle = '#ffdb00';
    ctx.fill();
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  function updateBanner(frame) {
    // Show nearest action within ±3 frames
    var near = null, bestDist = 4;
    for (var k = 0; k < matches.length; k++) {
      var m = matches[k];
      var d = Math.abs(m.f - frame);
      if (d < bestDist) { near = m; bestDist = d; }
    }
    if (!near) { banner.style.display = 'none'; return {pid: null, errPid: null}; }
    var color = CAT_COLORS[near.c] || '#888';
    var lbl = CAT_LABELS[near.c] || near.c;
    var gtTeam = teams[String(near.gp)] || '?';
    var plTeam = teams[String(near.pp)] || '?';
    var isErr = near.c !== 'correct';
    var html = "<span class='cat' style='background:" + color + ";color:#fff;'>" + lbl + "</span>"
             + "<b>f" + near.f + "</b> · " + (near.a || '—') + "<br>"
             + "GT: pid " + near.gp + " (team " + gtTeam + ")";
    if (near.pp != null) {
      html += " &nbsp;|&nbsp; Pipeline: pid " + near.pp + " (team " + plTeam + ")";
    } else {
      html += " &nbsp;|&nbsp; <i>no pipeline pick</i>";
    }
    banner.innerHTML = html;
    banner.style.display = 'block';
    banner.style.borderLeftColor = color;
    return {pid: near.gp, errPid: isErr ? near.pp : null};
  }

  function render() {
    var t = v.currentTime;
    var frame = Math.round(t * FPS);
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    var hi = updateBanner(frame);
    var positionsHere = posByFrame[frame] || [];
    if (positionsHere.length === 0) {
      // nearest frame fallback
      for (var off = 1; off < 4 && positionsHere.length === 0; off++) {
        positionsHere = posByFrame[frame - off] || posByFrame[frame + off] || [];
      }
    }
    for (var i = 0; i < positionsHere.length; i++) {
      var p = positionsHere[i];
      drawBox(p, p.t === hi.pid, p.t === hi.errPid);
    }
    drawBall(ballByFrame[frame]
      || ballByFrame[frame - 1] || ballByFrame[frame + 1]
      || ballByFrame[frame - 2] || ballByFrame[frame + 2]);

    // Swap flash: highlight whole frame if within ±2 frames of any swap event
    for (var si = 0; si < swaps.length; si++) {
      var s = swaps[si];
      if (Math.abs(s.frame - frame) <= 2) {
        ctx.save();
        ctx.lineWidth = 6;
        ctx.strokeStyle = 'rgba(255, 0, 180, 0.8)';
        ctx.strokeRect(2, 2, canvas.width / (window.devicePixelRatio || 1) - 4,
                                canvas.height / (window.devicePixelRatio || 1) - 4);
        ctx.fillStyle = 'rgba(255, 0, 180, 0.9)';
        ctx.font = 'bold 13px -apple-system, system-ui, sans-serif';
        ctx.fillText('⚡ ID SWAP: pid ' + s.tid + ' jumped Δ=' + s.dist.toFixed(2), 12,
                     canvas.height / (window.devicePixelRatio || 1) - 14);
        ctx.restore();
        break;
      }
    }

    requestAnimationFrame(render);
  }
  requestAnimationFrame(render);

  // Seeking helpers
  function seekTo(s) {
    if (isFinite(s)) { v.currentTime = s; v.play().catch(function(){}); }
  }
  document.querySelectorAll('[data-seek]').forEach(function(el){
    el.addEventListener('click', function(ev){
      ev.preventDefault();
      seekTo(parseFloat(el.getAttribute('data-seek')));
      window.scrollTo({top: 0, behavior: 'smooth'});
    });
  });
  var ph = document.getElementById('playhead');
  var tl = document.getElementById('timeline');
  v.addEventListener('timeupdate', function(){
    if (ph && v.duration) ph.style.left = (v.currentTime / v.duration * 100) + '%';
  });
  if (tl) {
    tl.addEventListener('click', function(ev){
      if (ev.target.classList.contains('tick')) return;
      var rect = tl.getBoundingClientRect();
      var frac = (ev.clientX - rect.left) / rect.width;
      if (v.duration) seekTo(frac * v.duration);
    });
  }
  document.addEventListener('keydown', function(ev){
    if (ev.target.tagName === 'INPUT' || ev.target.tagName === 'TEXTAREA') return;
    if (ev.key === 'j') v.currentTime = Math.max(0, v.currentTime - 1);
    if (ev.key === 'l') v.currentTime = Math.min(v.duration, v.currentTime + 1);
    if (ev.key === ' ') { ev.preventDefault(); v.paused ? v.play() : v.pause(); }
  });
})();
</script>
"""

    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{_esc(fx)} · {rid[:8]}</title><style>{CSS}</style></head>
<body>
<a class="back" href="_index.html">← all {_esc(fx)} rallies</a>
<h1>{_esc(fx)} · rally {rid[:8]}</h1>
<div class="subtitle">
  frames: {rally.get('start_ms')}–{rally.get('end_ms')}ms &nbsp;·&nbsp;
  serving team: <b>{_esc(serving_team) if serving_team else '—'}</b> &nbsp;·&nbsp;
  primary tids: {ptid_note}
</div>

{_primitive_badges}

{clip_block}

<div class="summary-strip">
  {_fmt_delta(correct, n, CATEGORY['correct']['color'])}
  {_fmt_delta(wrong, n, CATEGORY['wrong_cross_team']['color'])}
  {_fmt_delta(missing, n, CATEGORY['missing']['color'])}
  {_fmt_delta(abstained, n, CATEGORY['abstained']['color']) if abstained else ''}
</div>

<h2>Roster</h2>
<div class="roster">{roster_chips}</div>

<h2>Errors ({len(errors)})</h2>
<div class='err-list'>
  {''.join(errors) if errors else '<div style="color:var(--muted); font-style: italic;">No errors on this rally. 🎉</div>'}
</div>

<details style="margin-top: 20px;">
  <summary>Show correct actions ({len(corrects)})</summary>
  <div class='err-list' style="margin-top: 10px;">{''.join(corrects)}</div>
</details>

<div class='nav'>
  {prev_link}
  <a href='_index.html'>↑ rally list</a>
  {next_link}
</div>
{js}
</body></html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc)
    return {
        "primitive_fail": _rally_primitive_fail,
        "audit": primitive_audit,
    }


def render_fixture_index(
    fixture: str,
    rallies: list[dict[str, Any]],
    out_path: Path,
    audits: dict[str, dict[str, Any]] | None = None,
) -> None:
    audits = audits or {}
    rows: list[str] = []
    totals = {
        "n_gt_actions": 0,
        "correct": 0,
        "wrong": 0,
        "missing": 0,
        "abstained": 0,
        "prim_fail": 0,
    }
    for r in rallies:
        t = r.get("rally_totals") or {}
        n = t.get("n_gt_actions", 0)
        w = sum(t.get(k, 0) for k in WRONG_CATEGORIES)
        audit = audits.get(r["rally_id"], {})
        pa = audit.get("audit", {})
        prim_fail = bool(audit.get("primitive_fail"))
        for key, add in (("n_gt_actions", n), ("correct", t.get("correct", 0)),
                         ("wrong", w), ("missing", t.get("missing", 0)),
                         ("abstained", t.get("abstained", 0))):
            totals[key] += add
        if prim_fail:
            totals["prim_fail"] += 1
        prim_cell = ""
        if pa:
            parts = []
            n_swaps = len(pa.get("swap_events", []))
            cross_r = pa.get("label_flip", {}).get("cross_team_ratio", 0)
            if n_swaps:
                parts.append(f"<span style='color:#c23b3b;'>⚡{n_swaps} swap</span>")
            if pa.get("team_flip_suspect"):
                if cross_r > 0.3:
                    parts.append("<span style='color:#c23b3b;'>⚠ team pair</span>")
                else:
                    parts.append("<span style='color:#a56611;'>◐ cosmetic</span>")
            if pa.get("label_flip", {}).get("label_flip_suspect"):
                parts.append("<span style='color:#c23b3b;'>🔄 label flip</span>")
            if not parts:
                parts.append("<span style='color:#2f7d2f;'>✓</span>")
            prim_cell = " · ".join(parts)
        row_class = "rally-row" + (" prim-fail-row" if prim_fail else "")
        rows.append(
            f"<tr class='{row_class}'>"
            f"<td><a href='{r['rally_id']}.html'>{r['rally_id'][:8]}</a></td>"
            f"<td class='num' style='color:var(--muted);'>{r.get('start_ms')}–{r.get('end_ms')}ms</td>"
            f"<td class='num'>{n}</td>"
            f"<td class='num' style='color:{CATEGORY['correct']['color']};'>{t.get('correct', 0)}</td>"
            f"<td class='num' style='color:{CATEGORY['wrong_cross_team']['color']}; font-weight: 600;'>{w}</td>"
            f"<td class='num' style='color:{CATEGORY['missing']['color']};'>{t.get('missing', 0)}</td>"
            f"<td class='prim-cell'>{prim_cell}</td>"
            "</tr>"
        )
    n = totals["n_gt_actions"] or 1
    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>{_esc(fixture)} rallies</title><style>{CSS}</style></head>
<body>
<a class="back" href="../index.html">← all fixtures</a>
<h1>{_esc(fixture)}</h1>
<div class="subtitle">{len(rallies)} rallies · {totals['n_gt_actions']} GT actions</div>

<div class="summary-strip">
  {_fmt_delta(totals['correct'], n, CATEGORY['correct']['color'])}
  {_fmt_delta(totals['wrong'], n, CATEGORY['wrong_cross_team']['color'])}
  {_fmt_delta(totals['missing'], n, CATEGORY['missing']['color'])}
</div>

<h2>Rallies (click to inspect)</h2>
<div class='subtitle' style='margin-bottom:10px;'>
  primitive failures: <b>{totals['prim_fail']} / {len(rallies)}</b> rallies flagged
</div>
<table class='rally-table'>
<thead><tr>
  <th>rally</th><th>range</th><th>n_gt</th>
  <th class='num'>correct</th><th class='num'>wrong</th><th class='num'>missing</th>
  <th>primitives</th>
</tr></thead>
<tbody>
{''.join(rows)}
</tbody></table>
</body></html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc)


def render_index(
    by_fx: dict[str, list[dict[str, Any]]],
    agg: dict[str, Any],
    out_path: Path,
    source_label: str,
    prim_fails_by_fx: dict[str, int] | None = None,
) -> None:
    prim_fails_by_fx = prim_fails_by_fx or {}
    combined = agg["combined"]
    cc = combined["counts"]
    cr = combined["rates"]
    cw = sum(cc[k] for k in WRONG_CATEGORIES)

    cards: list[str] = []
    for fx, rallies in sorted(by_fx.items()):
        fx_agg = agg["per_fixture"].get(fx, {"counts": {}, "rates": {}})
        c = fx_agg["counts"]
        rates = fx_agg["rates"]
        w = sum(c.get(k, 0) for k in WRONG_CATEGORIES)
        n = c.get("n_gt_actions", 0) or 1
        correct_pct = rates.get("correct_rate", 0) * 100
        wrong_pct = rates.get("wrong_rate", 0) * 100
        missing_pct = rates.get("missing_rate", 0) * 100
        abstain_pct = rates.get("abstained_rate", 0) * 100
        prim_fails = prim_fails_by_fx.get(fx, 0)
        prim_badge = (
            f"<span class='pbadge fail' style='font-size:11px;'>⚠ {prim_fails} primitive fail</span>"
            if prim_fails else "<span class='pbadge ok' style='font-size:11px;'>✓ primitives OK</span>"
        )
        cards.append(
            f"""<div class='fixture-card'>
              <h3><a href="{fx}/_index.html">{_esc(fx)}</a></h3>
              <div class="subtitle" style="margin: 0;">
                {len(rallies)} rallies · {c.get('n_gt_actions', 0)} GT actions
              </div>
              <div style="margin-top:6px;">{prim_badge}</div>
              <div class="rate-row">
                <span class="rate-val" style="color:{CATEGORY['correct']['color']};">{_fmt_pct(rates.get('correct_rate', 0))}</span>
                <span class="rate-label">correct</span>
              </div>
              <div class="rate-row">
                <span class="rate-val" style="color:{CATEGORY['wrong_cross_team']['color']};">{_fmt_pct(rates.get('wrong_rate', 0))}</span>
                <span class="rate-label">wrong</span>
                &nbsp;·&nbsp;
                <span style="color:var(--muted); font-size: 12px;">{c.get('wrong_cross_team', 0)} cross · {c.get('wrong_same_team', 0)} same · {c.get('wrong_unknown_team', 0)} unk</span>
              </div>
              <div class="rate-row">
                <span class="rate-val" style="color:{CATEGORY['missing']['color']};">{_fmt_pct(rates.get('missing_rate', 0))}</span>
                <span class="rate-label">missing</span>
              </div>
              <div class="bar">
                <div style="width:{correct_pct}%;background:{CATEGORY['correct']['color']};"></div>
                <div style="width:{wrong_pct}%;background:{CATEGORY['wrong_cross_team']['color']};"></div>
                <div style="width:{missing_pct}%;background:{CATEGORY['missing']['color']};"></div>
                <div style="width:{abstain_pct}%;background:{CATEGORY['abstained']['color']};"></div>
              </div>
            </div>"""
        )

    aggregate_bar = ""
    if cc.get("n_gt_actions", 0):
        correct_pct = cr.get("correct_rate", 0) * 100
        wrong_pct = cr.get("wrong_rate", 0) * 100
        missing_pct = cr.get("missing_rate", 0) * 100
        aggregate_bar = (
            f"<div class='bar' style='height: 12px;'>"
            f"<div style='width:{correct_pct}%;background:{CATEGORY['correct']['color']};' title='correct'></div>"
            f"<div style='width:{wrong_pct}%;background:{CATEGORY['wrong_cross_team']['color']};' title='wrong'></div>"
            f"<div style='width:{missing_pct}%;background:{CATEGORY['missing']['color']};' title='missing'></div>"
            f"</div>"
        )

    html_doc = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Attribution audit</title><style>{CSS}</style></head>
<body>
<h1>Attribution audit</h1>
<div class="subtitle">source: <code>{_esc(source_label)}</code></div>

<h2>Overall ({cc.get('n_gt_actions', 0)} GT actions)</h2>
<div class="summary-strip">
  {_fmt_delta(cc.get('correct', 0), cc.get('n_gt_actions', 0), CATEGORY['correct']['color'])}
  {_fmt_delta(cw, cc.get('n_gt_actions', 0), CATEGORY['wrong_cross_team']['color'])}
  {_fmt_delta(cc.get('missing', 0), cc.get('n_gt_actions', 0), CATEGORY['missing']['color'])}
</div>
{aggregate_bar}

<h2>Fixtures</h2>
<div class='fixture-grid'>
{''.join(cards)}
</div>
</body></html>
"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html_doc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    data = json.loads(args.input.read_text())
    rallies = data["rallies"]
    agg = aggregate(rallies)
    args.out.mkdir(parents=True, exist_ok=True)

    by_fx: dict[str, list[dict]] = {}
    for r in rallies:
        by_fx.setdefault(r["fixture"], []).append(r)

    # Two passes: render per-rally pages first (computes audits), then fixture
    # indexes + top-level (use the audits we just gathered).
    audits_by_fx: dict[str, dict[str, dict[str, Any]]] = {}
    prim_fails_by_fx: dict[str, int] = {}
    for fx, fx_rallies in by_fx.items():
        fx_dir = args.out / fx
        fx_dir.mkdir(parents=True, exist_ok=True)
        clips_dir = fx_dir / "clips"
        rids = [r["rally_id"] for r in fx_rallies]
        audits = {}
        for i, r in enumerate(fx_rallies):
            prev_rid = rids[i - 1] if i > 0 else None
            next_rid = rids[i + 1] if i < len(rids) - 1 else None
            result = render_rally(
                r, fx_dir / f"{r['rally_id']}.html", prev_rid, next_rid, clips_dir
            )
            audits[r["rally_id"]] = result
        audits_by_fx[fx] = audits
        prim_fails_by_fx[fx] = sum(1 for a in audits.values() if a.get("primitive_fail"))
        render_fixture_index(fx, fx_rallies, fx_dir / "_index.html", audits)

    render_index(
        by_fx, agg, args.out / "index.html",
        data.get("source", str(args.input)),
        prim_fails_by_fx,
    )
    # Dump combined audit to JSON for Phase-1 consumption
    audit_json_path = args.out / "primitive_audit.json"
    audit_json_path.write_text(json.dumps({
        "fixtures": {fx: {rid: {"primitive_fail": a["primitive_fail"],
                                "audit": a["audit"]}
                          for rid, a in aud.items()}
                     for fx, aud in audits_by_fx.items()},
        "prim_fails_by_fx": prim_fails_by_fx,
    }, indent=2))
    print(f"  primitive audit: {audit_json_path}")
    total_fails = sum(prim_fails_by_fx.values())
    total_rallies = sum(len(v) for v in by_fx.values())
    print(f"  primitives: {total_fails}/{total_rallies} rallies flagged")
    for fx in sorted(prim_fails_by_fx):
        print(f"    {fx:6s}: {prim_fails_by_fx[fx]}/{len(by_fx[fx])} flagged")
    print(f"wrote {args.out}/index.html")
    print(f"  {sum(len(v) for v in by_fx.values())} rally pages across {len(by_fx)} fixtures")
    return 0


if __name__ == "__main__":
    sys.exit(main())
