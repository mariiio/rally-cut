"""Phase 1.4 — Ball court_side per contact audit.

For each cross_team_wrong contact in the baseline, compute ball-side 3 ways and
compare to the GT actor's actual side (foot projection).

Methods:
- **(A) Instantaneous**: pipeline's `contacts_json.contacts[i].courtSide` at
  contact frame — today's production signal.
- **(B) Trajectory median**: median ball-y over ±3 frames around contact →
  "near" if median > 0.5 (midline).
- **(C) Net-line crossing**: whether ball trajectory crossed the net line
  (vertical midline in image space) in the 10 frames before contact.

"Actual side" = GT actor's foot-y (bbox center_y + height/2) at the contact
frame compared to midline.

Emits:
- `reports/phase1_4_ball_court_side_2026_04_24.md` — method comparison.
- `reports/attribution_rebuild/phase1_4_ball_court_side.json` — full data.

Gate: whichever method wins ≥95% match with actual side is adopted. Phase 2
chooser consumes the winner.

Usage:
    uv run python scripts/phase1_4_ball_court_side.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

from rallycut.evaluation.db import get_connection

BASELINE_PATH = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "attribution_rebuild"
    / "baseline_2026_04_24.json"
)
OUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "phase1_4_ball_court_side_2026_04_24.md"
)
AUDIT_JSON_PATH = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "attribution_rebuild"
    / "phase1_4_ball_court_side.json"
)

METHOD_GATE = 0.95
WINDOW = 3


def _side_from_y(y: float | None, midline: float) -> str | None:
    if y is None:
        return None
    return "near" if y > midline else "far"


def _compute_midline(positions: list[dict]) -> float:
    """Per-rally adaptive midline: median foot-y across all 4 players' rally-start
    positions. Players straddle the midline; the median of their feet is the
    natural net-level in image space."""
    foot_ys: list[float] = []
    for p in positions:
        if p.get("trackId") in (1, 2, 3, 4) and p.get("frameNumber", 0) < 15:
            foot_ys.append(p["y"] + p.get("height", 0) / 2)
    if len(foot_ys) < 4:
        # Fallback: all frames
        foot_ys = [
            p["y"] + p.get("height", 0) / 2
            for p in positions
            if p.get("trackId") in (1, 2, 3, 4)
        ]
    return median(foot_ys) if foot_ys else 0.5


def _method_a(contact: dict) -> str | None:
    """Pipeline's instantaneous courtSide field."""
    return contact.get("courtSide")  # "near" / "far" / None


def _method_b(
    ball_positions: list[dict], frame: int, midline: float
) -> str | None:
    """Trajectory-window median ball-y ±WINDOW frames."""
    ys = [
        b["y"]
        for b in ball_positions
        if b.get("frameNumber") is not None
        and abs(b["frameNumber"] - frame) <= WINDOW
    ]
    if not ys:
        return None
    return _side_from_y(median(ys), midline)


def _method_c(
    ball_positions: list[dict], frame: int, midline: float
) -> str | None:
    """Net-line crossing: did the ball cross midline in the 10 frames before
    contact? Return the side ball is on at the contact frame if it crossed.
    """
    window = [
        b for b in ball_positions
        if b.get("frameNumber") is not None
        and frame - 10 <= b["frameNumber"] <= frame
    ]
    if len(window) < 3:
        return None
    window.sort(key=lambda b: b["frameNumber"])
    sides = [_side_from_y(b["y"], midline) for b in window]
    crossed = any(
        s0 is not None and s1 is not None and s0 != s1
        for s0, s1 in zip(sides, sides[1:])
    )
    if not crossed:
        return None
    closest = min(window, key=lambda b: abs(b["frameNumber"] - frame))
    return _side_from_y(closest["y"], midline)


def _actual_side(
    positions: list[dict],
    gt_pid: int,
    frame: int,
    midline: float,
) -> str | None:
    """Foot-y of GT actor at contact frame vs per-rally midline."""
    candidates = [
        p for p in positions
        if p.get("trackId") == gt_pid
        and p.get("frameNumber") is not None
        and abs(p["frameNumber"] - frame) <= 3
    ]
    if not candidates:
        return None
    best = min(candidates, key=lambda p: abs(p["frameNumber"] - frame))
    foot_y = best["y"] + best.get("height", 0) / 2
    return _side_from_y(foot_y, midline)


def main() -> int:
    baseline = json.loads(BASELINE_PATH.read_text())
    rallies = baseline["rallies"]
    rally_ids = [r["rally_id"] for r in rallies]

    tracking_by_rally: dict[str, tuple[list, list]] = {}
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT rally_id, positions_json, ball_positions_json "
            "FROM player_tracks WHERE rally_id = ANY(%s)",
            (rally_ids,),
        )
        for rid, pos, ball in cur.fetchall():
            tracking_by_rally[rid] = (pos or [], ball or [])

    # For each wrong_cross_team match, compute all three methods + actual.
    # Also measure on correct matches as a sanity baseline (should agree ~100%).
    per_method_stats: dict[str, dict[str, int]] = {
        "A": {"eligible": 0, "match_actual": 0},
        "B": {"eligible": 0, "match_actual": 0},
        "C": {"eligible": 0, "match_actual": 0},
    }
    rows: list[dict] = []
    for r in rallies:
        positions, ball = tracking_by_rally.get(r["rally_id"], ([], []))
        contacts = r.get("pipeline_contacts", [])
        midline = _compute_midline(positions)
        for m in r.get("matches", []):
            cat = m["category"]
            if cat not in ("wrong_cross_team", "correct"):
                continue
            gt_frame = m["gt_frame"]
            gt_pid = m["gt_pid"]
            pl_frame = m.get("pl_frame")

            pipeline_contact = next(
                (c for c in contacts if c.get("frame") == pl_frame),
                None,
            ) or {}

            sa = _method_a(pipeline_contact)
            sb = _method_b(ball, pl_frame if pl_frame is not None else gt_frame, midline)
            sc = _method_c(ball, pl_frame if pl_frame is not None else gt_frame, midline)
            actual = _actual_side(positions, gt_pid, gt_frame, midline)

            row = {
                "rally_id": r["rally_id"],
                "fixture": r["fixture"],
                "gt_frame": gt_frame,
                "gt_pid": gt_pid,
                "category": cat,
                "midline": round(midline, 3),
                "method_a": sa,
                "method_b": sb,
                "method_c": sc,
                "actual": actual,
            }
            rows.append(row)

            if actual is not None:
                if sa is not None:
                    per_method_stats["A"]["eligible"] += 1
                    if sa == actual:
                        per_method_stats["A"]["match_actual"] += 1
                if sb is not None:
                    per_method_stats["B"]["eligible"] += 1
                    if sb == actual:
                        per_method_stats["B"]["match_actual"] += 1
                if sc is not None:
                    per_method_stats["C"]["eligible"] += 1
                    if sc == actual:
                        per_method_stats["C"]["match_actual"] += 1

    # Also separate stats for just the cross_team_wrong subset
    def _stats_on(rows_subset: list[dict]) -> dict[str, dict[str, int]]:
        stats = {m: {"eligible": 0, "match_actual": 0} for m in ("A", "B", "C")}
        for r in rows_subset:
            if r["actual"] is None:
                continue
            for key, val in (("A", r["method_a"]), ("B", r["method_b"]),
                             ("C", r["method_c"])):
                if val is None:
                    continue
                stats[key]["eligible"] += 1
                if val == r["actual"]:
                    stats[key]["match_actual"] += 1
        return stats

    subset_wrong = [r for r in rows if r["category"] == "wrong_cross_team"]
    subset_correct = [r for r in rows if r["category"] == "correct"]
    stats_wrong = _stats_on(subset_wrong)
    stats_correct = _stats_on(subset_correct)

    def _rate(d: dict[str, int]) -> float:
        return d["match_actual"] / d["eligible"] if d["eligible"] else 0.0

    print()
    print("=== ALL matches (wrong_cross_team + correct) ===")
    for m in "ABC":
        s = per_method_stats[m]
        print(f"  Method {m}: {s['match_actual']}/{s['eligible']} ({_rate(s):.1%})")
    print()
    print("=== CROSS_TEAM_WRONG subset ===")
    for m in "ABC":
        s = stats_wrong[m]
        print(f"  Method {m}: {s['match_actual']}/{s['eligible']} ({_rate(s):.1%})")
    print()
    print("=== CORRECT subset (should be near 100%) ===")
    for m in "ABC":
        s = stats_correct[m]
        print(f"  Method {m}: {s['match_actual']}/{s['eligible']} ({_rate(s):.1%})")

    best_method = max("ABC", key=lambda m: _rate(per_method_stats[m]))
    best_rate = _rate(per_method_stats[best_method])
    gate_pass = best_rate >= METHOD_GATE

    # Report
    lines = [
        "# Phase 1.4 — Ball Court-Side Audit",
        "",
        f"**Date:** 2026-04-24  ",
        f"**Scope:** all `correct` + `wrong_cross_team` matches from baseline, "
        "measured against GT-actor foot projection as ground truth.  ",
        f"**Gate:** best method ≥ {METHOD_GATE:.0%}  ",
        f"**Winner:** Method **{best_method}** at {best_rate:.1%} — "
        f"{'**PASS**' if gate_pass else '**FAIL**'}",
        "",
        "## Methods",
        "",
        "- **(A) Instantaneous**: pipeline's `contacts_json.contacts[i].courtSide`.",
        f"- **(B) Trajectory median**: median ball-y over ±{WINDOW} frames vs midline 0.5.",
        "- **(C) Net-line crossing**: explicit midline crossing in 10f pre-contact.",
        "",
        "*Ground truth* = GT actor's foot-y (`position.y + height/2`) vs midline 0.5.",
        "",
        "## Results (all matches)",
        "",
        "| method | eligible | match actual | rate |",
        "|---|---|---|---|",
    ]
    for m in "ABC":
        s = per_method_stats[m]
        marker = "✅" if _rate(s) >= METHOD_GATE else "⚠️"
        lines.append(
            f"| **{m}** | {s['eligible']} | {s['match_actual']} | "
            f"{_rate(s):.1%} {marker} |"
        )

    lines.extend(["", "## Results (wrong_cross_team subset, 48 actions)", "",
                  "| method | eligible | match actual | rate |",
                  "|---|---|---|---|"])
    for m in "ABC":
        s = stats_wrong[m]
        lines.append(
            f"| **{m}** | {s['eligible']} | {s['match_actual']} | {_rate(s):.1%} |"
        )

    lines.extend(["", "## Results (correct subset — sanity, should be high)", "",
                  "| method | eligible | match actual | rate |",
                  "|---|---|---|---|"])
    for m in "ABC":
        s = stats_correct[m]
        lines.append(
            f"| **{m}** | {s['eligible']} | {s['match_actual']} | {_rate(s):.1%} |"
        )

    lines.extend(["", "## Decision", ""])
    if gate_pass:
        lines.append(
            f"**Method {best_method} wins at {best_rate:.1%} ≥ {METHOD_GATE:.0%}.** "
            "Phase 2 chooser adopts this for ball-side inference. The signal "
            "alone does not solve attribution (memory `attribution_ballside_oracle"
            "_2026_04_23.md` showed +3.82pp ceiling) but is a trusted primitive "
            "for Phase 3 cross-checks."
        )
    else:
        lines.append(
            f"**Best method {best_method} at {best_rate:.1%} < {METHOD_GATE:.0%}.** "
            "No method is reliable enough to be a single-source-of-truth for "
            "ball side. Phase 2 chooser must not rely on ball-side as a hard "
            "gate; treat as soft evidence only."
        )

    OUT_PATH.write_text("\n".join(lines))
    AUDIT_JSON_PATH.write_text(
        json.dumps(
            {
                "gate": METHOD_GATE,
                "best_method": best_method,
                "best_rate": best_rate,
                "gate_pass": gate_pass,
                "stats_all": per_method_stats,
                "stats_wrong": stats_wrong,
                "stats_correct": stats_correct,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"\nwrote {OUT_PATH}")
    print(f"wrote {AUDIT_JSON_PATH}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
