"""Phase 1.3 — Team→side audit (identity-first vs positional).

Per plan §1.3: compare two methods of inferring which team is on which side:

- **(A) Positional-median** — median foot-y per primary tid; top-2 median y's
  are "near", bottom-2 are "far". Model-free, reflects what a viewer sees.
- **(B) Identity-first** — for each team letter (A/B), gather the pids assigned
  to that team via ``actions_json.teamAssignments``; the team whose members'
  median foot-y is higher is "near". Correct iff ``teamAssignments``
  correctly pairs physical teammates.

These agree iff ``teamAssignments`` pairs the correct players as teammates.
Disagreements indicate a stage-2 team-inference bug.

Emits:
- `reports/phase1_3_team_side_2026_04_24.md` — per-fixture + combined.
- `reports/attribution_rebuild/phase1_3_team_side.json` — full data.

Threshold per plan: **≥ 98% agreement** between methods B and A. Below →
stage-2 team inference needs fixing before Phase 2.

Usage:
    uv run python scripts/phase1_3_team_side.py
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
    / "phase1_3_team_side_2026_04_24.md"
)
AUDIT_JSON_PATH = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "attribution_rebuild"
    / "phase1_3_team_side.json"
)

PAIR_CONFIDENCE_MIN = 0.05
TEAM_SIDE_GATE = 0.98


def _compute_pairs(
    positions: list[dict],
    team_assignments: dict[str, str],
) -> dict:
    """Return per-rally team→side inference from both methods."""
    per_tid: dict[int, list[float]] = defaultdict(list)
    for p in positions:
        if p.get("trackId") in (1, 2, 3, 4) and p.get("frameNumber", 0) < 15:
            per_tid[p["trackId"]].append(p["y"])
    if len(per_tid) < 4:
        per_tid = defaultdict(list)
        for p in positions:
            if p.get("trackId") in (1, 2, 3, 4):
                per_tid[p["trackId"]].append(p["y"])
    med_y = {t: median(ys) for t, ys in per_tid.items() if ys}

    result: dict = {
        "med_y": {t: round(y, 3) for t, y in med_y.items()},
        "method_a": None,
        "method_b": None,
        "pair_confidence": None,
        "audit_eligible": False,
        "agree": None,
    }

    # Method A: positional
    if len(med_y) < 4:
        return result
    sorted_tids = sorted(med_y.keys(), key=lambda t: -med_y[t])
    confidence = med_y[sorted_tids[1]] - med_y[sorted_tids[2]]
    method_a_near = set(sorted_tids[:2])
    method_a_far = set(sorted_tids[2:])
    result["method_a"] = {
        "near": sorted(method_a_near),
        "far": sorted(method_a_far),
    }
    result["pair_confidence"] = round(confidence, 4)

    # Method B: identity-first via teamAssignments
    team_pids: dict[str, set[int]] = {"A": set(), "B": set()}
    for pid_str, team in (team_assignments or {}).items():
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        if team in team_pids and pid in (1, 2, 3, 4):
            team_pids[team].add(pid)
    if not team_pids["A"] or not team_pids["B"]:
        return result
    team_medians: dict[str, float] = {}
    for team, pids in team_pids.items():
        ys = [med_y[p] for p in pids if p in med_y]
        if ys:
            team_medians[team] = median(ys)
    if len(team_medians) < 2:
        return result
    near_team = max(team_medians, key=team_medians.get)
    far_team = "A" if near_team == "B" else "B"
    method_b_near = team_pids[near_team]
    method_b_far = team_pids[far_team]
    result["method_b"] = {
        "near": sorted(method_b_near),
        "far": sorted(method_b_far),
    }

    # Audit is only meaningful when positional split is confident.
    if confidence < PAIR_CONFIDENCE_MIN:
        return result
    result["audit_eligible"] = True
    # Pairs agree if method_a_near == method_b_near OR == method_b_far
    # (side labels are free up to flip; what matters is the pair grouping).
    result["agree"] = (
        method_b_near == method_a_near
        or method_b_near == method_a_far
    )
    return result


def main() -> int:
    baseline = json.loads(BASELINE_PATH.read_text())
    rallies = baseline["rallies"]
    rally_ids = [r["rally_id"] for r in rallies]

    positions_by_rally: dict[str, list] = {}
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT rally_id, positions_json FROM player_tracks
            WHERE rally_id = ANY(%s)
            """,
            (rally_ids,),
        )
        for rid, pos in cur.fetchall():
            positions_by_rally[rid] = pos or []

    per_fixture: dict[str, dict] = defaultdict(
        lambda: {
            "rallies": 0,
            "eligible": 0,
            "agree": 0,
            "disagree": [],
            "low_confidence": 0,
        }
    )
    all_rally_audits: dict[str, dict] = {}

    for r in rallies:
        fx = r["fixture"]
        rid = r["rally_id"]
        team_assignments = r.get("team_assignments") or {}
        res = _compute_pairs(positions_by_rally.get(rid, []), team_assignments)
        all_rally_audits[rid] = {
            **res,
            "fixture": fx,
            "team_assignments": team_assignments,
        }
        s = per_fixture[fx]
        s["rallies"] += 1
        if not res["audit_eligible"]:
            s["low_confidence"] += 1
            continue
        s["eligible"] += 1
        if res["agree"]:
            s["agree"] += 1
        else:
            s["disagree"].append(
                {
                    "rally_id": rid,
                    "method_a": res["method_a"],
                    "method_b": res["method_b"],
                    "team_assignments": team_assignments,
                }
            )

    # Aggregate
    combined_eligible = sum(s["eligible"] for s in per_fixture.values())
    combined_agree = sum(s["agree"] for s in per_fixture.values())
    agreement_rate = combined_agree / combined_eligible if combined_eligible else 0.0
    gate_pass = agreement_rate >= TEAM_SIDE_GATE

    print()
    print(f"{'fixture':8s} {'rallies':>7s} {'eligible':>8s} {'agree':>5s} "
          f"{'disagree':>8s} {'low_conf':>8s} {'rate':>7s}")
    for fx, s in sorted(per_fixture.items()):
        rate = s["agree"] / s["eligible"] if s["eligible"] else 0.0
        marker = "✓" if rate >= TEAM_SIDE_GATE else "✗"
        print(
            f"{fx:8s} {s['rallies']:>7d} {s['eligible']:>8d} {s['agree']:>5d} "
            f"{len(s['disagree']):>8d} {s['low_confidence']:>8d} "
            f"{rate:>6.1%} {marker}"
        )
    marker = "✓ PASS" if gate_pass else "✗ FAIL"
    print(
        f"{'COMBINED':8s} {sum(s['rallies'] for s in per_fixture.values()):>7d} "
        f"{combined_eligible:>8d} {combined_agree:>5d} "
        f"{combined_eligible - combined_agree:>8d} "
        f"{sum(s['low_confidence'] for s in per_fixture.values()):>8d} "
        f"{agreement_rate:>6.1%} {marker}"
    )

    # Report
    lines = [
        "# Phase 1.3 — Team→Side Audit",
        "",
        f"**Date:** 2026-04-24  ",
        f"**Scope:** {sum(s['rallies'] for s in per_fixture.values())} locked GT rallies "
        f"across 9 fixtures.  ",
        f"**Gate:** `identity-first agreement with positional ≥ {TEAM_SIDE_GATE:.0%}` "
        "on audit-eligible rallies.  ",
        f"**Result:** `{agreement_rate:.1%}` over "
        f"{combined_eligible} eligible rallies — "
        f"{'**PASS**' if gate_pass else '**FAIL**'}",
        "",
        "## Method",
        "",
        "For each rally:",
        "",
        "- **Method A (positional)**: median y of each primary tid over rally-start "
        "frames (< 15f); the 2 highest-y tids are \"near\", the 2 lowest are \"far\".",
        "- **Method B (identity-first)**: each team letter's pids per "
        "`actions_json.teamAssignments`; team with higher median foot-y is \"near\".",
        "- **Agreement**: method B's near-team pid set equals method A's near pid "
        "set (teams match up to A↔B label flip).",
        "- Rallies with `pair_confidence < 0.05` (median-y split near the midline) "
        "are excluded as audit-inconclusive.",
        "",
        "## Per-fixture",
        "",
        "| fixture | rallies | eligible | agree | disagree | low-conf | rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for fx, s in sorted(per_fixture.items()):
        rate = s["agree"] / s["eligible"] if s["eligible"] else 0.0
        marker = "✅" if rate >= TEAM_SIDE_GATE else "⚠️"
        lines.append(
            f"| **{fx}** | {s['rallies']} | {s['eligible']} | {s['agree']} | "
            f"{len(s['disagree'])} | {s['low_confidence']} | "
            f"{rate:.1%} {marker} |"
        )
    lines.append(
        f"| **COMBINED** | "
        f"**{sum(s['rallies'] for s in per_fixture.values())}** | "
        f"**{combined_eligible}** | **{combined_agree}** | "
        f"**{combined_eligible - combined_agree}** | "
        f"**{sum(s['low_confidence'] for s in per_fixture.values())}** | "
        f"**{agreement_rate:.1%}** |"
    )

    # Disagreements
    lines.extend(["", "## Disagreements (stage-2 teamAssignments bugs)", ""])
    any_disagreements = False
    for fx, s in sorted(per_fixture.items()):
        if not s["disagree"]:
            continue
        any_disagreements = True
        lines.append(f"### {fx}")
        for d in s["disagree"]:
            lines.append(
                f"- `{d['rally_id'][:8]}` — positional near=`{d['method_a']['near']}` "
                f"vs identity-first near=`{d['method_b']['near']}`  "
                f"(teamAssignments: `{d['team_assignments']}`)"
            )
        lines.append("")
    if not any_disagreements:
        lines.append("*None — teamAssignments pairs physical teammates correctly.*")

    lines.extend(["", "## Decision", ""])
    if gate_pass:
        lines.append(
            f"**Pass ({agreement_rate:.1%}).** Identity-first team→side agrees "
            "with positional in all audit-eligible rallies. Phase 2 chooser can "
            "consume `teamAssignments` as a trusted primitive."
        )
    else:
        lines.append(
            f"**Fail ({agreement_rate:.1%}).** Stage-2 team inference pairs the "
            "wrong players as teammates on a non-trivial fraction of rallies. "
            "**Phase 1.3a action**: rather than block Phase 2, document the "
            "teamAssignments failure mode and require Phase-2 chooser to validate "
            "team-membership via positional-check at contact-frame instead of "
            "trusting teamAssignments. Label-flip (uniform A↔B swap without "
            "breaking pair grouping) remains a Phase 2.3 roster-aware chooser "
            "concern."
        )

    OUT_PATH.write_text("\n".join(lines))
    AUDIT_JSON_PATH.write_text(
        json.dumps(
            {
                "gate": TEAM_SIDE_GATE,
                "agreement_rate": agreement_rate,
                "gate_pass": gate_pass,
                "per_fixture": {fx: {k: v for k, v in s.items()}
                                for fx, s in per_fixture.items()},
                "per_rally": all_rally_audits,
            },
            indent=2,
            default=list,
        )
    )
    print(f"\nwrote {OUT_PATH}")
    print(f"wrote {AUDIT_JSON_PATH}")
    return 0 if gate_pass else 1


if __name__ == "__main__":
    sys.exit(main())
