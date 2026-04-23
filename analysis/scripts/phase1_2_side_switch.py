"""Phase 1.2 — Side-switch reliability audit.

Reads `videos.match_analysis_json` + baseline positions per fixture. For each
rally, extracts the pipeline's `sideSwitchDetected` boolean and checks whether
it correlates with observed positional team layout change relative to the
previous rally.

Emits:
- `reports/phase1_2_side_switch_2026_04_24.md` — per-fixture summary + user
  audit table (rallies where sideSwitchDetected=true + their prev/next
  positional pair, for eyeballing).
- `reports/attribution_rebuild/phase1_2_side_switch.json` — full data for
  Phase-2 chooser consumption.

Side-switch reliability threshold per plan: **≥98%**. Events are rare
(~1 per match in beach VB), so mistakes should be near-zero.

Usage:
    uv run python scripts/phase1_2_side_switch.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import median

from rallycut.evaluation.db import get_connection

FIXTURE_MAP_PATH = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "attribution_rebuild"
    / "fixture_video_ids_2026_04_24.json"
)
OUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "phase1_2_side_switch_2026_04_24.md"
)
AUDIT_JSON_PATH = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "attribution_rebuild"
    / "phase1_2_side_switch.json"
)

PAIR_CONFIDENCE_MIN = 0.05
SIDE_SWITCH_GATE = 0.98


def _start_pair(positions: list[dict]) -> tuple[set[int], set[int], float] | None:
    """Return (near_pids, far_pids, confidence). Uses first 15 frames."""
    per_tid: dict[int, list[float]] = defaultdict(list)
    for p in positions:
        if p.get("trackId") in (1, 2, 3, 4) and p.get("frameNumber", 0) < 15:
            per_tid[p["trackId"]].append(p["y"])
    if len(per_tid) < 4:
        # Fall back to all frames
        per_tid = defaultdict(list)
        for p in positions:
            if p.get("trackId") in (1, 2, 3, 4):
                per_tid[p["trackId"]].append(p["y"])
    if len(per_tid) < 4:
        return None
    med_y = {t: median(ys) for t, ys in per_tid.items() if ys}
    sorted_tids = sorted(med_y.keys(), key=lambda t: -med_y[t])
    confidence = med_y[sorted_tids[1]] - med_y[sorted_tids[2]]
    return (set(sorted_tids[:2]), set(sorted_tids[2:]), confidence)


def main() -> int:
    fixture_map = json.loads(FIXTURE_MAP_PATH.read_text())["fixtures"]

    per_fixture: dict[str, dict] = {}
    per_fixture_disagreements: dict[str, list[dict]] = defaultdict(list)

    with get_connection() as conn, conn.cursor() as cur:
        for fx_name, info in fixture_map.items():
            vid = info["video_id"]
            cur.execute(
                "SELECT match_analysis_json FROM videos WHERE id = %s",
                (vid,),
            )
            (ma,) = cur.fetchone()
            if not ma or "rallies" not in ma:
                print(f"[{fx_name}] no match_analysis_json")
                continue
            rallies_in_order = sorted(
                ma["rallies"], key=lambda r: r.get("startMs", 0)
            )

            # Pull positions per rally (only GT rallies matter for audit,
            # but we need all rallies for side-switch state tracking).
            rally_ids = [r["rallyId"] for r in rallies_in_order]
            cur.execute(
                "SELECT rally_id, positions_json FROM player_tracks "
                "WHERE rally_id = ANY(%s)",
                (rally_ids,),
            )
            positions_by_rally = {rid: pos or [] for rid, pos in cur.fetchall()}

            n_total = len(rallies_in_order)
            n_flagged = 0
            switch_events: list[dict] = []
            prev_pair: tuple[set[int], set[int], float] | None = None
            prev_rally_id: str | None = None

            for rally_record in rallies_in_order:
                rid = rally_record["rallyId"]
                flagged = bool(rally_record.get("sideSwitchDetected", False))
                pair = _start_pair(positions_by_rally.get(rid, []))

                if flagged:
                    n_flagged += 1
                    # Sanity check: compare to previous rally's pair
                    disagreement = None
                    if prev_pair is not None and pair is not None:
                        prev_near, prev_far, prev_conf = prev_pair
                        curr_near, curr_far, curr_conf = pair
                        # A real side switch means near-set flipped to far-set
                        # (and vice versa). No switch means near stays near.
                        if prev_conf >= PAIR_CONFIDENCE_MIN and curr_conf >= PAIR_CONFIDENCE_MIN:
                            flipped = prev_near == curr_far and prev_far == curr_near
                            same = prev_near == curr_near
                            if same and not flipped:
                                disagreement = {
                                    "type": "switch_flag_but_same_side",
                                    "prev_rally": prev_rally_id,
                                    "prev_near": sorted(prev_near),
                                    "curr_near": sorted(curr_near),
                                }
                    if disagreement:
                        per_fixture_disagreements[fx_name].append(
                            {"rally_id": rid, **disagreement}
                        )
                    switch_events.append(
                        {
                            "rally_id": rid,
                            "rally_index": rally_record.get("rallyIndex"),
                            "prev_near": sorted(prev_pair[0]) if prev_pair else None,
                            "curr_near": sorted(pair[0]) if pair else None,
                            "prev_conf": round(prev_pair[2], 3) if prev_pair else None,
                            "curr_conf": round(pair[2], 3) if pair else None,
                        }
                    )

                # Also detect the inverse: positional flip without flag.
                if not flagged and prev_pair is not None and pair is not None:
                    prev_near, _, prev_conf = prev_pair
                    curr_near, _, curr_conf = pair
                    if prev_conf >= PAIR_CONFIDENCE_MIN and curr_conf >= PAIR_CONFIDENCE_MIN:
                        flipped = prev_near != curr_near and len(prev_near & curr_near) == 0
                        if flipped:
                            per_fixture_disagreements[fx_name].append(
                                {
                                    "rally_id": rid,
                                    "type": "position_flip_no_flag",
                                    "prev_rally": prev_rally_id,
                                    "prev_near": sorted(prev_near),
                                    "curr_near": sorted(curr_near),
                                }
                            )

                prev_pair = pair
                prev_rally_id = rid

            per_fixture[fx_name] = {
                "n_rallies": n_total,
                "n_flagged": n_flagged,
                "switch_events": switch_events,
                "n_disagreements": len(per_fixture_disagreements[fx_name]),
                "disagreements": per_fixture_disagreements[fx_name],
            }

    # Summary
    print()
    print(f"{'fixture':8s} {'rallies':>7s} {'flagged':>7s} {'disagree':>8s}")
    total_flagged = 0
    total_disagree = 0
    total_rallies = 0
    for fx, s in sorted(per_fixture.items()):
        total_flagged += s["n_flagged"]
        total_disagree += s["n_disagreements"]
        total_rallies += s["n_rallies"]
        print(
            f"{fx:8s} {s['n_rallies']:>7d} {s['n_flagged']:>7d} {s['n_disagreements']:>8d}"
        )
    agreement_rate = 1 - (total_disagree / max(total_rallies, 1))
    marker = "✓ PASS" if agreement_rate >= SIDE_SWITCH_GATE else "✗ FAIL"
    print(
        f"{'TOTAL':8s} {total_rallies:>7d} {total_flagged:>7d} {total_disagree:>8d}  "
        f"agreement={agreement_rate:.2%} {marker}"
    )

    # Report
    lines = [
        "# Phase 1.2 — Side-Switch Reliability Audit",
        "",
        f"**Date:** 2026-04-24  ",
        f"**Scope:** all rallies across 9 fixtures (not just GT-locked ones — "
        f"side-switch state depends on full per-match history).  ",
        f"**Gate:** `agreement_rate ≥ {SIDE_SWITCH_GATE:.0%}`  ",
        f"**Result:** `{agreement_rate:.2%}` — "
        f"{'**PASS**' if agreement_rate >= SIDE_SWITCH_GATE else '**FAIL**'}",
        "",
        "## Method",
        "",
        "- Extract `sideSwitchDetected` boolean from "
        "`videos.match_analysis_json.rallies[*]` per rally, in time order.",
        "- Compute positional near/far pair per rally from rally-start positions "
        "(first 15 frames, median y-coord grouping).",
        "- **Disagreement types counted:**",
        "  - `switch_flag_but_same_side` — pipeline claims a switch but "
        "positional near-pair is unchanged from previous rally.",
        "  - `position_flip_no_flag` — pipeline says no switch, but the "
        "positional near-pair fully flipped.",
        "",
        "## Per-fixture",
        "",
        "| fixture | rallies | flagged | disagreements |",
        "|---|---|---|---|",
    ]
    for fx, s in sorted(per_fixture.items()):
        lines.append(
            f"| {fx} | {s['n_rallies']} | {s['n_flagged']} | "
            f"{s['n_disagreements']} |"
        )
    lines.append(
        f"| **TOTAL** | **{total_rallies}** | **{total_flagged}** | "
        f"**{total_disagree}** ({agreement_rate:.2%} agreement) |"
    )

    # Disagreement detail
    lines.extend(["", "## Disagreements (user audit needed)", ""])
    any_disagreements = False
    for fx, dlist in sorted(per_fixture_disagreements.items()):
        if not dlist:
            continue
        any_disagreements = True
        lines.append(f"### {fx}")
        for d in dlist:
            lines.append(
                f"- `{d['rally_id'][:8]}` — **{d['type']}**: "
                f"prev rally `{(d.get('prev_rally') or '—')[:8]}` "
                f"near={d.get('prev_near')} → curr near={d.get('curr_near')}"
            )
        lines.append("")
    if not any_disagreements:
        lines.append("*None — pipeline side-switch state agrees with positional evidence.*")

    lines.extend(
        [
            "",
            "## Decision",
            "",
        ]
    )
    if agreement_rate >= SIDE_SWITCH_GATE:
        lines.append(
            f"**Pass ({agreement_rate:.2%}).** Pipeline's `sideSwitchDetected` "
            "boolean is internally consistent with observed team positions. "
            "Phase 1.3 (team→side audit) can lean on side-switch state as a "
            "trusted primitive input."
        )
    else:
        lines.append(
            f"**Fail ({agreement_rate:.2%}).** Side-switch state is unreliable "
            "on {total_disagree}/{total_rallies} rallies. Phase 1.3 must compute "
            "team→side via identity-first with independent side inference, "
            "not by consuming `sideSwitchDetected`."
        )

    OUT_PATH.write_text("\n".join(lines))
    AUDIT_JSON_PATH.write_text(
        json.dumps(
            {
                "gate": SIDE_SWITCH_GATE,
                "agreement_rate": agreement_rate,
                "pass": agreement_rate >= SIDE_SWITCH_GATE,
                "per_fixture": per_fixture,
            },
            indent=2,
            default=list,
        )
    )
    print(f"\nwrote {OUT_PATH}")
    print(f"wrote {AUDIT_JSON_PATH}")
    return 0 if agreement_rate >= SIDE_SWITCH_GATE else 1


if __name__ == "__main__":
    sys.exit(main())
