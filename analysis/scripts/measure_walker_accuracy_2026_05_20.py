#!/usr/bin/env python3
"""Stage-1 walker-decision accuracy replay (no scorer interaction).

Reads the 1265-pair GT dataset from probe_gt_net_crossings_2026_05_20
(events.csv) and replays each of 4 ChainWalkerConfig combinations
against it:

  cfg_00: both flags OFF (= v13 baseline)
  cfg_10: B.1 ON, B.2 OFF
  cfg_01: B.1 OFF, B.2 ON
  cfg_11: both ON

For each config, recomputes the walker's flip decision per pair and
tallies correct_flip / missed_flip / correct_stay / over_flip. Compares
against v13 baseline (482 / 64 / 699 / 20).

CAVEAT: The events.csv does NOT carry production Contact.court_side
(which is derived from ball Y vs net_y at the contact frame). For
Stage-1 replay, we reconstruct court_side from team identity
(team A = near, team B = far) — this is the GT-derived signal that
DEFINES whether the ball crossed. As a result, B.2 (which uses
court_side as override signal) will appear perfect on this substrate
by construction. The realistic-vs-production gap for B.2 lives in
Stage-2's end-to-end A/B, which uses real Contact.court_side from
the live pipeline. cfg_10 (B.1 only) and cfg_00 baseline are not
affected by this caveat since they don't consume court_side.

Output:
  reports/walker_accuracy_2026_05_20/summary.json
  reports/walker_accuracy_2026_05_20/summary.md
  Console: side-by-side table.
"""
from __future__ import annotations

import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ANALYSIS_DIR))

from rallycut.tracking.action_classifier import (  # noqa: E402
    ActionType,
    ChainWalkerConfig,
    _possession_flips_after,
)

IN_CSV = ANALYSIS_DIR / "reports" / "gt_net_crossings_2026_05_20" / "events.csv"
OUT_DIR = ANALYSIS_DIR / "reports" / "walker_accuracy_2026_05_20"
OUT_DIR.mkdir(parents=True, exist_ok=True)


_STR_TO_ACTION_TYPE = {
    "serve":   ActionType.SERVE,
    "receive": ActionType.RECEIVE,
    "set":     ActionType.SET,
    "attack":  ActionType.ATTACK,
    "dig":     ActionType.DIG,
    "block":   ActionType.BLOCK,
}


@dataclass
class _Action:
    action_type: ActionType
    frame: int
    is_synthetic: bool = False


@dataclass
class _Contact:
    frame: int
    court_side: str
    is_synthetic: bool = False


def _row_to_objs(row: dict[str, str]) -> tuple[_Action, _Action, list[_Contact]]:
    """Reconstruct (curr_action, next_action, contacts) from an events.csv row.

    The events.csv has prev/curr action + frame + pid + team but NOT
    court_side per contact. Infer court_side from team_prev/team_curr
    using a fixed convention (team A = near). This correctly tracks
    same-side vs different-side for the verifier; absolute side labels
    don't matter to _possession_flips_after.
    """
    side_of = {"A": "near", "B": "far"}
    prev_side = side_of.get(row["prev_team"], "unknown")
    curr_side = side_of.get(row["curr_team"], "unknown")
    prev_at = _STR_TO_ACTION_TYPE.get(row["prev_action"].lower())
    curr_at = _STR_TO_ACTION_TYPE.get(row["curr_action"].lower())
    if prev_at is None or curr_at is None:
        raise ValueError(f"Unknown action type in row: {row}")
    a = _Action(action_type=prev_at, frame=int(row["prev_frame"]))
    nxt = _Action(action_type=curr_at, frame=int(row["curr_frame"]))
    contacts = [
        _Contact(frame=int(row["prev_frame"]), court_side=prev_side),
        _Contact(frame=int(row["curr_frame"]), court_side=curr_side),
    ]
    return a, nxt, contacts


def _accuracy_for_config(
    rows: list[dict[str, str]], cfg: ChainWalkerConfig,
) -> dict[str, int]:
    correct_flip = 0
    missed_flip = 0
    correct_stay = 0
    over_flip = 0
    for row in rows:
        a, nxt, contacts = _row_to_objs(row)
        walker_says_flip = _possession_flips_after(a, nxt, contacts, cfg)
        gt_says_flip = (row["event_type"] == "crossing")
        if gt_says_flip and walker_says_flip:
            correct_flip += 1
        elif gt_says_flip and not walker_says_flip:
            missed_flip += 1
        elif (not gt_says_flip) and (not walker_says_flip):
            correct_stay += 1
        else:
            over_flip += 1
    return {
        "correct_flip": correct_flip,
        "missed_flip": missed_flip,
        "correct_stay": correct_stay,
        "over_flip": over_flip,
        "total_correct": correct_flip + correct_stay,
        "total_pairs": len(rows),
    }


def main() -> int:
    if not IN_CSV.exists():
        print(f"ERROR: {IN_CSV} not found. Run probe_gt_net_crossings_2026_05_20 first.",
              file=sys.stderr)
        return 1

    with open(IN_CSV) as fh:
        rows = list(csv.DictReader(fh))
    print(f"Loaded {len(rows)} GT-pair events", flush=True)

    configs = {
        "cfg_00 (v13 baseline)": ChainWalkerConfig(False, False),
        "cfg_10 (B.1 only)":     ChainWalkerConfig(True, False),
        "cfg_01 (B.2 only)":     ChainWalkerConfig(False, True),
        "cfg_11 (B.1+B.2)":      ChainWalkerConfig(True, True),
    }

    results: dict[str, dict[str, int]] = {}
    for name, cfg in configs.items():
        r = _accuracy_for_config(rows, cfg)
        results[name] = r
        print(f"  {name}: correct_flip={r['correct_flip']}, "
              f"missed_flip={r['missed_flip']}, "
              f"correct_stay={r['correct_stay']}, "
              f"over_flip={r['over_flip']}, "
              f"total_correct={r['total_correct']}/{r['total_pairs']}",
              flush=True)

    out = {"configs": results}
    (OUT_DIR / "summary.json").write_text(json.dumps(out, indent=2))

    md = ["# Walker Accuracy Replay — Summary (2026-05-20)", ""]
    md.append(f"Substrate: {len(rows)} GT-pair events from "
              "reports/gt_net_crossings_2026_05_20/events.csv.")
    md.append("")
    md.append("| Config | correct_flip | missed_flip | correct_stay | "
              "over_flip | total_correct |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for name, r in results.items():
        md.append(
            f"| {name} | {r['correct_flip']} | {r['missed_flip']} | "
            f"{r['correct_stay']} | {r['over_flip']} | "
            f"{r['total_correct']}/{r['total_pairs']} |"
        )
    md.append("")
    md.append("## Sanity check")
    md.append("")
    md.append("cfg_00 must produce numbers identical to the v13 baseline "
              "(482 correct_flip, 64 missed_flip, 699 correct_stay, "
              "20 over_flip). If not, the refactor broke v13-equivalent "
              "behavior — STOP and debug before A/B.")
    md.append("")
    md.append("## CAVEAT: B.2 tautology on this substrate")
    md.append("")
    md.append("The events.csv reconstructs `court_side` from team identity "
              "(team A = near, team B = far) because the original GT-pair "
              "probe didn't capture Contact.court_side. B.2 (ball-trajectory "
              "verifier) consumes court_side as the override signal — so "
              "on this substrate, B.2 mechanically matches the GT-derived "
              "signal and appears perfect by construction. The realistic "
              "B.2 ceiling lives in Stage-2 A/B which uses production "
              "Contact.court_side (derived from ball Y vs net_y). cfg_00 "
              "baseline and cfg_10 (B.1 only) are unaffected — they don't "
              "consume court_side as a primary signal.")
    md.append("")
    (OUT_DIR / "summary.md").write_text("\n".join(md))
    print(f"\nWrote {OUT_DIR/'summary.md'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
