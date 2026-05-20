#!/usr/bin/env python3
"""GT-derived net-crossing event enumeration probe (2026-05-20).

Walks rally_action_ground_truth in frame order per trusted-32 rally. For
each consecutive GT contact pair (i, i+1):
  - team_prev = team of GT[i].resolved_track_id
  - team_curr = team of GT[i+1].resolved_track_id
  - If team_prev != team_curr -> ball crossed the net between them.

For each crossing event, check whether `prev_action_type` is in the
walker's _NET_CROSSING_ACTIONS = {SERVE, ATTACK}. If yes, walker would
correctly flip. If no, walker MISSES this flip — pinpoints the rule gap.

Also enumerates the reverse: non-crossing consecutive pairs where prev
is SERVE/ATTACK → walker would OVER-flip (false flip).

Output: reports/gt_net_crossings_2026_05_20/{summary.md, events.csv}
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, cast

import psycopg

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ANALYSIS_DIR / "scripts"))

from _upstream_probe_common import TRUSTED_32  # noqa: E402

DB_DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5436/rallycut",
)
OUT_DIR = ANALYSIS_DIR / "reports" / "gt_net_crossings_2026_05_20"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Mirror of action_classifier._NET_CROSSING_ACTIONS
NET_CROSSING_ACTIONS = {"serve", "attack"}


def main() -> int:
    print("Loading trusted-32 GT contacts + team assignments...", flush=True)

    # Pull GT + team_assignments per rally
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            """
            SELECT v.name, r.id,
                   rg.frame, rg.action::text, rg.resolved_track_id,
                   pt.actions_json
            FROM rally_action_ground_truth rg
            JOIN rallies r ON rg.rally_id = r.id
            JOIN videos v ON r.video_id = v.id
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE v.name = ANY(%s) AND rg.resolved_track_id IS NOT NULL
            ORDER BY v.name, r.id, rg.frame
            """,
            [list(TRUSTED_32)],
        )
        rows = cur.fetchall()

    # Group by rally
    rally_data: dict[str, dict[str, Any]] = {}
    for vname, rid, frame, action, tid, aj in rows:
        rid_s = str(rid)
        if rid_s not in rally_data:
            teams_for_rally: dict[str, str] = {}
            if isinstance(aj, dict):
                teams_for_rally = aj.get("teamAssignments") or {}
            elif isinstance(aj, str):
                try:
                    teams_for_rally = (json.loads(aj) or {}).get("teamAssignments") or {}
                except Exception:
                    teams_for_rally = {}
            rally_data[rid_s] = {
                "video": str(vname),
                "teams": teams_for_rally,
                "gt_contacts": [],
            }
        rally_data[rid_s]["gt_contacts"].append((int(frame), action.lower(), int(tid)))

    print(f"  {len(rally_data)} rallies loaded", flush=True)

    # Enumerate crossings + non-crossings
    crossings: list[dict] = []
    non_crossings: list[dict] = []
    skipped_no_team: int = 0

    for rid, data in rally_data.items():
        teams = cast(dict[str, str], data["teams"])
        contacts_raw = cast(list[tuple[int, str, int]], data["gt_contacts"])
        contacts = sorted(contacts_raw, key=lambda c: c[0])
        for i in range(1, len(contacts)):
            f_prev, a_prev, t_prev = contacts[i - 1]
            f_curr, a_curr, t_curr = contacts[i]
            team_prev = teams.get(str(t_prev))
            team_curr = teams.get(str(t_curr))
            if team_prev not in ("A", "B") or team_curr not in ("A", "B"):
                skipped_no_team += 1
                continue
            event = {
                "rally_id": rid,
                "video": data["video"],
                "prev_frame": f_prev,
                "curr_frame": f_curr,
                "prev_action": a_prev,
                "curr_action": a_curr,
                "prev_pid": t_prev,
                "curr_pid": t_curr,
                "prev_team": team_prev,
                "curr_team": team_curr,
                "walker_would_flip": a_prev in NET_CROSSING_ACTIONS,
            }
            if team_prev != team_curr:
                crossings.append(event)
            else:
                non_crossings.append(event)

    print(f"  total pairs: {len(crossings) + len(non_crossings)}")
    print(f"  GT crossings (team_prev != team_curr): {len(crossings)}")
    print(f"  GT non-crossings: {len(non_crossings)}")
    print(f"  skipped (missing team_assignments): {skipped_no_team}", flush=True)

    # Walker accuracy on crossings
    walker_correct_flip = sum(1 for e in crossings if e["walker_would_flip"])
    walker_missed_flip = sum(1 for e in crossings if not e["walker_would_flip"])

    # Walker accuracy on non-crossings
    walker_correct_stay = sum(1 for e in non_crossings if not e["walker_would_flip"])
    walker_over_flip = sum(1 for e in non_crossings if e["walker_would_flip"])

    # Prev-action histograms
    missed_flip_actions: Counter = Counter()
    for e in crossings:
        if not e["walker_would_flip"]:
            missed_flip_actions[e["prev_action"]] += 1

    over_flip_actions: Counter = Counter()
    for e in non_crossings:
        if e["walker_would_flip"]:
            over_flip_actions[e["prev_action"]] += 1

    # Prev->curr action transition for missed flips (informative)
    missed_flip_transitions: Counter = Counter()
    for e in crossings:
        if not e["walker_would_flip"]:
            missed_flip_transitions[f"{e['prev_action']}->{e['curr_action']}"] += 1

    over_flip_transitions: Counter = Counter()
    for e in non_crossings:
        if e["walker_would_flip"]:
            over_flip_transitions[f"{e['prev_action']}->{e['curr_action']}"] += 1

    # Output
    summary = {
        "n_rallies": len(rally_data),
        "n_total_pairs": len(crossings) + len(non_crossings),
        "n_crossings": len(crossings),
        "n_non_crossings": len(non_crossings),
        "n_skipped_no_team": skipped_no_team,
        "walker_on_crossings": {
            "correct_flip": walker_correct_flip,
            "missed_flip": walker_missed_flip,
            "accuracy_pct": 100 * walker_correct_flip / max(len(crossings), 1),
        },
        "walker_on_non_crossings": {
            "correct_stay": walker_correct_stay,
            "over_flip": walker_over_flip,
            "accuracy_pct": 100 * walker_correct_stay / max(len(non_crossings), 1),
        },
        "missed_flip_by_prev_action": dict(missed_flip_actions),
        "over_flip_by_prev_action": dict(over_flip_actions),
        "missed_flip_transitions": dict(missed_flip_transitions),
        "over_flip_transitions": dict(over_flip_transitions),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))

    # Per-event CSV
    csv_path = OUT_DIR / "events.csv"
    all_events = crossings + non_crossings
    for e in all_events:
        e["event_type"] = "crossing" if e in crossings else "non_crossing"
    with open(csv_path, "w", newline="") as fh:
        if all_events:
            w = csv.DictWriter(fh, fieldnames=list(all_events[0].keys()))
            w.writeheader()
            w.writerows(all_events)

    # Summary markdown
    md = ["# GT-derived Net-Crossing Probe — Summary (2026-05-20)", ""]
    md.append(f"Substrate: {len(rally_data)} trusted-32 rallies, "
              f"{summary['n_total_pairs']} consecutive GT-contact pairs.")
    md.append(f"GT crossings: **{len(crossings)}**, non-crossings: "
              f"**{len(non_crossings)}**, skipped: {skipped_no_team}.")
    md.append("")
    md.append("## Walker accuracy on GT crossings")
    md.append("")
    crossings_acc = 100 * walker_correct_flip / max(len(crossings), 1)
    md.append(f"- correct_flip (prev_action in {{serve,attack}}): "
              f"{walker_correct_flip}/{len(crossings)} "
              f"({crossings_acc:.1f}%)")
    md.append(f"- **missed_flip (walker didn't flip but should have): "
              f"{walker_missed_flip}/{len(crossings)}**")
    md.append("")
    md.append("### Missed flips — by prev_action_type:")
    md.append("")
    md.append("| prev_action | count |")
    md.append("|---|---:|")
    for act, n in missed_flip_actions.most_common():
        md.append(f"| {act} | {n} |")
    md.append("")
    md.append("### Missed flips — by transition pattern:")
    md.append("")
    md.append("| prev->curr | count |")
    md.append("|---|---:|")
    for trans, n in missed_flip_transitions.most_common():
        md.append(f"| {trans} | {n} |")
    md.append("")
    md.append("## Walker accuracy on GT non-crossings")
    md.append("")
    non_crossings_acc = 100 * walker_correct_stay / max(len(non_crossings), 1)
    md.append(f"- correct_stay (prev_action NOT in {{serve,attack}}): "
              f"{walker_correct_stay}/{len(non_crossings)} "
              f"({non_crossings_acc:.1f}%)")
    md.append(f"- **over_flip (walker would flip but shouldn't): "
              f"{walker_over_flip}/{len(non_crossings)}**")
    md.append("")
    md.append("### Over-flips — by prev_action_type:")
    md.append("")
    md.append("| prev_action | count |")
    md.append("|---|---:|")
    for act, n in over_flip_actions.most_common():
        md.append(f"| {act} | {n} |")
    md.append("")
    md.append("### Over-flips — by transition pattern:")
    md.append("")
    md.append("| prev->curr | count |")
    md.append("|---|---:|")
    for trans, n in over_flip_transitions.most_common():
        md.append(f"| {trans} | {n} |")
    md.append("")
    md.append("## Interpretation")
    md.append("")
    md.append("- High `missed_flip` count for a non-{serve,attack} action → "
              "that action should be added to `_NET_CROSSING_ACTIONS`, OR "
              "the action is being mis-typed by the classifier when it "
              "actually IS the net-crossing event.")
    md.append("- High `over_flip` count for a transition pattern → walker "
              "incorrectly flips on those rallies; needs a guard.")
    md.append("- BLOCK is documented as non-net-crossing in the walker; if "
              "many crossings have prev=block, the documentation/code is "
              "wrong (block-cover does cross sometimes).")
    md.append("")

    (OUT_DIR / "summary.md").write_text("\n".join(md))
    print(f"\nWrote {OUT_DIR/'summary.md'}", flush=True)
    print()
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
