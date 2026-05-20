#!/usr/bin/env python3
"""L6: team-chain accuracy probe.

For each wrong-attribution contact:
  - Pipeline chain-derived expected_team comes from walking pipeline
    actions_json and tagging each contact with the team of its actor.
  - GT-derived expected_team comes from walking rally_action_ground_truth
    and tagging each contact with the team of the GT actor.
  - Disagreement: chain != GT-derived.

Oracle: substitute GT-derived expected_team into scorer feature, re-score.

Output: reports/upstream_bottleneck_2026_05_20/L6.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import psycopg

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ANALYSIS_DIR / "scripts"))

from _upstream_probe_common import (  # noqa: E402
    DB_DSN,
    fetch_rally_state,
    load_wrong_attribution_corpus,
    rescore_contact,
)

OUT_DIR = ANALYSIS_DIR / "reports" / "upstream_bottleneck_2026_05_20"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def gt_team_chain_for_rally(
    rally_id: str, team_assignments: dict[str, str],
) -> dict[int, str | None]:
    """Return {gt_frame: team} from rally_action_ground_truth."""
    out: dict[int, str | None] = {}
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            """
            SELECT frame, resolved_track_id FROM rally_action_ground_truth
            WHERE rally_id = %s AND resolved_track_id IS NOT NULL
            ORDER BY frame
            """,
            [rally_id],
        )
        for frame, tid in cur.fetchall():
            t = team_assignments.get(str(tid))
            out[int(frame)] = t if t in ("A", "B") else None
    return out


def pipeline_team_chain_for_rally(
    actions: list[dict], team_assignments: dict[str, str],
) -> dict[int, str | None]:
    """Return {action_frame: team} from pipeline actions_json."""
    out: dict[int, str | None] = {}
    for a in actions:
        tid = a.get("playerTrackId")
        if tid is None or tid == -1:
            continue
        t = team_assignments.get(str(tid))
        out[int(a.get("frame", -1))] = t if t in ("A", "B") else None
    return out


def main() -> int:
    print("Loading wrong-attribution corpus...", flush=True)
    rows = load_wrong_attribution_corpus()
    print(f"  {len(rows)} wrong-attribution contacts", flush=True)

    chain_disagreements = 0
    oracle_recoveries = 0
    by_disagreement: Counter = Counter()

    for i, row in enumerate(rows):
        rally = fetch_rally_state(row.rally_id)
        if rally is None:
            continue
        teams = rally["teams"]
        gt_chain = gt_team_chain_for_rally(row.rally_id, teams)
        pipe_chain = pipeline_team_chain_for_rally(rally["actions"], teams)
        pipe_team = pipe_chain.get(row.action_frame)
        gt_team = None
        for gf, gt in gt_chain.items():
            if abs(gf - row.action_frame) <= 5:
                gt_team = gt
                break
        if pipe_team is None or gt_team is None:
            continue
        agree = pipe_team == gt_team
        if not agree:
            chain_disagreements += 1
            by_disagreement[f"pipe={pipe_team}, gt={gt_team}"] += 1
            contact = next(
                (c for c in rally["contacts"]
                 if abs(int(c.get("frame", -1)) - row.action_frame) <= 3),
                None,
            )
            if contact is None:
                continue
            cand_tids = [int(pc[0]) for pc in (contact.get("playerCandidates") or [])]
            team_assignments_int = {
                int(k): (0 if v == "A" else 1)
                for k, v in teams.items()
            }
            expected_team_int = 0 if gt_team == "A" else 1
            pick = rescore_contact(
                rally, contact, row.action_type, cand_tids,
                expected_team=expected_team_int,
                team_assignments_int=team_assignments_int,
            )
            if pick == row.gt_pid:
                oracle_recoveries += 1
        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(rows)}] processed", flush=True)

    out = {
        "n_total_wrong": len(rows),
        "chain_disagreements": chain_disagreements,
        "oracle_recoveries_at_chain_disagreements": oracle_recoveries,
        "by_disagreement_pattern": dict(by_disagreement),
    }
    (OUT_DIR / "L6.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_DIR/'L6.json'}", flush=True)
    print(f"  chain disagreements: {chain_disagreements}/{len(rows)}", flush=True)
    print(f"  oracle recoveries (substitute GT team): {oracle_recoveries}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
