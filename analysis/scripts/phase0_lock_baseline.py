"""Phase 0.1 — lock baseline JSON on disk.

Reads current pipeline state from DB (off-gate stage-3 production state) for the 9
Phase-0 fixtures. Scoring and matching delegated to
``rallycut.evaluation.attribution_bench``.

Output: reports/attribution_rebuild/baseline_2026_04_24.json
(SOURCE OF TRUTH for all A/B going forward).

Run:
    uv run python scripts/phase0_lock_baseline.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from rallycut.evaluation.attribution_bench import (
    MATCH_TOLERANCE_FRAMES,
    WRONG_CATEGORIES,
    aggregate,
    score_rally,
)
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
    / "attribution_rebuild"
    / "baseline_2026_04_24.json"
)


def main() -> int:
    fixture_map = json.loads(FIXTURE_MAP_PATH.read_text())["fixtures"]

    rallies: list[dict[str, Any]] = []

    with get_connection() as conn, conn.cursor() as cur:
        for fixture_name, finfo in fixture_map.items():
            video_id = finfo["video_id"]
            cur.execute(
                """
                SELECT
                    r.id, r.start_ms, r.end_ms,
                    pt.action_ground_truth_json,
                    pt.actions_json,
                    pt.contacts_json,
                    pt.primary_track_ids
                FROM rallies r
                JOIN player_tracks pt ON pt.rally_id = r.id
                WHERE r.video_id = %s
                  AND pt.action_ground_truth_json IS NOT NULL
                  AND jsonb_array_length(pt.action_ground_truth_json::jsonb) > 0
                ORDER BY r.start_ms
                """,
                (video_id,),
            )
            rows = cur.fetchall()
            print(f"[{fixture_name}] {len(rows)} GT rallies")

            for rid, start_ms, end_ms, gt, actions_json, contacts_json, ptids in rows:
                pipeline_actions = actions_json.get("actions", []) if actions_json else []
                team_assignments = actions_json.get("teamAssignments", {}) if actions_json else {}
                serving_team = actions_json.get("servingTeam") if actions_json else None
                contacts = contacts_json.get("contacts", []) if contacts_json else []

                rally = {
                    "rally_id": rid,
                    "video_id": video_id,
                    "fixture": fixture_name,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
                    "primary_track_ids": ptids,
                    "team_assignments": team_assignments,
                    "serving_team": serving_team,
                    "gt_actions": gt,
                    "pipeline_actions": pipeline_actions,
                    "pipeline_contacts": contacts,
                }
                scored = score_rally(rally)
                rally["matches"] = scored["matches"]
                rally["rally_totals"] = scored["rally_totals"]
                rallies.append(rally)

    agg = aggregate(rallies)
    payload = {
        "generated_at": "2026-04-24",
        "match_tolerance_frames": MATCH_TOLERANCE_FRAMES,
        "source": (
            "DB read from player_tracks.{action_ground_truth_json, actions_json, "
            "contacts_json, primary_track_ids} — off-gate stage-3 production state "
            "after stage-2/3 refresh (match-players + reattribute-actions)."
        ),
        "fixtures": {fx: v["counts"] for fx, v in agg["per_fixture"].items()},
        "fixture_rates": {fx: v["rates"] for fx, v in agg["per_fixture"].items()},
        "aggregate": agg["combined"],
        "rallies": rallies,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2))

    counts = agg["combined"]["counts"]
    rates = agg["combined"]["rates"]
    n = counts["n_gt_actions"]
    wrong = sum(counts[k] for k in WRONG_CATEGORIES)
    print()
    print(
        f"=== PER-FIXTURE (n_gt / correct / wrong(X+same+unk) / missing / abstained) ==="
    )
    for fx, v in agg["per_fixture"].items():
        c = v["counts"]
        r = v["rates"]
        w = sum(c[k] for k in WRONG_CATEGORIES)
        print(
            f"  {fx:6s}  n={c['n_gt_actions']:>3d}  "
            f"correct={c['correct']:>3d} ({r['correct_rate']:.1%})  "
            f"wrong={w:>3d} ({r['wrong_rate']:.1%}) "
            f"[{c['wrong_cross_team']}/{c['wrong_same_team']}/{c['wrong_unknown_team']}]  "
            f"miss={c['missing']:>2d}  abs={c['abstained']:>2d}"
        )
    print()
    print(f"=== AGGREGATE (n={n}) ===")
    print(f"correct:  {counts['correct']:>4d} ({rates['correct_rate']:.1%})")
    print(f"wrong:    {wrong:>4d} ({rates['wrong_rate']:.1%})  "
          f"[cross={counts['wrong_cross_team']} "
          f"same={counts['wrong_same_team']} "
          f"unk={counts['wrong_unknown_team']}]")
    print(f"missing:  {counts['missing']:>4d} ({rates['missing_rate']:.1%})")
    print(f"abstain:  {counts['abstained']:>4d} ({rates['abstained_rate']:.1%})")
    print(f"\nwrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
