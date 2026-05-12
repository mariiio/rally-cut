"""Re-measure attribution post-v1.3 — fresh snapshot for 9 Phase-0 fixtures.

Mirrors `scripts/phase0_lock_baseline.py` but writes to a NEW file (does not
overwrite the locked baseline) and produces a per-fixture + aggregate
diff vs the 2026-04-24 baseline.
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
from rallycut.training.action_gt_query import load_for_rallies

FIXTURE_MAP_PATH = (
    Path(__file__).resolve().parent.parent
    / "reports" / "attribution_rebuild" / "fixture_video_ids_2026_04_24.json"
)
BASELINE_PATH = (
    Path(__file__).resolve().parent.parent
    / "reports" / "attribution_rebuild" / "baseline_2026_04_24.json"
)
OUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "reports" / "attribution_rebuild" / "post_v13_2026_05_11.json"
)


def main() -> int:
    fixture_map = json.loads(FIXTURE_MAP_PATH.read_text())["fixtures"]
    rallies: list[dict[str, Any]] = []

    with get_connection() as conn:
        for fixture_name, finfo in fixture_map.items():
            video_id = finfo["video_id"]
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT r.id, r.start_ms, r.end_ms,
                           pt.actions_json,
                           pt.contacts_json,
                           pt.primary_track_ids
                    FROM rallies r
                    JOIN player_tracks pt ON pt.rally_id = r.id
                    WHERE r.video_id = %s
                      AND EXISTS (
                          SELECT 1 FROM rally_action_ground_truth gt
                          WHERE gt.rally_id = r.id
                      )
                    ORDER BY r.start_ms
                    """,
                    (video_id,),
                )
                rows = cur.fetchall()

            rally_ids = [str(row[0]) for row in rows]
            gt_by_rally = load_for_rallies(conn, rally_ids)

            print(f"[{fixture_name}] {len(rows)} GT rallies")
            for rid, start_ms, end_ms, actions_json, contacts_json, ptids in rows:
                pipeline_actions = actions_json.get("actions", []) if actions_json else []
                team_assignments = actions_json.get("teamAssignments", {}) if actions_json else {}
                serving_team = actions_json.get("servingTeam") if actions_json else None
                contacts = contacts_json.get("contacts", []) if contacts_json else []
                normalised_gt = gt_by_rally.get(str(rid), [])
                rally = {
                    "rally_id": rid, "video_id": video_id,
                    "fixture": fixture_name,
                    "start_ms": start_ms, "end_ms": end_ms,
                    "primary_track_ids": ptids,
                    "team_assignments": team_assignments,
                    "serving_team": serving_team,
                    "gt_actions": normalised_gt,
                    "pipeline_actions": pipeline_actions,
                    "pipeline_contacts": contacts,
                }
                scored = score_rally(rally)
                rally["matches"] = scored["matches"]
                rally["rally_totals"] = scored["rally_totals"]
                rallies.append(rally)

    agg = aggregate(rallies)
    payload = {
        "generated_at": "2026-05-11",
        "match_tolerance_frames": MATCH_TOLERANCE_FRAMES,
        "source": "DB read post-v1.3 fleet deploy (commit df4d2d3)",
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
    print("=== POST-v1.3 PER-FIXTURE ===")
    print(f"{'fix':<7} {'n_gt':>5} {'correct':>9} {'wrong':>8} "
          f"{'miss':>5} {'abs':>4}")
    for fx, v in agg["per_fixture"].items():
        c = v["counts"]; r = v["rates"]
        w = sum(c[k] for k in WRONG_CATEGORIES)
        print(f"  {fx:5s} {c['n_gt_actions']:>5d}  "
              f"{c['correct']:>3d} ({r['correct_rate']:>5.1%})  "
              f"{w:>3d} ({r['wrong_rate']:>5.1%})  "
              f"{c['missing']:>3d}  {c['abstained']:>3d}")
    print()
    print(f"=== POST-v1.3 AGGREGATE (n={n}) ===")
    print(f"correct:  {counts['correct']:>4d} ({rates['correct_rate']:.1%})")
    print(f"wrong:    {wrong:>4d} ({rates['wrong_rate']:.1%})")
    print(f"missing:  {counts['missing']:>4d} ({rates['missing_rate']:.1%})")
    print(f"abstain:  {counts['abstained']:>4d} ({rates['abstained_rate']:.1%})")

    # Diff vs the 2026-04-24 baseline
    baseline = json.loads(BASELINE_PATH.read_text())
    b_counts = baseline["aggregate"]["counts"]
    b_rates = baseline["aggregate"]["rates"]
    b_wrong = sum(b_counts[k] for k in WRONG_CATEGORIES)
    print()
    print(f"=== DIFF vs 2026-04-24 BASELINE ===")
    print(f"{'metric':<18} {'baseline':>10} {'post-v1.3':>10} {'delta':>10}")
    print(f"{'n_gt_actions':<18} {b_counts['n_gt_actions']:>10d} "
          f"{counts['n_gt_actions']:>10d} "
          f"{counts['n_gt_actions'] - b_counts['n_gt_actions']:>+10d}")
    print(f"{'correct':<18} {b_counts['correct']:>10d} {counts['correct']:>10d} "
          f"{counts['correct'] - b_counts['correct']:>+10d}")
    print(f"{'correct_rate':<18} {b_rates['correct_rate']:>10.3f} "
          f"{rates['correct_rate']:>10.3f} "
          f"{rates['correct_rate'] - b_rates['correct_rate']:>+10.3f}")
    print(f"{'wrong':<18} {b_wrong:>10d} {wrong:>10d} "
          f"{wrong - b_wrong:>+10d}")
    print(f"{'wrong_cross_team':<18} {b_counts['wrong_cross_team']:>10d} "
          f"{counts['wrong_cross_team']:>10d} "
          f"{counts['wrong_cross_team'] - b_counts['wrong_cross_team']:>+10d}")
    print(f"{'wrong_same_team':<18} {b_counts['wrong_same_team']:>10d} "
          f"{counts['wrong_same_team']:>10d} "
          f"{counts['wrong_same_team'] - b_counts['wrong_same_team']:>+10d}")
    print(f"{'missing':<18} {b_counts['missing']:>10d} {counts['missing']:>10d} "
          f"{counts['missing'] - b_counts['missing']:>+10d}")
    print()
    print("=== PER-FIXTURE DIFF (correct_rate, post − baseline) ===")
    b_per_fix = baseline.get("fixture_rates", baseline.get("aggregate", {}))
    for fx in sorted(agg["per_fixture"].keys()):
        post_r = agg["per_fixture"][fx]["rates"]["correct_rate"]
        if "fixture_rates" in baseline:
            base_r = baseline["fixture_rates"].get(fx, {}).get("correct_rate", 0.0)
        else:
            # fallback path: read fixture counts
            base_c = baseline["fixtures"].get(fx, {})
            base_r = base_c.get("correct", 0) / max(1, base_c.get("n_gt_actions", 1))
        delta = post_r - base_r
        print(f"  {fx:5s}  base={base_r:.1%}  post={post_r:.1%}  delta={delta:+.1%}")

    print(f"\nWrote: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
