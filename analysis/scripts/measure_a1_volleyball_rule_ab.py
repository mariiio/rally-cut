"""A/B measurement for A1 volleyball-rule attribution pass.

Re-runs `reattribute_players` in-process on the 3 fresh-GT videos
(cece / gigi / wawa = the "22-rally panel") with
``USE_VOLLEYBALL_RULE_ATTRIBUTION`` toggled OFF then ON, and reports
attribution metrics (correct / wrong_cross_team / wrong_same_team /
missing / abstained) plus the count of ``attribution_uncertain`` flags
set during the A1 pass.

Modelled after ``scripts/measure_attribution_team_chain_ab.py``;
mirrors ``scripts/measure_attribution_fresh_gt.py`` for GT
normalisation.

Run from analysis/:
    uv run python scripts/measure_a1_volleyball_rule_ab.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from rallycut.evaluation.attribution_bench import (
    WRONG_CATEGORIES,
    aggregate,
    score_rally,
)
from rallycut.evaluation.db import get_connection
from rallycut.tracking.action_classifier import (
    ClassifiedAction,
    reattribute_players,
)
from rallycut.tracking.contact_detector import Contact
from rallycut.training.action_gt_query import load_for_videos

FRESH_GT_VIDEOS = {
    "cece": "950fbe5d-fdad-4862-b05d-8b374bdd5ec6",
    "gigi": "b097dd2a-6953-4e0e-a603-5be3552f462e",
    "wawa": "5c756c41-1cc1-4486-a95c-97398912cfbe",
}

OUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "reports" / "a1_volleyball_rule" / "ab_panel_2026_05_13.json"
)


def _build_rally_remap(match_analysis: dict | None, rally_id: str) -> dict[int, int]:
    if not match_analysis:
        return {}
    for r in match_analysis.get("rallies") or []:
        rid = r.get("rallyId") or r.get("rally_id")
        if rid != rally_id:
            continue
        afm = r.get("appliedFullMapping") or r.get("trackToPlayer") or {}
        out: dict[int, int] = {}
        for k, v in afm.items():
            try:
                out[int(k)] = int(v)
            except (TypeError, ValueError):
                continue
        return out
    return {}


def _reconstruct_contacts(contacts_json: dict[str, Any]) -> list[Contact]:
    contacts: list[Contact] = []
    for c in (contacts_json or {}).get("contacts", []):
        candidates = [
            (int(p[0]), float(p[1])) for p in c.get("playerCandidates", [])
            if p[1] is not None
        ]
        contacts.append(Contact(
            frame=c.get("frame", 0),
            ball_x=c.get("ballX", 0.0),
            ball_y=c.get("ballY", 0.0),
            velocity=c.get("velocity", 0.0),
            direction_change_deg=c.get("directionChangeDeg", 0.0),
            player_track_id=c.get("playerTrackId", -1),
            player_distance=(
                float("inf")
                if c.get("playerDistance") is None
                else c["playerDistance"]
            ),
            player_candidates=candidates,
            court_side=c.get("courtSide", "unknown"),
            is_at_net=c.get("isAtNet", False),
            is_validated=c.get("isValidated", False),
            confidence=c.get("confidence", 0.0),
            arc_fit_residual=c.get("arcFitResidual", 0.0),
        ))
    return contacts


def _reconstruct_actions(actions_json: dict[str, Any]) -> list[ClassifiedAction]:
    from rallycut.tracking.action_classifier import ActionType
    raw = (actions_json or {}).get("actions", [])
    if isinstance(raw, dict):
        raw = raw.get("actions", [])
    out: list[ClassifiedAction] = []
    for a in raw:
        if not isinstance(a, dict):
            continue
        try:
            t = ActionType(a["action"])
        except (KeyError, ValueError):
            t = ActionType.UNKNOWN
        out.append(ClassifiedAction(
            action_type=t,
            frame=a.get("frame", 0),
            ball_x=a.get("ballX", 0.0),
            ball_y=a.get("ballY", 0.0),
            velocity=a.get("velocity", 0.0),
            player_track_id=a.get("playerTrackId", -1),
            court_side=a.get("courtSide", "unknown"),
            confidence=a.get("confidence", 0.0),
            is_synthetic=a.get("isSynthetic", False),
            team=a.get("team", "unknown"),
        ))
    return out


def _score(rallies_data: list[dict[str, Any]], a1_on: bool) -> dict[str, Any]:
    if a1_on:
        os.environ["USE_VOLLEYBALL_RULE_ATTRIBUTION"] = "1"
    else:
        os.environ.pop("USE_VOLLEYBALL_RULE_ATTRIBUTION", None)

    scored_rallies: list[dict[str, Any]] = []
    n_uncertain = 0
    for r in rallies_data:
        actions = _reconstruct_actions(r["actions_json"])
        contacts = _reconstruct_contacts(r["contacts_json"])
        team_assignments_raw = (r["actions_json"] or {}).get("teamAssignments", {})
        team_assignments = {
            int(k): (0 if v == "A" else 1) for k, v in team_assignments_raw.items()
        }
        if team_assignments and actions:
            reattribute_players(actions, contacts, team_assignments)

        # Count attribution_uncertain flips (set by the A1 pass).
        n_uncertain += sum(1 for a in actions if a.attribution_uncertain)

        pipeline_actions = [a.to_dict() for a in actions]

        # Normalise GT: raw track id -> canonical PID via applied mapping.
        normalised_gt = []
        for ga in r["gt"]:
            raw_tid = ga.get("trackId", ga.get("playerTrackId"))
            try:
                raw_int = int(raw_tid) if raw_tid is not None else None
            except (TypeError, ValueError):
                raw_int = None
            canonical = r["remap"].get(raw_int) if raw_int is not None else None
            if canonical is None and raw_int is not None and 1 <= raw_int <= 4:
                canonical = raw_int
            entry = dict(ga)
            entry["playerTrackId"] = canonical
            normalised_gt.append(entry)

        rally_record = {
            "rally_id": r["rally_id"],
            "video_id": r["video_id"],
            "fixture": r["fixture"],
            "team_assignments": team_assignments_raw,
            "gt_actions": normalised_gt,
            "pipeline_actions": pipeline_actions,
        }
        scored = score_rally(rally_record)
        rally_record["matches"] = scored["matches"]
        rally_record["rally_totals"] = scored["rally_totals"]
        scored_rallies.append(rally_record)

    agg = aggregate(scored_rallies)
    return {
        "agg": agg,
        "rallies": scored_rallies,
        "n_uncertain": n_uncertain,
    }


def _summary_block(label: str, agg: dict[str, Any], n_uncertain: int) -> dict[str, Any]:
    c = agg["combined"]["counts"]
    r = agg["combined"]["rates"]
    wrong = sum(c[k] for k in WRONG_CATEGORIES)
    n_total = c["n_gt_actions"]
    abst_rate = (n_uncertain / n_total) if n_total else 0.0
    print(f"\n=== {label} ===", flush=True)
    print(f"  n_gt_actions:                  {n_total:>4d}", flush=True)
    print(f"  correct:                       {c['correct']:>4d}  ({r['correct_rate']:>6.1%})", flush=True)
    print(f"  wrong (any):                   {wrong:>4d}  ({r['wrong_rate']:>6.1%})", flush=True)
    print(f"    wrong_cross_team:            {c['wrong_cross_team']:>4d}", flush=True)
    print(f"    wrong_same_team:             {c['wrong_same_team']:>4d}", flush=True)
    print(f"    wrong_unknown_team:          {c['wrong_unknown_team']:>4d}", flush=True)
    print(f"  missing:                       {c['missing']:>4d}  ({r['missing_rate']:>6.1%})", flush=True)
    print(f"  abstained (bench-level):       {c['abstained']:>4d}  ({r['abstained_rate']:>6.1%})", flush=True)
    print(f"  attribution_uncertain (A1):    {n_uncertain:>4d}  ({abst_rate:>6.1%})", flush=True)
    return {
        "counts": c, "rates": r, "wrong": wrong,
        "n_uncertain": n_uncertain, "uncertain_rate": abst_rate,
    }


def main() -> int:
    print("Loading rally data for A/B (cece + gigi + wawa)...", flush=True)
    rallies_data: list[dict[str, Any]] = []
    with get_connection() as conn:
        match_analyses: dict[str, dict] = {}
        with conn.cursor() as cur:
            for fixture, vid in FRESH_GT_VIDEOS.items():
                cur.execute("SELECT match_analysis_json FROM videos WHERE id = %s", (vid,))
                ma = cur.fetchone()
                match_analyses[vid] = ma[0] if ma and ma[0] else {}

        gt_by_rally = load_for_videos(conn, list(FRESH_GT_VIDEOS.values()))

        for fixture, vid in FRESH_GT_VIDEOS.items():
            with conn.cursor() as cur:
                cur.execute(
                    """SELECT r.id, pt.actions_json, pt.contacts_json
                       FROM rallies r JOIN player_tracks pt ON pt.rally_id = r.id
                       WHERE r.video_id = %s
                         AND EXISTS (
                             SELECT 1 FROM rally_action_ground_truth gt
                             WHERE gt.rally_id = r.id
                         )
                       ORDER BY r.start_ms""",
                    (vid,),
                )
                for rid, aj, cj in cur.fetchall():
                    rallies_data.append({
                        "rally_id": str(rid),
                        "video_id": vid,
                        "fixture": fixture,
                        "gt": gt_by_rally.get(str(rid), []),
                        "actions_json": aj,
                        "contacts_json": cj,
                        "remap": _build_rally_remap(match_analyses[vid], str(rid)),
                    })
    print(f"Loaded {len(rallies_data)} rallies across {len(FRESH_GT_VIDEOS)} videos",
          flush=True)

    off = _score(rallies_data, a1_on=False)
    on = _score(rallies_data, a1_on=True)

    s_off = _summary_block("BASELINE (USE_VOLLEYBALL_RULE_ATTRIBUTION unset)", off["agg"], off["n_uncertain"])
    s_on  = _summary_block("A1 ON    (USE_VOLLEYBALL_RULE_ATTRIBUTION=1)",      on["agg"],  on["n_uncertain"])

    print("\n=== DELTA (on - off) ===", flush=True)
    d_correct = s_on["counts"]["correct"] - s_off["counts"]["correct"]
    print(f"  correct:           {d_correct:+d} "
          f"({s_on['rates']['correct_rate'] - s_off['rates']['correct_rate']:+.1%})", flush=True)
    print(f"  wrong_cross_team:  {s_on['counts']['wrong_cross_team'] - s_off['counts']['wrong_cross_team']:+d}", flush=True)
    print(f"  wrong_same_team:   {s_on['counts']['wrong_same_team']  - s_off['counts']['wrong_same_team']:+d}", flush=True)
    print(f"  wrong_unknown:     {s_on['counts']['wrong_unknown_team'] - s_off['counts']['wrong_unknown_team']:+d}", flush=True)
    print(f"  missing:           {s_on['counts']['missing'] - s_off['counts']['missing']:+d}", flush=True)
    print(f"  attribution_uncertain: {s_on['n_uncertain'] - s_off['n_uncertain']:+d}", flush=True)

    print("\n=== PER-FIXTURE ===", flush=True)
    for fx in sorted(off["agg"]["per_fixture"].keys()):
        off_c = off["agg"]["per_fixture"][fx]["counts"]
        on_c  = on["agg"]["per_fixture"][fx]["counts"]
        n = off_c["n_gt_actions"]
        print(f"  {fx:<6} n={n:>3d}  correct: {off_c['correct']:>3d} → {on_c['correct']:>3d} "
              f"({on_c['correct'] - off_c['correct']:+d})  "
              f"same_team: {off_c['wrong_same_team']} → {on_c['wrong_same_team']}  "
              f"cross_team: {off_c['wrong_cross_team']} → {on_c['wrong_cross_team']}", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({
        "off": {"agg": off["agg"], "n_uncertain": off["n_uncertain"]},
        "on":  {"agg": on["agg"],  "n_uncertain": on["n_uncertain"]},
        "summary_off": s_off,
        "summary_on": s_on,
    }, indent=2, default=str))
    print(f"\nWrote {OUT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
