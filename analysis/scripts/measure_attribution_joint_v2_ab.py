"""A/B harness: re-run reattribute_players in memory on the 3 GT videos
with JOINT_ATTRIBUTION_V2 OFF vs ON. Mirrors the v1 A/B harness pattern.

Run from analysis/:
    uv run python scripts/measure_attribution_joint_v2_ab.py
"""
from __future__ import annotations

import json
import logging
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
    ActionType,
    ClassifiedAction,
    reattribute_players,
)
from rallycut.tracking.contact_detector import Contact

FRESH_GT_VIDEOS = {
    "cece": "950fbe5d-fdad-4862-b05d-8b374bdd5ec6",
    "gigi": "b097dd2a-6953-4e0e-a603-5be3552f462e",
    "wawa": "5c756c41-1cc1-4486-a95c-97398912cfbe",
}

OUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "reports" / "attribution_baseline" / "joint_v2_ab_2026_05_11.json"
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


class _FallbackCounter(logging.Handler):
    """Counts WARNING log records from joint_attribute fallback."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.count = 0

    def emit(self, record: logging.LogRecord) -> None:
        if "joint_attribute fallback" in record.getMessage():
            self.count += 1


def _score(rallies_data: list[dict[str, Any]], joint_v2_on: bool) -> dict[str, Any]:
    os.environ["JOINT_ATTRIBUTION_V2"] = "1" if joint_v2_on else "0"

    # Install fallback counter on the joint_attribution logger for ON runs.
    fallback_counter: _FallbackCounter | None = None
    ja_logger: logging.Logger | None = None
    if joint_v2_on:
        fallback_counter = _FallbackCounter()
        ja_logger = logging.getLogger("rallycut.tracking.joint_attribution")
        ja_logger.addHandler(fallback_counter)

    try:
        scored_rallies: list[dict[str, Any]] = []
        for r in rallies_data:
            actions = _reconstruct_actions(r["actions_json"])
            contacts = _reconstruct_contacts(r["contacts_json"])
            team_assignments_raw = (r["actions_json"] or {}).get("teamAssignments", {})
            team_assignments = {
                int(k): (0 if v == "A" else 1)
                for k, v in team_assignments_raw.items()
            }
            if team_assignments and actions:
                reattribute_players(actions, contacts, team_assignments)
            pipeline_actions = [a.to_dict() for a in actions]
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
    finally:
        if ja_logger is not None and fallback_counter is not None:
            ja_logger.removeHandler(fallback_counter)

    fallback_count = fallback_counter.count if fallback_counter is not None else 0
    return {"agg": agg, "rallies": scored_rallies, "fallback_count": fallback_count}


def _summary(label: str, agg: dict[str, Any]) -> dict[str, Any]:
    c = agg["combined"]["counts"]
    r = agg["combined"]["rates"]
    wrong = sum(c[k] for k in WRONG_CATEGORIES)
    print(f"\n=== {label} ===")
    print(f"  correct: {c['correct']:>3d}  ({r['correct_rate']:>6.1%})")
    print(f"  wrong:   {wrong:>3d}  ({r['wrong_rate']:>6.1%})  "
          f"[cross={c['wrong_cross_team']} same={c['wrong_same_team']} "
          f"unk={c['wrong_unknown_team']}]")
    print(f"  missing: {c['missing']:>3d}  ({r['missing_rate']:>6.1%})")
    return {"counts": c, "rates": r, "wrong": wrong}


def main() -> int:
    rallies_data: list[dict[str, Any]] = []
    with get_connection() as conn, conn.cursor() as cur:
        match_analyses: dict[str, dict] = {}
        for fixture, vid in FRESH_GT_VIDEOS.items():
            cur.execute(
                "SELECT match_analysis_json FROM videos WHERE id = %s", (vid,),
            )
            ma = cur.fetchone()
            match_analyses[vid] = ma[0] if ma and ma[0] else {}
        for fixture, vid in FRESH_GT_VIDEOS.items():
            cur.execute(
                """SELECT r.id, pt.action_ground_truth_json,
                          pt.actions_json, pt.contacts_json
                   FROM rallies r JOIN player_tracks pt ON pt.rally_id = r.id
                   WHERE r.video_id = %s
                     AND pt.action_ground_truth_json IS NOT NULL
                     AND jsonb_array_length(pt.action_ground_truth_json::jsonb) > 0
                   ORDER BY r.start_ms""",
                (vid,),
            )
            for rid, gt, aj, cj in cur.fetchall():
                rallies_data.append({
                    "rally_id": str(rid),
                    "fixture": fixture,
                    "gt": gt or [],
                    "actions_json": aj,
                    "contacts_json": cj,
                    "remap": _build_rally_remap(match_analyses[vid], str(rid)),
                })
    print(f"Loaded {len(rallies_data)} rallies across {len(FRESH_GT_VIDEOS)} videos",
          flush=True)

    off = _score(rallies_data, joint_v2_on=False)
    on = _score(rallies_data, joint_v2_on=True)

    s_off = _summary("OFF (v2 disabled)", off["agg"])
    s_on  = _summary("ON  (v2 enabled)",   on["agg"])

    print("\n=== DELTA (on - off) ===")
    print(f"  correct:           {s_on['counts']['correct'] - s_off['counts']['correct']:+d} "
          f"({s_on['rates']['correct_rate'] - s_off['rates']['correct_rate']:+.1%})")
    print(f"  wrong_cross_team: {s_on['counts']['wrong_cross_team'] - s_off['counts']['wrong_cross_team']:+d}")
    print(f"  wrong_same_team:  {s_on['counts']['wrong_same_team']  - s_off['counts']['wrong_same_team']:+d}")
    print(f"  wrong_unknown:    {s_on['counts']['wrong_unknown_team'] - s_off['counts']['wrong_unknown_team']:+d}")

    print("\n=== PER-FIXTURE DELTA ===")
    for fx in sorted(off["agg"]["per_fixture"].keys()):
        off_c = off["agg"]["per_fixture"][fx]["counts"]["correct"]
        on_c  = on["agg"]["per_fixture"][fx]["counts"]["correct"]
        print(f"  {fx:<6} correct: {off_c} → {on_c} ({on_c - off_c:+d})")

    print(f"\n=== FALLBACK RATE (G-F) ===")
    print(f"  joint_attribute fallback WARN lines: {on['fallback_count']} "
          f"(of {len(rallies_data)} rallies; gate ≤ 1)")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"off": off["agg"], "on": on["agg"]}, indent=2, default=str))
    print(f"\nWrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
