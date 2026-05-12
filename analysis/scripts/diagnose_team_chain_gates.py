"""Diagnostic: per-gate breakdown for wrong_cross_team errors.

For each of the 16 wrong_cross_team errors on the 3 GT videos, compute which
of G1/G2/G3/G4 would PASS or FAIL the _team_chain_override_allowed predicate,
then report counterfactual override counts under 6 gate configurations.

Run from analysis/:
    uv run python scripts/diagnose_team_chain_gates.py
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path
from typing import Any

# Add analysis to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rallycut.evaluation.attribution_bench import (
    MATCH_TOLERANCE_FRAMES,
    match_gt_to_pipeline,
    classify_action,
)
from rallycut.evaluation.db import get_connection
from rallycut.training.action_gt_query import load_for_rallies
from rallycut.tracking.action_classifier import (
    ActionType,
    ClassifiedAction,
    _chain_integrity,
    _compute_expected_teams,
)
from rallycut.tracking.contact_detector import Contact

FRESH_GT_VIDEOS = {
    "cece": "950fbe5d-fdad-4862-b05d-8b374bdd5ec6",
    "gigi": "b097dd2a-6953-4e0e-a603-5be3552f462e",
    "wawa": "5c756c41-1cc1-4486-a95c-97398912cfbe",
}


# ─── reconstruction helpers (copied from measure_attribution_team_chain_ab.py) ───

def _build_rally_remap(match_analysis: dict | None, rally_id: str) -> dict[int, int]:
    if not match_analysis:
        return {}
    for r in (match_analysis.get("rallies") or []):
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


# ─── gate evaluation helpers ───────────────────────────────────────────────────

def _eval_G3(
    contact: Contact,
    expected_team: int,
    team_assignments: dict[int, int],
    dist_ratio: float,
) -> tuple[bool, int | None, float | None]:
    """Returns (passes, best_correct_tid, best_correct_dist)."""
    if not math.isfinite(contact.player_distance):
        return False, None, None
    dist_cap = dist_ratio * contact.player_distance
    best_tid: int | None = None
    best_dist: float | None = None
    for tid, dist in contact.player_candidates:
        if team_assignments.get(tid) == expected_team and dist <= dist_cap:
            if best_dist is None or dist < best_dist:
                best_tid = tid
                best_dist = dist
    return best_tid is not None, best_tid, best_dist


def _eval_G4(contact: Contact, expected_team: int) -> tuple[bool, str]:
    """Returns (passes, reason)."""
    expected_side = "near" if expected_team == 0 else "far"
    cs = contact.court_side
    if cs == "unknown":
        return True, "soft-pass (unknown)"
    if cs == expected_side:
        return True, f"matches ({cs})"
    return False, f"disagrees: got {cs!r}, want {expected_side!r}"


# ─── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    # Load data from DB
    rallies_data: list[dict[str, Any]] = []
    with get_connection() as conn, conn.cursor() as cur:
        match_analyses: dict[str, dict] = {}
        for fixture, vid in FRESH_GT_VIDEOS.items():
            cur.execute("SELECT match_analysis_json FROM videos WHERE id = %s", (vid,))
            ma = cur.fetchone()
            match_analyses[vid] = ma[0] if ma and ma[0] else {}

        for fixture, vid in FRESH_GT_VIDEOS.items():
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
            rows = cur.fetchall()
            rally_ids = [str(row[0]) for row in rows]
            gt_by_rally = load_for_rallies(conn, rally_ids)
            for rid, aj, cj in rows:
                rallies_data.append({
                    "rally_id": str(rid),
                    "fixture": fixture,
                    "gt": gt_by_rally.get(str(rid), []),
                    "actions_json": aj,
                    "contacts_json": cj,
                    "remap": _build_rally_remap(match_analyses[vid], str(rid)),
                })

    print(f"Loaded {len(rallies_data)} rallies across {len(FRESH_GT_VIDEOS)} videos",
          flush=True)

    # ── per-error rows ──────────────────────────────────────────────────────────
    error_rows: list[dict[str, Any]] = []

    for r in rallies_data:
        actions = _reconstruct_actions(r["actions_json"])
        contacts = _reconstruct_contacts(r["contacts_json"])

        team_assignments_raw: dict[str, str] = (r["actions_json"] or {}).get("teamAssignments", {})
        team_assignments: dict[int, int] = {
            int(k): (0 if v == "A" else 1) for k, v in team_assignments_raw.items()
        }

        # contact index by frame
        contact_by_frame: dict[int, Contact] = {c.frame: c for c in contacts}

        # Pre-compute expected_teams + chain_integrity (over ORIGINAL actions,
        # before any reattribute_players — this mirrors the code path)
        expected_teams = _compute_expected_teams(actions, team_assignments) if team_assignments else [None] * len(actions)
        chain_integrity_flags = _chain_integrity(actions)

        # Normalise GT pids via appliedFullMapping
        remap = r["remap"]
        normalised_gt: list[dict[str, Any]] = []
        for ga in r["gt"]:
            raw_tid = ga.get("trackId", ga.get("playerTrackId"))
            try:
                raw_int = int(raw_tid) if raw_tid is not None else None
            except (TypeError, ValueError):
                raw_int = None
            canonical = remap.get(raw_int) if raw_int is not None else None
            if canonical is None and raw_int is not None and 1 <= raw_int <= 4:
                canonical = raw_int
            entry = dict(ga)
            entry["playerTrackId"] = canonical
            normalised_gt.append(entry)

        # Pipeline actions as dicts (pre-reattribute — we read as-stored)
        pipeline_actions_dicts = [a.to_dict() for a in actions]

        # Match GT → pipeline
        pairs = match_gt_to_pipeline(normalised_gt, pipeline_actions_dicts)

        for gt_action, pl_action in pairs:
            cat, _ = classify_action(gt_action, pl_action, team_assignments_raw)
            if cat != "wrong_cross_team":
                continue

            # Find the action index in the actions list
            pl_frame = pl_action["frame"] if pl_action else None
            action_idx: int | None = None
            if pl_frame is not None:
                for idx, a in enumerate(actions):
                    if a.frame == pl_frame:
                        action_idx = idx
                        break

            if action_idx is None:
                error_rows.append({
                    "fixture": r["fixture"],
                    "rally_id": r["rally_id"][-8:],
                    "frame": pl_frame,
                    "action_type": pl_action.get("action") if pl_action else None,
                    "gt_pid": gt_action.get("playerTrackId"),
                    "pl_pid": pl_action.get("playerTrackId") if pl_action else None,
                    "expected_team": None,
                    "current_team": None,
                    "court_side": None,
                    "current_dist": None,
                    "G1": None,
                    "G2": None,
                    "G3_1.5x": None,
                    "G3_2.0x": None,
                    "G4": None,
                    "G4_reason": "action_idx not found",
                    "best_correct_tid_1.5x": None,
                    "best_correct_dist_1.5x": None,
                    "anomaly": "action_idx not found",
                })
                continue

            action = actions[action_idx]
            contact = contact_by_frame.get(action.frame)
            expected_team = expected_teams[action_idx]
            chain_ok = chain_integrity_flags[action_idx] if action_idx < len(chain_integrity_flags) else False
            current_team = team_assignments.get(action.player_track_id)

            # ── gate evaluations ────────────────────────────────────────────────
            g1 = action.confidence >= 0.7
            g2 = chain_ok

            g3_15_pass = g3_20_pass = False
            best_tid_15 = best_dist_15 = None
            anomaly_note = ""

            if contact is None:
                g3_15_pass = g3_20_pass = False
                anomaly_note = "no contact at action frame"
            elif expected_team is None:
                g3_15_pass = g3_20_pass = False
                anomaly_note = "expected_team is None"
            else:
                g3_15_pass, best_tid_15, best_dist_15 = _eval_G3(
                    contact, expected_team, team_assignments, 1.5
                )
                g3_20_pass, _, _ = _eval_G3(
                    contact, expected_team, team_assignments, 2.0
                )

            g4_pass = False
            g4_reason = "no contact"
            if contact is not None and expected_team is not None:
                g4_pass, g4_reason = _eval_G4(contact, expected_team)

            court_side = contact.court_side if contact else None
            current_dist = contact.player_distance if contact else None
            if current_dist is not None and not math.isfinite(current_dist):
                current_dist = None  # display as None

            error_rows.append({
                "fixture": r["fixture"],
                "rally_id": r["rally_id"][-8:],
                "frame": action.frame,
                "action_type": action.action_type.value,
                "gt_pid": gt_action.get("playerTrackId"),
                "pl_pid": action.player_track_id,
                "expected_team": expected_team,
                "current_team": current_team,
                "court_side": court_side,
                "current_dist": current_dist,
                "G1": g1,
                "G2": g2,
                "G3_1.5x": g3_15_pass,
                "G3_2.0x": g3_20_pass,
                "G4": g4_pass,
                "G4_reason": g4_reason,
                "best_correct_tid_1.5x": best_tid_15,
                "best_correct_dist_1.5x": best_dist_15,
                "anomaly": anomaly_note,
            })

    # ── Print table ─────────────────────────────────────────────────────────────
    print()
    print(f"Found {len(error_rows)} wrong_cross_team errors")
    print()

    # Header
    hdr = (
        f"{'#':>2}  {'fixture':<6}  {'rally':>8}  {'frm':>5}  {'action':<10}  "
        f"{'gt_pid':>6}  {'pl_pid':>6}  "
        f"{'exp_tm':>6}  {'cur_tm':>6}  "
        f"{'court_side':<12}  {'cur_dist':>9}  "
        f"{'G1':>4}  {'G2':>4}  {'G3@1.5':>6}  {'G3@2.0':>6}  {'G4':>4}  "
        f"{'best_cand_tid':>13}  {'best_cand_dist':>14}  {'anomaly'}"
    )
    print(hdr)
    print("-" * len(hdr))

    for i, row in enumerate(error_rows, 1):
        def _bool(v: bool | None) -> str:
            if v is None:
                return "  N/A"
            return "PASS" if v else "FAIL"

        dist_s = f"{row['current_dist']:.4f}" if row["current_dist"] is not None else "    inf"
        bcd_s = f"{row['best_correct_dist_1.5x']:.4f}" if row["best_correct_dist_1.5x"] is not None else "           N/A"
        bct_s = str(row["best_correct_tid_1.5x"]) if row["best_correct_tid_1.5x"] is not None else "N/A"

        print(
            f"{i:>2}  {row['fixture']:<6}  {row['rally_id']:>8}  {row['frame']:>5}  "
            f"{str(row['action_type']):<10}  "
            f"{str(row['gt_pid']):>6}  {str(row['pl_pid']):>6}  "
            f"{str(row['expected_team']):>6}  {str(row['current_team']):>6}  "
            f"{str(row['court_side']):<12}  {dist_s:>9}  "
            f"{_bool(row['G1']):>4}  {_bool(row['G2']):>4}  "
            f"{_bool(row['G3_1.5x']):>6}  {_bool(row['G3_2.0x']):>6}  "
            f"{_bool(row['G4']):>4}  "
            f"{bct_s:>13}  {bcd_s:>14}  {row['anomaly']}"
        )

    # ── Counterfactual scenarios ─────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("COUNTERFACTUAL OVERRIDE-FIRE COUNTS")
    print("(rows where override predicate would FIRE → correction possible)")
    print("=" * 60)

    scenarios: list[tuple[str, Any]] = [
        # name, lambda(row) -> bool (would override fire?)
        ("Current  G1∧G2∧G3@1.5∧G4",
         lambda r: r["G1"] and r["G2"] and r["G3_1.5x"] and r["G4"]),
        ("Drop G4  G1∧G2∧G3@1.5",
         lambda r: r["G1"] and r["G2"] and r["G3_1.5x"]),
        ("Drop G4, loosen G3  G1∧G2∧G3@2.0",
         lambda r: r["G1"] and r["G2"] and r["G3_2.0x"]),
        ("Drop G4+G1  G2∧G3@2.0",
         lambda r: r["G2"] and r["G3_2.0x"]),
        ("G2 only",
         lambda r: r["G2"]),
        ("G2∧G3@1.5",
         lambda r: r["G2"] and r["G3_1.5x"]),
    ]

    total = len(error_rows)
    for name, pred in scenarios:
        fires = [r for r in error_rows if all(r[k] is not None for k in ("G1","G2","G3_1.5x","G3_2.0x","G4")) and pred(r)]
        skipped = sum(1 for r in error_rows if any(r[k] is None for k in ("G1","G2","G3_1.5x","G3_2.0x","G4")))
        eligible = total - skipped
        print(f"  {name:<40}  {len(fires):>2}/{eligible:>2} would fire  "
              f"(+{skipped} skipped due to anomaly)")

    # ── Anomaly summary ──────────────────────────────────────────────────────────
    anomalies = [r for r in error_rows if r["anomaly"]]
    if anomalies:
        print()
        print("ANOMALIES:")
        for r in anomalies:
            print(f"  fixture={r['fixture']} rally={r['rally_id']} frame={r['frame']}: {r['anomaly']}")
    else:
        print()
        print("No anomalies.")

    # ── G4 diagnostic: how often does G4 veto? ──────────────────────────────────
    valid = [r for r in error_rows if r["G4"] is not None]
    g4_fail = [r for r in valid if not r["G4"]]
    g4_pass_list = [r for r in valid if r["G4"]]
    g3_15_would_pass = [r for r in valid if r["G3_1.5x"]]
    g3_g4_both_pass = [r for r in valid if r["G3_1.5x"] and r["G4"]]
    g3_pass_g4_fail = [r for r in valid if r["G3_1.5x"] and not r["G4"]]

    print()
    print("G4 BREAKDOWN (of valid rows, i.e. no anomaly in G4):")
    print(f"  G4 PASS:  {len(g4_pass_list)}/{len(valid)}")
    print(f"  G4 FAIL:  {len(g4_fail)}/{len(valid)}")
    print(f"  G4 fail breakdown:")
    for r in g4_fail:
        print(f"    fixture={r['fixture']} rally={r['rally_id']} frame={r['frame']} "
              f"court_side={r['court_side']!r} expected_team={r['expected_team']} "
              f"reason={r['G4_reason']}")

    print()
    print("G3@1.5 vs G4 interaction:")
    print(f"  G3@1.5 passes:                 {len(g3_15_would_pass)}/{len(valid)}")
    print(f"  G3@1.5 ∧ G4 both pass:         {len(g3_g4_both_pass)}/{len(valid)}")
    print(f"  G3@1.5 passes but G4 blocks:   {len(g3_pass_g4_fail)}/{len(valid)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
