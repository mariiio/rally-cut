"""Attribution baseline on the 3 fresh-GT videos (cece/gigi/wawa).

These are the only videos with current, retrack-aligned GT as of 2026-05-11
(action_ground_truth_json gets wiped on retrack — see redetect_all_actions
memory entry).

Reports:
- Per-video: correct / wrong_cross_team / wrong_same_team / wrong_unknown / missing / abstained
- Per-action-type breakdown of the same categories
- Per-rally diagnostic table (one row per rally so anomalies are visible)
- Per-action error rows (gt vs pl) for the workstream's "where do errors live" analysis

Run from analysis/:
    uv run python scripts/measure_attribution_fresh_gt.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from rallycut.evaluation.attribution_bench import (
    CATEGORIES,
    MATCH_TOLERANCE_FRAMES,
    WRONG_CATEGORIES,
    aggregate,
    score_rally,
)
from rallycut.evaluation.db import get_connection

FRESH_GT_VIDEOS = {
    "cece": "950fbe5d-fdad-4862-b05d-8b374bdd5ec6",
    "gigi": "b097dd2a-6953-4e0e-a603-5be3552f462e",
    "wawa": "5c756c41-1cc1-4486-a95c-97398912cfbe",
}

OUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "reports" / "attribution_baseline" / "fresh_gt_2026_05_11.json"
)
ACTION_TYPES = ("serve", "receive", "set", "attack", "dig", "block")


def _build_rally_remap(
    match_analysis: dict[str, Any] | None,
    rally_id: str,
) -> dict[int, int]:
    """Per-rally raw-track-id → canonical-PID map (`appliedFullMapping`).

    GT for these 3 videos was labeled against RAW tracker IDs visible in the
    labeling UI. After `remap-track-ids` runs, `actions_json.playerTrackId`
    holds canonical PIDs (1–4). To compare GT vs current attribution we need
    to translate GT through the per-rally remap. The mapping lives on
    `match_analysis_json.rallies[].appliedFullMapping`. Falls back to
    `trackToPlayer` when appliedFullMapping is absent, then to identity.
    """
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


def main() -> int:
    rallies: list[dict[str, Any]] = []
    n_rows_per_video: dict[str, int] = {}
    n_gt_unmappable_per_video: dict[str, int] = {}

    with get_connection() as conn, conn.cursor() as cur:
        # Load match_analysis once per video for the per-rally remap
        match_analyses: dict[str, dict[str, Any]] = {}
        for fixture, video_id in FRESH_GT_VIDEOS.items():
            cur.execute(
                "SELECT match_analysis_json FROM videos WHERE id = %s",
                (video_id,),
            )
            ma_row = cur.fetchone()
            match_analyses[video_id] = ma_row[0] if ma_row and ma_row[0] else {}

        for fixture, video_id in FRESH_GT_VIDEOS.items():
            cur.execute(
                """
                SELECT r.id, r.start_ms, r.end_ms,
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
            n_rows_per_video[fixture] = len(rows)
            n_unmappable = 0
            print(f"[{fixture}] {len(rows)} GT rallies", flush=True)

            for rid, start_ms, end_ms, gt, actions_json, contacts_json, ptids in rows:
                pipeline_actions = actions_json.get("actions", []) if actions_json else []
                team_assignments = actions_json.get("teamAssignments", {}) if actions_json else {}
                serving_team = actions_json.get("servingTeam") if actions_json else None
                contacts = contacts_json.get("contacts", []) if contacts_json else []

                # Per-rally raw-track-id → canonical-PID map
                remap = _build_rally_remap(match_analyses[video_id], str(rid))

                # Normalise GT schema: action_ground_truth_json stores raw `trackId`;
                # translate through appliedFullMapping to canonical PID. Mark
                # untranslatable entries with playerTrackId=None so the bench
                # categorises them as wrong_unknown_team (visible signal, not
                # silent failure).
                normalised_gt = []
                for a in (gt or []):
                    raw_tid = a.get("trackId", a.get("playerTrackId"))
                    try:
                        raw_int = int(raw_tid) if raw_tid is not None else None
                    except (TypeError, ValueError):
                        raw_int = None
                    canonical = remap.get(raw_int) if raw_int is not None else None
                    if canonical is None and raw_int is not None:
                        # If GT pid is already in 1-4 and remap missing it,
                        # accept as-is (defensive).
                        if 1 <= raw_int <= 4:
                            canonical = raw_int
                        else:
                            n_unmappable += 1
                    entry = dict(a)
                    entry["playerTrackId"] = canonical
                    entry["raw_trackId"] = raw_int
                    normalised_gt.append(entry)

                rally = {
                    "rally_id": str(rid),
                    "video_id": video_id,
                    "fixture": fixture,
                    "start_ms": start_ms,
                    "end_ms": end_ms,
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
            n_gt_unmappable_per_video[fixture] = n_unmappable

    agg = aggregate(rallies)
    combined = agg["combined"]["counts"]
    combined_rates = agg["combined"]["rates"]
    n_total = combined["n_gt_actions"]
    wrong_total = sum(combined[k] for k in WRONG_CATEGORIES)

    # ---- Per-action-type breakdown (across all 3 videos) ----
    per_type: dict[str, dict[str, int]] = defaultdict(
        lambda: dict.fromkeys(CATEGORIES, 0) | {"n": 0}
    )
    for r in rallies:
        for m in r["matches"]:
            t = m["gt_action"] or "unknown"
            per_type[t]["n"] += 1
            per_type[t][m["category"]] += 1

    # ---- Per-rally summary table ----
    per_rally_rows: list[dict[str, Any]] = []
    for r in rallies:
        t = r["rally_totals"]
        per_rally_rows.append({
            "fixture": r["fixture"],
            "rally_id": r["rally_id"][:8],
            "n": t["n_gt_actions"],
            "correct": t["correct"],
            "wrong_cross": t["wrong_cross_team"],
            "wrong_same": t["wrong_same_team"],
            "wrong_unk": t["wrong_unknown_team"],
            "missing": t["missing"],
            "abstain": t["abstained"],
        })

    # ---- Per-error detail rows (only wrong-category matches) ----
    error_rows: list[dict[str, Any]] = []
    for r in rallies:
        ta = r["team_assignments"]
        for m in r["matches"]:
            if m["category"] in WRONG_CATEGORIES:
                gt_pid = m["gt_pid"]
                pl_pid = m["pl_pid"]
                error_rows.append({
                    "fixture": r["fixture"],
                    "rally_id": r["rally_id"][:8],
                    "gt_frame": m["gt_frame"],
                    "pl_frame": m["pl_frame"],
                    "gt_action": m["gt_action"],
                    "pl_action": m["pl_action"],
                    "gt_pid": gt_pid,
                    "pl_pid": pl_pid,
                    "gt_team": ta.get(str(gt_pid)),
                    "pl_team": ta.get(str(pl_pid)),
                    "category": m["category"],
                })

    # ---- Print summary ----
    print()
    print("=" * 78)
    print("ATTRIBUTION BASELINE — fresh GT (3 videos, 2026-05-11)")
    print(f"Match tolerance: ±{MATCH_TOLERANCE_FRAMES} frames")
    print("=" * 78)
    print()
    print("PER-VIDEO")
    print(f"  {'fix':<6} {'n':>4}  {'correct':>14}  {'wrong':>10}  "
          f"{'miss':>5}  {'abs':>4}")
    for fx in sorted(agg["per_fixture"].keys()):
        c = agg["per_fixture"][fx]["counts"]
        r = agg["per_fixture"][fx]["rates"]
        w = sum(c[k] for k in WRONG_CATEGORIES)
        print(f"  {fx:<6} {c['n_gt_actions']:>4d}  "
              f"{c['correct']:>3d} ({r['correct_rate']:>6.1%})  "
              f"{w:>3d} ({r['wrong_rate']:>5.1%})  "
              f"{c['missing']:>3d}    {c['abstained']:>3d}")
    print()
    print(f"COMBINED (n={n_total} GT actions)")
    print(f"  correct:          {combined['correct']:>4d} ({combined_rates['correct_rate']:>6.1%})")
    print(f"  wrong (any):      {wrong_total:>4d} ({combined_rates['wrong_rate']:>6.1%})")
    print(f"    cross_team:     {combined['wrong_cross_team']:>4d}")
    print(f"    same_team:      {combined['wrong_same_team']:>4d}")
    print(f"    unknown_team:   {combined['wrong_unknown_team']:>4d}")
    print(f"  missing:          {combined['missing']:>4d} ({combined_rates['missing_rate']:>6.1%})")
    print(f"  abstained:        {combined['abstained']:>4d} ({combined_rates['abstained_rate']:>6.1%})")

    print()
    print("PER-ACTION-TYPE (across all 3 videos)")
    print(f"  {'type':<8} {'n':>4}  {'correct':>14}  {'cross':>5}  "
          f"{'same':>5}  {'unk':>4}  {'miss':>5}  {'abs':>4}")
    for t in ACTION_TYPES:
        if t not in per_type:
            continue
        d = per_type[t]
        n = d["n"]
        c = d["correct"]
        rate = c / n if n else 0.0
        print(f"  {t:<8} {n:>4d}  {c:>3d} ({rate:>6.1%})  "
              f"{d['wrong_cross_team']:>5d}  "
              f"{d['wrong_same_team']:>5d}  "
              f"{d['wrong_unknown_team']:>4d}  "
              f"{d['missing']:>5d}  {d['abstained']:>4d}")

    print()
    print("PER-RALLY")
    print(f"  {'fix':<6} {'rally':<10} {'n':>3}  {'correct':>3}  "
          f"{'xt':>2}  {'st':>2}  {'unk':>3}  {'miss':>4}  {'abs':>3}")
    for row in sorted(per_rally_rows, key=lambda r: (r["fixture"], r["rally_id"])):
        print(f"  {row['fixture']:<6} {row['rally_id']:<10} "
              f"{row['n']:>3d}  {row['correct']:>3d}  "
              f"{row['wrong_cross']:>2d}  {row['wrong_same']:>2d}  "
              f"{row['wrong_unk']:>3d}  {row['missing']:>4d}  {row['abstain']:>3d}")

    print()
    print(f"PER-ERROR (showing all {len(error_rows)} wrong attributions)")
    print(f"  {'fix':<6} {'rally':<10} {'gt_f':>4} {'pl_f':>4}  "
          f"{'gt_act':<8} {'pl_act':<8}  "
          f"{'gt_pid':>3}({'T':<1}) → {'pl_pid':>3}({'T':<1})  cat")
    for e in sorted(error_rows, key=lambda r: (r["fixture"], r["rally_id"], r["gt_frame"] or 0)):
        gt_t = e["gt_team"] or "?"
        pl_t = e["pl_team"] or "?"
        pl_f = "----" if e["pl_frame"] is None else f"{e['pl_frame']:>4d}"
        pl_act = e["pl_action"] or "----"
        pl_pid = e["pl_pid"] if e["pl_pid"] is not None else "?"
        gt_pid = e["gt_pid"] if e["gt_pid"] is not None else "?"
        print(f"  {e['fixture']:<6} {e['rally_id']:<10} "
              f"{e['gt_frame']:>4d} {pl_f}  "
              f"{e['gt_action']:<8} {pl_act:<8}  "
              f"{str(gt_pid):>3}({gt_t}) → {str(pl_pid):>3}({pl_t})  "
              f"{e['category']}")

    # ---- Persist JSON for downstream analysis / A-B comparison ----
    payload = {
        "generated_at": "2026-05-11",
        "match_tolerance_frames": MATCH_TOLERANCE_FRAMES,
        "videos": FRESH_GT_VIDEOS,
        "n_rallies_per_video": n_rows_per_video,
        "n_gt_unmappable_per_video": n_gt_unmappable_per_video,
        "per_fixture": {fx: v for fx, v in agg["per_fixture"].items()},
        "combined": agg["combined"],
        "per_type": dict(per_type),
        "per_rally": per_rally_rows,
        "errors": error_rows,
        "rallies": rallies,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, default=str))
    print()
    print(f"Wrote: {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
