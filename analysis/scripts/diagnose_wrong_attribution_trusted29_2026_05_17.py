#!/usr/bin/env python3
"""Categorize the 105 wrong-attribution residual on trusted-29 (v4+v2 state).

For each GT row where the contact IS matched (within ±5f) but the
playerTrackId picked is wrong, classify by:
  - CROSS_TEAM: picked player is on the opposite team from GT player
  - WITHIN_TEAM_WRONG: picked player is teammate of GT player
  - GT_NOT_IN_CANDIDATES: GT player wasn't even in the action's candidate list
  - CANDIDATE_RANK: how far down the candidate list the GT player ranked
  - SCORE_GAP: gap between picked candidate's score and GT candidate's score

The output tells us:
  - How much is fixable by adding new attribution features (within-team errors)
  - How much is fixable by team-aware rules (cross-team errors)
  - How much is structural (GT not in candidates → upstream tracking)
"""
from __future__ import annotations

import json
import os
from collections import defaultdict

import psycopg

DB_DSN = "postgresql://postgres:postgres@localhost:5436/rallycut"
TRUSTED_29 = (
    "titi", "toto", "lulu", "wawa", "caco", "cece", "cici", "cuco",
    "gaga", "kaka", "kiki", "juju", "yeye", "keke",
    "gigi", "gugu", "mame", "meme", "mimi", "moma", "mumu",
    "papa", "pepe", "pipi", "popo", "pupu", "veve", "vivi", "vovo",
)
MATCH_WINDOW = 5


def main() -> int:
    print("Fetching trusted-29 GT + match analysis data...", flush=True)
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            """
            SELECT v.name, r.id, rg.action::text, rg.frame, rg.resolved_track_id,
                   pt.actions_json, v.match_analysis_json
            FROM rally_action_ground_truth rg
            JOIN rallies r ON rg.rally_id = r.id
            JOIN videos v ON r.video_id = v.id
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE v.name = ANY(%s) AND rg.resolved_track_id IS NOT NULL
            """,
            [list(TRUSTED_29)],
        )
        rows = cur.fetchall()
    print(f"Got {len(rows)} GT rows", flush=True)

    # Build team assignments per rally from actions_json.teamAssignments
    teams_by_rally: dict[str, dict[int, str]] = {}
    for _video, rid, _ga, _gf, _gtid, aj, _ma in rows:
        if rid in teams_by_rally:
            continue
        aj_dict = aj if isinstance(aj, dict) else (json.loads(aj) if aj else {})
        ta = aj_dict.get("teamAssignments") or {}
        teams_by_rally[rid] = {int(k): v for k, v in ta.items()}

    # Now analyze each GT row
    categories = defaultdict(int)
    by_action_category = defaultdict(lambda: defaultdict(int))
    gt_rank_dist = defaultdict(int)
    sample_cases = defaultdict(list)
    n_matched = 0
    n_correct = 0
    n_wrong = 0
    n_unmatched = 0
    n_no_team_info = 0

    for video, rid, gt_action_raw, gt_frame, gt_tid, aj, _ma in rows:
        gt_action = gt_action_raw.upper()
        aj = aj if isinstance(aj, dict) else (json.loads(aj) if aj else {})
        actions = aj.get("actions") or []

        # Find matched action (same logic as measure_attribution_trusted_29)
        best = None
        best_delta = MATCH_WINDOW + 1
        for a in actions:
            if (a.get("action") or "").upper() != gt_action:
                continue
            delta = abs(int(a.get("frame", -10**9)) - gt_frame)
            if delta < best_delta:
                best_delta = delta
                best = a
        if best is None:
            best_delta = MATCH_WINDOW + 1
            for a in actions:
                delta = abs(int(a.get("frame", -10**9)) - gt_frame)
                if delta < best_delta:
                    best_delta = delta
                    best = a

        if best is None:
            n_unmatched += 1
            continue
        n_matched += 1

        picked = int(best.get("playerTrackId", -1))
        if picked == gt_tid:
            n_correct += 1
            continue

        n_wrong += 1

        # Categorize via per-rally teamAssignments
        teams = teams_by_rally.get(rid, {})
        gt_team = teams.get(gt_tid)
        picked_team = teams.get(picked)

        if not teams:
            cat = "NO_TEAM_INFO"
            n_no_team_info += 1
        elif gt_team is None or picked_team is None:
            cat = "TEAM_LOOKUP_FAILED"
        elif gt_team != picked_team:
            cat = "CROSS_TEAM"
        else:
            cat = "WITHIN_TEAM_WRONG"

        categories[cat] += 1
        by_action_category[gt_action][cat] += 1

        if len(sample_cases[cat]) < 5:
            sample_cases[cat].append({
                "video": video, "frame": gt_frame, "action": gt_action,
                "gt_tid": gt_tid, "picked": picked,
                "gt_team": gt_team, "picked_team": picked_team,
            })

    print(f"\nTotals: matched={n_matched}, correct={n_correct}, wrong={n_wrong}, unmatched={n_unmatched}")
    print(f"NO_TEAM_INFO videos: {n_no_team_info} (excluded from team-aware categorization)")
    print()
    print("Wrong-attribution category breakdown:")
    for cat, n in sorted(categories.items(), key=lambda x: -x[1]):
        pct = n / max(1, n_wrong) * 100
        print(f"  {cat:<40s}{n:>4d}  ({pct:>5.1f}%)")

    print(f"\nPer action type wrong-attribution category:")
    for act in ("SERVE", "RECEIVE", "SET", "ATTACK", "DIG", "BLOCK"):
        if act not in by_action_category:
            continue
        total = sum(by_action_category[act].values())
        print(f"  {act} ({total} wrong):")
        for cat, n in sorted(by_action_category[act].items(), key=lambda x: -x[1]):
            print(f"    {cat:<40s}{n:>4d}  ({n/total*100:.1f}%)")

    print(f"\nSample cases per category (top 5):")
    for cat in ("CROSS_TEAM", "WITHIN_TEAM_WRONG", "NO_TEAM_INFO", "TEAM_LOOKUP_FAILED"):
        if cat not in sample_cases:
            continue
        print(f"\n  {cat}:")
        for c in sample_cases[cat]:
            print(f"    {c['video']:<8s} f{c['frame']:>4d} {c['action']:<8s} "
                  f"gt_tid={c['gt_tid']} picked={c['picked']} "
                  f"gt_team={c['gt_team']} picked_team={c['picked_team']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
