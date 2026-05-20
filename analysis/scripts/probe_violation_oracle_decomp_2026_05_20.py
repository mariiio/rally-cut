#!/usr/bin/env python3
"""Oracle decomposition of v11 coherence violations on trusted-32.

For every consecutive-action pair on trusted-32, count violations under
four oracle conditions:

  Baseline   – production actions_json as-is
  Oracle-A   – substitute action_type with GT (player kept from pipeline)
  Oracle-B   – substitute playerTrackId with GT (action kept from pipeline)
  Oracle-AB  – substitute both

For each baseline violation, classify which oracle "fixes" it
(fixed_by_A_only / fixed_by_B_only / fixed_by_both / fixed_by_neither).
The 4-cell histogram per bucket is the diagnostic payload — it tells us
whether the lever is upstream (action GBM) or downstream (scorer), or
whether the violations are genuinely coupled and require joint reasoning.

Substitution semantics: for each pipeline action, find the closest GT row
within ±5 frames preferring same action_type. If no GT match within
window, keep the pipeline field unchanged for that contact.

Buckets:
  set_attack_xteam      – prev=SET, curr=ATTACK, teams differ
  attack_dig_sameteam   – prev=ATTACK, curr=DIG, teams same
  serve_receive_sameteam– prev=SERVE, curr=RECEIVE, teams same
  set_set_anyteam       – prev=SET, curr=SET (any team)
  C-4                   – same-player back-to-back (production definition)
  C-5                   – cross-team without possession-transfer prev (prod def)

Output:
  reports/violation_oracle_decomp_2026_05_20/summary.json
  reports/violation_oracle_decomp_2026_05_20/per_violation.csv
  Console: per-bucket totals + 4-cell histogram.
"""
from __future__ import annotations

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import psycopg

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rallycut.tracking.coherence_invariants import (  # noqa: E402
    check_c4_no_same_player_back_to_back,
    check_c5_mid_possession_crossover,
)
from rallycut.tracking.pid_invariants import Violation  # noqa: E402

DB_DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5436/rallycut",
)

TRUSTED_32 = (
    "titi", "toto", "lulu", "wawa", "caco", "cece", "cici", "cuco",
    "gaga", "gigi", "kaka", "kiki", "keke", "koko", "kuku",
    "juju", "yeye", "gugu", "mame", "meme", "mimi", "moma", "mumu",
    "papa", "pepe", "pipi", "popo", "pupu", "veve", "vivi", "vovo",
    "haha",
)

OUT_DIR = Path("reports/violation_oracle_decomp_2026_05_20")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONDITIONS = ("baseline", "oracle_A", "oracle_B", "oracle_AB")

NAMED_BUCKETS = (
    "set_attack_xteam",
    "attack_dig_sameteam",
    "serve_receive_sameteam",
    "set_set_anyteam",
)
ALL_BUCKETS = (*NAMED_BUCKETS, "C-4", "C-5")


def match_gt_to_pipeline(
    pipeline_actions: list[dict[str, Any]],
    gt_rows: list[tuple[int, str, int]],
) -> dict[int, tuple[int, str, int]]:
    """Map pipeline-action index -> (gt_frame, gt_action_lower, gt_resolved_tid).

    Match rule: closest GT row within ±5 frames, preferring same action_type.
    """
    out: dict[int, tuple[int, str, int]] = {}
    for i, a in enumerate(pipeline_actions):
        p_frame = int(a.get("frame", -10**9))
        p_action = str(a.get("action", "")).lower()
        best = None
        best_delta = 6
        # First pass: same action_type
        for g_frame, g_action, g_tid in gt_rows:
            if g_action.lower() != p_action:
                continue
            d = abs(g_frame - p_frame)
            if d < best_delta:
                best_delta = d
                best = (g_frame, g_action.lower(), g_tid)
        if best is None:
            best_delta = 6
            for g_frame, g_action, g_tid in gt_rows:
                d = abs(g_frame - p_frame)
                if d < best_delta:
                    best_delta = d
                    best = (g_frame, g_action.lower(), g_tid)
        if best is not None:
            out[i] = best
    return out


def apply_oracle(
    pipeline_actions: list[dict[str, Any]],
    matches: dict[int, tuple[int, str, int]],
    *,
    sub_action: bool,
    sub_player: bool,
) -> list[dict[str, Any]]:
    """Return a copy of pipeline_actions with selected fields swapped to GT.

    Only contacts that matched a GT row are mutated; unmatched stay as-is.
    """
    out: list[dict[str, Any]] = []
    for i, a in enumerate(pipeline_actions):
        a_new = deepcopy(a)
        if i in matches:
            _, gt_action, gt_tid = matches[i]
            if sub_action:
                a_new["action"] = gt_action
            if sub_player:
                a_new["playerTrackId"] = int(gt_tid)
        out.append(a_new)
    return out


def detect_named_violations(
    rally_id: str,
    actions: list[dict[str, Any]],
    team_assignments: dict[str, str],
) -> list[tuple[str, int, int]]:
    """Return list of (bucket_name, prev_index, curr_index) for the 4 named buckets."""
    sorted_actions = sorted(actions, key=lambda x: int(x.get("frame", 0)))
    # Build idx -> team
    teams: list[str | None] = []
    for a in sorted_actions:
        tid = a.get("playerTrackId")
        if tid is None or tid == -1:
            teams.append(None)
            continue
        t = team_assignments.get(str(tid))
        teams.append(t if t in ("A", "B") else None)

    out: list[tuple[str, int, int]] = []
    for i in range(1, len(sorted_actions)):
        prev = sorted_actions[i - 1]
        curr = sorted_actions[i]
        prev_a = str(prev.get("action", "")).lower()
        curr_a = str(curr.get("action", "")).lower()
        t_prev = teams[i - 1]
        t_curr = teams[i]
        if t_prev is None or t_curr is None:
            continue
        same_team = t_prev == t_curr
        if prev_a == "set" and curr_a == "attack" and not same_team:
            out.append(("set_attack_xteam", i - 1, i))
        if prev_a == "attack" and curr_a == "dig" and same_team:
            out.append(("attack_dig_sameteam", i - 1, i))
        if prev_a == "serve" and curr_a == "receive" and same_team:
            out.append(("serve_receive_sameteam", i - 1, i))
        if prev_a == "set" and curr_a == "set":
            out.append(("set_set_anyteam", i - 1, i))
    return out


def detect_all_violations(
    rally_id: str,
    actions: list[dict[str, Any]],
    team_assignments: dict[str, str],
) -> set[tuple[str, int, int]]:
    """Union of (bucket, prev_index, curr_index) keys across 6 buckets.

    Note: we use prev/curr index pairs as the violation key so we can
    track the same logical violation across baseline+oracles, regardless
    of whether actions/players changed.
    """
    keys: set[tuple[str, int, int]] = set()
    for bucket, i_prev, i_curr in detect_named_violations(
        rally_id, actions, team_assignments,
    ):
        keys.add((bucket, i_prev, i_curr))

    c4_violations: list[Violation] = check_c4_no_same_player_back_to_back(
        rally_id=rally_id, actions=actions, team_assignments=team_assignments,
    )
    for v in c4_violations:
        p = v.payload or {}
        keys.add(("C-4", int(p["prev_index"]), int(p["curr_index"])))

    c5_violations: list[Violation] = check_c5_mid_possession_crossover(
        rally_id=rally_id, actions=actions, team_assignments=team_assignments,
    )
    for v in c5_violations:
        p = v.payload or {}
        keys.add(("C-5", int(p["prev_index"]), int(p["curr_index"])))
    return keys


def fetch_corpus() -> list[dict[str, Any]]:
    """One row per (rally, video) with actions_json + team_assignments + GT rows."""
    out: list[dict[str, Any]] = []
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            """
            SELECT v.id, v.name, r.id, pt.actions_json
            FROM videos v
            JOIN rallies r ON r.video_id = v.id
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE v.name = ANY(%s)
              AND (r.status = 'CONFIRMED' OR r.status IS NULL)
            ORDER BY v.name, r.start_ms
            """,
            [list(TRUSTED_32)],
        )
        rally_rows = cur.fetchall()

        cur = conn.execute(
            """
            SELECT r.id, rg.frame, rg.action::text, rg.resolved_track_id
            FROM rally_action_ground_truth rg
            JOIN rallies r ON rg.rally_id = r.id
            JOIN videos v ON r.video_id = v.id
            WHERE v.name = ANY(%s)
              AND rg.resolved_track_id IS NOT NULL
            ORDER BY r.id, rg.frame
            """,
            [list(TRUSTED_32)],
        )
        gt_by_rally: dict[str, list[tuple[int, str, int]]] = defaultdict(list)
        for rid, frame, action, tid in cur.fetchall():
            gt_by_rally[str(rid)].append((int(frame), str(action), int(tid)))

    for vid, vname, rid, aj in rally_rows:
        if not isinstance(aj, dict):
            try:
                aj = json.loads(aj) if isinstance(aj, str) else None
            except Exception:
                aj = None
        if not isinstance(aj, dict):
            continue
        actions = aj.get("actions") or []
        teams = aj.get("teamAssignments") or {}
        gt = gt_by_rally.get(str(rid), [])
        if not gt:
            continue  # skip rallies with no attribution GT
        # Sort actions by frame (defensive) and keep stable index ordering
        actions_sorted = sorted(actions, key=lambda x: int(x.get("frame", 0)))
        out.append({
            "video": vname,
            "rally_id": str(rid),
            "actions": actions_sorted,
            "teams": teams,
            "gt": gt,
        })
    return out


def main() -> int:
    print("Loading trusted-32 corpus from production DB...", flush=True)
    rallies = fetch_corpus()
    print(f"  {len(rallies)} rallies with attribution GT loaded", flush=True)

    # Per-bucket counters per condition
    counts: dict[str, dict[str, int]] = {
        b: {c: 0 for c in CONDITIONS} for b in ALL_BUCKETS
    }
    # Per-baseline-violation: which oracles fixed it?
    classifications: dict[str, Counter] = {b: Counter() for b in ALL_BUCKETS}
    per_violation_rows: list[dict[str, Any]] = []

    for idx, r in enumerate(rallies):
        rally_id = r["rally_id"]
        vname = r["video"]
        actions = r["actions"]
        teams = r["teams"]
        gt = r["gt"]
        matches = match_gt_to_pipeline(actions, gt)

        # Build per-condition action lists
        variants = {
            "baseline":   actions,
            "oracle_A":   apply_oracle(actions, matches, sub_action=True,  sub_player=False),
            "oracle_B":   apply_oracle(actions, matches, sub_action=False, sub_player=True),
            "oracle_AB":  apply_oracle(actions, matches, sub_action=True,  sub_player=True),
        }
        viol_sets: dict[str, set[tuple[str, int, int]]] = {}
        for cond, acts in variants.items():
            viol_sets[cond] = detect_all_violations(rally_id, acts, teams)
            for key in viol_sets[cond]:
                bucket = key[0]
                counts[bucket][cond] += 1

        # Classify each baseline violation
        for key in viol_sets["baseline"]:
            bucket, i_prev, i_curr = key
            in_A  = key not in viol_sets["oracle_A"]   # i.e., FIXED by A
            in_B  = key not in viol_sets["oracle_B"]
            in_AB = key not in viol_sets["oracle_AB"]
            if in_A and in_B:
                cls = "fixed_by_both"
            elif in_A and not in_B:
                cls = "fixed_by_A_only"
            elif in_B and not in_A:
                cls = "fixed_by_B_only"
            elif in_AB:
                # neither A nor B alone fixed it, but AB did -> genuinely coupled
                cls = "fixed_only_jointly"
            else:
                cls = "fixed_by_neither"
            classifications[bucket][cls] += 1

            prev = actions[i_prev]
            curr = actions[i_curr]
            per_violation_rows.append({
                "video": vname,
                "rally_id": rally_id,
                "bucket": bucket,
                "prev_frame": int(prev.get("frame", 0)),
                "curr_frame": int(curr.get("frame", 0)),
                "prev_action": str(prev.get("action", "")),
                "curr_action": str(curr.get("action", "")),
                "prev_pid": int(prev.get("playerTrackId", -1)),
                "curr_pid": int(curr.get("playerTrackId", -1)),
                "classification": cls,
                "prev_matched_gt": i_prev in matches,
                "curr_matched_gt": i_curr in matches,
            })

        if (idx + 1) % 20 == 0:
            print(f"  [{idx+1}/{len(rallies)}] processed (cum baseline violations: "
                  f"{sum(counts[b]['baseline'] for b in ALL_BUCKETS)})", flush=True)

    print("\n=== Per-condition violation counts ===\n", flush=True)
    print(f"{'bucket':<26s} {'base':>6s} {'orA':>6s} {'orB':>6s} {'orAB':>6s}")
    for b in ALL_BUCKETS:
        c = counts[b]
        print(f"{b:<26s} "
              f"{c['baseline']:>6d} "
              f"{c['oracle_A']:>6d} "
              f"{c['oracle_B']:>6d} "
              f"{c['oracle_AB']:>6d}")

    print("\n=== Per-baseline-violation classification ===\n", flush=True)
    print(f"{'bucket':<26s} {'total':>6s} {'A_only':>7s} {'B_only':>7s} "
          f"{'both':>6s} {'joint':>6s} {'neither':>8s}")
    for b in ALL_BUCKETS:
        cls = classifications[b]
        total = sum(cls.values())
        print(f"{b:<26s} "
              f"{total:>6d} "
              f"{cls.get('fixed_by_A_only', 0):>7d} "
              f"{cls.get('fixed_by_B_only', 0):>7d} "
              f"{cls.get('fixed_by_both', 0):>6d} "
              f"{cls.get('fixed_only_jointly', 0):>6d} "
              f"{cls.get('fixed_by_neither', 0):>8d}")

    # JSON snapshot
    out_summary = {
        "counts": counts,
        "classifications": {b: dict(c) for b, c in classifications.items()},
        "n_rallies": len(rallies),
        "n_violations_total": sum(counts[b]["baseline"] for b in ALL_BUCKETS),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(out_summary, indent=2))
    print(f"\nWrote summary -> {OUT_DIR/'summary.json'}", flush=True)

    csv_path = OUT_DIR / "per_violation.csv"
    if per_violation_rows:
        with open(csv_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per_violation_rows[0].keys()))
            w.writeheader()
            w.writerows(per_violation_rows)
        print(f"Wrote per-violation CSV -> {csv_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
