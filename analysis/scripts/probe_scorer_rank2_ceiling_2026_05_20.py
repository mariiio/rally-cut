#!/usr/bin/env python3
"""B1 ceiling probe: can the v2 scorer's rank-2 candidate recover B-only violations?

Reads the per_violation.csv from `probe_violation_oracle_decomp_2026_05_20.py`,
filters to baseline violations classified as `fixed_by_B_only`, then for each
violation:

  1. Determines which contact (prev / curr / both) needs its player flipped
     by re-running Oracle-prev-only and Oracle-curr-only and checking if the
     violation clears.
  2. For each "needs-flip" contact, looks up the contact in contacts_json,
     and reports:
       - Is GT player even in `playerCandidates`? (candidate-generation ceiling)
       - Re-runs the v2 scorer with the pipeline action_type → ranks GT player
         (scorer ranking ceiling)

Output:
  reports/scorer_rank2_ceiling_2026_05_20/summary.json
  reports/scorer_rank2_ceiling_2026_05_20/per_contact.csv
  Console: histogram of GT-rank under the v2 scorer.

The scorer is run with expected_team=None (uninformative 0.5) — this is a
LOWER bound on production scorer rank, since the production team-context
feature, when correctly chain-derived, only helps. If GT lands in scorer
top-3 even without team context, a reranker can recover it; if not, the
scorer's feature space genuinely can't see GT and we need a new signal
class (VLM probe).
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
from rallycut.tracking.dynamic_attribution_scorer import (  # noqa: E402
    DynamicAttributionScorer,
    extract_features,
    position_from_dict,
)
from rallycut.tracking.pid_invariants import Violation  # noqa: E402

DB_DSN = os.environ.get(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5436/rallycut",
)

IN_CSV = Path("reports/violation_oracle_decomp_2026_05_20/per_violation.csv")
OUT_DIR = Path("reports/scorer_rank2_ceiling_2026_05_20")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def detect_named_buckets_set(
    actions: list[dict[str, Any]],
    team_assignments: dict[str, str],
) -> set[tuple[str, int, int]]:
    """Replicates probe_violation_oracle_decomp's named-bucket detector."""
    keys: set[tuple[str, int, int]] = set()
    sorted_actions = sorted(actions, key=lambda x: int(x.get("frame", 0)))
    teams: list[str | None] = []
    for a in sorted_actions:
        tid = a.get("playerTrackId")
        if tid is None or tid == -1:
            teams.append(None)
            continue
        t = team_assignments.get(str(tid))
        teams.append(t if t in ("A", "B") else None)
    for i in range(1, len(sorted_actions)):
        prev_a = str(sorted_actions[i - 1].get("action", "")).lower()
        curr_a = str(sorted_actions[i].get("action", "")).lower()
        t_prev = teams[i - 1]
        t_curr = teams[i]
        if t_prev is None or t_curr is None:
            continue
        same_team = t_prev == t_curr
        if prev_a == "set" and curr_a == "attack" and not same_team:
            keys.add(("set_attack_xteam", i - 1, i))
        if prev_a == "attack" and curr_a == "dig" and same_team:
            keys.add(("attack_dig_sameteam", i - 1, i))
        if prev_a == "serve" and curr_a == "receive" and same_team:
            keys.add(("serve_receive_sameteam", i - 1, i))
        if prev_a == "set" and curr_a == "set":
            keys.add(("set_set_anyteam", i - 1, i))
    return keys


def all_violations(
    rally_id: str,
    actions: list[dict[str, Any]],
    team_assignments: dict[str, str],
) -> set[tuple[str, int, int]]:
    keys = detect_named_buckets_set(actions, team_assignments)
    for v in check_c4_no_same_player_back_to_back(
        rally_id=rally_id, actions=actions, team_assignments=team_assignments,
    ):
        p = v.payload or {}
        keys.add(("C-4", int(p["prev_index"]), int(p["curr_index"])))
    for v in check_c5_mid_possession_crossover(
        rally_id=rally_id, actions=actions, team_assignments=team_assignments,
    ):
        p = v.payload or {}
        keys.add(("C-5", int(p["prev_index"]), int(p["curr_index"])))
    return keys


def match_gt_to_pipeline(
    pipeline_actions: list[dict[str, Any]],
    gt_rows: list[tuple[int, str, int]],
) -> dict[int, tuple[int, str, int]]:
    out: dict[int, tuple[int, str, int]] = {}
    for i, a in enumerate(pipeline_actions):
        p_frame = int(a.get("frame", -10**9))
        p_action = str(a.get("action", "")).lower()
        best = None
        best_delta = 6
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


def fetch_rally(rally_id: str) -> dict[str, Any] | None:
    with psycopg.connect(DB_DSN) as conn:
        cur = conn.execute(
            """
            SELECT v.name, pt.actions_json, pt.contacts_json,
                   pt.positions_json, pt.ball_positions_json
            FROM rallies r
            JOIN videos v ON r.video_id = v.id
            JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE r.id = %s
            """,
            [rally_id],
        )
        row = cur.fetchone()
        if not row:
            return None
        vname, aj, cj, pj, bj = row
        cur = conn.execute(
            """
            SELECT rg.frame, rg.action::text, rg.resolved_track_id
            FROM rally_action_ground_truth rg
            WHERE rg.rally_id = %s AND rg.resolved_track_id IS NOT NULL
            ORDER BY rg.frame
            """,
            [rally_id],
        )
        gt = [(int(f), str(a), int(t)) for f, a, t in cur.fetchall()]

    if isinstance(aj, str):
        aj = json.loads(aj)
    if isinstance(cj, str):
        cj = json.loads(cj)
    if isinstance(pj, str):
        pj = json.loads(pj)
    if isinstance(bj, str):
        bj = json.loads(bj)
    actions = aj.get("actions") or []
    teams = aj.get("teamAssignments") or {}
    contacts = (cj or {}).get("contacts") or []
    positions = pj or []
    ball_positions = bj or []
    actions_sorted = sorted(actions, key=lambda x: int(x.get("frame", 0)))
    return {
        "video": vname,
        "actions": actions_sorted,
        "teams": teams,
        "contacts": contacts,
        "positions": positions,
        "ball_positions": ball_positions,
        "gt": gt,
    }


def find_contact_for_frame(
    contacts: list[dict[str, Any]], frame: int, tol: int = 3,
) -> dict[str, Any] | None:
    """Pipeline-action and contact may differ by 1-2 frames; nearest within tol."""
    best = None
    best_delta = tol + 1
    for c in contacts:
        d = abs(int(c.get("frame", -10**9)) - frame)
        if d < best_delta:
            best_delta = d
            best = c
    return best


def rerank_with_scorer(
    rally: dict[str, Any],
    contact: dict[str, Any],
    action_type: str,
    scorer: DynamicAttributionScorer,
) -> list[tuple[int, float]] | None:
    """Return list of (track_id, prob) sorted desc. None if scorer unavailable."""
    cand_tids = [int(pc[0]) for pc in (contact.get("playerCandidates") or [])]
    if not cand_tids:
        return None
    positions_like = [position_from_dict(p) for p in rally["positions"]]
    cf_list = []
    for tid in cand_tids:
        cf = extract_features(
            positions_like,
            tid,
            int(contact["frame"]),
            float(contact.get("ballX", 0.5)),
            float(contact.get("ballY", 0.5)),
            prev_action_tid=-1,
            post_ball_x=None,
            post_ball_y=None,
            expected_team=None,
            team_assignments=None,
        )
        if cf is not None:
            cf_list.append(cf)
    if not cf_list:
        return None
    probs = scorer.score(action_type, cf_list)
    if probs is None:
        return None
    ranked = sorted(
        zip([c.track_id for c in cf_list], probs, strict=True),
        key=lambda x: -x[1],
    )
    return ranked


def gt_rank_in(ranked: list[tuple[int, float]], gt_tid: int) -> int:
    """1-indexed rank of GT in ranked list. 0 if not present."""
    for i, (tid, _) in enumerate(ranked, start=1):
        if tid == gt_tid:
            return i
    return 0


def main() -> int:
    if not IN_CSV.exists():
        print(f"ERROR: {IN_CSV} not found. Run probe_violation_oracle_decomp first.",
              file=sys.stderr)
        return 1
    rows = list(csv.DictReader(open(IN_CSV)))
    b_only = [r for r in rows if r["classification"] == "fixed_by_B_only"]
    print(f"Loading scorer...", flush=True)
    os.environ.setdefault("USE_DYNAMIC_ATTRIBUTION_SCORER", "1")
    scorer = DynamicAttributionScorer()
    print(f"Found {len(b_only)} B-only violations across "
          f"{len({r['rally_id'] for r in b_only})} rallies", flush=True)

    by_rally: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in b_only:
        by_rally[r["rally_id"]].append(r)

    per_contact_rows: list[dict[str, Any]] = []
    flip_targets: list[dict[str, Any]] = []  # (rally_id, frame, action, gt_tid)

    print(f"\n=== Phase 1: identify which contact needs flipping per violation ===", flush=True)
    n_rallies = len(by_rally)
    for ridx, (rally_id, viols) in enumerate(by_rally.items()):
        rally = fetch_rally(rally_id)
        if rally is None:
            continue
        actions = rally["actions"]
        teams = rally["teams"]
        matches = match_gt_to_pipeline(actions, rally["gt"])

        baseline_set = all_violations(rally_id, actions, teams)
        for v in viols:
            bucket = v["bucket"]
            # Re-locate the violation key from prev/curr frames
            target_pf = int(v["prev_frame"])
            target_cf = int(v["curr_frame"])
            # Find indices in actions list
            i_prev = i_curr = -1
            for idx, a in enumerate(actions):
                if int(a.get("frame", -1)) == target_pf:
                    i_prev = idx
                if int(a.get("frame", -1)) == target_cf:
                    i_curr = idx
            if i_prev < 0 or i_curr < 0:
                continue
            key = (bucket, i_prev, i_curr)
            if key not in baseline_set:
                continue

            # Oracle-prev-only
            acts_p = deepcopy(actions)
            if i_prev in matches:
                acts_p[i_prev]["playerTrackId"] = int(matches[i_prev][2])
            prev_only_fixes = key not in all_violations(rally_id, acts_p, teams)

            # Oracle-curr-only
            acts_c = deepcopy(actions)
            if i_curr in matches:
                acts_c[i_curr]["playerTrackId"] = int(matches[i_curr][2])
            curr_only_fixes = key not in all_violations(rally_id, acts_c, teams)

            needs: list[tuple[int, int]] = []  # (action_idx, gt_tid)
            if prev_only_fixes and i_prev in matches:
                needs.append((i_prev, int(matches[i_prev][2])))
            if curr_only_fixes and i_curr in matches:
                needs.append((i_curr, int(matches[i_curr][2])))
            if not needs:
                # Must require both to flip
                if i_prev in matches:
                    needs.append((i_prev, int(matches[i_prev][2])))
                if i_curr in matches:
                    needs.append((i_curr, int(matches[i_curr][2])))

            for action_idx, gt_tid in needs:
                a = actions[action_idx]
                flip_targets.append({
                    "rally_id": rally_id,
                    "video": rally["video"],
                    "bucket": bucket,
                    "action_frame": int(a.get("frame", -1)),
                    "action_type": str(a.get("action", "")),
                    "current_pid": int(a.get("playerTrackId", -1)),
                    "gt_tid": gt_tid,
                    "fix_side": ("prev" if action_idx == i_prev else "curr"),
                    "_contacts": rally["contacts"],
                    "_positions": rally["positions"],
                    "_ball_positions": rally["ball_positions"],
                })
        if (ridx + 1) % 20 == 0:
            print(f"  [{ridx+1}/{n_rallies}] rallies; {len(flip_targets)} flip-targets so far",
                  flush=True)

    print(f"\n  Total flip-targets: {len(flip_targets)}", flush=True)

    print(f"\n=== Phase 2: score each flip-target ===", flush=True)
    rank_hist: Counter = Counter()
    by_bucket: dict[str, Counter] = defaultdict(Counter)
    not_in_candidates = 0
    no_scorer_output = 0

    for tgt in flip_targets:
        # Reconstruct rally dict for rerank_with_scorer
        rally_lite = {
            "positions": tgt["_positions"],
            "ball_positions": tgt["_ball_positions"],
        }
        contact = find_contact_for_frame(tgt["_contacts"], tgt["action_frame"], tol=3)
        if contact is None:
            no_scorer_output += 1
            continue
        cand_tids = [int(pc[0]) for pc in (contact.get("playerCandidates") or [])]
        gt_in_candidates = tgt["gt_tid"] in cand_tids
        if not gt_in_candidates:
            not_in_candidates += 1
            rank_hist["not_in_candidates"] += 1
            by_bucket[tgt["bucket"]]["not_in_candidates"] += 1
            per_contact_rows.append({
                "rally_id": tgt["rally_id"],
                "video": tgt["video"],
                "bucket": tgt["bucket"],
                "fix_side": tgt["fix_side"],
                "action_frame": tgt["action_frame"],
                "action_type": tgt["action_type"],
                "current_pid": tgt["current_pid"],
                "gt_tid": tgt["gt_tid"],
                "n_candidates": len(cand_tids),
                "gt_in_candidates": False,
                "gt_rank": 0,
                "scorer_top1": -1,
            })
            continue

        ranked = rerank_with_scorer(rally_lite, contact, tgt["action_type"], scorer)
        if ranked is None:
            no_scorer_output += 1
            rank_hist["scorer_unavailable"] += 1
            by_bucket[tgt["bucket"]]["scorer_unavailable"] += 1
            continue
        rank = gt_rank_in(ranked, tgt["gt_tid"])
        if rank == 0:
            rank_hist["scored_but_dropped"] += 1
            by_bucket[tgt["bucket"]]["scored_but_dropped"] += 1
        elif rank == 1:
            rank_hist["rank_1"] += 1
            by_bucket[tgt["bucket"]]["rank_1"] += 1
        elif rank == 2:
            rank_hist["rank_2"] += 1
            by_bucket[tgt["bucket"]]["rank_2"] += 1
        elif rank == 3:
            rank_hist["rank_3"] += 1
            by_bucket[tgt["bucket"]]["rank_3"] += 1
        else:
            rank_hist[f"rank_{rank}_plus"] += 1
            by_bucket[tgt["bucket"]][f"rank_{rank}_plus"] += 1

        per_contact_rows.append({
            "rally_id": tgt["rally_id"],
            "video": tgt["video"],
            "bucket": tgt["bucket"],
            "fix_side": tgt["fix_side"],
            "action_frame": tgt["action_frame"],
            "action_type": tgt["action_type"],
            "current_pid": tgt["current_pid"],
            "gt_tid": tgt["gt_tid"],
            "n_candidates": len(cand_tids),
            "gt_in_candidates": True,
            "gt_rank": rank,
            "scorer_top1": int(ranked[0][0]),
        })

    n_total = len(flip_targets)
    n_in_cands = n_total - not_in_candidates - no_scorer_output
    print(f"\n=== Histogram (n={n_total} flip-targets) ===", flush=True)
    for k, v in sorted(rank_hist.items(), key=lambda x: (
        x[0] != "rank_1", x[0] != "rank_2", x[0] != "rank_3", x[0],
    )):
        print(f"  {k:24s} {v:>4d}  ({100*v/n_total:.1f}%)")
    print()
    print(f"  GT in candidates AT ALL:   {n_in_cands}/{n_total} ({100*n_in_cands/n_total:.1f}%)",
          flush=True)
    top3 = rank_hist["rank_1"] + rank_hist["rank_2"] + rank_hist["rank_3"]
    print(f"  GT in scorer top-3:        {top3}/{n_total} ({100*top3/n_total:.1f}%)",
          flush=True)
    print(f"  Reranker max-recoverable:  {top3}/{n_total} ({100*top3/n_total:.1f}%) "
          f"of B-only flip-targets", flush=True)

    print(f"\n=== Per-bucket histogram ===", flush=True)
    print(f"{'bucket':<26s} {'tot':>4s} {'r1':>4s} {'r2':>4s} {'r3':>4s} "
          f"{'r4+':>4s} {'dropped':>8s} {'not_in':>7s}")
    for b in sorted(by_bucket.keys()):
        c = by_bucket[b]
        tot = sum(c.values())
        r4p = sum(v for k, v in c.items() if k.startswith("rank_") and "plus" in k)
        print(f"{b:<26s} {tot:>4d} "
              f"{c.get('rank_1',0):>4d} "
              f"{c.get('rank_2',0):>4d} "
              f"{c.get('rank_3',0):>4d} "
              f"{r4p:>4d} "
              f"{c.get('scored_but_dropped',0):>8d} "
              f"{c.get('not_in_candidates',0):>7d}")

    out = {
        "n_flip_targets": n_total,
        "rank_histogram": dict(rank_hist),
        "by_bucket": {b: dict(c) for b, c in by_bucket.items()},
        "gt_in_candidates_pct": 100 * n_in_cands / max(n_total, 1),
        "gt_in_scorer_top3_pct": 100 * top3 / max(n_total, 1),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote summary -> {OUT_DIR/'summary.json'}", flush=True)
    if per_contact_rows:
        with open(OUT_DIR / "per_contact.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per_contact_rows[0].keys()))
            w.writeheader()
            w.writerows(per_contact_rows)
        print(f"Wrote per-contact CSV -> {OUT_DIR/'per_contact.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
