"""Inspect dd042609 r10 cost matrix from MatchSolver probe sidecar.

Determines whether the wrong assignment is "matcher chose lower cost"
(cost matrix supports the wrong answer; profile contamination or
cost-construction issue) vs "matcher chose worse than GT-correct"
(impossible since Hungarian is optimal — would indicate hard
constraint blocking the right answer).

Run after `MATCH_PLAYERS_PROBE=1 rallycut match-players <vid>`.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


def main() -> None:
    if len(sys.argv) > 1:
        sidecar_path = Path(sys.argv[1])
    else:
        # Most recent dd042609 sidecar
        probe_dir = Path(__file__).resolve().parent.parent / "reports" / "profile_drift_probe"
        candidates = sorted(probe_dir.glob("dd042609_*.json"))
        if not candidates:
            sys.exit("no dd042609 sidecar in reports/profile_drift_probe/")
        sidecar_path = candidates[-1]
    print(f"Sidecar: {sidecar_path}")

    data = json.loads(sidecar_path.read_text())
    rally_ids = data["rally_ids"]
    print(f"Total rallies: {len(rally_ids)}")

    # Find r10 (rally starting with f811cc63)
    r10_idx = None
    for i, rid in enumerate(rally_ids):
        if rid.startswith("f811cc63"):
            r10_idx = i
            break
    if r10_idx is None:
        sys.exit("r10 (f811cc63) not found in rally_ids")
    print(f"r10 rally_idx: {r10_idx} (full rally_id: {rally_ids[r10_idx]})")

    iter_records = data["iter_records"]
    r10_iters = [r for r in iter_records if r["rally_idx"] == r10_idx]
    print(f"r10 has {len(r10_iters)} iter records")

    # Show assignment trajectory across iterations
    print("\n=== r10 assignment trajectory ===")
    for rec in r10_iters:
        print(f"  iter {rec['iteration']}: assignment={rec['assignment']}, "
              f"changes_from_prev={list(rec['changed_from_prev'].keys())}")

    final = r10_iters[-1]
    print()
    print("=" * 60)
    print("=== r10 FINAL ITERATION ===")
    print("=" * 60)
    print(f"iteration: {final['iteration']}")
    print(f"top_tracks: {final['top_tracks']}")
    print(f"cluster_ids: {final['cluster_ids']}")
    print(f"assignment: {final['assignment']}")

    cm = np.array(final["cost_matrix"])
    print(f"\ncost_matrix shape: {cm.shape}")
    print(f"\nFull cost matrix (rows = tracks, cols = clusters):")
    header = "          " + "  ".join(f"C{c}".rjust(8) for c in final["cluster_ids"])
    print(header)
    for ti, t in enumerate(final["top_tracks"]):
        row = "  ".join(f"{cm[ti, ci]:.4f}".rjust(8) for ci in range(cm.shape[1]))
        print(f"  T{t:3d}    {row}")

    print(f"\nrow_margins (best vs 2nd-best per track): {final['row_margins']}")

    # Cost of the matcher's chosen assignment
    chosen_cost = 0.0
    print("\n=== CHOSEN assignment cost breakdown ===")
    for tid_str, cid in final["assignment"].items():
        tid = int(tid_str)
        ti = final["top_tracks"].index(tid)
        ci = final["cluster_ids"].index(cid)
        chosen_cost += cm[ti, ci]
        print(f"  T{tid} -> cluster {cid}: cost {cm[ti, ci]:.4f}")
    print(f"CHOSEN total cost: {chosen_cost:.4f}")

    # GT-correct assignment per Phase 2 diagnostic:
    # T1->PID4 (presumed; not in BAD list, only 3 GT samples), T2->PID2,
    # T3->PID3, T16->PID1
    gt = {1: 4, 2: 2, 3: 3, 16: 1}
    print("\n=== GT-CORRECT assignment cost breakdown ===")
    print(f"GT mapping: {gt}")
    gt_cost = 0.0
    missing = False
    for tid, cid in gt.items():
        if tid not in final["top_tracks"] or cid not in final["cluster_ids"]:
            print(f"  T{tid} -> cluster {cid}: MISSING (track or cluster not in matrix)")
            missing = True
            continue
        ti = final["top_tracks"].index(tid)
        ci = final["cluster_ids"].index(cid)
        gt_cost += cm[ti, ci]
        print(f"  T{tid} -> cluster {cid}: cost {cm[ti, ci]:.4f}")
    if missing:
        print("WARN: GT mapping cannot be fully scored against this cost matrix")
    print(f"GT-CORRECT total cost: {gt_cost:.4f}")

    print()
    gap = gt_cost - chosen_cost
    print(f"Cost gap (GT - chosen): {gap:.4f}")
    if gap > 0:
        print(f"  -> Cost matrix supports the WRONG answer by {gap:.4f}.")
        print(f"     Matcher is acting optimally given the cost matrix.")
        print(f"     Diagnosis: profile contamination OR cost-construction issue.")
    elif gap < 0:
        print(f"  -> GT-correct has LOWER cost than chosen ({-gap:.4f} delta).")
        print(f"     Hungarian wouldn't choose this without a hard constraint blocking GT.")
        print(f"     Diagnosis: team-pair partition guard is excluding the GT solution.")
    else:
        print(f"  -> Cost-tied. Hungarian tiebreaker decided.")


if __name__ == "__main__":
    main()
