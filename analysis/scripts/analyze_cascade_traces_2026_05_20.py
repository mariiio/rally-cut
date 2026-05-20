#!/usr/bin/env python3
"""Analyze cascade traces from audit_cascade_override_2026_05_20.py and
identify which stage overrode the v2 scorer's correct top-1 pick on each
of the 28 rank_1 flip-target contacts.

Reads:
  - reports/scorer_rank2_ceiling_2026_05_20/per_contact.csv  (rank_1 rows)
  - reports/cascade_override_audit_2026_05_20/traces/{rally_id}.trace.json

For each flip-target contact, walks the snapshots in order and identifies:
  - `scorer_pick`: the playerTrackId at `after_dynamic_scorer` snapshot
  - `final_pick`: the playerTrackId at `final` snapshot
  - `override_stage`: the FIRST stage AFTER `after_dynamic_scorer` where
                     playerTrackId changes from the scorer pick.

Tags each flip-target with:
  - `kind="scorer_was_overridden"`: scorer_pick == gt_tid, final_pick != gt_tid.
                                    Identifies override stage.
  - `kind="scorer_already_wrong"`:  scorer_pick != gt_tid. Means the
                                    probe's expected_team=None scoring
                                    disagreed with production scoring
                                    (chain-context confound).
  - `kind="no_override_in_trace"`:  scorer_pick == gt_tid AND final_pick == gt_tid.
                                    B-only flag may be stale (DB state drifted).
  - `kind="trace_missing"`:          trace file not found for the rally.
  - `kind="match_failed"`:           trace contact not found by frame within ±3.

Output:
  reports/cascade_override_audit_2026_05_20/per_contact_override.csv
  reports/cascade_override_audit_2026_05_20/summary.json
  Console: histogram by override_stage and by kind.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
IN_CSV = ANALYSIS_DIR / "reports" / "scorer_rank2_ceiling_2026_05_20" / "per_contact.csv"
TRACE_DIR = ANALYSIS_DIR / "reports" / "cascade_override_audit_2026_05_20" / "traces"
OUT_DIR = ANALYSIS_DIR / "reports" / "cascade_override_audit_2026_05_20"

# Stages in order — must match the labels passed to _tr.snapshot() in
# action_classifier.classify_rally_actions().
STAGE_ORDER = (
    "after_classify_rally",
    "after_serve_prepend",
    "after_repair_action_sequence",
    "after_viterbi_decode_actions",
    "after_validate_action_sequence",
    "after_assign_court_side_from_teams",
    "after_reattribute_players",
    "after_dynamic_scorer",
    "after_visual_reattribute",
    "after_apply_sequence_override",
    "after_apply_decoder_labels",
    "final",
)
SCORER_STAGE = "after_dynamic_scorer"
POST_SCORER = STAGE_ORDER[STAGE_ORDER.index(SCORER_STAGE) + 1:]


def find_contact_in_trace(
    per_contact: dict[str, dict[str, dict[str, Any]]],
    target_frame: int,
    tol: int = 3,
) -> dict[str, dict[str, Any]] | None:
    """Pipeline action frame may differ from snapshot frame by ≤2; tolerant match."""
    best = None
    best_delta = tol + 1
    for frame_str, by_stage in per_contact.items():
        d = abs(int(frame_str) - target_frame)
        if d < best_delta:
            best_delta = d
            best = by_stage
    return best


def main() -> int:
    if not IN_CSV.exists():
        print(f"ERROR: {IN_CSV} not found", file=sys.stderr)
        return 1
    if not TRACE_DIR.exists():
        print(f"ERROR: {TRACE_DIR} not found. Run audit_cascade_override first.",
              file=sys.stderr)
        return 1

    with open(IN_CSV) as fh:
        rank1 = [r for r in csv.DictReader(fh) if r["gt_rank"] == "1"]
    print(f"{len(rank1)} rank_1 flip-targets across "
          f"{len({r['rally_id'] for r in rank1})} rallies", flush=True)

    rows_out: list[dict[str, Any]] = []
    by_stage: Counter = Counter()
    by_kind: Counter = Counter()

    for r in rank1:
        rally_id = r["rally_id"]
        target_frame = int(r["action_frame"])
        gt_tid = int(r["gt_tid"])
        trace_path = TRACE_DIR / f"{rally_id}.trace.json"
        if not trace_path.exists():
            kind = "trace_missing"
            by_kind[kind] += 1
            rows_out.append({**r, "kind": kind, "scorer_pick": "",
                             "final_pick": "", "override_stage": ""})
            continue
        trace = json.loads(trace_path.read_text())
        by_contact = find_contact_in_trace(trace["per_contact"], target_frame)
        if by_contact is None:
            kind = "match_failed"
            by_kind[kind] += 1
            rows_out.append({**r, "kind": kind, "scorer_pick": "",
                             "final_pick": "", "override_stage": ""})
            continue
        scorer_snap = by_contact.get(SCORER_STAGE)
        final_snap = by_contact.get("final")
        scorer_pick = scorer_snap["player_track_id"] if scorer_snap else -1
        final_pick = final_snap["player_track_id"] if final_snap else -1

        if scorer_pick != gt_tid:
            kind = "scorer_already_wrong"
            by_kind[kind] += 1
            rows_out.append({
                **r, "kind": kind,
                "scorer_pick": scorer_pick,
                "final_pick": final_pick,
                "override_stage": "",
            })
            continue

        # scorer_pick == gt_tid: find the first downstream stage where pid flips
        override_stage = ""
        cur = scorer_pick
        for stage in POST_SCORER:
            snap = by_contact.get(stage)
            if snap is None:
                continue
            if snap["player_track_id"] != cur:
                override_stage = stage
                break
        if final_pick == gt_tid:
            kind = "no_override_in_trace"
            by_kind[kind] += 1
        else:
            kind = "scorer_was_overridden"
            by_kind[kind] += 1
            by_stage[override_stage or "unknown"] += 1
        rows_out.append({
            **r, "kind": kind,
            "scorer_pick": scorer_pick,
            "final_pick": final_pick,
            "override_stage": override_stage,
        })

    print("\n=== By kind ===")
    for k, n in by_kind.most_common():
        print(f"  {k:32s} {n:>3d}")

    print("\n=== Override stage histogram (scorer_was_overridden only) ===")
    for s, n in by_stage.most_common():
        print(f"  {s:38s} {n:>3d}")

    out_csv = OUT_DIR / "per_contact_override.csv"
    if rows_out:
        with open(out_csv, "w", newline="") as fh:
            fieldnames: list[str] = []
            for r in rows_out:
                for k in r:
                    if k not in fieldnames:
                        fieldnames.append(k)
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows_out)
        print(f"\nWrote per-contact override CSV -> {out_csv}")
    summary = {
        "by_kind": dict(by_kind),
        "by_override_stage": dict(by_stage),
        "n_flip_targets": len(rank1),
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Wrote summary -> {OUT_DIR/'summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
