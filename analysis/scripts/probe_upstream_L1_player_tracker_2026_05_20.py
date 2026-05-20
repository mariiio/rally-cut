#!/usr/bin/env python3
"""L1: player-tracker contact-coverage probe.

For each wrong-attribution contact:
  - FAIL definition: GT player has no bbox within ±5 of GT contact frame
    (scorer's _find_pos returns None, GT candidate dropped from scoring).
  - Oracle: substitute any GT-player bbox available in rally, re-run scorer.
  - Realistic interventions:
    R1.a: widen _find_pos tolerance from ±5 to ±10
    R1.b: widen to ±15
    R1.c: interpolate bbox across gaps (≤10 frames)
    R1.d: detect ID-switch (GT player tracked under different track_id)

Failure-mode categorization per contact:
  never_tracked / short_gap_le_10 / long_gap_gt_10

Output: reports/upstream_bottleneck_2026_05_20/L1.json
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ANALYSIS_DIR / "scripts"))

from _upstream_probe_common import (  # noqa: E402
    fetch_rally_state,
    load_wrong_attribution_corpus,
    rescore_contact,
)

OUT_DIR = ANALYSIS_DIR / "reports" / "upstream_bottleneck_2026_05_20"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def find_player_tracked_frames(positions: list[dict], gt_pid: int) -> list[int]:
    """Return sorted list of frames where gt_pid has a bbox."""
    frames = sorted({int(p.get("frameNumber", -1))
                     for p in positions
                     if int(p.get("trackId", -1)) == gt_pid})
    return [f for f in frames if f >= 0]


def categorize_failure(
    tracked_frames: list[int], contact_frame: int,
) -> tuple[str, int | None]:
    """Return (failure_mode, gap_to_contact). gap is the smallest |frame - contact_frame|
    where gt_pid is tracked. None if never tracked."""
    if not tracked_frames:
        return "never_tracked", None
    gaps = [abs(f - contact_frame) for f in tracked_frames]
    min_gap = min(gaps)
    if min_gap <= 5:
        return "tracked_at_contact_unexpected", min_gap  # shouldn't be a L1 fail
    if min_gap <= 10:
        return "short_gap_le_10", min_gap
    return "long_gap_gt_10", min_gap


def main() -> int:
    print("Loading wrong-attribution corpus...", flush=True)
    rows = load_wrong_attribution_corpus()
    print(f"  {len(rows)} wrong-attribution contacts", flush=True)

    l1_failures: list[dict] = []
    by_category: Counter = Counter()
    oracle_recoveries = 0
    realistic_recoveries: dict[str, int] = {
        "widen_pm10": 0, "widen_pm15": 0, "interpolate_short_gap": 0,
    }

    for i, row in enumerate(rows):
        rally = fetch_rally_state(row.rally_id)
        if rally is None:
            continue
        tracked = find_player_tracked_frames(rally["positions"], row.gt_pid)
        category, min_gap = categorize_failure(tracked, row.action_frame)
        if category == "tracked_at_contact_unexpected":
            continue  # GT player IS tracked within ±5; this contact's L1 doesn't fail
        l1_failures.append({
            "rally_id": row.rally_id,
            "video": row.video,
            "action_frame": row.action_frame,
            "action_type": row.action_type,
            "pipeline_pid": row.pipeline_pid,
            "gt_pid": row.gt_pid,
            "category": category,
            "min_gap": min_gap,
        })
        by_category[category] += 1

        # Oracle: forge a bbox for GT player at contact frame using any tracked bbox
        cand_tids: list[int] = []
        for c in rally["contacts"]:
            if abs(int(c.get("frame", -1)) - row.action_frame) <= 3:
                cand_tids = [int(pc[0]) for pc in (c.get("playerCandidates") or [])]
                break
        if row.gt_pid not in cand_tids:
            cand_tids = [*cand_tids, row.gt_pid]

        gt_bbox = next(
            (p for p in rally["positions"]
             if int(p.get("trackId", -1)) == row.gt_pid),
            None,
        )
        if gt_bbox is not None:
            patched_positions = list(rally["positions"])
            patched_positions.append({**gt_bbox, "frameNumber": row.action_frame})
            rally_oracle = {**rally, "positions": patched_positions}
            contact_dict = next(
                (c for c in rally["contacts"]
                 if abs(int(c.get("frame", -1)) - row.action_frame) <= 3),
                None,
            )
            if contact_dict:
                pick = rescore_contact(
                    rally_oracle, contact_dict, row.action_type, cand_tids,
                    expected_team=None,
                )
                if pick == row.gt_pid:
                    oracle_recoveries += 1

        # Realistic interventions
        if min_gap is not None:
            if min_gap <= 10:
                realistic_recoveries["widen_pm10"] += 1
            if min_gap <= 15:
                realistic_recoveries["widen_pm15"] += 1
            before = [f for f in tracked if f < row.action_frame and row.action_frame - f <= 15]
            after = [f for f in tracked if f > row.action_frame and f - row.action_frame <= 15]
            if before and after:
                realistic_recoveries["interpolate_short_gap"] += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(rows)}] processed", flush=True)

    out = {
        "n_total_wrong": len(rows),
        "n_l1_failures": len(l1_failures),
        "by_category": dict(by_category),
        "oracle_recoveries": oracle_recoveries,
        "realistic_recoveries": realistic_recoveries,
        "failures": l1_failures,
    }
    (OUT_DIR / "L1.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {OUT_DIR/'L1.json'}", flush=True)
    print(f"L1 failures: {len(l1_failures)}/{len(rows)}", flush=True)
    print(f"  by category: {dict(by_category)}", flush=True)
    print(f"  oracle recoveries: {oracle_recoveries}", flush=True)
    print(f"  realistic recoveries: {realistic_recoveries}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
