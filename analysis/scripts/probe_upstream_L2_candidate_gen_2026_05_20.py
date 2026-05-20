#!/usr/bin/env python3
"""L2: candidate generation in contact_detector probe.

For each wrong-attribution contact, check if GT player is in
contact.playerCandidates. If not (L2 fail), compute oracle: force GT
into candidates, re-run scorer with chain context. Categorize the
eliminating rule (distance / court-side / not-found-in-positions).

Output: reports/upstream_bottleneck_2026_05_20/L2.json
"""
from __future__ import annotations

import json
import math
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


def categorize_elimination(
    rally: dict, contact: dict, gt_pid: int,
) -> str:
    """Best-effort: identify why GT player was excluded from candidates.

    Categories:
      not_in_positions: gt_pid never appears in positions for this rally
      no_bbox_at_contact: gt_pid exists but has no bbox within ±5 of contact
      too_far: gt_pid bbox exists at contact but is far from ball (> 5x nearest)
      unknown: catch-all
    """
    contact_frame = int(contact.get("frame", 0))
    bboxes_at_frame = [
        p for p in rally["positions"]
        if abs(int(p.get("frameNumber", -1)) - contact_frame) <= 5
        and int(p.get("trackId", -1)) == gt_pid
    ]
    if not any(
        int(p.get("trackId", -1)) == gt_pid for p in rally["positions"]
    ):
        return "not_in_positions"
    if not bboxes_at_frame:
        return "no_bbox_at_contact"
    # Compute distance from gt bbox to ball
    gt_box = bboxes_at_frame[0]
    bx, by = float(contact.get("ballX", 0.5)), float(contact.get("ballY", 0.5))
    gx = float(gt_box.get("x", 0)) + float(gt_box.get("width", 0)) / 2
    gy = float(gt_box.get("y", 0)) + float(gt_box.get("height", 0)) / 2
    gt_dist = math.hypot(gx - bx, gy - by)
    cand_dists = []
    for pc in (contact.get("playerCandidates") or []):
        cand_dists.append(float(pc[1]))
    if cand_dists:
        nearest = min(cand_dists)
        if gt_dist > 5 * nearest:
            return "too_far"
    return "unknown"


def main() -> int:
    print("Loading wrong-attribution corpus...", flush=True)
    rows = load_wrong_attribution_corpus()
    print(f"  {len(rows)} wrong-attribution contacts", flush=True)

    l2_failures: list[dict] = []
    by_category: Counter = Counter()
    oracle_recoveries = 0

    for i, row in enumerate(rows):
        rally = fetch_rally_state(row.rally_id)
        if rally is None:
            continue
        contact = next(
            (c for c in rally["contacts"]
             if abs(int(c.get("frame", -1)) - row.action_frame) <= 3),
            None,
        )
        if contact is None:
            continue
        cand_tids = [int(pc[0]) for pc in (contact.get("playerCandidates") or [])]
        if row.gt_pid in cand_tids:
            continue  # L2 doesn't fail; GT is in candidates already
        category = categorize_elimination(rally, contact, row.gt_pid)
        l2_failures.append({
            "rally_id": row.rally_id,
            "video": row.video,
            "action_frame": row.action_frame,
            "action_type": row.action_type,
            "pipeline_pid": row.pipeline_pid,
            "gt_pid": row.gt_pid,
            "category": category,
            "cand_tids": cand_tids,
        })
        by_category[category] += 1

        # Oracle: force GT into candidates
        cand_with_gt = [*cand_tids, row.gt_pid]
        pick = rescore_contact(
            rally, contact, row.action_type, cand_with_gt, expected_team=None,
        )
        if pick == row.gt_pid:
            oracle_recoveries += 1

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(rows)}] processed", flush=True)

    out = {
        "n_total_wrong": len(rows),
        "n_l2_failures": len(l2_failures),
        "by_category": dict(by_category),
        "oracle_recoveries": oracle_recoveries,
        "failures": l2_failures,
    }
    (OUT_DIR / "L2.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {OUT_DIR/'L2.json'}", flush=True)
    print(f"L2 failures: {len(l2_failures)}/{len(rows)}", flush=True)
    print(f"  by category: {dict(by_category)}", flush=True)
    print(f"  oracle recoveries: {oracle_recoveries}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
