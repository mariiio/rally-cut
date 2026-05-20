#!/usr/bin/env python3
"""L3: contact-frame regression accuracy probe.

For each wrong-attribution contact, compute |predicted_frame - GT_frame|
(captured in `pipeline_match_delta` at corpus-load time). Histogram the
distribution; correlate magnitude with attribution error rate. Oracle:
substitute GT frame (both ± delta directions), re-extract features at
GT frame, re-score. Count recoveries.

Output: reports/upstream_bottleneck_2026_05_20/L3.json
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


def main() -> int:
    print("Loading wrong-attribution corpus + computing frame deltas...", flush=True)
    rows = load_wrong_attribution_corpus()
    deltas: list[int] = []
    oracle_recoveries = 0
    delta_histogram: Counter = Counter()

    for i, row in enumerate(rows):
        rally = fetch_rally_state(row.rally_id)
        if rally is None:
            continue
        delta = row.pipeline_match_delta
        deltas.append(delta)
        delta_histogram[delta] += 1

        contact = next(
            (c for c in rally["contacts"]
             if abs(int(c.get("frame", -1)) - row.action_frame) <= 3),
            None,
        )
        if contact is None or delta == 0:
            continue
        cand_tids = [int(pc[0]) for pc in (contact.get("playerCandidates") or [])]
        # Try both delta directions; recovery counted if EITHER produces GT
        for sign in (-1, +1):
            gt_frame_guess = row.action_frame + sign * delta
            pick = rescore_contact(
                rally, contact, row.action_type, cand_tids,
                expected_team=None,
                contact_frame_override=gt_frame_guess,
            )
            if pick == row.gt_pid:
                oracle_recoveries += 1
                break

        if (i + 1) % 50 == 0:
            print(f"  [{i+1}/{len(rows)}] processed", flush=True)

    out = {
        "n_total_wrong": len(rows),
        "delta_histogram": dict(delta_histogram),
        "mean_delta": sum(deltas) / max(len(deltas), 1),
        "max_delta": max(deltas) if deltas else 0,
        "oracle_recoveries": oracle_recoveries,
    }
    (OUT_DIR / "L3.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_DIR/'L3.json'}", flush=True)
    print(f"  mean |Δframe|: {out['mean_delta']:.2f}", flush=True)
    print(f"  max |Δframe|: {out['max_delta']}", flush=True)
    print(f"  delta histogram: {dict(delta_histogram)}", flush=True)
    print(f"  oracle recoveries (frame substitution): {oracle_recoveries}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
