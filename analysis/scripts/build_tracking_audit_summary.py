"""Build reports/tracking_audit/_summary.json from per-rally audit JSONs.

Mirrors the `_summary.json` shape the `rallycut evaluate-tracking audit`
subcommand emits, but works for the retrack-based flow too, where
`--retrack --cached --audit-out <dir>` writes only the per-rally files.

Usage:
    uv run python scripts/build_tracking_audit_summary.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("summary")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=Path("reports/tracking_audit"))
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    out = args.out or (args.audit_dir / "_summary.json")

    summary: list[dict] = []
    for path in sorted(args.audit_dir.glob("*.json")):
        if path.name == "_summary.json":
            continue
        with open(path) as f:
            a = json.load(f)

        per_gt = a.get("perGt", [])
        missed_ranges_total = sum(
            sum(len(ranges) for ranges in g.get("missedByCause", {}).values())
            for g in per_gt
        )
        worst_coverage = min((g["coverage"] for g in per_gt), default=1.0)

        summary.append({
            "rallyId": a["rallyId"],
            "videoId": a["videoId"],
            "hota": a.get("hota"),
            "mota": a.get("mota"),
            "realSwitches": a.get("aggregateRealSwitches", 0),
            "missedRanges": missed_ranges_total,
            "worstCoverage": worst_coverage,
            "courtSideFlip": a.get("convention", {}).get("courtSideFlip", False),
            "teamLabelFlip": a.get("convention", {}).get("teamLabelFlip", False),
            "perGt": [
                {
                    "gtTrackId": g["gtTrackId"],
                    "coverage": g["coverage"],
                    "distinctPredIds": g["distinctPredIds"],
                    "realSwitchCount": len(g.get("realSwitches", [])),
                    "missCauseCounts": {
                        cause: sum(end - start + 1 for start, end in ranges)
                        for cause, ranges in g.get("missedByCause", {}).items()
                    },
                }
                for g in per_gt
            ],
        })

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"rallies": summary}, indent=2))
    logger.info(f"Wrote {out} with {len(summary)} rallies")


if __name__ == "__main__":
    main()
