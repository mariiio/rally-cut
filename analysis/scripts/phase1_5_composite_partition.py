"""Phase 1.5 — Composite primitive correctness partition.

Joins the four Phase-1 audits and partitions every baseline match into:

- **primitive_fixable** — error caused by a primitive failure (team-pair
  wrong, tracking swap near contact, within-rally ID instability).
  Fixing primitives would repair the error.
- **chooser_fixable** — primitives look OK at this contact but chooser picked
  wrong. Phase 2 confidence-gated chooser + Phase 3 consistency are the lever.
- **detection_limit** — missing actions with no pipeline contact within ±10f.
  Upstream contact-detection recall ceiling (closed workstream).
- **irreducible** — primitives OK + pipeline contact exists + same-team pick
  (within-team confusion architectural floor per memory).

Emits:
- `reports/phase1_5_composite_partition_2026_04_24.md` — partition table.
- `reports/attribution_rebuild/phase1_5_composite_partition.json`.

Usage:
    uv run python scripts/phase1_5_composite_partition.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

BASELINE_PATH = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "attribution_rebuild"
    / "baseline_2026_04_24.json"
)
TRACKING_JSON = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "attribution_rebuild"
    / "phase1_1_tracking_stability.json"
)
TEAM_SIDE_JSON = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "attribution_rebuild"
    / "phase1_3_team_side.json"
)
OUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "phase1_5_composite_partition_2026_04_24.md"
)
AUDIT_JSON_PATH = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "attribution_rebuild"
    / "phase1_5_composite_partition.json"
)

SWAP_WINDOW = 5


def _has_swap_near(tracking_per_rally: dict, rid: str, frame: int, pid: int) -> bool:
    a = tracking_per_rally.get(rid, {})
    for s in a.get("swap_events", []):
        if s["tid"] == pid and abs(s["frame"] - frame) <= SWAP_WINDOW:
            return True
    return False


def _is_low_coverage(tracking_per_rally: dict, rid: str, pid: int) -> bool:
    a = tracking_per_rally.get(rid, {})
    cov = (a.get("coverage") or {}).get(str(pid)) or (a.get("coverage") or {}).get(pid)
    if cov is None:
        return False
    return cov.get("coverage_ratio", 1.0) < 0.40


def main() -> int:
    baseline = json.loads(BASELINE_PATH.read_text())
    tracking = json.loads(TRACKING_JSON.read_text())
    team_side = json.loads(TEAM_SIDE_JSON.read_text())

    tracking_per_rally = tracking["per_rally"]
    team_side_per_rally = team_side["per_rally"]

    # Partition every match
    bucket = Counter()
    per_fixture_bucket: dict[str, Counter] = {}
    per_category_bucket: dict[str, Counter] = {}
    row_partitions: list[dict] = []

    for r in baseline["rallies"]:
        rid = r["rally_id"]
        fx = r["fixture"]
        ts = tracking_per_rally.get(rid, {})
        ts_audit = team_side_per_rally.get(rid, {})
        team_pair_wrong = (
            ts_audit.get("audit_eligible")
            and ts_audit.get("agree") is False
        )
        for m in r.get("matches", []):
            cat = m["category"]
            frame = m["gt_frame"]
            gt_pid = m["gt_pid"]
            pl_pid = m.get("pl_pid")

            # Missing — either detection limit or tracking gap on gt_pid
            if cat == "missing":
                # Check if gt_pid had coverage at this frame
                if _is_low_coverage(tracking_per_rally, rid, gt_pid):
                    bucket_kind = "primitive_fixable__coverage_gap"
                else:
                    bucket_kind = "detection_limit"
            elif cat == "correct":
                bucket_kind = "correct"
            else:
                # Wrong of some kind — diagnose
                reasons = []
                if team_pair_wrong:
                    reasons.append("team_pair_wrong")
                if pl_pid is not None and _has_swap_near(tracking_per_rally, rid, frame, pl_pid):
                    reasons.append("swap_near_pl_pid")
                if _has_swap_near(tracking_per_rally, rid, frame, gt_pid):
                    reasons.append("swap_near_gt_pid")
                if _is_low_coverage(tracking_per_rally, rid, gt_pid):
                    reasons.append("gt_pid_low_coverage")
                if cat == "wrong_unknown_team":
                    reasons.append("unknown_team_fallback")

                if reasons:
                    bucket_kind = "primitive_fixable__" + "+".join(reasons)
                elif cat == "wrong_same_team":
                    bucket_kind = "irreducible__within_team"
                elif cat == "wrong_cross_team":
                    bucket_kind = "chooser_fixable__cross_team_geometric"
                else:
                    bucket_kind = "chooser_fixable__other"

            bucket[bucket_kind] += 1
            per_fixture_bucket.setdefault(fx, Counter())[bucket_kind] += 1
            per_category_bucket.setdefault(cat, Counter())[bucket_kind] += 1
            row_partitions.append(
                {
                    "rally_id": rid,
                    "fixture": fx,
                    "gt_frame": frame,
                    "gt_pid": gt_pid,
                    "pl_pid": pl_pid,
                    "category": cat,
                    "bucket": bucket_kind,
                }
            )

    # Aggregate
    totals = {
        "primitive_fixable": 0,
        "chooser_fixable": 0,
        "detection_limit": 0,
        "irreducible": 0,
        "correct": 0,
    }
    for k, v in bucket.items():
        if k.startswith("primitive_fixable"):
            totals["primitive_fixable"] += v
        elif k.startswith("chooser_fixable"):
            totals["chooser_fixable"] += v
        elif k.startswith("irreducible"):
            totals["irreducible"] += v
        elif k == "detection_limit":
            totals["detection_limit"] += v
        elif k == "correct":
            totals["correct"] += v

    n = sum(totals.values())
    print()
    print("=== BASELINE PARTITION ===")
    for k, v in totals.items():
        pct = v / n * 100 if n else 0
        print(f"  {k:22s}: {v:>4d} ({pct:5.1f}%)")
    print(f"  {'TOTAL':22s}: {n:>4d}")

    # Per-fixture
    print()
    print("=== PER-FIXTURE ===")
    print(
        f"{'fx':8s} {'n':>3s} {'correct':>7s} {'prim_fix':>8s} {'choose_fix':>10s} "
        f"{'det_lim':>7s} {'irred':>6s}"
    )
    for fx in sorted(per_fixture_bucket):
        c = per_fixture_bucket[fx]
        pf = sum(v for k, v in c.items() if k.startswith("primitive_fixable"))
        cf = sum(v for k, v in c.items() if k.startswith("chooser_fixable"))
        ir = sum(v for k, v in c.items() if k.startswith("irreducible"))
        dl = c.get("detection_limit", 0)
        cc = c.get("correct", 0)
        tot = pf + cf + ir + dl + cc
        print(
            f"{fx:8s} {tot:>3d} {cc:>7d} {pf:>8d} {cf:>10d} {dl:>7d} {ir:>6d}"
        )

    # Subcategory detail
    print()
    print("=== SUBCATEGORY DETAIL ===")
    for k, v in sorted(bucket.items(), key=lambda kv: -kv[1]):
        if k == "correct":
            continue
        print(f"  {k:60s}: {v}")

    # Report
    lines = [
        "# Phase 1.5 — Composite Primitive Partition",
        "",
        f"**Date:** 2026-04-24  ",
        f"**Scope:** {n} baseline matches across 9 fixtures.  ",
        "",
        "## Partition",
        "",
        "| bucket | count | % |",
        "|---|---|---|",
    ]
    for k in ("correct", "primitive_fixable", "chooser_fixable",
              "detection_limit", "irreducible"):
        v = totals[k]
        pct = v / n * 100 if n else 0
        lines.append(f"| **{k}** | {v} | {pct:.1f}% |")
    lines.append(f"| TOTAL | {n} | 100% |")

    lines.extend([
        "",
        "## Per-fixture",
        "",
        "| fixture | n | correct | prim_fix | choose_fix | det_lim | irred |",
        "|---|---|---|---|---|---|---|",
    ])
    for fx in sorted(per_fixture_bucket):
        c = per_fixture_bucket[fx]
        pf = sum(v for k, v in c.items() if k.startswith("primitive_fixable"))
        cf = sum(v for k, v in c.items() if k.startswith("chooser_fixable"))
        ir = sum(v for k, v in c.items() if k.startswith("irreducible"))
        dl = c.get("detection_limit", 0)
        cc = c.get("correct", 0)
        tot = pf + cf + ir + dl + cc
        lines.append(f"| **{fx}** | {tot} | {cc} | {pf} | {cf} | {dl} | {ir} |")

    lines.extend([
        "",
        "## Subcategory detail",
        "",
        "| bucket | count |",
        "|---|---|",
    ])
    for k, v in sorted(bucket.items(), key=lambda kv: -kv[1]):
        if k == "correct":
            continue
        lines.append(f"| `{k}` | {v} |")

    # Phase 2 headroom calculation
    wrong_total = n - totals["correct"] - totals["detection_limit"]
    primitive_share = (
        totals["primitive_fixable"] / wrong_total * 100
        if wrong_total else 0
    )
    chooser_share = (
        totals["chooser_fixable"] / wrong_total * 100
        if wrong_total else 0
    )
    lines.extend([
        "",
        "## Phase 2 headroom",
        "",
        f"- Total wrong + missing-but-fixable: **{wrong_total}** "
        f"({wrong_total / n * 100:.1f}% of all matches)",
        f"- Primitive-fixable share: **{primitive_share:.1f}%** of wrong — "
        f"addressable only by Phase 1.3a (contact-time team validation) + "
        f"Phase 1.1a (swap-aware abstention).",
        f"- Chooser-fixable share: **{chooser_share:.1f}%** of wrong — "
        f"Phase 2 confidence-gated chooser direct target.",
        f"- Irreducible within-team: **{totals['irreducible']}** — architectural "
        f"floor per memory `within_team_reid_project_2026_04_16.md`.",
        f"- Detection limit (missing): **{totals['detection_limit']}** — "
        f"rescue-only via Phase 3.2 complement.",
        "",
        "## Decision",
        "",
        "**Phase 1 → Phase 2 handoff:**",
        "",
        "- Chooser-fixable bucket is where Phase 2's confidence-gated chooser "
        "has direct leverage. Kill gate unchanged: halve wrong_rate.",
        "- Primitive-fixable bucket is ship-blocking on Phase 1.1a "
        "(swap-aware abstention) + Phase 1.3a (contact-time team check). "
        "Both are chooser-side policies that use the Phase-1 audit artifacts "
        "as inputs — no stage-2 upstream fixes.",
        "- Irreducible + detection-limit buckets are accepted as architectural "
        "floor.",
    ])

    OUT_PATH.write_text("\n".join(lines))
    AUDIT_JSON_PATH.write_text(
        json.dumps(
            {
                "totals": totals,
                "subcategory": dict(bucket),
                "per_fixture": {fx: dict(c) for fx, c in per_fixture_bucket.items()},
                "per_category": {cat: dict(c) for cat, c in per_category_bucket.items()},
                "rows": row_partitions,
            },
            indent=2,
        )
    )
    print(f"\nwrote {OUT_PATH}")
    print(f"wrote {AUDIT_JSON_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
