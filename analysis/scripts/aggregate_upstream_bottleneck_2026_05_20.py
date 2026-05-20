#!/usr/bin/env python3
"""Aggregator for upstream-pipeline bottleneck probes.

Reads L1..L6 JSONs from reports/upstream_bottleneck_2026_05_20/, computes:
  - Per-layer table (oracle, realistic, gap, cost, confidence)
  - Ranking via formula: realistic * (1 - gap_ratio) / cost
  - L5 learning-curve slope reported separately

Output: summary.md
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ANALYSIS_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = ANALYSIS_DIR / "reports" / "upstream_bottleneck_2026_05_20"

# Cost estimates per the spec
COST = {
    "L1": 3,   # widening tolerance is a config tweak; retraining tracker is ~10
    "L2": 3,   # relaxing candidate-gen rule is a config tweak
    "L3": 10,  # retraining frame regressor is a small ML training cycle
    "L4": 10,  # WASB retraining is a larger cycle (but out of scope here)
    "L5": 30,  # 5-10k more labels at human-labeling rates
    "L6": 3,   # chain confidence-threshold tweak is config; chain rewrite is 10
}


def load_layer(label: str) -> dict[str, Any]:
    p = REPORT_DIR / f"{label}.json"
    if not p.exists():
        return {}
    result: dict[str, Any] = json.loads(p.read_text())
    return result


def compute_realistic_l1(d: dict[str, Any]) -> int:
    r = d.get("realistic_recoveries") or {}
    return max(
        int(r.get("widen_pm10", 0)),
        int(r.get("widen_pm15", 0)),
        int(r.get("interpolate_short_gap", 0)),
    )


def main() -> int:
    layers = {}
    for k in ("L1", "L2", "L3", "L4", "L5", "L6"):
        layers[k] = load_layer(k)

    n_total = max(
        int(layers["L1"].get("n_total_wrong", 0)),
        int(layers["L2"].get("n_total_wrong", 0)),
        int(layers["L3"].get("n_total_wrong", 0)),
        int(layers["L6"].get("n_total_wrong", 0)),
        1,
    )

    table: list[dict[str, Any]] = []

    # L1
    o = int(layers["L1"].get("oracle_recoveries", 0))
    r = compute_realistic_l1(layers["L1"])
    gap_ratio = (o - r) / max(o, 1)
    table.append({
        "layer": "L1 player-tracker coverage",
        "oracle": o, "realistic": r, "gap_ratio": gap_ratio,
        "cost": COST["L1"],
        "rank_score": r * (1 - gap_ratio) / COST["L1"],
    })

    # L2 — oracle IS realistic (forcing GT into candidates is the intervention)
    o = int(layers["L2"].get("oracle_recoveries", 0))
    table.append({
        "layer": "L2 candidate generation",
        "oracle": o, "realistic": o, "gap_ratio": 0.0,
        "cost": COST["L2"],
        "rank_score": o / COST["L2"],
    })

    # L3 — realistic = retraining regressor; we report oracle as proxy
    o = int(layers["L3"].get("oracle_recoveries", 0))
    table.append({
        "layer": "L3 contact-frame regression",
        "oracle": o, "realistic": o, "gap_ratio": 0.0,
        "cost": COST["L3"],
        "rank_score": o / COST["L3"],
    })

    # L4 — realistic = WASB retrain; oracle as proxy on partial corpus
    o = int(layers["L4"].get("oracle_recoveries", 0))
    n_overlap = int(layers["L4"].get("n_overlap", 0))
    table.append({
        "layer": f"L4 ball-tracking (overlap n={n_overlap})",
        "oracle": o, "realistic": o, "gap_ratio": 0.0,
        "cost": COST["L4"],
        "rank_score": o / COST["L4"] if n_overlap >= 30 else 0,
    })

    # L5: special handling — learning curve, no count
    l5 = layers["L5"]
    l5_summary = "no data"
    if l5:
        slopes = []
        for action, vals in l5.items():
            if not isinstance(vals, dict) or "frac_1.0" not in vals:
                continue
            v100_raw = vals.get("frac_1.0", 0)
            v75_raw = vals.get("frac_0.75", v100_raw)
            try:
                v100 = float(v100_raw) if v100_raw is not None else 0.0
                v75 = float(v75_raw) if v75_raw is not None else v100
            except (TypeError, ValueError):
                continue
            slope = v100 - v75
            slopes.append((action, v75, v100, slope))
        if slopes:
            still_sloping = [s for s in slopes if s[3] > 0.005]
            l5_summary = (f"{len(still_sloping)}/{len(slopes)} actions still "
                          f"sloping at frac=1.00 (slope > 0.005)")

    # L6
    o = int(layers["L6"].get("oracle_recoveries_at_chain_disagreements", 0))
    table.append({
        "layer": "L6 team-chain accuracy",
        "oracle": o, "realistic": o, "gap_ratio": 0.0,
        "cost": COST["L6"],
        "rank_score": o / COST["L6"],
    })

    table.sort(key=lambda x: -float(x["rank_score"]))

    md = ["# Upstream Bottleneck Probe — Summary (2026-05-20)", ""]
    md.append(f"Substrate: {n_total} wrong-attribution contacts on trusted-32.")
    md.append("")
    md.append("## Per-layer ranking")
    md.append("")
    md.append("| Rank | Layer | Oracle | Realistic | Gap | Cost | Score |")
    md.append("|---|---|---:|---:|---:|---:|---:|")
    for i, row in enumerate(table, start=1):
        md.append(
            f"| {i} | {row['layer']} | {row['oracle']} | "
            f"{row['realistic']} | {row['gap_ratio']:.2f} | "
            f"{row['cost']} | {row['rank_score']:.2f} |"
        )
    md.append("")
    md.append(f"## L5 (GT-scale learning curve)\n\n{l5_summary}\n")
    md.append("")

    if l5:
        md.append("Per-action curves:")
        md.append("")
        md.append("| Action | n_contacts | frac_0.25 | frac_0.5 | frac_0.75 | frac_1.0 | Δ(1.0-0.75) |")
        md.append("|---|---:|---:|---:|---:|---:|---:|")
        for action in ("SERVE", "RECEIVE", "SET", "ATTACK", "DIG", "BLOCK"):
            vals = l5.get(action, {})
            if not isinstance(vals, dict):
                continue
            n = vals.get("n_contacts", "—")
            f25 = vals.get("frac_0.25")
            f50 = vals.get("frac_0.5")
            f75 = vals.get("frac_0.75")
            f100 = vals.get("frac_1.0")

            def fmt(v: Any) -> str:
                if v is None:
                    return "—"
                try:
                    fv = float(v)
                    if fv != fv:  # NaN
                        return "NaN"
                    return f"{fv:.3f}"
                except (TypeError, ValueError):
                    return "—"

            delta = "—"
            try:
                if f100 is not None and f75 is not None:
                    fv100 = float(f100)
                    fv75 = float(f75)
                    if fv100 == fv100 and fv75 == fv75:
                        delta = f"{fv100 - fv75:+.3f}"
            except (TypeError, ValueError):
                pass
            md.append(
                f"| {action} | {n} | {fmt(f25)} | {fmt(f50)} | "
                f"{fmt(f75)} | {fmt(f100)} | {delta} |"
            )
        md.append("")

    md.append("## Decision rule application")
    md.append("")
    if table:
        top = table[0]
        md.append(f"**Top recommendation:** invest in **{top['layer']}** "
                  f"(realistic ceiling {top['realistic']}, cost {top['cost']}, "
                  f"rank score {top['rank_score']:.2f}).")
    md.append("")
    md.append("Caveats:")
    md.append("- Gap ratio > 0.5 on any layer = projection-trap candidate "
              "(audit ceiling >> realistic; treat with skepticism).")
    md.append("- L4 reported on partial corpus when overlap < 30 rallies "
              "(rank_score forced to 0 if so).")
    md.append("- All ranks are confounded if a contact fails at multiple "
              "layers; aggregator does not materialize a per-contact "
              "multi-layer-fail Venn (would be a follow-up).")
    md.append("- L1's realistic ceiling uses the BEST of "
              "widen_pm10/widen_pm15/interpolate; L2-L6 use oracle as "
              "realistic-proxy (intervention cost reflected in COST table).")
    md.append("")

    summary_path = REPORT_DIR / "summary.md"
    summary_path.write_text("\n".join(md))
    print(f"Wrote {summary_path}")
    print()
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
