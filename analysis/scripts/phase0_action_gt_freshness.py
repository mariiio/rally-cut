"""Phase 0.4 — action-type GT freshness audit.

For each matched (gt, pipeline) action pair from the baseline JSON, compare
action-type (serve/receive/set/attack/dig/block). If pipeline action-type
disagrees with GT action-type, surface it. Use this to decide whether Phase 3
Pattern A preconditions can lean on pipeline action types.

Scope: 9 fixtures, baseline JSON. The other 59 corpus videos are deferred to
Phase 4 pre-ship.

Usage:
    uv run python scripts/phase0_action_gt_freshness.py
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
OUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "action_gt_freshness_2026_04_24.md"
)


def main() -> int:
    data = json.loads(BASELINE_PATH.read_text())

    per_fixture: dict[str, dict] = {}
    transitions: Counter[tuple[str, str]] = Counter()
    stale_rallies: dict[str, list[dict]] = {}

    for rally in data["rallies"]:
        fx = rally["fixture"]
        fx_stat = per_fixture.setdefault(
            fx,
            {"matched": 0, "action_type_match": 0, "action_type_mismatch": 0},
        )
        rally_mismatches: list[dict] = []
        for m in rally.get("matches", []):
            if m.get("pl_action") is None:
                # missing — out of scope for action-type freshness
                continue
            fx_stat["matched"] += 1
            gt_a = m.get("gt_action")
            pl_a = m.get("pl_action")
            if gt_a == pl_a:
                fx_stat["action_type_match"] += 1
            else:
                fx_stat["action_type_mismatch"] += 1
                transitions[(gt_a, pl_a)] += 1
                rally_mismatches.append(
                    {"frame": m["gt_frame"], "gt": gt_a, "pl": pl_a}
                )
        if rally_mismatches:
            stale_rallies.setdefault(fx, []).append(
                {
                    "rally_id": rally["rally_id"],
                    "start_ms": rally.get("start_ms"),
                    "mismatches": rally_mismatches,
                }
            )

    combined = {"matched": 0, "action_type_match": 0, "action_type_mismatch": 0}
    for v in per_fixture.values():
        for k in combined:
            combined[k] += v[k]

    # Summary table
    lines = [
        "# Phase 0.4 — Action-Type GT Freshness Audit",
        "",
        f"**Generated:** 2026-04-24  ",
        f"**Source:** `{BASELINE_PATH.relative_to(BASELINE_PATH.parent.parent.parent)}`  ",
        f"**Scope:** 9 Phase-0 fixtures (matched GT↔pipeline action pairs only; "
        f"missing actions excluded since they have no pipeline action-type to compare).",
        "",
        "## Per-fixture disagreement rate",
        "",
        "| fixture | matched | agree | disagree | rate |",
        "|---|---|---|---|---|",
    ]
    for fx, s in sorted(per_fixture.items()):
        n = s["matched"]
        rate = s["action_type_mismatch"] / n if n else 0.0
        lines.append(
            f"| {fx} | {n} | {s['action_type_match']} | "
            f"{s['action_type_mismatch']} | **{rate:.1%}** |"
        )
    n = combined["matched"]
    rate = combined["action_type_mismatch"] / n if n else 0.0
    lines.append(
        f"| **COMBINED** | **{n}** | **{combined['action_type_match']}** | "
        f"**{combined['action_type_mismatch']}** | **{rate:.1%}** |"
    )

    # Transition matrix
    lines.extend([
        "",
        "## Disagreement pairs (gt → pipeline)",
        "",
        "| GT action | Pipeline action | count |",
        "|---|---|---|",
    ])
    for (gt_a, pl_a), c in transitions.most_common():
        lines.append(f"| {gt_a} | {pl_a} | {c} |")

    # Decision block
    lines.extend([
        "",
        "## Decision (Phase 3 Pattern A precondition)",
        "",
    ])
    if rate <= 0.05:
        lines.append(
            "**Under 5% — pipeline action types are trustworthy as Pattern A "
            "preconditions.** Stored `action_ground_truth.json` does NOT need a "
            "refresh pass before Phase 3 chain-rescue landing."
        )
    elif rate <= 0.15:
        lines.append(
            f"**{rate:.1%} disagreement — borderline.** Use pipeline action types "
            "only when they carry high confidence (≥ conf threshold tbd). Below "
            "threshold, abstain rather than infer."
        )
    else:
        lines.append(
            f"**{rate:.1%} disagreement — too high for Pattern A to lean on "
            "pipeline action types naively.** Either refresh action-type GT on "
            "the 9 fixtures, or scope Pattern A to action types the MS-TCN++/"
            "classifier emits at ≥0.9 confidence only."
        )

    # Per-rally detail
    if stale_rallies:
        lines.extend([
            "",
            "## Rallies with action-type disagreements",
            "",
        ])
        for fx in sorted(stale_rallies):
            lines.append(f"### {fx}")
            for rec in stale_rallies[fx]:
                lines.append(
                    f"- `{rec['rally_id'][:8]}` @ {rec['start_ms']}ms — "
                    f"{len(rec['mismatches'])} mismatch(es): "
                    + ", ".join(
                        f"f{m['frame']} {m['gt']}→{m['pl']}"
                        for m in rec["mismatches"][:5]
                    )
                    + ("…" if len(rec["mismatches"]) > 5 else "")
                )
            lines.append("")

    OUT_PATH.write_text("\n".join(lines))
    print(f"wrote {OUT_PATH}")
    print(f"  combined disagreement: {combined['action_type_mismatch']}/{n} "
          f"({rate:.1%})")
    for fx, s in sorted(per_fixture.items()):
        n_fx = s["matched"]
        r = s["action_type_mismatch"] / n_fx if n_fx else 0.0
        print(f"  {fx:6s}: {s['action_type_mismatch']:>3d}/{n_fx:>3d} ({r:.1%})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
