"""Standard attribution benchmark runner — Phase 0.2.

Reads a run artifact (same schema as reports/attribution_rebuild/baseline_2026_04_24.json)
and reports per-fixture + aggregate metrics. When --experiment is given, computes a
transition matrix vs --baseline.

This is the only runner downstream phases should use. Do not add bespoke measurement
code in each experiment — produce an artifact in the baseline schema instead, then
call this.

Usage:
    # Just print baseline metrics
    uv run python scripts/bench_attribution.py

    # A/B experiment vs locked baseline
    uv run python scripts/bench_attribution.py --experiment reports/attribution_rebuild/phase2_confidence_gated.json

    # Save the summary for programmatic use
    uv run python scripts/bench_attribution.py --experiment ... --out reports/attribution_rebuild/bench_phase2.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from rallycut.evaluation.attribution_bench import (
    CATEGORIES,
    WRONG_CATEGORIES,
    aggregate,
    score_rally,
    transition_matrix,
)

DEFAULT_BASELINE = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "attribution_rebuild"
    / "baseline_2026_04_24.json"
)


def _load(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    # Ensure matches/rally_totals are populated (they are in the locked baseline,
    # but experiments might emit raw rally records without pre-scoring).
    for rally in data["rallies"]:
        if "matches" not in rally or "rally_totals" not in rally:
            scored = score_rally(rally)
            rally["matches"] = scored["matches"]
            rally["rally_totals"] = scored["rally_totals"]
    return data


def _print_per_fixture(agg: dict[str, Any], label: str) -> None:
    print(f"\n=== {label} — per-fixture ===")
    print(f"{'fixture':8s} {'n_gt':>4s}  {'correct':>9s}  {'wrong':>9s}  "
          f"{'missing':>9s}  {'abstain':>9s}  [X/same/unk]")
    for fx, v in sorted(agg["per_fixture"].items()):
        c = v["counts"]
        r = v["rates"]
        wrong = sum(c[k] for k in WRONG_CATEGORIES)
        print(
            f"{fx:8s} {c['n_gt_actions']:>4d}  "
            f"{c['correct']:>4d} ({r['correct_rate']:.1%})  "
            f"{wrong:>4d} ({r['wrong_rate']:.1%})  "
            f"{c['missing']:>4d} ({r['missing_rate']:.1%})  "
            f"{c['abstained']:>4d} ({r['abstained_rate']:.1%})  "
            f"[{c['wrong_cross_team']}/{c['wrong_same_team']}/{c['wrong_unknown_team']}]"
        )


def _print_aggregate(agg: dict[str, Any], label: str) -> None:
    c = agg["combined"]["counts"]
    r = agg["combined"]["rates"]
    wrong = sum(c[k] for k in WRONG_CATEGORIES)
    print(f"\n=== {label} — aggregate (n={c['n_gt_actions']}) ===")
    print(f"correct:  {c['correct']:>4d} ({r['correct_rate']:.1%})")
    print(f"wrong:    {wrong:>4d} ({r['wrong_rate']:.1%})  "
          f"[cross={c['wrong_cross_team']} same={c['wrong_same_team']} unk={c['wrong_unknown_team']}]")
    print(f"missing:  {c['missing']:>4d} ({r['missing_rate']:.1%})")
    print(f"abstain:  {c['abstained']:>4d} ({r['abstained_rate']:.1%})")


def _print_transition_matrix(
    baseline_rallies: list[dict[str, Any]],
    experiment_rallies: list[dict[str, Any]],
) -> None:
    matrix = transition_matrix(baseline_rallies, experiment_rallies)
    print("\n=== Transition matrix (rows=baseline, cols=experiment) ===")
    print(f"{'':22s}" + "".join(f"{c:>12s}" for c in CATEGORIES))
    for c_from in CATEGORIES:
        row = matrix[c_from]
        total = sum(row.values())
        line = f"{c_from:22s}" + "".join(f"{row[c]:>12d}" for c in CATEGORIES)
        print(f"{line}   total={total}")


def _diff_agg(
    baseline: dict[str, Any], experiment: dict[str, Any]
) -> dict[str, Any]:
    out: dict[str, Any] = {"combined": {}, "per_fixture": {}}
    for metric in ("correct_rate", "wrong_rate", "missing_rate", "abstained_rate"):
        out["combined"][metric] = (
            experiment["combined"]["rates"][metric]
            - baseline["combined"]["rates"][metric]
        )
    for fx in sorted(set(baseline["per_fixture"]) | set(experiment["per_fixture"])):
        fx_diff = {}
        for metric in ("correct_rate", "wrong_rate", "missing_rate", "abstained_rate"):
            b = baseline["per_fixture"].get(fx, {"rates": {metric: 0.0}})["rates"][metric]
            e = experiment["per_fixture"].get(fx, {"rates": {metric: 0.0}})["rates"][metric]
            fx_diff[metric] = e - b
        out["per_fixture"][fx] = fx_diff
    return out


def _print_diff(diff: dict[str, Any]) -> None:
    def _fmt(v: float) -> str:
        sign = "+" if v >= 0 else ""
        return f"{sign}{v * 100:.1f}pp"

    print("\n=== Diff (experiment − baseline) ===")
    print(f"{'fixture':8s}  {'Δ correct':>10s}  {'Δ wrong':>10s}  {'Δ missing':>10s}  {'Δ abstain':>10s}")
    for fx, d in sorted(diff["per_fixture"].items()):
        print(
            f"{fx:8s}  {_fmt(d['correct_rate']):>10s}  "
            f"{_fmt(d['wrong_rate']):>10s}  "
            f"{_fmt(d['missing_rate']):>10s}  "
            f"{_fmt(d['abstained_rate']):>10s}"
        )
    d = diff["combined"]
    print(
        f"{'COMBINED':8s}  {_fmt(d['correct_rate']):>10s}  "
        f"{_fmt(d['wrong_rate']):>10s}  "
        f"{_fmt(d['missing_rate']):>10s}  "
        f"{_fmt(d['abstained_rate']):>10s}"
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--experiment", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None, help="Save JSON summary here")
    args = ap.parse_args()

    baseline = _load(args.baseline)
    b_agg = aggregate(baseline["rallies"])
    _print_per_fixture(b_agg, "BASELINE")
    _print_aggregate(b_agg, "BASELINE")

    out: dict[str, Any] = {
        "baseline_path": str(args.baseline),
        "baseline": b_agg,
    }

    if args.experiment:
        experiment = _load(args.experiment)
        e_agg = aggregate(experiment["rallies"])
        _print_per_fixture(e_agg, "EXPERIMENT")
        _print_aggregate(e_agg, "EXPERIMENT")
        _print_transition_matrix(baseline["rallies"], experiment["rallies"])
        diff = _diff_agg(b_agg, e_agg)
        _print_diff(diff)
        out["experiment_path"] = str(args.experiment)
        out["experiment"] = e_agg
        out["diff"] = diff
        out["transition_matrix"] = transition_matrix(
            baseline["rallies"], experiment["rallies"]
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
