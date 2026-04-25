"""Compare two production_eval runs against the ref-crop canonical-identity gate.

Pre-registered gates (per docs/superpowers/plans/2026-04-25-ref-crop-canonical-identity.md):
- court_side_accuracy, score_accuracy: within -1pp of baseline
- player_attribution_accuracy, serve_attr_accuracy: within -2pp of baseline (primary)

Usage:
    cd analysis
    uv run python scripts/compare_canonical_pid_ab.py \\
        outputs/production_eval/baseline_before_ref_crop_canonical_<stamp>.json \\
        outputs/production_eval/run_<new>.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

GATES = {
    "player_attribution_accuracy": ("primary", -0.02),
    "serve_attr_accuracy":         ("primary", -0.02),
    "court_side_accuracy":         ("secondary", -0.01),
    "score_accuracy":              ("secondary", -0.01),
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("baseline", type=Path)
    parser.add_argument("after", type=Path)
    args = parser.parse_args()

    base = json.loads(args.baseline.read_text())
    new = json.loads(args.after.read_text())

    print(f"baseline: {args.baseline.name}  git_sha={base.get('git_sha', '?')}")
    print(f"after:    {args.after.name}  git_sha={new.get('git_sha', '?')}")
    print(f"rallies:  {base['n_rallies_evaluated']} → {new['n_rallies_evaluated']}")
    print()

    print(f"{'metric':<35} {'baseline':>10} {'after':>10} {'delta':>10} {'gate':<10} {'verdict':<8}")
    print("-" * 90)

    all_pass = True
    for metric, (tier, gate_delta) in GATES.items():
        b = float(base["metrics"][metric]["mean"])
        a = float(new["metrics"][metric]["mean"])
        d = a - b
        passed = d >= gate_delta
        all_pass = all_pass and passed
        verdict = "PASS" if passed else "FAIL"
        print(f"{metric:<35} {b:>9.2%}  {a:>9.2%}  {d:>+9.2%}  {tier:<10} {verdict}")

    # Diagnostic-only metrics (no gate)
    print()
    print("Diagnostic (no gate):")
    for metric in ("contact_f1", "action_accuracy",
                   "player_attribution_oracle", "serve_attr_oracle"):
        if metric not in base.get("metrics", {}):
            continue
        b = float(base["metrics"][metric]["mean"])
        a = float(new["metrics"][metric]["mean"])
        d = a - b
        print(f"  {metric:<33} {b:>9.2%}  {a:>9.2%}  {d:>+9.2%}")

    print()
    print(f"OVERALL: {'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
