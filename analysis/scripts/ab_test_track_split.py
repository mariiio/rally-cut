#!/usr/bin/env python3
"""A/B test pre-Hungarian within-track segmentation against the v3 baseline.

Runs `lock_baseline_v2.py` twice — once with ENABLE_REF_CROP_TRACK_SPLIT=0
(reproduces baseline_v3) and once with ENABLE_REF_CROP_TRACK_SPLIT=1.
Computes deltas and evaluates pre-registered ship gates.

Usage:
    cd analysis
    uv run python scripts/ab_test_track_split.py
    uv run python scripts/ab_test_track_split.py --fixtures cuco wawa
    uv run python scripts/ab_test_track_split.py --reuse-off
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

_ANALYSIS_DIR = Path(__file__).resolve().parents[1]

ATTR_REPORTS = _ANALYSIS_DIR / "reports" / "attribution_rebuild"
OUT_JSON = ATTR_REPORTS / "track_split_ab_2026_04_26.json"
OFF_JSON = ATTR_REPORTS / "track_split_ab_off.json"
ON_JSON = ATTR_REPORTS / "track_split_ab_on.json"

GATE_PER_FIXTURE_CORRECT_DELTA = -0.005   # ≥ -0.5pp
GATE_AGG_CORRECT_DELTA = +0.005           # ≥ +0.5pp
GATE_AGG_WRONG_DELTA = 0.0                # ≤ +0.0pp


def _run_arm(out_json: Path, log_path: Path, flag_value: str,
             fixtures: list[str] | None) -> None:
    env = os.environ.copy()
    env["ENABLE_REF_CROP_TRACK_SPLIT"] = flag_value
    cmd = ["uv", "run", "python", "scripts/lock_baseline_v2.py", "--out", str(out_json)]
    if fixtures:
        cmd += ["--fixtures", *fixtures]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  [arm flag={flag_value}] running {' '.join(shlex.quote(c) for c in cmd)}")
    with log_path.open("w") as fh:
        fh.write(f"$ ENABLE_REF_CROP_TRACK_SPLIT={flag_value} {' '.join(shlex.quote(c) for c in cmd)}\n\n")
        proc = subprocess.run(cmd, cwd=_ANALYSIS_DIR, capture_output=True, text=True, env=env)
        fh.write(proc.stdout)
        if proc.stderr:
            fh.write("\n--- stderr ---\n" + proc.stderr)
    if proc.returncode != 0:
        raise RuntimeError(f"lock_baseline_v2.py (flag={flag_value}) exited {proc.returncode}")


def _evaluate_gates(off: dict[str, Any], on: dict[str, Any]) -> dict[str, Any]:
    off_agg = off["aggregate"]["rates"]
    on_agg = on["aggregate"]["rates"]
    agg_correct_delta = on_agg["correct_rate"] - off_agg["correct_rate"]
    agg_wrong_delta = on_agg["wrong_rate"] - off_agg["wrong_rate"]
    agg_missing_delta = on_agg["missing_rate"] - off_agg["missing_rate"]
    per_fixture: dict[str, dict[str, float]] = {}
    failures: list[tuple[str, float]] = []
    for fx in off["fixture_rates"]:
        if fx not in on["fixture_rates"]:
            continue
        d_correct = on["fixture_rates"][fx]["correct_rate"] - off["fixture_rates"][fx]["correct_rate"]
        d_wrong = on["fixture_rates"][fx]["wrong_rate"] - off["fixture_rates"][fx]["wrong_rate"]
        d_missing = on["fixture_rates"][fx]["missing_rate"] - off["fixture_rates"][fx]["missing_rate"]
        per_fixture[fx] = {
            "correct_off": off["fixture_rates"][fx]["correct_rate"],
            "correct_on": on["fixture_rates"][fx]["correct_rate"],
            "correct_delta": d_correct,
            "wrong_delta": d_wrong,
            "missing_delta": d_missing,
        }
        if d_correct < GATE_PER_FIXTURE_CORRECT_DELTA:
            failures.append((fx, d_correct))
    return {
        "aggregate": {
            "correct_off": off_agg["correct_rate"],
            "correct_on": on_agg["correct_rate"],
            "correct_delta": agg_correct_delta,
            "wrong_delta": agg_wrong_delta,
            "missing_delta": agg_missing_delta,
        },
        "per_fixture": per_fixture,
        "per_fixture_failures": failures,
        "gates": {
            "per_fixture_correct_delta_min": GATE_PER_FIXTURE_CORRECT_DELTA,
            "aggregate_correct_delta_min": GATE_AGG_CORRECT_DELTA,
            "aggregate_wrong_delta_max": GATE_AGG_WRONG_DELTA,
            "all_pass": (
                not failures
                and agg_correct_delta >= GATE_AGG_CORRECT_DELTA
                and agg_wrong_delta <= GATE_AGG_WRONG_DELTA
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixtures", nargs="*", default=None)
    parser.add_argument("--out", type=Path, default=OUT_JSON)
    parser.add_argument("--off-out", type=Path, default=OFF_JSON)
    parser.add_argument("--on-out", type=Path, default=ON_JSON)
    parser.add_argument("--reuse-off", action="store_true",
                        help="Reuse existing off-arm JSON if present.")
    args = parser.parse_args()

    log_dir = ATTR_REPORTS / "track_split_ab_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if args.reuse_off and args.off_out.exists():
        print(f"[1/2] off-arm: reusing {args.off_out}")
    else:
        print(f"[1/2] off-arm (flag=0)…")
        _run_arm(args.off_out, log_dir / "off.log", "0", args.fixtures)

    print(f"[2/2] on-arm (flag=1)…")
    _run_arm(args.on_out, log_dir / "on.log", "1", args.fixtures)

    off = json.loads(args.off_out.read_text())
    on = json.loads(args.on_out.read_text())
    eval_ = _evaluate_gates(off, on)

    out = {
        "off_arm_json": str(args.off_out),
        "on_arm_json": str(args.on_out),
        "eval": eval_,
        "ran_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))

    print(f"\nWrote {args.out}")
    agg = eval_["aggregate"]
    print(f"\nAggregate:")
    print(f"  correct: off={agg['correct_off']:.4f}  on={agg['correct_on']:.4f}  "
          f"Δ={agg['correct_delta']:+.4f}")
    print(f"  Δwrong  = {agg['wrong_delta']:+.4f}")
    print(f"  Δmissing= {agg['missing_delta']:+.4f}")
    print("\nPer-fixture:")
    for fx, d in eval_["per_fixture"].items():
        marker = " ⚠" if d["correct_delta"] < GATE_PER_FIXTURE_CORRECT_DELTA else ""
        print(f"  {fx:<6s}: off={d['correct_off']:.4f}  on={d['correct_on']:.4f}  "
              f"Δ={d['correct_delta']:+.4f}{marker}")

    print(f"\nGates:")
    print(f"  per-fixture failures: {eval_['per_fixture_failures'] or 'none'}")
    print(f"  aggregate Δcorrect ≥ {GATE_AGG_CORRECT_DELTA:+.4f}? "
          f"{agg['correct_delta'] >= GATE_AGG_CORRECT_DELTA}")
    print(f"  aggregate Δwrong   ≤ {GATE_AGG_WRONG_DELTA:+.4f}? "
          f"{agg['wrong_delta'] <= GATE_AGG_WRONG_DELTA}")
    print(f"\n{'PASS — ship' if eval_['gates']['all_pass'] else 'FAIL — keep flag default 0'}")


if __name__ == "__main__":
    main()
