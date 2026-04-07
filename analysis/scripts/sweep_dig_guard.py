"""Sweep dig_guard_ratio for apply_sequence_override.

Mirrors production_eval._run_once for one rerun, with a monkey-patched
DIG_GUARD_RATIO. Reports headline metrics + dig F1 + per-class F1 for each τ.

Constraint: dig F1 ≥ baseline (run_2026-04-07-160920.json) + 0.5pp.
Optimize:    action_accuracy.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPTS_DIR))

from eval_action_detection import (  # noqa: E402
    _load_match_team_assignments,
    compute_metrics,
    load_rallies_with_action_gt,
)
from production_eval import (  # noqa: E402
    PipelineContext,
    _build_calibrators,
    _parse_positions,
    _run_once,
    _serve_metrics,
)

import rallycut.tracking.sequence_action_runtime as sar  # noqa: E402

CLASSES = ["serve", "receive", "set", "attack", "block", "dig"]
TAUS = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0]  # 1.0 = current behavior baseline


def evaluate(rallies, team_map, calibrators, tau: float) -> dict[str, float]:
    sar.DIG_GUARD_RATIO = tau
    matches, unmatched, _ = _run_once(
        rallies, team_map, calibrators, PipelineContext(), print_progress=False,
    )
    m = compute_metrics(matches, unmatched)
    serve_id, _, serve_attr, _ = _serve_metrics(matches)
    out: dict[str, float] = {
        "tau": tau,
        "contact_f1": float(m["f1"]),
        "action_accuracy": float(m["action_accuracy"]),
        "court_side": float(m["court_side_accuracy"]),
        "player_attribution": float(m["player_evaluable_accuracy"]),
        "serve_id": float(serve_id),
        "serve_attr": float(serve_attr),
    }
    for cls, stats in m["per_class"].items():
        out[f"f1::{cls}"] = float(stats["f1"])
    return out


def main() -> int:
    print("Loading rallies...")
    rallies = load_rallies_with_action_gt()
    rally_pos_lookup = {
        r.rally_id: _parse_positions(r.positions_json)
        for r in rallies if r.positions_json
    }
    video_ids = {r.video_id for r in rallies}
    team_map = _load_match_team_assignments(video_ids, rally_positions=rally_pos_lookup)
    calibrators = _build_calibrators(video_ids)
    print(f"  {len(rallies)} rallies, {len(team_map)} team maps")

    rows = []
    for tau in TAUS:
        print(f"\n=== τ = {tau} ===")
        r = evaluate(rallies, team_map, calibrators, tau)
        rows.append(r)
        print(
            f"  action_acc={r['action_accuracy']*100:.2f}% "
            f"contact_f1={r['contact_f1']*100:.2f}% "
            f"dig_f1={r['f1::dig']*100:.2f}% "
            f"set_f1={r['f1::set']*100:.2f}% "
            f"block_f1={r['f1::block']*100:.2f}%"
        )

    # Summary table
    baseline = rows[0]  # τ=1.0 — same as current production
    dig_target = baseline["f1::dig"] + 0.005

    print("\n\n=== sweep summary ===")
    print(f"baseline (τ=1.0) dig_f1 = {baseline['f1::dig']*100:.2f}%, target ≥ {dig_target*100:.2f}%")
    print(f"{'τ':>5}  {'action_acc':>10}  {'dig_f1':>8}  {'set_f1':>8}  {'block_f1':>9}  {'contact_f1':>10}  {'meets_dig':>9}")
    for r in rows:
        meets = "✓" if r["f1::dig"] >= dig_target else "✗"
        print(
            f"{r['tau']:>5.1f}  "
            f"{r['action_accuracy']*100:>9.2f}%  "
            f"{r['f1::dig']*100:>7.2f}%  "
            f"{r['f1::set']*100:>7.2f}%  "
            f"{r['f1::block']*100:>8.2f}%  "
            f"{r['contact_f1']*100:>9.2f}%  "
            f"{meets:>9}"
        )

    # Pick winner: highest action_accuracy among rows meeting dig constraint
    qualifying = [r for r in rows if r["f1::dig"] >= dig_target]
    if not qualifying:
        print("\n[FAIL] no τ meets dig F1 ≥ baseline + 0.5pp constraint")
        return 1
    winner = max(qualifying, key=lambda r: r["action_accuracy"])
    print(
        f"\n[WINNER] τ = {winner['tau']}: "
        f"action_acc {winner['action_accuracy']*100:.2f}% "
        f"(baseline {baseline['action_accuracy']*100:.2f}%, "
        f"Δ {(winner['action_accuracy']-baseline['action_accuracy'])*100:+.2f}pp), "
        f"dig F1 {winner['f1::dig']*100:.2f}% "
        f"(Δ {(winner['f1::dig']-baseline['f1::dig'])*100:+.2f}pp)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
