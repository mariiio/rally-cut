"""Calibrate Joint Attribution PGM factor weights via coordinate ascent
on the 22-rally fresh-GT panel.

Loads the panel from DB; for each weight in fixed order, sweeps -50%,
-25%, +0%, +25%, +50% holding others fixed; picks the best step;
advances. Repeats up to 3 cycles or until no weight changes.

Outputs the calibrated FactorWeights as a Python literal that can be
pasted into joint_attribution_weights.py.

Run from analysis/:
    uv run python -u scripts/calibrate_joint_attribution_weights.py
    uv run python -u scripts/calibrate_joint_attribution_weights.py --smoke
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

from rallycut.evaluation.attribution_bench import score_rally
from rallycut.evaluation.db import get_connection
from rallycut.tracking.joint_attribution import (
    RallyContext,
    apply_pgm_result_to_actions,
    joint_attribute_rally,
)
from rallycut.tracking.joint_attribution_weights import (
    DEFAULT_WEIGHTS,
    FactorWeights,
)
from rallycut.training.action_gt_query import load_for_videos

PANEL_VIDEOS = {
    "cece": "950fbe5d-fdad-4862-b05d-8b374bdd5ec6",
    "gigi": "b097dd2a-6953-4e0e-a603-5be3552f462e",
    "wawa": "5c756c41-1cc1-4486-a95c-97398912cfbe",
}
SWEEP_FRACTIONS = (-0.5, -0.25, 0.0, 0.25, 0.5)
WEIGHT_ORDER = (
    "w_proximity", "w_dist", "w_dist_team", "w_visual", "w_pose",
    "w_prior", "w_action",
    "w_back_to_back", "w_alternation", "w_team_consistency", "w_absent_pair",
    "w_3_contact", "w_serve_first",
)
MAX_CYCLES = 3


def load_panel() -> list[dict]:
    """Load 22 panel rallies with all data needed to build RallyContext + score."""
    rallies = []
    with get_connection() as conn:
        for vname, vid in PANEL_VIDEOS.items():
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT r.id::text, r.start_ms, r.end_ms,
                           pt.actions_json, pt.contacts_json
                    FROM rallies r JOIN player_tracks pt ON pt.rally_id = r.id
                    WHERE r.video_id = %s AND EXISTS (
                        SELECT 1 FROM rally_action_ground_truth
                        WHERE rally_id = r.id
                    )
                    """,
                    (vid,),
                )
                rows = cur.fetchall()
            for rid, sm, em, aj, cj in rows:
                rallies.append({
                    "rally_id": rid, "video_id": vid, "fixture": vname,
                    "start_ms": sm, "end_ms": em,
                    "contacts": (cj if isinstance(cj, dict) else {}).get("contacts", []),
                    "actions": (aj if isinstance(aj, dict) else {}).get("actions", []),
                    "team_assignments_str_keys": (
                        aj if isinstance(aj, dict) else {}
                    ).get("teamAssignments", {}),
                    "serving_team": (aj if isinstance(aj, dict) else {}).get("servingTeam"),
                })
        # Load GT per video
        gt_by_rally = load_for_videos(conn, list(PANEL_VIDEOS.values()))
    for r in rallies:
        r["gt_actions"] = gt_by_rally.get(r["rally_id"], [])
    return rallies


def evaluate_weights(weights: FactorWeights, rallies: list[dict]) -> int:
    """Run PGM with these weights on all panel rallies; return total correct count."""
    total_correct = 0
    for r in rallies:
        team_assignments = {int(k): v for k, v in r["team_assignments_str_keys"].items()}
        rally_ctx = RallyContext(
            rally_id=r["rally_id"], contacts=r["contacts"],
            initial_actions=list(r["actions"]),
            team_assignments=team_assignments, serving_team=r["serving_team"],
        )
        result = joint_attribute_rally(rally_ctx, weights=weights)
        actions_copy = [dict(a) for a in r["actions"]]  # deep copy actions; don't mutate
        apply_pgm_result_to_actions(actions_copy, result, rally_ctx)
        rally_record = {
            "rally_id": r["rally_id"], "fixture": r["fixture"],
            "team_assignments": {str(k): v for k, v in team_assignments.items()},
            "serving_team": r["serving_team"],
            "gt_actions": r["gt_actions"],
            "pipeline_actions": actions_copy,
        }
        scored = score_rally(rally_record)
        total_correct += scored["rally_totals"]["correct"]
    return total_correct


def sanity_check(rallies: list[dict]) -> None:
    """Print one-line summary per rally to catch shape issues before the sweep."""
    print("Sanity check (first 3 rallies):", flush=True)
    for r in rallies[:3]:
        n_contacts = len(r["contacts"])
        n_actions = len(r["actions"])
        n_gt = len(r["gt_actions"])
        ta = r["team_assignments_str_keys"]
        st = r["serving_team"]
        print(
            f"  {r['fixture']}/{r['rally_id'][:8]}: "
            f"contacts={n_contacts} actions={n_actions} gt={n_gt} "
            f"team_assignments={ta} serving={st}",
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run abbreviated sweep (1 cycle, first 2 weights only) for end-to-end validation",
    )
    args = parser.parse_args()

    print("Loading panel...", flush=True)
    rallies = load_panel()
    n_gt = sum(len(r["gt_actions"]) for r in rallies)
    print(
        f"Loaded {len(rallies)} rallies, {n_gt} GT actions "
        f"(across {len(PANEL_VIDEOS)} videos)",
        flush=True,
    )
    sanity_check(rallies)

    weight_order = WEIGHT_ORDER[:2] if args.smoke else WEIGHT_ORDER
    max_cycles = 1 if args.smoke else MAX_CYCLES

    weights = DEFAULT_WEIGHTS
    t0 = time.time()
    baseline = evaluate_weights(weights, rallies)
    eval_dt = time.time() - t0
    print(
        f"Baseline (default weights): {baseline}/{n_gt} = {baseline / n_gt:.1%} "
        f"(eval took {eval_dt:.2f}s)",
        flush=True,
    )

    history = [{"cycle": 0, "weights": dataclasses.asdict(weights), "correct": baseline}]
    n_evals = 1
    sweep_t0 = time.time()
    for cycle in range(1, max_cycles + 1):
        cycle_changed = False
        for wname in weight_order:
            current_value = getattr(weights, wname)
            best_value, best_correct = current_value, evaluate_weights(weights, rallies)
            n_evals += 1
            for frac in SWEEP_FRACTIONS:
                if frac == 0.0:
                    continue
                new_value = max(0.1, min(10.0, current_value * (1.0 + frac)))
                trial_weights = dataclasses.replace(weights, **{wname: new_value})
                trial_correct = evaluate_weights(trial_weights, rallies)
                n_evals += 1
                if trial_correct > best_correct:
                    best_value, best_correct = new_value, trial_correct
            if best_value != current_value:
                weights = dataclasses.replace(weights, **{wname: best_value})
                cycle_changed = True
                print(
                    f"  cycle {cycle} {wname}: {current_value:.3f} -> {best_value:.3f}  "
                    f"correct={best_correct}",
                    flush=True,
                )
            else:
                print(
                    f"  cycle {cycle} {wname}: kept {current_value:.3f}  "
                    f"correct={best_correct}",
                    flush=True,
                )
        cycle_correct = evaluate_weights(weights, rallies)
        n_evals += 1
        history.append({
            "cycle": cycle,
            "weights": dataclasses.asdict(weights),
            "correct": cycle_correct,
        })
        print(
            f"End of cycle {cycle}: correct={cycle_correct}/{n_gt} "
            f"({cycle_correct / n_gt:.1%}); changed={cycle_changed}",
            flush=True,
        )
        if not cycle_changed:
            print(f"Cycle {cycle}: no changes; converged.", flush=True)
            break

    sweep_dt = time.time() - sweep_t0

    print()
    print("=" * 60)
    print("CALIBRATED WEIGHTS")
    print("=" * 60)
    print(f"  baseline correct: {baseline}/{n_gt} = {baseline / n_gt:.1%}")
    print(f"  final correct:    {history[-1]['correct']}/{n_gt} = "
          f"{history[-1]['correct'] / n_gt:.1%}")
    print(f"  evaluations:      {n_evals}")
    print(f"  sweep runtime:    {sweep_dt:.1f}s")
    print()
    print("Paste into joint_attribution_weights.py DEFAULT_WEIGHTS:")
    print()
    for wname in WEIGHT_ORDER:
        v = getattr(weights, wname)
        print(f"    {wname}: float = {v:.4f}")

    suffix = "_smoke" if args.smoke else ""
    out_path = Path(f"reports/joint_attribution_calibration_2026_05_12{suffix}.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "baseline_correct": baseline,
        "final_correct": history[-1]["correct"],
        "n_gt": n_gt,
        "n_evals": n_evals,
        "sweep_runtime_s": sweep_dt,
        "history": history,
        "final_weights": dataclasses.asdict(weights),
    }, indent=2))
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
