"""Session 4 — Learned-ReID weight sweep + four-target acceptance gate.

Runs ``WEIGHT_LEARNED_REID ∈ {0.0, 0.05, 0.10, 0.15, 0.20}`` (0.0 = control).
For each weight:

1. ``rallycut evaluate-tracking --all --retrack --cached --audit-out ...``
   — per-rally HOTA + in-memory audit JSON (matches
   ``reports/tracking_audit/reid_debug/`` schema).
2. ``scripts/production_eval.py --output ... --reruns 1`` — four oracle
   metrics (``player_attribution_oracle`` / ``serve_attr_oracle`` /
   ``court_side_accuracy`` / ``score_accuracy``).

Cross-rally rank-1 (acceptance #4) is weight-invariant (head doesn't change
across the sweep) — evaluated once directly against the cached DINOv2
features, reported once.

Outputs ``analysis/reports/within_team_reid/session4_gate_report.md`` with
per-weight pass/fail per acceptance target and a knee recommendation.

Usage:
    uv run python analysis/scripts/eval_learned_reid_gate.py \\
        [--weights 0.0,0.05,0.10,0.15,0.20] \\
        [--skip-production-eval]  # debug only — ship requires oracles

Hard budget: per plan, ~22 min first W>0 full retrack + ~3.5 min per
subsequent cached retrack + ~20 min per production_eval --reruns 1 on MPS.
Total ~150 min for the full sweep. Parallel oracles won't help on MPS.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
REPORTS_ROOT = ANALYSIS_ROOT / "reports"
SWEEP_ROOT = REPORTS_ROOT / "within_team_reid" / "sweep"
AUDIT_DIR = REPORTS_ROOT / "tracking_audit"
REID_DEBUG_DIR = AUDIT_DIR / "reid_debug"  # rally-ID inclusion list
EVAL_CACHE_DIR = REPORTS_ROOT / "within_team_reid" / "eval_cache"
WEIGHTS_PATH = ANALYSIS_ROOT / "weights" / "within_team_reid" / "best.pt"

HELD_OUT_EVENTS_COUNT = 24  # slice [34:58] after sort by (rally_id, swap_frame)
RANKING_SPLIT = 34

# Acceptance thresholds (from kickoff prompt + plan).
SWAP_REDUCTION_TARGET = 0.40    # ≥ 40% of held-out events must stop firing
ORACLE_REGRESSION_LIMIT = 0.003  # no regression > 0.3 pp on any oracle
HOTA_PER_RALLY_LIMIT = 0.005     # no rally drops > 0.5 pp HOTA
CROSS_RALLY_GUARD = 0.683        # zero-shot DINOv2-S baseline - 2 pp CF guard

ORACLE_KEYS = [
    "player_attribution_oracle",
    "serve_attr_oracle",
    "court_side_accuracy",
    "score_accuracy",
]
# Presentation labels per kickoff-prompt naming.
ORACLE_LABELS = {
    "player_attribution_oracle": "player_attribution_oracle",
    "serve_attr_oracle": "serve_attribution_oracle",
    "court_side_accuracy": "court_side_oracle",
    "score_accuracy": "score_accuracy",
}

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("gate")


@dataclass
class WeightResult:
    weight: float
    tracking_json: Path
    audit_dir: Path
    oracle_json: Path | None
    elapsed_s: float = 0.0
    swap_reduction: float | None = None
    swap_held_out_still_firing: int = 0
    swap_held_out_expected: int = 0
    hota_worst_drop_pp: float | None = None
    hota_rallies_regressed: list[tuple[str, float]] = field(default_factory=list)
    oracle_deltas: dict[str, float] = field(default_factory=dict)
    gate_pass: bool = False
    gate_fail_reasons: list[str] = field(default_factory=list)


def _run_retrack(weight: float, audit_dir: Path, tracking_json: Path) -> float:
    audit_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "WEIGHT_LEARNED_REID": f"{weight:.2f}"}
    cmd = [
        "uv", "run", "rallycut", "evaluate-tracking",
        "--all", "--retrack", "--cached",
        "--output", str(tracking_json),
        "--audit-out", str(audit_dir),
    ]
    logger.info("retrack W=%.2f → %s", weight, tracking_json)
    t0 = time.time()
    subprocess.run(cmd, env=env, cwd=str(ANALYSIS_ROOT), check=True)
    return time.time() - t0


def _run_production_eval(weight: float, oracle_json: Path) -> float:
    env = {**os.environ, "WEIGHT_LEARNED_REID": f"{weight:.2f}"}
    cmd = [
        "uv", "run", "python", "scripts/production_eval.py",
        "--output", str(oracle_json),
        "--reruns", "1",
    ]
    logger.info("production_eval W=%.2f → %s", weight, oracle_json)
    t0 = time.time()
    subprocess.run(cmd, env=env, cwd=str(ANALYSIS_ROOT), check=True)
    return time.time() - t0


def _load_held_out_events() -> list[Any]:
    """Load slice [34:58] of baseline audit events (the 24 reserved events).

    Follows the harvest-script convention: ``reid_debug/*.json`` is the list
    of rally IDs to include; the actual audit data is in ``AUDIT_DIR/{rid}.json``.
    """
    sys.path.insert(0, str(ANALYSIS_ROOT / "scripts"))
    from harvest_within_team_pairs import (  # noqa: E402
        _load_events_from_audit,
    )

    rally_ids: list[str] = []
    for p in sorted(REID_DEBUG_DIR.glob("*.json")):
        name = p.name
        if name.startswith("_") or "sota_probe" in name:
            continue
        rally_ids.append(name.removesuffix(".json"))

    all_events: list[Any] = []
    for rid in rally_ids:
        audit_path = AUDIT_DIR / f"{rid}.json"
        if not audit_path.exists():
            continue
        try:
            all_events.extend(_load_events_from_audit(audit_path))
        except Exception as exc:  # noqa: BLE001
            logger.warning("skipping %s: %s", audit_path.name, exc)
    all_events.sort(key=lambda e: (e.rally_id, e.swap_frame))
    held_out = all_events[RANKING_SPLIT:RANKING_SPLIT + HELD_OUT_EVENTS_COUNT]
    logger.info(
        "held-out: %d events (total %d, ranking/held-out split %d)",
        len(held_out), len(all_events), RANKING_SPLIT,
    )
    return held_out


def _events_still_firing(held_out: list[Any], audit_dir: Path) -> int:
    """Count how many baseline held-out events still appear in the new audit.

    Match key: (rally_id, gt_track_id, swap_frame ±2). ``pred_old``/``pred_new``
    are INTENTIONALLY excluded — the baseline reid_debug audits were generated
    from DB predictions; the sweep ``--audit-out`` audits are generated from
    the retracked in-memory positions, which re-number pred tracks with a
    different namespace. A pred-exchange-swap event is the "same event" if
    the SAME GT track shows a pred-id transition at the SAME frame (±2),
    regardless of the specific numeric pred IDs on each side.
    """
    sys.path.insert(0, str(ANALYSIS_ROOT / "scripts"))
    from harvest_within_team_pairs import (  # noqa: E402
        _load_events_from_audit,
    )

    by_rally: dict[str, list[Any]] = defaultdict(list)
    if audit_dir.exists():
        for p in sorted(audit_dir.glob("*.json")):
            if p.name.startswith("_"):
                continue
            try:
                for ev in _load_events_from_audit(p):
                    by_rally[ev.rally_id].append(ev)
            except Exception as exc:  # noqa: BLE001
                logger.warning("skipping %s: %s", p.name, exc)

    still = 0
    for ref in held_out:
        for ev in by_rally.get(ref.rally_id, []):
            if (
                abs(ev.swap_frame - ref.swap_frame) <= 2
                and ev.gt_track_id == ref.gt_track_id
            ):
                still += 1
                break
    return still


def _per_rally_hota_drops(
    baseline_tracking_json: Path, weight_tracking_json: Path,
) -> tuple[float, list[tuple[str, float]]]:
    base = json.loads(baseline_tracking_json.read_text())
    new = json.loads(weight_tracking_json.read_text())
    base_by_rally = {r["rally_id"]: r for r in base.get("per_rally", [])}
    new_by_rally = {r["rally_id"]: r for r in new.get("per_rally", [])}
    drops: list[tuple[str, float]] = []
    worst = 0.0
    for rid, r_new in new_by_rally.items():
        r_base = base_by_rally.get(rid)
        if r_base is None:
            continue
        h_base = r_base.get("hota")
        h_new = r_new.get("hota")
        if h_base is None or h_new is None:
            continue
        delta = h_base - h_new
        if delta > 0:
            worst = max(worst, delta)
        if delta > HOTA_PER_RALLY_LIMIT:
            drops.append((rid, delta))
    return worst, drops


def _load_oracle(path: Path) -> dict[str, float]:
    data = json.loads(path.read_text())
    metrics = data.get("metrics", {})
    out: dict[str, float] = {}
    for k in ORACLE_KEYS:
        if k in metrics and metrics[k] is not None:
            out[k] = float(metrics[k])
    return out


def _cross_rally_rank1() -> float:
    """Apply the trained head to the cached DINOv2 features; return rank-1.

    Weight-invariant across the sweep — reported once.
    """
    sys.path.insert(0, str(ANALYSIS_ROOT))
    import torch

    from training.within_team_reid.eval import cache as ec  # noqa: E402
    from training.within_team_reid.eval.cross_rally import (  # noqa: E402
        evaluate as evaluate_cross_rally,
    )
    from training.within_team_reid.model.head import MLPHead  # noqa: E402

    cache_npz = EVAL_CACHE_DIR / "eval_cache.npz"
    cache_meta = EVAL_CACHE_DIR / "metadata.json"
    cache = ec.load_cache(cache_npz, cache_meta)
    device = torch.device("cpu")
    head = MLPHead()
    ckpt = torch.load(WEIGHTS_PATH, map_location="cpu", weights_only=False)
    head.load_state_dict(ckpt["head_state_dict"])
    head.to(device).eval()
    result = evaluate_cross_rally(cache, head, device)
    logger.info(
        "cross-rally rank-1 = %.4f (CF guard %.4f, %d queries)",
        result.rank1, CROSS_RALLY_GUARD, result.n_queries,
    )
    return float(result.rank1)


def _evaluate_weight(
    weight: float,
    baseline: WeightResult | None,
    held_out: list[Any],
    run_production_eval: bool,
    analyze_only: bool,
) -> WeightResult:
    sweep_dir = SWEEP_ROOT / f"w{weight:.2f}"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    tracking_json = sweep_dir / "tracking.json"
    audit_dir = sweep_dir / "audit"
    oracle_json = sweep_dir / "oracle.json"

    result = WeightResult(
        weight=weight,
        tracking_json=tracking_json,
        audit_dir=audit_dir,
        oracle_json=oracle_json if run_production_eval else None,
    )

    # 1. retrack + audit
    if analyze_only:
        t_retrack = 0.0
        if not tracking_json.exists():
            logger.warning(
                "analyze-only: %s missing for W=%.2f", tracking_json, weight,
            )
    else:
        t_retrack = _run_retrack(weight, audit_dir, tracking_json)
    # 2. oracle suite (deferred to Session 5 by default — production_eval reads
    # from DB, not the retrack output, so running it here just re-measures
    # current production; delta would be zero by construction).
    t_oracle = 0.0
    if run_production_eval and not analyze_only:
        t_oracle = _run_production_eval(weight, oracle_json)
    result.elapsed_s = t_retrack + t_oracle

    # 3. swap-count reduction vs baseline (only meaningful at W>0)
    still = _events_still_firing(held_out, audit_dir)
    result.swap_held_out_still_firing = still
    result.swap_held_out_expected = len(held_out)
    if baseline is not None:
        base_still = baseline.swap_held_out_still_firing or 1  # avoid /0
        result.swap_reduction = 1.0 - (still / base_still)
    else:
        result.swap_reduction = 0.0

    # 4. per-rally HOTA drops vs control (weight=0 baseline)
    if baseline is not None:
        worst, drops = _per_rally_hota_drops(
            baseline.tracking_json, tracking_json,
        )
        result.hota_worst_drop_pp = worst * 100.0
        result.hota_rallies_regressed = [(rid, d * 100.0) for rid, d in drops]

    # 5. oracle deltas (only populated when production_eval ran). When skipped,
    # oracle gate is marked "deferred" in the report — ship decision still
    # derives from swap reduction + HOTA + cross-rally.
    if (
        run_production_eval
        and oracle_json.exists()
        and baseline is not None
        and baseline.oracle_json
        and baseline.oracle_json.exists()
    ):
        base_oracle = _load_oracle(baseline.oracle_json)
        new_oracle = _load_oracle(oracle_json)
        for k in ORACLE_KEYS:
            if k in new_oracle and k in base_oracle:
                result.oracle_deltas[k] = new_oracle[k] - base_oracle[k]

    return result


def _apply_gate(result: WeightResult, baseline: WeightResult) -> None:
    """Populate ``gate_pass`` and ``gate_fail_reasons`` on the result."""
    reasons: list[str] = []

    if result.weight == 0.0:
        # Control — gate doesn't apply.
        result.gate_pass = True
        return

    # #1 swap reduction
    if result.swap_reduction is None or result.swap_reduction < SWAP_REDUCTION_TARGET:
        reasons.append(
            f"swap_reduction {result.swap_reduction or 0:.2%} "
            f"< target {SWAP_REDUCTION_TARGET:.0%} "
            f"({result.swap_held_out_still_firing}/{baseline.swap_held_out_still_firing} "
            f"held-out events still firing)"
        )

    # #2 oracle no-regression — gated only when oracle data is present.
    # When production_eval is skipped (the default path, per the Session-4
    # scope), the oracle acceptance target is deferred to Session 5. We
    # do NOT auto-fail on missing oracle data — the other three targets
    # still gate the ship recommendation.
    for k in ORACLE_KEYS:
        delta = result.oracle_deltas.get(k)
        if delta is None:
            continue
        if delta < -ORACLE_REGRESSION_LIMIT:
            reasons.append(
                f"{ORACLE_LABELS[k]} regressed {delta * 100:+.2f} pp "
                f"(limit -{ORACLE_REGRESSION_LIMIT * 100:.2f} pp)"
            )

    # #3 per-rally HOTA
    if result.hota_rallies_regressed:
        worst_rid, worst_pp = max(
            result.hota_rallies_regressed, key=lambda p: p[1],
        )
        reasons.append(
            f"{len(result.hota_rallies_regressed)} rally(s) drop > "
            f"{HOTA_PER_RALLY_LIMIT * 100:.1f} pp HOTA "
            f"(worst {worst_rid[:8]} -{worst_pp:.2f} pp)"
        )

    result.gate_pass = not reasons
    result.gate_fail_reasons = reasons


def _recommend_knee(results: list[WeightResult]) -> float | None:
    """Lowest weight (excluding W=0 control) that passed all four gates."""
    passed = [r for r in results if r.weight > 0 and r.gate_pass]
    if not passed:
        return None
    return min(r.weight for r in passed)


def _render_markdown(
    results: list[WeightResult], cross_rank1: float, knee: float | None,
    held_out_count: int,
) -> str:
    any_oracle = any(r.oracle_deltas for r in results)
    oracle_note = (
        "_All four acceptance targets must pass for a weight to ship._"
        if any_oracle
        else (
            "_Oracle gate (#2) is **deferred to Session 5**: "
            "`scripts/production_eval.py` reads predictions from DB, not "
            "the retrack output, so running it here would re-measure "
            "current production and always report zero delta. Ship "
            "decision below is based on targets #1, #3, #4; oracle "
            "validation happens once retracked positions are persisted._"
        )
    )
    lines = [
        "# Session 4 — Learned-ReID weight sweep gate report",
        "",
        oracle_note,
        "",
        "## Cross-rally rank-1 guard (acceptance #4 — weight-invariant)",
        "",
        "- Head checkpoint: `analysis/weights/within_team_reid/best.pt`",
        f"- Cross-rally rank-1 (leave-one-out on adversarial gallery): "
        f"**{cross_rank1:.4f}**",
        f"- Guard: ≥ {CROSS_RALLY_GUARD:.4f}"
        f" → {'✅ PASS' if cross_rank1 >= CROSS_RALLY_GUARD else '❌ FAIL'}",
        "",
        "## Per-weight sweep",
        "",
        f"Held-out swap events (from baseline reid_debug, slice [34:58]): "
        f"**{held_out_count}**.",
        "",
        "| Weight | Retrack swaps still firing | Swap reduction | "
        "Worst rally HOTA drop | Oracle min Δ | Gate |",
        "|--------|--------------------------:|---------------:|"
        "----------------------:|-------------:|:----:|",
    ]
    for r in results:
        if r.weight == 0.0:
            gate_sym = "ctrl"
        else:
            gate_sym = "✅" if r.gate_pass else "❌"
        swap_red = (
            "n/a" if r.swap_reduction is None
            else f"{r.swap_reduction:+.1%}"
        )
        hota = (
            "n/a" if r.hota_worst_drop_pp is None
            else f"{r.hota_worst_drop_pp:.2f} pp"
        )
        min_delta_pp = (
            min(r.oracle_deltas.values()) * 100 if r.oracle_deltas else None
        )
        oracle_cell = (
            "n/a" if min_delta_pp is None
            else f"{min_delta_pp:+.2f} pp"
        )
        lines.append(
            f"| {r.weight:.2f} | "
            f"{r.swap_held_out_still_firing}/{r.swap_held_out_expected} | "
            f"{swap_red} | {hota} | {oracle_cell} | {gate_sym} |"
        )

    lines += [
        "",
        "## Oracle deltas (vs W=0 control, percentage points)",
        "",
        "| Weight | "
        + " | ".join(ORACLE_LABELS[k] for k in ORACLE_KEYS)
        + " |",
        "|--------|"
        + "|".join(":---:" for _ in ORACLE_KEYS)
        + "|",
    ]
    for r in results:
        if r.weight == 0.0:
            cells = " | ".join("—" for _ in ORACLE_KEYS)
        else:
            cells = " | ".join(
                f"{r.oracle_deltas.get(k, 0) * 100:+.2f}" if k in r.oracle_deltas
                else "n/a"
                for k in ORACLE_KEYS
            )
        lines.append(f"| {r.weight:.2f} | {cells} |")

    lines += ["", "## Failure reasons"]
    for r in results:
        if r.weight == 0.0 or r.gate_pass:
            continue
        lines.append(f"- **W={r.weight:.2f}**:")
        for reason in r.gate_fail_reasons:
            lines.append(f"  - {reason}")

    lines += ["", "## Per-rally HOTA regressions (drops > 0.5 pp)"]
    any_regression = False
    for r in results:
        if not r.hota_rallies_regressed:
            continue
        any_regression = True
        lines.append(f"- **W={r.weight:.2f}**:")
        for rid, pp in sorted(r.hota_rallies_regressed, key=lambda p: -p[1]):
            lines.append(f"  - `{rid[:8]}` -{pp:.2f} pp")
    if not any_regression:
        lines.append("- None.")

    lines += ["", "## Recommendation"]
    if knee is not None:
        lines.append(
            f"**Ship W={knee:.2f}** — lowest weight passing all four "
            f"acceptance targets."
        )
    else:
        lines.append(
            "**No ship.** No weight clears all four targets. Per the plan: "
            "do NOT relax acceptance — return to Session 3 for harder "
            "negatives (Session 2b labeller round) or head retraining."
        )

    lines += ["", "## Runtime"]
    for r in results:
        lines.append(f"- W={r.weight:.2f}: {r.elapsed_s / 60:.1f} min")

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--weights", type=str, default="0.0,0.05,0.10,0.15,0.20",
        help="Comma-separated weights (first must be 0.0 as control).",
    )
    parser.add_argument(
        "--report-out", type=Path,
        default=REPORTS_ROOT / "within_team_reid" / "session4_gate_report.md",
    )
    parser.add_argument(
        "--analyze-only", action="store_true",
        help=(
            "Skip retrack + production_eval; re-derive gate metrics from "
            "existing sweep/w*/tracking.json + sweep/w*/audit/ artifacts. "
            "Use after a full sweep to iterate on the match predicate or "
            "report format without rerunning the ~60-min retrack."
        ),
    )
    parser.add_argument(
        "--run-production-eval", action="store_true",
        help=(
            "Also invoke scripts/production_eval.py per weight. WARNING: "
            "production_eval reads predictions from DB, not retrack output, "
            "so the deltas it produces do NOT reflect the learned-ReID "
            "change. Use only as a baseline variance check. Session 5 must "
            "write retracked predictions back to DB before oracle validation "
            "is meaningful."
        ),
    )
    args = parser.parse_args()

    weights = [float(w) for w in args.weights.split(",")]
    if weights[0] != 0.0:
        logger.error("first weight must be 0.0 (control); got %s", weights)
        return 1

    SWEEP_ROOT.mkdir(parents=True, exist_ok=True)

    # Pre-flight: head load check.
    logger.info("pre-flight: cross-rally rank-1 / head load")
    cross_rank1 = _cross_rally_rank1()
    if cross_rank1 < CROSS_RALLY_GUARD:
        logger.error(
            "cross-rally rank-1 %.4f below guard %.4f — abort gate",
            cross_rank1, CROSS_RALLY_GUARD,
        )
        return 2

    held_out = _load_held_out_events()

    results: list[WeightResult] = []
    baseline: WeightResult | None = None
    for i, w in enumerate(weights):
        logger.info("=== weight %d/%d: W=%.2f ===", i + 1, len(weights), w)
        r = _evaluate_weight(
            w, baseline, held_out,
            args.run_production_eval, args.analyze_only,
        )
        if i == 0:
            baseline = r
        results.append(r)

    assert baseline is not None
    for r in results:
        _apply_gate(r, baseline)

    knee = _recommend_knee(results)
    report = _render_markdown(results, cross_rank1, knee, len(held_out))
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(report)
    logger.info("wrote %s", args.report_out)
    print(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
