"""Offline cross-product sweep of MS-TCN++ override guard constraints.

Operates on the per-contact probe JSON produced by
`scripts/diagnose_action_type_errors.py`. Each probe records the
pre-override action_type, the MS-TCN++ non-background probability vector
at the contact frame, and the existing post-override outcome, so we can
resimulate `apply_sequence_override` under any guard combination without
rerunning the pipeline.

Held fixed:
    * `DIG_GUARD_RATIO = 2.5` (existing dig-guard for dig->set overrides).

Swept knobs (all default off = `None`, meaning "current behaviour"):
    1. global_floor       — skip any override if seq_peak < τ
    2. relative_k         — skip if seq_peak < k * pre_override_confidence
    3. attack_preserve    — refuse attack->{set,dig} unless argmax_prob
                             >= τ * seq_prob[attack]
    4. set_preserve       — refuse set->dig unless argmax_prob
                             >= τ * seq_prob[set]
    5. dig_attack_preserve — refuse dig->attack unless argmax_prob
                             >= τ * seq_prob[dig]

Outputs:
    * console Pareto frontier table on
      (action_accuracy, dig_f1, set_f1, attack_f1)
    * JSON at outputs/override_guard_sweep_<date>.json with every cell

Usage:
    cd analysis
    uv run python scripts/sweep_override_guards.py \\
        --probes outputs/action_error_probes_2026-04-14.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from rich.console import Console
from rich.table import Table

console = Console()

# ACTION_TYPES order (from rallycut.actions.trajectory_features):
#   0 serve | 1 receive | 2 set | 3 attack | 4 dig | 5 block
# sequence_probs[1:, f] (non-background) follows this same index order.
ACTION_TYPES = ["serve", "receive", "set", "attack", "dig", "block"]
TARGET_CLASSES = {"dig", "set", "attack"}
ALL_CLASSES_FOR_F1 = ("serve", "receive", "set", "attack", "dig")

# Held fixed: current production value.
DIG_GUARD_RATIO = 2.5


# --------------------------------------------------------------------------- #
# Guard configuration                                                         #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class GuardConfig:
    """A single cell in the sweep grid. None = guard off."""

    global_floor: float | None
    relative_k: float | None
    attack_preserve: float | None
    set_preserve: float | None
    dig_attack_preserve: float | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "global_floor": self.global_floor,
            "relative_k": self.relative_k,
            "attack_preserve": self.attack_preserve,
            "set_preserve": self.set_preserve,
            "dig_attack_preserve": self.dig_attack_preserve,
        }

    def short(self) -> str:
        parts = []
        def fmt(name: str, v: float | None) -> str:
            return f"{name}={v}" if v is not None else f"{name}=off"
        parts.append(fmt("gf", self.global_floor))
        parts.append(fmt("rk", self.relative_k))
        parts.append(fmt("ap", self.attack_preserve))
        parts.append(fmt("sp", self.set_preserve))
        parts.append(fmt("dap", self.dig_attack_preserve))
        return " ".join(parts)


BASELINE_GUARDS = GuardConfig(None, None, None, None, None)


# Cross-product grid (5 knobs).
GRID: dict[str, list[float | None]] = {
    "global_floor":       [None, 0.80, 0.85, 0.90, 0.95],
    "relative_k":         [None, 1.0, 1.2, 1.5],
    "attack_preserve":    [None, 1.5, 2.0, 2.5, 3.0],
    "set_preserve":       [None, 2.0, 2.5, 3.0],
    "dig_attack_preserve": [None, 2.0, 2.5, 3.0],
}


def iter_grid() -> list[GuardConfig]:
    keys = list(GRID.keys())
    values = [GRID[k] for k in keys]
    cells: list[GuardConfig] = []
    for combo in itertools.product(*values):
        cells.append(GuardConfig(**dict(zip(keys, combo))))
    return cells


# --------------------------------------------------------------------------- #
# Override simulation on a single probe                                       #
# --------------------------------------------------------------------------- #


def _simulate_override(probe: dict[str, Any], g: GuardConfig) -> str:
    """Return post-override action_type under the guard config `g`.

    Faithful to `apply_sequence_override` in
    `rallycut/tracking/sequence_action_runtime.py`:
    serve-exempt, synthetic-exempt (captured via override_would_apply),
    existing DIG_GUARD_RATIO held fixed, new guards applied additively.
    """
    if not probe["seq_present"]:
        return probe["pre_override_type"]
    if not probe["override_would_apply"]:
        # Serve would be manufactured, or synthetic, or out-of-range —
        # override doesn't fire for this action regardless of new guards.
        return probe["pre_override_type"]

    argmax = probe["override_argmax"]
    if argmax == "":
        return probe["pre_override_type"]

    probs = probe["seq_probs_nonbg"]  # length 6
    if not probs:
        return probe["pre_override_type"]

    # Resolve indices within the non-background slice (same as ACTION_TYPES).
    try:
        argmax_idx = ACTION_TYPES.index(argmax)
    except ValueError:
        return probe["pre_override_type"]
    argmax_prob = probs[argmax_idx]

    pre = probe["pre_override_type"]
    gbm_conf = probe["pre_override_confidence"]

    # -- New constraint 1: global seq_peak floor.
    if g.global_floor is not None and argmax_prob < g.global_floor:
        return pre

    # -- New constraint 2: relative-confidence gate.
    if g.relative_k is not None and argmax_prob < g.relative_k * gbm_conf:
        return pre

    # -- New constraint 3: attack-preserving guard (covers attack->{set,dig}).
    if (
        g.attack_preserve is not None
        and pre == "attack"
        and argmax in ("set", "dig")
    ):
        attack_prob = probs[ACTION_TYPES.index("attack")]
        if argmax_prob < g.attack_preserve * attack_prob:
            return pre

    # -- New constraint 4a: set-preserving guard (set->dig).
    if (
        g.set_preserve is not None
        and pre == "set"
        and argmax == "dig"
    ):
        set_prob = probs[ACTION_TYPES.index("set")]
        if argmax_prob < g.set_preserve * set_prob:
            return pre

    # -- New constraint 4b: dig-preserving guard (dig->attack).
    if (
        g.dig_attack_preserve is not None
        and pre == "dig"
        and argmax == "attack"
    ):
        dig_prob = probs[ACTION_TYPES.index("dig")]
        if argmax_prob < g.dig_attack_preserve * dig_prob:
            return pre

    # -- Existing DIG_GUARD_RATIO (held fixed, dig->set).
    if pre == "dig" and argmax == "set":
        dig_prob = probs[ACTION_TYPES.index("dig")]
        set_prob = probs[ACTION_TYPES.index("set")]
        if set_prob < DIG_GUARD_RATIO * dig_prob:
            return pre

    return argmax


# --------------------------------------------------------------------------- #
# Metrics on a full probe set                                                 #
# --------------------------------------------------------------------------- #


@dataclass
class CellResult:
    guards: GuardConfig
    action_accuracy: float
    f1: dict[str, float]                 # per-class F1 on TARGET + serve/receive
    n_matched: int
    n_correct: int
    n_errors_target: int                 # dig/set/attack errors
    bucket_A: int                        # override regressions vs pre-override GT
    bucket_B: int
    bucket_C: int
    deltas_vs_baseline: dict[str, float] = None  # populated after baseline known

    def to_dict(self) -> dict[str, Any]:
        return {
            "guards": self.guards.as_dict(),
            "action_accuracy": self.action_accuracy,
            "f1": self.f1,
            "n_matched": self.n_matched,
            "n_correct": self.n_correct,
            "n_errors_target": self.n_errors_target,
            "bucket_A": self.bucket_A,
            "bucket_B": self.bucket_B,
            "bucket_C": self.bucket_C,
            "deltas_vs_baseline": self.deltas_vs_baseline,
        }


def _evaluate_cell(probes: list[dict[str, Any]], g: GuardConfig) -> CellResult:
    """Simulate override on every probe, compute metrics."""
    # We only evaluate GT-matched contacts (probes are all GT-matched by
    # construction — the diagnostic script skipped FNs).
    n_matched = len(probes)
    if n_matched == 0:
        raise RuntimeError("No probes to evaluate.")

    # Counts for per-class F1 across all 5 reported classes.
    per_class_counts = {
        c: {"tp": 0, "fp": 0, "fn": 0} for c in ALL_CLASSES_FOR_F1
    }
    n_correct = 0
    bucket_counts: Counter[str] = Counter()

    for p in probes:
        gt = p["gt_action"]
        pre = p["pre_override_type"]
        post = _simulate_override(p, g)

        if gt == post:
            n_correct += 1

        # Per-class F1 counts
        for c in ALL_CLASSES_FOR_F1:
            if gt == c and post == c:
                per_class_counts[c]["tp"] += 1
            elif gt != c and post == c:
                per_class_counts[c]["fp"] += 1
            elif gt == c and post != c:
                per_class_counts[c]["fn"] += 1

        # Bucket assignment (subset of the original diagnostic's logic, only
        # for dig/set/attack pairs — same scope as the ship criterion).
        if gt in TARGET_CLASSES and post in TARGET_CLASSES and gt != post:
            if not p["seq_present"]:
                bucket_counts["E"] += 1
            elif pre == gt:
                bucket_counts["A"] += 1
            elif pre == post:
                bucket_counts["B"] += 1
            else:
                bucket_counts["C"] += 1

    f1 = {}
    for c in ALL_CLASSES_FOR_F1:
        tp = per_class_counts[c]["tp"]
        fp = per_class_counts[c]["fp"]
        fn = per_class_counts[c]["fn"]
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1[c] = 2 * precision * recall / max(1e-9, precision + recall)

    n_errors_target = sum(
        1 for p in probes
        if p["gt_action"] in TARGET_CLASSES
        and _simulate_override(p, g) in TARGET_CLASSES
        and _simulate_override(p, g) != p["gt_action"]
    )

    return CellResult(
        guards=g,
        action_accuracy=n_correct / n_matched,
        f1=f1,
        n_matched=n_matched,
        n_correct=n_correct,
        n_errors_target=n_errors_target,
        bucket_A=bucket_counts["A"],
        bucket_B=bucket_counts["B"],
        bucket_C=bucket_counts["C"],
    )


# --------------------------------------------------------------------------- #
# Pareto frontier                                                             #
# --------------------------------------------------------------------------- #


PARETO_KEYS: tuple[str, ...] = ("action_accuracy", "dig_f1", "set_f1", "attack_f1")


def _score_tuple(cell: CellResult) -> tuple[float, ...]:
    return (
        cell.action_accuracy,
        cell.f1["dig"],
        cell.f1["set"],
        cell.f1["attack"],
    )


def _pareto_front(cells: list[CellResult]) -> list[CellResult]:
    """Return Pareto-optimal cells (maximize all objectives)."""
    out: list[CellResult] = []
    scores = [_score_tuple(c) for c in cells]
    for i, si in enumerate(scores):
        dominated = False
        for j, sj in enumerate(scores):
            if i == j:
                continue
            # sj dominates si iff sj >= si in every objective and > in at least one
            if all(b >= a for a, b in zip(si, sj)) and any(
                b > a for a, b in zip(si, sj)
            ):
                dominated = True
                break
        if not dominated:
            out.append(cells[i])
    return out


# --------------------------------------------------------------------------- #
# Reporting                                                                   #
# --------------------------------------------------------------------------- #


def _print_pareto(front: list[CellResult], baseline: CellResult) -> None:
    # Sort: primary by action_accuracy desc, then by sum of F1s.
    front = sorted(
        front,
        key=lambda c: (c.action_accuracy, sum(c.f1[k] for k in ("dig", "set", "attack"))),
        reverse=True,
    )

    tbl = Table(title=f"Pareto frontier (n={len(front)}), sorted by action_accuracy")
    tbl.add_column("guards", no_wrap=False)
    tbl.add_column("act_acc", justify="right")
    tbl.add_column("Δ", justify="right")
    tbl.add_column("dig F1", justify="right")
    tbl.add_column("set F1", justify="right")
    tbl.add_column("atk F1", justify="right")
    tbl.add_column("A", justify="right")
    tbl.add_column("B", justify="right")
    tbl.add_column("C", justify="right")

    base = baseline.action_accuracy
    for c in front:
        d = (c.action_accuracy - base) * 100
        tbl.add_row(
            c.guards.short(),
            f"{c.action_accuracy*100:.2f}%",
            f"{d:+.2f}pp",
            f"{c.f1['dig']*100:.1f}%",
            f"{c.f1['set']*100:.1f}%",
            f"{c.f1['attack']*100:.1f}%",
            str(c.bucket_A),
            str(c.bucket_B),
            str(c.bucket_C),
        )
    console.print(tbl)


def _print_baseline(baseline: CellResult) -> None:
    tbl = Table(title="Baseline (all new guards OFF — mirrors current production)")
    tbl.add_column("Metric")
    tbl.add_column("Value", justify="right")
    tbl.add_row("action_accuracy", f"{baseline.action_accuracy*100:.2f}%")
    for c in ALL_CLASSES_FOR_F1:
        tbl.add_row(f"{c} F1", f"{baseline.f1[c]*100:.2f}%")
    tbl.add_row("bucket A (override regressions)", str(baseline.bucket_A))
    tbl.add_row("bucket B (both wrong same)", str(baseline.bucket_B))
    tbl.add_row("bucket C (both wrong diff)", str(baseline.bucket_C))
    tbl.add_row("n_matched", str(baseline.n_matched))
    tbl.add_row("n_correct", str(baseline.n_correct))
    console.print(tbl)


def _compute_deltas(cells: list[CellResult], baseline: CellResult) -> None:
    """Populate deltas_vs_baseline in-place."""
    for c in cells:
        c.deltas_vs_baseline = {
            "action_accuracy": c.action_accuracy - baseline.action_accuracy,
            "bucket_A": c.bucket_A - baseline.bucket_A,
            **{f"{k}_f1": c.f1[k] - baseline.f1[k] for k in ALL_CLASSES_FOR_F1},
        }


def _filter_to_skip_session(
    probes: list[dict[str, Any]],
    poor_session_id: str | None,
) -> list[dict[str, Any]]:
    if not poor_session_id:
        return probes
    return [p for p in probes if p.get("session_id") != poor_session_id]


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probes",
        type=str,
        default="outputs/action_error_probes_2026-04-14.json",
        help="Path to probes JSON from diagnose_action_type_errors.py",
    )
    parser.add_argument(
        "--skip-session",
        type=str,
        default="6f599a0e-b8ea-4bf0-a331-ce7d9ef88164",
        help="Session id to exclude (poor session). Pass '' to disable.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path. Defaults to "
             "outputs/override_guard_sweep_<date>.json",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="How many Pareto cells to print beyond the frontier.",
    )
    args = parser.parse_args()

    probes_path = Path(args.probes)
    with probes_path.open("r", encoding="utf-8") as fh:
        probes_raw = json.load(fh)
    console.print(
        f"[bold]Loaded {len(probes_raw)} probes from {probes_path}[/bold]"
    )

    skip_id = args.skip_session or None
    probes = _filter_to_skip_session(probes_raw, skip_id)
    console.print(
        f"[dim]Evaluating on {len(probes)} probes after skipping session "
        f"'{skip_id[:8] if skip_id else 'none'}'[/dim]\n"
    )

    # Baseline (all new guards off — DIG_GUARD_RATIO=2.5 still applies).
    baseline = _evaluate_cell(probes, BASELINE_GUARDS)
    _print_baseline(baseline)

    # Sanity: verify baseline simulation matches stored post_override_type.
    mismatches = 0
    for p in probes:
        simulated = _simulate_override(p, BASELINE_GUARDS)
        if simulated != p["post_override_type"]:
            mismatches += 1
    console.print(
        f"\n[dim]Baseline parity check: {mismatches} / {len(probes)} "
        f"probes differ from stored post_override_type.[/dim]"
    )
    if mismatches > 5:
        console.print(
            "[red]ERROR: baseline simulation diverges from stored production "
            "outcomes beyond tolerance — aborting.[/red]"
        )
        return

    # Evaluate every cell in the grid.
    cells_all: list[CellResult] = []
    grid = iter_grid()
    console.print(f"\n[bold]Sweeping {len(grid)} cells...[/bold]")
    for i, g in enumerate(grid, start=1):
        res = _evaluate_cell(probes, g)
        cells_all.append(res)
        if i % 200 == 0:
            console.print(f"  ... {i}/{len(grid)}")

    _compute_deltas(cells_all, baseline)

    # Filter to "weakly improving" cells (don't regress action_accuracy by
    # more than a noise-level 0.1pp — keeps the Pareto set bounded to
    # meaningful candidates).
    improving = [
        c for c in cells_all
        if c.action_accuracy >= baseline.action_accuracy
    ]
    console.print(
        f"[bold]{len(improving)} / {len(cells_all)} cells improve or tie "
        f"action_accuracy.[/bold]"
    )

    front = _pareto_front(improving)
    console.print(f"[bold]Pareto frontier: {len(front)} cells.[/bold]\n")
    _print_pareto(front, baseline)

    # Also surface the top-N by action_accuracy among improving cells.
    topn = sorted(
        improving,
        key=lambda c: (c.action_accuracy, sum(c.f1[k] for k in ("dig", "set", "attack"))),
        reverse=True,
    )[: args.top_n]
    if len(topn) > len(front):
        console.print(
            f"\n[bold]Top {args.top_n} cells by action_accuracy[/bold]"
        )
        _print_pareto(topn, baseline)

    # Persist all results.
    if args.output:
        out_path = Path(args.output)
    else:
        date = datetime.now().strftime("%Y-%m-%d")
        out_path = Path("outputs") / f"override_guard_sweep_{date}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_payload = {
        "probes_path": str(probes_path),
        "n_probes": len(probes),
        "skip_session": skip_id,
        "dig_guard_ratio_fixed": DIG_GUARD_RATIO,
        "baseline": baseline.to_dict(),
        "pareto_front": [c.to_dict() for c in front],
        "all_cells": [c.to_dict() for c in cells_all],
    }
    out_path.write_text(json.dumps(out_payload, indent=2, default=str), encoding="utf-8")
    console.print(f"\n[green]Full sweep results written to {out_path}[/green]")


if __name__ == "__main__":
    main()
