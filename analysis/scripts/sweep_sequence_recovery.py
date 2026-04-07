"""Sweep sequence-recovery thresholds against the production dashboard.

Runs `production_eval.py` in-process over a grid of
`(SEQ_RECOVERY_TAU, SEQ_RECOVERY_CLF_FLOOR)` by monkey-patching the
module-level knobs between runs. Reports a table of headline metrics per
cell so the best precision/recall trade-off can be read directly.

Read-only on the canonical GT pool. One run ≈25s on CPU with cached tracks.

Usage:
    cd analysis
    uv run python scripts/sweep_sequence_recovery.py                  # default grid
    uv run python scripts/sweep_sequence_recovery.py --tau 0.8 0.9 --floor 0.0 0.05 0.10
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# Import the eval harness at module level so we can drive it in-process.
import production_eval  # noqa: E402
from eval_action_detection import (  # noqa: E402
    _build_player_positions,  # noqa: F401  (re-exported via production_eval)
    _load_match_team_assignments,
    load_rallies_with_action_gt,
)

from rallycut.tracking.player_tracker import PlayerPosition  # noqa: E402
from rallycut.tracking import sequence_action_runtime  # noqa: E402

console = Console()

BASELINE_FILE = "run_2026-04-07-162648.json"  # canonical baseline for Δ report


@dataclass
class CellResult:
    tau: float
    floor: float
    contact_f1: float
    contact_recall: float
    contact_precision: float
    action_accuracy: float
    court_side: float
    player_attr: float
    serve_id: float
    serve_attr: float
    set_f1: float
    dig_f1: float
    block_f1: float
    serve_f1: float
    receive_f1: float
    attack_f1: float


def _run_cell(
    rallies: list,
    team_map: dict,
    calibrators: dict,
    tau: float,
    floor: float,
) -> CellResult:
    """Run one (tau, floor) cell in-process. Returns a CellResult."""
    # Monkey-patch the two module-level constants the contact detector
    # reads at call time.
    sequence_action_runtime.SEQ_RECOVERY_TAU = tau
    sequence_action_runtime.SEQ_RECOVERY_CLF_FLOOR = floor

    ctx = production_eval.PipelineContext()
    matches, unmatched, rejections = production_eval._run_once(
        rallies, team_map, calibrators, ctx,
    )
    assert not rejections, f"cell tau={tau} floor={floor} rejected {len(rejections)}"
    flat = production_eval._flatten_run(matches, unmatched)
    return CellResult(
        tau=tau,
        floor=floor,
        contact_f1=flat["contact_f1"],
        contact_recall=flat["contact_recall"],
        contact_precision=flat["contact_precision"],
        action_accuracy=flat["action_accuracy"],
        court_side=flat["court_side_accuracy"],
        player_attr=flat["player_attribution_accuracy"],
        serve_id=flat["serve_id_accuracy"],
        serve_attr=flat["serve_attr_accuracy"],
        set_f1=flat.get("per_class::set::f1", 0.0),
        dig_f1=flat.get("per_class::dig::f1", 0.0),
        block_f1=flat.get("per_class::block::f1", 0.0),
        serve_f1=flat.get("per_class::serve::f1", 0.0),
        receive_f1=flat.get("per_class::receive::f1", 0.0),
        attack_f1=flat.get("per_class::attack::f1", 0.0),
    )


def _fmt_delta(val: float, base: float) -> str:
    d = (val - base) * 100
    if abs(d) < 0.05:
        return "   -"
    return f"{d:+.1f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tau", type=float, nargs="+", default=[0.70, 0.80, 0.90])
    parser.add_argument("--floor", type=float, nargs="+", default=[0.00, 0.05, 0.10])
    args = parser.parse_args()

    console.print("[bold]Loading rallies and team map...[/bold]")
    rallies = load_rallies_with_action_gt()
    console.print(f"  {len(rallies)} rallies")

    rally_pos_lookup: dict[str, list[PlayerPosition]] = {}
    for r in rallies:
        if r.positions_json:
            rally_pos_lookup[r.rally_id] = [
                PlayerPosition(
                    frame_number=pp["frameNumber"],
                    track_id=pp["trackId"],
                    x=pp["x"], y=pp["y"],
                    width=pp["width"], height=pp["height"],
                    confidence=pp.get("confidence", 1.0),
                    keypoints=pp.get("keypoints"),
                )
                for pp in r.positions_json
            ]
    video_ids = {r.video_id for r in rallies if r.video_id}
    team_map = _load_match_team_assignments(video_ids, rally_positions=rally_pos_lookup)
    calibrators = production_eval._build_calibrators(video_ids)
    console.print(
        f"  court calibration: {len(calibrators)}/{len(video_ids)} videos\n"
    )

    # Baseline from disk for Δ reporting.
    import json  # noqa: PLC0415
    baseline_path = Path(__file__).resolve().parents[1] / "outputs" / "production_eval" / BASELINE_FILE
    if baseline_path.exists():
        base = json.loads(baseline_path.read_text())["metrics"]
        base_f1 = base["contact_f1"]["mean"]
        base_rec = base["contact_recall"]["mean"]
        base_prec = base["contact_precision"]["mean"]
        base_acc = base["action_accuracy"]["mean"]
        base_court = base["court_side_accuracy"]["mean"]
        base_attr = base["player_attribution_accuracy"]["mean"]
        base_serve_id = base["serve_id_accuracy"]["mean"]
        base_serve_attr = base["serve_attr_accuracy"]["mean"]
        per_class = base["action_accuracy_per_class"]
        base_block = per_class["block"]["f1"]["mean"]
        base_dig = per_class["dig"]["f1"]["mean"]
        base_set = per_class["set"]["f1"]["mean"]
        console.print(
            f"[dim]Baseline ({BASELINE_FILE}): f1={base_f1*100:.1f} "
            f"recall={base_rec*100:.1f} prec={base_prec*100:.1f} "
            f"act={base_acc*100:.1f}[/dim]\n"
        )
    else:
        base_f1 = base_rec = base_prec = base_acc = 0.0
        base_court = base_attr = base_serve_id = base_serve_attr = 0.0
        base_block = base_dig = base_set = 0.0
        console.print(f"[yellow]Baseline file not found: {baseline_path}[/yellow]\n")

    console.print(
        f"[bold]Sweeping {len(args.tau)}x{len(args.floor)} = "
        f"{len(args.tau) * len(args.floor)} cells[/bold]\n"
    )

    results: list[CellResult] = []
    for tau in args.tau:
        for floor in args.floor:
            console.print(f"  tau={tau:.2f} floor={floor:.2f} ...", end="")
            cell = _run_cell(rallies, team_map, calibrators, tau, floor)
            results.append(cell)
            console.print(
                f" contact_f1={cell.contact_f1*100:.1f} "
                f"(rec {cell.contact_recall*100:.1f} / prec {cell.contact_precision*100:.1f}) "
                f"act={cell.action_accuracy*100:.1f}"
            )

    console.print()
    tbl = Table(title="Sequence-recovery sweep (Δ vs baseline)")
    tbl.add_column("τ", justify="right")
    tbl.add_column("floor", justify="right")
    tbl.add_column("ΔcF1", justify="right")
    tbl.add_column("Δrec", justify="right")
    tbl.add_column("Δprec", justify="right")
    tbl.add_column("Δact", justify="right")
    tbl.add_column("Δcourt", justify="right")
    tbl.add_column("Δattr", justify="right")
    tbl.add_column("Δsid", justify="right")
    tbl.add_column("Δsattr", justify="right")
    tbl.add_column("Δdig", justify="right")
    tbl.add_column("Δblock", justify="right")

    for c in results:
        tbl.add_row(
            f"{c.tau:.2f}", f"{c.floor:.2f}",
            _fmt_delta(c.contact_f1, base_f1),
            _fmt_delta(c.contact_recall, base_rec),
            _fmt_delta(c.contact_precision, base_prec),
            _fmt_delta(c.action_accuracy, base_acc),
            _fmt_delta(c.court_side, base_court),
            _fmt_delta(c.player_attr, base_attr),
            _fmt_delta(c.serve_id, base_serve_id),
            _fmt_delta(c.serve_attr, base_serve_attr),
            _fmt_delta(c.dig_f1, base_dig),
            _fmt_delta(c.block_f1, base_block),
        )
    console.print(tbl)

    # Identify best cell by contact_f1 (the hard floor from memory).
    best = max(results, key=lambda c: c.contact_f1)
    console.print(
        f"\n[bold]Best by contact_f1:[/bold] tau={best.tau:.2f} floor={best.floor:.2f} "
        f"→ contact_f1={best.contact_f1*100:.2f}% "
        f"(Δ={((best.contact_f1 - base_f1) * 100):+.2f}pp)"
    )
    # Ship gates from memory/fn_sequence_signal_2026_04.md.
    gates = {
        "contact_recall ≥ 80%": best.contact_recall >= 0.80,
        "contact_precision ≥ 88%": best.contact_precision >= 0.88,
        "contact_f1 ≥ 83.5%": best.contact_f1 >= 0.835,
        "action_accuracy ≥ 92.3%": best.action_accuracy >= 0.923,
    }
    console.print("\n[bold]Ship gates (best cell):[/bold]")
    for name, passed in gates.items():
        mark = "[green]✓[/green]" if passed else "[red]✗[/red]"
        console.print(f"  {mark} {name}")


if __name__ == "__main__":
    main()
