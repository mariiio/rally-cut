"""Sweep the Pattern-A two-arm rescue gate to pick (FLOOR_MULTIGEN, MIN_GENERATORS).

Runs `scripts/build_action_error_corpus.py` once per grid cell with monkey-
patched module constants, captures per-rally totals, and writes a markdown
report comparing each cell to the Arm-B-disabled baseline.

Per-rally regression rule: flag a rally if `(TP + wrong_player)` drops by
>=1 OR `extra_pred` count rises by >=2.

Threshold selection (applied by the human, not this script): maximize ΔTP
subject to Δextra_pred / ΔTP <= 0.5, n_rallies_regressed <= 5, ΔFN <= 0,
and all four user-confirmed rescues recover.

Usage (from analysis/):
    uv run python scripts/sweep_rescue_gate.py
    uv run python scripts/sweep_rescue_gate.py --dry-run   # list cells, exit
    uv run python scripts/sweep_rescue_gate.py --cells 0.08x3,0.10x3
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ANALYSIS_DIR = Path(__file__).parent.parent
OUTPUT_DIR = ANALYSIS_DIR / "outputs" / "action_errors"
SWEEP_DIR = OUTPUT_DIR / "sweep"
REPORT_PATH = ANALYSIS_DIR / "reports" / "pattern_a_rescue_sweep_2026_04_17.md"

# Grid: (FLOOR_MULTIGEN, MIN_GENERATORS). "Baseline" disables Arm B by
# requiring an unreachable generator count, preserving Arm A only.
GRID: list[tuple[float, int]] = [
    (0.05, 2),
    (0.05, 3),
    (0.08, 2),
    (0.08, 3),
    (0.10, 2),
    (0.10, 3),
    (0.12, 2),
    (0.12, 3),
    (0.15, 2),
    (0.15, 3),
]
BASELINE_CELL = ("baseline", "baseline")  # Arm B disabled (MIN_GENERATORS=999).

# Rallies corresponding to the four user-confirmed Pattern-A rescues. The
# sweep checks that each cell rescues them (i.e., they no longer appear as
# FN_contact at the listed gt_frame).
USER_CONFIRMED_RESCUES: list[tuple[str, int]] = [
    ("fb7f9c23-3544-48bd-910d-10a8f12fd594", 230),
    ("99a01ce4-4a00-4e54-b015-80f843ff21fc", 371),
    ("99a01ce4-4a00-4e54-b015-80f843ff21fc", 813),
    ("71c5d769-581e-4302-ad79-87cfdcb71b80", 234),
]


def _runner_code(floor: float | str, min_gens: int | str) -> str:
    """Return a Python program string that runs the corpus builder with the
    two rescue constants overridden. Executed in a clean subprocess per cell.
    """
    if floor == "baseline":
        floor_stmt = ""  # leave module default
        gens_stmt = "_sar.SEQ_RECOVERY_MIN_GENERATORS = 999  # Arm B disabled"
    else:
        floor_stmt = f"_sar.SEQ_RECOVERY_CLF_FLOOR_MULTIGEN = {float(floor)}"
        gens_stmt = f"_sar.SEQ_RECOVERY_MIN_GENERATORS = {int(min_gens)}"
    return (
        "import sys\n"
        "sys.argv = ['build_action_error_corpus.py']\n"
        "import rallycut.tracking.sequence_action_runtime as _sar\n"
        f"{floor_stmt}\n{gens_stmt}\n"
        "from scripts.build_action_error_corpus import main\n"
        "main()\n"
    )


def _cell_id(floor: float | str, min_gens: int | str) -> str:
    if floor == "baseline":
        return "baseline"
    return f"floor{float(floor):.2f}_gens{int(min_gens)}"


def _run_cell(floor: float | str, min_gens: int | str) -> Path:
    """Run the corpus builder with the given constants and snapshot outputs.

    Returns the cell output directory.
    """
    cell_id = _cell_id(floor, min_gens)
    cell_dir = SWEEP_DIR / cell_id
    cell_dir.mkdir(parents=True, exist_ok=True)

    code = _runner_code(floor, min_gens)
    print(f"\n=== Cell {cell_id} ===")
    print(
        f"    FLOOR_MULTIGEN={floor} MIN_GENERATORS="
        f"{'disabled (999)' if floor == 'baseline' else min_gens}"
    )
    t0 = time.monotonic()
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=ANALYSIS_DIR,
        capture_output=True,
        text=True,
    )
    elapsed = time.monotonic() - t0
    (cell_dir / "stdout.log").write_text(proc.stdout)
    (cell_dir / "stderr.log").write_text(proc.stderr)
    if proc.returncode != 0:
        print(f"  FAILED in {elapsed:.1f}s; see {cell_dir}/stderr.log")
        raise SystemExit(1)

    for name in ("corpus.jsonl", "rally_quality.json", "corpus_annotated.jsonl"):
        src = OUTPUT_DIR / name
        if src.exists():
            shutil.copy2(src, cell_dir / name)
    print(f"  ok in {elapsed:.1f}s; snapshot -> {cell_dir}")
    return cell_dir


def _load_cell(cell_dir: Path) -> dict:
    """Load a cell's per-rally totals and aggregates."""
    rows = [
        json.loads(line)
        for line in (cell_dir / "corpus.jsonl").read_text().splitlines()
        if line
    ]
    quality = json.loads((cell_dir / "rally_quality.json").read_text())

    rally_stats: dict[str, dict] = {
        rid: {"tp": 0, "fn": 0, "wrong_action": 0, "wrong_player": 0}
        for rid in quality.keys()
    }

    # Errors give us FN/wrong_action/wrong_player directly.
    for r in rows:
        rid = r["rally_id"]
        if rid not in rally_stats:
            rally_stats[rid] = {"tp": 0, "fn": 0, "wrong_action": 0, "wrong_player": 0}
        ec = r["error_class"]
        if ec == "FN_contact":
            rally_stats[rid]["fn"] += 1
        elif ec == "wrong_action":
            rally_stats[rid]["wrong_action"] += 1
        elif ec == "wrong_player":
            rally_stats[rid]["wrong_player"] += 1

    # TP = gt_contact_count - (FN + wrong_action + wrong_player)
    for rid, st in rally_stats.items():
        gt = quality.get(rid, {}).get("gt_contact_count", 0)
        st["tp"] = max(0, gt - st["fn"] - st["wrong_action"] - st["wrong_player"])
        st["extra_pred"] = quality.get(rid, {}).get("extra_predictions", 0)

    totals = {
        "tp": sum(s["tp"] for s in rally_stats.values()),
        "fn": sum(s["fn"] for s in rally_stats.values()),
        "wrong_action": sum(s["wrong_action"] for s in rally_stats.values()),
        "wrong_player": sum(s["wrong_player"] for s in rally_stats.values()),
        "extra_pred": sum(s["extra_pred"] for s in rally_stats.values()),
    }

    # Did each user-confirmed rescue land as TP (no FN at that frame)?
    confirmed = {}
    for rid, frame in USER_CONFIRMED_RESCUES:
        fn_at_frame = any(
            r["rally_id"] == rid
            and r["gt_frame"] == frame
            and r["error_class"] == "FN_contact"
            for r in rows
        )
        confirmed[f"{rid[:8]}:{frame}"] = not fn_at_frame

    return {"totals": totals, "rally_stats": rally_stats, "confirmed": confirmed}


def _compare(baseline: dict, cell: dict) -> dict:
    b = baseline["totals"]
    c = cell["totals"]
    delta = {
        "tp": c["tp"] - b["tp"],
        "fn": c["fn"] - b["fn"],
        "wrong_action": c["wrong_action"] - b["wrong_action"],
        "wrong_player": c["wrong_player"] - b["wrong_player"],
        "extra_pred": c["extra_pred"] - b["extra_pred"],
    }
    # Per-rally regression: (TP + wrong_player) drops by >=1 OR extra_pred rises by >=2.
    regressed: list[str] = []
    for rid, b_st in baseline["rally_stats"].items():
        c_st = cell["rally_stats"].get(rid)
        if c_st is None:
            continue
        b_score = b_st["tp"] + b_st["wrong_player"]
        c_score = c_st["tp"] + c_st["wrong_player"]
        extra_delta = c_st["extra_pred"] - b_st["extra_pred"]
        if c_score - b_score <= -1 or extra_delta >= 2:
            regressed.append(rid)
    return {"delta": delta, "regressed": regressed, "confirmed": cell["confirmed"]}


def _write_report(results: dict[str, dict]) -> None:
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    baseline = results[_cell_id(*BASELINE_CELL)]
    b = baseline["totals"]

    lines: list[str] = []
    lines.append("# Pattern A rescue-gate sweep — 2026-04-17\n")
    lines.append("## Setup\n")
    lines.append(
        "Two-arm rescue gate (Arm A unchanged at `SEQ_RECOVERY_CLF_FLOOR=0.20`). "
        "Arm B activates when `conf >= FLOOR_MULTIGEN AND n_generators >= MIN_GENERATORS "
        "AND player_distance <= 0.15 AND seq_peak >= 0.80`. Baseline disables Arm B "
        "(`SEQ_RECOVERY_MIN_GENERATORS=999`).\n"
    )
    lines.append("## Baseline (Arm A only)\n")
    lines.append(
        f"TP={b['tp']}  FN={b['fn']}  wrong_action={b['wrong_action']}  "
        f"wrong_player={b['wrong_player']}  extra_pred={b['extra_pred']}\n"
    )
    lines.append("## Grid\n")
    lines.append(
        "| FLOOR | MIN_GENS | ΔTP | ΔFN | Δwrong_action | Δwrong_player | "
        "Δextra_pred | n_regressed | user_rescues (4 max) |"
    )
    lines.append(
        "|------:|---------:|----:|----:|--------------:|--------------:|"
        "------------:|------------:|:---------------------|"
    )
    for floor, min_gens in GRID:
        cid = _cell_id(floor, min_gens)
        if cid not in results:
            continue
        cell = results[cid]
        cmp = _compare(baseline, cell)
        d = cmp["delta"]
        n_reg = len(cmp["regressed"])
        confirmed = sum(1 for v in cmp["confirmed"].values() if v)
        lines.append(
            f"| {floor:.2f} | {min_gens} | "
            f"{d['tp']:+d} | {d['fn']:+d} | {d['wrong_action']:+d} | "
            f"{d['wrong_player']:+d} | {d['extra_pred']:+d} | "
            f"{n_reg} | {confirmed}/4 |"
        )
    lines.append("\n## Threshold-selection guidance\n")
    lines.append(
        "- Maximize ΔTP subject to Δextra_pred / ΔTP ≤ 0.5, n_regressed ≤ 5, "
        "ΔFN ≤ 0, and user_rescues = 4/4.\n"
        "- Tiebreak toward higher FLOOR (less invasive) then higher MIN_GENS "
        "(stronger trajectory evidence).\n"
        "- If no cell clears the bar, raise MIN_GENS to 4 and rerun the floor "
        "sweep; else document NO SHIP.\n"
    )
    lines.append("\n## Per-rally regressions (top cells)\n")
    for floor, min_gens in GRID:
        cid = _cell_id(floor, min_gens)
        if cid not in results:
            continue
        cmp = _compare(baseline, results[cid])
        if cmp["regressed"]:
            lines.append(
                f"- **{cid}** ({len(cmp['regressed'])} rallies): "
                + ", ".join(sorted(rid[:8] for rid in cmp["regressed"])[:10])
                + ("…" if len(cmp["regressed"]) > 10 else "")
            )
    REPORT_PATH.write_text("\n".join(lines) + "\n")
    print(f"\nReport written to {REPORT_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Print cells and exit without running")
    ap.add_argument("--cells", type=str, default="",
                    help="Comma list of 'floor x gens' (e.g. '0.08x3,0.10x3'). "
                         "If empty, run full grid + baseline.")
    ap.add_argument("--skip-baseline", action="store_true",
                    help="Skip baseline cell (use an existing snapshot).")
    args = ap.parse_args()

    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    if args.cells:
        wanted = set()
        for tok in args.cells.split(","):
            f, g = tok.strip().split("x")
            wanted.add((float(f), int(g)))
        cells: list[tuple[float | str, int | str]] = [c for c in GRID if c in wanted]
    else:
        cells = [BASELINE_CELL] if not args.skip_baseline else []
        cells += list(GRID)

    if args.dry_run:
        for c in cells:
            print(_cell_id(*c), c)
        return

    results: dict[str, dict] = {}
    t0 = time.monotonic()
    for i, (floor, min_gens) in enumerate(cells):
        cell_dir = _run_cell(floor, min_gens)
        results[_cell_id(floor, min_gens)] = _load_cell(cell_dir)
        print(f"Progress: {i+1}/{len(cells)} cells complete")
    print(f"\nTotal sweep time: {time.monotonic() - t0:.1f}s")

    # If baseline was skipped, load from an existing snapshot.
    if _cell_id(*BASELINE_CELL) not in results:
        bl_dir = SWEEP_DIR / _cell_id(*BASELINE_CELL)
        if bl_dir.exists():
            results[_cell_id(*BASELINE_CELL)] = _load_cell(bl_dir)

    if _cell_id(*BASELINE_CELL) in results:
        _write_report(results)
    else:
        print("No baseline snapshot; skipping report.")


if __name__ == "__main__":
    main()
