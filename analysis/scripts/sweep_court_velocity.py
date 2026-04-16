"""Threshold sweep for court-plane velocity gate.

Runs rallycut evaluate-tracking --all --retrack --cached under a grid of
``RALLYCUT_MAX_MERGE_VELOCITY_METERS`` values, captures per-rally HOTA,
and produces a compact table for go/no-go decision.

Includes a baseline cell (``RALLYCUT_DISABLE_COURT_VELOCITY_GATE=1``) that
reproduces the pre-change image-plane gate (ablation wire).

Usage:
    uv run python scripts/sweep_court_velocity.py \
        --out-dir outputs/velocity_sweep_2026-04-16/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("sweep")

DEFAULT_GRID_METERS = [1.5, 2.0, 2.5, 3.0, 3.5]


def run_cell(
    label: str,
    out_json: Path,
    env_override: dict[str, str],
) -> dict[str, Any]:
    """Run one evaluate-tracking cell; returns parsed JSON output."""
    cmd = [
        "uv", "run", "rallycut", "evaluate-tracking",
        "--all", "--retrack", "--cached",
        "-o", str(out_json),
    ]
    env = os.environ.copy()
    env.update(env_override)
    logger.info(f"[{label}] env={env_override} -> {out_json}")
    start = time.time()
    result = subprocess.run(cmd, env=env, capture_output=True, text=True)
    elapsed = time.time() - start
    if result.returncode != 0:
        logger.error(f"[{label}] FAILED in {elapsed:.0f}s")
        logger.error(f"  stderr tail: {result.stderr[-2000:]}")
        raise SystemExit(1)
    logger.info(f"[{label}] done in {elapsed:.0f}s")
    with out_json.open() as f:
        data: dict[str, Any] = json.load(f)
    return data


def extract_per_rally_hota(cell_json: dict[str, Any]) -> dict[str, float]:
    """Pull per-rally HOTA from evaluate-tracking output JSON."""
    out: dict[str, float] = {}
    rallies = cell_json.get("rallies", [])
    for r in rallies:
        rid = r.get("rallyId") or r.get("rally_id")
        hota_val: float | None = None
        hm = r.get("hota") or r.get("hotaMetrics") or r.get("hota_metrics")
        if isinstance(hm, dict):
            nested = hm.get("hota")
            if isinstance(nested, (int, float)):
                hota_val = float(nested)
        elif isinstance(hm, (int, float)):
            hota_val = float(hm)
        if rid is not None and hota_val is not None:
            out[str(rid)] = hota_val
    return out


def aggregate_hota(per_rally: dict[str, float]) -> float:
    if not per_rally:
        return 0.0
    return sum(per_rally.values()) / len(per_rally)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--grid", type=float, nargs="+", default=DEFAULT_GRID_METERS,
        help="Court-plane threshold grid in metres.",
    )
    parser.add_argument(
        "--baseline-only", action="store_true",
        help="Only run the disable-court-gate baseline.",
    )
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    cells: list[tuple[str, Path, dict[str, str]]] = [
        (
            "baseline_image_020",
            args.out_dir / "baseline_image_020.json",
            {"RALLYCUT_DISABLE_COURT_VELOCITY_GATE": "1"},
        ),
    ]
    if not args.baseline_only:
        for m in args.grid:
            cells.append((
                f"court_{m:.1f}m",
                args.out_dir / f"court_{m:.1f}m.json",
                {"RALLYCUT_MAX_MERGE_VELOCITY_METERS": str(m)},
            ))

    results: dict[str, dict[str, Any]] = {}
    for label, out_json, env in cells:
        cell_json = run_cell(label, out_json, env)
        per_rally = extract_per_rally_hota(cell_json)
        results[label] = {
            "env": env,
            "per_rally_hota": per_rally,
            "aggregate_hota": aggregate_hota(per_rally),
            "n_rallies": len(per_rally),
        }

    # Compact summary
    baseline = results.get("baseline_image_020", {})
    baseline_hota = baseline.get("per_rally_hota", {})

    print("\n=== Sweep summary ===")
    print(f"{'cell':<24} {'agg_HOTA':>10} {'ΔbaseHOTA':>12} "
          f"{'worst_rally':>14} {'worst_Δ':>10} {'e5c1a9b3':>10}")
    for label, data in results.items():
        agg = data["aggregate_hota"]
        per = data["per_rally_hota"]
        delta_base = agg - baseline.get("aggregate_hota", 0.0)
        worst_delta = 0.0
        worst_rid = ""
        for rid, h in per.items():
            bh = baseline_hota.get(rid)
            if bh is None:
                continue
            d = h - bh
            if d < worst_delta:
                worst_delta = d
                worst_rid = rid[:8]
        e5 = None
        for rid, h in per.items():
            if rid.startswith("e5c1a9b3"):
                e5 = h
                break
        print(
            f"{label:<24} {agg:>10.4f} {delta_base:>+12.4f} "
            f"{worst_rid:>14} {worst_delta:>+10.4f} "
            f"{(f'{e5:.4f}' if e5 is not None else '-'):>10}"
        )

    summary_path = args.out_dir / "sweep_summary.json"
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSummary: {summary_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
