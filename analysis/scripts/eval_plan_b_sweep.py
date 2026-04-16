"""Session 8 Plan B — 2D sweep: learned-ReID veto × court-plane velocity gate.

Task 2 diagnostic (diagnose_per_pass_swaps.py) falsified the multi-site thesis:
10 of 12 SAME_TEAM_SWAPs come from a single pass (link_tracklets_by_appearance,
step 0c). The other 2 come from raw BoT-SORT (unfixable by merge veto). All 5
other passes contribute 0 swaps — adapter work there would be wasted.

Plan B: stay at one merge site, stack two orthogonal signals:
  1. Learned-ReID cosine veto  (LEARNED_MERGE_VETO_COS — Session 6)
  2. Court-plane velocity gate (ENABLE_COURT_VELOCITY_GATE + MAX_VEL_M)

8-cell grid:
  baseline            | cos=0.0, gate=0, vel=—
  learned_t080        | cos=0.8, gate=0, vel=—    (Session 6 best reference)
  velocity_v25        | cos=0.0, gate=1, vel=2.5   (velocity-only @ default)
  velocity_v35        | cos=0.0, gate=1, vel=3.5   (velocity-only @ looser)
  combined_070_v25    | cos=0.7, gate=1, vel=2.5
  combined_070_v35    | cos=0.7, gate=1, vel=3.5
  combined_080_v25    | cos=0.8, gate=1, vel=2.5
  combined_080_v35    | cos=0.8, gate=1, vel=3.5

Ship gate (all 3 must hold vs baseline):
  1. SAME_TEAM_SWAP reduction ≥ 50%
  2. No per-rally HOTA drop > 0.5 pp
  3. Fragmentation delta ≤ 20%
  (Gate 4 = player_attribution_oracle: deferred to Task 9)

Usage:
    uv run python scripts/eval_plan_b_sweep.py
    uv run python scripts/eval_plan_b_sweep.py --baseline-only   # validate first
    uv run python scripts/eval_plan_b_sweep.py --reuse-existing  # skip completed cells
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ANALYSIS_ROOT / "reports" / "merge_veto"
SWEEP_DIR = OUT_DIR / "session8_sweep"

SWAP_REDUCTION_FLOOR = 0.50
PER_RALLY_HOTA_LIMIT = 0.005   # 0.5 pp
FRAGMENTATION_DELTA_LIMIT = 0.20  # 20 %

# Expected baseline swap count for sanity check
EXPECTED_BASELINE_SWAPS = 12

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("eval_plan_b_sweep")


class CellSpec(NamedTuple):
    name: str
    learned_cos: float   # 0.0 = disabled
    enable_gate: bool    # ENABLE_COURT_VELOCITY_GATE
    max_vel_m: float     # RALLYCUT_MAX_MERGE_VELOCITY_METERS (ignored when gate off)


CELLS: list[CellSpec] = [
    CellSpec("baseline",         0.0,  False, 2.5),
    CellSpec("learned_t080",     0.80, False, 2.5),
    CellSpec("velocity_v25",     0.0,  True,  2.5),
    CellSpec("velocity_v35",     0.0,  True,  3.5),
    CellSpec("combined_070_v25", 0.70, True,  2.5),
    CellSpec("combined_070_v35", 0.70, True,  3.5),
    CellSpec("combined_080_v25", 0.80, True,  2.5),
    CellSpec("combined_080_v35", 0.80, True,  3.5),
]


@dataclass
class CellResult:
    spec: CellSpec
    tracking_json: Path
    audit_dir: Path
    elapsed_s: float = 0.0
    aggregate_hota: float = 0.0
    same_team_swaps: int = 0
    net_crossing: int = 0
    total_real_switches: int = 0
    per_rally_hota: dict[str, float] = field(default_factory=dict)
    per_rally_unique_pred_ids: dict[str, int] = field(default_factory=dict)


def _build_env(spec: CellSpec) -> dict[str, str]:
    env = dict(os.environ)
    env["LEARNED_MERGE_VETO_COS"] = f"{spec.learned_cos:.3f}"
    env["ENABLE_COURT_VELOCITY_GATE"] = "1" if spec.enable_gate else "0"
    if spec.enable_gate:
        env["RALLYCUT_MAX_MERGE_VELOCITY_METERS"] = f"{spec.max_vel_m:.2f}"
    else:
        env.pop("RALLYCUT_MAX_MERGE_VELOCITY_METERS", None)
    # Ensure the learned-ReID embedding head is wired (Session 4 cache key)
    env.setdefault("WEIGHT_LEARNED_REID", "0.05")
    return env


def _run_cell(spec: CellSpec, sweep_dir: Path) -> float:
    audit_dir = sweep_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    tracking_json = sweep_dir / "tracking.json"

    env = _build_env(spec)
    cmd = [
        "uv", "run", "rallycut", "evaluate-tracking",
        "--all", "--retrack", "--cached",
        "--output", str(tracking_json),
        "--audit-out", str(audit_dir),
    ]
    logger.info(
        "[cell=%s] cos=%.2f gate=%s vel=%.1f → %s",
        spec.name, spec.learned_cos,
        "on" if spec.enable_gate else "off",
        spec.max_vel_m if spec.enable_gate else 0.0,
        sweep_dir,
    )
    t0 = time.time()
    subprocess.run(cmd, env=env, cwd=str(ANALYSIS_ROOT), check=True)
    return time.time() - t0


def _parse_cell(spec: CellSpec, sweep_dir: Path) -> CellResult:
    tracking_json = sweep_dir / "tracking.json"
    audit_dir = sweep_dir / "audit"
    result = CellResult(spec=spec, tracking_json=tracking_json, audit_dir=audit_dir)

    # Per-rally HOTA from tracking.json
    # Structure: {"rallies": [{rallyId, hota: {hota, deta, assa}, ...}], "aggregate": {...}}
    if tracking_json.exists():
        data = json.loads(tracking_json.read_text())
        hotas: list[float] = []
        for r in data.get("rallies", []):
            rid = r.get("rallyId")
            # hota is a nested dict: {"hota": float, "deta": float, "assa": float}
            hota_obj = r.get("hota") or {}
            h = hota_obj.get("hota") or 0.0 if isinstance(hota_obj, dict) else (hota_obj or 0.0)
            if rid is not None:
                result.per_rally_hota[rid] = h
                hotas.append(h)
        result.aggregate_hota = sum(hotas) / len(hotas) if hotas else 0.0

    # SAME_TEAM_SWAP counts + fragmentation (unique pred IDs)
    for p in sorted(audit_dir.glob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        rid = d.get("rallyId")
        for g in d.get("perGt", []):
            for sw in g.get("realSwitches", []):
                cause = sw.get("cause")
                if cause == "same_team_swap":
                    result.same_team_swaps += 1
                elif cause == "net_crossing":
                    result.net_crossing += 1
                result.total_real_switches += 1
        pred_ids: set[int] = set()
        for g in d.get("perGt", []):
            for _, _, pid in g.get("predIdSpans", []):
                if pid >= 0:
                    pred_ids.add(pid)
        if rid is not None:
            result.per_rally_unique_pred_ids[rid] = len(pred_ids)

    return result


def _evaluate_gate(
    baseline: CellResult, cell: CellResult
) -> tuple[bool, str]:
    """Evaluate the three ship gates vs baseline.

    Returns (True, 'clears all gates') or (False, reason).
    Gate 4 (player_attribution_oracle) is deferred to Task 9.
    """
    base_sw = baseline.same_team_swaps

    # Gate 1: ≥ 50% SAME_TEAM_SWAP reduction
    if base_sw > 0:
        reduction = (base_sw - cell.same_team_swaps) / base_sw
        if reduction < SWAP_REDUCTION_FLOOR:
            return (
                False,
                f"Gate 1 FAIL: swap reduction {reduction:.1%} < {SWAP_REDUCTION_FLOOR:.0%} "
                f"({cell.same_team_swaps}/{base_sw} remain)",
            )
    else:
        # No baseline swaps — trivially passes gate 1
        pass

    # Gate 2: no per-rally HOTA drop > 0.5 pp
    regressions: list[tuple[str, float]] = []
    for rid, h_new in cell.per_rally_hota.items():
        h_base = baseline.per_rally_hota.get(rid)
        if h_base is None:
            continue
        delta = h_base - h_new
        if delta > PER_RALLY_HOTA_LIMIT:
            regressions.append((rid, delta))
    if regressions:
        regressions.sort(key=lambda p: -p[1])
        worst_rid, worst_delta = regressions[0]
        return (
            False,
            f"Gate 2 FAIL: {len(regressions)} rally/rallies regressed > 0.5 pp; "
            f"worst `{worst_rid[:8]}` −{worst_delta * 100:.2f} pp",
        )

    # Gate 3: fragmentation delta ≤ 20% per rally
    frag_failures: list[tuple[str, int, int]] = []
    for rid, count in cell.per_rally_unique_pred_ids.items():
        base = baseline.per_rally_unique_pred_ids.get(rid)
        if base is None or base == 0:
            continue
        ratio = (count - base) / base
        if ratio > FRAGMENTATION_DELTA_LIMIT:
            frag_failures.append((rid, base, count))
    if frag_failures:
        frag_failures.sort(key=lambda r: -(r[2] - r[1]) / r[1])
        worst_rid, base_n, new_n = frag_failures[0]
        return (
            False,
            f"Gate 3 FAIL: {len(frag_failures)} rally/rallies exceed 20% fragmentation; "
            f"worst `{worst_rid[:8]}` {base_n}→{new_n} pred-IDs "
            f"(+{((new_n - base_n) / base_n) * 100:.0f} %)",
        )

    return (True, "clears all gates")


def _hota_regressions(
    baseline: CellResult, cell: CellResult
) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for rid, h_new in cell.per_rally_hota.items():
        h_base = baseline.per_rally_hota.get(rid)
        if h_base is None:
            continue
        delta = h_base - h_new
        if delta > PER_RALLY_HOTA_LIMIT:
            out.append((rid, delta))
    out.sort(key=lambda p: -p[1])
    return out


def _fragmentation_regressions(
    baseline: CellResult, cell: CellResult
) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    for rid, count in cell.per_rally_unique_pred_ids.items():
        base = baseline.per_rally_unique_pred_ids.get(rid)
        if base is None or base == 0:
            continue
        ratio = (count - base) / base
        if ratio > FRAGMENTATION_DELTA_LIMIT:
            out.append((rid, base, count))
    out.sort(key=lambda r: -(r[2] - r[1]) / r[1])
    return out


def _render_report(
    baseline: CellResult,
    cells: list[CellResult],
    path: Path,
) -> None:
    base_sw = baseline.same_team_swaps
    base_hota = baseline.aggregate_hota
    base_frag_total = sum(baseline.per_rally_unique_pred_ids.values())

    lines = [
        "# Session 8 Plan B — 2D Sweep: Learned-ReID × Court-Velocity Gate",
        "",
        "## Methodology",
        "",
        "Task 2 diagnostic (per-pass swap attribution, commit 350cd3a) falsified the",
        "Session-8 multi-site thesis. Attribution showed:",
        "",
        "- **10 of 12 SAME_TEAM_SWAPs** come from a single pass: `link_tracklets_by_appearance` (step 0c).",
        "- **2 of 12** come from raw BoT-SORT + filtering (pre-merge-chain; no merge veto can fix these).",
        "- **5 other merge/rename passes contribute 0 swaps** — no adapter work needed there.",
        "",
        "Plan B: stay at one merge site, stack two orthogonal vetoes (both already implemented",
        "and env-gated; no new code required):",
        "",
        "1. **Learned-ReID cosine veto** (`LEARNED_MERGE_VETO_COS`) — Session 6 ship",
        "2. **Court-plane velocity gate** (`ENABLE_COURT_VELOCITY_GATE` + `RALLYCUT_MAX_MERGE_VELOCITY_METERS`)",
        "",
        "When both env vars are on, either veto can block a merge (OR-logic). Combined cells",
        "should satisfy `swaps ≤ min(learned_only, velocity_only)` — verified in results below.",
        "",
        "**Baseline**: all merge passes on, no vetoes (LEARNED_MERGE_VETO_COS=0, ENABLE_COURT_VELOCITY_GATE=0).",
        f"Baseline SAME_TEAM_SWAPs: **{base_sw}** (expected 12; {'OK' if base_sw == EXPECTED_BASELINE_SWAPS else 'MISMATCH'}).",
        "",
        "## Ship Gate",
        "",
        "All three must hold vs baseline on 43 GT rallies:",
        "1. SAME_TEAM_SWAP reduction ≥ 50% (≤6 remaining)",
        "2. No per-rally HOTA drop > 0.5 pp",
        "3. Fragmentation delta ≤ 20%",
        "4. `player_attribution_oracle` — **deferred to Task 9**",
        "",
        "## 8-Cell Summary",
        "",
        "| Cell | LEARNED | GATE | VEL_M | HOTA agg | HOTA Δ pp | Swaps | Swap Δ | Frag | Frag Δ | Gate | Reason |",
        "|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|",
    ]

    for c in cells:
        hota_delta_pp = (c.aggregate_hota - base_hota) * 100
        hota_delta_str = f"{hota_delta_pp:+.2f}" if c is not baseline else "—"

        if base_sw > 0:
            swap_reduction = (base_sw - c.same_team_swaps) / base_sw
            swap_delta_str = f"{c.same_team_swaps - base_sw:+d} ({-swap_reduction:.0%})" if c is not baseline else "—"
        else:
            swap_delta_str = "—"

        cell_frag_total = sum(c.per_rally_unique_pred_ids.values())
        if base_frag_total > 0 and c is not baseline:
            frag_ratio = (cell_frag_total - base_frag_total) / base_frag_total
            frag_delta_str = f"{frag_ratio:+.1%}"
        else:
            frag_delta_str = "—"

        if c is baseline:
            gate_sym = "ctrl"
            gate_reason = "baseline"
        else:
            passed, reason = _evaluate_gate(baseline, c)
            gate_sym = "✅" if passed else "❌"
            gate_reason = reason

        vel_str = f"{c.spec.max_vel_m:.1f}" if c.spec.enable_gate else "—"

        lines.append(
            f"| `{c.spec.name}` | {c.spec.learned_cos:.2f} | {'1' if c.spec.enable_gate else '0'} | {vel_str} | "
            f"{c.aggregate_hota * 100:.2f}% | {hota_delta_str} | "
            f"{c.same_team_swaps} | {swap_delta_str} | "
            f"{cell_frag_total} | {frag_delta_str} | "
            f"{gate_sym} | {gate_reason} |"
        )

    # Per-cell detail
    lines += ["", "## Per-Cell Detail", ""]
    for c in cells:
        if c is baseline:
            continue
        hota_regr = _hota_regressions(baseline, c)
        frag_regr = _fragmentation_regressions(baseline, c)
        passed, reason = _evaluate_gate(baseline, c)

        swap_delta = c.same_team_swaps - base_sw
        reduction_str = f"{(base_sw - c.same_team_swaps) / base_sw:.1%}" if base_sw > 0 else "n/a"

        lines.append(f"### `{c.spec.name}`")
        lines.append("")
        lines.append(
            f"- Config: LEARNED_MERGE_VETO_COS={c.spec.learned_cos:.2f}, "
            f"ENABLE_COURT_VELOCITY_GATE={'1' if c.spec.enable_gate else '0'}"
            + (f", RALLYCUT_MAX_MERGE_VELOCITY_METERS={c.spec.max_vel_m:.1f}" if c.spec.enable_gate else "")
        )
        lines.append(
            f"- SAME_TEAM_SWAP: **{c.same_team_swaps}** "
            f"(baseline {base_sw}, Δ {swap_delta:+d}, reduction {reduction_str})"
        )
        lines.append(f"- NET_CROSSING: {c.net_crossing}")
        lines.append(f"- Total real switches: {c.total_real_switches}")
        lines.append(f"- Aggregate HOTA: {c.aggregate_hota * 100:.2f}% (baseline {base_hota * 100:.2f}%)")
        lines.append(f"- Elapsed: {c.elapsed_s:.1f}s")
        lines.append(f"- Gate: {'PASS' if passed else 'FAIL'} — {reason}")

        if hota_regr:
            lines.append(f"- HOTA regressions > 0.5 pp ({len(hota_regr)} rallies):")
            for rid, delta in hota_regr:
                lines.append(f"  - `{rid[:8]}` −{delta * 100:.2f} pp")
        else:
            lines.append("- HOTA regressions > 0.5 pp: **none**")

        if frag_regr:
            lines.append(f"- Fragmentation exceedances > 20% ({len(frag_regr)} rallies):")
            for rid, base_n, new_n in frag_regr:
                lines.append(
                    f"  - `{rid[:8]}` {base_n}→{new_n} unique pred-IDs "
                    f"(+{((new_n - base_n) / base_n) * 100:.0f}%)"
                )
        else:
            lines.append("- Fragmentation exceedances > 20%: **none**")

        lines.append("")

    # OR-logic sanity check: combined ≤ min(learned_only, velocity_only)
    lines += ["## OR-Logic Sanity Check", ""]
    learned_only = next((c for c in cells if c.spec.name == "learned_t080"), None)
    velocity_v25 = next((c for c in cells if c.spec.name == "velocity_v25"), None)
    velocity_v35 = next((c for c in cells if c.spec.name == "velocity_v35"), None)
    combined_070_v25 = next((c for c in cells if c.spec.name == "combined_070_v25"), None)
    combined_070_v35 = next((c for c in cells if c.spec.name == "combined_070_v35"), None)
    combined_080_v25 = next((c for c in cells if c.spec.name == "combined_080_v25"), None)
    combined_080_v35 = next((c for c in cells if c.spec.name == "combined_080_v35"), None)

    checks: list[tuple[str, int | None, int | None, int | None]] = []
    if learned_only and velocity_v25 and combined_080_v25:
        checks.append((
            "combined_080_v25 ≤ min(learned_t080, velocity_v25)",
            combined_080_v25.same_team_swaps,
            learned_only.same_team_swaps,
            velocity_v25.same_team_swaps,
        ))
    if learned_only and velocity_v35 and combined_080_v35:
        checks.append((
            "combined_080_v35 ≤ min(learned_t080, velocity_v35)",
            combined_080_v35.same_team_swaps,
            learned_only.same_team_swaps,
            velocity_v35.same_team_swaps,
        ))

    for label, combo, l_only, v_only in checks:
        if combo is not None and l_only is not None and v_only is not None:
            expected_max = min(l_only, v_only)
            ok = combo <= expected_max
            sym = "OK" if ok else "BUG"
            lines.append(
                f"- {label}: combo={combo}, learned={l_only}, vel={v_only} → "
                f"min={expected_max} [{sym}]"
            )
        else:
            lines.append(f"- {label}: data missing")

    lines.append("")

    # Knee recommendation
    lines += ["## Knee Recommendation", ""]
    gate_passers: list[CellResult] = []
    for c in cells:
        if c is baseline:
            continue
        passed, _ = _evaluate_gate(baseline, c)
        if passed:
            gate_passers.append(c)

    if gate_passers:
        # Pick the most conservative (lowest learned_cos, then lowest vel)
        def _conservativeness(c: CellResult) -> tuple[float, float]:
            return (c.spec.learned_cos, c.spec.max_vel_m if c.spec.enable_gate else 0.0)

        knee = min(gate_passers, key=_conservativeness)
        lines.append(
            f"**SHIP CANDIDATE: `{knee.spec.name}`** — clears all 3 measurable gate targets."
        )
        lines.append("")
        lines.append(
            f"Config: LEARNED_MERGE_VETO_COS={knee.spec.learned_cos:.2f}, "
            f"ENABLE_COURT_VELOCITY_GATE={'1' if knee.spec.enable_gate else '0'}"
            + (f", RALLYCUT_MAX_MERGE_VELOCITY_METERS={knee.spec.max_vel_m:.1f}" if knee.spec.enable_gate else "")
        )
        lines.append(f"SAME_TEAM_SWAPs: {baseline.same_team_swaps} → {knee.same_team_swaps} "
                     f"({(baseline.same_team_swaps - knee.same_team_swaps) / baseline.same_team_swaps:.0%} reduction)")
        lines.append(f"Aggregate HOTA delta: {(knee.aggregate_hota - base_hota) * 100:+.2f} pp")
        lines.append("")
        if len(gate_passers) > 1:
            lines.append("All gate-passing cells:")
            for c in gate_passers:
                lines.append(
                    f"- `{c.spec.name}`: swaps={c.same_team_swaps}, "
                    f"HOTA={c.aggregate_hota * 100:.2f}%"
                )
        lines.append("")
        lines.append(
            "**Note**: Gate 4 (`player_attribution_oracle`) is deferred to Task 9 — "
            "requires `match-players` + `reattribute-actions` re-run on new track IDs."
        )
    else:
        # Find best-case reduction
        best_cell = min(cells[1:], key=lambda c: c.same_team_swaps)
        best_reduction = (base_sw - best_cell.same_team_swaps) / base_sw if base_sw > 0 else 0.0
        lines.append(
            f"**NO SHIP.** No cell clears all measurable gate targets simultaneously."
        )
        lines.append("")
        lines.append(
            f"Best-case reduction: `{best_cell.spec.name}` with {best_cell.same_team_swaps} swaps "
            f"({best_reduction:.0%} reduction) — did not clear all gates."
        )
        lines.append("")
        lines.append("Next-step options:")
        lines.append(
            "- **3a**: Continue looser learned-only sweep (e.g. LEARNED_MERGE_VETO_COS ∈ {0.60, 0.65, 0.70}) "
            "— may find a threshold with fewer HOTA regressions."
        )
        lines.append(
            "- **3b**: Different signal stack — e.g. bbox-size continuity, pose-keypoint distance, "
            "per-frame appearance trajectory rather than median."
        )
        lines.append(
            "- **3c**: Accept ceiling per Session 7 — redirect effort to ball/action/score/court workstreams. "
            "12 swaps across 43 rallies is 4.4% of identities affected; HOTA is already 91.6%."
        )

    # Runtime
    lines += ["", "## Runtime", ""]
    total_s = sum(c.elapsed_s for c in cells)
    for c in cells:
        lines.append(
            f"- `{c.spec.name}`: {c.elapsed_s:.1f}s"
        )
    lines.append(f"- **Total**: {total_s:.0f}s ({total_s / 60:.1f} min)")

    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-only",
        action="store_true",
        help="Run only the baseline cell for sanity check before full sweep.",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Skip cells that already have tracking.json + audit/ directory.",
    )
    parser.add_argument(
        "--report-out",
        type=Path,
        default=OUT_DIR / "session8_plan_b_report.md",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    cells_to_run = [CELLS[0]] if args.baseline_only else CELLS

    results: list[CellResult] = []
    baseline: CellResult | None = None
    total_t0 = time.time()

    for i, spec in enumerate(cells_to_run):
        sweep_dir = SWEEP_DIR / spec.name
        sweep_dir.mkdir(parents=True, exist_ok=True)
        tracking_json = sweep_dir / "tracking.json"
        audit_dir = sweep_dir / "audit"

        if args.reuse_existing and tracking_json.exists() and audit_dir.exists() and any(audit_dir.glob("*.json")):
            logger.info("[%d/%d] reusing existing %s", i + 1, len(cells_to_run), spec.name)
            elapsed = 0.0
        else:
            elapsed = _run_cell(spec, sweep_dir)

        cell = _parse_cell(spec, sweep_dir)
        cell.elapsed_s = elapsed
        results.append(cell)

        if baseline is None:
            baseline = cell

        elapsed_total = time.time() - total_t0
        logger.info(
            "[%d/%d] cell=%-25s swaps=%2d  hota=%.2f%%  elapsed=%.1fs  total=%.1fmin",
            i + 1, len(cells_to_run),
            spec.name,
            cell.same_team_swaps,
            cell.aggregate_hota * 100,
            elapsed,
            elapsed_total / 60,
        )

        # Baseline sanity check — abort early if mismatch
        if baseline is cell:
            if cell.same_team_swaps != EXPECTED_BASELINE_SWAPS:
                logger.error(
                    "BASELINE SANITY FAIL: expected %d SAME_TEAM_SWAPs, got %d. "
                    "Audit parsing or env-var defaults are off. STOPPING.",
                    EXPECTED_BASELINE_SWAPS,
                    cell.same_team_swaps,
                )
                return 1
            else:
                logger.info(
                    "BASELINE SANITY OK: %d SAME_TEAM_SWAPs as expected.",
                    cell.same_team_swaps,
                )
            if args.baseline_only:
                logger.info("--baseline-only complete. Re-run without flag for full sweep.")
                return 0

    assert baseline is not None

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    _render_report(baseline, results, args.report_out)
    logger.info("wrote %s", args.report_out)
    print(args.report_out.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
