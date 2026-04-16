"""Session 6 — sweep `LEARNED_MERGE_VETO_COS` across thresholds, measure
per-rally HOTA + SAME_TEAM_SWAP count, emit `session6_gate_report.md`.

The single-rally trace on ``fad29c31`` (HOTA 71.4% → 76.0% at cos≥0.90,
Real IDsw 1→0) established that the veto architecture works. This sweep
measures how the knee behaves across all 43 GT rallies.

Gate:
1. SAME_TEAM_SWAP count reduction ≥ 50% vs baseline (LEARNED_MERGE_VETO_COS=0).
2. No per-rally HOTA drop > 0.5 pp.
3. Fragmentation delta per rally ≤ 20%.
4. player_attribution_oracle — DEFERRED to Session 7 (requires re-attribution
   pipeline since retrack changes track IDs; DB mapping would be invalid).

Usage:
    uv run python scripts/eval_learned_merge_veto.py
    uv run python scripts/eval_learned_merge_veto.py --thresholds 0.0,0.85,0.90,0.95
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

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ANALYSIS_ROOT / "reports" / "merge_veto"

SWAP_REDUCTION_FLOOR = 0.50
PER_RALLY_HOTA_LIMIT = 0.005    # 0.5 pp
FRAGMENTATION_DELTA_LIMIT = 0.20  # 20 % per-rally max

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("eval_learned_merge_veto")


@dataclass
class CellResult:
    threshold: float
    tracking_json: Path
    audit_dir: Path
    elapsed_s: float = 0.0
    same_team_swaps: int = 0
    net_crossing: int = 0
    total_real_switches: int = 0
    per_rally_hota: dict[str, float] = field(default_factory=dict)
    per_rally_unique_pred_ids: dict[str, int] = field(default_factory=dict)


def _run_retrack(threshold: float, sweep_dir: Path) -> float:
    tracking_json = sweep_dir / "tracking.json"
    audit_dir = sweep_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, "LEARNED_MERGE_VETO_COS": f"{threshold:.3f}"}
    # Force the Session-4 cache with embeddings.
    env.setdefault("WEIGHT_LEARNED_REID", "0.05")
    cmd = [
        "uv", "run", "rallycut", "evaluate-tracking",
        "--all", "--retrack", "--cached",
        "--output", str(tracking_json),
        "--audit-out", str(audit_dir),
    ]
    logger.info("cell t=%.2f → %s", threshold, sweep_dir)
    t0 = time.time()
    subprocess.run(cmd, env=env, cwd=str(ANALYSIS_ROOT), check=True)
    return time.time() - t0


def _parse_cell(threshold: float, sweep_dir: Path) -> CellResult:
    tracking_json = sweep_dir / "tracking.json"
    audit_dir = sweep_dir / "audit"
    result = CellResult(
        threshold=threshold,
        tracking_json=tracking_json,
        audit_dir=audit_dir,
    )

    # Per-rally HOTA from tracking.json
    if tracking_json.exists():
        data = json.loads(tracking_json.read_text())
        for r in data.get("per_rally", []):
            rid = r.get("rally_id")
            h = r.get("hota") or 0.0
            if rid is not None:
                result.per_rally_hota[rid] = h

    # SAME_TEAM_SWAP counts + per-rally fragmentation (unique pred IDs)
    for p in sorted(audit_dir.glob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        rid = d.get("rallyId")
        # Count SAME_TEAM_SWAP and NET_CROSSING separately
        for g in d.get("perGt", []):
            for sw in g.get("realSwitches", []):
                cause = sw.get("cause")
                if cause == "same_team_swap":
                    result.same_team_swaps += 1
                elif cause == "net_crossing":
                    result.net_crossing += 1
                result.total_real_switches += 1
        # Unique pred IDs across all GT tracks
        pred_ids: set[int] = set()
        for g in d.get("perGt", []):
            for _, _, pid in g.get("predIdSpans", []):
                if pid >= 0:
                    pred_ids.add(pid)
        if rid is not None:
            result.per_rally_unique_pred_ids[rid] = len(pred_ids)
    return result


def _hota_regressions(
    baseline: CellResult, cell: CellResult, limit: float,
) -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    for rid, h_new in cell.per_rally_hota.items():
        h_base = baseline.per_rally_hota.get(rid)
        if h_base is None:
            continue
        delta = h_base - h_new
        if delta > limit:
            out.append((rid, delta))
    out.sort(key=lambda p: -p[1])
    return out


def _fragmentation_regressions(
    baseline: CellResult, cell: CellResult, limit: float,
) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    for rid, count in cell.per_rally_unique_pred_ids.items():
        base = baseline.per_rally_unique_pred_ids.get(rid)
        if base is None or base == 0:
            continue
        ratio = (count - base) / base
        if ratio > limit:
            out.append((rid, base, count))
    out.sort(key=lambda r: -(r[2] - r[1]) / r[1])
    return out


def _render_report(
    baseline: CellResult,
    cells: list[CellResult],
    path: Path,
) -> None:
    lines = [
        "# Session 6 — Learned-head merge-veto gate report",
        "",
        "Integration: `LEARNED_MERGE_VETO_COS` in `tracklet_link.link_tracklets_by_appearance` — "
        "for each candidate merge pair, block the merge when "
        "`cos(median_learned_emb_a, median_learned_emb_b) < threshold`. "
        "Abstains when < 5 embeddings per side.",
        "",
        "## Gate summary",
        "",
        "| Threshold | SAME_TEAM_SWAP | Δ vs baseline | Worst rally HOTA drop | Fragmentation-exceeded rallies | Gate |",
        "|----------:|--------------:|:-------------:|:---------------------:|:------------------------------:|:----:|",
    ]

    base_sw = baseline.same_team_swaps
    for c in cells:
        if c is baseline:
            lines.append(
                f"| **{c.threshold:.2f}** (baseline) | {c.same_team_swaps} | — | — | — | ctrl |"
            )
            continue
        reduction = (
            "n/a" if base_sw == 0 else f"{(1 - c.same_team_swaps / base_sw):+.1%}"
        )
        hota_regr = _hota_regressions(baseline, c, PER_RALLY_HOTA_LIMIT)
        worst_hota = hota_regr[0][1] * 100 if hota_regr else 0.0
        frag_regr = _fragmentation_regressions(baseline, c, FRAGMENTATION_DELTA_LIMIT)
        # Gate pass: all three
        sw_ok = base_sw > 0 and c.same_team_swaps / base_sw <= 1 - SWAP_REDUCTION_FLOOR
        hota_ok = len(hota_regr) == 0
        frag_ok = len(frag_regr) == 0
        passed = sw_ok and hota_ok and frag_ok
        sym = "✅" if passed else "❌"
        lines.append(
            f"| {c.threshold:.2f} | {c.same_team_swaps} | {reduction} | "
            f"{worst_hota:.2f} pp (on {len(hota_regr)} rally/rallies) | "
            f"{len(frag_regr)} | {sym} |"
        )

    lines += ["", "## Per-cell detail", ""]
    for c in cells:
        if c is baseline:
            continue
        hota_regr = _hota_regressions(baseline, c, PER_RALLY_HOTA_LIMIT)
        frag_regr = _fragmentation_regressions(baseline, c, FRAGMENTATION_DELTA_LIMIT)
        lines.append(f"### threshold = {c.threshold:.2f}")
        lines.append("")
        lines.append(
            f"- SAME_TEAM_SWAP: **{c.same_team_swaps}** "
            f"(baseline {base_sw}, "
            f"Δ {c.same_team_swaps - base_sw:+d})"
        )
        lines.append(f"- NET_CROSSING: {c.net_crossing}")
        lines.append(f"- Total real switches: {c.total_real_switches}")
        lines.append(f"- Retrack elapsed: {c.elapsed_s:.1f} s")
        if hota_regr:
            lines.append("- HOTA regressions > 0.5 pp:")
            for rid, delta in hota_regr:
                lines.append(f"  - `{rid[:8]}` −{delta * 100:.2f} pp")
        else:
            lines.append("- HOTA regressions > 0.5 pp: **none**")
        if frag_regr:
            lines.append("- Fragmentation exceedances > 20 %:")
            for rid, base_n, new_n in frag_regr:
                lines.append(
                    f"  - `{rid[:8]}` {base_n} → {new_n} unique pred-IDs "
                    f"(+{((new_n - base_n) / base_n) * 100:.0f} %)"
                )
        else:
            lines.append("- Fragmentation exceedances > 20 %: **none**")
        lines.append("")

    # Knee pick
    candidates = []
    for c in cells:
        if c is baseline:
            continue
        if base_sw == 0:
            continue
        sw_ok = c.same_team_swaps / base_sw <= 1 - SWAP_REDUCTION_FLOOR
        hota_ok = not _hota_regressions(baseline, c, PER_RALLY_HOTA_LIMIT)
        frag_ok = not _fragmentation_regressions(baseline, c, FRAGMENTATION_DELTA_LIMIT)
        if sw_ok and hota_ok and frag_ok:
            candidates.append(c.threshold)

    lines += ["", "## Recommendation", ""]
    if candidates:
        knee = min(candidates)
        lines.append(
            f"**SHIP at LEARNED_MERGE_VETO_COS = {knee:.2f}** — lowest "
            f"threshold clearing all three measurable gate targets "
            f"(swap reduction ≥ {int(SWAP_REDUCTION_FLOOR * 100)} %, "
            f"per-rally HOTA ≥ baseline − 0.5 pp, "
            f"fragmentation delta ≤ 20 %). **Gate target #4 "
            f"(`player_attribution_oracle`) deferred to Session 7** — "
            f"requires `match-players` + `reattribute-actions` to re-run on "
            f"the retracked track IDs before `production_eval` can be used "
            f"meaningfully."
        )
    else:
        lines.append(
            "**NO SHIP.** No threshold clears all measurable gate targets "
            "simultaneously. Review per-cell detail above — most likely "
            "either (a) swap reduction floor too tight for the head's "
            "signal, (b) HOTA regression on specific rallies indicates "
            "legit merges being rejected, or (c) fragmentation explosion "
            "on high thresholds. Iterate on threshold range, or combine "
            "with `ENABLE_COURT_VELOCITY_GATE=1` in a 2D sweep."
        )

    lines += ["", "## Runtime", ""]
    for c in cells:
        lines.append(f"- t={c.threshold:.2f}: {c.elapsed_s / 60:.1f} min")

    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--thresholds", type=str,
        default="0.0,0.80,0.85,0.88,0.90,0.92,0.95",
        help="Comma-separated thresholds (first must be 0.0 for baseline).",
    )
    parser.add_argument(
        "--report-out", type=Path,
        default=OUT_DIR / "session6_gate_report.md",
    )
    parser.add_argument(
        "--reuse-existing", action="store_true",
        help="Skip retracks for cells that already have tracking.json + audit/",
    )
    args = parser.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]
    if thresholds[0] != 0.0:
        logger.error("first threshold must be 0.0 (baseline)")
        return 1

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    cells: list[CellResult] = []
    baseline: CellResult | None = None

    for t in thresholds:
        sweep_dir = OUT_DIR / "sweep" / f"t{t:.2f}"
        sweep_dir.mkdir(parents=True, exist_ok=True)
        tracking_json = sweep_dir / "tracking.json"
        audit_dir = sweep_dir / "audit"

        if args.reuse_existing and tracking_json.exists() and audit_dir.exists():
            logger.info("reusing existing %s", sweep_dir)
            elapsed = 0.0
        else:
            elapsed = _run_retrack(t, sweep_dir)

        cell = _parse_cell(t, sweep_dir)
        cell.elapsed_s = elapsed
        cells.append(cell)
        if baseline is None:
            baseline = cell
        logger.info(
            "t=%.2f → SAME_TEAM_SWAP=%d, total_real_switches=%d",
            t, cell.same_team_swaps, cell.total_real_switches,
        )

    assert baseline is not None
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    _render_report(baseline, cells, args.report_out)
    logger.info("wrote %s", args.report_out)
    print(args.report_out.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
