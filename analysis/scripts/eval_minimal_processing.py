"""Session 7 — test whether the multi-stage post-processing chain is net-
positive for identity accuracy, or whether it over-processes and creates
the very swaps it's meant to repair.

Runs two cells on 43 GT rallies:

1. **baseline** — every post-YOLO merge/rename pass enabled (current
   production behaviour).
2. **minimal** — every merge/rename pass disabled via
   ``SKIP_ALL_MERGE_PASSES=1``. Only YOLO detection + filtering + team
   classification + canonical cross-rally remap (done later by
   ``match-players``) remains.

Both cells reuse the same pre-Session-4 raw cache (``WEIGHT_LEARNED_REID``
unset → default off), so the experiment runs in seconds.

Measurements:
- Per-rally HOTA (from tracking.json)
- SAME_TEAM_SWAP count (from audit JSONs)
- Fragmentation (unique pred IDs per rally)
- F1 / precision / recall (from tracking.json)
- Mostly Tracked / Mostly Lost (from tracking.json)

Writes ``reports/minimal_processing/session7_report.md`` with a three-
outcome decision: simplify, keep, or mixed.

Usage:
    uv run python scripts/eval_minimal_processing.py
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
OUT_DIR = ANALYSIS_ROOT / "reports" / "minimal_processing"

HOTA_FLAT_BAND = 0.005   # ± 0.5 pp considered "flat"
HOTA_MAJOR_DROP = 0.01   # ≥ 1 pp drop = merge passes doing real work

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("eval_minimal_processing")


@dataclass
class CellResult:
    label: str
    env_overrides: dict[str, str]
    tracking_json: Path
    audit_dir: Path
    elapsed_s: float = 0.0
    aggregate_hota: float = 0.0
    aggregate_f1: float = 0.0
    aggregate_idsw: int = 0
    same_team_swaps: int = 0
    net_crossing: int = 0
    total_real_switches: int = 0
    per_rally_hota: dict[str, float] = field(default_factory=dict)
    per_rally_unique_pred_ids: dict[str, int] = field(default_factory=dict)
    per_rally_f1: dict[str, float] = field(default_factory=dict)


def _run(cell: CellResult) -> None:
    cell.audit_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, **cell.env_overrides}
    cmd = [
        "uv", "run", "rallycut", "evaluate-tracking",
        "--all", "--retrack", "--cached",
        "--output", str(cell.tracking_json),
        "--audit-out", str(cell.audit_dir),
    ]
    logger.info("cell %s → %s", cell.label, cell.tracking_json)
    t0 = time.time()
    subprocess.run(cmd, env=env, cwd=str(ANALYSIS_ROOT), check=True)
    cell.elapsed_s = time.time() - t0


def _parse(cell: CellResult) -> None:
    if cell.tracking_json.exists():
        data = json.loads(cell.tracking_json.read_text())
        agg = data.get("aggregate") or {}
        # Aggregate HOTA is NOT in `aggregate` block; compute mean of
        # per-rally HOTAs to match the "Aggregate HOTA Metrics" console line.
        cell.aggregate_f1 = float(agg.get("f1") or 0.0)
        cell.aggregate_idsw = int(agg.get("idSwitches") or 0)
        hotas: list[float] = []
        for r in data.get("rallies", []):
            rid = r.get("rallyId")
            if rid is None:
                continue
            h = r.get("hota")
            if isinstance(h, dict):
                h = h.get("hota") or h.get("value")
            h_val = float(h or 0.0)
            cell.per_rally_hota[rid] = h_val
            hotas.append(h_val)
            r_agg = r.get("aggregate") or {}
            cell.per_rally_f1[rid] = float(r_agg.get("f1") or 0.0)
        if hotas:
            cell.aggregate_hota = sum(hotas) / len(hotas)

    for p in sorted(cell.audit_dir.glob("*.json")):
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
                    cell.same_team_swaps += 1
                elif cause == "net_crossing":
                    cell.net_crossing += 1
                cell.total_real_switches += 1
        pred_ids: set[int] = set()
        for g in d.get("perGt", []):
            for _, _, pid in g.get("predIdSpans", []):
                if pid >= 0:
                    pred_ids.add(pid)
        if rid is not None:
            cell.per_rally_unique_pred_ids[rid] = len(pred_ids)


def _verdict(baseline: CellResult, minimal: CellResult) -> tuple[str, str]:
    """Return (verdict_tag, rationale)."""
    delta_hota = baseline.aggregate_hota - minimal.aggregate_hota
    delta_swaps = baseline.same_team_swaps - minimal.same_team_swaps

    # Count per-rally winners/losers at HOTA 0.5 pp band
    losers = 0  # rallies where minimal is worse by > 0.5 pp
    winners = 0  # rallies where minimal is better by > 0.5 pp
    for rid, h_base in baseline.per_rally_hota.items():
        h_min = minimal.per_rally_hota.get(rid, h_base)
        delta = h_base - h_min
        if delta > HOTA_FLAT_BAND:
            losers += 1
        elif delta < -HOTA_FLAT_BAND:
            winners += 1

    flat_rallies = len(baseline.per_rally_hota) - losers - winners

    if abs(delta_hota) < HOTA_FLAT_BAND and delta_swaps >= baseline.same_team_swaps * 0.5:
        return (
            "SIMPLIFY",
            f"Aggregate HOTA stayed within ±{HOTA_FLAT_BAND * 100:.1f} pp "
            f"(Δ {delta_hota * 100:+.2f} pp) and minimal processing cut "
            f"SAME_TEAM_SWAP from {baseline.same_team_swaps} to "
            f"{minimal.same_team_swaps}. The merge chain costs more than "
            f"it buys — rip it out.",
        )
    if delta_hota > HOTA_MAJOR_DROP:
        return (
            "KEEP",
            f"Aggregate HOTA dropped {delta_hota * 100:.2f} pp under "
            f"minimal processing. The merge passes ARE doing real work. "
            f"Identity correction needs to happen INSIDE them (e.g. extend "
            f"Session-6 learned-head veto to all merge passes).",
        )
    return (
        "MIXED",
        f"HOTA Δ = {delta_hota * 100:+.2f} pp (aggregate), {winners} rallies "
        f"better / {losers} rallies worse / {flat_rallies} flat under "
        f"minimal processing. SAME_TEAM_SWAP "
        f"{baseline.same_team_swaps} → {minimal.same_team_swaps}. "
        f"Merge passes help some rallies, hurt others — per-rally policy "
        f"or targeted merge gating is the lever, not a blanket on/off.",
    )


def _render_report(baseline: CellResult, minimal: CellResult, path: Path) -> None:
    verdict_tag, rationale = _verdict(baseline, minimal)

    losers: list[tuple[str, float]] = []
    winners: list[tuple[str, float]] = []
    flat: list[tuple[str, float]] = []
    for rid, h_base in sorted(baseline.per_rally_hota.items()):
        h_min = minimal.per_rally_hota.get(rid, h_base)
        delta = h_base - h_min
        if delta > HOTA_FLAT_BAND:
            losers.append((rid, delta))
        elif delta < -HOTA_FLAT_BAND:
            winners.append((rid, -delta))
        else:
            flat.append((rid, delta))
    losers.sort(key=lambda p: -p[1])
    winners.sort(key=lambda p: -p[1])

    base_frag = sum(baseline.per_rally_unique_pred_ids.values())
    min_frag = sum(minimal.per_rally_unique_pred_ids.values())

    lines = [
        "# Session 7 — Minimal-processing A/B",
        "",
        "Tests whether the multi-stage post-YOLO merge/rename chain is net-"
        "positive or net-negative for identity accuracy. Two cells compared "
        "on 43 GT rallies.",
        "",
        "## Headline",
        "",
        "| Metric | Baseline (all passes on) | Minimal (`SKIP_ALL_MERGE_PASSES=1`) | Δ |",
        "|---|---:|---:|---:|",
        f"| Aggregate HOTA | {baseline.aggregate_hota * 100:.2f}% | "
        f"{minimal.aggregate_hota * 100:.2f}% | "
        f"{(minimal.aggregate_hota - baseline.aggregate_hota) * 100:+.2f} pp |",
        f"| Aggregate F1 | {baseline.aggregate_f1 * 100:.2f}% | "
        f"{minimal.aggregate_f1 * 100:.2f}% | "
        f"{(minimal.aggregate_f1 - baseline.aggregate_f1) * 100:+.2f} pp |",
        f"| Aggregate ID switches | {baseline.aggregate_idsw} | "
        f"{minimal.aggregate_idsw} | {minimal.aggregate_idsw - baseline.aggregate_idsw:+d} |",
        f"| SAME_TEAM_SWAP | {baseline.same_team_swaps} | "
        f"{minimal.same_team_swaps} | {minimal.same_team_swaps - baseline.same_team_swaps:+d} |",
        f"| NET_CROSSING | {baseline.net_crossing} | {minimal.net_crossing} | "
        f"{minimal.net_crossing - baseline.net_crossing:+d} |",
        f"| Total real switches | {baseline.total_real_switches} | "
        f"{minimal.total_real_switches} | "
        f"{minimal.total_real_switches - baseline.total_real_switches:+d} |",
        f"| Total unique pred-IDs (fragmentation) | {base_frag} | {min_frag} | {min_frag - base_frag:+d} |",
        "",
        f"## Verdict: **{verdict_tag}**",
        "",
        rationale,
        "",
        "## Per-rally HOTA breakdown",
        "",
        f"- Rallies UNCHANGED (|Δ| ≤ {HOTA_FLAT_BAND * 100:.1f} pp): **{len(flat)}**",
        f"- Rallies WORSE under minimal processing: **{len(losers)}**",
        f"- Rallies BETTER under minimal processing: **{len(winners)}**",
        "",
    ]
    if losers:
        lines += ["### Rallies where merge passes help (HOTA drops by > 0.5 pp under minimal)", ""]
        for rid, delta in losers[:10]:
            lines.append(f"- `{rid[:8]}`: {baseline.per_rally_hota[rid] * 100:.1f}% → "
                         f"{minimal.per_rally_hota.get(rid, 0) * 100:.1f}% "
                         f"(−{delta * 100:.2f} pp)")
        if len(losers) > 10:
            lines.append(f"- ... and {len(losers) - 10} more")
        lines.append("")
    if winners:
        lines += ["### Rallies where merge passes hurt (HOTA rises by > 0.5 pp under minimal)", ""]
        for rid, delta in winners[:10]:
            lines.append(f"- `{rid[:8]}`: {baseline.per_rally_hota[rid] * 100:.1f}% → "
                         f"{minimal.per_rally_hota.get(rid, 0) * 100:.1f}% "
                         f"(+{delta * 100:.2f} pp)")
        if len(winners) > 10:
            lines.append(f"- ... and {len(winners) - 10} more")
        lines.append("")

    lines += [
        "## Runtime",
        "",
        f"- Baseline: {baseline.elapsed_s:.1f} s",
        f"- Minimal: {minimal.elapsed_s:.1f} s",
    ]
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-out", type=Path,
        default=OUT_DIR / "session7_report.md",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sweep = OUT_DIR / "sweep"
    sweep.mkdir(parents=True, exist_ok=True)

    baseline = CellResult(
        label="baseline",
        env_overrides={},
        tracking_json=sweep / "baseline" / "tracking.json",
        audit_dir=sweep / "baseline" / "audit",
    )
    minimal = CellResult(
        label="minimal",
        env_overrides={"SKIP_ALL_MERGE_PASSES": "1"},
        tracking_json=sweep / "minimal" / "tracking.json",
        audit_dir=sweep / "minimal" / "audit",
    )

    for c in (baseline, minimal):
        _run(c)
        _parse(c)
        logger.info(
            "%s: HOTA=%.3f F1=%.3f SAME_TEAM_SWAP=%d elapsed=%.1fs",
            c.label, c.aggregate_hota, c.aggregate_f1,
            c.same_team_swaps, c.elapsed_s,
        )

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    _render_report(baseline, minimal, args.report_out)
    logger.info("wrote %s", args.report_out)
    print(args.report_out.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
