"""Session 8 diagnostic — attribute each SAME_TEAM_SWAP to the pass that introduced it.

Runs evaluate-tracking 8 times with cumulative subsets of post-processing
passes enabled (K=0: none; K=1: only pass 0; …; K=7: all passes). For each
K, parses SAME_TEAM_SWAP counts per rally from the audit JSONs. Attribution
for pass K = (swaps_at_K - swaps_at_K-1), both aggregate and per-rally.

Writes reports/merge_veto/per_pass_swap_attribution.md.

Usage:
    uv run python scripts/diagnose_per_pass_swaps.py [--dry-run]

Options:
    --dry-run    Run only K=0 and K=7 for sanity-check before full sweep.
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
SWEEP_DIR = OUT_DIR / "per_pass_sweep"

# 7 passes in execution order: (env-flag-suffix, pipeline-step-label)
PASSES: list[tuple[str, str]] = [
    ("ENFORCE_SPATIAL_CONSISTENCY", "0"),
    ("FIX_HEIGHT_SWAPS", "0a"),
    ("SPLIT_TRACKS_BY_COLOR", "0b"),
    ("RELINK_SPATIAL_SPLITS", "0b2"),
    ("RELINK_PRIMARY_FRAGMENTS", "0b3"),
    ("LINK_TRACKLETS_BY_APPEARANCE", "0c"),
    ("STABILIZE_TRACK_IDS", "1"),
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("diagnose_per_pass_swaps")


@dataclass
class CellResult:
    """Result for one K-cell (passes 0..K-1 enabled)."""

    passes_enabled: int  # K: number of passes enabled (0..7)
    tracking_json: Path
    audit_dir: Path
    elapsed_s: float = 0.0
    aggregate_swaps: int = 0
    # per_rally_swaps: rallyId -> list of (gt_id, swap_frame) tuples
    per_rally_swaps: dict[str, list[tuple[str, int]]] = field(default_factory=dict)


def _build_env(passes_enabled: int) -> dict[str, str]:
    """Build env dict: enable passes 0..K-1, skip passes K..6."""
    env = dict(os.environ)
    for pass_idx, (suffix, _step) in enumerate(PASSES):
        flag = f"SKIP_{suffix}"
        if pass_idx >= passes_enabled:
            env[flag] = "1"
        else:
            env.pop(flag, None)  # unset = enabled
    # Ensure SKIP_ALL_MERGE_PASSES is not set (individual flags take precedence,
    # but belt-and-suspenders: remove it so _skip() uses per-flag values)
    env.pop("SKIP_ALL_MERGE_PASSES", None)
    return env


def _run(cell: CellResult) -> None:
    cell.audit_dir.mkdir(parents=True, exist_ok=True)
    cell.tracking_json.parent.mkdir(parents=True, exist_ok=True)
    env = _build_env(cell.passes_enabled)

    enabled_names = [PASSES[i][0] for i in range(cell.passes_enabled)]
    skipped_names = [PASSES[i][0] for i in range(cell.passes_enabled, len(PASSES))]
    logger.info(
        "[K=%d] passes_enabled=%d  enabled=%s  skipped=%s",
        cell.passes_enabled,
        cell.passes_enabled,
        enabled_names or ["<none>"],
        skipped_names or ["<none>"],
    )

    cmd = [
        "uv", "run", "rallycut", "evaluate-tracking",
        "--all", "--retrack", "--cached",
        "--output", str(cell.tracking_json),
        "--audit-out", str(cell.audit_dir),
    ]
    t0 = time.time()
    subprocess.run(cmd, env=env, cwd=str(ANALYSIS_ROOT), check=True)
    cell.elapsed_s = time.time() - t0
    logger.info("[K=%d] finished in %.1fs", cell.passes_enabled, cell.elapsed_s)


def _parse(cell: CellResult) -> None:
    """Parse SAME_TEAM_SWAP events from audit JSONs."""
    for p in sorted(cell.audit_dir.glob("*.json")):
        if p.name.startswith("_"):
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        rid = d.get("rallyId")
        if rid is None:
            continue
        swaps: list[tuple[str, int]] = []
        for g in d.get("perGt", []):
            gt_id = g.get("gtId", "")
            for sw in g.get("realSwitches", []):
                if sw.get("cause") == "same_team_swap":
                    frame = int(sw.get("frame", sw.get("switchFrame", -1)))
                    swaps.append((gt_id, frame))
        if swaps:
            cell.per_rally_swaps[rid] = swaps
        elif rid not in cell.per_rally_swaps:
            cell.per_rally_swaps[rid] = []
    cell.aggregate_swaps = sum(len(v) for v in cell.per_rally_swaps.values())


def _sanity_check(cells: list[CellResult]) -> bool:
    """Return True if K=0 and K=7 (first and last) look plausible."""
    k0 = cells[0]
    k7 = cells[-1]
    ok = True
    if k0.aggregate_swaps > 4:
        logger.warning(
            "SANITY FAIL: K=0 (no passes) has %d SAME_TEAM_SWAPs — expected ≤2 "
            "(raw BoT-SORT). Something is wrong with the SKIP flags or cache.",
            k0.aggregate_swaps,
        )
        ok = False
    else:
        logger.info("SANITY OK: K=0 → %d swaps (expected ~0)", k0.aggregate_swaps)

    if k7.aggregate_swaps < 8:
        logger.warning(
            "SANITY FAIL: K=7 (all passes) has only %d SAME_TEAM_SWAPs — expected "
            "~12 (current baseline). Something may be off.",
            k7.aggregate_swaps,
        )
        ok = False
    else:
        logger.info("SANITY OK: K=7 → %d swaps (expected ~12)", k7.aggregate_swaps)

    return ok


def _attribute_swaps(
    cells: list[CellResult],
) -> tuple[
    dict[str, int],                             # pass_name → delta_swaps
    list[tuple[str, str, int, str]],            # (rally_id, gt_id, frame, pass_name)
]:
    """Attribution logic.

    For each (gt_id, frame) swap present at cell K but not K-1, it was
    introduced by pass K-1 (0-indexed). We use a frozenset key of
    (rally_id, gt_id, frame) to track individual events across cells.
    """
    pass_attribution: dict[str, int] = {name: 0 for name, _ in PASSES}
    per_swap_attribution: list[tuple[str, str, int, str]] = []

    # Build set of swap keys per cell
    def _swap_keys(cell: CellResult) -> set[tuple[str, str, int]]:
        keys: set[tuple[str, str, int]] = set()
        for rid, swaps in cell.per_rally_swaps.items():
            for gt_id, frame in swaps:
                keys.add((rid, gt_id, frame))
        return keys

    prev_keys: set[tuple[str, str, int]] = set()
    for k, cell in enumerate(cells):
        curr_keys = _swap_keys(cell)
        if k == 0:
            # K=0 swaps have no preceding pass — attribute to "raw" (shouldn't exist)
            for rid, gt_id, frame in curr_keys:
                per_swap_attribution.append((rid, gt_id, frame, "<raw_botsort>"))
        else:
            new_keys = curr_keys - prev_keys
            removed_keys = prev_keys - curr_keys
            pass_name = PASSES[k - 1][0]  # pass K-1 was just enabled
            for rid, gt_id, frame in sorted(new_keys):
                pass_attribution[pass_name] += 1
                per_swap_attribution.append((rid, gt_id, frame, pass_name))
            if removed_keys:
                logger.info(
                    "  K=%d (%s): %d swaps REMOVED by this pass: %s",
                    k, pass_name, len(removed_keys),
                    sorted(removed_keys)[:5],
                )
        prev_keys = curr_keys

    return pass_attribution, per_swap_attribution


def _render_report(
    cells: list[CellResult],
    pass_attribution: dict[str, int],
    per_swap_attribution: list[tuple[str, str, int, str]],
    path: Path,
) -> None:
    lines = [
        "# Session 8 — Per-Pass SAME_TEAM_SWAP Attribution",
        "",
        "Incremental-skip diagnostic: runs `evaluate-tracking` 8 times with cumulative",
        "subsets of post-processing passes enabled. Attribution for pass K = swaps at K",
        "minus swaps at K-1.",
        "",
        "## 8-Cell Summary",
        "",
        "| K | Passes Enabled | Pass Just Added | Aggregate SAME_TEAM_SWAPs | Δ vs prev |",
        "|---|---|---|---:|---:|",
    ]
    for k, cell in enumerate(cells):
        pass_just_added = PASSES[k - 1][0] if k > 0 else "—"
        step_label = PASSES[k - 1][1] if k > 0 else "—"
        if k > 0:
            delta = cell.aggregate_swaps - cells[k - 1].aggregate_swaps
            delta_str = f"{delta:+d}"
            pass_label = f"`{pass_just_added}` (step {step_label})"
        else:
            delta_str = "—"
            pass_label = "— (raw BoT-SORT)"
        lines.append(
            f"| {k} | {cell.passes_enabled} | {pass_label} | "
            f"{cell.aggregate_swaps} | {delta_str} |"
        )

    lines += [
        "",
        "## Per-Pass Attribution",
        "",
        "| Pass | Step | Swaps Introduced | Verdict |",
        "|---|---|---:|---|",
    ]
    for pass_name, step in PASSES:
        n = pass_attribution.get(pass_name, 0)
        verdict = "**SWAP CREATOR**" if n > 0 else "innocent"
        lines.append(f"| `{pass_name}` | {step} | {n} | {verdict} |")

    # Also check for raw (K=0) swaps
    raw_swaps = sum(1 for _, _, _, p in per_swap_attribution if p == "<raw_botsort>")
    if raw_swaps:
        lines.append(f"| `<raw_botsort>` | — | {raw_swaps} | **unexpected** |")

    lines += [
        "",
        "## Per-Rally Swap Detail",
        "",
        "| Rally ID | GT ID | Frame | Introduced By |",
        "|---|---|---:|---|",
    ]
    for rid, gt_id, frame, pass_name in sorted(per_swap_attribution):
        lines.append(f"| `{rid[:8]}` | `{gt_id[:12] if gt_id else '—'}` | {frame} | `{pass_name}` |")

    # Compute which passes need adapters (Tasks 3-6 mapping)
    # Task 3 = FIX_HEIGHT_SWAPS, Task 4 = RELINK_SPATIAL_SPLITS,
    # Task 5 = RELINK_PRIMARY_FRAGMENTS, Task 6 = STABILIZE_TRACK_IDS
    # Session 6 already covered LINK_TRACKLETS_BY_APPEARANCE
    task_map = {
        "FIX_HEIGHT_SWAPS": "Task 3",
        "RELINK_SPATIAL_SPLITS": "Task 4",
        "RELINK_PRIMARY_FRAGMENTS": "Task 5",
        "STABILIZE_TRACK_IDS": "Task 6",
        "LINK_TRACKLETS_BY_APPEARANCE": "Session 6 (already implemented)",
        "ENFORCE_SPATIAL_CONSISTENCY": "No task assigned",
        "SPLIT_TRACKS_BY_COLOR": "No task assigned",
    }

    lines += [
        "",
        "## Decision Gate — Which Adapter Tasks to Execute",
        "",
        "Based on the attribution above:",
        "",
    ]
    for pass_name, step in PASSES:
        n = pass_attribution.get(pass_name, 0)
        task = task_map.get(pass_name, "No task assigned")
        if n > 0:
            lines.append(
                f"- **`{pass_name}`** (step {step}): **{n} swap(s) introduced** → "
                f"**EXECUTE {task}**"
            )
        else:
            lines.append(
                f"- `{pass_name}` (step {step}): 0 swaps → skip / low priority "
                f"({task})"
            )

    lines += [
        "",
        "## Runtime",
        "",
        f"{'K':>2} | {'Pass Added':<35} | {'Elapsed':>8}",
        f"{'--':>2} | {'----------':<35} | {'-------':>8}",
    ]
    for k, cell in enumerate(cells):
        pass_just_added = PASSES[k - 1][0] if k > 0 else "<none>"
        lines.append(f"{k:>2} | {pass_just_added:<35} | {cell.elapsed_s:>7.1f}s")

    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run only K=0 and K=7 for sanity-check before full sweep.",
    )
    parser.add_argument(
        "--report-out", type=Path,
        default=OUT_DIR / "per_pass_swap_attribution.md",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    # Build all 8 cells (K=0..7)
    all_cells = [
        CellResult(
            passes_enabled=k,
            tracking_json=SWEEP_DIR / f"k{k}" / "tracking.json",
            audit_dir=SWEEP_DIR / f"k{k}" / "audit",
        )
        for k in range(8)
    ]

    if args.dry_run:
        logger.info("DRY-RUN: running only K=0 and K=7")
        dry_cells = [all_cells[0], all_cells[7]]
        for cell in dry_cells:
            _run(cell)
            _parse(cell)
            logger.info(
                "[K=%d] SAME_TEAM_SWAP=%d  elapsed=%.1fs",
                cell.passes_enabled, cell.aggregate_swaps, cell.elapsed_s,
            )
        _sanity_check(dry_cells)
        logger.info("Dry-run complete. Re-run without --dry-run for full sweep.")
        return 0

    # Full 8-cell sweep
    total_t0 = time.time()
    for k, cell in enumerate(all_cells):
        _run(cell)
        _parse(cell)
        prev_swaps = all_cells[k - 1].aggregate_swaps if k > 0 else 0
        delta = cell.aggregate_swaps - prev_swaps
        pass_just_added = PASSES[k - 1][0] if k > 0 else "<none>"
        elapsed_min = (time.time() - total_t0) / 60
        logger.info(
            "[%d/8] K=%d  pass_added=%-35s  swaps=%2d  Δ=%+d  total_elapsed=%.1f min",
            k + 1, k, pass_just_added, cell.aggregate_swaps, delta, elapsed_min,
        )

    total_elapsed = time.time() - total_t0
    logger.info("All 8 cells done in %.1f min", total_elapsed / 60)

    # Sanity check
    if not _sanity_check(all_cells):
        logger.error(
            "Sanity check FAILED — results may be unreliable. "
            "Inspect the SKIP flag wiring and cache behavior before trusting the report."
        )
        # Still emit the report for debugging
    else:
        logger.info("Sanity check PASSED")

    pass_attribution, per_swap_attribution = _attribute_swaps(all_cells)

    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    _render_report(all_cells, pass_attribution, per_swap_attribution, args.report_out)
    logger.info("wrote %s", args.report_out)
    print(args.report_out.read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
