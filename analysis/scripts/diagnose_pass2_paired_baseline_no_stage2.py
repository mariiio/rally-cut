#!/usr/bin/env python3
"""Session 7b-pass2 — paired baseline + no_stage2 in ONE Python process.

Pre-work guidance (user): "OSNet / BoT-SORT per-run variance is real. Always
compare against an in-run baseline from the same invocation, never across
runs." My first two runs (attribution baseline 74 + no_stage2 79) were in
separate processes and showed an apparent net +5 Stage-2 FIX, directly
contradicting pre-work's 100 → 45 (−55) damage. Before calling NO-GO I
need a paired measurement in a single process: load the model once, iterate
the 51-video pool twice (baseline, then no_stage2 via monkey-patch), same
process, same random state.

Writes a single summary JSON with both runs and per-video deltas.
Read-only. ~56 min total.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_cross_team_birth import analyse_video  # noqa: E402
from diagnose_pass2_ablations import patch_pass2  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    real_cats: Counter[str] = Counter()
    per_video_real: dict[str, int] = {}
    for r in results:
        is_art = r["rally0_is_perm_artifact"]
        n_real = 0
        for b in r["births"]:
            if is_art and b["category"] == "rally_0":
                continue
            real_cats[b["category"]] += 1
            n_real += 1
        per_video_real[r["short"]] = n_real
    return {
        "total_real": sum(real_cats.values()),
        "per_category": dict(real_cats),
        "per_video_real": per_video_real,
    }


def _run_pass(label: str, targets: list[str], reid_model: Any) -> list[dict[str, Any]]:
    logger.info("\n%s\n# PASS: %s (%d videos)\n%s", "#" * 70, label, len(targets), "#" * 70)
    results: list[dict[str, Any]] = []
    with patch_pass2(label):
        for i, vid in enumerate(targets, start=1):
            logger.info("\n[%d/%d %s | %s] running...", i, len(targets), vid, label)
            r = analyse_video(vid, reid_model)
            if r is None:
                continue
            results.append(r)
            logger.info(
                "  rallies=%d  cross-team births=%d  switch_rallies=%s",
                r["n_rallies"], r["n_cross_team_births"], r["switch_rallies"],
            )
    return results


def main() -> None:
    from rallycut.evaluation.db import get_connection
    from rallycut.evaluation.gt_loader import load_all_from_db
    from rallycut.tracking.reid_general import (
        WEIGHTS_PATH as REID_WEIGHTS_PATH,
    )
    from rallycut.tracking.reid_general import GeneralReIDModel

    if not REID_WEIGHTS_PATH.exists():
        logger.error("OSNet weights not found at %s", REID_WEIGHTS_PATH)
        sys.exit(1)
    reid_model = GeneralReIDModel(weights_path=REID_WEIGHTS_PATH)
    logger.info("Loaded general ReID (OSNet SupCon)")

    with get_connection() as conn:
        with conn.cursor() as cur:
            rows = load_all_from_db(cur)
    targets = [r.video_id[:8] for r in rows if r.gt.rallies]
    logger.info("Loaded %d GT videos", len(targets))

    started = time.time()
    baseline_results = _run_pass("baseline", targets, reid_model)
    no_stage2_results = _run_pass("no_stage2", targets, reid_model)
    elapsed = time.time() - started

    base_agg = _aggregate(baseline_results)
    nos2_agg = _aggregate(no_stage2_results)

    logger.info("\n%s", "=" * 70)
    logger.info("PAIRED IN-RUN SUMMARY  (elapsed %.1fs)", elapsed)
    logger.info("=" * 70)
    logger.info("baseline  real=%d  per_category=%s",
                base_agg["total_real"], base_agg["per_category"])
    logger.info("no_stage2 real=%d  per_category=%s",
                nos2_agg["total_real"], nos2_agg["per_category"])
    logger.info("delta (base − no_stage2): %+d",
                base_agg["total_real"] - nos2_agg["total_real"])

    # Per-video delta
    all_shorts = sorted(set(base_agg["per_video_real"]) | set(nos2_agg["per_video_real"]))
    deltas: list[tuple[int, str, int, int]] = []
    for s in all_shorts:
        b = base_agg["per_video_real"].get(s, 0)
        n = nos2_agg["per_video_real"].get(s, 0)
        if b != n:
            deltas.append((b - n, s, b, n))
    deltas.sort()

    logger.info("\nPer-video delta (only non-zero):")
    logger.info("%10s  %4s %5s %6s", "short", "base", "noS2", "delta")
    n_damage = 0
    n_fix = 0
    dmg_total = 0
    fix_total = 0
    for d, s, b, n in deltas:
        tag = "damage" if d > 0 else "fix"
        logger.info("  %s  %4d %5d %+6d  %s", s, b, n, d, tag)
        if d > 0:
            n_damage += 1
            dmg_total += d
        else:
            n_fix += 1
            fix_total += -d
    logger.info(
        "\nDamage: %d videos (+%d births)  Fix: %d videos (-%d births)  Net: %+d",
        n_damage, dmg_total, n_fix, fix_total, dmg_total - fix_total,
    )

    out_dir = Path("outputs/cross_rally_errors")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d-%H%M%S")
    out_path = out_dir / f"pass2_paired_baseline_no_stage2_{ts}.json"
    with out_path.open("w") as fh:
        json.dump({
            "elapsed_sec": elapsed,
            "baseline": base_agg,
            "no_stage2": nos2_agg,
            "delta_total": base_agg["total_real"] - nos2_agg["total_real"],
            "per_video_damage": [
                {"short": s, "baseline": b, "no_stage2": n, "delta": d}
                for d, s, b, n in deltas if d > 0
            ],
            "per_video_fix": [
                {"short": s, "baseline": b, "no_stage2": n, "delta": d}
                for d, s, b, n in deltas if d < 0
            ],
            "baseline_per_video": [
                {"video_id": r["video_id"], "short": r["short"],
                 "births": [{k: v for k, v in b.items() if k != "prev_state"} for b in r["births"]],
                 "rally0_is_perm_artifact": r["rally0_is_perm_artifact"]}
                for r in baseline_results
            ],
            "no_stage2_per_video": [
                {"video_id": r["video_id"], "short": r["short"],
                 "births": [{k: v for k, v in b.items() if k != "prev_state"} for b in r["births"]],
                 "rally0_is_perm_artifact": r["rally0_is_perm_artifact"]}
                for r in no_stage2_results
            ],
        }, fh, indent=2)
    logger.info("\nWrote %s", out_path)


if __name__ == "__main__":
    main()
