#!/usr/bin/env python3
"""Session 7b-pass2 pre-work — which Pass-2 stage(s) cause cross-team births?

The 2026-04-09 full-pool birth decomposition (cross_team_birth_2026-04-09-135625.json)
found 70 real cross-team births across the 51-video pool, with 79% (55/70)
clustering in mechanisms produced by `MatchPlayerTracker.refine_assignments`
(pair_swap 26 + solo_flip 22 + switch_rally 13). Three independent diagnostics
have already killed rally-0 anchoring as the lever.

This script ablates Pass-2 stages **without editing match_tracker.py** by
monkey-patching `MatchPlayerTracker` class methods for the duration of each
ablation run, then re-using `diagnose_cross_team_birth.analyse_video` to
recompute real cross-team births under the perm-artifact filter.

Pass 2 has three stages (see match_tracker.py:1495-1788):
  Stage 0 — `_detect_side_switches_combinatorial`
  Stage 1 — re-score loop inside `refine_assignments` (lines 1551-1590)
  Stage 2 — `_global_within_team_voting`

Ablations:
  baseline   — no patch (sanity reproduction of 70 baseline births)
  no_stage0  — Stage 0 returns []
  no_stage1  — refine_assignments runs Stage 0 + Stage 2 only
  no_stage2  — _global_within_team_voting is identity
  no_pass2   — refine_assignments is identity (Pass-2 ceiling)

Decision gate: a single ablation OR no_pass2 reducing real cross-team births
by ≥30 (≥43% of 70) → write follow-up Session 7b-pass2 fix. Otherwise the
cause is upstream of Pass 2 → reroute to a Pass 1 audit.

Read-only diagnostic. ~30-40 min per full ablation; backgrounded.
Use --focus-pair for the b03b461b + fb83f876 dry-run sanity check (~2 min).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterator

# Reuse the baseline diagnostic helpers
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_cross_team_birth import (  # noqa: E402
    DEFAULT_FOCUS,
    analyse_video,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

ABLATIONS = ["baseline", "no_stage0", "no_stage1", "no_stage2", "no_pass2"]


@contextlib.contextmanager
def patch_pass2(label: str) -> Iterator[None]:
    """Monkey-patch MatchPlayerTracker for the duration of an ablation run.

    Patches are applied at the class level so they take effect inside
    `match_players_across_rallies` without touching analyse_video.
    """
    from rallycut.tracking import match_tracker as mt

    cls = mt.MatchPlayerTracker
    saved: dict[str, Any] = {}

    def _save(name: str) -> None:
        saved[name] = cls.__dict__[name]

    if label == "baseline":
        yield
        return

    if label == "no_pass2":
        _save("refine_assignments")
        cls.refine_assignments = lambda self, initial_results: initial_results  # type: ignore[assignment]

    elif label == "no_stage0":
        _save("_detect_side_switches_combinatorial")
        cls._detect_side_switches_combinatorial = lambda self: []  # type: ignore[assignment]

    elif label == "no_stage2":
        _save("_global_within_team_voting")
        cls._global_within_team_voting = lambda self, results: results  # type: ignore[assignment]

    elif label == "no_stage1":
        # Reimplement refine_assignments verbatim minus the Stage-1 re-score loop.
        _save("refine_assignments")
        original_refine = cls.refine_assignments

        def refine_no_stage1(self, initial_results):  # type: ignore[no-untyped-def]
            if len(self.stored_rally_data) != len(initial_results):
                logger.warning(
                    "stored_rally_data length mismatch: %d vs %d results",
                    len(self.stored_rally_data),
                    len(initial_results),
                )
                return initial_results
            if len(initial_results) <= 1:
                return initial_results

            # Stage 0 (verbatim from match_tracker.py:1523-1546)
            switches = self._detect_side_switches_combinatorial()
            switch_set = set(switches)
            if switches:
                flipped = False
                for i, data in enumerate(self.stored_rally_data):
                    if i in switch_set:
                        flipped = not flipped
                    if flipped:
                        data.player_side_assignment = {
                            pid: (1 - team)
                            for pid, team in data.player_side_assignment.items()
                        }
                from rallycut.tracking.match_tracker import RallyTrackingResult
                for i in switches:
                    if i < len(initial_results):
                        r = initial_results[i]
                        initial_results[i] = RallyTrackingResult(
                            rally_index=r.rally_index,
                            track_to_player=r.track_to_player,
                            server_player_id=r.server_player_id,
                            side_switch_detected=True,
                            assignment_confidence=r.assignment_confidence,
                        )

            # Skip Stage 1; jump straight to Stage 2 with the unmodified
            # initial_results.
            return self._global_within_team_voting(initial_results)

        cls.refine_assignments = refine_no_stage1  # type: ignore[assignment]
        # Keep a reference so the linter doesn't elide it
        del original_refine

    else:
        raise ValueError(f"unknown ablation label: {label}")

    try:
        yield
    finally:
        for name, fn in saved.items():
            setattr(cls, name, fn)


def run_ablation(
    label: str,
    targets: list[str],
    reid_model: Any,
) -> dict[str, Any]:
    logger.info("\n" + "#" * 70)
    logger.info("# ABLATION: %s  (%d videos)", label, len(targets))
    logger.info("#" * 70)

    results: list[dict[str, Any]] = []
    started = time.time()
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
    elapsed = time.time() - started

    raw_cats: Counter[str] = Counter()
    real_cats: Counter[str] = Counter()
    n_artifact_videos = 0
    videos_by_mechanism: dict[str, list[str]] = defaultdict(list)
    for r in results:
        is_artifact = r["rally0_is_perm_artifact"]
        if is_artifact:
            n_artifact_videos += 1
        for b in r["births"]:
            raw_cats[b["category"]] += 1
            if is_artifact and b["category"] == "rally_0":
                continue
            real_cats[b["category"]] += 1
            videos_by_mechanism[b["category"]].append(r["short"])

    total_real = sum(real_cats.values())
    total_raw = sum(raw_cats.values())

    logger.info("\n" + "=" * 70)
    logger.info("ABLATION %s SUMMARY", label)
    logger.info("=" * 70)
    logger.info("Videos: %d  Elapsed: %.1fs", len(results), elapsed)
    logger.info("RAW total: %d", total_raw)
    logger.info("REAL total (perm-artifact filtered): %d", total_real)
    for cat in ("rally_0", "new_track", "switch_rally", "pair_swap", "solo_flip"):
        n = real_cats.get(cat, 0)
        pct = n / total_real * 100 if total_real else 0
        logger.info("  %-14s : %3d  (%.1f%%)", cat, n, pct)

    out_dir = Path("outputs/cross_rally_errors")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d-%H%M%S")
    out_path = out_dir / f"cross_team_birth_pass2_{label}_{ts}.json"
    with out_path.open("w") as fh:
        json.dump(
            {
                "ablation": label,
                "n_videos": len(results),
                "elapsed_sec": elapsed,
                "raw_aggregate": dict(raw_cats),
                "real_aggregate": dict(real_cats),
                "n_artifact_videos": n_artifact_videos,
                "videos_by_mechanism": {
                    k: sorted(set(v)) for k, v in videos_by_mechanism.items()
                },
                "per_video": results,
            },
            fh,
            indent=2,
        )
    logger.info("Wrote %s", out_path)

    return {
        "label": label,
        "n_videos": len(results),
        "total_real_births": total_real,
        "total_raw_births": total_raw,
        "per_category_real": dict(real_cats),
        "n_artifact_videos": n_artifact_videos,
        "json_path": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ablation", choices=ABLATIONS, default=None,
                        help="Run only one ablation. Default: run all 5 sequentially.")
    parser.add_argument("--video-id", type=str, default=None,
                        help="Single video ID prefix.")
    parser.add_argument("--focus-pair", action="store_true",
                        help="Run only the b03b461b + fb83f876 focus pair (dry run).")
    args = parser.parse_args()

    from rallycut.evaluation.db import get_connection
    from rallycut.evaluation.gt_loader import load_all_from_db
    from rallycut.tracking.reid_general import (
        WEIGHTS_PATH as REID_WEIGHTS_PATH,
    )
    from rallycut.tracking.reid_general import (
        GeneralReIDModel,
    )

    if not REID_WEIGHTS_PATH.exists():
        logger.error("OSNet weights not found at %s", REID_WEIGHTS_PATH)
        sys.exit(1)
    reid_model = GeneralReIDModel(weights_path=REID_WEIGHTS_PATH)
    logger.info("Loaded general ReID (OSNet SupCon)")

    if args.video_id:
        targets = [args.video_id]
    elif args.focus_pair:
        targets = list(DEFAULT_FOCUS)
    else:
        with get_connection() as conn:
            with conn.cursor() as cur:
                rows = load_all_from_db(cur)
        targets = [r.video_id[:8] for r in rows if r.gt.rallies]
        logger.info("Loaded %d GT videos from the pool", len(targets))

    ablations_to_run = [args.ablation] if args.ablation else ABLATIONS

    summaries: list[dict[str, Any]] = []
    for label in ablations_to_run:
        summary = run_ablation(label, targets, reid_model)
        summaries.append(summary)

    # Aggregate summary
    if len(summaries) > 1:
        baseline_total = next(
            (s["total_real_births"] for s in summaries if s["label"] == "baseline"),
            None,
        )
        logger.info("\n" + "=" * 70)
        logger.info("PASS-2 ABLATION SUMMARY  (baseline_real=%s)", baseline_total)
        logger.info("=" * 70)
        logger.info("%-12s %6s %8s  %s",
                    "label", "real", "delta", "per_category")
        for s in summaries:
            delta = (
                s["total_real_births"] - baseline_total
                if baseline_total is not None else None
            )
            cats = " ".join(
                f"{k}={v}" for k, v in sorted(s["per_category_real"].items())
            )
            logger.info(
                "%-12s %6d %+8s  %s",
                s["label"], s["total_real_births"],
                f"{delta:+d}" if delta is not None else "n/a",
                cats,
            )

        out_dir = Path("outputs/cross_rally_errors")
        ts = time.strftime("%Y-%m-%d-%H%M%S")
        summary_path = out_dir / f"pass2_ablation_summary_{ts}.json"
        with summary_path.open("w") as fh:
            json.dump(
                {
                    "baseline_real": baseline_total,
                    "n_videos": len(targets),
                    "ablations": summaries,
                },
                fh,
                indent=2,
            )
        logger.info("\nWrote %s", summary_path)


if __name__ == "__main__":
    main()
