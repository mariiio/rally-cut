#!/usr/bin/env python3
"""Session 7b-pass2 — per-birth attribution of Stage 2 (_global_within_team_voting).

Pre-work (pass2_ablation_summary_2026-04-09-174026.json) showed disabling
Stage 2 cuts real cross-team births 100 → 45 on the 51-video GT pool.
This script attributes each of the 100 baseline cross-team births to one of:

  * sub_step_A — voting loop swap: a track was flipped by the apply-swaps
                 block (match_tracker.py:1758-1778), and the flip was ALREADY
                 in the pre-orientation-flip label vector. The voting
                 iteration chose the swap on its own.
  * sub_step_B — orientation flip: the apply-swap was entirely driven by the
                 global `cost_flipped < cost_current` inversion
                 (match_tracker.py:1749-1751). Would not have fired without
                 the orientation flip.
  * sub_step_both — orientation flip fired AND the per-rally label also
                    toggled, i.e., both sub-steps conspired.
  * neither — track was not swapped by Stage 2; the birth escaped via some
              other Pass-2 path. Part of the 45-floor. Bounds the ceiling
              of any Stage-2 gate.

Implementation: monkey-patch `MatchPlayerTracker._global_within_team_voting`
with an instrumented copy that logs per-team state (rally_pairs,
label_preflip, orientation_flip_fired, label_final) onto the instance as
`_stage2_attribution`. Then re-use `diagnose_cross_team_birth.analyse_video`
to get the baseline birth list and correlate each birth's (rally_index, tid)
against the logged apply-swaps.

Read-only. ~1 hr on full pool (same as baseline). Background it.
Output: outputs/cross_rally_errors/pass2_stage2_attribution_<ts>.json
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_cross_team_birth import DEFAULT_FOCUS, analyse_video  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def _make_instrumented_voting():  # type: ignore[no-untyped-def]
    """Return an instrumented replacement for
    MatchPlayerTracker._global_within_team_voting.

    Mirrors the real method body but captures per-team:
      - rally_pairs: list of (rally_index, t_lo, t_hi)
      - label_preflip: np.ndarray (the vote-loop output before orientation)
      - orientation_flipped: bool
      - label_final: np.ndarray (labels actually used for apply-swaps)
      - applied_swaps: list of (rally_index, t_lo, t_hi)
    stored on self._stage2_attribution.
    """
    from rallycut.tracking.match_tracker import (
        REID_BLEND,
        RallyTrackingResult,
        compute_appearance_similarity,
        compute_track_similarity,
    )

    def _global_within_team_voting_instrumented(self, results):  # type: ignore[no-untyped-def]
        log: list[dict[str, Any]] = []
        self._stage2_attribution = log

        if len(results) < 3:
            return results

        for team in [0, 1]:
            team_player_ids = sorted(
                pid for pid, t in self.state.current_side_assignment.items()
                if t == team
            )
            if len(team_player_ids) != 2:
                continue
            p_lo, p_hi = team_player_ids

            rally_pairs: list[tuple[int, int, int]] = []
            for i, (data, result) in enumerate(
                zip(self.stored_rally_data, results)
            ):
                t_lo = None
                t_hi = None
                for tid, pid in result.track_to_player.items():
                    if pid == p_lo:
                        t_lo = tid
                    elif pid == p_hi:
                        t_hi = tid
                if t_lo is not None and t_hi is not None:
                    if t_lo in data.track_stats and t_hi in data.track_stats:
                        rally_pairs.append((i, t_lo, t_hi))

            if len(rally_pairs) < 3:
                continue

            n = len(rally_pairs)
            preference = np.zeros((n, n))

            for a in range(n):
                ri_a, t_lo_a, t_hi_a = rally_pairs[a]
                stats_lo_a = self.stored_rally_data[ri_a].track_stats[t_lo_a]
                stats_hi_a = self.stored_rally_data[ri_a].track_stats[t_hi_a]

                for b in range(a + 1, n):
                    ri_b, t_lo_b, t_hi_b = rally_pairs[b]
                    stats_lo_b = self.stored_rally_data[ri_b].track_stats[t_lo_b]
                    stats_hi_b = self.stored_rally_data[ri_b].track_stats[t_hi_b]

                    _rb = REID_BLEND if (
                        stats_lo_a.reid_embedding is not None
                        or stats_lo_b.reid_embedding is not None
                    ) else 0.0

                    same_cost = (
                        compute_track_similarity(stats_lo_a, stats_lo_b, _rb)
                        + compute_track_similarity(stats_hi_a, stats_hi_b, _rb)
                    )
                    swap_cost = (
                        compute_track_similarity(stats_lo_a, stats_hi_b, _rb)
                        + compute_track_similarity(stats_hi_a, stats_lo_b, _rb)
                    )
                    pref = swap_cost - same_cost
                    preference[a, b] = pref
                    preference[b, a] = pref

            labels = np.zeros(n, dtype=int)
            for _iteration in range(10):
                changed = False
                for k in range(n):
                    score = 0.0
                    for j in range(n):
                        if j == k:
                            continue
                        p = preference[k, j]
                        if labels[j] == 1:
                            p = -p
                        score += p
                    new_label = 0 if score >= 0 else 1
                    if new_label != labels[k]:
                        labels[k] = new_label
                        changed = True
                if not changed:
                    break

            label_preflip = labels.copy()

            cost_current = 0.0
            cost_flipped = 0.0
            for idx in range(n):
                ri, t_lo, t_hi = rally_pairs[idx]
                data = self.stored_rally_data[ri]
                if labels[idx] == 0:
                    c_lo, c_hi = t_lo, t_hi
                else:
                    c_lo, c_hi = t_hi, t_lo
                if p_lo in self.state.players and p_hi in self.state.players:
                    cost_current += (
                        compute_appearance_similarity(
                            self.state.players[p_lo], data.track_stats[c_lo]
                        )
                        + compute_appearance_similarity(
                            self.state.players[p_hi], data.track_stats[c_hi]
                        )
                    )
                    cost_flipped += (
                        compute_appearance_similarity(
                            self.state.players[p_hi], data.track_stats[c_lo]
                        )
                        + compute_appearance_similarity(
                            self.state.players[p_lo], data.track_stats[c_hi]
                        )
                    )

            orientation_flipped = cost_flipped < cost_current
            if orientation_flipped:
                labels = 1 - labels

            applied: list[tuple[int, int, int]] = []
            for idx in range(n):
                if labels[idx] == 1:
                    ri, t_lo, t_hi = rally_pairs[idx]
                    result = results[ri]
                    new_t2p = dict(result.track_to_player)
                    new_t2p[t_lo] = p_hi
                    new_t2p[t_hi] = p_lo
                    results[ri] = RallyTrackingResult(
                        rally_index=result.rally_index,
                        track_to_player=new_t2p,
                        server_player_id=result.server_player_id,
                        side_switch_detected=result.side_switch_detected,
                        assignment_confidence=result.assignment_confidence,
                    )
                    applied.append((ri, t_lo, t_hi))

            log.append({
                "team": team,
                "p_lo": p_lo,
                "p_hi": p_hi,
                "rally_pairs": rally_pairs,
                "label_preflip": label_preflip.tolist(),
                "label_final": labels.tolist(),
                "orientation_flipped": bool(orientation_flipped),
                "cost_current": float(cost_current),
                "cost_flipped": float(cost_flipped),
                "applied_swaps": applied,
            })

        return results

    return _global_within_team_voting_instrumented


@contextlib.contextmanager
def instrument_stage2() -> Iterator[None]:
    from rallycut.tracking import match_tracker as mt

    cls = mt.MatchPlayerTracker
    original = cls.__dict__["_global_within_team_voting"]
    cls._global_within_team_voting = _make_instrumented_voting()  # type: ignore[method-assign]
    try:
        yield
    finally:
        cls._global_within_team_voting = original  # type: ignore[method-assign]


def attribute_birth(
    birth: dict[str, Any],
    stage2_log: list[dict[str, Any]],
) -> str:
    """Classify a birth's attribution against the instrumented Stage-2 log.

    Returns one of: sub_step_A, sub_step_B, sub_step_both, neither.
    """
    rally_idx = birth["rally_index"]
    tid = birth["tid"]
    try:
        tid_int = int(tid)
    except (ValueError, TypeError):
        tid_int = tid  # match_tracker track IDs are int; fall back if not

    for team_entry in stage2_log:
        applied = team_entry["applied_swaps"]
        pairs = team_entry["rally_pairs"]
        preflip = team_entry["label_preflip"]
        final = team_entry["label_final"]
        flipped = team_entry["orientation_flipped"]

        for (ri, t_lo, t_hi) in applied:
            if ri != rally_idx:
                continue
            if t_lo != tid_int and t_hi != tid_int:
                continue

            # Find the index in rally_pairs for (ri, t_lo, t_hi).
            idx = None
            for k, (ri_k, tlo_k, thi_k) in enumerate(pairs):
                if ri_k == ri and tlo_k == t_lo and thi_k == t_hi:
                    idx = k
                    break
            if idx is None:
                return "sub_step_A"  # should not happen, default to A

            pre = preflip[idx]
            post = final[idx]
            # post must be 1 since it was applied.
            if not flipped:
                return "sub_step_A"  # pure voting decision
            if pre == 1 and post == 1:
                # flip inverted to 0, but applied=1 means another flip? impossible
                return "sub_step_both"
            if pre == 0 and post == 1:
                # label only became 1 due to orientation flip
                return "sub_step_B"
            return "sub_step_A"

    return "neither"


def run(targets: list[str], reid_model: Any) -> dict[str, Any]:
    per_video: list[dict[str, Any]] = []
    attribution_counter: Counter[str] = Counter()
    per_cat_attribution: dict[str, Counter[str]] = defaultdict(Counter)
    total_births = 0
    started = time.time()

    with instrument_stage2():
        for i, vid in enumerate(targets, start=1):
            logger.info("\n[%d/%d %s] running...", i, len(targets), vid)
            # The instrument_stage2 context manager has installed the
            # instrumented voting method on the class. We wrap it once more
            # to stash the per-tracker attribution log onto a class-level
            # slot so we can retrieve it after analyse_video returns.
            import rallycut.tracking.match_tracker as mt
            mt.MatchPlayerTracker._latest_stage2_log = None  # type: ignore[attr-defined]
            instrumented_voting = mt.MatchPlayerTracker._global_within_team_voting

            def _voting_stashing(self, results, _inner=instrumented_voting):  # type: ignore[no-untyped-def]
                out = _inner(self, results)
                mt.MatchPlayerTracker._latest_stage2_log = getattr(
                    self, "_stage2_attribution", []
                )
                return out

            mt.MatchPlayerTracker._global_within_team_voting = _voting_stashing  # type: ignore[method-assign]
            try:
                r = analyse_video(vid, reid_model)
            finally:
                mt.MatchPlayerTracker._global_within_team_voting = instrumented_voting  # type: ignore[method-assign]

            if r is None:
                continue

            stage2_log = mt.MatchPlayerTracker._latest_stage2_log or []  # type: ignore[attr-defined]

            is_artifact = r["rally0_is_perm_artifact"]
            births_attr: list[dict[str, Any]] = []
            for b in r["births"]:
                if is_artifact and b["category"] == "rally_0":
                    continue  # filtered perm artifact (matches pre-work)
                attr = attribute_birth(b, stage2_log)
                attribution_counter[attr] += 1
                per_cat_attribution[b["category"]][attr] += 1
                births_attr.append({**b, "attribution": attr})
                total_births += 1

            per_video.append({
                "video_id": r["video_id"],
                "short": r["short"],
                "n_rallies": r["n_rallies"],
                "rally0_is_perm_artifact": is_artifact,
                "births": births_attr,
                "stage2_log": [
                    {**e, "label_preflip": list(e["label_preflip"]),
                     "label_final": list(e["label_final"])}
                    for e in stage2_log
                ],
            })

            logger.info(
                "  real births=%d  attributions so far: %s",
                len(births_attr), dict(attribution_counter),
            )

    elapsed = time.time() - started

    logger.info("\n" + "=" * 70)
    logger.info("STAGE-2 ATTRIBUTION SUMMARY (%d videos, %.1fs)",
                len(per_video), elapsed)
    logger.info("=" * 70)
    logger.info("Total real births: %d", total_births)
    for attr in ("sub_step_A", "sub_step_B", "sub_step_both", "neither"):
        n = attribution_counter.get(attr, 0)
        pct = n / total_births * 100 if total_births else 0
        logger.info("  %-14s : %3d  (%.1f%%)", attr, n, pct)

    logger.info("\nPer-category breakdown:")
    logger.info("  %-14s %6s %6s %8s %8s",
                "category", "A", "B", "both", "neither")
    for cat in ("rally_0", "new_track", "switch_rally", "pair_swap", "solo_flip"):
        c = per_cat_attribution.get(cat, Counter())
        logger.info("  %-14s %6d %6d %8d %8d",
                    cat, c.get("sub_step_A", 0), c.get("sub_step_B", 0),
                    c.get("sub_step_both", 0), c.get("neither", 0))

    out_dir = Path("outputs/cross_rally_errors")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d-%H%M%S")
    out_path = out_dir / f"pass2_stage2_attribution_{ts}.json"
    with out_path.open("w") as fh:
        json.dump({
            "n_videos": len(per_video),
            "total_real_births": total_births,
            "attribution_aggregate": dict(attribution_counter),
            "per_category_attribution": {
                k: dict(v) for k, v in per_cat_attribution.items()
            },
            "per_video": per_video,
            "elapsed_sec": elapsed,
        }, fh, indent=2)
    logger.info("\nWrote %s", out_path)

    return {
        "attribution": dict(attribution_counter),
        "total": total_births,
        "path": str(out_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", type=str, default=None)
    parser.add_argument("--focus-pair", action="store_true")
    args = parser.parse_args()

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

    if args.video_id:
        targets = [args.video_id]
    elif args.focus_pair:
        targets = list(DEFAULT_FOCUS)
    else:
        with get_connection() as conn:
            with conn.cursor() as cur:
                rows = load_all_from_db(cur)
        targets = [r.video_id[:8] for r in rows if r.gt.rallies]
        logger.info("Loaded %d GT videos", len(targets))

    run(targets, reid_model)


if __name__ == "__main__":
    main()
