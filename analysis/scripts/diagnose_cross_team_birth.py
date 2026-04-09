#!/usr/bin/env python3
"""Session 7 follow-up — where are cross-team errors born mid-match?

The 7b-rally0-anchor pre-work (2026-04-09) killed both the side-switch
attack and the rally-0 team→side anchor. On b03b461b (9 cross-team errors)
and fb83f876 (10 cross-team errors) the match_tracker rally-0 layout is
already correct, yet cross-team errors still appear downstream. The open
question: at which rally does each cross-team error first appear, and by
what mechanism?

For each cross-team error (track_id, player_id) observed at any rally,
identify the **birth rally** — the earliest rally where the mapped pred
pid disagrees with GT at the cross-team level. Classify the birth:

  * `rally_0`           — born in rally 0 (first appearance of the track)
  * `new_track`         — track first appears in this rally (Pass-1 Hungarian
                          on a fresh track); previous rallies did not have it
  * `switch_rally`      — rally has `side_switch_detected=True`
  * `pair_swap`         — two same-team teammates flip simultaneously with
                          their team→side layout intact (Pass-2 within-team
                          vote swap signature)
  * `solo_flip`         — single track's mapped pid crosses team boundary
                          on an existing track without a switch or pair

Includes a perm-artifact check: the raw rally_0 bucket inflates because
the global 4-permutation steers toward the post-mid-match-flip
orientation, making a locally-correct rally 0 look wrong. Any video
where rally-0 births vanish under a rally-0-optimal perm is flagged as
a perm artifact, and those rally_0 births are filtered out of the
"real" aggregate.

Read-only. Defaults to the full 51-video GT pool (~30-40 min OSNet
re-run; background it). Use --focus-pair for the fast b03b461b +
fb83f876 sanity check (~2 min), or --video-id for a single video.
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

DEFAULT_FOCUS = ["b03b461b", "fb83f876"]


def _team(pid: int) -> int:
    return 0 if pid <= 2 else 1


def _find_best_permutation(
    gt_rallies: dict[str, dict[str, int]],
    pred_rallies: dict[str, dict[str, int]],
) -> dict[int, int]:
    player_ids = [1, 2, 3, 4]
    best_perm: dict[int, int] = {pid: pid for pid in player_ids}
    best_c = -1
    for perm in itertools.permutations(player_ids):
        pm = {pid: gt for pid, gt in zip(player_ids, perm)}
        c = 0
        for rid, gt_map in gt_rallies.items():
            pred_map = pred_rallies.get(rid, {})
            for tid in gt_map:
                if tid in pred_map and pm.get(pred_map[tid]) == gt_map[tid]:
                    c += 1
        if c > best_c:
            best_c = c
            best_perm = pm
    return best_perm


def analyse_video(
    video_id: str,
    reid_model: Any,
) -> dict[str, Any] | None:
    from rallycut.evaluation.db import get_connection
    from rallycut.evaluation.gt_loader import load_all_from_db
    from rallycut.evaluation.tracking.db import (
        get_video_path,
        load_rallies_for_video,
    )
    from rallycut.tracking.match_tracker import match_players_across_rallies

    with get_connection() as conn:
        with conn.cursor() as cur:
            rows = load_all_from_db(cur, video_id_prefix=video_id)
    if not rows:
        logger.warning("  no GT for %s", video_id)
        return None
    row = rows[0]
    full_video_id = row.video_id
    gt_rallies = row.gt.rallies

    rallies = load_rallies_for_video(full_video_id)
    video_path = get_video_path(full_video_id)
    if not rallies or not video_path:
        logger.warning("  no rallies/video")
        return None

    result = match_players_across_rallies(
        video_path=video_path,
        rallies=rallies,
        reid_model=reid_model,
    )

    preds_by_rally: dict[str, dict[str, int]] = {
        rallies[i].rally_id: {str(k): v for k, v in r.track_to_player.items()}
        for i, r in enumerate(result.rally_results)
    }
    switch_flag = [r.side_switch_detected for r in result.rally_results]
    perm = _find_best_permutation(gt_rallies, preds_by_rally)

    # Perm artifact check: compute the rally-0-optimal perm (ignores all
    # other rallies). If it differs from the global perm AND gives 0
    # cross-team errors at rally 0, then the "rally 0 cross-team" births
    # we report are global-perm artifacts (a later side switch made the
    # global perm prefer the post-switch orientation, which makes rally 0
    # look wrong even though it was locally correct).
    rally0_id = rallies[0].rally_id
    rally0_gt = gt_rallies.get(rally0_id, {})
    rally0_preds = preds_by_rally.get(rally0_id, {})
    r0_only_gt = {rally0_id: rally0_gt} if rally0_gt else {}
    rally0_perm = (
        _find_best_permutation(r0_only_gt, preds_by_rally) if r0_only_gt else perm
    )

    def _crossteam_at_rally0(p: dict[int, int]) -> int:
        n = 0
        for tid_str, gt_pid in rally0_gt.items():
            pp = rally0_preds.get(tid_str)
            if pp is None:
                continue
            m = p.get(pp)
            if m is None:
                continue
            if _team(m) != _team(gt_pid):
                n += 1
        return n

    global_rally0_cross = _crossteam_at_rally0(perm)
    local_rally0_cross = _crossteam_at_rally0(rally0_perm)
    perm_artifact = (
        perm != rally0_perm
        and global_rally0_cross > 0
        and local_rally0_cross == 0
    )

    # Walk rallies in order. For each (rally_idx, tid) compute the cross-team
    # state. Track per-tid birth rally.
    tid_history: dict[str, list[tuple[int, str]]] = defaultdict(list)
    # tid_history[tid] = [(rally_idx, state), ...] where state ∈
    #   {correct, within-team, cross-team}
    for rally_idx, rally in enumerate(rallies):
        rid = rally.rally_id
        gt_map = gt_rallies.get(rid, {})
        pred_map = preds_by_rally.get(rid, {})
        if not gt_map:
            continue
        for tid_str, gt_pid in gt_map.items():
            pred_pid = pred_map.get(tid_str)
            if pred_pid is None:
                tid_history[tid_str].append((rally_idx, "absent"))
                continue
            mapped = perm.get(pred_pid)
            if mapped is None:
                tid_history[tid_str].append((rally_idx, "absent"))
                continue
            if mapped == gt_pid:
                state = "correct"
            elif _team(mapped) == _team(gt_pid):
                state = "within-team"
            else:
                state = "cross-team"
            tid_history[tid_str].append((rally_idx, state))

    # Detect birth rallies: first cross-team transition for each tid.
    births: list[dict[str, Any]] = []
    for tid_str, hist in tid_history.items():
        prev = "absent"
        for rally_idx, state in hist:
            if state == "cross-team" and prev != "cross-team":
                births.append({
                    "tid": tid_str,
                    "rally_index": rally_idx,
                    "prev_state": prev,
                })
            prev = state

    # Group births by rally_index to spot pair swaps.
    per_rally_births: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for b in births:
        per_rally_births[b["rally_index"]].append(b)

    # Classify each birth.
    classified: list[dict[str, Any]] = []
    for rally_idx, bs in sorted(per_rally_births.items()):
        rid = rallies[rally_idx].rally_id
        gt_map = gt_rallies.get(rid, {})
        pred_map = preds_by_rally.get(rid, {})
        tids_born = [b["tid"] for b in bs]

        # Pair-swap signature: ≥2 births same rally, and the set of mapped
        # pids among the born tids is the same set as the gt pids (i.e.,
        # it's a permutation within the same team set).
        is_pair_swap = False
        if len(bs) >= 2:
            mapped_pids = []
            gt_pids = []
            for t in tids_born:
                pp = pred_map.get(t)
                if pp is not None:
                    m = perm.get(pp)
                    if m is not None:
                        mapped_pids.append(m)
                gp = gt_map.get(t)
                if gp is not None:
                    gt_pids.append(gp)
            if set(mapped_pids) == set(gt_pids) and len(mapped_pids) >= 2:
                is_pair_swap = True

        for b in bs:
            if rally_idx == 0:
                cat = "rally_0"
            elif b["prev_state"] == "absent":
                cat = "new_track"
            elif switch_flag[rally_idx]:
                cat = "switch_rally"
            elif is_pair_swap:
                cat = "pair_swap"
            else:
                cat = "solo_flip"
            classified.append({
                **b,
                "category": cat,
                "switch_flag": switch_flag[rally_idx],
                "rally_id": rid[:8],
                "pair_rally_size": len(bs),
            })

    return {
        "video_id": full_video_id,
        "short": full_video_id[:8],
        "n_rallies": len(rallies),
        "n_cross_team_births": len(classified),
        "births": classified,
        "switch_rallies": [i for i, s in enumerate(switch_flag) if s],
        "global_perm": {str(k): v for k, v in perm.items()},
        "rally0_perm": {str(k): v for k, v in rally0_perm.items()},
        "global_rally0_cross_team": global_rally0_cross,
        "local_rally0_cross_team": local_rally0_cross,
        "rally0_is_perm_artifact": perm_artifact,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", type=str, default=None,
                        help="Single video ID prefix; if unset, runs the full 51-video GT pool.")
    parser.add_argument("--focus-pair", action="store_true",
                        help="Run only the b03b461b + fb83f876 focus pair (fast).")
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
        targets = DEFAULT_FOCUS
    else:
        with get_connection() as conn:
            with conn.cursor() as cur:
                rows = load_all_from_db(cur)
        targets = [r.video_id[:8] for r in rows if r.gt.rallies]
        logger.info("Loaded %d GT videos from the pool", len(targets))
    results: list[dict[str, Any]] = []
    started = time.time()
    for i, vid in enumerate(targets, start=1):
        logger.info("\n[%d/%d %s] running...", i, len(targets), vid)
        r = analyse_video(vid, reid_model)
        if r is None:
            continue
        results.append(r)
        logger.info(
            "  rallies=%d  cross-team births=%d  switch_rallies=%s",
            r["n_rallies"], r["n_cross_team_births"], r["switch_rallies"],
        )
        logger.info(
            "  perm: global=%s rally0-optimal=%s  rally0_cross_team: global=%d local=%d  artifact=%s",
            r["global_perm"], r["rally0_perm"],
            r["global_rally0_cross_team"], r["local_rally0_cross_team"],
            r["rally0_is_perm_artifact"],
        )
        logger.info("  %-4s %-10s %-14s %-14s %-6s",
                    "ridx", "rally_id", "prev_state", "category", "tid")
        for b in r["births"]:
            logger.info(
                "  %-4d %-10s %-14s %-14s %-6s",
                b["rally_index"], b["rally_id"], b["prev_state"],
                b["category"], b["tid"],
            )
    elapsed = time.time() - started
    logger.info("\nFinished %d videos in %.1fs", len(results), elapsed)

    if not results:
        return

    # Aggregate: raw, real (perm-artifact filtered), and per-video
    raw_cats: Counter[str] = Counter()
    real_cats: Counter[str] = Counter()
    videos_by_mechanism: dict[str, list[str]] = defaultdict(list)
    n_artifact_videos = 0
    for r in results:
        is_artifact = r["rally0_is_perm_artifact"]
        if is_artifact:
            n_artifact_videos += 1
        real_by_video: Counter[str] = Counter()
        for b in r["births"]:
            raw_cats[b["category"]] += 1
            if is_artifact and b["category"] == "rally_0":
                continue  # filtered perm artifact
            real_cats[b["category"]] += 1
            real_by_video[b["category"]] += 1
        for cat in real_by_video:
            videos_by_mechanism[cat].append(r["short"])

    logger.info("\n" + "=" * 70)
    logger.info("RAW CROSS-TEAM BIRTH CATEGORIES (all %d videos)", len(results))
    logger.info("=" * 70)
    total_raw = sum(raw_cats.values())
    for cat in ("rally_0", "new_track", "switch_rally", "pair_swap", "solo_flip"):
        n = raw_cats.get(cat, 0)
        pct = n / total_raw * 100 if total_raw else 0
        logger.info("  %-14s : %3d  (%.1f%%)", cat, n, pct)
    logger.info("  %-14s : %3d", "TOTAL", total_raw)

    logger.info("\n" + "=" * 70)
    logger.info(
        "REAL CROSS-TEAM BIRTHS (perm-artifact filtered; %d/%d videos were artifacts)",
        n_artifact_videos, len(results),
    )
    logger.info("=" * 70)
    total_real = sum(real_cats.values())
    for cat in ("rally_0", "new_track", "switch_rally", "pair_swap", "solo_flip"):
        n = real_cats.get(cat, 0)
        pct = n / total_real * 100 if total_real else 0
        vids = videos_by_mechanism.get(cat, [])
        vids_str = (
            ", ".join(sorted(set(vids))[:8])
            + (f" +{len(set(vids)) - 8} more" if len(set(vids)) > 8 else "")
        )
        logger.info("  %-14s : %3d  (%.1f%%)  [%s]", cat, n, pct, vids_str)
    logger.info("  %-14s : %3d", "TOTAL", total_real)

    # Verdict hint
    logger.info("\n" + "=" * 70)
    all_cats = real_cats  # drive verdict off real counts
    dominant = all_cats.most_common(1)[0][0] if all_cats else "none"
    logger.info("Dominant mechanism: %s", dominant)
    if dominant == "pair_swap":
        logger.info("→ Pass-2 within-team voting is flipping teammates across sides.")
        logger.info("  Lever: gate pass-2 within-team voting, or add a side-consistency check.")
    elif dominant == "new_track":
        logger.info("→ Errors are born when fresh tracks enter mid-match.")
        logger.info("  Lever: strengthen Pass-1 side penalty for new tracks.")
    elif dominant == "switch_rally":
        logger.info("→ Errors cluster on switch rallies despite correct detection.")
        logger.info("  Lever: audit how side switches remap player ids.")
    elif dominant == "solo_flip":
        logger.info("→ Single-track cross-team drift on existing tracks.")
        logger.info("  Lever: stronger per-track side memory across rallies.")
    logger.info("=" * 70)

    out_dir = Path("outputs/cross_rally_errors")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d-%H%M%S")
    out_path = out_dir / f"cross_team_birth_{ts}.json"
    with out_path.open("w") as fh:
        json.dump(
            {
                "raw_aggregate": dict(raw_cats),
                "real_aggregate": dict(real_cats),
                "n_artifact_videos": n_artifact_videos,
                "videos_by_mechanism": {k: sorted(set(v)) for k, v in videos_by_mechanism.items()},
                "per_video": results,
            },
            fh,
            indent=2,
        )
    logger.info("\nWrote %s", out_path)


if __name__ == "__main__":
    main()
