#!/usr/bin/env python3
"""Session 7b-rally0-anchor pre-work — is the serve anchor right at rally 0?

The 7b-prework locality diagnostic killed the side-switch attack: only 28%
of cross-team errors are near switch boundaries; 72% are from-rally-0
inversions. Predicted switches match GT exactly on 5 of the 6 worst
cross-team videos yet team identities still flip. Conclusion: errors are
born at rally 0 with the wrong team-to-side anchor and propagate forward.

The proposed lever is rally-0 anchoring via `detect_serve_anchor`
(match_tracker.py:496-517 already fires it but only uses it as a fallback
for server recording, never to constrain rally-0 team assignment). Before
touching any production code, we need to know:

  1. Does the serve anchor fire at rally 0?
  2. Is it right — i.e., does the side it attributes the serve to match
     the GT-labelled server's side?
  3. Does match_tracker agree with it at rally 0, or override it?

For each video, classify rally 0 into one of:

  * anchor_right_mt_right    : both correct (no problem at rally 0)
  * anchor_right_mt_wrong    : THE LEVER — rally-0 override would fix it
  * anchor_wrong_mt_right    : mt recovers without the anchor
  * anchor_wrong_mt_wrong    : upstream serve-detection problem
  * anchor_absent_mt_right   : anchor silent; n/a
  * anchor_absent_mt_wrong   : anchor silent; can't help here

Go/no-go gate (focused on 5 structural-dominant videos):
  fb83f876, b03b461b, 1efa35cf, ce4c67a1, 840e8b6b

  ≥3 of 5 == anchor_right_mt_wrong  → GO (build rally-0 constraint)
  <3 of 5                            → NO-GO (route to Session 6)

Read-only. Reuses loaders + global-perm logic from
`diagnose_cross_team_switch_locality.py`. OSNet re-run ~30 min.

Usage:
    cd analysis
    uv run python scripts/diagnose_rally0_anchor.py
    uv run python scripts/diagnose_rally0_anchor.py --video-id fb83f876
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

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

FOCUS_VIDEOS = {"fb83f876", "b03b461b", "1efa35cf", "ce4c67a1", "840e8b6b"}


def _team(pid: int) -> int:
    return 0 if pid <= 2 else 1


def _find_best_permutation(
    gt_rallies: dict[str, dict[str, int]],
    pred_rallies: dict[str, dict[str, int]],
) -> dict[int, int]:
    """Global 4-permutation pred_pid→gt_pid (same as 7a/7b scripts)."""
    player_ids = [1, 2, 3, 4]
    best_perm: dict[int, int] = {pid: pid for pid in player_ids}
    best_c = -1
    for perm in itertools.permutations(player_ids):
        pm = {pid: gt for pid, gt in zip(player_ids, perm)}
        c = 0
        for rid, gt_map in gt_rallies.items():
            if rid not in pred_rallies:
                continue
            pred_map = pred_rallies[rid]
            for tid in gt_map:
                if tid in pred_map and pm.get(pred_map[tid]) == gt_map[tid]:
                    c += 1
        if c > best_c:
            best_c = c
            best_perm = pm
    return best_perm


def _track_sides_from_positions(
    positions: Any,
    court_split_y: float | None,
) -> dict[int, int]:
    """Replicate match_tracker.py:499-508 — avg_y vs court_split_y."""
    if court_split_y is None:
        return {}
    track_ys: dict[int, list[float]] = defaultdict(list)
    for p in positions:
        if p.track_id >= 0:
            track_ys[p.track_id].append(p.y)
    sides: dict[int, int] = {}
    for tid, ys in track_ys.items():
        avg_y = float(np.mean(ys))
        sides[tid] = 0 if avg_y > court_split_y else 1
    return sides


def _majority_side(tracks: list[int], track_sides: dict[int, int]) -> int | None:
    sides = [track_sides[t] for t in tracks if t in track_sides]
    if not sides:
        return None
    c = Counter(sides)
    return c.most_common(1)[0][0]


def classify_rally0(
    rally0: Any,
    rally0_preds: dict[str, int],  # track_id_str → pred_player_id (from mt)
    gt_rally0: dict[str, int],      # track_id_str → gt_player_id
    perm: dict[int, int],           # pred_pid → gt_pid
    anchor: Any,                    # ServeAnchor | None
) -> dict[str, Any]:
    """Compute per-video rally-0 classification."""
    track_sides = _track_sides_from_positions(rally0.positions, rally0.court_split_y)

    # GT team→side: group tracks by gt team, majority side per team
    gt_team_tracks: dict[int, list[int]] = defaultdict(list)
    for tid_str, gt_pid in gt_rally0.items():
        try:
            tid = int(tid_str)
        except ValueError:
            continue
        gt_team_tracks[_team(gt_pid)].append(tid)
    gt_team_side = {
        tm: _majority_side(tids, track_sides) for tm, tids in gt_team_tracks.items()
    }

    # MT team→side: apply perm to pred player ids, then group by gt-team label
    mt_team_tracks: dict[int, list[int]] = defaultdict(list)
    for tid_str, pred_pid in rally0_preds.items():
        mapped_pid = perm.get(pred_pid)
        if mapped_pid is None:
            continue
        try:
            tid = int(tid_str)
        except ValueError:
            continue
        mt_team_tracks[_team(mapped_pid)].append(tid)
    mt_team_side = {
        tm: _majority_side(tids, track_sides) for tm, tids in mt_team_tracks.items()
    }

    # Which team (in gt label space) served, according to GT?
    # Use the anchor's server_track_id if anchor fired; otherwise pick the
    # track on the serving side closest to the baseline — but we need a
    # server track to compare. If anchor is absent we still grade mt by
    # whether its team→side layout matches GT (that's the from-rally-0
    # inversion question).
    anchor_absent = anchor is None or anchor.server_track_id < 0
    anchor_side: int | None = None
    gt_server_team: int | None = None
    mt_server_team: int | None = None
    gt_side_for_server: int | None = None
    mt_side_for_server: int | None = None

    if not anchor_absent:
        server_tid = anchor.server_track_id
        anchor_side = anchor.server_team  # 0=near, 1=far (that's a side)
        gt_pid_opt = gt_rally0.get(str(server_tid))
        if gt_pid_opt is not None:
            gt_server_team = _team(gt_pid_opt)
            gt_side_for_server = gt_team_side.get(gt_server_team)
        pred_pid_opt = rally0_preds.get(str(server_tid))
        if pred_pid_opt is not None:
            mapped = perm.get(pred_pid_opt)
            if mapped is not None:
                mt_server_team = _team(mapped)
                mt_side_for_server = mt_team_side.get(mt_server_team)

    # Grade the mt rally-0 layout overall: did mt place gt-team-0 on the
    # same side GT says? (Checks from-rally-0 inversion directly.)
    mt_layout_correct = (
        gt_team_side.get(0) is not None
        and mt_team_side.get(0) is not None
        and gt_team_side.get(0) == mt_team_side.get(0)
        and gt_team_side.get(1) == mt_team_side.get(1)
    )

    # Anchor correctness: serve anchor side matches gt server's team side
    if anchor_absent:
        anchor_correct: bool | None = None
    else:
        anchor_correct = (
            anchor_side is not None
            and gt_side_for_server is not None
            and anchor_side == gt_side_for_server
        )

    # Labels
    if anchor_absent:
        label = "anchor_absent_mt_right" if mt_layout_correct else "anchor_absent_mt_wrong"
    elif anchor_correct and mt_layout_correct:
        label = "anchor_right_mt_right"
    elif anchor_correct and not mt_layout_correct:
        label = "anchor_right_mt_wrong"
    elif not anchor_correct and mt_layout_correct:
        label = "anchor_wrong_mt_right"
    else:
        label = "anchor_wrong_mt_wrong"

    return {
        "label": label,
        "anchor_absent": anchor_absent,
        "anchor_side": anchor_side,
        "anchor_confidence": float(anchor.confidence) if not anchor_absent else None,
        "anchor_server_track_id": (
            int(anchor.server_track_id) if not anchor_absent else None
        ),
        "gt_team_side": {str(k): v for k, v in gt_team_side.items()},
        "mt_team_side": {str(k): v for k, v in mt_team_side.items()},
        "gt_side_for_server": gt_side_for_server,
        "mt_side_for_server": mt_side_for_server,
        "mt_layout_correct": mt_layout_correct,
        "court_split_y": rally0.court_split_y,
    }


def run_video(video_id: str, reid_model: Any) -> dict[str, Any] | None:
    from rallycut.evaluation.tracking.db import (
        get_video_path,
        load_rallies_for_video,
    )
    from rallycut.tracking.identity_anchor import detect_serve_anchor
    from rallycut.tracking.match_tracker import match_players_across_rallies

    rallies = load_rallies_for_video(video_id)
    video_path = get_video_path(video_id)
    if not rallies or not video_path:
        logger.warning("  cannot load rallies/video")
        return None

    # Run the full OSNet baseline match (so perm is computed against the
    # entire match, same as 7a/7b scripts).
    try:
        result = match_players_across_rallies(
            video_path=video_path,
            rallies=rallies,
            reid_model=reid_model,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("  match failed: %s", exc)
        return None

    preds = {
        rallies[i].rally_id: {str(k): v for k, v in r.track_to_player.items()}
        for i, r in enumerate(result.rally_results)
    }

    # Compute serve anchor on rally 0 using the same inputs process_rally sees.
    rally0 = rallies[0]
    track_sides = _track_sides_from_positions(rally0.positions, rally0.court_split_y)
    anchor = None
    if track_sides:
        anchor = detect_serve_anchor(
            rally0.positions,
            track_sides,
            ball_positions=rally0.ball_positions,
            calibrator=None,
            serve_window_frames=30,
        )

    return {
        "rallies": rallies,
        "rally0": rally0,
        "preds": preds,
        "anchor": anchor,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video-id", type=str, default=None)
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

    with get_connection() as conn:
        with conn.cursor() as cur:
            gt_rows = load_all_from_db(cur, video_id_prefix=args.video_id)

    if not gt_rows:
        logger.error("No GT videos found.")
        sys.exit(1)

    logger.info("Found %d GT videos\n", len(gt_rows))

    per_video: list[dict[str, Any]] = []
    started = time.time()

    for idx, row in enumerate(gt_rows, start=1):
        video_id = row.video_id
        short = video_id[:8]
        gt_rallies = row.gt.rallies
        if not gt_rallies:
            logger.warning("[%d/%d %s] empty GT, skipping", idx, len(gt_rows), short)
            continue
        logger.info(
            "[%d/%d %s] running OSNet...", idx, len(gt_rows), short,
        )

        out = run_video(video_id, reid_model)
        if out is None:
            continue

        rally0 = out["rally0"]
        rally0_rid = rally0.rally_id
        if rally0_rid not in gt_rallies:
            logger.warning("  rally0 %s not in GT, skipping", rally0_rid[:8])
            continue
        gt_rally0 = gt_rallies[rally0_rid]
        rally0_preds = out["preds"].get(rally0_rid, {})
        perm = _find_best_permutation(gt_rallies, out["preds"])

        cls = classify_rally0(rally0, rally0_preds, gt_rally0, perm, out["anchor"])
        cls["video_id"] = video_id
        cls["short"] = short
        per_video.append(cls)

        a_side = cls["anchor_side"]
        a_side_s = "near" if a_side == 0 else "far" if a_side == 1 else "—"
        gt_s = cls["gt_side_for_server"]
        gt_s_s = "near" if gt_s == 0 else "far" if gt_s == 1 else "—"
        logger.info(
            "    anchor_side=%s gt_server_side=%s mt_layout=%s → %s",
            a_side_s, gt_s_s,
            "ok" if cls["mt_layout_correct"] else "FLIP",
            cls["label"],
        )

    elapsed = time.time() - started
    logger.info("\nFinished %d videos in %.1fs\n", len(per_video), elapsed)

    if not per_video:
        logger.error("No per-video results.")
        sys.exit(1)

    # ---- Focus table ----
    logger.info("=" * 90)
    logger.info("5 STRUCTURAL-DOMINANT VIDEOS (gate)")
    logger.info("=" * 90)
    logger.info(
        "%-10s %-10s %-10s %-8s %s", "Video", "AnchorSide", "GTSrvSide", "MT", "Label",
    )
    logger.info("-" * 90)
    focus_rows = [v for v in per_video if v["short"] in FOCUS_VIDEOS]
    focus_go = 0
    for v in sorted(focus_rows, key=lambda x: x["short"]):
        a = v["anchor_side"]
        g = v["gt_side_for_server"]
        a_s = "near" if a == 0 else "far" if a == 1 else "—"
        g_s = "near" if g == 0 else "far" if g == 1 else "—"
        mt_s = "ok" if v["mt_layout_correct"] else "FLIP"
        logger.info("%-10s %-10s %-10s %-8s %s", v["short"], a_s, g_s, mt_s, v["label"])
        if v["label"] == "anchor_right_mt_wrong":
            focus_go += 1
    logger.info("-" * 90)
    logger.info("Focus anchor_right_mt_wrong count: %d / %d", focus_go, len(focus_rows))

    # ---- Aggregate ----
    labels = Counter(v["label"] for v in per_video)
    logger.info("\n" + "=" * 90)
    logger.info("AGGREGATE — %d videos", len(per_video))
    logger.info("=" * 90)
    for lab in (
        "anchor_right_mt_right",
        "anchor_right_mt_wrong",
        "anchor_wrong_mt_right",
        "anchor_wrong_mt_wrong",
        "anchor_absent_mt_right",
        "anchor_absent_mt_wrong",
    ):
        logger.info("  %-28s : %d", lab, labels.get(lab, 0))

    # ---- Verdict ----
    logger.info("\n" + "=" * 90)
    if focus_go >= 3:
        verdict = (
            f"GO — {focus_go}/{len(focus_rows)} focus videos show "
            "anchor_right_mt_wrong. Rally-0 anchoring is a real lever."
        )
        next_step = (
            "→ Next session: build the rally-0 constraint in "
            "match_tracker._initialize_first_rally using the serve anchor."
        )
    else:
        verdict = (
            f"NO-GO — only {focus_go}/{len(focus_rows)} focus videos show "
            "anchor_right_mt_wrong. Anchor is absent or wrong on the dominant slice."
        )
        next_step = (
            "→ Route to Session 6 (serve-detection rewrite). The rally-0 "
            "signal isn't reliable enough to anchor on."
        )
    logger.info("VERDICT: %s", verdict)
    logger.info("        %s", next_step)
    logger.info("=" * 90)

    # ---- JSON dump ----
    out_dir = Path("outputs/cross_rally_errors")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%d-%H%M%S")
    out_path = out_dir / f"rally0_anchor_{ts}.json"
    with out_path.open("w") as fh:
        json.dump(
            {
                "n_videos": len(per_video),
                "focus_videos": sorted(FOCUS_VIDEOS),
                "focus_anchor_right_mt_wrong": focus_go,
                "aggregate_labels": dict(labels),
                "verdict": verdict,
                "per_video": per_video,
            },
            fh,
            indent=2,
        )
    logger.info("\nWrote %s", out_path)


if __name__ == "__main__":
    main()
