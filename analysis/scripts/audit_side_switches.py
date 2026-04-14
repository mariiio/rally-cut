"""Audit sideSwitches GT for correctness.

For each video with formation errors, shows all rallies with their:
- Rally index, GT serving team, formation prediction, side_flipped state
- Whether this rally is at or near a sideSwitch boundary
- Visual marker for suspected GT errors (prediction correct but marked wrong)

This helps identify incorrect sideSwitch indices that create phantom errors.

Usage:
    cd analysis
    uv run python scripts/audit_side_switches.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rallycut.evaluation.tracking.db import get_connection  # noqa: E402
from rallycut.tracking.action_classifier import (  # noqa: E402
    _find_serving_side_by_formation,
)
from rallycut.tracking.player_tracker import PlayerPosition  # noqa: E402

EXCLUDED_VIDEOS = {
    "0a383519",  # yoyo / IMG_2313 — low camera angle
    "627c1add",  # caca — tilted camera
}


def _parse_positions(raw: list[dict]) -> list[PlayerPosition]:
    return [
        PlayerPosition(
            frame_number=p["frameNumber"],
            track_id=p["trackId"],
            x=p["x"],
            y=p["y"],
            width=p.get("width", 0.05),
            height=p.get("height", 0.10),
            confidence=p.get("confidence", 1.0),
            keypoints=p.get("keypoints"),
        )
        for p in raw
    ]


def _gt_physical_side(
    gt_serving_team: str, side_flipped: bool, initial_near_is_a: bool,
) -> str:
    near_is_a = initial_near_is_a != side_flipped
    if gt_serving_team == "A":
        return "near" if near_is_a else "far"
    else:
        return "far" if near_is_a else "near"


def main() -> int:
    from scripts.eval_score_tracking import load_score_gt

    video_rallies = load_score_gt()
    video_rallies = {
        vid: rallies for vid, rallies in video_rallies.items()
        if not any(vid.startswith(ex) for ex in EXCLUDED_VIDEOS)
    }
    total = sum(len(v) for v in video_rallies.values())
    print(f"Loaded {total} rallies from {len(video_rallies)} videos")

    # Load video names and sideSwitches
    video_names: dict[str, str] = {}
    video_switches: dict[str, list[int]] = {}
    with get_connection() as conn, conn.cursor() as cur:
        vids = list(video_rallies.keys())
        ph = ", ".join(["%s"] * len(vids))
        cur.execute(
            f"SELECT id, s3_key, player_matching_gt_json FROM videos WHERE id IN ({ph})",
            vids,
        )
        for vid, s3_key, gt_json in cur.fetchall():
            video_names[vid] = Path(s3_key).stem if s3_key else vid[:8]
            sw = list(gt_json.get("sideSwitches", [])) if isinstance(gt_json, dict) else []
            video_switches[vid] = sorted(sw)

    # Per-video convention calibration
    initial_near_is_a: dict[str, bool] = {}
    for vid, rallies in video_rallies.items():
        vt, vf = 0, 0
        for r in rallies:
            pos = _parse_positions(r.positions)
            ny = r.court_split_y or 0.5
            ps, _ = _find_serving_side_by_formation(pos, net_y=ny, start_frame=0)
            if ps is None:
                continue
            if ps == _gt_physical_side(r.gt_serving_team, r.side_flipped, True):
                vt += 1
            if ps == _gt_physical_side(r.gt_serving_team, r.side_flipped, False):
                vf += 1
        initial_near_is_a[vid] = vt >= vf

    # Analyze each video
    videos_with_errors: list[tuple[str, int]] = []  # (vid, n_errors)

    for vid in sorted(video_rallies.keys(), key=lambda v: video_names.get(v, "")):
        rallies = video_rallies[vid]
        vname = video_names.get(vid, vid[:8])
        nia = initial_near_is_a[vid]
        switches = video_switches.get(vid, [])

        # Run formation on all rallies
        rally_results: list[dict] = []
        n_errors = 0
        for r in rallies:
            pos = _parse_positions(r.positions)
            ny = r.court_split_y or 0.5
            pred_side, conf = _find_serving_side_by_formation(
                pos, net_y=ny, start_frame=0,
            )
            gt_phys = _gt_physical_side(r.gt_serving_team, r.side_flipped, nia)
            is_correct = pred_side == gt_phys if pred_side is not None else None

            if not is_correct:
                n_errors += 1

            rally_results.append({
                "idx": r.rally_index,
                "rally_id": r.rally_id[:8],
                "gt_team": r.gt_serving_team,
                "flipped": r.side_flipped,
                "gt_phys": gt_phys,
                "pred_side": pred_side,
                "conf": conf,
                "correct": is_correct,
                "at_switch": r.rally_index in switches,
                "start_ms": r.start_ms,
            })

        if n_errors == 0:
            continue

        videos_with_errors.append((vid, n_errors))

        # Print video header
        n_correct = sum(1 for r in rally_results if r["correct"] is True)
        acc = n_correct / len(rally_results) * 100
        print(f"\n{'='*80}")
        print(f"{vname}  ({vid[:8]})  "
              f"{n_correct}/{len(rally_results)} correct ({acc:.0f}%)  "
              f"near_is_a={nia}  switches={switches}")
        print(f"{'='*80}")

        # Column headers
        print(f"{'idx':>3s}  {'rally':>8s}  {'gt_tm':>5s}  {'flip':>4s}  "
              f"{'gt_ph':>5s}  {'pred':>5s}  {'conf':>5s}  {'ok':>5s}  "
              f"{'switch':>6s}  {'suspect':>7s}")
        print("-" * 75)

        for r in rally_results:
            ok_str = "Y" if r["correct"] is True else ("N" if r["correct"] is False else "-")
            switch_str = "<< SW" if r["at_switch"] else ""
            # Suspect: prediction looks right but GT says wrong
            # (pred has reasonable confidence and disagrees with GT)
            suspect = ""
            if r["correct"] is False and r["pred_side"] is not None and r["conf"] > 0.1:
                suspect = "CHECK"
            elif r["correct"] is None:
                suspect = "ABST"

            # Color markers for readability
            marker = ""
            if r["at_switch"]:
                marker = " <<"

            print(f"{r['idx']:>3d}  {r['rally_id']:>8s}  {r['gt_team']:>5s}  "
                  f"{'Y' if r['flipped'] else 'N':>4s}  "
                  f"{r['gt_phys']:>5s}  {(r['pred_side'] or '-'):>5s}  "
                  f"{r['conf']:>5.2f}  {ok_str:>5s}  "
                  f"{switch_str:>6s}  {suspect:>7s}")

        # Detect streaks of errors that suggest switch is at wrong index
        print(f"\n  Error streaks (consecutive errors suggest switch-off-by-one):")
        streak_start = None
        streak_len = 0
        for i, r in enumerate(rally_results):
            if r["correct"] is not True:  # error or abstain
                if streak_start is None:
                    streak_start = i
                streak_len += 1
            else:
                if streak_len >= 2:
                    streak_idxs = [rally_results[j]["idx"] for j in range(streak_start, streak_start + streak_len)]
                    nearby_switch = any(
                        abs(s - rally_results[streak_start]["idx"]) <= 2
                        for s in switches
                    )
                    flag = " ← NEAR SWITCH" if nearby_switch else ""
                    print(f"    rallies {streak_idxs[0]}-{streak_idxs[-1]}: "
                          f"{streak_len} consecutive errors{flag}")
                streak_start = None
                streak_len = 0
        if streak_len >= 2:
            streak_idxs = [rally_results[j]["idx"] for j in range(streak_start, streak_start + streak_len)]
            nearby_switch = any(
                abs(s - rally_results[streak_start]["idx"]) <= 2
                for s in switches
            )
            flag = " ← NEAR SWITCH" if nearby_switch else ""
            print(f"    rallies {streak_idxs[0]}-{streak_idxs[-1]}: "
                  f"{streak_len} consecutive errors{flag}")

        # Test alternative switch configurations
        print(f"\n  Switch sensitivity analysis:")
        current_acc = n_correct / len(rallies) * 100

        # Try removing each switch
        for sw in switches:
            alt_switches = [s for s in switches if s != sw]
            alt_correct = _count_correct_with_switches(rallies, alt_switches, nia)
            alt_acc = alt_correct / len(rallies) * 100
            delta = alt_acc - current_acc
            flag = " ← IMPROVES" if delta > 0 else ""
            print(f"    Remove switch@{sw}: {alt_correct}/{len(rallies)} "
                  f"({alt_acc:.0f}%) delta={delta:+.1f}pp{flag}")

        # Try shifting each switch ±1
        for sw in switches:
            for offset in [-1, +1]:
                alt_sw = sw + offset
                if alt_sw < 0 or alt_sw >= len(rallies):
                    continue
                alt_switches = [alt_sw if s == sw else s for s in switches]
                alt_correct = _count_correct_with_switches(rallies, alt_switches, nia)
                alt_acc = alt_correct / len(rallies) * 100
                delta = alt_acc - current_acc
                flag = " ← IMPROVES" if delta > 0 else ""
                print(f"    Move switch {sw}→{alt_sw}: {alt_correct}/{len(rallies)} "
                      f"({alt_acc:.0f}%) delta={delta:+.1f}pp{flag}")

        # Try adding a switch at each position with errors
        error_idxs = [r["idx"] for r in rally_results if r["correct"] is not True]
        tried: set[int] = set()
        for eidx in error_idxs:
            for try_sw in [eidx, eidx + 1]:
                if try_sw in tried or try_sw in switches or try_sw >= len(rallies):
                    continue
                tried.add(try_sw)
                alt_switches = sorted(switches + [try_sw])
                alt_correct = _count_correct_with_switches(rallies, alt_switches, nia)
                alt_acc = alt_correct / len(rallies) * 100
                delta = alt_acc - current_acc
                if delta > 0:
                    print(f"    Add switch@{try_sw}: {alt_correct}/{len(rallies)} "
                          f"({alt_acc:.0f}%) delta={delta:+.1f}pp ← IMPROVES")

    # Summary
    print(f"\n{'='*80}")
    print(f"Videos with errors: {len(videos_with_errors)}")
    for vid, n_err in sorted(videos_with_errors, key=lambda x: -x[1]):
        vname = video_names.get(vid, vid[:8])
        print(f"  {vname:<25s}  {n_err} errors")

    return 0


def _count_correct_with_switches(
    rallies: list,
    switches: list[int],
    initial_near_is_a: bool,
) -> int:
    """Count correct predictions with alternative switch configuration."""
    switch_set = set(switches)
    flipped = False
    correct = 0
    for idx, r in enumerate(rallies):
        if idx in switch_set:
            flipped = not flipped
        pos = _parse_positions(r.positions)
        ny = r.court_split_y or 0.5
        pred_side, _ = _find_serving_side_by_formation(pos, net_y=ny, start_frame=0)
        gt_phys = _gt_physical_side(r.gt_serving_team, flipped, initial_near_is_a)
        if pred_side == gt_phys:
            correct += 1
    return correct


if __name__ == "__main__":
    raise SystemExit(main())
