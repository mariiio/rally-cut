"""Evaluate cross-rally Viterbi with user correction anchors.

Tests multiple configurations:
  1. Formation-only Viterbi (no anchors) — baseline
  2. + First-serve anchor (user specifies who served rally 1)
  3. + Count constraint (user provides final score)
  4. + Both first-serve + count constraint

All use GT side switches and GT-calibrated initial_near_is_a for oracle
measurement of the formation signal ceiling.

Usage:
    cd analysis
    uv run python scripts/eval_score_viterbi.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from rallycut.evaluation.tracking.db import get_connection  # noqa: E402
from rallycut.scoring.cross_rally_viterbi import (  # noqa: E402
    RallyObservation,
    calibrate_from_noisy_predictions,
    calibrate_initial_side,
    decode_video,
    decode_video_dual_hypothesis,
)
from rallycut.tracking.action_classifier import (  # noqa: E402
    _find_serving_side_by_formation,
)
from rallycut.tracking.player_tracker import PlayerPosition  # noqa: E402

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class EvalRally:
    rally_id: str
    video_id: str
    start_ms: int
    gt_serving_team: str
    positions: list[PlayerPosition]
    court_split_y: float | None
    rally_index: int = 0
    side_flipped: bool = False
    # Production signals (from match_analysis, not GT)
    prod_serving_team: str | None = None  # from formation + team_assignments + semantic_flip
    prod_semantic_flip: bool = False


def _parse_positions(raw: list[dict[str, Any]]) -> list[PlayerPosition]:
    return [
        PlayerPosition(
            frame_number=pp["frameNumber"],
            track_id=pp["trackId"],
            x=pp["x"],
            y=pp["y"],
            width=pp.get("width", 0.05),
            height=pp.get("height", 0.10),
            confidence=pp.get("confidence", 1.0),
            keypoints=pp.get("keypoints"),
        )
        for pp in raw
    ]


def _load_eval_data() -> dict[str, list[EvalRally]]:
    """Load all rallies with gt_serving_team, positions, and production signals."""
    from rallycut.tracking.action_classifier import (
        _find_serving_team_by_formation,
    )
    from rallycut.tracking.match_tracker import build_match_team_assignments

    # GT side switches
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT id, player_matching_gt_json FROM videos
            WHERE id IN (SELECT DISTINCT video_id FROM rallies WHERE gt_serving_team IS NOT NULL)
        """)
        video_switches: dict[str, set[int]] = {}
        for row in cur.fetchall():
            vid = str(row[0])
            gt = row[1]
            sw = list(gt.get("sideSwitches", [])) if isinstance(gt, dict) else []
            video_switches[vid] = set(sw)

    # Production signals: team_assignments + semantic_flip from match_analysis
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT id, match_analysis_json FROM videos
            WHERE id IN (SELECT DISTINCT video_id FROM rallies WHERE gt_serving_team IS NOT NULL)
              AND match_analysis_json IS NOT NULL
        """)
        prod_teams: dict[str, dict[int, int]] = {}
        prod_flips: dict[str, bool] = {}
        for row in cur.fetchall():
            ma = row[1]
            if not isinstance(ma, dict):
                continue
            prod_teams.update(build_match_team_assignments(ma, 0.70))
            # Compute automated semantic_flip per rally
            count = 0
            for entry in ma.get("rallies") or []:
                rid = entry.get("rallyId") or entry.get("rally_id")
                if rid:
                    prod_flips[rid] = (count % 2 == 1)
                if entry.get("sideSwitchDetected") or entry.get("side_switch_detected"):
                    count += 1

    # Rally data
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute("""
            SELECT r.id, r.video_id, r.start_ms, r.gt_serving_team,
                   pt.positions_json, pt.court_split_y
            FROM rallies r LEFT JOIN player_tracks pt ON pt.rally_id = r.id
            WHERE r.video_id IN (SELECT DISTINCT video_id FROM rallies WHERE gt_serving_team IS NOT NULL)
              AND r.gt_serving_team IS NOT NULL
            ORDER BY r.video_id, r.start_ms
        """)
        raw: dict[str, list[tuple[Any, ...]]] = defaultdict(list)
        for row in cur.fetchall():
            raw[str(row[1])].append(row)

    out: dict[str, list[EvalRally]] = {}
    for vid, rows in raw.items():
        rows.sort(key=lambda r: r[2])
        switches = video_switches.get(vid, set())
        flipped = False
        vid_out: list[EvalRally] = []
        for idx, (rid, _, sms, gt, pj, split_y) in enumerate(rows):
            rid_str = str(rid)
            if idx in switches:
                flipped = not flipped

            # Compute production serving_team (formation + team_assignments + semantic_flip)
            positions = _parse_positions(pj or [])
            net_y = split_y if split_y else 0.5
            prod_team, _ = _find_serving_team_by_formation(
                positions, start_frame=0, net_y=net_y,
                team_assignments=prod_teams.get(rid_str),
                semantic_flip=prod_flips.get(rid_str, False),
            )

            vid_out.append(EvalRally(
                rally_id=rid_str,
                video_id=vid,
                start_ms=sms or 0,
                gt_serving_team=gt,
                positions=positions,
                court_split_y=split_y,
                rally_index=idx,
                side_flipped=flipped,
                prod_serving_team=prod_team,
                prod_semantic_flip=prod_flips.get(rid_str, False),
            ))
        out[vid] = vid_out
    return out


def _gt_team_to_side(gt_team: str, side_flipped: bool) -> str:
    if not side_flipped:
        return "near" if gt_team == "A" else "far"
    return "far" if gt_team == "A" else "near"


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _eval_config(
    video_rallies: dict[str, list[EvalRally]],
    use_first_serve: bool = False,
    use_count_constraint: bool = False,
) -> tuple[int, int, dict[str, dict[str, int]]]:
    """Run Viterbi with a specific anchor configuration. Returns (correct, total, per_video)."""
    correct = 0
    total = 0
    per_video: dict[str, dict[str, int]] = {}

    for vid, rallies in sorted(video_rallies.items()):
        observations: list[RallyObservation] = []
        gt_teams: list[str | None] = []

        for rally in rallies:
            net_y = rally.court_split_y or 0.5
            formation_side, formation_conf = _find_serving_side_by_formation(
                rally.positions, net_y=net_y, start_frame=0,
            )
            observations.append(RallyObservation(
                rally_id=rally.rally_id,
                formation_side=formation_side,
                formation_confidence=formation_conf,
                gt_serving_team=rally.gt_serving_team,
            ))
            gt_teams.append(rally.gt_serving_team)

        # Per-video calibration
        initial_near_is_a = calibrate_initial_side(observations, gt_teams)

        # GT side switches
        switch_indices: set[int] = set()
        for i in range(1, len(rallies)):
            if rallies[i].side_flipped != rallies[i - 1].side_flipped:
                switch_indices.add(i)

        # Convert GT team → physical side using calibrated mapping.
        # Must account for initial_near_is_a + side switches.
        def gt_to_physical(rally_idx: int, gt_team: str) -> str:
            # Compute near_is_a at this rally by replaying side switches.
            # Switch at index j means the mapping changes BEFORE rally j.
            near_is_a = initial_near_is_a
            for j in range(rally_idx):
                if j + 1 in switch_indices:
                    near_is_a = not near_is_a
            if near_is_a:
                return "near" if gt_team == "A" else "far"
            return "far" if gt_team == "A" else "near"

        # Build anchors
        first_serve_side: str | None = None
        if use_first_serve:
            first_serve_side = gt_to_physical(0, rallies[0].gt_serving_team)

        n_near_target: int | None = None
        if use_count_constraint:
            n_near_target = sum(
                1 for i, r in enumerate(rallies)
                if gt_to_physical(i, r.gt_serving_team) == "near"
            )

        decoded = decode_video(
            observations,
            initial_near_is_a=initial_near_is_a,
            side_switch_rallies=switch_indices,
            first_serve_side=first_serve_side,
            n_near_target=n_near_target,
        )

        vid_correct = 0
        for rally, dec in zip(rallies, decoded):
            total += 1
            if dec.serving_team == rally.gt_serving_team:
                correct += 1
                vid_correct += 1

        per_video[vid] = {"total": len(rallies), "correct": vid_correct}

    return correct, total, per_video


def _eval_selfcal(
    video_rallies: dict[str, list[EvalRally]],
) -> tuple[int, int, dict[str, dict[str, int]]]:
    """Viterbi with self-calibration from noisy production predictions."""
    correct = 0
    total = 0
    per_video: dict[str, dict[str, int]] = {}

    for vid, rallies in sorted(video_rallies.items()):
        observations: list[RallyObservation] = []
        noisy_teams: list[str | None] = []

        for rally in rallies:
            net_y = rally.court_split_y or 0.5
            formation_side, formation_conf = _find_serving_side_by_formation(
                rally.positions, net_y=net_y, start_frame=0,
            )
            observations.append(RallyObservation(
                rally_id=rally.rally_id,
                formation_side=formation_side,
                formation_confidence=formation_conf,
            ))
            noisy_teams.append(rally.prod_serving_team)

        # Calibrate from noisy production predictions (no GT!)
        initial_near_is_a = calibrate_from_noisy_predictions(observations, noisy_teams)

        # Use automated side switches from match_analysis
        switch_indices: set[int] = set()
        for i in range(1, len(rallies)):
            if rallies[i].prod_semantic_flip != rallies[i - 1].prod_semantic_flip:
                switch_indices.add(i)

        decoded = decode_video(
            observations,
            initial_near_is_a=initial_near_is_a,
            side_switch_rallies=switch_indices,
        )

        vid_correct = 0
        for rally, dec in zip(rallies, decoded):
            total += 1
            if dec.serving_team == rally.gt_serving_team:
                correct += 1
                vid_correct += 1
        per_video[vid] = {"total": len(rallies), "correct": vid_correct}

    return correct, total, per_video


def _eval_dual_hypothesis(
    video_rallies: dict[str, list[EvalRally]],
) -> tuple[int, int, dict[str, dict[str, int]]]:
    """Viterbi trying both near=A and near=B, picks more plausible score."""
    correct = 0
    total = 0
    per_video: dict[str, dict[str, int]] = {}

    for vid, rallies in sorted(video_rallies.items()):
        observations: list[RallyObservation] = []

        for rally in rallies:
            net_y = rally.court_split_y or 0.5
            formation_side, formation_conf = _find_serving_side_by_formation(
                rally.positions, net_y=net_y, start_frame=0,
            )
            observations.append(RallyObservation(
                rally_id=rally.rally_id,
                formation_side=formation_side,
                formation_confidence=formation_conf,
            ))

        # Use automated side switches
        switch_indices: set[int] = set()
        for i in range(1, len(rallies)):
            if rallies[i].prod_semantic_flip != rallies[i - 1].prod_semantic_flip:
                switch_indices.add(i)

        decoded = decode_video_dual_hypothesis(
            observations, side_switch_rallies=switch_indices,
        )

        vid_correct = 0
        for rally, dec in zip(rallies, decoded):
            total += 1
            if dec.serving_team == rally.gt_serving_team:
                correct += 1
                vid_correct += 1
        per_video[vid] = {"total": len(rallies), "correct": vid_correct}

    return correct, total, per_video


def main() -> int:
    print("=" * 70)
    print("Production-Viable Calibration: Self-Cal vs Dual-Hypothesis")
    print("=" * 70)

    print("\nLoading eval data...")
    video_rallies = _load_eval_data()
    total_rallies = sum(len(v) for v in video_rallies.values())
    print(f"  {len(video_rallies)} videos, {total_rallies} rallies\n")

    print(f"{'Config':<45s}  {'Correct':>7s}  {'Total':>5s}  {'Accuracy':>8s}")
    print(f"{'-' * 45}  {'-' * 7}  {'-' * 5}  {'-' * 8}")

    all_results: dict[str, dict[str, Any]] = {}
    all_per_video: dict[str, dict[str, dict[str, int]]] = {}

    # 1. GT-calibrated (oracle ceiling)
    c, t, pv = _eval_config(video_rallies)
    acc = c / max(t, 1) * 100
    print(f"  {'GT-calibrated (oracle ceiling)':<43s}  {c:7d}  {t:5d}  {acc:7.1f}%")
    all_results["gt_calibrated"] = {"correct": c, "total": t, "accuracy": acc}
    all_per_video["gt_calibrated"] = pv

    # 2. Self-cal from noisy production predictions
    c, t, pv = _eval_selfcal(video_rallies)
    acc = c / max(t, 1) * 100
    print(f"  {'Self-cal (noisy prod predictions)':<43s}  {c:7d}  {t:5d}  {acc:7.1f}%")
    all_results["self_cal"] = {"correct": c, "total": t, "accuracy": acc}
    all_per_video["self_cal"] = pv

    # 3. Dual hypothesis (no external calibration)
    c, t, pv = _eval_dual_hypothesis(video_rallies)
    acc = c / max(t, 1) * 100
    print(f"  {'Dual-hypothesis (plausibility)':<43s}  {c:7d}  {t:5d}  {acc:7.1f}%")
    all_results["dual_hyp"] = {"correct": c, "total": t, "accuracy": acc}
    all_per_video["dual_hyp"] = pv

    # 4. Baseline: current production (no Viterbi)
    c_prod = sum(
        1 for rallies in video_rallies.values()
        for r in rallies if r.prod_serving_team == r.gt_serving_team
    )
    t_prod = sum(
        1 for rallies in video_rallies.values()
        for r in rallies if r.prod_serving_team is not None
    )
    acc_prod = c_prod / max(t_prod, 1) * 100
    print(f"\n  {'Current production (no Viterbi)':<43s}  {c_prod:7d}  {t_prod:5d}  {acc_prod:7.1f}%")
    print("  (Canonical score_accuracy baseline: 57.6% on 92-rally subset)")

    # Per-video comparison
    print(f"\n{'=' * 70}")
    print("PER-VIDEO COMPARISON")
    print(f"{'=' * 70}")
    print(f"  {'video':>10s}  {'n':>3s}  {'GT-cal':>6s}  {'self':>6s}  {'dual':>6s}  {'prod':>6s}")
    print(f"  {'-' * 10}  {'-' * 3}  {'-' * 6}  {'-' * 6}  {'-' * 6}  {'-' * 6}")

    for vid in sorted(video_rallies.keys()):
        n = len(video_rallies[vid])
        gt_acc = all_per_video["gt_calibrated"][vid]["correct"] / n * 100
        sc_acc = all_per_video["self_cal"][vid]["correct"] / n * 100
        dh_acc = all_per_video["dual_hyp"][vid]["correct"] / n * 100
        prod_c = sum(1 for r in video_rallies[vid] if r.prod_serving_team == r.gt_serving_team)
        pr_acc = prod_c / n * 100
        print(f"  {vid[:10]}  {n:3d}  {gt_acc:5.1f}%  {sc_acc:5.1f}%  {dh_acc:5.1f}%  {pr_acc:5.1f}%")

    # ---- Greedy cascade simulation ----
    print(f"\n{'=' * 70}")
    print("CASCADE SIMULATION: greedy user corrections")
    print(f"{'=' * 70}")
    print("  (Each step: correct the rally that improves accuracy the most)\n")

    # Run per-video greedy correction
    total_correct_after = 0
    total_n = 0

    print(f"  {'video':>10s}  {'n':>3s}  {'base':>5s}  {'1 fix':>5s}  "
          f"{'3 fix':>5s}  {'5 fix':>5s}  {'to95%':>5s}")
    print(f"  {'-' * 10}  {'-' * 3}  {'-' * 5}  {'-' * 5}  "
          f"{'-' * 5}  {'-' * 5}  {'-' * 5}")

    for vid, rallies in sorted(video_rallies.items()):
        observations: list[RallyObservation] = []
        gt_teams: list[str | None] = []
        for rally in rallies:
            net_y = rally.court_split_y or 0.5
            fs, fc = _find_serving_side_by_formation(
                rally.positions, net_y=net_y, start_frame=0,
            )
            observations.append(RallyObservation(
                rally_id=rally.rally_id,
                formation_side=fs,
                formation_confidence=fc,
                gt_serving_team=rally.gt_serving_team,
            ))
            gt_teams.append(rally.gt_serving_team)

        initial_near_is_a = calibrate_initial_side(observations, gt_teams)
        switch_indices: set[int] = set()
        for i in range(1, len(rallies)):
            if rallies[i].side_flipped != rallies[i - 1].side_flipped:
                switch_indices.add(i)

        def gt_to_phys(idx: int, gt_team: str) -> str:
            nia = initial_near_is_a
            for j in range(idx):
                if j + 1 in switch_indices:
                    nia = not nia
            return "near" if (gt_team == "A") == nia else "far"

        # Baseline (no corrections)
        decoded = decode_video(
            observations, initial_near_is_a=initial_near_is_a,
            side_switch_rallies=switch_indices,
        )
        base_correct = sum(
            1 for r, d in zip(rallies, decoded)
            if d.serving_team == r.gt_serving_team
        )

        # Greedy corrections
        user_anchors: dict[int, str] = {}
        accuracies = [base_correct / len(rallies) * 100]
        corrections_to_95 = -1

        for step in range(min(20, len(rallies))):
            # Find the rally correction that gives the best accuracy gain
            best_gain = 0
            best_idx = -1
            for i in range(len(rallies)):
                if i in user_anchors:
                    continue
                trial = dict(user_anchors)
                trial[i] = gt_to_phys(i, rallies[i].gt_serving_team)
                dec_trial = decode_video(
                    observations, initial_near_is_a=initial_near_is_a,
                    side_switch_rallies=switch_indices, anchors=trial,
                )
                trial_correct = sum(
                    1 for r, d in zip(rallies, dec_trial)
                    if d.serving_team == r.gt_serving_team
                )
                if trial_correct > best_gain:
                    best_gain = trial_correct
                    best_idx = i

            if best_idx < 0:
                break
            user_anchors[best_idx] = gt_to_phys(
                best_idx, rallies[best_idx].gt_serving_team,
            )
            dec_new = decode_video(
                observations, initial_near_is_a=initial_near_is_a,
                side_switch_rallies=switch_indices, anchors=user_anchors,
            )
            new_correct = sum(
                1 for r, d in zip(rallies, dec_new)
                if d.serving_team == r.gt_serving_team
            )
            accuracies.append(new_correct / len(rallies) * 100)
            if corrections_to_95 < 0 and accuracies[-1] >= 95.0:
                corrections_to_95 = len(user_anchors)
            if new_correct == len(rallies):
                break

        # Report per-video
        acc_1 = accuracies[1] if len(accuracies) > 1 else accuracies[0]
        acc_3 = accuracies[3] if len(accuracies) > 3 else accuracies[-1]
        acc_5 = accuracies[5] if len(accuracies) > 5 else accuracies[-1]
        to95 = str(corrections_to_95) if corrections_to_95 > 0 else ">20"
        print(f"  {vid[:10]}  {len(rallies):3d}  "
              f"{accuracies[0]:4.0f}%  {acc_1:4.0f}%  "
              f"{acc_3:4.0f}%  {acc_5:4.0f}%  {to95:>5s}")

        total_n += len(rallies)
        total_correct_after += (
            int(accuracies[min(5, len(accuracies) - 1)] * len(rallies) / 100)
        )

    print(f"\n  Overall after <=5 corrections/video: "
          f"~{total_correct_after}/{total_n} "
          f"= ~{total_correct_after / max(total_n, 1) * 100:.1f}%")

    # Save results
    output_path = REPO / "outputs" / "score_viterbi_eval.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
