"""Validate independent court_side signals against GT player team labels.

Tests three signals for determining which court side the ball is on,
without depending on player identity (breaking the circular dependency):

1. Y-threshold: ball_y >= net_y → "near" (team 0), ball_y < net_y → "far" (team 1)
2. Net crossing detection: ball_crossed_net() recall between contacts
3. Ball vertical velocity direction: vy sign at contact

For each signal, reports accuracy against the GT player's actual team
from match_team_assignments.

Also computes a ceiling analysis: if Y-threshold is correct AND we
constrain to the 2 same-team players, how often is the GT player the
nearest among same-team candidates?

Usage:
    cd analysis
    uv run python scripts/validate_court_side_signals.py
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass

from rich.console import Console
from rich.table import Table

from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.db import get_connection
from rallycut.evaluation.tracking.db import load_court_calibration
from rallycut.tracking.contact_detector import ball_crossed_net
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.contact_detector import (
    Contact,
    ContactDetectionConfig,
    _compute_velocities,
    detect_contacts,
)
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos

from scripts.eval_action_detection import (
    GtLabel,
    RallyData,
    _load_match_team_assignments,
    load_rallies_with_action_gt,
    match_contacts,
)

console = Console()


@dataclass
class SignalResult:
    """Result of validating a single contact's court_side signal."""

    rally_id: str
    frame: int
    gt_action: str
    gt_track_id: int
    gt_team: int  # 0=near, 1=far
    ball_y: float
    net_y: float

    # Signal 1: Y-threshold
    y_threshold_side: str  # "near" or "far"
    y_threshold_correct: bool

    # Signal 3: velocity direction
    vy: float | None  # vertical velocity at contact
    vy_sign: str  # "up" (vy < 0), "down" (vy > 0), "unknown"

    # Team-constrained ceiling
    gt_is_nearest_on_team: bool  # GT player is nearest among same-team players
    same_team_candidates: int  # how many same-team players in candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate independent court_side signals against GT",
    )
    parser.add_argument("--rally", type=str, help="Specific rally ID")
    parser.add_argument(
        "--tolerance-ms", type=int, default=167,
        help="Matching tolerance in ms (default: 167)",
    )
    args = parser.parse_args()

    rallies = load_rallies_with_action_gt(rally_id=args.rally)
    if not rallies:
        console.print("[red]No rallies found with action GT.[/red]")
        return

    # Load calibrations and match-level team assignments
    calibrators: dict[str, CourtCalibrator | None] = {}
    video_ids = {r.video_id for r in rallies}
    for vid in video_ids:
        corners = load_court_calibration(vid)
        if corners and len(corners) == 4:
            cal = CourtCalibrator()
            cal.calibrate([(c["x"], c["y"]) for c in corners])
            calibrators[vid] = cal
        else:
            calibrators[vid] = None

    match_teams_by_rally = _load_match_team_assignments(video_ids, min_confidence=0.70)

    console.print(
        f"\n[bold]Validating court_side signals across {len(rallies)} rallies[/bold]\n",
    )

    all_results: list[SignalResult] = []
    n_skipped_no_teams = 0
    n_net_crossing_gt = 0  # GT says ball crossed net (action transition)
    n_net_crossing_detected = 0  # ball_crossed_net() detected it
    n_net_crossing_checked = 0  # pairs where we could check

    for i, rally in enumerate(rallies):
        if not rally.ball_positions_json:
            continue

        teams = match_teams_by_rally.get(rally.rally_id)
        if not teams:
            n_skipped_no_teams += 1
            continue

        cal = calibrators.get(rally.video_id)

        ball_positions = [
            BallPos(
                frame_number=bp["frameNumber"],
                x=bp["x"],
                y=bp["y"],
                confidence=bp.get("confidence", 1.0),
            )
            for bp in rally.ball_positions_json
            if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
        ]

        player_positions: list[PlayerPos] = []
        if rally.positions_json:
            player_positions = [
                PlayerPos(
                    frame_number=pp["frameNumber"],
                    track_id=pp["trackId"],
                    x=pp["x"],
                    y=pp["y"],
                    width=pp["width"],
                    height=pp["height"],
                    confidence=pp.get("confidence", 1.0),
                )
                for pp in rally.positions_json
            ]

        # Re-run contact detection
        contact_seq = detect_contacts(
            ball_positions=ball_positions,
            player_positions=player_positions,
            config=ContactDetectionConfig(),
            net_y=rally.court_split_y,
            frame_count=rally.frame_count or None,
            court_calibrator=cal,
        )
        contacts = contact_seq.contacts
        net_y = contact_seq.net_y

        # Compute velocities for signal 3
        velocities = _compute_velocities(ball_positions)

        # Build contact lookup
        contact_by_frame: dict[int, Contact] = {c.frame: c for c in contacts}

        # Match to GT
        from rallycut.tracking.action_classifier import classify_rally_actions

        rally_actions = classify_rally_actions(
            contact_seq, rally.rally_id,
            match_team_assignments=teams,
        )
        pred_actions = [a.to_dict() for a in rally_actions.actions]
        real_pred = [a for a in pred_actions if not a.get("isSynthetic")]

        fps = rally.fps or 30.0
        tol = max(1, round(fps * args.tolerance_ms / 1000))
        avail_tids: set[int] | None = None
        if rally.positions_json:
            avail_tids = {pp["trackId"] for pp in rally.positions_json}

        matches, _ = match_contacts(
            rally.gt_labels, real_pred,
            tolerance=tol, available_track_ids=avail_tids,
        )

        # Signal 2: Net crossing between consecutive matched contacts
        # Check GT action transitions to determine expected crossings
        side_change_actions = {"serve", "attack"}
        matched_with_pred = [
            m for m in matches if m.pred_frame is not None
        ]
        matched_with_pred.sort(key=lambda m: m.gt_frame)

        for j in range(1, len(matched_with_pred)):
            prev_m = matched_with_pred[j - 1]
            curr_m = matched_with_pred[j]

            # GT says crossing if previous action is serve or attack
            gt_expects_crossing = prev_m.gt_action in side_change_actions

            if prev_m.pred_frame is not None and curr_m.pred_frame is not None:
                detected = ball_crossed_net(
                    ball_positions,
                    prev_m.pred_frame,
                    curr_m.pred_frame,
                    net_y,
                    min_frames_per_side=2,
                )

                if detected is not None:
                    n_net_crossing_checked += 1
                    if gt_expects_crossing:
                        n_net_crossing_gt += 1
                        if detected:
                            n_net_crossing_detected += 1

        # Process each matched contact for signals 1 and 3
        for m in matches:
            if m.pred_frame is None:
                continue

            gt_tid = next(
                (gt.player_track_id for gt in rally.gt_labels if gt.frame == m.gt_frame),
                -1,
            )
            if gt_tid < 0 or gt_tid not in teams:
                continue

            # Check if GT track_id exists in current tracking
            if avail_tids is not None and gt_tid not in avail_tids:
                continue  # stale GT

            gt_team = teams[gt_tid]  # 0=near, 1=far

            contact = contact_by_frame.get(m.pred_frame)
            if contact is None:
                continue

            ball_y = contact.ball_y

            # Signal 1: Y-threshold
            y_side = "near" if ball_y >= net_y else "far"
            y_team = 0 if y_side == "near" else 1
            y_correct = y_team == gt_team

            # Signal 3: Velocity direction
            vel = velocities.get(contact.frame)
            vy_val: float | None = None
            vy_sign = "unknown"
            if vel is not None:
                _, _, vy = vel
                vy_val = vy
                if abs(vy) > 0.001:
                    vy_sign = "down" if vy > 0 else "up"

            # Ceiling analysis: is GT player nearest among same-team candidates?
            gt_nearest_on_team = False
            same_team_count = 0
            if contact.player_candidates:
                same_team_cands = [
                    (tid, dist) for tid, dist in contact.player_candidates
                    if teams.get(tid) == gt_team
                ]
                same_team_count = len(same_team_cands)
                if same_team_cands:
                    nearest_on_team_tid = same_team_cands[0][0]
                    gt_nearest_on_team = nearest_on_team_tid == gt_tid

            all_results.append(SignalResult(
                rally_id=rally.rally_id,
                frame=m.gt_frame,
                gt_action=m.gt_action,
                gt_track_id=gt_tid,
                gt_team=gt_team,
                ball_y=ball_y,
                net_y=net_y,
                y_threshold_side=y_side,
                y_threshold_correct=y_correct,
                vy=vy_val,
                vy_sign=vy_sign,
                gt_is_nearest_on_team=gt_nearest_on_team,
                same_team_candidates=same_team_count,
            ))

        if (i + 1) % 20 == 0:
            console.print(f"  [{i + 1}/{len(rallies)}] {len(all_results)} contacts validated")

    # === Report ===
    console.print(f"\n[bold]Results: {len(all_results)} contacts with team data[/bold]")
    console.print(f"  Skipped {n_skipped_no_teams} rallies without match-level team assignments\n")

    # Signal 1: Y-threshold accuracy
    y_correct = sum(1 for r in all_results if r.y_threshold_correct)
    y_total = len(all_results)
    console.print(f"[bold cyan]Signal 1: Y-threshold (ball_y vs net_y)[/bold cyan]")
    console.print(f"  Overall: {y_correct}/{y_total} = {y_correct / max(1, y_total):.1%}")

    # Per-action breakdown
    action_table = Table(title="Y-threshold accuracy by action type")
    action_table.add_column("Action")
    action_table.add_column("Correct", justify="right")
    action_table.add_column("Total", justify="right")
    action_table.add_column("Accuracy", justify="right")
    action_table.add_column("Avg |ball_y - net_y|", justify="right")

    for action in ["serve", "receive", "set", "attack", "block", "dig"]:
        action_results = [r for r in all_results if r.gt_action == action]
        if not action_results:
            continue
        ac = sum(1 for r in action_results if r.y_threshold_correct)
        at = len(action_results)
        avg_margin = sum(abs(r.ball_y - r.net_y) for r in action_results) / at
        action_table.add_row(
            action, str(ac), str(at),
            f"{ac / at:.1%}", f"{avg_margin:.3f}",
        )
    console.print(action_table)

    # Margin analysis: accuracy by distance from net
    console.print("\n[bold cyan]Y-threshold accuracy by margin (|ball_y - net_y|)[/bold cyan]")
    margin_table = Table()
    margin_table.add_column("Margin range")
    margin_table.add_column("Correct", justify="right")
    margin_table.add_column("Total", justify="right")
    margin_table.add_column("Accuracy", justify="right")

    margins = [(0.0, 0.02), (0.02, 0.05), (0.05, 0.10), (0.10, 0.20), (0.20, 1.0)]
    for lo, hi in margins:
        in_range = [r for r in all_results if lo <= abs(r.ball_y - r.net_y) < hi]
        if not in_range:
            continue
        c = sum(1 for r in in_range if r.y_threshold_correct)
        t = len(in_range)
        margin_table.add_row(f"{lo:.2f}-{hi:.2f}", str(c), str(t), f"{c / t:.1%}")
    console.print(margin_table)

    # Signal 2: Net crossing
    console.print(f"\n[bold cyan]Signal 2: Net crossing detection[/bold cyan]")
    console.print(f"  GT crossings (serve/attack transitions): {n_net_crossing_gt}")
    console.print(f"  Detected by ball_crossed_net(): {n_net_crossing_detected}")
    crossing_recall = n_net_crossing_detected / max(1, n_net_crossing_gt)
    console.print(f"  Recall: {crossing_recall:.1%}")
    console.print(f"  Total pairs checked: {n_net_crossing_checked}")

    # Also test with min_frames_per_side=1 to see if that helps
    console.print("  [dim]Re-checking with min_frames_per_side=1...[/dim]")
    # Re-run with relaxed threshold
    n_gt2 = 0
    n_det2 = 0
    n_checked2 = 0
    for rally in rallies:
        if not rally.ball_positions_json:
            continue
        teams = match_teams_by_rally.get(rally.rally_id)
        if not teams:
            continue
        ball_positions = [
            BallPos(
                frame_number=bp["frameNumber"],
                x=bp["x"], y=bp["y"],
                confidence=bp.get("confidence", 1.0),
            )
            for bp in rally.ball_positions_json
            if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
        ]
        contact_seq = detect_contacts(
            ball_positions=ball_positions,
            player_positions=[
                PlayerPos(
                    frame_number=pp["frameNumber"],
                    track_id=pp["trackId"],
                    x=pp["x"], y=pp["y"],
                    width=pp["width"], height=pp["height"],
                    confidence=pp.get("confidence", 1.0),
                )
                for pp in (rally.positions_json or [])
            ],
            config=ContactDetectionConfig(),
            net_y=rally.court_split_y,
            frame_count=rally.frame_count or None,
        )
        net_y = contact_seq.net_y
        rally_actions = classify_rally_actions(
            contact_seq, rally.rally_id, match_team_assignments=teams,
        )
        pred_actions = [a.to_dict() for a in rally_actions.actions]
        real_pred = [a for a in pred_actions if not a.get("isSynthetic")]
        fps = rally.fps or 30.0
        tol = max(1, round(fps * args.tolerance_ms / 1000))
        matches, _ = match_contacts(rally.gt_labels, real_pred, tolerance=tol)
        matched_sorted = sorted(
            [m for m in matches if m.pred_frame is not None],
            key=lambda m: m.gt_frame,
        )
        for j in range(1, len(matched_sorted)):
            prev_m = matched_sorted[j - 1]
            curr_m = matched_sorted[j]
            gt_cross = prev_m.gt_action in side_change_actions
            detected = ball_crossed_net(
                ball_positions, prev_m.pred_frame, curr_m.pred_frame,
                net_y, min_frames_per_side=1,
            )
            if detected is not None:
                n_checked2 += 1
                if gt_cross:
                    n_gt2 += 1
                    if detected:
                        n_det2 += 1

    console.print(f"  With min_frames=1: {n_det2}/{n_gt2} = {n_det2 / max(1, n_gt2):.1%} recall")

    # Signal 3: Velocity direction
    console.print(f"\n[bold cyan]Signal 3: Ball vertical velocity at contact[/bold cyan]")
    with_vy = [r for r in all_results if r.vy is not None and r.vy_sign != "unknown"]
    console.print(f"  Contacts with measurable vy: {len(with_vy)}/{len(all_results)}")

    # For same-side actions (dig, set, receive): ball should go UP (vy < 0 in image coords)
    same_side = [r for r in with_vy if r.gt_action in ("dig", "set", "receive")]
    up_on_same = sum(1 for r in same_side if r.vy < 0)
    console.print(
        f"  Same-side actions (dig/set/receive): vy<0 (up) = {up_on_same}/{len(same_side)} "
        f"= {up_on_same / max(1, len(same_side)):.1%}",
    )

    # For crossing actions (attack, serve): ball should go DOWN (vy > 0) if near, UP if far
    crossing = [r for r in with_vy if r.gt_action in ("attack", "serve")]
    # After serve/attack from near side (team 0): ball goes up (toward far = lower y)
    # After serve/attack from far side (team 1): ball goes down (toward near = higher y)
    near_cross = [r for r in crossing if r.gt_team == 0]
    far_cross = [r for r in crossing if r.gt_team == 1]
    near_up = sum(1 for r in near_cross if r.vy < 0)
    far_down = sum(1 for r in far_cross if r.vy > 0)
    console.print(
        f"  Near-team serve/attack: vy<0 (toward far) = {near_up}/{len(near_cross)} "
        f"= {near_up / max(1, len(near_cross)):.1%}",
    )
    console.print(
        f"  Far-team serve/attack: vy>0 (toward near) = {far_down}/{len(far_cross)} "
        f"= {far_down / max(1, len(far_cross)):.1%}",
    )

    # === Ceiling Analysis ===
    console.print(f"\n[bold green]Ceiling Analysis: Team-Constrained Player Selection[/bold green]")

    # If Y-threshold correct → constrain to same-team → is GT player the nearest?
    y_correct_results = [r for r in all_results if r.y_threshold_correct]
    y_correct_and_nearest = sum(1 for r in y_correct_results if r.gt_is_nearest_on_team)
    y_correct_with_cands = [r for r in y_correct_results if r.same_team_candidates > 0]

    console.print(
        f"  Y-threshold correct: {len(y_correct_results)}/{len(all_results)} "
        f"= {len(y_correct_results) / max(1, len(all_results)):.1%}",
    )
    console.print(
        f"  + GT nearest on team: {y_correct_and_nearest}/{len(y_correct_with_cands)} "
        f"= {y_correct_and_nearest / max(1, len(y_correct_with_cands)):.1%}",
    )

    # Overall ceiling = Y-threshold correct AND GT nearest on team
    # Plus: Y-threshold wrong but still get correct player (e.g., missing_player cases)
    ceiling = y_correct_and_nearest
    console.print(
        f"  [bold]Projected attribution accuracy: {ceiling}/{len(all_results)} "
        f"= {ceiling / max(1, len(all_results)):.1%}[/bold]",
    )

    # Breakdown of failures in ceiling
    y_wrong = [r for r in all_results if not r.y_threshold_correct]
    y_correct_wrong_nearest = [
        r for r in y_correct_results
        if not r.gt_is_nearest_on_team and r.same_team_candidates > 0
    ]
    y_correct_no_cands = [r for r in y_correct_results if r.same_team_candidates == 0]
    console.print(f"\n  Ceiling failure breakdown:")
    console.print(f"    Y-threshold wrong: {len(y_wrong)}")
    console.print(f"    Y correct but GT not nearest on team: {len(y_correct_wrong_nearest)}")
    console.print(f"    Y correct but no same-team candidates: {len(y_correct_no_cands)}")

    # Dead zone analysis: contacts where ball_y is close to net_y
    dead_zone = 0.03
    in_dead_zone = [r for r in all_results if abs(r.ball_y - r.net_y) < dead_zone]
    dz_correct = sum(1 for r in in_dead_zone if r.y_threshold_correct)
    console.print(
        f"\n  Dead zone (|ball_y - net_y| < {dead_zone}): "
        f"{len(in_dead_zone)} contacts, {dz_correct}/{len(in_dead_zone)} correct "
        f"= {dz_correct / max(1, len(in_dead_zone)):.1%}",
    )

    # === Per-rally consistency check ===
    # If Y-threshold is consistently right or wrong within a rally,
    # it's a mapping issue. If random, Y-threshold is truly unreliable.
    console.print(f"\n[bold yellow]Per-rally Y-threshold consistency[/bold yellow]")
    rally_results: dict[str, list[bool]] = defaultdict(list)
    for r in all_results:
        rally_results[r.rally_id].append(r.y_threshold_correct)

    consistent_right = 0
    consistent_wrong = 0
    mixed = 0
    for rid, corrections in rally_results.items():
        if not corrections:
            continue
        acc = sum(corrections) / len(corrections)
        if acc >= 0.8:
            consistent_right += 1
        elif acc <= 0.2:
            consistent_wrong += 1
        else:
            mixed += 1

    console.print(f"  Rallies consistently right (>=80%): {consistent_right}")
    console.print(f"  Rallies consistently wrong (<=20%): {consistent_wrong}")
    console.print(f"  Rallies mixed: {mixed}")
    console.print(f"  Total: {len(rally_results)}")

    # If many are consistently wrong, it's a team mapping issue
    if consistent_wrong > 0:
        console.print(
            f"\n  [bold]With auto-inversion (flip rallies where Y-thresh <20%):[/bold]",
        )
        corrected = 0
        for r in all_results:
            rally_acc = sum(rally_results[r.rally_id]) / len(rally_results[r.rally_id])
            if rally_acc <= 0.2:
                # Invert for this rally
                corrected += not r.y_threshold_correct
            else:
                corrected += r.y_threshold_correct
        console.print(f"  Corrected accuracy: {corrected}/{len(all_results)} = {corrected / max(1, len(all_results)):.1%}")

    # === Per-rally team_assignments from median Y ===
    # Compare match_team_assignments vs per-rally classify_teams
    console.print(f"\n[bold yellow]Per-rally team check: median Y vs match teams[/bold yellow]")
    from rallycut.tracking.player_filter import classify_teams

    n_agree = 0
    n_disagree = 0
    n_comparable = 0

    for rally in rallies:
        teams = match_teams_by_rally.get(rally.rally_id)
        if not teams or not rally.positions_json:
            continue

        positions = [
            PlayerPos(
                frame_number=pp["frameNumber"],
                track_id=pp["trackId"],
                x=pp["x"], y=pp["y"],
                width=pp["width"], height=pp["height"],
                confidence=pp.get("confidence", 1.0),
            )
            for pp in rally.positions_json
        ]

        net_y = rally.court_split_y or 0.5
        per_rally_teams = classify_teams(positions, net_y)

        # Compare: for each track in both dicts, do they agree?
        for tid in set(teams.keys()) & set(per_rally_teams.keys()):
            n_comparable += 1
            if teams[tid] == per_rally_teams[tid]:
                n_agree += 1
            else:
                n_disagree += 1

    console.print(f"  Match teams vs per-rally median Y: {n_agree} agree, {n_disagree} disagree (of {n_comparable})")
    if n_comparable > 0:
        console.print(f"  Agreement rate: {n_agree / n_comparable:.1%}")

    # === Test with per-rally teams instead of match teams ===
    console.print(f"\n[bold yellow]Y-threshold with per-rally teams (median Y)[/bold yellow]")
    per_rally_correct = 0
    per_rally_total = 0
    per_rally_nearest = 0
    per_rally_nearest_total = 0

    for rally in rallies:
        if not rally.ball_positions_json or not rally.positions_json:
            continue

        positions = [
            PlayerPos(
                frame_number=pp["frameNumber"],
                track_id=pp["trackId"],
                x=pp["x"], y=pp["y"],
                width=pp["width"], height=pp["height"],
                confidence=pp.get("confidence", 1.0),
            )
            for pp in rally.positions_json
        ]

        net_y_val = rally.court_split_y or 0.5
        per_rally_teams = classify_teams(positions, net_y_val)

        # Get contacts and match to GT
        ball_positions = [
            BallPos(
                frame_number=bp["frameNumber"],
                x=bp["x"], y=bp["y"],
                confidence=bp.get("confidence", 1.0),
            )
            for bp in rally.ball_positions_json
            if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
        ]

        contact_seq = detect_contacts(
            ball_positions=ball_positions,
            player_positions=positions,
            config=ContactDetectionConfig(),
            net_y=rally.court_split_y,
            frame_count=rally.frame_count or None,
        )
        contacts = contact_seq.contacts
        net_y = contact_seq.net_y

        contact_by_frame = {c.frame: c for c in contacts}

        from rallycut.tracking.action_classifier import classify_rally_actions
        rally_actions = classify_rally_actions(contact_seq, rally.rally_id)
        pred_actions = [a.to_dict() for a in rally_actions.actions]
        real_pred = [a for a in pred_actions if not a.get("isSynthetic")]

        fps = rally.fps or 30.0
        tol = max(1, round(fps * args.tolerance_ms / 1000))
        avail_tids = {pp["trackId"] for pp in rally.positions_json}

        matches, _ = match_contacts(rally.gt_labels, real_pred, tolerance=tol, available_track_ids=avail_tids)

        for m in matches:
            if m.pred_frame is None:
                continue
            gt_tid = next(
                (gt.player_track_id for gt in rally.gt_labels if gt.frame == m.gt_frame),
                -1,
            )
            if gt_tid < 0 or gt_tid not in per_rally_teams:
                continue
            if gt_tid not in avail_tids:
                continue

            gt_team = per_rally_teams[gt_tid]
            contact = contact_by_frame.get(m.pred_frame)
            if contact is None:
                continue

            y_side = "near" if contact.ball_y >= net_y else "far"
            y_team = 0 if y_side == "near" else 1
            y_correct = y_team == gt_team
            per_rally_total += 1
            if y_correct:
                per_rally_correct += 1

            # Ceiling: GT nearest on team?
            if contact.player_candidates:
                same_team_cands = [
                    (tid, dist) for tid, dist in contact.player_candidates
                    if per_rally_teams.get(tid) == gt_team
                ]
                if same_team_cands:
                    per_rally_nearest_total += 1
                    if same_team_cands[0][0] == gt_tid:
                        per_rally_nearest += 1

    console.print(
        f"  Y-threshold accuracy: {per_rally_correct}/{per_rally_total} "
        f"= {per_rally_correct / max(1, per_rally_total):.1%}",
    )
    console.print(
        f"  GT nearest on correct team: {per_rally_nearest}/{per_rally_nearest_total} "
        f"= {per_rally_nearest / max(1, per_rally_nearest_total):.1%}",
    )


if __name__ == "__main__":
    main()
