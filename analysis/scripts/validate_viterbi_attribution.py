"""Validate Viterbi-based joint team+player attribution.

Uses a Hidden Markov Model where:
- States: (team, player, touch_count) = 12 states
  - 2 teams × 2 players per team × 3 touch counts
- Emissions: distance from each player to the ball at each contact
- Transitions: volleyball rules (1→2→3→flip, alternating players)

Jointly optimizes team and player assignment across the entire contact
sequence, avoiding the circular dependency of per-contact decisions.

Usage:
    cd analysis
    uv run python scripts/validate_viterbi_attribution.py
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from rich.console import Console
from rich.table import Table

from rallycut.court.calibration import CourtCalibrator
from rallycut.evaluation.tracking.db import load_court_calibration
from rallycut.tracking.action_classifier import classify_rally_actions
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.contact_detector import (
    Contact,
    ContactDetectionConfig,
    detect_contacts,
)
from rallycut.tracking.player_filter import classify_teams
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos

from scripts.eval_action_detection import (
    load_rallies_with_action_gt,
    match_contacts,
)

console = Console()


@dataclass
class ViterbiState:
    """A state in the HMM: which team, which player, touch count on side."""

    team: int  # 0=near, 1=far
    track_id: int  # Player track ID
    touch_count: int  # 1, 2, or 3

    def __hash__(self) -> int:
        return hash((self.team, self.track_id, self.touch_count))


def viterbi_attribute(
    contacts: list[Contact],
    per_rally_teams: dict[int, int],
    distance_sigma: float = 0.10,
    same_player_penalty: float = -5.0,
    touch_overflow_penalty: float = -3.0,
) -> list[int]:
    """Viterbi decode the most likely player sequence for a rally.

    Args:
        contacts: Contact sequence.
        per_rally_teams: track_id -> team (0=near, 1=far).
        distance_sigma: Scale for distance-based emission (higher = more tolerant).
        same_player_penalty: Log-probability penalty for same player consecutive
            on same team (double contact is rare in volleyball).
        touch_overflow_penalty: Log-probability penalty for exceeding 3 touches.

    Returns:
        List of track_ids (one per contact) for the most likely assignment.
    """
    if not contacts:
        return []

    # Build state space
    team_players: dict[int, list[int]] = {0: [], 1: []}
    for tid, team in per_rally_teams.items():
        if team in team_players:
            team_players[team].append(tid)

    # Need at least 1 player per team
    if not team_players[0] or not team_players[1]:
        return [c.player_track_id for c in contacts]

    # Limit to 2 players per team (take the ones most often seen)
    for team in [0, 1]:
        if len(team_players[team]) > 2:
            team_players[team] = team_players[team][:2]

    states: list[ViterbiState] = []
    for team in [0, 1]:
        for tid in team_players[team]:
            for tc in [1, 2, 3]:
                states.append(ViterbiState(team=team, track_id=tid, touch_count=tc))

    n_states = len(states)
    n_contacts = len(contacts)

    # Build candidate distance lookup per contact
    # contact.player_candidates is sorted by distance
    candidate_dists: list[dict[int, float]] = []
    for contact in contacts:
        dists: dict[int, float] = {}
        for tid, dist in contact.player_candidates:
            dists[tid] = dist
        # Add a large distance for players not in candidates
        candidate_dists.append(dists)

    # Emission probabilities: log P(observation | state)
    # Based on distance from player to ball
    def log_emission(contact_idx: int, state: ViterbiState) -> float:
        dist = candidate_dists[contact_idx].get(state.track_id, 0.5)
        return -(dist / distance_sigma) ** 2

    # Transition probabilities: log P(next_state | curr_state)
    def log_transition(curr: ViterbiState, nxt: ViterbiState) -> float:
        log_p = 0.0

        if curr.team == nxt.team:
            # Same team — sequential touches
            if nxt.touch_count == curr.touch_count + 1:
                # Normal progression (1→2, 2→3)
                if curr.track_id == nxt.track_id:
                    log_p += same_player_penalty  # Same player consecutive = rare
                else:
                    log_p += 0.0  # Normal alternation
            elif nxt.touch_count == 1 and curr.touch_count == 3:
                # Touch overflow: 3→1 on same team (shouldn't happen normally,
                # but could happen if we detected a block or FP contact)
                log_p += touch_overflow_penalty
            elif nxt.touch_count == curr.touch_count:
                # Same touch count, same team — unusual
                log_p += -4.0
            else:
                # Invalid progression (e.g., 1→3, 2→1 on same team)
                log_p += -8.0
        else:
            # Different team — ball crossed net
            if curr.touch_count == 3 and nxt.touch_count == 1:
                # Normal attack→receive transition
                log_p += 0.0
            elif curr.touch_count < 3 and nxt.touch_count == 1:
                # Early crossing (e.g., free ball, tip) — possible but unusual
                log_p += -1.5
            elif nxt.touch_count == 1:
                # Block scenario: curr was attack (tc=3), block happened,
                # ball returns. Modeled as another tc=1 on blocking team.
                log_p += -1.0
            else:
                # Unusual team switch (not starting at tc=1)
                log_p += -6.0

        return log_p

    # Viterbi algorithm
    # V[t][s] = log probability of best path ending at state s at time t
    V = np.full((n_contacts, n_states), -np.inf)
    backptr = np.zeros((n_contacts, n_states), dtype=int)

    # Initialize: first contact (serve) — prefer tc=1
    for si, state in enumerate(states):
        if state.touch_count == 1:
            V[0][si] = log_emission(0, state)
        else:
            V[0][si] = log_emission(0, state) - 10.0  # Strong penalty for non-serve start

    # Forward pass
    for t in range(1, n_contacts):
        for sj, nxt_state in enumerate(states):
            best_log_p = -np.inf
            best_prev = 0
            emit = log_emission(t, nxt_state)

            for si, curr_state in enumerate(states):
                log_p = V[t - 1][si] + log_transition(curr_state, nxt_state) + emit
                if log_p > best_log_p:
                    best_log_p = log_p
                    best_prev = si

            V[t][sj] = best_log_p
            backptr[t][sj] = best_prev

    # Backtrace
    best_final = int(np.argmax(V[n_contacts - 1]))
    path = [best_final]
    for t in range(n_contacts - 1, 0, -1):
        path.append(backptr[t][path[-1]])
    path.reverse()

    return [states[s].track_id for s in path]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rally", type=str)
    parser.add_argument("--tolerance-ms", type=int, default=167)
    parser.add_argument("--distance-sigma", type=float, default=0.10)
    parser.add_argument("--same-player-penalty", type=float, default=-5.0)
    parser.add_argument("--touch-overflow-penalty", type=float, default=-3.0)
    args = parser.parse_args()

    rallies = load_rallies_with_action_gt(rally_id=args.rally)
    if not rallies:
        console.print("[red]No rallies with action GT.[/red]")
        return

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

    console.print(
        f"\n[bold]Validating Viterbi attribution across {len(rallies)} rallies[/bold]"
        f"\n  sigma={args.distance_sigma}, same_player_pen={args.same_player_penalty}, "
        f"overflow_pen={args.touch_overflow_penalty}\n",
    )

    total = 0
    current_correct = 0
    viterbi_correct = 0
    current_team_correct = 0
    viterbi_team_correct = 0
    # Per-action
    per_action: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    # Track fixes and regressions
    fixes = 0  # Viterbi correct where current was wrong
    regressions = 0  # Viterbi wrong where current was correct

    for i, rally in enumerate(rallies):
        if not rally.ball_positions_json or not rally.positions_json:
            continue

        cal = calibrators.get(rally.video_id)

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
        ball_positions = [
            BallPos(
                frame_number=bp["frameNumber"],
                x=bp["x"], y=bp["y"],
                confidence=bp.get("confidence", 1.0),
            )
            for bp in rally.ball_positions_json
            if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
        ]

        net_y_val = rally.court_split_y or 0.5
        per_rally_teams = classify_teams(positions, net_y_val)

        contact_seq = detect_contacts(
            ball_positions=ball_positions,
            player_positions=positions,
            config=ContactDetectionConfig(),
            net_y=rally.court_split_y,
            frame_count=rally.frame_count or None,
            court_calibrator=cal,
        )
        contacts = contact_seq.contacts

        if not contacts:
            continue

        # Run Viterbi attribution
        viterbi_tids = viterbi_attribute(
            contacts, per_rally_teams,
            distance_sigma=args.distance_sigma,
            same_player_penalty=args.same_player_penalty,
            touch_overflow_penalty=args.touch_overflow_penalty,
        )

        # Build lookup
        contact_by_frame: dict[int, tuple[Contact, int]] = {}
        for ci, (contact, vtid) in enumerate(zip(contacts, viterbi_tids)):
            contact_by_frame[contact.frame] = (contact, vtid)

        # Classify actions for baseline
        rally_actions = classify_rally_actions(contact_seq, rally.rally_id)
        pred_actions_list = [a.to_dict() for a in rally_actions.actions]
        real_pred = [a for a in pred_actions_list if not a.get("isSynthetic")]

        fps = rally.fps or 30.0
        tol = max(1, round(fps * args.tolerance_ms / 1000))
        avail_tids = {pp["trackId"] for pp in rally.positions_json}

        matches, _ = match_contacts(
            rally.gt_labels, real_pred,
            tolerance=tol, available_track_ids=avail_tids,
        )

        for m in matches:
            if m.pred_frame is None:
                continue

            gt_tid = next(
                (gt.player_track_id for gt in rally.gt_labels if gt.frame == m.gt_frame),
                -1,
            )
            if gt_tid < 0 or gt_tid not in avail_tids or gt_tid not in per_rally_teams:
                continue

            lookup = contact_by_frame.get(m.pred_frame)
            if lookup is None:
                continue

            contact, vtid = lookup
            pred_tid = next(
                (a.get("playerTrackId", -1) for a in real_pred if a.get("frame") == m.pred_frame),
                -1,
            )

            gt_team = per_rally_teams[gt_tid]

            total += 1

            # Current system
            cur_ok = pred_tid == gt_tid
            if cur_ok:
                current_correct += 1
            pred_team = per_rally_teams.get(pred_tid)
            if pred_team == gt_team:
                current_team_correct += 1

            # Viterbi system
            vit_ok = vtid == gt_tid
            if vit_ok:
                viterbi_correct += 1
            vit_team = per_rally_teams.get(vtid)
            if vit_team == gt_team:
                viterbi_team_correct += 1

            # Track fixes and regressions
            if vit_ok and not cur_ok:
                fixes += 1
            elif not vit_ok and cur_ok:
                regressions += 1

            # Per-action
            per_action[m.gt_action]["total"] += 1
            if cur_ok:
                per_action[m.gt_action]["cur_correct"] += 1
            if vit_ok:
                per_action[m.gt_action]["vit_correct"] += 1

        if (i + 1) % 20 == 0:
            console.print(f"  [{i + 1}/{len(rallies)}] {total} contacts, "
                          f"cur={current_correct}/{total} vit={viterbi_correct}/{total}")

    # Report
    console.print(f"\n[bold]Results: {total} evaluable contacts[/bold]\n")
    console.print(f"  Current attribution: {current_correct}/{total} = {current_correct / max(1, total):.1%}")
    console.print(f"  Viterbi attribution: {viterbi_correct}/{total} = {viterbi_correct / max(1, total):.1%}")
    console.print(f"  Current team accuracy: {current_team_correct}/{total} = {current_team_correct / max(1, total):.1%}")
    console.print(f"  Viterbi team accuracy: {viterbi_team_correct}/{total} = {viterbi_team_correct / max(1, total):.1%}")
    console.print(f"\n  Fixes (viterbi right, current wrong): {fixes}")
    console.print(f"  Regressions (viterbi wrong, current right): {regressions}")
    console.print(f"  Net improvement: {fixes - regressions}")

    # Per-action
    console.print(f"\n[bold cyan]Per-action breakdown[/bold cyan]")
    act_table = Table()
    act_table.add_column("Action")
    act_table.add_column("Current", justify="right")
    act_table.add_column("Viterbi", justify="right")
    act_table.add_column("Total", justify="right")
    act_table.add_column("Cur Acc", justify="right")
    act_table.add_column("Vit Acc", justify="right")
    act_table.add_column("Delta", justify="right")

    for action in ["serve", "receive", "set", "attack", "block", "dig"]:
        cc = per_action[action].get("cur_correct", 0)
        vc = per_action[action].get("vit_correct", 0)
        t = per_action[action].get("total", 0)
        if t > 0:
            delta = vc - cc
            style = "green" if delta > 0 else ("red" if delta < 0 else "")
            act_table.add_row(
                action, str(cc), str(vc), str(t),
                f"{cc / t:.1%}", f"{vc / t:.1%}",
                f"{delta:+d}",
                style=style,
            )
    console.print(act_table)


if __name__ == "__main__":
    main()
