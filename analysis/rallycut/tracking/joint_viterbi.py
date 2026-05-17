"""Joint Viterbi over (action_type, player_track_id) — task #72.

Env-var probe support (2026-05-17): if `JOINT_VITERBI_TRANSITIONS_PATH`
is set, load empirical transitions from that JSON file (overrides the
hand-coded `_TYPICAL_NEXT_GIVEN_PREV` defaults). Used to validate
whether learned transitions would close the naive-Viterbi vs cascade
gap before committing to full CRF training.


Replaces the cascade of repair rules + scorer + sequence-override with a
single structured prediction. State per contact is the joint
(action_type, player_track_id) pair. Emissions combine the action-type
classifier's probability with the dynamic-attribution scorer's per-
candidate probability. Transitions encode volleyball rules:

- Hard (transition prob = epsilon):
    * Net-crossing action (SERVE / ATTACK) MUST flip team
    * Non-net-crossing action MUST stay on same team
    * 3rd consecutive contact on same team MUST be ATTACK
    * Same-player back-to-back is strongly penalised

- Soft (typical transitions):
    * SERVE → RECEIVE most likely (0.8)
    * RECEIVE → SET most likely (0.7)
    * SET → ATTACK most likely (0.7)
    * ATTACK → DIG most likely (0.5)
    * etc.

This module is PHASE 1 (skeleton + smoke test). Production integration
is deferred until empirical A/B validation on trusted-29 confirms the
joint Viterbi produces at least matching attribution + cleaner coherence
vs the current v3.2 pipeline.

Architectural design: [[repair_cascade_audit_2026_05_17]]
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

from rallycut.tracking.action_classifier import ActionType

_LOG_EPS = -50.0  # log of "effectively zero"
_EPS = 1e-9


# Net-crossing actions flip team possession.
_NET_CROSSING: set[ActionType] = {ActionType.SERVE, ActionType.ATTACK}

# Hand-coded typical-transition prior matrix. Row = previous action,
# col = new action. Values are loose volleyball-rule priors and will be
# refined empirically from trusted-GT statistics in a follow-up. The
# matrix is only consulted when team constraints permit the transition
# (so e.g. SERVE→SERVE same-team is already disallowed by the team
# constraint and never reaches the typical-transitions lookup).
def _load_empirical_transitions_if_available() -> (
    dict[ActionType, dict[ActionType, float]] | None
):
    """If JOINT_VITERBI_TRANSITIONS_PATH env var is set, load empirical
    transition probs from that JSON. Otherwise return None (caller uses
    hand-coded defaults)."""
    path_str = os.environ.get("JOINT_VITERBI_TRANSITIONS_PATH")
    if not path_str:
        return None
    path = Path(path_str)
    if not path.exists():
        return None
    raw = json.loads(path.read_text())
    out: dict[ActionType, dict[ActionType, float]] = {}
    name_to_enum = {at.value.upper(): at for at in ActionType}
    for prev_name, this_dict in raw.items():
        prev = name_to_enum.get(prev_name.upper())
        if prev is None:
            continue
        out[prev] = {}
        for this_name, prob in this_dict.items():
            this = name_to_enum.get(this_name.upper())
            if this is None:
                continue
            out[prev][this] = float(prob)
    return out


_TYPICAL_NEXT_GIVEN_PREV_DEFAULT: dict[ActionType, dict[ActionType, float]] = {
    ActionType.SERVE: {
        ActionType.RECEIVE: 0.85,
        ActionType.DIG: 0.10,  # dig on a hard serve
        ActionType.BLOCK: 0.03,
        ActionType.SET: 0.01,
        ActionType.ATTACK: 0.01,
    },
    ActionType.RECEIVE: {
        ActionType.SET: 0.75,
        ActionType.ATTACK: 0.20,  # 2-touch on a tight pass
        ActionType.DIG: 0.04,
        ActionType.BLOCK: 0.01,
    },
    ActionType.SET: {
        ActionType.ATTACK: 0.85,
        ActionType.SET: 0.05,  # rare second touch correction
        ActionType.DIG: 0.05,
        ActionType.RECEIVE: 0.05,
    },
    ActionType.ATTACK: {
        ActionType.DIG: 0.65,  # opponent digs the attack
        ActionType.BLOCK: 0.20,
        ActionType.RECEIVE: 0.10,
        ActionType.SET: 0.05,
    },
    ActionType.DIG: {
        ActionType.SET: 0.70,
        ActionType.ATTACK: 0.20,
        ActionType.DIG: 0.07,
        ActionType.RECEIVE: 0.03,
    },
    ActionType.BLOCK: {
        ActionType.SET: 0.40,
        ActionType.DIG: 0.30,
        ActionType.ATTACK: 0.20,
        ActionType.BLOCK: 0.05,
        ActionType.RECEIVE: 0.05,
    },
}

# Load empirical transitions if env override is set; else use defaults.
_TYPICAL_NEXT_GIVEN_PREV: dict[ActionType, dict[ActionType, float]] = (
    _load_empirical_transitions_if_available() or _TYPICAL_NEXT_GIVEN_PREV_DEFAULT
)

# Discounts applied multiplicatively on top of the typical transition.
_DISCOUNT_SAME_PLAYER_BACK_TO_BACK = 0.05
_DISCOUNT_THIRD_CONTACT_NOT_ATTACK = 0.10
_DISCOUNT_WRONG_TEAM_TRANSITION = 0.001  # effectively a hard constraint


@dataclass(frozen=True)
class StateCandidate:
    """One state in the joint (action_type, player_track_id) space."""
    action_type: ActionType
    player_track_id: int
    emission_prob: float


def _log(p: float) -> float:
    """Safe log."""
    if p <= 0:
        return _LOG_EPS
    return math.log(p)


def transition_log_prob(
    prev: StateCandidate,
    prev_count_on_team: int,
    new: StateCandidate,
    team_assignments: dict[int, int],
) -> float:
    """Log P(new_state | prev_state, chain_context).

    Encodes volleyball rules as discount factors on a typical-transitions
    prior. Returns log-probability ready for Viterbi accumulation.
    """
    base = _TYPICAL_NEXT_GIVEN_PREV.get(prev.action_type, {}).get(
        new.action_type, 0.01,
    )

    # Same-player back-to-back — strong discount
    if prev.player_track_id == new.player_track_id:
        base *= _DISCOUNT_SAME_PLAYER_BACK_TO_BACK

    # Team-transition constraint (the volleyball-rule core)
    prev_team = team_assignments.get(prev.player_track_id)
    new_team = team_assignments.get(new.player_track_id)
    if prev_team is not None and new_team is not None:
        same_team = (prev_team == new_team)
        prev_net_crossing = prev.action_type in _NET_CROSSING
        # Net-crossing must flip; non-net-crossing must NOT flip
        if prev_net_crossing == same_team:
            base *= _DISCOUNT_WRONG_TEAM_TRANSITION

    # 3rd-contact-must-be-ATTACK rule. Only fires when the previous
    # contact was the 2nd on a team (so this is the 3rd on same team).
    # Counter wraps at net-crossing in the caller.
    same_team = (
        prev_team is not None and new_team is not None
        and prev_team == new_team
    )
    if same_team and prev_count_on_team == 2 and new.action_type != ActionType.ATTACK:
        base *= _DISCOUNT_THIRD_CONTACT_NOT_ATTACK

    return _log(base)


def joint_viterbi(
    emissions_per_contact: list[list[StateCandidate]],
    team_assignments: dict[int, int] | None = None,
    seed_serve_team: int | None = None,
) -> list[StateCandidate]:
    """Run Viterbi over the joint state space and return the best path.

    Args:
        emissions_per_contact: For each contact, the list of (action_type,
            player_track_id) state candidates with their emission probs.
            Emission prob should already combine action-type classifier
            probability with player-attribution scorer probability.
        team_assignments: track_id → team (0 or 1). When None, team
            transition constraints are skipped (degenerate to action-type
            Viterbi over the joint).
        seed_serve_team: When known, constrains the first state to be a
            SERVE on this team. Otherwise the first contact is free to
            be any state but typically the highest-emission SERVE wins.

    Returns:
        The best-probability sequence of StateCandidate, one per contact.
    """
    if not emissions_per_contact:
        return []
    if team_assignments is None:
        team_assignments = {}

    n = len(emissions_per_contact)

    # Initialise dp[0] from emissions of the first contact.
    # Each dp entry: (cum_log_prob, prev_state_index, state, count_on_team)
    first_states = [
        s for s in emissions_per_contact[0]
        if s.action_type == ActionType.SERVE
    ]
    if not first_states:
        first_states = list(emissions_per_contact[0])

    dp: list[list[tuple[float, int, StateCandidate, int]]] = [[] for _ in range(n)]
    for s in first_states:
        log_em = _log(s.emission_prob)
        # If we know the serving team, penalise wrong-team SERVE candidates.
        if seed_serve_team is not None:
            tid_team = team_assignments.get(s.player_track_id)
            if tid_team is not None and int(tid_team) != int(seed_serve_team):
                log_em += _log(_DISCOUNT_WRONG_TEAM_TRANSITION)
        dp[0].append((log_em, -1, s, 1))

    # Forward pass
    for i in range(1, n):
        for new_s in emissions_per_contact[i]:
            log_em_new = _log(new_s.emission_prob)
            best_log_prob = -math.inf
            best_prev_idx = -1
            best_count = 1
            for prev_idx, (prev_log_prob, _, prev_s, prev_count) in enumerate(dp[i - 1]):
                t_log = transition_log_prob(
                    prev_s, prev_count, new_s, team_assignments,
                )
                cum = prev_log_prob + t_log + log_em_new
                # Determine count_on_team for the new state.
                prev_team = team_assignments.get(prev_s.player_track_id)
                new_team = team_assignments.get(new_s.player_track_id)
                if (prev_team is not None and new_team is not None
                        and prev_team == new_team
                        and prev_s.action_type not in _NET_CROSSING):
                    new_count = prev_count + 1
                else:
                    new_count = 1
                if cum > best_log_prob:
                    best_log_prob = cum
                    best_prev_idx = prev_idx
                    best_count = new_count
            if best_prev_idx >= 0:
                dp[i].append((best_log_prob, best_prev_idx, new_s, best_count))

    # Backtrace from the best final state.
    if not dp[n - 1]:
        return []
    end_idx = max(range(len(dp[n - 1])), key=lambda j: dp[n - 1][j][0])
    path: list[StateCandidate] = []
    cur_i = n - 1
    cur_j = end_idx
    while cur_i >= 0 and cur_j >= 0:
        cum, prev_idx, state, _count = dp[cur_i][cur_j]
        path.append(state)
        cur_j = prev_idx
        cur_i -= 1
    path.reverse()
    return path


__all__ = [
    "StateCandidate",
    "joint_viterbi",
    "transition_log_prob",
]
