"""Viterbi MAP decoder over the candidate lattice.

Replaces the GBM's per-candidate hard-threshold decision with a global
sequence decode that uses the volleyball action grammar as structural
prior. Documented in plan file ``contact-crf-sequence-decoder.md`` Phase
CRF-1.

Emission (per candidate i):
    log_emit_accept[i, action] = log P_gbm_contact(i) + log P_action(action | frame_i)
    log_emit_skip[i]          = log (1 - P_gbm_contact(i))

Where:
    P_gbm_contact: from ContactClassifier.predict_proba (the GBM we already
        train in LOO-per-video — phase unchanged).
    P_action: from MS-TCN++ sequence_probs at the candidate frame, renorm-
        alised over positive action classes.

Transition:
    log_trans(a_prev, a_cur, gap, cross) from an empirical probability matrix
    keyed by (prev_action, gap_bucket, cross_team). Learned from GT rally
    sequences in Phase CRF-0.

Decoding:
    Dynamic programming over candidates sorted by frame. State =
    (last_accepted_candidate_idx, last_accepted_action). Backtrace yields
    the MAP sequence of (candidate_idx, action) pairs.
"""
from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

ACTIONS = ["serve", "receive", "set", "attack", "dig", "block"]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
NUM_ACTIONS = len(ACTIONS)

# Gap buckets in frames (matches scripts/transition_analysis.py)
GAP_BUCKETS = [(0, 5), (6, 15), (16, 40), (41, 120), (121, 10_000)]

# Default transition matrix shipped alongside the code. Learned from GT rally
# sequences via scripts/transition_analysis.py (Phase CRF-0, 2026-04-20).
DEFAULT_TRANSITION_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "contact_transitions.json"
)


def _bucket(gap: int) -> int:
    """Map gap in frames to bucket index (0..len(GAP_BUCKETS)-1)."""
    for idx, (lo, hi) in enumerate(GAP_BUCKETS):
        if lo <= gap <= hi:
            return idx
    return len(GAP_BUCKETS) - 1


@dataclass
class TransitionMatrix:
    """Learned P(action_j | action_i, gap_bucket, cross_team) matrix.

    Loaded from Phase CRF-0's transition_matrix JSON. `probs` keys are
    ``"<action_i>|<bucket_idx>|<cross>"``, values are dicts mapping
    action_j -> probability.
    """
    probs: dict[str, dict[str, float]]
    num_actions: int = NUM_ACTIONS
    # Fallback P when we have no prior context (first contact) or no entry
    uniform: float = 1.0 / NUM_ACTIONS
    # Minimum probability floor for stability (avoid -inf log)
    eps: float = 1e-5

    @classmethod
    def from_json(cls, path: str | Path) -> TransitionMatrix:
        data = json.loads(Path(path).read_text())
        return cls(probs=data["probs"])

    @classmethod
    def default(cls) -> TransitionMatrix:
        """Load the transition matrix shipped alongside the module."""
        return cls.from_json(DEFAULT_TRANSITION_PATH)

    def log_trans(
        self, prev_action: str | None, cur_action: str,
        gap: int, cross: str,
    ) -> float:
        """Log transition probability.

        Args:
            prev_action: Previous accepted action, or None if no prior
                acceptance (first contact in rally).
            cur_action: Candidate action for the current position.
            gap: Frame gap between previous and current candidate.
            cross: "cross" (different team) / "same" (same team) /
                "unknown" (team not determined).
        """
        if prev_action is None:
            # Uniform prior over first contact types. Serves get a light
            # preference; the emission will do the heavy lifting.
            base = 0.35 if cur_action == "serve" else (0.65 / (NUM_ACTIONS - 1))
            return float(np.log(max(self.eps, base)))

        key = f"{prev_action}|{_bucket(gap)}|{cross}"
        row = self.probs.get(key)
        if row is None:
            # Back off: ignore cross-team
            for alt in ("cross", "same", "unknown"):
                alt_key = f"{prev_action}|{_bucket(gap)}|{alt}"
                row = self.probs.get(alt_key)
                if row is not None:
                    break
        if row is None:
            return float(np.log(self.uniform))
        p = row.get(cur_action, self.uniform)
        return float(np.log(max(self.eps, p)))


@dataclass
class CandidateFeatures:
    """Minimal per-candidate info the decoder needs."""
    frame: int
    gbm_contact_prob: float  # P(contact | features), raw (not thresholded)
    action_probs: np.ndarray  # (NUM_ACTIONS,) P(action | frame) normalised
    team: int                 # 0 = near, 1 = far, -1 = unknown


@dataclass
class DecodedContact:
    """One accepted contact from Viterbi."""
    candidate_idx: int
    frame: int
    action: str
    action_idx: int
    score: float  # log-probability contribution at this step


def decode_rally(
    candidates: Sequence[CandidateFeatures],
    transitions: TransitionMatrix,
    *,
    skip_penalty: float = 0.0,
    emission_floor: float = 0.02,
    min_accept_prob: float = 0.0,
) -> list[DecodedContact]:
    """Run Viterbi over a single rally's candidates. Returns accepted list.

    Args:
        candidates: Candidates sorted by frame, with features populated.
        transitions: Learned transition matrix.
        skip_penalty: Extra log-cost added when skipping a candidate. 0 =
            pure log(1-P_contact). Positive values discourage over-skipping.
        emission_floor: Probability floor for GBM/action probs to avoid
            -inf log-values.
        min_accept_prob: Reject candidates whose GBM prob is below this
            floor before decoding. Bounds worst-case FPs.

    Returns:
        List of DecodedContact (accepted candidates), in frame order.
    """
    if not candidates:
        return []

    # Filter out hopeless candidates early (emission floor).
    usable = [
        (idx, c) for idx, c in enumerate(candidates)
        if c.gbm_contact_prob >= min_accept_prob
    ]
    if not usable:
        return []

    n = len(usable)

    def _log(p: float) -> float:
        return float(np.log(max(emission_floor, p)))

    # Emission tables (aligned to usable[])
    log_emit_accept = np.full((n, NUM_ACTIONS), -np.inf, dtype=np.float64)
    log_emit_skip = np.zeros(n, dtype=np.float64)
    for i, (_orig_idx, c) in enumerate(usable):
        p_contact = float(c.gbm_contact_prob)
        for a_idx in range(NUM_ACTIONS):
            p_action = float(c.action_probs[a_idx])
            log_emit_accept[i, a_idx] = _log(p_contact) + _log(p_action)
        log_emit_skip[i] = _log(1.0 - p_contact) - skip_penalty

    # DP
    # best[i, a] = best log-prob of sequences over usable[0..i] where
    #              candidate i is the LAST ACCEPTED with action a.
    # back[i, a] = (prev_i, prev_a) — None if i is the first accepted.
    best = np.full((n, NUM_ACTIONS), -np.inf, dtype=np.float64)
    back: list[list[tuple[int, int] | None]] = [
        [None] * NUM_ACTIONS for _ in range(n)
    ]

    for i in range(n):
        _orig_i, c_i = usable[i]
        # Case 1: candidate i is the FIRST accepted (no predecessor).
        # Need to include log-emit-skip of all earlier candidates.
        prefix_skip = float(np.sum(log_emit_skip[:i]))
        for a in range(NUM_ACTIONS):
            # No-prior transition: use uniform-ish prior via transitions.
            score = (prefix_skip + log_emit_accept[i, a]
                     + transitions.log_trans(None, ACTIONS[a], 0, "unknown"))
            if score > best[i, a]:
                best[i, a] = score
                back[i][a] = None

        # Case 2: some earlier j < i is the previous accepted.
        for j in range(i):
            _orig_j, c_j = usable[j]
            gap = c_i.frame - c_j.frame
            cross = _cross_label(c_i.team, c_j.team)
            # Candidates between j and i are skipped
            mid_skip = float(np.sum(log_emit_skip[j + 1:i]))
            for a_prev in range(NUM_ACTIONS):
                if best[j, a_prev] == -np.inf:
                    continue
                for a in range(NUM_ACTIONS):
                    score = (
                        best[j, a_prev] + mid_skip
                        + log_emit_accept[i, a]
                        + transitions.log_trans(
                            ACTIONS[a_prev], ACTIONS[a], gap, cross,
                        )
                    )
                    if score > best[i, a]:
                        best[i, a] = score
                        back[i][a] = (j, a_prev)

    # Find best endpoint. Also consider the "accept nothing" outcome.
    final_best = float(np.sum(log_emit_skip))  # skip everything
    final_i, final_a = -1, -1
    for i in range(n):
        # Include skip log-emits from i+1..n-1 (tail)
        tail_skip = float(np.sum(log_emit_skip[i + 1:]))
        for a in range(NUM_ACTIONS):
            score = best[i, a] + tail_skip
            if score > final_best:
                final_best = score
                final_i, final_a = i, a

    if final_i < 0:
        return []

    # Backtrace
    accepted: list[DecodedContact] = []
    i, a = final_i, final_a
    while True:
        _orig_idx, c = usable[i]
        accepted.append(DecodedContact(
            candidate_idx=_orig_idx,
            frame=c.frame,
            action=ACTIONS[a],
            action_idx=a,
            score=float(best[i, a]),
        ))
        prev = back[i][a]
        if prev is None:
            break
        i, a = prev
    accepted.reverse()
    return accepted


def _cross_label(team_a: int, team_b: int) -> str:
    """Map a pair of team labels to cross/same/unknown string."""
    if team_a in (0, 1) and team_b in (0, 1):
        return "cross" if team_a != team_b else "same"
    return "unknown"


def infer_team_from_player_track(player_track_id: int) -> int:
    """Convention: track 1-2 = near (team 0), 3-4 = far (team 1)."""
    if player_track_id < 1 or player_track_id > 4:
        return -1
    return (player_track_id - 1) // 2
