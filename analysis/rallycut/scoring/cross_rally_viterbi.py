"""Cross-rally Viterbi decoder with hard transition constraints.

Operates on PHYSICAL SIDES (near/far) rather than team labels (A/B) to
avoid the team-mapping errors identified in Phase 0.

Observation stream per rally:
  Formation: which physical side is serving? (near/far, ~97% coverage)

User correction anchors (Phase 3):
  - first_serve_side: hard-lock rally 0
  - anchors: hard-lock arbitrary rallies (user corrections)
  - n_near_target: count constraint (from final score)

After decoding physical sides, converts to team labels (A/B) using a
per-video calibration that determines which team started on which side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RallyObservation:
    """Per-rally observations for the Viterbi decoder."""

    rally_id: str
    # Formation predictor: which physical side is serving?
    formation_side: str | None  # "near" or "far" (or None if abstained)
    formation_confidence: float  # 0-1
    # GT (for eval only)
    gt_serving_team: str | None = None  # "A" or "B"


@dataclass
class DecodedRally:
    """Viterbi output for one rally."""

    rally_id: str
    serving_side: str  # "near" or "far"
    serving_team: str  # "A" or "B"
    confidence: float
    score_a: int = 0
    score_b: int = 0


SIDES = ("near", "far")


def _log(x: float) -> float:
    return math.log(max(x, 1e-30))


def decode_video(
    observations: list[RallyObservation],
    p_stay: float = 0.515,
    initial_near_is_a: bool = True,
    side_switch_rallies: set[int] | None = None,
    first_serve_side: str | None = None,
    anchors: dict[int, str] | None = None,
    n_near_target: int | None = None,
) -> list[DecodedRally]:
    """Decode serving side for all rallies in a video using Viterbi.

    Args:
        observations: Per-rally observations (formation).
        p_stay: Soft prior P(same side serves next). Default 0.515.
        initial_near_is_a: Whether "near" = team "A" at video start.
        side_switch_rallies: Rally indices where teams swap sides.
        first_serve_side: User anchor for rally 0 ("near" or "far").
            Hard-locks the first rally's serving side.
        anchors: User corrections mapping rally_index → "near"/"far".
            Hard-locks the specified rallies.
        n_near_target: Count constraint — total rallies where near side
            serves. Derived from final score + first server. When set,
            uses a 3D DP (rally × side × near_count) for exact count.

    Returns:
        List of DecodedRally with serving side, team label, and score.
    """
    n = len(observations)
    if n == 0:
        return []

    # Merge first_serve_side into anchors.
    all_anchors: dict[int, int] = {}  # rally_index → side_idx (0=near, 1=far)
    if first_serve_side is not None:
        all_anchors[0] = 0 if first_serve_side == "near" else 1
    if anchors:
        for idx, side in anchors.items():
            all_anchors[idx] = 0 if side == "near" else 1

    switches = side_switch_rallies or set()

    if n_near_target is not None:
        path = _decode_count_constrained(
            observations, p_stay, all_anchors, n_near_target,
        )
    else:
        path = _decode_standard(observations, p_stay, all_anchors)

    return _build_results(observations, path, initial_near_is_a, switches)


def _decode_standard(
    observations: list[RallyObservation],
    p_stay: float,
    anchors: dict[int, int],
) -> list[int]:
    """Standard Viterbi (no count constraint)."""
    n = len(observations)
    neg_inf = -1e18
    dp = [[neg_inf, neg_inf] for _ in range(n)]
    bp = [[-1, -1] for _ in range(n)]

    def emit(obs: RallyObservation, s: int) -> float:
        if obs.formation_side is None:
            return _log(0.5)
        obs_idx = 0 if obs.formation_side == "near" else 1
        c = max(0.0, min(1.0, obs.formation_confidence))
        if obs_idx == s:
            return _log(0.5 + 0.45 * c)
        return _log(0.5 - 0.45 * c)

    log_stay = _log(p_stay)
    log_flip = _log(1.0 - p_stay)

    # Initialize.
    for s in range(2):
        if 0 in anchors and anchors[0] != s:
            dp[0][s] = neg_inf  # anchor blocks this state
        else:
            dp[0][s] = _log(0.5) + emit(observations[0], s)

    # Forward.
    for i in range(1, n):
        for s in range(2):
            if i in anchors and anchors[i] != s:
                dp[i][s] = neg_inf
                continue
            em = emit(observations[i], s)
            for prev_s in range(2):
                trans = log_stay if prev_s == s else log_flip
                score = dp[i - 1][prev_s] + trans + em
                if score > dp[i][s]:
                    dp[i][s] = score
                    bp[i][s] = prev_s

    # Backtrack.
    path = [0] * n
    path[-1] = 0 if dp[-1][0] >= dp[-1][1] else 1
    for i in range(n - 2, -1, -1):
        path[i] = bp[i + 1][path[i + 1]]
    return path


def _decode_count_constrained(
    observations: list[RallyObservation],
    p_stay: float,
    anchors: dict[int, int],
    n_near_target: int,
) -> list[int]:
    """Count-constrained Viterbi: exactly n_near_target rallies with near serving.

    DP state: dp[i][s][k] = log prob at rally i, state s, having seen k near-serves.
    """
    n = len(observations)
    neg_inf = -1e18
    max_k = n_near_target + 1

    def emit(obs: RallyObservation, s: int) -> float:
        if obs.formation_side is None:
            return _log(0.5)
        obs_idx = 0 if obs.formation_side == "near" else 1
        c = max(0.0, min(1.0, obs.formation_confidence))
        if obs_idx == s:
            return _log(0.5 + 0.45 * c)
        return _log(0.5 - 0.45 * c)

    log_stay = _log(p_stay)
    log_flip = _log(1.0 - p_stay)

    # dp[i][s][k], bp[i][s][k] = prev_s
    dp = [[[neg_inf] * max_k for _ in range(2)] for _ in range(n)]
    bp = [[[-1] * max_k for _ in range(2)] for _ in range(n)]

    # Init rally 0.
    for s in range(2):
        if 0 in anchors and anchors[0] != s:
            continue
        k = 1 if s == 0 else 0  # s=0 is near
        if k < max_k:
            dp[0][s][k] = _log(0.5) + emit(observations[0], s)

    # Forward.
    for i in range(1, n):
        for s in range(2):
            if i in anchors and anchors[i] != s:
                continue
            em = emit(observations[i], s)
            dk = 1 if s == 0 else 0  # near serves add 1 to k
            for k in range(max_k):
                prev_k = k - dk
                if prev_k < 0 or prev_k >= max_k:
                    continue
                for prev_s in range(2):
                    pv = dp[i - 1][prev_s][prev_k]
                    if pv <= neg_inf / 2:
                        continue
                    trans = log_stay if prev_s == s else log_flip
                    score = pv + trans + em
                    if score > dp[i][s][k]:
                        dp[i][s][k] = score
                        bp[i][s][k] = prev_s

    # Find best final state at k = n_near_target.
    final_k = n_near_target
    best = neg_inf
    best_s = 0
    for s in range(2):
        if final_k < max_k and dp[-1][s][final_k] > best:
            best = dp[-1][s][final_k]
            best_s = s

    if best <= neg_inf / 2:
        # Infeasible — fall back to standard decode.
        return _decode_standard(observations, p_stay, anchors)

    # Backtrack.
    path = [0] * n
    path[-1] = best_s
    k = final_k
    for i in range(n - 1, 0, -1):
        prev_s = bp[i][path[i]][k]
        if path[i] == 0:  # near → decrement k
            k -= 1
        path[i - 1] = prev_s
    return path


def _build_results(
    observations: list[RallyObservation],
    path: list[int],
    initial_near_is_a: bool,
    switches: set[int],
) -> list[DecodedRally]:
    """Convert physical side path to team labels + score."""
    near_is_a = initial_near_is_a
    score_a = 0
    score_b = 0
    results: list[DecodedRally] = []

    for i, obs in enumerate(observations):
        if i in switches:
            near_is_a = not near_is_a

        side = SIDES[path[i]]
        if near_is_a:
            team = "A" if side == "near" else "B"
        else:
            team = "B" if side == "near" else "A"

        # score_a/score_b represent the score BEFORE this rally is played.
        results.append(DecodedRally(
            rally_id=obs.rally_id,
            serving_side=side,
            serving_team=team,
            confidence=0.0,
            score_a=score_a,
            score_b=score_b,
        ))

        # Update score from serving team progression.
        if i > 0:
            prev_team = results[i - 1].serving_team
            cur_team = results[i].serving_team
            if cur_team != prev_team:
                if cur_team == "A":
                    score_a += 1
                else:
                    score_b += 1
            else:
                if prev_team == "A":
                    score_a += 1
                else:
                    score_b += 1
            results[i].score_a = score_a
            results[i].score_b = score_b

    return results


def calibrate_initial_side(
    observations: list[RallyObservation],
    gt_serving_teams: list[str | None] | None = None,
) -> bool:
    """Determine whether near=A at the start of the video.

    If GT is available (eval), uses majority vote from GT + formation.
    If not (production), assumes near=A (standard convention).

    Returns:
        True if near corresponds to team A at rally 0.
    """
    if gt_serving_teams is None:
        return True

    votes_near_is_a = 0
    votes_near_is_b = 0
    for obs, gt in zip(observations, gt_serving_teams):
        if obs.formation_side is None or gt is None:
            continue
        if obs.formation_side == "near" and gt == "A":
            votes_near_is_a += 1
        elif obs.formation_side == "near" and gt == "B":
            votes_near_is_b += 1
        elif obs.formation_side == "far" and gt == "A":
            votes_near_is_b += 1
        elif obs.formation_side == "far" and gt == "B":
            votes_near_is_a += 1

    return votes_near_is_a >= votes_near_is_b
