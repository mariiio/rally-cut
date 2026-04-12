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
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rallycut.tracking.player_tracker import PlayerPosition


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
class RallyPositionData:
    """Per-rally position and identity data for switch/convention detection."""

    positions: list[PlayerPosition] = field(default_factory=list)
    track_to_player: dict[int, int] = field(default_factory=dict)
    court_split_y: float | None = None


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


def calibrate_from_noisy_predictions(
    observations: list[RallyObservation],
    noisy_serving_teams: list[str | None],
) -> bool:
    """Self-calibrate near=A/B from existing noisy per-rally predictions.

    Uses the same majority-vote logic as ``calibrate_initial_side`` but
    substitutes GT labels with the existing pipeline's (noisy) predictions.
    Works in production without any GT.

    Args:
        observations: Per-rally formation observations.
        noisy_serving_teams: Existing per-rally serving_team predictions
            from the formation + team_assignments + semantic_flip path.
    """
    return calibrate_initial_side(observations, noisy_serving_teams)


def decode_video_dual_hypothesis(
    observations: list[RallyObservation],
    p_stay: float = 0.515,
    side_switch_rallies: set[int] | None = None,
) -> list[DecodedRally]:
    """Decode by trying both near=A and near=B, pick the better hypothesis.

    Scores each hypothesis by how plausible the resulting score progression
    looks: penalizes long serve runs (>8 consecutive), extreme score
    imbalance, and deviation from expected mean run length (~2).

    No GT or external calibration needed — purely self-contained.
    """
    switches = side_switch_rallies or set()

    results_a = decode_video(
        observations, p_stay=p_stay, initial_near_is_a=True,
        side_switch_rallies=switches,
    )
    results_b = decode_video(
        observations, p_stay=p_stay, initial_near_is_a=False,
        side_switch_rallies=switches,
    )

    score_a = _score_plausibility(results_a)
    score_b = _score_plausibility(results_b)
    return results_a if score_a >= score_b else results_b


def _score_plausibility(decoded: list[DecodedRally]) -> float:
    """Score how plausible a decoded serving sequence looks.

    Higher = more plausible. Based on volleyball priors:
    - Serve runs should average ~2 rallies (p_stay ≈ 0.515)
    - Neither team should dominate unrealistically
    - Score should be roughly balanced (beach volleyball sets are to 21)
    """
    if len(decoded) < 2:
        return 0.0

    # Compute serve run lengths.
    runs: list[int] = []
    cur_team = decoded[0].serving_team
    cur_len = 1
    for d in decoded[1:]:
        if d.serving_team == cur_team:
            cur_len += 1
        else:
            runs.append(cur_len)
            cur_team = d.serving_team
            cur_len = 1
    runs.append(cur_len)

    if not runs:
        return 0.0

    # Penalty for unrealistic run lengths.
    mean_run = sum(runs) / len(runs)
    # Expected mean run is ~2.06 (from p_stay=0.515: 1/(1-0.515)).
    run_penalty = abs(mean_run - 2.06)

    # Penalty for extreme score imbalance.
    n_a = sum(1 for d in decoded if d.serving_team == "A")
    n_b = len(decoded) - n_a
    balance = min(n_a, n_b) / max(n_a, n_b, 1)
    # Perfect balance = 1.0, extreme imbalance → 0.
    balance_penalty = 1.0 - balance

    # Penalty for very long runs (>8 is suspicious for beach volleyball).
    max_run = max(runs) if runs else 0
    long_run_penalty = max(0, max_run - 8) * 0.5

    return -(run_penalty + balance_penalty * 2.0 + long_run_penalty)


# ---------------------------------------------------------------------------
# Position-based side-switch detection (Phase 1)
# ---------------------------------------------------------------------------


def _classify_tracks_to_sides(
    positions: list[PlayerPosition],
    court_split_y: float | None,
    window_frames: int = 120,
) -> tuple[set[int], set[int]]:
    """Classify track_ids into near (high Y) and far (low Y) groups.

    Uses court_split_y if available and valid (splits players into 2 groups).
    Falls back to biggest-gap clustering when court_split_y puts all players
    on one side.

    Returns:
        (near_tids, far_tids) — sets of track_ids.
    """
    track_ys: dict[int, list[float]] = defaultdict(list)
    for p in positions:
        if p.frame_number > window_frames or p.track_id < 0:
            continue
        foot_y = p.y + p.height / 2.0
        track_ys[p.track_id].append(foot_y)

    if len(track_ys) < 2:
        return set(), set()

    means = {tid: sum(ys) / len(ys) for tid, ys in track_ys.items()}

    # Try court_split_y first.
    if court_split_y is not None:
        near = {t for t, y in means.items() if y > court_split_y}
        far = {t for t, y in means.items() if y <= court_split_y}
        if near and far:
            return near, far

    # Fallback: biggest-gap split.
    sorted_items = sorted(means.items(), key=lambda kv: kv[1])
    best_gap = 0.0
    best_idx = 0
    for i in range(len(sorted_items) - 1):
        gap = sorted_items[i + 1][1] - sorted_items[i][1]
        if gap > best_gap:
            best_gap = gap
            best_idx = i

    far = {t for t, _ in sorted_items[: best_idx + 1]}
    near = {t for t, _ in sorted_items[best_idx + 1 :]}
    return near, far


def _pids_on_side(
    tids: set[int], track_to_player: dict[int, int],
) -> list[int]:
    """Map track_ids to player_ids, return sorted list."""
    return sorted(
        track_to_player[t] for t in tids if t in track_to_player
    )


def detect_side_switches_from_positions(
    position_data: list[RallyPositionData],
    min_persist: int = 2,
) -> set[int]:
    """Detect side switches from player position changes across rallies.

    Tracks which player_ids are on the near vs far side. A switch is
    detected when the near-side player group flips to far and vice versa,
    persisting for at least ``min_persist`` consecutive rallies.

    Only uses "full" observations (2+ players on each side) to avoid
    noise from partial tracking. The reference grouping is never updated
    from partial observations.

    Args:
        position_data: Per-rally position + track_to_player data.
        min_persist: Minimum consecutive rallies the new grouping must
            persist before confirming a switch. Prevents false triggers
            from tracking noise.

    Returns:
        Set of rally indices where a side switch occurs (the mapping
        changes at the START of that rally).
    """
    n = len(position_data)
    if n < 2:
        return set()

    # Step 1: Compute per-rally near/far player_ids.
    # Only keep "full" observations where both sides have players.
    observations: list[tuple[frozenset[int], frozenset[int]] | None] = []
    for pd in position_data:
        near_tids, far_tids = _classify_tracks_to_sides(
            pd.positions, pd.court_split_y,
        )
        near_pids = frozenset(_pids_on_side(near_tids, pd.track_to_player))
        far_pids = frozenset(_pids_on_side(far_tids, pd.track_to_player))
        if len(near_pids) >= 1 and len(far_pids) >= 1:
            observations.append((near_pids, far_pids))
        else:
            observations.append(None)

    # Step 2: Establish initial reference from earliest full observation.
    ref_near: frozenset[int] | None = None
    ref_far: frozenset[int] | None = None
    for obs in observations:
        if obs is not None and len(obs[0]) >= 2 and len(obs[1]) >= 2:
            ref_near, ref_far = obs
            break

    if ref_near is None:
        # No full observation found — try best available.
        for obs in observations:
            if obs is not None:
                ref_near, ref_far = obs
                break
        if ref_near is None:
            return set()

    # Step 3: Detect transitions with hysteresis.
    # "Switched" means the old near-side players are now mostly on the far
    # side and vice versa. We use a voting approach: count how many of the
    # current near players were in ref_near vs ref_far.
    switches: set[int] = set()
    candidate_switch: int | None = None
    persist_count = 0

    def _is_switched(near: frozenset[int], far: frozenset[int]) -> bool | None:
        """Check if current arrangement is switched relative to reference.

        Returns True (switched), False (same), or None (ambiguous).
        """
        if ref_near is None or ref_far is None:
            return None
        all_ref = ref_near | ref_far
        # Only consider players we have reference for.
        known_near = near & all_ref
        known_far = far & all_ref
        # Count how many known-near players were originally near vs far.
        near_was_near = len(known_near & ref_near)
        near_was_far = len(known_near & ref_far)
        far_was_near = len(known_far & ref_near)
        far_was_far = len(known_far & ref_far)
        # Vote: "same" means near players are still mostly from ref_near.
        same_votes = near_was_near + far_was_far
        switch_votes = near_was_far + far_was_near
        if same_votes > switch_votes:
            return False
        if switch_votes > same_votes:
            return True
        return None  # Tie — ambiguous.

    for i in range(1, n):
        obs = observations[i]
        if obs is None:
            continue

        status = _is_switched(obs[0], obs[1])

        if status is None:
            continue  # Ambiguous — skip.

        if not status:
            # Same as reference — reset any pending candidate.
            if candidate_switch is not None:
                candidate_switch = None
                persist_count = 0
            # Update ref from full observations to track player_id changes
            # across rallies (match_tracker may assign different IDs).
            if len(obs[0]) >= 2 and len(obs[1]) >= 2:
                ref_near, ref_far = obs
        else:
            # Potential switch.
            if candidate_switch is None:
                candidate_switch = i
                persist_count = 1
            else:
                persist_count += 1

            if persist_count >= min_persist:
                switches.add(candidate_switch)
                # After confirming switch, update ref (swapped).
                ref_near, ref_far = ref_far, ref_near
                # Also update from latest full observation if available.
                if len(obs[0]) >= 2 and len(obs[1]) >= 2:
                    ref_near, ref_far = obs
                candidate_switch = None
                persist_count = 0

    return switches


# ---------------------------------------------------------------------------
# Team-identity convention anchor (Phase 2)
# ---------------------------------------------------------------------------


def calibrate_from_player_identity(
    observations: list[RallyObservation],
    position_data: list[RallyPositionData],
    side_switch_rallies: set[int] | None = None,
    min_confidence: float = 0.1,
) -> tuple[bool, float]:
    """Determine near=A/B convention from player identity on each side.

    **NOTE**: This function does NOT work for initial convention
    determination because match_tracker assigns player IDs {1,2} to
    whichever team is initially near — making the near=A/B question
    circular. Kept for potential future use with externally-grounded
    team identity (e.g., jersey color → team mapping).

    For each rally, identifies which player_ids are on the near vs far
    side, and votes on whether near=A (players {1,2}) or near=B ({3,4}).
    Accounts for side switches.

    Falls back to True (default convention) if insufficient signal.

    Args:
        observations: Per-rally formation observations.
        position_data: Per-rally position + track_to_player data.
        side_switch_rallies: Rally indices where sides swap.
        min_confidence: Minimum vote margin to return a confident answer.

    Returns:
        (initial_near_is_a, confidence) — confidence in [0, 1].
    """
    switches = side_switch_rallies or set()
    cumulative_switches = 0
    votes_near_a = 0.0
    votes_near_b = 0.0

    for i, (obs, pd) in enumerate(zip(observations, position_data)):
        if i in switches:
            cumulative_switches += 1
        flipped = cumulative_switches % 2 == 1

        near_tids, far_tids = _classify_tracks_to_sides(
            pd.positions, pd.court_split_y,
        )
        near_pids = _pids_on_side(near_tids, pd.track_to_player)
        if not near_pids:
            continue

        # Determine which team is on the near side.
        a_count = sum(1 for p in near_pids if p <= 2)
        b_count = sum(1 for p in near_pids if p >= 3)
        if a_count == b_count:
            continue  # Ambiguous — skip.

        near_team = "A" if a_count > b_count else "B"

        # Account for side switch: if flipped, near physically is the
        # opposite semantic side from the initial convention.
        if flipped:
            near_team = "B" if near_team == "A" else "A"

        weight = max(obs.formation_confidence, 0.3) if obs.formation_side else 0.3
        if near_team == "A":
            votes_near_a += weight
        else:
            votes_near_b += weight

    total = votes_near_a + votes_near_b
    if total < 1e-6:
        return True, 0.0  # No signal — default.

    near_is_a = votes_near_a >= votes_near_b
    confidence = abs(votes_near_a - votes_near_b) / total

    if confidence < min_confidence:
        return True, 0.0  # Too close to call — default.

    return near_is_a, confidence
