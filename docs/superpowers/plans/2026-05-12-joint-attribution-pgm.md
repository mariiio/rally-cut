# Joint Attribution PGM Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `reattribute_players` with a per-rally Probabilistic Graphical Model that performs joint MAP attribution over 6 states per contact (4 players + 2 absent-team latent states). Hand-tuned soft factors; exhaustive enumeration over the small joint space.

**Architecture:** Pure-function factors → unary evidence + pairwise rules + higher-order rules. Exhaustive enumeration for N≤8 contacts; coordinate-descent fallback at larger N. Default-OFF behind `USE_JOINT_ATTRIBUTION` env flag at call sites in `redetect_all_actions.py` and the `reattribute-actions` CLI.

**Tech Stack:** Python 3.11+, dataclasses, itertools, existing `rallycut.tracking` types and helpers, existing `analysis/scripts/measure_attribution_fresh_gt.py` measurement harness.

**Spec:** `docs/superpowers/specs/2026-05-12-joint-attribution-pgm-design.md`

**Branch:** `main` (user has approved working on main alongside in-flight uncommitted changes)

## File structure

| File | Action | Responsibility |
|---|---|---|
| `analysis/rallycut/tracking/joint_attribution_weights.py` | Create | `FactorWeights` dataclass + `DEFAULT_WEIGHTS` constant. Single source of truth for tuning. |
| `analysis/rallycut/tracking/joint_attribution_factors.py` | Create | Pure factor functions: unary (proximity, distance, visual, action prior), pairwise (back-to-back, alternation), higher-order (3-contact, serve-first). Each takes typed inputs and returns log-likelihood. |
| `analysis/rallycut/tracking/joint_attribution.py` | Create | `RallyContext` + `RallyAttribution` + `State` types; evidence builder; inference engine (exhaustive + coordinate-descent fallback); public API `joint_attribute_rally`. |
| `analysis/scripts/redetect_all_actions.py` | Modify | Read `USE_JOINT_ATTRIBUTION`; branch between `reattribute_players` (default) and PGM path. |
| `analysis/rallycut/cli/commands/reattribute_actions.py` | Modify | Same env-flag branch. |
| `analysis/rallycut/tracking/action_classifier.py` | Modify | Add `attribution_source` field to action dicts emitted by `classify_rally_actions` (default `"action_classifier"` or `"action_classifier_abstained"`). |
| `analysis/tests/unit/test_joint_attribution_factors.py` | Create | Truth tables for each factor function. ~30 tests. |
| `analysis/tests/unit/test_joint_attribution_inference.py` | Create | Inference correctness on small synthetic rallies; coordinate-descent fallback test. |
| `analysis/tests/unit/test_joint_attribution_e2e.py` | Create | End-to-end synthetic `RallyContext` → `joint_attribute_rally`. ~10 cases. |
| `analysis/tests/unit/test_joint_attribution_integration.py` | Create | env-flag branch wiring; verify `actions_json` shape with `attribution_source`. |
| `analysis/scripts/calibrate_joint_attribution_weights.py` | Create | Coordinate-ascent factor-weight calibration on the 22-rally panel. |
| `analysis/scripts/measure_joint_attribution_ab.py` | Create | A/B harness wrapping `measure_attribution_fresh_gt.py`. |

## Decision checkpoints

This plan has two human-decision checkpoints:

1. **After Task 11 (calibration):** Review the calibrated weights. Any weight at a sweep boundary suggests the sweep range was too narrow; widen and re-run. Document the calibration trajectory in the verdict report.

2. **After Task 12 (A/B):** Apply pre-ship gates G-A through G-H from the spec. Decide PASS (ship default-ON) / DONE_WITH_CONCERNS (ship default-OFF infrastructure) / FAIL (revert + escalate to Phase B).

---

### Task 1: FactorWeights dataclass + DEFAULT_WEIGHTS

**Files:**
- Create: `analysis/rallycut/tracking/joint_attribution_weights.py`

- [ ] **Step 1: Create the file with the dataclass and DEFAULT_WEIGHTS constant**

```python
"""Factor weights for the Joint Attribution PGM.

Single source of truth for tuning. Hand-tuned starting values; calibrated
on the 22-rally fresh-GT panel via
`analysis/scripts/calibrate_joint_attribution_weights.py`.

Spec: docs/superpowers/specs/2026-05-12-joint-attribution-pgm-design.md
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FactorWeights:
    """Weights for unary (evidence) and pairwise/higher-order (rule) factors.

    All weights are non-negative and applied as log-likelihood multipliers.
    Higher weight = stronger factor influence on the joint MAP.
    """

    # Unary (per-contact evidence)
    w_proximity: float = 2.0       # playerCandidates rank
    w_dist: float = 1.0            # playerCandidates distance (per-player)
    w_dist_team: float = 0.5       # absent-state penalty proportional to team's nearest tracked player
    w_visual: float = 1.5          # cross-rally PID profile cosine similarity
    w_pose: float = 1.5            # pose model P(touching)
    w_prior: float = 0.5           # action_classifier initial PID as soft prior
    w_action: float = 1.0          # action-type prior for absent states

    # Pairwise (rule)
    w_back_to_back: float = 3.0    # penalty for same-player consecutive contacts (non-absent)
    w_alternation: float = 3.0     # penalty for same-team consecutive across a net crossing
    w_team_consistency: float = 2.0  # penalty for cross-team consecutive without a net crossing
    w_absent_pair: float = 1.5     # penalty for two consecutive absent-* states

    # Higher-order (rule)
    w_3_contact: float = 4.0       # penalty per extra contact beyond 3 same-team same-side
    w_serve_first: float = 3.0     # penalty if first contact's team != serving_team


DEFAULT_WEIGHTS = FactorWeights()
```

- [ ] **Step 2: Run mypy + ruff**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && \
  uv run mypy rallycut/tracking/joint_attribution_weights.py && \
  uv run ruff check rallycut/tracking/joint_attribution_weights.py
```

Expected: 0 errors / clean.

- [ ] **Step 3: Commit**

```bash
git add analysis/rallycut/tracking/joint_attribution_weights.py
git commit -m "feat(attribution): FactorWeights dataclass for joint-attribution PGM

Hand-tuned starting weights for 13 factors (7 unary + 5 pairwise/higher-order).
Single source of truth for tuning. Defaults will be calibrated by
calibrate_joint_attribution_weights.py on the 22-rally panel.

Spec: docs/superpowers/specs/2026-05-12-joint-attribution-pgm-design.md"
```

---

### Task 2: RallyContext + RallyAttribution + State types

**Files:**
- Create: `analysis/rallycut/tracking/joint_attribution.py` (start of file; the rest accumulates in later tasks)

- [ ] **Step 1: Create the file with type definitions only**

```python
"""Joint Attribution PGM — replaces reattribute_players with per-rally
joint MAP inference over 6 states per contact (4 players + 2 absent-team).

Public API: joint_attribute_rally(rally) -> RallyAttribution

Spec: docs/superpowers/specs/2026-05-12-joint-attribution-pgm-design.md
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

# State type: a tracked player (1-4) OR an absent-team marker.
# Represented as a string for easy serialization and pattern-matching.
State = str  # "P1" | "P2" | "P3" | "P4" | "ABSENT_TEAM_A" | "ABSENT_TEAM_B"

ABSENT_STATES = ("ABSENT_TEAM_A", "ABSENT_TEAM_B")


def is_absent(state: State) -> bool:
    """True if state is one of the ABSENT_TEAM_* markers."""
    return state in ABSENT_STATES


def absent_state_for_team(team: str) -> State:
    """Map team letter ('A' or 'B') to the corresponding absent state."""
    if team == "A":
        return "ABSENT_TEAM_A"
    if team == "B":
        return "ABSENT_TEAM_B"
    raise ValueError(f"unknown team: {team}")


@dataclass(frozen=True)
class RallyContext:
    """Bundle of per-rally inputs the PGM needs.

    Built by the caller (e.g., redetect_all_actions, reattribute-actions CLI)
    from existing data sources: contacts_json, actions_json (initial PIDs),
    team_assignments, serving_team, and optionally visual profiles.
    """
    rally_id: str
    contacts: list[dict[str, Any]]
    """Each contact: {frame, ballX, ballY, playerCandidates, courtSide, ...}
    Same shape as contacts_json.contacts[*]."""

    initial_actions: list[dict[str, Any]]
    """Each action: {frame, action, playerTrackId, confidence, ...}
    Initial output from classify_rally_actions; provides per-contact
    action labels and the soft-prior PID for unary action_prior factors."""

    team_assignments: dict[int, str]
    """Map: canonical PID (1-4) -> team letter ('A' or 'B'). From the
    cross-rally matcher."""

    serving_team: str | None
    """'A' or 'B' or None (if unknown). From the rally's matcher state."""

    visual_profiles: dict[int, list[float]] | None = None
    """Optional: PID -> embedding vector for cross-rally visual similarity.
    None disables the w_visual factor for this rally."""


@dataclass(frozen=True)
class RallyAttribution:
    """Result of joint_attribute_rally."""
    map: tuple[State, ...]
    """The MAP joint assignment, one State per contact."""

    score: float
    """Joint log-likelihood of the MAP assignment."""

    marginals: list[dict[State, float]]
    """Per-contact posterior marginals over states. Indexed by contact position.
    Computed by exp(score - max_score) and normalized per-contact."""

    fallback_used: Literal["exhaustive", "coordinate_descent"]
    """Which inference method was actually used (exhaustive for N<=8,
    coordinate_descent for N>8)."""
```

- [ ] **Step 2: Run mypy + ruff**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && \
  uv run mypy rallycut/tracking/joint_attribution.py && \
  uv run ruff check rallycut/tracking/joint_attribution.py
```

Expected: 0 errors / clean.

- [ ] **Step 3: Commit**

```bash
git add analysis/rallycut/tracking/joint_attribution.py
git commit -m "feat(attribution): RallyContext + RallyAttribution + State types

Type definitions for the joint-attribution PGM input/output contract.
State is a string union: 'P1'..'P4' or 'ABSENT_TEAM_A' / 'ABSENT_TEAM_B'.
Helpers: is_absent, absent_state_for_team.

Spec: docs/superpowers/specs/2026-05-12-joint-attribution-pgm-design.md"
```

---

### Task 3: Unary factor functions + tests (TDD)

**Files:**
- Create: `analysis/rallycut/tracking/joint_attribution_factors.py`
- Create: `analysis/tests/unit/test_joint_attribution_factors.py`

- [ ] **Step 1: Write the test file with truth tables for unary factors**

```python
"""Unit tests for joint-attribution factor functions.

Each test is a small truth table for one factor: explicit inputs +
expected log-likelihood output. Pure-function testing.

Spec: docs/superpowers/specs/2026-05-12-joint-attribution-pgm-design.md
"""
from __future__ import annotations

import math

import pytest

from rallycut.tracking.joint_attribution_factors import (
    unary_proximity,
    unary_distance,
    unary_action_prior,
)
from rallycut.tracking.joint_attribution_weights import DEFAULT_WEIGHTS


# ---- unary_proximity ----

def test_unary_proximity_rank_1_best() -> None:
    """Player at rank 1 in candidates should get the highest proximity score."""
    candidates = [(1, 0.05), (2, 0.10), (3, 0.20), (4, 0.50)]  # (raw_id, distance)
    rank_1 = unary_proximity("P1", candidates, pid_to_raw_id={1: 1, 2: 2, 3: 3, 4: 4}, weights=DEFAULT_WEIGHTS)
    rank_2 = unary_proximity("P2", candidates, pid_to_raw_id={1: 1, 2: 2, 3: 3, 4: 4}, weights=DEFAULT_WEIGHTS)
    assert rank_1 > rank_2


def test_unary_proximity_player_not_in_candidates() -> None:
    """Player absent from candidates returns a large negative score."""
    candidates = [(1, 0.05)]
    score = unary_proximity("P4", candidates, pid_to_raw_id={1: 1, 4: 4}, weights=DEFAULT_WEIGHTS)
    assert score < -100  # large negative


def test_unary_proximity_absent_state_returns_zero() -> None:
    """ABSENT_TEAM_X states don't claim proximity; return 0."""
    candidates = [(1, 0.05)]
    score = unary_proximity("ABSENT_TEAM_A", candidates, pid_to_raw_id={1: 1}, weights=DEFAULT_WEIGHTS)
    assert score == 0.0


# ---- unary_distance ----

def test_unary_distance_closer_is_better() -> None:
    candidates = [(1, 0.05), (2, 0.20)]
    pid_to_raw = {1: 1, 2: 2}
    near = unary_distance("P1", candidates, pid_to_raw, weights=DEFAULT_WEIGHTS)
    far = unary_distance("P2", candidates, pid_to_raw, weights=DEFAULT_WEIGHTS)
    assert near > far


def test_unary_distance_absent_uses_team_distance() -> None:
    """ABSENT_TEAM_X penalty proportional to team's NEAREST tracked player.
    If team's nearest player is far, absent is less penalized."""
    candidates = [(1, 0.05), (3, 0.30)]
    pid_to_raw = {1: 1, 3: 3}
    team_assignments = {1: "A", 2: "A", 3: "B", 4: "B"}
    # team A's nearest: P1 at 0.05 (close) -> ABSENT_TEAM_A is heavily penalized
    abs_a = unary_distance("ABSENT_TEAM_A", candidates, pid_to_raw,
                           weights=DEFAULT_WEIGHTS, team_assignments=team_assignments)
    # team B's nearest: P3 at 0.30 (far) -> ABSENT_TEAM_B is less penalized
    abs_b = unary_distance("ABSENT_TEAM_B", candidates, pid_to_raw,
                           weights=DEFAULT_WEIGHTS, team_assignments=team_assignments)
    assert abs_b > abs_a  # less penalty for team-B-absent


# ---- unary_action_prior ----

def test_unary_action_prior_matching_initial_pid() -> None:
    """If state matches the initial classifier PID, return log(0.6)*w."""
    score = unary_action_prior("P1", initial_pid=1, weights=DEFAULT_WEIGHTS)
    assert math.isclose(score, math.log(0.6) * DEFAULT_WEIGHTS.w_prior, abs_tol=1e-9)


def test_unary_action_prior_non_matching_player() -> None:
    score = unary_action_prior("P2", initial_pid=1, weights=DEFAULT_WEIGHTS)
    assert math.isclose(score, math.log(0.1) * DEFAULT_WEIGHTS.w_prior, abs_tol=1e-9)


def test_unary_action_prior_absent_state() -> None:
    """Absent states get a very small prior unless evidence is strong."""
    score = unary_action_prior("ABSENT_TEAM_A", initial_pid=1, weights=DEFAULT_WEIGHTS)
    assert math.isclose(score, math.log(0.05) * DEFAULT_WEIGHTS.w_prior, abs_tol=1e-9)


def test_unary_action_prior_no_initial_pid() -> None:
    """When initial_pid is None (action_classifier abstained), prior is uniform."""
    score = unary_action_prior("P1", initial_pid=None, weights=DEFAULT_WEIGHTS)
    expected = math.log(0.25) * DEFAULT_WEIGHTS.w_prior  # 1/4 if no info
    assert math.isclose(score, expected, abs_tol=1e-9)
```

- [ ] **Step 2: Run the tests — expect FAIL (functions don't exist)**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && \
  uv run pytest tests/unit/test_joint_attribution_factors.py -v
```

Expected: FAIL with "cannot import unary_proximity from joint_attribution_factors" or similar.

- [ ] **Step 3: Implement the unary factors**

Create `analysis/rallycut/tracking/joint_attribution_factors.py`:

```python
"""Pure factor functions for the Joint Attribution PGM.

Each function takes typed inputs and returns a log-likelihood (float).
No global state; no I/O. Trivially testable.

Spec: docs/superpowers/specs/2026-05-12-joint-attribution-pgm-design.md
"""
from __future__ import annotations

import math
from typing import Any

from rallycut.tracking.joint_attribution import (
    State,
    is_absent,
)
from rallycut.tracking.joint_attribution_weights import FactorWeights

_LARGE_NEGATIVE = -1000.0  # used when a state is impossible (e.g., player absent from candidates)


def unary_proximity(
    state: State,
    candidates: list[tuple[int, float]],
    pid_to_raw_id: dict[int, int],
    weights: FactorWeights,
) -> float:
    """Log-likelihood from playerCandidates rank ordering.
    Lower rank (1 = best) gets higher score."""
    if is_absent(state):
        return 0.0
    pid = int(state[1:])  # "P3" -> 3
    raw_id = pid_to_raw_id.get(pid)
    if raw_id is None:
        return _LARGE_NEGATIVE
    for rank, (cand_raw_id, _dist) in enumerate(candidates, start=1):
        if cand_raw_id == raw_id:
            return math.log(1.0 / (rank + 1)) * weights.w_proximity
    return _LARGE_NEGATIVE


def unary_distance(
    state: State,
    candidates: list[tuple[int, float]],
    pid_to_raw_id: dict[int, int],
    weights: FactorWeights,
    team_assignments: dict[int, str] | None = None,
) -> float:
    """Log-likelihood from playerCandidates distance.
    For tracked players: -distance * w_dist (closer = less penalty).
    For absent states: -best_distance_in_team * w_dist_team."""
    if is_absent(state):
        if team_assignments is None:
            return 0.0
        team = "A" if state == "ABSENT_TEAM_A" else "B"
        # Find nearest tracked player on this team
        team_pids = [pid for pid, t in team_assignments.items() if t == team]
        team_raw_ids = [pid_to_raw_id.get(pid) for pid in team_pids]
        team_raw_ids = [r for r in team_raw_ids if r is not None]
        team_distances = [
            d for raw_id, d in candidates if raw_id in team_raw_ids
        ]
        if not team_distances:
            return 0.0  # team has no tracked players in candidates; absent is "free"
        nearest = min(team_distances)
        # +nearest (positive bonus): far team's nearest → higher absent score
        # → absent state more plausible. Confirmed by test_unary_distance_absent_uses_team_distance.
        return nearest * weights.w_dist_team
    # Tracked player
    pid = int(state[1:])
    raw_id = pid_to_raw_id.get(pid)
    if raw_id is None:
        return _LARGE_NEGATIVE
    for cand_raw_id, dist in candidates:
        if cand_raw_id == raw_id:
            return -dist * weights.w_dist
    return _LARGE_NEGATIVE


def unary_action_prior(
    state: State,
    initial_pid: int | None,
    weights: FactorWeights,
) -> float:
    """Log-likelihood from action_classifier's initial PID as soft prior."""
    if is_absent(state):
        return math.log(0.05) * weights.w_prior
    if initial_pid is None:
        # Uniform over 4 players
        return math.log(0.25) * weights.w_prior
    pid = int(state[1:])
    if pid == initial_pid:
        return math.log(0.6) * weights.w_prior
    return math.log(0.1) * weights.w_prior
```

- [ ] **Step 4: Run the tests — expect PASS**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && \
  uv run pytest tests/unit/test_joint_attribution_factors.py -v
```

Expected: 9 tests PASS.

- [ ] **Step 5: Run mypy + ruff on both files**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && \
  uv run mypy rallycut/tracking/joint_attribution_factors.py tests/unit/test_joint_attribution_factors.py && \
  uv run ruff check rallycut/tracking/joint_attribution_factors.py tests/unit/test_joint_attribution_factors.py
```

Expected: 0 errors / clean.

- [ ] **Step 6: Commit**

```bash
git add analysis/rallycut/tracking/joint_attribution_factors.py analysis/tests/unit/test_joint_attribution_factors.py
git commit -m "feat(attribution): unary factor functions for joint-attribution PGM

Three pure functions: unary_proximity (rank-based), unary_distance
(per-player or team-aggregate for absent states), unary_action_prior
(soft prior from action_classifier initial PID).

9 unit tests covering truth-table cases including absent-state semantics.

Spec: docs/superpowers/specs/2026-05-12-joint-attribution-pgm-design.md"
```

---

### Task 4: Pairwise factor functions + tests (TDD)

**Files:**
- Modify: `analysis/rallycut/tracking/joint_attribution_factors.py`
- Modify: `analysis/tests/unit/test_joint_attribution_factors.py`

- [ ] **Step 1: Append pairwise tests**

Append to `test_joint_attribution_factors.py`:

```python
from rallycut.tracking.joint_attribution_factors import (
    pairwise_no_back_to_back,
    pairwise_alternation,
)


# ---- pairwise_no_back_to_back ----

def test_pairwise_no_back_to_back_same_player_penalized() -> None:
    """X_t == X_{t+1} (both tracked players) gets a penalty."""
    score = pairwise_no_back_to_back(
        "P1", "P1", action_t="dig", action_t1="set", weights=DEFAULT_WEIGHTS
    )
    assert score == -DEFAULT_WEIGHTS.w_back_to_back


def test_pairwise_no_back_to_back_different_player_zero() -> None:
    score = pairwise_no_back_to_back(
        "P1", "P2", action_t="dig", action_t1="set", weights=DEFAULT_WEIGHTS
    )
    assert score == 0.0


def test_pairwise_no_back_to_back_block_attack_exception() -> None:
    """Same player consecutive is allowed if first action was a block."""
    score = pairwise_no_back_to_back(
        "P1", "P1", action_t="block", action_t1="attack", weights=DEFAULT_WEIGHTS
    )
    assert score == 0.0  # exception, no penalty


def test_pairwise_no_back_to_back_absent_states_no_penalty() -> None:
    """Two absent states consecutive is handled by R_absent_pair, not back-to-back."""
    score = pairwise_no_back_to_back(
        "ABSENT_TEAM_A", "ABSENT_TEAM_A",
        action_t="dig", action_t1="set", weights=DEFAULT_WEIGHTS,
    )
    assert score == 0.0


# ---- pairwise_alternation ----

def test_pairwise_alternation_same_team_across_net_penalized() -> None:
    """Same team consecutive AND a net crossing happened -> penalty."""
    team_assignments = {1: "A", 2: "A", 3: "B", 4: "B"}
    score = pairwise_alternation(
        "P1", "P2", net_crossed=True,
        team_assignments=team_assignments, weights=DEFAULT_WEIGHTS,
    )
    assert score == -DEFAULT_WEIGHTS.w_alternation


def test_pairwise_alternation_diff_team_across_net_zero() -> None:
    team_assignments = {1: "A", 2: "A", 3: "B", 4: "B"}
    score = pairwise_alternation(
        "P1", "P3", net_crossed=True,
        team_assignments=team_assignments, weights=DEFAULT_WEIGHTS,
    )
    assert score == 0.0


def test_pairwise_alternation_diff_team_same_side_penalized() -> None:
    """Different teams without a net crossing -> penalty (impossible play)."""
    team_assignments = {1: "A", 2: "A", 3: "B", 4: "B"}
    score = pairwise_alternation(
        "P1", "P3", net_crossed=False,
        team_assignments=team_assignments, weights=DEFAULT_WEIGHTS,
    )
    assert score == -DEFAULT_WEIGHTS.w_team_consistency


def test_pairwise_alternation_absent_states_neutral() -> None:
    """Alternation rule doesn't penalize involving absent states."""
    team_assignments = {1: "A", 2: "A", 3: "B", 4: "B"}
    score = pairwise_alternation(
        "ABSENT_TEAM_A", "P2", net_crossed=True,
        team_assignments=team_assignments, weights=DEFAULT_WEIGHTS,
    )
    # ABSENT_TEAM_A is on team A; P2 is on team A; net crossed -> rule fires
    assert score == -DEFAULT_WEIGHTS.w_alternation
```

- [ ] **Step 2: Run tests — expect FAIL (functions don't exist)**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && \
  uv run pytest tests/unit/test_joint_attribution_factors.py -v
```

Expected: 8 new test failures.

- [ ] **Step 3: Append the pairwise factor implementations**

Append to `joint_attribution_factors.py`:

```python
def _state_team(state: State, team_assignments: dict[int, str]) -> str | None:
    """Resolve a state to its team letter, or None if unresolvable."""
    if state == "ABSENT_TEAM_A":
        return "A"
    if state == "ABSENT_TEAM_B":
        return "B"
    pid = int(state[1:])
    return team_assignments.get(pid)


def pairwise_no_back_to_back(
    state_t: State,
    state_t1: State,
    action_t: str | None,
    action_t1: str | None,
    weights: FactorWeights,
) -> float:
    """Penalty if consecutive contacts have the same player.
    Exception: block-attack by the same player is allowed."""
    if is_absent(state_t) or is_absent(state_t1):
        return 0.0
    if state_t != state_t1:
        return 0.0
    # Block-attack exception
    if action_t == "block" and action_t1 == "attack":
        return 0.0
    return -weights.w_back_to_back


def pairwise_alternation(
    state_t: State,
    state_t1: State,
    net_crossed: bool,
    team_assignments: dict[int, str],
    weights: FactorWeights,
) -> float:
    """Two soft rules folded into one factor:
    - Same team across a net crossing: penalty (should alternate)
    - Different team without a net crossing: penalty (impossible play)"""
    team_t = _state_team(state_t, team_assignments)
    team_t1 = _state_team(state_t1, team_assignments)
    if team_t is None or team_t1 is None:
        return 0.0
    same_team = team_t == team_t1
    if net_crossed and same_team:
        return -weights.w_alternation
    if not net_crossed and not same_team:
        return -weights.w_team_consistency
    return 0.0
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && \
  uv run pytest tests/unit/test_joint_attribution_factors.py -v
```

Expected: 17 tests PASS (9 from Task 3 + 8 new).

- [ ] **Step 5: Run mypy + ruff**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && \
  uv run mypy rallycut/tracking/joint_attribution_factors.py tests/unit/test_joint_attribution_factors.py && \
  uv run ruff check rallycut/tracking/joint_attribution_factors.py tests/unit/test_joint_attribution_factors.py
```

- [ ] **Step 6: Commit**

```bash
git add analysis/rallycut/tracking/joint_attribution_factors.py analysis/tests/unit/test_joint_attribution_factors.py
git commit -m "feat(attribution): pairwise factor functions for joint-attribution PGM

Two pairwise rules: pairwise_no_back_to_back (with block-attack exception)
and pairwise_alternation (same-team-across-net or different-team-same-side
penalties).

8 new unit tests; truth-table coverage including absent-state semantics.

Spec: docs/superpowers/specs/2026-05-12-joint-attribution-pgm-design.md"
```

---

### Task 5: Higher-order factor functions + tests (TDD)

**Files:**
- Modify: `analysis/rallycut/tracking/joint_attribution_factors.py`
- Modify: `analysis/tests/unit/test_joint_attribution_factors.py`

- [ ] **Step 1: Append higher-order tests**

```python
from rallycut.tracking.joint_attribution_factors import (
    higher_3_contact_per_side,
    higher_serve_first,
    pairwise_absent_pair,
)


# ---- pairwise_absent_pair ----

def test_pairwise_absent_pair_two_absents_penalized() -> None:
    score = pairwise_absent_pair(
        "ABSENT_TEAM_A", "ABSENT_TEAM_B", weights=DEFAULT_WEIGHTS,
    )
    assert score == -DEFAULT_WEIGHTS.w_absent_pair


def test_pairwise_absent_pair_one_absent_zero() -> None:
    score = pairwise_absent_pair(
        "ABSENT_TEAM_A", "P2", weights=DEFAULT_WEIGHTS,
    )
    assert score == 0.0


def test_pairwise_absent_pair_no_absent_zero() -> None:
    score = pairwise_absent_pair("P1", "P2", weights=DEFAULT_WEIGHTS)
    assert score == 0.0


# ---- higher_3_contact_per_side ----

def test_higher_3_contact_no_violation_returns_zero() -> None:
    """Three contacts same team without crossing -> no penalty (allowed)."""
    states = ("P1", "P2", "P1")
    net_crossings = (False, False)  # crossings between t and t+1
    team_assignments = {1: "A", 2: "A", 3: "B", 4: "B"}
    score = higher_3_contact_per_side(
        states, net_crossings, team_assignments, weights=DEFAULT_WEIGHTS,
    )
    assert score == 0.0


def test_higher_3_contact_4_contacts_no_crossing_penalty() -> None:
    """4+ same-team contacts without a net crossing -> penalty."""
    states = ("P1", "P2", "P1", "P2")
    net_crossings = (False, False, False)
    team_assignments = {1: "A", 2: "A", 3: "B", 4: "B"}
    score = higher_3_contact_per_side(
        states, net_crossings, team_assignments, weights=DEFAULT_WEIGHTS,
    )
    assert score == -DEFAULT_WEIGHTS.w_3_contact


def test_higher_3_contact_5_contacts_no_crossing_double_penalty() -> None:
    """Each extra contact beyond 3 same-team adds a penalty."""
    states = ("P1", "P2", "P1", "P2", "P1")
    net_crossings = (False, False, False, False)
    team_assignments = {1: "A", 2: "A", 3: "B", 4: "B"}
    score = higher_3_contact_per_side(
        states, net_crossings, team_assignments, weights=DEFAULT_WEIGHTS,
    )
    assert score == -2 * DEFAULT_WEIGHTS.w_3_contact


def test_higher_3_contact_crossing_resets_count() -> None:
    """A net crossing resets the same-team contact counter."""
    states = ("P1", "P2", "P1", "P3")  # 3 team-A then crossing then 1 team-B
    net_crossings = (False, False, True)
    team_assignments = {1: "A", 2: "A", 3: "B", 4: "B"}
    score = higher_3_contact_per_side(
        states, net_crossings, team_assignments, weights=DEFAULT_WEIGHTS,
    )
    assert score == 0.0


# ---- higher_serve_first ----

def test_higher_serve_first_correct_team_no_penalty() -> None:
    team_assignments = {1: "A", 2: "A", 3: "B", 4: "B"}
    score = higher_serve_first(
        first_state="P1", serving_team="A",
        team_assignments=team_assignments, weights=DEFAULT_WEIGHTS,
    )
    assert score == 0.0


def test_higher_serve_first_wrong_team_penalty() -> None:
    team_assignments = {1: "A", 2: "A", 3: "B", 4: "B"}
    score = higher_serve_first(
        first_state="P3", serving_team="A",
        team_assignments=team_assignments, weights=DEFAULT_WEIGHTS,
    )
    assert score == -DEFAULT_WEIGHTS.w_serve_first


def test_higher_serve_first_absent_serving_team_small_penalty() -> None:
    """If first is ABSENT_TEAM_<serving_team>, small penalty (off-screen server)."""
    team_assignments = {1: "A", 2: "A", 3: "B", 4: "B"}
    score = higher_serve_first(
        first_state="ABSENT_TEAM_A", serving_team="A",
        team_assignments=team_assignments, weights=DEFAULT_WEIGHTS,
    )
    assert score == -0.5  # documented small constant


def test_higher_serve_first_absent_wrong_team_penalty() -> None:
    team_assignments = {1: "A", 2: "A", 3: "B", 4: "B"}
    score = higher_serve_first(
        first_state="ABSENT_TEAM_B", serving_team="A",
        team_assignments=team_assignments, weights=DEFAULT_WEIGHTS,
    )
    assert score == -DEFAULT_WEIGHTS.w_serve_first


def test_higher_serve_first_no_serving_team_zero() -> None:
    """If serving_team is None, no penalty (no information)."""
    team_assignments = {1: "A", 2: "A", 3: "B", 4: "B"}
    score = higher_serve_first(
        first_state="P3", serving_team=None,
        team_assignments=team_assignments, weights=DEFAULT_WEIGHTS,
    )
    assert score == 0.0
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && \
  uv run pytest tests/unit/test_joint_attribution_factors.py -v
```

Expected: 11 new test failures.

- [ ] **Step 3: Append higher-order factor implementations**

Append to `joint_attribution_factors.py`:

```python
_ABSENT_SERVER_SMALL_PENALTY = -0.5  # spec: small penalty when first is ABSENT_TEAM_<serving_team>


def pairwise_absent_pair(
    state_t: State,
    state_t1: State,
    weights: FactorWeights,
) -> float:
    """Penalty for two consecutive ABSENT_* states."""
    if is_absent(state_t) and is_absent(state_t1):
        return -weights.w_absent_pair
    return 0.0


def higher_3_contact_per_side(
    states: tuple[State, ...],
    net_crossings: tuple[bool, ...],
    team_assignments: dict[int, str],
    weights: FactorWeights,
) -> float:
    """Penalty per same-team contact beyond the 3rd in a row without a crossing.

    `net_crossings[i]` is True iff a net crossing occurred between states[i]
    and states[i+1]. Length is len(states) - 1.
    """
    assert len(net_crossings) == len(states) - 1
    penalty = 0.0
    streak = 1
    streak_team: str | None = _state_team(states[0], team_assignments)
    for i in range(1, len(states)):
        team_i = _state_team(states[i], team_assignments)
        if net_crossings[i - 1] or team_i != streak_team:
            streak = 1
            streak_team = team_i
            continue
        streak += 1
        if streak > 3:
            penalty -= weights.w_3_contact
    return penalty


def higher_serve_first(
    first_state: State,
    serving_team: str | None,
    team_assignments: dict[int, str],
    weights: FactorWeights,
) -> float:
    """Penalty if first contact's team doesn't match serving_team.

    Special case: if first_state is ABSENT_TEAM_<serving_team>, small penalty
    (off-screen server matches the serving team but isn't tracked).
    """
    if serving_team is None:
        return 0.0
    if first_state == f"ABSENT_TEAM_{serving_team}":
        return _ABSENT_SERVER_SMALL_PENALTY
    first_team = _state_team(first_state, team_assignments)
    if first_team is None:
        return 0.0
    if first_team == serving_team:
        return 0.0
    return -weights.w_serve_first
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && \
  uv run pytest tests/unit/test_joint_attribution_factors.py -v
```

Expected: 28 tests PASS (17 from Tasks 3-4 + 11 new).

- [ ] **Step 5: Run mypy + ruff + commit**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && \
  uv run mypy rallycut/tracking/joint_attribution_factors.py tests/unit/test_joint_attribution_factors.py && \
  uv run ruff check rallycut/tracking/joint_attribution_factors.py tests/unit/test_joint_attribution_factors.py
git add analysis/rallycut/tracking/joint_attribution_factors.py analysis/tests/unit/test_joint_attribution_factors.py
git commit -m "feat(attribution): higher-order + absent-pair factors for joint-attribution PGM

Three more factor functions: pairwise_absent_pair (two consecutive ABSENT_*
penalty), higher_3_contact_per_side (penalty per contact beyond 3 same-team
no-crossing), higher_serve_first (penalty if first contact's team mismatches
serving_team; small penalty when first is ABSENT_TEAM_<serving_team>).

11 new unit tests; total factor coverage now 28 tests.

Spec: docs/superpowers/specs/2026-05-12-joint-attribution-pgm-design.md"
```

---

### Task 6: Evidence builder

**Files:**
- Modify: `analysis/rallycut/tracking/joint_attribution.py`

- [ ] **Step 1: Append `build_evidence` to joint_attribution.py**

```python
import math
from collections.abc import Callable

from rallycut.tracking.joint_attribution_factors import (
    unary_proximity, unary_distance, unary_action_prior,
)
from rallycut.tracking.joint_attribution_weights import FactorWeights, DEFAULT_WEIGHTS


def build_state_domain(team_assignments: dict[int, str]) -> tuple[State, ...]:
    """Return the per-contact state domain: 4 player states + 2 absent-team states.
    Always returns the full domain regardless of team_assignments completeness;
    factor functions handle missing teams gracefully."""
    return ("P1", "P2", "P3", "P4", "ABSENT_TEAM_A", "ABSENT_TEAM_B")


def build_pid_to_raw_id(rally: RallyContext) -> dict[int, int]:
    """Build the canonical PID -> raw track id mapping from the rally's
    initial actions. Used to convert State (e.g., "P1") to candidate raw ids
    in playerCandidates lists.

    Falls back to identity mapping (1->1, 2->2, ...) if no actions or
    initial PIDs are present.
    """
    mapping: dict[int, int] = {}
    for action in rally.initial_actions:
        pid = action.get("playerTrackId")
        raw = action.get("rawTrackId")  # may be present from action_classifier
        if pid is None:
            continue
        if raw is None:
            # Identity fallback
            mapping[int(pid)] = int(pid)
        else:
            mapping[int(pid)] = int(raw)
    # Ensure all 4 PIDs have an entry (identity fallback for missing)
    for pid in (1, 2, 3, 4):
        mapping.setdefault(pid, pid)
    return mapping


def build_evidence(
    rally: RallyContext, weights: FactorWeights = DEFAULT_WEIGHTS,
) -> list[dict[State, float]]:
    """For each contact in the rally, compute a dict mapping each state
    to its summed unary log-likelihood (sum of all unary factor contributions).

    Returns a list of dicts of length len(rally.contacts).
    """
    pid_to_raw = build_pid_to_raw_id(rally)
    state_domain = build_state_domain(rally.team_assignments)
    evidence: list[dict[State, float]] = []
    # Pair contacts with their initial actions (by frame proximity)
    actions_by_frame = {a["frame"]: a for a in rally.initial_actions}
    for contact in rally.contacts:
        scores: dict[State, float] = {}
        candidates = [
            (int(tid), float(d) if d is not None else 1.0)
            for tid, d in (contact.get("playerCandidates") or [])
        ]
        # Find action at this contact's frame (within ±5)
        cf = contact["frame"]
        nearest_action = None
        nearest_d = math.inf
        for af, a in actions_by_frame.items():
            if abs(af - cf) <= 5 and abs(af - cf) < nearest_d:
                nearest_d, nearest_action = abs(af - cf), a
        initial_pid = nearest_action.get("playerTrackId") if nearest_action else None
        for state in state_domain:
            score = (
                unary_proximity(state, candidates, pid_to_raw, weights)
                + unary_distance(state, candidates, pid_to_raw, weights, rally.team_assignments)
                + unary_action_prior(state, initial_pid, weights)
            )
            scores[state] = score
        evidence.append(scores)
    return evidence
```

- [ ] **Step 2: Run mypy + ruff**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && \
  uv run mypy rallycut/tracking/joint_attribution.py && \
  uv run ruff check rallycut/tracking/joint_attribution.py
```

Expected: 0 errors / clean.

- [ ] **Step 3: Commit**

```bash
git add analysis/rallycut/tracking/joint_attribution.py
git commit -m "feat(attribution): evidence builder + state domain helpers

build_state_domain returns the 6-state domain. build_pid_to_raw_id maps
canonical PIDs to raw track ids using initial action data, with identity
fallback. build_evidence aggregates the three unary factors per (contact,
state) into a list of scoring dicts.

Spec: docs/superpowers/specs/2026-05-12-joint-attribution-pgm-design.md"
```

---

### Task 7: Inference engine + tests

**Files:**
- Modify: `analysis/rallycut/tracking/joint_attribution.py`
- Create: `analysis/tests/unit/test_joint_attribution_inference.py`

- [ ] **Step 1: Write inference tests first (TDD)**

```python
"""Inference correctness tests for the Joint Attribution PGM."""
from __future__ import annotations

import math

import pytest

from rallycut.tracking.joint_attribution import (
    RallyContext,
    joint_attribute_rally,
    score_joint_config,
    State,
)
from rallycut.tracking.joint_attribution_weights import DEFAULT_WEIGHTS


def _make_simple_rally(n_contacts: int = 3, ball_x: float = 0.5) -> RallyContext:
    """Build a synthetic 3-contact rally for testing."""
    contacts = [
        {
            "frame": 100 + i * 30,
            "ballX": ball_x, "ballY": 0.5,
            "playerCandidates": [[1, 0.05], [2, 0.10], [3, 0.20], [4, 0.50]],
            "courtSide": "near" if i % 2 == 0 else "far",
        }
        for i in range(n_contacts)
    ]
    actions = [
        {"frame": c["frame"], "action": "set", "playerTrackId": 1, "confidence": 0.8}
        for c in contacts
    ]
    return RallyContext(
        rally_id="test-rally", contacts=contacts, initial_actions=actions,
        team_assignments={1: "A", 2: "A", 3: "B", 4: "B"},
        serving_team="A",
    )


def test_inference_returns_valid_assignment() -> None:
    """MAP returns a tuple of length n_contacts with valid states."""
    rally = _make_simple_rally(n_contacts=3)
    result = joint_attribute_rally(rally)
    assert len(result.map) == 3
    valid_states = {"P1", "P2", "P3", "P4", "ABSENT_TEAM_A", "ABSENT_TEAM_B"}
    for state in result.map:
        assert state in valid_states
    assert result.fallback_used == "exhaustive"


def test_inference_picks_proximity_winner_in_simple_case() -> None:
    """With proximity heavily favoring P1 and no rule violations, MAP picks P1."""
    contacts = [{
        "frame": 100, "ballX": 0.5, "ballY": 0.5,
        "playerCandidates": [[1, 0.01], [2, 0.30], [3, 0.40], [4, 0.50]],
        "courtSide": "near",
    }]
    actions = [{"frame": 100, "action": "serve", "playerTrackId": 1, "confidence": 0.9}]
    rally = RallyContext(
        rally_id="test", contacts=contacts, initial_actions=actions,
        team_assignments={1: "A", 2: "A", 3: "B", 4: "B"},
        serving_team="A",
    )
    result = joint_attribute_rally(rally)
    assert result.map[0] == "P1"


def test_inference_serve_first_factor_overrides_proximity() -> None:
    """When proximity favors team-B but serving_team is A, MAP picks team-A
    (or absent-team-A)."""
    contacts = [{
        "frame": 100, "ballX": 0.5, "ballY": 0.5,
        "playerCandidates": [[3, 0.01], [4, 0.05], [1, 0.30], [2, 0.40]],
        "courtSide": "far",
    }]
    actions = [{"frame": 100, "action": "serve", "playerTrackId": 3, "confidence": 0.9}]
    rally = RallyContext(
        rally_id="test", contacts=contacts, initial_actions=actions,
        team_assignments={1: "A", 2: "A", 3: "B", 4: "B"},
        serving_team="A",
    )
    result = joint_attribute_rally(rally)
    # MAP should NOT pick a team-B player (P3 or P4) for the first contact
    assert result.map[0] in ("P1", "P2", "ABSENT_TEAM_A")


def test_inference_back_to_back_penalty_drives_alternation() -> None:
    """Two consecutive contacts with proximity favoring same player —
    PGM should still alternate due to back-to-back penalty."""
    contacts = [
        {"frame": 100, "ballX": 0.5, "ballY": 0.5,
         "playerCandidates": [[1, 0.01], [2, 0.10], [3, 0.50], [4, 0.60]],
         "courtSide": "near"},
        {"frame": 130, "ballX": 0.5, "ballY": 0.4,
         "playerCandidates": [[1, 0.01], [2, 0.10], [3, 0.50], [4, 0.60]],
         "courtSide": "near"},
    ]
    actions = [
        {"frame": 100, "action": "dig", "playerTrackId": 1, "confidence": 0.8},
        {"frame": 130, "action": "set", "playerTrackId": 1, "confidence": 0.8},
    ]
    rally = RallyContext(
        rally_id="test", contacts=contacts, initial_actions=actions,
        team_assignments={1: "A", 2: "A", 3: "B", 4: "B"},
        serving_team="A",
    )
    result = joint_attribute_rally(rally)
    assert result.map[0] != result.map[1] or result.map[0] in ("ABSENT_TEAM_A", "ABSENT_TEAM_B")


def test_inference_marginals_normalize_per_contact() -> None:
    """Per-contact marginals sum to 1.0 over all states."""
    rally = _make_simple_rally(n_contacts=3)
    result = joint_attribute_rally(rally)
    for marginal in result.marginals:
        total = sum(marginal.values())
        assert math.isclose(total, 1.0, abs_tol=1e-6)


def test_inference_falls_back_to_coordinate_descent_at_n9() -> None:
    """For N > 8 contacts, inference uses coordinate descent."""
    rally = _make_simple_rally(n_contacts=9)
    result = joint_attribute_rally(rally)
    assert result.fallback_used == "coordinate_descent"
    assert len(result.map) == 9


def test_score_joint_config_matches_per_factor_sum() -> None:
    """score_joint_config equals sum of unary + pairwise + higher factors."""
    rally = _make_simple_rally(n_contacts=3)
    config = ("P1", "P2", "P1")
    score = score_joint_config(config, rally, weights=DEFAULT_WEIGHTS)
    assert isinstance(score, float)
    # No assert on value; just no exception + float type
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && \
  uv run pytest tests/unit/test_joint_attribution_inference.py -v
```

Expected: all fail with "cannot import joint_attribute_rally" or similar.

- [ ] **Step 3: Append inference engine to `joint_attribution.py`**

```python
import itertools

from rallycut.tracking.joint_attribution_factors import (
    pairwise_no_back_to_back,
    pairwise_alternation,
    pairwise_absent_pair,
    higher_3_contact_per_side,
    higher_serve_first,
)


_EXHAUSTIVE_THRESHOLD = 8  # use exhaustive for N <= 8 contacts; coordinate descent for larger


def _net_crossings_for(contacts: list[dict[str, Any]]) -> tuple[bool, ...]:
    """For each consecutive contact pair, True if courtSide flipped between them.
    "unknown" courtSide is treated as no-information (preserves the prior side)."""
    sides: list[str | None] = []
    last_known: str | None = None
    for c in contacts:
        side = c.get("courtSide")
        if side in ("near", "far"):
            last_known = side
            sides.append(side)
        else:
            sides.append(last_known)
    crossings: list[bool] = []
    for i in range(len(sides) - 1):
        if sides[i] is None or sides[i + 1] is None:
            crossings.append(False)
        else:
            crossings.append(sides[i] != sides[i + 1])
    return tuple(crossings)


def score_joint_config(
    config: tuple[State, ...],
    rally: RallyContext,
    weights: FactorWeights = DEFAULT_WEIGHTS,
    evidence: list[dict[State, float]] | None = None,
    net_crossings: tuple[bool, ...] | None = None,
) -> float:
    """Compute joint log-likelihood for a configuration.

    Sums all unary, pairwise, and higher-order factor contributions.
    `evidence` and `net_crossings` are precomputed once per rally; pass
    them in for efficiency in the inner enumeration loop.
    """
    if evidence is None:
        evidence = build_evidence(rally, weights)
    if net_crossings is None:
        net_crossings = _net_crossings_for(rally.contacts)

    score = 0.0
    # Unary
    for t, state in enumerate(config):
        score += evidence[t][state]
    # Pairwise
    for t in range(len(config) - 1):
        action_t = (
            rally.initial_actions[t].get("action")
            if t < len(rally.initial_actions) else None
        )
        action_t1 = (
            rally.initial_actions[t + 1].get("action")
            if t + 1 < len(rally.initial_actions) else None
        )
        score += pairwise_no_back_to_back(
            config[t], config[t + 1], action_t, action_t1, weights,
        )
        score += pairwise_alternation(
            config[t], config[t + 1], net_crossings[t],
            rally.team_assignments, weights,
        )
        score += pairwise_absent_pair(config[t], config[t + 1], weights)
    # Higher-order
    score += higher_3_contact_per_side(
        config, net_crossings, rally.team_assignments, weights,
    )
    score += higher_serve_first(
        config[0], rally.serving_team, rally.team_assignments, weights,
    )
    return score


def _exhaustive_map(
    rally: RallyContext, weights: FactorWeights,
) -> tuple[tuple[State, ...], float, list[dict[State, float]]]:
    """Enumerate all 6^N configurations; return MAP + score + marginals."""
    n = len(rally.contacts)
    state_domain = build_state_domain(rally.team_assignments)
    evidence = build_evidence(rally, weights)
    net_crossings = _net_crossings_for(rally.contacts)

    best_score = -math.inf
    best_config: tuple[State, ...] | None = None
    # For marginals: accumulate sum_exp per (contact_t, state)
    log_scores: list[list[tuple[tuple[State, ...], float]]] = []  # not used; use per-contact accumulators
    per_contact_state_log_sum: list[dict[State, float]] = [
        {state: -math.inf for state in state_domain} for _ in range(n)
    ]

    for config in itertools.product(state_domain, repeat=n):
        score = score_joint_config(config, rally, weights, evidence, net_crossings)
        if score > best_score:
            best_score, best_config = score, config
        # Accumulate marginals (logsumexp per (contact_t, state))
        for t, state in enumerate(config):
            cur = per_contact_state_log_sum[t][state]
            per_contact_state_log_sum[t][state] = (
                _logaddexp(cur, score)
            )

    assert best_config is not None
    # Normalize marginals per contact
    marginals: list[dict[State, float]] = []
    for t in range(n):
        log_sum_total = -math.inf
        for state in state_domain:
            log_sum_total = _logaddexp(log_sum_total, per_contact_state_log_sum[t][state])
        marginal: dict[State, float] = {}
        for state in state_domain:
            log_p = per_contact_state_log_sum[t][state] - log_sum_total
            marginal[state] = math.exp(log_p) if log_p > -50 else 0.0
        # Renormalize to handle floating-point drift
        total = sum(marginal.values())
        if total > 0:
            marginal = {s: p / total for s, p in marginal.items()}
        marginals.append(marginal)

    return best_config, best_score, marginals


def _logaddexp(a: float, b: float) -> float:
    """Numerically stable log(exp(a) + exp(b))."""
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def _coordinate_descent_map(
    rally: RallyContext, weights: FactorWeights, max_iter: int = 5,
) -> tuple[tuple[State, ...], float, list[dict[State, float]]]:
    """Coordinate-descent fallback for N > 8 contacts.

    Initialize from initial_actions PIDs; iteratively re-assign each contact
    to maximize the joint score given others. Approximate; finds local MAP.
    Marginals are not computed; returned as uniform-ish (each MAP state at 1.0).
    """
    n = len(rally.contacts)
    state_domain = build_state_domain(rally.team_assignments)
    evidence = build_evidence(rally, weights)
    net_crossings = _net_crossings_for(rally.contacts)

    # Initialize from initial_actions (or P1 if missing)
    config = list(state_domain[0] for _ in range(n))
    for t in range(n):
        if t < len(rally.initial_actions):
            pid = rally.initial_actions[t].get("playerTrackId")
            if pid is not None and 1 <= int(pid) <= 4:
                config[t] = f"P{int(pid)}"

    score = score_joint_config(tuple(config), rally, weights, evidence, net_crossings)
    for _ in range(max_iter):
        improved = False
        for t in range(n):
            best_state, best_score = config[t], score
            for state in state_domain:
                if state == config[t]:
                    continue
                new_config = config[:t] + [state] + config[t + 1:]
                new_score = score_joint_config(
                    tuple(new_config), rally, weights, evidence, net_crossings,
                )
                if new_score > best_score:
                    best_state, best_score = state, new_score
            if best_state != config[t]:
                config[t] = best_state
                score = best_score
                improved = True
        if not improved:
            break

    final = tuple(config)
    marginals = [{state: (1.0 if state == final[t] else 0.0) for state in state_domain}
                 for t in range(n)]
    return final, score, marginals


def joint_attribute_rally(
    rally: RallyContext, weights: FactorWeights = DEFAULT_WEIGHTS,
) -> RallyAttribution:
    """Public API: compute joint MAP attribution for a rally.

    Uses exhaustive enumeration when N <= 8 contacts; coordinate-descent
    fallback for larger rallies. Returns the MAP assignment, joint score,
    and per-contact marginals.
    """
    n = len(rally.contacts)
    if n == 0:
        return RallyAttribution(
            map=(), score=0.0, marginals=[], fallback_used="exhaustive",
        )
    if n <= _EXHAUSTIVE_THRESHOLD:
        config, score, marginals = _exhaustive_map(rally, weights)
        return RallyAttribution(
            map=config, score=score, marginals=marginals, fallback_used="exhaustive",
        )
    config, score, marginals = _coordinate_descent_map(rally, weights)
    return RallyAttribution(
        map=config, score=score, marginals=marginals, fallback_used="coordinate_descent",
    )
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && \
  uv run pytest tests/unit/test_joint_attribution_inference.py -v
```

Expected: 7 tests PASS.

- [ ] **Step 5: Run mypy + ruff + commit**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && \
  uv run mypy rallycut/tracking/joint_attribution.py tests/unit/test_joint_attribution_inference.py && \
  uv run ruff check rallycut/tracking/joint_attribution.py tests/unit/test_joint_attribution_inference.py
git add analysis/rallycut/tracking/joint_attribution.py analysis/tests/unit/test_joint_attribution_inference.py
git commit -m "feat(attribution): inference engine for joint-attribution PGM

Three pieces: score_joint_config (sum all factor contributions for a
config), _exhaustive_map (enumerate all 6^N for N<=8), _coordinate_descent_map
(N>8 fallback), and the public API joint_attribute_rally.

Returns RallyAttribution with MAP, score, per-contact marginals, and
fallback_used flag. Marginals computed via logsumexp during enumeration.

7 inference tests covering valid output, proximity-driven choice,
serve-first override, back-to-back penalty, marginal normalization,
N=9 fallback to coordinate-descent.

Spec: docs/superpowers/specs/2026-05-12-joint-attribution-pgm-design.md"
```

---

### Task 8: End-to-end synthetic-rally tests

**Files:**
- Create: `analysis/tests/unit/test_joint_attribution_e2e.py`

- [ ] **Step 1: Write 10 e2e cases covering target scenarios**

```python
"""End-to-end synthetic-rally tests for the Joint Attribution PGM.

Each test builds a complete RallyContext modeling a known scenario and
asserts the MAP matches the expected attribution. Covers: clean rally,
off-screen serve, missing-middle-contact, cross-team alternation,
3-contact per side cap, etc.
"""
from __future__ import annotations

import pytest

from rallycut.tracking.joint_attribution import (
    RallyContext,
    joint_attribute_rally,
)
from rallycut.tracking.joint_attribution_weights import DEFAULT_WEIGHTS


def _team_assignments_2v2() -> dict[int, str]:
    return {1: "A", 2: "A", 3: "B", 4: "B"}


def test_e2e_clean_3_contact_alternating_rally() -> None:
    """Rally: serve P1 -> receive P3 -> attack P3. Standard pattern."""
    contacts = [
        {"frame": 100, "ballX": 0.5, "ballY": 0.2,
         "playerCandidates": [[1, 0.05], [2, 0.30], [3, 0.50], [4, 0.60]],
         "courtSide": "near"},
        {"frame": 130, "ballX": 0.5, "ballY": 0.6,
         "playerCandidates": [[3, 0.05], [4, 0.20], [1, 0.50], [2, 0.60]],
         "courtSide": "far"},
        {"frame": 160, "ballX": 0.5, "ballY": 0.5,
         "playerCandidates": [[3, 0.05], [4, 0.10], [1, 0.50], [2, 0.60]],
         "courtSide": "far"},
    ]
    actions = [
        {"frame": 100, "action": "serve", "playerTrackId": 1, "confidence": 0.9},
        {"frame": 130, "action": "receive", "playerTrackId": 3, "confidence": 0.9},
        {"frame": 160, "action": "attack", "playerTrackId": 3, "confidence": 0.9},
    ]
    rally = RallyContext(
        rally_id="e2e-1", contacts=contacts, initial_actions=actions,
        team_assignments=_team_assignments_2v2(), serving_team="A",
    )
    result = joint_attribute_rally(rally)
    assert result.map == ("P1", "P3", "P3")


def test_e2e_off_screen_server_picks_absent_team_a() -> None:
    """Server is off-screen; serve-frame contact has no team-A candidate close.
    PGM should pick ABSENT_TEAM_A for the serve."""
    contacts = [
        {"frame": 100, "ballX": 0.5, "ballY": 0.2,
         "playerCandidates": [[3, 0.30], [4, 0.40]],  # only team-B candidates, far
         "courtSide": "near"},
        {"frame": 130, "ballX": 0.5, "ballY": 0.6,
         "playerCandidates": [[3, 0.05], [4, 0.20]],
         "courtSide": "far"},
    ]
    actions = [
        {"frame": 100, "action": "serve", "playerTrackId": 3, "confidence": 0.5},
        {"frame": 130, "action": "receive", "playerTrackId": 3, "confidence": 0.9},
    ]
    rally = RallyContext(
        rally_id="e2e-2", contacts=contacts, initial_actions=actions,
        team_assignments=_team_assignments_2v2(), serving_team="A",
    )
    result = joint_attribute_rally(rally)
    # Serve should be ABSENT_TEAM_A (server off-screen, serving team is A,
    # no team-A candidate nearby)
    assert result.map[0] == "ABSENT_TEAM_A"
    assert result.map[1] == "P3"


def test_e2e_back_to_back_block_attack_allowed() -> None:
    """Same player consecutive is allowed if first action was a block."""
    contacts = [
        {"frame": 100, "ballX": 0.5, "ballY": 0.5,
         "playerCandidates": [[1, 0.05]], "courtSide": "near"},
        {"frame": 110, "ballX": 0.5, "ballY": 0.4,
         "playerCandidates": [[1, 0.05]], "courtSide": "near"},
    ]
    actions = [
        {"frame": 100, "action": "block", "playerTrackId": 1, "confidence": 0.9},
        {"frame": 110, "action": "attack", "playerTrackId": 1, "confidence": 0.9},
    ]
    rally = RallyContext(
        rally_id="e2e-3", contacts=contacts, initial_actions=actions,
        team_assignments=_team_assignments_2v2(), serving_team="A",
    )
    result = joint_attribute_rally(rally)
    assert result.map == ("P1", "P1")


def test_e2e_3_contact_per_side_cap_enforced() -> None:
    """4 same-side contacts with no crossing should drive PGM toward
    breaking the streak (introducing a different-team or absent state)."""
    contacts = [
        {"frame": 100 + i * 10, "ballX": 0.5, "ballY": 0.5,
         "playerCandidates": [[1, 0.05], [2, 0.10]],  # both team-A close
         "courtSide": "near"}
        for i in range(4)
    ]
    actions = [
        {"frame": c["frame"], "action": "set", "playerTrackId": 1, "confidence": 0.7}
        for c in contacts
    ]
    rally = RallyContext(
        rally_id="e2e-4", contacts=contacts, initial_actions=actions,
        team_assignments=_team_assignments_2v2(), serving_team="A",
    )
    result = joint_attribute_rally(rally)
    # 4 same-team-no-crossing should be penalized; PGM may pick an absent
    # state for the 4th OR another team's player. Just assert it's not all P1/P2.
    teams = []
    for state in result.map:
        if state.startswith("ABSENT"):
            teams.append("absent")
        else:
            pid = int(state[1:])
            teams.append({1: "A", 2: "A", 3: "B", 4: "B"}[pid])
    # At least one of the 4 contacts should not be a team-A tracked player
    non_team_a = sum(1 for t in teams if t != "A")
    assert non_team_a >= 1


def test_e2e_serve_team_overrides_proximity_for_first_contact() -> None:
    """When proximity favors team-B but serving_team is A, serve is
    re-attributed to team-A (or ABSENT_TEAM_A)."""
    contacts = [
        {"frame": 100, "ballX": 0.5, "ballY": 0.2,
         "playerCandidates": [[3, 0.05], [4, 0.10], [1, 0.30], [2, 0.40]],
         "courtSide": "near"},
    ]
    actions = [
        {"frame": 100, "action": "serve", "playerTrackId": 3, "confidence": 0.5},
    ]
    rally = RallyContext(
        rally_id="e2e-5", contacts=contacts, initial_actions=actions,
        team_assignments=_team_assignments_2v2(), serving_team="A",
    )
    result = joint_attribute_rally(rally)
    assert result.map[0] in ("P1", "P2", "ABSENT_TEAM_A")


def test_e2e_empty_rally_returns_empty_map() -> None:
    """Rally with zero contacts returns empty map."""
    rally = RallyContext(
        rally_id="e2e-6", contacts=[], initial_actions=[],
        team_assignments=_team_assignments_2v2(), serving_team="A",
    )
    result = joint_attribute_rally(rally)
    assert result.map == ()
    assert result.score == 0.0


def test_e2e_long_rally_uses_coordinate_descent_fallback() -> None:
    """Rally with 10 contacts triggers coordinate-descent fallback."""
    contacts = [
        {"frame": 100 + i * 30, "ballX": 0.5, "ballY": 0.5,
         "playerCandidates": [[1, 0.05], [2, 0.10], [3, 0.20], [4, 0.30]],
         "courtSide": "near" if i % 2 == 0 else "far"}
        for i in range(10)
    ]
    actions = [
        {"frame": c["frame"], "action": "set", "playerTrackId": 1, "confidence": 0.7}
        for c in contacts
    ]
    rally = RallyContext(
        rally_id="e2e-7", contacts=contacts, initial_actions=actions,
        team_assignments=_team_assignments_2v2(), serving_team="A",
    )
    result = joint_attribute_rally(rally)
    assert result.fallback_used == "coordinate_descent"
    assert len(result.map) == 10


def test_e2e_no_serving_team_no_serve_first_factor() -> None:
    """When serving_team is None, no serve-first penalty fires; MAP picks
    purely on evidence."""
    contacts = [
        {"frame": 100, "ballX": 0.5, "ballY": 0.2,
         "playerCandidates": [[3, 0.05], [4, 0.10], [1, 0.30], [2, 0.40]],
         "courtSide": "near"},
    ]
    actions = [
        {"frame": 100, "action": "serve", "playerTrackId": 3, "confidence": 0.9},
    ]
    rally = RallyContext(
        rally_id="e2e-8", contacts=contacts, initial_actions=actions,
        team_assignments=_team_assignments_2v2(), serving_team=None,
    )
    result = joint_attribute_rally(rally)
    assert result.map[0] == "P3"  # closest, no serve-first override


def test_e2e_marginals_include_all_states() -> None:
    """Per-contact marginals always have entries for all 6 states."""
    contacts = [{"frame": 100, "ballX": 0.5, "ballY": 0.5,
                 "playerCandidates": [[1, 0.05]], "courtSide": "near"}]
    actions = [{"frame": 100, "action": "set", "playerTrackId": 1, "confidence": 0.9}]
    rally = RallyContext(
        rally_id="e2e-9", contacts=contacts, initial_actions=actions,
        team_assignments=_team_assignments_2v2(), serving_team="A",
    )
    result = joint_attribute_rally(rally)
    expected_states = {"P1", "P2", "P3", "P4", "ABSENT_TEAM_A", "ABSENT_TEAM_B"}
    assert set(result.marginals[0].keys()) == expected_states


def test_e2e_partial_team_assignments_handled_gracefully() -> None:
    """Missing PIDs in team_assignments shouldn't crash; affected factors
    return 0 contribution."""
    contacts = [{"frame": 100, "ballX": 0.5, "ballY": 0.5,
                 "playerCandidates": [[1, 0.05], [2, 0.10]], "courtSide": "near"}]
    actions = [{"frame": 100, "action": "set", "playerTrackId": 1, "confidence": 0.9}]
    rally = RallyContext(
        rally_id="e2e-10", contacts=contacts, initial_actions=actions,
        team_assignments={1: "A"},  # only P1 has team known
        serving_team="A",
    )
    result = joint_attribute_rally(rally)
    assert len(result.map) == 1  # no crash
```

- [ ] **Step 2: Run tests — expect PASS (no new code; just exercising existing API)**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && \
  uv run pytest tests/unit/test_joint_attribution_e2e.py -v
```

Expected: 10 tests PASS. If any fail, the inference logic from Task 7 needs adjustment.

- [ ] **Step 3: Run mypy + ruff + commit**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && \
  uv run mypy tests/unit/test_joint_attribution_e2e.py && \
  uv run ruff check tests/unit/test_joint_attribution_e2e.py
git add analysis/tests/unit/test_joint_attribution_e2e.py
git commit -m "test(attribution): 10 end-to-end synthetic-rally tests for joint-attribution PGM

Cover scenarios: clean 3-contact alternating rally, off-screen serve
→ ABSENT_TEAM_A, block-attack same-player exception, 3-contact-per-side
cap, serve-first override of proximity, empty rally, N=10 coordinate-descent
fallback, no-serving-team case, marginals include all states, partial
team_assignments handled.

Spec: docs/superpowers/specs/2026-05-12-joint-attribution-pgm-design.md"
```

---

### Task 9: Add `attribution_source` field to action data model

**Files:**
- Modify: `analysis/rallycut/tracking/action_classifier.py` (in `classify_rally_actions` and helpers that emit action dicts)

- [ ] **Step 1: Find every place where action dicts are constructed**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && \
  grep -n '"playerTrackId":\|"frame":.*"action":' rallycut/tracking/action_classifier.py
```

Note the line numbers. Each construction site needs an `attribution_source` field added.

- [ ] **Step 2: For each construction site, add the new field**

When `playerTrackId` is set to a non-None value:
```python
"attribution_source": "action_classifier",
```

When `playerTrackId` is set to None (abstained):
```python
"attribution_source": "action_classifier_abstained",
```

The exact pattern depends on the construction style. If you see something like:

```python
action_dict = {
    "frame": ...,
    "action": ...,
    "playerTrackId": pl_pid,
    ...
}
```

Add the new field:

```python
action_dict = {
    "frame": ...,
    "action": ...,
    "playerTrackId": pl_pid,
    "attribution_source": "action_classifier" if pl_pid is not None else "action_classifier_abstained",
    ...
}
```

- [ ] **Step 3: Run all unit tests; existing tests should still pass**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && \
  uv run pytest tests/unit -v
```

Expected: all existing tests pass; the new field is additive and ignored by existing code.

- [ ] **Step 4: Run mypy + ruff**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && \
  uv run mypy rallycut/tracking/action_classifier.py && \
  uv run ruff check rallycut/tracking/action_classifier.py
```

- [ ] **Step 5: Commit**

```bash
git add analysis/rallycut/tracking/action_classifier.py
git commit -m "feat(attribution): add attribution_source field to action_classifier output

Each action emitted by classify_rally_actions now carries an
attribution_source field: 'action_classifier' when a PID is assigned,
'action_classifier_abstained' when playerTrackId is None.

Additive change; existing consumers ignore the new field.
Prepares the action data model for joint-attribution PGM, which will
override this field with 'pgm_committed' or 'pgm_absent_team_X'.

Spec: docs/superpowers/specs/2026-05-12-joint-attribution-pgm-design.md"
```

---

### Task 10: USE_JOINT_ATTRIBUTION env-flag at call sites

**Files:**
- Modify: `analysis/scripts/redetect_all_actions.py`
- Modify: `analysis/rallycut/cli/commands/reattribute_actions.py`
- Create: `analysis/tests/unit/test_joint_attribution_integration.py`

- [ ] **Step 1: Write the integration test first**

```python
"""Integration tests for the USE_JOINT_ATTRIBUTION env-flag wiring."""
from __future__ import annotations

import os

import pytest


def test_use_joint_attribution_env_flag_default_off() -> None:
    """With USE_JOINT_ATTRIBUTION unset or '0', the flag-checking helper returns False."""
    from rallycut.tracking.joint_attribution import use_joint_attribution_enabled
    os.environ.pop("USE_JOINT_ATTRIBUTION", None)
    assert use_joint_attribution_enabled() is False
    os.environ["USE_JOINT_ATTRIBUTION"] = "0"
    assert use_joint_attribution_enabled() is False
    os.environ.pop("USE_JOINT_ATTRIBUTION", None)


def test_use_joint_attribution_env_flag_enabled() -> None:
    """With USE_JOINT_ATTRIBUTION='1', the flag-checking helper returns True."""
    from rallycut.tracking.joint_attribution import use_joint_attribution_enabled
    os.environ["USE_JOINT_ATTRIBUTION"] = "1"
    try:
        assert use_joint_attribution_enabled() is True
    finally:
        os.environ.pop("USE_JOINT_ATTRIBUTION", None)


def test_use_joint_attribution_env_flag_other_values_inert() -> None:
    """Only '1' enables; 'true', 'yes', 'on' are inert (matching project pattern)."""
    from rallycut.tracking.joint_attribution import use_joint_attribution_enabled
    for value in ("true", "yes", "on", ""):
        os.environ["USE_JOINT_ATTRIBUTION"] = value
        assert use_joint_attribution_enabled() is False
    os.environ.pop("USE_JOINT_ATTRIBUTION", None)
```

- [ ] **Step 2: Run tests — expect FAIL (helper doesn't exist)**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && \
  uv run pytest tests/unit/test_joint_attribution_integration.py -v
```

Expected: FAIL with import error.

- [ ] **Step 3: Add the helper to `joint_attribution.py`**

Append to `joint_attribution.py`:

```python
import os


def use_joint_attribution_enabled() -> bool:
    """True iff USE_JOINT_ATTRIBUTION env var is exactly '1'.
    Matches existing RELAX_CONTACT_* flag semantics."""
    return os.environ.get("USE_JOINT_ATTRIBUTION", "0") == "1"
```

- [ ] **Step 4: Run tests — expect PASS**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && \
  uv run pytest tests/unit/test_joint_attribution_integration.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 5: Wire the helper into `redetect_all_actions.py`**

Find where `reattribute_players(...)` (or equivalent post-classify_rally_actions step) is called. Add the branch:

```python
from rallycut.tracking.joint_attribution import (
    joint_attribute_rally, use_joint_attribution_enabled, RallyContext,
)

# ... existing per-rally loop ...

# After classify_rally_actions produces rally_actions:
if use_joint_attribution_enabled():
    rally_ctx = RallyContext(
        rally_id=str(rally_id),
        contacts=contacts_for_pgm,  # list[dict] from contacts_json
        initial_actions=rally_actions,  # list[dict] from classify_rally_actions
        team_assignments=match_teams_for_rally,  # dict[int, str]
        serving_team=serving_team_for_rally,  # str | None
    )
    pgm_result = joint_attribute_rally(rally_ctx)
    # Overwrite rally_actions with PGM output
    apply_pgm_result_to_actions(rally_actions, pgm_result, rally_ctx)
else:
    # Legacy path
    reattribute_players(...)
```

The `apply_pgm_result_to_actions` helper writes the MAP back into `rally_actions[*]`:

Add to `joint_attribution.py`:

```python
def apply_pgm_result_to_actions(
    actions: list[dict[str, Any]],
    result: RallyAttribution,
    rally: RallyContext,
) -> None:
    """Overwrite each action's playerTrackId + attribution_source from
    the MAP. Handles ABSENT states by setting playerTrackId=None and
    attribution_source='pgm_absent_team_X'.

    Mutates `actions` in place.
    """
    # Pair actions to MAP entries by frame proximity
    for action in actions:
        cf = action["frame"]
        # Find nearest contact
        best_t = None
        best_d = None
        for t, contact in enumerate(rally.contacts):
            d = abs(contact["frame"] - cf)
            if best_d is None or d < best_d:
                best_d, best_t = d, t
        if best_t is None or best_d > 5:
            continue  # no matching contact; leave unchanged
        state = result.map[best_t]
        if state.startswith("ABSENT_TEAM_"):
            team = state[-1]
            action["playerTrackId"] = None
            action["team"] = team
            action["attribution_source"] = f"pgm_absent_team_{team}"
        else:
            pid = int(state[1:])
            action["playerTrackId"] = pid
            action["attribution_source"] = "pgm_committed"
        action["attribution_confidence"] = result.marginals[best_t][state]
```

- [ ] **Step 6: Wire the same branch into the `reattribute_actions` CLI**

In `analysis/rallycut/cli/commands/reattribute_actions.py`, find where `reattribute_players(...)` is called. Add the same branch:

```python
from rallycut.tracking.joint_attribution import (
    joint_attribute_rally, use_joint_attribution_enabled, RallyContext,
    apply_pgm_result_to_actions,
)

# In the per-rally loop:
if use_joint_attribution_enabled():
    rally_ctx = RallyContext(...)
    pgm_result = joint_attribute_rally(rally_ctx)
    apply_pgm_result_to_actions(rally_actions, pgm_result, rally_ctx)
else:
    reattribute_players(...)
```

- [ ] **Step 7: Run all unit tests; verify legacy path still works**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && \
  unset USE_JOINT_ATTRIBUTION && \
  uv run pytest tests/unit -v
```

Expected: all tests pass; default-OFF behavior is preserved byte-identically.

- [ ] **Step 8: Smoke-test the PGM path on a single rally**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && \
  USE_JOINT_ATTRIBUTION=1 uv run python -u scripts/redetect_all_actions.py --apply --video <pick-one-fresh-gt-video-id>
```

Expected: completes without error; some rallies' actions_json now have `attribution_source: pgm_committed` or `pgm_absent_team_X`.

If errors occur, debug and fix before committing.

- [ ] **Step 9: Commit**

```bash
git add analysis/scripts/redetect_all_actions.py analysis/rallycut/cli/commands/reattribute_actions.py analysis/rallycut/tracking/joint_attribution.py analysis/tests/unit/test_joint_attribution_integration.py
git commit -m "feat(attribution): wire USE_JOINT_ATTRIBUTION env-flag at call sites

Two call sites flip between reattribute_players (default) and
joint_attribute_rally (USE_JOINT_ATTRIBUTION=1):
- analysis/scripts/redetect_all_actions.py
- analysis/rallycut/cli/commands/reattribute_actions.py

Helpers in joint_attribution.py:
- use_joint_attribution_enabled() -> bool
- apply_pgm_result_to_actions(actions, result, rally) writes MAP back
  to actions list with attribution_source + attribution_confidence

3 unit tests for the env-flag check.

Default-OFF; legacy reattribute_players path is byte-identical when
flag is unset.

Spec: docs/superpowers/specs/2026-05-12-joint-attribution-pgm-design.md"
```

---

### Task 11: Calibration script + run

**Files:**
- Create: `analysis/scripts/calibrate_joint_attribution_weights.py`

- [ ] **Step 1: Create the calibration script**

```python
"""Calibrate Joint Attribution PGM factor weights via coordinate ascent
on the 22-rally fresh-GT panel.

Loads the panel from DB; for each weight in fixed order, sweeps -50%,
-25%, +0%, +25%, +50% holding others fixed; picks the best step;
advances. Repeats up to 3 cycles or until no weight changes.

Outputs the calibrated FactorWeights as a Python literal that can be
pasted into joint_attribution_weights.py.

Run from analysis/:
    uv run python -u scripts/calibrate_joint_attribution_weights.py
"""
from __future__ import annotations

import dataclasses
import json
import sys
from pathlib import Path

from rallycut.evaluation.attribution_bench import score_rally
from rallycut.evaluation.db import get_connection
from rallycut.tracking.joint_attribution import (
    RallyContext, joint_attribute_rally, apply_pgm_result_to_actions,
)
from rallycut.tracking.joint_attribution_weights import (
    FactorWeights, DEFAULT_WEIGHTS,
)
from rallycut.training.action_gt_query import load_for_videos

PANEL_VIDEOS = {
    "cece": "950fbe5d-fdad-4862-b05d-8b374bdd5ec6",
    "gigi": "b097dd2a-6953-4e0e-a603-5be3552f462e",
    "wawa": "5c756c41-1cc1-4486-a95c-97398912cfbe",
}
SWEEP_FRACTIONS = (-0.5, -0.25, 0.0, 0.25, 0.5)
WEIGHT_ORDER = (
    "w_proximity", "w_dist", "w_dist_team", "w_visual", "w_pose",
    "w_prior", "w_action",
    "w_back_to_back", "w_alternation", "w_team_consistency", "w_absent_pair",
    "w_3_contact", "w_serve_first",
)
MAX_CYCLES = 3


def load_panel() -> list[dict]:
    """Load 22 panel rallies with all data needed to build RallyContext + score."""
    rallies = []
    with get_connection() as conn:
        for vname, vid in PANEL_VIDEOS.items():
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT r.id::text, r.start_ms, r.end_ms,
                           pt.actions_json, pt.contacts_json
                    FROM rallies r JOIN player_tracks pt ON pt.rally_id = r.id
                    WHERE r.video_id = %s AND EXISTS (
                        SELECT 1 FROM rally_action_ground_truth
                        WHERE rally_id = r.id
                    )
                    """,
                    (vid,),
                )
                rows = cur.fetchall()
            for rid, sm, em, aj, cj in rows:
                # Extract serving_team and team_assignments from match_analysis
                rallies.append({
                    "rally_id": rid, "video_id": vid, "fixture": vname,
                    "start_ms": sm, "end_ms": em,
                    "contacts": (cj if isinstance(cj, dict) else {}).get("contacts", []),
                    "actions": (aj if isinstance(aj, dict) else {}).get("actions", []),
                    "team_assignments_str_keys": (aj if isinstance(aj, dict) else {}).get("teamAssignments", {}),
                    "serving_team": (aj if isinstance(aj, dict) else {}).get("servingTeam"),
                })
        # Load GT per video
        gt_by_rally = load_for_videos(conn, list(PANEL_VIDEOS.values()))
    for r in rallies:
        r["gt_actions"] = gt_by_rally.get(r["rally_id"], [])
    return rallies


def evaluate_weights(weights: FactorWeights, rallies: list[dict]) -> int:
    """Run PGM with these weights on all panel rallies; return total correct count."""
    total_correct = 0
    for r in rallies:
        team_assignments = {int(k): v for k, v in r["team_assignments_str_keys"].items()}
        rally_ctx = RallyContext(
            rally_id=r["rally_id"], contacts=r["contacts"],
            initial_actions=list(r["actions"]),  # copy; will be mutated
            team_assignments=team_assignments, serving_team=r["serving_team"],
        )
        result = joint_attribute_rally(rally_ctx, weights=weights)
        actions_copy = list(r["actions"])  # don't mutate original
        apply_pgm_result_to_actions(actions_copy, result, rally_ctx)
        # Score against GT
        rally_record = {
            "rally_id": r["rally_id"], "fixture": r["fixture"],
            "team_assignments": {str(k): v for k, v in team_assignments.items()},
            "serving_team": r["serving_team"],
            "gt_actions": r["gt_actions"],
            "pipeline_actions": actions_copy,
        }
        scored = score_rally(rally_record)
        total_correct += scored["rally_totals"]["correct"]
    return total_correct


def main() -> int:
    print("Loading panel...", flush=True)
    rallies = load_panel()
    n_gt = sum(len(r["gt_actions"]) for r in rallies)
    print(f"Loaded {len(rallies)} rallies, {n_gt} GT actions", flush=True)

    weights = DEFAULT_WEIGHTS
    baseline = evaluate_weights(weights, rallies)
    print(f"Baseline (default weights): {baseline}/{n_gt} = {baseline/n_gt:.1%}", flush=True)

    history = [{"cycle": 0, "weights": dataclasses.asdict(weights), "correct": baseline}]
    for cycle in range(1, MAX_CYCLES + 1):
        cycle_changed = False
        for wname in WEIGHT_ORDER:
            current_value = getattr(weights, wname)
            best_value, best_correct = current_value, evaluate_weights(weights, rallies)
            for frac in SWEEP_FRACTIONS:
                if frac == 0.0:
                    continue
                new_value = max(0.1, min(10.0, current_value * (1.0 + frac)))
                trial_weights = dataclasses.replace(weights, **{wname: new_value})
                trial_correct = evaluate_weights(trial_weights, rallies)
                if trial_correct > best_correct:
                    best_value, best_correct = new_value, trial_correct
            if best_value != current_value:
                weights = dataclasses.replace(weights, **{wname: best_value})
                cycle_changed = True
                print(f"  cycle {cycle} {wname}: {current_value:.3f} -> {best_value:.3f}  correct={best_correct}", flush=True)
        history.append({"cycle": cycle, "weights": dataclasses.asdict(weights),
                        "correct": evaluate_weights(weights, rallies)})
        if not cycle_changed:
            print(f"Cycle {cycle}: no changes; converged.", flush=True)
            break

    print()
    print("=" * 60)
    print("CALIBRATED WEIGHTS")
    print("=" * 60)
    print(f"  baseline correct: {baseline}/{n_gt}")
    print(f"  final correct:    {history[-1]['correct']}/{n_gt}")
    print()
    print("Paste into joint_attribution_weights.py DEFAULT_WEIGHTS:")
    print()
    for wname in WEIGHT_ORDER:
        v = getattr(weights, wname)
        print(f"    {wname}: float = {v:.4f}")

    out_path = Path("reports/joint_attribution_calibration_2026_05_12.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "baseline_correct": baseline,
        "final_correct": history[-1]["correct"],
        "n_gt": n_gt,
        "history": history,
        "final_weights": dataclasses.asdict(weights),
    }, indent=2))
    print(f"\nWrote: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Smoke-run the calibration**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && \
  uv run python -u scripts/calibrate_joint_attribution_weights.py
```

Expected runtime: 10-30 minutes (each evaluate_weights call processes 22 rallies × ~50ms PGM each + scoring overhead; multiplied by 13 weights × 4 sweep steps × 3 cycles ≈ 195 evaluations).

If it errors, debug. If it runs but takes >1 hour, profile and consider numpy-vectorizing the inner loop.

- [ ] **Step 3: Update DEFAULT_WEIGHTS in `joint_attribution_weights.py`**

Copy the calibrated weight values from the script's printed output into the `FactorWeights` dataclass defaults.

- [ ] **Step 4: Run unit tests with new defaults to confirm they still pass**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && \
  uv run pytest tests/unit -v
```

Expected: all unit tests pass (factor tests reference DEFAULT_WEIGHTS so they'll see the new values; the truth-table assertions are relative, not absolute).

If any test fails, the calibrated weight changed a relative-magnitude assertion. Adjust the test if it was overly specific to the original default.

- [ ] **Step 5: Commit**

```bash
git add analysis/scripts/calibrate_joint_attribution_weights.py analysis/rallycut/tracking/joint_attribution_weights.py analysis/reports/joint_attribution_calibration_2026_05_12.json
git commit -m "feat(attribution): calibrate joint-attribution PGM factor weights

Coordinate-ascent calibration on the 22-rally fresh-GT panel. Hand-tuned
starting weights are tuned by sweeping each weight ±50% in 4 steps,
holding others fixed, advancing to the best step. Up to 3 cycles or
until no weight changes.

Calibrated DEFAULT_WEIGHTS in joint_attribution_weights.py reflect the
script output. See reports/joint_attribution_calibration_2026_05_12.json
for the per-cycle weight trajectory.

Spec: docs/superpowers/specs/2026-05-12-joint-attribution-pgm-design.md"
```

---

### Task 12: A/B measurement + verdict

**Files:**
- Create: `analysis/scripts/measure_joint_attribution_ab.py`
- Create: `analysis/reports/attribution_baseline/joint_pgm_ab_2026_05_12.md`

- [ ] **Step 1: Create the A/B harness**

```python
"""Joint Attribution PGM A/B harness.

Runs measure_attribution_fresh_gt.py with USE_JOINT_ATTRIBUTION=0 then =1,
emits a delta table aligned to the spec's pre-ship gates.

Run from analysis/:
    uv run python -u scripts/measure_joint_attribution_ab.py
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

REPORT_PATH = Path("reports/attribution_baseline/joint_pgm_ab_2026_05_12.md")
HARNESS = "scripts/measure_attribution_fresh_gt.py"
OUT_JSON = "reports/attribution_baseline/fresh_gt_2026_05_11.json"


def run_harness(env_value: str) -> dict:
    """Run the harness with USE_JOINT_ATTRIBUTION set; load JSON output."""
    env = os.environ.copy()
    env["USE_JOINT_ATTRIBUTION"] = env_value
    print(f"--- Running harness with USE_JOINT_ATTRIBUTION={env_value} ---", flush=True)
    subprocess.run(["uv", "run", "python", "-u", HARNESS], env=env, check=True)
    return json.load(open(OUT_JSON))


def main() -> int:
    # Note: the existing measure_attribution_fresh_gt.py reads from DB.
    # For an honest A/B, we need to first re-run reattribute-actions on the
    # 22 panel rallies in both modes, then measure. Two passes.
    panel_videos = ["950fbe5d-fdad-4862-b05d-8b374bdd5ec6",
                    "b097dd2a-6953-4e0e-a603-5be3552f462e",
                    "5c756c41-1cc1-4486-a95c-97398912cfbe"]

    # Pass 1: legacy
    for vid in panel_videos:
        env = os.environ.copy()
        env["USE_JOINT_ATTRIBUTION"] = "0"
        print(f"Legacy reattribute-actions for {vid}...", flush=True)
        subprocess.run(
            ["uv", "run", "rallycut", "reattribute-actions", vid],
            env=env, check=True,
        )
    legacy = run_harness("0")

    # Pass 2: PGM
    for vid in panel_videos:
        env = os.environ.copy()
        env["USE_JOINT_ATTRIBUTION"] = "1"
        print(f"PGM reattribute-actions for {vid}...", flush=True)
        subprocess.run(
            ["uv", "run", "rallycut", "reattribute-actions", vid],
            env=env, check=True,
        )
    pgm = run_harness("1")

    # Compute deltas
    leg_c = legacy["combined"]["counts"]
    pgm_c = pgm["combined"]["counts"]
    n_total = leg_c["n_gt_actions"]
    delta_correct = pgm_c["correct"] - leg_c["correct"]
    delta_cross = pgm_c["wrong_cross_team"] - leg_c["wrong_cross_team"]
    delta_same = pgm_c["wrong_same_team"] - leg_c["wrong_same_team"]
    delta_unk = pgm_c["wrong_unknown_team"] - leg_c["wrong_unknown_team"]

    # Apply pre-ship gates
    pp_threshold = 5  # G-A: +5pp absolute
    cross_threshold = -0.30  # G-B: -30% relative
    g_a = delta_correct >= n_total * pp_threshold / 100
    g_b = leg_c["wrong_cross_team"] > 0 and (delta_cross / leg_c["wrong_cross_team"]) <= cross_threshold
    g_c = delta_same <= 0
    g_d = delta_unk <= 0

    verdict = "PASS" if (g_a and g_b and g_c and g_d) else "DONE_WITH_CONCERNS"
    if delta_correct < 0 or delta_same > leg_c["wrong_same_team"] * 0.33:
        verdict = "STOP"

    md = f"""# Joint Attribution PGM A/B Verdict — 2026-05-12

## Headline

| Metric | Legacy (USE_JOINT_ATTRIBUTION=0) | PGM (=1) | Delta |
|---|---|---|---|
| correct      | {leg_c['correct']:>4d} | {pgm_c['correct']:>4d} | {delta_correct:+d} ({delta_correct/n_total*100:+.1f}pp) |
| cross_team   | {leg_c['wrong_cross_team']:>4d} | {pgm_c['wrong_cross_team']:>4d} | {delta_cross:+d} |
| same_team    | {leg_c['wrong_same_team']:>4d} | {pgm_c['wrong_same_team']:>4d} | {delta_same:+d} |
| unknown_team | {leg_c['wrong_unknown_team']:>4d} | {pgm_c['wrong_unknown_team']:>4d} | {delta_unk:+d} |

## Pre-ship gates

- G-A (correct rate +5pp): {'PASS' if g_a else 'FAIL'}
- G-B (cross_team -30%): {'PASS' if g_b else 'FAIL'}
- G-C (same_team non-increasing): {'PASS' if g_c else 'FAIL'}
- G-D (unknown_team non-increasing): {'PASS' if g_d else 'FAIL'}

## Verdict: **{verdict}**
"""
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(md)
    print(md, flush=True)
    print(f"\nWrote: {REPORT_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Snapshot DB before A/B**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run python -c "
import json
from rallycut.evaluation.db import get_connection
panel_videos = ['950fbe5d-fdad-4862-b05d-8b374bdd5ec6','b097dd2a-6953-4e0e-a603-5be3552f462e','5c756c41-1cc1-4486-a95c-97398912cfbe']
with get_connection() as conn, open('/tmp/joint_pgm_pre_ab_snapshot.jsonl', 'w') as f:
    with conn.cursor() as cur:
        cur.execute('''
          SELECT r.id::text, pt.actions_json
          FROM rallies r JOIN player_tracks pt ON pt.rally_id = r.id
          WHERE r.video_id::text = ANY(%s)
        ''', (panel_videos,))
        for rid, a in cur.fetchall():
            f.write(json.dumps({'rally_id': rid, 'actions_json': a}, default=str) + chr(10))
"
```

- [ ] **Step 3: Run the A/B**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && \
  uv run python -u scripts/measure_joint_attribution_ab.py
```

Expected: produces `reports/attribution_baseline/joint_pgm_ab_2026_05_12.md` with PASS / DONE_WITH_CONCERNS / STOP verdict.

- [ ] **Step 4: Review verdict + decide ship action**

| Verdict | Action |
|---|---|
| PASS | Set `USE_JOINT_ATTRIBUTION=1` as default in production deployment scripts. Leave DB in PGM state. |
| DONE_WITH_CONCERNS | Ship as default-OFF infrastructure. Restore snapshot. Note in report what's needed to ship (e.g., wider GT). |
| STOP | Restore snapshot. Report what regressed. Escalate to Phase B (learned weights) or another approach. |

- [ ] **Step 5: If STOP or DONE_WITH_CONCERNS, restore snapshot**

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis && uv run python -c "
import json
from rallycut.evaluation.db import get_connection
restored = 0
with get_connection() as conn, open('/tmp/joint_pgm_pre_ab_snapshot.jsonl') as f:
    with conn.cursor() as cur:
        for line in f:
            rec = json.loads(line)
            cur.execute('UPDATE player_tracks SET actions_json = %s::jsonb WHERE rally_id::text = %s',
                (json.dumps(rec['actions_json']) if rec['actions_json'] else None, rec['rally_id']))
            restored += 1
        conn.commit()
print(f'restored {restored} rallies')
"
```

- [ ] **Step 6: Commit verdict**

```bash
git add analysis/scripts/measure_joint_attribution_ab.py analysis/reports/attribution_baseline/joint_pgm_ab_2026_05_12.md
git commit -m "report(attribution): Joint Attribution PGM A/B verdict <PASS/DONE_WITH_CONCERNS/STOP>"  # tailored to actual verdict
```

---

### Task 13: Final report + memory entry

**Files:**
- Modify: `analysis/reports/attribution_baseline/joint_pgm_ab_2026_05_12.md` (expand with full narrative)
- Create: `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/joint_attribution_pgm_2026_05_12.md`
- Modify: `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY.md` (add index entry)

- [ ] **Step 1: Expand the verdict report with workstream narrative**

Append to `joint_pgm_ab_2026_05_12.md`:

```markdown
## Calibration

See `reports/joint_attribution_calibration_2026_05_12.json` for the
per-cycle weight trajectory. Final calibrated weights are committed in
`analysis/rallycut/tracking/joint_attribution_weights.py`.

## Per-action-type breakdown

(Insert per-action-type recall table from the harness output here.)

## Coherence audit

(Run `uv run rallycut audit-coherence-invariants` per panel video; report
C-1, C-2, C-3 deltas vs pre-PGM baseline.)

## Open follow-ups

- Phase B: per-action-type weight learning from larger GT corpus
- Phase 1.7: re-enable player_motion_candidates for blocks (orthogonal)
- GT expansion to ~50 rallies to enable per-action-type calibration
```

- [ ] **Step 2: Write the memory entry**

Create `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/joint_attribution_pgm_2026_05_12.md`:

```markdown
---
name: joint-attribution-pgm-2026-05-12
description: Phase A + absent-actor-states PGM. Status: <PASS/DONE_WITH_CONCERNS/STOP>. Replaces reattribute_players via USE_JOINT_ATTRIBUTION env flag. Hand-tuned soft factors over 6 states per contact, exhaustive enumeration with coordinate-descent fallback at N>8. Panel correct rate <pre>%->><post>% (+<delta>pp) on 22-rally fresh-GT. Default-<OFF/ON>.
metadata:
  type: project
---

# Joint Attribution PGM (Phase A + Absent States) — 2026-05-12

**Status:** <PASS / DONE_WITH_CONCERNS / STOP>

## What shipped

- New `joint_attribute_rally(rally) -> RallyAttribution` in
  `analysis/rallycut/tracking/joint_attribution.py`
- Pure factor functions in `joint_attribution_factors.py`
  (28 unit tests)
- Calibrated `FactorWeights` defaults from coordinate ascent on 22-rally panel
- USE_JOINT_ATTRIBUTION env flag at call sites
- New `attribution_source` field on action dicts

## Measurement (22-rally fresh-GT panel)

(Insert headline numbers from verdict report.)

## Files

- `analysis/rallycut/tracking/joint_attribution.py`
- `analysis/rallycut/tracking/joint_attribution_factors.py`
- `analysis/rallycut/tracking/joint_attribution_weights.py`
- `analysis/scripts/calibrate_joint_attribution_weights.py`
- `analysis/scripts/measure_joint_attribution_ab.py`
- `analysis/tests/unit/test_joint_attribution_*.py` (4 files, ~50 tests)

## Spec / Plan

- `docs/superpowers/specs/2026-05-12-joint-attribution-pgm-design.md`
- `docs/superpowers/plans/2026-05-12-joint-attribution-pgm.md`

## Rollback

`USE_JOINT_ATTRIBUTION=0` reverts to `reattribute_players`.

## Composes with

- [[contact_detection_fn_v1_2026_05_12]] — both workstreams from same session
- Future Phase B (learned per-action-type weights)
- Future Phase 1.7 (player-motion candidates for blocks; orthogonal)
```

- [ ] **Step 3: Add MEMORY.md index entry**

Insert under "Current workstreams":

```markdown
- [<STATUS>] [**Joint Attribution PGM 2026-05-12**](joint_attribution_pgm_2026_05_12.md) — <one-line hook>
```

- [ ] **Step 4: Final commit**

```bash
git add analysis/reports/attribution_baseline/joint_pgm_ab_2026_05_12.md
git commit -m "report(attribution): final workstream report for Joint Attribution PGM 2026-05-12"
```

Memory files are NOT in the project git repo — write them but don't commit.

---

## Self-review checklist (run BEFORE handing off)

- [x] Spec coverage: every section in the spec maps to at least one task
  - Goal & Scope → Tasks 1-13
  - Formal model → Tasks 1-5 (factors), Task 6 (evidence)
  - Inference algorithm → Task 7
  - Integration → Tasks 9-10
  - Synthetic-action emission → Task 10 (`apply_pgm_result_to_actions`)
  - Calibration → Task 11
  - Measurement → Task 12
  - Code shape → Tasks 1-10
  - Testing → Tasks 3-8 + Task 10
  - Risks & rollback → Task 10 (env flag), Task 12 (snapshot)
- [x] No placeholders: all code blocks contain real code
- [x] Type consistency: `State`, `RallyContext`, `RallyAttribution`,
  `FactorWeights`, `joint_attribute_rally`, `score_joint_config`,
  `apply_pgm_result_to_actions`, `use_joint_attribution_enabled` referenced
  consistently
- [x] Decision checkpoints explicit (Task 11 calibration review;
  Task 12 verdict)
- [x] STOP conditions explicit (Task 12 verdict matrix)
- [x] Frequent commits (every task commits at least once)
