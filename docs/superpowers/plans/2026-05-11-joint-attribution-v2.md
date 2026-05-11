# Joint Rule-Aware Attribution (v2.0) Implementation Plan

> **STATUS: SUPERSEDED / NO-GO.** Plan executed through Task 7. Hard-rule beam-search proved fundamentally incompatible with the pipeline's upstream noise (servingTeam ~72-76% accurate, action types <100% accurate). Gates G-A, G-B, G-C, G-F all failed; cross_team errors INCREASED. Code removed from codebase 2026-05-11. v3 (local coherence-repair loop) supersedes. Plan retained as historical record.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace per-contact local attribution with a beam-search joint solver over the entire rally that enforces volleyball game rules (R1–R5) as HARD constraints. Targets the 17 `wrong_cross_team` errors on the 3 fresh-GT videos that v1's local predicate can't reach.

**Architecture:** Single new module `analysis/rallycut/tracking/joint_attribution.py` exposing `joint_attribute(actions, contacts, team_assignments, serving_team) -> list[ClassifiedAction]`. Internally: a rally-state machine, a hard-constraint predicate, a soft proximity scorer, and a beam search that prunes rule-violating partial assignments. Layered after the v1 team-chain predicate inside `reattribute_players`, behind env flag `JOINT_ATTRIBUTION_V2`.

**Tech Stack:** Python 3.11, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-11-joint-attribution-v2-design.md`

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `analysis/rallycut/tracking/joint_attribution.py` | Create | `RallyState` dataclass, `_derive_state_after`, `_is_valid_candidate`, `_score_candidate`, `_beam_search`, public `joint_attribute()`. Single file; ≤500 lines projected. |
| `analysis/rallycut/tracking/action_classifier.py` | Modify | Inside `reattribute_players`, call `joint_attribute` behind env flag `JOINT_ATTRIBUTION_V2`, AFTER Pass 1 server-exclusion and the v1 team-chain predicate, BEFORE `_attribute_synthetic_serves`. ~15 lines added. |
| `analysis/tests/unit/test_joint_attribution.py` | Create | Unit tests for state machine, hard constraints, soft scoring, beam search, and one end-to-end integration test on a constructed canonical-bug-pattern rally. |
| `analysis/scripts/measure_attribution_joint_v2_ab.py` | Create | A/B harness: re-runs `reattribute_players` in-memory on the 3 GT videos with `JOINT_ATTRIBUTION_V2` env flag OFF vs ON. Mirrors the existing v1 A/B harness. |
| `analysis/reports/attribution_baseline/joint_v2_2026_05_11.md` | Create | Pre-ship gate verdicts + A/B harness output. |
| `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/joint_attribution_v2_2026_05_11.md` | Create | Post-ship memory entry. |
| `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY.md` | Modify | Add index entry for the v2 ship. |

---

## Task 1: `RallyState` dataclass + `_derive_state_after` (state machine foundation)

**Files:**
- Create: `analysis/rallycut/tracking/joint_attribution.py`
- Create: `analysis/tests/unit/test_joint_attribution.py`

- [ ] **Step 1: Create the new test file**

Create `analysis/tests/unit/test_joint_attribution.py`:

```python
"""Unit tests for the joint rule-aware attribution v2 solver.

Spec: docs/superpowers/specs/2026-05-11-joint-attribution-v2-design.md
"""

from __future__ import annotations

from rallycut.tracking.action_classifier import ActionType, ClassifiedAction
from rallycut.tracking.joint_attribution import (
    RallyState,
    _derive_state_after,
)


def _a(action_type: ActionType, frame: int = 0) -> ClassifiedAction:
    """Minimal ClassifiedAction; player_track_id is unused by the state machine."""
    return ClassifiedAction(
        action_type=action_type,
        frame=frame,
        ball_x=0.5,
        ball_y=0.5,
        velocity=0.02,
        player_track_id=0,
        court_side="near",
        confidence=0.9,
    )


class TestRallyStateMachine:
    """Truth table for _derive_state_after under R1-R5 semantics."""

    def test_initial_state_seeded_by_serving_team_0(self) -> None:
        """A SERVE by team 0 sets serving_team=0, count=1, expected_team=1 for next."""
        prior = RallyState(
            expected_team=None, count_consecutive_same_team=0,
            last_was_block=False, serving_team=0,
        )
        state = _derive_state_after(_a(ActionType.SERVE), team_at_action=0, prior=prior)
        assert state.expected_team == 1   # net crossed; next contact is opposing team
        assert state.count_consecutive_same_team == 1  # opposing team's first contact starts at 1 (reset semantics)
        assert state.last_was_block is False
        assert state.serving_team == 0

    def test_initial_state_serving_team_seeds_when_prior_serving_team_none(self) -> None:
        """If serving_team is None pre-rally, first SERVE seeds it from team_at_action."""
        prior = RallyState(
            expected_team=None, count_consecutive_same_team=0,
            last_was_block=False, serving_team=None,
        )
        state = _derive_state_after(_a(ActionType.SERVE), team_at_action=1, prior=prior)
        assert state.serving_team == 1

    def test_receive_preserves_possession_and_increments_count(self) -> None:
        """RECEIVE keeps same team and increments count."""
        prior = RallyState(
            expected_team=1, count_consecutive_same_team=1,
            last_was_block=False, serving_team=0,
        )
        state = _derive_state_after(_a(ActionType.RECEIVE), team_at_action=1, prior=prior)
        assert state.expected_team == 1
        assert state.count_consecutive_same_team == 2

    def test_attack_flips_possession_and_resets_count(self) -> None:
        """ATTACK is net-crossing; expected_team flips and count resets to 1."""
        prior = RallyState(
            expected_team=1, count_consecutive_same_team=3,
            last_was_block=False, serving_team=0,
        )
        state = _derive_state_after(_a(ActionType.ATTACK), team_at_action=1, prior=prior)
        assert state.expected_team == 0  # ball crossed
        assert state.count_consecutive_same_team == 1
        assert state.last_was_block is False

    def test_block_does_not_flip_possession_and_resets_count(self) -> None:
        """BLOCK is a free pass: expected_team stays at the blocker's team; count resets to 0."""
        prior = RallyState(
            expected_team=1, count_consecutive_same_team=1,
            last_was_block=False, serving_team=0,
        )
        # The block is by the team that was the receiver of the prior attack.
        # Per R5, expected_team is now the blocker's team (cover follows).
        state = _derive_state_after(_a(ActionType.BLOCK), team_at_action=0, prior=prior)
        assert state.expected_team == 0  # cover by the blocking team
        assert state.count_consecutive_same_team == 0  # block itself doesn't count
        assert state.last_was_block is True

    def test_action_after_block_starts_counting_at_1(self) -> None:
        """First contact after a block (the cover) starts count at 1, not 2."""
        prior = RallyState(
            expected_team=0, count_consecutive_same_team=0,
            last_was_block=True, serving_team=0,
        )
        state = _derive_state_after(_a(ActionType.DIG), team_at_action=0, prior=prior)
        assert state.count_consecutive_same_team == 1
        assert state.last_was_block is False

    def test_unknown_action_passes_through(self) -> None:
        """UNKNOWN action: state carries over unchanged."""
        prior = RallyState(
            expected_team=1, count_consecutive_same_team=2,
            last_was_block=False, serving_team=0,
        )
        state = _derive_state_after(_a(ActionType.UNKNOWN), team_at_action=0, prior=prior)
        assert state == prior  # no change
```

- [ ] **Step 2: Run the tests to verify they fail with import error**

Run: `cd analysis && uv run pytest tests/unit/test_joint_attribution.py -v`
Expected: ImportError — `RallyState` and `_derive_state_after` don't exist yet.

- [ ] **Step 3: Implement `RallyState` and `_derive_state_after`**

Create `analysis/rallycut/tracking/joint_attribution.py`:

```python
"""Joint rule-aware action attribution (v2.0).

Spec: docs/superpowers/specs/2026-05-11-joint-attribution-v2-design.md

Replaces per-contact local attribution with a beam-search joint solver
that enforces beach volleyball game rules (R1-R5) as HARD constraints
over the entire rally. The soft scoring reuses existing proximity
ranking from Contact.player_candidates.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

from rallycut.tracking.action_classifier import ActionType, ClassifiedAction
from rallycut.tracking.contact_detector import Contact

logger = logging.getLogger(__name__)

# Action category sets — these are the canonical R2/R3 partitions.
_NET_CROSSING_ACTIONS = {ActionType.SERVE, ActionType.ATTACK}
_SAME_SIDE_ACTIONS = {ActionType.RECEIVE, ActionType.SET, ActionType.DIG}

# Block is its own category (R5). UNKNOWN is passthrough.

# Soft-scoring epsilon for -log(dist + ε) stability.
_SCORE_EPSILON = 1e-3

# R4 hard limit: max consecutive same-team non-block contacts.
_MAX_SAME_TEAM = 3


@dataclass(frozen=True)
class RallyState:
    """State after processing action[i]; constrains action[i+1].

    Attributes
    ----------
    expected_team : int | None
        The team (0=near=A, 1=far=B) that should perform the NEXT action,
        per R2/R3/R5. None pre-rally (before the first SERVE seeds it).
    count_consecutive_same_team : int
        Number of consecutive non-block same-team contacts ending at this
        action. Used for R4. Resets to 1 after a net-crossing (R2 flip)
        and to 0 after a BLOCK (R5).
    last_was_block : bool
        True iff the most recent processed action was a BLOCK. Used by
        R5 to give the next same-team action a free-pass (count starts
        fresh at 1).
    serving_team : int | None
        The team that opened the rally with the serve. Seeded by the
        first SERVE if not pre-set. Constant for the rest of the rally.
    """

    expected_team: int | None
    count_consecutive_same_team: int
    last_was_block: bool
    serving_team: int | None


def _derive_state_after(
    action: ClassifiedAction,
    team_at_action: int,
    prior: RallyState,
) -> RallyState:
    """Compute the rally state AFTER processing this action, given its team.

    Implements the R2/R3/R4/R5 semantics described in the spec. UNKNOWN
    actions pass through (state unchanged). The first SERVE seeds
    serving_team if prior.serving_team is None.

    Parameters
    ----------
    action : ClassifiedAction
        The action being processed. Only `action_type` is read.
    team_at_action : int
        Team assignment of the player performing this action (0 or 1).
    prior : RallyState
        State BEFORE this action.

    Returns
    -------
    RallyState
        State AFTER this action — constrains the next action.
    """
    if action.action_type == ActionType.UNKNOWN:
        return prior

    serving_team = (
        prior.serving_team
        if prior.serving_team is not None
        else (team_at_action if action.action_type == ActionType.SERVE else None)
    )

    if action.action_type == ActionType.BLOCK:
        # R5: block does not flip possession (cover by same team is legal)
        # and does not count toward the 3-contact limit (count resets to 0).
        return RallyState(
            expected_team=team_at_action,
            count_consecutive_same_team=0,
            last_was_block=True,
            serving_team=serving_team,
        )

    if action.action_type in _NET_CROSSING_ACTIONS:
        # R2: net crossing flips possession; opposing team's first contact starts count at 1.
        return RallyState(
            expected_team=1 - team_at_action,
            count_consecutive_same_team=1,
            last_was_block=False,
            serving_team=serving_team,
        )

    # _SAME_SIDE_ACTIONS (RECEIVE, SET, DIG): R3 + R5 cover-reset semantics.
    new_count = 1 if prior.last_was_block else prior.count_consecutive_same_team + 1
    return RallyState(
        expected_team=team_at_action,
        count_consecutive_same_team=new_count,
        last_was_block=False,
        serving_team=serving_team,
    )
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd analysis && uv run pytest tests/unit/test_joint_attribution.py::TestRallyStateMachine -v`
Expected: all 7 tests PASS.

- [ ] **Step 5: Run mypy + ruff**

Run: `cd analysis && uv run mypy rallycut/tracking/joint_attribution.py`
Run: `cd analysis && uv run ruff check rallycut/tracking/joint_attribution.py tests/unit/test_joint_attribution.py`
Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add analysis/rallycut/tracking/joint_attribution.py analysis/tests/unit/test_joint_attribution.py
git commit -m "$(cat <<'EOF'
feat(joint-attribution): RallyState + _derive_state_after foundation

Pure state machine encoding R2 (net-crossing flips possession), R3
(same-side preserves possession), R4 (max-3-same-side counter), and R5
(block-cover free pass). UNKNOWN actions pass through. First SERVE seeds
serving_team if not pre-set.

Spec: docs/superpowers/specs/2026-05-11-joint-attribution-v2-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Hard-constraint predicate `_is_valid_candidate`

**Files:**
- Modify: `analysis/rallycut/tracking/joint_attribution.py` (add `_is_valid_candidate`)
- Modify: `analysis/tests/unit/test_joint_attribution.py` (add `TestIsValidCandidate`)

- [ ] **Step 1: Add the truth-table tests**

Append to `analysis/tests/unit/test_joint_attribution.py`:

```python
from rallycut.tracking.joint_attribution import _is_valid_candidate


class TestIsValidCandidate:
    """Truth table for _is_valid_candidate: per-rule pass/fail cases."""

    def test_first_action_must_be_serve_by_serving_team(self) -> None:
        """R1: with serving_team known, the first action's team must match."""
        prior = RallyState(
            expected_team=None, count_consecutive_same_team=0,
            last_was_block=False, serving_team=0,
        )
        assert _is_valid_candidate(ActionType.SERVE, candidate_team=0, prior=prior) is True
        assert _is_valid_candidate(ActionType.SERVE, candidate_team=1, prior=prior) is False

    def test_first_action_serve_seeds_when_serving_team_none(self) -> None:
        """If serving_team is unset, any team is valid for the first SERVE."""
        prior = RallyState(
            expected_team=None, count_consecutive_same_team=0,
            last_was_block=False, serving_team=None,
        )
        assert _is_valid_candidate(ActionType.SERVE, candidate_team=0, prior=prior) is True
        assert _is_valid_candidate(ActionType.SERVE, candidate_team=1, prior=prior) is True

    def test_first_action_must_be_serve_not_other(self) -> None:
        """First action is SERVE — non-SERVE first actions are rejected."""
        prior = RallyState(
            expected_team=None, count_consecutive_same_team=0,
            last_was_block=False, serving_team=None,
        )
        assert _is_valid_candidate(ActionType.RECEIVE, candidate_team=1, prior=prior) is False
        assert _is_valid_candidate(ActionType.ATTACK, candidate_team=0, prior=prior) is False

    def test_r2_net_crossing_flips_required(self) -> None:
        """After SERVE/ATTACK, next non-BLOCK action must be on opposite team."""
        # State after a serve by team 0 (expected_team=1 set by state machine)
        prior = RallyState(
            expected_team=1, count_consecutive_same_team=1,
            last_was_block=False, serving_team=0,
        )
        # Valid: receive on opposite team
        assert _is_valid_candidate(ActionType.RECEIVE, candidate_team=1, prior=prior) is True
        # Invalid: receive on same team (R2 violation — serving team can't receive own serve)
        assert _is_valid_candidate(ActionType.RECEIVE, candidate_team=0, prior=prior) is False

    def test_r3_same_side_preserves(self) -> None:
        """After RECEIVE/SET/DIG, next non-BLOCK action must be on same team."""
        prior = RallyState(
            expected_team=1, count_consecutive_same_team=1,
            last_was_block=False, serving_team=0,
        )
        # Valid: set on same team
        assert _is_valid_candidate(ActionType.SET, candidate_team=1, prior=prior) is True
        # Invalid: set jumps team
        assert _is_valid_candidate(ActionType.SET, candidate_team=0, prior=prior) is False

    def test_r5_block_passes_through_team_constraint(self) -> None:
        """A BLOCK action can be on either team — the rule constraint is on
        the NEXT action's team (cover by blocker's team).

        However, in beach VB the block is always by the team OPPOSITE the
        attacker. The prior state's expected_team after an ATTACK is the
        receiving team; a BLOCK at this point should be by the receiving
        team. So the constraint actually does apply: block team == expected_team.
        """
        # State after ATTACK by team 0 — expected_team=1 (receiving)
        prior = RallyState(
            expected_team=1, count_consecutive_same_team=1,
            last_was_block=False, serving_team=0,
        )
        # Valid: block by receiving team 1
        assert _is_valid_candidate(ActionType.BLOCK, candidate_team=1, prior=prior) is True
        # Invalid: block by attacking team 0 (the attacker can't block their own attack)
        assert _is_valid_candidate(ActionType.BLOCK, candidate_team=0, prior=prior) is False

    def test_r4_max_3_same_side_blocks_4th_same_team(self) -> None:
        """After 3 same-team contacts, the next non-attack same-team action is rejected.

        In practice the 3rd same-team contact is usually an ATTACK which is net-crossing
        (R2 forces next to flip). R4 catches the abnormal case where the chain has
        3 same-side actions and a 4th would push the count to 4.
        """
        # After 3 consecutive same-team contacts
        prior = RallyState(
            expected_team=1, count_consecutive_same_team=3,
            last_was_block=False, serving_team=0,
        )
        # The 4th same-team action is rejected by R4
        assert _is_valid_candidate(ActionType.SET, candidate_team=1, prior=prior) is False
        # But an attack by same team is still allowed (it crosses; R2 fires next)
        assert _is_valid_candidate(ActionType.ATTACK, candidate_team=1, prior=prior) is True

    def test_r5_after_block_cover_allowed_same_team(self) -> None:
        """After a BLOCK, the cover (same team as blocker) is allowed and
        the count resets — so the cover plus 3 more same-team contacts are legal."""
        prior = RallyState(
            expected_team=0, count_consecutive_same_team=0,
            last_was_block=True, serving_team=0,
        )
        # Cover by same team as blocker (team 0)
        assert _is_valid_candidate(ActionType.DIG, candidate_team=0, prior=prior) is True
        # Cover by opposing team (team 1) — invalid; ball is still on blocker's side
        assert _is_valid_candidate(ActionType.DIG, candidate_team=1, prior=prior) is False

    def test_unknown_always_valid(self) -> None:
        """UNKNOWN actions are passthrough — any team valid (state-machine skip)."""
        prior = RallyState(
            expected_team=1, count_consecutive_same_team=1,
            last_was_block=False, serving_team=0,
        )
        assert _is_valid_candidate(ActionType.UNKNOWN, candidate_team=0, prior=prior) is True
        assert _is_valid_candidate(ActionType.UNKNOWN, candidate_team=1, prior=prior) is True
```

- [ ] **Step 2: Run the tests to verify they fail with import error**

Run: `cd analysis && uv run pytest tests/unit/test_joint_attribution.py::TestIsValidCandidate -v`
Expected: ImportError — `_is_valid_candidate` not defined.

- [ ] **Step 3: Implement `_is_valid_candidate`**

Append to `analysis/rallycut/tracking/joint_attribution.py`:

```python
def _is_valid_candidate(
    action_type: ActionType,
    candidate_team: int,
    prior: RallyState,
) -> bool:
    """Return True iff assigning ``candidate_team`` to this action obeys R1-R5.

    Used by the beam search to prune rule-violating partial assignments.
    """
    # UNKNOWN passes through — no team constraint.
    if action_type == ActionType.UNKNOWN:
        return True

    # R1: first action of a rally must be SERVE.
    if prior.expected_team is None and prior.serving_team is None:
        if action_type != ActionType.SERVE:
            return False
        # Serving team unset → any team valid for the seed.
        return True

    # First action with known serving_team: must be SERVE and team must match.
    if prior.expected_team is None and prior.serving_team is not None:
        if action_type != ActionType.SERVE:
            return False
        return candidate_team == prior.serving_team

    # Non-first action: prior.expected_team is set.
    # R4: max 3 same-team contacts unless the next action is ATTACK (which crosses).
    if (
        prior.count_consecutive_same_team >= _MAX_SAME_TEAM
        and candidate_team == prior.expected_team
        and action_type not in _NET_CROSSING_ACTIONS
        and action_type != ActionType.BLOCK
    ):
        return False

    # R2/R3/R5: candidate_team must match prior.expected_team.
    # (For BLOCK after an ATTACK: prior.expected_team is the receiving team,
    # which is exactly the team that does the block.)
    return candidate_team == prior.expected_team
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd analysis && uv run pytest tests/unit/test_joint_attribution.py -v`
Expected: 7 + 9 = 16 tests PASS.

- [ ] **Step 5: Run mypy + ruff**

Run: `cd analysis && uv run mypy rallycut/tracking/joint_attribution.py`
Run: `cd analysis && uv run ruff check rallycut/tracking/joint_attribution.py tests/unit/test_joint_attribution.py`
Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add analysis/rallycut/tracking/joint_attribution.py analysis/tests/unit/test_joint_attribution.py
git commit -m "$(cat <<'EOF'
feat(joint-attribution): _is_valid_candidate predicate for R1-R5

Pure hard-constraint check. Returns True iff assigning a candidate_team
to an action obeys R1 (first=serve-by-serving-team), R2/R3 (expected_team
match), R4 (max-3-same-side unless crossing), R5 (block-cover free pass).
UNKNOWN actions are passthrough.

Spec: docs/superpowers/specs/2026-05-11-joint-attribution-v2-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Soft scoring `_score_candidate`

**Files:**
- Modify: `analysis/rallycut/tracking/joint_attribution.py` (add `_score_candidate`)
- Modify: `analysis/tests/unit/test_joint_attribution.py` (add `TestScoreCandidate`)

- [ ] **Step 1: Add the soft-scoring tests**

Append to `analysis/tests/unit/test_joint_attribution.py`:

```python
from rallycut.tracking.joint_attribution import _score_candidate


def _contact(
    frame: int = 0,
    candidates: list[tuple[int, float]] | None = None,
    court_side: str = "near",
) -> Contact:
    return Contact(
        frame=frame,
        ball_x=0.5,
        ball_y=0.5,
        velocity=0.02,
        direction_change_deg=60.0,
        player_track_id=(candidates[0][0] if candidates else -1),
        player_distance=(candidates[0][1] if candidates else float("inf")),
        player_candidates=candidates or [],
        court_side=court_side,
        is_validated=True,
    )


class TestScoreCandidate:
    """Truth table for the soft proximity scorer."""

    def test_returns_neg_log_distance_when_pid_in_candidates(self) -> None:
        """Score = -log(dist + ε) for a pid present in player_candidates."""
        contact = _contact(candidates=[(1, 0.05), (2, 0.10)])
        score_1 = _score_candidate(contact, candidate_pid=1)
        score_2 = _score_candidate(contact, candidate_pid=2)
        # Nearer pid scores higher (smaller distance → larger -log)
        assert score_1 > score_2
        # Sanity-check exact values
        assert score_1 == pytest.approx(-math.log(0.05 + 1e-3))
        assert score_2 == pytest.approx(-math.log(0.10 + 1e-3))

    def test_returns_neg_inf_when_pid_not_in_candidates(self) -> None:
        """A pid not in player_candidates → -inf score (effectively rejected)."""
        contact = _contact(candidates=[(1, 0.05), (2, 0.10)])
        assert _score_candidate(contact, candidate_pid=3) == float("-inf")

    def test_handles_zero_distance(self) -> None:
        """Zero distance is well-defined via the epsilon guard."""
        contact = _contact(candidates=[(1, 0.0)])
        score = _score_candidate(contact, candidate_pid=1)
        assert math.isfinite(score)
        assert score == pytest.approx(-math.log(1e-3))

    def test_empty_candidates_returns_neg_inf(self) -> None:
        """A contact with no candidates rejects everything."""
        contact = _contact(candidates=[])
        assert _score_candidate(contact, candidate_pid=1) == float("-inf")
```

Also add three new imports at the top of the test file (where the other imports live):

```python
import math

import pytest

from rallycut.tracking.contact_detector import Contact
```

`pytest` is needed by `pytest.approx` calls in this task's tests. `math` is needed for the `-math.log(...)` assertions. `Contact` is needed by the new `_contact` helper.

- [ ] **Step 2: Run the tests to verify they fail with import error**

Run: `cd analysis && uv run pytest tests/unit/test_joint_attribution.py::TestScoreCandidate -v`
Expected: ImportError — `_score_candidate` not defined.

- [ ] **Step 3: Implement `_score_candidate`**

Append to `analysis/rallycut/tracking/joint_attribution.py`:

```python
def _score_candidate(
    contact: Contact,
    candidate_pid: int,
) -> float:
    """Soft proximity score for assigning ``candidate_pid`` to this contact.

    Returns ``-log(rank_distance + ε)`` for a pid present in
    ``contact.player_candidates`` (the depth-corrected proximity ranking
    populated by ``detect_contacts``). Returns ``-inf`` for pids not in
    the candidates list — effectively rejecting them from the beam.

    Higher scores are better (smaller distance → larger -log).
    """
    for tid, dist in contact.player_candidates:
        if tid == candidate_pid:
            return -math.log(dist + _SCORE_EPSILON)
    return float("-inf")
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd analysis && uv run pytest tests/unit/test_joint_attribution.py -v`
Expected: 16 + 4 = 20 tests PASS.

- [ ] **Step 5: Run mypy + ruff**

Run: `cd analysis && uv run mypy rallycut/tracking/joint_attribution.py`
Run: `cd analysis && uv run ruff check rallycut/tracking/joint_attribution.py tests/unit/test_joint_attribution.py`
Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add analysis/rallycut/tracking/joint_attribution.py analysis/tests/unit/test_joint_attribution.py
git commit -m "$(cat <<'EOF'
feat(joint-attribution): _score_candidate proximity scorer

Soft scoring used by the beam search to rank rule-valid assignments.
Returns -log(rank_distance + ε) when the candidate_pid is in
Contact.player_candidates; -inf otherwise. Higher scores are better.

Spec: docs/superpowers/specs/2026-05-11-joint-attribution-v2-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Beam search `_beam_search`

**Files:**
- Modify: `analysis/rallycut/tracking/joint_attribution.py` (add `_beam_search`)
- Modify: `analysis/tests/unit/test_joint_attribution.py` (add `TestBeamSearch`)

- [ ] **Step 1: Add the beam-search tests**

Append to `analysis/tests/unit/test_joint_attribution.py`:

```python
from rallycut.tracking.joint_attribution import _beam_search


class TestBeamSearch:
    """End-to-end tests of the beam search returning a per-action pid list."""

    def test_canonical_bug_pattern_returns_rule_valid_assignment(self) -> None:
        """The user-quoted bug: serve by team B, receive currently attributed to team B.
        Beam search should pick the team-A receiver (which is in the candidate list)
        because that's the only R2-compliant assignment."""
        actions = [
            _a(ActionType.SERVE, frame=50),
            _a(ActionType.RECEIVE, frame=90),
        ]
        contacts = [
            _contact(frame=50, candidates=[(3, 0.04)]),
            # The wrong-team nearest candidate (4) is first; the correct candidate (1) is second.
            _contact(frame=90, candidates=[(4, 0.05), (1, 0.07)]),
        ]
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}  # teams 0=near=A, 1=far=B

        assignment = _beam_search(
            actions, contacts, team_assignments,
            serving_team=1,  # team B serves
            beam_width=50,
        )
        # Serve by track 3 (team B); receive by track 1 (team A, the correct team).
        assert assignment == [3, 1]

    def test_returns_none_when_no_valid_assignment_exists(self) -> None:
        """If every candidate path violates a rule, return None for the fallback."""
        actions = [
            _a(ActionType.SERVE, frame=50),
            _a(ActionType.RECEIVE, frame=90),
        ]
        contacts = [
            _contact(frame=50, candidates=[(3, 0.04)]),
            # Only team-B candidates exist for the receive — violates R2.
            _contact(frame=90, candidates=[(3, 0.05), (4, 0.07)]),
        ]
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}

        assignment = _beam_search(
            actions, contacts, team_assignments,
            serving_team=1, beam_width=50,
        )
        assert assignment is None  # fallback signal

    def test_picks_proximity_best_within_rule_valid_space(self) -> None:
        """Among rule-valid assignments, the highest soft-score (closest) wins."""
        actions = [
            _a(ActionType.SERVE, frame=50),
            _a(ActionType.RECEIVE, frame=90),
        ]
        contacts = [
            _contact(frame=50, candidates=[(3, 0.04), (4, 0.20)]),  # both team B; 3 is closer
            _contact(frame=90, candidates=[(1, 0.05), (2, 0.10)]),  # both team A; 1 is closer
        ]
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}

        assignment = _beam_search(
            actions, contacts, team_assignments,
            serving_team=1, beam_width=50,
        )
        # Both serve candidates are team B (rule-valid); 3 is closer.
        # Both receive candidates are team A (rule-valid); 1 is closer.
        assert assignment == [3, 1]

    def test_unknown_action_passes_through_any_pid(self) -> None:
        """UNKNOWN actions accept any candidate; beam search picks the proximity-best."""
        actions = [
            _a(ActionType.SERVE, frame=50),
            _a(ActionType.UNKNOWN, frame=70),
            _a(ActionType.RECEIVE, frame=90),
        ]
        contacts = [
            _contact(frame=50, candidates=[(3, 0.04)]),
            _contact(frame=70, candidates=[(2, 0.05), (3, 0.08)]),  # UNKNOWN: either valid
            _contact(frame=90, candidates=[(1, 0.05), (4, 0.07)]),  # receive: must be team A
        ]
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}

        assignment = _beam_search(
            actions, contacts, team_assignments,
            serving_team=1, beam_width=50,
        )
        # UNKNOWN: 2 is closer → picked.
        # Receive: 1 (team A) is valid; 4 (team B) violates R2.
        assert assignment == [3, 2, 1]

    def test_block_and_cover_legal_sequence(self) -> None:
        """A block followed by a same-team cover is legal under R5.

        Sequence: SERVE(B) → RECEIVE(A) → SET(A) → ATTACK(A) → BLOCK(B) → DIG(B) → SET(B) → ATTACK(B)
        """
        actions = [
            _a(ActionType.SERVE, frame=50),
            _a(ActionType.RECEIVE, frame=90),
            _a(ActionType.SET, frame=130),
            _a(ActionType.ATTACK, frame=170),
            _a(ActionType.BLOCK, frame=180),
            _a(ActionType.DIG, frame=200),
            _a(ActionType.SET, frame=240),
            _a(ActionType.ATTACK, frame=280),
        ]
        contacts = [
            _contact(frame=50, candidates=[(3, 0.04)]),
            _contact(frame=90, candidates=[(1, 0.05)]),
            _contact(frame=130, candidates=[(2, 0.06)]),
            _contact(frame=170, candidates=[(1, 0.07)]),
            _contact(frame=180, candidates=[(3, 0.05)]),  # block by team B
            _contact(frame=200, candidates=[(4, 0.06)]),  # cover by team B (same as blocker)
            _contact(frame=240, candidates=[(3, 0.05)]),  # set by team B
            _contact(frame=280, candidates=[(4, 0.07)]),  # attack by team B
        ]
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}

        assignment = _beam_search(
            actions, contacts, team_assignments,
            serving_team=1, beam_width=50,
        )
        assert assignment == [3, 1, 2, 1, 3, 4, 3, 4]
```

- [ ] **Step 2: Run the tests to verify they fail with import error**

Run: `cd analysis && uv run pytest tests/unit/test_joint_attribution.py::TestBeamSearch -v`
Expected: ImportError — `_beam_search` not defined.

- [ ] **Step 3: Implement `_beam_search`**

Append to `analysis/rallycut/tracking/joint_attribution.py`:

```python
def _beam_search(
    actions: list[ClassifiedAction],
    contacts: list[Contact],
    team_assignments: dict[int, int],
    serving_team: int | None,
    beam_width: int = 50,
) -> list[int] | None:
    """Beam search over per-action player assignments under R1-R5.

    Walks the action sequence left-to-right. At each step, expands each
    surviving partial assignment by every pid in the corresponding
    contact's player_candidates, filters by ``_is_valid_candidate``, scores
    by ``_score_candidate``, and retains the top ``beam_width`` partials.

    Returns the highest-scoring full assignment (list of pids parallel to
    actions) or ``None`` if no rule-valid full assignment exists.

    Notes
    -----
    - The contact for ``actions[i]`` is ``contacts[i]``. Lengths must match.
    - Rallies with fewer contacts than actions get -inf score and fall back.
    - UNKNOWN actions accept any pid in their contact's candidates (no
      team constraint), but still receive a proximity score.
    """
    if len(actions) != len(contacts):
        logger.warning(
            "joint_attribute: action/contact length mismatch (%d vs %d), "
            "falling back",
            len(actions), len(contacts),
        )
        return None
    if not actions:
        return []

    # Initial beam: one entry per valid candidate for actions[0].
    initial_state = RallyState(
        expected_team=None,
        count_consecutive_same_team=0,
        last_was_block=False,
        serving_team=serving_team,
    )

    # Each beam entry is (cumulative_score, assignment_so_far, state_after).
    beam: list[tuple[float, list[int], RallyState]] = []
    for tid, _dist in contacts[0].player_candidates:
        tid_team = team_assignments.get(tid)
        if tid_team is None:
            continue
        if not _is_valid_candidate(actions[0].action_type, tid_team, initial_state):
            continue
        score = _score_candidate(contacts[0], tid)
        if not math.isfinite(score):
            continue
        new_state = _derive_state_after(actions[0], tid_team, initial_state)
        beam.append((score, [tid], new_state))

    if not beam:
        return None

    # Expand for actions[1:].
    for i in range(1, len(actions)):
        next_beam: list[tuple[float, list[int], RallyState]] = []
        action_i = actions[i]
        contact_i = contacts[i]
        for cum_score, partial, state in beam:
            for tid, _dist in contact_i.player_candidates:
                tid_team = team_assignments.get(tid)
                if tid_team is None:
                    continue
                if not _is_valid_candidate(action_i.action_type, tid_team, state):
                    continue
                inc_score = _score_candidate(contact_i, tid)
                if not math.isfinite(inc_score):
                    continue
                new_score = cum_score + inc_score
                new_partial = partial + [tid]
                new_state = _derive_state_after(action_i, tid_team, state)
                next_beam.append((new_score, new_partial, new_state))
        if not next_beam:
            return None
        # Prune to beam_width by descending score.
        next_beam.sort(key=lambda x: x[0], reverse=True)
        beam = next_beam[:beam_width]

    # Return the highest-scoring full assignment.
    beam.sort(key=lambda x: x[0], reverse=True)
    return beam[0][1]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd analysis && uv run pytest tests/unit/test_joint_attribution.py::TestBeamSearch -v`
Expected: 5 tests PASS.

- [ ] **Step 5: Run the full file's tests as a regression check**

Run: `cd analysis && uv run pytest tests/unit/test_joint_attribution.py -v`
Expected: 20 + 5 = 25 tests PASS.

- [ ] **Step 6: Run mypy + ruff**

Run: `cd analysis && uv run mypy rallycut/tracking/joint_attribution.py`
Run: `cd analysis && uv run ruff check rallycut/tracking/joint_attribution.py tests/unit/test_joint_attribution.py`
Expected: both clean.

- [ ] **Step 7: Commit**

```bash
git add analysis/rallycut/tracking/joint_attribution.py analysis/tests/unit/test_joint_attribution.py
git commit -m "$(cat <<'EOF'
feat(joint-attribution): _beam_search over per-action pid assignments

Left-to-right beam search. Expands each partial by every pid in the
current contact's player_candidates, prunes rule-violators via
_is_valid_candidate, scores via _score_candidate, retains top beam_width.
Returns the highest-scoring full assignment, or None when the beam empties
(no rule-valid path) — the fallback signal for joint_attribute.

Spec: docs/superpowers/specs/2026-05-11-joint-attribution-v2-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Public entry `joint_attribute`

**Files:**
- Modify: `analysis/rallycut/tracking/joint_attribution.py` (add `joint_attribute`)
- Modify: `analysis/tests/unit/test_joint_attribution.py` (add `TestJointAttribute`)

- [ ] **Step 1: Add the public-entry tests**

Append to `analysis/tests/unit/test_joint_attribution.py`:

```python
from rallycut.tracking.joint_attribution import joint_attribute


class TestJointAttribute:
    """End-to-end public entry: rewrites action.player_track_id from the beam result.
    Falls back to the input unchanged if no valid assignment exists."""

    def test_overrides_cross_team_receive_on_canonical_bug(self) -> None:
        """The user-quoted bug: receive currently attributed to team B; v2
        rewrites it to the team-A candidate that satisfies R2."""
        actions = [
            ClassifiedAction(
                action_type=ActionType.SERVE,
                frame=50, ball_x=0.5, ball_y=0.5, velocity=0.02,
                player_track_id=3, court_side="far", confidence=0.95,
            ),
            ClassifiedAction(
                action_type=ActionType.RECEIVE,
                frame=90, ball_x=0.5, ball_y=0.5, velocity=0.02,
                player_track_id=4, court_side="near", confidence=0.9,
            ),
        ]
        contacts = [
            _contact(frame=50, candidates=[(3, 0.04)]),
            _contact(frame=90, candidates=[(4, 0.05), (1, 0.07)]),
        ]
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}

        result = joint_attribute(actions, contacts, team_assignments, serving_team=1)
        assert result[0].player_track_id == 3
        assert result[1].player_track_id == 1  # was 4, now 1

    def test_fallback_preserves_input_when_no_valid_assignment(self) -> None:
        """When the beam empties, return the input unchanged."""
        actions = [
            ClassifiedAction(
                action_type=ActionType.SERVE,
                frame=50, ball_x=0.5, ball_y=0.5, velocity=0.02,
                player_track_id=3, court_side="far", confidence=0.95,
            ),
            ClassifiedAction(
                action_type=ActionType.RECEIVE,
                frame=90, ball_x=0.5, ball_y=0.5, velocity=0.02,
                player_track_id=4, court_side="near", confidence=0.9,
            ),
        ]
        contacts = [
            _contact(frame=50, candidates=[(3, 0.04)]),
            # Both candidates on serving team — no rule-valid receive exists.
            _contact(frame=90, candidates=[(3, 0.05), (4, 0.07)]),
        ]
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}

        result = joint_attribute(actions, contacts, team_assignments, serving_team=1)
        # Input preserved
        assert result[0].player_track_id == 3
        assert result[1].player_track_id == 4

    def test_empty_actions_returns_empty(self) -> None:
        """Empty input returns empty list (no crash)."""
        result = joint_attribute([], [], {}, serving_team=None)
        assert result == []

    def test_serving_team_none_seeds_from_first_serve(self) -> None:
        """If serving_team is None, the first SERVE seeds it; valid assignment found."""
        actions = [
            ClassifiedAction(
                action_type=ActionType.SERVE,
                frame=50, ball_x=0.5, ball_y=0.5, velocity=0.02,
                player_track_id=3, court_side="far", confidence=0.95,
            ),
            ClassifiedAction(
                action_type=ActionType.RECEIVE,
                frame=90, ball_x=0.5, ball_y=0.5, velocity=0.02,
                player_track_id=4, court_side="near", confidence=0.9,
            ),
        ]
        contacts = [
            _contact(frame=50, candidates=[(3, 0.04)]),
            _contact(frame=90, candidates=[(4, 0.05), (1, 0.07)]),
        ]
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}

        result = joint_attribute(actions, contacts, team_assignments, serving_team=None)
        assert result[0].player_track_id == 3
        assert result[1].player_track_id == 1  # team A; R2 satisfied even without pre-set serving_team
```

- [ ] **Step 2: Run the tests to verify they fail with import error**

Run: `cd analysis && uv run pytest tests/unit/test_joint_attribution.py::TestJointAttribute -v`
Expected: ImportError — `joint_attribute` not defined.

- [ ] **Step 3: Implement `joint_attribute`**

Append to `analysis/rallycut/tracking/joint_attribution.py`:

```python
def joint_attribute(
    actions: list[ClassifiedAction],
    contacts: list[Contact],
    team_assignments: dict[int, int],
    serving_team: int | None,
    beam_width: int = 50,
) -> list[ClassifiedAction]:
    """Joint rule-aware action attribution over a single rally.

    Reuses existing proximity ranking (Contact.player_candidates) as the
    soft signal. Enforces R1-R5 as hard constraints via beam search.
    Mutates each ClassifiedAction in place to rewrite ``player_track_id``
    per the beam-search-best assignment. Returns the same list (for
    chainable use).

    If no rule-valid assignment exists, returns ``actions`` UNCHANGED and
    emits a single WARN log line. Never silently corrupts attribution.

    Parameters
    ----------
    actions, contacts
        Parallel lists. ``contacts[i]`` corresponds to ``actions[i]``.
    team_assignments
        Map from player track id to team (0=near=A, 1=far=B).
    serving_team
        The team that opened the rally (0 or 1). If None, the first
        SERVE's chosen pid seeds it.
    beam_width
        Number of partial assignments retained per step. Default 50.
    """
    if not actions:
        return actions

    assignment = _beam_search(
        actions, contacts, team_assignments,
        serving_team=serving_team, beam_width=beam_width,
    )
    if assignment is None:
        logger.warning(
            "joint_attribute fallback: no rule-valid assignment for "
            "rally with %d actions; preserving input attributions",
            len(actions),
        )
        return actions

    for i, action in enumerate(actions):
        action.player_track_id = assignment[i]
    return actions
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd analysis && uv run pytest tests/unit/test_joint_attribution.py -v`
Expected: 25 + 4 = 29 tests PASS.

- [ ] **Step 5: Run mypy + ruff**

Run: `cd analysis && uv run mypy rallycut/tracking/joint_attribution.py`
Run: `cd analysis && uv run ruff check rallycut/tracking/joint_attribution.py tests/unit/test_joint_attribution.py`
Expected: both clean.

- [ ] **Step 6: Commit**

```bash
git add analysis/rallycut/tracking/joint_attribution.py analysis/tests/unit/test_joint_attribution.py
git commit -m "$(cat <<'EOF'
feat(joint-attribution): public joint_attribute entry with safe fallback

Public entry function. Calls _beam_search; on success rewrites
player_track_id per the best assignment. On beam-empty failure, returns
actions unchanged and emits a single WARN — the "never silently corrupt"
guarantee.

Spec: docs/superpowers/specs/2026-05-11-joint-attribution-v2-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Wire `joint_attribute` into `reattribute_players` behind env flag

**Files:**
- Modify: `analysis/rallycut/tracking/action_classifier.py` (call joint_attribute inside reattribute_players)

- [ ] **Step 1: Locate the insertion point**

Run: `cd analysis && grep -n "_attribute_synthetic_serves\|Pass 3 \(ReID\)\|_reattribute_reid\|return actions$" rallycut/tracking/action_classifier.py`

Find the line at the end of `reattribute_players` that returns `actions`. The new call goes IMMEDIATELY BEFORE Pass 3 (ReID re-attribution), AFTER the team-chain v1 work and the `correct_team_from_propagation` adjustments. Search the file for `# Pass 3: ReID re-attribution` to find the boundary. If no such comment, find the call to `_reattribute_reid` — the new v2 call goes immediately before it.

- [ ] **Step 2: Add the import**

Open `analysis/rallycut/tracking/action_classifier.py` and add the import alongside the other tracking imports (top of file, alphabetical):

```python
from rallycut.tracking.joint_attribution import joint_attribute
```

- [ ] **Step 3: Add the env-flag-gated call**

Inside `reattribute_players`, immediately before the Pass 3 ReID call (line varies after Tasks 1-5; locate via the search in Step 1), insert:

```python
    # Pass 2b (v2.0, 2026-05-11): joint rule-aware attribution.
    # Default-OFF; enable via JOINT_ATTRIBUTION_V2=1.
    # Layered after v1 team-chain predicate so the v1 fixes feed v2's
    # starting state — they compose. Spec:
    # docs/superpowers/specs/2026-05-11-joint-attribution-v2-design.md
    if os.environ.get("JOINT_ATTRIBUTION_V2", "0") == "1" and team_assignments:
        # serving_team is the int form: 0=near=A, 1=far=B. The CLI sees the
        # string form ("A"/"B"/None). When called from production with team_assignments
        # already populated, we don't have direct access to servingTeam here —
        # joint_attribute handles serving_team=None by seeding from the first SERVE.
        joint_attribute(
            actions, contacts, team_assignments, serving_team=None,
        )
```

Note: this call does NOT pass `serving_team` because the existing `reattribute_players` signature doesn't accept it. The function handles `serving_team=None` by seeding from the first SERVE's chosen pid, which is the right behavior here (any well-attributed serve already has the right team).

- [ ] **Step 4: Run the existing action-classifier suite to verify no regression**

Run: `cd analysis && uv run pytest tests/unit/test_action_classifier.py -v`
Expected: all 163 existing tests PASS.

- [ ] **Step 5: Run the new joint-attribution suite to confirm import chain is clean**

Run: `cd analysis && uv run pytest tests/unit/test_joint_attribution.py -v`
Expected: 29 tests PASS.

- [ ] **Step 6: Run mypy + ruff on the modified file**

Run: `cd analysis && uv run mypy rallycut/tracking/action_classifier.py`
Run: `cd analysis && uv run ruff check rallycut/tracking/action_classifier.py`
Expected: both clean.

- [ ] **Step 7: Commit**

```bash
git add analysis/rallycut/tracking/action_classifier.py
git commit -m "$(cat <<'EOF'
feat(joint-attribution): wire joint_attribute into reattribute_players

New Pass 2b in reattribute_players, gated by env flag JOINT_ATTRIBUTION_V2
(default OFF). Calls joint_attribute() with the existing team_assignments
and the actions/contacts list, AFTER the v1 team-chain predicate and the
correct_team_from_propagation pass, but BEFORE Pass 3 ReID. serving_team
is left as None — joint_attribute seeds from the first SERVE's chosen pid.

Spec: docs/superpowers/specs/2026-05-11-joint-attribution-v2-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: A/B harness + measurement on 3 GT videos

**Files:**
- Create: `analysis/scripts/measure_attribution_joint_v2_ab.py`
- Create: `analysis/reports/attribution_baseline/joint_v2_2026_05_11.md`

- [ ] **Step 1: Create the A/B harness script**

Create `analysis/scripts/measure_attribution_joint_v2_ab.py`:

```python
"""A/B harness: re-run reattribute_players in memory on the 3 GT videos
with JOINT_ATTRIBUTION_V2 OFF vs ON. Mirrors the v1 A/B harness pattern.

Run from analysis/:
    uv run python scripts/measure_attribution_joint_v2_ab.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from rallycut.evaluation.attribution_bench import (
    WRONG_CATEGORIES,
    aggregate,
    score_rally,
)
from rallycut.evaluation.db import get_connection
from rallycut.tracking.action_classifier import (
    ActionType,
    ClassifiedAction,
    reattribute_players,
)
from rallycut.tracking.contact_detector import Contact

FRESH_GT_VIDEOS = {
    "cece": "950fbe5d-fdad-4862-b05d-8b374bdd5ec6",
    "gigi": "b097dd2a-6953-4e0e-a603-5be3552f462e",
    "wawa": "5c756c41-1cc1-4486-a95c-97398912cfbe",
}

OUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "reports" / "attribution_baseline" / "joint_v2_ab_2026_05_11.json"
)


def _build_rally_remap(match_analysis: dict | None, rally_id: str) -> dict[int, int]:
    if not match_analysis:
        return {}
    for r in match_analysis.get("rallies") or []:
        rid = r.get("rallyId") or r.get("rally_id")
        if rid != rally_id:
            continue
        afm = r.get("appliedFullMapping") or r.get("trackToPlayer") or {}
        out: dict[int, int] = {}
        for k, v in afm.items():
            try:
                out[int(k)] = int(v)
            except (TypeError, ValueError):
                continue
        return out
    return {}


def _reconstruct_contacts(contacts_json: dict[str, Any]) -> list[Contact]:
    contacts: list[Contact] = []
    for c in (contacts_json or {}).get("contacts", []):
        candidates = [
            (int(p[0]), float(p[1])) for p in c.get("playerCandidates", [])
            if p[1] is not None
        ]
        contacts.append(Contact(
            frame=c.get("frame", 0),
            ball_x=c.get("ballX", 0.0),
            ball_y=c.get("ballY", 0.0),
            velocity=c.get("velocity", 0.0),
            direction_change_deg=c.get("directionChangeDeg", 0.0),
            player_track_id=c.get("playerTrackId", -1),
            player_distance=(
                float("inf")
                if c.get("playerDistance") is None
                else c["playerDistance"]
            ),
            player_candidates=candidates,
            court_side=c.get("courtSide", "unknown"),
            is_at_net=c.get("isAtNet", False),
            is_validated=c.get("isValidated", False),
            confidence=c.get("confidence", 0.0),
            arc_fit_residual=c.get("arcFitResidual", 0.0),
        ))
    return contacts


def _reconstruct_actions(actions_json: dict[str, Any]) -> list[ClassifiedAction]:
    raw = (actions_json or {}).get("actions", [])
    if isinstance(raw, dict):
        raw = raw.get("actions", [])
    out: list[ClassifiedAction] = []
    for a in raw:
        if not isinstance(a, dict):
            continue
        try:
            t = ActionType(a["action"])
        except (KeyError, ValueError):
            t = ActionType.UNKNOWN
        out.append(ClassifiedAction(
            action_type=t,
            frame=a.get("frame", 0),
            ball_x=a.get("ballX", 0.0),
            ball_y=a.get("ballY", 0.0),
            velocity=a.get("velocity", 0.0),
            player_track_id=a.get("playerTrackId", -1),
            court_side=a.get("courtSide", "unknown"),
            confidence=a.get("confidence", 0.0),
            is_synthetic=a.get("isSynthetic", False),
            team=a.get("team", "unknown"),
        ))
    return out


def _score(rallies_data, joint_v2_on: bool) -> dict[str, Any]:
    os.environ["JOINT_ATTRIBUTION_V2"] = "1" if joint_v2_on else "0"
    scored_rallies: list[dict[str, Any]] = []
    for r in rallies_data:
        actions = _reconstruct_actions(r["actions_json"])
        contacts = _reconstruct_contacts(r["contacts_json"])
        team_assignments_raw = (r["actions_json"] or {}).get("teamAssignments", {})
        team_assignments = {
            int(k): (0 if v == "A" else 1)
            for k, v in team_assignments_raw.items()
        }
        if team_assignments and actions:
            reattribute_players(actions, contacts, team_assignments)
        pipeline_actions = [a.to_dict() for a in actions]
        normalised_gt = []
        for ga in r["gt"]:
            raw_tid = ga.get("trackId", ga.get("playerTrackId"))
            try:
                raw_int = int(raw_tid) if raw_tid is not None else None
            except (TypeError, ValueError):
                raw_int = None
            canonical = r["remap"].get(raw_int) if raw_int is not None else None
            if canonical is None and raw_int is not None and 1 <= raw_int <= 4:
                canonical = raw_int
            entry = dict(ga)
            entry["playerTrackId"] = canonical
            normalised_gt.append(entry)
        rally_record = {
            "rally_id": r["rally_id"],
            "fixture": r["fixture"],
            "team_assignments": team_assignments_raw,
            "gt_actions": normalised_gt,
            "pipeline_actions": pipeline_actions,
        }
        scored = score_rally(rally_record)
        rally_record["matches"] = scored["matches"]
        rally_record["rally_totals"] = scored["rally_totals"]
        scored_rallies.append(rally_record)
    agg = aggregate(scored_rallies)
    return {"agg": agg, "rallies": scored_rallies}


def _summary(label: str, agg: dict[str, Any]) -> dict[str, Any]:
    c = agg["combined"]["counts"]
    r = agg["combined"]["rates"]
    wrong = sum(c[k] for k in WRONG_CATEGORIES)
    print(f"\n=== {label} ===")
    print(f"  correct: {c['correct']:>3d}  ({r['correct_rate']:>6.1%})")
    print(f"  wrong:   {wrong:>3d}  ({r['wrong_rate']:>6.1%})  "
          f"[cross={c['wrong_cross_team']} same={c['wrong_same_team']} "
          f"unk={c['wrong_unknown_team']}]")
    print(f"  missing: {c['missing']:>3d}  ({r['missing_rate']:>6.1%})")
    return {"counts": c, "rates": r, "wrong": wrong}


def main() -> int:
    rallies_data: list[dict[str, Any]] = []
    with get_connection() as conn, conn.cursor() as cur:
        match_analyses: dict[str, dict] = {}
        for fixture, vid in FRESH_GT_VIDEOS.items():
            cur.execute(
                "SELECT match_analysis_json FROM videos WHERE id = %s", (vid,),
            )
            ma = cur.fetchone()
            match_analyses[vid] = ma[0] if ma and ma[0] else {}
        for fixture, vid in FRESH_GT_VIDEOS.items():
            cur.execute(
                """SELECT r.id, pt.action_ground_truth_json,
                          pt.actions_json, pt.contacts_json
                   FROM rallies r JOIN player_tracks pt ON pt.rally_id = r.id
                   WHERE r.video_id = %s
                     AND pt.action_ground_truth_json IS NOT NULL
                     AND jsonb_array_length(pt.action_ground_truth_json::jsonb) > 0
                   ORDER BY r.start_ms""",
                (vid,),
            )
            for rid, gt, aj, cj in cur.fetchall():
                rallies_data.append({
                    "rally_id": str(rid),
                    "fixture": fixture,
                    "gt": gt or [],
                    "actions_json": aj,
                    "contacts_json": cj,
                    "remap": _build_rally_remap(match_analyses[vid], str(rid)),
                })
    print(f"Loaded {len(rallies_data)} rallies across {len(FRESH_GT_VIDEOS)} videos",
          flush=True)

    off = _score(rallies_data, joint_v2_on=False)
    on = _score(rallies_data, joint_v2_on=True)

    s_off = _summary("OFF (v2 disabled)", off["agg"])
    s_on  = _summary("ON  (v2 enabled)",   on["agg"])

    print("\n=== DELTA (on - off) ===")
    print(f"  correct:           {s_on['counts']['correct'] - s_off['counts']['correct']:+d} "
          f"({s_on['rates']['correct_rate'] - s_off['rates']['correct_rate']:+.1%})")
    print(f"  wrong_cross_team: {s_on['counts']['wrong_cross_team'] - s_off['counts']['wrong_cross_team']:+d}")
    print(f"  wrong_same_team:  {s_on['counts']['wrong_same_team']  - s_off['counts']['wrong_same_team']:+d}")
    print(f"  wrong_unknown:    {s_on['counts']['wrong_unknown_team'] - s_off['counts']['wrong_unknown_team']:+d}")

    print("\n=== PER-FIXTURE DELTA ===")
    for fx in sorted(off["agg"]["per_fixture"].keys()):
        off_c = off["agg"]["per_fixture"][fx]["counts"]["correct"]
        on_c  = on["agg"]["per_fixture"][fx]["counts"]["correct"]
        print(f"  {fx:<6} correct: {off_c} → {on_c} ({on_c - off_c:+d})")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps({"off": off["agg"], "on": on["agg"]}, indent=2, default=str))
    print(f"\nWrote {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Create the report file with baseline reference**

```bash
cat > analysis/reports/attribution_baseline/joint_v2_2026_05_11.md <<'EOF'
# Joint Attribution v2.0 — A/B Measurement Report (2026-05-11)

Spec: docs/superpowers/specs/2026-05-11-joint-attribution-v2-design.md
Plan: docs/superpowers/plans/2026-05-11-joint-attribution-v2.md

## Reference: v1 baseline (post-deploy DB read, 2026-05-11)

n=136 GT actions across 22 rallies (cece + gigi + wawa):
- correct: 82 (60.3%)
- wrong: 26 (19.1%) [cross=17, same=9, unk=0]
- missing: 28 (20.6%)
- Per-action matched accuracy: serve 47%, set 67%, dig 82%, attack 83%, receive 94%

## A/B harness output

(Filled in below from `scripts/measure_attribution_joint_v2_ab.py` stdout.)

EOF
```

- [ ] **Step 3: Run the A/B harness**

Run from `analysis/`: `uv run python -u scripts/measure_attribution_joint_v2_ab.py`
Expected: counts for OFF and ON, plus deltas. Save full stdout to the report file's "## A/B harness output" section.

- [ ] **Step 4: Verify the 6 pre-ship gates**

Append a gate checklist to the report file:

```markdown
## Pre-ship gates (A/B in-memory, env flag OFF vs ON)

- [ ] G-A: Combined `correct_rate` improves by ≥ +10pp (60.3% → ≥ 70%).
      Result: OFF=__%, ON=__%, delta=__pp.
- [ ] G-B: `wrong_cross_team` reduces by ≥ 75% (17 → ≤ 4).
      Result: OFF=__, ON=__.
- [ ] G-C: No per-fixture `correct` regression (cece ≥ 22, gigi ≥ 35, wawa ≥ 25).
      Result: cece __→__, gigi __→__, wawa __→__.
- [ ] G-D: `wrong_same_team` non-increasing (9 today).
      Result: OFF=__, ON=__.
- [ ] G-E: `audit-coherence-invariants` C-2 on the 3 videos reduces by ≥ 50% post-deploy.
      Result: (Task 8).
- [ ] G-F: Fallback rate (WARN log lines from joint_attribute) ≤ 5% of rallies (≤ 1 of 22).
      Result: __ WARN lines in deploy log.
```

Fill in the result blanks. Mark each gate PASS or FAIL.

**STOP CONDITIONS:** If G-A, G-B, or G-D fail, do NOT proceed to Task 8. Diagnose:
- G-A or G-B too small: solver isn't firing enough. Check whether the env flag is actually picked up; check whether `Contact.player_candidates` has alternative candidates available.
- G-D fails: rule-driven swaps are flipping cross-team to same-team. The soft scoring may be wrong; consider tightening scoring or adding a rank-0-preference bonus.
- G-F too high (>5%): the rule set is too strict for real data. Investigate which rallies fall back; the cause is usually missing actions or UNKNOWN types breaking the chain.

- [ ] **Step 5: Commit the harness and report**

```bash
git add analysis/scripts/measure_attribution_joint_v2_ab.py analysis/reports/attribution_baseline/joint_v2_2026_05_11.md analysis/reports/attribution_baseline/joint_v2_ab_2026_05_11.json
git commit -m "$(cat <<'EOF'
report(joint-attribution): A/B harness + v2 measurement on 3 GT videos

scripts/measure_attribution_joint_v2_ab.py re-runs reattribute_players
in-memory on the 3 fresh-GT videos for both JOINT_ATTRIBUTION_V2 env
states and reports per-fixture deltas. Output captured alongside the
6 pre-ship gate verdicts.

Spec: docs/superpowers/specs/2026-05-11-joint-attribution-v2-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Flip default ON, fleet deploy, memory entry

**Files:**
- Modify: `analysis/rallycut/tracking/action_classifier.py` (flip env flag default to ON)
- Modify: `analysis/reports/attribution_baseline/joint_v2_2026_05_11.md` (fleet results)
- Create: `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/joint_attribution_v2_2026_05_11.md`
- Modify: `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY.md`

- [ ] **Step 1: Pre-deploy DB snapshot of the 3 GT videos**

```bash
mkdir -p analysis/reports/attribution_baseline/db_snapshots
PGPASSWORD=postgres psql -h localhost -p 5436 -U postgres -d rallycut \
  -t -A -c "
SELECT json_build_object(
  'video_id', r.video_id,
  'rally_id', r.id,
  'pt_id', pt.id,
  'actions_json', pt.actions_json
)
FROM rallies r
JOIN player_tracks pt ON pt.rally_id = r.id
WHERE r.video_id IN (
  '950fbe5d-fdad-4862-b05d-8b374bdd5ec6',
  'b097dd2a-6953-4e0e-a603-5be3552f462e',
  '5c756c41-1cc1-4486-a95c-97398912cfbe'
)
ORDER BY r.video_id, r.start_ms" \
  > analysis/reports/attribution_baseline/db_snapshots/pre_joint_v2_2026_05_11.jsonl
wc -l analysis/reports/attribution_baseline/db_snapshots/pre_joint_v2_2026_05_11.jsonl
```

Expected: 22 lines.

- [ ] **Step 2: Flip the env flag default to ON in action_classifier.py**

In `analysis/rallycut/tracking/action_classifier.py`, find the line added in Task 6:

```python
    if os.environ.get("JOINT_ATTRIBUTION_V2", "0") == "1" and team_assignments:
```

Change `"0"` to `"1"`:

```python
    if os.environ.get("JOINT_ATTRIBUTION_V2", "1") == "1" and team_assignments:
```

This means: default ON, explicit `JOINT_ATTRIBUTION_V2=0` disables for rollback.

- [ ] **Step 3: Verify unit tests still pass after flag flip**

Run: `cd analysis && uv run pytest tests/unit/test_joint_attribution.py tests/unit/test_action_classifier.py -v`
Expected: 29 + 163 = 192 tests PASS.

- [ ] **Step 4: Deploy to the 3 GT videos**

```bash
cd analysis
uv run rallycut reattribute-actions 950fbe5d-fdad-4862-b05d-8b374bdd5ec6
uv run rallycut reattribute-actions b097dd2a-6953-4e0e-a603-5be3552f462e
uv run rallycut reattribute-actions 5c756c41-1cc1-4486-a95c-97398912cfbe
```

Per-video output: re-attribution counts. Look for `joint_attribute fallback` WARN lines in stderr — count them. STOP if any video errors or shows >10 re-attributions per rally.

- [ ] **Step 5: Re-run baseline harness against DB**

```bash
cd analysis && uv run python -u scripts/measure_attribution_fresh_gt.py
```

Append the output to `analysis/reports/attribution_baseline/joint_v2_2026_05_11.md` under "## Post-deploy (DB read, env flag default ON)".

- [ ] **Step 6: Run coherence-invariants on the 3 GT videos**

```bash
cd analysis
uv run rallycut audit-coherence-invariants 950fbe5d-fdad-4862-b05d-8b374bdd5ec6
uv run rallycut audit-coherence-invariants b097dd2a-6953-4e0e-a603-5be3552f462e
uv run rallycut audit-coherence-invariants 5c756c41-1cc1-4486-a95c-97398912cfbe
```

Record the C-1/C-2/C-3 counts. Compare to the v1 post-deploy state (cece C-2=2, gigi C-2=3, wawa C-2=7).

Expected: C-2 violations DROP significantly (the joint solver prevents the very violations C-2 detects). Pre-ship gate G-E: ≥ 50% C-2 reduction on the 3 videos.

- [ ] **Step 7: Fleet deploy**

Run the fleet-deploy script pattern (single bash script in background, like `/tmp/fleet_deploy_team_chain_v1.sh` from the v1 workstream). Adapt for v2:

```bash
cat > /tmp/fleet_deploy_joint_v2.sh <<'EOF'
#!/usr/bin/env bash
set -u
cd /Users/mario/Personal/Projects/RallyCut/analysis
VIDEOS_FILE=/tmp/fleet_video_ids.txt
PRE_LOG=/tmp/coherence_pre_joint_v2.log
DEPLOY_LOG=/tmp/reattribute_joint_v2.log
POST_LOG=/tmp/coherence_post_joint_v2.log
SENTINEL=/tmp/joint_v2_done
PROGRESS=/tmp/joint_v2_progress.log
rm -f "$SENTINEL"; : > "$PROGRESS"

# Fleet ID list (assumes the same 70 videos as v1; re-derive if needed)
PGPASSWORD=postgres psql -h localhost -p 5436 -U postgres -d rallycut -t -A -c "
SELECT v.id FROM videos v
WHERE EXISTS (SELECT 1 FROM rallies r JOIN player_tracks pt ON pt.rally_id = r.id WHERE r.video_id = v.id)
  AND v.match_analysis_json IS NOT NULL
ORDER BY v.filename" > "$VIDEOS_FILE"

echo "[$(date +%H:%M:%S)] Starting v2 fleet deploy" | tee -a "$PROGRESS"

# Step 1: pre-coherence
: > "$PRE_LOG"; COUNT=0
while read -r vid; do
  COUNT=$((COUNT + 1)); echo "=== $vid ===" >> "$PRE_LOG"
  uv run rallycut audit-coherence-invariants "$vid" >> "$PRE_LOG" 2>&1
  [ $((COUNT % 10)) -eq 0 ] && echo "[$(date +%H:%M:%S)]   pre: $COUNT/$(wc -l < "$VIDEOS_FILE")" | tee -a "$PROGRESS"
done < "$VIDEOS_FILE"
echo "[$(date +%H:%M:%S)] Step 1 done" | tee -a "$PROGRESS"

# Step 2: fleet deploy
: > "$DEPLOY_LOG"; COUNT=0
while read -r vid; do
  COUNT=$((COUNT + 1)); echo "=== $vid ===" >> "$DEPLOY_LOG"
  uv run rallycut reattribute-actions "$vid" >> "$DEPLOY_LOG" 2>&1
  [ $((COUNT % 5)) -eq 0 ] && echo "[$(date +%H:%M:%S)]   deploy: $COUNT/$(wc -l < "$VIDEOS_FILE")" | tee -a "$PROGRESS"
done < "$VIDEOS_FILE"
echo "[$(date +%H:%M:%S)] Step 2 done" | tee -a "$PROGRESS"

# Step 3: post-coherence
: > "$POST_LOG"; COUNT=0
while read -r vid; do
  COUNT=$((COUNT + 1)); echo "=== $vid ===" >> "$POST_LOG"
  uv run rallycut audit-coherence-invariants "$vid" >> "$POST_LOG" 2>&1
  [ $((COUNT % 10)) -eq 0 ] && echo "[$(date +%H:%M:%S)]   post: $COUNT/$(wc -l < "$VIDEOS_FILE")" | tee -a "$PROGRESS"
done < "$VIDEOS_FILE"
echo "[$(date +%H:%M:%S)] Step 3 done" | tee -a "$PROGRESS"

echo "[$(date +%H:%M:%S)] ALL DONE" | tee -a "$PROGRESS"
touch "$SENTINEL"
EOF
chmod +x /tmp/fleet_deploy_joint_v2.sh
bash /tmp/fleet_deploy_joint_v2.sh &
echo "Fleet deploy PID: $!"
```

Use `run_in_background: true` if invoking through the agent.

Monitor /tmp/joint_v2_progress.log for milestones; sentinel /tmp/joint_v2_done indicates completion.

Expected: ~5-10 minutes total (3-step structure, similar to v1).

- [ ] **Step 8: Diff fleet coherence counts pre vs post**

Reuse the analysis script pattern from v1's `/tmp/diff_coherence.py`. Compute:
- Total C-1, C-2, C-3 PRE vs POST fleet-wide.
- Per-video delta (flag any video that INCREASED C-2 by > 2).
- Per-video re-attribution counts (parse deploy log).
- Cross-reference re-attributions vs C-2 deltas (like v1's cross-check diagnostic).

Append the fleet summary to `analysis/reports/attribution_baseline/joint_v2_2026_05_11.md` under "## Fleet deploy results".

**Expectation:** unlike v1 (where C-2 went UP due to teamAssignments-coverage visibility), v2 should produce a measurable C-2 DECREASE fleet-wide — the joint solver prevents the very violations the audit detects. If C-2 doesn't decrease, the solver isn't firing or its impact is small enough that the visibility effect still dominates.

- [ ] **Step 9: Write the memory entry**

Create `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/joint_attribution_v2_2026_05_11.md`:

```markdown
---
name: Joint attribution v2.0 2026-05-11
description: Beam-search joint rule-aware attribution. Enforces volleyball rules R1-R5 as HARD constraints over the whole rally. Default ON behind JOINT_ATTRIBUTION_V2 env flag. Targets the 17 wrong_cross_team errors that v1's local predicate couldn't reach.
type: project
---
# Joint Rule-Aware Attribution v2.0 — 2026-05-11

**Shipped:** 2026-05-11

## What it does

`reattribute_players` Pass 2b runs a beam-search joint solver over the entire rally that enforces R1-R5 as HARD constraints. New module `analysis/rallycut/tracking/joint_attribution.py` exposes `joint_attribute(actions, contacts, team_assignments, serving_team)`.

Rules:
- **R1** first action is SERVE by serving_team
- **R2** SERVE/ATTACK flip possession to opposite team
- **R3** RECEIVE/SET/DIG preserve possession on same team
- **R4** max 3 same-team consecutive contacts (unless 3rd is an ATTACK that crosses)
- **R5** BLOCK is a free pass — does not flip possession, does not count toward R4 (cover by blocker's team is legal)

Soft scoring: `-log(rank_distance + ε)` from existing `Contact.player_candidates`. No new feature engineering.

Beam width 50; ~4000 partial expansions per rally; fast.

Fallback: if no rule-valid assignment exists, return input unchanged and emit WARN log line ("never silently corrupt").

## Why this works where v1 didn't

v1's `_team_chain_override_allowed` predicate reasons LOCALLY per-contact and only fires when ALL of G1+G2+G3 pass. Cross-team errors where the local nearest is correct-team-equivalent-but-rule-violating still slip through.

v2 reasons over the WHOLE rally. The 17 cross-team errors on the 3 GT panel are textbook R2/R3 violations — by construction, no rule-violating assignment survives the beam.

## Measured impact (fill in after Tasks 7+8)

| Metric | Pre-v2 | Post-v2 | Delta |
|---|---:|---:|---:|
| Combined correct_rate (3 GT videos) | 60.3% | __% | __pp |
| wrong_cross_team | 17 | __ | __ |
| wrong_same_team (must not increase) | 9 | __ | __ |
| Fleet C-2 violations | 338 | __ | __ |
| Beam-empty fallbacks (3 GT) | — | __ | — |

Per-action matched accuracy:
- receive 94% → __%, attack 83% → __%, dig 82% → __%, set 67% → __%, serve 47% → __%

## Env flag

`JOINT_ATTRIBUTION_V2=0` restores prior behavior (v1 team-chain predicate + existing reattribute_players passes). Read at call time inside `reattribute_players`. Default ON post-ship.

## Files touched

- `analysis/rallycut/tracking/joint_attribution.py` — new module (~250 lines).
- `analysis/rallycut/tracking/action_classifier.py` — added env-flag-gated Pass 2b call.
- `analysis/tests/unit/test_joint_attribution.py` — 29 unit + integration tests.
- `analysis/scripts/measure_attribution_joint_v2_ab.py` — A/B harness.
- `analysis/reports/attribution_baseline/joint_v2_2026_05_11.md` — measurement report.
- `analysis/reports/attribution_baseline/db_snapshots/pre_joint_v2_2026_05_11.jsonl` — rollback snapshot.

## Commits

(Fill in commit SHAs from Tasks 1-8.)

## Open follow-ups (v2.1+)

- **Cross-rally server consistency** (v2.1): same player serves across consecutive same-team rallies. Targets the 4 wrong-of-two-teammates serve errors.
- **Role priors** (v2.2): setter usually at net; attacker on attacking side; digger in back-court. Targets the 5 same-team setter/dig errors.
- **Trajectory hints** (v2.3): ball direction post-contact constrains which player CAN have hit it. Additional signal for set/attack disambiguation.
- **Independent ball-only court_side helper** would let v1's G4 come back as a true 4th gate in the team-chain predicate. Different workstream.
```

Fill in measured-impact values from Tasks 7+8 results.

- [ ] **Step 10: Add MEMORY.md index entry**

In `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY.md`, under "## Current workstreams", add as the top entry (most recent first):

```markdown
- [SHIPPED] [**Joint attribution v2.0 2026-05-11**](joint_attribution_v2_2026_05_11.md) — Beam-search joint solver over the whole rally enforcing volleyball rules R1-R5 as HARD constraints. New module `joint_attribution.py`; Pass 2b in `reattribute_players` gated by `JOINT_ATTRIBUTION_V2` (default ON). Targets cross-team errors v1's local predicate couldn't reach. Panel correct __→__%; fleet C-2 __→__.
```

- [ ] **Step 11: Commit the production-code flag flip + reports + rollback snapshot**

```bash
git add analysis/rallycut/tracking/action_classifier.py analysis/reports/attribution_baseline/joint_v2_2026_05_11.md analysis/reports/attribution_baseline/db_snapshots/
git commit -m "$(cat <<'EOF'
ship(joint-attribution): flip JOINT_ATTRIBUTION_V2 default ON + fleet deploy

Pre-ship gates passed on the 3 GT videos. Flag flipped to default-on
in reattribute_players. Fleet of 70 videos re-attributed; coherence C-2
diff captured. DB snapshot of the 3 GT videos pre-deploy retained for
rollback.

Rollback: JOINT_ATTRIBUTION_V2=0 restores prior behavior without code
revert.

Spec: docs/superpowers/specs/2026-05-11-joint-attribution-v2-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(Memory files live OUTSIDE the repo — no git commit for them; the only repo commits are the in-repo files above.)

---

## Summary of touched files

In-repo (committed):
- `analysis/rallycut/tracking/joint_attribution.py` — new module.
- `analysis/rallycut/tracking/action_classifier.py` — Pass 2b wiring + env-flag flip.
- `analysis/tests/unit/test_joint_attribution.py` — new test file.
- `analysis/scripts/measure_attribution_joint_v2_ab.py` — new A/B harness.
- `analysis/reports/attribution_baseline/joint_v2_2026_05_11.md` — verdict report.
- `analysis/reports/attribution_baseline/joint_v2_ab_2026_05_11.json` — raw A/B output.
- `analysis/reports/attribution_baseline/db_snapshots/pre_joint_v2_2026_05_11.jsonl` — rollback snapshot.

Out-of-repo (user memory, not committed):
- `~/.claude/projects/.../memory/joint_attribution_v2_2026_05_11.md`
- `~/.claude/projects/.../memory/MEMORY.md` — index entry.

## Rollback procedure

1. Set env: `JOINT_ATTRIBUTION_V2=0` on the host running `reattribute-actions`.
2. Re-run `rallycut reattribute-actions <video>` on the regressed video(s). The env flag short-circuits the v2 Pass 2b, restoring prior behavior.
3. If the env-flag rollback is insufficient (e.g., the v2 deploy corrupted stored `actions_json` in a way the OFF path can't fix), restore from `db_snapshots/pre_joint_v2_2026_05_11.jsonl` via psql `UPDATE player_tracks SET actions_json = ... WHERE id = pt_id` per row.
4. If the underlying code change itself is suspect, `git revert <commits-from-tasks-1-6>` and redeploy. The flag-flip commit is separable.
