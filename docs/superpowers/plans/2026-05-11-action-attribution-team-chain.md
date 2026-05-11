# Action Attribution Team-Chain Override (v1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the unconditional "don't override the nearest" guard in `reattribute_players` Pass 2 with a 4-gate trust-the-chain predicate. Reduce cross-team attribution errors on the 3 fresh-GT videos from 16 → ≤8 while not regressing same-team errors or coherence-invariant counts.

**Architecture:** Add two pure helpers (`_chain_integrity`, `_team_chain_override_allowed`) to `analysis/rallycut/tracking/action_classifier.py`. Wire the predicate into `reattribute_players` (line ~2963). Gate the new behavior behind env flag `RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN` (default ON). All other passes (server exclusion, ReID, formation serving-team, Viterbi) untouched.

**Tech Stack:** Python 3.11, pytest, the existing `attribution_bench` evaluation primitives. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-11-action-attribution-team-chain-design.md`

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `analysis/rallycut/tracking/action_classifier.py` | Modify | Add `_chain_integrity` and `_team_chain_override_allowed` helpers near `_compute_expected_teams` (lines ~2808+). Replace guard at ~2963-2968 inside `reattribute_players` with the new predicate. |
| `analysis/tests/unit/test_action_attribution_team_chain.py` | Create | All new unit tests: chain-integrity truth table, 4-gate predicate truth table, end-to-end `reattribute_players` integration test on a constructed rally. |
| `analysis/scripts/measure_attribution_fresh_gt.py` | Existing | Baseline / A/B harness on the 3 fresh-GT videos. Reused unchanged. |
| `analysis/reports/attribution_baseline/team_chain_v1_2026_05_11.md` | Create | The before/after report with measured deltas and the pre-ship-gate verdict. |
| `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/attribution_team_chain_v1_2026_05_11.md` | Create | Post-ship memory entry. |
| `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY.md` | Modify | Add index entry pointing to the new memory file. |

---

## Task 1: Add `_chain_integrity` helper (pure function, TDD)

**Files:**
- Modify: `analysis/rallycut/tracking/action_classifier.py` (add new function below `_compute_expected_teams` at line ~2855)
- Create: `analysis/tests/unit/test_action_attribution_team_chain.py`

- [ ] **Step 1: Create the new test file with the chain-integrity truth-table tests**

Create `analysis/tests/unit/test_action_attribution_team_chain.py`:

```python
"""Unit tests for the team-chain override predicate and its helpers.

Spec: docs/superpowers/specs/2026-05-11-action-attribution-team-chain-design.md
"""

from __future__ import annotations

import math
from unittest.mock import patch

import pytest

from rallycut.tracking.action_classifier import (
    ActionType,
    ClassifiedAction,
    _chain_integrity,
)
from rallycut.tracking.contact_detector import Contact


def _action(
    action_type: ActionType,
    frame: int,
    player_track_id: int = 1,
    confidence: float = 0.9,
    is_synthetic: bool = False,
    court_side: str = "near",
) -> ClassifiedAction:
    return ClassifiedAction(
        action_type=action_type,
        frame=frame,
        ball_x=0.5,
        ball_y=0.5,
        velocity=0.02,
        player_track_id=player_track_id,
        court_side=court_side,
        confidence=confidence,
        is_synthetic=is_synthetic,
    )


class TestChainIntegrity:
    """Truth table for _chain_integrity."""

    def test_clean_chain_after_seed_serve(self) -> None:
        """Serve seeds the chain; subsequent known non-synthetic actions stay intact."""
        actions = [
            _action(ActionType.SERVE, 10, player_track_id=1),
            _action(ActionType.RECEIVE, 30, player_track_id=2),
            _action(ActionType.SET, 50, player_track_id=3),
            _action(ActionType.ATTACK, 70, player_track_id=2),
        ]
        result = _chain_integrity(actions)
        assert result == [True, True, True, True]

    def test_unknown_breaks_chain_from_that_point(self) -> None:
        """An UNKNOWN action sets all subsequent positions to False."""
        actions = [
            _action(ActionType.SERVE, 10, player_track_id=1),
            _action(ActionType.RECEIVE, 30, player_track_id=2),
            _action(ActionType.UNKNOWN, 50, player_track_id=3),
            _action(ActionType.ATTACK, 70, player_track_id=2),
        ]
        result = _chain_integrity(actions)
        assert result == [True, True, False, False]

    def test_synthetic_non_seed_breaks_chain(self) -> None:
        """A synthetic dig/set in the middle of a chain breaks downstream."""
        actions = [
            _action(ActionType.SERVE, 10, player_track_id=1),
            _action(ActionType.RECEIVE, 30, player_track_id=2),
            _action(ActionType.DIG, 50, player_track_id=3, is_synthetic=True),
            _action(ActionType.ATTACK, 70, player_track_id=2),
        ]
        result = _chain_integrity(actions)
        assert result == [True, True, False, False]

    def test_no_seed_serve_means_all_false(self) -> None:
        """Actions before the first valid SERVE all read False."""
        actions = [
            _action(ActionType.RECEIVE, 30, player_track_id=2),
            _action(ActionType.SET, 50, player_track_id=3),
        ]
        result = _chain_integrity(actions)
        assert result == [False, False]

    def test_serve_with_unattributed_player_does_not_seed(self) -> None:
        """A SERVE with player_track_id=-1 does NOT seed the chain."""
        actions = [
            _action(ActionType.SERVE, 10, player_track_id=-1),
            _action(ActionType.RECEIVE, 30, player_track_id=2),
        ]
        result = _chain_integrity(actions)
        assert result == [False, False]

    def test_synthetic_serve_seeds_chain(self) -> None:
        """A synthetic SERVE with player_track_id >= 0 still seeds the chain.

        Rationale: synthetic serves are seeded with a server identity by
        _make_synthetic_serve. The downstream chain is observable from there.
        """
        actions = [
            _action(ActionType.SERVE, 10, player_track_id=1, is_synthetic=True),
            _action(ActionType.RECEIVE, 30, player_track_id=2),
        ]
        result = _chain_integrity(actions)
        assert result == [True, True]

    def test_empty_actions(self) -> None:
        """Empty input returns empty output."""
        assert _chain_integrity([]) == []
```

- [ ] **Step 2: Run the tests and verify they fail with import error**

Run: `cd analysis && uv run pytest tests/unit/test_action_attribution_team_chain.py -v`
Expected: ImportError (`_chain_integrity` not yet defined).

- [ ] **Step 3: Implement `_chain_integrity` in `action_classifier.py`**

Add directly after `_compute_expected_teams` (currently ends at line ~2855), before `def reattribute_players` (line ~2858):

```python
def _chain_integrity(actions: list[ClassifiedAction]) -> list[bool]:
    """Return a parallel list[bool] marking each action's chain integrity.

    The chain is seeded by the first action that is a SERVE with
    ``player_track_id >= 0`` (synthetic or otherwise). All actions before the
    seed are marked False. The seed itself is True. After the seed, an action
    is True iff no UNKNOWN action and no non-seed synthetic action appears
    between the seed and that action (inclusive of any breaks at the action
    itself).

    Used by ``_team_chain_override_allowed`` (gate G2) to decide whether the
    serve-derived ``expected_team`` chain is trustworthy enough to override
    the nearest-player guard inside ``reattribute_players``.
    """
    n = len(actions)
    result = [False] * n
    seen_seed = False
    broken = False
    for i, a in enumerate(actions):
        if not seen_seed:
            if a.action_type == ActionType.SERVE and a.player_track_id >= 0:
                seen_seed = True
                result[i] = True
            continue
        if a.action_type == ActionType.UNKNOWN or a.is_synthetic:
            broken = True
            result[i] = False
            continue
        result[i] = not broken
    return result
```

- [ ] **Step 4: Run the tests and verify they pass**

Run: `cd analysis && uv run pytest tests/unit/test_action_attribution_team_chain.py::TestChainIntegrity -v`
Expected: all 7 tests PASS.

- [ ] **Step 5: Run the full action_classifier test suite to verify no regression**

Run: `cd analysis && uv run pytest tests/unit/test_action_classifier.py -v`
Expected: all tests PASS (no behavior change to existing code yet).

- [ ] **Step 6: Commit**

```bash
git add analysis/rallycut/tracking/action_classifier.py analysis/tests/unit/test_action_attribution_team_chain.py
git commit -m "$(cat <<'EOF'
feat(attribution): add _chain_integrity helper for team-chain trust gating

Pure function returning a parallel list[bool] marking which actions have
an unbroken chain from the seed serve. Chain breaks on UNKNOWN actions
and non-seed synthetic actions. Used by the upcoming
_team_chain_override_allowed predicate.

Spec: docs/superpowers/specs/2026-05-11-action-attribution-team-chain-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Add `_team_chain_override_allowed` predicate (4-gate combiner)

**Files:**
- Modify: `analysis/rallycut/tracking/action_classifier.py` (add new function below `_chain_integrity`)
- Modify: `analysis/tests/unit/test_action_attribution_team_chain.py` (add `TestTeamChainOverrideAllowed`)

- [ ] **Step 1: Add the truth-table tests**

Append to `analysis/tests/unit/test_action_attribution_team_chain.py`:

```python
def _contact_with_candidates(
    frame: int,
    nearest_dist: float,
    candidates: list[tuple[int, float]],
    court_side: str = "near",
) -> Contact:
    """Build a Contact with ranked player_candidates and player_distance set."""
    return Contact(
        frame=frame,
        ball_x=0.5,
        ball_y=0.5,
        velocity=0.02,
        direction_change_deg=60.0,
        player_track_id=candidates[0][0] if candidates else -1,
        player_distance=nearest_dist,
        player_candidates=candidates,
        court_side=court_side,
        is_validated=True,
    )


class TestTeamChainOverrideAllowed:
    """Truth table for the 4-gate predicate.

    Convention used in these tests: team_assignments[tid] = 0 means near
    (team A), = 1 means far (team B). The current (wrong) attribution is
    track 4 on team 1; the correct attribution should be track 1 on team 0.
    expected_team = 0 (near). The Contact reports court_side="near".
    """

    def _setup(
        self,
        *,
        confidence: float = 0.9,
        chain_ok: bool = True,
        candidate_dist: float = 0.06,
        current_dist: float = 0.05,
        court_side: str = "near",
        env_off: bool = False,
    ):
        from rallycut.tracking.action_classifier import (
            _team_chain_override_allowed,
        )
        action = _action(
            ActionType.RECEIVE, 100, player_track_id=4,
            confidence=confidence,
        )
        contact = _contact_with_candidates(
            frame=100,
            nearest_dist=current_dist,
            candidates=[(4, current_dist), (1, candidate_dist)],
            court_side=court_side,
        )
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}
        expected_team = 0  # The CORRECT team for the receive
        return (
            _team_chain_override_allowed,
            action, contact, expected_team, chain_ok, team_assignments,
            env_off,
        )

    def test_all_gates_pass_allows_override(self) -> None:
        fn, action, contact, expected, chain_ok, ta, _ = self._setup()
        with patch.dict("os.environ", {"RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN": "1"}):
            assert fn(action, contact, expected, chain_ok, ta) is True

    def test_env_flag_off_denies_override(self) -> None:
        fn, action, contact, expected, chain_ok, ta, _ = self._setup()
        with patch.dict("os.environ", {"RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN": "0"}):
            assert fn(action, contact, expected, chain_ok, ta) is False

    def test_low_action_confidence_denies_override(self) -> None:
        fn, action, contact, expected, chain_ok, ta, _ = self._setup(
            confidence=0.5,
        )
        with patch.dict("os.environ", {"RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN": "1"}):
            assert fn(action, contact, expected, chain_ok, ta) is False

    def test_broken_chain_denies_override(self) -> None:
        fn, action, contact, expected, _, ta, _ = self._setup()
        with patch.dict("os.environ", {"RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN": "1"}):
            assert fn(action, contact, expected, False, ta) is False

    def test_no_candidate_within_distance_cap_denies_override(self) -> None:
        # candidate is 4x further than current → > 1.5x cap
        fn, action, contact, expected, chain_ok, ta, _ = self._setup(
            current_dist=0.05, candidate_dist=0.25,
        )
        with patch.dict("os.environ", {"RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN": "1"}):
            assert fn(action, contact, expected, chain_ok, ta) is False

    def test_court_side_disagrees_denies_override(self) -> None:
        # expected_team=0 → expected_side="near"; contact reports "far"
        fn, action, contact, expected, chain_ok, ta, _ = self._setup(
            court_side="far",
        )
        with patch.dict("os.environ", {"RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN": "1"}):
            assert fn(action, contact, expected, chain_ok, ta) is False

    def test_court_side_unknown_is_soft_pass(self) -> None:
        # Allows override when court_side cannot corroborate (no calibration)
        fn, action, contact, expected, chain_ok, ta, _ = self._setup(
            court_side="unknown",
        )
        with patch.dict("os.environ", {"RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN": "1"}):
            assert fn(action, contact, expected, chain_ok, ta) is True

    def test_current_player_distance_infinite_denies_override(self) -> None:
        # No current distance → cannot enforce distance cap → deny
        fn, action, _contact, expected, chain_ok, ta, _ = self._setup()
        contact_inf = _contact_with_candidates(
            frame=100,
            nearest_dist=math.inf,
            candidates=[(4, math.inf), (1, 0.06)],
            court_side="near",
        )
        with patch.dict("os.environ", {"RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN": "1"}):
            assert fn(action, contact_inf, expected, chain_ok, ta) is False
```

- [ ] **Step 2: Run the tests and verify they fail with import error**

Run: `cd analysis && uv run pytest tests/unit/test_action_attribution_team_chain.py::TestTeamChainOverrideAllowed -v`
Expected: ImportError (`_team_chain_override_allowed` not defined).

- [ ] **Step 3: Implement `_team_chain_override_allowed`**

Add directly after `_chain_integrity` in `action_classifier.py`:

```python
def _team_chain_override_allowed(
    action: ClassifiedAction,
    contact: Contact,
    expected_team: int,
    chain_integrity_i: bool,
    team_assignments: dict[int, int],
    max_distance_ratio: float = 1.5,
) -> bool:
    """The 4-gate predicate that relaxes the nearest-player guard.

    Returns True iff the team-chain-derived `expected_team` is trustworthy
    enough to override the current (wrong-team) nearest-player attribution
    at this contact. Spec:
    docs/superpowers/specs/2026-05-11-action-attribution-team-chain-design.md

    Gates:
      G0 (env flag): RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN != "0" (default ON).
      G1 (confidence): action.confidence >= 0.7.
      G2 (chain integrity): chain_integrity_i is True.
      G3 (candidate-on-expected-team within distance cap): some
        Contact.player_candidate has team == expected_team and distance
        <= max_distance_ratio * contact.player_distance.
      G4 (ball-trajectory corroborator): Contact.court_side either agrees
        with expected_team (near<->0, far<->1) OR is "unknown" (soft pass —
        no corroboration available, defer to chain alone).
    """
    # G0
    if os.environ.get("RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN", "1") == "0":
        return False
    # G1
    if action.confidence < 0.7:
        return False
    # G2
    if not chain_integrity_i:
        return False
    # G3
    current_dist = contact.player_distance
    if not math.isfinite(current_dist):
        return False
    dist_cap = max_distance_ratio * current_dist
    has_correct_team_candidate = any(
        team_assignments.get(tid) == expected_team and dist <= dist_cap
        for tid, dist in contact.player_candidates
    )
    if not has_correct_team_candidate:
        return False
    # G4
    expected_side = "near" if expected_team == 0 else "far"
    if contact.court_side in ("near", "far") and contact.court_side != expected_side:
        return False
    # court_side == "unknown" → soft pass (chain trust alone is sufficient)
    return True
```

Also ensure these imports are present at the top of `action_classifier.py` (check the existing imports first):

```python
import math
import os
```

If either is missing, add it in the existing import block, alphabetically. (Likely `math` is already there; `os` may not be.)

- [ ] **Step 4: Run the tests and verify they pass**

Run: `cd analysis && uv run pytest tests/unit/test_action_attribution_team_chain.py -v`
Expected: all 15 tests PASS (7 chain-integrity + 8 predicate).

- [ ] **Step 5: Run the full action_classifier test suite (no regressions yet)**

Run: `cd analysis && uv run pytest tests/unit/test_action_classifier.py -v`
Expected: all tests PASS.

- [ ] **Step 6: Commit**

```bash
git add analysis/rallycut/tracking/action_classifier.py analysis/tests/unit/test_action_attribution_team_chain.py
git commit -m "$(cat <<'EOF'
feat(attribution): add _team_chain_override_allowed 4-gate predicate

Pure function that decides whether the serve-chain-derived expected_team
is trustworthy enough to override the nearest-player guard inside
reattribute_players Pass 2. Four gates: G1 action confidence >= 0.7,
G2 chain integrity, G3 candidate-on-expected-team within distance cap,
G4 Contact.court_side corroborator. Env flag
RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN (default "1") supports fast rollback.

Spec: docs/superpowers/specs/2026-05-11-action-attribution-team-chain-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Wire the predicate into `reattribute_players` Pass 2

**Files:**
- Modify: `analysis/rallycut/tracking/action_classifier.py` inside `reattribute_players` (currently lines 2858-3034). The guard to replace is at lines 2963-2968.
- Modify: `analysis/tests/unit/test_action_attribution_team_chain.py` (add `TestReattributePlayersIntegration`)

- [ ] **Step 1: Add the integration test**

Append to `analysis/tests/unit/test_action_attribution_team_chain.py`:

```python
class TestReattributePlayersIntegration:
    """End-to-end test that reattribute_players Pass 2 fires when the new
    predicate passes — even when the wrong-team current attribution is the
    spatially nearest candidate (the canonical bug pattern)."""

    def test_cross_team_receive_overridden_when_chain_trustworthy(self) -> None:
        """The user-quoted bug: P2 receives P1's serve (same team) — should
        be overridden to a candidate on the receiving team when all gates
        pass.

        Setup (PIDs already in canonical 1-4 space):
          team 0 (near, A) = {1, 2}, team 1 (far, B) = {3, 4}
          actions:
            - serve by track 3 (team B, far)
            - receive currently attributed to track 4 (team B, NEAREST in
              candidates) — wrong, expected_team is 0 (team A)
            - receive's contact has candidates: [(4, 0.05), (1, 0.07)]
              ⇒ track 1 (team A) is within 1.5x distance of track 4.
            - Contact.court_side = "near" — corroborates expected_team=0
        """
        from rallycut.tracking.action_classifier import reattribute_players

        actions = [
            _action(
                ActionType.SERVE, 50, player_track_id=3, confidence=0.95,
                court_side="far",
            ),
            _action(
                ActionType.RECEIVE, 90, player_track_id=4, confidence=0.9,
                court_side="near",
            ),
        ]
        contacts = [
            _contact_with_candidates(
                frame=50, nearest_dist=0.04,
                candidates=[(3, 0.04)],
                court_side="far",
            ),
            _contact_with_candidates(
                frame=90, nearest_dist=0.05,
                candidates=[(4, 0.05), (1, 0.07)],
                court_side="near",
            ),
        ]
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}

        with patch.dict("os.environ", {"RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN": "1"}):
            reattribute_players(actions, contacts, team_assignments)

        # The receive should now be attributed to track 1 (team A), not 4.
        assert actions[1].action_type == ActionType.RECEIVE
        assert actions[1].player_track_id == 1

    def test_env_flag_off_preserves_old_behavior(self) -> None:
        """With env flag off, the old unconditional nearest-guard blocks
        the override and the wrong-team attribution is preserved."""
        from rallycut.tracking.action_classifier import reattribute_players

        actions = [
            _action(
                ActionType.SERVE, 50, player_track_id=3, confidence=0.95,
                court_side="far",
            ),
            _action(
                ActionType.RECEIVE, 90, player_track_id=4, confidence=0.9,
                court_side="near",
            ),
        ]
        contacts = [
            _contact_with_candidates(
                frame=50, nearest_dist=0.04,
                candidates=[(3, 0.04)],
                court_side="far",
            ),
            _contact_with_candidates(
                frame=90, nearest_dist=0.05,
                candidates=[(4, 0.05), (1, 0.07)],
                court_side="near",
            ),
        ]
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}

        with patch.dict("os.environ", {"RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN": "0"}):
            reattribute_players(actions, contacts, team_assignments)

        # Old behavior: wrong-team nearest attribution is preserved.
        assert actions[1].player_track_id == 4

    def test_broken_chain_blocks_override(self) -> None:
        """An UNKNOWN action between the serve and the receive breaks the
        chain — the override is blocked even with env flag on."""
        from rallycut.tracking.action_classifier import reattribute_players

        actions = [
            _action(
                ActionType.SERVE, 50, player_track_id=3, confidence=0.95,
                court_side="far",
            ),
            _action(
                ActionType.UNKNOWN, 70, player_track_id=2, confidence=0.4,
            ),
            _action(
                ActionType.RECEIVE, 90, player_track_id=4, confidence=0.9,
                court_side="near",
            ),
        ]
        contacts = [
            _contact_with_candidates(
                frame=50, nearest_dist=0.04,
                candidates=[(3, 0.04)],
                court_side="far",
            ),
            _contact_with_candidates(
                frame=70, nearest_dist=0.05,
                candidates=[(2, 0.05)],
                court_side="near",
            ),
            _contact_with_candidates(
                frame=90, nearest_dist=0.05,
                candidates=[(4, 0.05), (1, 0.07)],
                court_side="near",
            ),
        ]
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}

        with patch.dict("os.environ", {"RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN": "1"}):
            reattribute_players(actions, contacts, team_assignments)

        # Chain broken by UNKNOWN at frame 70 → no override fires.
        assert actions[2].player_track_id == 4
```

- [ ] **Step 2: Run the tests and verify they fail in the way you expect**

Run: `cd analysis && uv run pytest tests/unit/test_action_attribution_team_chain.py::TestReattributePlayersIntegration -v`
Expected:
- `test_cross_team_receive_overridden_when_chain_trustworthy` FAILS (current code's unconditional guard still blocks the override).
- `test_env_flag_off_preserves_old_behavior` PASSES (no change yet).
- `test_broken_chain_blocks_override` PASSES (current code already blocks via the unconditional guard).

The failing first test is what we're about to make pass.

- [ ] **Step 3: Modify `reattribute_players` to use the new predicate**

In `analysis/rallycut/tracking/action_classifier.py`, locate `reattribute_players` at line ~2858. Find the block at lines ~2912-2918 that computes `expected_teams`:

```python
    expected_teams = _compute_expected_teams(actions, team_assignments)
```

Immediately below that line, add the chain-integrity computation:

```python
    chain_integrity = _chain_integrity(actions)
```

Then locate lines ~2963-2968 — the unconditional nearest-guard:

```python
        # Guard: don't override the nearest candidate. Proximity is hard
        # physical evidence; the expected_team chain drifts when contacts
        # are missed or action types are wrong. Only override non-nearest
        # attributions (unmapped tracks or clear team mismatches).
        if (
            not is_unmapped
            and contact.player_candidates
            and contact.player_candidates[0][0] == action.player_track_id
        ):
            continue
```

Replace those 9 lines with:

```python
        # Guard: don't override the nearest candidate UNLESS the team-chain
        # trust gates pass. The 4-gate predicate is the strict relaxation of
        # the prior unconditional guard — env flag
        # RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN=0 restores the old behavior.
        # Spec: docs/superpowers/specs/2026-05-11-action-attribution-team-chain-design.md
        current_is_nearest = (
            bool(contact.player_candidates)
            and contact.player_candidates[0][0] == action.player_track_id
        )
        if not is_unmapped and current_is_nearest:
            chain_ok_i = (
                chain_integrity[i] if i < len(chain_integrity) else False
            )
            if not _team_chain_override_allowed(
                action=action,
                contact=contact,
                expected_team=expected_team,
                chain_integrity_i=chain_ok_i,
                team_assignments=team_assignments,
                max_distance_ratio=max_distance_ratio,
            ):
                continue
            logger.info(
                "team_chain_override frame=%d action=%s old_pid=%d "
                "expected_team=%d (relaxed nearest-guard via chain trust)",
                action.frame, action.action_type.value,
                action.player_track_id, expected_team,
            )
```

- [ ] **Step 4: Run the integration tests and verify they pass**

Run: `cd analysis && uv run pytest tests/unit/test_action_attribution_team_chain.py::TestReattributePlayersIntegration -v`
Expected: all 3 tests PASS.

- [ ] **Step 5: Run the full action_classifier test suite — guard against regressions**

Run: `cd analysis && uv run pytest tests/unit/test_action_classifier.py -v`
Expected: all tests PASS. If any existing test fails, it likely depends on the old unconditional-guard behavior — investigate before proceeding.

- [ ] **Step 6: Run the full unit test suite as a final smoke check**

Run: `cd analysis && uv run pytest tests/unit -v`
Expected: all tests PASS (excluding pre-existing failures from other unrelated workstreams, which should be empty on a clean main).

- [ ] **Step 7: Type check + lint**

Run: `cd analysis && uv run mypy rallycut/tracking/action_classifier.py`
Expected: no new errors.

Run: `cd analysis && uv run ruff check rallycut/tracking/action_classifier.py analysis/tests/unit/test_action_attribution_team_chain.py`
Expected: no new errors.

If `analysis/tests/unit/test_action_attribution_team_chain.py` is rejected by ruff for missing-future-annotations or similar, the test file already includes `from __future__ import annotations` per Step 1 — if a different rule fires, fix inline (likely line-length).

- [ ] **Step 8: Commit**

```bash
git add analysis/rallycut/tracking/action_classifier.py analysis/tests/unit/test_action_attribution_team_chain.py
git commit -m "$(cat <<'EOF'
feat(attribution): relax nearest-guard via team-chain trust predicate

reattribute_players Pass 2 now overrides the nearest-player guard when
the 4-gate predicate _team_chain_override_allowed returns True
(confidence >= 0.7, intact action chain from seed serve, candidate on
expected team within 1.5x distance, Contact.court_side corroborates or
unknown). Env flag RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN=0 restores the
prior unconditional guard for fast rollback.

Spec: docs/superpowers/specs/2026-05-11-action-attribution-team-chain-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: A/B measurement on the 3 fresh-GT videos

**Files:**
- Read-only on `analysis/scripts/measure_attribution_fresh_gt.py` (existing harness).
- Create: `analysis/reports/attribution_baseline/team_chain_v1_2026_05_11.md` (the A/B report).

The harness reads `actions_json` from the DB. For the A/B comparison the in-DB attributions need to reflect the new code path. We do NOT redeploy to DB yet — instead this task uses an in-memory re-run. Task 5 handles the DB deploy.

- [ ] **Step 1: Create the report file and capture the baseline**

```bash
mkdir -p analysis/reports/attribution_baseline
cat > analysis/reports/attribution_baseline/team_chain_v1_2026_05_11.md <<'EOF'
# Attribution Team-Chain v1 — Measurement Report (2026-05-11)

Spec: docs/superpowers/specs/2026-05-11-action-attribution-team-chain-design.md
Plan: docs/superpowers/plans/2026-05-11-action-attribution-team-chain.md

## Baseline (DB read — current production)

(Filled in below from `scripts/measure_attribution_fresh_gt.py` stdout.)

EOF
```

Then from `analysis/`:

```bash
uv run python -u scripts/measure_attribution_fresh_gt.py
```

Expected: combined correct 60.3%, wrong 19.1%, missing 20.6% (the 2026-05-11 baseline). Append the stdout under the "## Baseline" section in the report. The script reads from DB; the env flag doesn't affect this run.

- [ ] **Step 2: Build a small in-memory A/B harness**

Create `analysis/scripts/measure_attribution_team_chain_ab.py`:

```python
"""A/B measurement: re-run reattribute_players on the 3 GT videos in
memory (with and without the env flag set) and report the deltas.

Reads contacts_json + the pre-reattribute actions_json (the on-disk row
already had reattribute_players applied, but we re-run it locally to
exercise the new predicate against the env flag).

Run from analysis/:
    uv run python scripts/measure_attribution_team_chain_ab.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from rallycut.evaluation.attribution_bench import (
    MATCH_TOLERANCE_FRAMES,
    WRONG_CATEGORIES,
    aggregate,
    score_rally,
)
from rallycut.evaluation.db import get_connection
from rallycut.tracking.action_classifier import (
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
    / "reports" / "attribution_baseline" / "team_chain_ab_2026_05_11.json"
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
    from rallycut.tracking.action_classifier import ActionType
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


def _score(rallies_data, team_chain_on: bool) -> dict[str, Any]:
    os.environ["RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN"] = "1" if team_chain_on else "0"
    scored_rallies: list[dict[str, Any]] = []
    for r in rallies_data:
        actions = _reconstruct_actions(r["actions_json"])
        contacts = _reconstruct_contacts(r["contacts_json"])
        team_assignments_raw = (r["actions_json"] or {}).get("teamAssignments", {})
        team_assignments = {
            int(k): (0 if v == "A" else 1) for k, v in team_assignments_raw.items()
        }
        if team_assignments and actions:
            reattribute_players(actions, contacts, team_assignments)
        pipeline_actions = [a.to_dict() for a in actions]
        # Normalise GT once
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


def main() -> int:
    rallies_data: list[dict[str, Any]] = []
    with get_connection() as conn, conn.cursor() as cur:
        match_analyses: dict[str, dict] = {}
        for fixture, vid in FRESH_GT_VIDEOS.items():
            cur.execute("SELECT match_analysis_json FROM videos WHERE id = %s", (vid,))
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

    off = _score(rallies_data, team_chain_on=False)
    on = _score(rallies_data, team_chain_on=True)

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

    s_off = _summary("OFF (production baseline)", off["agg"])
    s_on  = _summary("ON  (team-chain v1)",       on["agg"])

    print("\n=== DELTA (on − off) ===")
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

- [ ] **Step 2: Run the A/B harness**

Run from `analysis/`: `uv run python -u scripts/measure_attribution_team_chain_ab.py`
Expected output: counts for OFF and ON, plus deltas. Save full stdout to the report file.

- [ ] **Step 3: Verify the 5 pre-ship gates**

Append a checklist section to `analysis/reports/attribution_baseline/team_chain_v1_2026_05_11.md` (created in Step 1):

```markdown
## Pre-ship gates (A/B in-memory, env flag OFF vs ON)

- [ ] G-A: Combined `correct_rate` improves by ≥ +5pp (60.3% → ≥ 65.3%).
      Result: OFF=__%, ON=__%, delta=__pp.
- [ ] G-B: `wrong_cross_team` ≥ 50% reduction (16 → ≤ 8).
      Result: OFF=__, ON=__.
- [ ] G-C: No per-fixture `correct` regression (cece ≥ 22, gigi ≥ 35, wawa ≥ 25).
      Result: cece __→__, gigi __→__, wawa __→__.
- [ ] G-D: `wrong_same_team` count non-increasing (7 today).
      Result: OFF=__, ON=__.
- [ ] G-E: `audit-coherence-invariants` C-2 violation count on the 3 videos ≤ current baseline.
      Result: (filled in next step).
```

Fill in the result blanks from the A/B harness output. Check off any passing gate.

**STOP CONDITIONS:** If any of G-A through G-D fail, do NOT proceed to Task 5. Diagnose:
- G-A or G-C fails: the new behavior may be over-aggressive — investigate which rallies regress and which gate (G1/G2/G3/G4) is being too permissive.
- G-B fails: the new behavior is too conservative — too many gates vetoing the override.
- G-D fails: cross-team errors are being flipped to same-team errors — investigate the candidate selection.

- [ ] **Step 4: Run coherence-invariants A/B on the 3 videos**

The audit reads from DB and the DB doesn't yet reflect the new code. We need to either (a) deploy to the 3 GT videos first (Task 5) then audit, or (b) run a code-level dry-run. Skip the coherence step here and check it AFTER Task 5 instead. Mark G-E as "deferred to post-deploy".

- [ ] **Step 5: Commit the report**

```bash
git add analysis/scripts/measure_attribution_team_chain_ab.py analysis/reports/attribution_baseline/team_chain_v1_2026_05_11.md analysis/reports/attribution_baseline/team_chain_ab_2026_05_11.json
git commit -m "$(cat <<'EOF'
report(attribution): A/B harness + team-chain v1 measurement on 3 GT videos

scripts/measure_attribution_team_chain_ab.py re-runs reattribute_players
in-memory on the 3 fresh-GT videos for both env-flag states and reports
per-fixture deltas. Output committed for record alongside the pre-ship
gate verdicts.

Spec: docs/superpowers/specs/2026-05-11-action-attribution-team-chain-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Deploy to the 3 GT videos and re-validate via DB

**Files:**
- Read-only on `analysis/rallycut/cli/commands/reattribute_actions.py`.
- Edit: `analysis/reports/attribution_baseline/team_chain_v1_2026_05_11.md` (final verdict).

The new code already runs whenever `rallycut reattribute-actions` is invoked. This task writes the new attributions to DB on just the 3 GT videos (smallest possible blast radius before fleet) and re-verifies via the existing `measure_attribution_fresh_gt.py`.

- [ ] **Step 1: Take a DB safety snapshot of the 3 videos' actions_json**

Run:

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
  > analysis/reports/attribution_baseline/db_snapshots/pre_team_chain_2026_05_11.jsonl
```

Expected: 22 lines of JSON written to the snapshot file (one rally per line; `-t -A` disables headers + alignment so psql emits raw rows). `pt_id` is included so a rollback `UPDATE player_tracks SET actions_json = ... WHERE id = pt_id` is trivial.

Verify:

```bash
wc -l analysis/reports/attribution_baseline/db_snapshots/pre_team_chain_2026_05_11.jsonl
```

Expected: `22 ...`.

- [ ] **Step 2: Deploy via reattribute-actions on cece**

Run from `analysis/`:

```bash
RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN=1 uv run rallycut reattribute-actions 950fbe5d-fdad-4862-b05d-8b374bdd5ec6
```

Expected: console output listing per-rally re-attributions. Look for the `team_chain_override` log lines from Task 3, Step 3. STOP IF the command errors or reports >5 re-attributions per rally — that's a sign the predicate is firing too often.

- [ ] **Step 3: Deploy on gigi**

```bash
RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN=1 uv run rallycut reattribute-actions b097dd2a-6953-4e0e-a603-5be3552f462e
```

- [ ] **Step 4: Deploy on wawa**

```bash
RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN=1 uv run rallycut reattribute-actions 5c756c41-1cc1-4486-a95c-97398912cfbe
```

- [ ] **Step 5: Re-run the baseline harness against DB**

Run from `analysis/`: `uv run python -u scripts/measure_attribution_fresh_gt.py`
Expected: numbers match the "ON" side of the A/B harness output from Task 4 (small variance acceptable if `reattribute-actions` invoked other passes — server exclusion, ReID, formation — that the in-memory A/B harness didn't run).

Append the printed combined / per-fixture / per-action / per-rally output to `analysis/reports/attribution_baseline/team_chain_v1_2026_05_11.md` under "## Post-deploy (DB)".

- [ ] **Step 6: Run coherence-invariants on the 3 videos**

```bash
cd analysis && uv run rallycut audit-coherence-invariants 950fbe5d-fdad-4862-b05d-8b374bdd5ec6
uv run rallycut audit-coherence-invariants b097dd2a-6953-4e0e-a603-5be3552f462e
uv run rallycut audit-coherence-invariants 5c756c41-1cc1-4486-a95c-97398912cfbe
```

Capture the C-1 / C-2 / C-3 violation counts per video. Compare to the 2026-05-10 fleet baseline portion attributable to these 3 videos (look up in the coherence-invariants memory entry — `coherence_invariants_v1_2026_05_10.md`).

**Pre-ship gate G-E:** C-2 violation count on each of the 3 videos ≤ its prior baseline.

- [ ] **Step 7: Final verdict on the 3 GT videos**

In `analysis/reports/attribution_baseline/team_chain_v1_2026_05_11.md`, fill in the final verdicts for G-A through G-E. If ALL pass, proceed to Task 6 (fleet deploy). If any regress, STOP and investigate.

**ROLLBACK** if needed:

```bash
# Restore the 3 videos' actions_json from snapshot
# (Manual: psql with UPDATE per row from the snapshot file. Out of scope
#  for this plan to script — the snapshot is the source-of-truth.)
```

If rollback happens, also set `RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN=0` in production env (or revert the code commit from Task 3).

- [ ] **Step 8: Commit the post-deploy verification**

```bash
git add analysis/reports/attribution_baseline/team_chain_v1_2026_05_11.md analysis/reports/attribution_baseline/db_snapshots/
git commit -m "$(cat <<'EOF'
report(attribution): team-chain v1 post-deploy verification on 3 GT videos

reattribute-actions re-run on cece/gigi/wawa with the new predicate
active. DB snapshot pre-deploy retained for rollback. All 5 pre-ship
gates documented.

Spec: docs/superpowers/specs/2026-05-11-action-attribution-team-chain-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Fleet deploy (~70 videos)

**Files:** None modified; only DB.

- [ ] **Step 1: List all fleet videos with tracked rallies**

Run:

```bash
PGPASSWORD=postgres psql -h localhost -p 5436 -U postgres -d rallycut -c "\
SELECT DISTINCT v.id, v.filename \
FROM videos v \
JOIN rallies r ON r.video_id = v.id \
JOIN player_tracks pt ON pt.rally_id = r.id \
WHERE v.match_analysis_json IS NOT NULL \
ORDER BY v.filename"
```

Expected: ~63-70 video IDs. Save the list to `/tmp/fleet_video_ids.txt` (one ID per line).

- [ ] **Step 2: Pre-deploy fleet C-2 baseline snapshot**

```bash
cd analysis
for vid in $(cat /tmp/fleet_video_ids.txt); do
  uv run rallycut audit-coherence-invariants "$vid" \
    >> /tmp/coherence_pre_team_chain_v1.log 2>&1
done
```

Expected: log file with per-video C-1/C-2/C-3 counts.

- [ ] **Step 3: Fleet deploy of reattribute-actions**

```bash
cd analysis
for vid in $(cat /tmp/fleet_video_ids.txt); do
  echo "=== $vid ==="
  RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN=1 uv run rallycut reattribute-actions "$vid" \
    2>&1 | tee -a /tmp/reattribute_team_chain_v1.log
done
```

Expected: ~70 video × ~1 min ≈ ~1 hour. Run with `run_in_background: true` if invoked through the agent.

STOP CONDITIONS during deploy: any video errors out, or `team_chain_override` log lines exceed ~30 per rally (would indicate the predicate is firing pathologically on that video's chain).

- [ ] **Step 4: Post-deploy fleet C-2 baseline**

```bash
cd analysis
for vid in $(cat /tmp/fleet_video_ids.txt); do
  uv run rallycut audit-coherence-invariants "$vid" \
    >> /tmp/coherence_post_team_chain_v1.log 2>&1
done
```

- [ ] **Step 5: Diff coherence counts pre vs post**

Compare `/tmp/coherence_pre_team_chain_v1.log` vs `/tmp/coherence_post_team_chain_v1.log`. The expectation is that C-2 (alternating-possession) violations should DROP — the new override fixes cross-team errors that are themselves the cause of many C-2 failures. C-1 and C-3 should be ~unchanged.

Write the deltas to `analysis/reports/attribution_baseline/team_chain_v1_2026_05_11.md` under "## Fleet deploy C-2 deltas".

Acceptable outcomes:
- C-2 drops fleet-wide: expected; the workstream's secondary signal.
- C-2 holds steady fleet-wide: acceptable; the fix is local to attribution.
- C-2 rises on any individual video by > 10%: investigate that video before declaring success.

- [ ] **Step 6: Run measure_pid_accuracy as a paranoia check**

```bash
cd analysis && uv run python scripts/measure_pid_accuracy.py
```

Expected: PERMUTED PID accuracy holds at or above 95.1% (the 2026-05-11 baseline). This is a sanity check — the new code doesn't touch tracking, so any drop > 0.5pp is suspicious.

- [ ] **Step 7: Commit the fleet-deploy report**

```bash
git add analysis/reports/attribution_baseline/team_chain_v1_2026_05_11.md
git commit -m "$(cat <<'EOF'
report(attribution): team-chain v1 fleet deploy complete (~70 videos)

reattribute-actions re-run fleet-wide with the new predicate active.
Coherence-invariant C-2 deltas captured pre/post. PERMUTED PID accuracy
held at >= 95.1%.

Spec: docs/superpowers/specs/2026-05-11-action-attribution-team-chain-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Memory entry + MEMORY.md index update

**Files:**
- Create: `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/attribution_team_chain_v1_2026_05_11.md`
- Modify: `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY.md`

- [ ] **Step 1: Write the memory entry**

Create `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/attribution_team_chain_v1_2026_05_11.md`:

```markdown
---
name: Attribution team-chain v1 2026-05-11
description: Relaxed the unconditional nearest-player guard in reattribute_players Pass 2. 4-gate trust-the-chain predicate (confidence, chain integrity, distance-cap, court_side corroborator). Default-on; env flag RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN for rollback.
type: project
---
# Attribution Team-Chain Override v1 — 2026-05-11

**Shipped:** 2026-05-11

**What it does:** `reattribute_players` Pass 2 in `analysis/rallycut/tracking/action_classifier.py` no longer unconditionally skips override when the wrong-team current attribution is the nearest candidate. Instead, the new `_team_chain_override_allowed` predicate must return True. Four gates:
- G1: `action.confidence >= 0.7`
- G2: `_chain_integrity` from seed serve unbroken (no UNKNOWN, no non-seed synthetic)
- G3: a candidate on the expected team within `1.5 ×` the current distance
- G4: `Contact.court_side` agrees with `expected_team` (soft-pass when "unknown")

**Why:** Pre-ship baseline on the 3 fresh-GT videos (cece/gigi/wawa, 136 GT actions): attribution accuracy 76% among matched; 16 cross-team errors (62% of wrong); per-action serve 47% / set 67% / dig 82% / attack 83% / receive 94%. The 2026-04-12 investigation pegged team assignment as the #1 attribution lever; this is the cleanest possible application of that lever now that team_assignments are reliably populated (post `redetect_all_actions` fix) and tracking PERMUTED PID 95.1%.

**Measured impact** (fill in after Task 4 / Task 5 complete):
- Combined `correct_rate`: 60.3% → __% (Δ +__pp)
- `wrong_cross_team`: 16 → __
- `wrong_same_team`: 7 → __ (non-increasing required)
- Fleet C-2 violations: 542 → __

**Env flag:** `RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN=0` restores the pre-2026-05-11 unconditional guard for fast rollback without redeploy.

**Spec:** `docs/superpowers/specs/2026-05-11-action-attribution-team-chain-design.md`
**Plan:** `docs/superpowers/plans/2026-05-11-action-attribution-team-chain.md`
**Commits:** (fill in after Task 7 — final 7 commits from this plan, sha range)
**Files:** `analysis/rallycut/tracking/action_classifier.py`, `analysis/tests/unit/test_action_attribution_team_chain.py`, `analysis/scripts/measure_attribution_fresh_gt.py`, `analysis/scripts/measure_attribution_team_chain_ab.py`

**Open follow-ups:**
- Same-team-as-server attribution (4 of 7 same-team errors). Different lever (pose / ball-toss).
- Synthetic-serve unattributed `pl_pid=-1` cases (3 errors). Belongs to synthetic-serve workstream.
- Architectural: decouple `action_ground_truth_json` from `PlayerTrack` so retracks don't wipe GT. Recommend a `RallyActionGroundTruth` table keyed on `rally_id`.
```

Fill in the "Measured impact" and "Commits" lines from the Task 4/5/6 results once available.

- [ ] **Step 2: Update MEMORY.md index**

Open `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY.md`. Under the "## Current workstreams" section, add as the top entry (most recent first):

```markdown
- [SHIPPED] [**Attribution team-chain v1 2026-05-11**](attribution_team_chain_v1_2026_05_11.md) — 4-gate trust-the-chain predicate in reattribute_players Pass 2 relaxes the nearest-player guard when chain is intact + court_side corroborates. Default-on; env flag RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN. Fleet correct_rate __% → __% (n=136 GT).
```

Keep the entry to ≤ 200 characters per the memory-hygiene feedback rule.

- [ ] **Step 3: Commit memory updates**

```bash
git add ~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/attribution_team_chain_v1_2026_05_11.md ~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY.md
# Note: these files live OUTSIDE the repo. They're written but NOT
# committed via git in the repo. Memory files are per-user, not part of
# the project. Skip the git commit; only the in-repo files were
# committed in Tasks 1-6.
```

(No commit — memory files live outside the repo. The previous steps in Tasks 1-6 are the only repo commits.)

- [ ] **Step 4: Final verification — `pre-commit` skill check**

Run the project's pre-commit skill checks:

```bash
cd analysis && uv run pytest tests/unit -v
uv run mypy rallycut/tracking/action_classifier.py
uv run ruff check rallycut/tracking/action_classifier.py tests/unit/test_action_attribution_team_chain.py
```

Expected: all green.

---

## Summary of touched files

In-repo changes (committed):
- `analysis/rallycut/tracking/action_classifier.py` — `+_chain_integrity`, `+_team_chain_override_allowed`, modified `reattribute_players` guard.
- `analysis/tests/unit/test_action_attribution_team_chain.py` — new test file.
- `analysis/scripts/measure_attribution_team_chain_ab.py` — new A/B harness.
- `analysis/scripts/measure_attribution_fresh_gt.py` — existing (committed in spec-commit).
- `analysis/reports/attribution_baseline/team_chain_v1_2026_05_11.md` — verdict report.
- `analysis/reports/attribution_baseline/team_chain_ab_2026_05_11.json` — raw A/B output.
- `analysis/reports/attribution_baseline/db_snapshots/pre_team_chain_2026_05_11.json` — rollback snapshot.

Out-of-repo (user memory, not committed):
- `~/.claude/projects/.../memory/attribution_team_chain_v1_2026_05_11.md`
- `~/.claude/projects/.../memory/MEMORY.md` — index entry.

## Rollback procedure (if Task 5 or Task 6 surfaces a regression)

1. Set env: `RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN=0` (in production env or shell that runs `reattribute-actions`).
2. Re-run `rallycut reattribute-actions <video>` on each regressed video — the env flag short-circuits the predicate, restoring prior behavior.
3. If the env-flag rollback is insufficient (e.g., the new behavior corrupted stored actions_json in a way the OFF path can't fix), restore from `db_snapshots/pre_team_chain_2026_05_11.json` via `psql` UPDATE statements.
4. If the underlying code change itself is suspect, `git revert <commit-from-task-3>` and redeploy.
