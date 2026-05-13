# Sub-2.B Coherence Repair — Phase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship Phase 1 of Sub-2.B — add the C-4 game-rule invariant (no-same-player-back-to-back, prev=block exempt) to the audit framework and produce a pre-scored evidence catalog of fleet C-4 violations that will drive Phase 2 repair-pass design at the gated review.

**Architecture:** Three components: (1) a new check function in `coherence_invariants.py` mirroring C-1/C-2/C-3's pattern, with a small optional-`payload` extension to the shared `Violation` dataclass for structured downstream consumption; (2) a catalog harness script that walks the fleet and writes per-violation rows with pre-scored evidence signals (action-type fit, team-geometry, alt-ratio, confidence, co-violations) plus a placeholder repair recommendation that the gated review will falsify; (3) a no-op `Pass 2d` stub in `reattribute_players` gated by env flag `COHERENCE_C4_REPAIR=0` so Phase 2 can fill in the rule body without touching the integration points again.

**Tech Stack:** Python 3.11, Typer CLI, frozen dataclasses, pytest, psycopg2, CSV stdlib. No new dependencies.

**Spec:** [docs/superpowers/specs/2026-05-13-sub-2b-coherence-repair-design.md](../specs/2026-05-13-sub-2b-coherence-repair-design.md)

---

## File Structure

**Modified files:**
- `analysis/rallycut/tracking/pid_invariants.py` — add optional `payload` field to `Violation` (no behavior change).
- `analysis/rallycut/tracking/coherence_invariants.py` — add `check_c4_no_same_player_back_to_back` + wire into `run_all`.
- `analysis/rallycut/tracking/action_classifier.py` — add `_coherence_c4_repair_pass` no-op stub + env-flag-gated call site between Pass 2c and Pass 3 in `reattribute_players`.
- `analysis/tests/unit/test_coherence_invariants.py` — add `TestC4SamePlayerBackToBack` class + extend `TestRunAll`.

**New files:**
- `analysis/scripts/catalog_c4_violations.py` — fleet-walking catalog harness; pure signal helpers + DB orchestration.
- `analysis/tests/unit/test_catalog_c4_violations.py` — tests for the pure signal helpers (type-fit, team-geometry, alt-ratio, placeholder recommendation).
- `analysis/tests/unit/test_action_classifier_pass2d_stub.py` — verifies the stub is a no-op at both flag settings.

**Output directory** (created at script run-time, gitignored content):
- `analysis/reports/coherence_c4_catalog/` — holds the CSV catalog + summary markdown produced by Task 7.

---

## Task 1: Extend `Violation` with optional `payload` field

**Files:**
- Modify: `analysis/rallycut/tracking/pid_invariants.py:30-35`
- Test: `analysis/tests/unit/test_pid_invariants.py` (existing tests must still pass)

- [ ] **Step 1: Run existing tests to capture green baseline**

Run: `cd analysis && uv run pytest tests/unit/test_pid_invariants.py tests/unit/test_coherence_invariants.py -v`
Expected: all green.

- [ ] **Step 2: Add `payload` field to `Violation`**

Edit `analysis/rallycut/tracking/pid_invariants.py`. Replace the existing `Violation` dataclass:

```python
@dataclass(frozen=True)
class Violation:
    invariant: str
    rally_id: str
    detail: str
    severity: Literal["error", "warn"] = "error"
    payload: dict[str, Any] | None = None
```

The `payload` default is `None`. Existing callers construct `Violation(invariant=..., rally_id=..., detail=...)` with no payload; that stays valid because the field is optional.

- [ ] **Step 3: Run existing tests to verify no regression**

Run: `cd analysis && uv run pytest tests/unit/test_pid_invariants.py tests/unit/test_coherence_invariants.py -v`
Expected: all green. No existing test references `payload`.

- [ ] **Step 4: Run type check**

Run: `cd analysis && uv run mypy rallycut/tracking/pid_invariants.py`
Expected: success (no new errors).

- [ ] **Step 5: Commit**

```bash
git add analysis/rallycut/tracking/pid_invariants.py
git commit -m "$(cat <<'EOF'
feat(violation): add optional payload field for structured downstream consumption

Lets coherence-invariant checks emit structured evidence (frame indices,
player ids, action types) without forcing every consumer to regex-parse
the detail string. Default None preserves byte-identical behavior for
all existing C-1/C-2/C-3 and I-1..I-8 producers.
EOF
)"
```

---

## Task 2: TDD C-4 detector — write failing tests

**Files:**
- Modify: `analysis/tests/unit/test_coherence_invariants.py` (add `TestC4SamePlayerBackToBack` class; extend imports)

- [ ] **Step 1: Add the failing test class**

Append to `analysis/tests/unit/test_coherence_invariants.py` (and extend the import at the top of the file to include `check_c4_no_same_player_back_to_back`):

```python
# Update the imports near the top of the file:
from rallycut.tracking.coherence_invariants import (
    check_c1_three_contact_rule,
    check_c2_alternating_possessions,
    check_c3_first_action_is_serve,
    check_c4_no_same_player_back_to_back,
    run_all,
)


class TestC4NoSamePlayerBackToBack:
    """C-4: consecutive actions must be by different players.

    Exception: prev action is 'block' (block→cover by same player is legal).
    """

    def test_different_players_passes(self) -> None:
        actions = [
            _action(100, "serve", 3),
            _action(140, "receive", 1),
            _action(170, "set", 2),
            _action(200, "attack", 1),
        ]
        team_assignments = {"1": "B", "2": "B", "3": "A", "4": "A"}
        result = check_c4_no_same_player_back_to_back(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        assert result == []

    def test_same_player_consecutive_fires(self) -> None:
        # Player 1 sets then attacks back-to-back — illegal (no block exception).
        actions = [
            _action(100, "serve", 3),    # A
            _action(140, "receive", 1),  # B
            _action(170, "set", 1),      # B (same player as prev)
            _action(200, "attack", 2),   # B
        ]
        team_assignments = {"1": "B", "2": "B", "3": "A", "4": "A"}
        result = check_c4_no_same_player_back_to_back(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        assert len(result) == 1
        v = result[0]
        assert v.invariant == "C-4"
        assert v.rally_id == "r1"
        assert v.payload is not None
        assert v.payload["prev_index"] == 1
        assert v.payload["curr_index"] == 2
        assert v.payload["prev_frame"] == 140
        assert v.payload["curr_frame"] == 170
        assert v.payload["prev_action"] == "receive"
        assert v.payload["curr_action"] == "set"
        assert v.payload["player_id"] == 1

    def test_block_exception_block_then_same_player(self) -> None:
        # Player 4 blocks, then player 4 sets the cover — exempt.
        actions = [
            _action(100, "serve", 3),    # A
            _action(140, "attack", 1),   # B (after receive/set elsewhere, simplified)
            _action(170, "block", 4),    # A blocks
            _action(200, "set", 4),      # A — same player as block → exempt
        ]
        team_assignments = {"1": "B", "2": "B", "3": "A", "4": "A"}
        result = check_c4_no_same_player_back_to_back(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        assert result == []

    def test_block_then_block_same_player_exempt(self) -> None:
        # block→block by same player: prev=block exempts the pair (strict reading).
        actions = [
            _action(100, "attack", 1),   # B
            _action(140, "block", 4),    # A
            _action(170, "block", 4),    # A — exempt because prev is block
        ]
        team_assignments = {"1": "B", "2": "B", "4": "A"}
        result = check_c4_no_same_player_back_to_back(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        assert result == []

    def test_set_then_block_same_player_fires(self) -> None:
        # curr is block but prev is NOT block → not exempt.
        actions = [
            _action(100, "serve", 3),    # A
            _action(140, "set", 4),      # A
            _action(170, "block", 4),    # A — prev=set, NOT exempt
        ]
        team_assignments = {"1": "B", "3": "A", "4": "A"}
        result = check_c4_no_same_player_back_to_back(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        assert len(result) == 1
        assert result[0].invariant == "C-4"
        assert result[0].payload is not None
        assert result[0].payload["prev_action"] == "set"
        assert result[0].payload["curr_action"] == "block"

    def test_missing_pid_skips(self) -> None:
        # action[1].playerTrackId == -1 → skip the (0,1) pair check.
        actions = [
            _action(100, "serve", 3),
            _action(140, "receive", -1),
            _action(170, "set", 3),  # would fire vs action[1] if not for the -1 skip
        ]
        team_assignments = {"3": "A"}
        result = check_c4_no_same_player_back_to_back(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        assert result == []

    def test_unmapped_pid_skips(self) -> None:
        # action[1].playerTrackId == 99 not in team_assignments → skip.
        actions = [
            _action(100, "serve", 3),
            _action(140, "receive", 99),
            _action(170, "set", 99),
        ]
        team_assignments = {"3": "A"}
        result = check_c4_no_same_player_back_to_back(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        assert result == []

    def test_cascade_three_same_player_fires_twice(self) -> None:
        # X → X → X with no block prev → pair (0,1) and pair (1,2) both fire.
        actions = [
            _action(100, "receive", 1),
            _action(140, "set", 1),
            _action(170, "attack", 1),
        ]
        team_assignments = {"1": "B"}
        result = check_c4_no_same_player_back_to_back(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        assert len(result) == 2
        assert all(v.invariant == "C-4" for v in result)
        assert result[0].payload["prev_index"] == 0
        assert result[0].payload["curr_index"] == 1
        assert result[1].payload["prev_index"] == 1
        assert result[1].payload["curr_index"] == 2

    def test_cascade_with_middle_block_one_violation(self) -> None:
        # X(non-block) → X(block) → X — only (0,1) fires; (1,2) exempt because prev=block.
        actions = [
            _action(100, "set", 4),
            _action(140, "block", 4),   # (0,1): prev=set, NOT exempt → fires
            _action(170, "dig", 4),     # (1,2): prev=block → exempt
        ]
        team_assignments = {"4": "A"}
        result = check_c4_no_same_player_back_to_back(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        assert len(result) == 1
        assert result[0].payload["prev_index"] == 0
        assert result[0].payload["curr_index"] == 1

    def test_zero_actions_skips(self) -> None:
        result = check_c4_no_same_player_back_to_back(
            rally_id="r1", actions=[], team_assignments={},
        )
        assert result == []

    def test_one_action_skips(self) -> None:
        actions = [_action(100, "serve", 1)]
        result = check_c4_no_same_player_back_to_back(
            rally_id="r1", actions=actions, team_assignments={"1": "A"},
        )
        assert result == []

    def test_defensive_sort_by_frame(self) -> None:
        # DB returns out-of-order actions; detector must sort first.
        actions = [
            _action(170, "set", 1),
            _action(140, "receive", 1),  # same player as the "later" action
            _action(100, "serve", 3),
        ]
        team_assignments = {"1": "B", "3": "A"}
        result = check_c4_no_same_player_back_to_back(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        assert len(result) == 1
        assert result[0].payload["prev_frame"] == 140
        assert result[0].payload["curr_frame"] == 170
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd analysis && uv run pytest tests/unit/test_coherence_invariants.py::TestC4NoSamePlayerBackToBack -v`
Expected: every test in the class fails with `ImportError` (the symbol `check_c4_no_same_player_back_to_back` doesn't exist yet).

- [ ] **Step 3: Commit the failing tests**

```bash
git add analysis/tests/unit/test_coherence_invariants.py
git commit -m "$(cat <<'EOF'
test(coherence): failing tests for C-4 no-same-player-back-to-back

12 tests covering: clean pass, basic violation, block-exception (strict prev=block),
block→block exempt, set→block fires, missing/-1 pid skip, unmapped pid skip,
cascade (X→X→X), cascade-with-middle-block, zero/one action skip, defensive sort.

Symbol does not yet exist; tests will pass after Task 3 implements the detector.
EOF
)"
```

---

## Task 3: Implement C-4 detector

**Files:**
- Modify: `analysis/rallycut/tracking/coherence_invariants.py` (add function below `check_c3_first_action_is_serve`)

- [ ] **Step 1: Add the detector function**

Edit `analysis/rallycut/tracking/coherence_invariants.py`. Insert *after* `check_c3_first_action_is_serve` and *before* `_UPSTREAM_BLOCKER_INVARIANTS`:

```python
def check_c4_no_same_player_back_to_back(
    *,
    rally_id: str,
    actions: list[dict[str, Any]],
    team_assignments: dict[str, str],
) -> list[Violation]:
    """C-4: consecutive actions must be by different players.

    Exception: when ``action[i-1].action == 'block'`` the pair is exempt —
    block→cover by the same player is legal (and so is the rarer
    block→block by the same player, since the prev is still block).

    Skip pair semantics: if either action's ``playerTrackId`` is missing,
    -1, or not in ``team_assignments``, the comparison is meaningless and
    the pair is skipped. This is the *audit* side, which surfaces all
    same-player pairs regardless of action confidence; the Phase 2 repair
    pass mirrors Pass 2c's ``confidence < 0.3`` floor instead.
    """
    if len(actions) < 2:
        return []
    sorted_actions = _actions_sorted_by_frame(actions)

    violations: list[Violation] = []
    for i in range(1, len(sorted_actions)):
        prev = sorted_actions[i - 1]
        curr = sorted_actions[i]
        prev_pid = prev.get("playerTrackId")
        curr_pid = curr.get("playerTrackId")

        if prev_pid is None or curr_pid is None:
            continue
        if prev_pid == -1 or curr_pid == -1:
            continue
        if str(prev_pid) not in team_assignments:
            continue
        if str(curr_pid) not in team_assignments:
            continue

        prev_action = str(prev.get("action", ""))
        if prev_action == "block":
            continue  # strict block exception

        if prev_pid != curr_pid:
            continue

        curr_action = str(curr.get("action", ""))
        violations.append(
            Violation(
                invariant="C-4",
                rally_id=rally_id,
                detail=(
                    f"action[{i - 1}] (frame {prev.get('frame')}, "
                    f"{prev_action}, player {prev_pid}) and "
                    f"action[{i}] (frame {curr.get('frame')}, "
                    f"{curr_action}, player {curr_pid}) "
                    f"attributed to same player; max is 1 unless prev is block"
                ),
                payload={
                    "prev_index": i - 1,
                    "curr_index": i,
                    "prev_frame": int(prev.get("frame", 0)),
                    "curr_frame": int(curr.get("frame", 0)),
                    "prev_action": prev_action,
                    "curr_action": curr_action,
                    "player_id": int(prev_pid),
                },
            )
        )
    return violations
```

- [ ] **Step 2: Run the C-4 tests to verify they pass**

Run: `cd analysis && uv run pytest tests/unit/test_coherence_invariants.py::TestC4NoSamePlayerBackToBack -v`
Expected: all 12 tests pass.

- [ ] **Step 3: Run the rest of the file's tests to verify no regression**

Run: `cd analysis && uv run pytest tests/unit/test_coherence_invariants.py -v`
Expected: all existing C-1/C-2/C-3 tests still pass alongside the new C-4 tests.

- [ ] **Step 4: Type check**

Run: `cd analysis && uv run mypy rallycut/tracking/coherence_invariants.py`
Expected: success.

- [ ] **Step 5: Commit**

```bash
git add analysis/rallycut/tracking/coherence_invariants.py
git commit -m "$(cat <<'EOF'
feat(coherence): C-4 no-same-player-back-to-back invariant

Adds the rule (and structured payload via Violation.payload) but does NOT
yet wire it into run_all. Skip rules: missing/-1 player_track_id, unmapped
in team_assignments, strict prev=block exception. Audit surfaces all pairs
regardless of action.confidence — the 0.3 floor lives in the (gated)
Phase 2 repair pass, not in the detector.

Spec: docs/superpowers/specs/2026-05-13-sub-2b-coherence-repair-design.md
EOF
)"
```

---

## Task 4: Wire C-4 into `run_all`

**Files:**
- Modify: `analysis/rallycut/tracking/coherence_invariants.py:200-262` (the `run_all` body)
- Modify: `analysis/tests/unit/test_coherence_invariants.py` (extend `TestRunAll`)

- [ ] **Step 1: Add failing assertion to `TestRunAll`**

Find the existing `TestRunAll` class in `analysis/tests/unit/test_coherence_invariants.py`. Add this test method (alongside existing run_all tests):

```python
    def test_run_all_dispatches_c4(self) -> None:
        # A single rally with a same-player back-to-back pair (no block prev)
        # should produce at least one C-4 violation from run_all.
        actions = [
            {"frame": 100, "action": "receive", "playerTrackId": 1},
            {"frame": 140, "action": "set", "playerTrackId": 1},  # C-4 pair
        ]
        team_assignments = {"1": "B"}
        actions_json = {"actions": actions, "teamAssignments": team_assignments}

        mock_conn = MagicMock()
        mock_cur = MagicMock()
        mock_cur.fetchall.return_value = [("rally_xyz", actions_json)]
        mock_conn.cursor.return_value.__enter__.return_value = mock_cur

        with (
            patch(
                "rallycut.tracking.coherence_invariants.get_connection"
            ) as mock_get_conn,
            patch(
                "rallycut.tracking.coherence_invariants.pid_run_all",
                return_value=[],
            ),
        ):
            mock_get_conn.return_value.__enter__.return_value = mock_conn
            violations = run_all(video_id="vid_abc")

        c4 = [v for v in violations if v.invariant == "C-4"]
        assert len(c4) == 1, f"expected one C-4 from run_all, got {c4!r}"
        assert c4[0].rally_id == "rally_xyz"
```

- [ ] **Step 2: Run the new test to verify it fails**

Run: `cd analysis && uv run pytest tests/unit/test_coherence_invariants.py::TestRunAll::test_run_all_dispatches_c4 -v`
Expected: FAIL — assertion `len(c4) == 1` fails because `run_all` doesn't dispatch C-4 yet.

- [ ] **Step 3: Wire C-4 into `run_all`**

Edit `analysis/rallycut/tracking/coherence_invariants.py`. In the `run_all` function, find the per-rally dispatch block (the three `violations.extend(...)` calls inside the `for row in rally_rows:` loop) and add a fourth call right after the C-3 dispatch:

```python
        violations.extend(
            check_c3_first_action_is_serve(
                rally_id=rally_id, actions=actions,
            )
        )
        violations.extend(
            check_c4_no_same_player_back_to_back(
                rally_id=rally_id, actions=actions,
                team_assignments=team_assignments,
            )
        )
```

Also update the module docstring at the top of the file to mention C-4. Find the `Rules:` block in the existing module docstring and add a 4th line:

```python
"""...
Rules:
  C-1: A team can have at most 3 consecutive contacts before the ball
       must cross to the other team.
  C-2: Possessions alternate teams.
  C-3: First action of a rally is `serve`.
  C-4: Consecutive actions must be by different players (exception: prev
       action is `block`).

Skip semantics: ..."""
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `cd analysis && uv run pytest tests/unit/test_coherence_invariants.py::TestRunAll::test_run_all_dispatches_c4 -v`
Expected: PASS.

- [ ] **Step 5: Run the full test module to verify no regression**

Run: `cd analysis && uv run pytest tests/unit/test_coherence_invariants.py -v`
Expected: all green (C-1/C-2/C-3/C-4 + run_all suite).

- [ ] **Step 6: Smoke-test the CLI on a known video**

Pick any video_id from a recent fleet run, e.g.:

```bash
cd analysis && uv run rallycut audit-coherence-invariants <video-id-here>
```

Expected: command runs successfully and the violation table now potentially includes C-4 rows alongside C-1/C-2/C-3. Exit code 0 if no errors, 1 if any. (Don't be alarmed if C-4 fires a lot — that's the whole reason we're building the catalog.)

- [ ] **Step 7: Commit**

```bash
git add analysis/rallycut/tracking/coherence_invariants.py analysis/tests/unit/test_coherence_invariants.py
git commit -m "$(cat <<'EOF'
feat(coherence): wire C-4 into audit-coherence-invariants run_all

C-4 now ships in the audit CLI alongside C-1/C-2/C-3. Inherits the
upstream-PID-skip gate (I-1/I-3/I-6 still exclude rallies). Single-rally
dispatch only. The fleet baseline will be measured by the catalog
harness in Task 7.
EOF
)"
```

---

## Task 5: Catalog signal helpers (pure functions)

**Files:**
- Create: `analysis/scripts/catalog_c4_violations.py` (initial: only pure helpers + the `EXPECTED_TRANSITIONS` constant)
- Create: `analysis/tests/unit/test_catalog_c4_violations.py`

- [ ] **Step 1: Write failing tests for the pure helpers**

Create `analysis/tests/unit/test_catalog_c4_violations.py`:

```python
"""Tests for the pure signal-computation helpers in catalog_c4_violations.

The DB-read + CSV-write orchestration in that script is integration-only
(verified by Task 7's smoke run); the pure helpers below are unit-tested.
"""

from __future__ import annotations

import math

import pytest

from scripts.catalog_c4_violations import (
    EXPECTED_TRANSITIONS,
    best_same_team_alt_ratio,
    placeholder_repair_recommendation,
    signal_team_geometry,
    signal_type_fit,
)


class TestSignalTypeFit:
    def test_known_ok_pair(self) -> None:
        # Serve → receive is in the table with label 'ok'.
        assert signal_type_fit("serve", "receive") == "ok"

    def test_known_wrong_pair(self) -> None:
        # Serve → set is in the table with label 'wrong'.
        assert signal_type_fit("serve", "set") == "wrong"

    def test_unknown_pair_returns_unknown(self) -> None:
        assert signal_type_fit("attack", "set") == "unknown"  # not in table

    def test_block_prev_always_ok(self) -> None:
        # block → anything is 'ok' (C-4 detector exempts the pair anyway,
        # but the type-fit signal must agree, not return 'wrong').
        assert signal_type_fit("block", "receive") == "ok"
        assert signal_type_fit("block", "set") == "ok"
        assert signal_type_fit("block", "attack") == "ok"  # fallback for any block→X

    def test_expected_transitions_constant_has_required_keys(self) -> None:
        # Sanity: the table must include the volleyball happy-path
        # transitions that the spec's expected-transitions section listed.
        required = [
            ("serve", "receive"),
            ("serve", "dig"),
            ("receive", "set"),
            ("set", "attack"),
            ("attack", "dig"),
            ("attack", "block"),
            ("dig", "set"),
        ]
        for key in required:
            assert key in EXPECTED_TRANSITIONS, f"missing transition {key}"


class TestBestSameTeamAltRatio:
    def test_no_same_team_alt_returns_nan(self) -> None:
        # Only opposite-team candidates → no alt → NaN.
        candidates = [(101, 0.50, "B"), (102, 0.55, "B")]
        ratio = best_same_team_alt_ratio(
            candidates=candidates,
            current_dist=0.45,
            current_team="A",
            current_tid=100,
        )
        assert math.isnan(ratio)

    def test_same_team_closer_returns_ratio_below_one(self) -> None:
        # Same-team candidate at 0.20 vs current at 0.40 → ratio 0.5.
        candidates = [(200, 0.20, "A"), (201, 0.30, "B")]
        ratio = best_same_team_alt_ratio(
            candidates=candidates,
            current_dist=0.40,
            current_team="A",
            current_tid=100,
        )
        assert ratio == pytest.approx(0.5)

    def test_same_team_farther_returns_ratio_above_one(self) -> None:
        candidates = [(200, 0.80, "A"), (201, 0.30, "B")]
        ratio = best_same_team_alt_ratio(
            candidates=candidates,
            current_dist=0.40,
            current_team="A",
            current_tid=100,
        )
        assert ratio == pytest.approx(2.0)

    def test_excludes_current_tid(self) -> None:
        # The current player is in the candidate list (rank 1) but must be
        # excluded from the "alternative" search.
        candidates = [(100, 0.20, "A"), (200, 0.50, "A")]
        ratio = best_same_team_alt_ratio(
            candidates=candidates,
            current_dist=0.20,
            current_team="A",
            current_tid=100,
        )
        # Alt is tid=200 at 0.50, ratio = 0.50/0.20 = 2.5.
        assert ratio == pytest.approx(2.5)


class TestSignalTeamGeometry:
    def test_rank1_on_expected_team_matches(self) -> None:
        candidates = [(101, 0.30, "A"), (200, 0.50, "B")]
        result = signal_team_geometry(
            candidates=candidates, expected_team="A",
        )
        assert result == "matches"

    def test_rank1_wrong_team_rank2_within_2x_violates(self) -> None:
        # Rank1 is on team B (wrong), rank2 is team A at 0.50/0.30 = 1.67x → violates.
        candidates = [(101, 0.30, "B"), (200, 0.50, "A")]
        result = signal_team_geometry(
            candidates=candidates, expected_team="A",
        )
        assert result == "violates"

    def test_rank1_wrong_team_rank2_beyond_2x_ambiguous(self) -> None:
        # Rank1 team B at 0.20, rank2 team A at 0.50 → 2.5x → outside 2x → ambiguous.
        candidates = [(101, 0.20, "B"), (200, 0.50, "A")]
        result = signal_team_geometry(
            candidates=candidates, expected_team="A",
        )
        assert result == "ambiguous"

    def test_empty_candidates_returns_ambiguous(self) -> None:
        assert signal_team_geometry(candidates=[], expected_team="A") == "ambiguous"

    def test_no_expected_team_returns_ambiguous(self) -> None:
        candidates = [(101, 0.30, "A")]
        assert signal_team_geometry(candidates=candidates, expected_team=None) == "ambiguous"


class TestPlaceholderRepairRecommendation:
    def _row(self, **overrides: object) -> dict[str, object]:
        defaults: dict[str, object] = {
            "signal_type_fit_prev": "ok",
            "signal_type_fit_curr": "ok",
            "signal_team_geometry_prev": "matches",
            "signal_team_geometry_curr": "matches",
            "prev_best_same_team_alt_ratio": float("nan"),
            "curr_best_same_team_alt_ratio": float("nan"),
            "conf_prev": 0.9,
            "conf_curr": 0.9,
        }
        defaults.update(overrides)
        return defaults

    def test_all_signals_ok_recommends_skip(self) -> None:
        row = self._row()
        assert placeholder_repair_recommendation(row) == "skip"

    def test_two_strong_against_curr_recommends_repair_curr(self) -> None:
        row = self._row(
            signal_type_fit_curr="wrong",
            signal_team_geometry_curr="violates",
        )
        assert placeholder_repair_recommendation(row) == "repair_curr"

    def test_two_strong_against_prev_recommends_repair_prev(self) -> None:
        row = self._row(
            signal_type_fit_prev="wrong",
            conf_prev=0.3,
        )
        assert placeholder_repair_recommendation(row) == "repair_prev"

    def test_both_strong_returns_ambiguous(self) -> None:
        row = self._row(
            signal_type_fit_prev="wrong",
            signal_team_geometry_prev="violates",
            signal_type_fit_curr="wrong",
            signal_team_geometry_curr="violates",
        )
        assert placeholder_repair_recommendation(row) == "ambiguous"

    def test_alt_ratio_under_06_counts_against_side(self) -> None:
        # Curr has a same-team alt at 0.5x current — that's one strike. Plus
        # type_fit=wrong → 2 strong, recommend repair_curr.
        row = self._row(
            signal_type_fit_curr="wrong",
            curr_best_same_team_alt_ratio=0.4,
        )
        assert placeholder_repair_recommendation(row) == "repair_curr"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `cd analysis && uv run pytest tests/unit/test_catalog_c4_violations.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'scripts.catalog_c4_violations'` (or similar) — the script doesn't exist yet.

- [ ] **Step 3: Create the script with pure helpers**

Create `analysis/scripts/catalog_c4_violations.py`:

```python
"""Catalog harness for C-4 (no-same-player-back-to-back) coherence violations.

Walks the labeled fleet, runs `coherence_invariants.run_all` per video,
filters to C-4 violations, computes per-pair evidence signals, writes a
CSV row + a fleet-aggregate summary markdown. The CSV is the input to the
Phase 1 → Phase 2 gated review (see spec).

Pure signal-computation helpers live at module top so they're unit-testable
without DB access. The DB orchestration is in `main`.

Spec: docs/superpowers/specs/2026-05-13-sub-2b-coherence-repair-design.md
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Expected-transitions table for signal_type_fit. Conservative: only known
# pairs are labeled; unknown pairs return 'unknown' (NOT 'wrong').
# ---------------------------------------------------------------------------
EXPECTED_TRANSITIONS: dict[tuple[str, str], tuple[str | None, str]] = {
    # (prev_type, curr_type): (expected_curr_team_relation, label)
    ("serve",   "receive"): ("other",  "ok"),
    ("serve",   "dig"):     ("other",  "ok"),
    ("serve",   "set"):     (None,     "wrong"),
    ("serve",   "attack"):  (None,     "wrong"),
    ("serve",   "block"):   ("other",  "ok"),
    ("receive", "set"):     ("same",   "ok"),
    ("receive", "attack"):  ("same",   "ok"),
    ("set",     "attack"):  ("same",   "ok"),
    ("set",     "set"):     (None,     "wrong"),
    ("attack",  "dig"):     ("other",  "ok"),
    ("attack",  "block"):   ("other",  "ok"),
    ("attack",  "receive"): ("other",  "ok"),
    ("dig",     "set"):     ("same",   "ok"),
    ("dig",     "attack"):  ("same",   "ok"),
}
# Block-prev fallback: any block→X is 'ok' (block exempts the C-4 pair).
_BLOCK_PREV_FALLBACK_LABEL = "ok"


def signal_type_fit(prev_action: str, curr_action: str) -> str:
    """Return 'ok' | 'wrong' | 'unknown' for the action-type transition.

    Block-prev pairs are always 'ok' (the C-4 detector exempts them).
    """
    if prev_action == "block":
        return _BLOCK_PREV_FALLBACK_LABEL
    entry = EXPECTED_TRANSITIONS.get((prev_action, curr_action))
    if entry is None:
        return "unknown"
    _, label = entry
    return label


def best_same_team_alt_ratio(
    *,
    candidates: list[tuple[int, float, str]],
    current_dist: float,
    current_team: str,
    current_tid: int,
) -> float:
    """Return best same-team alternative's distance / current_dist.

    < 1.0 means a same-team alternative is closer than the current player.
    NaN if no same-team alternative exists or current_dist is non-finite.
    """
    if not math.isfinite(current_dist) or current_dist <= 0:
        return math.nan
    best: float = math.inf
    for tid, dist, team in candidates:
        if tid == current_tid:
            continue
        if team != current_team:
            continue
        if dist < best:
            best = dist
    if not math.isfinite(best):
        return math.nan
    return best / current_dist


def signal_team_geometry(
    *,
    candidates: list[tuple[int, float, str]],
    expected_team: str | None,
) -> str:
    """Return 'matches' | 'violates' | 'ambiguous'.

    'matches' if rank-1 candidate is on the expected team.
    'violates' if rank-1 is wrong team but a rank-2+ same-team candidate
    is within 2x distance of rank-1.
    'ambiguous' otherwise (no candidates, no expected team, no nearby
    same-team alternative).
    """
    if expected_team is None or not candidates:
        return "ambiguous"
    rank1_tid, rank1_dist, rank1_team = candidates[0]
    if rank1_team == expected_team:
        return "matches"
    # rank1 is wrong team — does a same-team candidate sit within 2x?
    if not math.isfinite(rank1_dist) or rank1_dist <= 0:
        return "ambiguous"
    threshold = 2.0 * rank1_dist
    for tid, dist, team in candidates[1:]:
        if team == expected_team and math.isfinite(dist) and dist <= threshold:
            return "violates"
    return "ambiguous"


def placeholder_repair_recommendation(row: dict[str, Any]) -> str:
    """Hypothesis-only rule scoring evidence on each side of the pair.

    Counts "strong-against" signals per side: type_fit=='wrong',
    team_geometry=='violates', alt_ratio < 0.6 (closer same-team alt
    exists), confidence < 0.5. If one side has ≥2 strong-against and the
    other has ≤1, recommend repairing that side. If both have 0,
    recommend 'skip'. Otherwise 'ambiguous'.

    Not shipped to production. Exists so the gated review can falsify it
    against hand-classified root_cause labels.
    """
    def strong_against(prefix: str) -> int:
        count = 0
        if row.get(f"signal_type_fit_{prefix}") == "wrong":
            count += 1
        if row.get(f"signal_team_geometry_{prefix}") == "violates":
            count += 1
        alt = row.get(f"{prefix}_best_same_team_alt_ratio")
        if isinstance(alt, (int, float)) and math.isfinite(alt) and alt < 0.6:
            count += 1
        conf = row.get(f"conf_{prefix}")
        if isinstance(conf, (int, float)) and conf < 0.5:
            count += 1
        return count

    prev_count = strong_against("prev")
    curr_count = strong_against("curr")

    if prev_count >= 2 and curr_count <= 1:
        return "repair_prev"
    if curr_count >= 2 and prev_count <= 1:
        return "repair_curr"
    if prev_count == 0 and curr_count == 0:
        return "skip"
    return "ambiguous"


# ---------------------------------------------------------------------------
# DB orchestration lands in Task 7. Stub `main` for now so the module is
# importable but cannot be run as a script yet.
# ---------------------------------------------------------------------------
def main() -> int:
    print("catalog_c4_violations: DB orchestration not yet implemented (Task 7).",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd analysis && uv run pytest tests/unit/test_catalog_c4_violations.py -v`
Expected: all 14 tests pass.

- [ ] **Step 5: Type check**

Run: `cd analysis && uv run mypy scripts/catalog_c4_violations.py`
Expected: success.

- [ ] **Step 6: Commit**

```bash
git add analysis/scripts/catalog_c4_violations.py analysis/tests/unit/test_catalog_c4_violations.py
git commit -m "$(cat <<'EOF'
feat(catalog): C-4 pattern-catalog signal helpers (pure functions)

Three pure helpers + a placeholder repair-recommendation rule, fully
unit-tested without DB access:
  - signal_type_fit(prev, curr) → ok | wrong | unknown
  - best_same_team_alt_ratio(...) → float (NaN if no alt)
  - signal_team_geometry(...) → matches | violates | ambiguous
  - placeholder_repair_recommendation(row) → skip | repair_prev |
    repair_curr | ambiguous

The placeholder rule is a hypothesis the gated review will falsify
against hand-classified root_cause labels. DB orchestration lands in
the next task.

Spec: docs/superpowers/specs/2026-05-13-sub-2b-coherence-repair-design.md
EOF
)"
```

---

## Task 6: Pass 2d stub in `action_classifier.py`

**Files:**
- Modify: `analysis/rallycut/tracking/action_classifier.py` (add `_coherence_c4_repair_pass` and the env-flag-gated call site after Pass 2c)
- Create: `analysis/tests/unit/test_action_classifier_pass2d_stub.py`

The stub lands here (before the catalog orchestrator) so the Phase 1 plumbing is fully in place when the catalog runs — making Task 7's smoke run double as a "stub doesn't break the pipeline" verification.

- [ ] **Step 1: Write failing tests for the stub**

Create `analysis/tests/unit/test_action_classifier_pass2d_stub.py`:

```python
"""Pass 2d stub no-op tests.

Phase 1 ships only the env-flag-gated stub. Verifies that at both flag
settings (default OFF and explicitly ON) the function is byte-identical
to a no-op — the actual repair body lands in Phase 2.
"""

from __future__ import annotations

import os
from unittest.mock import patch

from rallycut.tracking.action_classifier import _coherence_c4_repair_pass


class TestCoherenceC4RepairPassStub:
    def test_returns_zero_on_empty_inputs(self) -> None:
        n = _coherence_c4_repair_pass(
            actions=[],
            contact_by_frame={},
            team_assignments={},
            chain_integrity=[],
            expected_teams=[],
        )
        assert n == 0

    def test_returns_zero_on_nonempty_inputs(self) -> None:
        # Stub MUST NOT mutate actions or do anything observable — Phase 2
        # fills in the body. This test exists to lock that contract.
        from rallycut.tracking.action_classifier import (
            ActionType,
            ClassifiedAction,
        )

        actions = [
            ClassifiedAction(
                frame=100, action_type=ActionType.SET, confidence=0.9,
                player_track_id=5,
            ),
            ClassifiedAction(
                frame=140, action_type=ActionType.ATTACK, confidence=0.9,
                player_track_id=5,
            ),
        ]
        n = _coherence_c4_repair_pass(
            actions=actions,
            contact_by_frame={},
            team_assignments={5: 0},
            chain_integrity=[True, True],
            expected_teams=[0, 0],
        )
        assert n == 0
        # Actions must be byte-identical (no mutations).
        assert actions[0].player_track_id == 5
        assert actions[1].player_track_id == 5


class TestCoherenceC4RepairFlagDefaultOff:
    """The env flag default-OFF must keep reattribute_players behavior
    byte-identical to pre-workstream. We don't run reattribute_players
    end-to-end here; we just verify the flag check reads correctly."""

    def test_flag_default_is_off(self) -> None:
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("COHERENCE_C4_REPAIR", None)
            assert os.environ.get("COHERENCE_C4_REPAIR", "0") == "0"

    def test_flag_explicit_one_reads_one(self) -> None:
        with patch.dict(os.environ, {"COHERENCE_C4_REPAIR": "1"}):
            assert os.environ.get("COHERENCE_C4_REPAIR", "0") == "1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd analysis && uv run pytest tests/unit/test_action_classifier_pass2d_stub.py -v`
Expected: FAIL with `ImportError: cannot import name '_coherence_c4_repair_pass'`.

- [ ] **Step 3: Add the stub + call site to `action_classifier.py`**

Edit `analysis/rallycut/tracking/action_classifier.py`. Find `reattribute_players` (around line 2949). Locate the end of the Pass 2c block (around line 3184, just before the `# Pass 3: ReID re-attribution` comment). Insert this new block BETWEEN Pass 2c and Pass 3:

```python
    # Pass 2d (Sub-2.B, gated, default-OFF): C-4 same-player-back-to-back
    # repair. Multi-signal evidence-based. Exact rule designed at Phase 1→2
    # gate from the violation-pattern catalog
    # (analysis/reports/coherence_c4_catalog/). Flip default-ON only after
    # A/B passes on the 22-rally panel.
    # Spec: docs/superpowers/specs/2026-05-13-sub-2b-coherence-repair-design.md
    if os.environ.get("COHERENCE_C4_REPAIR", "0") == "1":
        n_c4_repairs = _coherence_c4_repair_pass(
            actions=actions,
            contact_by_frame=contact_by_frame,
            team_assignments=team_assignments,
            chain_integrity=chain_integrity,
            expected_teams=expected_teams,
        )
        if n_c4_repairs > 0:
            logger.info(
                "C-4 repair: re-attributed %d/%d actions",
                n_c4_repairs, len(actions),
            )
```

Then add the helper function at module scope. Insert it AFTER `reattribute_players` ends (right before `def correct_team_from_propagation` around line 3193):

```python
def _coherence_c4_repair_pass(
    *,
    actions: list["ClassifiedAction"],
    contact_by_frame: dict[int, "Contact"],
    team_assignments: dict[int, int] | None,
    chain_integrity: list[bool],
    expected_teams: list[int | None],
) -> int:
    """Phase 1: no-op stub. Returns 0.

    Phase 2 (gated on Phase 1 pattern-catalog review) fills in the body
    using the multi-signal evidence rule designed from real fleet
    patterns. Constraints the eventual rule must respect are documented
    in the spec (section 'Phase 2 detail (placeholder)'):
      - skip when action.confidence < 0.3 (mirrors Pass 2c)
      - skip when action.player_track_id < 0 (mirrors Pass 2c)
      - strict prev=block exception (matches C-4 detector)
      - distance cap candidate.dist <= 2.0 * current_player.dist
      - upstream-error skip: skip when this rally fires C-3 or when
        either action has is_synthetic=True
      - multi-signal convergence (>=2 of type_fit / team_geometry /
        alt_ratio / confidence agree)
      - default-OFF until A/B passes on the 22-rally panel
      - forward iteration order; fixed-point pass (re-run is no-op)

    Spec: docs/superpowers/specs/2026-05-13-sub-2b-coherence-repair-design.md
    """
    return 0
```

- [ ] **Step 4: Run the new test file to verify it passes**

Run: `cd analysis && uv run pytest tests/unit/test_action_classifier_pass2d_stub.py -v`
Expected: all tests pass.

- [ ] **Step 5: Run the existing action_classifier tests to verify no regression**

Run: `cd analysis && uv run pytest tests/unit/test_action_classifier.py -v`
Expected: all green. The default-OFF flag means production behavior is byte-identical to pre-workstream.

- [ ] **Step 6: Type check**

Run: `cd analysis && uv run mypy rallycut/tracking/action_classifier.py`
Expected: success (or no new errors beyond any pre-existing ones).

- [ ] **Step 7: Commit**

```bash
git add analysis/rallycut/tracking/action_classifier.py analysis/tests/unit/test_action_classifier_pass2d_stub.py
git commit -m "$(cat <<'EOF'
feat(action_classifier): Pass 2d C-4 repair stub (default-OFF)

Adds the integration point for the Phase 2 C-4 repair pass in
reattribute_players, gated by COHERENCE_C4_REPAIR=0 default. Stub body
returns 0 — Phase 2 fills in the multi-signal rule after Phase 1's
catalog review.

Production behavior is byte-identical to pre-workstream at default-OFF.
At COHERENCE_C4_REPAIR=1 the stub still returns 0, so flipping the flag
during Phase 1 is also a no-op.

Spec: docs/superpowers/specs/2026-05-13-sub-2b-coherence-repair-design.md
EOF
)"
```

---

## Task 7: Catalog DB orchestration — fleet sweep, CSV + summary markdown

**Files:**
- Modify: `analysis/scripts/catalog_c4_violations.py` (replace stub `main` with real orchestrator)

The orchestration is integration code (talks to Postgres, walks the fleet). Pure helpers are already TDD'd in Task 5. The smoke verification is running it end-to-end on the panel.

- [ ] **Step 1: Confirm DB connectivity from the analysis package**

Run: `cd analysis && uv run python -c "from rallycut.evaluation.tracking.db import get_connection; conn = get_connection().__enter__(); print('OK', conn)"`
Expected: prints `OK <connection-object>`. If the connection fails, fix env vars before proceeding.

- [ ] **Step 2: Identify the fleet video IDs to walk**

Run: `cd analysis && uv run python -c "
from rallycut.evaluation.tracking.db import get_connection
with get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute('SELECT id FROM videos WHERE id IN (SELECT DISTINCT video_id FROM rallies WHERE status = %s OR status IS NULL) ORDER BY id', ['CONFIRMED'])
        rows = cur.fetchall()
print(f'{len(rows)} videos with confirmed-or-null rallies')
"`
Expected: prints the fleet count (around 70 videos per the Sub-2.A baseline).

- [ ] **Step 3: Replace the stub `main` with the orchestrator**

In `analysis/scripts/catalog_c4_violations.py`, replace the stub `def main()` and the `if __name__ == "__main__":` block with this full orchestrator. (Imports at the top of the file already have `argparse`, `csv`, `json`, `math`, `sys`, `Path`, `Counter`, `Any` — add additional imports listed below the orchestrator's top docstring as needed.)

Add these additional imports near the top of the file (after the existing imports):

```python
from rallycut.evaluation.tracking.db import get_connection
from rallycut.tracking.coherence_invariants import run_all as coherence_run_all
from rallycut.tracking.pid_invariants import run_all as pid_run_all
```

Then add the orchestrator and the rewritten `main`:

```python
CSV_COLUMNS = [
    "rally_id", "video_id", "pair_idx",
    "frame_prev", "frame_curr",
    "action_prev_type", "action_curr_type",
    "player_id", "team_label",
    "conf_prev", "conf_curr",
    "prev_player_dist", "curr_player_dist",
    "prev_top3_candidates", "curr_top3_candidates",
    "prev_best_same_team_alt_ratio", "curr_best_same_team_alt_ratio",
    "signal_type_fit_prev", "signal_type_fit_curr",
    "signal_team_geometry_prev", "signal_team_geometry_curr",
    "co_c1_fires", "co_c2_fires", "co_c3_fires",
    "co_pid_invariant_fires",
    "repair_recommendation",
    "root_cause", "notes",
]


def _load_video_ids() -> list[str]:
    """All videos that have at least one CONFIRMED-or-NULL rally."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT DISTINCT video_id FROM rallies "
                "WHERE status = %s OR status IS NULL "
                "ORDER BY video_id",
                ["CONFIRMED"],
            )
            return [row[0] for row in cur.fetchall()]


def _load_rally_payloads(
    video_id: str,
) -> dict[str, dict[str, Any]]:
    """For each rally in the video, load actions_json + contacts_json.

    Returns {rally_id: {"actions": ..., "team_assignments": ..., "contacts": ...}}.
    """
    query = """
        SELECT
            r.id AS rally_id,
            pt.actions_json,
            pt.contacts_json
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE r.video_id = %s
          AND (r.status = 'CONFIRMED' OR r.status IS NULL)
        ORDER BY r.start_ms
    """
    out: dict[str, dict[str, Any]] = {}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, [video_id])
            for rally_id, actions_json, contacts_json in cur.fetchall():
                if not isinstance(actions_json, dict):
                    continue
                actions = actions_json.get("actions") or []
                team_assignments = actions_json.get("teamAssignments") or {}
                contacts: list[dict[str, Any]] = []
                if isinstance(contacts_json, dict):
                    contacts = contacts_json.get("contacts") or []
                elif isinstance(contacts_json, list):
                    contacts = contacts_json
                out[rally_id] = {
                    "actions": actions,
                    "team_assignments": team_assignments,
                    "contacts": contacts,
                }
    return out


def _contact_by_frame(
    contacts: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    """Index contacts by frame for O(1) lookup."""
    by_frame: dict[int, dict[str, Any]] = {}
    for c in contacts:
        frame = c.get("frame")
        if isinstance(frame, int):
            by_frame[frame] = c
    return by_frame


def _extract_top3_candidates(
    contact: dict[str, Any] | None,
    team_assignments: dict[str, str],
) -> list[tuple[int, float, str]]:
    """Return up to 3 (tid, dist, team_label) tuples from contact.player_candidates."""
    if contact is None:
        return []
    raw = contact.get("player_candidates") or contact.get("playerCandidates") or []
    out: list[tuple[int, float, str]] = []
    for entry in raw[:3]:
        if isinstance(entry, dict):
            tid = entry.get("track_id") or entry.get("trackId") or entry.get("tid")
            dist = entry.get("distance") or entry.get("dist")
        elif isinstance(entry, (list, tuple)) and len(entry) >= 2:
            tid, dist = entry[0], entry[1]
        else:
            continue
        if tid is None or dist is None:
            continue
        try:
            tid_i = int(tid)
            dist_f = float(dist)
        except (TypeError, ValueError):
            continue
        team = team_assignments.get(str(tid_i), "?")
        out.append((tid_i, dist_f, team))
    return out


def _expected_team_for_curr(
    prev_action: str, prev_team: str,
) -> str | None:
    """Derive expected curr-action team from prev's action + team using the
    EXPECTED_TRANSITIONS relations ('same' / 'other' / 'either' / None)."""
    if prev_action == "block":
        return None  # block exempts; geometry is 'ambiguous' for curr
    for (p_a, _c_a), (relation, _label) in EXPECTED_TRANSITIONS.items():
        # We don't need curr_action for the team derivation — the relation
        # is shared per prev row in the table. But the table is keyed
        # (prev,curr), so we pick the first 'ok' row for this prev:
        if p_a == prev_action and relation in ("same", "other"):
            return prev_team if relation == "same" else (
                "B" if prev_team == "A" else "A"
            )
    return None


def _row_from_violation(
    *,
    rally_id: str,
    video_id: str,
    payload: dict[str, Any],
    rally_payload: dict[str, Any],
    co_violations: dict[str, bool],
    co_pid_invariants: list[str],
) -> dict[str, Any]:
    """Build one CSV row for a C-4 violation."""
    actions: list[dict[str, Any]] = rally_payload["actions"]
    team_assignments: dict[str, str] = rally_payload["team_assignments"]
    contacts_by_frame = _contact_by_frame(rally_payload["contacts"])

    prev_idx = payload["prev_index"]
    curr_idx = payload["curr_index"]
    # Re-sort the actions the same way the detector did, so indices align.
    sorted_actions = sorted(actions, key=lambda a: int(a.get("frame", 0)))
    prev = sorted_actions[prev_idx]
    curr = sorted_actions[curr_idx]

    prev_pid = int(prev.get("playerTrackId", -1))
    curr_pid = int(curr.get("playerTrackId", -1))
    team_label = team_assignments.get(str(prev_pid), "?")

    prev_contact = contacts_by_frame.get(int(prev.get("frame", -1)))
    curr_contact = contacts_by_frame.get(int(curr.get("frame", -1)))

    prev_top3 = _extract_top3_candidates(prev_contact, team_assignments)
    curr_top3 = _extract_top3_candidates(curr_contact, team_assignments)

    prev_player_dist = float(
        (prev_contact or {}).get("player_distance",
                                  (prev_contact or {}).get("playerDistance",
                                                            math.nan))
    )
    curr_player_dist = float(
        (curr_contact or {}).get("player_distance",
                                  (curr_contact or {}).get("playerDistance",
                                                            math.nan))
    )

    prev_alt_ratio = best_same_team_alt_ratio(
        candidates=prev_top3,
        current_dist=prev_player_dist,
        current_team=team_label,
        current_tid=prev_pid,
    )
    curr_alt_ratio = best_same_team_alt_ratio(
        candidates=curr_top3,
        current_dist=curr_player_dist,
        current_team=team_label,
        current_tid=curr_pid,
    )

    # Team geometry on each side uses the volleyball-expected team for
    # that action-slot. Prev's expected team is derived from action[i-2]
    # if available; for the catalog we keep it simple and derive curr's
    # expected team from prev (the table). Prev's expected team is left
    # as None (i.e., 'ambiguous') when we can't look back two steps.
    expected_curr_team = _expected_team_for_curr(
        prev_action=payload["prev_action"], prev_team=team_label,
    )
    expected_prev_team: str | None = None
    if prev_idx >= 1:
        action_before_prev = sorted_actions[prev_idx - 1]
        before_prev_pid = action_before_prev.get("playerTrackId")
        if isinstance(before_prev_pid, int):
            before_prev_team = team_assignments.get(str(before_prev_pid))
            if before_prev_team:
                expected_prev_team = _expected_team_for_curr(
                    prev_action=str(action_before_prev.get("action", "")),
                    prev_team=before_prev_team,
                )

    signal_type_fit_prev_v = "unknown"  # cannot compute without action[i-2]
    if prev_idx >= 1:
        signal_type_fit_prev_v = signal_type_fit(
            str(sorted_actions[prev_idx - 1].get("action", "")),
            payload["prev_action"],
        )
    signal_type_fit_curr_v = signal_type_fit(
        payload["prev_action"], payload["curr_action"],
    )

    row: dict[str, Any] = {
        "rally_id": rally_id,
        "video_id": video_id,
        "pair_idx": curr_idx,
        "frame_prev": payload["prev_frame"],
        "frame_curr": payload["curr_frame"],
        "action_prev_type": payload["prev_action"],
        "action_curr_type": payload["curr_action"],
        "player_id": payload["player_id"],
        "team_label": team_label,
        "conf_prev": prev.get("confidence", math.nan),
        "conf_curr": curr.get("confidence", math.nan),
        "prev_player_dist": prev_player_dist,
        "curr_player_dist": curr_player_dist,
        "prev_top3_candidates": json.dumps(prev_top3),
        "curr_top3_candidates": json.dumps(curr_top3),
        "prev_best_same_team_alt_ratio": prev_alt_ratio,
        "curr_best_same_team_alt_ratio": curr_alt_ratio,
        "signal_type_fit_prev": signal_type_fit_prev_v,
        "signal_type_fit_curr": signal_type_fit_curr_v,
        "signal_team_geometry_prev": signal_team_geometry(
            candidates=prev_top3, expected_team=expected_prev_team,
        ),
        "signal_team_geometry_curr": signal_team_geometry(
            candidates=curr_top3, expected_team=expected_curr_team,
        ),
        "co_c1_fires": co_violations.get("C-1", False),
        "co_c2_fires": co_violations.get("C-2", False),
        "co_c3_fires": co_violations.get("C-3", False),
        "co_pid_invariant_fires": ",".join(co_pid_invariants),
        "root_cause": "",
        "notes": "",
    }
    row["repair_recommendation"] = placeholder_repair_recommendation(row)
    return row


def _summarize(rows: list[dict[str, Any]]) -> str:
    """Build the fleet-aggregate summary markdown."""
    total = len(rows)
    pair_counter: Counter[tuple[str, str]] = Counter(
        (r["action_prev_type"], r["action_curr_type"]) for r in rows
    )
    co_c2 = sum(1 for r in rows if r["co_c2_fires"])
    co_c3 = sum(1 for r in rows if r["co_c3_fires"])
    rec_counter: Counter[str] = Counter(r["repair_recommendation"] for r in rows)
    by_video: Counter[str] = Counter(r["video_id"] for r in rows)
    top_videos = by_video.most_common(10)

    lines: list[str] = []
    lines.append(f"# C-4 Fleet Catalog Summary\n")
    lines.append(f"**Total C-4 violations:** {total}\n")
    lines.append(f"**Co-violation rates:** C-2 fires on {co_c2} ({100*co_c2/total if total else 0:.1f}%), "
                 f"C-3 fires on {co_c3} ({100*co_c3/total if total else 0:.1f}%).\n")
    lines.append("## (prev_action, curr_action) breakdown\n")
    lines.append("| prev | curr | count |\n|---|---|---:|")
    for (prev, curr), count in pair_counter.most_common():
        lines.append(f"| {prev} | {curr} | {count} |")
    lines.append("\n## Placeholder repair_recommendation distribution\n")
    lines.append("| recommendation | count |\n|---|---:|")
    for rec, count in rec_counter.most_common():
        lines.append(f"| {rec} | {count} |")
    lines.append("\n## Top 10 worst videos\n")
    lines.append("| video_id | C-4 count |\n|---|---:|")
    for video_id, count in top_videos:
        lines.append(f"| {video_id} | {count} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Walk the fleet, write a per-pair C-4 violation catalog.",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Path to write the per-violation CSV.",
    )
    parser.add_argument(
        "--summary", type=Path, required=True,
        help="Path to write the fleet-aggregate markdown summary.",
    )
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    video_ids = _load_video_ids()
    print(f"[catalog] fleet has {len(video_ids)} videos", flush=True)

    all_rows: list[dict[str, Any]] = []
    for idx, video_id in enumerate(video_ids, start=1):
        try:
            coherence_violations = coherence_run_all(video_id=video_id)
        except Exception as exc:
            print(f"[{idx}/{len(video_ids)}] video={video_id} ERROR: {exc}",
                  flush=True)
            continue
        try:
            pid_violations = pid_run_all(video_id=video_id)
        except Exception:
            pid_violations = []

        rally_payloads = _load_rally_payloads(video_id)

        # Build per-rally co-violation and PID-violation lookup tables.
        co_by_rally: dict[str, dict[str, bool]] = {}
        for v in coherence_violations:
            entry = co_by_rally.setdefault(v.rally_id, {
                "C-1": False, "C-2": False, "C-3": False, "C-4": False,
            })
            entry[v.invariant] = True
        pid_by_rally: dict[str, list[str]] = {}
        for v in pid_violations:
            pid_by_rally.setdefault(v.rally_id, []).append(v.invariant)

        c4_violations = [v for v in coherence_violations if v.invariant == "C-4"]
        rallies_with_c4 = {v.rally_id for v in c4_violations}
        print(
            f"[{idx}/{len(video_ids)}] video={video_id} "
            f"c4_pairs={len(c4_violations)} "
            f"rallies_with_c4={len(rallies_with_c4)}",
            flush=True,
        )

        for v in c4_violations:
            if v.payload is None:
                continue
            rally_payload = rally_payloads.get(v.rally_id)
            if rally_payload is None:
                continue
            row = _row_from_violation(
                rally_id=v.rally_id,
                video_id=video_id,
                payload=v.payload,
                rally_payload=rally_payload,
                co_violations=co_by_rally.get(v.rally_id, {}),
                co_pid_invariants=pid_by_rally.get(v.rally_id, []),
            )
            all_rows.append(row)

    # Write CSV.
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    print(f"[catalog] wrote {len(all_rows)} rows to {args.output}", flush=True)

    # Write summary markdown.
    args.summary.write_text(_summarize(all_rows))
    print(f"[catalog] wrote summary to {args.summary}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Re-run the helper tests to confirm no regression from the merge**

Run: `cd analysis && uv run pytest tests/unit/test_catalog_c4_violations.py -v`
Expected: all 14 tests still pass.

- [ ] **Step 5: Type check the script**

Run: `cd analysis && uv run mypy scripts/catalog_c4_violations.py`
Expected: success (some `Any`-typed JSON access is unavoidable; if mypy complains about specific `Any` returns, suppress with `# type: ignore[...]` only on the specific lines, not blanket).

- [ ] **Step 6: Smoke run on the 4-rally panel**

Run:

```bash
mkdir -p /tmp/c4_catalog_smoke
cd analysis && uv run python -u scripts/catalog_c4_violations.py \
    --output /tmp/c4_catalog_smoke/catalog.csv \
    --summary /tmp/c4_catalog_smoke/summary.md
```

Expected: per-video progress lines stream (`[N/M] video=<id> c4_pairs=X rallies_with_c4=Y`). Final lines:
```
[catalog] wrote <N> rows to /tmp/c4_catalog_smoke/catalog.csv
[catalog] wrote summary to /tmp/c4_catalog_smoke/summary.md
```

Sanity-check the outputs:

```bash
wc -l /tmp/c4_catalog_smoke/catalog.csv         # header + N rows
cat /tmp/c4_catalog_smoke/summary.md            # readable markdown
```

If the CSV has 0 rows (only the header), STOP and investigate — either (a) the fleet truly has no C-4 violations (unlikely given Sub-2.A's 744-violation baseline), or (b) something is broken in the orchestrator. Don't proceed to commit until the smoke run looks plausible.

- [ ] **Step 7: Commit**

```bash
git add analysis/scripts/catalog_c4_violations.py
git commit -m "$(cat <<'EOF'
feat(catalog): C-4 fleet orchestrator — CSV + summary markdown

Walks every video with confirmed-or-null rallies, runs the coherence
audit, filters to C-4 violations, joins with PID-invariant fires for
co-violation columns, computes per-pair evidence (type-fit /
team-geometry / alt-ratio / placeholder repair recommendation), writes
CSV + fleet-aggregate summary markdown.

Per-video progress streams to stdout (CLAUDE.md rule).

Spec: docs/superpowers/specs/2026-05-13-sub-2b-coherence-repair-design.md
EOF
)"
```

---

## Task 8: Run the fleet catalog and prepare for the gated review

This task is *operational*, not code. It produces the artifacts (CSV, summary, hand-classified rows) that the Phase 1 → Phase 2 review consumes. No commits required for Steps 1-3 (the outputs are gitignored content under `analysis/reports/coherence_c4_catalog/`). Step 4 commits the review markdown that closes Phase 1.

- [ ] **Step 1: Run the full fleet catalog**

Run:

```bash
mkdir -p analysis/reports/coherence_c4_catalog
cd analysis && uv run python -u scripts/catalog_c4_violations.py \
    --output analysis/reports/coherence_c4_catalog/2026-05-13.csv \
    --summary analysis/reports/coherence_c4_catalog/2026-05-13_summary.md
```

Expected: per-video progress + final write confirmations. Inspect `analysis/reports/coherence_c4_catalog/2026-05-13_summary.md`. The header tells you total fleet C-4 count and co-violation rates with C-2 / C-3. If C-3 co-fires on a majority, that's a strong signal the synthetic-serve cascade is dominant — flag this for the gated-review write-up.

- [ ] **Step 2: Hand-classify ≥30 rows**

Open `analysis/reports/coherence_c4_catalog/2026-05-13.csv` in your editor of choice. Sort by `repair_recommendation` to bias toward interesting rows (`ambiguous` and `skip` first; they're where the placeholder rule is uncertain), or sort by `rally_id` to keep video-level coherence.

For each of ≥30 rows, fill `root_cause` from this fixed vocabulary (also in the spec):

- `synthetic_serve_cascade` — prev or curr is a mis-attributed synthetic serve.
- `attribution_swap_prev` — prev should be a teammate (current is correct).
- `attribution_swap_curr` — curr should be a teammate (prev is correct).
- `genuine_double_touch` — both correctly attributed; rule should not have fired.
- `ghost_contact_prev` / `ghost_contact_curr` — one is a spurious contact-detection FP.
- `wrong_action_type` — attribution is fine but action types are misclassified.
- `block_exception_miss` — prev should have been classified as block (rule applied correctly but action type is wrong).
- `other` — anything else.

Fill `notes` for non-obvious rows. **Do not look at `repair_recommendation` before filling `root_cause`** — that's the catalog-bias risk called out in the spec (Risk 1). Hide or freeze the column during the first pass through.

- [ ] **Step 3: Compute placeholder-vs-truth agreement**

Quick pivot in your editor or a one-liner:

```bash
cd analysis && uv run python -c "
import csv
from collections import Counter
rows = list(csv.DictReader(open('analysis/reports/coherence_c4_catalog/2026-05-13.csv')))
labeled = [r for r in rows if r['root_cause']]
print(f'hand-classified: {len(labeled)} of {len(rows)}')
print('root_cause distribution:', Counter(r['root_cause'] for r in labeled))
print('placeholder distribution:', Counter(r['repair_recommendation'] for r in labeled))
"
```

This is the data the Phase 2 rule will be designed against.

- [ ] **Step 4: Write the gated review markdown and commit**

Create `analysis/reports/coherence_c4_catalog/2026-05-13_review.md`. Address:

1. **Fleet baseline:** total C-4 violations, dominant `(prev_type, curr_type)` cells, co-violation rates.
2. **Root-cause distribution** from the hand-classified rows. Which buckets dominate?
3. **Placeholder rule audit:** for each row hand-classified, did the placeholder `repair_recommendation` agree with the implied correct repair? Which signals fired correctly, which were silent or misleading?
4. **Phase 2 viability:** based on the distribution, is a Phase 2 repair pass worth building? If `synthetic_serve_cascade` dominates, the right fix is upstream and Phase 2 should be parked. If `attribution_swap_prev/curr` dominates, Phase 2 has clear leverage and the next step is designing the empirical rule.
5. **Phase 2 rule sketch** (only if §4 is "yes"): which signals carry weight, what thresholds/combination logic, what skip predicates. Stay grounded in the catalog data.

Commit the review:

```bash
git add analysis/reports/coherence_c4_catalog/2026-05-13_review.md
git commit -m "$(cat <<'EOF'
report(coherence): C-4 fleet catalog review (Phase 1 gate)

Baseline + root-cause distribution + placeholder-rule audit + Phase 2
viability call. Closes the Phase 1 → Phase 2 review gate for Sub-2.B.

Spec: docs/superpowers/specs/2026-05-13-sub-2b-coherence-repair-design.md
EOF
)"
```

- [ ] **Step 5: Update the memory index**

Add an entry under "Current workstreams" in `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY.md` (or update the existing Sub-2.A entry). Create a topic file `coherence_repair_sub_2b_2026_05_13.md` summarizing Phase 1 outputs and the Phase 2 viability call. Follow the existing memory entry style (status tag, short description, links to spec/plan/review).

---

## Final Verification

After all tasks land:

- [ ] All unit tests green: `cd analysis && uv run pytest tests/unit/test_coherence_invariants.py tests/unit/test_catalog_c4_violations.py tests/unit/test_action_classifier_pass2d_stub.py tests/unit/test_action_classifier.py tests/unit/test_pid_invariants.py -v`
- [ ] Type checks green: `cd analysis && uv run mypy rallycut/tracking/coherence_invariants.py rallycut/tracking/pid_invariants.py rallycut/tracking/action_classifier.py scripts/catalog_c4_violations.py`
- [ ] CLI works: `cd analysis && uv run rallycut audit-coherence-invariants <any-fleet-video-id>` exits 0 or 1, table includes C-4 rows when present.
- [ ] Production behavior at `COHERENCE_C4_REPAIR=0`: byte-identical to pre-workstream (Pass 2d stub returns 0).
- [ ] Production behavior at `COHERENCE_C4_REPAIR=1`: also byte-identical (stub returns 0 unconditionally).
- [ ] Catalog CSV + summary markdown exist under `analysis/reports/coherence_c4_catalog/`.
- [ ] ≥30 rows have `root_cause` filled.
- [ ] Gated review markdown drafted and committed.
- [ ] Memory entry recorded.

Phase 2 is gated on the review's verdict. If "go", write a Phase 2 plan against this same spec; if "no-go for now," park and revisit when upstream fixes (synthetic-serve placement, etc.) change the catalog distribution.
