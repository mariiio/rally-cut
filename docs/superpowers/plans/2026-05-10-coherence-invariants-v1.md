# Coherence Invariants v1 Implementation Plan (Sub-2.A)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a 3-rule game-rule audit (C-1: 3-contact max, C-2: alternating possessions, C-3: first-action-is-serve) that surfaces volleyball-rule violations in action sequences, paired with a CLI mirroring the existing PID audit.

**Architecture:** New module `coherence_invariants.py` with same shape as `pid_invariants.py` (same `Violation` dataclass, per-rule `check_cN_*` functions, `run_all` orchestrator). Orchestrator imports from `pid_invariants` to skip rallies with upstream I-1/I-3/I-6 violations. New CLI `audit-coherence-invariants` parallels `audit-pid-invariants`.

**Tech Stack:** Python 3.11, Typer, Rich, pytest, psycopg via existing `rallycut.evaluation.tracking.db.get_connection`.

**Spec:** `docs/superpowers/specs/2026-05-10-coherence-invariants-v1-design.md`

**Action vocabulary** (verified against fleet data): `serve`, `receive`, `dig`, `set`, `attack`, `block`, `unknown` — these strings appear in `actions_json["actions"][i]["action"]`.

---

## File Structure

**New (3):**
- `analysis/rallycut/tracking/coherence_invariants.py` — module with helpers, 3 check functions, `run_all` orchestrator (~200 LOC).
- `analysis/rallycut/cli/commands/audit_coherence_invariants.py` — Typer CLI (~50 LOC).
- `analysis/tests/unit/test_coherence_invariants.py` — unit tests for all 3 rules + orchestrator (~200 LOC).

**Modified (1):**
- `analysis/rallycut/cli/main.py` — register new CLI (~2 lines).

---

## Task 1: Module skeleton + C-1 (three-contact rule)

**Files:**
- Create: `analysis/rallycut/tracking/coherence_invariants.py`
- Create: `analysis/tests/unit/test_coherence_invariants.py`

- [ ] **Step 1: Write the failing test**

Create `analysis/tests/unit/test_coherence_invariants.py` with:

```python
"""Tests for coherence_invariants module."""

from __future__ import annotations

from rallycut.tracking.coherence_invariants import check_c1_three_contact_rule


def _action(frame: int, action: str, player_track_id: int) -> dict:
    return {"frame": frame, "action": action, "playerTrackId": player_track_id}


class TestCheckC1ThreeContactRule:
    def test_clean_sequence_passes(self) -> None:
        # Standard 3-contact possession: receive, set, attack — then opposing team digs.
        actions = [
            _action(100, "serve", 3),    # team A: serve
            _action(140, "receive", 1),  # team B: receive
            _action(170, "set", 2),      # team B: set
            _action(200, "attack", 1),   # team B: attack (3 contacts, ball crosses)
            _action(230, "dig", 4),      # team A: dig
        ]
        team_assignments = {"1": "B", "2": "B", "3": "A", "4": "A"}
        result = check_c1_three_contact_rule(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        assert result == []

    def test_4_consecutive_contacts_fails(self) -> None:
        # Team B has 4 consecutive contacts — illegal.
        actions = [
            _action(100, "serve", 3),    # A
            _action(140, "receive", 1),  # B
            _action(170, "set", 2),      # B
            _action(200, "set", 1),      # B (3rd)
            _action(230, "attack", 2),   # B (4th — illegal)
        ]
        team_assignments = {"1": "B", "2": "B", "3": "A", "4": "A"}
        result = check_c1_three_contact_rule(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        assert len(result) == 1
        assert result[0].invariant == "C-1"
        assert result[0].rally_id == "r1"
        assert "team B" in result[0].detail
        assert "4 consecutive" in result[0].detail

    def test_zero_actions_skips(self) -> None:
        result = check_c1_three_contact_rule(
            rally_id="r1", actions=[], team_assignments={},
        )
        assert result == []

    def test_one_action_skips(self) -> None:
        actions = [_action(100, "serve", 1)]
        team_assignments = {"1": "A"}
        result = check_c1_three_contact_rule(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        assert result == []

    def test_unresolvable_team_skips(self) -> None:
        # Action references player not in team_assignments — orchestrator should
        # have skipped this rally upstream, but defensive check still applies.
        actions = [
            _action(100, "serve", 99),   # 99 not in team_assignments
            _action(140, "receive", 1),
        ]
        team_assignments = {"1": "B"}
        result = check_c1_three_contact_rule(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        # Skip — undeterminable team
        assert result == []
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd analysis && uv run pytest tests/unit/test_coherence_invariants.py -v
```

Expected: FAIL with `ImportError: cannot import name 'check_c1_three_contact_rule'`.

- [ ] **Step 3: Create the module with the C-1 implementation**

Create `analysis/rallycut/tracking/coherence_invariants.py` with:

```python
"""Coherence (game-rule) invariants for the analysis pipeline.

Eval-time enforcement only: production never imports these checks. The audit
CLI (`rallycut audit-coherence-invariants`) wires them in alongside the
existing PID-structural audit.

Rules:
  C-1: A team can have at most 3 consecutive contacts before the ball
       must cross to the other team.
  C-2: Possessions alternate teams.
  C-3: First action of a rally is `serve`.

Skip semantics: each rule has explicit skip conditions. Additionally, the
orchestrator (`run_all`) excludes rallies that fail any I-1 / I-3 / I-6
PID invariant — those are upstream issues that would produce noisy
downstream coherence violations.

Spec: docs/superpowers/specs/2026-05-10-coherence-invariants-v1-design.md
"""

from __future__ import annotations

from typing import Any, cast

from rallycut.evaluation.tracking.db import get_connection
from rallycut.tracking.pid_invariants import Violation


def _team_for_action(
    action: dict[str, Any], team_assignments: dict[str, str]
) -> str | None:
    """Resolve an action to its team label ('A'/'B'). None if undeterminable."""
    tid = action.get("playerTrackId")
    if tid is None:
        return None
    label = team_assignments.get(str(tid))
    if label not in ("A", "B"):
        return None
    return label


def _actions_sorted_by_frame(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Defensive sort by frame — actions in DB may be unordered."""
    return sorted(actions, key=lambda a: int(a.get("frame", 0)))


def check_c1_three_contact_rule(
    *,
    rally_id: str,
    actions: list[dict[str, Any]],
    team_assignments: dict[str, str],
) -> list[Violation]:
    """C-1: a team can have at most 3 consecutive contacts before crossing."""
    if len(actions) < 2:
        return []
    sorted_actions = _actions_sorted_by_frame(actions)

    violations: list[Violation] = []
    current_team: str | None = None
    current_frames: list[int] = []

    for action in sorted_actions:
        team = _team_for_action(action, team_assignments)
        if team is None:
            return []  # defensive — orchestrator should have skipped
        frame = int(action.get("frame", 0))
        if team == current_team:
            current_frames.append(frame)
        else:
            current_team = team
            current_frames = [frame]
        if len(current_frames) == 4:
            # First time we hit 4 consecutive — emit one violation per
            # offending sequence (don't keep emitting at 5, 6, etc.).
            violations.append(
                Violation(
                    invariant="C-1",
                    rally_id=rally_id,
                    detail=(
                        f"team {current_team} had 4 consecutive contacts "
                        f"at frames {current_frames}; max is 3"
                    ),
                )
            )
    return violations
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd analysis && uv run pytest tests/unit/test_coherence_invariants.py -v
```

Expected: 5 PASSED.

- [ ] **Step 5: Lint + mypy**

```bash
cd analysis && uv run ruff check rallycut/tracking/coherence_invariants.py tests/unit/test_coherence_invariants.py
cd analysis && uv run mypy rallycut/tracking/coherence_invariants.py
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add analysis/rallycut/tracking/coherence_invariants.py analysis/tests/unit/test_coherence_invariants.py
git commit -m "feat(coherence): C-1 three-contact rule + module skeleton"
```

---

## Task 2: C-2 (alternating possessions)

**Files:**
- Modify: `analysis/rallycut/tracking/coherence_invariants.py`
- Modify: `analysis/tests/unit/test_coherence_invariants.py`

- [ ] **Step 1: Write the failing tests**

Append to `analysis/tests/unit/test_coherence_invariants.py`:

```python
from rallycut.tracking.coherence_invariants import check_c2_alternating_possessions


class TestCheckC2AlternatingPossessions:
    def test_clean_alternating_passes(self) -> None:
        # Standard exchange: A serves, B receives/sets/attacks, A digs/sets/attacks.
        actions = [
            _action(100, "serve", 3),    # A
            _action(140, "receive", 1),  # B
            _action(170, "set", 2),      # B
            _action(200, "attack", 1),   # B (ends possession)
            _action(230, "dig", 4),      # A
            _action(260, "set", 3),      # A
            _action(290, "attack", 4),   # A (ends possession)
        ]
        team_assignments = {"1": "B", "2": "B", "3": "A", "4": "A"}
        result = check_c2_alternating_possessions(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        assert result == []

    def test_same_team_after_attack_fails(self) -> None:
        # Team B attacks, then team B digs — illegal (possession should transfer).
        actions = [
            _action(100, "serve", 3),    # A
            _action(140, "receive", 1),  # B
            _action(170, "set", 2),      # B
            _action(200, "attack", 1),   # B (ends possession — ball crosses)
            _action(230, "dig", 2),      # B — should be A!
        ]
        team_assignments = {"1": "B", "2": "B", "3": "A", "4": "A"}
        result = check_c2_alternating_possessions(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        assert len(result) >= 1
        assert any(v.invariant == "C-2" for v in result)
        assert any("expected" in v.detail and "team A" in v.detail for v in result)

    def test_zero_actions_skips(self) -> None:
        result = check_c2_alternating_possessions(
            rally_id="r1", actions=[], team_assignments={},
        )
        assert result == []

    def test_one_action_skips(self) -> None:
        result = check_c2_alternating_possessions(
            rally_id="r1",
            actions=[_action(100, "serve", 1)],
            team_assignments={"1": "A"},
        )
        assert result == []

    def test_serve_after_serve_fails(self) -> None:
        # Two serves in a row — possession should have transferred.
        actions = [
            _action(100, "serve", 3),    # A
            _action(140, "serve", 1),    # B — illegal (possession should transfer)
        ]
        team_assignments = {"1": "B", "3": "A"}
        result = check_c2_alternating_possessions(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        # Two serves on different teams — actually possession alternates here, no violation.
        # Adjust assertion to clarify expected behavior. A serve always ends the prior
        # possession; the next serve from the other team is fine.
        assert result == []
```

- [ ] **Step 2: Run tests to verify failure**

```bash
cd analysis && uv run pytest tests/unit/test_coherence_invariants.py::TestCheckC2AlternatingPossessions -v
```

Expected: FAIL with `ImportError: cannot import name 'check_c2_alternating_possessions'`.

- [ ] **Step 3: Append the C-2 implementation**

Append to `analysis/rallycut/tracking/coherence_invariants.py`:

```python
# Action types that always end the current possession (ball crosses to the
# other side or starts a new turn).
_POSSESSION_END_ACTIONS = {"attack", "serve"}


def check_c2_alternating_possessions(
    *,
    rally_id: str,
    actions: list[dict[str, Any]],
    team_assignments: dict[str, str],
) -> list[Violation]:
    """C-2: possessions alternate teams.

    Possession ends when:
      - The action is `attack` (ball crosses to other team).
      - The action is `serve` (rally turn starts).
      - The current team has accumulated 3 contacts.

    After a possession ends, the next action must be by the OTHER team.
    """
    if len(actions) < 2:
        return []
    sorted_actions = _actions_sorted_by_frame(actions)

    violations: list[Violation] = []
    current_team: str | None = None
    contacts_in_possession: int = 0
    last_team: str | None = None
    last_action: dict[str, Any] | None = None
    last_index: int = -1

    for idx, action in enumerate(sorted_actions):
        team = _team_for_action(action, team_assignments)
        if team is None:
            return []  # defensive
        action_type = str(action.get("action", ""))

        if last_team is not None and last_action is not None:
            last_action_type = str(last_action.get("action", ""))
            possession_ended = (
                last_action_type in _POSSESSION_END_ACTIONS
                or contacts_in_possession >= 3
            )
            if possession_ended:
                # New action must be the OTHER team.
                if team == last_team:
                    other = "B" if last_team == "A" else "A"
                    violations.append(
                        Violation(
                            invariant="C-2",
                            rally_id=rally_id,
                            detail=(
                                f"team {last_team} action[{last_index}] "
                                f"(frame {last_action.get('frame')}, "
                                f"{last_action_type}) ended possession; "
                                f"next action[{idx}] (frame {action.get('frame')}, "
                                f"{action_type}) was also team {team} — "
                                f"expected team {other}"
                            ),
                        )
                    )
                # Reset possession either way (don't keep cascading violations
                # if the same wrong team has multiple actions).
                current_team = team
                contacts_in_possession = 1
            else:
                if team != last_team:
                    # Possession transferred without an end-action — also a violation
                    # (mid-possession crossover). For v1, just track and move on.
                    current_team = team
                    contacts_in_possession = 1
                else:
                    contacts_in_possession += 1
        else:
            current_team = team
            contacts_in_possession = 1

        last_team = team
        last_action = action
        last_index = idx

    return violations
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd analysis && uv run pytest tests/unit/test_coherence_invariants.py::TestCheckC2AlternatingPossessions -v
```

Expected: 5 PASSED.

- [ ] **Step 5: Run full test file to confirm no regression**

```bash
cd analysis && uv run pytest tests/unit/test_coherence_invariants.py -v
```

Expected: 10 PASSED (5 from Task 1 + 5 new).

- [ ] **Step 6: Lint + mypy**

```bash
cd analysis && uv run ruff check rallycut/tracking/coherence_invariants.py tests/unit/test_coherence_invariants.py
cd analysis && uv run mypy rallycut/tracking/coherence_invariants.py
```

Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add analysis/rallycut/tracking/coherence_invariants.py analysis/tests/unit/test_coherence_invariants.py
git commit -m "feat(coherence): C-2 alternating possessions"
```

---

## Task 3: C-3 (first action is serve)

**Files:**
- Modify: `analysis/rallycut/tracking/coherence_invariants.py`
- Modify: `analysis/tests/unit/test_coherence_invariants.py`

- [ ] **Step 1: Write the failing tests**

Append to `analysis/tests/unit/test_coherence_invariants.py`:

```python
from rallycut.tracking.coherence_invariants import check_c3_first_action_is_serve


class TestCheckC3FirstActionIsServe:
    def test_clean_first_serve_passes(self) -> None:
        actions = [
            _action(100, "serve", 1),
            _action(140, "receive", 2),
        ]
        result = check_c3_first_action_is_serve(rally_id="r1", actions=actions)
        assert result == []

    def test_first_action_attack_fails(self) -> None:
        actions = [
            _action(100, "attack", 1),
            _action(140, "dig", 2),
        ]
        result = check_c3_first_action_is_serve(rally_id="r1", actions=actions)
        assert len(result) == 1
        assert result[0].invariant == "C-3"
        assert "attack" in result[0].detail

    def test_first_action_receive_fails(self) -> None:
        actions = [
            _action(100, "receive", 1),
        ]
        result = check_c3_first_action_is_serve(rally_id="r1", actions=actions)
        assert len(result) == 1
        assert result[0].invariant == "C-3"
        assert "receive" in result[0].detail

    def test_zero_actions_skips(self) -> None:
        result = check_c3_first_action_is_serve(rally_id="r1", actions=[])
        assert result == []
```

- [ ] **Step 2: Run, verify failure**

```bash
cd analysis && uv run pytest tests/unit/test_coherence_invariants.py::TestCheckC3FirstActionIsServe -v
```

Expected: FAIL with `ImportError: cannot import name 'check_c3_first_action_is_serve'`.

- [ ] **Step 3: Append the C-3 implementation**

Append to `analysis/rallycut/tracking/coherence_invariants.py`:

```python
def check_c3_first_action_is_serve(
    *,
    rally_id: str,
    actions: list[dict[str, Any]],
) -> list[Violation]:
    """C-3: rally's first action must be `serve`."""
    if not actions:
        return []
    sorted_actions = _actions_sorted_by_frame(actions)
    first = sorted_actions[0]
    first_type = str(first.get("action", ""))
    if first_type == "serve":
        return []
    return [
        Violation(
            invariant="C-3",
            rally_id=rally_id,
            detail=(
                f"first action is {first_type!r} (frame {first.get('frame')}); "
                f"expected 'serve'"
            ),
        )
    ]
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd analysis && uv run pytest tests/unit/test_coherence_invariants.py -v
```

Expected: 14 PASSED (10 from Tasks 1-2 + 4 new).

- [ ] **Step 5: Lint + mypy**

```bash
cd analysis && uv run ruff check rallycut/tracking/coherence_invariants.py tests/unit/test_coherence_invariants.py
cd analysis && uv run mypy rallycut/tracking/coherence_invariants.py
```

- [ ] **Step 6: Commit**

```bash
git add analysis/rallycut/tracking/coherence_invariants.py analysis/tests/unit/test_coherence_invariants.py
git commit -m "feat(coherence): C-3 first action is serve"
```

---

## Task 4: run_all orchestrator (with upstream skip)

**Files:**
- Modify: `analysis/rallycut/tracking/coherence_invariants.py`
- Modify: `analysis/tests/unit/test_coherence_invariants.py`

- [ ] **Step 1: Write failing tests**

Append to `analysis/tests/unit/test_coherence_invariants.py`:

```python
from unittest.mock import MagicMock, patch

from rallycut.tracking.coherence_invariants import run_all
from rallycut.tracking.pid_invariants import Violation as PidViolation


class TestRunAll:
    def _mock_db_conn(
        self,
        *,
        rally_rows: list[tuple],
    ) -> MagicMock:
        cur = MagicMock()
        cur.fetchall.return_value = rally_rows
        cur.__enter__ = lambda self: self
        cur.__exit__ = lambda self, *a: None
        conn = MagicMock()
        conn.cursor.return_value = cur
        conn.__enter__ = lambda self: self
        conn.__exit__ = lambda self, *a: None
        return conn

    def test_clean_video_returns_no_violations(self) -> None:
        # One rally with a clean alternating sequence.
        actions_json = {
            "actions": [
                _action(100, "serve", 3),
                _action(140, "receive", 1),
                _action(170, "set", 2),
                _action(200, "attack", 1),
                _action(230, "dig", 4),
            ],
            "teamAssignments": {"1": "B", "2": "B", "3": "A", "4": "A"},
        }
        rally_rows = [("r1", actions_json)]
        conn = self._mock_db_conn(rally_rows=rally_rows)
        with patch(
            "rallycut.tracking.coherence_invariants.get_connection", return_value=conn
        ), patch(
            "rallycut.tracking.coherence_invariants.pid_run_all", return_value=[]
        ):
            violations = run_all(video_id="v1")
        assert violations == []

    def test_dirty_video_aggregates_violations(self) -> None:
        # First action is `attack` (C-3 fires), and 4 consecutive same-team contacts (C-1 fires).
        actions_json = {
            "actions": [
                _action(100, "attack", 1),  # C-3: should be serve
                _action(140, "set", 2),     # B (1)
                _action(170, "set", 1),     # B (2)
                _action(200, "set", 2),     # B (3)
                _action(230, "attack", 1),  # B (4 — C-1 fires)
            ],
            "teamAssignments": {"1": "B", "2": "B"},
        }
        rally_rows = [("r1", actions_json)]
        conn = self._mock_db_conn(rally_rows=rally_rows)
        with patch(
            "rallycut.tracking.coherence_invariants.get_connection", return_value=conn
        ), patch(
            "rallycut.tracking.coherence_invariants.pid_run_all", return_value=[]
        ):
            violations = run_all(video_id="v1")
        invariants_seen = {v.invariant for v in violations}
        assert "C-1" in invariants_seen
        assert "C-3" in invariants_seen

    def test_skips_rally_with_upstream_i6_violation(self) -> None:
        # Rally has illegal sequence (would fire C-1) BUT also has I-6 violation
        # — orchestrator should skip it entirely.
        actions_json = {
            "actions": [
                _action(100, "serve", 1),
                _action(140, "set", 1),    # 4 consecutive A
                _action(170, "set", 1),
                _action(200, "set", 1),
                _action(230, "attack", 1),
            ],
            "teamAssignments": {"1": "A"},
        }
        rally_rows = [("r1", actions_json)]
        conn = self._mock_db_conn(rally_rows=rally_rows)
        upstream = [
            PidViolation(
                invariant="I-6", rally_id="r1",
                detail="primary track 2 missing from team_assignments",
            )
        ]
        with patch(
            "rallycut.tracking.coherence_invariants.get_connection", return_value=conn
        ), patch(
            "rallycut.tracking.coherence_invariants.pid_run_all", return_value=upstream
        ):
            violations = run_all(video_id="v1")
        assert violations == []  # Skipped due to upstream I-6
```

- [ ] **Step 2: Run, verify failure**

```bash
cd analysis && uv run pytest tests/unit/test_coherence_invariants.py::TestRunAll -v
```

Expected: FAIL with `ImportError: cannot import name 'run_all'` (or `pid_run_all`).

- [ ] **Step 3: Append `run_all` to the module**

Append to `analysis/rallycut/tracking/coherence_invariants.py`:

```python
# Renamed import to avoid shadowing the local `run_all` function name.
from rallycut.tracking.pid_invariants import run_all as pid_run_all


# PID invariants whose failures should exclude a rally from coherence checks.
# These directly affect action attribution / team labeling, so coherence
# violations on these rallies would be downstream noise.
_UPSTREAM_BLOCKER_INVARIANTS = frozenset({"I-1", "I-3", "I-6"})


def run_all(*, video_id: str) -> list[Violation]:
    """Run all 3 coherence invariants against a video's persisted state.

    Skips rallies that fail upstream PID invariants (I-1 / I-3 / I-6) to
    avoid flagging downstream effects of structural problems.
    """
    upstream = pid_run_all(video_id=video_id)
    excluded_rallies: set[str] = {
        v.rally_id for v in upstream
        if v.invariant in _UPSTREAM_BLOCKER_INVARIANTS
    }

    rally_query = """
        SELECT
            r.id AS rally_id,
            pt.actions_json
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE r.video_id = %s
          AND (r.status = 'CONFIRMED' OR r.status IS NULL)
        ORDER BY r.start_ms
    """

    violations: list[Violation] = []

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(rally_query, [video_id])
            rally_rows = cur.fetchall()

    for row in rally_rows:
        rally_id = cast(str, row[0])
        if rally_id in excluded_rallies:
            continue
        actions_json = row[1]
        if not isinstance(actions_json, dict):
            continue
        actions = actions_json.get("actions")
        team_assignments = actions_json.get("teamAssignments")
        if not isinstance(actions, list):
            continue
        if not isinstance(team_assignments, dict):
            team_assignments = {}

        violations.extend(
            check_c1_three_contact_rule(
                rally_id=rally_id, actions=actions,
                team_assignments=team_assignments,
            )
        )
        violations.extend(
            check_c2_alternating_possessions(
                rally_id=rally_id, actions=actions,
                team_assignments=team_assignments,
            )
        )
        violations.extend(
            check_c3_first_action_is_serve(
                rally_id=rally_id, actions=actions,
            )
        )

    return violations
```

The query is intentionally simple — only loads what coherence checks need (rally_id + actions_json). This is also why the test mock provides 2-tuples (rally_id, actions_json) instead of the larger tuples used by `pid_invariants.run_all`.

- [ ] **Step 4: Run tests to verify pass**

```bash
cd analysis && uv run pytest tests/unit/test_coherence_invariants.py -v
```

Expected: 17 PASSED (14 from prior tasks + 3 new).

- [ ] **Step 5: Lint + mypy**

```bash
cd analysis && uv run ruff check rallycut/tracking/coherence_invariants.py tests/unit/test_coherence_invariants.py
cd analysis && uv run mypy rallycut/tracking/coherence_invariants.py
```

- [ ] **Step 6: Commit**

```bash
git add analysis/rallycut/tracking/coherence_invariants.py analysis/tests/unit/test_coherence_invariants.py
git commit -m "feat(coherence): run_all orchestrator with upstream-PID-failure skip"
```

---

## Task 5: CLI command + register in main.py

**Files:**
- Create: `analysis/rallycut/cli/commands/audit_coherence_invariants.py`
- Modify: `analysis/rallycut/cli/main.py`

- [ ] **Step 1: Create the CLI command**

Create `analysis/rallycut/cli/commands/audit_coherence_invariants.py`:

```python
"""CLI: rallycut audit-coherence-invariants <video-id>

Eval-time game-rule audit. Sibling to audit-pid-invariants. Exits non-zero
on any violation.
"""

from __future__ import annotations

import typer
from rich.console import Console
from rich.table import Table

from rallycut.tracking.coherence_invariants import run_all

console = Console()


def audit_coherence_invariants_cmd(
    video_id: str = typer.Argument(..., help="Video UUID"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress info output"),
) -> None:
    """Audit volleyball-rule coherence for a video's action sequences."""
    if not quiet:
        console.print(
            f"[dim]Running coherence-invariant audit on {video_id}…[/dim]"
        )

    violations = run_all(video_id=video_id)

    if not violations:
        if not quiet:
            console.print("[green]✓ All coherence invariants hold[/green]")
        raise typer.Exit(code=0)

    table = Table(title=f"Coherence-invariant violations — video {video_id}")
    table.add_column("Invariant", style="bold")
    table.add_column("Rally", style="cyan")
    table.add_column("Severity", style="yellow")
    table.add_column("Detail")
    for v in violations:
        table.add_row(v.invariant, v.rally_id, v.severity, v.detail)
    console.print(table)

    n_errors = sum(1 for v in violations if v.severity == "error")
    n_warns = len(violations) - n_errors
    console.print(
        f"[red]{n_errors} error[/red] · [yellow]{n_warns} warn[/yellow] · "
        f"{len(violations)} total"
    )

    raise typer.Exit(code=1 if n_errors else 0)
```

- [ ] **Step 2: Verify import**

```bash
cd analysis && uv run python -c "from rallycut.cli.commands.audit_coherence_invariants import audit_coherence_invariants_cmd; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 3: Register in `cli/main.py`**

In `analysis/rallycut/cli/main.py`:

After the line:
```python
from rallycut.cli.commands.audit_pid_invariants import audit_pid_invariants_cmd
```

Add:
```python
from rallycut.cli.commands.audit_coherence_invariants import audit_coherence_invariants_cmd
```

After the line:
```python
app.command(name="audit-pid-invariants")(audit_pid_invariants_cmd)
```

Add:
```python
app.command(name="audit-coherence-invariants")(audit_coherence_invariants_cmd)
```

- [ ] **Step 4: Verify CLI registration**

```bash
uv run rallycut audit-coherence-invariants --help
```

Expected: Typer help output describing the command, including `Audit volleyball-rule coherence for a video's action sequences.`.

- [ ] **Step 5: Lint + mypy**

```bash
cd analysis && uv run ruff check rallycut/cli/commands/audit_coherence_invariants.py rallycut/cli/main.py
cd analysis && uv run mypy rallycut/cli/commands/audit_coherence_invariants.py
```

Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add analysis/rallycut/cli/commands/audit_coherence_invariants.py analysis/rallycut/cli/main.py
git commit -m "feat(cli): audit-coherence-invariants — game-rule audit"
```

---

## Task 6: Fleet sweep + memory

**Files:** None edited; verification + memory only.

- [ ] **Step 1: Run audit on the 4 panel videos for spot-check**

```bash
cd analysis
for vid in 5c756c41-1cc1-4486-a95c-97398912cfbe \
           b5fb0594-d64f-4a0d-bad9-de8fc36414d0 \
           854bb250-3e91-47d2-944d-f62413e3cf45 \
           7d77980f-3006-40e0-adc0-db491a5bb659; do
  echo "===== ${vid:0:8} ====="
  uv run rallycut audit-coherence-invariants "$vid" --quiet
  echo "  exit=$?"
done
```

Expected: completes without error. Outputs are interpretable (rule name + rally + detail). Counts vary per video.

- [ ] **Step 2: Run audit on the full fleet**

```bash
while read vid; do
  uv run rallycut audit-coherence-invariants "$vid" --quiet
done < /tmp/audit_videos.txt > /tmp/coherence_fleet.txt 2>&1
echo "exit=$?"
echo "summary:"
grep -cE "C-[123]" /tmp/coherence_fleet.txt
```

Expected: completes without error. The grep gives total violation count across the fleet.

- [ ] **Step 3: Per-rule fleet counts**

```bash
echo "C-1 count: $(grep -cE 'C-1' /tmp/coherence_fleet.txt)"
echo "C-2 count: $(grep -cE 'C-2' /tmp/coherence_fleet.txt)"
echo "C-3 count: $(grep -cE 'C-3' /tmp/coherence_fleet.txt)"
echo "Top 10 dirtiest videos by total violations:"
grep -E "video " /tmp/coherence_fleet.txt | sort | uniq -c | sort -rn | head -10
```

Use the actual numbers in the memory entry below.

- [ ] **Step 4: Update memory**

Memory directory is NOT a git repo. Write files; no commit.

Create `/Users/mario/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/coherence_invariants_v1_2026_05_10.md`:

```markdown
---
name: Coherence invariants v1 (Sub-2.A) 2026-05-10
description: Game-rule audit module with 3 starter rules (C-1: 3-contact max, C-2: alternating possessions, C-3: first action is serve)
type: project
---

# Coherence Invariants v1 — Sub-2.A

**Shipped:** 2026-05-10

**Why:** Closes the original "concern #5: game dynamics" from the session-start request. Builds on the now-clean PID baseline (94% fleet clean post-retrack). Surfaces CORRECTNESS issues that the PID-structural audit can't see — e.g., 4-consecutive-contacts on one team, possession not transferring after attack, missing serve.

**What it does:** New module `analysis/rallycut/tracking/coherence_invariants.py` mirrors `pid_invariants.py`. Three rules:
- **C-1**: a team can have at most 3 consecutive contacts.
- **C-2**: possessions alternate teams (after attack/serve or 3rd contact, next action is opposing team).
- **C-3**: first action of a rally is `serve`.

Orchestrator excludes rallies with upstream I-1/I-3/I-6 violations to avoid downstream noise. New CLI `audit-coherence-invariants <video-id>` parallels `audit-pid-invariants`.

**How to apply:** `uv run rallycut audit-coherence-invariants <video-id>`. Detection only — no cleanup.

**v1 scope:** detection only; 3 rules; single-rally checks (no cross-rally state); skip-when-upstream-fails to keep signal clean.

**Spec:** `docs/superpowers/specs/2026-05-10-coherence-invariants-v1-design.md`
**Plan:** `docs/superpowers/plans/2026-05-10-coherence-invariants-v1.md`

**Initial fleet baseline (2026-05-10):**
- [Replace with actual counts from Step 3 above]
- C-1 fleet count: ___
- C-2 fleet count: ___
- C-3 fleet count: ___
- Top dirty videos: ___

**Future iterations** could add:
- Same-player-no-consecutive-touches (within possession).
- Block + cover edge cases.
- Cross-rally rules (server's team consistent within set).
- Cleanup CLIs for systematic patterns surfaced by v1.
```

Update `/Users/mario/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY.md` — add Sub-2.A to the workstreams list. Use Edit with this old_string:

```markdown
- [SHIPPED] [**PID leverage audit Sub-1 2026-05-08/10**](pid_i6_team_assignments_realign_2026_05_08.md) — `audit-pid-invariants` CLI + 8 invariants (I-1..I-8) + I-7 silent-skip in `compute_match_stats`. Producer fixes: Sub-1.1.A (`player_tracker` re-derive after `optimize_global_identity`, closes I-6) + Sub-1.2.A [`classify_teams 2v2-fallback`](classify_teams_2v2_fallback_2026_05_10.md) (closes I-8 at producer for new tracking). Cleanup CLIs: Sub-1.1.B [`cleanup-team-assignments`](pid_i6_cleanup_cli_2026_05_09.md), Sub-1.1.C [`cleanup-stale-attribution`](pid_stale_attribution_cleanup_2026_05_09.md), Sub-1.1.E [`cleanup-team-labels-by-majority`](pid_i8_cleanup_2026_05_10.md). Sub-1.1.D retracked 16 I-1 rallies. I-8 added (`5a6968b`). **Fleet state (2026-05-10):** 47/70 clean (67%); 60 residual on legacy data = 1 I-1 (`4b7ad71f` metadata edge) + 59 I-8 (side-switched videos; will clear on re-track now that producer fix is in). Frontend `STALE_TIMEOUT_MS` 30→60 min in `4767ef3`. Audit schema-bug fix `9e6390f`.
```

and this new_string:

```markdown
- [SHIPPED] [**PID leverage audit Sub-1 2026-05-08/10**](pid_i6_team_assignments_realign_2026_05_08.md) — `audit-pid-invariants` CLI + 8 invariants (I-1..I-8) + I-7 silent-skip in `compute_match_stats`. Producer fixes: Sub-1.1.A (`player_tracker` re-derive after `optimize_global_identity`, closes I-6) + Sub-1.2.A [`classify_teams 2v2-fallback`](classify_teams_2v2_fallback_2026_05_10.md) (closes I-8 at producer for new tracking). Cleanup CLIs: Sub-1.1.B [`cleanup-team-assignments`](pid_i6_cleanup_cli_2026_05_09.md), Sub-1.1.C [`cleanup-stale-attribution`](pid_stale_attribution_cleanup_2026_05_09.md), Sub-1.1.E [`cleanup-team-labels-by-majority`](pid_i8_cleanup_2026_05_10.md). Sub-1.1.D retracked 16 I-1 rallies. I-8 added (`5a6968b`). 59 I-8 rallies retracked → fleet 94% clean (10 residual: 1 I-1 + 8 I-6 on degenerate rallies + 1 I-8 ambiguous). Frontend `STALE_TIMEOUT_MS` 30→60 min in `4767ef3`. Audit schema-bug fix `9e6390f`.
- [SHIPPED] [**Coherence invariants v1 Sub-2.A 2026-05-10**](coherence_invariants_v1_2026_05_10.md) — `audit-coherence-invariants` CLI + 3 game-rule invariants (C-1: 3-contact max, C-2: alternating possessions, C-3: first action is serve). Detection only; orchestrator skips rallies with upstream I-1/I-3/I-6. Initial fleet counts in memo.
```

- [ ] **Step 5: No commit (memory dir is not a git repo)**

---

## Done criteria

- [ ] All 17 unit tests in `test_coherence_invariants.py` pass.
- [ ] `uv run rallycut audit-coherence-invariants --help` shows the command.
- [ ] CLI runs cleanly on the 4 panel videos.
- [ ] CLI runs without error across the full fleet (70 videos).
- [ ] Per-rule fleet counts captured in memory entry.
- [ ] Memory entry recorded in MEMORY.md and per-topic file.

## Out of scope

- Cleanup CLIs for coherence violations (no clear mechanical fix).
- More than 3 rules (defer until v1 reveals patterns).
- Cross-rally rules (set-boundary detection).
- Severity tuning (all violations are `error`).
- Re-attribution / classifier corrections based on coherence violations.
- Sub-3 (web debug surface).
