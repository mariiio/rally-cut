# PID Leverage Audit Sub-1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an eval-time invariant audit for canonical PID attribution across the analysis pipeline; expose it as a CLI; close the one known leakage point at `compute_match_stats:156`; wire into the panel eval script.

**Architecture:** Pure-function check module (`pid_invariants.py`) with one function per invariant returning `list[Violation]`, an orchestrator (`run_all`) that loads from the database, a Typer CLI command (`audit-pid-invariants`) that prints a Rich table and exits non-zero on errors, and a surgical silent-skip fix in `compute_match_stats.py` that closes I-7. Wiring into `scripts/eval_cross_fixture.sh` makes the audit a regression gate for panel runs.

**Tech Stack:** Python 3.11, Typer (CLI), Rich (tables), pytest, psycopg (DB via existing `rallycut.evaluation.tracking.db.get_connection`).

**Spec:** `docs/superpowers/specs/2026-05-08-pid-leverage-audit-sub1-design.md`

---

## File Structure

**New files (3):**
- `analysis/rallycut/tracking/pid_invariants.py` — `Violation` dataclass + 7 pure check functions + `run_all` orchestrator
- `analysis/rallycut/cli/commands/audit_pid_invariants.py` — Typer command, Rich table output, exit-code propagation
- `analysis/tests/unit/test_pid_invariants.py` — one test class per check function, clean + bad input cases

**Modified files (3):**
- `analysis/rallycut/cli/main.py` — register new CLI command
- `analysis/rallycut/cli/commands/compute_match_stats.py:155-156` — silent-skip when track_id not in `player_map`
- `analysis/scripts/eval_cross_fixture.sh` — call audit after match-players; propagate exit code

---

## Task 1: Module skeleton — `Violation` dataclass and module docstring

**Files:**
- Create: `analysis/rallycut/tracking/pid_invariants.py`

- [ ] **Step 1: Create the module with the dataclass**

```python
"""PID-attribution invariants for the analysis pipeline.

Eval-time enforcement only: production never imports these checks. The audit
CLI (`rallycut audit-pid-invariants`) and the panel eval script wire them in.

Each invariant has a dedicated `check_iN_*` function returning a list of
Violation records. `run_all` orchestrates DB loading and aggregation.

Invariants (see docs/superpowers/specs/2026-05-08-pid-leverage-audit-sub1-design.md):
  I-1: len(primary_track_ids) == 4 (or 0 if filter disabled)
  I-2: every trackId in positionsJson ∈ primary_track_ids
  I-3: every action's playerTrackId ∈ primary_track_ids ∪ {-1}
  I-4: every contact's playerTrackId ∈ primary_track_ids ∪ {-1}
  I-5: trackToPlayer is total over primary_track_ids
  I-6: team_assignments is total over primary_track_ids
  I-7: after stats mapping, every action's player_track_id ∈ {1..4} ∪ {-1}
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal


@dataclass(frozen=True)
class Violation:
    invariant: str
    rally_id: str
    detail: str
    severity: Literal["error", "warn"] = "error"
```

- [ ] **Step 2: Verify the file imports cleanly**

Run: `uv run python -c "from rallycut.tracking.pid_invariants import Violation; print(Violation(invariant='I-1', rally_id='r', detail='d'))"`
Expected: prints `Violation(invariant='I-1', rally_id='r', detail='d', severity='error')`.

- [ ] **Step 3: Commit**

```bash
git add analysis/rallycut/tracking/pid_invariants.py
git commit -m "feat(pid-invariants): add Violation dataclass and module skeleton"
```

---

## Task 2: Test file skeleton + I-1 (primary_track_ids size)

**Files:**
- Create: `analysis/tests/unit/test_pid_invariants.py`
- Modify: `analysis/rallycut/tracking/pid_invariants.py` (add `check_i1_primary_set_size`)

- [ ] **Step 1: Write the failing test**

```python
"""Tests for pid_invariants module."""

from __future__ import annotations

from rallycut.tracking.pid_invariants import (
    Violation,
    check_i1_primary_set_size,
)


class TestCheckI1PrimarySetSize:
    def test_clean_size_4_passes(self) -> None:
        violations = check_i1_primary_set_size(rally_id="r1", primary_track_ids=[3, 7, 12, 15])
        assert violations == []

    def test_clean_size_0_passes(self) -> None:
        # filter disabled is allowed
        violations = check_i1_primary_set_size(rally_id="r1", primary_track_ids=[])
        assert violations == []

    def test_size_3_fails(self) -> None:
        violations = check_i1_primary_set_size(rally_id="r1", primary_track_ids=[3, 7, 12])
        assert len(violations) == 1
        assert violations[0].invariant == "I-1"
        assert violations[0].rally_id == "r1"
        assert "size 3" in violations[0].detail

    def test_size_5_fails(self) -> None:
        violations = check_i1_primary_set_size(rally_id="r2", primary_track_ids=[3, 7, 12, 15, 22])
        assert len(violations) == 1
        assert violations[0].invariant == "I-1"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest analysis/tests/unit/test_pid_invariants.py::TestCheckI1PrimarySetSize -v`
Expected: FAIL with `ImportError: cannot import name 'check_i1_primary_set_size'`.

- [ ] **Step 3: Implement the check**

Append to `analysis/rallycut/tracking/pid_invariants.py`:

```python
def check_i1_primary_set_size(
    *,
    rally_id: str,
    primary_track_ids: list[int],
) -> list[Violation]:
    """I-1: primary_track_ids must have exactly 4 entries (or 0 if filter disabled)."""
    n = len(primary_track_ids)
    if n in (0, 4):
        return []
    return [
        Violation(
            invariant="I-1",
            rally_id=rally_id,
            detail=f"primary_track_ids has size {n}, expected 4 (or 0 if filter disabled)",
        )
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest analysis/tests/unit/test_pid_invariants.py::TestCheckI1PrimarySetSize -v`
Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add analysis/rallycut/tracking/pid_invariants.py analysis/tests/unit/test_pid_invariants.py
git commit -m "feat(pid-invariants): I-1 primary_track_ids size check"
```

---

## Task 3: I-2 (positions_json track IDs ⊆ primary)

**Files:**
- Modify: `analysis/rallycut/tracking/pid_invariants.py`
- Modify: `analysis/tests/unit/test_pid_invariants.py`

- [ ] **Step 1: Write the failing test**

Append to `test_pid_invariants.py`:

```python
from rallycut.tracking.pid_invariants import check_i2_positions_in_primary


class TestCheckI2PositionsInPrimary:
    def test_clean_passes(self) -> None:
        positions = [
            {"trackId": 3, "frameNumber": 0},
            {"trackId": 7, "frameNumber": 0},
            {"trackId": 12, "frameNumber": 1},
        ]
        violations = check_i2_positions_in_primary(
            rally_id="r1", primary_track_ids=[3, 7, 12, 15], positions_json=positions,
        )
        assert violations == []

    def test_empty_passes(self) -> None:
        violations = check_i2_positions_in_primary(
            rally_id="r1", primary_track_ids=[3, 7, 12, 15], positions_json=[],
        )
        assert violations == []

    def test_none_passes(self) -> None:
        violations = check_i2_positions_in_primary(
            rally_id="r1", primary_track_ids=[3, 7, 12, 15], positions_json=None,
        )
        assert violations == []

    def test_non_primary_track_fails(self) -> None:
        positions = [
            {"trackId": 3, "frameNumber": 0},
            {"trackId": 99, "frameNumber": 0},  # non-primary
            {"trackId": 99, "frameNumber": 1},  # same offender, second sighting
        ]
        violations = check_i2_positions_in_primary(
            rally_id="r1", primary_track_ids=[3, 7, 12, 15], positions_json=positions,
        )
        assert len(violations) == 1  # one violation per offending trackId
        assert violations[0].invariant == "I-2"
        assert "99" in violations[0].detail
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest analysis/tests/unit/test_pid_invariants.py::TestCheckI2PositionsInPrimary -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the check**

Append to `pid_invariants.py`:

```python
def check_i2_positions_in_primary(
    *,
    rally_id: str,
    primary_track_ids: list[int],
    positions_json: list[dict[str, Any]] | None,
) -> list[Violation]:
    """I-2: every trackId in positions_json must be in primary_track_ids."""
    if not positions_json or not primary_track_ids:
        return []
    primary = set(primary_track_ids)
    offenders: set[int] = set()
    for p in positions_json:
        tid = p.get("trackId")
        if tid is None:
            continue
        if int(tid) not in primary:
            offenders.add(int(tid))
    return [
        Violation(
            invariant="I-2",
            rally_id=rally_id,
            detail=f"positions_json contains non-primary trackId={tid} (primary={sorted(primary)})",
        )
        for tid in sorted(offenders)
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest analysis/tests/unit/test_pid_invariants.py::TestCheckI2PositionsInPrimary -v`
Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add analysis/rallycut/tracking/pid_invariants.py analysis/tests/unit/test_pid_invariants.py
git commit -m "feat(pid-invariants): I-2 positions_json track IDs in primary set"
```

---

## Task 4: I-3 (action attribution)

**Files:**
- Modify: `analysis/rallycut/tracking/pid_invariants.py`
- Modify: `analysis/tests/unit/test_pid_invariants.py`

- [ ] **Step 1: Write the failing test**

```python
from rallycut.tracking.pid_invariants import check_i3_action_attribution


class TestCheckI3ActionAttribution:
    def test_clean_passes(self) -> None:
        actions = [
            {"playerTrackId": 3, "action": "spike", "frame": 10},
            {"playerTrackId": -1, "action": "serve", "frame": 0, "isSynthetic": True},
        ]
        violations = check_i3_action_attribution(
            rally_id="r1", primary_track_ids=[3, 7, 12, 15], actions_json=actions,
        )
        assert violations == []

    def test_none_passes(self) -> None:
        violations = check_i3_action_attribution(
            rally_id="r1", primary_track_ids=[3, 7, 12, 15], actions_json=None,
        )
        assert violations == []

    def test_non_primary_attribution_fails(self) -> None:
        actions = [
            {"playerTrackId": 99, "action": "set", "frame": 5},
            {"playerTrackId": 101, "action": "dig", "frame": 8},
        ]
        violations = check_i3_action_attribution(
            rally_id="r1", primary_track_ids=[3, 7, 12, 15], actions_json=actions,
        )
        assert len(violations) == 2
        assert all(v.invariant == "I-3" for v in violations)
        assert any("99" in v.detail for v in violations)
        assert any("101" in v.detail for v in violations)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest analysis/tests/unit/test_pid_invariants.py::TestCheckI3ActionAttribution -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the check**

```python
def check_i3_action_attribution(
    *,
    rally_id: str,
    primary_track_ids: list[int],
    actions_json: list[dict[str, Any]] | None,
) -> list[Violation]:
    """I-3: every action's playerTrackId must be in primary ∪ {-1}."""
    if not actions_json or not primary_track_ids:
        return []
    allowed = set(primary_track_ids) | {-1}
    violations: list[Violation] = []
    for idx, a in enumerate(actions_json):
        tid = a.get("playerTrackId")
        if tid is None:
            continue
        if int(tid) not in allowed:
            violations.append(
                Violation(
                    invariant="I-3",
                    rally_id=rally_id,
                    detail=(
                        f"action[{idx}] playerTrackId={tid} not in primary "
                        f"{sorted(primary_track_ids)} ∪ {{-1}} "
                        f"(action={a.get('action')!r}, frame={a.get('frame')})"
                    ),
                )
            )
    return violations
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest analysis/tests/unit/test_pid_invariants.py::TestCheckI3ActionAttribution -v`
Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add analysis/rallycut/tracking/pid_invariants.py analysis/tests/unit/test_pid_invariants.py
git commit -m "feat(pid-invariants): I-3 action attribution in primary"
```

---

## Task 5: I-4 (contact attribution)

**Files:**
- Modify: `analysis/rallycut/tracking/pid_invariants.py`
- Modify: `analysis/tests/unit/test_pid_invariants.py`

- [ ] **Step 1: Write the failing test**

```python
from rallycut.tracking.pid_invariants import check_i4_contact_attribution


class TestCheckI4ContactAttribution:
    def test_clean_passes(self) -> None:
        contacts = [
            {"playerTrackId": 7, "frame": 12},
            {"playerTrackId": -1, "frame": 20},
        ]
        violations = check_i4_contact_attribution(
            rally_id="r1", primary_track_ids=[3, 7, 12, 15], contacts_json=contacts,
        )
        assert violations == []

    def test_none_passes(self) -> None:
        violations = check_i4_contact_attribution(
            rally_id="r1", primary_track_ids=[3, 7, 12, 15], contacts_json=None,
        )
        assert violations == []

    def test_non_primary_contact_fails(self) -> None:
        contacts = [{"playerTrackId": 88, "frame": 30}]
        violations = check_i4_contact_attribution(
            rally_id="r1", primary_track_ids=[3, 7, 12, 15], contacts_json=contacts,
        )
        assert len(violations) == 1
        assert violations[0].invariant == "I-4"
        assert "88" in violations[0].detail
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest analysis/tests/unit/test_pid_invariants.py::TestCheckI4ContactAttribution -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the check**

```python
def check_i4_contact_attribution(
    *,
    rally_id: str,
    primary_track_ids: list[int],
    contacts_json: list[dict[str, Any]] | None,
) -> list[Violation]:
    """I-4: every contact's playerTrackId must be in primary ∪ {-1}."""
    if not contacts_json or not primary_track_ids:
        return []
    allowed = set(primary_track_ids) | {-1}
    violations: list[Violation] = []
    for idx, c in enumerate(contacts_json):
        tid = c.get("playerTrackId")
        if tid is None:
            continue
        if int(tid) not in allowed:
            violations.append(
                Violation(
                    invariant="I-4",
                    rally_id=rally_id,
                    detail=(
                        f"contact[{idx}] playerTrackId={tid} not in primary "
                        f"{sorted(primary_track_ids)} ∪ {{-1}} (frame={c.get('frame')})"
                    ),
                )
            )
    return violations
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest analysis/tests/unit/test_pid_invariants.py::TestCheckI4ContactAttribution -v`
Expected: 3 PASSED.

- [ ] **Step 5: Commit**

```bash
git add analysis/rallycut/tracking/pid_invariants.py analysis/tests/unit/test_pid_invariants.py
git commit -m "feat(pid-invariants): I-4 contact attribution in primary"
```

---

## Task 6: I-5 (trackToPlayer total over primary)

**Files:**
- Modify: `analysis/rallycut/tracking/pid_invariants.py`
- Modify: `analysis/tests/unit/test_pid_invariants.py`

- [ ] **Step 1: Write the failing test**

```python
from rallycut.tracking.pid_invariants import check_i5_track_to_player_total


class TestCheckI5TrackToPlayerTotal:
    def test_clean_total_passes(self) -> None:
        # Note: trackToPlayer keys are str in JSON
        violations = check_i5_track_to_player_total(
            rally_id="r1",
            primary_track_ids=[3, 7, 12, 15],
            track_to_player={"3": 1, "7": 2, "12": 3, "15": 4},
        )
        assert violations == []

    def test_missing_primary_fails(self) -> None:
        violations = check_i5_track_to_player_total(
            rally_id="r1",
            primary_track_ids=[3, 7, 12, 15],
            track_to_player={"3": 1, "7": 2, "12": 3},  # 15 missing
        )
        assert len(violations) == 1
        assert violations[0].invariant == "I-5"
        assert "15" in violations[0].detail

    def test_pid_out_of_range_fails(self) -> None:
        violations = check_i5_track_to_player_total(
            rally_id="r1",
            primary_track_ids=[3, 7, 12, 15],
            track_to_player={"3": 1, "7": 2, "12": 3, "15": 7},  # 7 not in {1..4}
        )
        assert len(violations) == 1
        assert violations[0].invariant == "I-5"
        assert "pid=7" in violations[0].detail or "7" in violations[0].detail

    def test_empty_mapping_with_empty_primary_passes(self) -> None:
        violations = check_i5_track_to_player_total(
            rally_id="r1", primary_track_ids=[], track_to_player={},
        )
        assert violations == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest analysis/tests/unit/test_pid_invariants.py::TestCheckI5TrackToPlayerTotal -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the check**

```python
def check_i5_track_to_player_total(
    *,
    rally_id: str,
    primary_track_ids: list[int],
    track_to_player: dict[str, int] | None,
) -> list[Violation]:
    """I-5: trackToPlayer must map every primary_track_id to a PID in {1..4}."""
    if not primary_track_ids:
        return []
    mapping = track_to_player or {}
    # Normalize keys: JSON serializes int keys as strings.
    normalized = {int(k): int(v) for k, v in mapping.items()}
    violations: list[Violation] = []
    for tid in primary_track_ids:
        if tid not in normalized:
            violations.append(
                Violation(
                    invariant="I-5",
                    rally_id=rally_id,
                    detail=f"primary track {tid} missing from trackToPlayer (have keys {sorted(normalized.keys())})",
                )
            )
            continue
        pid = normalized[tid]
        if pid not in (1, 2, 3, 4):
            violations.append(
                Violation(
                    invariant="I-5",
                    rally_id=rally_id,
                    detail=f"trackToPlayer[{tid}]=pid={pid}, expected 1..4",
                )
            )
    return violations
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest analysis/tests/unit/test_pid_invariants.py::TestCheckI5TrackToPlayerTotal -v`
Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add analysis/rallycut/tracking/pid_invariants.py analysis/tests/unit/test_pid_invariants.py
git commit -m "feat(pid-invariants): I-5 trackToPlayer total over primary"
```

---

## Task 7: I-6 (team_assignments total over primary)

**Files:**
- Modify: `analysis/rallycut/tracking/pid_invariants.py`
- Modify: `analysis/tests/unit/test_pid_invariants.py`

- [ ] **Step 1: Write the failing test**

```python
from rallycut.tracking.pid_invariants import check_i6_team_assignments_total


class TestCheckI6TeamAssignmentsTotal:
    def test_clean_total_passes(self) -> None:
        violations = check_i6_team_assignments_total(
            rally_id="r1",
            primary_track_ids=[3, 7, 12, 15],
            team_assignments={"3": "A", "7": "A", "12": "B", "15": "B"},
        )
        assert violations == []

    def test_missing_primary_fails(self) -> None:
        violations = check_i6_team_assignments_total(
            rally_id="r1",
            primary_track_ids=[3, 7, 12, 15],
            team_assignments={"3": "A", "7": "A", "12": "B"},
        )
        assert len(violations) == 1
        assert violations[0].invariant == "I-6"
        assert "15" in violations[0].detail

    def test_invalid_team_label_fails(self) -> None:
        violations = check_i6_team_assignments_total(
            rally_id="r1",
            primary_track_ids=[3, 7, 12, 15],
            team_assignments={"3": "A", "7": "A", "12": "B", "15": "X"},
        )
        assert len(violations) == 1
        assert violations[0].invariant == "I-6"
        assert "X" in violations[0].detail

    def test_empty_mapping_with_empty_primary_passes(self) -> None:
        violations = check_i6_team_assignments_total(
            rally_id="r1", primary_track_ids=[], team_assignments={},
        )
        assert violations == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest analysis/tests/unit/test_pid_invariants.py::TestCheckI6TeamAssignmentsTotal -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the check**

```python
def check_i6_team_assignments_total(
    *,
    rally_id: str,
    primary_track_ids: list[int],
    team_assignments: dict[str, str] | None,
) -> list[Violation]:
    """I-6: team_assignments must label every primary_track_id with team A or B."""
    if not primary_track_ids:
        return []
    mapping = team_assignments or {}
    normalized = {int(k): str(v) for k, v in mapping.items()}
    violations: list[Violation] = []
    for tid in primary_track_ids:
        if tid not in normalized:
            violations.append(
                Violation(
                    invariant="I-6",
                    rally_id=rally_id,
                    detail=f"primary track {tid} missing from team_assignments (have keys {sorted(normalized.keys())})",
                )
            )
            continue
        team = normalized[tid]
        if team not in ("A", "B"):
            violations.append(
                Violation(
                    invariant="I-6",
                    rally_id=rally_id,
                    detail=f"team_assignments[{tid}]={team!r}, expected 'A' or 'B'",
                )
            )
    return violations
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest analysis/tests/unit/test_pid_invariants.py::TestCheckI6TeamAssignmentsTotal -v`
Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add analysis/rallycut/tracking/pid_invariants.py analysis/tests/unit/test_pid_invariants.py
git commit -m "feat(pid-invariants): I-6 team_assignments total over primary"
```

---

## Task 8: I-7 (post-mapping canonical PID)

**Files:**
- Modify: `analysis/rallycut/tracking/pid_invariants.py`
- Modify: `analysis/tests/unit/test_pid_invariants.py`

I-7 is checked against the *output* of the stats pipeline (the `mapped_track_id` after applying `trackToPlayer`). The check function takes a list of mapped track IDs (already passed through the mapping) and validates each is in `{1, 2, 3, 4} ∪ {-1}`.

- [ ] **Step 1: Write the failing test**

```python
from rallycut.tracking.pid_invariants import check_i7_stats_canonical_pid


class TestCheckI7StatsCanonicalPid:
    def test_clean_passes(self) -> None:
        violations = check_i7_stats_canonical_pid(
            rally_id="r1", mapped_track_ids=[1, 2, 3, 4, -1, 1, 2],
        )
        assert violations == []

    def test_unmapped_fails(self) -> None:
        # An unmapped raw track_id (e.g., 12) leaks through
        violations = check_i7_stats_canonical_pid(
            rally_id="r1", mapped_track_ids=[1, 2, 12, 4],
        )
        assert len(violations) == 1
        assert violations[0].invariant == "I-7"
        assert "12" in violations[0].detail

    def test_collision_shifted_fails(self) -> None:
        # 101 = collision-shifted unmapped ID
        violations = check_i7_stats_canonical_pid(
            rally_id="r1", mapped_track_ids=[1, 101, 3, 4],
        )
        assert len(violations) == 1
        assert violations[0].invariant == "I-7"
        assert "101" in violations[0].detail

    def test_empty_passes(self) -> None:
        violations = check_i7_stats_canonical_pid(rally_id="r1", mapped_track_ids=[])
        assert violations == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest analysis/tests/unit/test_pid_invariants.py::TestCheckI7StatsCanonicalPid -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement the check**

```python
def check_i7_stats_canonical_pid(
    *,
    rally_id: str,
    mapped_track_ids: list[int],
) -> list[Violation]:
    """I-7: post-mapping, every action's mapped_track_id must be in {1..4} ∪ {-1}."""
    allowed = {1, 2, 3, 4, -1}
    offenders: set[int] = set()
    for tid in mapped_track_ids:
        if int(tid) not in allowed:
            offenders.add(int(tid))
    return [
        Violation(
            invariant="I-7",
            rally_id=rally_id,
            detail=f"post-mapping mapped_track_id={tid} not in {{1..4}} ∪ {{-1}}",
        )
        for tid in sorted(offenders)
    ]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest analysis/tests/unit/test_pid_invariants.py::TestCheckI7StatsCanonicalPid -v`
Expected: 4 PASSED.

- [ ] **Step 5: Commit**

```bash
git add analysis/rallycut/tracking/pid_invariants.py analysis/tests/unit/test_pid_invariants.py
git commit -m "feat(pid-invariants): I-7 post-mapping canonical PID"
```

---

## Task 9: `run_all` orchestrator

**Files:**
- Modify: `analysis/rallycut/tracking/pid_invariants.py`
- Modify: `analysis/tests/unit/test_pid_invariants.py`

`run_all` loads from the database for a single video, runs all 7 checks per rally, and returns aggregated violations. It has its own integration-style test using a stubbed connection.

- [ ] **Step 1: Write the failing test**

Append to `test_pid_invariants.py`:

```python
from unittest.mock import MagicMock, patch

from rallycut.tracking.pid_invariants import run_all


class TestRunAll:
    def _mock_conn(
        self,
        *,
        rallies: list[tuple],
        match_analysis: dict | None,
    ) -> MagicMock:
        """Build a mock connection that yields the given rally rows + video row."""
        cur = MagicMock()
        # Two execute calls happen: one for rallies, one for video.
        # fetchall returns rallies; fetchone returns (match_analysis_json,) tuple.
        cur.fetchall.return_value = rallies
        cur.fetchone.return_value = (match_analysis,)
        cur.__enter__ = lambda self: self
        cur.__exit__ = lambda self, *a: None

        conn = MagicMock()
        conn.cursor.return_value = cur
        conn.__enter__ = lambda self: self
        conn.__exit__ = lambda self, *a: None
        return conn

    def test_clean_video_returns_no_violations(self) -> None:
        rallies = [
            (
                "r1",
                [3, 7, 12, 15],  # primary_track_ids
                [{"trackId": 3, "frameNumber": 0}],  # positions_json
                {
                    "actions": [{"playerTrackId": 3, "action": "spike", "frame": 5}],
                    "teamAssignments": {"3": "A", "7": "A", "12": "B", "15": "B"},
                },  # actions_json
                [{"playerTrackId": 3, "frame": 5}],  # contacts_json
            ),
        ]
        match_analysis = {
            "rallies": [
                {
                    "rally_id": "r1",
                    "track_to_player": {"3": 1, "7": 2, "12": 3, "15": 4},
                }
            ]
        }
        conn = self._mock_conn(rallies=rallies, match_analysis=match_analysis)

        with patch("rallycut.tracking.pid_invariants.get_connection", return_value=conn):
            violations = run_all(video_id="v1")

        assert violations == []

    def test_dirty_video_aggregates_violations(self) -> None:
        rallies = [
            (
                "r1",
                [3, 7, 12],  # I-1: only 3 primary tracks
                [{"trackId": 99, "frameNumber": 0}],  # I-2: 99 not in primary
                {
                    "actions": [{"playerTrackId": 99, "action": "spike", "frame": 5}],  # I-3
                    "teamAssignments": {"3": "A", "7": "A"},  # I-6: 12 missing
                },
                [{"playerTrackId": 88, "frame": 5}],  # I-4: 88 not in primary
            ),
        ]
        match_analysis = {
            "rallies": [
                {
                    "rally_id": "r1",
                    "track_to_player": {"3": 1, "7": 2},  # I-5: 12 missing
                }
            ]
        }
        conn = self._mock_conn(rallies=rallies, match_analysis=match_analysis)

        with patch("rallycut.tracking.pid_invariants.get_connection", return_value=conn):
            violations = run_all(video_id="v1")

        invariants_seen = {v.invariant for v in violations}
        # Expect I-1, I-2, I-3, I-4, I-5, I-6 to all fire
        assert {"I-1", "I-2", "I-3", "I-4", "I-5", "I-6"}.issubset(invariants_seen)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest analysis/tests/unit/test_pid_invariants.py::TestRunAll -v`
Expected: FAIL with `ImportError: cannot import name 'run_all'`.

- [ ] **Step 3: Implement `run_all`**

Append to `pid_invariants.py`:

```python
def run_all(*, video_id: str) -> list[Violation]:
    """Run all 7 PID invariants against a video's persisted state.

    Loads rallies + player_tracks for `video_id`, plus the video's
    match_analysis_json. Calls each check_iN function and aggregates results.
    """
    from rallycut.evaluation.tracking.db import get_connection

    rally_query = """
        SELECT
            r.id AS rally_id,
            pt.primary_track_ids,
            pt.positions_json,
            pt.actions_json,
            pt.contacts_json
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE r.video_id = %s
          AND (r.status = 'CONFIRMED' OR r.status IS NULL)
        ORDER BY r.start_ms
    """
    video_query = "SELECT match_analysis_json FROM videos WHERE id = %s"

    violations: list[Violation] = []

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(rally_query, [video_id])
            rally_rows = cur.fetchall()
            cur.execute(video_query, [video_id])
            video_row = cur.fetchone()

    match_analysis: dict[str, Any] = {}
    if video_row and video_row[0]:
        match_analysis = video_row[0] if isinstance(video_row[0], dict) else {}

    rally_to_track_to_player: dict[str, dict[str, int]] = {}
    if isinstance(match_analysis.get("rallies"), list):
        for entry in match_analysis["rallies"]:
            rid = entry.get("rally_id")
            ttp = entry.get("track_to_player")
            if rid and isinstance(ttp, dict):
                rally_to_track_to_player[rid] = ttp

    for row in rally_rows:
        rally_id, primary_raw, positions_json, actions_json, contacts_json = row
        primary_track_ids: list[int] = []
        if isinstance(primary_raw, list):
            primary_track_ids = [int(t) for t in primary_raw]

        actions_list = None
        team_assignments = None
        if isinstance(actions_json, dict):
            actions_list = actions_json.get("actions")
            team_assignments = actions_json.get("teamAssignments")

        track_to_player = rally_to_track_to_player.get(rally_id)

        violations.extend(
            check_i1_primary_set_size(
                rally_id=rally_id, primary_track_ids=primary_track_ids,
            )
        )
        violations.extend(
            check_i2_positions_in_primary(
                rally_id=rally_id,
                primary_track_ids=primary_track_ids,
                positions_json=positions_json if isinstance(positions_json, list) else None,
            )
        )
        violations.extend(
            check_i3_action_attribution(
                rally_id=rally_id,
                primary_track_ids=primary_track_ids,
                actions_json=actions_list,
            )
        )
        violations.extend(
            check_i4_contact_attribution(
                rally_id=rally_id,
                primary_track_ids=primary_track_ids,
                contacts_json=contacts_json if isinstance(contacts_json, list) else None,
            )
        )
        violations.extend(
            check_i5_track_to_player_total(
                rally_id=rally_id,
                primary_track_ids=primary_track_ids,
                track_to_player=track_to_player,
            )
        )
        violations.extend(
            check_i6_team_assignments_total(
                rally_id=rally_id,
                primary_track_ids=primary_track_ids,
                team_assignments=team_assignments,
            )
        )
        # I-7 is checked using mapped_track_ids derived from actions + track_to_player.
        # An unmapped raw track_id falls through to itself, so we rebuild the same
        # mapping logic compute_match_stats uses.
        if actions_list and track_to_player:
            normalized_ttp = {int(k): int(v) for k, v in track_to_player.items()}
            mapped_track_ids: list[int] = []
            for a in actions_list:
                tid = a.get("playerTrackId")
                if tid is None:
                    continue
                tid_int = int(tid)
                if tid_int == -1:
                    mapped_track_ids.append(-1)
                elif tid_int in normalized_ttp:
                    mapped_track_ids.append(normalized_ttp[tid_int])
                else:
                    mapped_track_ids.append(tid_int)  # fall-through (current bug)
            violations.extend(
                check_i7_stats_canonical_pid(
                    rally_id=rally_id, mapped_track_ids=mapped_track_ids,
                )
            )

    return violations
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest analysis/tests/unit/test_pid_invariants.py::TestRunAll -v`
Expected: 2 PASSED.

- [ ] **Step 5: Run the entire test file to confirm nothing regressed**

Run: `uv run pytest analysis/tests/unit/test_pid_invariants.py -v`
Expected: ALL PASSED (24+ tests).

- [ ] **Step 6: Commit**

```bash
git add analysis/rallycut/tracking/pid_invariants.py analysis/tests/unit/test_pid_invariants.py
git commit -m "feat(pid-invariants): run_all orchestrator with DB load"
```

---

## Task 10: CLI command — `audit-pid-invariants`

**Files:**
- Create: `analysis/rallycut/cli/commands/audit_pid_invariants.py`

This task does not include a unit test — the CLI is a thin shell over `run_all` (which is fully tested) and Rich/Typer rendering. We verify it manually in this task and via the `eval_cross_fixture.sh` wiring in Task 13.

- [ ] **Step 1: Create the CLI command**

```python
"""CLI: rallycut audit-pid-invariants <video-id>

Eval-time invariant audit. Exits non-zero on any error-severity violation.
"""

from __future__ import annotations

import sys

import typer
from rich.console import Console
from rich.table import Table

from rallycut.tracking.pid_invariants import run_all

console = Console()


def audit_pid_invariants_cmd(
    video_id: str = typer.Argument(..., help="Video UUID"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Suppress info output"),
) -> None:
    """Audit PID-attribution invariants for a video's match-analysis state."""
    if not quiet:
        console.print(f"[dim]Running PID-invariant audit on {video_id}…[/dim]")

    violations = run_all(video_id=video_id)

    if not violations:
        if not quiet:
            console.print("[green]✓ All invariants hold[/green]")
        raise typer.Exit(code=0)

    table = Table(title=f"PID-invariant violations — video {video_id}")
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

- [ ] **Step 2: Verify the module imports cleanly**

Run: `uv run python -c "from rallycut.cli.commands.audit_pid_invariants import audit_pid_invariants_cmd; print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add analysis/rallycut/cli/commands/audit_pid_invariants.py
git commit -m "feat(cli): audit-pid-invariants command"
```

---

## Task 11: Register CLI in `main.py`

**Files:**
- Modify: `analysis/rallycut/cli/main.py`

- [ ] **Step 1: Add import and registration**

Edit `analysis/rallycut/cli/main.py`:

After the line:
```python
from rallycut.cli.commands.analyze import app as analyze_app
```

Add:
```python
from rallycut.cli.commands.audit_pid_invariants import audit_pid_invariants_cmd
```

After the line:
```python
app.command(name="repair-identities")(repair_identities_cmd)
```

Add:
```python
app.command(name="audit-pid-invariants")(audit_pid_invariants_cmd)
```

- [ ] **Step 2: Verify CLI registration**

Run: `uv run rallycut audit-pid-invariants --help`
Expected: Typer help output describing the command, including `Audit PID-attribution invariants for a video's match-analysis state.`.

- [ ] **Step 3: Commit**

```bash
git add analysis/rallycut/cli/main.py
git commit -m "feat(cli): register audit-pid-invariants in main entry point"
```

---

## Task 12: Production fix — close I-7 leakage at `compute_match_stats:156`

**Files:**
- Modify: `analysis/rallycut/cli/commands/compute_match_stats.py:155-156`

This is the one known violation. Surgical silent-skip — production trusts producers; if a non-primary track somehow flows in, drop the action rather than counting it.

- [ ] **Step 1: Read current code**

Run: `uv run python -c "import inspect; from rallycut.cli.commands import compute_match_stats; src=inspect.getsource(compute_match_stats); print(src)" | grep -n "orig_track_id" -A 2`
Expected: lines around 155-156 showing `orig_track_id = a.get("playerTrackId", -1)` then `mapped_track_id = player_map.get(orig_track_id, orig_track_id)`.

- [ ] **Step 2: Apply the fix**

In `analysis/rallycut/cli/commands/compute_match_stats.py`, locate the action-loop body (around line 155). Replace the existing block:

```python
            orig_track_id = a.get("playerTrackId", -1)
            mapped_track_id = player_map.get(orig_track_id, orig_track_id)

            actions.append(ClassifiedAction(
```

with:

```python
            orig_track_id = a.get("playerTrackId", -1)
            if orig_track_id == -1:
                mapped_track_id = -1
            elif orig_track_id in player_map:
                mapped_track_id = player_map[orig_track_id]
            else:
                # Non-primary track leaked into actions — silent skip per
                # PID-invariant I-7. Surfaced by `rallycut audit-pid-invariants`.
                continue

            actions.append(ClassifiedAction(
```

- [ ] **Step 3: Run existing match-stats tests to confirm no regression**

Run: `uv run pytest analysis/tests/ -k "match_stats or compute_match" -v`
Expected: all pass (or zero collected if no such tests exist — that's also OK; the fix is a strictly tighter guard).

- [ ] **Step 4: Commit**

```bash
git add analysis/rallycut/cli/commands/compute_match_stats.py
git commit -m "fix(stats): silent-skip non-primary tracks (closes invariant I-7)"
```

---

## Task 13: Wire audit into `eval_cross_fixture.sh`

**Files:**
- Modify: `analysis/scripts/eval_cross_fixture.sh`

The eval script already runs match-players per video, then `measure_pid_accuracy.py`. Insert the audit between those two steps so the panel run fails fast on invariant regressions.

- [ ] **Step 1: Read the current script structure**

Run: `cat /Users/mario/Personal/Projects/RallyCut/analysis/scripts/eval_cross_fixture.sh`
Expected: a bash script with a per-video loop calling `reset_matcher_state.py`, then a match-players invocation, then `measure_pid_accuracy.py`. Note the surrounding bash style (variable names, quoting).

- [ ] **Step 2: Add audit invocation**

In `analysis/scripts/eval_cross_fixture.sh`, locate the line that calls `measure_pid_accuracy.py` (around line 76). Immediately *before* it, add:

```bash
    # PID-invariant audit (Sub-1: catches regressions in primary-set, attribution, and stats mapping)
    if ! uv run rallycut audit-pid-invariants "$vid" --quiet; then
        echo "  [FAIL] PID-invariant audit failed for $vid"
        AUDIT_FAILED=1
    fi
```

At the top of the script, near the other variable initializations (e.g., next to `set -e` or just after the usage comment), add:

```bash
AUDIT_FAILED=0
```

At the bottom of the script (after the per-video loop), add:

```bash
if [ "$AUDIT_FAILED" = "1" ]; then
    echo ""
    echo "[!] One or more videos failed the PID-invariant audit. Run:"
    echo "    uv run rallycut audit-pid-invariants <video_id>"
    echo "    for full details on the offending video."
    exit 1
fi
```

- [ ] **Step 3: Verify the script parses (no execution)**

Run: `bash -n /Users/mario/Personal/Projects/RallyCut/analysis/scripts/eval_cross_fixture.sh`
Expected: no output (no syntax errors). Non-zero exit means a typo to fix.

- [ ] **Step 4: Commit**

```bash
git add analysis/scripts/eval_cross_fixture.sh
git commit -m "chore(eval): wire audit-pid-invariants into eval_cross_fixture.sh"
```

---

## Task 14: End-to-end smoke — run audit on one panel video

**Files:** None edited; this is a manual verification step.

This is where any *unexpected* leakage (beyond I-7) surfaces. Per the spec's risk-mitigation policy: if 4+ unexpected violation classes appear, **stop and re-scope** — don't grind through fixes blind.

- [ ] **Step 1: Pick a panel video and run the audit**

Run: `uv run rallycut audit-pid-invariants 5c756c41`
Expected: either `✓ All invariants hold` (likely outcome — Task 12 closed I-7 and the others were "likely held") OR a violations table.

- [ ] **Step 2: Categorize any violations**

If violations appear:
- Note which invariants fired and how many rallies are affected.
- If only 1-3 unexpected violation classes appear: open a follow-up task to investigate and fix (out of scope for this plan; create a brief note in your session memory or a follow-up issue).
- If 4+ unexpected violation classes appear: **stop**, surface the findings to the user, and re-brainstorm scope.

If no violations: proceed.

- [ ] **Step 3: Run the full panel audit via the eval script**

Run: `bash /Users/mario/Personal/Projects/RallyCut/analysis/scripts/eval_cross_fixture.sh`
Expected: completes without `[FAIL] PID-invariant audit failed` lines; exit code 0.

- [ ] **Step 4: No commit (verification only)**

If anything failed, surface to the user before continuing.

---

## Task 15: Module docstring — final invariant catalog

**Files:**
- Modify: `analysis/rallycut/tracking/pid_invariants.py` (top docstring)

Final touch: refresh the module docstring with the implementation status of each invariant after Task 14's smoke test.

- [ ] **Step 1: Update the docstring**

In `analysis/rallycut/tracking/pid_invariants.py`, replace the existing top-of-file docstring with:

```python
"""PID-attribution invariants for the analysis pipeline.

Eval-time enforcement only: production never imports these checks. The audit
CLI (`rallycut audit-pid-invariants`) and the panel eval script wire them in.

Invariants:
  I-1: len(primary_track_ids) == 4 (or 0 if filter disabled)
  I-2: every trackId in positionsJson ∈ primary_track_ids
  I-3: every action's playerTrackId ∈ primary_track_ids ∪ {-1}
  I-4: every contact's playerTrackId ∈ primary_track_ids ∪ {-1}
  I-5: trackToPlayer is total over primary_track_ids (each maps to PID 1..4)
  I-6: team_assignments is total over primary_track_ids (team in {A, B})
  I-7: post-mapping, every action's mapped_track_id ∈ {1..4} ∪ {-1}
       (closed by silent-skip in compute_match_stats — see commit history)

Spec: docs/superpowers/specs/2026-05-08-pid-leverage-audit-sub1-design.md
"""
```

- [ ] **Step 2: Run unit tests once more to confirm nothing regressed**

Run: `uv run pytest analysis/tests/unit/test_pid_invariants.py -v`
Expected: ALL PASSED.

- [ ] **Step 3: Commit**

```bash
git add analysis/rallycut/tracking/pid_invariants.py
git commit -m "docs(pid-invariants): finalize invariant catalog in module docstring"
```

---

## Done criteria

- [ ] Audit CLI exists and runs cleanly: `uv run rallycut audit-pid-invariants <panel-video> --quiet` exits 0.
- [ ] All unit tests pass: `uv run pytest analysis/tests/unit/test_pid_invariants.py -v` shows all green.
- [ ] `eval_cross_fixture.sh` runs the audit per video and surfaces failures.
- [ ] `compute_match_stats.py:155-156` closes the I-7 fall-through.
- [ ] Module docstring lists all 7 invariants with status.

## Out of scope (per spec)

- Coherence rules (3-contact, alternating teams, server ≠ next contact) — Sub-2.
- Web/API debug surface — Sub-3.
- Re-enabling point-winner detector.
- Any production hot-path logging or defensive guards beyond the surgical silent-skip.
