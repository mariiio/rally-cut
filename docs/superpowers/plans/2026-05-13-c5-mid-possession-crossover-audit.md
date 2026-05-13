# C-5 Mid-Possession Crossover Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new coherence invariant (C-5) that fires when consecutive actions cross teams without a possession-end action (attack/serve/block) in between. Detection-only.

**Architecture:** New function `check_c5_mid_possession_crossover` in `coherence_invariants.py` alongside C-1..C-4. Wire into `run_all` dispatch. Update the audit CLI (no code change needed if `run_all` already collects all violations). Regenerate fleet baseline.

**Tech Stack:** Python 3.11+, pytest, Typer CLI, PostgreSQL via `rallycut.evaluation.tracking.db.get_connection`.

**Spec:** `docs/superpowers/specs/2026-05-13-action-attribution-root-causes-design.md` (workstream C).

---

### Task 1: Write failing tests for C-5

**Files:**
- Create / modify: `analysis/tests/unit/test_coherence_invariants.py` — add `TestC5MidPossessionCrossover` class at the end of the file (mirrors `TestC4NoSamePlayerBackToBack` structure).

- [ ] **Step 1: Add the test class**

Append to `analysis/tests/unit/test_coherence_invariants.py`:

```python
class TestC5MidPossessionCrossover:
    """C-5: consecutive same-team possession passes (receive/set/dig) must
    not cross to the other team without a possession-end action between
    them. attack, serve, block are possession-end actions that legally
    transfer the ball.
    """

    def test_cross_team_after_attack_passes(self) -> None:
        from rallycut.tracking.coherence_invariants import (
            check_c5_mid_possession_crossover,
        )
        actions = [
            {"frame": 100, "action": "attack", "playerTrackId": 1},
            {"frame": 130, "action": "dig", "playerTrackId": 3},
        ]
        team_assignments = {"1": "A", "2": "A", "3": "B", "4": "B"}
        v = check_c5_mid_possession_crossover(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        assert v == [], "attack→dig cross-team is legal; should not fire"

    def test_cross_team_after_serve_passes(self) -> None:
        from rallycut.tracking.coherence_invariants import (
            check_c5_mid_possession_crossover,
        )
        actions = [
            {"frame": 50, "action": "serve", "playerTrackId": 1},
            {"frame": 78, "action": "receive", "playerTrackId": 3},
        ]
        team_assignments = {"1": "A", "2": "A", "3": "B", "4": "B"}
        v = check_c5_mid_possession_crossover(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        assert v == [], "serve→receive cross-team is legal"

    def test_cross_team_after_block_passes(self) -> None:
        from rallycut.tracking.coherence_invariants import (
            check_c5_mid_possession_crossover,
        )
        actions = [
            {"frame": 100, "action": "block", "playerTrackId": 1},
            {"frame": 110, "action": "dig", "playerTrackId": 3},
        ]
        team_assignments = {"1": "A", "2": "A", "3": "B", "4": "B"}
        v = check_c5_mid_possession_crossover(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        assert v == [], "block→dig cross-team is legal"

    def test_cross_team_after_receive_fires(self) -> None:
        from rallycut.tracking.coherence_invariants import (
            check_c5_mid_possession_crossover,
        )
        actions = [
            {"frame": 100, "action": "receive", "playerTrackId": 1},
            {"frame": 130, "action": "set", "playerTrackId": 3},
        ]
        team_assignments = {"1": "A", "2": "A", "3": "B", "4": "B"}
        v = check_c5_mid_possession_crossover(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        assert len(v) == 1
        assert v[0].invariant == "C-5"
        assert v[0].rally_id == "r1"
        assert "100" in v[0].detail and "130" in v[0].detail

    def test_cross_team_after_set_fires(self) -> None:
        from rallycut.tracking.coherence_invariants import (
            check_c5_mid_possession_crossover,
        )
        actions = [
            {"frame": 100, "action": "set", "playerTrackId": 1},
            {"frame": 130, "action": "attack", "playerTrackId": 3},
        ]
        team_assignments = {"1": "A", "2": "A", "3": "B", "4": "B"}
        v = check_c5_mid_possession_crossover(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        assert len(v) == 1
        assert v[0].invariant == "C-5"

    def test_cross_team_after_dig_fires(self) -> None:
        from rallycut.tracking.coherence_invariants import (
            check_c5_mid_possession_crossover,
        )
        actions = [
            {"frame": 100, "action": "dig", "playerTrackId": 1},
            {"frame": 130, "action": "set", "playerTrackId": 3},
        ]
        team_assignments = {"1": "A", "2": "A", "3": "B", "4": "B"}
        v = check_c5_mid_possession_crossover(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        assert len(v) == 1
        assert v[0].invariant == "C-5"

    def test_same_team_transitions_no_fire(self) -> None:
        from rallycut.tracking.coherence_invariants import (
            check_c5_mid_possession_crossover,
        )
        actions = [
            {"frame": 100, "action": "receive", "playerTrackId": 1},
            {"frame": 130, "action": "set", "playerTrackId": 2},
            {"frame": 160, "action": "attack", "playerTrackId": 1},
        ]
        team_assignments = {"1": "A", "2": "A", "3": "B", "4": "B"}
        v = check_c5_mid_possession_crossover(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        assert v == [], "all same-team — no crossover"

    def test_multiple_crossovers_in_rally(self) -> None:
        from rallycut.tracking.coherence_invariants import (
            check_c5_mid_possession_crossover,
        )
        actions = [
            {"frame": 100, "action": "receive", "playerTrackId": 1},
            {"frame": 130, "action": "set", "playerTrackId": 3},
            {"frame": 160, "action": "dig", "playerTrackId": 1},
        ]
        team_assignments = {"1": "A", "2": "A", "3": "B", "4": "B"}
        v = check_c5_mid_possession_crossover(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        assert len(v) == 2, "two crossovers should fire two violations"

    def test_missing_pid_skips_pair(self) -> None:
        from rallycut.tracking.coherence_invariants import (
            check_c5_mid_possession_crossover,
        )
        actions = [
            {"frame": 100, "action": "receive", "playerTrackId": None},
            {"frame": 130, "action": "set", "playerTrackId": 3},
        ]
        team_assignments = {"1": "A", "2": "A", "3": "B", "4": "B"}
        v = check_c5_mid_possession_crossover(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        assert v == []

    def test_negative_pid_skips_pair(self) -> None:
        from rallycut.tracking.coherence_invariants import (
            check_c5_mid_possession_crossover,
        )
        actions = [
            {"frame": 100, "action": "receive", "playerTrackId": -1},
            {"frame": 130, "action": "set", "playerTrackId": 3},
        ]
        team_assignments = {"1": "A", "2": "A", "3": "B", "4": "B"}
        v = check_c5_mid_possession_crossover(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        assert v == []

    def test_unmapped_pid_skips_pair(self) -> None:
        from rallycut.tracking.coherence_invariants import (
            check_c5_mid_possession_crossover,
        )
        actions = [
            {"frame": 100, "action": "receive", "playerTrackId": 99},
            {"frame": 130, "action": "set", "playerTrackId": 3},
        ]
        team_assignments = {"1": "A", "2": "A", "3": "B", "4": "B"}
        v = check_c5_mid_possession_crossover(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        assert v == []

    def test_zero_actions_skips(self) -> None:
        from rallycut.tracking.coherence_invariants import (
            check_c5_mid_possession_crossover,
        )
        v = check_c5_mid_possession_crossover(
            rally_id="r1", actions=[], team_assignments={"1": "A"},
        )
        assert v == []

    def test_one_action_skips(self) -> None:
        from rallycut.tracking.coherence_invariants import (
            check_c5_mid_possession_crossover,
        )
        actions = [{"frame": 100, "action": "receive", "playerTrackId": 1}]
        v = check_c5_mid_possession_crossover(
            rally_id="r1", actions=actions, team_assignments={"1": "A"},
        )
        assert v == []

    def test_defensive_sort_by_frame(self) -> None:
        from rallycut.tracking.coherence_invariants import (
            check_c5_mid_possession_crossover,
        )
        # Out-of-order input; the check should sort by frame.
        actions = [
            {"frame": 130, "action": "set", "playerTrackId": 3},
            {"frame": 100, "action": "receive", "playerTrackId": 1},
        ]
        team_assignments = {"1": "A", "2": "A", "3": "B", "4": "B"}
        v = check_c5_mid_possession_crossover(
            rally_id="r1", actions=actions, team_assignments=team_assignments,
        )
        assert len(v) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run from `analysis/`:

```
uv run pytest tests/unit/test_coherence_invariants.py::TestC5MidPossessionCrossover -v
```

Expected: all FAIL with `ImportError` / `AttributeError` on `check_c5_mid_possession_crossover` (not yet defined).

---

### Task 2: Implement `check_c5_mid_possession_crossover`

**Files:**
- Modify: `analysis/rallycut/tracking/coherence_invariants.py` — add the new function after `check_c4_no_same_player_back_to_back` (around line 265), before `_UPSTREAM_BLOCKER_INVARIANTS`.

- [ ] **Step 1: Add the constant for possession-end action types**

Find this near the top of the file (around line 91-93):

```python
# Action types that always end the current possession (ball crosses to the
# other side or starts a new turn).
_POSSESSION_END_ACTIONS = {"attack", "serve"}
```

**Add a new constant directly below it** (do not modify the existing `_POSSESSION_END_ACTIONS` — C-2's semantics depend on it):

```python
# Action types that legally transfer possession to the other team. C-5
# fires when a cross-team transition occurs without one of these as the
# *prev* action. `block` is included here even though it's not a
# possession-end action for C-2 (a block doesn't end your team's
# possession if your team also blocked-cover) — for C-5, block→opponent
# is always legal because the block itself transferred the ball.
_C5_POSSESSION_TRANSFER_ACTIONS = frozenset({"attack", "serve", "block"})
```

- [ ] **Step 2: Add the new function**

Add after `check_c4_no_same_player_back_to_back` (after line 264, before `_UPSTREAM_BLOCKER_INVARIANTS`):

```python
def check_c5_mid_possession_crossover(
    *,
    rally_id: str,
    actions: list[dict[str, Any]],
    team_assignments: dict[str, str],
) -> list[Violation]:
    """C-5: consecutive actions that cross teams must follow a
    possession-transfer action (attack / serve / block).

    Rationale: in volleyball, possession only changes legally via attack,
    serve, or block. A team's receive / set / dig sequence stays within
    the team. A cross-team transition without one of those three as the
    *prev* action means either: (a) the prev action was mis-typed, or
    (b) one of the two actions has wrong team attribution.

    This complements C-2 which only catches **possession-end** alternation
    failures (C-2 punts on mid-possession crossover by design — see the v1
    comment in check_c2_alternating_possessions).

    Skip semantics: pair is skipped if either ``playerTrackId`` is
    missing, -1, or not in ``team_assignments``.
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

        prev_team = team_assignments[str(prev_pid)]
        curr_team = team_assignments[str(curr_pid)]
        if prev_team == curr_team:
            continue  # same-team transitions handled by C-1/C-4

        prev_action = str(prev.get("action", ""))
        if prev_action in _C5_POSSESSION_TRANSFER_ACTIONS:
            continue  # legal crossover

        curr_action = str(curr.get("action", ""))
        violations.append(
            Violation(
                invariant="C-5",
                rally_id=rally_id,
                detail=(
                    f"action[{i - 1}] (frame {prev.get('frame')}, "
                    f"{prev_action}, player {prev_pid} team {prev_team}) "
                    f"crossed to action[{i}] (frame {curr.get('frame')}, "
                    f"{curr_action}, player {curr_pid} team {curr_team}) "
                    f"without a possession-transfer action"
                ),
                payload={
                    "prev_index": i - 1,
                    "curr_index": i,
                    "prev_frame": int(prev.get("frame", 0)),
                    "curr_frame": int(curr.get("frame", 0)),
                    "prev_action": prev_action,
                    "curr_action": curr_action,
                    "prev_team": prev_team,
                    "curr_team": curr_team,
                    "prev_player_id": int(prev_pid),
                    "curr_player_id": int(curr_pid),
                },
            )
        )
    return violations
```

- [ ] **Step 3: Run the C-5 tests, verify they pass**

```
uv run pytest tests/unit/test_coherence_invariants.py::TestC5MidPossessionCrossover -v
```

Expected: all 14 tests PASS.

- [ ] **Step 4: Commit**

```
git add analysis/rallycut/tracking/coherence_invariants.py analysis/tests/unit/test_coherence_invariants.py
git commit -m "$(cat <<'EOF'
feat(coherence): add C-5 mid-possession crossover invariant

Detection-only invariant: consecutive cross-team actions must follow a
possession-transfer action (attack/serve/block). Surfaces F3-shape
(set→attack cross-team via occlusion) and F5-shape (mid-rally attack
that's really a block) patterns that C-2's possession-end-only check
deliberately punts on.

Spec: docs/superpowers/specs/2026-05-13-action-attribution-root-causes-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Wire C-5 into `run_all` dispatch

**Files:**
- Modify: `analysis/rallycut/tracking/coherence_invariants.py` — update `run_all` to also call `check_c5_mid_possession_crossover`.

- [ ] **Step 1: Write a failing dispatch test**

Find `class TestRunAll` (around line 200-318 — the section between C-3 and C-4 tests). Add a new method to that class:

```python
    def test_run_all_dispatches_c5(self) -> None:
        """C-5 violations should surface in run_all output."""
        from unittest.mock import patch
        from rallycut.tracking.coherence_invariants import run_all
        from rallycut.tracking.pid_invariants import StaleVersionReport

        # Sequence with a C-5 violation: receive → set cross-team.
        actions = [
            {"frame": 100, "action": "receive", "playerTrackId": 1},
            {"frame": 130, "action": "set", "playerTrackId": 3},
        ]
        actions_json = {
            "actions": actions,
            "teamAssignments": {"1": "A", "2": "A", "3": "B", "4": "B"},
        }

        with patch(
            "rallycut.tracking.coherence_invariants.pid_run_all",
            return_value=(
                [],
                StaleVersionReport(
                    actions_pipeline_version="v1",
                    contacts_pipeline_version="v1",
                    skipped_stale_actions=set(),
                    skipped_stale_contacts=set(),
                ),
            ),
        ), patch(
            "rallycut.tracking.coherence_invariants.get_connection"
        ) as mock_conn:
            cur = mock_conn.return_value.__enter__.return_value.cursor.return_value.__enter__.return_value
            cur.fetchall.return_value = [("rally-1", actions_json)]
            violations, _ = run_all(video_id="vid-1")

        c5_violations = [v for v in violations if v.invariant == "C-5"]
        assert len(c5_violations) == 1
```

(Look at the existing `test_run_all_dispatches_c4` method in the same class for the exact mocking shape — copy that style.)

- [ ] **Step 2: Run to verify it fails**

```
uv run pytest tests/unit/test_coherence_invariants.py::TestRunAll::test_run_all_dispatches_c5 -v
```

Expected: FAIL — `run_all` doesn't yet call `check_c5_mid_possession_crossover`.

- [ ] **Step 3: Wire C-5 into `run_all`**

In `coherence_invariants.py`, find the `run_all` function (around line 273). Inside the loop that iterates over `rally_rows`, after the C-4 dispatch (search for `check_c4_no_same_player_back_to_back`), add:

```python
            violations.extend(
                check_c5_mid_possession_crossover(
                    rally_id=rally_id,
                    actions=actions_list,
                    team_assignments=team_assignments,
                )
            )
```

(The existing dispatch site adds C-1..C-4 the same way; mirror the indentation and the call shape.)

- [ ] **Step 4: Run all coherence tests**

```
uv run pytest tests/unit/test_coherence_invariants.py -v
```

Expected: all tests PASS (the new dispatch test + existing C-1..C-4 + existing run_all tests).

- [ ] **Step 5: Update module docstring**

At the top of `coherence_invariants.py` (line 8-14), update the rules summary to include C-5:

Find:

```python
Rules:
  C-1: A team can have at most 3 consecutive contacts before the ball
       must cross to the other team.
  C-2: Possessions alternate teams.
  C-3: First action of a rally is `serve`.
  C-4: Consecutive actions must be by different players (exception: prev
       action is `block`).
```

Replace with:

```python
Rules:
  C-1: A team can have at most 3 consecutive contacts before the ball
       must cross to the other team.
  C-2: Possessions alternate teams.
  C-3: First action of a rally is `serve`.
  C-4: Consecutive actions must be by different players (exception: prev
       action is `block`).
  C-5: Cross-team transitions must follow a possession-transfer action
       (attack / serve / block).
```

- [ ] **Step 6: Commit**

```
git add analysis/rallycut/tracking/coherence_invariants.py analysis/tests/unit/test_coherence_invariants.py
git commit -m "$(cat <<'EOF'
feat(coherence): wire C-5 into run_all dispatch

The audit CLI's run_all now collects C-5 violations alongside C-1..C-4.
No CLI changes required — the existing renderer iterates over all
violations and groups by invariant.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: Verify CLI reports C-5

**Files:**
- No changes (the existing `audit_coherence_invariants.py` CLI iterates over all violations from `run_all` and renders them in a table; new invariants surface automatically).

- [ ] **Step 1: Smoke-test CLI on F5 rally's video**

The F5 rally `99091ec6` is in video `bfd1decd` (keke). Run from `analysis/`:

```
uv run rallycut audit-coherence-invariants bfd1decd-2a7d-4ca0-a395-a0c1a7d1a07f
```

(If the UUID above doesn't match, find the real one: `PGPASSWORD=postgres psql -h localhost -p 5436 -U postgres -d rallycut -c "SELECT id, name FROM videos WHERE name='keke';"`.)

Expected: output includes at least one C-5 violation for rally `99091ec6` (frame 184 attack p2 A surrounded by team-B receives — exactly the F5 case).

- [ ] **Step 2: Smoke-test CLI on F3 rally's video**

F3 rally `0144acfb` is on the same `keke` video. The same CLI invocation should report at least one C-5 on rally `0144acfb` (frame 174 set p4 B → frame 223 attack p1 A — cross-team without a possession-transfer prev).

If you find F3's pair classifies prev as `set` (not attack/serve/block), C-5 fires. If C-5 doesn't fire on F3, re-read the catalog row to see what prev_action_type actually is — the C-5 detector relies on the type label being correct.

- [ ] **Step 3: No commit** — this task is a smoke-test only.

---

### Task 5: Fleet baseline catalog regeneration

**Files:**
- Modify: `analysis/scripts/catalog_c4_violations.py` (or create a sibling `catalog_c5_violations.py`) — extend it to also catalog C-5 if helpful, OR just re-run `audit-coherence-invariants` over all 72 videos and tally C-5 counts.

**Note:** A full pre-scored catalog (like `2026-05-13.csv` for C-4) is *not* required for ship; the spec ship gate only needs the fleet violation count. A minimal tally is sufficient.

- [ ] **Step 1: Run the audit across all videos**

From `analysis/`:

```
uv run python -c "
from rallycut.evaluation.tracking.db import get_connection
from rallycut.tracking.coherence_invariants import run_all

with get_connection() as conn:
    with conn.cursor() as cur:
        cur.execute('SELECT id, name FROM videos WHERE deleted_at IS NULL ORDER BY name')
        videos = cur.fetchall()

total_c5 = 0
videos_with_c5 = 0
for vid, name in videos:
    violations, _ = run_all(video_id=vid)
    c5 = [v for v in violations if v.invariant == 'C-5']
    if c5:
        videos_with_c5 += 1
        total_c5 += len(c5)
        print(f'  {name} ({vid[:8]}): {len(c5)} C-5')

print(f'\\nFleet total: {total_c5} C-5 violations across {videos_with_c5} videos')
"
```

Expected: per-video counts stream to stdout. Final tally line.

- [ ] **Step 2: Save the tally**

Save the output to `analysis/reports/coherence_c5_baseline_2026_05_13.md` (one short markdown file: fleet total, top-5 videos, F3/F5 confirmation).

Create the file with:

```
# C-5 fleet baseline — 2026-05-13

Total C-5 violations: <N>
Videos with at least one C-5: <K>

## Top videos by C-5 count

<paste top-5 from the per-video output>

## F3 confirmation

Rally 0144acfb (video keke): <N> C-5 violations — including the
documented F3 set→attack cross-team pattern.

## F5 confirmation

Rally 99091ec6 (video keke): <N> C-5 violations — including the
documented F5 mid-rally team-A attack pattern.
```

- [ ] **Step 3: Commit the baseline**

```
git add analysis/reports/coherence_c5_baseline_2026_05_13.md
git commit -m "$(cat <<'EOF'
docs(coherence): C-5 fleet baseline 2026-05-13

Detection-only baseline measurement. Spec: WS-C ship gate item 2.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: Update memory index

**Files:**
- Modify: `/Users/mario/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY.md` — add a single index line.
- Create: `/Users/mario/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/c5_mid_possession_crossover_2026_05_13.md` — short topic file.

- [ ] **Step 1: Create the topic file**

```markdown
---
name: c5-mid-possession-crossover-2026-05-13
description: "C-5 coherence invariant added 2026-05-13. Fires when consecutive actions cross teams without a possession-transfer action (attack/serve/block) as prev. Detection-only; complements C-2 which deliberately punts on mid-possession crossovers. Fleet baseline: <fill in from Task 5>."
metadata: 
  type: project
---

# C-5 Mid-Possession Crossover Invariant — SHIPPED 2026-05-13

Detection-only volleyball-rule audit. Companion to C-1..C-4 in
`coherence_invariants.py`. Run via `rallycut audit-coherence-invariants
<video-id>` or `run_all`.

Surfaces F3-shape (set→attack cross-team via occlusion) and F5-shape
(mid-rally attack that's really a block) patterns that C-2's
possession-end-only check ignores.

Composes with [[coherence_invariants_v1_2026_05_10]] (C-1..C-3),
[[coherence_repair_sub_2b_2026_05_13]] (C-4), and the wider
[[action-attribution-root-causes-2026-05-13]] design.

## Files

- `analysis/rallycut/tracking/coherence_invariants.py` (check + dispatch)
- `analysis/tests/unit/test_coherence_invariants.py::TestC5MidPossessionCrossover`
- `analysis/reports/coherence_c5_baseline_2026_05_13.md` (fleet baseline)
```

Fill in `<fill in from Task 5>` with the actual baseline numbers.

- [ ] **Step 2: Add the MEMORY.md index line**

In the "Current workstreams" section, add a line near the top:

```
- [SHIPPED] [**C-5 mid-possession crossover 2026-05-13**](c5_mid_possession_crossover_2026_05_13.md) — Detection-only invariant. Fires on cross-team transitions without a possession-transfer prev (attack/serve/block). Fleet baseline: <N> violations across <K> videos.
```

- [ ] **Step 3: Commit the memory update**

(Memory dir is not in the repo; just save the files. No git commit.)

---

## Ship gate verification

- [x] Tests pass (Task 2 + 3).
- [ ] Fleet baseline produced (Task 5).
- [ ] F3 and F5 rallies fire C-5 (Task 4 smoke-test).

If any of those fail, do not mark workstream C as shipped.
