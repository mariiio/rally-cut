# classify_teams 2v2-Fallback Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the existing median-index fallback in `classify_teams` to fire on any non-2v2 partition, ensuring future tracking runs always emit 2v2 team assignments for 4-track rallies.

**Architecture:** One-line conditional change in `analysis/rallycut/tracking/player_filter.py:721`. The fallback body (Y-rank + median-index split) is unchanged — already tested for the 0v4 case. New unit tests cover 1A+3B and 3A+1B inputs, plus regression tests for clean 2v2 and edge cases.

**Tech Stack:** Python 3.11, pytest, numpy, existing `PlayerPosition` dataclass.

**Spec:** `docs/superpowers/specs/2026-05-10-classify-teams-2v2-fallback-design.md`

---

## File Structure

**New (1):**
- `analysis/tests/unit/test_classify_teams_2v2_fallback.py` — focused unit tests for the new fallback trigger.

**Modified (1):**
- `analysis/rallycut/tracking/player_filter.py:721` — single conditional change.

---

## Task 1: TDD the 2v2-fallback extension

**Files:**
- Create: `analysis/tests/unit/test_classify_teams_2v2_fallback.py`
- Modify: `analysis/rallycut/tracking/player_filter.py:721`

- [ ] **Step 1: Write the failing tests**

Create `analysis/tests/unit/test_classify_teams_2v2_fallback.py` with this EXACT content:

```python
"""Test classify_teams forces 2v2 partition for any 4-track input.

Regression test for the producer fix that closes I-8 violations on new
tracking runs. The existing fallback at player_filter.py:721 forced 2v2
ONLY when all tracks landed on the same side (0v4). The fix extends it
to fire on any non-2v2 partition (1v3, 3v1, 0v4, 4v0).
"""

from __future__ import annotations

from rallycut.tracking.player_filter import classify_teams
from rallycut.tracking.player_tracker import PlayerPosition


def _make_position(frame: int, track_id: int, y: float) -> PlayerPosition:
    return PlayerPosition(
        frame_number=frame,
        track_id=track_id,
        x=0.5,
        y=y,
        width=0.05,
        height=0.10,
        confidence=0.9,
    )


def _build_positions(track_ys: dict[int, float], n_frames: int = 60) -> list[PlayerPosition]:
    """Build n_frames of positions per track at the given y."""
    out: list[PlayerPosition] = []
    for tid, y in track_ys.items():
        for f in range(n_frames):
            out.append(_make_position(f, tid, y))
    return out


def _team_counts(team_assignments: dict[int, int]) -> tuple[int, int]:
    """Return (near_count, far_count) where near=0, far=1."""
    near = sum(1 for v in team_assignments.values() if v == 0)
    far = sum(1 for v in team_assignments.values() if v == 1)
    return near, far


class TestClassifyTeams2v2Fallback:
    def test_clean_2v2_unchanged(self) -> None:
        # Two tracks above split (near, team 0), two below (far, team 1).
        # Y-classification per track produces 2v2 directly; fallback shouldn't trigger.
        positions = _build_positions({1: 0.8, 2: 0.7, 3: 0.3, 4: 0.2})
        result = classify_teams(positions, court_split_y=0.5)
        near, far = _team_counts(result)
        assert near == 2 and far == 2
        assert result[1] == 0  # high y → near
        assert result[2] == 0
        assert result[3] == 1  # low y → far
        assert result[4] == 1

    def test_1A3B_input_forced_to_2v2(self) -> None:
        # Three tracks below the split, one above. Per-track Y classification
        # produces 1 near + 3 far. Fallback must reshuffle to 2v2.
        positions = _build_positions({1: 0.8, 2: 0.4, 3: 0.3, 4: 0.2})
        result = classify_teams(positions, court_split_y=0.5)
        near, far = _team_counts(result)
        assert near == 2 and far == 2
        # Median-index split sorts by Y. Top-2 (highest y) = near (1, 2). Bottom-2 = far (3, 4).
        assert result[1] == 0  # highest y
        assert result[2] == 0  # second highest
        assert result[3] == 1
        assert result[4] == 1

    def test_3A1B_input_forced_to_2v2(self) -> None:
        # Three tracks above the split, one below. Per-track Y classification
        # produces 3 near + 1 far. Fallback must reshuffle to 2v2.
        positions = _build_positions({1: 0.9, 2: 0.8, 3: 0.7, 4: 0.2})
        result = classify_teams(positions, court_split_y=0.5)
        near, far = _team_counts(result)
        assert near == 2 and far == 2
        assert result[1] == 0  # highest y → near
        assert result[2] == 0  # second highest → near
        assert result[3] == 1  # third highest → far (median-index split)
        assert result[4] == 1  # lowest → far

    def test_0A4B_input_forced_to_2v2_existing_behavior(self) -> None:
        # All four tracks below split — existing 0v4 fallback case still works.
        positions = _build_positions({1: 0.4, 2: 0.3, 3: 0.2, 4: 0.1})
        result = classify_teams(positions, court_split_y=0.5)
        near, far = _team_counts(result)
        assert near == 2 and far == 2
        assert result[1] == 0  # highest among the 4 → near
        assert result[2] == 0
        assert result[3] == 1
        assert result[4] == 1

    def test_4A0B_input_forced_to_2v2_existing_behavior(self) -> None:
        # All four tracks above split — existing 4v0 fallback case still works.
        positions = _build_positions({1: 0.9, 2: 0.8, 3: 0.7, 4: 0.6})
        result = classify_teams(positions, court_split_y=0.5)
        near, far = _team_counts(result)
        assert near == 2 and far == 2
        assert result[1] == 0  # highest → near
        assert result[2] == 0
        assert result[3] == 1
        assert result[4] == 1

    def test_three_tracks_no_fallback(self) -> None:
        # 3 tracks: fallback requires len == 4. With 3 tracks, per-track Y
        # classification stands. 1 above, 2 below → 1 near + 2 far (NOT 2v2,
        # but acceptable because the rally is structurally I-1 territory).
        positions = _build_positions({1: 0.8, 2: 0.4, 3: 0.3})
        result = classify_teams(positions, court_split_y=0.5)
        near, far = _team_counts(result)
        # Whatever per-track Y produces — fallback does NOT fire.
        assert near + far == 3
        assert result[1] == 0  # high y → near
        assert result[2] == 1  # low y → far
        assert result[3] == 1

    def test_precomputed_assignments_takes_precedence(self) -> None:
        # When precomputed_assignments is provided, those labels are used
        # directly. If precomputed produces 2v2, fallback doesn't fire.
        positions = _build_positions({1: 0.8, 2: 0.4, 3: 0.3, 4: 0.2})
        precomputed = {1: 0, 2: 0, 3: 1, 4: 1}  # forces 2v2 partition not matching Y
        result = classify_teams(
            positions, court_split_y=0.5, precomputed_assignments=precomputed,
        )
        near, far = _team_counts(result)
        assert near == 2 and far == 2
        # Precomputed labels survive — fallback only fires for non-2v2.
        assert result == precomputed
```

- [ ] **Step 2: Run the new tests to verify they FAIL on current code**

```bash
cd analysis && uv run pytest tests/unit/test_classify_teams_2v2_fallback.py -v
```

Expected: `test_1A3B_input_forced_to_2v2` and `test_3A1B_input_forced_to_2v2` FAIL because the current fallback only fires on 0v4 / 4v0. The other tests should already pass (they cover unchanged behavior).

If `test_clean_2v2_unchanged`, `test_0A4B_input_forced_to_2v2_existing_behavior`, `test_4A0B_input_forced_to_2v2_existing_behavior`, `test_three_tracks_no_fallback`, or `test_precomputed_assignments_takes_precedence` fails, STOP — they describe existing behavior that the fix should preserve. A failure here means an unexpected regression.

- [ ] **Step 3: Apply the producer fix**

In `analysis/rallycut/tracking/player_filter.py`, locate line 721. The current code is:

```python
    # Fallback: if all players ended up on the same team (court_split_y
    # is too extreme, e.g. above/below all players), split by median Y
    # index.  This guarantees a valid 2-team split.
    if len(team_assignments) >= 4 and (near_count == 0 or far_count == 0):
```

Replace it with:

```python
    # Fallback: if the per-track Y classification did not produce an exact
    # 2v2 partition (the structurally valid shape for 4-player beach
    # volleyball), force 2v2 by sorting tracks by median-Y and splitting at
    # the midpoint. Triggers on 1v3, 3v1, 0v4, and 4v0 inputs. Closes I-8
    # at the producer (companion to Sub-1.1.E cleanup CLI for legacy data).
    if len(team_assignments) == 4 and not (near_count == 2 and far_count == 2):
```

The body of the conditional (the median-index split that follows) is unchanged.

- [ ] **Step 4: Run the new tests to verify they now PASS**

```bash
cd analysis && uv run pytest tests/unit/test_classify_teams_2v2_fallback.py -v
```

Expected: 7 PASSED.

- [ ] **Step 5: Run regression suite to confirm no other tests broke**

```bash
cd analysis && uv run pytest tests/ -k "classify_teams or player_filter or team_pair_partition" -v
```

Expected: all pass. The fix is a strict tightening of structural correctness; existing tests should be agnostic to the change (they test either the precomputed path or scenarios unchanged by the fix).

If a test fails, read its assertion — if it asserted a non-2v2 output for a 4-track input, the test was wrong (asserting structurally invalid behavior). Update it to the corrected assertion.

- [ ] **Step 6: Lint + mypy**

```bash
cd analysis && uv run ruff check rallycut/tracking/player_filter.py tests/unit/test_classify_teams_2v2_fallback.py
cd analysis && uv run mypy rallycut/tracking/player_filter.py
```

Expected: clean. If mypy flags pre-existing issues in `player_filter.py` that aren't introduced by the fix (the file is large), document them but don't fix — out of scope.

- [ ] **Step 7: Commit**

```bash
git add analysis/rallycut/tracking/player_filter.py analysis/tests/unit/test_classify_teams_2v2_fallback.py
git commit -m "fix(player-filter): classify_teams forces 2v2 for 4-track rallies (closes I-8 at producer)"
```

---

## Task 2: Update memory + final verification

**Files:** None edited; memory and verification only.

- [ ] **Step 1: Run the full pid_invariants test suite to confirm no regression**

```bash
cd analysis && uv run pytest tests/unit/test_pid_invariants.py -v
```

Expected: ALL PASSED (39 tests — 38 prior + 0 new since I-8 logic itself didn't change). The producer fix doesn't touch the audit module.

- [ ] **Step 2: Run fleet audit to confirm no fleet-state regression**

```bash
cd analysis && uv run python scripts/fleet_pid_audit.py 2>&1 > /tmp/fleet_after_producer_fix.txt
sed -n '70,$p' /tmp/fleet_after_producer_fix.txt
```

Expected: fleet state unchanged from pre-fix baseline (60 violations: 1 I-1 + 59 I-8). The producer fix only affects FUTURE tracking runs; existing data in DB is unchanged.

If the fleet state degraded (more violations than before), STOP and investigate — something unexpected happened.

- [ ] **Step 3: Update memory**

The memory directory is NOT a git repo. Just write files; no commit needed.

Create `/Users/mario/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/classify_teams_2v2_fallback_2026_05_10.md`:

```markdown
---
name: classify_teams 2v2-fallback producer fix 2026-05-10
description: Sub-1.2.A — extends classify_teams fallback to fire on any non-2v2 partition (1v3, 3v1, 0v4, 4v0)
type: project
---

# classify_teams 2v2-Fallback Producer Fix (Sub-1.2.A)

**Shipped:** 2026-05-10

**Why:** I-8 (added 2026-05-10) detects scrambled team partitions. Sub-1.1.E (`cleanup-team-labels-by-majority`) cleared 53 of 112 legacy violations via cross-rally majority cleanup. The remaining 59 violations on side-switched videos require the producer fix to prevent NEW I-8 violations on future tracking runs.

**What it does:** In `analysis/rallycut/tracking/player_filter.py:721`, extends the existing median-index fallback to fire on any non-2v2 partition (not just 0v4 / 4v0 imbalance). When per-track Y-classification produces 1v3 or 3v1, the fallback sorts tracks by median-Y and splits at the midpoint, guaranteeing 2v2 for any 4-track rally.

**How to apply:** Automatic on next tracking run. The fix takes effect for any video re-tracked after this commit. Existing DB data is unchanged (handled by Sub-1.1.E cleanup CLI).

**Layered architecture:**
- Producer (this fix) enforces 2v2 SHAPE.
- Sub-1.1.E cleanup CLI enforces team IDENTITY (which player is on which team) via cross-rally majority.
- I-8 audit guards against regression.

**Caveat:** The fallback uses each track's first `window_frames` Y positions. For late arrivers / occluded-at-rally-start / off-frame servers, this Y data may be unrepresentative — the rank-based assignment may put a player on the wrong team. Identity errors of this kind are downstream Sub-1.1.E territory; producer fix only ensures shape.

**Spec:** `docs/superpowers/specs/2026-05-10-classify-teams-2v2-fallback-design.md`
**Plan:** `docs/superpowers/plans/2026-05-10-classify-teams-2v2-fallback.md`
**Commit:** see git history for `fix(player-filter): classify_teams forces 2v2 for 4-track rallies`

**Post-fix fleet state:** unchanged from pre-fix (legacy data not affected). New tracking runs will produce 2v2-clean output by construction. Re-tracking any video that previously had I-8 violations should clear them on the producer side.
```

Update `/Users/mario/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY.md` — find the existing PID-leverage workstream entry and add Sub-1.2.A reference. Use the Edit tool with this old_string:

```markdown
- [SHIPPED] [**PID leverage audit Sub-1 2026-05-08/10**](pid_i6_team_assignments_realign_2026_05_08.md) — `audit-pid-invariants` CLI + 8 invariants (I-1..I-8) + I-7 silent-skip in `compute_match_stats`. Sub-1.1.A `player_tracker.py` re-derive after `optimize_global_identity` closes I-6 for new tracking. Sub-1.1.B [`cleanup-team-assignments`](pid_i6_cleanup_cli_2026_05_09.md), Sub-1.1.C [`cleanup-stale-attribution`](pid_stale_attribution_cleanup_2026_05_09.md), Sub-1.1.E [`cleanup-team-labels-by-majority`](pid_i8_cleanup_2026_05_10.md) — three cleanup CLIs covering legacy I-6 / I-2-I-3-I-7 / I-8 respectively. Sub-1.1.D retracked 16 I-1 rallies. I-8 added (`5a6968b`) — beach VB requires 2A+2B partition. **Fleet state (post all cleanups, 2026-05-10):** 47/70 clean (67%); 60 residual = 1 I-1 (`4b7ad71f` metadata edge) + 59 I-8 (mostly side-switched videos awaiting producer fix in `classify_teams`). Frontend `STALE_TIMEOUT_MS` 30→60 min in `4767ef3`. Audit schema-bug fix `9e6390f`.
```

and this new_string:

```markdown
- [SHIPPED] [**PID leverage audit Sub-1 2026-05-08/10**](pid_i6_team_assignments_realign_2026_05_08.md) — `audit-pid-invariants` CLI + 8 invariants (I-1..I-8) + I-7 silent-skip in `compute_match_stats`. Sub-1.1.A `player_tracker.py` re-derive after `optimize_global_identity` closes I-6 for new tracking. Sub-1.1.B [`cleanup-team-assignments`](pid_i6_cleanup_cli_2026_05_09.md), Sub-1.1.C [`cleanup-stale-attribution`](pid_stale_attribution_cleanup_2026_05_09.md), Sub-1.1.E [`cleanup-team-labels-by-majority`](pid_i8_cleanup_2026_05_10.md) — three cleanup CLIs covering legacy I-6 / I-2-I-3-I-7 / I-8 respectively. Sub-1.1.D retracked 16 I-1 rallies. I-8 added (`5a6968b`). Sub-1.2.A [classify_teams 2v2-fallback](classify_teams_2v2_fallback_2026_05_10.md) — producer fix prevents new I-8 violations on future tracking runs. **Fleet state (2026-05-10, post all cleanups + producer fix):** 47/70 clean (67%); 60 residual on legacy data = 1 I-1 (`4b7ad71f` metadata edge) + 59 I-8 (side-switched videos; will clear automatically on re-track). Frontend `STALE_TIMEOUT_MS` 30→60 min in `4767ef3`. Audit schema-bug fix `9e6390f`.
```

- [ ] **Step 4: No commit (memory dir is not a git repo)**

---

## Done criteria

- [ ] `analysis/rallycut/tracking/player_filter.py:721` updated with the new conditional.
- [ ] All 7 unit tests in `test_classify_teams_2v2_fallback.py` pass.
- [ ] No regression in existing tests (`pytest tests/ -k "classify_teams or player_filter or team_pair_partition"` clean).
- [ ] Fleet audit unchanged (60 violations, no new ones introduced).
- [ ] Memory entry recorded.

## Out of scope

- Improving the Y-signal source (full-track median, multi-window aggregation, etc.).
- Adding bbox-area or color-based tiebreaks for ambiguous Y.
- Re-tracking existing data (cleanup-team-labels-by-majority handles legacy I-8 already; new tracking benefits from this fix).
- Sub-2 (coherence validator) and Sub-3 (web debug surface).
