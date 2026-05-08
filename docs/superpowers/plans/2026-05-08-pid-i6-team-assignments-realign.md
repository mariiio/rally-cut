# PID Invariant I-6 Team Assignments Realign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Re-derive `team_assignments` after `optimize_global_identity` so its keys match the post-global track set; close systematic I-6 violations on the panel.

**Architecture:** Single call-site addition in `player_tracker.py` immediately after `optimize_global_identity` returns. Uses the existing `classify_teams` precomputed-passthrough branch — preserves trusted labels for surviving track_ids and falls back to median-Y for newly introduced IDs. No signature changes anywhere; no new code paths.

**Tech Stack:** Python 3.11, pytest, numpy.

**Spec:** `docs/superpowers/specs/2026-05-08-pid-i6-team-assignments-realign-design.md`

---

## File Structure

**New (1):**
- `analysis/tests/unit/test_classify_teams_post_global.py` — focused unit test exercising the precomputed-passthrough behavior under simulated global-identity rewrite.

**Modified (1):**
- `analysis/rallycut/tracking/player_tracker.py` — re-derive `team_assignments` immediately after the `optimize_global_identity` call (line 1761-1768).

---

## Task 1: TDD the player_tracker fix

**Files:**
- Create: `analysis/tests/unit/test_classify_teams_post_global.py`
- Modify: `analysis/rallycut/tracking/player_tracker.py:1768` (insert ~5 lines after the closing `)` of the `optimize_global_identity` call)

- [ ] **Step 1: Write the failing test**

Create `analysis/tests/unit/test_classify_teams_post_global.py` with:

```python
"""Test classify_teams precomputed-passthrough under simulated global-identity rewrite.

Regression test for PID invariant I-6: after optimize_global_identity rewrites
track_ids on positions, classify_teams called with the OLD team_assignments as
precomputed_assignments must produce a dict whose keys match the new track set,
preserving labels for surviving tracks and classifying new tracks via median-Y.
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


class TestClassifyTeamsPostGlobalRewrite:
    def test_rewrite_track_id_carries_label_for_survivors_and_classifies_new(self) -> None:
        # Pre-global track set: {1, 2, 3, 6}. team_assignments built BEFORE rewrite.
        pre_global_team_assignments = {1: 0, 2: 1, 3: 0, 6: 1}

        # Simulate optimize_global_identity rewriting all track_id=6 entries to track_id=4
        # in the position list. Tracks {1, 2, 3} survive untouched.
        # split_y = 0.5 (court middle); near team has higher y, far team has lower y.
        positions: list[PlayerPosition] = []
        for frame in range(60):
            positions.append(_make_position(frame, track_id=1, y=0.7))   # near (team 0)
            positions.append(_make_position(frame, track_id=2, y=0.3))   # far  (team 1)
            positions.append(_make_position(frame, track_id=3, y=0.8))   # near (team 0)
            positions.append(_make_position(frame, track_id=4, y=0.2))   # far  (team 1) — was 6 pre-rewrite

        result = classify_teams(
            positions,
            court_split_y=0.5,
            precomputed_assignments=pre_global_team_assignments,
        )

        # Keys match the post-global track set (no stale 6, includes new 4).
        assert set(result.keys()) == {1, 2, 3, 4}

        # Surviving tracks keep their pre-global labels via the precomputed branch.
        assert result[1] == 0
        assert result[2] == 1
        assert result[3] == 0

        # New track (4) is classified via median-Y fallback (not in precomputed dict).
        # y=0.2 with split_y=0.5 → median_y < split_y → far team (1).
        assert result[4] == 1

    def test_no_rewrite_is_idempotent(self) -> None:
        # If global_identity rewrote nothing, every track_id is in the old dict
        # → precomputed branch covers all → result equals input dict.
        team_assignments = {1: 0, 2: 1, 3: 0, 4: 1}
        positions: list[PlayerPosition] = []
        for frame in range(60):
            positions.append(_make_position(frame, track_id=1, y=0.7))
            positions.append(_make_position(frame, track_id=2, y=0.3))
            positions.append(_make_position(frame, track_id=3, y=0.8))
            positions.append(_make_position(frame, track_id=4, y=0.2))

        result = classify_teams(
            positions,
            court_split_y=0.5,
            precomputed_assignments=team_assignments,
        )

        assert result == team_assignments

    def test_stale_keys_in_precomputed_are_dropped(self) -> None:
        # Old dict has phantom key (5) that doesn't exist in post-global positions.
        # classify_teams iterates over track_positions (built from positions),
        # so the phantom is silently dropped.
        pre_global_team_assignments = {1: 0, 2: 1, 5: 0}  # 5 is phantom
        positions: list[PlayerPosition] = []
        for frame in range(60):
            positions.append(_make_position(frame, track_id=1, y=0.7))
            positions.append(_make_position(frame, track_id=2, y=0.3))

        result = classify_teams(
            positions,
            court_split_y=0.5,
            precomputed_assignments=pre_global_team_assignments,
        )

        assert set(result.keys()) == {1, 2}
        assert 5 not in result
```

- [ ] **Step 2: Run test to verify the FIRST test currently passes (no fix needed in classify_teams)**

```bash
cd analysis && uv run pytest tests/unit/test_classify_teams_post_global.py -v
```

Expected: All 3 tests PASSED.

**Note:** `classify_teams` itself already supports the precomputed-passthrough behavior. The tests here are a regression guard — they verify the function behaves the way the fix relies on. The actual fix is the call-site addition in player_tracker.py (Step 3). After Step 3, the production code will use this verified pattern.

- [ ] **Step 3: Apply the player_tracker fix**

In `analysis/rallycut/tracking/player_tracker.py`, find the block that calls `optimize_global_identity`. The call ends at line 1768 (closing paren). Immediately after the call returns and before the next existing line (currently `_num_global_segments = global_result.num_segments` at line 1769), insert the re-derive:

Replace this block:

```python
                positions, global_result = optimize_global_identity(
                    positions,
                    team_assignments,
                    color_store,
                    court_split_y=split_y,
                    appearance_store=appearance_store,
                    learned_store=learned_store,
                )
                _num_global_segments = global_result.num_segments
                _num_global_remapped = global_result.num_remapped
```

with:

```python
                positions, global_result = optimize_global_identity(
                    positions,
                    team_assignments,
                    color_store,
                    court_split_y=split_y,
                    appearance_store=appearance_store,
                    learned_store=learned_store,
                )
                # optimize_global_identity may rewrite track_ids on positions
                # via segment reassignment to canonical anchors. Re-derive
                # team_assignments so its keys match the post-global track set:
                # surviving tracks keep their labels via precomputed-passthrough,
                # new tracks fall back to median-Y. Restores PID-invariant I-6.
                team_assignments = classify_teams(
                    positions,
                    split_y,
                    precomputed_assignments=team_assignments,
                )
                _num_global_segments = global_result.num_segments
                _num_global_remapped = global_result.num_remapped
```

The indentation must match the surrounding block (16 spaces — inside the `if` at line 1746).

`classify_teams` is already imported in this scope (lazy import at line 1510-1513).

- [ ] **Step 4: Run the focused unit test again to confirm it still passes**

```bash
cd analysis && uv run pytest tests/unit/test_classify_teams_post_global.py -v
```

Expected: 3 PASSED.

- [ ] **Step 5: Run regression suite**

```bash
cd analysis && uv run pytest tests/ -k "team or filter or tracker"
```

Expected: All pass. The fix uses an existing path (precomputed branch) — no behavior change for clean rallies.

- [ ] **Step 6: Lint + mypy**

```bash
cd analysis && uv run ruff check rallycut/tracking/player_tracker.py tests/unit/test_classify_teams_post_global.py
cd analysis && uv run mypy rallycut/tracking/player_tracker.py
```

Expected: clean (or pre-existing mypy issues unchanged — don't fix any unrelated mypy issues that may already exist on player_tracker.py; document them but leave alone).

- [ ] **Step 7: Commit**

```bash
git add analysis/rallycut/tracking/player_tracker.py analysis/tests/unit/test_classify_teams_post_global.py
git commit -m "fix(player-tracker): re-derive team_assignments after global_identity (closes I-6)"
```

---

## Task 2: Re-match panel + verify audit clean

**Files:** None edited; this is integration verification.

This task confirms the fix actually clears the panel violations. If audit-clean isn't achieved, stop and surface findings rather than proceeding.

- [ ] **Step 1: Re-match the 4 panel videos**

```bash
bash analysis/scripts/eval_cross_fixture.sh
```

Expected (in order):
1. Per-video output: `[panel video N/4]` then match-players + remap + audit logs.
2. The audit step now passes for each video (no `[FAIL] PID-invariant audit failed for $vid` lines).
3. Final `==== Done. ====` and exit code 0.

If any video still shows audit failures, run the audit standalone for full detail:

```bash
uv run rallycut audit-pid-invariants <video-id>
```

- [ ] **Step 2: Run the audit standalone on each panel video to confirm zero violations**

```bash
for v in 5c756c41-1cc1-4486-a95c-97398912cfbe b5fb0594-d64f-4a0d-bad9-de8fc36414d0 854bb250-3e91-47d2-944d-f62413e3cf45 7d77980f-3006-40e0-adc0-db491a5bb659; do
  echo "===== $v ====="
  uv run rallycut audit-pid-invariants "$v" --quiet
  echo "  (exit=$?)"
done
```

Expected: each video shows `✓ All invariants hold` (without the `--quiet` flag) or no output with exit 0.

- [ ] **Step 3: If everything is clean, no commit (this task is verification only)**

If audit is clean, proceed to Step 4. If not, **stop** and report findings — don't attempt further fixes blind.

- [ ] **Step 4: Update memory with post-fix panel-clean baseline**

Add a single line to `/Users/mario/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY.md` under the "Current workstreams" section, near the existing 2026-05-08 entries. Use this exact format:

```markdown
- [SHIPPED] [**PID I-6 fix 2026-05-08**](pid_i6_team_assignments_realign_2026_05_08.md) — Re-derive team_assignments after `optimize_global_identity`; panel audit-clean post-rematch.
```

Then create the memory file at `/Users/mario/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/pid_i6_team_assignments_realign_2026_05_08.md` with this content:

```markdown
---
name: PID I-6 team_assignments realign 2026-05-08
description: Sub-1.1.A — re-derive team_assignments after optimize_global_identity to close systematic I-6 violations
type: project
---

# PID I-6 Team Assignments Realign (Sub-1.1.A)

**Shipped:** 2026-05-08

**Why:** Sub-1's audit (`audit-pid-invariants`) surfaced I-6 violations on 27 panel rallies — primary tracks missing from `teamAssignments`. Root cause traced empirically to `player_tracker.py:1733-1768`: `classify_teams` builds `team_assignments` keyed on pre-global track IDs, then `optimize_global_identity` rewrites `track_id` on positions but doesn't return updated `team_assignments`. Stale keys persisted through `remap-track-ids` to DB.

**Fix:** Single call-site addition immediately after `optimize_global_identity` returns. Re-derive `team_assignments` via `classify_teams(post_global_positions, split_y, precomputed_assignments=team_assignments)` — surviving tracks keep labels via precomputed-passthrough; new tracks fall back to median-Y. Idempotent for the no-rewrite case.

**How to apply:** When touching `player_tracker.py` near the global-identity block, treat the re-derive call as a load-bearing invariant. If the call signature of `optimize_global_identity` ever changes to return updated `team_assignments`, the re-derive call becomes redundant and can be dropped.

**Spec:** `docs/superpowers/specs/2026-05-08-pid-i6-team-assignments-realign-design.md`
**Plan:** `docs/superpowers/plans/2026-05-08-pid-i6-team-assignments-realign.md`

**Post-fix panel state (2026-05-08):** All 4 panel videos audit-clean. `eval_cross_fixture.sh` exits 0.
```

Commit the memory updates:

```bash
git add /Users/mario/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY.md /Users/mario/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/pid_i6_team_assignments_realign_2026_05_08.md
git commit -m "memory: record PID I-6 fix and post-rematch panel-clean baseline"
```

Note: the memory directory is OUTSIDE the project repo. Verify whether the memory dir is git-managed before running `git commit` — if it's a separate repo or not under git at all, the memory write succeeds via the file system but no commit is needed/possible. Use `git status /Users/mario/.claude/projects/...` to check; if not a repo, just write the files and skip the commit.

---

## Done criteria

- [ ] `cd analysis && uv run pytest tests/unit/test_classify_teams_post_global.py -v` shows 3 PASSED.
- [ ] `cd analysis && uv run pytest tests/ -k "team or filter or tracker"` shows no regressions.
- [ ] `cd analysis && uv run ruff check rallycut/tracking/player_tracker.py tests/unit/test_classify_teams_post_global.py` clean.
- [ ] `bash analysis/scripts/eval_cross_fixture.sh` exits 0.
- [ ] All 4 panel videos pass `uv run rallycut audit-pid-invariants <video> --quiet` with exit 0.
- [ ] Memory entry recorded.

## Out of scope

- Any change to `optimize_global_identity` (out per spec — the "purer" architectural fix).
- Any change to `remap-track-ids`, `match-players`, or the audit module.
- Sub-2 (team-coherence validator) and Sub-3 (web debug surface).
