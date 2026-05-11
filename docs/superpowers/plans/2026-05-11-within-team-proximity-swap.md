# Within-Team Proximity Swap (v3.1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Pass 2c to `reattribute_players` that swaps within-team attributions to the proximity-best (rank-1) candidate, unlocking the v3.0 candidate-pool improvements that Pass 2's `current_team == expected_team` skip leaves on the table.

**Architecture:** ~15-line block in `analysis/rallycut/tracking/action_classifier.py` immediately after Pass 2. Behind env flag `WITHIN_TEAM_PROXIMITY_SWAP` (default ON). Composes with v1 team-chain (cross-team) and Pass 3 ReID (within-team via fine-tuned classifier) — runs between them.

**Tech Stack:** Python 3.11, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-05-11-within-team-proximity-swap-design.md`

---

## File Structure

| File | Status | Responsibility |
|---|---|---|
| `analysis/rallycut/tracking/action_classifier.py` | Modify | Add Pass 2c block in `reattribute_players` after Pass 2. ~15 lines. |
| `analysis/tests/unit/test_reattribute_within_team.py` | Create | 4 truth-table tests for Pass 2c. |
| `analysis/reports/attribution_baseline/adaptive_window_v30_2026_05_11.md` | Modify | Append v3.1 post-deploy measurement. |
| `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/adaptive_candidate_window_v30_2026_05_11.md` | Modify | Update memory entry with v3.1 details. |

---

## Task 1: Pass 2c implementation + tests

**Files:**
- Modify: `analysis/rallycut/tracking/action_classifier.py`
- Create: `analysis/tests/unit/test_reattribute_within_team.py`

- [ ] **Step 1: Create the test file with 4 truth-table tests**

Create `analysis/tests/unit/test_reattribute_within_team.py`:

```python
"""Unit tests for v3.1 within-team proximity swap (Pass 2c) in reattribute_players.

Spec: docs/superpowers/specs/2026-05-11-within-team-proximity-swap-design.md
"""

from __future__ import annotations

from unittest.mock import patch

from rallycut.tracking.action_classifier import (
    ActionType, ClassifiedAction, reattribute_players,
)
from rallycut.tracking.contact_detector import Contact


def _action(
    action_type: ActionType,
    frame: int,
    player_track_id: int,
    confidence: float = 0.9,
) -> ClassifiedAction:
    return ClassifiedAction(
        action_type=action_type,
        frame=frame,
        ball_x=0.5, ball_y=0.5, velocity=0.02,
        player_track_id=player_track_id,
        court_side="near", confidence=confidence,
    )


def _contact(
    frame: int,
    candidates: list[tuple[int, float]],
) -> Contact:
    return Contact(
        frame=frame,
        ball_x=0.5, ball_y=0.5, velocity=0.02, direction_change_deg=60.0,
        player_track_id=(candidates[0][0] if candidates else -1),
        player_distance=(candidates[0][1] if candidates else float("inf")),
        player_candidates=candidates,
        court_side="near", is_validated=True,
    )


class TestPass2cWithinTeamSwap:
    """Truth table for Pass 2c."""

    def test_swaps_when_rank1_differs_same_team(self) -> None:
        """Same-team mismatch + rank-1 differs from current → swap to rank-1."""
        # Serve currently attributed to track 2 (team A). Rank-1 candidate is
        # track 1 (also team A). Pass 2 skipped (same team). Pass 2c swaps.
        actions = [
            _action(ActionType.SERVE, frame=50, player_track_id=2),
        ]
        contacts = [
            _contact(frame=50, candidates=[(1, 0.04), (2, 0.06)]),
        ]
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}  # 1, 2 = team A; 3, 4 = team B

        with patch.dict("os.environ", {"WITHIN_TEAM_PROXIMITY_SWAP": "1"}):
            reattribute_players(actions, contacts, team_assignments)

        assert actions[0].player_track_id == 1  # swapped from 2 to 1

    def test_no_swap_when_rank1_matches_current(self) -> None:
        """If current is already rank-1, no swap."""
        actions = [
            _action(ActionType.SERVE, frame=50, player_track_id=1),
        ]
        contacts = [
            _contact(frame=50, candidates=[(1, 0.04), (2, 0.06)]),
        ]
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}

        with patch.dict("os.environ", {"WITHIN_TEAM_PROXIMITY_SWAP": "1"}):
            reattribute_players(actions, contacts, team_assignments)

        assert actions[0].player_track_id == 1  # unchanged

    def test_no_swap_when_rank1_is_cross_team(self) -> None:
        """Cross-team rank-1 → Pass 2c declines (Pass 2's domain)."""
        # Current attribution is on the EXPECTED team (correct), but rank-1 is
        # on the OTHER team. Pass 2c should not swap.
        actions = [
            _action(ActionType.SERVE, frame=50, player_track_id=2),  # team A
        ]
        contacts = [
            _contact(frame=50, candidates=[(3, 0.04), (2, 0.06)]),  # rank-1=team B
        ]
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}

        with patch.dict("os.environ", {"WITHIN_TEAM_PROXIMITY_SWAP": "1"}):
            reattribute_players(actions, contacts, team_assignments)

        assert actions[0].player_track_id == 2  # unchanged

    def test_env_flag_off_disables_swap(self) -> None:
        """WITHIN_TEAM_PROXIMITY_SWAP=0 restores pre-v3.1 behavior."""
        actions = [
            _action(ActionType.SERVE, frame=50, player_track_id=2),
        ]
        contacts = [
            _contact(frame=50, candidates=[(1, 0.04), (2, 0.06)]),
        ]
        team_assignments = {1: 0, 2: 0, 3: 1, 4: 1}

        with patch.dict("os.environ", {"WITHIN_TEAM_PROXIMITY_SWAP": "0"}):
            reattribute_players(actions, contacts, team_assignments)

        assert actions[0].player_track_id == 2  # unchanged (env flag off)
```

- [ ] **Step 2: Run the tests to verify the first one FAILS (others pass already since they assert no-change)**

Run: `cd analysis && uv run pytest tests/unit/test_reattribute_within_team.py -v`
Expected:
- `test_swaps_when_rank1_differs_same_team`: FAIL (Pass 2c not implemented yet).
- `test_no_swap_when_rank1_matches_current`: PASS (no change behavior is correct).
- `test_no_swap_when_rank1_is_cross_team`: PASS (Pass 2 handles cross-team; not our domain).
- `test_env_flag_off_disables_swap`: PASS (no Pass 2c yet, so OFF is the default state).

- [ ] **Step 3: Add Pass 2c to `reattribute_players`**

In `analysis/rallycut/tracking/action_classifier.py`, locate the end of Pass 2 — search for the line `if n_reattributed > 0:` followed by the "Re-attributed %d/%d actions using team signal" log. This is around line 3024-3028. Pass 2c goes IMMEDIATELY AFTER that log block, BEFORE Pass 3 ReID.

Insert the v3.1 block:

```python
    # Pass 2c (v3.1, 2026-05-11): within-team proximity swap.
    # Targets wrong-of-two-teammates errors where Pass 2 correctly says
    # "player is on the right team, skip" but a CLOSER same-team candidate
    # exists in the contact's player_candidates list (typically because
    # v3.0's adaptive candidate-window widening added them). Default-ON;
    # env flag WITHIN_TEAM_PROXIMITY_SWAP=0 disables.
    # Spec: docs/superpowers/specs/2026-05-11-within-team-proximity-swap-design.md
    if os.environ.get("WITHIN_TEAM_PROXIMITY_SWAP", "1") != "0":
        n_within_team = 0
        for action in actions:
            if action.confidence < 0.6 or action.player_track_id < 0:
                continue
            contact = contact_by_frame.get(action.frame)
            if contact is None or not contact.player_candidates:
                continue
            rank1_tid = contact.player_candidates[0][0]
            if rank1_tid == action.player_track_id:
                continue  # already attributing to rank-1
            current_team = team_assignments.get(action.player_track_id)
            rank1_team = team_assignments.get(rank1_tid)
            if current_team is None or rank1_team is None:
                continue
            if current_team != rank1_team:
                continue  # cross-team is Pass 2's domain
            # Same-team, but rank-1 differs from current → swap to rank-1.
            logger.info(
                "within_team_swap frame=%d action=%s old_pid=%d new_pid=%d "
                "team=%d",
                action.frame, action.action_type.value,
                action.player_track_id, rank1_tid, current_team,
            )
            action.player_track_id = rank1_tid
            n_within_team += 1
        if n_within_team > 0:
            logger.info(
                "Within-team proximity swap: re-attributed %d/%d actions",
                n_within_team, len(actions),
            )
```

Verify `import os` is at top of file (it is — used by v1's team-chain predicate).

- [ ] **Step 4: Run the new tests and verify they pass**

Run: `cd analysis && uv run pytest tests/unit/test_reattribute_within_team.py -v`
Expected: all 4 tests PASS.

- [ ] **Step 5: Run the existing suites and confirm no regression**

Run: `cd analysis && uv run pytest tests/unit -q`
Expected: 1336 tests pass (was 1332; +4 from the new test file). No failures.

Spot-check: `cd analysis && uv run pytest tests/unit/test_action_classifier.py -v`
Expected: 163 pass.

Spot-check: `cd analysis && uv run pytest tests/unit/test_action_attribution_team_chain.py -v`
Expected: 17 pass (v1 team-chain — should not regress, Pass 2c operates on disjoint subset).

- [ ] **Step 6: Run mypy + ruff**

Run: `cd analysis && uv run mypy rallycut/tracking/action_classifier.py`
Run: `cd analysis && uv run ruff check rallycut/tracking/action_classifier.py tests/unit/test_reattribute_within_team.py`
Expected: both clean.

- [ ] **Step 7: Commit**

```bash
git add analysis/rallycut/tracking/action_classifier.py analysis/tests/unit/test_reattribute_within_team.py
git commit -m "$(cat <<'EOF'
feat(reattribute): v3.1 within-team proximity swap (Pass 2c)

Adds Pass 2c to reattribute_players: when an action's current attribution
is on the correct team but is NOT the proximity-best (rank-1) candidate
in the contact's player_candidates list, swap to rank-1. Targets
wrong-of-two-teammates errors that Pass 2 (cross-team) skips.

Composes with v3.0's adaptive-candidate-window: v3.0 widens the candidate
pool to include late-tracked players (server entering after the serve
frame); v3.1 USES the wider list to actually correct within-team
attributions. Together they form the end-to-end fix for absent-GT serves
where GT becomes rank-1 in the regenerated candidate list.

Composes with v1 team-chain predicate (cross-team) and Pass 3 ReID
(within-team via fine-tuned classifier). Disjoint pre-conditions; no
conflict in execution order.

Env flag WITHIN_TEAM_PROXIMITY_SWAP=0 disables for rollback.

Spec: docs/superpowers/specs/2026-05-11-within-team-proximity-swap-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: A/B re-measurement on 3 GT videos

**Files:**
- Modify: `analysis/reports/attribution_baseline/adaptive_window_v30_2026_05_11.md` (append v3.1 section)

- [ ] **Step 1: Re-run `reattribute-actions` on the 3 GT videos**

The v3.0 regeneration already wrote new `contacts_json` in Task 5. Pass 2c will pick up the rank-1 candidates from that regenerated state.

```bash
cd analysis
uv run rallycut reattribute-actions 950fbe5d-fdad-4862-b05d-8b374bdd5ec6
uv run rallycut reattribute-actions b097dd2a-6953-4e0e-a603-5be3552f462e
uv run rallycut reattribute-actions 5c756c41-1cc1-4486-a95c-97398912cfbe
```

Look for `within_team_swap` INFO log lines. Expected: 2 fires on the 3 GT videos (gigi/5b6f0474 and wawa/06c13117 serves). Capture the output.

- [ ] **Step 2: Re-run the baseline harness**

```bash
cd analysis && uv run python -u scripts/measure_attribution_fresh_gt.py
```

Capture the output. Expected:
- Combined correct: 82 → 84 (60.3% → 61.8%) — **+2 win**.
- wrong_same_team: 9 → 7 — 2 errors converted to correct.
- Per-fixture: gigi 35 → 35, wawa 25 → 26 (+1 from wawa/06c13117), cece 22 → 22 (no rank-1 cases in cece serve set), AND gigi's confidence-gate may block some.

Append the output to `analysis/reports/attribution_baseline/adaptive_window_v30_2026_05_11.md` under a NEW section "## v3.1 post-deploy measurement".

- [ ] **Step 3: Verify the 5 pre-ship gates**

In the same section, fill in:

```markdown
### v3.1 pre-ship gates

- [ ] G-A: correct_rate +1pp (60.3% → ≥61.3%).
      Result: pre=60.3%, post=__%, delta=__pp.
- [ ] G-B: no per-fixture regression (cece ≥22, gigi ≥35, wawa ≥25).
      Result: cece __, gigi __, wawa __.
- [ ] G-C: wrong_unknown_team non-increasing (0 today).
- [ ] G-D: no new test failures.
      Result: __ tests passed.
- [ ] G-E: within-team swap log lines fire on gigi/5b6f0474 + wawa/06c13117 (the rank-1 recovery cases).
      Result: __ within_team_swap log lines in deploy output.
```

Fill in result blanks. Mark each gate PASS or FAIL.

STOP CONDITIONS:
- G-A fails (correct_rate didn't move by ≥ +1pp) → report DONE_WITH_CONCERNS.
- G-B fails (any fixture regressed) → report DONE_WITH_CONCERNS.

If all gates pass, report DONE.

- [ ] **Step 4: Run coherence audit on the 3 GT videos**

```bash
cd analysis
uv run rallycut audit-coherence-invariants 950fbe5d-fdad-4862-b05d-8b374bdd5ec6
uv run rallycut audit-coherence-invariants b097dd2a-6953-4e0e-a603-5be3552f462e
uv run rallycut audit-coherence-invariants 5c756c41-1cc1-4486-a95c-97398912cfbe
```

Capture C-1 / C-2 / C-3 counts. Expected: same as post-v3.0 (the swaps are within-team, don't change cross-team alternation). If C-2 INCREASES, investigate.

Append to the report.

- [ ] **Step 5: Commit the verification**

```bash
git add analysis/reports/attribution_baseline/adaptive_window_v30_2026_05_11.md
git commit -m "$(cat <<'EOF'
report(within-team-swap): v3.1 post-deploy verification on 3 GT videos

Pass 2c fires on the 2 rank-1 recovery cases from v3.0 (gigi/5b6f0474,
wawa/06c13117) and corrects their attribution. Combined correct_rate
post-v3.1 captured.

Spec: docs/superpowers/specs/2026-05-11-within-team-proximity-swap-design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Update memory entry

**Files:**
- Modify: `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/adaptive_candidate_window_v30_2026_05_11.md`
- Modify: `~/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY.md`

- [ ] **Step 1: Update the v3 memory entry with v3.1 results**

Append to the v3 memory file (since v3.1 is the completion of v3.0's intent):

```markdown
## v3.1: Within-Team Proximity Swap — 2026-05-11

v3.0 widened the candidate pool but `reattribute_players` Pass 2 has a
hard `current_team == expected_team → skip` early return. For
wrong-of-two-teammates errors (which are most of the v3.0-recovered
cases), the team check passes and Pass 2 leaves the wrong attribution
alone. v3.1 adds Pass 2c that swaps to the rank-1 candidate when current
attribution differs but is on the same team.

**Composition:**
- v1 team-chain (cross-team): unchanged.
- Pass 2 (cross-team for non-nearest): unchanged.
- Pass 2c (v3.1, NEW): within-team rank-1 swap.
- Pass 3 ReID: composes — overrides Pass 2c on high-confidence ReID predictions.

**Measured impact on 3 GT videos** (fill in from Task 2):
- Combined correct: 82 → __ (+__ pp).
- wrong_same_team: 9 → __.
- Fires: gigi/5b6f0474 (GT=1, was 2 → now 1), wawa/06c13117 (GT=3, was 4 → now 3).
- gigi/72c8229b's rank-2 case still un-fixed (would need cross-rally server identity, v3.2 candidate).

**Env flag:** `WITHIN_TEAM_PROXIMITY_SWAP=0` disables.
```

- [ ] **Step 2: Update MEMORY.md index entry**

The MEMORY.md entry for v3.0 was added in v3.0's Task 8. Update it to reflect v3.1:

```markdown
- [SHIPPED] [**Adaptive candidate window v3.0 + within-team swap v3.1 2026-05-11**](adaptive_candidate_window_v30_2026_05_11.md) — v3.0: forward-only fallback in `_find_nearest_player(s)` catches late-tracked servers. v3.1: Pass 2c in `reattribute_players` swaps to rank-1 same-team candidate when current differs. Panel +__pp on 3 GT. Env flags `ADAPTIVE_PLAYER_SEARCH_WINDOW`, `WITHIN_TEAM_PROXIMITY_SWAP`.
```

- [ ] **Step 3: No git commit** (memory files are out of repo)

---

## Summary of touched files

In-repo (committed Tasks 1-2):
- `analysis/rallycut/tracking/action_classifier.py` — Pass 2c block.
- `analysis/tests/unit/test_reattribute_within_team.py` — 4 truth-table tests.
- `analysis/reports/attribution_baseline/adaptive_window_v30_2026_05_11.md` — v3.1 section appended.

Out-of-repo (Task 3):
- `~/.claude/projects/.../memory/adaptive_candidate_window_v30_2026_05_11.md` — v3.1 results.
- `~/.claude/projects/.../memory/MEMORY.md` — index entry updated.

## Rollback procedure

1. Set env: `WITHIN_TEAM_PROXIMITY_SWAP=0`. Pass 2c skipped entirely.
2. If Pass 2c already wrote bad attributions to actions_json, re-run `reattribute-actions` with the env flag set — Pass 2c won't fire, and other passes (Pass 1, Pass 2, Pass 3) will re-derive their decisions on the (still-v3.0-regenerated) candidates.
3. For full rollback to pre-v3.0 state: see v3.0's rollback procedure (restore contacts_json from before v3.0 regeneration).
