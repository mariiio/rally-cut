# Within-Team Proximity Swap (v3.1)

**Date:** 2026-05-11
**Status:** Design — pending implementation plan
**Workstream context:** Sub-iteration of v3 (`adaptive-candidate-window v3.0`). v3.0 widened the candidate pool to include late-tracked servers (validated: 3 of 4 absent-GT cases now have GT in candidates). But end-to-end attribution didn't move because `reattribute_players` Pass 2 skips any action where `current_team == expected_team` — for the recovered cases (which are wrong-of-two-teammates within the same team), the team check passes and Pass 2 leaves the wrong attribution alone. v3.1 adds a new Pass 2c that handles the within-team case.

## Goal

Recover 2 of 4 absent-GT serve cases (the rank-1 cases: gigi/5b6f0474, wawa/06c13117) on the 3 fresh-GT videos by adding a within-team proximity swap pass to `reattribute_players`. Targets +2 correct attributions → 60.3% → 61.8%. Env flag `WITHIN_TEAM_PROXIMITY_SWAP` (default ON).

## Motivation

### What v3.0 left on the table

After v3.0 ship + regeneration on the 3 GT videos:
- 3 of 4 absent-GT cases now have GT in `Contact.player_candidates` (G-E PASS).
- BUT: attribution accuracy is unchanged at 60.3% (G-A FAIL).
- Root cause: `reattribute_players` Pass 2 (`action_classifier.py:2917+`) has:
  ```python
  if current_team is not None and current_team == expected_team:
      continue
  ```
- For 6 of 8 absent-GT serve errors, both the wrong and correct attribution are on the SAME team. Pass 2's team check passes → skip. The new wider candidate list contains GT but no code considers within-team swapping.

### Where v3.1 will work

Per the rank-of-GT re-validation post v3.0 regeneration:

| Case | Pre-v3 GT rank | Post-v3 GT rank | Current attribution | v3.1 fires? |
|---|---|---|---|---|
| gigi/5b6f0474 | absent | **rank 1** | rank-N same-team | ✓ swap to rank-1 = GT |
| wawa/06c13117 | absent | **rank 1** | rank-N same-team | ✓ swap to rank-1 = GT |
| gigi/72c8229b | absent | rank 2 | rank 1 same-team | ✗ current is rank-1; no swap |
| gigi/bc9345c1 | absent | still absent (gap=85) | n/a | ✗ |

Expected: 2 of 4 cases convert from wrong to correct.

### Why this is the minimal viable fix

- Targets ONLY the rank-1 case: GT became proximity-best after v3.0's window widening. The CURRENT attribution is non-rank-1 within the same team. Swapping to rank-1 is the obvious correction.
- Does NOT touch the rank-2 case (would require additional signal — server consistency across rallies, pose, etc. Deferred).
- Does NOT touch cross-team swaps (v1 team-chain predicate + Pass 2 own that domain).
- Tight by design: ~15 lines of code in one Pass + tests.

## Scope

### In scope

- Add Pass 2c block to `reattribute_players` in `analysis/rallycut/tracking/action_classifier.py`, immediately AFTER the existing Pass 2 (line ~3024, the `if n_reattributed > 0:` log line).
- Gate the entire block behind env flag `WITHIN_TEAM_PROXIMITY_SWAP` (default `"1"` = ON; `"0"` disables).
- Pass 2c logic:
  - For each action where `action.confidence >= 0.6` AND `action.player_track_id >= 0`:
    - Get `contact = contact_by_frame.get(action.frame)`; skip if no contact or no candidates.
    - Get `rank1_tid = contact.player_candidates[0][0]`. If `rank1_tid == action.player_track_id`, skip (already rank-1).
    - Get `current_team = team_assignments.get(action.player_track_id)` and `rank1_team = team_assignments.get(rank1_tid)`. Skip if either None or different (cross-team is Pass 2's domain).
    - Otherwise: swap `action.player_track_id = rank1_tid`. Increment counter.
  - Log "Within-team proximity swap: re-attributed N/M actions" at INFO when N > 0.
- Tests at `analysis/tests/unit/test_reattribute_within_team.py`:
  - Truth-table for the swap condition (rank-1 matches current → no swap; cross-team rank-1 → no swap; same-team rank-1 differs from current → swap).
  - Env-flag OFF preserves old behavior.
- Re-run baseline harness on 3 GT videos. Verify G-A correct_rate +2pp target.

### Out of scope

- **Within-team rank-2 preference** (would handle gigi/72c8229b). Requires a different signal (cross-rally server consistency, pose, ball trajectory at toss). Separate workstream.
- **Lowering the rally-level confidence threshold** (would unblock gigi's 0-of-7 rallies-≥-0.70 issue). Broader concern; not v3.1.
- **Cross-rally server identity** (the proper fix for the rank-2 cases). v3.2 candidate workstream.

## Design

### §1 — Where the change lives

`analysis/rallycut/tracking/action_classifier.py`, inside `reattribute_players` at line ~3025 (immediately AFTER Pass 2's `n_reattributed` log line, BEFORE Pass 3 ReID).

### §2 — Pass 2c code

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

### §3 — Composition with other passes

| Pass | Domain | Affected by v3.1? |
|---|---|---|
| Pass 1 (server exclusion) | receive attribution defense | No — Pass 1 runs before, unchanged. |
| v1 team-chain predicate | cross-team override on nearest-guard relaxation | No — v1's gates check `expected_team != current_team`; v3.1 fires only when they're equal. |
| Pass 2 (team-based) | cross-team swap for non-nearest | No — Pass 2 only fires when `current_team != expected_team`; v3.1 fires when they're equal. |
| Pass 2b (v2 joint) | REMOVED — v2 was NO-GO | n/a |
| **Pass 2c (v3.1, NEW)** | within-team rank-1 swap | This. |
| Pass 3 (ReID) | within-team via fine-tuned classifier | Composes — if Pass 3 has high-confidence ReID prediction, it overrides Pass 2c's swap. |

The order matters: v1 → Pass 2 → v3.1 → Pass 3. Each pass is conditioned on different state, so they don't conflict.

### §4 — Env flag for rollback

`WITHIN_TEAM_PROXIMITY_SWAP=0` disables the block entirely. Default `"1"` (ON). Read at call time so monkeypatch works in tests.

## Validation

### Pre-ship gates (3 GT videos)

- **G-A** Combined `correct_rate` improves by **≥ +1pp** (60.3% → ≥ 61.3%). Lower bar than v3.0's +2pp because only 2 of 4 panel cases are addressable.
- **G-B** No per-fixture regression (cece ≥ 22, gigi ≥ 35, wawa ≥ 25).
- **G-C** `wrong_unknown_team` non-increasing (0).
- **G-D** No new test failures.
- **G-E** Within-team swap log lines fire on gigi/5b6f0474 and wawa/06c13117 specifically (the rank-1 recoveries).

### Specifically validated cases

For `gigi/5b6f0474` and `wawa/06c13117`:
- Pre-v3.1: action.player_track_id is a same-team-but-wrong player.
- Post-v3.1: action.player_track_id == rank-1 of regenerated candidates == GT.

## Risk

| Risk | Mitigation |
|---|---|
| Pass 2c swaps a currently-CORRECT attribution to wrong (the rank-1 is not actually the actor) | The rank-1 candidate is by depth-corrected proximity — the same metric that drives `_find_nearest_player` (which determines initial attribution at contact-detection time). If rank-1 is GT, the original `_find_nearest_player` would also have picked rank-1. Why didn't it? Because in the original window, rank-1 was a different player. v3.0's regeneration changed the candidate list. The fact that rank-1 changed means a closer player now exists in the candidate pool — likely the GT. |
| Pass 2c interacts badly with v1 team-chain predicate | v1 only fires for cross-team mismatches. v3.1 only fires for same-team current attribution. The two operate on disjoint subsets. |
| Pass 2c interacts badly with Pass 3 ReID | Pass 3 runs AFTER v3.1 and can override v3.1's swap when the ReID classifier has high confidence. Composition order is preserved. |
| Env flag default-ON regresses something we don't see in the panel | Env flag provides instant rollback. `WITHIN_TEAM_PROXIMITY_SWAP=0` restores prior behavior. |

## Roll-out

1. Land code + tests behind env-flag-ON default.
2. Re-run baseline harness on 3 GT videos.
3. Verify G-A through G-E.
4. If green, fleet deploy via `reattribute-actions` (existing CLI; no new tooling needed).
5. Memory entry post-ship.

## Non-goals (explicit)

- NOT a within-team rank-2 preference. Would need new signal (cross-rally server consistency, pose, trajectory).
- NOT a cross-rally server identity workstream. Separate v3.2 candidate.
- NOT a confidence-threshold change for `reattribute_players`. Separate concern.

## Open questions for the implementation plan

- Whether to also track and log when Pass 2c WOULD have fired but was vetoed by a condition (for diagnostic visibility). Suggested: no — only log actual swaps to keep deploy logs compact.
- Whether to also fire Pass 2c on UNKNOWN actions (action_type == UNKNOWN). Suggested: yes, since the swap is purely proximity-based and UNKNOWN actions still have valid contact + candidates. The action confidence guard already excludes low-confidence UNKNOWNs.
