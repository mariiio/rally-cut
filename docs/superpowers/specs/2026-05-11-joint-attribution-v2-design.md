# Joint Rule-Aware Attribution (v2.0)

**Date:** 2026-05-11
**Status:** Design — pending implementation plan
**Workstream context:** Second iteration of the action-attribution improvement ladder. v1 (`attribution_team_chain_v1_2026_05_11.md`) relaxed the local nearest-guard predicate; v2 reframes attribution as a rally-level joint optimization with volleyball rules as HARD constraints. Composes over the same upstream signals (contact detection F1 0.927, action classification, cross-rally PID 95.1% PERMUTED).

## Goal

Replace per-contact local attribution with a beam-search joint solver that enforces beach volleyball game rules as hard constraints over the entire rally. Targets the 17 `wrong_cross_team` errors measured on the 3 fresh-GT videos (cece, gigi, wawa) and the equivalent fleet-wide population. Default-OFF behind env flag `JOINT_ATTRIBUTION_V2` for the first ship; flip to default-ON after pre-ship gates pass.

## Motivation

### What v1 reached and where it stopped

v1 baseline (2026-05-11 fresh GT, 136 actions across 22 rallies):
- Attribution accuracy among matched actions: **76.0% (82/108)**
- `wrong_cross_team`: **17 errors** (after deploy categorical shift from `unknown_team`)
- `wrong_same_team`: 9 errors
- Per-action accuracy: serve 47%, set 67%, dig 82%, attack 83%, receive 94%

v1's `_team_chain_override_allowed` predicate fired on 4 of 22 panel rallies but produced zero net delta due to a single match-tolerance regression. The cause is structural: v1 reasons LOCALLY per-contact, applies team-chain as a SOFT tie-breaker, and only overrides when ALL of G1+G2+G3 pass. It cannot reason about rally-level rule violations.

### What v2 can do that v1 cannot

Walking the 17 cross-team errors:

- `cece/5c35e049 f=423 dig`: previous action was team-B attack → ball must cross → dig MUST be team A. Currently team B. **Hard violation of R2 (alternating possession after net-crossing).**
- `cece/f978201e f=92 receive`: serve was team B → receive MUST be team A. Currently team B. **Same R2 violation.** (User-quoted canonical bug.)
- `cece/f978201e f=195 set`: after attack, ball crossed → must be team B. Currently team A. **R2 again.**
- `gigi/39e866fd f=87 serve`: by definition, serve is by serving team. Wrong serving_team identified → wrong attribution. **R1 violation.**
- All 13 remaining cross-team errors follow the same R1 / R2 / R3 violation pattern.

A solver that REJECTS any rally-level assignment that violates volleyball rules cannot produce these 17 errors — they're rule-impossible. Within the rule-valid space, proximity-based scoring picks the proximity-best valid assignment.

**Expected impact (panel):** `wrong_cross_team` 17 → ≤ 4 (the residual is cases where `Contact.player_candidates` simply doesn't include the GT player — data quality, not solver quality). Combined `correct_rate` 60.3% → ~70% (the 13+ fixes ÷ 136 actions ≈ +10pp).

### What v2 deliberately does NOT do

- Same-team errors (9 of 26 wrong). Need cross-rally server consistency (v2.1) and role priors (v2.2). Out of v2.0 scope.
- Synthetic-serve `pl_pid=-1` cases (handled by `_attribute_synthetic_serves`, downstream).
- Missing-action layer (28 of 136 GT actions, 20.6%). Belongs to ball-tracker recall workstream.

## Scope

### In scope

- New module `analysis/rallycut/tracking/joint_attribution.py` exposing a single entry function `joint_attribute(...)`.
- Beam search over per-contact player assignments. Beam width configurable; default 50.
- Hard constraint set: R1–R5 (defined in §Design).
- Soft scoring: per-contact `-log(rank_distance + ε)` using existing `Contact.player_candidates` rank-distances.
- Fallback: if no valid assignment exists in the beam, preserve input actions and log warning.
- Env flag `JOINT_ATTRIBUTION_V2` (default `"0"` for v2.0 ship; flip to `"1"` after gates pass).
- Layered AFTER the v1 team-chain predicate inside `reattribute_players` Pass 2. v1's local fixes feed v2's starting state — they compose.
- Unit tests at `analysis/tests/unit/test_joint_attribution.py`:
  - Hard-constraint truth table (each rule fires when expected).
  - Block-and-cover edge case test (BLOCK does not flip possession, does not count toward 3-contact limit).
  - Beam-search end-to-end on the canonical bug-pattern rally (cece/f978201e style).
  - Fallback: rally where no valid assignment exists (chain too broken) — input preserved.
- A/B harness extension to evaluate `JOINT_ATTRIBUTION_V2=1` vs `=0` on the 3 GT videos.
- Pre-ship gates defined under §Validation.

### Out of scope (deferred)

- **v2.1: cross-rally server consistency.** Within a serving turn (a contiguous sequence of rallies where the same team is serving), the serve should be by the SAME player. This is a soft constraint that disambiguates the 4 wrong-of-two-teammates serve errors.
- **v2.2: role priors.** Setter usually at the net; attacker usually on attacking side; digger usually in back-court. These constrain the 5 setter/dig confusion errors.
- **v2.3: trajectory hints.** Ball direction post-contact constrains who CAN have hit it. E.g., a ball going from far-right to near-left implies the player was on the far side. Additional disambiguation signal.
- **Independent ball-only court_side helper** (would let v1's G4 come back as a true 4th gate). Different workstream; not blocking v2.

## Design

### §1 — Architecture (where the change lives)

Single new module `analysis/rallycut/tracking/joint_attribution.py`:

```python
def joint_attribute(
    actions: list[ClassifiedAction],
    contacts: list[Contact],
    team_assignments: dict[int, int],
    serving_team: int | None,
    beam_width: int = 50,
) -> list[ClassifiedAction]:
    """Joint rally-level player attribution under hard volleyball rules.

    Reads per-contact ranked Contact.player_candidates (proximity-scored,
    populated by detect_contacts). Returns actions with rewritten
    player_track_id per the beam-search-best assignment satisfying R1-R5.

    If no valid assignment exists, returns the input unchanged (never
    silently corrupts) and logs a single WARN line.
    """
```

Integration site: inside `reattribute_players` (`analysis/rallycut/tracking/action_classifier.py`), called AFTER Pass 1 server exclusion AND the team-chain v1 predicate path, but BEFORE `_attribute_synthetic_serves` and Pass 3 ReID:

```python
# ...existing Pass 1 + team-chain v1 work...

# Pass 2b (v2.0): joint rule-aware attribution.
if os.environ.get("JOINT_ATTRIBUTION_V2", "0") == "1":
    serving_team_int = (
        0 if serving_team_str == "A" else 1 if serving_team_str == "B" else None
    )
    actions = joint_attribute(
        actions, contacts, team_assignments,
        serving_team=serving_team_int,
    )

# ...existing Pass 3 ReID + synthetic serve...
```

No changes to `detect_contacts`, `_compute_expected_teams`, `_chain_integrity`, `_team_chain_override_allowed`, or any other existing function. v2 is additive.

### §2 — Hard constraint set (R1–R5)

Beam search maintains a partial assignment `(pid_0, pid_1, ..., pid_i)` and a derived rally-state `(current_team, count_consecutive_same_team, last_was_block, serving_team)`. The state is recomputed deterministically from the partial assignment + the (unchanged) action types. A candidate `pid` for action `i+1` is rejected unless ALL rules pass.

**R1 — first-action-is-serve-by-serving-team.**
- `actions[0].action_type == SERVE` (already true by pipeline construction; the rule effectively just says: the first action's pid must be on the serving team).
- `team_assignments[pid_0] == serving_team` (when `serving_team` is known).
- If `serving_team` is None at call time, infer from `pid_0` (the first action's chosen pid seeds it). All subsequent rule applications use this inferred serving_team.

**R2 — net-crossing-flips-possession.**
- If `actions[i].action_type in {SERVE, ATTACK}` and `actions[i+1].action_type != BLOCK`:
  - `team_assignments[pid_{i+1}] != team_assignments[pid_i]`

**R3 — same-side-preserves-possession.**
- If `actions[i].action_type in {RECEIVE, SET, DIG}` and `actions[i+1].action_type != BLOCK`:
  - `team_assignments[pid_{i+1}] == team_assignments[pid_i]`

**R4 — max-3-same-side.**
- `count_consecutive_same_team` resets to 0 after a SERVE, ATTACK, or BLOCK; and after a net-crossing (R2-induced) team switch.
- Otherwise increments per same-team contact.
- A candidate violates R4 if accepting it would push `count_consecutive_same_team` to 4 without first crossing the net.

**R5 — block-cover-exception.**
- `actions[i].action_type == BLOCK` is a free pass: does not flip possession (R2 skips it), does not count toward R4 limit (next action's same-team count starts fresh).
- The block itself MUST be on the receiving team relative to the prior ATTACK (i.e., the team that was NOT attacking).

**UNKNOWN action types** are passthrough — rules do not constrain them. The rally state carries over (last known non-UNKNOWN team, contact count) from the prior non-UNKNOWN action.

### §3 — Block-and-cover concrete walkthrough

Example legal sequence (post-block cover by team A):

```
[i=0] SERVE     team B  (serving_team = B)
[i=1] RECEIVE   team A  (R2: net crossed, opposite team)
[i=2] SET       team A  (R3: same team)
[i=3] ATTACK    team A  (R3: same team)
[i=4] BLOCK     team B  (R5: free pass; doesn't flip; receiving team relative to attack)
[i=5] DIG       team B  (R5: free pass continues; count starts at 1 from here)
[i=6] SET       team B  (R3 from i=5; count=2)
[i=7] ATTACK    team B  (R3 from i=6; count=3)
[i=8] RECEIVE   team A  (R2: net crossed)
```

All constraints satisfied. R4's "max 3" is enforced from the post-block contact (i=5) onward, so team B is allowed 3 contacts after the block.

### §4 — Soft scoring

Within the rule-valid candidate space, the beam picks the assignment maximizing total soft score. Per-contact score:

```python
def _score(contact: Contact, candidate_pid: int) -> float:
    """Soft score: lower rank_distance → higher score. ε avoids log(0)."""
    for tid, dist in contact.player_candidates:
        if tid == candidate_pid:
            return -math.log(dist + 1e-3)
    return float("-inf")  # candidate not in proximity list → rejected
```

`Contact.player_candidates` is the existing depth-corrected proximity ranking from `_find_nearest_players`. The scoring deliberately reuses the v1 proximity signal — v2.0's value-add is the hard constraint enforcement, not a new soft signal.

The beam search:
1. Start with `actions[0]`'s candidates as the initial beam (one element per pid in `actions[0]`'s contact candidates that pass R1).
2. For each subsequent action `i`, expand each beam element by every pid in `contact[i].player_candidates`; filter to those passing R2/R3/R4/R5; score each extension; retain top-`beam_width`.
3. After processing all actions, return the highest-scoring full assignment.
4. Fallback: if the beam empties before finishing, return the input unchanged.

For the typical rally (5-20 contacts, ≤4 candidates each, beam width 50), this is `~20 × 50 × 4 = 4000` partial expansions per rally — fast.

### §5 — Fallback + telemetry

If no valid assignment exists (beam empties or no full assignment survives), `joint_attribute` returns the input unchanged AND emits one structured log line:

```
WARN joint_attribute fallback rally=<rally_id> actions=<count> reason=beam_empty_at_index=<i>
```

This is the "never silently corrupt" rule. Causes:
- Rally state corrupted (action types missing/wrong, primary tracks incomplete).
- Bad `serving_team` identification at rally start.
- Unusual rally structure (e.g., multiple consecutive blocks the rules don't model — extension hook for v2.1+).

A counter tracks fallback rate. Pre-ship gate G-F requires the fleet-wide fallback rate ≤ 5% of rallies (if higher, the rule set is too strict or the data is more noisy than expected — investigate before relaxing).

### §6 — Env flag staging

- **v2.0 ship:** `JOINT_ATTRIBUTION_V2=0` (default OFF). Pre-ship A/B uses env=1 for the measurement.
- **After pre-ship gates pass:** flip default to `"1"`; ship as a separate small commit; document the flip in the memory entry.
- **Rollback:** `JOINT_ATTRIBUTION_V2=0` restores prior behavior (v1 team-chain predicate + existing reattribute_players passes). Fast rollback path.

## Validation

### Pre-ship gates (3 GT videos, A/B in-memory)

- **G-A** Combined `correct_rate` improves by **≥ +10pp** (60.3% → ≥ 70%).
- **G-B** `wrong_cross_team` reduces by **≥ 75%** (17 → ≤ 4).
- **G-C** No per-fixture `correct` regression (cece ≥ 22, gigi ≥ 35, wawa ≥ 25).
- **G-D** `wrong_same_team` non-increasing (9 today; v2.0 must not flip cross-team errors into same-team errors. If it does, the soft scoring is wrong.)
- **G-E** `audit-coherence-invariants` C-2 violations on the 3 videos reduce by ≥ 50% post-deploy (hard constraints prevent the very violations C-2 detects).
- **G-F** Fallback rate (beam-empty rallies) ≤ 5% on the 3 videos.

Each gate is a strict pass/fail before fleet deploy. A failed gate triggers the standard investigation loop (which gate, why, can we fix without relaxing the rules).

### Fleet deploy (Task 6 equivalent)

- Run `reattribute-actions` fleet-wide with `JOINT_ATTRIBUTION_V2=1`.
- Pre/post coherence audit. Expected: C-2 violations DROP fleet-wide (this is the workstream's primary fleet-level signal — unlike v1 where C-2 visibility-not-regression was the story, v2 should produce a real measurable C-2 reduction).
- PERMUTED PID paranoia check (unchanged from v1).

## Risk

| Risk | Mitigation |
|---|---|
| Beam search picks a rule-valid but proximity-bad assignment that introduces same-team errors | G-D gates `wrong_same_team` non-increasing. If violated, investigate which rally regressed and consider tightening soft scoring (e.g., add a multiplier penalty for non-rank-0 candidates). |
| Block-and-cover rule (R5) is too permissive — allows non-block contacts to skip the 3-count | R5 only fires when `actions[i].action_type == BLOCK`. The block must itself be on the receiving team relative to the prior attack. Unit test pins this. |
| Wrong serving_team at rally start cascades through all R1-R3 checks | If `serving_team` is None, the first action's pid seeds it. If `serving_team` is given but contradicts what the proximity-best `actions[0]` would say, beam search will explore both branches via candidate enumeration. The high-score branch wins. |
| Soft scoring's `-log(dist)` is unstable for very-close distances | `ε = 1e-3` guards. Unit tested. |
| Beam width too narrow misses globally-best assignment | Default 50 is generous for 5-20 contacts × 4 candidates. Configurable; can be increased if specific rallies show evidence of beam pruning. |
| Fallback rate > 5% (G-F fails) | Indicates the rule set is too strict or the data is noisier than expected. Don't relax rules to fit data — investigate root cause first (likely: action-type classification errors that put UNKNOWN where it shouldn't be, or missing actions breaking the chain). |

## Rollout

1. Land v2.0 code + tests behind env-flag-OFF default.
2. Run A/B harness on the 3 GT videos. Validate all 6 pre-ship gates (G-A through G-F).
3. If green, flip env-flag default to ON in a small follow-up commit.
4. Fleet deploy via `reattribute-actions`. Audit C-2 pre/post.
5. Memory entry post-ship summarizing measured impact + commits.

## Non-goals (explicit)

- NOT a rewrite of contact detection, action classification, or cross-rally PID. v2.0 reuses those as upstream signals.
- NOT a replacement for v1's team-chain predicate. v1 fires before v2; they compose.
- NOT a full Bayesian or ILP solver. Beam search is the right complexity for the rally size.
- NOT the final word on attribution accuracy. v2.1/v2.2/v2.3 will tackle same-team errors with cross-rally server consistency, role priors, and trajectory hints.

## Open questions for the implementation plan

- Best file structure: single `joint_attribution.py` vs. splitting `rules.py` + `beam.py` + `scoring.py`. Suggested: single file for v2.0 (≤500 lines total), refactor only if it grows.
- Rally-state representation: dataclass with `current_team`, `count_consecutive_same_team`, `last_was_block`, `serving_team`. Immutable; new instance per beam expansion.
- Whether to also expose `joint_attribute` as a public function for offline analysis (e.g., the A/B harness could call it directly). Suggested: yes — it's already a pure function.
