# PID Leverage Audit — Sub-1: Invariants & Leakage Fixes

**Date:** 2026-05-08
**Status:** Design — pending implementation plan
**Workstream context:** Sub-1 of 3 (Sub-2 = team-coherence validator, Sub-3 = in-editor debug surface). Sub-1 ships a regression gate + closes any leakage discovered during audit. Sub-2 and Sub-3 will be brainstormed separately after Sub-1 ships.

## Goal

Define and enforce — at evaluation time only — the canonical PID-attribution invariants across the analysis pipeline. Provide a CLI + integration test that catches regressions where actions, contacts, or stats reference track IDs outside the four primary on-court players. Close any leakage discovered during audit with surgical production fixes (silent skip, no defensive guards).

## Motivation

The user audit ("is player ID properly leveraged where relevant?") revealed a more nuanced state than expected:

- The infrastructure for primary-track filtering **is already built**: `PlayerTrack.primaryTrackIds` column, filtered `positionsJson` vs unfiltered `rawPositionsJson`, `validate_primary_track_ids` invariant guard, and `primary_track_ids` parameter threaded through every `detect_contacts` call site.
- One known gap exists: `compute_match_stats.py:156` falls through to the raw `track_id` when the `trackToPlayer` map lacks an entry. This is the only confirmed runtime leakage.
- Several other invariants (action attribution stays in primary, `trackToPlayer` is total over primary, etc.) are **likely held but never asserted**. Without explicit checks, future refactors can quietly break them.

The cost of not closing this:
- Sub-2 (coherence validator) and Sub-3 (web debug surface) build on the assumption that `playerTrackId` ∈ primary. If that assumption silently fails, both downstream layers carry hidden noise.
- A regression that introduces leakage today would not be caught by any existing test.

## Scope

**In scope:**
- New module `analysis/rallycut/tracking/pid_invariants.py` — pure functions, one per invariant, return `list[Violation]`.
- New CLI `rallycut audit-pid-invariants <video-id>` — runs all invariants, prints table, exits non-zero on errors.
- One known production fix: `compute_match_stats.py:156` (closes I-7).
- Additional production fixes for any other leakage points discovered during audit run on panel fixtures. Surgical skip pattern: `if pid not in primary_set: continue`. No new logging in production hot path.
- Unit tests (one per invariant, clean + bad input) and one panel integration test.
- Wiring into `scripts/eval_cross_fixture.sh` so panel runs assert these invariants.

**Out of scope (deferred):**
- Coherence rules (3-contact rule, alternating teams, server ≠ next contact) — Sub-2.
- Side-switch detection / cross-rally team-identity coherence — Sub-2.
- Visual debug surface (web rally editor panel) — Sub-3.
- Re-enabling the disabled point-winner detector.
- Unifying `playerTrackId == -1` semantics (synthetic vs unattributed).
- API or web changes.
- Modal tracking job code — only stages downstream of Modal.

## The invariants

Seven invariants, ordered by pipeline stage:

| # | Invariant | Where it should hold | Current status |
|---|-----------|----------------------|----------------|
| **I-1** | `len(primary_track_ids) == 4` (or 0 if filter disabled) | Post-tracking, on `PlayerTrack` row | Held — `validate_primary_track_ids` enforces in-process |
| **I-2** | Every `trackId` in `positionsJson` ∈ `primary_track_ids` | Storage time | Implied by filter; **never explicitly asserted** |
| **I-3** | Every action's `playerTrackId` ∈ `primary_track_ids` ∪ {-1} | On `actionsJson` after the full match-analysis pipeline completes (post `reattribute_actions`, pre `compute_match_stats`) | Likely held (synthetic = -1, real attribution uses filtered `_find_nearest_player`); **unverified** |
| **I-4** | Every contact's `playerTrackId` ∈ `primary_track_ids` ∪ {-1} | On `contactsJson` after the full match-analysis pipeline completes | Likely held (`_find_nearest_player` honors `primary_track_ids`); **unverified** |
| **I-5** | `trackToPlayer` is total over `primary_track_ids` (every primary maps to a PID 1–4) | On each rally entry in `matchAnalysisJson.rallies[]` after `match-players` stage | Likely held; **unverified** |
| **I-6** | `team_assignments` is total over `primary_track_ids` (every primary has a team in {0, 1}) | On each rally entry in `matchAnalysisJson.rallies[]` after `match-players` stage | Likely held; **unverified** |
| **I-7** | After `compute_match_stats` mapping, every kept action's `player_track_id` ∈ {1, 2, 3, 4} ∪ {-1} | Stats aggregation | **Known violation**: `compute_match_stats.py:156` falls through to raw track_id when unmapped |

`-1` is treated as a single allowed sentinel for "synthetic / unattributed." Pre-existing semantics; not unified in Sub-1.

## Architecture

### Module: `analysis/rallycut/tracking/pid_invariants.py`

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class Violation:
    invariant: str          # "I-3", "I-7", etc.
    rally_id: str
    detail: str             # "action[42] playerTrackId=12 not in primary {0,3,5,7}"
    severity: Literal["error", "warn"]

def check_i1_primary_set_size(track) -> list[Violation]: ...
def check_i2_positions_in_primary(track) -> list[Violation]: ...
def check_i3_action_attribution(track) -> list[Violation]: ...
def check_i4_contact_attribution(track) -> list[Violation]: ...
def check_i5_track_to_player_total(rally, primary) -> list[Violation]: ...
def check_i6_team_assignments_total(rally, primary) -> list[Violation]: ...
def check_i7_stats_canonical_pid(actions) -> list[Violation]: ...

def run_all(video_id: str, db) -> list[Violation]:
    """Orchestrator: load data, run all 7 checks, return aggregated violations."""
```

**Design choices:**
- One pure function per invariant. Explicit names (not a registry) for grep-ability and clean test boundaries.
- No I/O inside check functions. `run_all` is the only function that touches the DB.
- No mutation of input data. The module reads and reports.
- No imports into production hot paths. The module is consumed only by the audit CLI and tests.

### CLI: `rallycut audit-pid-invariants <video-id>`

- Loads the video's rallies + `PlayerTrack` rows + `matchAnalysisJson`.
- Calls `pid_invariants.run_all()`.
- Prints a Rich-formatted table: `invariant | rally_id | detail | severity`.
- Exit code: 1 if any error-severity violation, 0 otherwise.
- Standard rallycut conventions: `--quiet` flag, per-rally progress logging while loading.

### Production fix: `compute_match_stats.py:156`

Current:
```python
orig_track_id = a.get("playerTrackId", -1)
mapped_track_id = player_map.get(orig_track_id, orig_track_id)  # falls through
```

After:
```python
orig_track_id = a.get("playerTrackId", -1)
if orig_track_id == -1:
    mapped_track_id = -1
elif orig_track_id in player_map:
    mapped_track_id = player_map[orig_track_id]
else:
    continue  # silent skip — non-primary track leaked into actions
```

No logging in the production path. The audit CLI is the diagnostic surface.

### Additional production fixes (TBD per audit findings)

Same surgical-skip pattern at any other site the audit reveals. Implementation plan will enumerate after audit run on panel fixtures.

## Testing

**Unit tests** (`analysis/tests/unit/test_pid_invariants.py`)
- One test per invariant function. Two cases each: (a) clean input → zero violations, (b) constructed-bad input → expected violation count and detail.
- Constructed inputs are tiny in-memory dicts/dataclasses, not full DB fixtures. Each test runs in <50ms.

**Panel integration test** (`analysis/tests/integration/test_pid_invariants_panel.py`)
- Uses existing snapshot at `analysis/tests/fixtures/panel_player_tracks/`.
- Runs `pid_invariants.run_all()` against each of the 4 panel videos: `5c756c41`, `b5fb0594`, `854bb250`, `7d77980f`.
- Asserts zero error-severity violations on current state.
- Bootstrap: this test will fail today on I-7 until the `compute_match_stats:156` fix lands. Fix and test land together.

**Not tested:**
- No e2e or smoke tests in production paths.
- No tests requiring full Modal tracking pipeline runs.
- No performance benchmarks (invariant checks are O(rallies × actions); trivially fast).

## Rollout sequence

1. Land `pid_invariants.py` + unit tests (no production impact).
2. Land the audit CLI command + the `compute_match_stats:156` fix.
3. Run the audit against panel fixtures; categorize unexpected violations; surgical-fix each.
4. Land the panel integration test (now passing).
5. Wire into `scripts/eval_cross_fixture.sh` so panel runs fail on regressions.

## Done criteria

- Audit CLI exists and runs cleanly against the 4 panel fixtures.
- Panel integration test passes in CI.
- All 7 invariants documented in the module docstring; for each invariant that required a production fix, the docstring references the closing commit.

## Risks

1. **Audit may reveal more leakage than expected.** I-3/I-4/I-5/I-6 are unverified. If the panel test surfaces 4+ unexpected violation classes, the surgical-fix-per-site approach starts looking like whack-a-mole. **Mitigation:** if that happens, pause Sub-1 and re-scope. The audit CLI is still valuable as a diagnostic even without all fixes landing.

2. **`positions_json` invariant ambiguity (I-2).** The filter pipeline runs through several stages (post-filter → tracklet-link → match-tracker → identity-repair → remap). Some intermediate stage may legitimately write a non-primary track ID before later stages clean it up. **Mitigation:** during implementation, if I-2 reveals a legitimate intermediate state, narrow the invariant to "after stage N" rather than "always."

3. **Panel-only validation.** Four videos. Indoor-only or low-confidence-rally leakage paths won't be caught. **Mitigation:** Sub-1 ships a regression gate, not a comprehensive validator. Broader fixture coverage is a Sub-2/Sub-3 concern.

4. **`-1` overloading.** Used for both "synthetic" and "no detection." Invariants treat it as a single allowed sentinel. If audit reveals violations are actually `-1`-vs-`-2`-vs-other sentinels, accept the noise; do not unify in Sub-1.

## Open questions (deferred to implementation)

- The complete list of additional leakage points (depends on audit run).
- Whether I-2 needs to split into "raw filter output" vs. "post-remap state."
- Whether the audit CLI should write a JSON report file in addition to stdout + exit code.

These are small enough that the implementation plan can decide them without re-brainstorming.

## File-change summary (estimate)

- **New (2):** `analysis/rallycut/tracking/pid_invariants.py`, `analysis/rallycut/cli/commands/audit_pid_invariants.py`
- **Edited (2-4):** `analysis/rallycut/cli/commands/compute_match_stats.py` (certain), additional sites if audit reveals leakage, `scripts/eval_cross_fixture.sh` (eval wiring), CLI registration in the rallycut entry point
- **New tests (2):** `analysis/tests/unit/test_pid_invariants.py`, `analysis/tests/integration/test_pid_invariants_panel.py`
- **Total:** ~6-8 file changes for the certain part. More if audit reveals additional leakage.
