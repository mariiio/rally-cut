# Sub-2.A: Coherence Invariants v1 — Game-Rule Validator

**Date:** 2026-05-10
**Status:** Design — pending implementation plan
**Workstream context:** Sub-2.A is the first piece of the team-coherence validator (the original "concern #5: game dynamics" from the session-start request). Builds on the now-clean PID baseline established by Sub-1 (94% fleet clean post-retrack). Extends the audit framework with sequence-based volleyball-rule checks that catch CORRECTNESS issues the PID-structural audit can't see.

## Goal

Provide a CLI command — `rallycut audit-coherence-invariants <video-id>` — that walks every rally of a given video, applies 3 hard volleyball rules to the action sequence, and reports any violations. Detection only; no cleanup or correction. Establishes a measurement framework for game-dynamics gaps that future fixes can target.

## Motivation

Throughout this session we built measurement and cleanup infrastructure for **structural** PID correctness (every rally has 4 primary tracks, every track has a team label, every action's player is in primary, etc.). Eight invariants (I-1..I-8), three cleanup CLIs, and two producer fixes brought the fleet from 706 violations on 65 videos to 10 violations on 4 videos.

But "structurally correct PID attribution" doesn't imply "the action sequence makes volleyball sense." A rally where:
- One team has 4 consecutive contacts (illegal — max 3),
- Or contacts alternate weirdly (e.g., team A → team B → team A → team B without any "ball crossed" event between),
- Or the first action isn't a `serve`,

would pass all PID invariants but still represent attribution / contact-detection / classification errors that affect stats. Sub-2 surfaces these as coherence violations.

The original "concern #5" from the session-start asked: *"game dynamics make sense based on teams"*. Sub-2.A answers it with a measurement framework.

## Scope

**In scope:**
- New module `analysis/rallycut/tracking/coherence_invariants.py` mirroring the structure of `pid_invariants.py`. Imports `Violation` dataclass from `pid_invariants` for consistency.
- 3 starter rules (C-1, C-2, C-3) — see "The 3 rules" section below.
- `run_all(*, video_id: str) -> list[Violation]` orchestrator that loads from DB, dispatches to each rule, and skips rallies that have unresolved upstream PID-invariant violations.
- New CLI command `rallycut audit-coherence-invariants <video-id>` (Typer + Rich), mirroring `audit-pid-invariants` exactly.
- CLI registration in `cli/main.py`.
- Unit tests per rule (TDD-driven), mirroring `test_pid_invariants.py`.

**Out of scope:**
- Cleanup or auto-correction of coherence violations. There's no obvious mechanical fix (you can't tell which contact is wrong from rules alone). v1 is detection only.
- More than 3 rules. Defer until v1 reveals what's common.
- Cross-rally rules (e.g., "server's team consistent within a set"). Requires set-boundary detection.
- Severity tuning per rule. All v1 violations are `severity="error"`.
- Producer-side correction in `action_classifier` based on coherence findings.
- Re-attribution of contacts based on coherence rules.
- Sub-3 (web debug surface) — separate workstream.

## Architecture

### Module: `analysis/rallycut/tracking/coherence_invariants.py`

```python
"""Coherence (game-rule) invariants for the analysis pipeline.

Eval-time enforcement only: production never imports these checks. The audit
CLI (`rallycut audit-coherence-invariants`) wires them in alongside the
existing PID-structural audit (`rallycut audit-pid-invariants`).

These rules examine the ACTION SEQUENCE within a rally, applying volleyball
rules to surface attribution/classification errors that the PID-structural
audit can't see. Violations indicate either:
  - A team-label error (an action attributed to the wrong team),
  - An action-classification error (e.g., a "set" labeled as an "attack"),
  - A missing or extra action.

Rules:
  C-1: A team can have at most 3 consecutive contacts before the ball
       must cross to the other team.
  C-2: Possessions alternate teams. After a possession ends (3rd contact
       or any cross-net action), the next action must be by the other team.
  C-3: First action of a rally is `serve`.

Skip semantics: each rule has explicit skip conditions for degenerate
rallies (e.g., rally with 0 actions skips all rules). Additionally, rallies
that fail any I-1 / I-3 / I-6 PID invariant are EXCLUDED from coherence
checks at the orchestrator level — they're upstream issues that would
produce noisy coherence violations downstream.
"""
```

Imports `Violation` from `pid_invariants` (single source of truth for the dataclass). Defines:

- `_team_for_action(action: dict, team_assignments: dict[str, str]) -> str | None` — helper that resolves an action to its team (looks up `playerTrackId` in `teamAssignments`); returns None if undetermined.
- `_actions_sorted_by_frame(actions: list) -> list` — defensive sort for irregular DB ordering.
- `check_c1_three_contact_rule(*, rally_id, actions, team_assignments) -> list[Violation]`
- `check_c2_alternating_possessions(*, rally_id, actions, team_assignments) -> list[Violation]`
- `check_c3_first_action_is_serve(*, rally_id, actions) -> list[Violation]`
- `run_all(*, video_id: str) -> list[Violation]` orchestrator.

### CLI: `analysis/rallycut/cli/commands/audit_coherence_invariants.py`

Mirrors `audit_pid_invariants.py` exactly:
- Typer command with `video_id` (required) + `--quiet` flag
- Calls `coherence_invariants.run_all(video_id=video_id)`
- Renders Rich table (Invariant | Rally | Severity | Detail)
- Exits 1 if any violations, 0 otherwise

### Orchestrator skip-upstream gate

`run_all` for coherence does:

1. Call `pid_invariants.run_all(video_id=video_id)` to get the set of upstream-failing rallies.
2. Build `excluded_rallies = {v.rally_id for v in upstream_violations if v.invariant in ("I-1", "I-3", "I-6")}`.
3. Load rallies + actions + teamAssignments from DB.
4. For each rally NOT in excluded_rallies, dispatch to all 3 coherence checks.
5. Aggregate violations.

This dependency on `pid_invariants` is intentional and explicit: coherence rules assume structural correctness as a precondition. The skip is conservative (only I-1, I-3, I-6 — invariants that directly affect action attribution / team labeling), not a blanket "skip any failing rally."

### The 3 rules

#### C-1: Three-contact rule

A team can have at most 3 consecutive contacts before the ball must cross to the other team.

```python
def check_c1_three_contact_rule(
    *,
    rally_id: str,
    actions: list[dict[str, Any]],
    team_assignments: dict[str, str],
) -> list[Violation]:
    """C-1: a team can have at most 3 consecutive contacts.

    Walks the actions array sorted by frame. Tracks consecutive contacts
    by the same team. Flags any sequence of 4+ same-team contacts.
    """
```

**Skip if:**
- Fewer than 2 actions (no sequence to violate).
- Any action's team can't be resolved (defensive — coherence orchestrator already skips I-6 rallies, but extra guard).

**Violation detail format**: `"team A had 4 consecutive contacts at frames [100, 137, 187, 234]; max is 3"`.

#### C-2: Alternating possessions

After a possession ends (3rd contact OR any cross-net action), the next action must be by the opposing team.

```python
def check_c2_alternating_possessions(
    *,
    rally_id: str,
    actions: list[dict[str, Any]],
    team_assignments: dict[str, str],
) -> list[Violation]:
    """C-2: possessions alternate teams.

    Maintains per-possession state: starts with the receiving team after
    a `serve`. Possession ends when (a) team has had 3 contacts, OR (b) any
    action's `action` field is in {"attack", "serve"}, OR (c) the next
    action's court_side differs from the current possession's side. After
    possession end, next action must be the OTHER team.
    """
```

**Possession-end heuristic** (clarification):
- An `attack` ends possession (the ball crosses).
- A `serve` ends possession (this is the start of a new rally turn).
- After 3 contacts of one team, possession ends regardless of action type.
- A change in `court_side` between consecutive actions strongly suggests a possession crossed (use as fallback signal).

**Skip if:**
- Fewer than 2 actions.
- Any action's team can't be resolved.

**Violation detail format**: `"team A action[3] (frame 234, attack) ended possession; next action[4] (frame 260, dig) was also team A — expected team B"`.

#### C-3: First action is serve

The first action of a rally must be a `serve`.

```python
def check_c3_first_action_is_serve(
    *,
    rally_id: str,
    actions: list[dict[str, Any]],
) -> list[Violation]:
    """C-3: rally's first action must be `serve`.

    Sanity check on rally-start detection. Catches rallies where the
    first contact wasn't classified as a serve, or where the serve is
    missing entirely.
    """
```

**Skip if:**
- Rally has 0 actions (degenerate — likely zero-action rally already known).

**Violation detail format**: `"first action is 'attack' (frame 100); expected 'serve'"` or `"rally has 1 action 'attack' but no 'serve'"`.

### Composition with existing audit

```
rallycut audit-pid-invariants <video>          → I-1..I-8 (structural)
rallycut audit-coherence-invariants <video>    → C-1..C-3 (game-rule)
```

Both run independently. Future `audit-all` could run both, but v1 keeps them separate to make the layer firing obvious.

The fleet-audit script (`scripts/fleet_pid_audit.py`) can be extended later to call both modules and report combined counts. v1 doesn't extend it — keeps changes contained to the new module.

## Testing

### Unit tests (`analysis/tests/unit/test_coherence_invariants.py`)

Per-rule TDD pattern, mirrors `test_pid_invariants.py`. One `Test*` class per rule plus `TestRunAll` for the orchestrator.

For each rule:
- **Clean case**: a rally with a valid action sequence passes (returns []).
- **Invalid case**: a constructed bad sequence fails with expected violation count and detail string substring.
- **Skip cases**: degenerate inputs (empty actions, missing team labels) skip gracefully.

For `TestRunAll`:
- Mock connection (same `MagicMock` pattern as `test_pid_invariants.py`).
- Test that rallies failing upstream PID invariants are excluded from coherence checks.
- Test that orchestrator aggregates violations across rules correctly.

### No integration tests in v1

Like the existing PID work, integration is verified by running the CLI on real fleet data (post-implementation) and inspecting outputs.

## Rollout sequence

1. Land `coherence_invariants.py` + 3 check functions + unit tests per rule (single commit).
2. Land `run_all` orchestrator + its tests (single commit).
3. Land `audit_coherence_invariants.py` CLI + register in `main.py` (single commit).
4. Run `audit-coherence-invariants` on the 4 panel videos to spot-check output.
5. Run on the full fleet (`for vid in $(cat /tmp/audit_videos.txt); do uv run rallycut audit-coherence-invariants "$vid"; done`).
6. Tabulate findings: which rules fire most, on which videos, and how many.
7. Document in memory.

## Done criteria

- [ ] `uv run rallycut audit-coherence-invariants --help` shows the command.
- [ ] All unit tests pass (3 rule classes + orchestrator class).
- [ ] CLI runs cleanly on a clean rally (no violations).
- [ ] CLI runs cleanly on a known-bad rally (e.g., one of the 10 residual rallies in the fleet, if it has actions) — produces expected output.
- [ ] Fleet sweep produces interpretable output: per-rule counts, top affected videos.
- [ ] Memory entry recorded.

## Risks

1. **C-2's possession-end heuristic.** "Possession ends when..." is a heuristic — not all volleyball plays fit cleanly. Edge cases:
   - Block-and-cover: blocker touches ball, then teammates dig — looks like 4+ contacts but block doesn't count toward the 3.
   - Double touches: legal in some plays, illegal in others.
   - Action types could be misclassified upstream.
   **Mitigation**: keep the heuristic simple (3 contacts OR `attack`/`serve` action OR court_side change). Document edge cases explicitly. False positives are acceptable in v1 — they surface for review, not auto-correction.

2. **C-3 may be too strict.** Some rallies have the serve detection miss — the first stored action may be `receive` or `dig` instead of `serve`. We'd see lots of C-3 fires on those. This is INTENDED — surfacing serve-detection misses is a useful signal, not a false positive.

3. **Coupling with `pid_invariants`.** The orchestrator imports from `pid_invariants` to build the skip set. This creates a runtime dependency. Acceptable — both modules are eval-time-only and the skip is essential for signal/noise. If the dependency becomes painful, refactor later.

4. **Action type vocabulary.** The rules assume action types like `serve`, `attack`, `set`, `dig`, `receive`. If the action classifier ever introduces new types, rules need updating. Mitigation: rules use string literals for type checks; failures surface in CI tests.

5. **Performance.** Each rule iterates a single rally's actions. O(rally_actions) per rule, O(rallies × 3 rules) per video. Trivially fast.

## File-change summary

- New (3): `analysis/rallycut/tracking/coherence_invariants.py` (~200 LOC), `analysis/rallycut/cli/commands/audit_coherence_invariants.py` (~50 LOC), `analysis/tests/unit/test_coherence_invariants.py` (~200 LOC).
- Modified (1): `analysis/rallycut/cli/main.py` (~2 lines: import + `app.command`).
- **Total:** 4 files, ~450 LOC.
