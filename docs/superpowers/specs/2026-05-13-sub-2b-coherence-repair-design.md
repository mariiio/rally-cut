# Sub-2.B: Coherence Repair — C-4 Audit + Measurement-Gated Player Re-attribution

**Date:** 2026-05-13
**Status:** Design — pending implementation plan
**Workstream context:** Sub-2.B extends [Sub-2.A](2026-05-10-coherence-invariants-v1-design.md) (`audit-coherence-invariants`, shipped 2026-05-10, detection-only) with one new game-rule invariant and a measurement-gated player-attribution repair pass. Sits alongside the recent attribution work whose mechanical-rule ceilings were hit ([joint-attribution PGM](2026-05-12-joint-attribution-pgm-design.md), [contact-detection FN reduction Phase 1+1.5](2026-05-12-contact-detection-fn-reduction-design.md)). The audit-first discipline here is deliberate — design the repair pass from real fleet patterns, not from upfront hypotheses.

## Goal

1. Add **C-4** (no-same-player-back-to-back, exception: prev=block) to `audit-coherence-invariants`.
2. Build a fleet-wide pattern catalog of C-4 violations with pre-scored evidence signals.
3. After a gated review of the catalog, design and ship a **multi-signal repair pass** inside `action_classifier.reattribute_players` parallel to the existing Pass 2c (within-team proximity swap), default-OFF behind an env flag.

## Motivation

Sub-2.A surfaced 744 coherence violations across the fleet (179 C-1 / 542 C-2 / 23 C-3). C-2's possession-alternation breaks are dominant but heterogeneous — many root causes mix into a single bucket. C-4 (same player consecutive) is a narrower, more diagnostic rule: a single observable pattern with a small number of distinct root-cause families (synthetic-serve cascade, attribution swap, genuine double-touch, ghost contact).

The current attribution rules (Pass 2 team-chain override + Pass 2c within-team proximity swap) operate on per-action signals (team label, distance to ball). They do not consult adjacent-action context. A pair like `(set by player_5, attack by player_5)` is acceptable to Pass 2/2c if both attributions individually look fine — but volleyball-rule wise it should fire (a player who sets cannot also be the one attacking on the same possession, normally).

Sub-2.B's C-4 detector is the rule. The repair pass is the fix. The measurement gate between them ensures the repair design is empirically grounded, not invented.

## Constraints (locked before measurement)

- GT remains contact-based (all action types, including `block`, require ball contact). The rule's `block` exception reflects volleyball semantics, not GT semantics.
- The two repair functions stay separate by responsibility:
  - `repair_action_sequence` = action-type rules (existing).
  - `reattribute_players` = player-attribution rules (existing). Sub-2.B's repair lands here.
- The repair must NOT be a tactical patch elsewhere (e.g., not in `redetect_all_actions`, not in a fleet cleanup CLI).
- The repair must skip rather than over-correct when violations reflect upstream errors. Concretely: when this rally also fires C-3, or when either action has `is_synthetic=True` (e.g., synthetic-serve prepended by `_make_synthetic_serve`).
- Single-rally checks only (no cross-rally state). Mirrors Sub-2.A's scope.

## Scope

**In scope (Phase 1, designed in full):**

- C-4 detector in `analysis/rallycut/tracking/coherence_invariants.py`.
- One-line extension to the `Violation` dataclass in `pid_invariants.py` (optional structured `payload`) so the catalog harness consumes structured data instead of regex-parsing `detail` strings.
- Fleet pattern-catalog script `analysis/scripts/catalog_c4_violations.py` writing pre-scored evidence rows to `analysis/reports/coherence_c4_catalog/`.
- A no-op `Pass 2d` stub in `reattribute_players`, env-flag-gated default-OFF, ready for Phase 2 to fill in.
- Unit tests for C-4 mirroring Sub-2.A's pattern.

**Out of scope (Phase 2, gated):**

- The repair rule body itself. Designed at the Phase 1 → Phase 2 review gate from real catalog patterns, recorded as a follow-up plan against this same spec.
- A/B measurement harness for the repair pass.
- Default-ON flip. Only after A/B passes on the 22-rally panel.
- Cross-rally C-4 (server's team consistent across rallies). Sub-2.A scope discipline.
- C-5, C-6, … other future coherence invariants.
- Refactoring `reattribute_players` itself.

## Architecture

```
Phase 1 (this spec, designed in full):
  A. C-4 detector       → coherence_invariants.py        (+ pid_invariants.Violation.payload)
  B. Pattern catalog    → scripts/catalog_c4_violations.py
                         → reports/coherence_c4_catalog/2026-05-13.csv  (per-pair evidence)
                         → reports/coherence_c4_catalog/2026-05-13_summary.md  (fleet-aggregate)
  C. Pass 2d stub       → action_classifier.py            (no-op, COHERENCE_C4_REPAIR=0 default)

→ HARD STOP. Gated review:
   - Fill ≥30 rows' `root_cause` column in CSV (hand-classified).
   - Write reports/coherence_c4_catalog/2026-05-13_review.md.
   - Decide: Phase 2 viable? If yes, design the multi-signal rule from the patterns.

Phase 2 (gated; separate plan against this spec):
  D. Multi-signal repair → action_classifier.py            (Pass 2d body, signals from catalog)
  E. A/B harness         → scripts/measure_c4_repair_ab.py
  F. Default-ON flip     → only after A/B non-regressing on panel correct-rate.
```

Skip semantics mirror Sub-2.A: C-4 runs only on rallies that pass upstream PID invariants I-1 / I-3 / I-6 (those are upstream-driven and would produce noisy downstream coherence violations).

## Phase 1 detail

### A. C-4 detector

**Module:** `analysis/rallycut/tracking/coherence_invariants.py` (extends existing).

**Function:**

```python
def check_c4_no_same_player_back_to_back(
    *,
    rally_id: str,
    actions: list[dict[str, Any]],
    team_assignments: dict[str, str],
) -> list[Violation]:
    """C-4: consecutive actions must be by different players.

    Exception: when action[i-1].action == 'block', the pair is exempt
    (block→cover by the same player is legal — and so is the rarer
    block→block by the same player, since the prev action is still block).

    Skip pair if either action's player_track_id is missing, -1, or
    unmapped in team_assignments — the comparison is meaningless when
    one side is unattributed.

    Audit-side does NOT skip on action.confidence < 0.3 — surface all
    pairs so the catalog reflects the full distribution. The 0.3 floor
    applies only to the Phase 2 repair pass.
    """
```

**Logic** (after defensive sort by frame):

```
for i in 1..len(sorted_actions)-1:
    prev, curr = sorted_actions[i-1], sorted_actions[i]
    prev_pid = prev.get("playerTrackId")
    curr_pid = curr.get("playerTrackId")

    # Skip pair: missing/-1 ids
    if prev_pid is None or curr_pid is None: continue
    if prev_pid == -1 or curr_pid == -1: continue
    if str(prev_pid) not in team_assignments: continue
    if str(curr_pid) not in team_assignments: continue

    # Block exception (strict — only previous is block exempts the pair)
    if str(prev.get("action", "")) == "block": continue

    # Fire C-4
    if prev_pid == curr_pid:
        violations.append(Violation(
            invariant="C-4",
            rally_id=rally_id,
            detail=(
                f"action[{i-1}] (frame {prev.get('frame')}, "
                f"{prev.get('action')}, player {prev_pid}) and "
                f"action[{i}] (frame {curr.get('frame')}, "
                f"{curr.get('action')}, player {curr_pid}) "
                f"attributed to same player; max is 1 unless prev is block"
            ),
            payload={
                "prev_index": i - 1,
                "curr_index": i,
                "prev_frame": int(prev.get("frame", 0)),
                "curr_frame": int(curr.get("frame", 0)),
                "prev_action": str(prev.get("action", "")),
                "curr_action": str(curr.get("action", "")),
                "player_id": int(prev_pid),
            },
        ))
```

**Wiring:** add to `coherence_invariants.run_all` alongside C-1/C-2/C-3.

### B. `Violation.payload` extension

In `analysis/rallycut/tracking/pid_invariants.py`:

```python
@dataclass(frozen=True)
class Violation:
    invariant: str
    rally_id: str
    detail: str
    severity: Literal["error", "warn"] = "error"
    payload: dict[str, Any] | None = None   # NEW: optional structured data
```

`payload` defaults to `None`. C-1/C-2/C-3 do not populate it. C-4 does. The `audit-coherence-invariants` and `audit-pid-invariants` CLIs render `detail` only; `payload` is for programmatic consumers (the catalog script).

### C. Pattern catalog harness

**Script:** `analysis/scripts/catalog_c4_violations.py`.

**Invocation:**

```
uv run python -u analysis/scripts/catalog_c4_violations.py \
    --output analysis/reports/coherence_c4_catalog/2026-05-13.csv \
    --summary analysis/reports/coherence_c4_catalog/2026-05-13_summary.md
```

Walks every video in the labeled fleet, runs `coherence_invariants.run_all`, filters to C-4 violations, computes per-pair evidence, writes CSV + summary markdown. Pure DB read; no ML. Per-rally progress logging (`[3/70] video=<id> rallies_with_c4=<n>`).

**CSV schema (one row per C-4 violation pair):**

| Group | Column | Notes |
|---|---|---|
| Identity | `rally_id`, `video_id`, `pair_idx`, `frame_prev`, `frame_curr` | from DB + payload |
| Pair facts | `action_prev_type`, `action_curr_type`, `player_id`, `team_label` | from actions_json + team_assignments |
| Confidence | `conf_prev`, `conf_curr` | from actions_json |
| Distance | `prev_player_dist`, `curr_player_dist` | from contacts_json |
| Candidates | `prev_top3_candidates`, `curr_top3_candidates` | JSON `[(tid, dist, team), …]` |
| Alt-ratio | `prev_best_same_team_alt_ratio`, `curr_best_same_team_alt_ratio` | float; NaN if no same-team alt; <1.0 means alt is closer |
| Type-fit | `signal_type_fit_prev`, `signal_type_fit_curr` | `ok` / `wrong` / `unknown` (expected-transitions table) |
| Team-geometry | `signal_team_geometry_prev`, `signal_team_geometry_curr` | `matches` / `violates` / `ambiguous` |
| Co-violations | `co_c1_fires`, `co_c2_fires`, `co_c3_fires` (bool) | this rally's other coherence violations |
| | `co_pid_invariant_fires` | csv of I-* tags if any |
| Placeholder | `repair_recommendation` | `skip` / `repair_prev` / `repair_curr` / `ambiguous` (placeholder rule) |
| Hand-class | `root_cause` (blank), `notes` (blank) | filled at review |

**Expected-transitions table** (Python constant in the script):

```python
EXPECTED_TRANSITIONS = {
    # (prev_type, curr_type): (expected_curr_team_relation, label)
    ("serve",   "receive"): ("other",  "ok"),
    ("serve",   "dig"):     ("other",  "ok"),   # receive sometimes labeled dig
    ("serve",   "set"):     (None,     "wrong"),
    ("serve",   "attack"):  (None,     "wrong"),
    ("serve",   "block"):   ("other",  "ok"),
    ("receive", "set"):     ("same",   "ok"),
    ("receive", "attack"):  ("same",   "ok"),
    ("set",     "attack"):  ("same",   "ok"),
    ("set",     "set"):     (None,     "wrong"),
    ("attack",  "dig"):     ("other",  "ok"),
    ("attack",  "block"):   ("other",  "ok"),
    ("attack",  "receive"): ("other",  "ok"),
    ("block",   "receive"): ("either", "ok"),   # block prev exempts C-4 anyway
    ("block",   "dig"):     ("either", "ok"),
    ("block",   "set"):     ("either", "ok"),
    ("dig",     "set"):     ("same",   "ok"),
    ("dig",     "attack"):  ("same",   "ok"),
    # Unknown pairs → 'unknown' (NOT 'wrong' — table is conservative).
}
```

Pairs not in the table return `signal_type_fit = 'unknown'`. Keeps the table additive — adding rows can only sharpen the signal, never invalidate prior catalog runs.

**`signal_team_geometry` computation:**

For each side, look at the contact's `player_candidates`. If the rank-1 candidate's team matches the volleyball-expected team for this slot, geometry is `matches`. If rank-1 is on the wrong team but a same-team candidate is rank-2 within 2x distance, geometry is `violates` (the matcher picked across teams against the rules). Else `ambiguous`.

The "volleyball-expected team for this slot" is derived from the action sequence using the existing `_compute_expected_teams` helper (already in `action_classifier.py`).

**Placeholder `repair_recommendation` rule** (hypothesis only; not shipped):

```
strong_against(side) = count of signals in {
    signal_type_fit[side] == 'wrong',
    signal_team_geometry[side] == 'violates',
    signal_alt_ratio[side] < 0.6,         # closer same-team alt exists
    conf[side] < 0.5,
}

if strong_against(prev) >= 2 and strong_against(curr) <= 1:
    'repair_prev'
elif strong_against(curr) >= 2 and strong_against(prev) <= 1:
    'repair_curr'
elif strong_against(prev) == 0 and strong_against(curr) == 0:
    'skip'
else:
    'ambiguous'
```

The Phase 2 design checkpoint compares this placeholder against the hand-classified `root_cause` column. Mismatches reveal which signals are over- or under-weighted; the empirical Phase 2 rule is fit accordingly.

**Hand-classification labels** for the `root_cause` column:

- `synthetic_serve_cascade` — prev or curr is a mis-attributed synthetic serve.
- `attribution_swap_prev` — prev should be a teammate (current is correct).
- `attribution_swap_curr` — curr should be a teammate (prev is correct).
- `genuine_double_touch` — both correctly attributed; rule should not have fired.
- `ghost_contact_prev` / `ghost_contact_curr` — one is a spurious contact-detection FP.
- `wrong_action_type` — attribution is fine but action types are misclassified.
- `block_exception_miss` — prev should have been classified as block (rule applied correctly but action type is wrong).
- `other`.

Target ≥30 rows hand-classified.

**Summary markdown** (`2026-05-13_summary.md`):

- Total fleet C-4 count.
- Breakdown by `(action_prev_type, action_curr_type)` cell.
- Co-violation rates: % of C-4 rallies also firing C-2 / C-3 / any PID invariant.
- Placeholder `repair_recommendation` distribution.
- Per-video counts, top 10 worst videos.

This file is the gated review's input artifact.

### D. Pass 2d stub

In `analysis/rallycut/tracking/action_classifier.py:reattribute_players`, between the existing Pass 2c block and the Pass 3 ReID block:

```python
# Pass 2d (Sub-2.B, gated): C-4 same-player-back-to-back repair.
# Multi-signal evidence-based. Exact rule designed at Phase 1→2 gate from
# the violation-pattern catalog (analysis/reports/coherence_c4_catalog/).
# Spec: docs/superpowers/specs/2026-05-13-sub-2b-coherence-repair-design.md
# Default-OFF; flip to ON only after A/B passes on the 22-rally panel.
if os.environ.get("COHERENCE_C4_REPAIR", "0") == "1":
    n_c4_repairs = _coherence_c4_repair_pass(
        actions=actions,
        contact_by_frame=contact_by_frame,
        team_assignments=team_assignments,
        chain_integrity=chain_integrity,
        expected_teams=expected_teams,
    )
    if n_c4_repairs > 0:
        logger.info(
            "C-4 repair: re-attributed %d/%d actions",
            n_c4_repairs, len(actions),
        )
```

**Phase 1 ships the `_coherence_c4_repair_pass` function as a no-op stub:**

```python
def _coherence_c4_repair_pass(
    *,
    actions: list[ClassifiedAction],
    contact_by_frame: dict[int, Contact],
    team_assignments: dict[int, int],
    chain_integrity: list[bool],
    expected_teams: list[int | None],
) -> int:
    """Phase 1: not yet implemented. Returns 0.

    Phase 2 (gated on Phase 1 pattern catalog review) fills in the body
    using the multi-signal evidence rule designed from real fleet
    patterns. See spec section 'Phase 2 detail (placeholder)' for the
    constraint set the eventual rule must respect.
    """
    return 0
```

Production behavior at default-OFF is byte-identical to pre-workstream. Setting `COHERENCE_C4_REPAIR=1` in Phase 1 is also a no-op (the stub returns 0).

## Phase 2 detail (placeholder — actual rule designed from catalog)

The Phase 2 plan will design the repair rule from the Phase 1 catalog. The rule must respect these constraints (locked in this spec):

1. **Confidence floor:** skip when `action.confidence < 0.3` (mirrors Pass 2c).
2. **Player floor:** skip when `action.player_track_id < 0` (mirrors Pass 2c).
3. **Block exception:** skip pair when prev action is block (matches C-4 detector).
4. **Distance cap:** any candidate swap must satisfy `candidate.dist <= 2.0 * current_player.dist`. Matches the user's "within ~2x" guidance and the existing team-chain override's 2.0x cap (`_team_chain_override_allowed`, action_classifier.py:2877).
5. **Upstream-error skip:** skip when (a) this rally fires C-3, OR (b) either action has `is_synthetic=True` (catches synthetic-serve prepends; `ClassifiedAction.is_synthetic` is the canonical field — see action_classifier.py:89, set by `_make_synthetic_serve` at action_classifier.py:1487).
6. **Multi-signal convergence:** the rule must consult ≥2 of {action-type fit, team geometry, alt-ratio, confidence} to decide which side to repair. Distance alone is insufficient.
7. **Default-OFF until A/B passes** on the 22-rally panel (correct-rate non-regressing AND C-4 fleet count drops).
8. **Iteration order:** forward (low-index to high-index). Repair to `action[i].player_track_id` may affect the pair `(action[i], action[i+1])` — the rule must be idempotent on a fixed-point pass (re-run yields no further changes).

The empirical rule (which signals, what weights, what combination logic) is the Phase 2 design output, not committed in this spec.

## Testing

`analysis/tests/unit/test_coherence_invariants.py` extends with one `TestC4SamePlayerBackToBack` class:

- **Clean** — different players consecutively → no violation.
- **Bad** — same player consecutive non-block actions → exactly one violation, correct payload (frame indices, player_id, action types).
- **Block exception (block→X same player)** — no violation.
- **Block→block same player** — no violation (strict reading: prev=block exempts).
- **X→block same player (curr is block, prev is not)** — fires (only prev=block exempts).
- **Missing pid** — `playerTrackId = -1` on either side → no violation.
- **Unmapped pid** — `playerTrackId` not in `team_assignments` → no violation.
- **Cascade (X→X→X, none block)** — two violations (pair 0-1 and pair 1-2).
- **Cascade (X→block→X same player)** — one violation (pair 0-1 fires; pair 1-2 exempt because prev is block).

`TestRunAll` extends with one assertion: orchestrator includes C-4 in dispatch.

No new integration tests in Phase 1 (matches Sub-2.A pattern). Phase 2 will land integration tests for the repair pass + an A/B harness.

`Violation` field addition needs no new tests — adding an optional field with a default doesn't change existing constructor signatures.

## Rollout sequence (Phase 1)

1. Extend `Violation` dataclass with `payload: dict[str, Any] | None = None`. (Single commit; touches Sub-2.A code.)
2. Land `check_c4_no_same_player_back_to_back` + the `TestC4*` class. Wire into `coherence_invariants.run_all`. (Single commit.)
3. Land `analysis/scripts/catalog_c4_violations.py` with the `EXPECTED_TRANSITIONS` constant, per-rally progress logging, CSV + summary output. (Single commit.)
4. Land the Pass 2d stub in `action_classifier.py` (no-op behind `COHERENCE_C4_REPAIR` env flag). (Single commit.)
5. Run the catalog on the full fleet. Inspect `2026-05-13_summary.md`.
6. Hand-classify ≥30 rows in the CSV (`root_cause` column).
7. **GATED REVIEW** — write `analysis/reports/coherence_c4_catalog/2026-05-13_review.md` summarizing the pattern distribution and proposing the Phase 2 rule.
8. Open this spec for amendment with the Phase 2 rule design, then write a Phase 2 plan.

## Done criteria (Phase 1)

- [ ] `uv run rallycut audit-coherence-invariants <video>` reports C-4 violations alongside C-1/C-2/C-3.
- [ ] All unit tests pass (existing + new `TestC4*` class).
- [ ] CLI runs cleanly on a known clean rally (no C-4 violations); runs cleanly on a constructed rally with a same-player pair (one C-4 violation).
- [ ] Catalog script runs end-to-end on the 22-rally panel as a smoke test, produces a non-empty CSV.
- [ ] Catalog script runs on the full fleet, produces interpretable summary markdown.
- [ ] `COHERENCE_C4_REPAIR=0` and `COHERENCE_C4_REPAIR=1` both leave production behavior byte-identical (stub returns 0).
- [ ] ≥30 rows hand-classified.
- [ ] Gated review markdown drafted.
- [ ] Memory entry recorded.

## Risks

1. **Catalog placeholder rule could anchor design.** Looking at `repair_recommendation` during hand-classification may bias `root_cause` toward agreeing with it. *Mitigation:* fill `root_cause` BEFORE looking at `repair_recommendation`, or hide the column during the first pass through the CSV.
2. **Synthetic-serve cascade dominates the catalog.** If most C-4 fires are downstream of a wrong synthetic serve, the right fix is upstream (better synthetic-serve placement; ref the 2026-05-12 `d26c6e9e` commit), not a C-4 repair. *Mitigation:* the catalog will tell us; if `co_c3_fires` dominates, the Phase 1→2 review concludes "no Phase 2 needed yet — fix upstream first," and this spec parks at the gate.
3. **`Violation.payload` addition is mildly invasive.** Sub-2.A's existing CLI consumers (`audit_coherence_invariants.py`, `audit_pid_invariants.py`) read `detail` only. *Mitigation:* default `None`, never required; CLI code unchanged.
4. **22-rally panel may have ≤5 C-4 violations.** Phase 2 A/B might not have statistical power. *Mitigation:* the catalog tells us the fleet rate; if panel coverage is too thin, extend A/B to include fleet C-4-positive rallies even if they're not on the GT panel.
5. **`EXPECTED_TRANSITIONS` table will surface beach-2v2 edge cases.** Beach 2v2 has legitimate transitions that 6v6 doesn't (e.g., short-receive→attack by same team is common). *Mitigation:* table is data, not hardcoded behavior; update it as the catalog surfaces edge cases. Unknown pairs are conservative (`unknown`, not `wrong`).
6. **Phase 1 measurement could conclude "no Phase 2 needed."** If C-4 violations are rare or dominated by upstream causes, the right call is to park Phase 2 indefinitely. *This is a success, not a failure.* The audit ships value on its own (it's measurement infrastructure; future workstreams can use it).

## File-change summary

- **New (1 file + 1 dir):**
  - `analysis/scripts/catalog_c4_violations.py` (~250 LOC).
  - `analysis/reports/coherence_c4_catalog/` (output dir; gitignored content).
- **Modified (4 files):**
  - `analysis/rallycut/tracking/pid_invariants.py` (+`payload` field, ~3 lines).
  - `analysis/rallycut/tracking/coherence_invariants.py` (+C-4 check function + `run_all` dispatch, ~50 lines).
  - `analysis/rallycut/tracking/action_classifier.py` (+Pass 2d stub + env flag + helper function, ~25 lines).
  - `analysis/tests/unit/test_coherence_invariants.py` (+`TestC4*` class, ~130 lines).
- **Total Phase 1:** 5 files, ~460 LOC.

**Phase 2** (gated, separate plan): fill in `_coherence_c4_repair_pass` body in `action_classifier.py`, add `analysis/scripts/measure_c4_repair_ab.py`, add integration tests for the repair pass.

## Composes with

- [Sub-2.A coherence invariants v1](2026-05-10-coherence-invariants-v1-design.md) — direct extension.
- [Within-team proximity swap (Pass 2c)](2026-05-11-within-team-proximity-swap-design.md) — Pass 2d sits alongside this.
- [Action attribution team chain (Pass 2 override)](2026-05-11-action-attribution-team-chain-design.md) — shares the 2.0x distance cap pattern.
- [Synthetic serve placement v1.1](2026-05-10-synthetic-serve-placement-design.md) and the 2026-05-12 dual-signal guard (commit `d26c6e9e`) — the synthetic-serve-cascade root cause this catalog will measure.
- [Joint attribution PGM 2026-05-12](2026-05-12-joint-attribution-pgm-design.md) and [contact-detection FN Phase 1+1.5](2026-05-12-contact-detection-fn-reduction-design.md) — same architectural lesson (hand-tuned global rules cap fast); Sub-2.B uses measurement-first discipline to avoid the same trap.
