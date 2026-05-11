# Action Attribution: Team-Chain Override (v1)

**Date:** 2026-05-11
**Status:** Design — pending implementation plan
**Workstream context:** First targeted ship in the action-attribution improvement workstream. Composes downstream of the post-v1.3 contact-detection ceiling (fleet F1 0.927) and the 2026-05-11 `redetect_all_actions` fix that restored `teamAssignments` + `servingTeam` on every rally. Attribution is the next layer up.

## Goal

Reduce cross-team attribution errors on the action-attribution layer by relaxing the "don't override the nearest candidate" guard in `reattribute_players` Pass 2, gated on four trust-the-chain predicates. Default-on production change. The fix is surgical: 5–10 lines of altered control flow plus a small predicate helper.

## Motivation

### Baseline (2026-05-11, 3 fresh-GT videos)

`scripts/measure_attribution_fresh_gt.py` on cece + gigi + wawa (the 3 videos with current, retrack-aligned action GT — every other video's `action_ground_truth_json` was wiped by the v1.1 / v1.2 / v1.3 retracks):

| Video | n GT actions | correct | wrong | missing |
|---|---:|---:|---:|---:|
| cece (5 rallies) | 29 | 22 (75.9%) | 4 (13.8%) | 3 |
| gigi (7 rallies) | 56 | 35 (62.5%) | 12 (21.4%) | 9 |
| wawa (10 rallies) | 51 | 25 (49.0%) | 10 (19.6%) | 16 |
| **Combined** | **136** | **82 (60.3%)** | **26 (19.1%)** | **28 (20.6%)** |

`missing` is action-detection layer (no pipeline action within ±10 frames of GT) — out of scope. **Attribution accuracy among matched actions: 76.0%** (82/108).

### Error mix among 26 wrong attributions

- `wrong_cross_team`: **16 (62%)** — pipeline picked opposing team. THIS WORKSTREAM'S TARGET.
- `wrong_same_team`: 7 (27%) — right team, wrong player. Out of v1 scope (needs pose/reid).
- `wrong_unknown_team`: 3 (12%) — synthetic-serve attribution failed (`pl_pid=-1`). Out of v1 scope.

### Per-action accuracy among matched actions

| Action | matched | correct | accuracy |
|---|---:|---:|---:|
| receive | 16 | 15 | 93.8% |
| attack | 35 | 29 | 82.9% |
| dig | 17 | 14 | 82.4% |
| set | 24 | 16 | 66.7% |
| **serve** | **15** | **7** | **46.7%** |
| block | 1 | 1 | — |

Set and serve carry most of the cross-team error budget — both are actions where the "nearest player" intuition is weakest (set: receiver / target ambiguity near the net; serve: synthetic serve placement + wrong-teammate failures).

### The proximate cause (verified in code)

`reattribute_players` Pass 2 (`analysis/rallycut/tracking/action_classifier.py:2858`) already derives an `expected_team` per action via `_compute_expected_teams` (line 2808), using volleyball transition rules seeded by the serve identity. For an action whose current `playerTrackId` is on the **wrong team** per `expected_team`, the function tries to find a candidate on the **correct team** within `1.5 ×` the current distance.

But at lines 2963–2968:

```python
# Guard: don't override the nearest candidate. Proximity is hard
# physical evidence; the expected_team chain drifts when contacts
# are missed or action types are wrong. Only override non-nearest
# attributions (unmapped tracks or clear team mismatches).
if (
    not is_unmapped
    and contact.player_candidates
    and contact.player_candidates[0][0] == action.player_track_id
):
    continue
```

This unconditional guard fires whenever the wrong-team current attribution is ALSO the spatially nearest candidate — exactly the canonical failure mode (e.g., the user-quoted `07fedbd4` rally `5d35c3bf` where the wrong-team server's teammate was nearest to the ball at the receive frame). The 2026-04-12 attribution investigation introduced this guard because the chain drifts on missed contacts or action-type errors. Today the chain is significantly more reliable (action F1 0.927, contact F1 0.927, PERMUTED PID 95.1%), so the guard's blanket conservatism leaves cross-team wins on the table.

### Prior art — what NOT to repeat

- **Ball-Y net-crossing detection** as a team-transition signal (2026-04-12): -1.2pp attribution, -3.5pp court_side. Ball Y is too noisy at contact time. This design uses `Contact.court_side` (already computed by `_resolve_court_side`, ball-position + homography, NOT ball-Y crossings) only as a corroborator, never as the trigger.
- **Bare guard removal**: would regress on rallies with broken action chains (UNKNOWN, missed contacts). v1 keeps the guard except when ALL four trust gates pass.
- **Confidence-threshold sweep** on `pose_attribution_min_confidence` / `temporal_attribution_min_confidence` (2026-04-12): confirmed no-op. Don't re-explore.

## Scope

### In scope

- New helpers (both pure functions, isolated tests):
  - `_chain_integrity(actions, i) -> bool` — returns whether the chain from the seed serve to `actions[i]` is unbroken (no UNKNOWN action types, no `is_synthetic` actions intermediating). Computed once per rally; returned alongside the expected-team array.
  - `_team_chain_override_allowed(action, contact, expected_team, chain_integrity_i, team_assignments, max_distance_ratio) -> bool` — the four-gate predicate combining G1+G2+G3+G4 into one boolean. Tested as a truth table.
- Extension of `_compute_expected_teams` (line 2808) to also return a parallel `chain_integrity: list[bool]` array. Existing return signature changes from `list[int | None]` to `tuple[list[int | None], list[bool]]`. Two existing callers (Pass 2 in `reattribute_players`, and `correct_team_from_propagation` — currently doesn't use this signal) updated.
- Replace the unconditional guard at `action_classifier.py:2963-2968` with a four-gate predicate:
  - **G1**: `action.confidence ≥ 0.7`
  - **G2**: `chain_integrity[i] == True`
  - **G3**: a candidate on the expected team exists within `1.5 ×` the current distance (the same `max_distance_ratio` cap already used — this is just computing the cap *before* the guard fires)
  - **G4**: the corresponding `Contact.court_side` agrees with `expected_team` (`expected_team=0 ⇒ "near"`, `expected_team=1 ⇒ "far"`). If `court_side == "unknown"`, treat as a soft pass — emit a debug log noting the absence of corroboration but still allow the override.
- Env flag `RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN` (read at call time). Default ON. Set to `0` for fast rollback without redeploy.
- Unit tests at `analysis/tests/unit/test_reattribute_players_chain_trust.py`:
  - 4-gate predicate truth table (each gate fails individually, all pass, all fail).
  - Cross-team override fires on a constructed minimal rally where current=wrong-team-nearest, candidate=correct-team-second-nearest.
  - Override SUPPRESSED when chain has an UNKNOWN action between the serve and the current action.
  - Override SUPPRESSED when no candidate on the expected team is within the distance cap.
  - Override SUPPRESSED when `Contact.court_side` actively disagrees (not "unknown").
- Re-running the baseline script before/after to produce the A/B comparison.

### Out of scope (deferred to v2 or other workstreams)

- Same-team-as-server attribution errors (4 of 7 same-team errors, all serves). Different lever: pose / ball-toss / jump-detection at serve. Larger workstream — needs pose-feature pipeline integration.
- Synthetic-serve `pl_pid=-1` cases (3 unknown_team errors). Already handled by `_attribute_synthetic_serves` in `cli/commands/reattribute_actions.py`; failures are upstream (no candidate on the serving side within ±5 frames). Fix belongs in the synthetic-serve workstream, not here.
- Action-detection "missing" rate (28/136 = 20.6%). Belongs to ball-tracker / contact-detector workstreams (see `contact_detection_ceiling_2026_05_11.md`).
- Decoupling `action_ground_truth_json` from `PlayerTrack` lifecycle so retracks don't wipe GT. Flagged for follow-up; recommended approach is moving GT onto a `RallyActionGroundTruth` table keyed on `rally_id`. Not this workstream.

## Design

### §1 — Where the change lives

`analysis/rallycut/tracking/action_classifier.py`, inside `reattribute_players` Pass 2 (line ~2920–3025). The fix touches:

- `_compute_expected_teams` (line 2808) — extended return type to include `chain_integrity`.
- Add new module-level helper `_team_chain_trust` (alongside `_compute_expected_teams`).
- Replace lines 2963–2968 with the four-gate predicate.

The change does NOT touch:
- Contact detection / candidate ranking in `contact_detector.py`.
- `_compute_expected_teams`'s upstream serve identification.
- `correct_team_from_propagation` Pass (still runs after Pass 2; no behavior change expected).
- `_reattribute_server_exclusion` (Pass 1) or `_reattribute_reid` (Pass 3).
- Production deploy path: `rallycut reattribute-actions <video>` continues to be the single entry point. No retracking required, so no GT wipe risk on the 3 fresh-GT videos.

### §2 — The trust-the-chain predicate

For each action `i` where current_team ≠ expected_team and current attribution is the nearest candidate, override fires iff ALL of:

| Gate | Definition | Failure semantics |
|---|---|---|
| **G1** | `action.confidence ≥ 0.7` | Action type classification must be confident. Existing soft threshold; reused for symmetry. |
| **G2** | `chain_integrity[i] == True` | No UNKNOWN action types and no `is_synthetic` actions between the seed serve and `actions[i]`. Direct port of the 2026-04-12 warning into an explicit guard. |
| **G3** | `∃ candidate cand with team_assignments[cand.tid] == expected_team AND cand.dist ≤ 1.5 × current.dist` | The existing `max_distance_ratio` cap, but evaluated BEFORE the override decision. If no such candidate exists, override is meaningless. |
| **G4** | `Contact.court_side` at this frame is one of `{"near", "far"}` AND consistent with `expected_team` per `_SIDE_TO_TEAM` | Independent ball-trajectory corroborator. `_resolve_court_side` uses ball position + homography, NOT player identity, so this is a genuinely orthogonal signal. If `court_side == "unknown"` (no calibration / ambiguous), G4 passes with a debug-log marker indicating low-corroboration override. |

When ANY gate fails, the override is suppressed and the current attribution is left in place — the new behavior is a strict relaxation of the existing guard, never a strict tightening.

### §3 — Chain-integrity helper

```python
def _chain_integrity(
    actions: list[ClassifiedAction],
    i: int,
) -> bool:
    """Return True iff the chain from the seed serve to action[i] is unbroken.

    The chain is broken by any UNKNOWN action type or any synthetic action
    between the serve and action[i]. Returns False also when no serve has been
    seen yet at position i.
    """
    seen_serve = False
    for j in range(i + 1):
        a = actions[j]
        if a.action_type == ActionType.SERVE and a.player_track_id >= 0 and not a.is_synthetic:
            seen_serve = True
            continue
        if not seen_serve:
            continue
        if a.action_type == ActionType.UNKNOWN:
            return False
        if a.is_synthetic:
            return False
    return seen_serve
```

Computed for every `i` in one O(n²) pass inside the extended `_compute_expected_teams` (or O(n) by carrying a running "chain broken" flag). Returned alongside the expected-team array as a parallel `list[bool]`; no per-action recompute downstream.

### §4 — Code-shape of the predicate site

Replacing lines 2963–2968:

```python
# OLD: unconditional nearest-guard
# if (
#     not is_unmapped
#     and contact.player_candidates
#     and contact.player_candidates[0][0] == action.player_track_id
# ):
#     continue

# NEW: nearest-guard, relaxable when team-chain trust gates pass
current_is_nearest = (
    contact.player_candidates
    and contact.player_candidates[0][0] == action.player_track_id
)
if not is_unmapped and current_is_nearest:
    override_allowed = _team_chain_override_allowed(
        action=action,
        contact=contact,
        expected_team=expected_team,
        chain_integrity_i=chain_integrity[i],
        team_assignments=team_assignments,
        max_distance_ratio=max_distance_ratio,
    )
    if not override_allowed:
        continue
```

`_team_chain_override_allowed` is the pure-function combination of G1+G2+G3+G4. Tested as a truth table in isolation.

### §5 — Env flag for rollback

```python
RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN = (
    os.environ.get("RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN", "1") != "0"
)
```

Read at call time (inside the predicate) so test harnesses can flip it via `monkeypatch.setenv`. When `0`, the old unconditional guard behavior is restored — useful for fast rollback if a fleet deploy regresses unexpected metrics.

## Validation

### Pre-ship gates

Run `scripts/measure_attribution_fresh_gt.py` before and after on the 3 GT videos. All gates must pass to ship:

1. **Combined `correct_rate` improves by ≥ +5pp** (60.3% → ≥ 65%).
2. **`wrong_cross_team` reduces by ≥ 50%** (16 → ≤ 8).
3. **No per-fixture `correct` count regression** (cece 22 / gigi 35 / wawa 25 are floors).
4. **`wrong_same_team` count non-increasing** (7 today; guards against bad cross-team flips landing as bad same-team flips).
5. **`audit-coherence-invariants` on the 3 videos**: C-2 (alternating-possession) violation count ≤ current baseline on the 3 videos.

### Side-bench: panel PERMUTED PID accuracy

Run `scripts/measure_pid_accuracy.py` on the panel (which includes wawa = 5c756c41). Confirm PERMUTED accuracy ≥ 95.1% averaged across 7 Phase-0 fixtures (the 2026-05-11 baseline). The fix doesn't touch tracking, so this is a paranoia check, not an expected-effect check.

### Sanity probes

- For each cross-team error that the new logic fixes, log the gate-pass profile (which gates passed, which were soft-pass via court_side="unknown"). Used to characterize the win distribution.
- For each cross-team error that the new logic does NOT fix, log which gate vetoed it. Used to inform v2 design.

### Failure-mode telemetry

Emit a single structured INFO log line per override fire (and per gate-veto when the chain is broken). Format:

```
team_chain_override frame=92 rally=f978201e action=receive
    old_pid=4 old_team=B new_pid=1 new_team=A
    gates G1=pass G2=pass G3=pass(0.067/0.105) G4=pass(near)
```

Volume estimate: ≤ 1 line per ~5 actions ≈ ~30 lines per typical rally. Acceptable.

## Risk

| Risk | Mitigation |
|---|---|
| Chain trust fails in a way we don't catch — override flips a correct attribution to wrong | G4 (independent ball-trajectory corroborator) is the load-bearing safety net. The 2026-04-12 ball-Y experiment shows what happens when only the chain signal is used; G4 prevents the same failure mode here. |
| `Contact.court_side` is `"unknown"` frequently → G4 becomes a soft-pass too often → effectively reduces to chain-only signal | Measure `court_side` coverage on the 3 GT videos. If "unknown" rate is > 30% on the cross-team error rows, harden G4 to a hard-pass (no soft-pass when unknown). Listed as a follow-up calibration in §validation. |
| `chain_integrity` computation is wrong (off-by-one, mis-categorized "break") | Unit tests cover the chain function in isolation. |
| Deploy regresses some non-measured metric (e.g., score tracking via `compute-match-stats`) | Env flag `RELAX_NEAREST_GUARD_FOR_TEAM_CHAIN=0` provides instant rollback without code revert or redeploy. Memory entry post-ship will name the flag. |

## Roll-out

1. Land the code change + tests behind the env-flag-on default.
2. Run baseline script before/after on the 3 GT videos. Validate all 5 pre-ship gates.
3. Run `audit-coherence-invariants` before/after on the 3 GT videos.
4. If green, run `rallycut reattribute-actions <video>` on the fleet (~70 videos, ~1 min each).
5. After fleet deploy: re-run `audit-coherence-invariants` fleet-wide and compare to the 744-violation baseline. Expectation: net reduction concentrated in C-2.
6. Write memory entry `attribution_team_chain_v1_2026_05_11.md`. Schema mirrors `serve_peak_prepend_v13_2026_05_11.md` (date, scope, before/after, commits, env flag).

## Non-goals (explicit)

- **NOT** a rewrite of `reattribute_players`. Surgical relaxation of one guard.
- **NOT** introducing new tracking or pose pipelines. v1 uses only signals already produced by the current pipeline.
- **NOT** the final word on attribution accuracy. v2+ will tackle the same-team-server problem and the synthetic-serve attribution failures.

## Open questions for the implementation plan

- Cleanest split of the new helpers between `_compute_expected_teams` (extended to also return `chain_integrity`) and `_team_chain_override_allowed` (the 4-gate predicate). Suggested: keep chain-data in `_compute_expected_teams`, predicate in the new function. Names finalized in this spec.
- Whether to also add a small integration test that runs `reattribute_players` end-to-end on a constructed rally fixture (in addition to the predicate unit tests). Strongly recommended for review confidence; low cost.
- Where the log line goes: `logger.info` is conservative; `logger.debug` reduces fleet log volume but loses post-deploy telemetry. Suggested: `logger.info` for the first deploy, then downgrade to `debug` after we've characterized the win distribution.
