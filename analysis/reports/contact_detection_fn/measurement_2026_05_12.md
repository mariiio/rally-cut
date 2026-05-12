# Contact Detection FN Reduction (Phase 1) — Final Workstream Report 2026-05-12

## Summary

**Verdict: NO-SHIP for Phase 1. Validation-gate relaxation infrastructure ships
as default-OFF measurement hooks. Workstream escalates to Phase 1.5
(generator-threshold relaxations).**

The probe correctly identified that 173 of 217 missing GT actions on the
409-rally action-GT corpus had the ball tracked at the GT frame — pointing the
finger at contact detection (not ball tracking) as the FN bottleneck. The
probe's per-gate failure histogram (`direction_change_deg` 68.8%,
`inflection_angle_deg` 66.5%, `velocity` 44.5%) suggested validation-gate
relaxation as the lever.

Three A/B configurations were tested:
1. `RELAX_CONTACT_DIR_CHANGE` alone → recall **0.0pp** Δ (commit `84a40543`)
2. `DIR_CHANGE + INFLECTION + VELOCITY` → recall **+0.08pp** Δ (commit `e3807e92`)
3. All 5 flags ON → recall **-0.13pp** Δ (regression; addendum in same report)

A diagnostic confirmed the structural reason: only **14 of 173 (8.1%)** probe
FN cases had a contact within ±10 frames of `gt_frame` even with three flags
relaxed. **Generators are not firing at the GT frames in the first place** —
loosening the validation gates that would accept those candidates is moot
because no candidates exist.

## Probe summary

- N ball-tracked-no-contact cases identified: **173 of 217 missing GT actions**
- Dominant per-gate failures (independent counts; cases overlap):
  - `direction_change_deg` ≥ 20°: 119 (68.8%)
  - `inflection_angle_deg` ≥ 15°: 115 (66.5%)
  - `velocity` ≥ 0.008: 77 (44.5%)
  - `min_candidate_velocity` ≥ 0.003: 53 (30.6%)
  - `player_radius_depth_scaled`: 41 (23.7%; 50% on serves specifically)
- Per-action breakdown (most-missed types): dig (49), attack (37), serve (34),
  set (27), receive (25), block (1)

## Per-flag A/B results

| Flag(s) | Recall | Δ recall | C-1 | Verdict |
|---|---|---|---|---|
| Baseline (all OFF) | 0.8909 | — | 112 | — |
| `RELAX_CONTACT_DIR_CHANGE` | 0.8909 | 0.0pp | 104 | **FAIL G-A** |
| `DIR + INFL + VEL` | 0.8917 | +0.08pp | 88 | **FAIL G-A** |
| All 5 flags | 0.8896 | **-0.13pp** | 112 | **FAIL G-A (regression)** |

## Ship list

**Default-ON in production: NONE.** All `RELAX_CONTACT_*` flags remain
default-OFF. Production behavior unchanged.

**Held / failed:** All 5 flags. Reason: validation-gate relaxation alone
cannot move recall by ≥+3pp on the corpus.

## Composition check (Plan Task 10): N/A

Skipped — nothing shipped to compose with. The relaxation infrastructure that
landed is no-op at default flags; downstream pipelines (action_classifier,
reattribute_players, audit-coherence-invariants) see byte-identical input.

## What we built (infrastructure that stays)

Even though no flag ships, the workstream produced reusable measurement and
configuration infrastructure that future workstreams (Phase 1.5+) can pick up
without rebuilding:

1. **Probe** (`scripts/probe_contact_gate_failures.py`) — per-frame gate
   diagnosis on the 409-rally corpus. Reusable as-is for any future
   gate-relaxation hypothesis.
2. **Relaxation infrastructure** (`contact_detector.py` —
   `_resolve_effective_config(cfg)` + 6 `*_relaxed` fields). Pattern is
   trivially extensible: add a new `RELAX_CONTACT_<X>` flag by adding a
   `_relaxed` field and one block in the helper. Phase 1.5 will reuse this
   exact mechanism for generator thresholds.
3. **Measurement harness** (`scripts/measure_contact_recall_full.py`) —
   recall + precision proxy + per-action-type + coherence on the full
   action-GT corpus. ~1-2 minute baseline run; the standard A/B reference
   for any contact-detection change.
4. **DB snapshot/restore pattern** (`/tmp/combined_pre_ab_snapshot.jsonl` +
   inline restore script). Cheap rollback for any future workstream that
   needs to mutate `contacts_json`/`actions_json` and revert.
5. **Diagnostic-as-data** (the 14/173 contact-at-GT-frame check). Future A/Bs
   should use this as a primary pass/fail signal alongside recall —
   it directly tests "did this change put a candidate near the GT frame?"
   which is more diagnostic than aggregate recall.

## Per-action recall floor (informative for future workstreams)

Pre-existing baseline floors that this workstream did NOT move:

| Action | Recall | n_gt | Notes |
|---|---|---|---|
| **block** | **10.8%** | 37 | Striking floor; small sample size; mostly A_NO_BALL_AT_GT (ball not tracked at block frame) |
| dig | 83.6% | 377 | Soft contacts; trajectory bends are subtle |
| serve | 86.3% | 408 | Server often far from camera; player_radius issues |
| receive | 91.9% | 372 | After serve; usually visible |
| set | 93.4% | 527 | Soft, mid-court; some trajectory smoothing issues |
| attack | 93.4% | 653 | Sharpest signal; highest baseline |

Block recall (10.8%) is the most striking floor and the most likely
beneficiary of either GT expansion (Phase 3 retrain) or ball-tracking
improvements.

## Recommended next workstream — Phase 1.5

The validation-gate ladder is exhausted. The next investment should target
**generator-creation thresholds**, which determine whether candidates are
created in the first place:

| Field | Current default | Suggested relaxed | Generator |
|---|---|---|---|
| `direction_change_candidate_min_deg` | 25.0 | 15.0 | `_find_direction_change_candidates` |
| `min_peak_prominence` | 0.003 | 0.0015 | velocity-peak generator |
| `parabolic_min_residual` | 0.015 | 0.010 | parabolic-fit generator |
| `parabolic_min_prominence` | 0.008 | 0.004 | parabolic-fit generator |
| `min_candidate_velocity` | 0.003 | 0.0015 | velocity-floor generator gate |

Concrete plan:
1. Add `*_relaxed` fields for the above to `ContactDetectionConfig`
2. Extend `_resolve_effective_config` with new `RELAX_CONTACT_GENERATOR_*` flags
3. Re-run the same A/B loop with the SAME pre-ship gates
4. Use the **diagnostic count** (probe-cases-with-contact-at-GT-frame) as the
   PRIMARY success signal, with recall as confirmation. Target: ≥80 of 173
   probe cases now have a candidate at GT frame (vs 14 today).

If Phase 1.5 also caps at low recall gain, escalate to:
- **Phase 2 (soft-contact channel)** — architectural; tag relaxed candidates
  as `is_soft=True` so downstream layers (PGM-style attribution) can consume
  them without committing
- **Phase 3 (per-action-type GBM retrain)** — bottlenecked on GT expansion
  (separate workstream in flight)

Plus the orthogonal:
- **Block-recall investigation** — separate from this workstream's scope but
  the most striking single number on the board

## Architecture-of-attribution implication

The original session brainstormed PGM with absent-actor states (Framing 1)
as an attribution architecture. The +13pp ceiling for that work is unchanged
— but the relative attractiveness depends on what other ladders do. If
Phase 1.5 / Phase 2 close the contact-detection gap by even 5-10pp, the
remaining attribution-addressable error budget shrinks and the case for
investing in the PGM gets weaker. If contact detection caps near current
levels, the PGM stays the right next architecture eventually.

## Files

- Spec: `docs/superpowers/specs/2026-05-12-contact-detection-fn-reduction-design.md`
- Plan: `docs/superpowers/plans/2026-05-12-contact-detection-fn-reduction.md`
- Probe: `analysis/scripts/probe_contact_gate_failures.py`
- Measurement: `analysis/scripts/measure_contact_recall_full.py`
- Infrastructure: `analysis/rallycut/tracking/contact_detector.py`
- Unit tests: `analysis/tests/unit/test_contact_detector_relaxations.py`
- Reports: `analysis/reports/contact_detection_fn/`

## Commits (in order)

| SHA | Subject |
|---|---|
| `12621d6` | feat(contact-detection): probe script for gate-failure diagnosis |
| `3ab2c45` | fix(contact-detection): probe code-quality cleanup (mypy + loop refactor) |
| `d196d4a` | report(contact-detection): probe results + decision footer |
| `e31f393` | feat(contact-detection): add *_relaxed fields to ContactDetectionConfig |
| `16d0f3f7` | feat(contact-detection): add _resolve_effective_config helper |
| `56e6368` | feat(contact-detection): wire _resolve_effective_config into detect_contacts |
| `0a6c05e` | test(contact-detection): unit tests for _resolve_effective_config |
| `2c15fe3b` | feat(contact-detection): full-corpus recall + precision-proxy script |
| `17db1e43` | report(contact-detection): baseline measurement |
| `84a40543` | report(contact-detection): A/B verdict RELAX_CONTACT_DIR_CHANGE = FAIL |
| `e3807e92` | report(contact-detection): A/B verdict combined DIR+INFL+VEL = FAIL, pivot |
| `4c5ff913` | report(contact-detection): all-5-flags follow-up addendum |
| (this commit) | report(contact-detection): final workstream report 2026-05-12 |
