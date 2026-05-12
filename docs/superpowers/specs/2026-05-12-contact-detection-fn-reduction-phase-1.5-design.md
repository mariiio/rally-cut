# Contact Detection FN Reduction — Phase 1.5 Design

## Status

Delta from Phase 1 (`2026-05-12-contact-detection-fn-reduction-design.md`,
NO-SHIP verdict captured in
`analysis/reports/contact_detection_fn/measurement_2026_05_12.md`).

Phase 1 conclusively showed validation-gate relaxation alone cannot move recall
on the 409-rally action-GT corpus. The probe's diagnostic confirmed why: only
14 of 173 ball-tracked-no-contact cases gained a candidate within ±10 frames
of the GT frame with three validation gates relaxed simultaneously. **The
binding constraint is candidate CREATION, not validation.**

Phase 1.5 tests the next hypothesis: relax the **generator-creation
thresholds** that determine whether candidates are created in the first
place. Reuses Phase 1's infrastructure (env-flag mechanism, measurement
harness, A/B methodology, FP guards) verbatim.

## Goal

Surface more contact candidates AT the GT frames by relaxing the
generator-creation thresholds in `_find_direction_change_candidates`,
`_find_velocity_peak_candidates` (and its peak-prominence dependency), and
`_find_parabolic_breakpoints`. **Primary success signal: the diagnostic
count — how many of the 173 probe FN cases now have a contact within ±10 of
GT frame. Target ≥ 80 of 173 (vs 14 today with Phase 1's relaxations).**

## Hypothesis test

The same diagnostic as Phase 1's combined-flag A/B (count of probe cases
with contact at GT frame). Three outcomes:

| Diagnostic count after combined Phase 1.5 flags | Implication |
|---|---|
| **≥ 80 of 173** (≥46%) | Hypothesis confirmed — generators were the constraint. Phase 1.5 ships if FP gates also pass. |
| **40-80 of 173** (23-46%) | Partial confirmation. Per-flag A/B isolates which generator(s) contribute; mixed verdict per-flag. |
| **< 40 of 173** (<23%) | Hypothesis falsified — even relaxed generators don't fire at GT frames. The signals genuinely aren't there in stored ball trajectories. Escalate to Phase 2 (soft-contact channel) or Phase 1.7 (player-motion-candidates re-enable for blocks). |

## Scope IN

- Six new `*_relaxed` fields on `ContactDetectionConfig` for generator-creation
  thresholds
- Three new env flags following the established Phase 1 pattern:
  - `RELAX_CONTACT_DIR_GEN` (direction-change-candidate generator)
  - `RELAX_CONTACT_VEL_GEN` (velocity-peak generator)
  - `RELAX_CONTACT_PARABOLIC_GEN` (parabolic-breakpoint generator)
- Extension of `_resolve_effective_config` to handle them
- Per-flag and combined unit tests
- A/B measurement: combined flags first (most-likely-to-pass config), then
  per-flag isolation if combined PASSes
- Primary success signal: diagnostic count (probe-cases-with-contact-at-GT)
- Secondary success signals: action recall, action precision proxy,
  per-action-type recall, C-1/C-2/C-3 coherence (same Phase 1 gates)

## Scope OUT (same as Phase 1, plus)

- Player-motion-candidate generator re-enablement — **moved to a separate
  Phase 1.7 workstream** focused specifically on block recall
- GBM classifier threshold relaxation — same Phase 1 rationale (last resort;
  highest FP risk)
- Validation-gate relaxations (already tested in Phase 1; no-op)
- Attribution-layer resilience audit — separate small workstream (user-noted
  principle; only 2 known cascade points, not urgent)

## Generator thresholds to relax

| Generator (function) | Field | Current default | Phase 1.5 relaxed | Rationale |
|---|---|---|---|---|
| `_find_direction_change_candidates` | `direction_change_candidate_min_deg` | 25.0 | 15.0 | Per probe: 68.8% of FN cases failed `min_direction_change_deg=20.0` (validation gate); generator threshold is 25.0, even stricter. Lowering to 15° aligns with the inflection-angle validation gate's relaxed value (10°) without going below it. |
| `_find_direction_change_candidates` | `direction_change_candidate_prominence` | 10.0 | 5.0 | Half the prominence requirement; lets shallower direction-change peaks fire as candidates. |
| velocity-peak generator | `min_peak_prominence` | 0.003 | 0.0015 | Half the prominence; lets smaller velocity peaks register. |
| velocity-peak generator | `min_candidate_velocity` | 0.003 | 0.0015 | Velocity floor for inflection/reversal candidates; same proportional reduction. |
| `_find_parabolic_breakpoints` | `parabolic_min_residual` | 0.015 | 0.010 | Lower residual peak threshold for parabolic-fit breakpoint detection. |
| `_find_parabolic_breakpoints` | `parabolic_min_prominence` | 0.008 | 0.004 | Half the prominence requirement; parallel to other generator relaxations. |

Each relaxation is approximately a **2× reduction** (or 1.67× for direction
change, where 25 → 15 is a 40% drop). The principle: half the strictness;
let the downstream GBM and rescue paths filter the resulting noise.

## Env-flag groupings

Three flags rather than six so each flag tests one generator's contribution
independently:

| Flag | Fields swapped |
|---|---|
| `RELAX_CONTACT_DIR_GEN` | `direction_change_candidate_min_deg`, `direction_change_candidate_prominence` |
| `RELAX_CONTACT_VEL_GEN` | `min_peak_prominence`, `min_candidate_velocity` |
| `RELAX_CONTACT_PARABOLIC_GEN` | `parabolic_min_residual`, `parabolic_min_prominence` |

## Measurement methodology

Identical to Phase 1 (Section 4 of the Phase 1 spec). Same gates G-A through
G-E, same STOP conditions. The new primary signal is the **diagnostic count**
which gives a more direct readout than aggregate recall (recall changes can
be obscured by reshuffling; the diagnostic directly measures "did this
relaxation put a candidate at the GT frame").

Decision matrix:

| Diagnostic count | Aggregate recall Δ | Action |
|---|---|---|
| ≥ 80, recall Δ ≥ +3pp | PASS | Ship combined flags as default-ON; consider per-flag A/B for clarity |
| ≥ 80, recall Δ < +3pp | DONE_WITH_CONCERNS | Candidates surface but downstream filtering still kills them; ship as infrastructure; flag GBM threshold as Phase 1.6 candidate |
| 40-80, recall Δ ≥ +3pp | PASS | Same as first row |
| 40-80, recall Δ < +3pp | Per-flag A/B | Isolate which generator contributed |
| < 40 | NO-SHIP | Escalate to Phase 1.7 / Phase 2 |

## Files touched (delta from Phase 1)

**Modified:**
- `analysis/rallycut/tracking/contact_detector.py` — 6 new fields on
  `ContactDetectionConfig`, 3 new env-flag blocks in `_resolve_effective_config`
- `analysis/tests/unit/test_contact_detector_relaxations.py` — 3 new per-flag
  tests + 1 combined-3-flag test

**Created:**
- `analysis/reports/contact_detection_fn/measurement_phase15_combined_2026_05_12.{md,json}`
- (Conditional on combined PASS) `measurement_phase15_per_flag_*.md`

## Composition with Phase 1 infrastructure

- `_resolve_effective_config` already exists; just add 3 more `if env_get == "1"`
  blocks
- `measure_contact_recall_full.py` works unchanged
- Diagnostic-count script (inline in Phase 1 A/B) becomes the primary verdict
  signal; the queryable JSON contacts_json data is what it reads
- Snapshot/restore pattern works unchanged
- Unit-test fixture pattern reused

Estimated implementation cost: ~1 day (most infrastructure already exists).
A/B execution cost: ~1-2 hours (redetect_all_actions is fast given warm
models).
