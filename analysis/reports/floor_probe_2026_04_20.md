# SEQ_RECOVERY_CLF_FLOOR Probe — NO-GO (2026-04-20)

**Status:** NO-GO. The probe was designed on a premise that turned out to be wrong — the constant `SEQ_RECOVERY_CLF_FLOOR` is **not wired** into any production gate.

## What was planned

The 2026-04-20 audit's Section 2.1 proposed lowering `SEQ_RECOVERY_CLF_FLOOR` from 0.20 to 0.15 or 0.10 to rescue 26–55 FNs. Plan Task 3 scoped running `scripts/sweep_sequence_recovery.py` across the grid to see if any cell cleared the ship gate.

## What was found

A grep audit of the full codebase confirms `SEQ_RECOVERY_CLF_FLOOR` is referenced in:
- `rallycut/tracking/sequence_action_runtime.py:337` — constant definition (`= 0.20`)
- `rallycut/tracking/sequence_action_runtime.py:321` — docstring
- `rallycut/tracking/contact_detector.py:243, 1863` — docstrings describing the "two-signal agreement gate"
- `rallycut/cli/commands/track_player.py:977` — comment

**Zero call sites.** The `_has_sequence_support(frame)` helper at `contact_detector.py:2149` was defined but never called until the 2026-04-20 Task 2 wiring. The `SEQ_RECOVERY_CLF_FLOOR` constant was never consumed by any gate. It is dead configuration.

The `scripts/sweep_sequence_recovery.py` harness monkey-patches the constant value between runs, but since no code reads it, the sweep measures no signal — every cell would report the same F1.

## Retrospective on the audit's §2.1 simulation

The audit's crosstab-based simulation ("Lower floor 0.20 → 0.10 would admit 26/154 FNs") was a **counterfactual on a gate that does not exist**. The numbers are informative for *designing* a new gate, but cannot be achieved by tuning a constant — the gate logic needs to be written first.

## Path forward

There are two distinct gate designs worth wiring:

1. **Two-signal conf-band rescue (the originally-intended design):**
   - Gate: accept rejected candidates when `classifier_conf ≥ SEQ_RECOVERY_CLF_FLOOR AND _has_sequence_support(frame)`.
   - Targets the 26 FNs with `conf ∈ [0.10, 0.20)` AND `seq ≥ 0.80`.
   - Scope: 1-2h (wire gate + unit tests) + LOO A/B + orthogonality probe vs flat-floor.

2. **Distance-band rescue (shipped in Task 2 this cycle):**
   - Gate: accept rejected candidates when `player_dist ∈ (0.15, 0.20] AND _has_sequence_support(frame)`.
   - Targets the 38 seq-endorsed `no_player_nearby` FNs.
   - Already landed at `contact_detector.py:2377` (approximate).

Design (1) is complementary to design (2) — they rescue disjoint FN populations. Design (1) is a reasonable follow-up but would need its own plan + LOO budget. Deferred.

## Recommendation

Do not ship a bare constant change. Either:
- Wire the two-signal conf-band rescue properly (new task, ~1-2h + LOO), or
- Accept that the conf-band FNs are decoder-territory (per the shipped Viterbi integration in Task 1).

The latter is more honest — once the decoder flag flips default-on (after 2-week A/B), the 26 conf-band FNs become decoder emission territory and should not need a separate floor-gate rescue.

## Cross-references

- Plan: `docs/superpowers/plans/2026-04-20-action-detection-fixes.md` Task 3 (rescoped to NO-GO)
- Audit §2.1: `analysis/reports/action_detection_audit_2026_04_20.md`
- Dead-code-point: `analysis/rallycut/tracking/sequence_action_runtime.py:337`
