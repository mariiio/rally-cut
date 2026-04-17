# Pattern A rescue-gate sweep — 2026-04-17

## Context

The commit-65f4ca2 corpus (364 rallies / 2098 GT contacts / 395 FN_contact) showed 71 residual FNs with `classifier_conf ∈ [0.05, 0.20)` AND ≥3 trajectory generators firing AND a player within 0.15 — the user flagged 4 of these as visually obvious contacts missed because the existing one-arm rescue gate requires conf ≥ 0.20.

Hypothesis: a second rescue arm gated by multi-generator trajectory agreement + spatial proximity + sequence endorsement (same τ=0.80) can safely rescue these cases.

## Implementation

`rallycut/tracking/contact_detector.py` — added `_build_generators_by_frame` that attributes each merged candidate frame to the generators that claimed it (velocity_peak / inflection / reversal / deceleration / parabolic / net_crossing), using the same `min_peak_distance_frames` window as `_merge_candidates`. Threaded into the rescue site at `detect_contacts` alongside the existing Arm A gate:

- Arm A (unchanged): `seq_peak ≥ 0.80 AND conf ≥ 0.20` → rescue.
- Arm B (new): `seq_peak ≥ 0.80 AND conf ≥ FLOOR_MULTIGEN AND n_gens ≥ MIN_GENERATORS AND player_dist ≤ 0.15` → rescue.

Constants live in `rallycut/tracking/sequence_action_runtime.py` next to the existing `SEQ_RECOVERY_*` block so `scripts/sweep_rescue_gate.py` can monkeypatch them between cells.

## Sweep setup

Grid: `FLOOR_MULTIGEN ∈ {0.05, 0.08, 0.10, 0.12, 0.15}` × `MIN_GENERATORS ∈ {2, 3}`. Each cell reruns `scripts/build_action_error_corpus.py` on the full 364-rally corpus with the two constants monkeypatched. Baseline disables Arm B (`MIN_GENERATORS=999`) so Arm B's marginal impact is isolated from the companion Pattern-C serve anchor (also dormant-shipped — see the sibling report).

Snapshots land in `outputs/action_errors/sweep/<cell_id>/` (10 grid cells + 2 baselines). Per-rally regression rule: flag a rally if `(TP + wrong_player)` drops by ≥1 OR `extra_pred` rises by ≥2.

## Baseline — Arm A + Pattern C anchor, Arm B disabled

TP=935  FN=394  wrong_action=176  wrong_player=593  extra_pred=304

## Grid results (Δ vs baseline)

| FLOOR | MIN_GENS | ΔTP | ΔFN | Δwrong_action | Δwrong_player | Δextra_pred | n_regressed | Δextra / ΔTP |
|------:|---------:|----:|----:|--------------:|--------------:|------------:|------------:|-------------:|
| 0.05 | 2 | +27 | −48 | +7 | +14 | +126 | 33 | 4.7 |
| 0.05 | 3 | +21 | −44 | +8 | +15 | +101 | 28 | 4.8 |
| 0.08 | 2 | +14 | −31 | +4 | +13 | +74 | 24 | 5.3 |
| 0.08 | 3 | +10 | −27 | +5 | +12 | +60 | 21 | 6.0 |
| 0.10 | 2 | +18 | −23 | +0 | +5 | +53 | 19 | 2.9 |
| 0.10 | 3 | +13 | −19 | +1 | +5 | +41 | 17 | 3.2 |
| 0.12 | 2 | +17 | −20 | −2 | +5 | +30 | 15 | 1.8 |
| 0.12 | 3 | +7 | −16 | +6 | +3 | +40 | 10 | 5.7 |
| 0.15 | 2 | +15 | −17 | +1 | +1 | +22 | 5 | 1.5 |
| 0.15 | 3 | +8 | −14 | +3 | +3 | +19 | 4 | 2.4 |

## Decision

Pre-registered rule: maximize ΔTP subject to `Δextra_pred / ΔTP ≤ 0.5`, `n_rallies_regressed ≤ 5`, `ΔFN ≤ 0`, and all 4 user-confirmed rescues recovered.

- **`Δextra_pred / ΔTP ≤ 0.5`**: no cell cleared it. The best cell (0.15, 2) is at 1.5; most are 2+.
- **`n_rallies_regressed ≤ 5`**: only (0.15, 2) = 5 and (0.15, 3) = 4 cleared it.
- **`ΔFN ≤ 0`**: every cell cleared it (rescue reduces FNs by design).
- **User rescues = 4/4**: 0/4 across every cell. Post-mortem on the 4 cases:
    - 3 of the 4 are `rejected_by_classifier` with 2–3 generators fired per the `diagnose_fn_contacts` tolerance-based re-count; the 4th (`99a01ce4:813`) is `no_player_nearby` — structurally unreachable by Arm B.
    - The production `_build_generators_by_frame` attributes generators to *merged candidate frames*, not to GT frames. When a generator fires near GT but gets consolidated with an adjacent stronger peak by `_merge_candidates`, the count at the merged frame can diverge from the diagnostic's re-count, leaving `n_gens` below `MIN_GENERATORS` at the site where the gate actually evaluates. The hypothesis that "diagnostic-reported 3-gen cases will rescue" was not validated.

### Verdict: **DORMANT SHIP**

No cell clears the precision budget. The modest TP gain (8–27) is dwarfed by the extra_pred cost (19–126) at every threshold, and the 4 user-flagged cases don't recover at any cell due to the generator-attribution-site mismatch above. Shipping the gate live would trade contact_recall (+0.9–3.1pp depending on cell) against action_accuracy (flat-to-slightly-negative) and per-rally regressions — a bad bargain without a precision-improving follow-up.

`rallycut/tracking/sequence_action_runtime.py` sets `SEQ_RECOVERY_MIN_GENERATORS = 999` (unreachable) as the module default. Arm B, its tests, the `_build_generators_by_frame` helper, the sweep harness, and the per-rally snapshots all stay in place for future retuning — the measurement harness is reusable, and the honest negative result is preserved. Matches the `LEARNED_MERGE_VETO_COS=0.0` precedent in `rallycut/tracking/merge_veto.py` and the Session-4 within-team-ReID NO-SHIP.

## Follow-up directions

Two changes might close the precision gap enough to re-enable Arm B:

1. **Anchor `n_gens` at GT-tolerance semantics rather than merged-frame semantics.** Replace `_build_generators_by_frame` with a helper that counts how many generator frames lie within `±tolerance` of each merged candidate (matching the diagnostic's semantics). This directly tests whether the hypothesis failed because of the attribution-site mismatch or because multi-gen agreement is genuinely a weak FP-discriminator.
2. **Retrain the 25-dim GBM with the 71 user-sourced residual FNs as positives.** The current rescue gate fights the classifier's distribution; a retrain that shifts the conf distribution upward on these cases would let them exit through Arm A (the already-validated path) without introducing a second gate.

## Files

- Implementation: `rallycut/tracking/contact_detector.py` (`_build_generators_by_frame` at line ~1448, two-arm gate at line ~2154).
- Constants + dormant rationale: `rallycut/tracking/sequence_action_runtime.py:340-370`.
- Tests: `tests/unit/test_contact_detector.py` — `TestBuildGeneratorsByFrame` (3) + `TestSequenceRecoveryGate` (5).
- Sweep harness: `scripts/sweep_rescue_gate.py`; aggregator: `scripts/analyze_sweep_and_pattern_c.py`.
- Snapshots: `outputs/action_errors/sweep/<cell_id>/{corpus.jsonl,rally_quality.json,stdout.log}`.
- FP metric wiring: `scripts/build_action_error_corpus.py:357+` (persists `extra_predictions` per rally on `RallyQuality`).
