# Phase 3 Stage 1 — Rule L windowed-index probe: HYPOTHESIS REFUTED

**Date:** 2026-05-19
**Decision:** Rule L is innocent. The cross-model frame-index phase shift hypothesis is wrong.

## What we tested

The Phase 3 audit's top hypothesis was that `apply_sequence_override` in `sequence_action_runtime.py:260` reads MS-TCN++ per-frame probabilities at the contact's exact frame:

```python
per_frame = sequence_probs[1:, frame]  # OLD: exact-frame index
```

If MS-TCN++ peaks are aligned to v4-pipeline contact frames but our new pipeline shifts frames ~3f earlier, we'd be sampling off-peak. To test, we changed it to a windowed max over ±5 frames:

```python
lo = max(0, frame - 5)
hi = min(sequence_probs.shape[1], frame + 6)
per_frame = sequence_probs[1:, lo:hi].max(axis=1)  # NEW: windowed max
```

The pattern mirrors the SEQ_RECOVERY rescue path in the same file (line 319 uses `max(sequence_probs[1:, f-5:f+6])`), so it's an established idiom.

## Result: byte-identical to Phase 2.5

Ran trusted-31 eval with NEW contact + NEW regressor + V4 action + V4 scorer + windowed-fix:

| Metric | Phase 2.5 (exact frame) | Phase 3 Stage 1 (windowed) | Δ |
|--------|-------------------------|----------------------------|---|
| Contact F1 | 94.1% | 94.1% | **0** |
| Contact Recall | 92.9% | 92.9% | **0** |
| Contact Precision | 95.4% | 95.4% | **0** |
| Action Accuracy | 90.4% | 90.4% | **0** |
| Player Attribution | 87.1% | 87.1% | **0** |
| Court-Side | 86.8% | 86.8% | **0** |
| All per-class F1 | (unchanged) | (unchanged) | **0** |
| All per-class attribution | (unchanged) | (unchanged) | **0** |
| Sanity violations | 33 | 33 | **0** |

**Every single number is identical.** No metric moved by even 0.1pp.

## What this means

Either:
1. **MS-TCN++ peaks are already well-aligned** with the new pipeline's contact frames. The windowed max returns the same value as the exact-frame index because the peak IS at the contact frame.
2. The guards (`OVERRIDE_RELATIVE_CONF_K`, `DIG_GUARD_RATIO`, `ATTACK_PRESERVE_RATIO`) filter out any class-shifts the windowed max would produce.

Either way, **Rule L is not the source of the −0.6pp Action Acc regression.**

This is an unambiguous negative result. The cross-model frame-index phase shift hypothesis is refuted with a definitive A/B (byte-identical metrics across 1445 GT contacts).

## What this rules out

The audit listed Rule L as the **prime suspect** because:
- It's the only rule with INDEX_PHASE sensitivity (the others are RELATIVE/ABSOLUTE_FRAME)
- It cross-cuts two models (GBM contact + MS-TCN++ action), so a misalignment between them would cause exactly the structural regression we see
- It's cheap to test

That hypothesis is now ruled out by direct experiment. The Action Acc regression has a different cause.

## What's left

The remaining audit candidates from PHASE3_AUDIT.md:

| Rule | What it gates | Why it might cause Action Acc regression on Phase 2.5 |
|------|---------------|--------------------------------------------------------|
| Rule B | `serve_window_frames=60` (serve detection) | Serve F1 −1.7pp on Phase 2.5 suggests serve detection misfiring |
| Rule C | Pass 3 flight_window=45 | Receive F1 −2.6pp suggests serve/receive disambiguation drift |
| Rule E | Synthetic serve placement | Affects team_transition feature; serve+receive chain depends on it |
| Rule A | block_max_frame_gap=8 | Block F1 unchanged at 0% (not the issue, but still 30fps-specific) |
| Rule D | time-gap sanity guard | Probably not — uniform shift doesn't affect inter-contact gaps |
| Rule K | Viterbi HMM | Untested; learned component, would require retraining |

## Per-action regression pattern (Phase 2.5 vs v4)

| Action | v4 F1 | Phase 2.5 F1 | Δ |
|--------|-------|--------------|---|
| Serve | 89.5% | 87.8% | **−1.7pp** |
| Receive | 88.6% | 86.0% | **−2.6pp** |
| Set | 86.8% | 86.3% | −0.5pp |
| Attack | 85.6% | 86.2% | +0.6pp |
| Dig | 81.4% | 83.1% | +1.7pp |

**Serve and Receive — the EARLIEST actions in a rally — regress most.** Attack and Dig (later actions) improve. This is a strong pattern that points at **rally-start detection rules** (Rule B serve_window_frames, Rule E synthetic serve placement, Rule H server detection).

The pattern is also consistent with: new contact_classifier accepts slightly different INITIAL contacts → Serve detection picks the wrong first contact → Receive chains off the wrong serve → Set/Attack/Dig get attribution boost from cleaner mid-rally contacts.

## What we know now (concrete, evidence-based)

After Phase 1/2/2.5 + Phase 3 Stage 1:

1. **The 60fps degradation is real** (Phase 1): +4.5pp class-A FN gap, root cause is GBM rejection of valid candidates.
2. **Simple retrain of contact_classifier helps 60fps but causes Action Acc regression** (Phase 2/2.5).
3. **Coordinated retrain of contact + action + scorer doesn't fix Action Acc** (Phase 2).
4. **Coordinated retrain of contact + regressor doesn't fix Action Acc either** (Phase 2.5).
5. **Cross-model frame-index misalignment (Rule L) is NOT the cause** (Phase 3 Stage 1).
6. **The regression pattern shows it's specifically Serve/Receive detection** — early rally actions.

This narrows the field considerably. The next probe should target the early-rally rules.

## Production state

All 4 production models restored to v4 (`md5 -q` verified). `sequence_action_runtime.py` reverted to exact-frame index. No code changes outstanding.
