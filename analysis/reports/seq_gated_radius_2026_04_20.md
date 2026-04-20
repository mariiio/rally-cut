# Seq-Gated `player_contact_radius` Relaxation — NO-GO (2026-04-20)

**Status:** NO-GO. Reverted. Audit's +30-35 TP projection was too optimistic — the rescue interacts with the 12-frame dedup gate in ways the crosstab didn't model.

## What was tested

Post-GBM rescue branch at `contact_detector.py:2374` (now reverted): accept rejected candidates when
- `cfg.enable_sequence_recovery` (True by default)
- `player_dist ∈ (0.15, 0.20]` (normalized image coords)
- `_has_sequence_support(frame)` (MS-TCN++ non-bg peak ≥ `SEQ_RECOVERY_TAU=0.80` within ±5f)

The rescue stamped `confidence = 0.25` so rescued contacts could be tracked.

## Measured LOO-per-video deltas vs baseline

Baseline (`contact_baseline_loo_video_2026_04_19.md`):
- Contact F1 = 88.0% (P=91.1%, R=85.0%), TP/FP/FN = 1781/174/314
- Action Accuracy = 91.2%

With rescue active (`reports/eval_loo_video_flag_off_2026_04_20.md`):
- Contact F1 = **88.0%** (P=91.1%, R=85.0%), TP/FP/FN = **1781/174/314** — IDENTICAL to baseline
- Action Accuracy = **90.8%** (-0.4pp)

**Net: zero contact lift, slight action-acc drop.** Fails the plan's ship gate (F1 ≥ 87.8% + recall ≥ 85.3% = lift ≥ 0.3pp).

## Why the crosstab prediction missed

A standalone A/B probe (30 rallies, pre-loaded GBM) confirmed the rescue fires **~11 times per 30 rallies** (~133 rescues extrapolated to 364 rallies). But LOO's TP/FP/FN are bit-identical to baseline. The explanation:

- Rescued contacts arrive near existing accepted ones → the 12-frame dedup merges them.
- Rescued contacts near the dedup boundary DISPLACE pre-existing accepts (confidence 0.25 beats the real contact's 0.45 occasionally, flipping the accept pick).
- Net contact count change ≈ 0, but action-type assignments shift — hence the Action Acc drop.

The audit's Section 2.2 counted 38/43 `no_player_nearby` FNs as "rescue targets" based on seq endorsement, but did not account for the dedup interaction. The rescue fires far more broadly than on those 38 FNs (the seq model peaks near many non-FN candidate frames), and the dedup mostly absorbs the extras.

## What would actually work

To rescue the 38 specific FNs without dedup interference:
1. **Gate on "no accepted contact within 12 frames"** — only rescue when dedup won't immediately coalesce.
2. **Require `classifier_conf ≥ 0.10`** — adds the two-signal conf-band gate the dead `SEQ_RECOVERY_CLF_FLOOR` was originally designed for.
3. **Consume the rescue in the Viterbi decoder** (Task 1 v2) — the decoder's structural prior disambiguates "should this rescued contact replace an existing one, or sit alongside, or be rejected".

Of these, (3) is the cleanest. The conf-band rescue (option 2) would also need its own gate wiring — the `SEQ_RECOVERY_CLF_FLOOR` constant is dead code per `reports/floor_probe_2026_04_20.md`.

## Verdict

Do not ship this rescue shape. The cleanest path forward is Task 1 v2 (full decoder accept-loop restructure), which naturally handles the 92 conf<0.05/seq≥0.95 FNs AND the 38 distance-band FNs as a single problem.

## Cross-references

- Plan: `docs/superpowers/plans/2026-04-20-action-detection-fixes.md` Task 2 (rescoped to NO-GO)
- Audit §2.2: `analysis/reports/action_detection_audit_2026_04_20.md`
- Flag-off LOO measurement: `analysis/reports/eval_loo_video_flag_off_2026_04_20.md`
- Related: `analysis/reports/floor_probe_2026_04_20.md` (dead SEQ_RECOVERY_CLF_FLOOR)
