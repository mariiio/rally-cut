# Synthetic-serve contribution — LOO eval correction

**Date:** 2026-04-23
**Context:** NO-GO closure for `player_contact_radius` loose led to a question: does the LOO serve-recall number (65.1 %) reflect production behavior? The `_make_synthetic_serve` mechanism in `action_classifier.py` and Pass-2 synthetic matching in `eval_action_detection.py` are both in production, but `eval_loo_video.py` was filtering synthetic serves OUT before matching. This memo corrects that measurement and re-grades the Phase B (rally-start serve prior) ROI case.

## Code change

`scripts/eval_loo_video.py`:
- Added `--include-synthetic` flag (default off, preserving historical numbers).
- When enabled, runs Pass-2 `_match_synthetic_serves` against unmatched GT serves with ±1 s tolerance — matching the production-aligned eval in `eval_action_detection.py`.

## 4-way LOO comparison

| Metric | base r=0.15 no-synth | base r=0.15 +synth | loose r=0.20 no-synth | loose r=0.20 +synth |
|---|---:|---:|---:|---:|
| Contact F1 | 89.0 % | **89.8 %** | 88.9 % | 89.6 % |
| Action Acc | 91.7 % | 91.3 % | 91.6 % | 91.3 % |
| TP | 1809 | 1841 | 1810 | 1839 |
| FP | 159 | 164 | 166 | 172 |
| FN | 286 | **254** | 285 | 256 |
| Serve F1 | 71.7 % | **78.8 %** | 72.3 % | 78.9 % |
| Serve Recall | 65.1 % | **76.4 %** | 66.5 % | 76.9 % |
| Serve TP / FN | 237 / 127 | 278 / 86 | 242 / 122 | 280 / 84 |

## Findings

### 1. Synthetic-serve mechanism works

+41 serve TPs / −41 serve FNs vs. no-synth eval. Production-aligned serve F1 is **78.8 %** (not 71.7 %). All prior memo citations of 65.1 % / 71.7 % reflect the research LOO, not production.

### 2. `no_player_nearby` NO-GO verdict stands under synth-enabled eval

Head-to-head within synth-enabled regime:
- Contact F1 Δ: −0.2 pp
- Action Acc Δ: tied (TP −2)
- FP Δ: +8
- FN Δ: +2

Still net-negative. Original NO-GO verdict is correct under both evaluation regimes.

### 3. Synthetic-serve matches are partially imperfect

Adding synth matching contributes +32 TPs to the Action Acc denominator but only ~21 of those are fully-correct (right class, right side, right track). **~33 % of synthetic serve matches fire at the wrong spot or wrong side.** Action Acc actually drops slightly (91.7 → 91.3) because of this.

Root cause candidates:
- Anchor: `_make_synthetic_serve` uses `rally_start_frame` within 90f window, else `first_contact_frame - 30`. Hardcoded −30 is a median; actual serve-flight time varies 15–35 frames depending on serve type.
- Court-side inference: `_infer_serve_side` uses trajectory or position heuristics, both imperfect.

## Revised next-workstream ranking

1. **Improve synthetic-serve anchor + side inference (~3–4 hr).**
   - Replace hardcoded `first_contact − 30` with ballistic backward-fit using 3-5 frames of pre-first-contact ball trajectory.
   - Use Probe 1's net-line estimator (`analysis/rallycut/court/net_line_estimator.py`) for better court-side inference.
   - Expected: 10–15 partial-matches upgraded to full TPs → +0.3 to +0.5 pp Action Acc.
   - Zero cross-class regression risk (scoped to synthetic serves only).

2. **Phase B rally-start serve prior (~2–3 hr).**
   - Targets the remaining 86 serve FNs (cases where no first contact was detected AT ALL, so no phantom-serve path fires).
   - 86 is still the single largest class-FN pool (34 % of all FNs).
   - Boost serve-candidate acceptance when `frames_since_rally_start < 90` AND candidate pattern matches serve signature.
   - Expected: +15–25 serve TPs → +1.0 to +1.5 pp overall Contact F1.

Both are independent. Both should ship gated by `eval_loo_video.py --include-synthetic` as the canonical evaluation regime going forward.

## Action for existing memos

`memory/player_contact_radius_loose_nogo_2026_04_23.md` will be updated to note:
- Baseline serve recall under production-aligned eval is 76.4 %, not 65.1 %.
- NO-GO verdict still holds under synth-enabled eval (Contact F1 −0.2 pp).

## Convention going forward

- All future LOO evals for action/contact workstreams should use `--include-synthetic`.
- The research-only (no-synth) regime stays available for isolating the real-detection emission layer but is NOT the production-aligned gate.
- The `--include-synthetic` flag should be added to the canonical eval command list in `analysis/CLAUDE.md`.

## Artifacts

- `reports/loo_baseline_radius015_synth.{md,json}`
- `reports/loo_loose_radius020_synth.{md,json}`
- `scripts/eval_loo_video.py` — modified to support `--include-synthetic`
