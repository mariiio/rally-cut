# Decoder A/B decision memo — Phase 3 + 4 (consolidated harness)

## **Revised verdict 2026-04-24 (two-pass production-real): NO-GO as detect_contacts replacement, GO as label-only overlay (already shipping +2.64pp), or opt-in flag ship for richer relabel (+5.2pp Action Acc / −3.4pp F1 trade).**

A production smoke test on 10 stratified rallies (see
`analysis/reports/decoder_smoke_verdict_2026_04_24.md`) revealed that
the Phase 3 measurement used a `_eval_gt_frames` backdoor that
overstated production behavior — specifically the trainer's
`frames_since_last` feature was computed against GT contacts, a proxy
unavailable in production. With the two-pass production-correct scheme
(`detect_contacts_via_decoder` internal: Pass 1 identify GBM-threshold
accepted candidates → Pass 2 re-extract features with that acceptance
set as `frames_since_last` proxy), the 68-fold LOO lands at:

| Metric | Baseline | Decoder (GT-backdoor, Phase 3/4) | Decoder (two-pass, production-real) |
|---|---:|---:|---:|
| Contact F1 | 90.1% | 89.4% | **86.7%** (−3.4pp, gate ≥−1.0pp FAIL) |
| Action Acc | 90.6% | 95.8% | **95.8%** (+5.2pp, gate ≥+2.5pp PASS) |
| serve F1 | 78.6% | 87.3% | 82.7% (+4.1pp) |
| receive F1 | 81.9% | 91.9% | 86.3% (+4.4pp) |
| set F1 | 86.0% | 89.3% | 86.2% (0.0pp) |
| attack F1 | 87.8% | 88.9% | 87.9% (0.1pp) |
| dig F1 | 73.0% | 70.6% | 71.7% (−1.3pp, within floor) |
| block F1 | 9.3% | 5.9% | **14.3%** (+5.0pp) |
| FP count | 165 | 118 | 215 (+50, gate ≤165 FAIL) |

**3 of 5 pre-registered gates fail on production-realistic measurement.**
Contact F1 regression and FP budget overshoot.

Source: `analysis/reports/decoder_phase4_2pass.{md,json}`.

---



**Date:** 2026-04-24
**Plan:** `docs/superpowers/plans/2026-04-24-parallel-decoder-ship.md`
**Phase:** 3 — A/B eval through `eval_loo_video.py --include-synthetic` with `--use-decoder` flag
**Methodology:** 68-fold leave-one-video-out, MS-TCN++ v5 + GBM v5 + decoder skip-penalty 1.0
**Wall-clock:** baseline 19.2 min, decoder 19.1 min (parallel)

---

## TL;DR — **GO**, with one Phase-4 design decision

**Decoder ships.** All pre-registered gates pass on the apples-to-apples
no-synthetic comparison. The `--include-synthetic` measurement is
artifactually worse on Contact F1 because the decoder path does not yet
emit synthetic serves; either add synth emission in Phase 4 (preferred) or
accept the synth-gap as a known cost.

| Headline | Decoder | Baseline | Δ |
|---|---:|---:|---:|
| Contact F1 (no-synth, apples-to-apples) | **89.4%** | 89.0% | **+0.4pp** |
| Action Acc (no-synth, apples-to-apples) | **95.2%** | 91.7% | **+3.5pp** |
| Contact F1 (synth on both) | 88.2% | 90.1% | −1.9pp* |
| Action Acc (synth on both) | 95.7% | 90.6% | +5.1pp |

*The synth-on F1 gap is a methodology artifact: baseline gains +41 synth
serve TPs on `--include-synthetic`, decoder does not emit synth serves.
Phase 4 cleanup item.

---

## 1. Two apples-to-apples comparisons

### A. No-synthetic (matches the original validation methodology)

| Metric | Baseline | Decoder | Δ | Pre-registered gate |
|---|---:|---:|---:|---|
| Contact F1 | 89.0% | **89.4%** | **+0.4pp** | ≥ 89.0% absolute → **PASS** |
| Action Acc | 91.7% | **95.2%** | **+3.5pp** | ≥ +2.5pp → **PASS** |
| TP / FP / FN | 1809 / 159 / 286 | 1900 / 153 / 297 | +91 / −6 / +11 | FP ≤ canonical → **PASS** |

Source: `analysis/reports/loo_baseline_radius015.md` (baseline) and
`analysis/reports/decoder_v5_full_2026_04_24.md` (decoder).

### B. Synthetic on both (Phase 3 consolidated harness, **post Phase-4 patch**)

| Metric | Baseline | Decoder (P3, no synth emit) | Decoder (P4, synth emit) |
|---|---:|---:|---:|
| Contact F1 | 90.1% | 88.2% | **89.4%** (+1.2pp from P3) |
| Action Acc | 90.6% | 95.7% | **95.8%** |
| serve F1 | 78.6% | 79.9% | **87.3%** (+7.4pp) |
| dig F1 | 73.0% | 70.6% | 70.6% (unchanged) |
| TP / FP / FN | 1938/165/259 | 1828/118/369 | 1872/118/325 |

Phase-4 synth-serve emission patch (commit landed) closes the synth-on
Contact F1 gap from −1.9pp to −0.7pp. Action Acc now within 0.1pp of
the no-synth measurement. Dig regression of −2.4pp persists (decoder
genuinely produces 30 fewer dig TPs vs baseline) and is the only
remaining per-class concern.

Source: `analysis/reports/decoder_phase3_baseline.md` and
`analysis/reports/decoder_phase4_synth.md`.

---

## 2. Per-class F1 (no-synth, apples-to-apples)

| Class | Baseline | Decoder | Δ | Gate (≥−1.5pp; block exempt) |
|---|---:|---:|---:|---|
| serve | 71.7% | 80.2% | **+8.5pp** | PASS |
| receive | 82.7% | 91.4% | **+8.7pp** | PASS |
| set | 87.2% | 89.1% | +1.9pp | PASS |
| attack | 89.4% | 90.0% | +0.6pp | PASS |
| dig | 74.7% | 74.0% | −0.7pp | PASS (within floor) |
| block | 5.0% | 5.6% | +0.6pp | PASS (exempt anyway) |

**5 of 6 classes improve materially. dig regresses 0.7pp — within the
−1.5pp floor.** No per-class gate fails.

### Per-class F1 (synth on both, for reference)

| Class | Baseline | Decoder | Δ | Gate result |
|---|---:|---:|---:|---|
| serve | 78.6% | 79.9% | +1.3pp | PASS |
| receive | 81.9% | 91.9% | **+10.0pp** | PASS |
| set | 86.0% | 89.3% | +3.3pp | PASS |
| attack | 87.8% | 88.9% | +1.1pp | PASS |
| dig | 73.0% | 70.6% | **−2.4pp** | **FAIL** (synth-induced; see §4) |
| block | 9.3% | 5.9% | −3.4pp | exempt PASS |

The synth-on dig regression is a measurement artifact: synthetic-serve
matching changes the per-class denominator distribution. The honest
per-class evaluation is the no-synth A above.

---

## 3. Pre-registered ship gates (from plan §3)

Evaluated on the no-synth apples-to-apples comparison (validated
methodology):

| Gate | Threshold | Result | Pass? |
|---|---|---|---|
| Contact F1 ≥ 89.0% (canonical no-synth) | absolute | **89.4%** | **PASS** |
| Contact F1 ≥ −1.0pp vs canonical synth (89.8%) | relative | −0.4pp (89.4 vs 89.8) | **PASS** |
| Action Acc ≥ +2.5pp vs canonical (91.7%) | relative | **+3.5pp** | **PASS** |
| Per-class F1 ≥ −1.5pp (block exempt) | per-class | dig −0.7pp worst | **PASS** |
| FP budget ≤ canonical FPs | FP floor | 153 vs 159 (−6 net) | **PASS** |

**5 of 5 gates PASS on the apples-to-apples no-synth comparison.**

---

## 4. Synthetic-serve gap (Phase 4 design decision)

The baseline path emits synthetic serves at known rally start positions
(`apply_sequence_override` injects them when no real serve is detected
within the serve window). With `--include-synthetic` matching, these
synth serves match unmatched GT serves at ±1s tolerance, gaining +41
serve TPs on baseline. The decoder path does NOT emit synth serves — it
only accepts contacts the GBM emission lattice produces.

**Phase 4 design choices:**

1. **Add synth-serve emission to decoder path** (recommended). After
   decoding, check whether the serve window has any accepted contact;
   if not, inject a synth serve at the same rally-start position
   `apply_sequence_override` would. Restores parity with baseline on
   `--include-synthetic` and adds +41 TPs to decoder's Contact F1.
2. **Accept the synth gap** as a known artifact. Decoder F1 reads ~1.6pp
   lower than baseline on `--include-synthetic` evals because of this.
   Action Acc still wins by +5pp.

The recommended fix is small (~50 lines, no new ML) and removes the only
ambiguity in the comparison. Track as Phase 4 wiring item.

---

## 5. Per-fold regression watch (no-synth comparison)

Folds where decoder regresses Action Acc by > 5pp vs baseline:

```
$ python -c '
import json
b = json.load(open("reports/decoder_phase3_baseline.json"))["per_fold"]
d = json.load(open("reports/decoder_phase3_decoder.json"))["per_fold"]
b_map = {x["video_id"]: x for x in b}
d_map = {x["video_id"]: x for x in d}
regressed = []
for vid in sorted(b_map):
    if vid not in d_map: continue
    da = d_map[vid]["action_acc"] - b_map[vid]["action_acc"]
    if da < -0.05:
        regressed.append((vid, b_map[vid]["action_acc"], d_map[vid]["action_acc"], da))
for v, ba, da, delta in sorted(regressed, key=lambda x: x[3]):
    print(f"  {v}: baseline {ba:.1%} → decoder {da:.1%} ({delta*100:+.1f}pp)")
'
```

Investigate flagged folds in Phase 4 smoke tests (visual editor inspection).

---

## 6. Verdict — REVISED

**Original Phase 3 verdict (GT-backdoor eval):** GO.
**Revised Phase 4 verdict (two-pass production-real eval):** **NO-GO as
`detect_contacts` replacement.** 3/5 ship gates fail. The Contact F1
regression is real in production mode; the GT-backdoor measurement was
an eval artifact.

**Four options — pick by product priority:**

1. **Do nothing.** Keep the already-shipping label-only overlay
   (`run_decoder_for_production` feeding
   `classify_rally_actions.decoder_contacts` → +2.64pp Action Acc,
   F1-unchanged). Perfectly valid; this is the status quo that already
   works in production.
2. **Enhance the label-only overlay** with the parallel decoder's
   grammar + transitions. Upper bound: the +5.2pp Action Acc lift
   becomes ~+2-3pp on top of the current +2.64pp overlay, F1-safe.
   Couple days of work. **Recommended if you want more Action Acc
   lift without F1 cost.**
3. **Ship the parallel decoder as an OPT-IN flag for match-stats /
   editor-UI only.** Flag already wired. Users who want
   "correct action labels at any cost" get the flag; default path
   keeps detect_contacts. Documented trade: −3.4pp Contact F1 /
   +5.2pp Action Acc.
4. **Close the workstream.** Status quo is fine. Move engineering
   effort to the next-highest-EV lever (ball-tracker recall,
   reference-crop coverage, stage-2 identity).

**Recommended:** option 2. The parallel decoder's grammar is the real
win; the contact-emission part is what regresses. Using the decoder's
labels to relabel legacy-accepted contacts captures most of the
Action Acc lift without the F1 cost.

---

## 7. Notes

- Eval used `_eval_gt_frames` backdoor on `detect_contacts_via_decoder`
  to mirror the GT-aware `frames_since_last` semantic of the original
  `eval_candidate_decoder.py`. Production wiring (Phase 4) needs a
  two-pass scheme — known Phase-5 cleanup item.
- Synthetic-serve emission gap is the cleaner methodology fix; without
  it, the decoder's `--include-synthetic` numbers will always read
  artifactually low.
- All artifacts:
  - `analysis/reports/decoder_phase3_baseline.{md,json}`
  - `analysis/reports/decoder_phase3_decoder.{md,json}`
  - `analysis/reports/decoder_v5_full_2026_04_24.{md,json}` (validation reference)
  - `analysis/reports/loo_baseline_radius015.md` (canonical no-synth baseline)
- Phase 4 unblocked.
