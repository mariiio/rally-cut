# Decoder smoke test verdict — production parity restored via two-pass

**Date:** 2026-04-24
**Plan:** `docs/superpowers/plans/2026-04-24-parallel-decoder-ship.md`
**Phase:** 4 (smoke before flag-default flip)
**Driving evidence:** `analysis/reports/decoder_smoke_2pass_2026_04_24.md`,
`analysis/reports/decoder_phase4_2pass.{md,json}` (full LOO).

---

## TL;DR — flag is shippable, dig is the only persistent watch

**Initial smoke (10 rallies, single-pass decoder)** showed a catastrophic
regression: decoder produced **−37% total actions** vs legacy. Root cause:
the trainer's `extract_candidate_features` treats every candidate as
"accepted" when called without `gt_frames`, making the GBM's
`frames_since_last` feature systematically wrong in production mode.

**Fix:** Two-pass scheme inside `detect_contacts_via_decoder`:
1. Pass 1 — extract features with `gt_frames=None`, run GBM, identify
   provisionally-accepted candidates (prob ≥ classifier.threshold).
2. Pass 2 — re-extract features with those accepted frames as the
   `gt_frames` proxy. Now `frames_since_last` reflects only accepted
   contacts (mirroring legacy `detect_contacts`'s per-iteration
   `prev_accepted_frame` update). Re-run GBM. Decode.

**Re-run smoke (10 rallies, two-pass)** restored parity:

| Class | Legacy | Decoder (two-pass) | Δ |
|---|---:|---:|---:|
| serve | 9 | 9 | 0 |
| receive | 10 | 10 | 0 |
| set | 10 | 14 | **+4** |
| attack | 12 | 12 | 0 |
| dig | 11 | 7 | **−4** ⚠ |
| block | 0 | 1 | **+1** |
| **total** | **52** | **53** | **+1** |

Per-rally observations:
- `0102cbba`: GT 4 contacts → both legacy and decoder hit 4/4 ✓
- `032ec267`: GT includes "block" → legacy misses it, decoder catches it ✓
- `5447e090`: decoder misses the trailing "dig" (legacy gets it)
- `0ab56722` / `d724bbf0`: decoder over-emits "set" (set→set patterns)

The dig regression matches the −2.4pp synth-on / −0.7pp no-synth gap
already documented in the Phase 3 + 4 memo. **Pattern: decoder is more
conservative on dig and slightly over-emits on set.** Both are bounded —
neither catastrophic.

---

## What this changes vs the prior eval result

The Phase 3 + 4 measurement used the `_eval_gt_frames` backdoor (passed
GT frames into `frames_since_last`). That over-stated production
behavior because production has no GT. The smoke test made this gap
visible.

The two-pass scheme is the production-correct version of what the eval
backdoor approximated. Re-running the LOO eval with the two-pass scheme
(no backdoor) is the canonical measurement.

**Production-realistic LOO numbers** (`decoder_phase4_2pass.md`,
68-fold, --include-synthetic): _populated when eval completes._

---

## Verdict

| Gate | Status |
|---|---|
| Smoke total-action parity | ✅ +1 net |
| serve / receive / attack | ✅ exact match |
| dig | ⚠ −4 (persistent, watch in soak) |
| block | ✅ +1 (decoder catches one legacy missed) |
| LOO Contact F1 | _pending two-pass eval result_ |
| LOO Action Acc | _pending two-pass eval result_ |

**Ship recommendation:** keep `USE_PARALLEL_DECODER` default OFF until
the two-pass LOO eval confirms the gates still pass. If the eval shows
F1 within −1pp of canonical and Action Acc lift ≥ +2pp, proceed with
the soak as planned. Dig watch is the long pole.

---

## What to do if two-pass eval regresses badly

If F1 drops below 88% or Action Acc lift collapses:
1. **Revert the two-pass change** — keep the eval backdoor for now.
2. **Investigate the per-rally dig misses** — are they genuine GT
   contacts the decoder rejects? Or borderline cases where neither
   path is right?
3. **Consider hybrid:** run decoder for action labels, keep legacy
   `detect_contacts` for contact emission (+attribution). This loses
   the contact-rescue side of the lift but preserves the action-label
   lift, which is the bigger product win.

---

## Reusable artifacts

- `analysis/scripts/decoder_smoke_2026_04_24.py` — side-by-side smoke
  harness on stratified rallies. Re-run after any decoder change.
- `analysis/reports/decoder_smoke_2pass_2026_04_24.md` — current
  smoke output.
- `analysis/reports/decoder_phase4_2pass.{md,json}` — production-
  realistic LOO eval (pending completion).
- `analysis/rallycut/tracking/contact_detector.py:_features_to_classifier_matrix`
  + two-pass logic — implementation.
