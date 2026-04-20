# Serve Candidate Generator — Follow-up Plan

**Date**: 2026-04-20  
**Status**: planned, next session  
**Expected cost**: ~1 day  
**Expected lift**: +2-3pp aggregate Contact F1 (on top of shipped decoder baseline 88.2%)

## Motivation

Oracle diagnosis (`reports/oracle_candidate_coverage_2026_04_20.md`) proved that
**74 of 95 non-rescuable GBM-miss FNs are serves** where the candidate generator
never fired at all. These cannot be helped by downstream architecture changes
(decoder, action classifier) — there is no candidate for them to consider.

The existing synthetic-serve fix (`_make_synthetic_serve` in `action_classifier.py`)
is post-hoc action infill — it requires a subsequent detected contact as an anchor.
In the 74-FN set the pipeline detects nothing, so synthetic has no anchor and
doesn't fire.

## Failure mode (from `reports/no_candidate_fn_diagnosis_2026_04_20.md`)

| Category | Count | % of 95 | Serve subset |
|---|---|---|---|
| BALL_DROPOUT (ball sparse/occluded around GT) | 50 | 53% | 39/50 |
| BALL_PRESENT_NO_EVENT (tracked but no generator fires) | 31 | 33% | 23/31 |
| EDGE (total data gap) | 14 | 15% | 12/14 |

**Non-serve no-candidate FNs: only 21. Serves dominate (78%).**

## Design

A ninth candidate generator in `rallycut/tracking/contact_detector.py`, fired at
detection time alongside the existing 8 generators:

```
serve_candidate_generator(ball_positions, rally_frame_count, fps):
    # Only fire once per rally, in the serve window
    serve_window_end = min(rally_frame_count, int(fps * 2.0))  # first 2 seconds
    
    # Find the first stable ball track in the serve window
    for f in range(serve_window_end):
        if _is_stable_ball_track(f, ball_positions, min_len=3, min_conf=0.3):
            yield candidate(frame=f, is_serve_candidate=True)
            return
    
    # Fallback: fire at rally_start + 15 if ball never stabilizes
    yield candidate(frame=15, is_serve_candidate=True, confidence_override=0.4)
```

### Key decisions

- **Fires at most once per rally**: a rally has exactly one serve; a generator that
  fires repeatedly adds noise.
- **"Stable ball" = 3+ consecutive frames with conf ≥ 0.3**: filters out single-frame
  WASB false positives.
- **Serve window = first 2 seconds at fps**: matches observed toss duration.
- **Fallback placement**: if the ball never stabilizes in the window, emit a candidate at
  rally-start + 15 anyway. Lets the GBM/decoder decide if it's real.
- **`is_serve_candidate` flag on `CandidateFeatures`**: new boolean feature the GBM
  can use to discriminate serve-specific signal.

### Integration with existing synthetic-serve logic

The serve candidate and the synthetic serve are **complementary**, not redundant:
- Serve candidate gives the GBM something to score (classifier-level fix).
- Synthetic serve remains as the fallback when even the serve candidate is rejected.
- Both can ship; no conflict.

## Risks

1. **FPs on rallies with no real serve** (rallies that are mis-segmented, non-serve
   rallies in training data). Mitigation: emit only in the serve window, and the GBM
   will learn the `is_serve_candidate` feature to discriminate.
2. **Double-counting with synthetic serve**: already accepted via distinct-attribution
   logic in existing code; new candidate gets normal GBM scoring.
3. **Breaking baseline**: the new generator is additive. Existing candidates still fire.
   Net change: new candidates in the serve window that the GBM evaluates.

## Expected impact (estimated)

- **Upper bound**: 74 serve FNs recovered → +3.5pp aggregate F1.
- **Realistic**: 50 serves recovered at some FP cost → +2.0-2.5pp aggregate F1.
- **Serve F1 specifically**: 78% → ~88% (closer to other action classes).

## Implementation steps

1. Add `is_serve_candidate: bool` to `CandidateFeatures` dataclass (contact_classifier.py).
2. Implement `_find_serve_candidates()` in `contact_detector.py`.
3. Plumb the new generator through `detect_contacts()` merge order (priority: HIGH
   for the serve window, equivalent to direction-change priority).
4. Retrain contact classifier with the new feature present (feature version bump).
5. Re-eval with LOO-per-video; measure lift on serve F1 + aggregate.
6. Ship alongside the decoder.

## Eval harness

Reuse `scripts/eval_loo_video.py` and `scripts/eval_candidate_decoder.py`. No new eval
infrastructure needed. Expected wall-clock: 70 min per full run.

## Gate

- **Ship**: if aggregate F1 ≥ 89.5% and no per-class regression (serve/receive/set/attack/dig).
- **Iterate**: if F1 ≥ 88.5% but <89.5%, tune the stable-ball threshold and serve-window width.
- **Stop**: if F1 doesn't beat 88.2% (current decoder baseline) → the generator isn't
  adding value; investigate serve-specific recovery differently.
