# Phase 1.2 — Side-Switch Reliability Audit

**Date:** 2026-04-24  
**Scope:** all rallies across 9 fixtures (not just GT-locked ones — side-switch state depends on full per-match history).  
**Gate:** `agreement_rate ≥ 98%`  
**Result:** `100.00%` — **PASS**

## Method

- Extract `sideSwitchDetected` boolean from `videos.match_analysis_json.rallies[*]` per rally, in time order.
- Compute positional near/far pair per rally from rally-start positions (first 15 frames, median y-coord grouping).
- **Disagreement types counted:**
  - `switch_flag_but_same_side` — pipeline claims a switch but positional near-pair is unchanged from previous rally.
  - `position_flip_no_flag` — pipeline says no switch, but the positional near-pair fully flipped.

## Per-fixture

| fixture | rallies | flagged | disagreements |
|---|---|---|---|
| cece | 5 | 0 | 0 |
| cuco | 7 | 1 | 0 |
| lala | 21 | 2 | 0 |
| lulu | 22 | 4 | 0 |
| rere | 7 | 1 | 0 |
| tata | 20 | 3 | 0 |
| toto | 30 | 4 | 0 |
| wawa | 6 | 0 | 0 |
| yeye | 7 | 0 | 0 |
| **TOTAL** | **125** | **15** | **0** (100.00% agreement) |

## Disagreements (user audit needed)

*None — pipeline side-switch state agrees with positional evidence.*

## Decision

**Pass (100.00%).** Pipeline's `sideSwitchDetected` boolean is internally consistent with observed team positions. Phase 1.3 (team→side audit) can lean on side-switch state as a trusted primitive input.