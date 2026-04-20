# Oracle Candidate-Coverage Test

- GT contacts: 2095
- GBM hits (baseline 88.0% F1 reproducing): 1781 (85.0%)
- GBM misses: 314

## The decisive number

**Of 314 GBM-MISS contacts, 219 have a generator candidate within ±7 frames.**
= 69.7% of misses are recoverable by a perfect downstream filter.

## Interpretation
⚠️ **Partial signal.** 219 candidates exist for rescue, but 95 misses have no candidate nearby — those are generator failures.
Sequence decoding could help the first group. Candidate-generator improvements needed for the second.

## Sanity check (HIT coverage — should be ~100%)

Of 1781 GBM-HIT contacts, 1781 have a generator candidate within ±7f (100.0%).

## Per-action breakdown of misses

| Action | Misses | Candidate within ±7f | Coverage |
|---|---|---|---|
| attack | 32 | 26 | 81.2% |
| block | 28 | 27 | 96.4% |
| dig | 56 | 53 | 94.6% |
| receive | 42 | 39 | 92.9% |
| serve | 124 | 50 | 40.3% |
| set | 32 | 24 | 75.0% |

## Distance from GBM-MISS GT to nearest candidate (diagnostic)

- P25: 1 frames
- P50: 3 frames
- P75: 11 frames
- P90: 28 frames
- P95: 37 frames
