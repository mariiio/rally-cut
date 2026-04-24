# Phase CRF-1 — Viterbi Candidate Decoder Eval (LOO-per-video)

- Folds: 68
- Skip penalty: 1.0
- Min accept prob: 0.0
- Transition matrix: None
- Wall-clock: 18.8 min

## Aggregate

| Metric | Decoder | Phase 0 baseline | Δ |
|---|---|---|---|
| Contact F1 | **89.4%** | 88.0% | **+1.4%** |
| Action accuracy | **95.2%** | 91.2% | **+4.0%** |
| TP / FP / FN | 1900 / 153 / 297 | 1781/174/314 | |

## Verdict: WEAK

## Per-class F1

| Class | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|
| serve | 274 | 32 | 103 | 89.5% | 72.7% | 80.2% |
| receive | 320 | 37 | 23 | 89.6% | 93.3% | 91.4% |
| set | 444 | 59 | 50 | 88.3% | 89.9% | 89.1% |
| attack | 514 | 29 | 85 | 94.7% | 85.8% | 90.0% |
| dig | 256 | 85 | 95 | 75.1% | 72.9% | 74.0% |
| block | 1 | 2 | 32 | 33.3% | 3.0% | 5.6% |