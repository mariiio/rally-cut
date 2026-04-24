# Phase CRF-1 — Viterbi Candidate Decoder Eval (LOO-per-video)

- Folds: 5
- Skip penalty: 1.0
- Min accept prob: 0.0
- Transition matrix: None
- Wall-clock: 2.1 min

## Aggregate

| Metric | Decoder | Phase 0 baseline | Δ |
|---|---|---|---|
| Contact F1 | **86.8%** | 88.0% | **-1.2%** |
| Action accuracy | **97.4%** | 91.2% | **+6.2%** |
| TP / FP / FN | 151 / 17 / 29 | 1781/174/314 | |

## Verdict: REGRESSION

## Per-class F1

| Class | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|
| serve | 26 | 2 | 8 | 92.9% | 76.5% | 83.9% |
| receive | 29 | 3 | 3 | 90.6% | 90.6% | 90.6% |
| set | 36 | 4 | 2 | 90.0% | 94.7% | 92.3% |
| attack | 42 | 3 | 6 | 93.3% | 87.5% | 90.3% |
| dig | 14 | 9 | 11 | 60.9% | 56.0% | 58.3% |
| block | 0 | 0 | 3 | 0.0% | 0.0% | 0.0% |