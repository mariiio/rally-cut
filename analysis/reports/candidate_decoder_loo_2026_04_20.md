# Phase CRF-1 — Viterbi Candidate Decoder Eval (LOO-per-video)

- Folds: 68
- Skip penalty: 0.0
- Min accept prob: 0.0
- Transition matrix: reports/transition_matrix_2026_04_20.json
- Wall-clock: 69.0 min

## Aggregate

| Metric | Decoder | Phase 0 baseline | Δ |
|---|---|---|---|
| Contact F1 | **87.7%** | 88.0% | **-0.3%** |
| Action accuracy | **96.2%** | 91.2% | **+5.0%** |
| TP / FP / FN | 1690 / 68 / 405 | 1781/174/314 | |

## Verdict: NO LIFT

## Per-class F1

| Class | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|
| serve | 244 | 16 | 120 | 93.8% | 67.0% | 78.2% |
| receive | 276 | 20 | 55 | 93.2% | 83.4% | 88.0% |
| set | 408 | 39 | 63 | 91.3% | 86.6% | 88.9% |
| attack | 474 | 16 | 96 | 96.7% | 83.2% | 89.4% |
| dig | 223 | 41 | 106 | 84.5% | 67.8% | 75.2% |
| block | 0 | 1 | 30 | 0.0% | 0.0% | 0.0% |