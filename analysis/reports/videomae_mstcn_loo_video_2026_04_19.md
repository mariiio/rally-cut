# Phase 3 — VideoMAE MSTCN head (LOO-per-video)

- Model: MSTCN (2 stages × 8 layers, hidden=64) on 768-dim VideoMAE @ stride=4
- σ=3.0, tolerance=±233ms, NMS min_dist=3 windows
- Epochs/fold: 10, LR=0.0003, EMA=0.999
- Folds: 3

## Aggregate binary F1 (all classes collapsed)

| Thr | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|
| 0.20 | 85 | 480 | 21 | 15.0% | 80.2% | 25.3% ← best |
| 0.30 | 35 | 150 | 71 | 18.9% | 33.0% | 24.1% |
| 0.40 | 19 | 79 | 87 | 19.4% | 17.9% | 18.6% |
| 0.50 | 4 | 22 | 102 | 15.4% | 3.8% | 6.1% |
| 0.60 | 0 | 0 | 106 | 0.0% | 0.0% | 0.0% |
| 0.70 | 0 | 0 | 106 | 0.0% | 0.0% | 0.0% |

## Per-class F1 @ best threshold 0.20

| Class | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|
| serve | 0 | 0 | 22 | 0.0% | 0.0% | 0.0% |
| receive | 5 | 134 | 15 | 3.6% | 25.0% | 6.3% |
| set | 0 | 1 | 21 | 0.0% | 0.0% | 0.0% |
| attack | 3 | 67 | 25 | 4.3% | 10.7% | 6.1% |
| dig | 7 | 205 | 8 | 3.3% | 46.7% | 6.2% |
| block | 0 | 143 | 0 | 0.0% | 0.0% | 0.0% |

## Gate: FAIL

- ≥82% binary F1 → Phase 4/5 (PASS)
- 65–82% → flagged (FLAG)
- <65% → STOP (FAIL)