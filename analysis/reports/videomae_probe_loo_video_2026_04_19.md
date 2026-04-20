# Phase 2 — VideoMAE linear probe (LOO-per-video)

- Model: LogisticRegression (class-balanced)
- Stride: 4, Backbone: videomae-v1
- Label radius: ±3 effective-fps frames
- Match tolerance: ±233 ms
- Peak-NMS min distance: 3 windows
- Total windows: 178,722 (3,646 positive)

## Aggregate (all folds, best threshold)

| Metric | Value |
|---|---|
| Threshold | 0.30 |
| Contact F1 | **36.9%** |
| Precision | 27.8% |
| Recall | 54.7% |
| TP / FP / FN | 1146 / 2978 / 949 |
| Gate (≥60% → Phase 3) | **FAIL** |

## Threshold sweep

| Thr | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|
| 0.30 | 1146 | 2978 | 949 | 27.8% | 54.7% | 36.9% |
| 0.40 | 1057 | 2631 | 1038 | 28.7% | 50.5% | 36.6% |
| 0.50 | 987 | 2288 | 1108 | 30.1% | 47.1% | 36.8% |
| 0.60 | 899 | 1980 | 1196 | 31.2% | 42.9% | 36.1% |
| 0.70 | 809 | 1616 | 1286 | 33.4% | 38.6% | 35.8% |
| 0.80 | 675 | 1211 | 1420 | 35.8% | 32.2% | 33.9% |
| 0.90 | 476 | 799 | 1619 | 37.3% | 22.7% | 28.2% |
