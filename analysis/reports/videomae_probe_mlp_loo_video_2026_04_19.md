# Phase 2 — VideoMAE linear probe (LOO-per-video)

- Model: MLP 768→64→2
- Stride: 4, Backbone: videomae-v1
- Label radius: ±3 effective-fps frames
- Match tolerance: ±233 ms
- Peak-NMS min distance: 3 windows
- Total windows: 178,722 (3,646 positive)

## Aggregate (all folds, best threshold)

| Metric | Value |
|---|---|
| Threshold | 0.60 |
| Contact F1 | **33.3%** |
| Precision | 28.6% |
| Recall | 40.0% |
| TP / FP / FN | 837 / 2089 / 1258 |
| Gate (≥60% → Phase 3) | **FAIL** |

## Threshold sweep

| Thr | TP | FP | FN | P | R | F1 |
|---|---|---|---|---|---|---|
| 0.30 | 1087 | 3938 | 1008 | 21.6% | 51.9% | 30.5% |
| 0.40 | 1035 | 3337 | 1060 | 23.7% | 49.4% | 32.0% |
| 0.50 | 949 | 2718 | 1146 | 25.9% | 45.3% | 32.9% |
| 0.60 | 837 | 2089 | 1258 | 28.6% | 40.0% | 33.3% |
| 0.70 | 661 | 1438 | 1434 | 31.5% | 31.6% | 31.5% |
| 0.80 | 420 | 792 | 1675 | 34.7% | 20.0% | 25.4% |
| 0.90 | 176 | 245 | 1919 | 41.8% | 8.4% | 14.0% |
