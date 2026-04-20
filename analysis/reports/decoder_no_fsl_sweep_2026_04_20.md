# No-FSL Emission + Decoder Sweep (LOO-per-video)

Feature masked: `frames_since_last` (index 16)

## Reference lines

- Phase 0 GBM baseline (with FSL, no decoder): F1=88.0% P=91.1% R=85.0% acc=91.2%
- Previous best decoder (with FSL, sp1.0): F1=88.2% P=90.4% R=86.1% acc=95.5%
- No-FSL GBM standalone (no decoder): F1=76.5% P=68.7% R=86.3%

## No-FSL + decoder sweep

| Config | F1 | ΔF1 | P | R | Action Acc | Δacc | TP | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| no-FSL + sp0.0 | 77.9% | -10.1% | 97.1% | 65.1% | 96.6% | +5.4% | 1364 | 41 | 731 |
| no-FSL + sp0.5 | 83.9% | -4.1% | 94.4% | 75.5% | 96.3% | +5.1% | 1581 | 94 | 514 |
| no-FSL + sp1.0 | 85.8% | -2.2% | 89.2% | 82.7% | 95.8% | +4.6% | 1732 | 209 | 363 ← best |
| no-FSL + sp1.5 | 85.2% | -2.8% | 83.7% | 86.7% | 95.4% | +4.2% | 1817 | 354 | 278 |
| no-FSL + sp2.0 | 82.5% | -5.5% | 77.2% | 88.6% | 95.2% | +4.0% | 1857 | 549 | 238 |

## Per-class F1 (best: no-FSL + sp1.0)

| Class | TP | FP | FN | F1 |
|---|---|---|---|---|
| serve | 246 | 36 | 118 | 76.2% |
| receive | 290 | 55 | 41 | 85.8% |
| set | 414 | 70 | 57 | 86.7% |
| attack | 467 | 16 | 103 | 88.7% |
| dig | 241 | 91 | 88 | 72.9% |
| block | 1 | 14 | 29 | 4.4% |