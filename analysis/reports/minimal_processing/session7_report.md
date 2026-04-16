# Session 7 — Minimal-processing A/B

Tests whether the multi-stage post-YOLO merge/rename chain is net-positive or net-negative for identity accuracy. Two cells compared on 43 GT rallies.

## Headline

| Metric | Baseline (all passes on) | Minimal (`SKIP_ALL_MERGE_PASSES=1`) | Δ |
|---|---:|---:|---:|
| Aggregate HOTA | 91.63% | 86.44% | -5.19 pp |
| Aggregate F1 | 94.43% | 91.98% | -2.45 pp |
| Aggregate ID switches | 111 | 143 | +32 |
| SAME_TEAM_SWAP | 12 | 2 | -10 |
| NET_CROSSING | 0 | 0 | +0 |
| Total real switches | 12 | 2 | -10 |
| Total unique pred-IDs (fragmentation) | 176 | 239 | +63 |

## Verdict: **KEEP**

Aggregate HOTA dropped 5.19 pp under minimal processing. The merge passes ARE doing real work. Identity correction needs to happen INSIDE them (e.g. extend Session-6 learned-head veto to all merge passes).

## Per-rally HOTA breakdown

- Rallies UNCHANGED (|Δ| ≤ 0.5 pp): **7**
- Rallies WORSE under minimal processing: **32**
- Rallies BETTER under minimal processing: **4**

### Rallies where merge passes help (HOTA drops by > 0.5 pp under minimal)

- `740ffd88`: 99.3% → 79.2% (−20.06 pp)
- `e5c1a9b3`: 89.2% → 71.4% (−17.78 pp)
- `72c8229b`: 95.7% → 82.2% (−13.46 pp)
- `793625cd`: 95.0% → 81.7% (−13.37 pp)
- `c48eeb7d`: 98.2% → 84.9% (−13.29 pp)
- `87ce7bff`: 87.1% → 74.6% (−12.46 pp)
- `0a376585`: 88.5% → 76.7% (−11.87 pp)
- `21266995`: 94.3% → 84.6% (−9.72 pp)
- `8b0b9e13`: 97.7% → 88.2% (−9.50 pp)
- `b7f92cdc`: 86.7% → 77.3% (−9.37 pp)
- ... and 22 more

### Rallies where merge passes hurt (HOTA rises by > 0.5 pp under minimal)

- `8c2d30ce`: 95.0% → 97.4% (+2.41 pp)
- `fad29c31`: 71.4% → 72.8% (+1.42 pp)
- `2dff5eeb`: 79.1% → 80.4% (+1.31 pp)
- `9db9cb6b`: 90.9% → 91.9% (+0.92 pp)

## Runtime

- Baseline: 7.4 s
- Minimal: 7.1 s
