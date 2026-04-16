# Session 6 — Learned-head merge-veto gate report

Integration: `LEARNED_MERGE_VETO_COS` in `tracklet_link.link_tracklets_by_appearance` — for each candidate merge pair, block the merge when `cos(median_learned_emb_a, median_learned_emb_b) < threshold`. Abstains when < 5 embeddings per side.

## Gate summary

| Threshold | SAME_TEAM_SWAP | Δ vs baseline | Worst rally HOTA drop | Fragmentation-exceeded rallies | Gate |
|----------:|--------------:|:-------------:|:---------------------:|:------------------------------:|:----:|
| **0.00** (baseline) | 12 | — | — | — | ctrl |
| 0.80 | 9 | +25.0% | 0.00 pp (on 0 rally/rallies) | 2 | ❌ |
| 0.85 | 11 | +8.3% | 0.00 pp (on 0 rally/rallies) | 2 | ❌ |
| 0.88 | 9 | +25.0% | 0.00 pp (on 0 rally/rallies) | 7 | ❌ |
| 0.90 | 11 | +8.3% | 0.00 pp (on 0 rally/rallies) | 6 | ❌ |
| 0.92 | 11 | +8.3% | 0.00 pp (on 0 rally/rallies) | 6 | ❌ |
| 0.95 | 11 | +8.3% | 0.00 pp (on 0 rally/rallies) | 7 | ❌ |

## Per-cell detail

### threshold = 0.80

- SAME_TEAM_SWAP: **9** (baseline 12, Δ -3)
- NET_CROSSING: 0
- Total real switches: 9
- Retrack elapsed: 3982.5 s
- HOTA regressions > 0.5 pp: **none**
- Fragmentation exceedances > 20 %:
  - `209be896` 4 → 5 unique pred-IDs (+25 %)
  - `740ffd88` 4 → 5 unique pred-IDs (+25 %)

### threshold = 0.85

- SAME_TEAM_SWAP: **11** (baseline 12, Δ -1)
- NET_CROSSING: 0
- Total real switches: 11
- Retrack elapsed: 9.2 s
- HOTA regressions > 0.5 pp: **none**
- Fragmentation exceedances > 20 %:
  - `209be896` 4 → 6 unique pred-IDs (+50 %)
  - `c48eeb7d` 4 → 5 unique pred-IDs (+25 %)

### threshold = 0.88

- SAME_TEAM_SWAP: **9** (baseline 12, Δ -3)
- NET_CROSSING: 0
- Total real switches: 9
- Retrack elapsed: 8.2 s
- HOTA regressions > 0.5 pp: **none**
- Fragmentation exceedances > 20 %:
  - `209be896` 4 → 6 unique pred-IDs (+50 %)
  - `e5c1a9b3` 4 → 6 unique pred-IDs (+50 %)
  - `21029e9f` 4 → 5 unique pred-IDs (+25 %)
  - `53ca3586` 4 → 5 unique pred-IDs (+25 %)
  - `572bff7e` 4 → 5 unique pred-IDs (+25 %)
  - `740ffd88` 4 → 5 unique pred-IDs (+25 %)
  - `c48eeb7d` 4 → 5 unique pred-IDs (+25 %)

### threshold = 0.90

- SAME_TEAM_SWAP: **11** (baseline 12, Δ -1)
- NET_CROSSING: 0
- Total real switches: 11
- Retrack elapsed: 8.2 s
- HOTA regressions > 0.5 pp: **none**
- Fragmentation exceedances > 20 %:
  - `209be896` 4 → 6 unique pred-IDs (+50 %)
  - `e5c1a9b3` 4 → 6 unique pred-IDs (+50 %)
  - `21029e9f` 4 → 5 unique pred-IDs (+25 %)
  - `572bff7e` 4 → 5 unique pred-IDs (+25 %)
  - `740ffd88` 4 → 5 unique pred-IDs (+25 %)
  - `c48eeb7d` 4 → 5 unique pred-IDs (+25 %)

### threshold = 0.92

- SAME_TEAM_SWAP: **11** (baseline 12, Δ -1)
- NET_CROSSING: 0
- Total real switches: 11
- Retrack elapsed: 8.1 s
- HOTA regressions > 0.5 pp: **none**
- Fragmentation exceedances > 20 %:
  - `209be896` 4 → 6 unique pred-IDs (+50 %)
  - `e5c1a9b3` 4 → 6 unique pred-IDs (+50 %)
  - `21029e9f` 4 → 5 unique pred-IDs (+25 %)
  - `572bff7e` 4 → 5 unique pred-IDs (+25 %)
  - `740ffd88` 4 → 5 unique pred-IDs (+25 %)
  - `c48eeb7d` 4 → 5 unique pred-IDs (+25 %)

### threshold = 0.95

- SAME_TEAM_SWAP: **11** (baseline 12, Δ -1)
- NET_CROSSING: 0
- Total real switches: 11
- Retrack elapsed: 8.2 s
- HOTA regressions > 0.5 pp: **none**
- Fragmentation exceedances > 20 %:
  - `209be896` 4 → 6 unique pred-IDs (+50 %)
  - `e5c1a9b3` 4 → 6 unique pred-IDs (+50 %)
  - `21029e9f` 4 → 5 unique pred-IDs (+25 %)
  - `21266995` 4 → 5 unique pred-IDs (+25 %)
  - `21a9b203` 4 → 5 unique pred-IDs (+25 %)
  - `572bff7e` 4 → 5 unique pred-IDs (+25 %)
  - `740ffd88` 4 → 5 unique pred-IDs (+25 %)


## Recommendation

**NO SHIP.** No threshold clears all measurable gate targets simultaneously. Review per-cell detail above — most likely either (a) swap reduction floor too tight for the head's signal, (b) HOTA regression on specific rallies indicates legit merges being rejected, or (c) fragmentation explosion on high thresholds. Iterate on threshold range, or combine with `ENABLE_COURT_VELOCITY_GATE=1` in a 2D sweep.

## Runtime

- t=0.00: 0.1 min
- t=0.80: 66.4 min
- t=0.85: 0.2 min
- t=0.88: 0.1 min
- t=0.90: 0.1 min
- t=0.92: 0.1 min
- t=0.95: 0.1 min
