# Session 8 Plan B — 2D Sweep: Learned-ReID × Court-Velocity Gate

## Methodology

Task 2 diagnostic (per-pass swap attribution, commit 350cd3a) falsified the
Session-8 multi-site thesis. Attribution showed:

- **10 of 12 SAME_TEAM_SWAPs** come from a single pass: `link_tracklets_by_appearance` (step 0c).
- **2 of 12** come from raw BoT-SORT + filtering (pre-merge-chain; no merge veto can fix these).
- **5 other merge/rename passes contribute 0 swaps** — no adapter work needed there.

Plan B: stay at one merge site, stack two orthogonal vetoes (both already implemented
and env-gated; no new code required):

1. **Learned-ReID cosine veto** (`LEARNED_MERGE_VETO_COS`) — Session 6 ship
2. **Court-plane velocity gate** (`ENABLE_COURT_VELOCITY_GATE` + `RALLYCUT_MAX_MERGE_VELOCITY_METERS`)

When both env vars are on, either veto can block a merge (OR-logic). Combined cells
should satisfy `swaps ≤ min(learned_only, velocity_only)` — verified in results below.

**Baseline**: all merge passes on, no vetoes (LEARNED_MERGE_VETO_COS=0, ENABLE_COURT_VELOCITY_GATE=0).
Baseline SAME_TEAM_SWAPs: **12** (expected 12; OK).

## Ship Gate

All three must hold vs baseline on 43 GT rallies:
1. SAME_TEAM_SWAP reduction ≥ 50% (≤6 remaining)
2. No per-rally HOTA drop > 0.5 pp
3. Fragmentation delta ≤ 20%
4. `player_attribution_oracle` — **deferred to Task 9**

## 8-Cell Summary

| Cell | LEARNED | GATE | VEL_M | HOTA agg | HOTA Δ pp | Swaps | Swap Δ | Frag | Frag Δ | Gate | Reason |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|:---:|---|
| `baseline` | 0.00 | 0 | — | 91.63% | — | 12 | — | 176 | — | ctrl | baseline |
| `learned_t080` | 0.80 | 0 | — | 92.15% | +0.52 | 8 | -4 (-33%) | 176 | +0.0% | ❌ | Gate 1 FAIL: swap reduction 33.3% < 50% (8/12 remain) |
| `velocity_v25` | 0.00 | 1 | 2.5 | 91.20% | -0.44 | 11 | -1 (-8%) | 179 | +1.7% | ❌ | Gate 1 FAIL: swap reduction 8.3% < 50% (11/12 remain) |
| `velocity_v35` | 0.00 | 1 | 3.5 | 91.32% | -0.31 | 12 | +0 (-0%) | 176 | +0.0% | ❌ | Gate 1 FAIL: swap reduction 0.0% < 50% (12/12 remain) |
| `combined_070_v25` | 0.70 | 1 | 2.5 | 91.26% | -0.37 | 10 | -2 (-17%) | 180 | +2.3% | ❌ | Gate 1 FAIL: swap reduction 16.7% < 50% (10/12 remain) |
| `combined_070_v35` | 0.70 | 1 | 3.5 | 91.41% | -0.22 | 11 | -1 (-8%) | 177 | +0.6% | ❌ | Gate 1 FAIL: swap reduction 8.3% < 50% (11/12 remain) |
| `combined_080_v25` | 0.80 | 1 | 2.5 | 91.92% | +0.28 | 7 | -5 (-42%) | 179 | +1.7% | ❌ | Gate 1 FAIL: swap reduction 41.7% < 50% (7/12 remain) |
| `combined_080_v35` | 0.80 | 1 | 3.5 | 91.91% | +0.28 | 7 | -5 (-42%) | 177 | +0.6% | ❌ | Gate 1 FAIL: swap reduction 41.7% < 50% (7/12 remain) |

## Per-Cell Detail

### `learned_t080`

- Config: LEARNED_MERGE_VETO_COS=0.80, ENABLE_COURT_VELOCITY_GATE=0
- SAME_TEAM_SWAP: **8** (baseline 12, Δ -4, reduction 33.3%)
- NET_CROSSING: 0
- Total real switches: 8
- Aggregate HOTA: 92.15% (baseline 91.63%)
- Elapsed: 0.0s
- Gate: FAIL — Gate 1 FAIL: swap reduction 33.3% < 50% (8/12 remain)
- HOTA regressions > 0.5 pp (4 rallies):
  - `2dff5eeb` −4.34 pp
  - `572bff7e` −2.38 pp
  - `72c8229b` −1.83 pp
  - `87ce7bff` −0.53 pp
- Fragmentation exceedances > 20% (2 rallies):
  - `209be896` 4→5 unique pred-IDs (+25%)
  - `72c8229b` 4→5 unique pred-IDs (+25%)

### `velocity_v25`

- Config: LEARNED_MERGE_VETO_COS=0.00, ENABLE_COURT_VELOCITY_GATE=1, RALLYCUT_MAX_MERGE_VELOCITY_METERS=2.5
- SAME_TEAM_SWAP: **11** (baseline 12, Δ -1, reduction 8.3%)
- NET_CROSSING: 0
- Total real switches: 11
- Aggregate HOTA: 91.20% (baseline 91.63%)
- Elapsed: 8.5s
- Gate: FAIL — Gate 1 FAIL: swap reduction 8.3% < 50% (11/12 remain)
- HOTA regressions > 0.5 pp (7 rallies):
  - `740ffd88` −12.28 pp
  - `b7f92cdc` −7.32 pp
  - `72c8229b` −6.47 pp
  - `21266995` −5.54 pp
  - `87ce7bff` −3.83 pp
  - `572bff7e` −2.38 pp
  - `2dff5eeb` −1.54 pp
- Fragmentation exceedances > 20% (4 rallies):
  - `209be896` 4→5 unique pred-IDs (+25%)
  - `21266995` 4→5 unique pred-IDs (+25%)
  - `72c8229b` 4→5 unique pred-IDs (+25%)
  - `740ffd88` 4→5 unique pred-IDs (+25%)

### `velocity_v35`

- Config: LEARNED_MERGE_VETO_COS=0.00, ENABLE_COURT_VELOCITY_GATE=1, RALLYCUT_MAX_MERGE_VELOCITY_METERS=3.5
- SAME_TEAM_SWAP: **12** (baseline 12, Δ +0, reduction 0.0%)
- NET_CROSSING: 0
- Total real switches: 12
- Aggregate HOTA: 91.32% (baseline 91.63%)
- Elapsed: 7.8s
- Gate: FAIL — Gate 1 FAIL: swap reduction 0.0% < 50% (12/12 remain)
- HOTA regressions > 0.5 pp (6 rallies):
  - `740ffd88` −12.28 pp
  - `b7f92cdc` −7.32 pp
  - `21266995` −5.54 pp
  - `72c8229b` −4.71 pp
  - `87ce7bff` −2.40 pp
  - `572bff7e` −2.38 pp
- Fragmentation exceedances > 20% (2 rallies):
  - `21266995` 4→5 unique pred-IDs (+25%)
  - `740ffd88` 4→5 unique pred-IDs (+25%)

### `combined_070_v25`

- Config: LEARNED_MERGE_VETO_COS=0.70, ENABLE_COURT_VELOCITY_GATE=1, RALLYCUT_MAX_MERGE_VELOCITY_METERS=2.5
- SAME_TEAM_SWAP: **10** (baseline 12, Δ -2, reduction 16.7%)
- NET_CROSSING: 0
- Total real switches: 10
- Aggregate HOTA: 91.26% (baseline 91.63%)
- Elapsed: 8.0s
- Gate: FAIL — Gate 1 FAIL: swap reduction 16.7% < 50% (10/12 remain)
- HOTA regressions > 0.5 pp (8 rallies):
  - `740ffd88` −12.28 pp
  - `72c8229b` −6.47 pp
  - `b7f92cdc` −5.63 pp
  - `21266995` −5.54 pp
  - `e5c1a9b3` −3.87 pp
  - `572bff7e` −2.38 pp
  - `2dff5eeb` −1.54 pp
  - `87ce7bff` −0.53 pp
- Fragmentation exceedances > 20% (5 rallies):
  - `209be896` 4→5 unique pred-IDs (+25%)
  - `21266995` 4→5 unique pred-IDs (+25%)
  - `72c8229b` 4→5 unique pred-IDs (+25%)
  - `740ffd88` 4→5 unique pred-IDs (+25%)
  - `e5c1a9b3` 4→5 unique pred-IDs (+25%)

### `combined_070_v35`

- Config: LEARNED_MERGE_VETO_COS=0.70, ENABLE_COURT_VELOCITY_GATE=1, RALLYCUT_MAX_MERGE_VELOCITY_METERS=3.5
- SAME_TEAM_SWAP: **11** (baseline 12, Δ -1, reduction 8.3%)
- NET_CROSSING: 0
- Total real switches: 11
- Aggregate HOTA: 91.41% (baseline 91.63%)
- Elapsed: 8.0s
- Gate: FAIL — Gate 1 FAIL: swap reduction 8.3% < 50% (11/12 remain)
- HOTA regressions > 0.5 pp (6 rallies):
  - `740ffd88` −12.28 pp
  - `b7f92cdc` −5.63 pp
  - `21266995` −5.54 pp
  - `72c8229b` −4.71 pp
  - `e5c1a9b3` −3.87 pp
  - `572bff7e` −2.38 pp
- Fragmentation exceedances > 20% (3 rallies):
  - `21266995` 4→5 unique pred-IDs (+25%)
  - `740ffd88` 4→5 unique pred-IDs (+25%)
  - `e5c1a9b3` 4→5 unique pred-IDs (+25%)

### `combined_080_v25`

- Config: LEARNED_MERGE_VETO_COS=0.80, ENABLE_COURT_VELOCITY_GATE=1, RALLYCUT_MAX_MERGE_VELOCITY_METERS=2.5
- SAME_TEAM_SWAP: **7** (baseline 12, Δ -5, reduction 41.7%)
- NET_CROSSING: 0
- Total real switches: 7
- Aggregate HOTA: 91.92% (baseline 91.63%)
- Elapsed: 8.0s
- Gate: FAIL — Gate 1 FAIL: swap reduction 41.7% < 50% (7/12 remain)
- HOTA regressions > 0.5 pp (6 rallies):
  - `b7f92cdc` −5.63 pp
  - `21266995` −5.54 pp
  - `e5c1a9b3` −3.87 pp
  - `572bff7e` −2.38 pp
  - `2dff5eeb` −1.54 pp
  - `87ce7bff` −0.53 pp
- Fragmentation exceedances > 20% (4 rallies):
  - `209be896` 4→5 unique pred-IDs (+25%)
  - `21266995` 4→5 unique pred-IDs (+25%)
  - `53ca3586` 4→5 unique pred-IDs (+25%)
  - `e5c1a9b3` 4→5 unique pred-IDs (+25%)

### `combined_080_v35`

- Config: LEARNED_MERGE_VETO_COS=0.80, ENABLE_COURT_VELOCITY_GATE=1, RALLYCUT_MAX_MERGE_VELOCITY_METERS=3.5
- SAME_TEAM_SWAP: **7** (baseline 12, Δ -5, reduction 41.7%)
- NET_CROSSING: 0
- Total real switches: 7
- Aggregate HOTA: 91.91% (baseline 91.63%)
- Elapsed: 8.0s
- Gate: FAIL — Gate 1 FAIL: swap reduction 41.7% < 50% (7/12 remain)
- HOTA regressions > 0.5 pp (4 rallies):
  - `b7f92cdc` −5.63 pp
  - `21266995` −5.54 pp
  - `e5c1a9b3` −3.87 pp
  - `572bff7e` −2.38 pp
- Fragmentation exceedances > 20% (4 rallies):
  - `209be896` 4→5 unique pred-IDs (+25%)
  - `21266995` 4→5 unique pred-IDs (+25%)
  - `53ca3586` 4→5 unique pred-IDs (+25%)
  - `e5c1a9b3` 4→5 unique pred-IDs (+25%)

## OR-Logic Sanity Check

- combined_080_v25 ≤ min(learned_t080, velocity_v25): combo=7, learned=8, vel=11 → min=8 [OK]
- combined_080_v35 ≤ min(learned_t080, velocity_v35): combo=7, learned=8, vel=12 → min=8 [OK]

## Knee Recommendation

**NO SHIP.** No cell clears all measurable gate targets simultaneously.

Best-case reduction: `combined_080_v25` with 7 swaps (42% reduction) — did not clear all gates.

Next-step options:
- **3a**: Continue looser learned-only sweep (e.g. LEARNED_MERGE_VETO_COS ∈ {0.60, 0.65, 0.70}) — may find a threshold with fewer HOTA regressions.
- **3b**: Different signal stack — e.g. bbox-size continuity, pose-keypoint distance, per-frame appearance trajectory rather than median.
- **3c**: Accept ceiling per Session 7 — redirect effort to ball/action/score/court workstreams. 12 swaps across 43 rallies is 4.4% of identities affected; HOTA is already 91.6%.

## Runtime

- `baseline`: 0.0s
- `learned_t080`: 0.0s
- `velocity_v25`: 8.5s
- `velocity_v35`: 7.8s
- `combined_070_v25`: 8.0s
- `combined_070_v35`: 8.0s
- `combined_080_v25`: 8.0s
- `combined_080_v35`: 8.0s
- **Total**: 48s (0.8 min)
