# Session 8 — Per-Pass SAME_TEAM_SWAP Attribution

Incremental-skip diagnostic: runs `evaluate-tracking` 8 times with cumulative
subsets of post-processing passes enabled. Attribution for pass K = swaps at K
minus swaps at K-1.

## 8-Cell Summary

| K | Passes Enabled | Pass Just Added | Aggregate SAME_TEAM_SWAPs | Δ vs prev |
|---|---|---|---:|---:|
| 0 | 0 | — (raw BoT-SORT) | 2 | — |
| 1 | 1 | `ENFORCE_SPATIAL_CONSISTENCY` (step 0) | 2 | +0 |
| 2 | 2 | `FIX_HEIGHT_SWAPS` (step 0a) | 2 | +0 |
| 3 | 3 | `SPLIT_TRACKS_BY_COLOR` (step 0b) | 2 | +0 |
| 4 | 4 | `RELINK_SPATIAL_SPLITS` (step 0b2) | 2 | +0 |
| 5 | 5 | `RELINK_PRIMARY_FRAGMENTS` (step 0b3) | 2 | +0 |
| 6 | 6 | `LINK_TRACKLETS_BY_APPEARANCE` (step 0c) | 12 | +10 |
| 7 | 7 | `STABILIZE_TRACK_IDS` (step 1) | 12 | +0 |

## Per-Pass Attribution

| Pass | Step | Swaps Introduced | Verdict |
|---|---|---:|---|
| `ENFORCE_SPATIAL_CONSISTENCY` | 0 | 0 | innocent |
| `FIX_HEIGHT_SWAPS` | 0a | 0 | innocent |
| `SPLIT_TRACKS_BY_COLOR` | 0b | 0 | innocent |
| `RELINK_SPATIAL_SPLITS` | 0b2 | 0 | innocent |
| `RELINK_PRIMARY_FRAGMENTS` | 0b3 | 0 | innocent |
| `LINK_TRACKLETS_BY_APPEARANCE` | 0c | 5 | **SWAP CREATOR** |
| `STABILIZE_TRACK_IDS` | 1 | 0 | innocent |
| `<raw_botsort>` | — | 2 | **unexpected** |

## Per-Rally Swap Detail

| Rally ID | GT ID | Frame | Introduced By |
|---|---|---:|---|
| `209be896` | `—` | 263 | `LINK_TRACKLETS_BY_APPEARANCE` |
| `21a9b203` | `—` | 460 | `LINK_TRACKLETS_BY_APPEARANCE` |
| `29cb4e29` | `—` | 246 | `<raw_botsort>` |
| `53ca3586` | `—` | 244 | `LINK_TRACKLETS_BY_APPEARANCE` |
| `5e2e58fb` | `—` | 141 | `<raw_botsort>` |
| `b7f92cdc` | `—` | 207 | `LINK_TRACKLETS_BY_APPEARANCE` |
| `fad29c31` | `—` | 223 | `LINK_TRACKLETS_BY_APPEARANCE` |

## Decision Gate — Which Adapter Tasks to Execute

Based on the attribution above:

- `ENFORCE_SPATIAL_CONSISTENCY` (step 0): 0 swaps → skip / low priority (No task assigned)
- `FIX_HEIGHT_SWAPS` (step 0a): 0 swaps → skip / low priority (Task 3)
- `SPLIT_TRACKS_BY_COLOR` (step 0b): 0 swaps → skip / low priority (No task assigned)
- `RELINK_SPATIAL_SPLITS` (step 0b2): 0 swaps → skip / low priority (Task 4)
- `RELINK_PRIMARY_FRAGMENTS` (step 0b3): 0 swaps → skip / low priority (Task 5)
- **`LINK_TRACKLETS_BY_APPEARANCE`** (step 0c): **5 swap(s) introduced** → **EXECUTE Session 6 (already implemented)**
- `STABILIZE_TRACK_IDS` (step 1): 0 swaps → skip / low priority (Task 6)

## Runtime

 K | Pass Added                          |  Elapsed
-- | ----------                          |  -------
 0 | <none>                              |     7.5s
 1 | ENFORCE_SPATIAL_CONSISTENCY         |     7.2s
 2 | FIX_HEIGHT_SWAPS                    |     7.2s
 3 | SPLIT_TRACKS_BY_COLOR               |     6.9s
 4 | RELINK_SPATIAL_SPLITS               |     7.1s
 5 | RELINK_PRIMARY_FRAGMENTS            |     7.2s
 6 | LINK_TRACKLETS_BY_APPEARANCE        |     7.4s
 7 | STABILIZE_TRACK_IDS                 |     7.4s
