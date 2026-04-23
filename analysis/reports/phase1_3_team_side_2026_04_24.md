# Phase 1.3 — Team→Side Audit

**Date:** 2026-04-24  
**Scope:** 69 locked GT rallies across 9 fixtures.  
**Gate:** `identity-first agreement with positional ≥ 98%` on audit-eligible rallies.  
**Result:** `26.1%` over 23 eligible rallies — **FAIL**

## Method

For each rally:

- **Method A (positional)**: median y of each primary tid over rally-start frames (< 15f); the 2 highest-y tids are "near", the 2 lowest are "far".
- **Method B (identity-first)**: each team letter's pids per `actions_json.teamAssignments`; team with higher median foot-y is "near".
- **Agreement**: method B's near-team pid set equals method A's near pid set (teams match up to A↔B label flip).
- Rallies with `pair_confidence < 0.05` (median-y split near the midline) are excluded as audit-inconclusive.

## Per-fixture

| fixture | rallies | eligible | agree | disagree | low-conf | rate |
|---|---|---|---|---|---|---|
| **cece** | 4 | 2 | 0 | 2 | 2 | 0.0% ⚠️ |
| **cuco** | 5 | 0 | 0 | 0 | 5 | 0.0% ⚠️ |
| **lala** | 7 | 1 | 0 | 1 | 6 | 0.0% ⚠️ |
| **lulu** | 9 | 7 | 6 | 1 | 2 | 85.7% ⚠️ |
| **rere** | 5 | 0 | 0 | 0 | 5 | 0.0% ⚠️ |
| **tata** | 16 | 2 | 0 | 2 | 14 | 0.0% ⚠️ |
| **toto** | 11 | 2 | 0 | 2 | 9 | 0.0% ⚠️ |
| **wawa** | 5 | 3 | 0 | 3 | 2 | 0.0% ⚠️ |
| **yeye** | 7 | 6 | 0 | 6 | 1 | 0.0% ⚠️ |
| **COMBINED** | **69** | **23** | **6** | **17** | **46** | **26.1%** |

## Disagreements (stage-2 teamAssignments bugs)

### cece
- `5c35e049` — positional near=`[2, 3]` vs identity-first near=`[1, 2]`  (teamAssignments: `{'1': 'A', '2': 'A', '3': 'B', '4': 'B'}`)
- `8f17a3ed` — positional near=`[2, 3]` vs identity-first near=`[1, 2]`  (teamAssignments: `{'1': 'A', '2': 'A', '3': 'B', '4': 'B'}`)

### lala
- `6a467cf6` — positional near=`[2, 3]` vs identity-first near=`[3, 4]`  (teamAssignments: `{'1': 'A', '2': 'A', '3': 'B', '4': 'B'}`)

### lulu
- `2adb9d80` — positional near=`[2, 4]` vs identity-first near=`[1, 2]`  (teamAssignments: `{'1': 'B', '2': 'B', '3': 'A', '4': 'A'}`)

### tata
- `ad7cccbf` — positional near=`[1, 4]` vs identity-first near=`[3, 4]`  (teamAssignments: `{'1': 'A', '2': 'A', '3': 'B', '4': 'B'}`)
- `d5c51d52` — positional near=`[2, 3]` vs identity-first near=`[1, 2]`  (teamAssignments: `{'1': 'B', '2': 'B', '3': 'A', '4': 'A'}`)

### toto
- `e5a1da4f` — positional near=`[2, 4]` vs identity-first near=`[1, 2]`  (teamAssignments: `{'1': 'B', '2': 'B', '3': 'A', '4': 'A'}`)
- `67b3e1ad` — positional near=`[2, 4]` vs identity-first near=`[1, 2]`  (teamAssignments: `{'1': 'B', '2': 'B', '3': 'A', '4': 'A'}`)

### wawa
- `8c49e480` — positional near=`[1, 4]` vs identity-first near=`[1, 2]`  (teamAssignments: `{'1': 'A', '2': 'A', '3': 'B', '4': 'B'}`)
- `7f0f540a` — positional near=`[1, 4]` vs identity-first near=`[3, 4]`  (teamAssignments: `{'1': 'A', '2': 'A', '3': 'B', '4': 'B'}`)
- `06c13117` — positional near=`[1, 4]` vs identity-first near=`[1, 2]`  (teamAssignments: `{'1': 'A', '2': 'A', '3': 'B', '4': 'B'}`)

### yeye
- `df3ba73b` — positional near=`[2, 4]` vs identity-first near=`[3, 4]`  (teamAssignments: `{'1': 'A', '2': 'A', '3': 'B', '4': 'B'}`)
- `a67c04fb` — positional near=`[2, 4]` vs identity-first near=`[3, 4]`  (teamAssignments: `{'1': 'A', '2': 'A', '3': 'B', '4': 'B'}`)
- `4c0f4c83` — positional near=`[2, 4]` vs identity-first near=`[1, 2]`  (teamAssignments: `{'1': 'A', '2': 'A', '3': 'B', '4': 'B'}`)
- `cbf17cce` — positional near=`[2, 4]` vs identity-first near=`[3, 4]`  (teamAssignments: `{'1': 'A', '2': 'A', '3': 'B', '4': 'B'}`)
- `721bb968` — positional near=`[2, 4]` vs identity-first near=`[3, 4]`  (teamAssignments: `{'1': 'A', '2': 'A', '3': 'B', '4': 'B'}`)
- `61f79c70` — positional near=`[2, 4]` vs identity-first near=`[3, 4]`  (teamAssignments: `{'1': 'A', '2': 'A', '3': 'B', '4': 'B'}`)


## Decision

**Fail (26.1%).** Stage-2 team inference pairs the wrong players as teammates on a non-trivial fraction of rallies. **Phase 1.3a action**: rather than block Phase 2, document the teamAssignments failure mode and require Phase-2 chooser to validate team-membership via positional-check at contact-frame instead of trusting teamAssignments. Label-flip (uniform A↔B swap without breaking pair grouping) remains a Phase 2.3 roster-aware chooser concern.