# Phase 1.3 — Team→Side Audit

**Date:** 2026-04-24  
**Scope:** 69 locked GT rallies across 9 fixtures.  
**Gate:** `identity-first agreement with positional ≥ 98%` on audit-eligible rallies.  
**Result:** `100.0%` over 23 eligible rallies — **PASS**

## Method

For each rally:

- **Method A (positional)**: median y of each primary tid over rally-start frames (< 15f); the 2 highest-y tids are "near", the 2 lowest are "far".
- **Method B (identity-first)**: each team letter's pids per `actions_json.teamAssignments`; team with higher median foot-y is "near".
- **Agreement**: method B's near-team pid set equals method A's near pid set (teams match up to A↔B label flip).
- Rallies with `pair_confidence < 0.05` (median-y split near the midline) are excluded as audit-inconclusive.

## Per-fixture

| fixture | rallies | eligible | agree | disagree | low-conf | rate |
|---|---|---|---|---|---|---|
| **cece** | 4 | 2 | 2 | 0 | 2 | 100.0% ✅ |
| **cuco** | 5 | 0 | 0 | 0 | 5 | 0.0% ⚠️ |
| **lala** | 7 | 1 | 1 | 0 | 6 | 100.0% ✅ |
| **lulu** | 9 | 7 | 7 | 0 | 2 | 100.0% ✅ |
| **rere** | 5 | 0 | 0 | 0 | 5 | 0.0% ⚠️ |
| **tata** | 16 | 2 | 2 | 0 | 14 | 100.0% ✅ |
| **toto** | 11 | 2 | 2 | 0 | 9 | 100.0% ✅ |
| **wawa** | 5 | 3 | 3 | 0 | 2 | 100.0% ✅ |
| **yeye** | 7 | 6 | 6 | 0 | 1 | 100.0% ✅ |
| **COMBINED** | **69** | **23** | **23** | **0** | **46** | **100.0%** |

## Disagreements (stage-2 teamAssignments bugs)

*None — teamAssignments pairs physical teammates correctly.*

## Decision

**Pass (100.0%).** Identity-first team→side agrees with positional in all audit-eligible rallies. Phase 2 chooser can consume `teamAssignments` as a trusted primitive.