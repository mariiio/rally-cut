# Phase 1.5 — Composite Primitive Partition

**Date:** 2026-04-24  
**Scope:** 468 baseline matches across 9 fixtures.  

## Partition

| bucket | count | % |
|---|---|---|
| **correct** | 206 | 44.0% |
| **primitive_fixable** | 43 | 9.2% |
| **chooser_fixable** | 117 | 25.0% |
| **detection_limit** | 71 | 15.2% |
| **irreducible** | 31 | 6.6% |
| TOTAL | 468 | 100% |

## Per-fixture

| fixture | n | correct | prim_fix | choose_fix | det_lim | irred |
|---|---|---|---|---|---|---|
| **cece** | 23 | 7 | 10 | 2 | 4 | 0 |
| **cuco** | 40 | 4 | 2 | 28 | 3 | 3 |
| **lala** | 71 | 28 | 5 | 19 | 10 | 9 |
| **lulu** | 37 | 4 | 1 | 23 | 8 | 1 |
| **rere** | 35 | 12 | 0 | 17 | 3 | 3 |
| **tata** | 108 | 75 | 3 | 10 | 13 | 7 |
| **toto** | 59 | 16 | 6 | 17 | 13 | 7 |
| **wawa** | 33 | 10 | 10 | 1 | 11 | 1 |
| **yeye** | 62 | 50 | 6 | 0 | 6 | 0 |

## Subcategory detail

| bucket | count |
|---|---|
| `chooser_fixable__cross_team_geometric` | 117 |
| `detection_limit` | 71 |
| `primitive_fixable__team_pair_wrong` | 38 |
| `irreducible__within_team` | 31 |
| `primitive_fixable__unknown_team_fallback` | 2 |
| `primitive_fixable__coverage_gap` | 1 |
| `primitive_fixable__swap_near_pl_pid` | 1 |
| `primitive_fixable__swap_near_gt_pid` | 1 |

## Phase 2 headroom

- Total wrong + missing-but-fixable: **191** (40.8% of all matches)
- Primitive-fixable share: **22.5%** of wrong — addressable only by Phase 1.3a (contact-time team validation) + Phase 1.1a (swap-aware abstention).
- Chooser-fixable share: **61.3%** of wrong — Phase 2 confidence-gated chooser direct target.
- Irreducible within-team: **31** — architectural floor per memory `within_team_reid_project_2026_04_16.md`.
- Detection limit (missing): **71** — rescue-only via Phase 3.2 complement.

## Decision

**Phase 1 → Phase 2 handoff:**

- Chooser-fixable bucket is where Phase 2's confidence-gated chooser has direct leverage. Kill gate unchanged: halve wrong_rate.
- Primitive-fixable bucket is ship-blocking on Phase 1.1a (swap-aware abstention) + Phase 1.3a (contact-time team check). Both are chooser-side policies that use the Phase-1 audit artifacts as inputs — no stage-2 upstream fixes.
- Irreducible + detection-limit buckets are accepted as architectural floor.