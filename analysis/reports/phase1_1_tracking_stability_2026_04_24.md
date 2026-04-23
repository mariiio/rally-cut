# Phase 1.1 — Tracking-ID Stability Audit

**Date:** 2026-04-24  
**Scope:** 69 rallies across 9 fixtures.  
**Gate:** `rally_stable_rate ≥ 70%`  
**Result:** `55.1%` — **FAIL** — upstream fixes required before Phase 2.

## Stability definition

A rally is *stable* iff **all** of:
- Zero silent swaps (position jump > 0.08/frame for any primary tid).
- Zero mid-rally dropouts (tid missing for ≥ 45 consecutive frames after first appearing).
- All 4 primary tids present for ≥ 40% of rally frame span.
- All of {1, 2, 3, 4} present as primary tids.

## Per-fixture

| fixture | rallies | stable | rate | swaps | dropouts | low_cov | missing_tids |
|---|---|---|---|---|---|---|---|
| **cece** | 4 | 3 | 75.0% ✅ | 1 | 0 | 0 | 0 |
| **cuco** | 5 | 2 | 40.0% ⚠️ | 10 | 1 | 0 | 0 |
| **lala** | 7 | 3 | 42.9% ⚠️ | 2 | 3 | 0 | 0 |
| **lulu** | 9 | 3 | 33.3% ⚠️ | 6 | 7 | 0 | 0 |
| **rere** | 5 | 4 | 80.0% ✅ | 0 | 1 | 0 | 0 |
| **tata** | 16 | 10 | 62.5% ⚠️ | 1 | 5 | 0 | 2 |
| **toto** | 11 | 9 | 81.8% ✅ | 0 | 4 | 1 | 0 |
| **wawa** | 5 | 1 | 20.0% ⚠️ | 0 | 5 | 0 | 0 |
| **yeye** | 7 | 3 | 42.9% ⚠️ | 5 | 10 | 0 | 0 |
| **COMBINED** | **69** | **38** | **55.1%** | 25 | 36 | 1 | 2 |

## Unstable rallies

### cece

- `8f17a3ed` — ⚡ 1 swap(s)

### cuco

- `d0d64de1` — ⚡ 1 swap(s)
- `f127f3d5` — ⚡ 9 swap(s)
- `d2995693` — ⏸ 1 dropout(s)

### lala

- `793625cd` — ⏸ 1 dropout(s)
- `a80db496` — ⏸ 1 dropout(s)
- `2eeb3ae6` — ⚡ 2 swap(s)
- `9666105f` — ⏸ 1 dropout(s)

### lulu

- `060e5898` — ⚡ 2 swap(s), ⏸ 1 dropout(s)
- `8da4f0dc` — ⏸ 1 dropout(s)
- `2adb9d80` — ⚡ 2 swap(s), ⏸ 1 dropout(s)
- `69117837` — ⏸ 1 dropout(s)
- `55565c2b` — ⚡ 2 swap(s)
- `71c642dc` — ⏸ 3 dropout(s)

### rere

- `c036a173` — ⏸ 1 dropout(s)

### tata

- `09553ef1` — ⏸ 1 dropout(s)
- `ad7cccbf` — ⏸ 1 dropout(s)
- `8c802c26` — ⏸ 1 dropout(s)
- `03144243` — ⚡ 1 swap(s)
- `4bfafd6d` — ⏸ 1 dropout(s), ❓ 1 missing tid(s)
- `689854e3` — ⏸ 1 dropout(s), ❓ 1 missing tid(s)

### toto

- `d24901dd` — ⏸ 1 dropout(s), 📉 1 low-coverage tid(s)
- `248aaf83` — ⏸ 3 dropout(s)

### wawa

- `7094136a` — ⏸ 1 dropout(s)
- `8c49e480` — ⏸ 1 dropout(s)
- `7f0f540a` — ⏸ 2 dropout(s)
- `06c13117` — ⏸ 1 dropout(s)

### yeye

- `a67c04fb` — ⚡ 1 swap(s), ⏸ 2 dropout(s)
- `4c0f4c83` — ⏸ 3 dropout(s)
- `cbf17cce` — ⏸ 3 dropout(s)
- `2d3cb54b` — ⚡ 4 swap(s), ⏸ 2 dropout(s)


## Decision

**Stability gate failed: 55.1% < 70%.** 6/9 fixtures below threshold. **Phase 1.1a policy response (not upstream fix):**

Per memory, upstream swap recovery is architecturally blocked — 5 sessions of ML attempts (Sessions 4-8, 2026-04) on within-team ReID all NO-GO'd; BoT-SORT's swap rate is the production ceiling. Tightening the dropout or coverage thresholds further would be measurement gaming on a real ceiling.

**Instead, Phase 1.1 hands forward a `swap_events` and `low_coverage` map to Phase 2.2 chooser** (see `phase1_1_tracking_stability.json`). The Phase-2 chooser MUST abstain on contacts landing within ±5 frames of a swap event on the candidate tid, and MUST abstain when the candidate tid has < 40% rally coverage. This converts per-rally tracking failures into `missing_rate` rather than `wrong_rate` — aligned with the north-star 'prefer miss over wrong.'

**Phase 1.2 proceeds on the current tracking foundation.**

Fixture-specific follow-ups (not blocking):
- **wawa** (20%, 0 swaps / 5 dropouts): systematic fragmentation — all 5 rallies have ≥1 long dropout. Worth a retrack pass to investigate.
- **cuco / lulu** (heavy swaps): most within-rally swap events concentrated here. Possible candidates for manual `swap-tracks` CLI.
- **tata**: 2 rallies with only 3 primary tids (4th player never tracked). Occlusion at serve; not fixable upstream.