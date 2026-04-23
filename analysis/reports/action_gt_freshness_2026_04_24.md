# Phase 0.4 — Action-Type GT Freshness Audit

**Generated:** 2026-04-24  
**Source:** `reports/attribution_rebuild/baseline_2026_04_24.json`  
**Scope:** 9 Phase-0 fixtures (matched GT↔pipeline action pairs only; missing actions excluded since they have no pipeline action-type to compare).

## Per-fixture disagreement rate

| fixture | matched | agree | disagree | rate |
|---|---|---|---|---|
| cece | 19 | 17 | 2 | **10.5%** |
| cuco | 37 | 37 | 0 | **0.0%** |
| lala | 61 | 49 | 12 | **19.7%** |
| lulu | 29 | 25 | 4 | **13.8%** |
| rere | 32 | 31 | 1 | **3.1%** |
| tata | 95 | 88 | 7 | **7.4%** |
| toto | 45 | 36 | 9 | **20.0%** |
| wawa | 22 | 18 | 4 | **18.2%** |
| yeye | 56 | 53 | 3 | **5.4%** |
| **COMBINED** | **396** | **354** | **42** | **10.6%** |

## Disagreement pairs (gt → pipeline)

| GT action | Pipeline action | count |
|---|---|---|
| dig | set | 9 |
| attack | dig | 7 |
| dig | attack | 7 |
| attack | set | 6 |
| set | receive | 4 |
| set | dig | 3 |
| receive | serve | 2 |
| attack | serve | 1 |
| dig | receive | 1 |
| set | attack | 1 |
| set | block | 1 |

## Decision (Phase 3 Pattern A precondition)

**10.6% disagreement — borderline.** Use pipeline action types only when they carry high confidence (≥ conf threshold tbd). Below threshold, abstain rather than infer.

## Rallies with action-type disagreements

### cece
- `f978201e` @ 3000ms — 1 mismatch(es): f195 set→block
- `5c35e049` @ 42200ms — 1 mismatch(es): f289 dig→set

### lala
- `276985b8` @ 114386ms — 2 mismatch(es): f219 attack→dig, f250 dig→set
- `a80db496` @ 146655ms — 1 mismatch(es): f181 set→receive
- `2eeb3ae6` @ 233781ms — 4 mismatch(es): f542 set→attack, f571 attack→dig, f617 dig→set, f809 dig→attack
- `9666105f` @ 521747ms — 5 mismatch(es): f379 dig→attack, f416 set→dig, f540 attack→set, f594 dig→attack, f639 set→dig

### lulu
- `060e5898` @ 84300ms — 2 mismatch(es): f340 receive→serve, f439 set→receive
- `35ca5d33` @ 147163ms — 1 mismatch(es): f300 receive→serve
- `71c642dc` @ 583599ms — 1 mismatch(es): f559 attack→dig

### rere
- `c036a173` @ 105400ms — 1 mismatch(es): f621 dig→set

### tata
- `8b0b9e13` @ 0ms — 1 mismatch(es): f655 attack→dig
- `e50f127e` @ 209400ms — 1 mismatch(es): f204 dig→set
- `8c802c26` @ 269400ms — 1 mismatch(es): f273 attack→set
- `724ead56` @ 304400ms — 1 mismatch(es): f176 attack→dig
- `fdc4375b` @ 402200ms — 1 mismatch(es): f167 set→receive
- `d5c51d52` @ 456200ms — 1 mismatch(es): f264 set→dig
- `4e7e589c` @ 516600ms — 1 mismatch(es): f478 dig→attack

### toto
- `b52bc536` @ 212053ms — 1 mismatch(es): f148 set→receive
- `1d316b85` @ 396191ms — 5 mismatch(es): f209 attack→serve, f245 dig→receive, f290 dig→set, f450 attack→set, f493 dig→attack
- `248aaf83` @ 432297ms — 3 mismatch(es): f137 attack→set, f193 dig→set, f471 dig→attack

### wawa
- `7094136a` @ 10627ms — 2 mismatch(es): f308 attack→set, f367 dig→attack
- `8c49e480` @ 37854ms — 2 mismatch(es): f434 dig→set, f684 attack→dig

### yeye
- `a67c04fb` @ 56810ms — 1 mismatch(es): f261 attack→dig
- `4c0f4c83` @ 89947ms — 1 mismatch(es): f172 dig→set
- `cbf17cce` @ 146927ms — 1 mismatch(es): f808 attack→set
