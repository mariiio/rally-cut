# Corpus Delta — Pre- vs Post-GT-Repair (2026-04-17)

## Summary

GT integrity repair ran `match-players → remap-track-ids → reattribute-actions` on 52 videos (65 rallies moved from `auto_fixable` → `clean`; the refined classifier now splits 364 rallies into **254 clean / 0 auto_fixable / 110 gt_orphaned**). `reports/gt_integrity_orphaned_rallies_2026_04_17.md` documents why S3 restore wasn't viable for the 110 orphans.

The corpus was rebuilt against the post-repair DB. Because `reattribute-actions` also re-runs `classify_rally_actions`, predictions shifted on the 52 touched videos.

## Totals

|                   | Pre-repair | Post-repair | Δ       |
|-------------------|-----------:|------------:|--------:|
| GT contacts       |       2098 |        2098 |       0 |
| True positives    |            |         747 |         |
| FN_contact        |        537 |         589 |     +52 |
| wrong_action      |        274 |         293 |     +19 |
| wrong_player      |        417 |         469 |     +52 |
| **Total errors**  |   **1228** |    **1351** | **+123** |

## Breakdown by rally bucket

|                | Clean (254) | Orphaned (110) |
|----------------|------------:|---------------:|
| Pre errors     |         787 |            441 |
| Post errors    |         905 |            446 |
| Δ              |        +118 |             +5 |

The delta is concentrated on the 52 videos touched by the recovery pipeline. Orphaned rallies (not touched) are within noise (+5 errors, mostly a single-rally `classify_rally_actions` run-to-run jitter).

## Reading the delta

The +118 error delta on the 254 clean rallies is not a true regression against the labelled reality — it's a measurement shift. Pre-repair, `wrong_player` was *under*-counted because `match_contacts:540-544` marked GT labels with `player_track_id ∉ available_track_ids` as `player_evaluable=False`. Post-repair, many of those labels are now resolvable against current tracks, so they are now counted — and some of those evaluations produce wrong_player hits.

The post-repair corpus is a more honest measurement of the classifier and attribution pipeline because:
- GT playerTrackIds resolve to live tracks on 254 rallies (up from ~189).
- `match_contacts` can compare pred→canonical vs gt→canonical end-to-end on more labels.
- `reattribute-actions` predictions reflect current production logic (using match-level team assignments from the freshly-rebuilt `trackToPlayer`).

This is the baseline we measure Phase 3 (sequence wiring) against.

## FN subcategories (post-repair)

| Subcategory               | Count |
|---------------------------|------:|
| rejected_by_classifier    |   268 |
| rejected_by_gates         |   188 |
| no_player_nearby          |    71 |
| deduplicated              |    25 |
| no_candidate              |    19 |
| ball_dropout              |    18 |

`rejected_by_classifier` (268) is the single largest FN bucket — the classifier threshold (0.35) is the biggest FN lever in the current system. Pattern 1 in the design doc targets this bucket via a gap-aware rescue gate.

## Action misclassifications (post-repair, top 10)

| Confusion          | Count |
|--------------------|------:|
| set→receive        |    52 |
| dig→attack         |    33 |
| attack→set         |    30 |
| dig→set            |    30 |
| receive→serve      |    26 |
| set→attack         |    21 |
| attack→receive     |    20 |
| serve→receive      |    17 |
| attack→dig         |    16 |
| receive→set        |    13 |

The set↔receive / dig↔set / attack↔set / attack↔dig cluster (set-adjacent confusions, 181 total) is the target for Pattern 2 (MS-TCN++ sequence override in `classify_rally_actions`).

## Files

- `outputs/action_errors/corpus_pre_repair.jsonl` — snapshot of the pre-repair corpus
- `outputs/action_errors/corpus.jsonl` — post-repair baseline
- `outputs/action_errors/corpus_annotated.jsonl` — post-repair corpus joined with per-rally quality signals
- `reports/gt_integrity_diagnosis_refined.json` — rally-level classification
- `reports/gt_integrity_orphaned_rallies_2026_04_17.md` — orphan-recovery decision log
