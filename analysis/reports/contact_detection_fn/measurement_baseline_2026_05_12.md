# Contact-Detection FN-Reduction — Baseline Measurement (2026-05-12)

All `RELAX_CONTACT_*` env flags **OFF**. Comparison reference for Phase 1 A/B
per-flag evaluation.

Source: `scripts/measure_contact_recall_full.py --label baseline`
Full JSON (gitignored): `reports/contact_detection_fn/measurement_baseline_2026_05_12.json`

## Corpus

| Metric        | Value |
| ------------- | ----- |
| Rallies       | 409   |
| Videos        | 69    |
| GT actions    | 2374  |
| Pipeline acts | 2263  |

## Headline

| Metric                 | Value  | n              |
| ---------------------- | ------ | -------------- |
| Recall                 | 0.8909 | 2115 / 2374    |
| Precision proxy        | 0.9346 | 2115 / 2263    |

Match tolerance: ±10 frames between pipeline action frame and GT frame.
Greedy nearest-match per GT, no per-pipeline-action reuse.

## Per-action-type recall

| Action  | Recall  | n             |
| ------- | ------- | ------------- |
| attack  | 0.9342  | 610 / 653     |
| block   | 0.1081  | 4 / 37        |
| dig     | 0.8355  | 315 / 377     |
| receive | 0.9194  | 342 / 372     |
| serve   | 0.8627  | 352 / 408     |
| set     | 0.9336  | 492 / 527     |

`block` recall (10.8%) is the standout floor — block events go almost entirely
undetected by the current contact pipeline.

## Coherence-violation counts (game-rule audit)

| Invariant | Count |
| --------- | ----- |
| C-1 (3-contact max)         | 112 |
| C-2 (alternating possessions) | 333 |
| C-3 (first action is serve) | 2   |

Audit ran cleanly across all 69 videos (0 per-video errors).

## Notes

- C-2 (333) is the dominant coherence violation, consistent with the broader
  fleet-wide pattern noted in the coherence-invariants v1 entry; same-team
  back-to-back attribution gaps the structural audit can see.
- The 89.09% recall + 93.46% precision proxy match the smoke-run output and
  the earlier `probe_2026_05_12` numbers — measurement is stable.
- This baseline is the comparison reference for Plan Task 9
  (per-flag A/B evaluation of `RELAX_CONTACT_*` flags).
