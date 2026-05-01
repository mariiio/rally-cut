# Phase-1 verdict counts

- baseline:         AGREES 12/13
- dropema:          AGREES 12/13
- baseline_restore: AGREES 0/13 (sanity: should match baseline)

# Per-rally comparison (baseline vs dropema)

| rally | kind | expected | baseline | dropema | flipped? |
|---|---|---|---|---|---|
| 5c756c41/r01 | CTRL | GOOD | GOOD (0      clean) | GOOD (0      clean) | — |
| 5c756c41/r03 | PNL | BAD | GOOD (0      clean) | GOOD (0      clean) | — |
| 5c756c41/r07 | PNL | BAD | BAD (0      slow_drift (PID4 half-shift=0.58, xrange-overlap=0.51)) | BAD (0      slow_drift (PID4 half-shift=0.58, xrange-overlap=0.51)) | — |
| 7d77980f/r01 | CTRL | GOOD | GOOD (0      clean) | GOOD (0      clean) | — |
| 7d77980f/r02 | PNL | BAD | BAD (1      within_rally_swap (1 per-frame swap events)) | BAD (1      within_rally_swap (1 per-frame swap events)) | — |
| 7d77980f/r13 | PNL | GOOD | GOOD (0      clean (extras: [10])) | GOOD (0      clean (extras: [10])) | — |
| 7d77980f/r19 | PNL | GOOD | GOOD (0      clean (extras: [14])) | GOOD (0      clean (extras: [14])) | — |
| 854bb250/r01 | PNL | GOOD | GOOD (0      clean) | GOOD (0      clean) | — |
| 854bb250/r02 | CTRL | GOOD | GOOD (0      clean) | GOOD (0      clean) | — |
| b5fb0594/r01 | CTRL | BAD | BAD (0      hungarian_drop (missing PIDs [3]; have [1, 2, 4, 18])) | BAD (0      hungarian_drop (missing PIDs [3]; have [1, 2, 4, 18])) | — |
| b5fb0594/r04 | PNL | GOOD | GOOD (0      clean) | GOOD (0      clean) | — |
| b5fb0594/r06 | PNL | GOOD | GOOD (0      clean) | GOOD (0      clean) | — |
| b5fb0594/r10 | PNL | BAD | BAD (0      slow_drift (PID3 half-shift=0.21, xrange-overlap=0.56)) | BAD (0      slow_drift (PID3 half-shift=0.21, xrange-overlap=0.56)) | — |

# Panel BAD rallies — flip status

- BAD→GOOD flips: 0 / 4 panel-BAD rallies
  - 5c756c41/r03: baseline_bad=False, dropema_good=True, flipped=False
  - 5c756c41/r07: baseline_bad=True, dropema_good=False, flipped=False
  - 7d77980f/r02: baseline_bad=True, dropema_good=False, flipped=False
  - b5fb0594/r10: baseline_bad=True, dropema_good=False, flipped=False

- GOOD→BAD regressions in dropema: 0

# Phase 1 verdict gate

**CASCADE FALSIFIED** — drop EMA hypothesis, write NO-GO memo.

# Probe sidecar drift summary

## 5c756c41/baseline (sidecar: 5c756c41_baseline_20260501T114139Z.json)
- video: 5c756c41
- drop_ema: 0
- iter_records: 40
- update_records: 10
- pid1_lower_l2_range: 0.1721
- pid2_lower_l2_range: 0.0977
- pid3_lower_l2_range: 0.0780
- pid4_lower_l2_range: 0.0317

## 5c756c41/dropema (sidecar: 5c756c41_dropema_20260501T114338Z.json)
- video: 5c756c41
- drop_ema: 1
- iter_records: 40
- update_records: 10
- pid1_lower_l2_range: 0.0000
- pid2_lower_l2_range: 0.0000
- pid3_lower_l2_range: 0.0000
- pid4_lower_l2_range: 0.0000

## 7d77980f/baseline (sidecar: 7d77980f_baseline_20260501T114222Z.json)
- video: 7d77980f
- drop_ema: 0
- iter_records: 84
- update_records: 21
- pid1_lower_l2_range: 0.0951
- pid2_lower_l2_range: 0.1776
- pid3_lower_l2_range: 0.2586
- pid4_lower_l2_range: 0.0705

## 7d77980f/dropema (sidecar: 7d77980f_dropema_20260501T114421Z.json)
- video: 7d77980f
- drop_ema: 1
- iter_records: 84
- update_records: 21
- pid1_lower_l2_range: 0.0000
- pid2_lower_l2_range: 0.0000
- pid3_lower_l2_range: 0.0000
- pid4_lower_l2_range: 0.0000

## 854bb250/baseline (sidecar: 854bb250_baseline_20260501T114209Z.json)
- video: 854bb250
- drop_ema: 0
- iter_records: 20
- update_records: 5
- pid1_lower_l2_range: 0.1295
- pid2_lower_l2_range: 0.0718
- pid3_lower_l2_range: 0.0340
- pid4_lower_l2_range: 0.0442

## 854bb250/dropema (sidecar: 854bb250_dropema_20260501T114407Z.json)
- video: 854bb250
- drop_ema: 1
- iter_records: 20
- update_records: 5
- pid1_lower_l2_range: 0.0000
- pid2_lower_l2_range: 0.0000
- pid3_lower_l2_range: 0.0000
- pid4_lower_l2_range: 0.0000

## b5fb0594/baseline (sidecar: b5fb0594_baseline_20260501T114507Z.json)
- video: b5fb0594
- drop_ema: 0
- iter_records: 110
- update_records: 11
- pid1_lower_l2_range: 0.1060
- pid2_lower_l2_range: 0.1183
- pid3_lower_l2_range: 0.0644
- pid4_lower_l2_range: 0.1842

## b5fb0594/dropema (sidecar: b5fb0594_dropema_20260501T114310Z.json)
- video: b5fb0594
- drop_ema: 1
- iter_records: 110
- update_records: 11
- pid1_lower_l2_range: 0.0000
- pid2_lower_l2_range: 0.0000
- pid3_lower_l2_range: 0.0000
- pid4_lower_l2_range: 0.0000

