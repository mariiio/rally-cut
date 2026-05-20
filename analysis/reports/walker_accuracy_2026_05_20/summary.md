# Walker Accuracy Replay — Summary (2026-05-20)

Substrate: 1265 GT-pair events from reports/gt_net_crossings_2026_05_20/events.csv.

| Config | correct_flip | missed_flip | correct_stay | over_flip | total_correct |
|---|---:|---:|---:|---:|---:|
| cfg_00 (v13 baseline) | 482 | 64 | 699 | 20 | 1181/1265 |
| cfg_10 (B.1 only) | 495 | 51 | 699 | 20 | 1194/1265 |
| cfg_01 (B.2 only) | 546 | 0 | 719 | 0 | 1265/1265 |
| cfg_11 (B.1+B.2) | 546 | 0 | 719 | 0 | 1265/1265 |

## Sanity check

cfg_00 must produce numbers identical to the v13 baseline (482 correct_flip, 64 missed_flip, 699 correct_stay, 20 over_flip). If not, the refactor broke v13-equivalent behavior — STOP and debug before A/B.

## CAVEAT: B.2 tautology on this substrate

The events.csv reconstructs `court_side` from team identity (team A = near, team B = far) because the original GT-pair probe didn't capture Contact.court_side. B.2 (ball-trajectory verifier) consumes court_side as the override signal — so on this substrate, B.2 mechanically matches the GT-derived signal and appears perfect by construction. The realistic B.2 ceiling lives in Stage-2 A/B which uses production Contact.court_side (derived from ball Y vs net_y). cfg_00 baseline and cfg_10 (B.1 only) are unaffected — they don't consume court_side as a primary signal.
