# Block-vs-attack signal diagnostic — 2026-05-14

## Class counts
- GT blocks (with pipeline candidate ±5): 12 / 13
- GT attacks (with pipeline candidate ±5): 162
- Both classes have candidate-frame data: yes

## Per-signal AUROC

| Signal | dir | Block median [Q1, Q3] | Attack median [Q1, Q3] | n_b | n_a | AUROC |
|---|---|---|---|---|---|---|
| `ms_tcn_block_prob` | high | 0.002 [0.001, 0.251] | 0.001 [0.000, 0.002] | 12 | 162 | 0.651 |
| `prev_team_cross` | high | 1.000 [0.000, 1.000] | 0.000 [0.000, 1.000] | 7 | 127 | 0.620 |
| `arc_fit_residual` | high | 0.004 [0.003, 0.005] | 0.003 [0.001, 0.004] | 12 | 162 | 0.606 |
| `pre_contact_ball_dy_sign` | low | 0.017 [0.015, 0.020] | 0.019 [0.013, 0.023] | 12 | 162 | 0.587 |
| `velocity` | low | 0.018 [0.016, 0.020] | 0.019 [0.015, 0.024] | 12 | 162 | 0.572 |
| `ms_tcn_block_minus_attack` | high | -0.900 [-0.980, 0.226] | -0.968 [-0.987, -0.302] | 12 | 162 | 0.567 |
| `wrist_y_image` | low | 0.442 [0.399, 0.486] | 0.427 [0.370, 0.488] | 11 | 152 | 0.433 |
| `body_center_minus_ball_y` | high | 0.129 [0.124, 0.152] | 0.128 [0.112, 0.154] | 11 | 161 | 0.559 |
| `player_bbox_top_y` | low | 0.427 [0.385, 0.464] | 0.411 [0.353, 0.468] | 11 | 161 | 0.441 |
| `ms_tcn_attack_prob` | low | 0.902 [0.020, 0.987] | 0.970 [0.304, 0.989] | 12 | 162 | 0.550 |
| `wrist_minus_net_y` | low | -0.249 [-0.272, -0.238] | -0.256 [-0.278, -0.189] | 11 | 152 | 0.530 |
| `ball_y_at_contact` | low | 0.396 [0.356, 0.463] | 0.394 [0.328, 0.464] | 12 | 162 | 0.471 |
| `direction_change_deg` | low | 153.500 [113.682, 173.748] | 157.602 [127.607, 172.532] | 12 | 162 | 0.508 |
| `wrist_above_net` | high | 1.000 [1.000, 1.000] | 1.000 [1.000, 1.000] | 11 | 152 | 0.500 |

Direction column meaning: `low` ⇒ smaller value predicts BLOCK; `high` ⇒ larger value predicts BLOCK. AUROC is computed in the direction-aware sense (AUROC > 0.5 = signal as informative).

## Top signals (by |AUROC − 0.5|)

- `ms_tcn_block_prob` (dir=high, AUROC=0.651, n_block=12, n_attack=162)
- `prev_team_cross` (dir=high, AUROC=0.620, n_block=7, n_attack=127)
- `arc_fit_residual` (dir=high, AUROC=0.606, n_block=12, n_attack=162)

## Best single-signal threshold rule
`ms_tcn_block_prob` > 0.0033

- TP=6, FP=24, FN=6
- precision=0.200, recall=0.500, F1=0.286

## Best combined-signal threshold rule
(`ms_tcn_block_prob` > 0.0010) AND (`prev_team_cross` > 0.0000)

- TP=2, FP=8, FN=5
- precision=0.200, recall=0.286, F1=0.235

## Visualization
`scatter.png` — top-2 signals coloured by class.

## Verdict

NO-SIGNAL-block-floor — Top single-signal AUROC = 0.651 (ms_tcn_block_prob). Best single-signal threshold rule: prec=0.200 rec=0.500 F1=0.286. Best 2-signal combined rule: prec=0.200 rec=0.286 F1=0.235.
