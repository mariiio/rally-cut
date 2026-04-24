# Pose-feature discrimination audit — TP vs FN (rejected-by-classifier)

Tolerance for TP/FN matching: ±7 frames
Total rows emitted: 2157
FN bucket (non-block, rejected_by_classifier) recovered: 128
FN with keypoints: 111 / 128 (target ~90/128 per pre-validated cov)

## Per-class FN coverage (in 128 bucket)

| Class | FN total | FN with kpts | TP sampled |
|---|---:|---:|---:|
| serve | 18 | 10 | 147 |
| receive | 25 | 22 | 206 |
| set | 15 | 13 | 300 |
| attack | 26 | 24 | 300 |
| dig | 44 | 42 | 239 |

## Per-feature ROC-AUC (TP vs FN, both with keypoints)

| Class | n_TP | n_FN | wrist_velocity_max | hand_ball_dist_min | arm_extension_change | pose_confidence_mean | both_arms_raised | best_AUC |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| serve | 147 | 10 | 0.615 | 0.573 | 0.630 | 0.562 | 0.531 | 0.630 |
| receive | 206 | 22 | 0.656 | 0.621 | 0.661 | 0.504 | 0.546 | 0.661 |
| set | 300 | 13 | 0.735 | 0.592 | 0.604 | 0.654 | 0.620 | 0.735 |
| attack | 300 | 24 | 0.587 | 0.645 | 0.663 | 0.502 | 0.668 | 0.668 |
| dig | 239 | 42 | 0.528 | 0.592 | 0.534 | 0.558 | 0.577 | 0.592 |

**Overall best feature/class:** wrist_velocity_max on set (AUC = 0.735)

## TP vs FN distributions (median [p25, p75])

### serve  (TP n=147, FN n=10)

| Feature | TP median [p25,p75] | FN median [p25,p75] | TP/FN ratio |
|---|---|---|---:|
| wrist_velocity_max | 0.0133 [0.0033, 0.0514] | 0.0044 [0.0037, 0.0163] | 3.03 |
| hand_ball_dist_min | 0.1303 [0.0464, 0.2512] | 0.1021 [0.0307, 0.2126] | 1.28 |
| arm_extension_change | 27.4266 [5.8629, 88.2274] | 5.8016 [2.0114, 44.7702] | 4.73 |
| pose_confidence_mean | 0.7770 [0.7462, 0.8308] | 0.7789 [0.7358, 0.7971] | 1.00 |
| both_arms_raised | 0.0000 [0.0000, 0.1056] | 0.0000 [0.0000, 0.4500] | nan |

### receive  (TP n=206, FN n=22)

| Feature | TP median [p25,p75] | FN median [p25,p75] | TP/FN ratio |
|---|---|---|---:|
| wrist_velocity_max | 0.0179 [0.0068, 0.0357] | 0.0074 [0.0022, 0.0209] | 2.43 |
| hand_ball_dist_min | 0.0797 [0.0423, 0.1537] | 0.1172 [0.0697, 0.2154] | 0.68 |
| arm_extension_change | 30.9636 [13.9719, 70.2025] | 12.5835 [3.7811, 32.4756] | 2.46 |
| pose_confidence_mean | 0.7913 [0.7335, 0.8692] | 0.7849 [0.7462, 0.8479] | 1.01 |
| both_arms_raised | 0.0000 [0.0000, 0.0000] | 0.0000 [0.0000, 0.0000] | nan |

### set  (TP n=300, FN n=13)

| Feature | TP median [p25,p75] | FN median [p25,p75] | TP/FN ratio |
|---|---|---|---:|
| wrist_velocity_max | 0.0193 [0.0098, 0.0324] | 0.0101 [0.0052, 0.0112] | 1.91 |
| hand_ball_dist_min | 0.0535 [0.0178, 0.1157] | 0.0900 [0.0199, 0.1786] | 0.59 |
| arm_extension_change | 39.2865 [13.5630, 83.7320] | 18.3906 [6.4779, 70.5586] | 2.14 |
| pose_confidence_mean | 0.7834 [0.7435, 0.8267] | 0.7717 [0.7394, 0.7774] | 1.02 |
| both_arms_raised | 0.0000 [0.0000, 0.8000] | 0.0000 [0.0000, 0.0000] | nan |

### attack  (TP n=300, FN n=24)

| Feature | TP median [p25,p75] | FN median [p25,p75] | TP/FN ratio |
|---|---|---|---:|
| wrist_velocity_max | 0.0273 [0.0146, 0.0515] | 0.0175 [0.0052, 0.0461] | 1.56 |
| hand_ball_dist_min | 0.0343 [0.0214, 0.1499] | 0.1695 [0.0384, 0.2238] | 0.20 |
| arm_extension_change | 46.8505 [23.3363, 110.8514] | 25.0429 [12.5228, 49.2379] | 1.87 |
| pose_confidence_mean | 0.7770 [0.7505, 0.8233] | 0.7808 [0.7315, 0.8667] | 1.00 |
| both_arms_raised | 0.3636 [0.0000, 0.8182] | 0.0000 [0.0000, 0.2500] | nan |

### dig  (TP n=239, FN n=42)

| Feature | TP median [p25,p75] | FN median [p25,p75] | TP/FN ratio |
|---|---|---|---:|
| wrist_velocity_max | 0.0185 [0.0078, 0.0373] | 0.0196 [0.0044, 0.0344] | 0.95 |
| hand_ball_dist_min | 0.0671 [0.0284, 0.1244] | 0.0987 [0.0505, 0.1536] | 0.68 |
| arm_extension_change | 41.9607 [14.5832, 112.1598] | 35.3379 [12.1059, 98.0848] | 1.19 |
| pose_confidence_mean | 0.7904 [0.7510, 0.8643] | 0.7832 [0.7443, 0.8256] | 1.01 |
| both_arms_raised | 0.0000 [0.0000, 0.0909] | 0.0000 [0.0000, 0.0000] | nan |
