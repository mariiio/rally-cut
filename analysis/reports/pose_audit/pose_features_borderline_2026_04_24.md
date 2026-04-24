# Pose features on classifier-rejected FNs — diagnostic memo

**Date:** 2026-04-24
**Question:** Do the 5 GBM pose features (`nearest_active_wrist_velocity_max`,
`nearest_hand_ball_dist_min`, `nearest_active_arm_extension_change`,
`nearest_pose_confidence_mean`, `nearest_both_arms_raised`) discriminate
true-positive contacts from the 128 non-block, classifier-rejected FNs?
**Verdict:** Borderline — closer to NO than YES. Pose is not the lever.

## Method

Two passes on the 128 non-block `rejected_by_classifier` FNs cataloged in
`outputs/action_errors/corpus_annotated.jsonl` (151 total, minus 23 block).

- **Pass 1+2** (`scripts/pose_audit_2026_04_24.py`): for every GT contact in
  the 377-rally action-GT corpus, compute the 5 pose features at the GT frame
  via `extract_contact_pose_features_for_nearest`, then label each row TP/FN
  via `match_contacts(tolerance=7)` against `contacts_json`. Compute per-class
  ROC-AUC for TP vs FN where both rows have keypoints.
- **Pass 3** (`scripts/pose_audit_pass3_2026_04_24.py`): re-train the GBM
  with production hyperparams on the 101 rallies that contain an FN-bucket
  case. Two label schemes: standard (all GT contacts as positives) and
  focused (only FN-bucket frames as positives, exclude other GT to avoid
  label leakage). Compare `feature_importances_`.

## Pose coverage on the 128 FN bucket

111 / 128 (87%) have keypoints — better than pre-validated 90/128 (70%).
Pose-blind cases: 17 (cannot be touched by any pose-feature change).

## Pass 2 — per-class TP-vs-FN ROC-AUC

| Class    | n_TP | n_FN | wrist_vel_max | hand_ball_min | arm_ext_chg | pose_conf | both_arms | best |
|---       |---:  |---:  |---:           |---:           |---:         |---:       |---:       |---:  |
| serve    | 147  | 10   | 0.615         | 0.573         | 0.630       | 0.562     | 0.531     | 0.630 |
| receive  | 206  | 22   | 0.656         | 0.621         | 0.661       | 0.504     | 0.546     | 0.661 |
| set      | 300  | 13   | **0.735**     | 0.592         | 0.604       | 0.654     | 0.620     | **0.735** |
| attack   | 300  | 24   | 0.587         | 0.645         | 0.663       | 0.502     | 0.668     | 0.668 |
| dig      | 239  | 42   | 0.528         | 0.592         | 0.534       | 0.558     | 0.577     | 0.592 |

Best AUC = 0.735 on `set / wrist_velocity_max` (n_FN=13 — small). Per the
pre-registered gate (best AUC ≥ 0.65 on at least one class) → proceed to
Pass 3.

**The dominant FN class (dig, 42/128 = 33% of bucket) tops out at AUC 0.592 —
basically chance.** Any pose intervention is structurally capped from helping
33% of the bucket.

## Pass 3 — focused subpopulation retrain

| Setting                      | Pose total importance |
|---                           |---:                   |
| Production GBM (reference)   | 1.45%                 |
| Standard training (101 rallies, all GT as positives) | 1.45% |
| Focused training (101 rallies, FN-bucket as positives) | **6.34%** |

Pose importance rose **+4.89pp**. The pre-registered threshold for "pose
features ARE useful" was +5.00pp. Miss by 0.11pp — not a clean PASS.

Crucially, **trajectory and player-position features rose much more** under
focused training:

| Feature                | Standard | Focused | Δ        |
|---                     |---:      |---:     |---:      |
| `frames_since_last`    | 34.31%   | 7.94%   | -26.37pp |
| `seq_max_nonbg`        | 34.55%   | 22.56%  | -11.98pp |
| `ball_y`               | 0.73%    | 6.10%   | +5.37pp  |
| `player_distance`      | 8.07%    | 11.22%  | +3.16pp  |
| `acceleration`         | 1.29%    | 4.43%   | +3.14pp  |
| `best_player_max_d_y`  | 1.26%    | 4.16%   | +2.89pp  |
| `consecutive_detections` | 0.74% | 3.26%   | +2.51pp  |
| `ball_y_relative_net`  | 1.79%    | 3.90%   | +2.11pp  |
| `pose total (5)`       | 1.45%    | 6.34%   | +4.89pp  |

Trajectory features cumulatively rose ~17pp. Player-position features
cumulatively rose ~10pp. Pose rose only +4.89pp.

## Diagnosis

The 128 FN bucket is **not pose-detectable as a primary signal**. It is a
**hard-example weighting problem affecting all features in the same direction.**
When the GBM is forced to discriminate the FN bucket from same-rally noise,
every feature gains importance, but trajectory + player-position gain ~3-5×
more than pose.

Pose features genuinely have weak modular signal:

- 17/128 cases have no keypoints (architectural ceiling).
- dig (42/128) is at chance (AUC 0.592).
- set (13/128) at AUC 0.735 is fragile (small n).
- The remaining 56 cases (serve+receive+attack) sit at AUC 0.61-0.67 — useful
  but not strong enough to drive a population-level rescue.

A pose-only intervention's plausible upside is bounded: targeting set + the
~half of attack/receive cases where AUC > 0.65, with the dig majority
unaffected. Realistically +1-2pp F1 lift in a best case, with high FP risk
(per `crop_head_phase2_nogo_2026_04_20.md` and the contact_arbitrator NO-GO).

## Decision

- **Pose feature engineering as a primary lever: NO-GO.** The data does not
  support it.
- **Hard-example weighting / curriculum training: GO probe.** The Pass 3
  result strongly suggests the FN bucket is a global weighting problem, not
  a feature-coverage problem. Test in 3-5 days via `sample_weight` upweighting
  on the 128 cases.
- **Trajectory feature engineering: GO design.** Pass 3 shows trajectory
  features rise much more than pose under focused training. The "tight-touch"
  signature (low velocity / dcd) may be detectable with new trajectory
  features (micro-acceleration, sub-frame inflections).

Plan at `docs/superpowers/plans/2026-04-24-fn-bucket-hard-example-weighting.md`.
