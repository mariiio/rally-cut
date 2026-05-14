# Pose v2 probe (scaled-down) — 2026-05-14

## Setup

- 35 stratified GT contacts × ~4 canonical-pid candidates each
- Pose model: cached `yolo11s-pose.pt` keypoints (NO re-inference)
- Total inferences: 0 (uses pre-existing pose cache)
- Runtime: 22.9s (0.4 min)
- Hard caps respected: MAX_CONTACTS=40, MAX_PER_VIDEO=4, HARD_TIMEOUT=900s
- Strict serial execution, gc.collect() every contact, intermediate dump every 10
- Pose data coverage: 35 rallies with pose, 0 contacts had no pose detections at contact frame

## Per-type precision (on 40-contact subset)

| Type | N | Baseline pipeline | Today's GBM | + Pose v2 | Δ vs today | Δ vs baseline |
|---|---|---|---|---|---|---|
| SERVE | 7 | 5/7 = 71.4% | 6/7 = 85.7% | 5/7 = 71.4% | -14.3pp | +0.0pp |
| RECEIVE | 7 | 6/7 = 85.7% | 5/7 = 71.4% | 5/7 = 71.4% | +0.0pp | -14.3pp |
| SET | 7 | 6/7 = 85.7% | 4/7 = 57.1% | 4/7 = 57.1% | +0.0pp | -28.6pp |
| ATTACK | 7 | 7/7 = 100.0% | 7/7 = 100.0% | 7/7 = 100.0% | +0.0pp | +0.0pp |
| DIG | 7 | 6/7 = 85.7% | 6/7 = 85.7% | 6/7 = 85.7% | +0.0pp | +0.0pp |

**Aggregate**: baseline 30/35 = 85.7% → today 28/35 = 80.0% → pose-v2 27/35 = 77.1%

## Per-type top-5 features (by permutation importance)

### SERVE
- `approach_speed_to_ball`: 0.2739
- `pv2_best_wrist_to_ball_dist`: 0.0411 **[pose]**
- `pv2_body_velocity_pre_contact`: 0.0028 **[pose]**
- `proximity_pre_contact_mean`: 0.0000
- `proximity_pre_contact_min`: 0.0000
_pose-feature-in-top5: True_

### RECEIVE
- `body_velocity_post`: 0.1333
- `approach_speed_to_ball`: 0.0463
- `bbox_motion_dx`: 0.0093
- `proximity_to_ball`: 0.0000
- `proximity_pre_contact_mean`: 0.0000
_pose-feature-in-top5: False_

### SET
- `candidate_dist_rank`: 0.1472
- `body_velocity_post`: 0.0556
- `team_match_expected`: 0.0532
- `pv2_body_velocity_post_contact`: 0.0361 **[pose]**
- `proximity_to_ball`: 0.0000
_pose-feature-in-top5: True_

### ATTACK
- `candidate_dist_rank`: 0.3903
- `proximity_to_ball`: 0.0000
- `proximity_pre_contact_mean`: 0.0000
- `proximity_pre_contact_min`: 0.0000
- `proximity_trajectory_endpoint`: 0.0000
_pose-feature-in-top5: False_

### DIG
- `candidate_dist_rank`: 0.2535
- `bbox_motion_dx`: 0.0208
- `pv2_best_wrist_to_ball_dist`: 0.0042 **[pose]**
- `proximity_to_ball`: 0.0000
- `proximity_pre_contact_mean`: 0.0000
_pose-feature-in-top5: True_

## Decision criteria (small-sample friendly)

Scaled-down criteria (40-contact subset is smaller than the 641-row experiment so
thresholds are looser than the canonical 5pp/2pp gates):
- ≥3pp lift on ≥2 action types (where n≥5)
- No type regresses by ≥3pp (where n≥5)
- ≥1 type has a pose feature in top-5

- Types lifting ≥3pp (n≥5): 0 (none)
- Types regressing ≥3pp (n≥5): 1 (SERVE)
- Types with pose feature in top-5: 3 (SERVE, SET, DIG)

## Verdict: **FLAT — pose adds noise, no lift**

## Artifacts

- Pose feature vectors per contact: `/Users/mario/Personal/Projects/RallyCut/analysis/reports/pose_v2_probe_2026_05_14/pose_features.jsonl`
- Intermediate progress mirror: `/tmp/pose_v2_intermediate.jsonl`
- Machine-readable results: `/Users/mario/Personal/Projects/RallyCut/analysis/reports/pose_v2_probe_2026_05_14/results.json`