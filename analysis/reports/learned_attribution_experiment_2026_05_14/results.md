# Per-action-type learned attribution — validation experiment (2026-05-14)

## Setup

- 641 GT rows across 12 trusted-GT videos: titi, toto, lulu, wawa, caco, cece, cici, cuco, gaga, kaka, juju, yeye
- 30 features per candidate (mixed per-candidate + per-contact-shared signals)
- HistGradientBoostingClassifier (sklearn) per-action-type — equivalent gradient boosting to LightGBM
- Per-candidate binary target `is_gt`; argmax(P(is_gt=1)) across 4 candidates picks the predicted attribution
- Leave-one-video-out CV across all 12 videos
- Anti-leakage: `is_pipeline_pick` is excluded from features

## Per-type precision

| Type | N GT (matched-to-contact) | Baseline pipeline | Learned | Δ (pp) |
|---|---|---|---|---|
| SERVE | 78 | 67/78 = 85.9% | 70/78 = 89.7% | +3.8pp |
| RECEIVE | 73 | 65/73 = 89.0% | 65/73 = 89.0% | +0.0pp |
| SET | 117 | 94/117 = 80.3% | 100/117 = 85.5% | +5.1pp |
| ATTACK | 141 | 122/141 = 86.5% | 117/141 = 83.0% | -3.5pp |
| DIG | 72 | 57/72 = 79.2% | 57/72 = 79.2% | +0.0pp |

**Aggregate** (excluding BLOCK): baseline 405/481 = 84.2% → learned 409/481 = 85.0% (+0.8pp)

Note: 'N GT' here is GT rows that matched a pipeline contact within ±5 frames
(the same denominator as the baseline-precision A-table in `precision_trusted_videos_2026_05_14`).
Baseline 'no pipeline pick' (pl_tid missing) counts as wrong; learned always picks 1 of 4.

## Per-type top-5 features (by permutation importance)

### SERVE
- `candidate_dist_rank`: 0.0887
- `body_velocity_pre`: 0.0310
- `bbox_motion_dx`: 0.0168
- `body_velocity_post`: 0.0166
- `team_match_expected`: 0.0140

### RECEIVE
- `candidate_dist_rank`: 0.1368
- `team_match_expected`: 0.0563
- `body_velocity_around`: 0.0215
- `proximity_to_ball`: 0.0159
- `approach_speed_to_ball`: 0.0110

### SET
- `candidate_dist_rank`: 0.2198
- `team_match_expected`: 0.0439
- `proximity_to_ball`: 0.0254
- `is_next_toucher`: 0.0208
- `body_velocity_post`: 0.0192

### ATTACK
- `candidate_dist_rank`: 0.1596
- `team_match_expected`: 0.0311
- `body_velocity_around`: 0.0143
- `proximity_trajectory_endpoint`: 0.0108
- `action_confidence`: 0.0102

### DIG
- `candidate_dist_rank`: 0.2102
- `direction_change_deg`: 0.0270
- `is_prev_toucher`: 0.0262
- `is_next_toucher`: 0.0218
- `bbox_motion_dy`: 0.0159

## Decision criteria

Ship gate for labeling investment:
- ≥ 5pp lift over baseline on **at least 2 action types**
- No action type regresses by more than 2pp

- Types with ≥5pp lift: 1 (SET)
- Types regressing by ≥2pp: 1 (ATTACK)

## Verdict: **NO-SIGNAL-DONT-LABEL**
