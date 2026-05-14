# 3D court projection probe — results 2026-05-14

## Setup

- 481 GT rows (matched-to-pipeline-contact), 12 trusted videos, leave-one-video-out CV.
- Baseline (today's experiment): 30 features. Aggregate precision: 85.0%.
- New: 30 + 6 court-projection features = 36 features. Aggregate precision: 85.4%.
- Pipeline-baseline (current production pick): 84.2%.

- Contacts with no usable homography (excluded from 3D feature contribution but kept in baseline + today's GBM eval): **4**. Court-projection-ok contacts: 477.

## Per-type precision lift

| Type | N GT | Baseline (today's GBM, 30f) | + 3D features (36f) | Δ vs today's GBM | Δ vs pipeline baseline |
|---|---|---|---|---|---|
| SERVE | 78 | 70/78 = 89.7% | 71/78 = 91.0% | +1.3pp | +5.1pp |
| RECEIVE | 73 | 65/73 = 89.0% | 66/73 = 90.4% | +1.4pp | +1.4pp |
| SET | 117 | 100/117 = 85.5% | 98/117 = 83.8% | -1.7pp | +3.4pp |
| ATTACK | 141 | 117/141 = 83.0% | 117/141 = 83.0% | +0.0pp | -3.5pp |
| DIG | 72 | 57/72 = 79.2% | 59/72 = 81.9% | +2.8pp | +2.8pp |

**Aggregate** (excluding BLOCK): pipeline 405/481=84.2% → today's GBM 409/481=85.0% → +3D 411/481=85.4% (Δ vs today's GBM +0.4pp, Δ vs pipeline +1.2pp)

## Top features per type (36-feature model, by permutation importance)

### SERVE
- `candidate_dist_rank`: 0.0812
- `body_velocity_pre`: 0.0215
- `team_match_expected`: 0.0143
- `body_velocity_post`: 0.0097
- `court_y`: 0.0068  ⭐3D

### RECEIVE
- `candidate_dist_rank`: 0.1514
- `team_match_expected`: 0.0529
- `proximity_to_ball`: 0.0162
- `body_velocity_around`: 0.0110
- `court_y`: 0.0104  ⭐3D

### SET
- `candidate_dist_rank`: 0.2049
- `team_match_expected`: 0.0422
- `is_next_toucher`: 0.0208
- `proximity_to_ball`: 0.0194
- `court_x`: 0.0170  ⭐3D

### ATTACK
- `candidate_dist_rank`: 0.1558
- `team_match_expected`: 0.0278
- `action_confidence`: 0.0080
- `court_x`: 0.0079  ⭐3D
- `proximity_trajectory_endpoint`: 0.0071

### DIG
- `candidate_dist_rank`: 0.2075
- `is_next_toucher`: 0.0192
- `is_prev_toucher`: 0.0185
- `bbox_motion_dy`: 0.0174
- `direction_change_deg`: 0.0159

## Per-error subset: F3-shape (cross-team-flow contacts)

Definition: contacts where the prior action's player was on the *other* team from the expected-team in the rally chain — the occlusion-flip cases.

### F3-shape, ATTACK-only (hypothesis-of-record)

- N = 38
- Pipeline baseline: 35/38 = 92.1%
- Today's GBM (30f): 34/38 = 89.5%
- +3D (36f): 35/38 = 92.1%
- Δ (36f vs 30f): +2.6pp

### F3-shape, all types pooled

- N = 170
- Pipeline baseline: 145/170 = 85.3%
- Today's GBM (30f): 147/170 = 86.5%
- +3D (36f): 148/170 = 87.1%
- Δ (36f vs 30f): +0.6pp

## Decision criteria

- **SHIP-3D-FEATURES** if ≥5pp lift on ≥2 action types AND no type regresses by >2pp (vs today's GBM).
- **TARGETED-3D** if F3-shape ATTACK subset lifts ≥10pp even without aggregate lift.
- **SKIP-3D** if flat across the board.

- Types with ≥5pp lift (vs today's GBM): 0 (none)
- Types regressing by ≥2pp (vs today's GBM): 0 (none)
- F3-shape ATTACK lift: +2.6pp (threshold +10pp)

## Verdict: **SKIP-3D**

## Recommended next probe

Try scaled-down Pose v2 next: ~100 contacts, ~5 min, pure-numpy pose-feature extraction over existing position snapshots (no model inference).

## Runtime

- Total: 125.1s
- Peak Python heap (tracemalloc): 186.2 MB
