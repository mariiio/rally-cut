# Dynamic Attribution Scorer Design (2026-05-14)

## Context

The 2026-05-14 attribution-ceiling memo concluded that classical-ML
attribution-flipping on the *static* feature space is empirically exhausted on
the 641-trusted-GT corpus. Per-action-type GBM on static features lifted only
+0.8pp (within noise). The remaining classes of error documented there:
attack-block contest ambiguity, edge-bbox quality, cross-team mid-rally
crossover.

This spec proposes a complementary intervention that exhausted the *dynamic*
feature space: per-action-type GBM scorer that augments existing static
features (`bbox_dist`, etc.) with temporal features computed across a small
window around each contact (`velocity_mag`, `top_y_change`, `height_change`,
`bbox_aspect_ratio` quality penalty) plus pose-derived signals (wrist
velocity, wrist-to-ball, body orientation, arms-raised, post-contact
alignment, pose confidence). End-to-end A/B on the trusted-21 corpus
(978 GT rows after a +7-video expansion landed 2026-05-15) shows
**+10.2pp** attribution accuracy lift (v2 pose; +2.2pp over the v1
bbox+sequence model). User-flagged worst rally (keke r1) fully fixed
(4/4 actions correct end-to-end).

## What ships

1. **6 per-action-type GBM models** (one each for SERVE, RECEIVE, SET,
   ATTACK, DIG, BLOCK) at
   `analysis/weights/dynamic_attribution_scorer/{ACTION}_v1.joblib`.
2. **Inference module** `analysis/rallycut/tracking/dynamic_attribution_scorer.py`
   providing lazy-loaded singleton scorer with `pick(action, candidates)`
   API.
3. **Pipeline integration**: new helper `_apply_dynamic_scorer_attribution`
   in `analysis/rallycut/tracking/action_classifier.py` runs after
   `reattribute_players` Pass 2, gated by env flag
   `USE_DYNAMIC_ATTRIBUTION_SCORER` (default OFF).
4. **Training infrastructure**:
   - `analysis/scripts/train_and_save_dynamic_scorer_2026_05_14.py`
     (production training, no hold-out)
   - `analysis/scripts/train_dynamic_attribution_scorer_2026_05_14.py`
     (LOO CV measurement)
5. **Verification scripts**:
   - `analysis/scripts/probe_dynamic_features_2026_05_14.py` — feature
     discrimination analysis
   - `analysis/scripts/probe_scorer_on_user_flagged_2026_05_14.py` —
     per-user-rally verification

## Feature space (17 features per candidate)

| Feature | Type | Description |
|---|---|---|
| `bbox_dist` | static | Upper-quarter bbox distance to ball (mirrors `contact_detector._player_to_ball_dist` bbox fallback) |
| `bbox_area` | static | `w * h` at contact |
| `bbox_aspect_ratio` | static | `w / h` — penalizes degenerate thin-strip detections |
| `bbox_inside_frame` | static | 1.0 if fully inside frame, 0.0 if extending off |
| `velocity_mag` | **dynamic** | `|dx,dy|` of bbox center across `f-5..f+5` |
| `velocity_toward_ball` | **dynamic** | Radial component of velocity vector toward ball |
| `top_y_at_contact` | **dynamic** | Bbox top-y at contact (smaller = higher in image = jumping) |
| `top_y_change` | **dynamic** | `y - y(f-5)` (negative = rising) |
| `height_change` | **dynamic** | `height(f+3) - height(f-3)` (positive = extending) |
| `same_as_prev` | **sequence** | 1.0 if candidate.tid == previous action's playerTrackId else 0.0 — discourages C-4 violations |
| `wrist_velocity_max` | **pose** | Max wrist speed across `f-3..f+3` (peak swing) |
| `wrist_to_ball_min` | **pose** | Min wrist-to-ball distance across `f-3..f+3` |
| `body_orientation_diff` | **pose** | |angle(left_shoulder→right_shoulder) − bearing(player→ball)| (rad) |
| `arms_raised` | **pose** | Both wrists above shoulder line at contact (0/1) |
| `wrist_post_alignment` | **pose** | Cosine alignment between wrist motion `f..f+3` and ball direction `f..f+3` |
| `pose_confidence_mean` | **pose** | Mean keypoint confidence over the 7-frame window |
| `wrist_y_velocity` | **pose v2.1** | Vertical component of wrist motion; positive = downward swing, added to disambiguate ATTACK contests |

Feature importance varies by action type:
- SERVE: dominated by `bbox_aspect_ratio` (0.45) — penalizes degenerate
  off-frame server bboxes — and `top_y_change` (0.21) — rising into toss.
- RECEIVE: balanced static + dynamic; `bbox_dist`, `height_change`, `top_y_change`.
- SET: dominated by `height_change` (0.38) — reaching extension is the dominant signal.
- ATTACK: 68% `bbox_dist` — dynamic features barely contribute; attack-block
  contest remains the residual.
- DIG: `bbox_dist` (0.33) + `top_y_change` (0.22) — rising motion matters.
- BLOCK: `bbox_dist` (0.46) + `velocity_mag`/`velocity_toward_ball` — net jumping motion.

## Training methodology

**Critical: production-matched feature extraction.** Earlier training used
GT-snapshot ball positions (cleaner but distribution-mismatched). Switching
to pipeline-action `frame` + `ballX/ballY` flipped end-to-end production
result from −3.1pp to **+15.6pp** on user-flagged contacts. Standard ML
deployment principle: train distribution must match inference distribution.

```python
for each GT row in rally_action_ground_truth:
    1. Match to corresponding pipeline action (prefer same action_type
       within ±5 frames; else closest by frame)
    2. Use the pipeline action's frame + ballX/ballY as input
    3. Label: 1 if candidate.tid == GT.resolved_track_id else 0
    4. Per action type: train GradientBoostingClassifier
       (n_estimators=80, max_depth=3, learning_rate=0.05)
```

Training corpus: 622 GT rows from trusted-14 (full corpus, no hold-out for
production). 125 GT rows skipped (contact-FN, no matching pipeline action).

LOO CV measurement (for honest generalization estimate):
- Train on N-1 videos, evaluate on held-out video
- Per-action rank-1 accuracy: +5.6pp over bbox_dist baseline, +12.4pp over
  current pipeline picker (on 678 contacts)

## Inference flow

```
classify_rally_actions():
    1. classify_rally() → action types + serve detection
    2. repair_action_sequence(Rule 1 + 4 + 8)
    3. viterbi_decode_actions()
    4. validate_action_sequence()
    5. assign_court_side_from_teams()
    6. reattribute_players()  # team-chain Pass 2
    7. _apply_dynamic_scorer_attribution()  # NEW — env-gated
    8. apply_sequence_override()  # MS-TCN++ argmax for non-serve types
```

For each action with `confidence ≥ 0.3` and `player_track_id ≥ 0` and
`action_type != UNKNOWN`:
- Build CandidateFeatures for each track in `contact.player_candidates`
- Call `scorer.pick(action_type, candidates)` → returns track_id of argmax
- If different from current `action.player_track_id`, swap

When flag is OFF or models are unavailable: no-op, byte-identical behavior.

## A/B results

### v3.1 trusted-29 (final ship-ready, 2026-05-17 PM)

Adds `team_matches_expected` (feature #18) with per-action masking — the
feature defaults to `0.5` (uninformative) for SET to avoid the team-chain
cascade regression that v3 introduced.

**Four-way matched-accuracy A/B (n=1027):**

| Action | OFF | v2 pose | v3 team | **v3.1 SET-mask** | v3.1 − v2 |
|---|---|---|---|---|---|
| ATTACK | 83.9% | 88.1% | 91.0% | **90.0%** | +1.9 |
| DIG | 82.6% | 91.3% | 93.8% | **94.4%** | +3.1 |
| RECEIVE | 87.6% | 92.2% | 93.5% | **94.1%** | +2.0 |
| SET | 86.1% | 91.2% | 89.1% | **90.3%** | −0.8 |
| SERVE | 81.5% | 84.2% | 84.2% | 84.2% | 0.0 |
| BLOCK | 55.6% | 33.3% | 27.8% | 33.3% | 0.0 (n=18 noise) |
| **TOTAL** | **83.9%** | **88.4%** | **89.3%** | **89.6%** | **+1.2** |

**Honest LOO CV (n=974, 29 folds, 2026-05-17):**

| Action | n | bbox baseline | LOO scorer | Δ |
|---|---|---|---|---|
| SERVE | 124 | 73.4% | 95.2% | +21.8pp |
| SET | 225 | 75.6% | 90.7% | +15.1pp |
| DIG | 158 | 77.2% | 90.5% | +13.3pp |
| RECEIVE | 143 | 78.3% | 90.2% | +11.9pp |
| ATTACK | 307 | 82.4% | 85.0% | +2.6pp |
| BLOCK | 17 | 41.2% | 47.1% | noise |
| **TOTAL** | **974** | **77.5%** | **88.6%** | **+11.1pp** |

**LOO 88.6% is within 1pp of full-pipeline matched 89.6% — the scorer
generalizes; the lift is not training-data memorization.**

**Coherence-invariant audit (v3.1 ON vs OFF on trusted-29):**

| Invariant | v3.1 ON | OFF | v3.1 − OFF |
|---|---|---|---|
| C-1 (3-contact rule) | 44 | 33 | +11 (small regression) |
| C-2 (alt. possessions) | 78 | 76 | +2 (noise) |
| C-4 (same-player back-to-back) | 83 | 85 | −2 |
| **C-5 (cross-team crossover)** | 163 | 198 | **−35** (big win) |
| **TOTAL** | **368** | **392** | **−24** (−6.1%) |

The C-5 reduction is the team-awareness feature working as designed.
C-1 +11 is the trade-off for stronger same-team bias. Net coherence
improves.

Reports: `analysis/reports/attribution_trusted_29_2026_05_17/v3_1_summary.md`,
`analysis/reports/coherence_trusted_29_2026_05_17/summary.md`.

### v2 trusted-29 retrain (historic, 2026-05-17 AM)

End-to-end `redetect_all_actions --apply` × 29 videos × two passes
(`USE_DYNAMIC_ATTRIBUTION_SCORER` = 0 then 1) using the v2 scorer
retrained on trusted-29 (~1017 matched GT rows, 972 positives).

| Action | OFF | ON | Δ |
|---|---|---|---|
| Total (matched) | 862/1027 (83.9%) | **908/1027 (88.4%)** | **+4.5pp** |
| DIG | 66.8% | 73.9% | +7.0pp ▲ |
| SET | 73.7% | 78.1% | +4.3pp ▲ |
| ATTACK | 73.3% | 77.0% | +3.7pp ▲ |
| RECEIVE | 69.4% | 73.1% | +3.6pp ▲ |
| SERVE | 59.8% | 61.8% | +2.0pp ▲ |
| BLOCK | 37.0% | 22.2% | −14.8pp ▼ (n=27 noise; 17 positives in training) |

Per-video on trusted-29: **21 wins, 5 small regressions (max −6.7pp on
vivi n=15), 3 unchanged**. Adding 8 videos of GT did NOT shift the
matched-accuracy headline (88.3% → 88.4%) — the feature space is the
binding constraint, not training-set size.

Full report: `analysis/reports/attribution_trusted_29_2026_05_17/`.

### v2 final (trusted-21, 17 features incl. pose, 2026-05-15)

End-to-end after `redetect_all_actions --apply` with flag OFF vs ON:

| Metric | Baseline (OFF) | v1 bbox+seq | **v2 bbox+seq+pose** | Δ vs baseline | Δ v2 vs v1 |
|---|---|---|---|---|---|
| Total | 590/755 (78.1%) | 650/755 (86.1%) | **666/754 (88.3%)** | **+10.2pp** | **+2.2pp** |
| SERVE | 81.6% | 81.6% | 82.5% | +0.9pp | +0.9pp |
| RECEIVE | — | 89.7% | **94.1%** | — | +4.4pp |
| SET | — | 84.7% | **89.1%** | — | +4.4pp |
| ATTACK | — | 88.4% | 88.7% | — | +0.3pp (contest residual unchanged) |
| DIG | — | 91.9% | 93.7% | — | +1.8pp |
| BLOCK | — | noisy (n=13) | noisy (n=13) | — | flat |

**Pose lift is concentrated on RECEIVE + SET** — exactly the actions where
wrist proximity / arms-raised / orientation disambiguate same-team
candidates that look identical to bbox-only features. ATTACK contests
(near-side blocker occludes far-side attacker) remain the unresolved
residual — pose features alone don't crack them, consistent with the
2026-05-14 ceiling memo's call for visual scene reasoning.

### v1 initial (trusted-14, 10 features, 2026-05-14, historic)

For reference, the first non-falsified version of the scorer:

| Metric | Baseline (OFF) | v1 (ON) | Δ |
|---|---|---|---|
| Total | 446/571 (78.1%) | 497/571 (87.0%) | +8.9pp |
| DIG | 71.3% | 92.0% | +20.7pp |
| ATTACK | 79.8% | 87.3% | +7.5pp |
| RECEIVE | 84.5% | 92.9% | +8.4pp |

Per-video on trusted-14: 13 wins, only wawa regresses (−7.1pp, after the
coverage-gate fix).

**Coherence audit (C-1..C-5 violation counts) — neutral:**

| Aspect | OFF | ON | Δ |
|---|---|---|---|
| Total violations across trusted-14 | 291 | 292 | **+1** (neutral) |
| Videos with fewer violations | — | titi −15, kiki −12, juju −7, lulu −4, cici −3 | |
| Videos with more violations | — | toto +11, yeye +7, cuco +3, cece +2, gaga +2 | |
| keke outlier | 18 | 35 | +17 (baseline muddied by earlier Stages 1+2 changes; not a clean before/after) |

No broad coherence regression — scorer doesn't break sequence-rule integrity.

## Risk assessment

- **Env-flag gated, default OFF** → zero risk for current users.
- **Reversible** → set `USE_DYNAMIC_ATTRIBUTION_SCORER=0` to disable.
- **Wawa regression** (−17.9pp, 5 contacts) — needs investigation but doesn't block flagged deployment.
- **BLOCK regression** with only 10 GT rows is noise-level; collect more BLOCK GT before drawing conclusions.
- **Pipeline-version impact**: when defaulting ON, bump `ACTION_PIPELINE_VERSION` v2→v3.

## Ship plan

1. ✅ Train + save models (production-matched, full corpus)
2. ✅ Inference module + env-flag integration
3. ✅ End-to-end A/B on trusted-14 corpus v1 (+8.9pp, 13/14 video wins)
4. ✅ Investigate wawa regression — root cause identified (feature-window mismatch)
5. ✅ Fix `_find_pos` tolerance (widen 2→5) + retrain → wawa regression closed
6. ✅ Add coverage gate (≥3 candidate features + pipeline pick in feature set) to avoid bad swaps
7. ✅ Backfill pose keypoints across the 21 trusted videos (242 rallies, 47,738 positions)
8. ✅ Retrain v2 with 17 features (7 pose) on the trusted-21 corpus → 88.3% (+10.2pp vs baseline, +2.2pp vs v1)
9. ✅ Fix `redetect_all_actions.py` keypoint-loading bug — `b4fcab68` (was reconstructing PlayerPosition without keypoints, causing v2 to under-perform v1 at inference)
10. ✅ Retrain on trusted-29 (+8 videos) and A/B (2026-05-17) — sustains 88.4% matched, generalizes cleanly
11. ✅ v3 team-awareness experiment (2026-05-17) — implemented + refined to v3.1 with SET-mask
12. ✅ Honest LOO CV (2026-05-17) — 88.6% vs full-pipeline 89.6% (within 1pp, generalization confirmed)
13. ✅ Coherence-audit baseline diff (2026-05-17) — v3.1 has 24 fewer total violations than OFF; C-5 −35
14. ✅ Push-weights pattern — `weights/dynamic_attribution_scorer/` already in `WEIGHT_GROUPS["trained"]`
15. ⏳ Bump `ACTION_PIPELINE_VERSION` v2→v3 + flip env flag default OFF→ON (single ship commit)
16. ⏳ Fleet refresh post-ship: `uv run python scripts/redetect_all_actions.py --apply`
17. ⏳ Periodic retraining as new GT is added (every ~100 new GT rows)

## Wawa regression root cause (added 2026-05-14 post-A/B)

5 contacts regressed in wawa (−17.9pp). Investigation showed:
- **3 of 5: feature-window mismatch.** Scorer uses `_find_pos(tolerance=2)`
  in feature extraction; pipeline's `contact.player_candidates` uses ±15
  frames. When GT player has positions at ±3..±15 of contact but not ±2,
  pipeline's Pass 2 can swap to them, scorer drops them from its candidate
  set entirely. These are unrecoverable cases at current tolerance.
- **1 of 5: close-call (r9 f395 SET).** GT=P2 in candidates with prob=0.445
  vs P3 prob=0.467 — scorer's choice is genuinely close. May resolve with
  better features or more training data.
- **1 of 5: stale earlier comparison** — re-running showed r6 f623 ATTACK
  is actually correct (scorer picks GT=P3). So real regression is 4/45, not
  5/45.

Fix: widen `_find_pos` tolerance from 2 to 5 in both `extract_features`
(inference) and `_compute_features` (training). Must retrain to match
distribution. Estimated lift recovery on wawa: ~+3-4 contacts (from
75.0% → ~85%).

## Future work

- **v3 team-awareness feature (highest-EV, 2026-05-17 diagnosis)**: ATTACK
  residual catalog on trusted-21 showed **63% of v2 ATTACK errors are
  CROSS_TEAM** (attacker-vs-blocker confusion across the net), and the v2
  pose features actively amplify the confusion (blockers near the net look
  pose-similar to attackers). Add `team_matches_expected_attacker` as
  feature #18, retrain, A/B. Expected ATTACK matched accuracy 88.1% →
  ~94-95% (+6-7pp ATTACK, ~+2pp total). Risk: misleads when team-chain
  itself is wrong → gate by team-chain confidence or default to 0.5 for
  uninformative. See `analysis/reports/attack_residual_2026_05_17/`.
- **ATTACK contest residual after v3**: if CROSS_TEAM is closed, the
  remaining ~11 errors are 5 keke OTHER (video-specific track-quality)
  + 4 GT_MISSING (upstream contact-detection) + 2 small residual. At
  that point visual scene reasoning (DINOv2/V-JEPA/VLM) becomes the
  next-EV experiment if more lift is wanted.
- **BLOCK GT expansion**: only 13 BLOCK examples in trusted-21; model can't
  learn a reliable head. Need ≥50 BLOCK GT before treating any BLOCK
  measurement as load-bearing.
- **Per-rally model selection**: some rallies may benefit from different
  feature weights (night-time vs day, beach vs indoor). Could become
  per-condition model selection if signal exists.

## Memory entry

[[dynamic_attribution_scorer_2026_05_14]] — captures findings, user-facing
validation on keke r1, integration details, and what's left before default-on.
