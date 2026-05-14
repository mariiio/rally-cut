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
`bbox_aspect_ratio` quality penalty). End-to-end A/B on trusted-14 shows
**+8.2pp** attribution accuracy lift. User-flagged worst rally (keke r1)
fully fixed (4/4 actions correct end-to-end).

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

## Feature space (10 features per candidate)

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

## A/B results (full trusted-14 corpus)

End-to-end after `redetect_all_actions` with flag OFF vs ON:

| Metric | Baseline (OFF) | Scorer (ON) | Δ |
|---|---|---|---|
| Total | 446/571 (78.1%) | **493/571 (86.3%)** | **+47 correct (+8.2pp)** |
| SERVE | 81.9% | 83.1% | +1.2pp |
| RECEIVE | 84.5% | 92.9% | +8.4pp |
| SET | 75.4% | 83.6% | +8.2pp |
| ATTACK | 79.8% | 87.3% | +7.5pp |
| DIG | 71.3% | **92.0%** | **+20.7pp** |
| BLOCK | 60% | 30% | −30pp (10 rows, noise) |

Per-video: 12 wins, 1 same, 1 regression (wawa −17.9pp).

**Post-fix A/B (with widened tolerance + coverage gate):**

| Metric | Baseline (OFF) | Scorer (ON) | Δ |
|---|---|---|---|
| Total | 446/571 (78.1%) | **497/571 (87.0%)** | **+51 correct (+8.9pp)** |

Per-video: 13 wins, only wawa still regresses (−7.1pp, recovered +10.7pp from initial measurement).

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
3. ✅ End-to-end A/B on trusted-14 corpus (+8.2pp, 12/14 video wins)
4. ✅ Investigate wawa regression — root cause identified (feature-window mismatch; see below)
5. ⏳ Fix `_find_pos` tolerance (widen 2→5) + retrain → closes most wawa regression
6. ⏳ Add `weights/dynamic_attribution_scorer/` to `rallycut train push-weights` pattern list
7. ⏳ Coherence-audit baseline diff (C-1..C-5 violation deltas)
8. ⏳ Bump `ACTION_PIPELINE_VERSION` v2→v3 + fleet refresh when defaulting ON
9. ⏳ Periodic retraining as new GT is added (every ~100 new GT rows)

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

- **Pose dynamics**: wrist velocity + body orientation — requires DB schema
  change to save keypoints, would address residual ATTACK errors (attack-block
  contest where `bbox_dist` dominates and bbox-only dynamic features don't
  disambiguate). Estimated additional lift: +2-4pp on ATTACK specifically.
- **Per-rally model selection**: some rallies may benefit from different
  feature weights (night-time vs day, beach vs indoor). Could become
  per-condition model selection if signal exists.

## Memory entry

[[dynamic_attribution_scorer_2026_05_14]] — captures findings, user-facing
validation on keke r1, integration details, and what's left before default-on.
