# Contact Detection Pipeline — Stage Reference

**Status:** Phase 1 deliverable for the 2026-04-21 contact-detection full review (brief: `docs/superpowers/briefs/2026-04-21-contact-detection-full-review.md`).
**Purpose:** Per-stage map of the production contact-detection code path. No analysis, no fixes — just "what does each stage do, what are its knobs, what can it and can't it catch." Phase 2 (visual inspection) consumes this.

All paths are relative to `analysis/` unless marked absolute. "Production" means the path `scripts/build_eval_reconciled_corpus.py` exercises (the harness Phase 0 reproduced the 88.15% / 93.15% baseline on).

## Stage map at a glance

| # | Stage | Module | Role |
|---|---|---|---|
| 1 | Ball tracker | `rallycut/tracking/ball_tracker.py` | WASB HRNet per-frame detection |
| 1b | Ball filter | `rallycut/tracking/ball_filter.py` | 12-step temporal cleanup + linear-interp gap fill (≤5f) |
| 2 | Player tracker | `rallycut/tracking/player_tracker.py` | YOLO11s + BoxMOT BoT-SORT + OSNet ReID |
| 3 | Sequence runtime | `rallycut/tracking/sequence_action_runtime.py` | MS-TCN++ per-frame action probs (7 classes × T) |
| 4 | Candidate generator | `rallycut/tracking/contact_detector.py::detect_contacts` steps 4-6c (and mirror `scripts/train_contact_classifier.py::extract_candidate_features`) | 10 generators + merge + 2 refinement passes |
| 5 | Feature extraction | same function, per-candidate feature compute | 26 features per candidate |
| 6 | GBM classifier | `rallycut/tracking/contact_classifier.py` | sklearn `GradientBoostingClassifier`, threshold 0.30 |
| 7 | Dedup + rescue | `detect_contacts` tail: `_apply_rescue_branch` + `_deduplicate_contacts` | Post-classifier rescue (off by default); single greedy adaptive-distance dedup |
| 8 | Action classifier | `rallycut/tracking/action_classifier.py::classify_rally_actions` | Re-labels action_type on existing contacts; NEVER adds/removes contacts (one exception: synthetic serve injection when serve missed) |
| 9 | Decoder overlay | `rallycut/tracking/decoder_runtime.py` + `decoder_overlay.py` + `candidate_decoder.py` | Viterbi over candidates, label-only overlay within ±3f. Does NOT add or remove contacts |
| 10 | Match (eval-only) | `scripts/eval_action_detection.py::match_contacts` | Hungarian assignment ±7f tol (≈233ms @ 30fps), then synthetic-serve 2nd pass at ~1s tol |

Where the 238 non-block FNs land per prior session (brief § "Previous session's coarse stage breakdown"):

| First stage lost at | FN count | File:function where the loss happens |
|---|---|---|
| Ball tracker (stage 1/1b) | 35 | `ball_filter.py` pruning/interpolation leaves gap; frame never carries a ball position |
| Candidate generator (stage 4) | 29 | All 10 generators + proximity/trajectory refinement fail to emit a frame within tol of GT |
| GBM classifier (stage 6) | 80 | Candidate exists within tol but `predict_proba[:,1] < threshold (0.30)` |
| Dedup (stage 7) | 84 | Contact was accepted by classifier, then removed by `_deduplicate_contacts` |
| Action labeling (stage 8) | 4 | Pre-serve FP filter or phantom-serve logic drops the contact |
| Matching-steal (stage 10, excl. block) | 3 | Two GT frames compete for the same pred; Hungarian assigns the other |

Priority for Phase 2: stages 4-7 (193/238 of losses).

---

## Stage 1 — Ball tracker (`ball_tracker.py`, 158 lines)

**Input:** Per-frame BGR video frames. Model path configured via `create_ball_tracker()` factory.
**Decision logic:**
- WASB HRNet keypoint detector (fine-tuned).
- Single-stage: per-frame inference produces `(x, y, confidence)` in normalized 0–1 coords.
- Confidence gate: `threshold=0.3` (factory default, `ball_tracker.py:154`).
- No temporal logic.

**Tunables:**

| Param | Default | Controls |
|---|---|---|
| `threshold` | 0.3 | Detection confidence gate for inference output |

**Output:** list of `BallPosition` (dataclass at `ball_tracker.py:27–45`) with fields `frame_number`, `x`, `y`, `confidence`, `motion_energy`. Frames with no detection are **missing from the list** — no zero-confidence placeholder.

**Handles:** nothing — pure detection only.

**Does not handle:** lock-on, oscillation, exit ghosts, static FPs, occlusion gaps, subtle deflection. All deferred to stage 1b.

---

## Stage 1b — Ball filter (`ball_filter.py`, 1272 lines)

**Input:** list of `BallPosition` from stage 1 + `BallFilterConfig` (lines 25–171). WASB production preset at `get_wasb_filter_config()` (lines 173–219).

**Decision logic — 12 filter steps in execution order** (see class docstring at line 222 and `filter_batch()` at line 242–442):

1. Motion energy filter — **disabled for WASB** (line 202). Removes stationary FPs at player positions.
2. Stationarity filter — **disabled for WASB** (line 204). Detects 12+ frames within 0.5% screen = lock-on.
3. **Exit ghost detection** (line 309–310) — marks frame ranges where ball approaches edge with consistent velocity then reverses. Thresholds: `exit_edge_zone=0.10`, `exit_approach_frames=3`, `exit_min_approach_speed=0.008`, `exit_max_ghost_frames=30`.
4. **Segment pruning** (line 312–316) — multi-substep:
   - 4a drop `confidence < min_output_confidence=0.05`.
   - 4b split at jumps ≥`segment_jump_threshold=0.20` or frame gaps ≥`max_segment_gap=15`.
   - 4c identify anchors (segments ≥`min_segment_frames=8`).
   - 4d exclude ghost-overlapping anchors.
   - 4e remove false start/tail anchors (warmup/cooldown).
   - 4f chain-based anchor pruning — build trajectory chain from longest/most-moving anchor; disconnected anchors killed unless ≥`min_disconnected_anchor_frames=30`. `max_chain_gap=30`.
   - 4g short-segment recovery within `proximity=threshold/2` of anchor endpoints.
5. Exit ghost removal (line 318–333) — apply detected ranges.
6. Oscillation pruning — **disabled for WASB** (line 210).
7. **Outlier removal** (line 345–347) — trajectory deviation > `max_trajectory_deviation=0.08`, edge-margin artifacts at `edge_margin=0.02`, velocity reversals > 120° with both speeds > `outlier_min_speed=0.02`.
8. Blip removal — **disabled for WASB** (line 215).
9. Re-pruning (conditional re-run of oscillation + segment if outlier removed any).
10. **Warmup protection re-injection** (line 371–396) — re-insert high-conf early detections (`frame<warmup_protect_frames=120`, `confidence>=warmup_protect_confidence=0.30`) pruned earlier; spatial-gated.
11. **Far-court protection re-injection** (line 398–411) — re-insert high-conf upper-band detections (`y<far_court_protect_y=0.33`, `confidence>=far_court_protect_confidence=0.30`); spatial-gated.
12. **Linear interpolation** (line 414–417) — fills gaps where gap ≤ `max_interpolation_gap=5`. Gaps > 5 frames left empty.

**Critical property for contact detection:** frames in a gap of length > 5 carry **no** ball position. Stages 4/5/6 treat a missing ball position as "this candidate cannot be scored at this frame" and fall through. See `detect_contacts` line 2204–2211 which probes `[-1, +1, -2, +2, -3, +3]` offsets to find a ball when the exact frame is empty — so the effective workable window around a frame is roughly ±3f of any ball detection, layered on top of the ±5f interpolation.

**Output:** list of `BallPosition` sorted by frame; all with `confidence>=0.05`. Frames with no post-filter detection are **still missing from the list**. Interpolated positions carry `interpolated_confidence=0.5` (config default, not actually set on inserted points in the WASB preset — the code passes through the surrounding confidences; verify in Phase 2 if it matters to a specific failure mode).

**Handles:** static FPs (structural, WASB-disabled but designed), exit ghosts, false start/tail segments, non-ball disconnected trajectories, trajectory blips via outlier removal, velocity reversals, small occlusion gaps (≤5f), serve-toss re-injection, far-court re-injection.

**Does not handle:**
- Occlusion gaps > 5 frames (left empty).
- **Subtle deflections at contact** — filter is purely geometric; no contact-aware behavior. A deflection that doesn't break the trajectory beyond `max_trajectory_deviation=0.08` or reverse velocity >120° is smoothed across.
- Hand-ball overlap at contact — if WASB drops the ball for a few frames around the hand overlap, interpolation fills it linearly, **erasing** the deflection signal before the candidate generator ever sees it. This is a failure mode to watch for in Phase 2.
- Ball exits frame and re-enters after >30-frame gap.

---

## Stage 2 — Player tracker (`player_tracker.py`, 2542 lines)

**Input:** iterator of BGR frames; optional court ROI polygon (normalized); optional ball positions for downstream filtering; optional `CourtCalibrator`.

**Detector:** `yolo11s` default (`player_tracker.py:41–43`), `imgsz=1280` (+8pp far-court recall vs 640). Confidence 0.15, NMS IoU 0.45. Optional pose variant (`yolo11s-pose`) emits COCO-17 keypoints per detection.

**Tracker:** `boxmot-botsort` default (`player_tracker.py:536`) using fine-tuned OSNet-x1.0 128-dim ReID (`weights/reid/general_reid.pt`). Config `botsort_volleyball.yaml`: `track_high=0.25`, `track_low=0.08`, `new_track=0.35`, `track_buffer=45` (≈1.5 s @ 30fps, fps-scaled), `match_thresh=0.90`, `appearance_thresh=0.30`, `proximity_thresh=0.5`.

**Per-frame pipeline** (`track_frames()` at line 2432, `track_video()` at line 1931):
1. Optional CLAHE contrast on sand backgrounds.
2. ROI mask from court polygon.
3. YOLO infer → filter to `PERSON_CLASS_ID`.
4. OSNet embeddings from each bbox.
5. Tracker update → track ids.
6. Normalize to 0–1 coords.
7. Sport filter: aspect ratio, zone-dependent min area, top-`MAX_DETECTIONS_PER_FRAME=8` by confidence (line 1371).

**Post-processing** (`apply_post_processing()` at line 1414, calls into `player_filter.py`): 13+ passes — stationary-background removal, spatial-consistency, height-swap fix, color-split + relink, appearance link, track-ID stabilization, primary-player analysis, convergence-swap detection, per-frame top-k filtering, team assignment, gap interpolation up to 30f. For contact detection, the consumer-relevant thresholds are:

| Param | Default | File:Line | Controls |
|---|---|---|---|
| `min_bbox_area` | 0.003 | `player_filter.py:105` | Min bbox area fraction |
| `min_presence_rate` | 0.20 | `player_filter.py:120` | Track must appear ≥20% of frames to be primary |
| `ball_proximity_radius` | 0.20 | `player_filter.py:131` | Boost track stability if near ball |
| `max_gap_frames` | 90 | `player_filter.py:154` | Max frames to relink fragments on same id |
| `max_interpolation_gap` | 30 | `player_filter.py:197` | Max gap to linearly interpolate player |

**Output:** `PlayerPosition` per detection (`frame_number, track_id, x, y, width, height, confidence, keypoints`). `PlayerTrackingResult` wraps with `primary_track_ids`, `team_assignments`, `quality_report`. Frames with no detection are gaps; interpolated fills carry conf 0.5 and are inserted by `interpolate_player_gaps` at step 5.

**Handles:** fragmented tracks (relink), same-team height swaps, jersey color splits, appearance-based merges, team assignment, primary-player identification, short gap interpolation.

**Does not handle:** long occlusions > 30f, subtle within-team swaps (see memory §player-attribution-restart), stationary players ambiguous with spectators without safety net.

---

## Stage 3 — MS-TCN++ sequence runtime (`sequence_action_runtime.py`, 337 lines)

**Input:** `ball_positions`, `player_positions`, `court_split_y` (net in image space, from `ContactSequence.net_y`), `frame_count`, optional `team_assignments`, optional `CourtCalibrator`.

**Decision logic — `get_sequence_probs(...)` (line 88–133):**
1. Lazy-load `weights/sequence_action/ms_tcn_production.pt` (`_load_sequence_model`, line 41–85). One-time WARNING if missing.
2. Guard: return `None` if `frame_count < 10` (line 108).
3. `extract_trajectory_features(...)` (line 121–125, in `sequence_action_features.py`) with optional homography enrichment.
4. Forward pass on PyTorch tensor (attention mask = 1), softmax over action dim.
5. Return shape `(NUM_CLASSES, T)` as NumPy, or `None`.

**Classes:** `["bg", "serve", "receive", "set", "attack", "dig", "block"]` (7 total, bg at index 0).

**Cached:** model loaded once per process (lines 38, 50–51); same weights reused for all rallies in a session.

**Downstream tunables (from this module, read at call time):**

| Constant | Default | Used at | Semantics |
|---|---|---|---|
| `DIG_GUARD_RATIO` | 2.5 | `apply_sequence_override:299` | Min set÷dig MS-TCN++ ratio to override GBM dig→set |
| `OVERRIDE_RELATIVE_CONF_K` | 1.2 | `apply_sequence_override:277` | Min (MS-TCN++ argmax ÷ GBM top-1) to override marginal GBM decisions |
| `ATTACK_PRESERVE_RATIO` | 2.5 | `apply_sequence_override:288` | Min (MS-TCN++ argmax ÷ attack conf) to override GBM attack→{set,dig} |
| `SEQ_RECOVERY_TAU` | 0.80 | Read by `detect_contacts` line 2172 via `from ... import SEQ_RECOVERY_TAU` | Documented as rescue threshold; see **§ dead-code warning** below |
| `SEQ_RECOVERY_CLF_FLOOR` | 0.20 | Docs only — see below | Documented as GBM floor for rescue; see below |

**What this stage decides vs. not:**
- Decides per-frame class posterior (used as `seq_max_nonbg` feature in stage 6 and as relabel signal in stage 8 `apply_sequence_override`).
- Does NOT add/remove contacts. Only re-labels via `apply_sequence_override` (line 201–302, mutates `rally_actions.actions[i].action_type`).
- Exempts SERVE actions (line 255) and synthetic actions (line 255).

---

## Stage 4 — Candidate generator (in `detect_contacts` + mirror `extract_candidate_features`)

Two code paths generate identical candidate frames. Production inference uses `detect_contacts` (`contact_detector.py:1845`). Training / corpus / decoder use `extract_candidate_features` (`train_contact_classifier.py:67`). Both must stay in sync — the trainer has a comment on every merge to that effect.

**Inputs:** `ball_positions` (post-filter), `player_positions`, `ContactDetectionConfig`, `frame_count`.

**10 candidate generators, in merge order:**

| # | Generator | Function | Purpose | Key thresholds |
|---|---|---|---|---|
| A | Velocity peaks | `scipy.signal.find_peaks` on smoothed speed | Velocity spike (effect of contact) | `min_peak_velocity=0.008`, `min_peak_prominence=0.003`, `smoothing_window=5`, `min_peak_distance_frames=12` |
| B | Inflection | `_find_inflection_candidates` | Trajectory direction change | `min_inflection_angle_deg=15`, `inflection_check_frames=5` |
| C | Velocity reversal | `_find_velocity_reversal_candidates` | Velocity flips sign | `min_peak_distance_frames=12` |
| D | Deceleration | `_find_deceleration_candidates` | Catches receives/digs that slow the ball | `deceleration_min_speed_before=0.008`, `deceleration_min_drop_ratio=0.3`, `deceleration_window=5` |
| E | Parabolic breakpoint | `_find_parabolic_breakpoints` | Breaks in free-flight parabola (soft touches) | `parabolic_window_frames=12`, `parabolic_stride=3`, `parabolic_min_residual=0.015`, `parabolic_min_prominence=0.008` |
| F | Net crossing | `_find_net_crossing_candidates` | Ball crosses `estimated_net_y` | `min_peak_distance_frames=12` |
| G | Direction-change peak | `_find_direction_change_candidates` | CAUSE of contact (trajectory angle peak, not velocity effect) | `direction_change_candidate_min_deg=25`, `direction_change_candidate_prominence=10` |
| H | Player motion | `_find_player_motion_candidates` | Bbox dy/dh spike near ball, catches blocks/soft touches | **Disabled by default** (`enable_player_motion_candidates=False`): +265 cands / 9 TPs hurts classifier |
| I | Post-serve receive | `_find_post_serve_receive_candidate` | Min player-ball proximity near net crossing after first candidate | `post_serve_search_window=15`, `player_contact_radius=0.15`, `player_search_frames=5` |
| J | Proximity | `_find_proximity_frame` | Shift/add candidate at frame of min player-ball distance | `proximity_search_window=8`, `player_search_frames=5`, `player_contact_radius=0.15` |

**Merge order (`_merge_candidates` preserves the first arg, adds from second arg only if no existing candidate is within `min_peak_distance_frames=12`):**

```
inflection_and_reversal = merge(B, C, 12)
traditional              = merge(A, inflection_and_reversal, 12)
with_deceleration        = merge(traditional, D, 12)
with_parabolic           = merge(with_deceleration, E, 12)
with_net_crossing        = merge(with_parabolic, F, 12)
candidate_frames         = merge(G, with_net_crossing, 12)  # G PRIORITIZED via arg-swap
[optional] merge in H if enabled
[optional] add I if no existing cand within 12f of receive_frame
[optional] trajectory refinement (step 6b) ± 5f, skipping first 60f (serve window)
[optional] merge in J (step 6c)
```

**Refinement passes (operate on the merged list):**

- **6b Trajectory-peak refinement** (`_refine_candidates_to_trajectory_peak`, line 1636–1690): shift each candidate to max `compute_direction_change` within ±`trajectory_refinement_window=5` frames. Serves (`frame - first_frame < serve_window_frames=60`) are skipped because serve trajectories have multiple peaks (toss, contact, arc) and refinement picks the wrong one 62% of the time.
- **6c Proximity refinement** (`_find_proximity_frame`, line 1693–1727): if `proximity_search_window=8`-frame search yields a closer player-ball distance frame, **add** that frame (doesn't replace — original stays).

**Gate at loop entry (`detect_contacts` lines 2191–2251, same in trainer 312–322):**

- Skip `frame - first_frame < warmup_skip_frames=5` (line 2193).
- Skip `frame > frame_count` (post-rally).
- Skip `velocity < min_candidate_velocity=0.003` (line 2250).
- Require ball position at the candidate frame or within ±3f (lines 2204-2211).

**Output:** `candidate_frames: list[int]` — the superset fed to stage 5.

**Handles:** both EFFECT signals (velocity peak/reversal, parabolic break) and CAUSE signals (direction-change peak, proximity). Post-serve structural candidate. Trajectory-peak refinement repositions to direction-change peak (helps classifier features). `min_peak_distance_frames=12` deduplication between generators.

**Does not handle:**
- Contacts on flat trajectories with no velocity spike, no direction change >15°, and no player proximity < 0.15 — i.e., extremely subtle touches. Generator H (player motion) would address this but is disabled for classifier-purity reasons.
- Two real contacts within 12 frames on the same side (`min_peak_distance_frames=12` forces a 0.4 s gap between candidates; pairs closer than this collapse into one at generation time).
- **Cross-side pairs within 12 frames survive** (because the merge is frame-gap-only, no side awareness) but the dedup at stage 7 is what chooses which to keep, governed by `_CROSS_SIDE_MIN_DISTANCE=4`.
- Contacts in a ball-detection hole > 3f (the ±3 fallback at line 2204 can't find a ball).

---

## Stage 5 — Feature extraction (26 features per candidate)

Produced by the same loop as stage 4 (`detect_contacts` lines 2253-2364; trainer path 333-414). `CandidateFeatures` dataclass at `contact_classifier.py:47–152`. Field list in order:

| # | Feature | Source | Notes |
|---|---|---|---|
| 1 | `velocity` | `velocity_lookup[frame]` (smoothed) | Normalized units/frame |
| 2 | `direction_change_deg` | `compute_direction_change(..., direction_check_frames=8)` | 0-180° |
| 3 | `arc_fit_residual` | `residual_by_frame[frame]` | From parabolic-break generator; 0 if not flagged |
| 4 | `acceleration` | `_compute_acceleration(velocities, frame, window=3)` | 2nd derivative |
| 5 | `trajectory_curvature` | `_compute_trajectory_curvature(ball_by_frame, frame, window=5)` | Path curvature |
| 6 | `velocity_y` | `velocities[frame][2]` | Signed, + = downward |
| 7 | `velocity_ratio` | `_compute_velocity_ratio(velocities, frame, window=5)` | speed_after ÷ speed_before |
| 8 | `player_distance` | `_find_nearest_player` search ±`player_search_frames=5` | Image-space; substituted to 1.0 if `inf` (line 93 of `to_array`) |
| 9 | `best_player_max_d_y` | `_compute_candidate_bbox_motion` over ranked cands | Peak bbox dy across all nearby players |
| 10 | `best_player_max_d_height` | ditto | Peak bbox dh across all nearby players |
| 11 | `nearest_player_max_d_y` | ditto, nearest only | dy of nearest player |
| 12 | `nearest_player_max_d_height` | ditto | dh of nearest player |
| 13 | `ball_x` | `ball.x` | Normalized 0-1 |
| 14 | `ball_y` | `ball.y` | Normalized 0-1 |
| 15 | `ball_y_relative_net` | `ball.y - estimated_net_y` | Negative = far side |
| 16 | `is_net_crossing` | `_check_net_crossing(..., window=5)` | Bool → float |
| 17 | `frames_since_last` | `frame - prev_accepted_frame` | Measured from last ACCEPTED contact (not last candidate) |
| 18 | `ball_detection_density` | fraction of ±10 frames with confident ball | **Z-score strongly negative on classifier-rejection FNs** (brief warns: correlation only) |
| 19 | `consecutive_detections` | `_count_consecutive_detections(ball_by_frame, frame)` | Around candidate |
| 20 | `frames_since_rally_start` | `frame - first_frame` | Early = serve |
| 21 | `nearest_active_wrist_velocity_max` | `extract_contact_pose_features_for_nearest` | 0.0 if no keypoints |
| 22 | `nearest_hand_ball_dist_min` | ditto | 0.0 if no keypoints |
| 23 | `nearest_active_arm_extension_change` | ditto | 0.0 if no keypoints |
| 24 | `nearest_pose_confidence_mean` | ditto | 0.0 if no keypoints |
| 25 | `nearest_both_arms_raised` | ditto | 0.0 if no keypoints |
| 26 | `seq_max_nonbg` | `compute_seq_max_nonbg(sequence_probs, frame, window=5)` | Max non-bg MS-TCN++ prob in ±5f. GBM feat importance ≈ 0.297 (historical); brief §what-we-ruled-out: adding this was already done |

**Critical invariants:**
- 7 old `seq_p_*` per-class features were **removed 2026-04-07** because the trainer was always passing zero-filled seq probs (0.0000 importance). MS-TCN++ signal now reaches via feature 26 only, plus stage 8 `apply_sequence_override`.
- The training-time version (`train_contact_classifier.py::extract_candidate_features`) uses **GT-anchored `frames_since_last`**: distance from the last GT-matched candidate within ±5f, not the last accepted inference-time contact. This divergence is annotated at line 294-307 as intentional ("prevents the classifier from learning that close candidates are always noise").

**Output:** `CandidateFeatures` instance per candidate → `.to_array()` → `np.float64` length-26 vector.

**Handles:** provides all features the GBM uses.

**Does not handle:** anything not encoded in these 26 features. Notably no ball-detection history beyond ±10f for density, no cross-rally context, no pose for non-nearest players, no per-class MS-TCN++ breakdown (just the max).

---

## Stage 6 — GBM classifier (`contact_classifier.py`, 384 lines)

**Input:** `list[CandidateFeatures]` (stage 5 output).

**Model:** `sklearn.ensemble.GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, min_samples_leaf=5, subsample=0.8, random_state=42)` (line 222-231). Threshold `0.40` in the `__init__` default but **production threshold is 0.30** — set via CLI (`--threshold 0.30` in `build_eval_reconciled_corpus.py:227`) and `_train_fold` plumbs it through to the classifier instance.

**Predict (line 172-200):**
```
is_trained ? probas = model.predict_proba(X)[:, 1] : return [(False, 0.0)]*N
return [(p >= threshold, p) for p in probas]
```
Feature-count mismatch between code and pickle is auto-padded/truncated (line 186-193) for backward compat.

**Training contract:** LOO-per-video in `scripts/eval_loo_video.py::_train_fold`. Each fold retrains from scratch on all other 67 videos, never on held-out. `positive_weight=1.0` by default (no class weighting). Feature importance available via `feature_importance()`.

**Output:** `list[tuple[bool, float]]` — `(is_validated, confidence)` per candidate.

**Handles:** non-linear feature-space boundary across the 26 features.

**Does not handle (from brief):**
- GT-frame test showed 3.8% rescue rate for the 80 classifier-rejection FNs → classifier **is not fooled by candidate-frame offset**; it's at feature-space ceiling on those cases.
- `seq_max_nonbg` already feature #26 → "add seq as a feature" is not a missing lever.

---

## Stage 7 — Rescue + dedup (`detect_contacts` tail)

### 7a. Per-candidate rescue (`_apply_rescue_branch`, line 1608-1633)

Invoked at line 2367 AFTER `classifier.predict`. Default OFF (`enable_rescue=False` kwarg on `detect_contacts`; the production harness does not set `enable_rescue=True`).

```python
def _apply_rescue_branch(is_validated, gbm_prob, seq_max_nonbg, *, enable_rescue):
    if is_validated:              return True
    if not enable_rescue:         return False
    return gbm_prob < _RESCUE_GBM_CEILING (0.10)
        and seq_max_nonbg >= _RESCUE_SEQ_FLOOR (0.95)
```

**Brief §what-we-ruled-out confirms:** this rule was tested — candidate-level +17pp F1 collapsed to −4.82pp at integration. 67/68 folds regressed. The machinery remains but is off. **Do not retry variants of this pattern** (brief §never-do). Constants at `contact_detector.py:46-47`.

### 7b. Sequential same-player override (line 2403-2414)

Not a dedup — a post-validation player-attribution swap. If nearest player equals previous contact's player AND 2+ candidates exist AND top-2 distance gap < 0.05, swap to the alternative. Applied after validation so classifier feature distribution is preserved.

### 7c. Attribution model override (line 2416-2498)

Pose attributor (preferred) or temporal attributor (fallback) can override `track_id` when ≥2 candidates + confidence ≥ `pose_attribution_min_confidence=0.5` or `temporal_attribution_min_confidence=0.6`. Does not affect whether the contact is kept — only which player it's assigned to. Out-of-scope for FN detection (brief explicitly de-scopes wrong_player).

### 7d. `_deduplicate_contacts` (line 1745-1797)

**Single pass**, not multi-stage. Input: validated `contacts` list. Sort by `confidence` DESC, greedy keep:

```python
for contact in sorted_by_conf_desc:
    for existing in kept:
        if adaptive and sides_known and contact.court_side != existing.court_side:
            effective_min = _CROSS_SIDE_MIN_DISTANCE  # 4
        elif both confidence >= _BOTH_CONFIDENT_FLOOR (999.0 → never fires):
            effective_min = _CROSS_SIDE_MIN_DISTANCE
        else:
            effective_min = min_peak_distance_frames  # 12
        if abs(contact.frame - existing.frame) < effective_min: drop
```

**Key values:**
- `adaptive=True` via `cfg.adaptive_dedup=True` default.
- Same-side min distance: `min_peak_distance_frames=12` (~0.4s @ 30fps).
- Cross-side min distance: `_CROSS_SIDE_MIN_DISTANCE=4` (~0.13s @ 30fps; attack→block/dig).
- "Both confident" bypass: **disabled** — `_BOTH_CONFIDENT_FLOOR=999.0` means the condition is never true (line 1739-1742 has a note: a real 0.50/0.80 threshold added 195+ FPs in a test).

**Handles:** same-side duplicates within 12f (proximity/velocity-peak pair on the same action); cross-side pairs ≥4f apart preserved (attack→block/dig rescue).

**Does not handle:**
- Same-side real pairs < 12f apart (impossible physically on same side, per config comment, but in practice two GT can be that close and one will be lost if same-side — brief states 2–10f pairs are real).
- Cross-side pairs < 4f apart (merges into one).
- Ordering by `confidence` DESC means a low-confidence real contact can be evicted by a high-confidence FP in a nearby frame.
- Per brief measurement: 84 non-block FNs lost here, and **0 have a surviving contact within tol of GT** — so the loss is not "wrong one kept, right one dropped" but rather "all of this GT's candidates were eliminated." That pattern is different from the block-steal mechanism; Phase 2 must observe it directly.

### 7e. Dead code — `_has_sequence_support`

`detect_contacts` defines `_has_sequence_support(frame, window=5)` (line 2174–2182) but **has no callers in the file**. The docstring on `detect_contacts` lines 1888–1897 describes a two-signal agreement gate using `SEQ_RECOVERY_TAU`/`SEQ_RECOVERY_CLF_FLOOR` — that gate is no longer implemented at contact-detection time. The seq signal reaches the classifier only via feature #26 (`seq_max_nonbg`) and reaches action relabeling via `apply_sequence_override` (stage 8). Flagging for Phase 2 so nobody chases a ghost lever.

**Output of stage 7:** final `ContactSequence(contacts, net_y, rally_start_frame, ball_positions, player_positions)` — fed to stages 8, 9, and 10.

---

## Stage 8 — Action classifier (`classify_rally_actions`, `action_classifier.py:3364–3553`)

**Input (most consequential):**
- `contact_sequence: ContactSequence` — contacts from stage 7 (fixed).
- `team_assignments` / `match_team_assignments`, `track_to_player`, `calibrator`.
- `sequence_probs` — MS-TCN++ probs for `apply_sequence_override`.
- `decoder_contacts` — **label-only** overlay from stage 9 (tol=3f).
- `use_classifier=True` — loads action GBM (dig/set/attack) or rule-based fallback.

**Decision flow (line 3391-3551):**
1. `classify_rally()` — per-contact initial action type + serve detection. 4-pass serve heuristic (position → arc → baseline+velocity → first-contact fallback), optional GBM arbitration on Pass 3.
2. `repair_action_sequence(Rule 1 only)` — consecutive receive/dig → set.
3. `viterbi_decode_actions()` — sequence-level smoothing on action labels (not contact frames).
4. `validate_action_sequence()` — constraint log-only.
5. `assign_court_side_from_teams()` — overwrite court_side from `match_team_assignments`.
6. `reattribute_players()` — server exclusion + server-seeded team chain.
7. Formation-based serving team (multi-feature logistic regression, 92.5-93.3% LOO-video CV on 620 GT rallies, 82% coverage when ball available).
8. `apply_sequence_override()` — MS-TCN++ relabels non-serve actions with `DIG_GUARD_RATIO/OVERRIDE_RELATIVE_CONF_K/ATTACK_PRESERVE_RATIO` guards.
9. `apply_decoder_labels()` — stage 9 overlay, label-only within ±3f.

**Contact-level impact:**
- Adds: **only** synthetic serves via `_make_synthetic_serve` (line 1886-1893) when the first real contact is phantom-rejected. Synthetic serve gets `is_synthetic=True`, inherits from or precedes the first real contact frame. Does not create a new real-contact frame — it prepends a separate synthetic action.
- Drops: `pre_serve_fp_filter` (line 1621-1622) skips any real contact that occurs before the identified serve frame (not appended to `actions`, though the Contact object in `contact_sequence` still exists in memory).
- Otherwise does not remove contacts.

**Phantom-serve rejection (lines 1670-1838):** checks whether "first contact" is really a serve.

- `_is_ball_on_serve_side` — rejects if ball clearly on wrong side at contact.
- Median of first-5 pre-contact positions on opposite side of net from contact → phantom.
- Ball not moving toward net → phantom (with rescue when WASB missed the serve trajectory).
- GBM arbitration can rescue phantom (line 1819) or reject non-phantom (line 1827, requires GBM ≥0.7 non-serve).

**Output:** `RallyActions(actions: list[ClassifiedAction], rally_id, team_assignments, formation_serving_team)`. `ClassifiedAction` adds `action_type`, `court_side`, `confidence`, `is_synthetic`, `team`, zones.

**Handles:** action-type re-labeling with Viterbi + MS-TCN++ guards; serve detection with 4 heuristics; synthetic serve for phantom cases.

**Does not handle:** adding missed real contacts; fixing upstream (stage 7) losses. A GT contact that made it through stages 1-7 but was labeled incorrectly is a **wrong_action** error, not a FN (brief de-scopes for this review).

**The 4 FNs attributed to this stage** in the brief are almost certainly pre-serve-FP filter drops (real GT contact before the identified serve frame gets filtered out). Phase 2 confirms.

---

## Stage 9 — Decoder overlay (trio: `decoder_runtime.py` + `decoder_overlay.py` + `candidate_decoder.py`)

**Input (`run_decoder_over_rally`, `decoder_runtime.py:97-184`):** same ball/player/seq inputs + the trained `ContactClassifier` + `ContactDetectionConfig`. Internally:
1. Build `_RallyDataShim` (`contact_detector.py:2554-2593`) to satisfy `extract_candidate_features`'s `RallyData` interface.
2. Call `extract_candidate_features` (stage 4+5) → `feats_list`, `cand_frames`.
3. Run GBM `predict_proba` on `feats_list` → `gbm_probs` (raw, not thresholded).
4. Build `DecoderCandidateFeatures` per candidate: `frame`, `gbm_contact_prob`, `action_probs` (renormalized MS-TCN++ over positive classes), `team` (near/far/unknown via `infer_team_from_player_track`: tracks 1-2 = near, 3-4 = far).
5. `decode_rally(...)` → Viterbi MAP.

**Viterbi (`candidate_decoder.py:159-284`):**

- Emission: `log P_gbm_contact(i) + log P_action(action|frame_i)` for each (candidate, action); skip emission = `log(1 - P_gbm_contact(i)) - skip_penalty`.
- Transition: learned `TransitionMatrix` keyed by `(prev_action, gap_bucket, cross_team)`. Buckets: `[(0,5),(6,15),(16,40),(41,120),(121,10000)]`. Default matrix at `data/contact_transitions.json` (Phase CRF-0).
- Cross/same/unknown from team labels on candidates.
- Production defaults: `skip_penalty=1.0`, `emission_floor=0.02`, `min_accept_prob=0.0`.
- DP state `best[i, a]`; backtrace yields MAP sequence in frame order.

**Overlay (`decoder_overlay.py:48-102`):**

- Greedy: for each decoder contact in frame order, find unmatched `ClassifiedAction` with smallest `|frame gap|` ≤ `tol_frames=3`. Ties broken by list order.
- If matched, swap `action_type` only via `dataclasses.replace` — all other fields (`playerTrackId`, `court_side`, `team`, `confidence`, zones) preserved byte-for-byte.
- Decoder contacts with no matching detected action are **dropped** (module docstring line 8-11 explicit: "we do not add contacts").

**Output:** `RallyActions` (re-labeled) + `OverlayStat(n_decoder_contacts, n_detected_contacts, n_matched, n_label_swapped)`.

**Handles:** improves action-type F1 (`Action Acc +2.64pp` @ zero F1 regression per brief §track-ID stability / decoder-integration memo).

**Does not handle:** the decoder cannot rescue a GT contact that stage 7 dedup dropped, because `apply_decoder_labels` never adds actions — it's a relabel-only overlay. This is the brief's "+17pp candidate F1 → −4.82pp production F1" lesson: candidate-level gains from the decoder don't translate to contacts unless a separate integration path also accepts (which is outside the current scope, and the previous session's attempt at that was NO-GO).

---

## Stage 10 — Matching (eval-only, `eval_action_detection.py`)

**Input:** `gt_labels: list[GtLabel]` from `rally.gt_labels`, `pred_actions: list[dict]` (from `ClassifiedAction.to_dict()`).

**Pass 1 — `match_contacts` (line 472-583):**
- Builds cost matrix `cost[i, j] = |gt[i].frame - pred[j].frame|` capped at `tolerance+1`.
- `scipy.optimize.linear_sum_assignment` — optimal bipartite Hungarian assignment.
- Keep pair iff `cost[gi][pi] <= tolerance`.
- `tolerance` is caller-supplied; the harness uses `tol_frames = max(1, round(fps * 233 / 1000))` = **7 frames @ 30fps** (not the library default `3`).
- Unmatched preds → `unmatched`, unmatched GT → FN (`pred_frame=None` in MatchResult).

**Pass 2 — `_match_synthetic_serves` (line 586-655):**
- For GT serves still unmatched (pass-1 FN) AND action type is "serve", try matching synthetic serves (`isSynthetic=True` in pred) with tolerance ≈ 1 s (caller-supplied, usually `synth_tolerance = fps`).
- Each synthetic serve matches at most one GT serve.
- Updates the MatchResult in place.

**Output:** `(list[MatchResult], list[unmatched_preds])`. MatchResult has `gt_frame, gt_action, pred_frame, pred_action, player_correct, player_evaluable, court_side_correct`.

**Handles:**
- Optimal assignment under frame-distance cost avoids the greedy pitfall of "first GT grabs the nearest pred, second GT is orphaned" when two GT frames are near the same pred.
- Pass-2 synthetic rescue for serves missed by real candidates (off-screen server, far-side occlusion — all in brief §known-realities).

**Does not handle:**
- Matching-steal across actions (brief: 3 non-block FNs here): two GT frames compete for the same pred; Hungarian assigns to the other. This is rare (brief: 3) but real.
- Wrong-action same-frame pairs (brief scope defers to action classifier review).

---

## Cross-stage invariants relevant to FN analysis

1. **"Stage where it's first lost" ≠ "root cause."** Brief makes this explicit. A candidate absent at stage 7 dedup may have been in a bad state since stage 4 (refined to the wrong frame), stage 5 (features compromised by the ball gap), or stage 6 (classifier boundary).

2. **Ball-position availability is a gating precondition at every stage 4-7 operation.** Any candidate without a ball position in ±3f is silently skipped at the `detect_contacts` loop head (line 2204–2211). This means a gap at the actual contact frame cascades: no candidate → no features → no classifier score → no contact.

3. **Interpolation is linear.** A real deflection inside a ≤5f WASB gap is smoothed across before stage 4 sees the positions. This is invisible to any stage 4+ mechanism.

4. **`min_peak_distance_frames=12` operates in three places:**
   - Between candidate generators (merge phase 4).
   - On velocity-peak `find_peaks` (stage 4 A generator).
   - Dedup same-side (stage 7d).
   And cross-side uses `_CROSS_SIDE_MIN_DISTANCE=4`. Two GT contacts within 12f same-side collapse at multiple points; neither the candidate pipeline nor dedup can separate them without side knowledge.

5. **Trajectory refinement skips serves** (first 60 frames). Serves come from generators A-G at the velocity peak or first net crossing, not refined to direction-change peak. This is intentional (multiple peaks in serve toss) but means serve candidates can sit farther from the GT frame than non-serve candidates; matching tolerance (7f) absorbs most of this.

6. **Training ↔ inference divergence on `frames_since_last`.** Trainer measures from last GT-matched candidate (±5f); inference measures from last accepted contact. Deliberate per line 294-307 comment. If the classifier learned "close candidates are always noise" from inference-style features, it would reject legitimate attack→block pairs.

7. **Decoder (stage 9) is label-only.** Cannot rescue a contact that stages 1-7 dropped. The +2.64pp Action Acc improvement comes from re-labeling accepted contacts; no FN is recovered by stage 9.

8. **Action classifier (stage 8) is label-only except for synthetic serves.** Cannot rescue a contact that stages 1-7 dropped. Synthetic serves are new actions, not rescued real contacts — they populate GT serve slots in the matcher's pass 2.

9. **Pre-serve FP filter (stage 8) is one of the 4 FN drop paths at action-labeling.** Contacts before the identified serve frame are dropped regardless of classifier confidence.

---

## Appendix: all `ContactDetectionConfig` tunables (line 127-255)

Full list in order, for cross-reference during Phase 4 quantification. Defaults shown.

| Param | Default | Stage |
|---|---|---|
| `min_peak_velocity` | 0.008 | 4A |
| `min_peak_prominence` | 0.003 | 4A |
| `smoothing_window` | 5 | pre-4 |
| `min_peak_distance_frames` | 12 | 4 merge, 4A, 7d same-side |
| `min_direction_change_deg` | 20.0 | 4B (ish; legacy) |
| `direction_check_frames` | 8 | 4G, 6b |
| `enable_inflection_detection` | True | 4B |
| `min_inflection_angle_deg` | 15.0 | 4B |
| `inflection_check_frames` | 5 | 4B |
| `enable_noise_filter` | True | pre-4 |
| `noise_spike_max_jump` | 0.20 | pre-4 |
| `player_contact_radius` | 0.15 | 4I, 4J, 6 fallback |
| `player_search_frames` | 5 | 4I, 4J, 6 |
| `player_candidate_search_frames` | 15 | 6 (bbox-motion ranking) |
| `high_velocity_threshold` | 0.025 | 6 fallback (no classifier) |
| `warmup_skip_frames` | 5 | loop gate |
| `min_candidate_velocity` | 0.003 | loop gate |
| `enable_parabolic_detection` | True | 4E |
| `parabolic_window_frames` | 12 | 4E |
| `parabolic_stride` | 3 | 4E |
| `parabolic_min_residual` | 0.015 | 4E |
| `parabolic_min_prominence` | 0.008 | 4E |
| `enable_deceleration_detection` | True | 4D |
| `deceleration_min_speed_before` | 0.008 | 4D |
| `deceleration_min_drop_ratio` | 0.3 | 4D |
| `deceleration_window` | 5 | 4D |
| `enable_post_serve_receive` | True | 4I |
| `post_serve_search_window` | 15 | 4I |
| `enable_proximity_candidates` | True | 4J |
| `proximity_search_window` | 8 | 4J |
| `enable_trajectory_refinement` | True | 6b |
| `trajectory_refinement_window` | 5 | 6b |
| `enable_direction_change_candidates` | True | 4G |
| `direction_change_candidate_min_deg` | 25.0 | 4G |
| `direction_change_candidate_prominence` | 10.0 | 4G |
| `baseline_y_near` | 0.82 | stage 8 fallback baselines |
| `baseline_y_far` | 0.18 | stage 8 fallback baselines |
| `serve_window_frames` | 60 | 6b skip, stage 8 serve detection |
| `enable_player_motion_candidates` | False | 4H (default OFF) |
| `player_motion_min_d_y` | 0.015 | 4H |
| `player_motion_min_d_height` | 0.015 | 4H |
| `player_motion_max_ball_distance` | 0.20 | 4H |
| `use_temporal_attribution` | True | 7c |
| `temporal_attribution_min_confidence` | 0.6 | 7c |
| `use_pose_attribution` | True | 7c |
| `pose_attribution_min_confidence` | 0.5 | 7c |
| `adaptive_dedup` | True | 7d |
| `enable_sequence_recovery` | True | 7e (dead, but flag read) |

Module-level constants:

| Name | Value | Stage |
|---|---|---|
| `_CONFIDENCE_THRESHOLD` | 0.3 | all; ball confidence gate for `ball_by_frame` |
| `_RESCUE_GBM_CEILING` | 0.10 | 7a (off by default) |
| `_RESCUE_SEQ_FLOOR` | 0.95 | 7a (off by default) |
| `_CROSS_SIDE_MIN_DISTANCE` | 4 | 7d |
| `_BOTH_CONFIDENT_FLOOR` | 999.0 | 7d — effectively disabled |
| `SEQ_RECOVERY_TAU` | 0.80 | imported by 7e (dead) |
| `SEQ_RECOVERY_CLF_FLOOR` | 0.20 | referenced in docstrings only; no live caller |
