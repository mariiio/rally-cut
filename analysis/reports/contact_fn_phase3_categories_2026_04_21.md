# Contact FN failure-mode categorization — 2026-04-21

**Phase 3 deliverable** of the contact-detection full review (brief: `docs/superpowers/briefs/2026-04-21-contact-detection-full-review.md`).

**Provenance.** Categories are derived from the Phase 2 sample of 20 FNs (`analysis/reports/contact_fn_visual_log_2026_04_21.md`). Each category has an **observable signature** at the pipeline-data level (reproducible in Phase 4) plus a **behavioral description** (what a human sees in the clip). Hypothesized mechanisms are flagged as hypotheses — they are candidates for Phase 4 verification, not conclusions.

**Note on scope.** Categories are grouped by the stage at which the contact is first lost (per the brief's coarse table). `dedup_survived=False` is treated as its own super-category because all 9 such cases showed a single consistent pattern distinct from the classifier-rejection cases.

---

## Summary table

| # | Category | Observed in | Observable signature | Stage |
|---|---|---|---|---|
| 1 | Interp-erases-deflection | C17 | `ball_gap_frames` 1-5 AND `nearest_cand_dist > min_peak_distance_frames=12` AND no gen-10 candidate at GT | Candidate gen |
| 2 | Kinematic-underreports-visual | C01, C02, C05, C09, C18 | `direction_change_deg ≤ 10°` AND `player_distance ≤ 0.10` AND `seq_peak_nonbg ≥ 0.85` AND `gbm < 0.30` | Classifier |
| 3 | Kinematic-maximal-GBM-rejects-deeply | C04, C06 | `direction_change_deg ≥ 170°` AND `gbm ≤ 0.05` AND correct `seq_peak_action` | Classifier |
| 4 | Heavy-dual-occlusion | C10, C12 | `player_distance = inf` AND `ball_gap_frames ≥ 1` | Classifier / dedup |
| 5 | Serve-ball-dropout-shifted-candidate | C14 | `gt_action=serve` AND `ball_gap_frames ≥ 5` AND `nearest_candidate_distance > 7` (matcher tol) | Ball tracker |
| 6 | Ball-gap-exceeds-interp | C16 | `ball_gap_frames > 5` AND `nearest_cand_dist ≥ 10` | Ball tracker |
| 7 | **Confident-accept-killed-by-dedup** | **C03, C07, C08, C11, C12, C13, C15, C19, C20** | `classifier_accepted=True` AND `dedup_survived=False` AND `accepted_in_window_nearest_gbm ≥ 0.30` AND `rally_actions_in_window = []` | Dedup |

Overlaps: C12 is both Category 4 (dual-occlusion) and Category 7 (dedup-kill). All other cases are in exactly one category.

Total accounted for: 20/20 (9 + 11; C12 double-counted).

---

## Category 1 — Interp-erases-deflection

**Observable signature.**
- `ball_tracked = True` (stage 1b interp kept positions continuous)
- `ball_gap_frames ∈ [1, 5]` (within `max_interpolation_gap=5`)
- `candidate_generated = False` (no candidate fires at or near GT)
- `nearest_cand_dist > min_peak_distance_frames = 12` (closest candidate is too far to matter)

**Behavioral description.** Ball is visible overall but WASB misses ~1-5 frames around contact. Stage 1b linearly interpolates the gap. The reconstructed trajectory is a straight line through the contact moment, so direction change = 0° — below every candidate generator's firing threshold (velocity peaks need `min_peak_prominence=0.003` in a smoothed signal; inflection needs `min_inflection_angle_deg=15°`; direction-change-peak needs `25°`; parabolic fit a straight line has zero residual).

**Hypothesized mechanism (requires Phase 4 verification).** Linear interpolation is contact-unaware. A real deflection inside the interp window is erased before any stage-4 generator sees the positions.

**Phase 4 quantification targets.**
- Of 18 `no_candidate` + 4 `rejected_by_gates` + 36 `ball_dropout` FNs (per corpus `fn_subcategory` tally), how many have `ball_gap_frames ∈ [1, 5]` at GT?
- For those, what's the distribution of `nearest_cand_dist`? (If most are > 12, Category 1 is the dominant sub-population.)
- How does the CANDIDATE-level score compare for matched-TP vs unmatched-FN at identical ball_gap_frames? (Establishes whether the interp-smoothing hypothesis is the mechanism, vs other correlated factors.)

---

## Category 2 — Kinematic-underreports-visual

**Observable signature.**
- `candidate_generated = True` (candidate fires within ≤3 frames of GT)
- `ball_gap_frames = 0` (no WASB gap at the candidate frame itself — distinct from Category 1)
- `direction_change_deg ≤ 10°` (feature measures flat)
- `player_distance ≤ 0.10` (player well within `player_contact_radius=0.15`)
- `seq_peak_nonbg_within_5f ≥ 0.85` (MS-TCN++ strongly endorses)
- `gbm < 0.30` (classifier rejects, often deeply — 0.011 to 0.252 in sample)
- `velocity ∈ [0.009, 0.018]` (low-to-mid range)

**Behavioral description.** User sees a clear trajectory change at contact; player is visibly in contact position; no occlusion. But `compute_direction_change` at the candidate frame (±8f check window) measures near-zero angle. The classifier's `direction_change_deg` feature, combined with low-to-mid velocity, dominates the other features (even strong `seq_max_nonbg` and close `player_distance`).

**Hypothesized mechanisms (requires Phase 4 verification).**
- **H2a:** WASB detections immediately around contact are sparse (dropped frames BELOW `_CONFIDENCE_THRESHOLD=0.3`, even if `ball_gap_frames` reports 0 at the candidate frame itself — the 0.3 floor is applied before the gap is measured).
- **H2b:** The ±8f check window for `compute_direction_change` is too wide — spans smoothed tracker positions that average down the contact-moment angle.
- **H2c:** The candidate was generated 1-3 frames off the actual contact (feature computed at candidate frame, not GT frame) and the off-by-2 frames matter for sharp angular signals. Brief's GT-frame test rescued only 3.8% → this is partially ruled out but not fully.

**Phase 4 quantification targets.**
- For each Category 2 candidate, inspect raw WASB confidences at the candidate frame ± 8f. Count how many are below 0.3 (the `ball_by_frame` inclusion floor).
- Re-compute `compute_direction_change` at windows [±3, ±5, ±8, ±12] — does a narrower window report the actual angle?
- Count across 80 classifier-rejection FNs (brief stage total, excl. block) how many match Category 2's observable signature.

---

## Category 3 — Kinematic-maximal-GBM-rejects-deeply

**Observable signature.**
- `candidate_generated = True` at `nearest_candidate_distance ≤ 2`
- `direction_change_deg ≥ 170°` (near-complete ball reversal — the ball bounced back)
- `player_distance ≤ 0.10`
- `seq_peak_nonbg_within_5f ≥ 0.90` with correct `seq_peak_action`
- `gbm ≤ 0.05` (deep rejection despite all those endorsements)
- `velocity` is LOW: 0.006-0.011 (below `high_velocity_threshold=0.025`)
- `ball_gap_frames = 0`

**Behavioral description.** User observes a ball that comes in fast and comes back out fast (dig/defensive contact). Ball briefly stops at the player's hands (low velocity AT contact frame) then accelerates away. Pipeline measures the trajectory correctly (170°+ direction change), player is close, MS-TCN++ agrees — yet GBM rejects at the most extreme kinematic signal in our sample.

**Hypothesized mechanisms (requires Phase 4 verification).**
- **H3a:** The classifier's `velocity` feature penalizes low-velocity candidates even when `direction_change_deg` is maximal. This is a learned feature-interaction that correlates low-velocity with tracker-noise FPs during training.
- **H3b:** Pose features (`nearest_pose_confidence_mean`, `nearest_hand_ball_dist_min`, etc.) default to 0.0 when YOLO-Pose keypoints aren't available (or are degraded by hand-ball overlap at contact — user observed this in C04). Five of 26 features at zero can push the GBM negative.
- **H3c:** `ball_detection_density` in the ±10f window — if the ball detection was marginal (confidence near 0.3) right around the contact frame, density drops. Brief §traps-to-avoid flags this feature with strong negative z-score on cls-rej FNs.

**Phase 4 quantification targets.**
- For each Category 3 candidate, list the 26 feature values and compute per-feature contribution to the log-odds via `model.staged_decision_function`.
- Count how many of the 80 classifier-rejection FNs (brief) match Category 3's signature (`dir_change ≥ 170°` AND `gbm ≤ 0.05`).
- Of those, how many have pose confidence 0? Ball density < 0.5?

---

## Category 4 — Heavy-dual-occlusion

**Observable signature.**
- `ball_tracked = True` AND `player_tracked = True` (both exist in rally)
- `ball_gap_frames ≥ 1` AND `player_distance = inf` (both lost AT contact frame)
- `direction_change_deg = 0°` (flat because no pre/post positions to compute from)
- `seq_peak_nonbg_within_5f ≥ 0.95` with correct `seq_peak_action`

**Behavioral description.** Far-side receiver is physically occluded by the near-side server's body (standard volleyball geometry). Both ball and receiving player are hidden from camera for 1-2 frames at contact. MS-TCN++ still identifies the action via its multi-frame temporal window. C10 had classifier reject (gbm=0.225 marginal). C12 had classifier accept (gbm=0.414) then dedup kill.

**Hypothesized mechanisms.**
- Category 4 outcomes depend on whether other features (seq, arc_residual, consecutive_detections) compensate for the `player_distance=1.0` default. When they don't, classifier rejects (C10). When they do, dedup usually eliminates anyway (C12) — aligning with Category 7.

**Phase 4 quantification targets.**
- Count FNs with `player_distance=inf` across all 238 non-block FNs. Sub-bucket by `ball_gap_frames` (0/1-2/3-5/>5).
- Of classifier-accepted + dedup-killed FNs (Category 7), how many ALSO have `player_distance=inf` (Category 4 ∩ 7)?

---

## Category 5 — Serve-ball-dropout-shifted-candidate

**Observable signature.**
- `gt_action = serve`
- `ball_tracked = False` at attribution (but ball detected elsewhere in rally)
- `ball_gap_frames ≥ 5` at GT
- `candidate_generated = True` but at `nearest_candidate_distance ≥ 7` (outside Hungarian matcher's 7-frame tolerance)
- `classifier_accepted = True` with high gbm

**Behavioral description.** WASB starts tracking the ball several frames after the serve contact, not at contact. A candidate is generated at the first tracked frame (≥7 frames after GT), classifier confidently endorses it as a serve, but the Hungarian matcher cannot pair the pred with GT within 7 frames. The synthetic-serve pass 2 rescue (1-second tolerance) should fire here but apparently did not for C14.

**Hypothesized mechanisms.**
- **H5a:** `estimate_net_position` or the serve-window-frames gate in the phantom-serve logic rejects the serve before it reaches action_classifier, preventing synthetic generation.
- **H5b:** The accepted candidate is dedup-killed, and action_classifier never sees it as a serve input.

**Phase 4 quantification targets.**
- Of 125 serve FNs (corpus), how many match the signature `ball_gap_frames ≥ 5` AND `nearest_cand_dist 7-30`? (expected: many — brief's "ball tracker → serves 23" sub-count is consistent)
- For each, inspect the `actions_json` (if present) for a synthetic serve and its distance to GT. If a synthetic exists but is > 1s from GT, rescue pass 2 couldn't match.

---

## Category 6 — Ball-gap-exceeds-interp

**Observable signature.**
- `ball_tracked = False`
- `ball_gap_frames > 5` (exceeds `max_interpolation_gap`)
- `candidate_generated = False`
- `nearest_cand_dist ≥ 10` (no candidate anywhere nearby)
- `seq_peak_nonbg_within_5f` still ≥ 0.90 with correct class

**Behavioral description.** User may describe "ball tracking looks lost." WASB genuinely missed 6+ frames around contact. Stage 1b can't interp more than 5. The candidate-loop head's ±3f ball-fallback (contact_detector.py:2204-2211) also can't reach. The entire downstream pipeline is starved.

**Hypothesized mechanisms.**
- **H6a:** Low-confidence WASB detections around contact — ball is visible to a human but below `_CONFIDENCE_THRESHOLD=0.3`.
- **H6b:** True ball occlusion (player body blocks line of sight for 6-9 frames).

**Phase 4 quantification targets.**
- Count FNs with `ball_gap_frames > 5`. Check raw WASB confidences in the gap — are they < 0.3 but > 0 (distinguishes H6a from H6b)?
- Of 36 `ball_tracked → FN` + 18 `candidate_generated → FN` (brief = 54 non-block), how many match Category 6 signature?

---

## Category 7 — Confident-accept-killed-by-dedup ⭐ PRIMARY TARGET

**9 of 9 dedup-elimination cases in our sample.**

**Observable signature.**
- `classifier_accepted = True` (the accepted one has `accepted_in_window_nearest_gbm ≥ 0.30`)
- `dedup_survived = False`
- `rally_actions_in_window = []` (brief's already-measured property: dedup winner is absent from final pred within tolerance)

**Quantitative scatter in our 9 cases:**
| Case | gt_action | gbm accepted | dir_change | player_dist | ball_gap | Notes |
|---|---|---|---|---|---|---|
| C03 | attack | 0.365 | 77.3° | 0.030 | 0 | gt_track=27 non-canonical |
| C07 | dig | 0.350 | 29.6° | 0.067 | 0 | user: attack before (pair) |
| C08 | dig | 0.670 | 5.0° | 0.172 | 0 | user: attack before (pair) |
| C11 | receive | 0.688 | 47.7° | 0.015 | 0 | 60fps video |
| C12 | receive | 0.414 | 0.0° | inf | 2 | also Category 4 |
| C13 | receive | 0.866 | 147.6° | 0.013 | 0 | highest-proximity case |
| C15 | serve | 0.965 | 126.3° | 0.116 | 0 | highest gbm in sample |
| C19 | set | 0.894 | 5.7° | 0.028 | 0 | user: suspected track swaps |
| C20 | set | 0.799 | 2.5° | 0.123 | 0 | ball enters/exits frame |

**Mean gbm = 0.647, min 0.350, max 0.965.** Every case had seq_peak_action correct (or at least strongly non-bg for one). Kinematic, proximity, and seq signals span their full ranges — the only common factor is the high-confidence acceptance followed by dedup elimination.

**Behavioral description.** A contact that was visually clear (C03/C07/C08/C13/C19/C20) or occluded-but-endorsed-by-seq (C11/C12/C15) got generated, feature-extracted, classifier-accepted with high confidence, then eliminated by `_deduplicate_contacts` against a neighboring candidate that itself doesn't appear in the final pred within 7 frames of GT.

**Hypothesized mechanisms (NOT verified; Phase 4 must identify for each case).**
- **H7a — Attack→dig/block pair collapse (C07, C08).** Attack candidate wins dedup via higher confidence; dig 5-12 frames later gets killed even though cross-side at the physical level. `court_side` resolution may have put both on same side → same-side `min_distance=12` triggers.
- **H7b — Pre-serve pseudo-peak (C15).** Serve-toss moment produces a velocity peak within 12 frames of the actual serve contact. Toss-peak wins dedup (higher confidence, earlier in sort), serve contact is killed.
- **H7c — Non-canonical track ID / court_side unknown (C03).** `gt_player_track_id=27` outside 1-4 → `team_assignments` lookup misses → `court_side` falls through to "near" or "far" via player_y with potentially wrong resolution → adaptive dedup uses same-side distance when cross-side was correct.
- **H7d — Track-ID switch mid-rally (C19 per user).** Fast player movement + occlusion → track id swap → "sequential same-player override" mis-fires or court_side flips.
- **H7e — Generic "confidence-DESC sort causes eviction" for set→attack pairs.** Set candidate at high confidence, attack within 6-10 frames same-side, set ties of keeps attack, or attack gets killed — neither survives to final pred within 7f of either GT.

**Phase 4 quantification targets (URGENT — this category has 9/20 of our sample and ~84 non-block FNs per brief).**
- For each of the 9 sampled dedup-kills: inspect the full candidate list in the rally (via re-running `detect_contacts` with instrumentation), identify the specific candidate that won dedup, its frame/gbm/court_side/action, and trace why it's absent from final pred within 7 frames of GT.
- Tabulate: does the "dedup winner" land in another GT's tolerance window (wrong_action, not FN), OR get filtered later (pre-serve FP, phantom rejection), OR end up outside matching tolerance?
- Compute: for each dedup-elimination FN, the frame gap between the accepted candidate and the nearest accepted-and-surviving candidate in the rally. Compare same-side-gap distribution vs cross-side-gap distribution.
- Check: how often is `court_side` the same for the FN candidate and its killer? If predominantly same-side, the hypothesis "court_side resolution forces same-side=12 where cross-side=4 was correct" is testable.

**This is the single highest-impact observation from Phase 2.** The classifier is doing the right thing on nine out of nine cases that dedup then ruins.

---

## Stage-coverage sanity check vs brief's 238 non-block FN total

Extrapolating from our 20-case sample (non-representative, just a sanity check — Phase 4 replaces this with real counts):

| Category | Our n | % of 20 | If proportional → 238 |
|---|---|---|---|
| 1 Interp-erases | 1 | 5% | ~12 |
| 2 Kinematic-underreports | 5 | 25% | ~60 |
| 3 Kinematic-maximal | 2 | 10% | ~24 |
| 4 Dual-occlusion | 2 | 10% | ~24 |
| 5 Serve-dropout | 1 | 5% | ~12 |
| 6 Gap-exceeds-interp | 1 | 5% | ~12 |
| 7 Dedup-kill | 9 | 45% | ~107 |

Phase 4 will compute real counts from the full corpus + attribution join. The above is directional — expect the real distribution to shift, particularly Category 7 which we expect to be a large share of the 84 `dedup_survived=False` FNs but not necessarily 107.

---

## What Phase 4 must produce

Per brief §Phase 4: "For each category identified in Phase 3, write a targeted detector… Count each category across the 238 non-block FNs. This converts the 20-case visual observations into a corpus-wide quantified breakdown."

Deliverable: a table of `failure-mode → count → examples`, with the observable-signature-based detectors specified above. Must be run against `fn_stage_attribution.jsonl` ∪ `corpus_eval_reconciled.jsonl` (joined), with each FN assigned to ≤1 primary category plus (if applicable) Category 7 as a cross-cutting tag.

---

## Meta-observations carried forward (data-quality + scope)

- **Corpus `fn_subcategory` taxonomy is unreliable.** It mislabels Category 7 cases as `rejected_by_classifier` because it reads the nearest-rejected candidate, not the actually-accepted one. Attribution's `lost_at_stage` is authoritative and must be used for Phase 4 quantification.
- **C11's misaligned overlays were stale tracking, not a renderer bug** (user confirmed after a rally retrack fixed it). The 60fps renderer concern is downgraded; C02/C09/C11 visual observations are reliable.
- **User's "template-answer" pattern on C02/C06/C09/C13/C15** (identical "very clear in all signals" text) was mostly accurate-to-pipeline for C06/C09/C13 but inconsistent with C15's pipeline state. Phase 4 should rely on the numerical features, not user qualitative descriptions, for category assignment on those cases.
