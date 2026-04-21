# Contact FN visual inspection log — 2026-04-21

Phase 2 deliverable of the contact-detection full review (brief: `docs/superpowers/briefs/2026-04-21-contact-detection-full-review.md`).

**Discipline (from brief § Phase 2):** answer the 5 questions for each case BEFORE opening the diagnostic data. Watch the clip first.

**Sample provenance:** seed=42 stratified draw from `analysis/outputs/fn_stage_attribution.jsonl` (238 non-block FNs), proportional to `lost_at_stage` shares within each action class. Serves stratified 1 near / 1 far via gt_player_track_id. Generator: `analysis/scripts/sample_phase2_fns.py`. Machine copy: `analysis/outputs/phase2_sample_2026_04_21.jsonl`.

## Case index

| Case | Action | Stage lost | Rally | GT frame | GT trk | Clip |
|---|---|---|---|---|---|---|
| C01 | attack | candidate_generated | `512f9467` | 169 | 2 | `512f9467-8bd9-465f-b737-a3b404763d7a_169.mp4` ✓ |
| C02 | attack | classifier_accepted | `8c49e480` | 338 | 1 | `8c49e480-407e-4118-8a9e-c4ed5172a7ce_338.mp4` ✓ |
| C03 | attack | dedup_survived | `4c27b635` | 140 | 27 | `4c27b635-fbab-4bcb-a30e-f82a87c223c2_140.mp4` ✓ |
| C04 | dig | classifier_accepted | `72c8229b` | 436 | 3 | `72c8229b-2993-4310-9b61-cd6162cc27fa_436.mp4` ✓ |
| C05 | dig | classifier_accepted | `89c53e08` | 225 | 4 | `89c53e08-09b8-49f8-9991-97d97b3616d0_225.mp4` ✓ |
| C06 | dig | classifier_accepted | `c4eb6810` | 356 | 3 | `c4eb6810-46f4-4e65-af9d-427e75717c8b_356.mp4` ✓ |
| C07 | dig | dedup_survived | `04ef801f` | 228 | 4 | `04ef801f-23af-4795-a34e-f02c9a68b842_228.mp4` ✓ |
| C08 | dig | dedup_survived | `d0d64de1` | 194 | 2 | `d0d64de1-b63e-4fff-8b13-cb0e2ed1a09d_194.mp4` ✓ |
| C09 | receive | classifier_accepted | `0ab56722` | 167 | 1 | `0ab56722-74a3-4f74-8a25-83ad2a529a3c_167.mp4` ✓ |
| C10 | receive | classifier_accepted | `85f7b2a6` | 146 | 4 | `85f7b2a6-4bb0-4496-8916-cf540540426b_146.mp4` ✓ |
| C11 | receive | dedup_survived | `3655eb69` | 633 | 3 | `3655eb69-b01f-431b-8d99-911ef4c414d7_633.mp4` ✓ |
| C12 | receive | dedup_survived | `5cc5127f` | 151 | 1 | `5cc5127f-2c64-465d-a289-ba5960e6125d_151.mp4` ✓ |
| C13 | receive | dedup_survived | `9bb60892` | 123 | 4 | `9bb60892-403c-449d-b088-d2284c87e7fd_123.mp4` ✓ |
| C14 | serve | ball_tracked | `0d84f858` | 46 | 4 | `0d84f858-afad-4cdc-9f72-e88b4137c313_46.mp4` ✓ |
| C15 | serve | dedup_survived | `ac84c527` | 118 | 1 | `ac84c527-9d95-4512-9b6f-8aa965bb7152_118.mp4` ✓ |
| C16 | set | ball_tracked | `fcabf602` | 132 | 2 | `fcabf602-113c-4de3-9595-86836c188198_132.mp4` ✓ |
| C17 | set | candidate_generated | `e5e4c0b7` | 168 | 1 | `e5e4c0b7-7f18-493f-b95b-574e51821452_168.mp4` ✓ |
| C18 | set | classifier_accepted | `cf33c17d` | 348 | 4 | `cf33c17d-7e25-43a4-9503-4d7b072662a6_348.mp4` ✓ |
| C19 | set | dedup_survived | `15a57d49` | 125 | 1 | `15a57d49-041b-400e-aa97-9f9c320950a1_125.mp4` ✓ |
| C20 | set | dedup_survived | `7a9aadfd` | 191 | 4 | `7a9aadfd-aa9b-4ed5-a50c-fee5e6c1ed5b_191.mp4` ✓ |

## Per-case log

Each case has two sections:

- **A. Visual observations (fill first, before opening diagnostics).** Answer the 5 brief questions.
- **B. Pipeline state (fill after A).** Diagnostic fields from the corpus + attribution. Note agreements/disagreements with A.

### C01 — attack @ frame 169 (rally `512f9467…`)

- **Clip:** `outputs/action_errors/clips/512f9467-8bd9-465f-b737-a3b404763d7a_169.mp4`
- **Video id:** `6d2f646c-1551-4917-a26c-7707a48ba0e9`
- **Lost at stage:** `candidate_generated`
- **GT player track id:** 2 (near)

#### A. Visual observations

1. **Ball visibly deflected?** — **subtle**
2. **Ball visible ±5f?** — **yes**
3. **Player in contact position?** — **yes**
4. **Other contacts within ±10f?** — ambiguous: "not sure detected because the other player slightly touches the ball trying to dig, ball moves fast but not change direction on touch, only one ground bounce"
5. **Vs. accepted contacts nearby in same rally?** — "clear only player close to ball in attack pose ball trajectory slightly changes"

#### B. Pipeline state (fill after A)

- `ball_tracked`: True  `player_tracked`: True
- `candidate_generated`: False  `classifier_accepted`: False  `dedup_survived`: False  `action_labeled`: False
- `nearest_candidate_distance`: 9999 frames  `nearest_candidate_gbm`: -1.000
- corpus `classifier_conf`: 0.252  `fn_subcategory`: deduplicated
- corpus `nearest_cand_dist`: **8 frames** (candidate existed at ~frame 177, outside the attribution's tolerance)
- `velocity`: 0.0105  `direction_change_deg`: **0.7°**  `player_distance`: **0.073**
- `ball_gap_frames` (corpus): 0
- `seq_peak_nonbg_within_5f`: **0.996**  `seq_peak_action`: **attack**
- `adjacent_gt_took_it_frame`: None  `adjacent_gt_took_it_action`: None
- `detected_contact_frames_in_window`: []  `rally_actions_in_window`: []

**Agreements / disagreements with Visual observations:**
- ✅ "Subtle deflection" matches `direction_change_deg=0.72°` (essentially flat trajectory — smallest kinematic signal possible).
- ✅ "Ball visible" matches `ball_gap_frames=0`.
- ✅ "Player in attack pose" matches `player_distance=0.073` (half of `player_contact_radius=0.15`).
- ⚠️ User noted ambiguous 2nd player touch. Pipeline `rally_actions_in_window=[]` and no `adjacent_gt` — not visible to either pred OR GT. Either GT blind-spot or the touch was too slight for anyone to label. No matching-steal here.
- 🔥 MS-TCN++ **endorses attack at 0.996**; classifier rejected at **0.252** (below 0.30 by 0.048). Brief §traps-to-avoid: seq-only rescue (`gbm<0.10 AND seq≥0.95`) would NOT fire here anyway (gbm=0.25 > 0.10), and that pattern was tested NO-GO.
- 🔸 Taxonomy split: attribution `lost_at_stage=candidate_generated` (no candidate survived within tol) vs corpus `fn_subcategory=deduplicated` (older diagnostic's label). Explained by `nearest_cand_dist=8` — a candidate fired ~8f from GT but was classifier-rejected; the two taxonomies bucket that event differently. Attribution is authoritative.

---

### C02 — attack @ frame 338 (rally `8c49e480…`)

- **Clip:** `outputs/action_errors/clips/8c49e480-407e-4118-8a9e-c4ed5172a7ce_338.mp4`
- **Video id:** `5c756c41-1cc1-4486-a95c-97398912cfbe`
- **Lost at stage:** `classifier_accepted`
- **GT player track id:** 1 (near)

#### A. Visual observations

1. **Ball visibly deflected?** — **yes**
2. **Ball visible ±5f?** — **yes**
3. **Player in contact position?** — **yes**
4. **Other contacts within ±10f?** — **no**
5. **Vs. accepted contacts nearby?** — "very clear in all signals (tracking looks good, player distinctive pose and hands near the ball making a big trajectory change, no other players around)"

#### B. Pipeline state

- `ball_tracked`: True  `player_tracked`: True
- `candidate_generated`: True  `classifier_accepted`: False  `dedup_survived`: False  `action_labeled`: False
- `nearest_candidate_distance`: **2 frames** (candidate right at GT)  `nearest_candidate_gbm`: **0.037** (deep rejection)
- corpus `classifier_conf`: 0.028  `fn_subcategory`: rejected_by_classifier
- `velocity`: 0.0144  `direction_change_deg`: **4.7°** (below all generator thresholds 15°/20°/25°)
- `player_distance`: **0.066** (~44% of radius)
- `ball_gap_frames`: 0
- `seq_peak_nonbg_within_5f`: **0.838**  `seq_peak_action`: **receive** (class-confused: GT is attack)
- `adjacent_gt_took_it_frame`: None  `adjacent_gt_took_it_action`: None
- `detected_contact_frames_in_window`: []  `rally_actions_in_window`: []

**Agreements / disagreements with Visual observations:**
- ✅ Ball visible, player close, no nearby contacts — all match.
- ❌ **STRONG DISAGREEMENT:** user saw "big trajectory change" + "very clear in all signals"; pipeline measured `direction_change_deg=4.7°` (flat) and rejected with `gbm=0.037` (deep, not marginal like C01).
- 🔎 **Plausible mechanisms (for Phase 4 quantification, not to fix now):** WASB may be sparse/noisy AROUND contact even though `ball_gap_frames=0` AT contact — the ±8f `compute_direction_change` window could straddle smoothing/interp regions that flatten a sharp real change. Candidate at 2f offset from GT: feature computation at candidate frame, not GT. MS-TCN++ class confusion ("receive" vs GT "attack") despite 0.838 non-bg suggests MS-TCN++ also saw something but couldn't lock the class.
- 🔸 Category candidate: **"visually clear deflection, kinematically flat."** C01 was a weaker version (0.72°); C02 is a clearer version (4.7° despite human-observed big change). Same underlying "feature underreports what a human sees" story worth quantifying in Phase 4.

**Agreements / disagreements with Visual observations:**
- _TBD_

---

### C03 — attack @ frame 140 (rally `4c27b635…`)

- **Clip:** `outputs/action_errors/clips/4c27b635-fbab-4bcb-a30e-f82a87c223c2_140.mp4`
- **Video id:** `43928971-2e07-4814-bb1a-3d91c7bf03b2`
- **Lost at stage:** `dedup_survived`
- **GT player track id:** **27** (non-canonical — outside 1-4; track-ID remapping suspect)

#### A. Visual observations

1. **Ball visibly deflected?** — **yes**
2. **Ball visible ±5f?** — **yes**
3. **Player in contact position?** — **yes**
4. **Other contacts within ±10f?** — **no**
5. **Vs. accepted contacts nearby?** — "very clear in all signals (tracking looks good, player distinctive pose and hands near the ball making a big trajectory change, no other players around)"

#### B. Pipeline state

- `ball_tracked`: True  `player_tracked`: True
- `candidate_generated`: True  `classifier_accepted`: **True**  `dedup_survived`: **False** ← lost here  `action_labeled`: False
- `nearest_candidate_distance`: **0 frames**  `nearest_candidate_gbm`: **0.365** (above 0.30 threshold)
- corpus `classifier_conf`: 0.020 (disagrees with attribution's 0.365 — probably a different nearby candidate)  `fn_subcategory`: rejected_by_classifier (corpus taxonomy mis-labels; attribution is authoritative)
- `velocity`: 0.0161  `direction_change_deg`: **77.3°** 💥  `player_distance`: **0.030** (~20% of radius)
- `ball_gap_frames`: 0
- `seq_peak_nonbg_within_5f`: **0.936**  `seq_peak_action`: **attack** (agrees with GT)
- `adjacent_gt_took_it_frame`: None  `adjacent_gt_took_it_action`: None
- `rally_actions_in_window`: []  `accepted_in_window_nearest_gbm`: 0.365

**Agreements / disagreements with Visual observations:**
- ✅ Every signal (kinematic, proximity, seq, classifier) matches user's "clear in all signals" report.
- ❌ Yet the contact was dropped at dedup. Brief's measured fact: dedup-elimination FNs have 0 surviving contact within tol of GT — consistent here (`rally_actions_in_window=[]`).
- 🔎 **Category hypothesis: "dedup killed a fully-endorsed contact."** Unlike C01/C02 (classifier-rejection of kinematically-flat candidates), this is a fully-endorsed contact lost during dedup. The dedup winner must have been either (a) itself absent from final pred (dropped by later stage), or (b) outside the matcher's 7-frame tolerance to GT.
- 🔸 `gt_player_track_id=27` is non-canonical. Hypothesis worth checking in Phase 4: track-ID hygiene may have affected `court_side` resolution, causing the adaptive dedup to use `min_distance=12` (same-side) instead of `_CROSS_SIDE_MIN_DISTANCE=4`. Also possible: if `court_side` came up as "unknown" for this contact, adaptive would fall through to same-side = 12f.

---

### C04 — dig @ frame 436 (rally `72c8229b…`)

- **Clip:** `outputs/action_errors/clips/72c8229b-2993-4310-9b61-cd6162cc27fa_436.mp4`
- **Video id:** `b097dd2a-6953-4e0e-a603-5be3552f462e`
- **Lost at stage:** `classifier_accepted`
- **GT player track id:** 3 (far)

#### A. Visual observations

1. **Ball visibly deflected?** — "not at the exact contact frame, ball occluded for a short moment during exact contact time by a player, but around the small occlusion the trajectory ball change is very clear"
2. **Ball visible ±5f?** — **occluded slightly**
3. **Player in contact position?** — "yes, but not exact contact point since ball occluded"
4. **Other contacts within ±10f?** — **no**
5. **Vs. accepted contacts nearby?** — "the small occlusion of the ball"

#### B. Pipeline state

- `ball_tracked`: True  `player_tracked`: True
- `candidate_generated`: True  `classifier_accepted`: **False** ← lost here  `dedup_survived`: False  `action_labeled`: False
- `nearest_candidate_distance`: **0 frames**  `nearest_candidate_gbm`: **0.038** (deep rejection)
- corpus `classifier_conf`: 0.033  `fn_subcategory`: rejected_by_classifier
- `velocity`: **0.0061** (below `min_peak_velocity=0.008`)  `direction_change_deg`: **175.9°** 🔥 (near-complete reversal)  `player_distance`: **0.059**
- `ball_gap_frames`: 0 (but this is the scalar at candidate frame; actual WASB positions around contact likely sparse/interpolated given user observation)
- `seq_peak_nonbg_within_5f`: **0.924**  `seq_peak_action`: **dig** (correct class)
- `adjacent_gt_took_it_frame`: None  `adjacent_gt_took_it_action`: None
- `rally_actions_in_window`: []  `accepted_in_window_nearest_gbm`: -1.000

**Agreements / disagreements with Visual observations:**
- ✅ "Trajectory change very clear around occlusion" matches `direction_change_deg=175.9°` — pipeline captured the reversal despite the occlusion because `compute_direction_change` samples pre/post within ±8f and finds positions on both sides of the brief hole.
- ❓ User said ball "occluded slightly" but `ball_gap_frames=0` (corpus scalar). Reconcilable: the scalar reports gap AT the candidate frame only; if WASB detected ball at 434 and 438 but missed 435-437, the ≤5f linear interp (stage 1b) would fill those frames so `ball_gap_frames` reads 0. Verifying this would require inspecting raw WASB positions around frame 436.
- ❌ **STRONG DISAGREEMENT at classifier level:** direction change 175.9° + correct MS-TCN++ class at 0.924 + close player → classifier rejected at **0.038** (deep rejection, not marginal).
- 🔎 **Category hypothesis for C04:** "ball-briefly-occluded-by-hand + low-velocity-at-contact → classifier rejects despite maximal kinematic signal." Candidates for why GBM dropped so low:
  - `velocity=0.0061` (ball momentarily stalled at hand contact) — may correlate with "tracker noise" in classifier's learned representation.
  - Pose features may be 0.0 if keypoints on the contacting hand were occluded/missing → no pose endorsement.
  - `ball_detection_density` may be reduced by the momentary occlusion in the ±10f window — brief §traps-to-avoid flags this feature with strong z-score on cls-rej FNs.
  - Aligns with brief §known-realities: "Hand-ball overlap at contact can briefly occlude the ball."
- 🔸 Distinct from C01/C02 (kinematically flat) and C03 (dedup killed clear contact). This makes 3 distinct category seeds from 4 cases.
- _TBD_

---

### C05 — dig @ frame 225 (rally `89c53e08…`)

- **Clip:** `outputs/action_errors/clips/89c53e08-09b8-49f8-9991-97d97b3616d0_225.mp4`
- **Video id:** `bbd880f2-2cc1-429c-96c0-9b72222607cb`
- **Lost at stage:** `classifier_accepted`
- **GT player track id:** 4 (far)

#### A. Visual observations

1. **Ball visibly deflected?** — **yes**
2. **Ball visible ±5f?** — **yes**
3. **Player in contact position?** — **yes**
4. **Other contacts within ±10f?** — **no**
5. **Vs. accepted contacts nearby?** — "very clear in all signals (tracking looks good, player distinctive pose and hands near the ball making a big trajectory change, no other players around)"

#### B. Pipeline state

- `ball_tracked`: True  `player_tracked`: True
- `candidate_generated`: True  `classifier_accepted`: **False** ← lost here (gbm=0.011)  `dedup_survived`: False  `action_labeled`: False
- `nearest_candidate_distance`: **1 frame**  `nearest_candidate_gbm`: **0.011** (very deep rejection)
- corpus `classifier_conf`: 0.012  `fn_subcategory`: rejected_by_classifier
- `velocity`: 0.0089  `direction_change_deg`: **≈0°** (8.5e-7)  `player_distance`: **0.045**
- `ball_gap_frames`: 0
- `seq_peak_nonbg_within_5f`: **0.949**  `seq_peak_action`: **dig** (correct class)
- `adjacent_gt_took_it_frame`: None  `adjacent_gt_took_it_action`: None
- `rally_actions_in_window`: []

**Agreements / disagreements with Visual observations:**
- ✅ Ball visible, player close, no nearby contacts — all match.
- ❌ **STRONG DISAGREEMENT (same pattern as C02):** user saw "big trajectory change" + "very clear in all signals"; pipeline measured `direction_change_deg ≈ 0°` and rejected with `gbm=0.011` (very deep).
- 🔎 **Pattern now consolidating (3 of 4 cls-rej cases so far — C02, C04 kinematically is opposite, C05):** user observes clear ball deflection at contact; pipeline's measured direction-change feature is <5° or near-zero; GBM rejects deeply. Likely mechanism: WASB sparsity around contact means `compute_direction_change` can't find close pre/post positions within ±8f that capture the actual reversal, so the angle gets averaged across a too-wide interval. Worth verifying in Phase 4 by inspecting raw ball_positions around each of these frames.

---

### C06 — dig @ frame 356 (rally `c4eb6810…`)

- **Clip:** `outputs/action_errors/clips/c4eb6810-46f4-4e65-af9d-427e75717c8b_356.mp4`
- **Video id:** `ae81fff5-a80e-4f77-8315-2e3377ce7737`
- **Lost at stage:** `classifier_accepted`
- **GT player track id:** 3 (far)

#### A. Visual observations

1. **Ball visibly deflected?** — **yes**
2. **Ball visible ±5f?** — **yes**
3. **Player in contact position?** — **yes**
4. **Other contacts within ±10f?** — **no**
5. **Vs. accepted contacts nearby?** — "very clear in all signals (tracking looks good, player distinctive pose and hands near the ball making a big trajectory change, no other players around)"

#### B. Pipeline state

- `ball_tracked`: True  `player_tracked`: True
- `candidate_generated`: True  `classifier_accepted`: **False** ← lost here (gbm=0.032)  `dedup_survived`: False  `action_labeled`: False
- `nearest_candidate_distance`: **0 frames**  `nearest_candidate_gbm`: **0.032** (deep rejection)
- corpus `classifier_conf`: 0.024  `fn_subcategory`: rejected_by_classifier
- `velocity`: 0.0113  `direction_change_deg`: **177.8°** 🔥  `player_distance`: **0.047**
- `ball_gap_frames`: 0
- `seq_peak_nonbg_within_5f`: **0.938**  `seq_peak_action`: **dig** (correct class)
- `adjacent_gt_took_it_frame`: None  `adjacent_gt_took_it_action`: None
- `rally_actions_in_window`: []

**Agreements / disagreements with Visual observations:**
- ✅ Every observable signal (kinematic, proximity, seq, user) matches. `direction_change_deg=177.8°` captures the clear reversal user saw.
- ❌ GBM rejects at **0.032** (deep). Duplicate of C04 pattern (175.9°/0.038) — two dig cases where kinematic signal is maximal but classifier confidently rejects.
- 🔎 **Pattern consolidated across 6 cls-rej cases:** splits into 2 sub-modes.
  - **Mode α — kinematically flat despite visible deflection:** C01 (0.7°), C02 (4.7°), C05 (0°). Hypothesis: WASB sparsity / interp smoothing around contact means `compute_direction_change` can't capture the angle.
  - **Mode β — kinematically maximal, GBM still rejects:** C04 (175.9°), C06 (177.8°). Both digs. Both have correct MS-TCN++ class at ≥0.92, close player, gbm<0.04. The GBM has 25 features beyond `direction_change_deg` — something in those (low velocity, low ball_detection_density, pose 0, …) dominates the decision. Worth unpacking per-feature in Phase 4.
- _TBD_

---

### C07 — dig @ frame 228 (rally `04ef801f…`)

- **Clip:** `outputs/action_errors/clips/04ef801f-23af-4795-a34e-f02c9a68b842_228.mp4`
- **Video id:** `d07ce9e1-aa5d-4875-97c4-f565bbfc14f3`
- **Lost at stage:** `dedup_survived`
- **GT player track id:** 4 (far)

#### A. Visual observations

1. **Ball visibly deflected?** — **yes**
2. **Ball visible ±5f?** — **yes**
3. **Player in contact position?** — **yes**
4. **Other contacts within ±10f?** — **yes (attack before the dig)** ← FIRST CASE WITH ADJACENT CONTACT OBSERVED
5. **Vs. accepted contacts nearby?** — "very clear in all signals (tracking looks good, player distinctive pose and hands near the ball making a big trajectory change, no other players around)"

#### B. Pipeline state

- `ball_tracked`: True  `player_tracked`: True
- `candidate_generated`: True  `classifier_accepted`: **True** (a candidate with gbm **0.350** was accepted in the window)  `dedup_survived`: **False** ← lost here  `action_labeled`: False
- `nearest_candidate_distance`: 2 frames  `nearest_candidate_gbm`: **0.067** (that one was rejected; the 0.350 one accepted per `accepted_in_window_nearest_gbm`)
- corpus `classifier_conf`: 0.139  `fn_subcategory`: rejected_by_classifier (corpus labels the nearest-rejected candidate, not the accepted one — another corpus-taxonomy mismatch)
- `velocity`: **0.0363** (above high-velocity 0.025)  `direction_change_deg`: **29.6°** (above 20° threshold)  `player_distance`: **0.067**
- `ball_gap_frames`: 0
- `seq_peak_nonbg_within_5f`: **0.922**  `seq_peak_action`: **dig** (correct class)
- `adjacent_gt_took_it_frame`: None  `adjacent_gt_took_it_action`: None  `rally_actions_in_window`: []

**Agreements / disagreements with Visual observations:**
- ✅ Every kinematic + proximity signal matches user's "clear in all signals."
- ✅ Classifier agreed too (accepted at 0.350) — then dedup eliminated it.
- 🔎 **This is a textbook attack→dig pair collapse.** User saw an attack preceding the dig. If the attack was at ~frame 215-220 and dig at 228 (gap 8-13f), and dedup used same-side distance 12f, the dig got killed as a "duplicate" of the attack. The adaptive cross-side rule (`_CROSS_SIDE_MIN_DISTANCE=4`) would preserve the pair IF court_side was resolved differently for the two contacts. Attack→dig is usually cross-side (attacker on one side, defender digging on the other), so the expected behavior would be to preserve both.
- 🔎 **Strong hypothesis emerging:** dedup-elimination FNs (C03, C07) both show the pattern "accepted candidate killed by dedup against a near-in-time winner that doesn't appear in final pred within GT tolerance." The lever is likely **court_side resolution failures that force same-side 12f merging** when the pair is actually cross-side.
- 🔸 Corpus `fn_subcategory=rejected_by_classifier` misleads here — the **nearest** candidate (at distance 2, gbm 0.067) was rejected, but a **different** in-window candidate (gbm 0.350) was accepted and then dedup-killed. Attribution's `classifier_accepted=True` + `dedup_survived=False` is the authoritative story.
- _TBD_

---

### C08 — dig @ frame 194 (rally `d0d64de1…`)

- **Clip:** `outputs/action_errors/clips/d0d64de1-b63e-4fff-8b13-cb0e2ed1a09d_194.mp4`
- **Video id:** `d3486f0b-6601-43d8-a60f-6ecc4874c408`
- **Lost at stage:** `dedup_survived`
- **GT player track id:** 2 (near)

#### A. Visual observations

1. **Ball visibly deflected?** — **yes**
2. **Ball visible ±5f?** — **yes**
3. **Player in contact position?** — **yes**
4. **Other contacts within ±10f?** — "yes (attack before the dig) - although not sure if exactly in the 10f window"
5. **Vs. accepted contacts nearby?** — "very clear in all signals (tracking looks good, player distinctive pose and hands near the ball making a big trajectory change, no other players around)"

#### B. Pipeline state

- `ball_tracked`: True  `player_tracked`: True
- `candidate_generated`: True  `classifier_accepted`: **True** (gbm **0.670** — confident!)  `dedup_survived`: **False** ← lost here  `action_labeled`: False
- `nearest_candidate_distance`: 2 frames  `nearest_candidate_gbm`: **0.670**
- corpus `classifier_conf`: 0.024  `fn_subcategory`: no_player_nearby (corpus labels the geometric-nearest candidate; attribution reflects the actually-accepted candidate)
- `velocity`: **0.042** (above 0.025 high-velocity threshold)  `direction_change_deg`: 5.0°  `player_distance`: **0.172** (marginally beyond `player_contact_radius=0.15`)
- `ball_gap_frames`: 0
- `seq_peak_nonbg_within_5f`: **0.982**  `seq_peak_action`: **dig** (correct class, very strong)
- `adjacent_gt_took_it_frame`: None  `adjacent_gt_took_it_action`: None  `rally_actions_in_window`: []

**Agreements / disagreements with Visual observations:**
- ✅ Ball visible, player in pose, clear kinematics (high velocity 0.042 carries the classifier even when direction_change is only 5°).
- ✅ Classifier confidently accepted (0.670) — this wasn't a marginal decision.
- ❌ Dedup eliminated it despite strong acceptance.
- 🔎 **Second attack→dig pair case in a row** (C07, C08). User confirmed attack before dig both times. Dedup's adaptive cross-side rule (4f) should preserve these; same-side 12f wouldn't. The `accepted_in_window_nearest_gbm` values confirm real candidates existed in the window but died at dedup.
- 🔸 Corpus `fn_subcategory=no_player_nearby` is about the `player_distance=0.172 > 0.15` geometric-nearest candidate. But that's not the candidate the pipeline actually lost — the 0.670 accepted one was. Corpus taxonomy limitation.
- _TBD_

---

### C09 — receive @ frame 167 (rally `0ab56722…`)

- **Clip:** `outputs/action_errors/clips/0ab56722-74a3-4f74-8a25-83ad2a529a3c_167.mp4`
- **Video id:** `0a383519-ecaa-411a-8e5e-e0aadc835725`
- **Lost at stage:** `classifier_accepted`
- **GT player track id:** 1 (near)

#### A. Visual observations

1. **Ball visibly deflected?** — **yes**
2. **Ball visible ±5f?** — **yes**
3. **Player in contact position?** — **yes**
4. **Other contacts within ±10f?** — **no**
5. **Vs. accepted contacts nearby?** — "very clear in all signals (tracking looks good, player distinctive pose and hands near the ball making a big trajectory change, no other players around)"

#### B. Pipeline state

- `ball_tracked`: True  `player_tracked`: True
- `candidate_generated`: True  `classifier_accepted`: **False** ← lost here (gbm=0.064)  `dedup_survived`: False  `action_labeled`: False
- `nearest_candidate_distance`: 3 frames  `nearest_candidate_gbm`: **0.064** (deep rejection)
- corpus `classifier_conf`: 0.016  `fn_subcategory`: rejected_by_classifier
- `velocity`: 0.0136  `direction_change_deg`: **7.8°**  `player_distance`: **0.024** (extremely close — 16% of radius)
- `ball_gap_frames`: 0
- `seq_peak_nonbg_within_5f`: **0.973**  `seq_peak_action`: **receive** (correct class, very strong)
- `adjacent_gt_took_it_frame`: None  `adjacent_gt_took_it_action`: None
- `rally_actions_in_window`: []  `accepted_in_window_nearest_gbm`: -1.0 (no candidate in the window was accepted)

**Agreements / disagreements with Visual observations:**
- ✅ Ball visible, player very close (`player_distance=0.024` — closest case so far), no nearby contacts, MS-TCN++ endorses correct class at 0.973.
- ❌ User saw "big trajectory change"; pipeline measured 7.8°. Same Mode α pattern.
- ❌ GBM rejected at 0.064 despite extreme player proximity + very strong seq + moderate velocity.
- 🔎 Video `0a383519` is the **night-stadium / crop-guided ReID** fixture per memory — ball tracking under low light may be noisier here, consistent with WASB-sparsity-around-contact hypothesis.
- 🔸 Strongest Mode α case yet: player_distance=0.024 (closer than any other case), yet GBM still rejects 0.064. If the GBM uses player_distance and seq_max_nonbg as major features, this case SHOULD have been accepted. Something else pushing hard.
- _TBD_

---

### C10 — receive @ frame 146 (rally `85f7b2a6…`)

- **Clip:** `outputs/action_errors/clips/85f7b2a6-4bb0-4496-8916-cf540540426b_146.mp4`
- **Video id:** `dd042609-e22e-4f60-83ed-038897c88c32`
- **Lost at stage:** `classifier_accepted`
- **GT player track id:** 4 (far)

#### A. Visual observations

1. **Ball visibly deflected?** — "not at the contact frame, heavy occlusion but before and after there's a clear direction change"
2. **Ball visible ±5f?** — **no, heavy occlusion by server on the near side**
3. **Player in contact position?** — **no, occluded both player and ball**
4. **Other contacts within ±10f?** — **no**
5. **Vs. accepted contacts nearby?** — "occlusion"

#### B. Pipeline state

- `ball_tracked`: True  `player_tracked`: True
- `candidate_generated`: True  `classifier_accepted`: **False** ← lost here (gbm=0.225)  `dedup_survived`: False  `action_labeled`: False
- `nearest_candidate_distance`: 2 frames  `nearest_candidate_gbm`: **0.225** (marginal — closest to threshold of any cls-rej so far)
- corpus `classifier_conf`: 0.010  `fn_subcategory`: **no_player_nearby** (consistent!)
- `velocity`: 0.0072  `direction_change_deg`: **0.0°**  `player_distance`: **inf** 🚨 (no player tracked near ball)
- `ball_gap_frames`: **1** (brief ball gap confirmed by corpus)
- `seq_peak_nonbg_within_5f`: **0.996**  `seq_peak_action`: **receive** (correct class, maximal — MS-TCN++ sees through occlusion via multi-frame context)
- `adjacent_gt_took_it_frame`: None  `adjacent_gt_took_it_action`: None
- `rally_actions_in_window`: []

**Agreements / disagreements with Visual observations:**
- ✅ All user observations match pipeline directly: heavy occlusion → `ball_gap_frames=1` AND `player_distance=inf`. Direction change 0° because occlusion broke the trajectory sample.
- ✅ MS-TCN++ still endorsed the correct class at 0.996 — its wider temporal window sees through the frame-level occlusion.
- 🆕 **Mode γ: "heavy dual-occlusion (ball + player)."** Distinct from Mode α (kinematically flat with player close) and Mode β (kinematically maximal with GBM still rejecting). Here the player tracker itself lost the player → classifier feature defaults to `player_distance=1.0` via the `to_array()` substitution → deep rejection.
- 🔎 This case aligns with brief §known-realities: "Far-side servers can be fully occluded by the net, body position, or other players." Here the receive-side equivalent: far-side receiver occluded by near-side server's body.
- 🔸 Related but milder version: C08 had `player_distance=0.172` (just-beyond-radius). The player-tracker-miss spectrum runs from "close but just beyond radius" to "lost entirely (inf)."
- _TBD_

---

### C11 — receive @ frame 633 (rally `3655eb69…`)

- **Clip:** `outputs/action_errors/clips/3655eb69-b01f-431b-8d99-911ef4c414d7_633.mp4`
- **Video id:** `b5fb0594-d64f-4a0d-bad9-de8fc36414d0`
- **Lost at stage:** `dedup_survived`
- **GT player track id:** 3 (far)
- **Video fps: 60** (outlier — most of dataset is 30 fps)
- **Rally context: 3 FNs in same rally** (serve @ 563, receive @ 633 this case, set @ 733)

#### A. Visual observations

1-5. **Combined meta-observation from user:** "this looks like a tracker bug, both player and ball are shifted probably related to something else that caused this"

User could not complete the 5-question template because the clip overlays show misaligned ball+player positions. This is a data-visualization/quality concern, not an FN mechanism observation.

#### B. Pipeline state

- `ball_tracked`: True  `player_tracked`: True
- `candidate_generated`: True  `classifier_accepted`: **True** (gbm **0.688** accepted in window)  `dedup_survived`: **False** ← lost here  `action_labeled`: False
- `nearest_candidate_distance`: 1 frame  `nearest_candidate_gbm`: 0.038 (secondary rejected)  `accepted_in_window_nearest_gbm`: **0.688** (the winner)
- corpus `classifier_conf`: 0.024  `fn_subcategory`: rejected_by_classifier (misleading corpus label again)
- `velocity`: 0.013  `direction_change_deg`: **47.7°**  `player_distance`: **0.015** (extraordinarily close, 10% of radius)
- `ball_gap_frames`: 0
- `seq_peak_nonbg_within_5f`: **0.985**  `seq_peak_action`: **receive** (correct class)
- `adjacent_gt_took_it_frame`: None  `adjacent_gt_took_it_action`: None  `rally_actions_in_window`: []

**Agreements / disagreements with Visual observations:**
- 🔴 **Inconclusive for Mode categorization.** User flagged a tracker/rendering bug that prevented visual verification. Numerically the pipeline has all signals strong (gbm 0.688 accepted, dir 47.7°, dist 0.015, seq 0.985) and dedup eliminated the accept — same pattern as C07/C08.
- 🔎 **Potential rendering bug hypothesis:** 60fps video + `render_action_error_strips.py` uses `CLIP_FPS=15` and a fixed `WINDOW_BEFORE=30` frames. If the renderer's frame-lookup logic assumes 30fps, a 60fps video would cause frames to be displayed at 2× offset — the GT contact frame shown in the clip would actually be frame `GT - 30` (not `GT`), and overlays would map to different underlying content. Worth verifying.
- 🔎 **Alternative: actual tracker drift.** If DB positions are genuinely shifted but consistently, the pipeline could still measure plausible relative geometry.
- 🔸 Rally has 3 FNs (serve/receive/set) at frames 563, 633, 733. Spacing of 70-100 frames (at 60fps = 1.17-1.67s apart). If multiple losses cluster in one rally, a rally-level data issue is more likely than independent per-contact failures.
- 🔸 Treating this case as **inconclusive** for Mode categorization. Phase 4 should (a) verify the clip renderer's fps handling and (b) quantify whether 60fps rallies or multi-FN rallies have systematically different error profiles.
- _TBD_

---

### C12 — receive @ frame 151 (rally `5cc5127f…`)

- **Clip:** `outputs/action_errors/clips/5cc5127f-2c64-465d-a289-ba5960e6125d_151.mp4`
- **Video id:** `9e9a2327-8766-4515-9188-47293f9aad85`
- **Lost at stage:** `dedup_survived`
- **GT player track id:** 1 (near)

#### A. Visual observations

1. **Ball visibly deflected?** — "not at the contact frame, heavy occlusion but before and after there's a clear direction change"
2. **Ball visible ±5f?** — **no, heavy occlusion by server on the near side**
3. **Player in contact position?** — **no, occluded both player and ball**
4. **Other contacts within ±10f?** — **no**
5. **Vs. accepted contacts nearby?** — "occlusion"

#### B. Pipeline state

- `ball_tracked`: True  `player_tracked`: True
- `candidate_generated`: True  `classifier_accepted`: **True** (gbm **0.414** above threshold)  `dedup_survived`: **False** ← lost here  `action_labeled`: False
- `nearest_candidate_distance`: 4 frames  `nearest_candidate_gbm`: **0.414**
- corpus `classifier_conf`: 0.006  `fn_subcategory`: **no_player_nearby** (consistent with occlusion)
- `velocity`: 0.0087  `direction_change_deg`: **0.0°** (flat due to occlusion)  `player_distance`: **inf** 🚨
- `ball_gap_frames`: **2** (bigger gap than C10's 1)
- `seq_peak_nonbg_within_5f`: **0.995**  `seq_peak_action`: **receive** (correct class, maximal)
- `adjacent_gt_took_it_frame`: None  `adjacent_gt_took_it_action`: None  `rally_actions_in_window`: []

**Agreements / disagreements with Visual observations:**
- ✅ All user observations match pipeline directly: dual occlusion → `ball_gap_frames=2` AND `player_distance=inf`. Direction change 0° because trajectory sample broken.
- ✅ MS-TCN++ endorsed correct class at 0.995 via multi-frame temporal context.
- 🆕 **Mode γ sub-split:** this is Mode γ₂ (dual-occlusion → classifier accept → dedup kill). C10 was Mode γ₁ (dual-occlusion → classifier reject). The difference: here the other features (seq 0.995, likely arc_residual or density) carried gbm above 0.30 despite `player_distance=1.0` default; dedup then eliminated.
- 🔎 **Dedup-elimination pattern now 5-for-5** (C03/C07/C08/C11/C12): each had a classifier-accepted candidate (gbm 0.35-0.69) that dedup killed. Brief's property that "0 surviving contact within tol of GT" holds: the dedup winner is not visible in the final pred. Phase 4 target: identify the dedup winner for each case and understand why it's absent from final pred within 7f tolerance.
- _TBD_

---

### C13 — receive @ frame 123 (rally `9bb60892…`)

- **Clip:** `outputs/action_errors/clips/9bb60892-403c-449d-b088-d2284c87e7fd_123.mp4`
- **Video id:** `1a5da176-8755-4e0d-8afd-ed1cab746fe3`
- **Lost at stage:** `dedup_survived`
- **GT player track id:** 4 (far)

#### A. Visual observations

1. **Ball visibly deflected?** — **yes**
2. **Ball visible ±5f?** — **yes**
3. **Player in contact position?** — **yes**
4. **Other contacts within ±10f?** — **no**
5. **Vs. accepted contacts nearby?** — "very clear in all signals (tracking looks good, player distinctive pose and hands near the ball making a big trajectory change, no other players around)"

#### B. Pipeline state

- `ball_tracked`: True  `player_tracked`: True
- `candidate_generated`: True  `classifier_accepted`: **True** (gbm **0.866** — highest accept so far)  `dedup_survived`: **False** ← lost here  `action_labeled`: False
- `nearest_candidate_distance`: 2 frames  `nearest_candidate_gbm`: 0.023 (secondary rejected)  `accepted_in_window_nearest_gbm`: **0.866** (the winner)
- corpus `classifier_conf`: 0.012  `fn_subcategory`: rejected_by_classifier (corpus mislabels again — labels the secondary, not the accepted)
- `velocity`: 0.017  `direction_change_deg`: **147.6°** 💥  `player_distance`: **0.013** (9% of radius — closest case yet)
- `ball_gap_frames`: 0
- `seq_peak_nonbg_within_5f`: **0.999**  `seq_peak_action`: **receive** (correct class, essentially 1.0)
- `adjacent_gt_took_it_frame`: None  `adjacent_gt_took_it_action`: None  `rally_actions_in_window`: []

**Agreements / disagreements with Visual observations:**
- ✅ Every signal maximal. Classifier very confidently accepted (0.866). MS-TCN++ at 0.999. Direction change 147.6°. Player distance 0.013.
- ❌ Dedup still killed it.
- 🔎 **Dedup-elimination pattern now 6/6** with increasing confidence: C03 (0.365) → C07 (0.350) → C08 (0.670) → C11 (0.688) → C12 (0.414) → **C13 (0.866)**. Every case had a classifier-accepted candidate that dedup eliminated against a "winner" not visible in the final pred within tolerance.
- 🔥 **Strongest single piece of evidence so far:** C13's GBM accept at 0.866 is NOT marginal. The classifier is calibrated high-confidence correct and dedup is vetoing it. Phase 4 must identify the dedup winner's frame/court_side/action and understand why it's not in the final pred output — this is the clearest surgical-fix target the diagnosis has surfaced.
- _TBD_

---

### C14 — serve @ frame 46 (rally `0d84f858…`)

- **Clip:** `outputs/action_errors/clips/0d84f858-afad-4cdc-9f72-e88b4137c313_46.mp4`
- **Video id:** `a7ee3d38-a3a9-4dcd-a2af-e0617997e708`
- **Lost at stage:** `ball_tracked`
- **GT player track id:** 4 (far server)

#### A. Visual observations

1. **Ball visibly deflected?** — "yes, but ball tracking looks jumpy"
2. **Ball visible ±5f?** — **yes**
3. **Player in contact position?** — **yes**
4. **Other contacts within ±10f?** — **no**
5. **Vs. accepted contacts nearby?** — "it's a serve so maybe ball tracking starts too sudden and not enough signal from ball to detect the trajectory change, player pose and contact very clear"

#### B. Pipeline state

- `ball_tracked`: **False** ← lost here (WASB missed frames around contact)  `player_tracked`: True
- `candidate_generated`: True  `classifier_accepted`: **True** (gbm **0.758** — shifted 7+f from GT)  `dedup_survived`: False  `action_labeled`: False
- `nearest_candidate_distance`: **7 frames** (attribution) / `nearest_cand_dist`: **11 frames** (corpus) — candidate exists but **outside matcher's 7f tolerance**
- corpus `classifier_conf`: 0.018  `fn_subcategory`: **ball_dropout** (consistent with user)
- `velocity`: 0.0  `direction_change_deg`: 0.0°  `player_distance`: inf (all degenerate at GT frame — ball dropout)
- `ball_gap_frames`: **7** (significant gap confirmed)
- `seq_peak_nonbg_within_5f`: **0.997**  `seq_peak_action`: **serve** (correct class)
- `adjacent_gt_took_it_frame`: None  `adjacent_gt_took_it_action`: None  `rally_actions_in_window`: []

**Agreements / disagreements with Visual observations:**
- ✅ User "ball tracking jumpy" matches `ball_gap_frames=7`. User "starts too sudden" aligns with brief §known-realities: "Ball tracking may start a few frames late at rally start."
- 🆕 **Mode δ: "serve ball-dropout + matcher-tolerance FN."** WASB missed frames right around serve contact (frame 46). A candidate generates 7-11 frames later (possibly at the first WASB detection); classifier confidently accepts it (gbm=0.758); but Hungarian matcher's 7f tolerance can't pair GT@46 with pred@53+. Something ALSO happens at dedup (`dedup_survived=False`).
- 🔎 **Brief §"Pass 2 — synthetic serve match"** exists at **~1s tolerance** specifically to rescue such cases. Apparently did NOT fire here — `rally_actions_in_window=[]` suggests the accepted candidate didn't survive as a synthetic-serve emission either, OR the synthetic wasn't within the 1s tolerance from GT@46. Phase 4 should inspect: for each serve FN with ball_dropout signature, was a synthetic serve generated and at what frame?
- 🔸 Mode δ is distinct from Mode γ (no occlusion — user said ball visible overall, just the tracker jumped at start).
- _TBD_

---

### C15 — serve @ frame 118 (rally `ac84c527…`)

- **Clip:** `outputs/action_errors/clips/ac84c527-9d95-4512-9b6f-8aa965bb7152_118.mp4`
- **Video id:** `211e2a4c-c9a3-4438-9b0c-bea4e7555ad0`
- **Lost at stage:** `dedup_survived`
- **GT player track id:** 1 (near server)

#### A. Visual observations

1. **Ball visibly deflected?** — "yes, but ball tracking looks jumpy"
2. **Ball visible ±5f?** — **yes**
3. **Player in contact position?** — **yes**
4. **Other contacts within ±10f?** — **no**
5. **Vs. accepted contacts nearby?** — "it's a serve so maybe ball tracking starts too sudden and not enough signal from ball to detect the trajectory change, player pose and contact very clear"

⚠️ **NOTE:** user answers are identical to C14's; pipeline state for C15 does NOT show jumpy tracking (ball_gap_frames=0, direction_change=126.3°, gbm=0.965). Possible user gave the C14 template by habit rather than the actual C15 clip observations. Recorded verbatim but treating this case's Mode categorization on pipeline data, not the (possibly-template) visual.

#### B. Pipeline state

- `ball_tracked`: True  `player_tracked`: True
- `candidate_generated`: True  `classifier_accepted`: **True** (gbm **0.965** — HIGHEST gbm accept in sample)  `dedup_survived`: **False** ← lost here  `action_labeled`: False
- `nearest_candidate_distance`: 2 frames  `nearest_candidate_gbm`: **0.965**
- corpus `classifier_conf`: 0.099  `fn_subcategory`: rejected_by_classifier (mislabels again — the 0.965 accept is the real story)
- `velocity`: **0.031** (above 0.025 high-vel, typical serve)  `direction_change_deg`: **126.3°**  `player_distance`: 0.116
- `ball_gap_frames`: 0 (no gap — contradicts user's "jumpy" observation)
- `seq_peak_nonbg_within_5f`: **0.996**  `seq_peak_action`: **serve** (correct)
- `adjacent_gt_took_it_frame`: None  `adjacent_gt_took_it_action`: None  `rally_actions_in_window`: []

**Agreements / disagreements with Visual observations:**
- 🔴 **User answer is identical to C14's template** but pipeline signals here do NOT show jumpy tracking. Either (a) user re-used the C14 answer by habit, or (b) the visual "jumpiness" is genuine but not captured in `ball_gap_frames` (e.g., rapid high-frequency position wobble within detected frames — which is not what `ball_gap_frames` measures).
- 🔎 On pipeline data alone: **this is Mode "confident accept killed by dedup" — the same pattern as C03/C07/C08/C11/C12/C13, NOT Mode δ ball-dropout.**
- 🔥 **gbm=0.965 is the highest acceptance score in all 15 cases so far.** The classifier is near-certain this is a real serve contact. Dedup still vetoed. Phase 4 must identify the dedup winner — given this is a near-side serve, candidates: (a) toss-moment pseudo-peak within 12 frames, (b) a phantom contact the phantom-serve rejection logic (stage 8) was designed to filter but didn't, (c) court_side misclassified.
- 🔸 **Dedup-elimination pattern updated to 7/7**, gbm range 0.350-0.965. The dedup_survived FN bucket (84 non-block FNs per brief) looks increasingly like "classifier did the right thing; dedup is wrong."
- _TBD_

---

### C16 — set @ frame 132 (rally `fcabf602…`)

- **Clip:** `outputs/action_errors/clips/fcabf602-113c-4de3-9595-86836c188198_132.mp4`
- **Video id:** `ae81fff5-a80e-4f77-8315-2e3377ce7737`
- **Lost at stage:** `ball_tracked`
- **GT player track id:** 2 (near)

#### A. Visual observations

1. **Ball visibly deflected?** — "yes, but ball tracking looks lost"
2. **Ball visible ±5f?** — "yes, but ball tracking looks lost"
3. **Player in contact position?** — **yes**
4. **Other contacts within ±10f?** — **no**
5. **Vs. accepted contacts nearby?** — "ball tracking looks like lost at contact time"

#### B. Pipeline state

- `ball_tracked`: **False** ← lost here  `player_tracked`: True
- `candidate_generated`: **False** (cascade from ball loss)  `classifier_accepted`: False  `dedup_survived`: False  `action_labeled`: False
- `nearest_candidate_distance`: **9999** frames (no candidate anywhere nearby)  `nearest_candidate_gbm`: -1.0
- corpus `classifier_conf`: 0.009  `fn_subcategory`: **ball_dropout** (consistent with user)
- `velocity`: 0.0  `direction_change_deg`: 0.0°  `player_distance`: inf (all degenerate)
- `ball_gap_frames`: **9** 🚨 (exceeds `max_interpolation_gap=5` at stage 1b)
- `seq_peak_nonbg_within_5f`: **0.988**  `seq_peak_action`: **set** (MS-TCN++ knew)
- `adjacent_gt_took_it_frame`: None  `adjacent_gt_took_it_action`: None  `rally_actions_in_window`: []

**Agreements / disagreements with Visual observations:**
- ✅ User "ball tracking lost" matches `ball_gap_frames=9` (> `max_interpolation_gap=5` → stage 1b can't fill; ±3f fallback at candidate-loop head also fails).
- ✅ User "ball visible but tracking lost" matches WASB missing detections despite visible ball (a tracker-confidence-threshold issue, not an occlusion issue).
- 🆕 **Mode ε: "ball-tracker gap > 5f, cascading candidate failure."** Distinct from Mode δ — no shifted candidate generated at all. Entire pipeline starved from stage 1b onward.
- 🔎 **MS-TCN++ at 0.988 "set" — correct class, independent of ball trajectory.** If a mechanism existed to inject candidates at MS-TCN++ confidence peaks for cases where `ball_gap_frames > 5`, this case is rescuable in principle. But brief §traps-to-avoid line 53 warns against "other variants" of seq-based rescue after prior NO-GO. The distinction: this would be candidate INJECTION from seq (not per-candidate rescue AFTER generation) — a subtly different mechanism. Phase 4 quantification + Phase 5 pre-registered validation would be needed before ever trying this.
- 🔸 The 9f gap isn't a stage-1 WASB inference failure per se — could be WASB detected the ball at confidence below 0.3 (the `_CONFIDENCE_THRESHOLD` at contact_detector.py:39) which excludes it from `ball_by_frame`. Worth inspecting raw WASB confidence values around frame 132 in Phase 4.
- _TBD_

---

### C17 — set @ frame 168 (rally `e5e4c0b7…`)

- **Clip:** `outputs/action_errors/clips/e5e4c0b7-7f18-493f-b95b-574e51821452_168.mp4`
- **Video id:** `a7ee3d38-a3a9-4dcd-a2af-e0617997e708`
- **Lost at stage:** `candidate_generated`
- **GT player track id:** 1 (near)

#### A. Visual observations

1. **Ball visibly deflected?** — "yes, but ball tracking looks lost"
2. **Ball visible ±5f?** — "yes, but ball tracking looks lost"
3. **Player in contact position?** — **yes**
4. **Other contacts within ±10f?** — **no**
5. **Vs. accepted contacts nearby?** — "ball tracking looks like lost at contact time"

#### B. Pipeline state

- `ball_tracked`: **True**  `player_tracked`: True
- `candidate_generated`: **False** ← lost here  `classifier_accepted`: False  `dedup_survived`: False  `action_labeled`: False
- `nearest_cand_dist` (corpus): **14 frames** (nearest cand in rally) | `nearest_candidate_distance` (attribution): 9999 (outside attribution's tol)
- corpus `classifier_conf`: 0.007  `fn_subcategory`: **no_candidate**
- `velocity`: **0.033** (high — ball was moving fast near contact)  `direction_change_deg`: 0.0°  `player_distance`: inf
- `ball_gap_frames`: **3** (within stage 1b's max_interpolation_gap=5 — interp DID fill)
- `seq_peak_nonbg_within_5f`: **0.993**  `seq_peak_action`: **set**
- `adjacent_gt_took_it_frame`: None  `adjacent_gt_took_it_action`: None  `rally_actions_in_window`: []

**Agreements / disagreements with Visual observations:**
- ⚠️ User said "ball tracking lost" but `ball_gap_frames=3` means WASB only missed 3 frames around contact — within stage 1b's 5f linear-interp window. User's observation that "tracking looks lost" is probably the brief interp hole being visually perceptible.
- 🆕 **Mode ζ: "WASB gap ≤5f filled by linear interp that erases the deflection."** Distinct from Mode ε (C16, 9f gap, no interp possible):
  - C16 (Mode ε): gap=9, no interp, no candidate anywhere → cascading failure.
  - C17 (Mode ζ): gap=3, **interp filled**, but the reconstructed linear trajectory is so smooth (zero curvature, zero direction change) that NONE of the 10 candidate generators fires:
    - Velocity peaks: linear interp → constant velocity → no peak.
    - Inflection/direction-change: linear → 0° → below 15°/25° thresholds.
    - Parabolic break: linear fits a straight line → no break.
    - Deceleration: smooth velocity → no drop.
    - Proximity: requires `player_distance ≤ 0.15` at the candidate frame, but no candidate frame in this neighborhood.
- 🔎 This is **exactly** the "subtle deflections at contact" / "hand-ball overlap" failure mode I flagged in Phase 1's pipeline doc (§ stage 1b limitations). The 5f interp window was designed to fill tracker gaps but it has no contact-awareness — a real deflection inside the interp window is erased before stage 4 ever sees the positions.
- 🔸 MS-TCN++ at 0.993 "set" — again knew the answer independently. Same candidate-injection conversation as C16, but this case is even more interesting: the ball IS tracked (gap within interp range), yet the DEFLECTION is erased. Worth a Phase 4 quantification: how many of the 18 `no_candidate` + 4 `rejected_by_gates` FNs have 1-5 frame gaps at GT? If most do, this is a large surgical-fix target — e.g., ballistic-aware (not linear) interp for small gaps, or a candidate generator that fires at WASB-gap EDGES.
- _TBD_

---

### C18 — set @ frame 348 (rally `cf33c17d…`)

- **Clip:** `outputs/action_errors/clips/cf33c17d-7e25-43a4-9503-4d7b072662a6_348.mp4`
- **Video id:** `1efa35cf-4edd-4504-b4a4-834eee9e5218`
- **Lost at stage:** `classifier_accepted`
- **GT player track id:** 4 (far)

#### A. Visual observations

1. **Ball visibly deflected?** — **yes**
2. **Ball visible ±5f?** — **yes**
3. **Player in contact position?** — "yes, the overlay highlights other player bbox but the player that did the action is tracked correctly"
4. **Other contacts within ±10f?** — **no**
5. **Vs. accepted contacts nearby?** — "ball tracking looks jumpy"

#### B. Pipeline state

- `ball_tracked`: True  `player_tracked`: True
- `candidate_generated`: True  `classifier_accepted`: **False** ← lost here (gbm 0.104, marginal)  `dedup_survived`: False  `action_labeled`: False
- `nearest_candidate_distance`: **1 frame** (perfect position)  `nearest_candidate_gbm`: **0.104** (marginal rejection, 0.20 from threshold)
- corpus `classifier_conf`: 0.014  `fn_subcategory`: deduplicated (corpus mislabels again — this is cls-rej per attribution)
- `velocity`: 0.018  `direction_change_deg`: **0.0°** (flat — interp effect)  `player_distance`: **0.074** (close)
- `ball_gap_frames`: **1** (brief hole, interp filled)
- `seq_peak_nonbg_within_5f`: **0.983**  `seq_peak_action`: **set** (correct class)
- `adjacent_gt_took_it_frame`: None  `adjacent_gt_took_it_action`: None  `rally_actions_in_window`: []

**Agreements / disagreements with Visual observations:**
- ✅ Player close (`player_distance=0.074`), correct class, no nearby contacts.
- ⚠️ "Ball tracking jumpy" is supported by the 1f gap — consistent with a visually-perceptible brief hole at contact, though 1f is technically in-spec.
- ❌ User saw clear deflection; pipeline measured `direction_change_deg=0.0°` because 1f interp flattened it. **Mode ζ again** (interp erasing the deflection), milder version than C17 (3f gap).
- 🆕 **User's "overlay highlights wrong player" observation** is a wrong_player concern, explicitly de-scoped by the brief (GT for player identity is unreliable). Noted but not counted against this FN.
- 🔸 This is a hybrid α/ζ case: small interp hole at contact + marginal GBM rejection. Less extreme than C01/C02/C05 (Mode α pure) or C17 (Mode ζ pure). Suggests Mode α and Mode ζ may share a common underlying mechanism: WASB sparsity at contact → direction_change feature underreports.
- _TBD_

---

### C19 — set @ frame 125 (rally `15a57d49…`)

- **Clip:** `outputs/action_errors/clips/15a57d49-041b-400e-aa97-9f9c320950a1_125.mp4`
- **Video id:** `b03b461b-b1c1-4f53-8cce-79c0afe8a049`
- **Lost at stage:** `dedup_survived`
- **GT player track id:** 1 (near)

#### A. Visual observations

1. **Ball visibly deflected?** — **yes**
2. **Ball visible ±5f?** — **yes**
3. **Player in contact position?** — **yes**
4. **Other contacts within ±10f?** — **no**
5. **Vs. accepted contacts nearby?** — "player moves fast to the ball where other players are to make the contact and moves away, looks like there might be tracks id switches due to fast movement and occlusion"

#### B. Pipeline state

- `ball_tracked`: True  `player_tracked`: True
- `candidate_generated`: True  `classifier_accepted`: **True** (gbm **0.894**)  `dedup_survived`: **False** ← lost here  `action_labeled`: False
- `nearest_candidate_distance`: **0 frames**  `nearest_candidate_gbm`: 0.013 (secondary rejected)  `accepted_in_window_nearest_gbm`: **0.894**
- corpus `classifier_conf`: 0.032  `fn_subcategory`: rejected_by_classifier (mislabels the accepted candidate)
- `velocity`: **0.026** (just above high-velocity threshold)  `direction_change_deg`: 5.7°  `player_distance`: **0.028** (~19% of radius)
- `ball_gap_frames`: 0
- `seq_peak_nonbg_within_5f`: **0.992**  `seq_peak_action`: **set**
- `adjacent_gt_took_it_frame`: None  `adjacent_gt_took_it_action`: None  `rally_actions_in_window`: []

**Agreements / disagreements with Visual observations:**
- ✅ Every measurable signal matches: close player, correct class at 0.992, confident classifier accept at 0.894.
- ❌ Dedup still eliminated it. **Dedup-elimination pattern: 8/8** (gbm accept range 0.350-0.965).
- 🔎 **User's track-ID-switch hypothesis** is interesting and not directly verifiable from corpus/attribution alone. Relevant because:
  - `contact_detector.py:2403-2414` has a "sequential same-player override" that swaps `track_id` when nearest player equals previous contact's player with close margin. If track IDs switch mid-rally between near neighbors, this override could mis-fire, affecting `court_side` via `_resolve_court_side`.
  - `court_side` resolution uses `team_assignments[track_id]` first, then court projection, then player_y. A mid-rally track swap could flip court_side mid-sequence, causing same-side 12f dedup where cross-side 4f was appropriate (or vice versa).
- 🔸 Multiple dedup-elimination cases now have plausible but different explanations: track-ID hygiene (C03 non-canonical track, C19 possible swap), attack→dig pair collapse (C07, C08), serve-specific (C15), dual-occlusion (C12). Phase 4 needs per-case "what was the dedup winner" trace to disambiguate.
- _TBD_

---

### C20 — set @ frame 191 (rally `7a9aadfd…`)

- **Clip:** `outputs/action_errors/clips/7a9aadfd-aa9b-4ed5-a50c-fee5e6c1ed5b_191.mp4`
- **Video id:** `dd042609-e22e-4f60-83ed-038897c88c32`
- **Lost at stage:** `dedup_survived`
- **GT player track id:** 4 (far)

#### A. Visual observations

1. **Ball visibly deflected?** — **yes**
2. **Ball visible ±5f?** — **yes**
3. **Player in contact position?** — **yes**
4. **Other contacts within ±10f?** — **no**
5. **Vs. accepted contacts nearby?** — "ball enters from outside the frame before contact and exits frame after contact, but the signals are there and clear"

#### B. Pipeline state

- `ball_tracked`: True  `player_tracked`: True
- `candidate_generated`: True  `classifier_accepted`: **True** (gbm **0.799**)  `dedup_survived`: **False** ← lost here  `action_labeled`: False
- `nearest_candidate_distance`: 2 frames  `nearest_candidate_gbm`: **0.799**  `accepted_in_window_nearest_gbm`: 0.799
- corpus `classifier_conf`: 0.035  `fn_subcategory`: rejected_by_classifier (mislabels)
- `velocity`: **0.044** (high — ball moving fast)  `direction_change_deg`: 2.5° (low)  `player_distance`: 0.123
- `ball_gap_frames`: 0
- `seq_peak_nonbg_within_5f`: **0.977**  `seq_peak_action`: **set**
- `adjacent_gt_took_it_frame`: None  `adjacent_gt_took_it_action`: None  `rally_actions_in_window`: []

**Agreements / disagreements with Visual observations:**
- ✅ User "signals are there and clear" matches pipeline's confident accept (gbm=0.799).
- 🔸 User noted ball enters/exits frame — this could trigger stage 1b's edge-margin / exit-ghost logic, but `ball_gap_frames=0` suggests no actual gap happened here. High-velocity trajectory across the frame → classifier accepted → dedup killed. Same pattern as the other 8 dedup cases.
- 🔎 **Dedup-elimination: 9/9 in sample.** The pattern is overwhelming at n=9.
- _TBD_

---
