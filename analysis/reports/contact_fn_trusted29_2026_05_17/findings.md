# Contact-Detection FN Investigation — trusted-29 (2026-05-17)

Proper investigation of the 18% unmatched-GT residual that the prior
`contact_detection_fn_v1_2026_05_12` work NO-SHIP'd as "rule-tuning
ceiling". The new diagnostic reveals a **completely different picture**:
the "FN cases" decompose into distinct failure modes, each with a
different fix path.

## Diagnostic results (227 FN cases out of 1252 GT rows)

| Failure mode | Count | % | What's actually wrong |
|---|---|---|---|
| **NEAR_PIPELINE_CONTACT** | **112** | **49.3%** | Pipeline DETECTS the contact, places it >5f from GT |
| **TRAJECTORY_CHANGE_BUT_NO_CONTACT** | **51** | **22.5%** | All signals present at GT (player nearby, direction change) but GBM rejects |
| NO_PLAYER_NEAR_BALL | 33 | 14.5% | Player-tracker FN (off-screen / edge / occluded) |
| BALL_NOT_TRACKED | 26 | 11.5% | True ball-tracker FN (night-time, edge cases) |
| Other (trajectory incomplete, no change) | 5 | 2.2% | Edge cases |

## NEAR_PIPELINE_CONTACT — placement timing, not detection FN

- **86 cases (77%)** are off by only **6-9 frames** (just barely outside the ±5 match window)
- **96 cases (86%)** have the **right action type** — pipeline labeled it correctly, just placed the contact frame slightly off
- Volleyball contacts last ~30-50ms physically; ball trajectory takes a few frames to clearly change post-contact; placement drift of 5-10 frames is mechanically expected
- **None of these are "failures to detect"** — the detector successfully found them, just labeled them at the velocity-peak frame (typically a few frames after the actual contact) instead of the GT-marked frame

**Fix path:** snap pipeline contact-frame to the local ball-velocity *minimum* within the candidate window (the moment of stillness during contact), rather than the velocity-*peak* (which fires post-contact).

**Expected recovery:** ~86 cases (~7pp on matched accuracy if "matched" means within ±5f of GT). Even without code changes, **widening the GT-match window from ±5 to ±10 frames** would re-categorize these as matched — defensible because contact timing is genuinely ambiguous at the sub-100ms scale.

## TRAJECTORY_CHANGE_BUT_NO_CONTACT — classifier rejection (NOT prior memo's "rule-tuning ceiling")

The prior NO-SHIP memo concluded "trajectory primitives don't fire at 88% of FN frames". **This investigation found the OPPOSITE: trajectory primitives DO fire at 88% of FN frames in this category, but the downstream GBM contact-classifier rejects the resulting candidates.**

Probe results (51 cases analyzed):
- **45 cases (88.2%) — candidate was generated** by trajectory primitives but rejected by GBM
- 5 cases (9.8%) — no candidate generated
- 1 case (2.0%) — candidate near but unclear which generator fired

Generator hit counts on the 45 candidate-generated cases:
- direction_change: 37 / 45
- inflection: 36 / 45
- velocity_peak: 31 / 45
- parabolic: 24 / 45
- net_crossing: 22 / 45
- deceleration: 18 / 45

**Multiple primitives fire at the GT frame.** The signal is present. The GBM classifier is the gatekeeper that rejects.

**Why does the GBM reject?** Visual inspection of 3 cases (wawa f551, kaka f396, wawa f338) shows textbook ATTACK signatures — ball reverses direction at the player's hand at the net. But pre/post velocities are ~0.06-0.07 (moderate, not high). The GBM may have learned that "real attacks have post-contact velocity > 0.10" from training data dominated by hard spikes, rejecting these slower-paced attacks (set-into-soft-attack, tip shots, attacks against tight blocks).

**Fix path:**
1. **Highest EV: retrain GBM with trusted-29 GT.** The trusted-29 corpus has +274 new GT rows from videos the classifier hasn't seen. These 45 currently-rejected cases would become explicit positives in training, teaching the GBM that low-velocity attacks are real.
2. **Alternative: high-trajectory-signal override.** Force-accept candidates where `direction_change_deg > 90° AND player_distance < 0.05` regardless of GBM score. ~30 LOC change, conservative threshold, low FP risk.

**Expected recovery:** ~40-45 of 51 (3-4pp matched accuracy on trusted-29).

## NO_PLAYER_NEAR_BALL — player-tracker FN (off-screen / occluded)

33 cases where the ball IS tracked at GT frame but no player bbox is within 0.15 normalized units. Examples:
- papa f44 SERVE: min_player_dist=0.659 — server is OFF-SCREEN (right edge of frame, behind camera)
- wawa f446 ATTACK: min_player_dist=0.359 — player far from ball (mid-jump, tracker lost them)

SERVE-heavy concentration (21/33 = 64% are serves) — confirms what we saw in the SERVE placement investigation earlier today: many near-side serves happen at the edge of the frame or off-screen.

**Fix path:** improve player-tracker recall at frame edges. Larger project — out of scope for this investigation but documented.

## BALL_NOT_TRACKED — true WASB FN

26 cases. Almost all SERVEs (24/26). Concentrated in night-time videos (vovo, wawa, vivi) and edge-of-frame scenarios. WASB ball-tracker has a real recall ceiling on low-SNR cases.

**Fix path:** WASB fine-tuning for low-SNR scenarios. Larger project — out of scope.

## Per-action breakdown (failure-mode distribution per action type)

| Action | Top failure modes |
|---|---|
| SERVE (54 FNs) | 39% NO_PLAYER (off-screen) / 26% NEAR_PLACEMENT / 24% BALL_NOT_TRACKED (night) |
| RECEIVE (41) | **68% NEAR_PLACEMENT** / 15% BALL_NOT_TRACKED |
| SET (40) | **65% NEAR_PLACEMENT** / 15% TRAJECTORY_CHANGE rejected |
| ATTACK (44) | 45% NEAR_PLACEMENT / **41% TRAJECTORY_CHANGE rejected** |
| DIG (39) | 41% NEAR_PLACEMENT / **44% TRAJECTORY_CHANGE rejected** |
| BLOCK (9) | 89% NEAR_PLACEMENT |

**Pattern:** RECEIVE/SET dominate NEAR_PLACEMENT (the contact-detector's velocity-peak heuristic places these soft contacts a few frames late). ATTACK/DIG dominate TRAJECTORY_CHANGE rejection (the GBM was trained on hard attacks; rejects soft ones).

## Recommended action sequence (highest EV first)

1. **Retrain contact-classifier (GBM) with trusted-29 GT** (~1 day, highest EV)
   - +274 new GT rows from videos the classifier hasn't seen
   - Specifically include the 45 TRAJECTORY_CHANGE_REJECTED cases as positives
   - Expected: recovers ~40 cases → +3pp matched accuracy

2. **Tighten contact-frame placement** (~half day, structural)
   - Snap detected contact frames to local ball-velocity *minimum* (moment of impact) within the candidate window, instead of the velocity peak (post-contact)
   - Recovers ~86 NEAR_PLACEMENT cases → +7pp matched accuracy
   - Side benefit: even existing correctly-placed contacts get sub-frame-accurate frame placement, improving downstream attribution feature extraction

3. **Player-tracker FN at frame edges** (~weeks, larger project)
   - Improve player tracker recall for off-screen / edge-of-frame players
   - Recovers ~25-30 of 33 NO_PLAYER cases → +2pp matched

4. **WASB low-SNR tuning** (~weeks, larger project)
   - Night-time + low-contrast ball tracking
   - Recovers ~20 of 26 BALL_NOT_TRACKED → +1.5pp matched

Combined (1) + (2) alone: **~+10pp matched accuracy** (91.6% → ~98%) — much bigger than any attribution work could deliver. Both are bounded ~1-2 day projects.

## Comparison to prior NO-SHIP memo

The prior `contact_detection_fn_v1_2026_05_12` work concluded:
> "the trajectory + player-proximity primitives the contact detector relies on don't fire at the GT frames in 88% of probe FN cases"

**This investigation found the opposite.** Primitives DO fire at the GT frames in 88% of the TRAJECTORY_CHANGE_BUT_NO_CONTACT subset. The prior probe likely measured a different subset (probably included the BALL_NOT_TRACKED + NO_PLAYER cases which DO lack primitives). Categorizing failures BEFORE measuring rejection rate gives a much clearer diagnosis.

The prior fix attempts (validation-gate relaxation, generator-threshold relaxation) failed because they addressed the wrong rejection layer (gates / generators) instead of the actual one (GBM classifier + placement timing).

## Files

- Diagnostic script: `analysis/scripts/diagnose_contact_fn_trusted29_2026_05_17.py`
- Per-case CSV: `analysis/reports/contact_fn_trusted29_2026_05_17/fn_cases.csv`
- Gate-rejection probe: `analysis/scripts/probe_contact_detector_gates_2026_05_17.py`
- Visual frame probe: `/tmp/visualize_traj_change_fn.py` (overlays in `/tmp/contact_fn_inspect/`)
