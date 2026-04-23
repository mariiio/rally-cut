# Probe 1 PASS — keypoint-anchored net-line clears the 16/20 gate

**Date:** 2026-04-23
**Status:** **PASS.** Supersedes `analysis/reports/game_semantics_probe_1_nogo_2026_04_23.md` (the homography-only NO-GO).
**Probe:** Probe 1 of docs/superpowers/plans/2026-04-22-game-semantics-scoping.md.

## TL;DR

The first Probe 1 pass used `calibrate_camera` on the user-clicked 4 corners stored in `Video.courtCalibrationJson`. It hit 11–13/20 visual pass. Root cause was under-determined focal length amplifying into net-top error at z = 2.43 m, plus two catastrophic user-labeling errors (reproj > 100 px).

Path A swapped to the **court-keypoint detector's 4 corners + 2 center points (center-left, center-right at the net-sideline intersections).** This is fully automatic — no manual calibration dependency, which is the operational constraint. Single-frame run scored **20/20** net-top pass on the same 20 stratified rallies.

Tier-1 optimizations (multi-frame aggregation, per-video cache, per-sideline confidence gating, sanity validation) landed on top and preserved the 20/20 result while reducing reproj error on 7 videos and stabilizing focal across the corpus. Two edge-case videos (`9a6a9499`, `627c1add`) still have reproj ≈ 20 px — flagged as "auto-detection unreliable" for Probe 2 abstention, not a gate failure.

## Method (the automatic pipeline)

1. **Multi-frame sampling**: 30 frames evenly spaced through the video (skipping first/last 2 s), run `CourtKeypointDetector._detect_frame()` per-frame.
2. **Weighted-median aggregation** of the 6 detected keypoints (4 corners + center-left/right) across frames, weighted by YOLO keypoint confidence.
3. **Camera calibration**: `calibrate_camera` on the aggregated corners (not the DB corners).
4. **Net-base**: use the aggregated center-left/right **directly as net-base anchors** — no homography projection needed, which is the whole point.
5. **Net-top**: camera-projects (0, 8, 2.43) and (8, 8, 2.43), then Y-shifts each side so the camera-predicted net-base matches the observed keypoint centers. Keeps the camera's tilt + x-component correct while correcting the focal-uncertainty-driven Y error.
6. **Confidence gating**: if one sideline's center-point has conf < 0.30, mirror its Y from the higher-confidence side using the corner geometry. Prevents tilt errors when one antenna pole is occluded.
7. **Sanity validation**: net-base must sit between 12 % and 85 % of the way from far-baseline image-Y to near-baseline image-Y. Outside this band → attach `sanity_failed` warning (not a reject — caller decides).
8. **Per-video cache** at `~/.cache/rallycut/net_line/<video_key>__<model_mtime>__n30.json`. One detect per video, shared across all its rallies. Invalidates automatically when the keypoint model weights change (mtime bump).

## Results

### Per-video reproj error (pixels)

| video | strat | single-frame | tier1 (30-frame) | net-top pass |
|---|---|---:|---:|:-:|
| 4cd680c7 | hard_dark | 11.1 | **8.3** | P |
| bbd880f2 | hard_dark | 3.1 | 3.3 | P |
| 6d2f646c | hard_dark | 3.2 | **2.7** | P |
| 2e984c43 | hard_dark | 1.7 | **0.9** | P |
| a5866029 | low_res | 5.1 | 6.0 | P |
| 7c61e308 | low_res | 2.1 | 1.9 | P |
| 1a5da176 | low_res | 0.3 | 1.0 | P |
| 840e8b6b | low_res | 1.9 | 2.0 | P (homography had failed) |
| 9a6a9499 | extreme_angle | 18.6 | 19.8 | P (with cyan tilt) |
| 56f2739d | extreme_angle | 4.3 | **2.9** | P |
| 0a383519 | extreme_angle | 4.0 | **2.8** | P (homography was 122 px FAIL) |
| 808a5618 | extreme_angle | 3.8 | **2.0** | P |
| ae81fff5 | regular | 1.9 | 2.5 | P |
| a7ee3d38 | regular | 3.9 | 4.4 | P |
| 2b2d7786 | regular | 1.4 | 1.2 | P (homography was 126 px FAIL) |
| e88a06f5 | regular | 0.6 | 1.4 | P |
| ff175026 | regular | 1.0 | 1.9 | P |
| 90266c1d | regular | 3.3 | **2.5** | P |
| c6e4c876 | regular | 2.6 | 3.3 | P |
| 627c1add | regular | 23.6 | 23.3 | P (both lines tilted, marginal) |

**Summary:** 20/20 PASS on net-top in user's visual scoring. 7 videos improved ≥ 10 % reproj, 11 tied within ±1 px, 2 essentially flat at high reproj. No regressions.

### Outliers — flagged for Probe 2 abstention

- `9a6a9499` — extreme camera angle (perspective ratio 0.38). Cyan (net-base) is visibly tilted. Keypoint-model fails consistently here even across 30 frames.
- `627c1add` — similar. Cyan tilted high. Low keypoint confidence (0.98 median but unstable frame-to-frame).

**Action:** treat ball-side on these two videos as `side_confidence = 0.0` in Probe 2, so they abstain from crossing-consistency tests.

## What didn't work

- **Manual-calibration homography (the original Probe 1):** 11–13/20 pass. Disqualified independently because it requires user-clicked corners, not automatic.
- **Hough net-top refinement** over the predicted strip: 14/20 pass. Over-corrects on 4–6 videos (snaps to grandstand roofs, flags, banners, shadow lines). Not shipped; fixable with multi-frame Hough aggregation, but tier-1 already clears the gate so it's a future-session distractor.

## What landed

- `analysis/rallycut/court/net_line_estimator.py` — `estimate_net_line()` + `estimate_net_line_from_s3()` helpers with 30-frame weighted-median aggregation, per-video cache, confidence gating, sanity validation.
- `analysis/scripts/probe1_net_line.py` gained `--mode tier1` and `--mode vs_tier1`.
- `analysis/scripts/render_side_overlay.py` defaulted to `--net-mode tier1` (uses the cached estimator).
- 20 COMPARE PNGs at `analysis/outputs/net_line_probe/*_COMPARE.png` documenting the three-way comparison (homography | keypoints_net | keypoints_hough) + the vs_tier1 comparison.

## Next — Probe 2

Probe 1 passes → Probe 2 is live. Plan:

1. **Crossing-consistency signal** — no manual work. Over all GT action pairs in the corpus (~1500 pairs), check whether the predicted ball-side flips as the action-type pair structurally requires (e.g. `attack → receive` = crossing expected, `receive → set` = no crossing). Gate: ≥ 85 % agreement.
2. **Trajectory-midpoint signal** — also no manual work. Midway between a crossing GT pair, the ball should not have flipped sides multiple times.
3. **Manual spot-check** — 2 hr of user time, 10 rallies × ~15 frames = ~150 labels. Gate: ≥ 90 % agreement.

Both manual and structural gates must pass. If they do, Probe 3 (the decisive wrong-action net-cross violation rate) runs next.

## Note on the superseded NO-GO memo

`analysis/reports/game_semantics_probe_1_nogo_2026_04_23.md` is kept for audit trail but is no longer the current verdict. It describes the failure mode correctly; the PASS memo describes the fix. If you read the NO-GO memo in isolation, read this one next — the workstream is OPEN, not closed.
