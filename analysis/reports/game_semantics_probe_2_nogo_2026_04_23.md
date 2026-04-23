# Probe 2 NO-GO — 2D ball-side classification ceilings at 67 %

**Date:** 2026-04-23
**Probe:** Probe 2 of `docs/superpowers/plans/2026-04-22-game-semantics-scoping.md`
**Status:** **NO-GO.** 85 % crossing-consistency gate missed at 67 %. Multiple signal variants swept — all cap at the same ceiling. Root cause is architectural (monocular 2D geometry), not implementation.
**Consequence:** game-semantics as a contact-detection HARD CONSTRAINT is dead. Infrastructure preserved for attribution Phase 4 (softer γ-weighted rule cost).

## TL;DR

- Probe 1 PASSED on net-line detection (20/20 visual).
- Probe 2 ran crossing-consistency over 357 rallies × ~1 680 GT action-pairs in the 68-video LOO corpus.
- No signal exceeds ~67 % overall agreement. The 85 % gate is unreachable from 2D.
- Per the scoping plan's Probe 2 kill gate: **close the workstream.** Probe 3 / 3b / Phase A–C / implementation brief all NOT run.

## Per-signal ceiling (on the 1 680 tested pairs)

| signal | total | crossing-only | no-crossing-only |
|---|---:|---:|---:|
| court_y span crosses 8 (any margin 0–3m) | 66.2 % | 49 % | 77 % |
| court_y span AND ball went above net-top mid-flight | 67.0 % | 48 % | 79 % |
| court_y span OR ball went above net-top | 38.8 % | 99 % | 1 % |
| ball went above net-top only (min delta < −0.03) | 38.2 % | 93 % | 4 % |
| ball went above net-top only (min delta < −0.05) | 35.5 % | 82 % | 6 % |
| image-Δy endpoint side classifier (net-top decisive) | 67.5 % | 23 % | 85 % |
| image-Δy endpoint side classifier (net-base decisive) | 70.3 % | 1.5 % | 100 % |

Every signal lies on a tradeoff curve where you cap around 66–70 % total. No combination breaks through.

## Tier 1.2 anchored 3D follow-up — also caps at ~68 %

Appended 2026-04-23 later session after the initial NO-GO. Per user request, explored SOTA monocular-3D techniques before closing.

### Approach

Replaced endpoint 2D classification with **camera-ray intersection at assumed hand-height plane**. For each GT contact, per-action hand-height table (serve=2.5m, attack=2.8m, block=2.6m, set=2.3m, receive=0.8m, dig=0.3m); intersect the camera ray through ball-image-xy with z=hand_height plane to recover court-xy at the contact. Linear interpolation between consecutive contacts for court-y; crossing iff interpolated court-y spans the net midline (y=8).

Implementation: `analysis/scripts/probe2_crossing_3d.py` using `calibrate_camera` + `image_ray` from `rallycut.court.camera_model`.

### Results — 1 667-pair sweep

| signal | overall | crossing | no-cross |
|---|---:|---:|---:|
| ray → z=hand_height (bounded, ball-only) | 68.5 % | 40 % | 82 % |
| ray → z=hand_height + player-foot fallback | 52.7 % | 43 % | 59 % |
| image-dx > 0.20 | 68.2 % | 35 % | 89 % |
| image-dx > 0.25 AND court-y spans 8 | 63.7 % | 9 % | 98 % |
| image-dx > 0.25 OR court-y spans 8 | 55.5 % | 66 % | 49 % |

**No Tier 1.2 signal or combination exceeds ~68 %** — identical to the 2D ceiling.

### Why Tier 1.2 didn't break through

1. **Camera-ray-z-plane intersection is numerically unstable at ball heights close to camera height.** Rays become nearly parallel to the z-plane, so small changes in ray direction produce enormous swings in intersection distance. Concrete observation: court-y estimates ranged from −5 m to 21 540 m before bounds enforcement. Even with bounds, the remaining ~50 % of valid intersections inherit the assumed-height bias (if actual z differs from assumed by 0.5 m, court-y is off by 1–2 m).

2. **Ball-tracker recall at contact frames is poor.** 873 / 1 672 pairs (52 %) had NO ball detection within ±5 frames of the GT contact; widening to ±12 frames recovered just 5 pairs. Contact moments are exactly where WASB misses (fast motion, hand occlusion).

3. **Player-foot fallback picks the wrong player.** At high contacts (attack, block), the nearest player in image space to the ball could be the blocker (other side) or the attacker (same side). Resolving this reliably requires attribution, which is documented-unreliable.

### Tier 1.1 parabolic-fit — analytical lower bound why it won't clear 85 %

Even under perfect physics-prior fitting (3D parabola under gravity between contact pairs):
- **Endpoint ambiguity is still the limiting factor.** The parabola is anchored by the two contact 3D positions; if those are wrong, the interpolant is wrong. Tier 1.2's endpoint estimates are already at 68 %.
- **Beach VB violates the gravity-only assumption.** Wind and ball spin introduce deviations the fitter can't model without more state.
- **Intermediate 2D observations are sparse.** With 52 % contact-frame dropouts, many intervals would have too few inbetween points for stable fit.

Expected Tier 1.1 ceiling given these: 72–78 %. Still below 85 % gate, and 1–2 days of implementation to find out.

### Tier 2 — DepthAnythingV2 / Metric3D — not pursued this session

Foundation monocular depth would give a z estimate per ball pixel, bypassing hand-height assumptions. Caveats: ball is ~3 px at 480p — sub-pixel depth noise compounds; per-frame GPU compute on Modal; 2–3 days to deploy, cache, integrate. Out-of-scope given Tier 1.2's architectural cap suggesting 3D reconstruction from RGB alone has a sports-specific ceiling that no monocular method reliably breaks.

### Final decision

Tier 1.2 explored, matches 2D ceiling at 68 %, 85 % gate unachievable without stereo / multi-camera or a domain-trained 3D ball reconstructor (which is the abandoned `ball_3d_phase_c` workstream). **Workstream closes.**

Artifacts retained:
- `analysis/scripts/probe2_crossing_3d.py` — Tier 1.2 implementation (run it again if a future session wants to re-validate).
- `analysis/outputs/game_semantics_probe/probe2_crossing_3d.jsonl` + summary — raw signal-sweep data.

## Root cause — two compounding 2D limits

### 1. Endpoint sidedness is ambiguous at player-head height

With camera behind the near baseline, a ball at court-y=4 (far, ground) and a ball at court-y=12 (near, head-height ≈1.8 m) project to nearly the same image-y. Receive / set / dig contacts happen in this ambiguous band. Endpoint classification (net-top-decisive OR net-base-decisive) collapses to a single side across most contact frames.

### 2. z=0 homography projection does not preserve far/near ordering for airborne balls

For a ball at actual (x, y_court, z), reverse-projecting from the image via the z=0 homography gives a biased court-y estimate. The bias from z ≠ 0 is LARGER than the signal from y_court differences for typical serve / set / attack heights (z ≈ 1.5–3 m).

Concretely, a ball at (x, 4, 3) — far side, 3 m up — and a ball at (x, 12, 3) — near side, 3 m up — can project to the same image-y and hence the same z=0-homography court-y. The estimated court-y rarely spans the net-midline (court-y=8) cleanly, even when the ball physically does cross. Hence 49 % agreement on crossing-expected pairs.

## What does work — and why it's insufficient

- `set→attack` pairs: 90 % agree on no-crossing (sets stay on same side; ball rarely goes above net-top on a set).
- `serve→receive` pairs: 50 % agree on crossing — the cleanest expected-crossing case, and half of it fails.
- `attack→dig` pairs: 44 % agree on crossing.

The asymmetry — good on "no crossing" no-cross classification, bad on "crossing" — means the classifier defaults to "no crossing" and misses most real crossings. Flipping the asymmetry (OR-logic) destroys no-crossing accuracy.

## What's salvaged

- **`analysis/rallycut/court/net_line_estimator.py`** stays. 20/20 net-line visual pass on Probe 1.
- **Per-video cache** (`~/.cache/rallycut/net_line/`) amortizes across any future consumer.
- **`render_side_overlay.py`** stays (useful for attribution debugging).
- **`probe1_net_line.py`** (visual gallery) stays as a reusable diagnostic.
- **`game_semantics_probe_0_coordination.md`** — the schema agreement with attribution — remains valid as the INPUT spec for attribution's Phase 4 rule-cost `γ·rule_violation` term, provided γ is weak enough to absorb the 33 % noise floor.

## What's dropped

- **Probe 3 / 3b / Phase A / B / C / implementation brief**: not run. Workstream closes.
- **Hough net-top refinement**: discarded — Path A (keypoints_net) cleared Probe 1 without it. Hough was 14/20 with over-correction failures; not needed.
- **3D ball reconstruction**: NOT reopened. `ball_3d_phase_c_audit_2026_04_11` ABANDON stands. Re-opening needs a fresh architectural hypothesis, not engineering.

## Next-best levers on the v5 contact-detection error budget

Per the original scoping plan's error table:

- **`no_player_nearby` distance-gate loosening** — 57 / 291 FN (20 %). Untouched. 4–6 hr implementation, ball-coord-based (GT-player-independent). This is the largest remaining single lever.
- **WASB ball-tracker recall** — 30 / 291 FN (10 %). User rejected retrain path; needs a smart mechanism, not more data. Backlog.
- **Rally-start serve prior** (Phase B from the original plan) — doesn't depend on ball-side. Still viable independently as a shrink of serve-FN count. ~2 hr implementation.

## Lessons

1. **2D sidedness ceilings at ~67 % on contact frames.** Don't try variants; the limit is architectural, not a threshold knob.
2. **Visual overlay passed (Probe 1) does not imply semantic signal passes (Probe 2).** Accurate net-line detection + unreliable ball-side = the two probes failed in different dimensions.
3. **The `set→attack: 90% no-cross` finding is the real takeaway.** When the ball stays on one side, the 2D test confidently says so. When the ball crosses, 2D cannot always confirm it. That asymmetry is the 2D failure signature.
4. **Soft γ beats hard rule.** For workstreams like attribution that can absorb noise via a weighted cost, a 67 % signal is still useful. For workstreams that need ≥ 85 % hard accept/reject, 2D is insufficient.

## Cross-reference

- Probe 1 PASS memo: `analysis/reports/game_semantics_probe_1_pass_2026_04_23.md`
- Probe 0 coordination memo: `analysis/reports/game_semantics_probe_0_coordination.md`
- Prior Probe-1 NO-GO (superseded): `analysis/reports/game_semantics_probe_1_nogo_2026_04_23.md`
- Prior 3D abandon: `memory/ball_3d_phase_c_audit_2026_04_11.md`
