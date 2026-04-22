# Game-semantics scoping — net-cross, ball-side, direction-of-play

**Owner:** fresh-session CV engineer picking up after the 2026-04-22 contact-detection session.
**Prior context:** `memory/mstcn_retrain_2026_04_22.md` (v5 shipped), `memory/contact_arbitrator_2026_04_22_nogo.md` (Option B closed), FN audit on v5 (§ below).
**Kind:** scoping plan — NOT an implementation plan. Output is a go/no-go verdict + phased design, not code.

## TL;DR

The contact-detection error budget at v5 has no single fat lever left on the classifier side. The untested lever is **game-semantics**: leveraging the sport's structural rules (net-cross direction, 3-touch limit, side-of-play, serve-at-rally-start) as constraints on the candidate decoder and/or classifier accept/reject. This plan covers the **probe phase** that decides whether game-semantics is a genuine lever or a red herring before any implementation work starts. The probe phase is ≤ 2 days total; the answer it returns is "yes, implement + gate" OR "no, here's why — pivot."

**Kill gate for the whole workstream:** if Probe 3 below shows that <25% of wrong_action errors violate reconstructable net-cross direction, AND <15% of FN serves live at rally starts where a serve-prior would fire, the lever is too thin to justify implementation. Close the workstream and pivot.

## Why NOW

v5 shipped with 88.87% F1 / 94.01% Action Acc. Error pool:

| Subcategory | Count | % | Status |
|---|---:|---:|---|
| rejected_by_classifier | 134 | 46% | Arbitrator NO-GO 2026-04-22 (replace + rescue-only both fail) |
| no_player_nearby | 57 | 20% | Untouched — brief-noted as 4-6h distance-gate lever; ball-coord-based |
| deduplicated | 48 | 16% | Mostly serves (36/48); dedup-bypass already NO-GO 2026-04-21 |
| ball_dropout | 30 | 10% | WASB-adjacent; user rejects retrain path, wants smart mechanisms |
| no_candidate | 18 | 6% | Ball-tracker adjacent |
| rejected_by_gates | 4 | 1% | Tiny |
| **TOTAL FN** | **291** | | |
| wrong_action | **108** | | Most interesting for game-semantics — these are detected-but-mis-labeled |

Serves dominate: 113 / 291 FNs (39%) span all five FN categories. The shared root cause for many serve FNs is that the pipeline doesn't know "rally is about to start, bias toward serve detection" — exactly what game-semantics would encode.

Of the 108 wrong_action errors, an unknown fraction would have been structurally impossible if the pipeline knew which side of the net the ball is on and which direction it last crossed. That fraction is Probe 3 below.

## Three probes, each ≤ 4 hours. Do in order; skip later probes if earlier ones kill.

### Probe 1 — Can we detect the net line reliably (2D, image space)?

**Hypothesis:** court-keypoint detection already has the four corners. The net base is the midline between the two near-net corner pairs. The net TOP in image space is the base line raised by a fixed height via perspective (net height = 2.43 m men / 2.24 m women; we have calibration / camera geometry for most videos).

**Method:**
1. Pick 20 rallies covering the 68-video corpus (stratified: 4 easy day, 4 hard night, 4 extreme camera angle, 4 low-res, 4 regular).
2. For each rally's first frame: extract court keypoints (existing detection), fit net-base line, compute net-top line via calibrator + net-height constant.
3. Overlay both lines on the frame as a PNG and inspect visually. Count: how many rallies have net-base correctly on the net's base (±5 px) AND net-top correctly within the net's top rope (±10 px)?

**Kill gate:** <16/20 rallies pass visual check → net-line detection is not reliable enough; workstream dies here. Move to pure 2D midline heuristics (below) OR close the workstream entirely.

**Output:** a diagnostic PNG gallery at `analysis/outputs/net_line_probe/<rally_id>.png` + a CSV of `(rally_id, net_base_ok, net_top_ok, notes)`. Roughly 2 hours.

**Fallback if calibration missing:** use pure 2D midline — the HALFWAY image-Y between the top two (far-side) court corners and the bottom two (near-side) is a crude net-base proxy. Worse accuracy but no calibration dependency. Sweep this fallback on videos without calibration and measure accuracy separately.

### Probe 2 — Can we classify "ball is on which side of net" per frame, reliably?

**Hypothesis:** once the net LINE in image space is known, a per-frame label `ball_side ∈ {far, near}` follows from the ball position's 2D pixel-Y relative to the net-top line. (Using net-top, not net-base, matters: a ball at mid-height above the net but image-Y below the net-top line is still on the "far" side by convention — we need to treat the net-top as the decisive horizontal in image space.)

**Method:**
1. On the 20 rallies from Probe 1, annotate ball-side frame-by-frame (or spot-check every 30 frames) as GT. ~1 hour of manual labeling.
2. Run the pipeline's `ball_by_frame` positions + Probe 1's net-line; label each ball position `predicted_side`.
3. Compute agreement %.

**Kill gate:** <90% agreement on this task. If the 2D net-line + ball-Y isn't enough to determine side, game-semantics can't be built on top. Need 3D or better.

**Why 90%:** at 88.87% F1 on contact detection, introducing a <90%-reliable side signal into the pipeline would muddy more than it clarifies (per `feedback_small_sample_probes.md` pattern).

**Output:** CSV of `(rally_id, frame, gt_side, pred_side)` + aggregate agreement %.

### Probe 3 — Are wrong_action errors actually violating net-cross direction? (The decisive probe.)

**Hypothesis:** real volleyball has structural sequences. A dig comes after the opposite-side attack; a set comes after a same-side receive/dig. If the pipeline mislabels a set as a dig (or similar), it often happens because it doesn't know the ball was just on the opponent's side. We can check this retroactively on v5's 108 wrong_action errors.

**Method:**
1. For each of the 108 wrong_action errors in v5 corpus:
   - Extract the 10-frame window around the pred contact.
   - Using Probe 2's per-frame ball-side, determine the ball's side trajectory: `side(t−5) → side(t) → side(t+5)`.
   - Classify: "matches predicted action's side requirement" (e.g., attack requires ball was on attacker's side before the contact) vs "violates it".
2. Count the violation rate per (gt_action, pred_action) pair.

**Decisive gate:**
- **If ≥ 40% of wrong_action errors violate net-cross direction** → game-semantics is a **big lever**; proceed to implementation plan (phased below).
- **If 25–40% violate** → modest lever; consider combining with other constraints (3-touch limit, serve-at-rally-start).
- **If < 25% violate** → game-semantics via net-cross is NOT the fix for wrong_action. Workstream closes; pivot to other levers (no_player_nearby distance-gate loosening is the next-biggest untouched lever at ~20% of FNs).

**Output:** a categorization dump at `analysis/outputs/game_semantics_probe/wrong_action_net_cross_violations.jsonl` + a summary table.

### Probe 3b — Validate the hypothesis against user-vetted FN cases

**Context:** during the 2026-04-22 session the user flagged six specific FN cases as "obviously easy" (ones the pipeline rejects at GBM confidence ≤ 0.062 despite MS-TCN++ endorsement ≥ 0.92):

| rally_id | frame | gt_action | classifier_conf | seq_peak_prob | direction_change | player_dist |
|---|---:|---|---:|---:|---:|---:|
| fb7f9c23 | 230 | receive | 0.004 | 0.919 | 3.12° | 0.103 |
| 8ce36875 | 72 | receive | 0.020 | 0.997 | 49.07° | 0.030 |
| a67c04fb | 143 | receive | 0.018 | 0.997 | 3.53° | 0.092 |
| 04ef801f | 228 | dig | 0.062 | 0.979 | 29.57° | 0.067 |
| 1a6e05d5 | 147 | serve | 0.031 | 0.992 | 9.85° | 0.145 |
| f978201e | 92 | receive | 0.010 | 0.997 | 7.99° | 0.113 |

Deep inspection of the raw ball trajectories (see `analysis/scripts/inspect_6_fn_cases.py`) showed 5 of 6 have strong physical contact signatures (direction change, speed discontinuity, or position jump). The monolithic GBM rejects them because the joint feature distribution at these values is ~50/50 contacts vs non-contacts in training. **Rally context (ball just crossed the net from opposite side, whose serve is it, what's the 3-touch count) is the missing signal that would make these acceptances decisive.**

**Probe 3b method:**
1. For each of the 6 cases, using Probes 1 + 2's net-line + per-frame ball-side output:
   - Compute the ball-side trajectory in the ±30-frame window around GT.
   - Compute the rally-state variables at GT: frames-since-serve, 3-touch-count-current-side, last-known-crossing-direction.
2. Classify each: "a rally-state-aware prior would have BOOSTED acceptance here" vs "rally state does not help."

**Validation gate (tight, because n=6):**
- **6/6 boost-eligible** → hypothesis fully validated on user-flagged evidence. Proceed with confidence.
- **4-5/6 boost-eligible** → hypothesis mostly validated; inspect the losers to understand.
- **≤ 3/6 boost-eligible** → hypothesis may not generalize; cross-check with Probe 3's broader 108-case result before committing.

**Output:** a short memo at `analysis/reports/game_semantics_probe_3b_6_cases.md` documenting the rally state at each of the 6 cases + the prior-boost verdict.

**Why this matters:** Probe 3 measures whether the LEVER exists on the wrong_action pool. Probe 3b measures whether the lever exists on the FN pool specifically — and it's validated against cases the domain expert (user) has personally vetted, making the probe's result immediately credible.

## Visual debugging tooling (required for every probe)

A mandatory deliverable regardless of gate outcome: **per-rally side-of-ball visual overlay**. A tool that takes a rally video + ball positions + net-line, and emits:

- An MP4 rendering with net-base line drawn in cyan, net-top line in yellow, and ball position colored by predicted side (red = near, blue = far).
- A per-frame timestamp overlay showing `side_pred` and `side_changed_vs_prev_frame`.

**Why:** the user explicitly said "I never visually assisted the direction." Before trusting any automated net-cross signal we need humans in the loop to sanity-check that the signal matches perception. This tool enables Probe 2's manual GT labeling and Probe 3's spot checks.

Estimated build: 3-4 hours. Location: `analysis/scripts/render_side_overlay.py`.

## Cross-workstream value — ball-side is a shared primitive

**Reliable per-frame ball-side classification (Probe 2's deliverable) feeds TWO workstreams, not just contact detection.** It is also a material lever for the player-attribution workstream (see `memory/player_attribution_day3_2026_04_22.md`):

- Today's attribution geometry is image-space distance between ball and player, without rally context. This produces misattribution when the geometrically-nearest player is on the wrong team (e.g., a ball crossing to the far side attributed to a near-side player who happens to be close in pixel space).
- With reliable ball-side + the existing team-to-side mapping from `match_tracker`, attribution can be constrained to the ball-side's team: "a contact with ball on far side must be attributed to a far-side player." This is a game-rule constraint, not a heuristic, and it should eliminate a meaningful class of wrong-player attribution errors.
- The Day-3 roadmap's Phase 4 (A5 motion-integral + game-semantics rule costs) explicitly lists "serve→opposite-team, 3-touch, net-cross, setter→teammate, block+dig coupling" as planned rule costs. **The ball-side signal is an input to those rule costs.** Building it once for contact detection and reusing it for attribution is the right factorization.

**Implication for prioritization:** Probes 1 and 2 pay for themselves even if Probe 3 kills the game-semantics contact-detection lever. The net-line detection + per-frame ball-side primitives still feed attribution Phase 4. So the minimum investment on this plan (~4 hours for overlay + 2 hours each for Probes 1, 2 = ~8 hours) produces a durable cross-workstream asset regardless of the contact-detection outcome.

**Coordination with attribution workstream:** before building the ball-side classifier, confirm with the attribution-workstream owner that its output schema matches what Day-3 Phase 4's rule-cost framework expects. The two workstreams should agree on (a) ball-side definition (net-top line vs net-base line), (b) sidedness semantics under ambiguity (at-net ball), (c) storage format if the signal is cached per rally.

## 3D reconstruction — should we consider it?

User reported a competitor doing full 3D reconstruction on the same domain, same challenges. That's a different architectural bet:

- **Pros:** unambiguous ball-side, ball-height, net-cross timing, could land on z=y=x rally reconstruction.
- **Cons:** monocular 3D ball reconstruction is an open research problem in beach volleyball specifically (per prior `ball_3d_phase_c_audit_2026_04_11.md` memory — ABANDONED). Requires either stereo / multi-camera, depth estimation model, or strong geometric priors (known ball size + camera intrinsics + ballistic trajectory model between contacts).

**Recommendation for THIS scoping:** stay 2D for Probes 1-3. If Probe 3 passes the 40% gate but the residual 60% is caused by ambiguous side decisions (ball near the net, pixel-Y ambiguous), THEN open a separate 3D-reconstruction scoping as a follow-up. Don't block on 3D here.

## Implementation plan — TRIGGERED ONLY BY Probe 3 pass

Only write this after Probe 3 returns. Sketched here so the scoping is complete.

### Phase A — wire net-cross signal into the candidate decoder

1. Add `ball_side_at_frame(frame) -> "near" | "far"` to the decoder runtime as a per-frame feature.
2. Add `ball_side_changed(frame_a, frame_b) -> bool` lookup.
3. Modify `TransitionMatrix.default()` to ADD a penalty on transitions that violate net-cross (e.g., attack_far → dig_near without an intervening near_contact is allowed; attack_far → set_far without a far-side receive/dig first is penalized).

### Phase B — serve-prior at rally start

When `frames_since_rally_start < 90` (3s at 30fps), boost serve candidate acceptance threshold downward by Δ — but only if candidate's `direction_change_deg` < 30 AND ball is on exactly one side for ≥ 1s before the candidate. Addresses the 113 serve FNs specifically.

### Phase C — 3-touch limit as hard constraint

Track contact counts per rally side since last net-cross. If ≥ 4 accepts on one side without crossing, downgrade the 4th+ confidence to the bottom quartile. Rare but would rescue ~2-5 edge cases.

### Gates per phase

Each phase is a separate A/B against v5. Pre-registered:
- Phase A: F1 Δ ≥ +0.3pp, Action Acc Δ ≥ +0.2pp, no fold > 0.8pp regression.
- Phase B: FN-serve count Δ ≤ −10 absolute, F1 Δ ≥ 0pp.
- Phase C: Contact-count-over-3 count Δ ≤ −3 absolute, F1 Δ ≥ 0pp.

## Dependencies / prior work to consult

- `memory/net_cross_homography_2026_04_15.md` — **z=0 projection** NO-GO. Don't re-run THAT variant; Probe 3 above uses 2D pixel net-line, architecturally different.
- `memory/ball_3d_phase_c_audit_2026_04_11.md` — 3D ball reconstruction ABANDONED. Relevant if Probe 3 passes but 2D ambiguity dominates the residual.
- `memory/landing_heatmap_2d_brief_2026_04_12.md` — shipped 2D heatmaps for landing. Same infrastructure (court calibration, image projection) is reusable for net-line detection.
- `memory/crop_head_phase2_nogo_2026_04_20.md` — crop-head specialist for block is NO-GO for swap but +10.69 block F1 as class specialist. If Phase A ships, the crop-head could be reconsidered as a BLOCK-ONLY specialist within the game-semantics layer (since blocks are constrained to one side of the net by rule).

## What NOT to do (tired-engineer traps)

- Do NOT build the net-line detection before Probe 1 confirms it's reliable.
- Do NOT build the direction signal into the pipeline before Probe 3 says yes.
- Do NOT reuse z=0 court-plane projection for net-cross (already NO-GO).
- Do NOT skip the visual overlay tool — it's the only way to sanity-check the signal with a human in the loop.
- Do NOT combine Probe 1, 2, 3 into one mega-diagnostic — the gates are sequential and the cost of merged-probe debugging exceeds the cost of the separate probes.
- Do NOT open 3D reconstruction in the same workstream — separate it behind a success gate.

## Success criteria for the scoping session

**Minimum:** Probes 1 + 2 + 3 run, results logged, kill-gate outcomes clear, visual overlay tool committed.
**Target:** Probe 3 passes the 40% gate → implementation plan for Phase A signed off.
**Stretch:** Phase A prototype landed and passes its pre-registered gate.

## Appendix — kickoff prompt for the next session

```
Read docs/superpowers/plans/2026-04-22-game-semantics-scoping.md.
Run Probes 1 → 2 → 3 → 3b in order. Skip later probes if earlier ones kill.
Probe 3b validates the hypothesis on 6 user-flagged FN cases and is the
immediate-credibility check on top of Probe 3's broader 108-case result.

Deliverable: go/no-go verdict on game-semantics lever + visual overlay tool
+ the Probe 3b 6-case rally-state memo.

Hard rules:
- Build the visual overlay tool FIRST (4 hours). All probes depend on it.
- Honor each probe's kill gate. Do NOT rationalize a miss into a pass.
- Do NOT write Phase A/B/C code in this session. Only Probes.
- If Probe 3 passes but 3b says ≤3/6 on user cases, STOP and reconcile
  before building. The user-vetted evidence is load-bearing; we cannot
  ship an implementation that doesn't fix cases the domain expert
  explicitly flagged.
- If Probes pass overall, author a separate implementation brief; do
  not start coding.
- If any probe kills, write a NO-GO memo naming the root cause.
```
