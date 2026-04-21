# Contact FN Phase 5 — Per-category hypotheses + surgical-fix proposals

**Phase 5 deliverable** of the contact-detection full review (brief: `docs/superpowers/briefs/2026-04-21-contact-detection-full-review.md`).

**Scope.** For each category from Phase 4, propose the smallest possible change to existing code (threshold tweak, parameter change, targeted rule — NOT a new component), with **integration-level** pre-registered gates. No implementation — the deliverable is this plan. Phase 6 decides whether the collective Path-A surgical fixes close the 92% F1 / 96% Action Acc gap, or whether Path B architectural work is needed.

**Hard rules (from brief).**
- Integration-level validation only (candidate-level metrics misled the previous session — +17pp cand F1 → -4.82pp integration F1).
- Pre-registered gates; honor them. Marginal result = NO-GO.
- No complexity layered without evidence each layer is needed.
- Brief §traps-to-avoid line 53: **do NOT try variants of the `gbm<0.10 AND seq≥0.95` rescue rule**; it and the pattern family were tested NO-GO.

**Target gap.** Baseline 88.15% F1 / 93.15% Action Acc. Target 92% F1 / 96% Action Acc. Gap: +3.85pp F1, +2.85pp Action Acc.

---

## Current baseline arithmetic (grounding for Phase 6)

```
total GT:            2095
current TP:          1782
current FN:            313
current FP:            166
current wrong_action:  122
F1 = 2·P·R / (P+R) = 88.15%
Action Acc = (TP - wrong_action) / TP = 93.15%
```

**To reach 92% F1:** need to recover ~88 FN or reduce ~76 FP (approximately — exact depends on precision/recall balance). Reducing only FN: if TP→1870 (recover 88), FP unchanged → F1 ≈ 92.2%.

**To reach 96% Action Acc:** reduce wrong_action from 122 to ~71 (recover 51 wrong_action → correct). Out of scope for this review (brief de-scopes action re-labeling), but catching contacts currently lost AS FNs into wrong_action would trade one error class for another.

**Implication.** Categories recovering ≥10 FNs are material. Category 7 (84), Cat 6 (34), Cat 2 (31) are the three levers that could individually move F1 meaningfully. Smaller categories contribute incrementally.

---

## Priority 1 — Category 7 (Dedup-kill) — 84 FNs / 35%

### Status before proposing a fix

**What we know:** 9/9 Phase-2 sample had `classifier_accepted=True` + `dedup_survived=False` with gbm 0.35–0.97 and correct MS-TCN++ class. Dedup winner (the candidate that caused the elimination) is **not visible in the final pred** within 7f of GT.

**What we do NOT know:** The dedup winner's identity, frame, confidence, and why it disappears from the final pred. This is critical — the surgical fix depends on which sub-mechanism is dominant.

### Phase 5 precondition — "dedup winner trace" diagnostic

**Before any fix can be designed, run an instrumented `_deduplicate_contacts` diagnostic that captures, for each dedup-elimination FN:**

1. The specific candidate that won dedup (its frame, confidence, court_side, bbox-derived player_track_id).
2. All candidates eliminated against it (their features).
3. The eventual fate of the winner: final pred? matched to another GT? filtered by action_classifier's pre-serve-FP rule? dropped by phantom-serve logic?

**Deliverable:** `analysis/scripts/trace_dedup_winners.py` runs over all 84 Category 7 FNs, produces `analysis/outputs/dedup_winner_trace_2026_04_21.jsonl` with one row per eliminated candidate + its winner's state. ~1-2 hours scripting.

**Until this diagnostic runs, we cannot distinguish the 5 hypothesized sub-mechanisms (H7a-e from Phase 3).** Proposing a fix without it risks the Phase 2 lesson in reverse: we'd pick a plausible intervention and not know which sub-population it helps vs hurts.

### Candidate surgical fixes (conditional on trace results)

These are exploratory proposals — validate sub-mechanism distribution via trace FIRST, then commit to 1 fix.

#### Fix A — Adaptive dedup: treat `court_side == "unknown"` as cross-side

**Code site:** `contact_detector.py:1770-1788` (`_deduplicate_contacts` adaptive block).

**Current behavior:**
```python
if sides_known and contact.court_side != existing.court_side:
    effective_min = _CROSS_SIDE_MIN_DISTANCE  # 4
elif both-confident:
    effective_min = _CROSS_SIDE_MIN_DISTANCE
else:
    effective_min = min_distance  # 12
```

`sides_known` requires BOTH contacts to have `court_side ∈ ("near", "far")`. If either is "unknown" (which happens when `team_assignments` lacks the track_id and calibration fails), it falls through to same-side 12f.

**Hypothesis H7c verification target:** if `court_side="unknown"` on at least one contact in ≥30% of Cat 7 dedup pairs (from trace), the same-side fall-through is contributing.

**Proposed change:** when one contact has `court_side="unknown"`, resolve the winner's court_side from the eliminated candidate's side (or vice versa) and apply adaptive distance. Or: treat pairs with `near` vs `unknown` / `far` vs `unknown` as cross-side. **Tiny 3-line change to the existing conditional.**

**Pre-registered integration gates:**
- F1 Δ ≥ +0.5pp (recover ≥12 FNs) at fold-average over 68 folds.
- NO regression on any class's per-class precision below −1.0pp.
- FP increase ≤ +10 absolute (~+6% of current 166 FPs).
- **NO-GO conditions:** any regression > 1.0pp F1 on any fold's action_acc, or FP increase > 20.

#### Fix B — Confidence bypass for classifier-accepted pairs near net-crossing

**Code site:** `contact_detector.py:1742` (`_BOTH_CONFIDENT_FLOOR = 999.0` — currently disabled).

**Hypothesis:** 9/9 Cat 7 cases had the eliminated candidate's classifier accept ≥ 0.35. If the surviving-winner also had gbm ≥ 0.35, setting `_BOTH_CONFIDENT_FLOOR = 0.5` (say) would keep both.

**Known prior result (line 1739-1742 comment in code):** "Confident dedup bypass tested (0.50/0.80 → skip same-side dedup when both contacts confident). Result: adds 195+ FPs because many FP candidates have high confidence. NOT shipped."

**Proposed refined variant:** add a geometric disambiguator — keep both IFF classifier-confident AND `|ball_y_1 - ball_y_2| ≥ 0.05` (contacts spatially separated by ≥5% screen vertical distance) OR `is_net_crossing=True` on either candidate. This tightens the filter that made the earlier attempt fail.

**Pre-registered integration gates:**
- F1 Δ ≥ +0.5pp.
- FP increase ≤ +20 absolute (tighter than Fix A because this lever historically added 195 FPs; we must prove the geometric tightener eliminates most of that).
- NO-GO: FP increase > 50, or regression > 1.0pp F1 on any fold.

**This fix is lower-priority than Fix A until the trace shows H7e (generic same-side eviction) is the dominant sub-mechanism.**

#### Fix C — Pre-serve FP filter relaxation

**Code site:** `action_classifier.py:1621-1622` pre-serve FP drop.

**Hypothesis:** if a serve's dedup winner is another pre-serve candidate (e.g., ball-toss peak), and that winner is then pre-serve-filtered at action_classifier, BOTH contacts are lost. Relaxing the pre-serve filter might rescue some cases. But this is a wrong_action risk — brief de-scopes.

**Defer.** Do not propose without trace evidence that pre-serve filtering is the downstream killer for dedup winners in serve cases.

### Phase 5 commitment for Category 7

- **Commit to trace diagnostic.** Pre-conditions Fix selection.
- **Fix A is the most likely candidate** (smallest, most targeted, no prior NO-GO stigma).
- **Fix B is a contingency** if trace shows H7e dominates.
- Fix C deferred pending evidence.

### Estimated FN recovery (if fixes land at their gates)

Pre-registered gate = ≥12 FN recovered. Realistic upper bound if Fix A handles H7c cases (maybe 40-60% of the 84) = ~35 FN. Combined (A + B if both land) = up to ~50 FN. This alone could move F1 ~+1.5 to +2.0pp.

---

## Priority 2 — Category 6 (Ball-gap ≥ 4f) — 34 FNs / 14%

### Status

**What we know:** 34 FNs have `ball_gap_frames ≥ 4` + no candidate at or near GT. 22 of 34 are serves (rally-start WASB lag); 12 are non-serve (mid-rally ball loss).

**What we don't know:** per-case — is WASB genuinely missing detections (below whatever internal confidence it uses), or are detections being dropped because confidence < `_CONFIDENCE_THRESHOLD=0.3`?

### Phase 5 precondition — raw-WASB-confidence inspection

**Before a fix: run a short diagnostic that reports, for each Cat 6 FN, WASB's raw per-frame confidence in the gap window (GT ± 10f).**

- Script: `analysis/scripts/inspect_wasb_gaps.py` — reads raw WASB inference (before the 0.3 gate) for each Cat 6 rally at the GT frame window.
- Output: distribution of raw confidences. If many are 0.1–0.3 (below gate, above noise), fix is "lower threshold + add candidate-level gate." If all are ~0 (genuine miss), fix is "re-train WASB on harder cases" (Path D, out of this review's Path A scope).

### Candidate surgical fix

#### Fix D — Lower `_CONFIDENCE_THRESHOLD` at ball_by_frame construction point, WITH a bridge guard

**Code site:** `contact_detector.py:39` (`_CONFIDENCE_THRESHOLD = 0.3`) and call site at line 1966-1970 (`ball_by_frame` comprehension).

**Hypothesis:** if raw WASB confidences in the gap window are 0.1-0.3, those are real detections currently discarded. Lowering the floor to 0.15 admits them, letting the candidate generators fire.

**Risk:** lower threshold admits more noisy positions → more FP candidates. Must gate integration-level, not candidate-level.

**Proposed concrete change:**
- Change `_CONFIDENCE_THRESHOLD` from 0.3 → 0.15 **only for the gap-bridging logic** — add a `_CONFIDENCE_THRESHOLD_BRIDGE = 0.15` applied conditionally when `ball_gap_frames` in the surrounding ±10f window would exceed 5 at the default threshold. This keeps 0.3 as the default precision floor and only relaxes it where stage 1b interp would otherwise fail.

**Pre-registered integration gates:**
- F1 Δ ≥ +0.3pp.
- FP increase ≤ +15.
- NO-GO: any fold regresses > 0.8pp F1, or FP increase > 30.

#### Fix E — Reduce `max_interpolation_gap` from 5 → 7 at stage 1b

**Code site:** `ball_filter.py:218` (`max_interpolation_gap=5`).

**Hypothesis:** if WASB reliably detects the ball on both sides of longer gaps, extending interpolation to 7 frames recovers some Cat 6 cases. BUT: longer linear interp also erases real deflections more broadly (Mode ζ risk — Cat 1 would get WORSE).

**Risk of interaction with Cat 1:** Cat 1 is already 5 FNs corpus-wide; Mode ζ was the mechanism. Extending the interp window would likely convert some Cat 6 cases to Cat 1 — we'd swap one FN type for another, potentially net-negative.

**Recommendation:** **do not propose Fix E standalone.** Only propose if paired with a generator improvement that fires on the interp-bridge edges (different Path B architectural change).

### Phase 5 commitment for Category 6

- **Commit to raw-WASB-confidence inspection.** If results support Fix D's premise (raw confidences 0.1-0.3 in the gaps), proceed with Fix D. If not, Cat 6 has no Path A surgical fix; document as ceiling and move on.
- Fix E not recommended.

### Estimated FN recovery (if Fix D lands)

Aggressive: ~10 FNs (30% of Cat 6) — the 22 serve cases are a known-hard area (brief §closed). Conservative: ~5-8 FNs. F1 lift: ~+0.3-0.5pp.

---

## Priority 3 — Category 2 (Kin-underreports) — 31 FNs / 13%

### Status

**What we know:** 31 classifier-rejection FNs with `dir_change ≤ 30°`, `player_distance ≤ 0.15`, `seq_peak ≥ 0.85`, `gbm < 0.30`.

**What we don't know:** per-case, WHICH feature is dominating the GBM rejection. The hypothesis is `direction_change_deg`, but `ball_detection_density` (brief §traps z-score -2.4 to -4.2) or pose features (0.0 when keypoints missing) could be the real drivers.

### Phase 5 precondition — per-feature SHAP/staged-decision attribution

**Before a fix: run `model.staged_decision_function` (or manual per-tree decomposition) on each Cat 2 FN's feature vector, producing per-feature log-odds contribution.**

- Script: `analysis/scripts/decompose_gbm_rejections.py` — for each Cat 2 FN's `CandidateFeatures` vector, compute per-feature contribution to the final log-odds. Rank features by negative contribution on rejections.
- Output: histogram of "most-negative-feature" per FN. If 70%+ are `direction_change_deg`, Fix F below. If `ball_detection_density` dominates, Fix G. If mixed, no clean surgical fix.

### Candidate surgical fixes

#### Fix F — Use a narrower `direction_check_frames` window in feature extraction

**Code site:** `contact_detector.py:421` (`compute_direction_change(..., check_frames=8)`) and its usage in candidate features.

**Hypothesis:** `direction_check_frames=8` (Phase 1 doc) averages pre/post contact positions across a too-wide window, smoothing out sharp real deflections. A shorter window (say 3-5 frames each side) would see the actual angle.

**Concrete change:** expose `direction_check_frames` in `ContactDetectionConfig`, set default to 5 for the classifier feature but keep 8 for candidate generator G (which needs prominence across a wider window to compete against neighbors). Feature-specific parameter.

**Risk:** shortening the window also makes the feature noisier on genuine FP candidates (trajectory noise jitter produces angle spikes in short windows). Net impact requires integration test.

**Pre-registered integration gates:**
- F1 Δ ≥ +0.3pp on fold-average.
- NO-GO: per-class precision regresses > 1.0pp on any class.
- NO-GO: FP increase > 15.

#### Fix G — Adjust `ball_detection_density` window or computation

**Code site:** `contact_detector.py:2288-2294` (density computed over ±10f window at candidate frame).

**Hypothesis (if attribution shows density dominates):** the ±10f density captures noise when WASB goes sparse around contact (momentary hand occlusion), conflating tracker-noise with contact-event. A shorter window (±5f) might correlate less with the specific "contact-momentary-occlusion" failure mode.

**Deferred until attribution results.** If density is not the dominant feature, don't touch it.

### Phase 5 commitment for Category 2

- **Commit to per-feature attribution diagnostic.**
- If direction_change dominates (>70%), proceed with Fix F.
- If density dominates, consider Fix G but with extreme caution — brief §traps-to-avoid line 206 warns density is a correlation, not a mechanism.
- If mixed / no single feature dominates, Cat 2 has **no surgical fix at feature level**; document as classifier-ceiling area and consider architectural (Path B): richer per-candidate emission (the brief's "per-candidate crop head" already-backlogged workstream).

### Estimated FN recovery (if Fix F lands)

Aggressive: ~15 FNs (50% of Cat 2) if direction_change is the dominant negative feature. Conservative: ~5-10 FNs. F1 lift: ~+0.2-0.5pp.

---

## Smaller category assessments (Path-A proposals gated by leverage)

### Category 4 (Dual-occlusion) — 13 primary + 4 Cat-7-overlap = 17 total

**Phase 3 observation:** `player_distance=inf` forces classifier's feature to 1.0 (line 93 of `contact_classifier.py:to_array`), which is a strong FP-like value.

**Surgical proposal — Fix H:** substitute `player_distance=inf` with a neutral value (the training-set median for matched TPs) instead of 1.0. Small code change.

**Gate:** F1 Δ ≥ +0.15pp, NO-GO if FP increase > 10.

**Estimated recovery:** ~5 FNs. Low leverage but cheap.

### Category 5b (Serve cand-gen other) — 17 FNs

**Brief §closed 2026-04-20:** "Serve candidate generator NO-GO... 53/59 Mode C serves are a WASB ball-tracker emergence-recall problem, not a generator problem. Future serve work → ball tracking / pose-based seeding."

**Phase 5 stance:** accept as ceiling; no Path A fix. Path D (more training data) or Path B (pose-seeded candidate generator) is the documented next step. Would recover up to 17 FNs at significant investment.

### Category 1, 3, 5, 8, 10, seq-signal — total 25 FNs

Each has 1-5 FNs; individual Path A fixes have negligible leverage. Accept as ceiling for this review.

---

## Consolidated Path A surgical-fix program

| Fix | Category | Pre-fix diagnostic | Gate F1 Δ | Est. FN recovery |
|---|---|---|---|---|
| **A** — adaptive dedup with unknown-side | Cat 7 (84) | dedup winner trace | +0.5pp | ~35 |
| **B** — confident-dedup-bypass + geo-guard (contingent) | Cat 7 (84) | dedup winner trace | +0.5pp | ~15 |
| **D** — conditional WASB threshold bridge | Cat 6 (34) | raw-WASB-conf inspection | +0.3pp | ~10 |
| **F** — narrower direction_check window | Cat 2 (31) | per-feature GBM attribution | +0.3pp | ~15 |
| **H** — neutralize player_distance=inf | Cat 4 (17) | — | +0.15pp | ~5 |

**If all 5 fixes land at gate minima:** ~80 FN recovered → F1 Δ ~+1.6pp (to ~89.7%).

**If Fix A + D + F + H land at realistic (non-minimum) recovery:** ~65 FN → F1 Δ ~+1.3pp (to ~89.4%). Still short of 92% target by ~2.6pp.

### Brutal gap assessment

**Path-A fixes alone do NOT close the +3.85pp gap to 92% F1.** Even aggressive assumptions top out ~90%. This matches the brief's design: the cases left after Path A are the ones requiring architectural work (Path B) or more training data (Path D).

**The residual gap after Path A is ~2.6pp = ~55 FNs**, concentrated in:
- 17 serve cand-gen cases (Cat 5b) — Path D/B documented, accepted-ceiling.
- 22 `u-classifier_other` + 7 `u-candidate_gen_other` — no coherent single mechanism; classifier-ceiling.
- Residual Cat 7 cases not addressed by Fix A/B (~35-45 of the 84).
- Residual Cat 6 cases not addressed by Fix D (~20 of 34).

---

## Deferred to Phase 6 (gap-assessment memo)

Phase 5 commitments to run before Phase 6:

1. **Dedup winner trace** (Cat 7 precondition) — 1-2 hours scripting.
2. **Raw-WASB-confidence inspection** (Cat 6 precondition) — 30 min.
3. **Per-feature GBM attribution on Cat 2** — 30 min.

After those three diagnostics run, Phase 6 can:
- Commit to (or reject) Fix A/B/D/F/H based on diagnostic evidence.
- Quantify cumulative Path-A recovery.
- Explicitly document the residual gap and whether Path B architectural work is justified (brief: "Architectural rework proposals need evidence: a specific stage that cannot close its residual gap surgically").

---

## What Phase 5 explicitly does NOT propose

Per brief §never-do and §traps-to-avoid:

- **No seq-only rescue variants** (brief line 53 — tested NO-GO on the +17pp candidate → −4.82pp integration lesson). Even though 8/9 Cat 7 cases and most Cat 2 cases have `seq_peak ≥ 0.85`, we do not propose rescuing candidates on seq-endorsement alone.
- **No candidate-frame offset rescue** (brief line 53 — 3.8% rescue at GT frame tested).
- **No `seq_max_nonbg` as a candidate-injection mechanism** (brief line 53 — "variants of this pattern" are banned).
- **No architectural work** in Phase 5 — surgical fixes only. Architectural proposals are Phase 6's territory IFF Path A cannot close the gap.
- **No fix for wrong_player** (brief de-scoped).
- **No fix for wrong_action** (brief de-scoped for this review).
