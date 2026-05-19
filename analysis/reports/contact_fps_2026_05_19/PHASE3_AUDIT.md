# Phase 3 — action_classifier rule audit: prioritized punch list

**Date:** 2026-05-19
**Scope:** `analysis/rallycut/tracking/action_classifier.py` (4197 lines) + `action_type_classifier.py` (872 lines). 5069 lines total.
**Goal:** Enumerate every rule, gate, and heuristic that depends on contact-frame positions, so a future workstream can redesign them for fps-invariance.

## Why this audit

Phase 1/2/2.5 (see `PHASE2_5_FINDINGS.md`) proved that retraining `contact_classifier` + downstream models doesn't close the −0.6pp Action Acc regression on trusted-31. The regression is invariant across three different retrain configurations — it just moves between action types (Phase 2 hit Dig; Phase 2.5 hit Serve/Receive). Hypothesis: rule-based logic in `action_classifier.py` is calibrated to v4-pipeline contact-frame distributions. This audit identifies the candidate rules.

## Top finding: Rule L is the prime suspect

The audit surfaced a hypothesis I hadn't considered: **`apply_sequence_override` (line 4187-4189) reads MS-TCN++ per-frame probabilities at the contact's frame index.** If MS-TCN++ was trained with peaks aligned to v4-pipeline contact frames, and we now query at NEW-pipeline frames (~3f earlier), we're sampling off-peak — MS-TCN++ may indicate the wrong action type because its peak is at frame+3 not frame.

This would explain the structural −0.6pp Action Acc regression: it's NOT a downstream model that needs retraining; it's a **cross-model frame-index phase shift** that no amount of retraining the contact/action/scorer models can fix. The fix is to either (a) retrain MS-TCN++ on the new contact frames or (b) make `apply_sequence_override` look at MS-TCN++ in a ±N window around the contact frame (not just the exact frame).

Validating this hypothesis is the highest-EV next probe.

## Prioritized rule punch list

### HIGH PRIORITY — Likely drivers of the Action Acc regression

#### Rule L: `apply_sequence_override` MS-TCN++ frame indexing
- **Location:** `action_classifier.py:4187-4189` (call site); `sequence_action_runtime.py` (impl)
- **What:** Replaces non-serve action types with MS-TCN++ per-frame argmax, indexed at the contact frame.
- **Sensitivity:** Implicit — `sequence_probs[:, contact.frame]` indexes MS-TCN++ at the contact's frame. If MS-TCN++ peak is calibrated to v4-pipeline frame placement, querying at new-pipeline frames samples off-peak.
- **Failure mode:** 3-frame phase shift between GBM contact frame and MS-TCN++ peak → wrong action argmax → mislabeled actions.
- **Fix:**
  - **WINDOW**: sample `max(sequence_probs[:, frame±5])` instead of `sequence_probs[:, frame]`
  - **RELATIVE_REORDER**: use ordinal contact index, not raw frame
  - **RETRAIN_MS_TCN**: retrain on new contact frames (but expensive; ms_tcn is the long-training model)
- **Risk if fixed wrong:** HIGH. apply_sequence_override is one of the highest-impact rules (per memory v5+v9+v11 history, large F1 swings tied to MS-TCN++).
- **Validation:** A/B with window vs exact-frame indexing on Phase 2.5 trusted-31 corpus. Quick — no retraining needed.

#### Rule A: `block_max_frame_gap` (Block detection)
- **Location:** `action_classifier.py:1754-1760`
- **What:** Marks contact as BLOCK if at net, ≤8 frames after an attack, on opposite court side.
- **Sensitivity:** RELATIVE_FRAME_DISTANCE (`<= 8`)
- **Constants:** `block_max_frame_gap = 8` (~267ms at 30fps, ~133ms at 60fps)
- **Failure mode:** At 60fps the gate covers half the physical time → blocks 9-15 frames after attack are missed.
- **Fix:** `PHYSICAL_TIME` — convert to `(contact.frame - prev.frame) / fps <= 0.3`. Or `RELATIVE_REORDER` — "next at-net contact after attack on opposite side, regardless of frame distance."
- **Risk:** MEDIUM. Block F1 is already 0% on trusted-31 (baseline issue), so widening the gate could help without making things much worse.

#### Rule B: `serve_window_frames` (Serve in-window detection)
- **Location:** `action_classifier.py:1723-1725, 1859, 2003-2005` (3 call sites)
- **What:** Limits heuristic serve detection to first 60 frames of rally; serves outside drop confidence.
- **Sensitivity:** ABSOLUTE_FRAME (`<= 60` from rally_start_frame)
- **Constants:** `serve_window_frames = 60` (~2s at 30fps, ~1s at 60fps)
- **Failure mode:** At 60fps the serve window covers half the physical time → late serves outside window misclassified.
- **Fix:** `PHYSICAL_TIME` — `(c.frame - start_frame) / fps <= 2.0`. Or `RELATIVE_REORDER` — "serve is the first contact, full stop."
- **Risk:** MEDIUM. Serve detection anchors the action sequence; loosening risks precision.

#### Rule C: Pass 3 serve fallback `flight_window`
- **Location:** `action_classifier.py:1900-1943`
- **What:** Validates Pass 3 (first-contact-is-serve) by checking pre-contact ball trajectory originates from server's side.
- **Sensitivity:** ABSOLUTE_FRAME (`contact.frame - 45`)
- **Constants:** `flight_window = 45` (~1.5s at 30fps, ~0.75s at 60fps)
- **Failure mode:** At 60fps the window samples half the trajectory → flight-origin check less reliable.
- **Fix:** `PHYSICAL_TIME` — `contact.frame - int(1.5 * fps)`.
- **Risk:** MEDIUM. Pass 3 is a fallback so impact is limited, but Phase 2.5 Receive −2.6pp suggests this gate matters.

#### Rule D: Time-gap sanity guard in repair-rule main pass
- **Location:** `action_classifier.py:2657-2661`
- **What:** Disables same-side repair rules (1, 2, 5, 6) when gap between consecutive contacts > 90 frames.
- **Sensitivity:** RELATIVE_FRAME_DISTANCE (`> 90`)
- **Constants:** `> 90` frames (explicit comment "3s at 30fps")
- **Failure mode:** At 60fps, 3-second gaps span 180 frames, never trigger the disable → repair rules fire on long-gap rallies that should be marked uncertain.
- **Fix:** `PHYSICAL_TIME` — `(a.frame - prev.frame) / fps > 3.0`.
- **Risk:** LOW. Conservative gate; over-firing rules is the failure mode, not under-firing.

### MEDIUM PRIORITY — Structural but indirect

#### Rule E: Synthetic serve placement fallback
- **Location:** `action_classifier.py:1595-1603`
- **What:** Places synthetic serve at `max(0, first_contact - 30)` or rally_start when first-contact gap ≤90.
- **Sensitivity:** RELATIVE_FRAME_DISTANCE + ABSOLUTE_FRAME (`-30` offset, `<= 90` threshold)
- **Constants:** `-30` (~1s), `<= 90` (~3s)
- **Failure mode:** At 60fps the synthetic placement is ~0.5s before first contact (intended ~1s); affects `team_transition` feature.
- **Fix:** `PHYSICAL_TIME` — `int(1.0 * fps)` offset, `(first - start) / fps <= 3.0` threshold.
- **Risk:** LOW. Synthetic serves are low-confidence fallbacks.

#### Rule J: Dynamic-scorer post-contact ball alignment offset
- **Location:** `action_classifier.py:3810-3815`
- **What:** Looks 5-15 frames post-contact for ball position to compute `wrist_post_alignment` feature.
- **Sensitivity:** RELATIVE_FRAME_DISTANCE (`range(5, 16)`)
- **Constants:** `5, 16` frames (~167-500ms at 30fps; ~83-250ms at 60fps)
- **Failure mode:** At 60fps the window is half the physical time → feature distribution shifts → scorer trained on 30fps misranks candidates.
- **Fix:** `PHYSICAL_TIME` — `range(int(0.17*fps), int(0.5*fps))`. Requires scorer retrain after change.
- **Risk:** MEDIUM. Changing feature window without retraining the scorer breaks it; coordinated change.

#### Rule K: Viterbi temporal smoothing (HMM transition costs)
- **Location:** `action_classifier.py` (viterbi_decode_actions — not deeply inspected)
- **What:** HMM smooths action sequence by penalizing unlikely transitions over N frames.
- **Sensitivity:** Implicit in transition cost matrix (frame-distance dependent)
- **Failure mode:** If HMM was trained on v4-pipeline frame distributions, new pipeline produces different inter-action gaps, suboptimal smoothing.
- **Fix:** Retrain HMM, OR convert transition costs to physical time.
- **Risk:** MEDIUM. Viterbi is learned; retraining is the standard fix.

### LOW PRIORITY — Marginal impact

#### Rule F: `late_entry_scan_frames`
- **Location:** `action_classifier.py:1299-1305`
- **What:** Scans 200 frames for late-entering near-side player (server toss timing).
- **Sensitivity:** ABSOLUTE_FRAME
- **Constants:** `200` (~6.7s at 30fps)
- **Failure mode:** At 60fps the window covers half the time → late entries past 100 frames missed.
- **Fix:** `PHYSICAL_TIME` — `int(6.7 * fps)`.
- **Risk:** LOW. Secondary signal; small documented impact.

#### Rule G: `min_first_frame=15`, `max_first_frame=window_frames`
- **Location:** `action_classifier.py:1337-1340`
- **What:** Late-entry detection only for players appearing between frames 15-120.
- **Fix:** `NO_OP`. Empirical validation showed near-zero sensitivity ("+2 fixes, 0 regressions" on 401-rally tune).

#### Rule H: `_find_server_by_position(window_frames=45)`
- **Location:** `action_classifier.py:456`
- **What:** First 45 frames of rally analyzed for server position (player furthest from net).
- **Sensitivity:** ABSOLUTE_FRAME
- **Fix:** `PHYSICAL_TIME` — `int(1.5 * fps)`. Or `NO_OP` (player positions are independent of contact-frame shifts).
- **Risk:** LOW.

#### Rule I: `formation_window_frames=120`
- **Location:** `action_classifier.py:381, 4171`
- **What:** Formation-based serving team prediction uses first 120 frames.
- **Sensitivity:** ABSOLUTE_FRAME — but independent of contact frames (uses player positions only).
- **Fix:** `NO_OP` for the contact-frame-shift concern. Separately, `PHYSICAL_TIME` would be ideal for fps-invariance.

## Summary table

| Rule | File:Lines | Constant | Sensitivity | Priority | Risk | Proposed Fix |
|------|-----------|----------|-------------|----------|------|--------------|
| **L** | 4187-4189 | (implicit MS-TCN frame index) | INDEX_PHASE | **HIGH** | HIGH | WINDOW (±5f) or retrain ms_tcn |
| **A** | 1754-1760 | `<= 8` | RELATIVE | HIGH | MED | PHYSICAL_TIME or RELATIVE_REORDER |
| **B** | 1723-2005 | `<= 60` | ABSOLUTE | HIGH | MED | PHYSICAL_TIME or RELATIVE_REORDER |
| **C** | 1900-1943 | `45` | ABSOLUTE | HIGH | MED | PHYSICAL_TIME |
| **D** | 2657-2661 | `> 90` | RELATIVE | HIGH | LOW | PHYSICAL_TIME |
| **E** | 1595-1603 | `-30`, `<= 90` | RELATIVE+ABSOLUTE | MED | LOW | PHYSICAL_TIME |
| **J** | 3810-3815 | `range(5,16)` | RELATIVE | MED | MED | PHYSICAL_TIME + scorer retrain |
| **K** | (viterbi) | HMM costs | INDEX_PHASE | MED | MED | retrain HMM or PHYSICAL_TIME |
| F | 1299-1305 | `200` | ABSOLUTE | LOW | LOW | PHYSICAL_TIME |
| G | 1337-1340 | `15, 120` | ABSOLUTE | LOW | very low | NO_OP |
| H | 456 | `45` | ABSOLUTE | LOW | LOW | PHYSICAL_TIME or NO_OP |
| I | 381, 4171 | `120` | ABSOLUTE | LOW | very low | NO_OP (independent of contact frames) |

## Recommended execution order

**Stage 1 (cheapest probes, validate the hypothesis):**
1. **Rule L probe** — modify `apply_sequence_override` to look at MS-TCN++ in `frame±5` window instead of exact frame. Re-run trusted-31 A/B with new contact_classifier + new regressor + new action_classifier + new scorer + this Rule-L fix. If Action Acc closes the gap, Rule L is the primary culprit; ship.
2. If Rule L probe doesn't close the gap, escalate to Stage 2.

**Stage 2 (coordinated rule fixes):**
1. Rules B, C, D, E — convert all to `PHYSICAL_TIME` semantics (need fps parameter threaded into `classify_rally_actions`).
2. Rule A — convert to `PHYSICAL_TIME` AND test `RELATIVE_REORDER` variant.
3. Rule J — convert to `PHYSICAL_TIME` and retrain dynamic_attribution_scorer.
4. Coordinated retrain (contact + action + scorer) + trusted-31 A/B.

**Stage 3 (if Stage 1+2 still don't close the gap):**
1. Rule K — retrain Viterbi HMM on new contact frames.
2. Consider per-fps separate model paths.

**Estimated effort:**
- Stage 1: ~2 hours (one rule change + eval)
- Stage 2: ~1 day (5 rule changes + coordinated retrain + eval)
- Stage 3: ~2-3 days (HMM retraining)

## Why this matters

The Phase 2.5 finding said "the fix is action_classifier rule redesign" but didn't enumerate the rules. This audit is the input: a concrete punch list of 12 rules with priorities, fix approaches, and risks. The biggest unlock is Rule L — it's both the most likely culprit AND the cheapest to test (no retraining, just changing the index lookup to a windowed max).

If Stage 1 (Rule L probe) closes the gap, we ship with:
- New contact_classifier (the candidate we preserved)
- New regressor (the candidate we preserved)
- New action_classifier (regenerable)
- New scorer (regenerable)
- One small change to `apply_sequence_override`
- Coordinated version bumps on CONTACT_PIPELINE_VERSION + ACTION_PIPELINE_VERSION

If Stage 1 doesn't help, we have evidence that retraining alone (even with Rule L fixed) isn't enough, and Stage 2 becomes the path.
