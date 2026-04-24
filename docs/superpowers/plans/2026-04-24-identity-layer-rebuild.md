# RallyCut Identity-Layer Rebuild — Architecture Plan

> **Date:** 2026-04-24
> **Driving evidence:** Day-4 ref-crop result (`memory/player_attribution_day4_2026_04_23.md`) + Phase-0/1/2/3 locked baseline (`analysis/reports/attribution_rebuild/`).
> **Validation surfaces:** 9-fixture locked baseline (468 actions), Day-4 click-GT (314 actions), low-confidence post-label subset, end-to-end latency.

---

## Context

Today: blind analysis emits player attributions at 43.8% accuracy on the 9-fixture locked baseline (`analysis/reports/attribution_rebuild/baseline_2026_04_24.json`). The Day-4 workstream proved that *post-analysis* reference-crop labeling drives click-GT direct accuracy from 33.8% to 95.22% on 8 fixtures (`memory/player_attribution_day4_2026_04_23.md`). The product needs the Day-4 workflow productionized: blind analysis runs first, the user provides 4 reference crops afterward, and identity is corrected across every action effectively instantly. Five within-team ReID sessions (4–9) have NO-GO'd ML improvements to blind pid; the architectural answer is reference-prototype authority applied through the existing two-pass match-tracker, not a per-frame Hungarian rebuild.

This plan scopes out detector-recall (22% architectural ceiling, four prior NO-GOs) and sequences the work so the highest-uncertainty bet (scratchpad replay determinism) lands first, the user-visible payoff (Day-4-style relabel) lands second, UX wires up third, and regime-3 (same-uniform teammates) is a stretch with a quantified scope-out cost.

---

## 1. Problem Decomposition

**What this layer owns.** Given the contacts, candidates, and tracks the pipeline emits, produce a `(contact → player_id 1..4)` assignment that matches GT for ≥90% of emitted actions after the user provides reference crops, and stays at blind-pid quality (today's 43.8%) when no crops are given.

**What this layer does NOT own.**
- **Contact-detector recall.** 72/468 actions (15.4%) are *missing* from the pipeline entirely (`phase1_5_composite_partition.json`). 105/468 have zero candidates. Four prior NO-GOs (VideoMAE, crop-head Phase 2, E2E-Spot, contact arbitrator — `memory/videomae_contact_nogo_2026_04_19.md`, etc.) closed this workstream; it needs a fresh architectural hypothesis. This plan scopes it out cleanly: the identity layer stamps identities on whatever contacts exist.
- **Score/serve-alternation state.** The `servingTeam` cascade bug (`phase3_serving_team_cascade_test_2026_04_24.md`) is a display/chain concern absorbed by `action_classifier.py:2930`'s nearest-guard. Not identity.
- **Tracking-layer same-team ID swaps** below the match-level view. 12 real IDsw across 7 rallies are created POST-YOLO by the consolidation chain (`memory/player_tracking_audit_2026_04_15.md`, `memory/within_team_reid_project_2026_04_16.md`). Sessions 4–9 NO-GO'd the ML fixes. The identity layer *can* fix a swap's labeling even if it can't prevent the swap (reassign pid to the correct post-swap track), and that's in scope.

**Ceiling after these scope-outs.** 43.8% correct + 117 chooser-fixable cross-team + 43 primitive-fixable team-pair + 31 irreducible within-team → theoretical identity-only ceiling ≈ **78% player_attr correct** (`phase1_5_composite_partition.json`). Day-4's 95.22% was measured on a *different surface* (click-GT direct accuracy on 8 fixtures, 314 actions) with the 24-permutation label-search applied. Both numbers are true; they measure different things, and the plan sets gates on both (§6).

---

## 2. Architecture

**User flow (target UX).**
```
[1] upload → [2] blind analyze (no identity) → [3] user sees rally timeline →
[4] user taps "select players" → provides 4 ref crops (possibly partial) →
[5] relabel pass (seconds, not minutes) → [6] per-player filtered action view
```

**Data flow (new/changed boxes in bold).**
```
 detect_contacts ─────────────┐
 track_player   ──► positions_json, primary_track_ids (per rally)
 match_players  ──► playerProfiles, teamTemplates, trackToPlayer (per rally)
                    + **rallyScratchpad** (new: persisted TrackAppearanceStats +
                      side classifications + side-switch partition)
 reattribute_actions ─────────► actions_json (pid stamped on each contact)

 USER PROVIDES REF CROPS (step 4)
        │
        ▼
 **relabel_with_crops** (new CLI / worker)
    ├── load rallyScratchpad + current trackToPlayer + side-switch partition
    ├── build_profiles_from_crops(new crops) → frozen anchors
    ├── re-run refine_assignments STAGES 1 + 2 (match_tracker.py:2022–2069)
    │     — stage 1 = _assign_tracks_to_players_global per rally
    │     — stage 2 = _global_within_team_voting (cross-rally consistency, line 2067)
    │     — side-switch partition preserved, EMA skipped on frozen pids
    │     — verified 2026-04-24: stage 2 reads only state already in snapshot
    ├── optional: 24-perm label-consistency check vs remaining blind rallies
    └── write back match_analysis_json + trigger reattribute_actions
```

**Why this mechanism (not per-frame Hungarian, not joint inference).**

Day-4 proved the *existing* two-pass system + frozen reference profiles reaches 95.22% direct accuracy on 8 fixtures (`player_attribution_day4_2026_04_23.md` §Headline). The architectural move that worked was **reference prototypes as authority**, not per-frame re-matching. The restart memo's per-frame/uniqueness-constraint proposal (`player_attribution_restart_2026_04_21.md` §1) was written 2 days *before* Day-4 and predicted crop-guided approaches wouldn't solve regime 3 — Day-4 confirmed regime 3 is indeed unsolved (2e984c43 residual) but also confirmed that *per-rally* Hungarian with ref anchors is sufficient for regimes 1–2.

A full per-frame rebuild is therefore not yet justified. The current machinery, exposed through a relabel seam, is the shortest path to the user-facing UX. Per-frame is kept on the table as a Phase-5 option if Phase 1 hits a ceiling the post-label pass can't breach.

**Reuse vs new.**
- **Reuse (no change):** `match_tracker.MatchPlayerTracker.__init__` (line 560, frozen-profile injection), `_load_db_reference_crops` (`match_players.py:18`), `build_profiles_from_crops` (`player_features.py:990`), `refine_assignments` stages 1+2 (`match_tracker.py:2022–2069`, including `_global_within_team_voting` at 2067 — verified 2026-04-24 by determinism probe to read only state already in the snapshot), `reattribute_actions` entry (`reattribute_actions.py:383`), `remap_reference_crops.py` as an operator-only tool.
- **Extend:** `match_players_across_rallies` persists `rallyScratchpad` into `match_analysis_json` (serialized `TrackAppearanceStats` + side-switch partition + final per-rally side assignments).
- **New:** `rallycut relabel-with-crops <video-id>` CLI + API handler. Single responsibility: rebuild frozen profiles from current DB crops, re-score rallies with stored scratchpad, write back. Does NOT re-run Pass 1, NOT re-detect side switches, NOT re-run track extraction.
- **Replace:** nothing. `--reid` flag on `reattribute_actions` and the visual-attribution Pass 3 stay default-off and unused (`memory/crop_guided_identity_nogo_2026_04_19.md`).

**Why not joint inference.** The 5-pass combinatorial side-switch detector (`_detect_side_switches_combinatorial`, `match_tracker.py:1644`) runs once per match and is cumulative. Recomputing it every time the user adds a crop is wasteful and introduces non-determinism. Keep it deterministic: blind pass runs full Pass 1 + Pass 2; relabel pass only re-scores with new anchors.

---

## 3. Feature Choices

| Feature | Where used | Justification | Memo/file cite |
|---|---|---|---|
| HSV lower + upper + head histograms | `match_tracker` Pass 1/2 cost | Current production signal; regime-1 parity at 93.8% on tata | `crop_guided_attribution_2026_04_19.md` Phase 7 |
| DINOv2 ViT-S/14 blend (`REID_BLEND=0.5`, `REID_MIN_MARGIN=0.08`) | Same | In production, net-positive when margin gate clears | `match_tracker.py:85,91` |
| **GrabCut + split-region HSV (shirt y=0.10–0.45, shorts y=0.45–0.75, 16×6×6)** | **NEW: reference-crop profile construction only** | 100% ref-crop purity on tata, 91.5% track-level attribution; never productionized; low risk because it runs offline on 4 user crops, not per-track | `crop_guided_attribution_2026_04_19.md` §Phase 7 |
| Motion-integral V1 A5 (F-8..F window) | **SKIP in Phase 1** | Measured +5.4pp on Day-3 baseline but Day-4 memo says "likely lower leverage now; re-measure against 95.22%" | `player_attribution_day4_2026_04_23.md` Phase 4 |
| Per-track temporal appearance trajectory (new) | Regime-3 supplement (§4) | No prior test; speculative lift | — |
| Per-action crop classifier (`--reid`) | **EXCLUDED** | −23.5pp on 0a383519; closed | `memory/crop_guided_identity_nogo_2026_04_19.md` §B |
| Individual-retrained DINOv2 head | **EXCLUDED** | Session 9: median cos 0.784 → 0.818 (worse) | `memory/session9_individual_reid_probe_2026_04_17.md` |
| VideoMAE per-player backbone | **EXCLUDED** | +4.5pp on one video, 17.7 min/video — cost-prohibitive | `crop_guided_attribution_2026_04_19.md` §Phase 1.5 |

**Addressing the two user-favored techniques explicitly.**
- **Per-frame identity over reference prototypes with uniqueness constraint** → **defer to Phase 5, conditionally**. Not validated on this codebase. Day-4 achieved the Day-4 result without it, and per-rally Hungarian + frozen ref anchors already encodes a uniqueness constraint at rally granularity. Elevate to Phase 5 only if Phase 1 hits a ceiling on regime-2 fixtures (mixed lighting) that per-rally assignment cannot reach.
- **GrabCut + split-region HSV** → **included in Phase 1 for ref-crop profile construction**. Narrow scope: runs ~16 times per match (4 crops × 4 players), not per track and not per frame. This is the cheapest proven lift available and was never shipped. Do NOT extend to per-track appearance extraction — Phase 7 measured it there but the lift was narrow to regime 1.

---

## 4. Regime-3 Strategy (Same-Uniform Teammates)

**The sensing limit.** Regime 3 fixtures (titi confirmed; 2e984c43 borderline) have grey shirts + grey shorts across both teammates. Phase 7 measured reference-crop cluster purity 47–53% (chance = 25%, i.e. 1.5–2× above floor). Day-4's ref-crop approach left 2e984c43 as a residual. No monolithic embedding (DINOv2 S/L, OpenCLIP, multi-frame aggregation, unsupervised clustering) separated them above ~50% (`crop_guided_attribution_2026_04_19.md` §Phase 1–6).

**Role priors are explicitly out.** Per user constraint.

**Proposed orthogonal signal: per-rally trajectory voting.** Each rally has ~4 tracks present for ≥60% of frames. Build a per-rally position-summary vector (mean x/y court-projected + trajectory entropy + per-second movement integral) and compare to reference-crop HSV-assigned pids' mean profile across confident rallies (where the HSV+DINOv2 margin is clean). On ambiguous same-uniform rallies, break ties by trajectory similarity to the reference rallies' confident assignments.

**Confidence this works.** Low-medium. Trajectory-as-a-tiebreaker is untested on this codebase for regime 3 specifically. A comparable signal (trajectory cost in `global_identity`) regressed −1.3pp HOTA when applied at the wrong layer (`memory/player_tracking_audit_2026_04_15.md`). Different scope here (tiebreaker only, post-assignment) avoids that failure mode, but the lift is speculative.

**Quantified scope-out if Phase-3 work doesn't land.** On the 9-fixture baseline, regime-3 residual is bounded by the 31/468 "irreducible within-team" count (6.6%). On Day-4's click-GT, 2e984c43-style residuals bounded the non-100% fixtures to ~4–8 errors each. Accepting regime 3 as residual → aggregate plan ceiling drops from a targeted 90%+ to ~85% on click-GT. Both are well above production's 55% end-to-end and 43.8% locked-baseline, so **shipping without regime-3 resolution is still a win**; Phase 3 is a stretch, not a blocker.

---

## 5. Partial-Labeling Semantics

**0 crops provided.** System emits blind pid (today's Pass 1 + Pass 2 output). UI surfaces a badge: "Identity quality: preview." No relabel pass runs.

**1–3 crops (partial).** The already-reported `build_profiles_from_crops` flow handles this — frozen pids get anchored, the remaining 1–3 pids stay as unfrozen profiles updated by EMA during Pass 1 (and preserved as-is during the relabel pass, since Pass 1 doesn't re-run). Explicit spec:
- Label the provided pids with frozen anchors.
- For un-anchored pids, inherit the current Pass-1 profiles. Do NOT run a second Pass 1.
- Side-switch partition stays fixed.
- After refine_assignments stages 1+2 re-runs, run a label-permutation consistency check on *just the unlabeled pids* (2! = 2 or 3! = 6 perms), scored against cross-rally team-template consistency. Surface as suggestion per Q1 refinement; don't auto-apply.

**Multi-crop-per-player intra-player anomaly check (Q1 refinement).** When the user provides ≥2 crops for the same pid, compute pairwise cosine distance between those crops' DINOv2 + split-HSV embeddings. If any single crop's median pairwise distance exceeds T standard deviations from the cohort median (initial T = 1.5σ, tunable), flag it: "This crop looks different from your other selections for P2 — confirm or replace?" Don't auto-discard; the user may have selected from very different lighting on purpose.

**4 crops (full).** Freeze all 4. Run refine_assignments stage 1. Run the full 24-perm check *only if* the user opts into automated label correction; otherwise surface the winning permutation as a suggestion ("did you mean to swap Player 2 ↔ Player 3?").

**User-provided crop is ambiguous/incorrect.** Out of scope to auto-detect. Three mitigations:
1. `validate-reference-crops` (`reference_crops.py:100`) already runs a DINOv2 anchor quality check as API pre-flight. Keep it.
2. Post-relabel, surface per-rally `assignmentConfidence` in the UI; flag rallies below a threshold.
3. When the 24-perm check finds a different global winner than the user's provided mapping, show the suggestion rather than auto-applying.

---

## 6. Validation Gates

Each phase MUST clear its gate on *all three* measurement surfaces before shipping.

**Surface A — 9-fixture locked baseline** (`baseline_2026_04_24.json`, 69 rallies, 468 actions).
- Headline: ≥ **58%** player_attr_correct (+15pp over 43.8%), in place of 43.8/40.8/15.4.
- Per-fixture non-regression: no fixture drops > 1pp from its baseline. Worst baseline fixtures (lulu 10.8%, cuco 10.0%) must show ≥ +5pp or the plan reconsiders; these are the fixtures ref-crops should most dramatically improve.
- Missing rate (15.4%) unchanged — this plan does not touch detector recall.

**Surface B — Day-4 click-GT** (8 fixtures, 314 actions, `player_attribution_day4_2026_04_23.md`).
- ≥ **90%** direct accuracy aggregate (below the 95.22% oracle, above today's production when ref-crops are supplied).
- ≥ 3 fixtures at 100%. No fixture below 75%.

**Surface C — Post-label quality on rallies blind-pid got wrong.** The key test that the relabel pass actually repairs rather than freezes errors. Procedure:
1. Identify rallies where blind pid has `assignmentConfidence < 0.7` OR `sideSwitchDetected = True`.
2. Measure player_attr_correct on those rallies pre- and post-relabel.
3. Gate: ≥ 10pp absolute improvement on the low-confidence subset, with no single rally regressing >15pp.

**Surface D — latency.** Relabel pass must complete in ≤ **5 seconds** for a 50-rally match on API hardware. This is the "effectively instant" product requirement.

---

## 7. Phases, Risk, and Sequencing

Two axes (user asked for explicit reasoning, not a default):
- **Uncertainty axis:** Highest on the scratchpad persistence (is the serialized state complete enough to replay stage 1 deterministically?) and on regime 3 (speculative). Lower on the relabel pass itself — Day-4 already validates the approach.
- **UX axis:** Mid. Too early → users see a half-integrated flow; too late → measurement trust is built in the dark without real usage.

**Recommended sequencing.**

**Phase 0 — Scratchpad persistence + determinism (week 1).**
- Serialize `TrackAppearanceStats`, per-rally `side_assignment`, `sides_from_calibration` flag, `top_tracks`, side-switch partition, and final Pass-2 profiles into `match_analysis_json.rallyScratchpad`.
- Add a "replay" test: load scratchpad, re-run `refine_assignments` stages 1+2 with identical frozen profiles, assert byte-identical `trackToPlayer` output.
- Gate: 9-fixture baseline reproduces to 0.0pp delta.
- Risk: match_tracker state surface is broad (4 numpy arrays per profile, variable track count per rally). Easy to miss a field. **Pre-validated 2026-04-24** by `analysis/scripts/probe_scratchpad_determinism.py`: tata 20/20 + cuco 7/7 byte-identical on stage-1 cost matrices and Hungarian outputs; rere "divergence" was a probe omission (replay missed `_global_within_team_voting`), no missing snapshot state.

**Phase 1 — `relabel-with-crops` CLI + API handler + smart crop suggester (weeks 2–3).**
- New command: `rallycut relabel-with-crops <video-id>`. Loads scratchpad, loads DB crops via `_load_db_reference_crops`, rebuilds frozen profiles with GrabCut + split-HSV addition, runs refine_assignments stages 1+2, calls `reattribute_actions`.
- 24-permutation label-consistency check as an optional second step (flag-gated, default on; surface-as-suggestion per §8 Q1).
- Smart crop suggester (Q5 refinement): new endpoint `rallycut suggest-reference-crops <video-id>` reuses `global_seed_from_rallies` (`match_tracker.py:935`) to cluster all viable detections into k=4 groups, then ranks candidates per group by `confidence × non-occluded × forward-facing × time-spread`. Returns 6–8 ranked candidates per slot.
- Intra-player anomaly check (Q1 refinement): post-submission, compute pairwise cosine distance between user-provided crops within each pid; flag any crop > T standard deviations from the median.
- Gate: Surface A (+15pp), Surface B (≥90%), Surface D (≤5s for relabel; suggester latency ≤2s).
- Risk: medium. Main failure modes — 24-perm over-correcting (mitigated by suggest-don't-apply); suggester clustering surfacing the wrong groupings on regime-3 fixtures (mitigated by surfacing 6–8 candidates per slot, not 1).

**Phase 2 — UX wiring (week 4).** *User-facing enters here.*
- "Select players" upload flow + validation gate surfacing + per-rally confidence badges.
- The previous phases proved the plumbing; now real users test workflows.
- Gate: internal dogfooding on 3 fresh videos hits ≥85% user-confirmed correct attribution (click-through sample).
- Risk: low on code, real on UX. Delay here is acceptable; don't delay Phase 1's measurement gate.

**Phase 3 — Regime-3 trajectory supplement (weeks 5–6).**
- Build per-rally position summary features. Add as a tiebreaker cost in `refine_assignments` stage 1 when HSV+DINOv2 margin < `REID_MIN_MARGIN`.
- Gate: Surface A improvement on `rere` (13 within-team wrongs) and `titi`-style fixtures in held-out. Scoped-out if lift < 2pp on regime-3 fixtures.
- Risk: HIGH. Speculative signal, historical precedent (trajectory cost in global_identity) regressed when mis-placed.

**Phase 4 — Held-out validation (week 7).**
- Label 2–3 fresh fixtures (target: one regime-1, one regime-2, one candidate regime-3). Re-run entire plan from scratch.
- Gate: no regression vs locked baseline; held-out matches the aggregate metric on Surface A within ±5pp.

**Phase 5 — Per-frame identity over prototypes (conditional, ≥ week 8).** Only if Phase 3 hits a ceiling that regime-2 fixtures demonstrably can't breach via per-rally Hungarian. Would require a full design pass (cost function, uniqueness constraint shape, warm-start policy); don't spec until the empirical case is made.

**Why UX mid and not early.** Day-4 workflow already works in operator mode with CLI scripts; users getting it early before scratchpad determinism is confirmed would see intermittent regressions when crops are updated. Phase 0 is load-bearing for reproducibility.

---

## 8. Locked Product Decisions (2026-04-24)

1. **Permutation suggestion, not auto-apply.** When the system's 24-perm consistency check finds a winner different from the user's labeling, surface a banner ("Looks like P2 and P3 are swapped. Apply?") with one-tap accept. Auto-apply only when score margin to second-best perm is ≥10×.
   - **Refinement:** also allow single-player selection (1, 2, or 3 crops). When the user provides multiple crops per pid, run an intra-player anomaly check — if one crop's embedding is visually distant from the others (cosine distance > T from the median), flag it: "This crop looks different from your other selections for P2 — confirm or replace?"
2. **Blind-pid preview hides per-player filter.** Show timeline + actions at blind quality (~44%) but disable the per-player filter until at least 1 ref crop exists. Banner: "Add reference crops to enable per-player filtering."
3. **2v2 / no-sub matches only in V1.** Hard 4-pid assumption stays. Substitutions documented as known limitation; sub support becomes Phase 6.
4. **Per-rally confidence only.** Flag low-confidence rallies with a "review" badge. No per-action confidence numbers in V1.
5. **Crops picked from analyzed video, with smart suggester.** No external image upload in V1.
   - **Refinement:** the suggester is load-bearing for UX. Auto-cluster every viable detection across the match into 4 groups (visual similarity, k=4 k-means over per-detection HSV+DINOv2 features). Within each group, rank crops by clarity: detection confidence × non-occluded × forward-facing × time-spread. Present 6–8 ranked candidates per slot. Goal: minimum clicks (4 total) to optimal accuracy.
   - **Reuse:** `global_seed_from_rallies` (`match_tracker.py:935`, env-gated dormant) already does the k=4 cluster across all rally HSV — repurpose as the suggester's clustering layer.
6. **Latency target: ≤5 seconds.** Aim for 5s end-to-end on a 50-rally match. Re-evaluate to <2s only if dogfooding shows users perceive it as slow.

---

## 9. What NOT to Do

All of the below have measured NO-GOs. Do not re-propose without new evidence:

- **Retrain DINOv2 for individual identity** (`session9_individual_reid_probe_2026_04_17.md`): median cos 0.784→0.818, worse. Closes DINOv2 within-team ReID.
- **Enable `--reid` flag on `reattribute_actions`** (`crop_guided_identity_nogo_2026_04_19.md`): −23.5pp on 0a383519.
- **In-pipeline track splits via DINOv2 argmax** (`crop_guided_identity_nogo_2026_04_19.md`): −8.3pp HOTA.
- **Learned-head veto in tracklet_link merge chain** (`within_team_reid_project_2026_04_16.md` Session 6): 25% of gate target, downstream passes compensate.
- **Learned-head as additive cost in `_compute_assignment_cost`** (Session 4): byte-identical output, signal swallowed.
- **Trajectory cost in `global_identity`** (`player_tracking_audit_2026_04_15.md`): +3× real IDsw when placed at segment-level Hungarian. This plan uses trajectory only as a *tiebreaker* post-assignment, not as a cost in global_identity.
- **Monolithic whole-body embedding prototypes** (DINOv2 S/L, OpenCLIP-L, VideoMAE, multi-frame aggregation, unsupervised k=4): saturate ~50% (`crop_guided_attribution_2026_04_19.md` §Phase 1–6).
- **Role priors** (user-excluded).
- **Per-action VideoMAE crop classifier** (crop-head Phase 2): +0.00pp F1, closed (`memory/crop_head_phase2_nogo_2026_04_20.md`).
- **Full per-frame Hungarian rebuild** (without empirical evidence that per-rally hits a ceiling it can't reach) — this is Phase 5 material, conditional on Phase 3 outcomes, not V1.
- **Re-run Pass 1 on every relabel** — expensive and introduces non-determinism in side-switch detection.
- **Touch `detect_contacts` / ball tracker in this plan** — separate workstream; four prior NO-GOs (`memory/videomae_contact_nogo_2026_04_19.md`, etc.).

---

## Verification Plan

After Phase 0–1 land, prove the plan end-to-end:

1. `make dev` (per `CLAUDE.md`) to bring up local stack.
2. Re-run `rallycut match-players <video>` on the 9-fixture baseline corpus; confirm `match_analysis_json.rallyScratchpad` is populated.
3. Run the new `rallycut relabel-with-crops <video>` against each fixture's existing DB ref crops; assert `actions_json` is updated and `trackToPlayer` shifted on at least the rallies with `assignmentConfidence < 0.7`.
4. Re-run the existing baseline harness used to produce `analysis/reports/attribution_rebuild/baseline_2026_04_24.json`; compare per-fixture `correct_rate` against Surface A gates.
5. Re-run the Day-4 click-GT harness referenced in `memory/player_attribution_day4_2026_04_23.md`; compare against Surface B gates.
6. Time the relabel pass on a synthetic 50-rally match against API hardware; assert ≤5s (Surface D).
7. Spot-check 3 low-confidence rallies in the web UI to confirm the per-player filter shows the correct actions.
