# Attribution — Primitive-First Rebuild

**Date:** 2026-04-24
**Status:** Plan — awaiting approval
**Owner:** Mario
**Baseline (measured 2026-04-23 on 8 fixtures, 314 click-GT actions):**
- correct: 227 (72.3%)
- wrong: 63 (20.1%) — CROSS_TEAM 47 + SAME_TEAM 13 + UNKNOWN_TEAM 3
- missing: 27 (8.6%) — pipeline emitted no action within ±10f of GT
- stage 2 (identity): 95.22%

**Target (revised per user direction):**
- **`wrong_rate` → near 0** (north-star UX principle: prefer miss over wrong)
- `missing_rate` tolerated, bounded, and reported alongside
- Visual debugging surface at every phase so user can eyeball per-rally correctness

---

## 1. Why this plan exists

Three consecutive attribution "fixes" reported large positive lifts that evaporated under honest measurement:

- **Phase 3 global-seed** — initial ON/OFF suggested a win; deeper audit showed permutation-agnostic GT scoring drifting into coincidental agreement with broken pipeline picks. NO-GO.
- **Team-gated attribution, wide** — pre/post on contaminated DB showed +4.1pp combined, +18.9pp on 44e89f6c. Clean apples-to-apples off-gate run on the same code path matched those numbers. NO-GO.
- **Team-gated attribution, narrowed** — transition matrix against calibration team membership revealed 47/47 CROSS_TEAM_WRONG errors survived the gate because all three signals (ball court_side, picked tid team, GT actor team) self-consistently pointed at the same wrong answer. Gate is a no-op on its target error class. NO-GO.

**Common root cause:** iterating on the chooser while the primitives (team, side, ball-side, tracking stability) are unaudited. The 47 "cross-team" errors aren't chooser mistakes — they're primitive mismatches the chooser faithfully propagated. This plan reorders the work: audit primitives first, then fix the chooser on trusted primitives, then layer rally-level consistency rules. Visual debugging at every phase prevents the blind-iteration pattern.

---

## 2. North-star UX principle

> **Prefer missing attribution over wrong attribution.**

Implications wired into every phase:

- **Metric split.** Every A/B reports both:
  - `wrong_rate` = `wrong_count / all_GT_actions`
  - `missing_rate` = `(not_detected + chooser_abstained) / all_GT_actions`
- **Chooser abstains when uncertain.** `player_track_id = None` is a first-class output.
- **Complement rescue only on unambiguous chains.** Never speculative.
- **UX surface** (downstream): high-confidence actions shown; uncertain bucket expandable on demand. Out of plan scope but sets the product bar.

---

## 3. Scope and non-scope

**In scope**
- Primitive audit (identity, team→side, ball→side, tracking stability) with visual debug.
- Attribution chooser: confidence-gated, identity-grounded, roster-aware.
- Rally-level consistency: unambiguous complement rescue only, soft volleyball-rule priors (not hard filters).
- Measurement discipline: locked baseline JSON, split wrong/missing metrics, standard benchmark runner.

**Out of scope (named, not addressed)**
- **Contact-detection recall.** 27/314 GT actions have no emitted contact. Complement rescues the unambiguous subset; rest is a known floor. Separate workstream (all recent attempts NO-GO'd: Phase B, crop-head, VideoMAE, serve-gen, contact arbitrator).
- **Within-team same-side disambiguation.** 13 errors where chooser picks teammate. Distance-only chooser's architectural floor. Deferred until after Phases 1-3 land and residual shape is known.
- **Generalization beyond 8 fixtures.** Production = 68+ videos. Held-out validation is Phase 4, not broader rollout.

---

## 4. Visual debugging surface (first-class deliverable)

Built in Phase 0, extended across every phase. Goal: user steps through rallies one at a time in a browser and eyeballs whether primitives and attributions look right.

**Per-rally HTML viewer** (`reports/attribution_audit/{vshort}/{rally_id}.html`) shows for each rally:

- **Rally clip** (embedded MP4, scrubbable).
- **Court calibration overlay** — net line, court boundaries, midline (8m court-Y) drawn.
- **Per-tid coloring** — each primary tid rendered with a color; legend shows pid→tid→team mapping from identity-first logic.
- **Team→side badges** — top corners: "team A = near (maj=3/3)" vs "team B = far (maj=2/3)" with majority-vote breakdown.
- **Side-switch state** — timeline ribbon showing cumulative side-switch count at rally start and whether this rally is flipped.
- **Ball trajectory** — ball path colored by inferred `court_side` at each frame; markers where `court_side` changes.
- **Contact markers** — every detected contact shown with: attributed pid, chooser confidence margin, GT pid (if present), whether this is an error.
- **Error highlights** — WRONG contacts red; MISSING GT (no contact within ±10f) shown as phantom marker; ABSTAINED as gray "?".
- **Primitive-level audit flags** — if identity swap detected in rally → badge; if team→side disagrees with positional fallback → badge; if ball `court_side` at any contact disagrees with GT actor foot-projection → badge.

**Index page** (`reports/attribution_audit/index.html`) — 8-fixture summary grid: per-rally thumbnail, per-rally `wrong_rate`/`missing_rate`, click-through to viewer.

This surface is the user's shared mental model. Every phase's deliverable renders into it. User scrolls rallies and tells me "this one looks wrong" — that's the primary feedback channel. No more text-only dashboards.

---

## 5. Phase structure

### Phase 0 — Lock baseline + dormant sweep (1 day)

**0.1 Lock baseline artifact on disk.**
- Run off-gate stage-3 on 8 fixtures cold.
- Save contacts+actions+team_assignments as JSON: `reports/baseline_2026_04_24.json`.
- Schema (flat, one record per rally):
  ```
  { rally_id, video_shortname, contacts: [...], actions: [...],
    team_assignments: {tid: team}, primary_track_ids: [...],
    gt: [{frame, actor_tid}], metrics: {wrong, missing, correct} }
  ```
- Every subsequent A/B reads this file for "pre", never DB.

**0.2 Standard benchmark runner** (`scripts/bench_attribution.py`).
- Inputs: a run mode (baseline | phaseX) and an artifact path.
- Outputs:
  - `wrong_rate`, `missing_rate`, `correct_rate` per-fixture and combined.
  - Transition matrix vs baseline.
  - Per-rally breakdown JSON.
  - Renders into the visual debug surface.
- Every experiment calls this runner. No bespoke per-experiment measurement code.

**0.3 Dormant code sweep.**
- Inventory every env-gated dormant workstream (`TEAM_GATED_ATTRIBUTION`, `JOINT_DECODE_IDENTITY`, `ENABLE_OCCLUSION_RESOLVER`, `LEARNED_MERGE_VETO_COS`, `MATCH_TRACKER_GLOBAL_SEED`, `WEIGHT_LEARNED_REID`, `CROP_HEAD_*`, others).
- Decision per-flag: remove (disproven NO-GO) vs keep-dormant (still plausible under new primitive foundation) vs promote (evidence supports enabling).
- Output: `reports/dormant_flag_decisions_2026_04_24.md`.

**0.4 Action-GT freshness audit.**
- Known issue: `action_ground_truth.json` (serve/receive/set/attack/dig/block labels) is stale on some rallies. Confirmed by user.
- Click-GT (actor identity at contact frame, 314 actions on 8 fixtures) is trusted and drives all `wrong_rate`/`missing_rate` numbers.
- Action-type GT is used by Phase 3 Pattern A ("receive/dig → attack/spike" anchors) and by any rule relying on action-type sequence.
- Audit scope: for each rally, compare pipeline-emitted action types against `action_ground_truth.json`; surface disagreement rate. If stale ≥5% of rallies, Pattern A preconditions must use pipeline action types (trusted if high-confidence), not stored GT. Refresh stored action-GT is optional ship-blocker, tracked as Phase 0.4a.
- Output: `reports/action_gt_freshness_2026_04_24.md` listing stale rallies, stale rate, decision on whether to refresh GT vs lean on pipeline action types for Phase 3.

**0.5 Worktree isolation.**
- Create `worktrees/attribution-rebuild-2026-04-24` via `superpowers:using-git-worktrees`.
- All Phase 1-3 code lands there; main stays clean.
- Rebase / merge back at Phase 4 ship time.

**Phase 0 deliverable:** locked baseline JSON, standard benchmark runner, dormant-flag decisions, action-GT freshness verdict, worktree ready. First version of visual debug surface rendered from baseline.

### Phase 1 — Primitive audit (2 days)

Phase-1 order respects dependencies (each primitive audit feeds the next).

**1.1 Tracking-ID stability within a rally (foundation — everything else depends on this).**
- For each rally: trace tid across frames; detect silent swaps, duplicate pids, mid-rally dropouts.
- Visual: per-rally viewer gains "tid timeline" row — each primary tid's frame coverage plotted, anomalies flagged.
- Measure: `rally_stable_rate` = rallies with no tracking anomalies / total rallies.
- Threshold: ≥95%. Below → fix upstream before Phase 2.

**1.2 Side-switch reliability (prerequisite to 1.3, not 1.4 as in v1 of this plan).**
- Extract `sideSwitchDetected` state per rally from `match_analysis_json`.
- Visual: timeline ribbon in viewer shows cumulative side-switch count + flipped state.
- Audit: user eyeballs 2-3 rallies per video via the viewer; marks correct/wrong.
- Measure: per-video side-switch agreement rate.
- Threshold: ≥98% (it's a binary over few events per match — mistakes rare expected).

**1.3 Team→side per rally (identity-first vs positional-median comparison).**
- Compute team→side two ways: (A) today's positional-median, (B) identity-first (pid → team via convention, team → side via majority-vote of team members' foot projections), honoring 1.2's side-switch state.
- Compare both to visual GT via the per-rally viewer.
- Measure: agreement rate per method per rally. Identify rallies where (A) and (B) disagree.
- Threshold: (B) ≥98% agreement with visual GT. Below → fix identity-first logic.
- Expected improvement over (A): primarily on rallies with midline-straddling players.

**1.4 Ball `court_side` per contact frame.**
- On the 47 CROSS_TEAM_WRONG contacts from baseline: compute ball `court_side` three ways: (A) today's instantaneous `ball.y`, (B) trajectory-window median (±3 frames), (C) explicit net-line crossing detection.
- Also compute GT actor's foot projection at contact frame ("actual side").
- Visual: per-contact sub-page shows ball trajectory ±1 second around contact with all three classifications + actual side.
- Measure: how often each method matches actual side.
- Threshold: whichever method best-wins ≥95%. Adopt that method.

**1.5 Composite primitive correctness per contact.**
- For every contact in baseline: does the triple (picked tid's team via 1.3, ball court_side via 1.4, GT actor's expected side) internally agree AND match the GT actor?
- If yes → primitive layer is clean; any remaining error is chooser.
- If no → primitive error, candidate for Phase 2 primitive fix vs Phase 3 consistency.
- This measurement partitions baseline errors into "primitive-fixable" vs "chooser-fixable" vs "irreducible."

**Phase 1 deliverable:** `reports/attribution_primitive_audit_2026_04_24.md` with per-primitive accuracy table, visual viewer populated with all audit overlays, error partition (primitive vs chooser vs irreducible). User reviews viewer and signs off before Phase 2.

**Phase 1 kill gate:** No fix ships in Phase 1 — it's audit-only. But if any primitive measures below threshold, *its fix is added as Phase 1.x and blocks Phase 2 start.*

### Phase 2 — Attribution chooser on trusted primitives (2-3 days)

Prerequisites: Phase 1 complete; primitives at or above thresholds.

**2.1 Identity-first team→side wired into `_classify_track_sides`.**
- Replace positional-first priority with identity-first; positional becomes fallback when identity unavailable.
- Visual: viewer's team→side badges auto-update; disagreements with positional flagged.

**2.2 Confidence-gated chooser.**
- For each contact: chooser margin = `(d_2 - d_1) / d_1` among team-consistent candidates (d_1=closest, d_2=2nd closest).
- If margin < `THR_MARGIN`: emit `player_track_id = None`.
- Sweep `THR_MARGIN` on locked baseline; plot `wrong_rate` vs `missing_rate` Pareto curve.
- Adopt the smallest threshold at which `wrong_rate` is ≤ half the baseline wrong_rate (target: 20.1% → ≤10%).
- Visual: viewer shows per-contact margin bar; abstained contacts rendered gray "?".

**2.3 Roster-aware chooser.**
- When expected team has 2 members visible and one is clearly non-contactor (large distance to ball + not moving toward ball), bias toward the other member.
- Fires only when signal is unambiguous (two members visible, clear distance separation).
- Visual: viewer annotates roster-complement decisions with "visible non-contactor → {pid} by elimination."

**Phase 2 deliverable:** confidence-gated chooser integrated, measured via standard benchmark runner.

**Phase 2 kill gate:**
- `wrong_rate` ≤ 10% (halve baseline 20.1%). Hard requirement.
- `missing_rate` ≤ 20% (baseline 8.6% + 11.4pp tolerance). Hard ceiling.
- `correct_rate` ≥ 70% (near baseline 72.3%). Soft — protects against total abstention.
- Per-fixture: no fixture regresses on `wrong_rate` by >2pp (no fixture gets actively worse).

### Phase 3 — Rally-level consistency and unambiguous complement rescue (3-4 days)

Prerequisites: Phase 2 lands cleanly.

**3.1 Serve side/team inference from (serving_team, team→side).**
- When serve contact is missing or low-confidence: attribute "serve by {serving_team}, {side}" as team+side level.
- Player-level only when one team-member is visible-and-far-from-ball at the serve frame.

**3.2 Obvious-case complement rescue — concrete patterns.**
- **Pattern A — bookend fill:** action `X(P_a)` at frame `f_1` → missing or abstained action at frame `f_m` → action `Y(P_a)` at frame `f_2`, where:
  - `f_2 - f_1 ≤ 120 frames` (~4 seconds).
  - Both anchors high-confidence (margin above threshold).
  - No net-crossing between `f_1` and `f_2` (ball stays on team's side).
  - X and Y are compatible with "middle is a set" volleyball-wise (X ∈ {receive, dig}, Y ∈ {attack, spike}).
  - Action types for X and Y come from the **pipeline** (high-confidence MS-TCN++/classifier output), not from stored `action_ground_truth.json` (known stale per Phase 0.4).
  - → Infer middle was P_b (the other team member on the same side).
- **Pattern B — server-occluded serve:** no contact at expected serve frame but next team-action within 60 frames is an across-net receive by the opposing team's visible player → infer serve by expected team's expected server (if unambiguous via 3.1).
- **Pattern C — teammate-role check:** a confidence-gated abstained contact on side S where only one team-S player is near the ball (other clearly non-contactor by distance/position) → infer the close one.
- Each pattern fires only when ALL preconditions satisfied. Single missing precondition → abstain.
- Visual: viewer renders rescued contacts in green with the rescue pattern name, so user can eyeball each rescue.

**3.3 Soft volleyball-rule priors (never hard filters).**
- Touch-count ≤3 same side, across-net alternation, serve alternation — used to nudge confidence, not reject attributions.
- If an attribution violates a soft rule, its margin is reduced (moving it toward abstention) but never flipped to a different tid.
- Amateurs break rules. Soft priors never hard-reject.

**3.4 Confidence propagation within rally.**
- High-confidence anchors bump priors of nearby low-confidence attributions on the same side.
- Only applies within a rally, never across rallies.

**Phase 3 deliverable:** consistency layer on Phase 2 chooser. Measured via standard runner.

**Phase 3 kill gate:**
- `wrong_rate` strictly monotonic-lower vs Phase 2 output (rescue cannot introduce new wrongs at the aggregate level).
- `missing_rate` lower than Phase 2 (rescue fills gaps).
- Rescue precision ≥95% — of actions filled by complement rescue, ≥95% must match GT. (Audit via visual viewer before computing aggregate.)
- Per-fixture: no fixture regresses on `wrong_rate` by any amount.

### Phase 4 — Held-out validation + ship (1-2 days + user labeling)

**4.1 Held-out click-GT labeling.**
- Select 2-3 fixtures from production's 68 videos, regime-diverse (one same-uniform-teammates, one not-yet-click-GT, one recently-added).
- User labels via existing click-GT tool.

**4.2 Run Phase 3 pipeline on held-out.**
- Measure `wrong_rate`, `missing_rate`.
- Kill gate: held-out `wrong_rate` within 3pp of 8-fixture number. Held-out `missing_rate` within 5pp.

**4.3 Ship.**
- Remove `TEAM_GATED_ATTRIBUTION` and other NO-GO flags.
- Merge worktree branch.
- Update memory with final numbers and close the workstream.

---

## 6. Visual debug surface evolution per phase

| Phase | Added to viewer |
|-------|-----------------|
| 0 | baseline contacts + attributions, court overlay, tid colors, per-rally and index pages |
| 1.1 | tid-timeline row, anomaly flags |
| 1.2 | side-switch ribbon |
| 1.3 | team→side badges with majority-vote breakdown |
| 1.4 | per-contact ball trajectory sub-page with 3-way side classification |
| 1.5 | composite primitive OK/FAIL flag per contact |
| 2.2 | chooser margin bar, abstained gray "?" markers |
| 2.3 | roster-complement annotations |
| 3.2 | rescued contacts in green with pattern name |
| 3.3 | soft-rule violation badges |

User can at any time open a rally in the viewer and see exactly what the current pipeline thinks, why, and where it's uncertain. Feedback channel: user says "rally X looks wrong," I open rally X in viewer and diagnose immediately.

---

## 7. Measurement discipline (meta-rule)

- **Lock baseline on disk as JSON** (Phase 0.1). All A/B comparisons read this file. Never DB.
- **Always fresh-vs-fresh.** Compare two fresh runs, not fresh vs stale DB.
- **Always report BOTH `wrong_rate` and `missing_rate`.** Never a single combined accuracy.
- **Per-fixture breakdown + COMBINED.** Aggregate hides regime-specific regressions.
- **Name categories precisely.** Every metric gets a one-line definition in its report. Last cycle's "CROSS_TEAM_WRONG" meant "static team labels differ" not "chooser picked wrong team" — that conflation burned 2 days.
- **Visual verification before numeric claim.** For every lift claim, open 3 affected rallies in the viewer and eyeball before reporting the Δpp.

---

## 8. Non-negotiables

- Primitives first, then chooser, then consistency. No chooser work before Phase 1 dashboard lands.
- No ship until 8-fixture passes AND held-out passes.
- `wrong_rate` is the optimization target. Lower at any cost of `missing_rate` within the stated ceilings.
- Complement rescue only on unambiguous chains. Single missing precondition → abstain.
- Soft volleyball rules never hard-reject. Amateurs break them.
- Every phase renders into the visual debug surface.

---

## 9. Estimated effort

| Phase | Days | Output |
|-------|------|--------|
| 0 | 1 | Locked baseline, benchmark runner, dormant decisions, worktree, v0 viewer |
| 1 | 2 | Primitive audits, thresholds met or fixes scoped |
| 2 | 2-3 | Confidence-gated chooser, `wrong_rate` ≤10% |
| 3 | 3-4 | Complement rescue + soft rules, `wrong_rate` monotonic lower |
| 4 | 1-2 + labeling | Held-out validation + ship |

**Total:** 9-12 focused days + held-out labeling time.

---

## 10. Decision points before Phase 0 starts

1. **North-star "miss > wrong" confirmed?** Drives every kill gate. (Assume yes unless user disagrees.)
2. **Primitive thresholds — are 95% (tracking stability), 98% (side-switch, team→side, ball-side) right?** Tighter = more fix work, higher bar. Looser = earlier Phase 2 start, risk of shaky foundation.
3. **OK to exclude NOT_DETECTED (27 actions) from direct targets, rescue-only via 3.2?**
4. **OK with Phase 4 held-out labeling cost** (user labels 2-3 fixtures)?
5. **Visual debug surface style** — HTML with embedded MP4 per rally is what I'm proposing. Alternative: static per-rally images. Prefer HTML+MP4 for scrubbability but it's heavier.
6. **Worktree isolation acceptable?**
7. **Action-GT staleness** — OK to lean on pipeline action types for Phase 3 Pattern A preconditions rather than refreshing `action_ground_truth.json` upfront? (Refresh is an extra GT-labeling pass if preferred.)

Once these are answered, Phase 0 starts.
