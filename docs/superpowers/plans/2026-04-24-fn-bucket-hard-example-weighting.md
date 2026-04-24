# Plan — FN-bucket lever: hard-example weighting probe + trajectory feature engineering

**Date:** 2026-04-24
**Scope:** the 128 non-block, classifier-rejected FN cases in
`outputs/action_errors/corpus_annotated.jsonl` — the bucket where
`gbm < 0.10 AND seq ≥ 0.95` on 88/128 cases (max GBM↔MS-TCN++ disagreement).
**Driving evidence:** `analysis/reports/pose_audit/pose_features_borderline_2026_04_24.md`
**Original hypothesis:** pose features exist, GBM under-weights them, fix
is feature engineering.
**Revised hypothesis (this plan):** the bucket is a *whole-feature-set
weighting problem*, not a pose-feature problem. Pose was the wrong lever.

---

## 1. Where the diagnostic lands

Pre-registered branches per the diagnostic spec:

| Branch | Condition | Evidence | Verdict |
|---|---|---|---|
| A — pose discriminates, underweighted | Pose AUC ≥ 0.65 + pose importance rises >5pp under focused retrain | Best AUC 0.735 (set/13 FNs); +4.89pp pose rise | **Borderline → reject as primary** |
| B — pose populates but needs better features | Same as A + cumulative pose rise dominant | Trajectory rose ~17pp / player-pos ~10pp / pose +4.89pp | **Reject — pose is dominated** |
| C — pose doesn't help; recommend the actual lever | Pose rise capped, other features rise more | This plan | **Adopt** |

Three reasons Branch C wins:

1. **Dig (42/128 cases, 33% of bucket) is at AUC 0.592 — chance.** Pose
   feature engineering cannot help a third of the bucket structurally.
2. **17 cases (13%) are pose-blind.** No pose intervention touches them.
3. **Under focused retraining, pose total importance is 6.34% — fourth-tier
   behind seq, frames_since_last, and player_distance.** Trajectory features
   (acceleration, ball_y, ball_y_relative_net, arc_fit_residual, velocity_y,
   trajectory_curvature, velocity_ratio, velocity, consecutive_detections)
   sum to ~17pp rise. Player-position features (player_distance + 4 bbox
   motion features) sum to ~10pp rise. Pose's +4.89pp is third-place at best.

The Pass 3 finding is the **important new thing**: when forced to
discriminate the bucket, the GBM CAN find separating signal — it just isn't
pose-shaped. The signal is broadly distributed across trajectory and player
features. This rules out "the GBM lacks information" and points to "the GBM
lacks weighting/loss-shape to prioritize the bucket."

---

## 2. Plan: three workstreams in priority order

### Workstream P0 — hard-example sample-weight probe (3-5 days)

**Hypothesis:** if the bucket is a weighting problem, simply upweighting the
128 FN positives during training should lift bucket recall at modest FP cost.
This is the cheapest possible test of the Pass 3 finding.

**Implementation (single script, no pipeline changes):**

1. Re-purpose `train_contact_classifier.py` with a new `--bucket-weight` flag
   that loads the FN bucket from `corpus_annotated.jsonl` and multiplies the
   sample weight on candidates whose frame matches an FN gt_frame within
   ±5 frames.
2. Run the LOO-CV harness in `scripts/eval_loo_video.py --include-synthetic`
   for bucket weights {1×, 3×, 5×, 10×, 20×}. Each weight is a single retrain
   + 68-fold LOO eval (~20 min wall-clock per weight).
3. Hold the threshold fixed at 0.40 for the primary measurement; sweep
   {0.30, 0.35, 0.40, 0.45} as a precision/recall tradeoff secondary.

**Pre-registered ship gates (matching shipped-precedent convention):**

- 68-fold LOO Contact F1 ≥ **88.0%** (production baseline; v5 was 88.87%)
- Per-class F1 floors vs production: serve / receive / set / attack / dig
  ≥ **−1.5pp**; block exempt at **−12pp**
- Action Acc ≥ production
- 9-fixture attribution baseline (43.8% / 40.8% / 15.4%): no meaningful
  regression
- FP budget: **≤ 50 added FPs per 2095 contacts** (the blanket-rescue NO-GO
  cost +284 FPs at +36 TPs — this is the empirical ceiling for "ship-able
  cost per rescue")

**Decision tree:**

- If any tested weight clears all gates → ship with that weight; close the
  bucket workstream. Expected lift +0.5 to +2.0pp F1 if Pass 3's "everything
  rises" hypothesis is correct.
- If F1 lifts but a per-class floor fails (likely for set or attack — they
  have the cleanest pose signature and may regress when the bucket pulls
  weight away) → fall back to per-class weighting (separate weight for
  bucket-by-class subsets), repeat.
- If no weight clears the FP gate → workstream P0 NO-GO; escalate to P1
  (trajectory feature engineering).

**Why this is the right first move:** zero code in production paths, zero
new features, the entire experiment is a sample-weight + retrain + LOO eval
on existing infra. If it works, it ships in days. If it doesn't, the NO-GO
cleanly rules out "bucket weighting" as the lever and we know feature
engineering is required.

---

### Workstream P1 — trajectory feature engineering (1-2 weeks, conditional)

**Triggered when:** P0 NO-GO confirms the bucket needs new feature signal,
not just weighting.

**Hypothesis:** the FN bucket is "tight-touch" contacts where ball
velocity / direction-change is below the candidate-generator's normal
trigger thresholds. Per the task description, the reference case has
velocity=0.007, dcd=3.1° — well below the 0.05 / 15° defaults. Pass 3 shows
that under focused training, `acceleration` (+3.14pp), `velocity_y`
(+1.64pp), `arc_fit_residual` (+1.60pp), `trajectory_curvature` (+1.95pp),
`ball_y_relative_net` (+2.11pp), and `consecutive_detections` (+2.51pp)
ALL rise dramatically — meaning useful trajectory information is present
but currently coarse-summarized.

**Candidate features to design (none of which require pose):**

1. **Sub-frame micro-acceleration** — second derivative over a tighter
   window than the current ±3-frame `acceleration` feature. The GBM's
   existing acceleration feature averages over 3 frames; a ±1-frame jerk
   estimate would catch tight contacts that don't deflect ball trajectory
   much but introduce a brief impulse.
2. **Ball-vertical-velocity inflection within ±3 frames** — pre-contact
   `velocity_y` reverses sign on most real contacts. Currently encoded as
   raw `velocity_y` only. A "sign flip in window" indicator + magnitude
   would surface tight-touch sets/digs that gently redirect the ball
   downward.
3. **Player-anchored ball-distance integral over ±5 frames** — current
   `player_distance` is a single-frame snapshot. A trajectory-area metric
   (∫|distance| dt) captures sustained close encounters that aren't single-
   frame nearest-distance minima.
4. **Net-line proximity at peak ball height** — many serves/sets in the FN
   bucket are at the apex; ball_y_relative_net + height proximity to local
   peak might encode "hand-up volley" geometry without pose.

**Validation gate per feature:** add to feature list, retrain GBM (200
estimators, depth 4, LR 0.05 — production hyperparams), measure
68-fold LOO F1 + per-class. Same gates as P0. Reject features that don't
clear individually before stacking.

**Risk:** historical precedent (memory: `action_fixes_attempt_2026_04_20.md`,
`contact_arbitrator_2026_04_22_nogo.md`) shows feature additions to the
GBM rarely move F1 by more than ±0.5pp in either direction. Expected best
case: +0.3 to +1.0pp F1 and 30-50% of the bucket recovered.

---

### Workstream P2 — set-class pose-targeted probe (3-4 days, low priority)

**Triggered when:** P0 + P1 both fail and a stretch experiment is wanted.

**Justification:** of the five non-block classes, `set` is the only one
where a single pose feature (`wrist_velocity_max`) clears AUC 0.70 (0.735).
But n_FN=13 → fragile. Worth a bounded ablation only if the larger workstreams
fail.

**Design:** add a single conditional rescue gate restricted to `set`
candidates (gated by MS-TCN++ argmax = set within ±5 frames):

```text
Rescue iff:
  seq_argmax_set ≥ 0.85
  AND wrist_velocity_max ≤ percentile(TP_set_wrist_vel, 25) ≈ 0.0098
  AND player_distance ≤ 0.15
  AND gbm_conf ≥ 0.05  (avoid pure-noise rescues)
```

**Gates:** same as P0/P1, except:
- per-class set F1 must rise ≥ +1.5pp
- Other classes must not regress (≥ -0.3pp each)

**Expected:** +0.1 to +0.3pp F1 from a set-specific rescue. Marginal.

---

## 3. Pre-registered ship gates (uniform across P0-P2)

- **Contact F1:** 68-fold LOO ≥ **88.0%** (`scripts/eval_loo_video.py
  --include-synthetic`).
- **Per-class F1 floors:** serve, receive, set, attack, dig each ≥ **−1.5pp**
  vs production v5; block exempt at **−12pp**.
- **Action Acc:** ≥ production (currently ~93.4%-94.0% depending on baseline).
- **Attribution baseline:** the 9-fixture click-GT baseline (43.8% /
  40.8% / 15.4%) must hold within measurement noise.
- **FP budget:** ≤ **50 added FPs per 2095 contacts** vs the chosen baseline.
- **Block specifically exempt** because of structural data starvation
  documented in `crop_head_phase2_nogo_2026_04_20.md`.

Any gate failure → workstream NO-GO, not a tweak-and-retry. Memory is
explicit on this pattern (`action_fixes_attempt_2026_04_20.md`).

---

## 4. Out of scope (do not retry)

Memory has explicit NO-GO evidence for:

- Blanket seq-only rescue (`action_fixes_attempt_2026_04_20.md`).
- 12-feature meta-classifier arbitrator
  (`contact_arbitrator_2026_04_22_nogo.md`).
- Crop-head as a label-only emitter (`crop_head_phase2_nogo_2026_04_20.md`).
- VideoMAE contact features (`videomae_contact_nogo_2026_04_19.md`).
- DINOv2 individual-identity ReID (`session9_individual_reid_probe_2026_04_17.md`).
- Decoder integration into `detect_contacts` (`action_fixes_attempt_2026_04_20.md`).
- Three-signal threshold rescue (conf × seq × player_dist — same memo).
- Per-candidate crop-head emitter (`crop_head_phase2_nogo_2026_04_20.md`).

**Pose features as a primary discriminator on the 128 FN bucket** — added
by this plan after the diagnostic.

---

## 5. North-star alignment

User's product goal: "pick a player, see all his actions accurately."
This plan touches **action detection**, not attribution. The 128 FN bucket
is missed contacts — they don't appear in the player timeline at all,
which is the worst presentation outcome (a real action invisible).
Recovering even half the bucket without FP bleed materially improves
"all his actions" coverage on every fixture. Attribution remains a
separate workstream gated on reference-crop coverage and within-team ReID
(see `player_attribution_day4_2026_04_23.md`).

---

## 6. Path-forward summary

1. **Spend 3-5 days on P0** (sample-weight probe). This is binary go/no-go.
2. **If P0 ships:** close the bucket workstream, redirect to attribution
   coverage or ball-tracker recall (the next two highest-EV levers per
   `videomae_contact_nogo_2026_04_19.md` and Day-4 attribution memo).
3. **If P0 fails:** spend 1-2 weeks on P1 (trajectory feature design).
   Run the four candidate features individually before any stacking.
4. **If P0 and P1 both fail:** accept the bucket as the architectural
   ceiling of the trajectory-GBM design, redirect to either parallel-decoder
   architecture (`action_fixes_attempt_2026_04_20.md`, ~2 days) or per-
   candidate crop-head detector (3-4 weeks, separate plan + brainstorm).
5. **P2 (set-class pose probe)** is a 3-4 day stretch only after P0+P1
   close.

The diagnostic that produced this plan is reproducible at
`scripts/pose_audit_2026_04_24.py` + `scripts/pose_audit_pass3_2026_04_24.py`.
Re-run before any future pose-related experiment to confirm the AUC + pose
importance numbers haven't shifted under a new GBM checkpoint or expanded GT.
