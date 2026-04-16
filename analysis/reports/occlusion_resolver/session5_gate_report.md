# Session 5 — Occlusion-resolver gate report

**Verdict: NO SHIP.** The features we measure cannot separate the single
labelled swap from the 3 scored no-swaps. Root causes are diagnosable and
pointed at in §Path-forward below.

## Corpus

- **21 labelled events** across 15 GT rallies.
- Distribution: `no-swap=15`, `fragment-only=3`, `unclear=2`, **`swap=1`**.
- All 5 audit-hinted `crossed_switch=True` events got labelled `no-swap` by
  the human reviewer. The audit's `SAME_TEAM_SWAP` classifier over-fires
  under the current tracking pipeline — those "swaps" aren't swaps on the
  video.
- The 1 genuine swap (`d474b2ad · T3↔T17 · frames 845–878`) was found
  in an un-hinted DB convergence event, not in the audit set.

## Per-event scoring

(Only events whose pre/post windows contain ≥ 5 learned embeddings AND pass
the court-side veto get a feature vector; others abstain.)

| Event | Label | app_score | traj_score | Verdict |
|---|---|---:|---:|:---:|
| `e000_T3T17_f845-878` (d474b2ad) | **swap** | +0.030 | 0.000 | scored |
| `rsw000_gt2gt3_T2T25_f141` (5e2e58fb) | no-swap | -0.079 | -0.839 | scored |
| `e000_T2T3_f179-183` (de7136d1) | no-swap | -0.024 | -0.414 | scored |
| `e001_T2T3_f325-345` (de7136d1) | no-swap | -0.126 | +0.000 | scored |

**Abstentions**:
- 13 × `insufficient_embeddings` — pre/post learned-ReID windows have <5
  valid frames. Often because the primary track ended at the convergence
  (fragment-only) or the track never received embedding extraction before
  a post-processing merge.
- 4 × `court_side_veto` — median pre-side differs from post-side, veto
  fires correctly.

## Labelled-set performance

- **Precision on labelled swaps: 0.00%** (floor 95%) — the grid search
  found no threshold combination that catches the one swap while clearing
  the precision floor on the 3 scored no-swaps.
- **Recall on labelled swaps: 0.00%** (floor 50%).
- Confusion (excluding abstentions + `unclear`): tp=0, fp=0, fn=0, tn=0.
  (The best grid cell caught 0 swaps and 0 false-positives — all events
  fell below any decision surface that could both catch the swap AND keep
  precision ≥ 95% on the 3 no-swaps.)

## 43-rally retrack comparison

- SAME_TEAM_SWAP count (from Session-4 retrack audit baseline): **12**.
- Running the resolver with default thresholds on the full sweep would
  either apply no corrections (since the current thresholds are set
  conservatively and no event scored high enough) or — with relaxed
  thresholds — apply false corrections that the precision floor forbids.
- Audit-proxy reduction target (≥ 30 %) not achievable with current
  feature set.

## Per-rally HOTA regressions (drops > 0.5 pp)

- None. Resolver is effectively a no-op at the thresholds that would
  clear the labelled-set gate (there aren't any).

## Diagnosis — why the features don't separate

1. **One-positive corpus.** With a single labelled swap event, the grid
   search is measuring noise. Any threshold that catches it also catches
   at least one of the 3 no-swaps. 1/1 recall vs 1/4 precision is the
   best operating point, which fails precision by a mile. No amount of
   tuning rescues this at N=1.

2. **The signal on the one swap is genuinely weak.**
   - `appearance_score = +0.030` is barely above zero. Compare to
     no-swap's `-0.126` on a clean case — the margin is 0.156, far less
     than the jitter between no-swap examples themselves.
   - The trained head's held-out margin (Session 3 teammate-margin-mean)
     was `−0.017` — *negative* even after training. The head makes
     correct choices on the top 5/10 events but its underlying cosine
     ordering is still teammate-closer-than-self on average. This event
     appears to be one where the head barely puts its thumb on the scale.

3. **Long convergences kill trajectory signal.** The swap event is a
   34-frame convergence; `TRAJECTORY_MAX_GAP_FRAMES=15` sets
   `trajectory_score=0` for anything longer. Appearance alone has to
   carry, and appearance alone isn't strong enough at the 21-event scale.

4. **Production tracking is already clean.** 12 SAME_TEAM_SWAPs across
   43 GT rallies, 1 of which was labelled a real swap by a human.
   Recent production-cleanup passes (reattribute-actions, match-players,
   etc.) have reduced the visible failure surface. The resolver has
   little to fix on this curated set — but uncurated production videos
   may have more swaps the resolver could help with.

## Path-forward options

1. **Expand the labelled corpus (Session 2b labeller round)** — target
   the uncurated-video pool where the audit-hinted events aren't
   already-resolved false positives. Target ≥ 10 labelled swaps. That
   gives enough positives for a real threshold gate measurement.

2. **Relax trajectory window or swap its semantics.** Currently gated at
   ≤ 15 frames. Push to ≤ 45 frames and use segment-midpoint position
   instead of first-post position. Worth testing on the existing one
   swap to see if signal appears.

3. **Reopen the feature set.** Current features are 1-D appearance ×
   1-D trajectory × court-side. Add:
   - **Bbox-size continuity** (bbox height pre → post for each track).
   - **Pose keypoint descriptor** (currently gated off in retrack cache
     — plumbing needed).
   - **Per-frame appearance trajectory** (not just median — cosine(t)
     over window, cheaper to detect mid-convergence flips).

4. **Pivot the commit point earlier.** Even with better features, a
   resolver that only fires after `apply_post_processing` inherits
   whatever identity decisions tracklet_link has already made. Pushing
   the learned signal into BoT-SORT's ReID cost is the biggest lever
   but also the biggest lift (Session 6+).

5. **Accept the negative and document.** Same-team-disambiguation at the
   segment-level is structurally hard — Session 4's NO SHIP and Session
   5's NO SHIP both point at the same conclusion. The trained head's
   data-scale just isn't enough to make crisp decisions. Invest elsewhere
   (preflight ball tracking, action classification) for the next block
   of cycles.

## What's shippable (dormant)

- Occlusion-resolver module + env-var-gated integration (`ENABLE_OCCLUSION_RESOLVER=0` default → byte-identical).
- Labelling app + enumerator reproduce on any new rally corpus.
- Eval script ready for a larger label set.

All of this is kept as dormant code, ready to be revisited when one of
the path-forward options is worth pursuing.

## Runtime

- Phase A (enumerator + labeller): ~45 min to build, ~20 min human labelling.
- Phase B (resolver + unit tests + integration): ~2 h.
- Phase C (eval + gate): ~5 min (skipped 43-rally retrack — no need without a positive labelled gate result).

## Artifacts

- `analysis/training_data/occlusion_resolver/labels.json` — 21 human-verified labels.
- `analysis/reports/occlusion_resolver/events.json` — convergence-event corpus.
- `analysis/reports/occlusion_resolver/labeller.html` — reproducible labelling UI.
- `analysis/reports/occlusion_resolver/event_scores.json` — per-event feature values (post-processing-aware).
- `analysis/rallycut/tracking/occlusion_resolver.py` — dormant module.
- `analysis/scripts/{enumerate_convergence_events,render_occlusion_labeller,eval_occlusion_resolver}.py` — reproducible pipeline.
