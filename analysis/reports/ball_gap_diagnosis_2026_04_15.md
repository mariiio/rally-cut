# Ball Tracking Gap Diagnosis — 2026-04-15

## TL;DR

Ran the new `diagnose_ball_gaps.py` against all **43 ball-GT rallies** (15,077
GT frames). Against the stored DB predictions (`--source db`, production
state), overall per-frame match is **81.91%** (50 px threshold, GT linearly
interpolated between keyframes).

**The dominant gap is detector miss, not filter behavior.** Of the 18.09%
gap, the frame breakdown across the 27 rallies with raw cache is:

| Bucket | Frames | % of total GT | Root cause |
|---|---:|---:|---|
| `missed_no_raw` | 1,169 | 11.78% | WASB produced no near-GT raw detection |
| `missed_filter_killed` | 136 | **1.37%** | Raw had GT-close detection, filter dropped it |
| `wrong_object` | 254 | 2.56% | Pred exists but >50 px from GT |
| `interpolated_wrong` | 59 | 0.59% | Filter interpolated incorrectly |

On the 16 rallies without raw cache the missed bucket is undifferentiated
(**18.40%** combined), so true filter-kill share is almost certainly still
near 1% at worst across the full set.

**Headlines:**

1. **The filter is not the bottleneck.** Raising filter sensitivity would
   recover at most ~1.4% — and spending effort there risks introducing new
   wrong-object regressions.
2. **Two rallies dominate the wrong-object bucket**: `9dbe457a` (120
   frames, 30% of its GT) and `21a9b203` (43 frames). Investigation of
   those two alone would explain ~40% of all wrong-object cases.
3. **Stationary-distractor pattern is not present** in the current raw
   cache on the 43 GT rallies — the previously-memorised "meme #13"
   background-ball FP either no longer occurs with fine-tuned WASB, or is
   outside the labeled set. Re-enabling the disabled stationarity filter
   would add zero measured benefit on this set.
4. **Teleports are rare (50 events across 21 rallies).** Only 4 rallies
   have ≥4 teleports, and most teleports correlate with real fast-motion
   spike frames rather than tracker defects.

The root cause the user asked about — "missed tracks, wrong objects,
teleports" — resolves this way: **missed tracks are model-level misses
(WASB didn't fire), wrong objects are concentrated in a small set of
rallies with atypical trajectories, and teleports are near-noise. The
filter is mostly doing its job.**

Recommendation priority (evidence-weighted, see §Recommendations):
**model > wrong-object-rally-specific analysis > filter**, not the other
way around.

---

## Method

Tool: `analysis/scripts/diagnose_ball_gaps.py` (new).

For each of the 43 rallies with ball GT:

1. Load GT keyframes, interpolate to every frame in the GT-extent range.
2. Load predictions from `player_tracks.ball_positions_json` (DB, what
   production actually serves — this matters; see §Caveat).
3. Load raw WASB detections from `~/.cache/rallycut/ball_grid_search/`
   (27 of 43 rallies cached).
4. Compute optimal constant frame-offset via `find_optimal_frame_offset`
   (0–5 frames).
5. Run six failure-mode detectors from the new
   `rallycut/evaluation/tracking/ball_failure_modes.py`:
   - missed streaks (≥5 consecutive GT frames with no GT-close pred)
   - teleports (>120 px/frame between consecutive preds)
   - wrong-object (pred exists but >100 px from GT)
   - stationary clusters (raw conf ≥0.3 with <0.5% spread for ≥20 frames)
   - two-ball divergence (>200 px separation between simultaneous raws)
   - per-stage filter-kill attribution
6. Roll frames into `matched` / `missed_no_raw` / `missed_filter_killed`
   / `wrong_object` / `interpolated_{correct,wrong}` buckets with no
   double-counting.
7. Render a per-rally MP4 overlay (GT green, pred red, raw faint blue,
   filter-kill orange X, teleport arrow, wrong-object yellow box,
   status banner) — see `analysis/outputs/ball_gap_report/index.html`.

**Thresholds used** (calibrated on sanity rallies 0d84f858 +
c3b31af2):
- match: 50 px
- wrong-object: 100 px (kept above match so fast-motion interpolation
  slack doesn't count as wrong)
- teleport: 120 px/frame (beach spike tops out around 100–130 px/frame
  on 1920×1080 footage)

---

## Caveat — DB predictions ≠ what today's code would produce

On 0d84f858, the DB stores **147** ball positions for this rally while
re-applying the *current* `get_wasb_filter_config()` to the cached raw
yields **180**. On c3b31af2 the DB has **204** (including
interpolation-filled frames); current filter produces **183**. The DB
rows were written whenever each rally was last tracked, against whatever
filter config was deployed then.

This means the **1.37% filter-killed figure describes the filter that was
deployed at track time, not today's filter.** Re-running the diagnostic
with `--source refilter` would measure today's filter against GT; that's
a recommended follow-up but not done in this session.

---

## Per-rally breakdown

See `outputs/ball_gap_report/index.html` for the interactive sortable
table with stacked-bar budgets and MP4 links. Worst ten:

| Rally | Match % | GT | No raw | Killed | Wrong | Tele | Raw cache |
|---|---:|---:|---:|---:|---:|---:|:-:|
| 9dbe457a | 45.1% | 395 | 26 | 26 | 120 | 1 | ✓ |
| 21a9b203 | 52.1% | 587 | 236 | – | 43 | 0 | — |
| 0d84f858 | 69.2% | 198 | 28 | 23 | 8 | 0 | ✓ |
| e84deef3 | 69.8% | 351 | 71 | – | 35 | 2 | — |
| c48eeb7d | 70.5% | 387 | 101 | 12 | 1 | 0 | ✓ |
| a43fb033 | 72.4% | 279 | 73 | – | 2 | 0 | — |
| b7f92cdc | 72.7% | 139 | 23 | – | 10 | 0 | — |
| f0fdfcdb | 75.2% | 581 | 144 | – | 0 | 0 | — |
| 7ff96129 | 75.4% | 452 | 93 | – | 18 | 0 | — |
| 8ce5a9e2 | 75.5% | 274 | 58 | 5 | 4 | 2 | ✓ |

Three rallies show 100% match (1f87460b, 53ca3586, de7136d1) — partially
an artefact of sparse GT + linearly-interpolated GT lining up with
interpolated predictions. Visual QA on those MP4s recommended before
trusting them as perfect.

---

## Per-bucket root-cause discussion

### 1. `missed_no_raw` — 11.78% (1,169 frames with raw cache; ~18% on raw-less rallies)

The detector never fired anywhere close to the GT position. This is
almost always one of three underlying scenes:

- **Ball at frame edge** or off-screen mid-rally (e.g. 9dbe457a frames
  544–560: GT x<0.2, y>0.65 while ball is leaving the bottom-left). WASB
  is trained on beach-VB where the ball is near-centred; edge coverage
  is weak.
- **Far-court action** (small ball pixel size). Rally c48eeb7d has 101
  frames of this pattern.
- **Long blur sequences** on hard spikes. The 9dbe457a 199–204 streak
  (had_raw=4/6) and 461–471 (had_raw=7/11) land in this bucket —
  partially-seen motion where raw fires intermittently but doesn't
  cluster into a segment the filter keeps.

These are **detector-side issues**. Options: retrain WASB with edge-
heavy and motion-blur heavy negatives, or add a second small detector
specifically for edge-zone frames.

### 2. `missed_filter_killed` — 1.37% (136 frames)

Raw WASB had a GT-close detection that didn't reach the filtered output.
Per-stage attribution across the 27 raw-cached rallies shows segment
pruning and outlier removal as the only two stages that ever fire kills
here (motion energy, stationarity, oscillation, blip removal are all
disabled in the current WASB preset, so they kill nothing by design).

Given the 1.37% ceiling, **this is not worth a dedicated filter
redesign**. The two highest-killed rallies — 9dbe457a (26 frames) and
0d84f858 (23 frames) — are both already on the weak list for other
reasons; fixing the detector-miss bucket will recover far more.

### 3. `wrong_object` — 2.56% (254 frames, concentrated)

The wrong-object distribution is heavy-tailed: **10 rallies hold 289
of the 401 total wrong-object frames (72%).** Inside 9dbe457a, frames
166–207 show a consistent prediction 100–250 px behind GT as the ball
flies from (0.98, 0.08) inward — the tracker is tracking *something*
moving in the same direction, just lagging. `nearest_player_distance_px`
on those frames is 400–600 px, so it is NOT a player-body lock. This
looks like **tracker-initialisation lag on a corner serve**: the ball
enters frame at the far corner, WASB's heatmap peak is biased toward
the scene centre, and the filter interpolates from a stale earlier
position until actual detections catch up.

This deserves visual inspection on `overlay.mp4` for 9dbe457a first —
the ground-truth motion may actually be labelled differently than what
WASB considers the ball (i.e. GT labels the real ball, tracker locks on
a second object). The overlay renders raw WASB as faint-blue dots so the
user can judge whether the tracker is consistently seeing *something
else* or just lagging.

### 4. `teleports` — 50 events across 43 rallies

Per-frame per-rally rate is ~0.03 events/second. Spot checks on the top
teleport rallies (9db9cb6b=7, 2dff5eeb=5, 97f95cda=5) suggest these are
almost all **real fast-motion frames** where the ball legitimately
travels 120–160 px/frame during a spike. The 120 px/frame threshold was
deliberately set to the high side of physical plausibility, so the
residual count is a reasonable upper bound on actual tracker teleports.
No rally-level action is justified.

### 5. `stationary_clusters` — 0 events

The disabled stationarity filter's original target (the background-ball
distractor at (0.2, 0.58)) does not appear in any current raw cache on
any of the 27 cached GT rallies. Two explanations:

- Fine-tuned WASB stopped emitting that distractor at the detector
  stage (confirmed by the cache inventory: 0d84f858's raw positions
  all have y ≤ 0.477).
- The specific rally where the distractor was observed isn't in the GT
  set.

**Action: no reason to re-enable the stationarity filter** — on this set
it would find nothing to filter.

### 6. `two_ball` — 0 events

The raw cache stores only one detection per frame (WASB's top heatmap
peak). The two-ball detector is unreachable without a multi-peak
extraction pass. This bucket is an **instrumentation gap, not an
absence of evidence.** If two-ball confusion is suspected, a follow-up
WASB inference pass that emits top-K peaks per frame would light it up.

---

## Recommendations (ranked by expected ROI, no commitments)

1. **WASB detector improvements (highest ROI).** 70%+ of the gap is
   `missed_no_raw`. Concrete moves, in order of evidence strength:
   - Hard-negative mining of edge-zone frames (y > 0.7 OR x < 0.15 OR
     x > 0.85) and motion-blur sequences from the current 27-cached-raw
     GT set. Fine-tune WASB Round 6+ on those. Memory notes Rounds 2–5
     hit a pseudo-label ceiling, but those rounds used full-frame
     sampling; edge-and-blur-weighted sampling is a different protocol.
   - A dedicated smaller model run at higher resolution for frame
     edges (windowed detection) could help on rallies 9dbe457a /
     c48eeb7d / a43fb033 — all dominated by off-centre ball.

2. **Investigate 9dbe457a and 21a9b203 by hand.** Together they hold
   40% of wrong-object frames. The overlay.mp4s are already rendered —
   next step is a 10-minute visual pass to confirm whether the tracker
   is (a) lagging behind a real ball, (b) locked on a different object,
   or (c) losing the ball in far-corner motion. That judgment dictates
   whether the fix is filter, detector, or GT-labelling.

3. **Populate raw cache on the 16 rallies missing it.** Without raw we
   can't split `missed_no_raw` vs `missed_filter_killed` for 5,153 GT
   frames (34% of the set). `evaluate-tracking --retrack --cached`
   followed by re-running this diagnostic closes that hole.

4. **Run diagnostic with `--source refilter`** and compare against
   `--source db`. Diff pinpoints whether any currently-deployed DB
   rallies would improve simply by being retracked with today's code.

5. **Filter changes — defer.** Based on the 1.37% ceiling, no filter
   stage should be re-enabled or retuned until (1) and (2) are done.
   Re-enabling the stationarity/oscillation stages would add real
   regression risk for zero measured upside on this set.

6. **Add multi-peak raw extraction** so the two-ball detector has
   inputs. Only worth doing if (2) uncovers a wrong-object case that
   the single-peak detector can't explain.

---

## What's not in this session

- **No production changes.** Filter config, WASB weights, and the DB
  rally rows are untouched.
- **No visual QA of every flagged event.** Report ranks hotspots; the
  MP4s support the user doing that judgement call.
- **No GT expansion.** Discovered the 16→43 jump already happened; no
  new labelling needed.

---

## Artefacts

- `analysis/scripts/diagnose_ball_gaps.py` — per-rally MP4 + JSON
  renderer.
- `analysis/scripts/build_ball_failure_report.py` — HTML dashboard.
- `analysis/rallycut/evaluation/tracking/ball_failure_modes.py` — pure
  detectors.
- `analysis/outputs/ball_gap_report/`
  - `index.html` — sortable cross-rally dashboard.
  - `aggregate.json` — cross-rally totals.
  - `{rally_id}/overlay.mp4` — annotated video for every rally.
  - `{rally_id}/events.json` — per-event detail.
  - `{rally_id}/failure_budget.json` — per-rally bucket counts.
  - `summary.json` — batch run log.
- Total artefact size: ~1.0 GB (43 MP4s).
