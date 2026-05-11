# Serve-Peak Prepend Synthesis (v1.3)

**Date:** 2026-05-11
**Status:** Design — pending implementation plan
**Workstream context:** Sub-2.C (extends v1.1 synthetic-serve placement and the v1.2 seq-anchored rescue). Targets rallies where the contact detector misses the serve **entirely** — not because synthesis fails to place it, but because synthesis never fires (the first detected contact is the *receive*, mis-classified as serve).

## Goal

When `classify_rally_actions` produces a first action labeled `serve` but MS-TCN++ has a strong serve-class peak significantly earlier in the rally, **prepend a synthetic serve at the peak frame** and re-classify the misidentified first action.

Default-on production change. Conservative gate (calibration-locked) prevents false-prepends on correctly classified rallies.

## Motivation

### The gap v1.1 doesn't cover

v1.1 (`pick_synthetic_serve_frame`) places synthetic serves correctly when synthesis *fires* — but synthesis only fires via two paths:
1. `_make_synthetic_serve` invoked in `classify_rally` when the first detected contact looks non-serve.
2. `repair_action_sequence` Rule 0 (disabled in production).

When the first detected contact is N seconds *after* the actual serve, the default labeling rule "first contact = serve" mis-labels the receive as a serve. Neither path fires. **No synthesis attempt happens — the serve is lost.**

Canonical case (`wawa/8c49e480`, visually verified far-side server, fully on-screen):
- GT first action: serve at frame 101 (`abs t = 39.54s`)
- Pipeline first detected contact: frame 426 (`abs t = 44.96s`) → labeled "serve" by default
- MS-TCN++ serve-class peak at frame **109**, prob **0.990**
- The signal exists. No code path consults it.

### Fleet diagnostic, 2026-05-11

Across all 328 GT rallies with stored tracking + `first_classified_action.type == "serve"`:

| Cluster | Count | %  | Recoverable by v1.3 |
|---|---:|---:|:---:|
| pipeline_already_correct (\|pred − GT\| ≤ 15) | 253 | 74.9% | ✗ already correct |
| near_miss_other (\|pred − GT\| 16–60) | 55 | 16.3% | partial |
| **target rallies (pred − GT > 60 + strong MS-TCN++ peak before pred)** | **27** | **8.0%** | ✓ |
| pre_serve_artifact (pred fires before GT) | 9 | 2.7% | ✗ different problem |
| late_real_serve other | 4 | 1.2% | ✓ caught by same gate |

### Visual validation (2026-05-11)

User spot-checked 8 candidate rallies. The "off-screen-server" label was misleading — the real pattern is: **MS-TCN++ catches serves the contact detector misses, regardless of server visibility.** 4 of 8 verified cases were off-screen; the other 4 were fully on-screen serves that the contact detector simply missed (weak parabolic break, fast flat serves, etc.).

### Static FP-sweep result (2026-05-11)

At the proposed gate (`peak_p ≥ 0.95, min_gap ≥ 25`):

| Outcome | Count | Notes |
|---|---:|---|
| **TPs** (placement within ±15 of GT serve) | **23** | Real wins |
| Near-misses (placement +30 to +36 from GT) | 4 | Better than the current state (no serve at all) |
| Catastrophic FPs (peak misidentified, e.g., wawa/fb6e37bf) | **0** | The 0.95 floor naturally filters MS-TCN++ uncertain cases |
| Fires on pipeline_already_correct rallies | **0** | The 25-frame min_gap filters normal serve-buildup peaks |

Peak_p threshold has near-zero discriminating power between 0.85 and 0.97 (loses only 1 TP). The 0.95 floor is the conservative choice; it cleanly filters the one uncertain case in the diagnostic (`wawa/fb6e37bf` with peak_p=0.82).

### Why this is structural, not a hack

The "first contact = serve" default is a fallback rule from an earlier era of the pipeline. With MS-TCN++ available, we have a stronger signal: "the model thinks the action at frame X was a serve". When MS-TCN++ very confidently asserts a serve frame that's far before what we labeled as the first contact, the default rule is wrong, not the model.

v1.3 is the symmetric counterpart to v1.1:
- v1.1: serve-placement helper, fires when synthesis *does* fire.
- v1.3: serve-prepend pre-pass, fires when synthesis *should* fire but doesn't.

Together they cover both failure modes of synthetic-serve handling.

## Scope

**In scope:**

- New module `rallycut/tracking/serve_prepend.py` with:
  - Pure predicate `should_prepend_serve(sequence_probs, first_action_frame, first_action_serve_prob, rally_start_frame) -> int | None` returning peak frame or `None`.
  - Constants `SERVE_PREPEND_PEAK_FLOOR = 0.95`, `SERVE_PREPEND_MIN_GAP = 25`, `SERVE_PREPEND_FIRST_ACTION_SERVE_CEIL = 0.50`, `SERVE_PREPEND_GUARD_FRAMES = 15`.
- Integration in `classify_rally_actions` (in `rallycut/tracking/action_classifier.py:3505-3527`) AFTER `classify_rally` returns and BEFORE `repair_action_sequence` / Viterbi:
  - If `result.actions` is non-empty AND `result.actions[0].action_type == SERVE`:
    - Compute `first_action_serve_prob` from `sequence_probs` at `result.actions[0].frame`.
    - Call `should_prepend_serve(...)`.
    - If returns a peak frame F:
      - **Inject a synthetic `Contact` at frame F into the `ContactSequence`** (interpolate ball x/y from `ball_positions` near F; player_track_id=-1; court_side inferred from ball-vs-net position; is_validated=True; confidence=peak_prob).
      - **Re-run `classify_rally` on the injected sequence**, passing the same args. The existing rule engine (first contact = serve, touch counting, court-side propagation, team assignment, etc.) re-classifies every contact in the new serve's context. Old first contact at frame 426 in the canonical wawa case naturally becomes a downstream action (receive/dig/set/attack depending on rally structure) — no manual re-labeling.
      - Tag the new first action as `is_synthetic=True` and `player_track_id=-1` (server unknown).
  - This is the **fully robust** path: every downstream label decision (including PID attribution, court_side, team_assignment, serving_team) comes from `classify_rally`'s established logic rather than from a v1.3-specific re-labeling heuristic. The cost is one extra `classify_rally` invocation on rallies where the gate fires (~27/338 = 8% of GT rallies).
- Flag `_DISABLE_V13_PREPEND` module-level boolean in `serve_prepend.py` (default False) for the clean A/B harness, parallel to v1.1's `_DISABLE_V11_PLACEMENT`.
- Unit tests in `analysis/tests/unit/test_serve_prepend.py`:
  - `TestShouldPrependServe`: pure-predicate tests for each gate condition + boundary cases + calibration-lock.
  - `TestPrependIntegration`: end-to-end test on `wawa/8c49e480` (asserts synth serve at frame ~110 in the resulting `RallyActions`).
- Clean A/B fleet measurement script `analysis/scripts/measure_serve_prepend_clean_ab.py` (parallel to `measure_synthetic_serve_placement_panel.py`).

**Out of scope:**
- Modifying `pick_synthetic_serve_frame` itself (v1.1's logic is unchanged).
- Modifying `detect_contacts` (v1.2's logic is unchanged).
- Handling `pre_serve_artifact` cluster (9 rallies where pipeline fires *before* GT — different problem, deferred).
- Improving MS-TCN++'s peak alignment for the 4 "+30-36 frame" near-misses — those are a sequence-model accuracy issue, not a synthesis issue.

## Gate

```
def should_prepend_serve(
    sequence_probs: np.ndarray | None,
    first_action_frame: int,
    first_action_serve_prob: float,
    rally_start_frame: int,
) -> int | None:
    if sequence_probs is None:
        return None
    if first_action_serve_prob >= SERVE_PREPEND_FIRST_ACTION_SERVE_CEIL:
        return None  # first action is itself confidently a serve — don't override
    upper_excl = first_action_frame - SERVE_PREPEND_GUARD_FRAMES
    if upper_excl <= rally_start_frame:
        return None
    serve_idx = ACTION_TYPES.index("serve") + 1
    window = sequence_probs[serve_idx, rally_start_frame:upper_excl]
    if window.size == 0:
        return None
    peak_offset = int(np.argmax(window))
    peak_prob = float(window[peak_offset])
    peak_frame = rally_start_frame + peak_offset
    if peak_prob < SERVE_PREPEND_PEAK_FLOOR:
        return None
    if first_action_frame - peak_frame < SERVE_PREPEND_MIN_GAP:
        return None
    return peak_frame
```

All five gate conditions are conjunctive. Calibration-locked: the unit test asserts these constants match the calibration in the sweep.

## Predicted impact

**Panel (5-video, ~95 rallies):**
- From the FP-sweep, fires on **wuwu/fb7f9c23** at minimum (+1 panel FN fix; landed in the v1.2 residual `no_candidate` bucket — see `seq_anchored_rescue_v12_2026_05_11.md`). Potentially more depending on which rallies land in the panel.
- Expected panel F1: 0.896 → ~0.900–0.905 (modest, in line with v1.2's marginal but real bumps).
- Real-serve regressions: **0 expected** (gate filters by `first_pred − peak ≥ 25` which doesn't fire on correct rallies).

**Fleet (66 videos, 338 GT rallies):**
- +23 placement TPs (HIT_TOLERANCE=15)
- +4 placement near-misses (still better than baseline's "no serve")
- 0 catastrophic regressions
- Cascade effect: fixing the first action propagates to `servingTeam`, score tracking, player attribution — same multiplier as v1.1.

## Ship gate

- Real-side: 0 measured regressions on the GT panel (fleet-wide real-serve hit rate must be ≥ baseline 89.9%).
- Synth-side: ≥18 placement TPs on the fleet (75% of the predicted 23 — a buffer in case fleet drift affects a few cases).
- Visual confirm: spot-check a fleet sample of 5 newly-firing rallies post-deploy.

## Risks

- **MS-TCN++ argmax in long windows.** For very long rallies (>1000 frames), the global argmax might land on a *receive* peak rather than a serve peak. Mitigation: serve-class is a distinct probability channel; receive peaks are in a different channel. The argmax of serve-class specifically should be the serve.
- **Cascading re-classification.** When we prepend a serve, the entire downstream action sequence is re-labeled by `classify_rally`. If the rule engine has a bug or edge case that gets triggered only when a new first contact appears, v1.3 could surface it. Mitigation: integration tests cover both the "fires correctly" case (wawa/8c49e480) and the "doesn't fire on already-correct rally" case; the clean A/B measurement on the full 338-rally fleet detects any cascading regressions before deploy.
- **Compute cost.** v1.3 runs `classify_rally` twice on rallies where the gate fires (27/338 fleet, 8%). Each invocation is fast (~tens of ms); total fleet overhead is negligible. Production paths run synchronously so no architectural change needed.
- **One outlier in the diagnostic** (`wawa/fb6e37bf`, peak_p=0.82) is filtered by the 0.95 floor — confirming the floor is at the right place. If this changes (e.g., from a MS-TCN++ retrain), the calibration-lock unit test will surface it.

## Files

- New: `rallycut/tracking/serve_prepend.py`
- Modify: `rallycut/tracking/action_classifier.py` — insert pre-pass call in `classify_rally_actions` between line 3513 (`classify_rally` return) and line 3517 (`repair_action_sequence`).
- New: `analysis/tests/unit/test_serve_prepend.py`
- New: `analysis/scripts/measure_serve_prepend_clean_ab.py`

## Diagnostic artifacts (already produced)

- `analysis/scripts/diagnose_offscreen_server_candidates.py` (cluster identification)
- `analysis/scripts/prep_offscreen_visual_check.py` (visual-check table)
- `analysis/scripts/sweep_offscreen_gate_fp.py` (calibration sweep)
- `reports/offscreen_server_diagnostic.json` (per-rally signals, 338 rallies)
- `reports/offscreen_gate_sweep.json` (sweep results)
