# Player Attribution Investigation Design

## Context

Player attribution accuracy is 84.1% (oracle, permutation-invariant) against a 92.7% oracle ceiling. The contact detector picks the nearest player to the ball at each contact, with pose-based and temporal model overrides. Two improvements shipped (+2.5pp wrist keypoints, +4.2pp pose GBM). Three approaches were tried and failed (trajectory, velocity, team chain). The remaining 8.6pp gap needs investigation before committing to a fix approach.

Attribution errors cascade downstream: wrong player leads to wrong team assignment, wrong court_side, wrong action classification, wrong score tracking, and wrong landing heatmap attribution.

## Goal

Categorize the 16% of attribution failures by error type, validate the untried Viterbi sequence-decoding approach, and identify which signals could close the gap. The investigation outputs a structured report that informs the fix design.

## Stage 1: Run Existing Diagnostics (No New Code)

### 1a. Error Taxonomy

Run `analysis/scripts/diagnose_attribution.py` on the full 62-video GT set.

```bash
cd analysis && uv run python scripts/diagnose_attribution.py
```

**Expected output:**
- Error type counts: `correct`, `wrong_nearest`, `wrong_team`, `missing_player`, `no_player`, `stale_gt`
- Per-action breakdown (serve/receive/set/attack/dig/block)
- Cascade analysis: attribution error -> court_side error -> action error
- Per-rally worst offenders (sorted by error rate)

**What we learn:** Where the 16% of errors are concentrated. This determines Stage 2 scope.

### 1b. Viterbi Sequence Validation

Run `analysis/scripts/validate_viterbi_attribution.py` with a sigma sweep.

```bash
cd analysis
uv run python scripts/validate_viterbi_attribution.py
uv run python scripts/validate_viterbi_attribution.py --distance-sigma 0.05
uv run python scripts/validate_viterbi_attribution.py --distance-sigma 0.20
```

**Expected output:**
- Current vs Viterbi attribution accuracy (total + per-action)
- Current vs Viterbi team accuracy
- Net fixes vs regressions
- Sensitivity to distance_sigma

**What we learn:** Whether volleyball sequence rules (serve->receive->set->attack, 3-touch limit, team alternation) improve attribution even with distance-only emissions. If yes, integrating pose scores as Viterbi emissions is the clear next step.

**Critical files:**
- `analysis/scripts/diagnose_attribution.py` — error taxonomy (run as-is)
- `analysis/scripts/validate_viterbi_attribution.py` — Viterbi comparison (run as-is)
- `analysis/scripts/eval_action_detection.py` — shared data loading (`load_rallies_with_action_gt`, `match_contacts`)

## Stage 2: Targeted Deep Dive (New Script)

Based on Stage 1 results, create `analysis/scripts/investigate_attribution.py` focused on the dominant error bucket.

### If `wrong_nearest` dominates (GT player was a candidate but not #1):

1. **Proximity margin histogram**: For each error, compute `dist(chosen) - dist(GT)`. Small margins = ambiguous, large = systematic.
2. **Pose model confidence at errors**: Extract pose attribution confidence for both the chosen and GT player. Categorize: (a) pose model not available, (b) pose model low confidence, (c) pose model confident but wrong.
3. **Feature delta analysis**: For wrong_nearest errors with 2+ candidates, extract full 30-dim pose vectors for chosen vs GT player. Identify which features differ most via permutation importance on the error subset.
4. **Viterbi cross-reference**: For each wrong_nearest error, check if Viterbi would have picked the correct player. Quantify the overlap.
5. **Action-conditioned analysis**: Break down errors by GT action type. Do sets have different error patterns than digs?

### If `wrong_team` dominates:

1. Cross-reference with score_investigation Phase 0 findings (TEAM_MAP 54%, ABSTENTION 40%)
2. Analyze team assignment confidence at error points
3. Check if `classify_teams()` or `match_team_assignments` is the source

### If `missing_player` dominates:

1. Frame-level player count at error contacts
2. Which players go missing (far-court? occluded? near net?)
3. Tracking dropout analysis (player_filter.py issues vs YOLO detection)

### Output format

The script produces a structured report:
```
=== Attribution Investigation Report ===

Error Taxonomy:
  correct:        N (X%)
  wrong_nearest:  N (X%)
  wrong_team:     N (X%)
  missing_player: N (X%)
  ...

Viterbi Comparison:
  Current:  X% (N/M contacts)
  Viterbi:  X% (N/M contacts)
  Fixes:    N, Regressions: N, Net: +/-N

Deep Dive (wrong_nearest):
  Proximity margin: median=X, p25=X, p75=X
  Pose confidence: unavailable=N%, low=N%, high-wrong=N%
  Top discriminative features: [feature_name: importance]
  Viterbi would fix: N/M (X%)

Recommended Fix Approach:
  [Based on findings]
```

## Verification

- Stage 1 outputs should be internally consistent: error counts should sum to ~16% of evaluable contacts
- Cross-validate against production_eval canonical metrics (84.1% oracle, 92.4% action_acc)
- Sanity check per-action patterns against known baselines (serve ~95%, dig ~71%)

## Critical Files

| File | Role |
|------|------|
| `analysis/scripts/diagnose_attribution.py` | Existing error taxonomy script |
| `analysis/scripts/validate_viterbi_attribution.py` | Existing Viterbi validation script |
| `analysis/scripts/investigate_attribution.py` | **New** — Stage 2 targeted analysis |
| `analysis/rallycut/tracking/pose_attribution/features.py` | Feature extraction (reuse) |
| `analysis/rallycut/tracking/contact_detector.py` | Attribution logic (reference) |
| `analysis/scripts/eval_action_detection.py` | Data loading utilities (reuse) |

## Non-Goals

- No changes to attribution logic in this investigation
- No model training
- No changes to production pipeline
- No new features or models — investigation only
