# Score Tracking Investigation — Design Spec

## Context

Beach volleyball score tracking from a single consumer camera currently achieves 57.6% accuracy (per-rally serving-team prediction). The formation-based predictor reaches 72-76% on LOO-CV but abstains ~35% of rallies, falling back to a 52% contact-based signal. Each rally is predicted independently with no temporal chaining.

The goal is near-perfect score tracking. The investigation revealed that the previous "HMM is dead" conclusion (p_stay=0.515) only applies to the soft transition prior. The deterministic volleyball rule `serving_team[i+1] = point_winner[i]` provides a hard constraint that was never exploited. This design pursues that insight through 4 incremental phases.

## Phase 0: Failure Analysis (0.5 days)

**Script**: `analysis/scripts/diagnose_formation_errors.py`

**Purpose**: Categorize the ~28% formation predictor errors to determine Phase 1 priorities.

**Error buckets**:

| Bucket | Description | Diagnosis method |
|--------|-------------|------------------|
| Abstention | Separation ratio below margin but > 1.0 | `1.0 < ratio < margin` |
| Team mapping | Right side identified, A/B label inverted | Flipping prediction matches GT |
| True formation error | Wrong side had larger separation | Correct side had smaller separation |
| Data quality | Missing players, positions, or court split | < 2 tracks or no split_y |

**Per-error data**: video_id, rally_id, near_sep, far_sep, ratio, n_tracks per side, semantic_flip value, whether flip-correction would fix it.

**Output**: Summary table with bucket counts + per-video breakdown (worst videos first).

**Decision matrix**:
- Abstention dominates -> graduated confidence + fallback signals (Phase 1a, 1b)
- Team mapping dominates -> fix semantic_flip logic (Phase 1d)
- True errors dominate -> formation ceiling reached, new signals critical (Phase 1b, 1c)
- Data quality dominates -> upstream tracking/court fixes needed first

**Reuses**: `_load_all()` pattern from `diagnose_player_at_serve.py`, calls `_find_serving_team_by_formation()` with instrumentation.

**Files**:
- Creates: `analysis/scripts/diagnose_formation_errors.py`
- Reads: `analysis/rallycut/tracking/action_classifier.py` (`_find_serving_team_by_formation`)

## Phase 1: Fix Per-Rally Signal (2 days, target 78-82%)

### 1a. Graduated Confidence

Replace the hard abstention (return `None` when ratio < margin) with a soft confidence:

```
confidence = clamp((ratio - 1.0) / (margin - 1.0), 0.0, 1.0)
```

Even 55% confidence is better than the 52% contact-based fallback. The soft confidence feeds naturally into the Phase 2 Viterbi as emission probability.

**Change in**: `action_classifier.py:_find_serving_team_by_formation()` — emit `(team, confidence)` instead of `(team | None, float)`.

### 1b. Ball-Direction Signal

Ball at serve moves toward the net (from server's side). Extract from existing `ContactSequence.ball_positions`:
- Direction of ball motion in first 10-15 frames after first detection
- Ball moving toward near side -> server is on far side (and vice versa)

Already partially prototyped in `_ball_moving_toward_net()` (action_classifier.py line 248).

**Caveat**: WASB warmup lag (~60 frames / 2s) means ball direction may be unavailable for early serves. Phase 0 reveals coverage.

### 1c. Ensemble Predictor

Lightweight logistic regression or GBM combining:
- `formation_ratio` (continuous)
- `formation_side` (near/far)
- `ball_initial_dy` (ball direction toward near/far)
- `ball_initial_side` (which side of net the ball starts on)
- `contact_serve_team` (from first "serve" action)
- `n_tracked_players` (data quality proxy)

Evaluated via LOO-video CV in the existing `eval_score_tracking.py` harness (supports pluggable `Callable[[RallyData], str | None]` predictors).

### 1d. Semantic Flip Fix (conditional on Phase 0)

If Phase 0 reveals team-mapping errors dominate: use the first high-confidence formation prediction per video as an anchor and propagate team identity from there, rather than trusting `sideSwitchDetected` from match_analysis_json.

**Files**:
- Modifies: `analysis/rallycut/tracking/action_classifier.py` (formation predictor, new ensemble)
- Modifies: `analysis/scripts/eval_score_tracking.py` (new ensemble predictor registration)
- Reads: `analysis/rallycut/statistics/landing_detector.py` (ball trajectory data)

## Phase 2: Point-Winner Detection + Cross-Rally Viterbi (3 days, target 88-92%)

### 2a. Point-Winner Detector

Determine which team won each rally from terminal action analysis:

**Primary signal** — Terminal action + landing side:
- Last classified action tells us what happened (attack, block, failed dig, etc.)
- `landing_detector` detects where attacks land (player feet at ground plane, <2px accuracy)
- Logic rules:
  - Terminal attack on team A's court + no subsequent action -> team A lost
  - Serve with no receive -> ace, server's team won
  - Attack error (out of bounds) -> attacking team lost

**Secondary signal** — Ball final position:
- Last few ball positions projected to court coordinates
- Ball on team A's court + rally ended -> team A likely lost
- Weaker signal, used as corroboration

**Estimated accuracy**: 65-75% point-winner detection. Terminal action classification is ~92%; the mapping to point winner depends on court-side accuracy.

**Output**: `PointWinnerResult(team: str, confidence: float, method: str)` per rally.

### 2b. Cross-Rally Viterbi with Hard Transition Constraint

**State**: `serving_team[i]` for each rally `i` in a video.

**Two observations per rally**:
1. Formation/ensemble serving-team prediction (Phase 1, ~80%, emission probability)
2. Point-winner-implied serving-team for next rally (~70%, transition constraint)

**Transition model**:
- When point_winner confidence is high: `P(serving[i+1] = winner[i]) = 1.0` (deterministic)
- When point_winner confidence is low: `P(stay) = 0.515` (soft prior, minimal information)

**Why this works**: A correct point_winner at rally i hard-locks `serving[i+1]`. With 70% point-winner accuracy, ~70% of transitions are deterministic. Errors only accumulate where BOTH point-winner and formation fail consecutively (~6% = 0.3 x 0.2).

**Side-switch checkpoints**: At known score thresholds (every 7 combined points sets 1-2, every 5 set 3), verify whether predicted side switch aligns with detected side switch. Mismatch signals accumulated error -> force Viterbi re-evaluation.

**Architecture note**: The cross-rally Viterbi is a standalone module that runs AFTER all per-rally classification is complete, taking per-rally RallyActions as input. It does NOT modify or couple into the within-rally `classify_rally_actions()` pipeline.

**Implementation**: Extend the existing Viterbi logic from `diagnose_serve_chain_viterbi.py` to support:
- Dual observation streams (formation + point-winner)
- Hard transition constraints when point-winner is confident
- Side-switch checkpoint verification

**Files**:
- Creates: `analysis/rallycut/scoring/point_winner_detector.py`
- Creates: `analysis/rallycut/scoring/cross_rally_viterbi.py`
- Reads: `analysis/rallycut/statistics/landing_detector.py` (landing positions)
- Reads: `analysis/rallycut/tracking/action_classifier.py` (terminal actions)
- References: `analysis/scripts/diagnose_serve_chain_viterbi.py` (existing Viterbi logic to extend)

## Phase 3: User Correction Layer (2 days, target 95-98%)

### 3a. User Anchors

Two user-provided inputs with outsized impact:

1. **First-serve anchor**: Which team served rally 1? Single click. Pins the start of the Viterbi chain, eliminates initial-team-assignment ambiguity.

2. **Final score** (optional): e.g., "21-18". Enables count-constrained Viterbi (already implemented in `diagnose_serve_chain_viterbi.py:viterbi_count_constrained()`). The constraint that exactly N_A rallies have serving_team=A bounds total errors.

### 3b. Confidence-Based Correction UI (backend)

After the automated pass (Phases 1+2):
- System identifies low-confidence rallies
- Exposes confidence scores per rally via API
- When user corrects a rally, that correction becomes a hard anchor
- Viterbi re-decodes surrounding rallies (cascade correction: 1 user fix may resolve 2-3 adjacent errors)

**Re-decode is pure DP** — no ML inference, instant response.

### 3c. Backend API

- `PATCH /api/rallies/:id/score-correction` — user provides serving_team override
- `POST /api/videos/:id/score-decode` — triggers re-decode with all anchors
- Response includes per-rally serving_team + confidence + changed flags

**Note**: Frontend UI (score timeline, confidence visualization, jump-to-uncertain) is a separate follow-up task, not in scope for this spec.

**Files**:
- Modifies: `api/src/routes/` (new score-correction endpoints)
- Creates: `analysis/rallycut/scoring/score_decoder.py` (entry point combining all phases)
- Reads: `analysis/scripts/diagnose_serve_chain_viterbi.py` (count-constrained Viterbi to reuse)

## Verification Plan

### Phase 0
- Run `diagnose_formation_errors.py` on all GT rallies
- Verify bucket counts sum to total errors
- Compare per-video error rates against known problem videos (0a383519, 4f2bd66a, b026dc6c)

### Phase 1
- LOO-video CV via `eval_score_tracking.py` for each signal independently and ensemble
- Gate: ensemble accuracy >= 78% on 304-rally score GT set
- Compare against baseline formation (72-76%) to confirm improvement is real

### Phase 2
- Evaluate point-winner detector accuracy against GT (derive GT point_winner from consecutive serving_team labels: `gt_winner[i] = gt_serving[i+1]`)
- Evaluate full cross-rally Viterbi via LOO-video CV
- Gate: score_accuracy >= 85% automated (no user anchors)
- Per-video breakdown to identify remaining problem videos

### Phase 3
- Evaluate with simulated user anchors (first-serve from GT + final score from GT)
- Gate: score_accuracy >= 95% with first-serve + final-score anchors
- Measure cascade effect: how many rallies are fixed per user correction

### End-to-end
- Run `production_eval.py` with full pipeline integration (minimum 2 reruns for variance check, std < 0.5pp)
- Confirm no regression on other metrics (action_accuracy, contact_f1, player_attr)

## Summary

| Phase | Target | Effort | Key Innovation |
|-------|--------|--------|----------------|
| 0: Failure analysis | Categorize errors | 0.5 days | Actionable error taxonomy |
| 1: Fix per-rally signal | 78-82% | 2 days | Graduated confidence + ball direction ensemble |
| 2: Point-winner + Viterbi | 88-92% | 3 days | Hard transition constraint (deterministic, not soft prior) |
| 3: User correction | 95-98% | 2 days | Anchor-based cascade re-decode |
| **Total** | **95-98%** | **~7.5 days** | |

Each phase ships independently and is measurable via existing eval infrastructure.
