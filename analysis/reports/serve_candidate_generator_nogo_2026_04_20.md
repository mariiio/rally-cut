# Serve Candidate Generator — NO-GO

**Date**: 2026-04-20
**Decision**: **NO-GO**. Closed. The 74 no-candidate serve FNs are not recoverable by a candidate-generation fix; they are a ball-tracker emergence-recall problem.
**Status supersedes**: `serve_candidate_generator_plan_2026_04_20.md` (plan file now reflects closure).

## TL;DR

Ran the gated 3-step validation from the follow-up plan: mode decomposition → heuristic measurement → oracle injection. All three gates passed their intermediate thresholds, but the aggregate oracle lift was only **+0.15pp F1** — far below the +1.5pp ship gate. Closing the workstream.

Root cause: Mode C ("tracker lock-on latency") accounts for 59/74 serves (80%) — they *look* like a candidate-generation problem (no generator fires, ball is present in frame) but the ball's first *tracked* frame is 2-5 frames after the true GT serve contact. Even a perfect first-stable-ball generator would fire systematically late, outside the ±7f match tolerance for many serves.

## Step 1 — Mode decomposition (PASS gate)

For each of the 74 no-candidate serve FNs, classified by ball-density trace in the ±30f window around the GT serve frame:

| Mode | Description | Count |
|---|---|---|
| A | Off-screen near-side server (ball appears from near baseline) | 5 |
| B | Occluded far-side server (ball emerges from far arc) | 1 |
| C | Tracker lock-on latency (ball tracked, no generator fires) | 59 |
| unknown | Doesn't match cleanly | 9 |

**Gate 1 (|A|+|B|+|C| ≥ 50): PASSED** (65).

## Step 2 — Heuristic measurement (PASS gate)

Swept stability thresholds `(N consecutive frames, min ball confidence T)` on Mode C. Best config `(N=5, T=0.20)`:
- Mode C ≤7f coverage: **71.2%** (42 of 59 serves reachable by a first-stable-ball generator within match tolerance)
- Modes A, B, unknown: **0%** ≤7f coverage (ball appears 13-28 frames after GT — structurally unrecoverable by first-stable-ball rule)

**Gate 2 (Mode C ≤7f coverage ≥ 60%): PASSED** (71.2%).

## Step 3 — Oracle injection (FAIL gate)

Injected the idealized "first-stable-ball" candidate at the Step 2 config per rally. Re-ran full GBM + decoder pipeline. Aggregate F1 lift across 68-fold LOO:

- Serve recall: +5.4pp (42 → ~45 TPs of 364 GT serves)
- **Aggregate Contact F1: +0.15pp** (88.0% → 88.15%)

Why the tiny aggregate lift despite 42 serves reachable? The per-rally oracle injection adds ~4 recovered serves per video but also introduces ~3 false positives per video (the first-stable-ball heuristic mis-fires on pre-serve ball reflections, practice throws, and the 9 Mode "unknown" rallies). Precision-recall balance is nearly neutral.

**Gate 3 (aggregate F1 lift ≥ +1.5pp): FAILED**.

## What this actually reveals

The 53 Mode C cases we can *almost* reach (but at distance >7f) aren't missed by the generator; they are missed by **ball tracking**. The ball's first visually-detected frame in a rally is systematically 2-5 frames after the actual serve contact (racket-hit instant). The tracker's emergence recall at low-confidence rally-start frames is the bottleneck.

Shipping a serve candidate generator on top of this tracker behavior merely shifts the noise vs signal tradeoff — can't recover what was never detected in the first place.

## Paths that would actually move the needle (not this cycle)

1. **WASB ball-tracker: low-confidence emergence recall**. Most of the 53 near-misses have the ball detected at `conf < 0.30` for 1-3 frames around the true GT serve, then stabilizing above threshold. Relaxing the tracker-emergence threshold (possibly with a rally-start-specific relaxation window) would lift Mode C directly.
2. **Pre-serve ball seed from pose**. Server's wrist-release frame is detectable from YOLO-Pose (77% coverage). Using the serving player's arm-swing apex to seed a candidate would catch Mode C even when ball-tracker lags.
3. **Cross-modality gate**: if both tracker confidence is low AND pose shows serve-arm trajectory, inject a candidate. Combines the two weak signals.

None of these are candidate-generator work — they're upstream (ball tracking) or cross-modality (pose integration) investments.

## Artifacts preserved

- `analysis/scripts/diagnose_serve_fn_modes.py` — Step 1 decomposition harness.
- `analysis/scripts/measure_serve_generator_heuristics.py` — Step 2 threshold sweep.
- `analysis/scripts/inject_oracle_serve_candidates.py` — Step 3 oracle injection.
- `analysis/reports/serve_fn_modes_2026_04_20.md` — per-mode ball-confidence traces.
- `analysis/reports/serve_generator_heuristics_2026_04_20.md` — (N, T) sweep results.

## What this closes

Serves no longer a near-term improvement target via candidate generation. Future serve-recall work should target WASB emergence recall or pose-based seeding. Memory updated at `memory/MEMORY.md` → Action Detection section.
