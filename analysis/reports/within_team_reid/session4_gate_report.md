# Session 4 — Learned-ReID weight sweep gate report

_Oracle gate (#2) is **deferred to Session 5**: `scripts/production_eval.py` reads predictions from DB, not the retrack output, so running it here would re-measure current production and always report zero delta. Ship decision below is based on targets #1, #3, #4; oracle validation happens once retracked positions are persisted._

## Cross-rally rank-1 guard (acceptance #4 — weight-invariant)

- Head checkpoint: `analysis/weights/within_team_reid/best.pt`
- Cross-rally rank-1 (leave-one-out on adversarial gallery): **0.6935**
- Guard: ≥ 0.6830 → ✅ PASS

## Per-weight sweep

Held-out swap events (from baseline reid_debug, slice [34:58]): **24**.

| Weight | Retrack swaps still firing | Swap reduction | Worst rally HOTA drop | Oracle min Δ | Gate |
|--------|--------------------------:|---------------:|----------------------:|-------------:|:----:|
| 0.00 | 15/24 | +0.0% | n/a | n/a | ctrl |
| 0.05 | 15/24 | +0.0% | 0.00 pp | n/a | ❌ |
| 0.10 | 15/24 | +0.0% | 0.00 pp | n/a | ❌ |
| 0.15 | 15/24 | +0.0% | 0.00 pp | n/a | ❌ |
| 0.20 | 15/24 | +0.0% | 0.00 pp | n/a | ❌ |

## Oracle deltas (vs W=0 control, percentage points)

| Weight | player_attribution_oracle | serve_attribution_oracle | court_side_oracle | score_accuracy |
|--------|:---:|:---:|:---:|:---:|
| 0.00 | — | — | — | — |
| 0.05 | n/a | n/a | n/a | n/a |
| 0.10 | n/a | n/a | n/a | n/a |
| 0.15 | n/a | n/a | n/a | n/a |
| 0.20 | n/a | n/a | n/a | n/a |

## Failure reasons
- **W=0.05**:
  - swap_reduction 0.00% < target 40% (15/15 held-out events still firing)
- **W=0.10**:
  - swap_reduction 0.00% < target 40% (15/15 held-out events still firing)
- **W=0.15**:
  - swap_reduction 0.00% < target 40% (15/15 held-out events still firing)
- **W=0.20**:
  - swap_reduction 0.00% < target 40% (15/15 held-out events still firing)

## Per-rally HOTA regressions (drops > 0.5 pp)
- None.

## Recommendation
**NO SHIP.** No weight clears acceptance target #1. Per the plan: do NOT
relax acceptance — return to Session 3 / propose architectural change.

## Diagnosis — why all weights produce byte-identical output

Verified via `scripts/debug_learned_reid_trace.py` on rally `072fb8c5`
(clean, global identity commits) and `fad29c31` (swap-heavy, global identity
reverts):

- **Learned cost IS firing.** On rally `072fb8c5` at W=0.20, 20/20
  `_compute_assignment_cost` calls had both embeddings active; learned
  contribution ranged 0.000–0.101 (mean 0.038).
- **…but doesn't flip the argmin.** Per-GT predIdSpans are byte-identical
  between W=0.00 and W=0.20 on the same rally. The HSV cost already
  dominates the assignment on clean rallies, where same-kit teammate
  Bhattacharyya distance is small AND correct (each GT maps to its
  intended pred cleanly).
- **On swap-heavy rallies, global identity REVERTS regardless of weight.**
  Rally `fad29c31` warns `Global identity: track 6 lost 191/299 frames
  (36% remaining) — reverting` at both W=0.00 and W=0.20. The coverage
  guard at `global_identity.py:261-273` nullifies any assignment that
  reduces a track's frame coverage by >50 %. The learned cost tries to
  reassign swap-frame spans to the correct canonical player, which
  naturally reduces the losing track's coverage — so the guard fires,
  and the ENTIRE global identity optimization is reverted back to the
  pre-optimization state (stabilize_track_ids output). The learned cost's
  decision never lands in the final output.

**Net effect**: at this integration point, segment-level cost tuning is
structurally unable to fix within-team swaps. The swaps we care about are
exactly the ones that trigger the coverage-revert guard.

## Implications for Session 5

1. **Learned ReID's signal is real** (cross-rally rank-1 0.6935 confirms
   the head didn't bit-rot; held-out rank-1 0.500 from Session 3 shows
   it separates teammates) but the plumbing surfaces where that signal
   can commit are NOT at `optimize_global_identity`.

2. **Candidate commit points, ranked**:
   - `stabilize_track_ids` / `tracklet_link.link_tracklets_by_appearance`
     — merge decisions happen BEFORE the coverage guard runs; learned
     cost could gate an appearance merge just like HSV does today.
   - `convergence_swap.detect_convergence_swaps` — operates on primary
     tracks with simple swap semantics (no coverage guard). A learned-ReID
     signal at convergence frames could disagree with the bbox+color
     signal that currently drives the swap decision.
   - `BoT-SORT ReID` — would require plumbing the head into `_extract_reid_embeddings`
     at `player_tracker.py:2090` and using it instead of (or in addition
     to) the OSNet embedding. Biggest potential lift; largest change.

3. **Global identity coverage guard** is the confounder. Any fix that
   lets segment-level cost changes commit needs to either relax the
   guard (dangerous — it protects against bad merges) or change the
   integration point (safer — do the work earlier, before the guard).

## Runtime
- W=0.00: 0.1 min (cached)
- W=0.05: 66.2 min (full YOLO + backbone+head on 42/43 rallies)
- W=0.10: 0.2 min (cached)
- W=0.15: 0.1 min (cached)
- W=0.20: 0.1 min (cached)
- Total: ~67 min end-to-end

## Artifacts
- `sweep/w{0.00..0.20}/tracking.json` — per-weight per-rally HOTA + metrics
- `sweep/w{0.00..0.20}/audit/*.json` — per-rally pred-exchange events
- `scripts/debug_learned_reid_trace.py` — single-rally activation tracer
- `analysis/weights/within_team_reid/best.pt` — head checkpoint (SHA `a1c62e3719e3`)
