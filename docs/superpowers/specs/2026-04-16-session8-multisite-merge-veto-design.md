# Session 8 — Multi-Site Learned-ReID Merge Veto

**Date**: 2026-04-16
**Status**: Design approved, pending implementation plan
**Predecessors**: Sessions 4 (additive cost, NO SHIP), 5 (occlusion resolver, NO SHIP), 6 (single-site merge veto, NO SHIP), 7 (minimal-processing A/B, KEEP merge chain)

## Motivation

Within-team identity accuracy in production tracking is limited by ~12 SAME_TEAM_SWAPs across the 43 GT rallies. A validation probe in Session 6 revealed the root cause:

- **Raw BoT-SORT output produces 0 real identity switches** across all 43 rallies (fragmentation only, never swaps).
- **All 12 SAME_TEAM_SWAPs are CREATED by the 7-pass post-processing merge/rename/split chain** that runs after BoT-SORT (5 merge/rename + 2 splits).
- The current DB state shows only ~3 swaps, but this is a *hybrid* of algorithmic output + the user's manual corrections. The honest algorithmic baseline is 12.

Session 6 proved the **architectural approach works**: adding a learned-ReID cosine veto at a single merge site (`tracklet_link.link_tracklets_by_appearance`, step 0c) reduced swaps by 25% with **zero HOTA regressions across every threshold tried**. The intervention is safe but narrow — downstream merge passes can recreate swaps that step 0c blocks (non-monotonic counts 9/11/9/11 across thresholds).

Session 7 ran a minimal-processing A/B (all merge passes off) and confirmed the merge chain **rescues 32/43 rallies** from fragmentation (HOTA drops 5.2 pp without it). The chain must stay; identity correction must happen **inside it**.

**Session 8's thesis**: extend the Session-6 veto from one merge site to every merge site that plausibly creates swaps. Same signal, more decision points covered, downstream passes can no longer recreate blocked merges.

## Ship Gate

All four must hold on 43 GT rallies:

1. **≥50% SAME_TEAM_SWAP reduction** (≤6 of 12 remaining).
2. **Zero per-rally HOTA drop > 0.5 pp** vs the post-Session-7 baseline (all merge passes enabled, `LEARNED_MERGE_VETO_COS=0`).
3. **Fragmentation delta ≤ 20%** (total unique pred-IDs across rallies).
4. **No `player_attribution_oracle` regression > 0.3 pp** via cached `production_eval`. Accepts the caveat that retracking loses action-to-player GT; comparison is against the current DB-backed oracle, not a post-retrack oracle.

Miss any one → NO SHIP, fall to Plan B.

## Architecture

### 1. Diagnostic pass (prerequisite)

Before any veto code, identify **which** merge passes create **which** swaps.

- Instrument `player_tracker.apply_post_processing` to call `build_rally_audit` after each of the 6 merge/rename passes.
- Attribute each of the 12 SAME_TEAM_SWAPs to the pass that introduced it (first pass where the GT↔pred-ID mapping flips for that track pair).
- Write `reports/merge_veto/per_pass_swap_attribution.md` — table of `{rally_id, gt_id, swap_frame, introduced_by_pass}`.
- **Decision branch**:
  - If all 12 swaps attributed to passes already covered (just `link_tracklets_by_appearance`) → Session 8 rolls directly into Plan B. Session 6's work was complete at its site; we need a stronger signal, not more sites.
  - If 2+ passes create swaps → proceed with multi-site integration.

### 2. Unified veto helper (factored)

Single implementation of the cosine veto, called from all adapter sites:

```python
def _learned_cosine_veto(
    store: LearnedEmbeddingStore,
    id_a: int,
    frames_a: Sequence[int],
    id_b: int,
    frames_b: Sequence[int],
    threshold: float,
    min_embeddings: int = 5,
) -> bool:
    """Return True iff the merge/rename should be BLOCKED.

    Abstains (returns False, allowing the merge) if either side has
    fewer than `min_embeddings` valid learned embeddings — the regime
    where the head's signal is unreliable.
    """
    med_a = _segment_median_embedding(store, id_a, frames_a, min_embeddings)
    med_b = _segment_median_embedding(store, id_b, frames_b, min_embeddings)
    if med_a is None or med_b is None:
        return False  # abstain
    cos = float(np.dot(med_a, med_b))
    return cos < threshold
```

Location: `rallycut/tracking/merge_veto.py` (new module — avoids circular imports with `tracklet_link.py`, `global_identity.py`, etc.). Reuses `_segment_median_embedding` helper from Session 6 (move it here, update import in `tracklet_link.py`).

### 3. Per-pass adapters

Each pass has a different decision structure. The veto call site differs per pass, but all call the same `_learned_cosine_veto`. Work depends on the diagnostic output; only passes flagged as swap-creators need adapters.

| Pass | File | Integration point | Adapter work |
|---|---|---|---|
| `link_tracklets_by_appearance` | `tracklet_link.py` | already done (Session 6) | none |
| `fix_height_swaps` | `global_identity.py` or `height_swap.py` | pre-swap decision (before rename table applies) | veto per proposed swap pair |
| `relink_spatial_splits` | `global_identity.py` (subfn) | per-pair cost matrix (similar to `tracklet_link`) | cost = 1.0 on veto |
| `relink_primary_fragments` | `global_identity.py` (subfn) | per-pair decision | similar to above |
| `stabilize_track_ids` | `global_identity.py` | pre-rename table check | veto per proposed canonical rename |
| `enforce_spatial_consistency` | — | splits tracks, not merges | likely no adapter |
| `split_tracks_by_color` | — | splits tracks, not merges | no adapter |

All adapters share the same veto signature and threshold env var.

### 4. Env var gating (preserve byte-identical defaults)

- **`LEARNED_MERGE_VETO_COS`** — already exists (Session 6). Reused as the **single** threshold across all adapter sites. `0.0` (default) disables veto everywhere.
- Config hash annotation: `merge_veto:enabled` (not threshold value — so threshold sweeps share the raw cache, fix proven in Session 6). No hash change when disabled.

### 5. Threshold sweep + gate

**Sweep harness**: `scripts/eval_multisite_merge_veto.py` (modeled on Session-6's `eval_learned_merge_veto.py`).

- 8 cells: `LEARNED_MERGE_VETO_COS ∈ {0.0, 0.40, 0.50, 0.55, 0.60, 0.65, 0.70, 0.80}`.
- Each cell: `evaluate-tracking --all --retrack --cached --audit-out ...`.
- Per cell: parse aggregate HOTA, per-rally HOTA, SAME_TEAM_SWAP count, unique pred-IDs per rally.
- Output: `reports/merge_veto/session8_gate_report.md` — per-cell table, knee recommendation, per-rally HOTA + swap deltas.
- **Knee pick**: lowest threshold that clears all four gate criteria.

## Plan B (if Session 8 fails)

**Single-site multi-signal at `link_tracklets_by_appearance`**. Stack learned ReID + court-plane velocity (already infra-ready, dormant since 2026-04-16). AND-logic: require BOTH signals to vote "different players" before the veto fires. Rationale:

- Independent failure modes — ReID weak on visually-identical teammates but strong on distinct poses; velocity strong on players who teleport (bad merge signature) but weak on adjacent convergent trajectories.
- Prior Session 3b probe showed velocity alone was 3 wins / 5-7 losses as lone veto. ANDing with ReID recovers legit merges the velocity gate rejected.
- Reuses existing `_would_create_velocity_anomaly` (tracklet_link.py:125-250) via `calibrator=court_calibrator`. No new primitives.

Plan B is NOT part of Session 8's scope — it's a contingency for if/when Session 8 ship gate fails.

## Kill Conditions

- **Diagnostic attribution test**: if all 12 swaps already attributed to `link_tracklets_by_appearance` → Session 8 rolls to Plan B (skip ahead, no wasted adapter work).
- **Sweep ceiling**: if best-case reduction across all 8 cells < 40% (< 5 swaps reduced) → veto mechanism is fundamentally underpowered regardless of coverage. Close within-team work per Session 7's "accept ceiling" framing.

## Budget

~10 h across one session:

| Task | Estimate |
|---|---|
| Diagnostic per-pass audit tool + run on 43 rallies | 1.5 h |
| Veto helper (factored) + unit tests | 1 h |
| Adapter: `fix_height_swaps` | 1 h |
| Adapters: `relink_spatial_splits` + `relink_primary_fragments` | 1.5 h |
| Adapter: `stabilize_track_ids` (only if diagnostic flags it) | 0.5 h |
| Regression parity smoke (default = byte-identical) | 0.5 h |
| 8-cell threshold sweep + production_eval | 2 h |
| Report + memory + commit | 1.5 h |
| Buffer | 0.5 h |

## Risks

| Risk | Mitigation |
|---|---|
| Diagnostic shows swaps concentrated in passes already covered | Narrows scope; redirect to Plan B. Good outcome. |
| Veto fires too aggressively → fragmentation > 20% | Gate #3 blocks ship. Per-pass threshold tuning as escape hatch. |
| Passes have interdependencies (vetoing pass N breaks pass N+1) | Gate #2 catches it; unit test per adapter with synthetic tracks. |
| Head weak at rename-time single-frame resolution | Multi-crop median over the ±10 frames adjacent to the decision boundary (track A's pre-boundary frames, track B's post-boundary frames, as in Session 6). Abstain if either side has < 5 valid embeddings in its window. |
| Action-to-player GT lost on retrack | Gate #4 compares against cached oracle pre-Session-8. If ship conditions met, re-attribution is targeted on NEW videos only; existing GT preserved. |

## Files to Touch

### New
- `analysis/rallycut/tracking/merge_veto.py` — factored veto helper + `_segment_median_embedding`.
- `analysis/scripts/eval_multisite_merge_veto.py` — 8-cell threshold sweep harness.
- `analysis/scripts/diagnose_per_pass_swaps.py` — diagnostic, attributes swaps to passes.

### Modified
- `analysis/rallycut/tracking/tracklet_link.py` — move `_segment_median_embedding` import to `merge_veto`; Session-6 veto call uses factored helper.
- `analysis/rallycut/tracking/global_identity.py` — adapters at `fix_height_swaps`, `relink_spatial_splits`, `relink_primary_fragments`, `stabilize_track_ids` (subject to diagnostic).
- `analysis/rallycut/tracking/player_tracker.py` — instrumentation hooks for diagnostic (optional, gated).
- `analysis/tests/unit/test_merge_veto.py` — new tests for factored helper.
- `analysis/tests/unit/test_global_identity.py` — per-adapter tests.

### Reused (no edits)
- `rallycut/tracking/color_repair.py::LearnedEmbeddingStore` — embedding store (Session 4).
- `rallycut/tracking/reid_embeddings.py` — head loader (Session 4).
- `rallycut/evaluation/tracking/retrack_cache.py` — cache schema (Session 4).

## Deliverables

- Multi-site merge veto behind existing `LEARNED_MERGE_VETO_COS` env var. Default 0.0 → byte-identical to post-Session-7 baseline.
- `reports/merge_veto/per_pass_swap_attribution.md` — diagnostic output (regardless of ship/no-ship).
- `reports/merge_veto/session8_gate_report.md` — per-cell table, knee recommendation.
- Memory update: SHIP verdict + threshold, or NO-SHIP verdict + pivot to Plan B / accept ceiling.

## What NOT to Change

- Session 4's additive cost at `_compute_assignment_cost` — stays dormant (`WEIGHT_LEARNED_REID=0`).
- Session 5's occlusion-resolver — stays dormant (`ENABLE_OCCLUSION_RESOLVER=0`).
- Session 7's per-pass skip flags — stay wired; useful for diagnostic.
- `optimize_global_identity` coverage-revert guard (line 261-273) — unchanged; we sit upstream of it.
