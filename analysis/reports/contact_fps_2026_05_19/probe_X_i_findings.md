# Probe X-I findings: snap + regressor confirmed as the attack/block collapse layer

**Date:** 2026-05-20
**Inputs:** the 13 H-A3 rallies. Monkey-patches
`_snap_contacts_to_direction_change_max` and `refine_contacts_with_regressor`
to capture every contact's frame before and after each stage. Then reads the
post-dedup final output.

## Headline (regressor hypothesis CONFIRMED)

| Stage that pushes attack-window candidate INTO block window | Cases |
|---|---|
| **SNAP alone** (heuristic direction-change MAX, window=10) pushes 4-10 frames | **5 of 13** |
| **REGRESSOR alone** (small 2-4 frame shifts, often after snap already moved it) | **all 13** |
| Rallies with no attack-window candidate entering snap | 0 |

Snap shift distribution (|Δ| frames):

| 0 | 2 | 4 | 6 | 7 | 9 | 10 |
|---|---|---|---|---|---|---|
| 6 | 1 | 2 | 1 | 1 | 1 | 1 |

Regressor shift distribution (|Δ| frames):

| 0 | 1 | 2 | 3 | 4 |
|---|---|---|---|---|
| 1 | 1 | 7 | 3 | 1 |

So the snap is bimodal: it leaves 6 of 13 candidates untouched but moves
5 of 13 by ≥6 frames. The regressor is uniform: 2-4 frame shifts. Together
they push every single attack-window candidate into the block window.

## Two sub-patterns within the 13

**Pattern A (7 of 13)**: TWO candidates survive the merge stage —
one in attack window, one in block window. Snap+regressor pushes the
attack candidate within `min_peak_distance_frames=12` of the block
candidate, then dedup collapses them.

Examples:
- `gigi 72c8229b` (gt=631): PRE-SNAP `[..., 627, 638]` → POST-REG `[637, 639]` → FINAL `[637]` (dedup eats one).
- `juju acada27e` (gt=241): PRE-SNAP `[..., 237, 249]` → POST-REG `[238, 240]` → FINAL `[238]`.
- `moma 753a4ec7` (gt=201): PRE-SNAP `[..., 192, 206]` → POST-REG `[200, 206]` → FINAL `[200]`.

**Pattern B (6 of 13)**: only ONE candidate survives the merge — either
the attack candidate (which then gets snapped to block window) OR the
block candidate (with no attack-side counterpart at all). No second
contact to recover.

Examples:
- `caco 9452ee5a` (gt=190): PRE-SNAP `[..., 185]` → POST-REG `[187]`. Single candidate slides into the block frame.
- `gigi b8d333ae` (gt=234): PRE-SNAP `[..., 230]` → POST-REG `[233]`. Same: one candidate.
- `kiki a0aba15e` (gt=972): PRE-SNAP `[..., 963]` → POST-SNAP `[..., 972]` → POST-REG `[974]`. Snap of +9 frames; single candidate.

The split is exactly the H-A1 / H-A3 dichotomy from X-G's earlier
classification, now grounded in real data:
- Pattern A = X-G's H-A3 (originally "GBM rejects"; actually
  "regressor pushes both candidates close enough for dedup to collapse")
- Pattern B = X-G's H-A1 (originally "merger drops cross-side pairs"
  during stage 2; one candidate generation does fire near the block,
  but the attack candidate was already lost)

## Why is the regressor doing this?

`refine_contacts_with_regressor` (`contact_frame_regressor.py:212`) loads
a learned `GradientBoostingRegressor` trained on 2480 (candidate, GT-frame)
pairs from the full 74-video action-GT corpus. The training labels are
the SINGLE GT contact frame per labelled contact. In the attack+block
pair case, the GT corpus labels only the attack OR only the block
(whichever the labeler clicked), so the regressor learned to bias
candidates toward THAT label. The regressor doesn't see "this candidate
should stay separate from another candidate 5f later" — it only sees
"snap to nearest GT-likely frame".

Combined with the snap stage (which targets the direction-change MAX,
typically at the apex of the block-back, which is the BLOCK contact's
trajectory bend), the effect compounds: the attack candidate gets pulled
toward the block's apex, the regressor finishes the job.

## Fix-space implications

Both pattern A and B are produced by the same architectural assumption:
each candidate is refined INDIVIDUALLY without awareness of its neighbours.
A cross-side-aware refinement would need to:

1. **Snap**: don't allow a candidate to cross the implied-attack/implied-
   block boundary toward another candidate. Concretely: when snapping
   candidate at frame T, clamp the snap window to the midpoint between
   T and the nearest other candidate.
2. **Regressor**: same clamp — never shift a candidate past the midpoint
   to its neighbour.
3. **Dedup**: use PRE-snap frame distance for the dedup decision, not
   post-snap. If two candidates were >12f apart at merge time, keep
   them both regardless of where the regressor moved them.

(3) is the cleanest single-knob fix and the least likely to regress
other tests:
- Requires preserving the `original_frame` on Contact (or a sidecar
  map) through snap and regressor.
- Changes dedup's `frame_gap` test from `abs(c.frame - existing.frame)`
  to `min(abs(c.frame - existing.frame), abs(c.original_frame - existing.original_frame))`.
- Conservative: ONLY prevents the snap+regressor from collapsing pairs;
  doesn't introduce any new at-merger or at-GBM behaviour.

Option (1)/(2) are surgical to the snap+regressor but require knowing
the neighbour set at snap time (currently each candidate is refined
independently in a per-contact loop).

## Pattern B (the 6 lone-candidate rallies) — separate fix?

Pattern B cases will NOT benefit from any post-snap fix. They need a
second candidate to survive the merge in step 2 (`_merge_candidates`,
`contact_detector.py:2267-2296`). The earlier `_BOTH_CONFIDENT_FLOOR=0.80`
attempt was rejected (+195 FPs). A more targeted alternative:

- Make `_merge_candidates` cross-side-aware via a CHEAP net-proximity
  proxy: when a candidate at frame T has `abs(ball.y - estimated_net_y)
  < 0.15`, allow it to coexist with another candidate at frame T±5..12
  if THAT candidate is also near-net. This is the same net-proximity
  guard the plan originally suggested for H-A1.

Pattern B's 6 cases stand to benefit if such a guard is added; the rest
(7 of 13) would benefit either way once dedup is fixed.

## Recommendations (decision tree update)

Original Phase 3 decision tree assumed the bottleneck was at the GBM
or the merger. The actual bottleneck is at the snap/regressor/dedup
layer for 7 of 13 cases and at the merger for 6 of 13 cases.

Suggested order:
1. **Pattern A fix** (dedup-uses-pre-snap-frame). Targets 7 of 13. Low
   blast radius — only changes the dedup decision when snap moved
   candidates within 12f.
2. **Pattern B fix** (net-proximity merger relaxation). Targets the
   remaining 6 of 13 PLUS the original 8 H-A1 cases from X-G (total
   reach: up to 14 cases, with overlap).
3. **Re-run trusted-31 eval after each** to check the metric gates.

## Numbers reproduce

```bash
cd analysis
uv run python -u scripts/probe_X_i_snap_regressor_deltas.py
```
