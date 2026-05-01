# W4 (`ENABLE_BBOX_SWAP_DETECTION=1`) on the 13-rally panel — NO-GO 2026-05-01

**Verdict**: do not enable W4 by default. Reproduces the exact UNLABELED
regression that prompted commit `7307c1d`'s revert; does not fix any of
the 3 within-rally drift rallies on the panel.

## Setup

- 4 fixtures × 2 passes (W4 OFF, W4 ON), `--no-ref-crops`,
  `ENABLE_POSITION_JUMP_SWAP=0`, `--reset-anchors` between passes for
  clean fresh solves.
- Final verdict via `panel_verdict_per_frame.py` (reads positions_json
  directly, the artifact the editor renders).
- Both passes converged to the same AGREES count.

## Result

| AGREES | Pass |
|---|---|
| 12/13 | W4 OFF |
| 12/13 | W4 ON |

But per-rally shapes differ — the AGREES count is preserved only because
the verdict tool's "BAD" bucket includes both `slow_drift` and
`unlabeled`, and the regression rally was already BAD.

| Rally | W4 OFF shape | W4 ON shape | Δ |
|---|---|---|---|
| 7d77980f/r02 (PNL-BAD) | within_rally_swap (1 event) | within_rally_swap (1 event) | unchanged |
| **b5fb0594/r10 (PNL-BAD)** | slow_drift (PID3 half-shift=0.21) | **unlabeled (1137 frames, 37.5% of positions)** | **REGRESSION** |
| 5c756c41/r07 (PNL-BAD) | slow_drift (PID4 half-shift=0.58) | slow_drift (PID4 half-shift=0.58) | unchanged |
| b5fb0594/r01 (CTRL-BAD) | hungarian_drop | hungarian_drop | shape-only minor |

The 1137 UNLABELED frames on b5fb0594/r10 matches almost exactly the
"1137 (37.4%)" regression documented in
`regression_2026_05_01_7307c1d_revert.md`. Same root cause: W4 emits
sub-tracks → parent removed from `top_tracks` (line 860 of
match_tracker.py, retained post-revert) → frames outside the sub-track
range go UNLABELED. The "Bucket 1" court-side propagation fix that
allowed sub-tracks to survive Hungarian without UNLABELED was reverted
in `ce7b08c` and not reinstated.

## Conclusion

W4 in its current (post-revert) form should remain default-OFF for the
blind regime. Re-enabling it would require either:

(a) Cherry-picking Bucket 1 from `7307c1d` (which is the path the
session that produced the original revert ALREADY tried — and which
caused the same regression by a different mechanism, per
`regression_2026_05_01_7307c1d_revert.md`), or

(b) Redesigning W4's parent-removal behavior so parents remain in
`top_tracks` and sub-tracks act as per-frame overrides only, not
replacement entries. This is a real engineering option but not in
scope here.

W4 also does not fix any of the 3 within-rally drift rallies — the two
`slow_drift` cases (b5fb0594/r10, 5c756c41/r07) don't have the
single-frame mirror-swap event that W4 detects, and 7d77980f/r02 has
exactly one swap event but W4's emitted sub-tracks don't reach a
clean enough split to fix the visible identity flip.

## Next step

Approach (2) from the within-rally drift plan: build a blind-regime
within-track segmenter targeting `slow_drift` specifically. Use the
same signal the verdict tool's `slow_drift` detector uses (per-PID
half-and-half centroid shift + x-range overlap), bisect at the drift
midpoint, and re-Hungarian each half independently. Targets b5fb0594/r10
and 5c756c41/r07 directly without touching W4.

## Pointers

- Orchestrator: `analysis/scripts/ab_test_w4.sh`
- Verdict snapshots: `analysis/reports/w4_ab_2026_05_01/verdict_w4{off,on}.txt`
- Per-fixture logs: `analysis/reports/w4_ab_2026_05_01/<short>_w4{off,on}_*.log`
- Original regression memo: `regression_2026_05_01_7307c1d_revert.md`
- Locked panel state: `panel_visual_verdict_2026_05_01.md`
