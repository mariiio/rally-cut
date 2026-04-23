# Phase 0 — Attribution Primitive-First Rebuild: Summary

**Date:** 2026-04-24  
**Plan:** `docs/superpowers/plans/2026-04-24-attribution-primitive-first.md`  
**Scope:** 9 fixtures / 69 GT rallies / 468 GT actions — tata, toto, lala, lulu, wawa, cece, cuco, rere, yeye.

## Baseline (locked)

`reports/attribution_rebuild/baseline_2026_04_24.json` — source-of-truth artifact for all A/B going forward.

- Full DB snapshot of `player_tracks.{action_ground_truth_json, actions_json, contacts_json, primary_track_ids}` after stage-2 → remap → stage-3 refresh.
- Per-rally scored with `rallycut.evaluation.attribution_bench` (±10 frame GT match tolerance).

### Aggregate

| metric | count | rate |
|---|---|---|
| correct | 313 | **66.9%** |
| wrong | 83 | **17.7%** — cross_team 48 / same_team 33 / unknown_team 2 |
| missing | 72 | 15.4% |
| abstained | 0 | 0.0% (chooser doesn't abstain at baseline) |

### Per-fixture

| fixture | n_gt | correct | wrong | missing | [X / same / unk] |
|---|---|---|---|---|---|
| tata | 108 | 68.5% | 19.4% | 12.0% | 12 / 9 / 0 |
| toto | 59 | 64.4% | 11.9% | 23.7% | 2 / 5 / 0 |
| lala | 71 | 63.4% | 22.5% | 14.1% | 11 / 4 / 1 |
| lulu | 37 | 54.1% | 24.3% | 21.6% | 5 / 4 / 0 |
| wawa | 33 | 51.5% | 15.2% | 33.3% | 5 / 0 / 0 |
| cece | 23 | 73.9% | 8.7% | 17.4% | 2 / 0 / 0 |
| cuco | 40 | 67.5% | 25.0% | 7.5% | 0 / 9 / 1 |
| rere | 35 | 71.4% | 20.0% | 8.6% | 6 / 1 / 0 |
| yeye | 62 | **80.6%** | 9.7% | 9.7% | 5 / 1 / 0 |

## Deliverables

- `reports/attribution_rebuild/fixture_video_ids_2026_04_24.json` — fixture → DB video_id mapping.
- `reports/attribution_rebuild/baseline_2026_04_24.json` — locked baseline snapshot (all 468 GT actions scored).
- `scripts/bench_attribution.py` — standard runner for all future A/B (computes per-fixture + aggregate, transition matrix vs baseline, optional summary JSON).
- `rallycut/evaluation/attribution_bench.py` — matching / classification primitives.
- `scripts/phase0_lock_baseline.py` — re-lockable baseline freezer.
- `scripts/render_attribution_viewer.py` — v0 visual viewer (HTML, 69 rally pages at `reports/attribution_audit/`).
- `reports/dormant_flag_decisions_2026_04_24.md` — 8 env-gated NO-GO flags marked REMOVE (global seed, learned ReID, occlusion resolver, merge veto, skip merge passes, court velocity gate ×3).
- `reports/action_gt_freshness_2026_04_24.md` — 10.6% action-type disagreement rate → Phase 3 Pattern A uses high-confidence pipeline action types only.

## Key learnings for Phase 1+

1. **Canonical pid pipeline must include `remap-track-ids` between match-players and reattribute-actions.** Without it, `primary_track_ids` + `teamAssignments` contain raw tracker tids (e.g. 18, 23, 59) while `actions_json.playerTrackId` emits canonical {1,2,3,4} → produces spurious `unknown_team` errors. Evidence: adding remap step moved +4.9pp correct, collapsed `unknown_team` from 53 → 2.
2. **Reference crops are load-bearing for cross-rally pid stability.** Before crops were fixed: lala had 21 distinct pids across 7 rallies (tracker raw IDs leaking). After ref-crop label + remap: 4 stable pids. wawa/cuco added crops → also canonicalized.
3. **Two residual primitive issues worth Phase 1.1 investigation:**
   - **Low-rally-count fixtures with 3-player-detection rallies**: tata/toto each have 1-2 rallies where `primary_track_ids` has only 3 pids (4th player missed entirely). Occlusion/coverage gap.
   - **cuco's same_team rate** (9/10 of its wrongs are within-team confusion) — geometry-only chooser floor, Phase 2 confidence gate is the lever.
4. **Baseline wrong_rate is 17.7%**, not the plan's pre-registered 20.1% — different fixture set + action-GT is a larger measurement universe than click-GT. Phase 2 kill gate (`wrong_rate ≤ 10%`) becomes `≤ 8.85%` at the new baseline's halving rule, or keep the absolute 10% target. Hold until Phase 1 measurements inform the tightness.
5. **Missing rate is 15.4%** — dominated by toto (14), tata (13), wawa (11). Upstream contact-detection recall ceiling, rescue-only via Phase 3.2 complement.

## Phase 1 entry criteria met

- ✅ Baseline locked on disk.
- ✅ Benchmark runner operational.
- ✅ Visual debug surface rendered.
- ✅ Dormant code flagged for removal (pending user sign-off).
- ✅ Action-type GT freshness measured and scoped.
- ✅ All 9 fixtures have stable 4-pid cross-rally identity after remap.
