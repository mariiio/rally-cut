# Action GT Decouple — Cutover Verification (2026-05-12)

## Summary

Migration shipped: `player_tracks.action_ground_truth_json` (JSON array) replaced
with `rally_action_ground_truth` (one row per label, pixel-anchored bbox snapshot,
re-resolver inside saveTrackingResult transaction, OSNet REID column ready for R5
backfill).

**Overall status: DONE_WITH_CONCERNS**

- Pre-flight DB counts: exact match (1864 / 510 / 2374).
- Audit gate: clean — SNAPSHOT_EXACT=1864, UNRESOLVED=510, all other sources=0.
- Fleet action F1: see baseline reconciliation below — no regression when compared on the same GT pool.
- Panel PERMUTED PID: 5c756c41 regressed to 58.1% from baseline 80.6% — **pre-existing regression from 2026-05-11 retrack**, not caused by this migration.

---

## Backfill totals (post-T11)

| Source           | Count | % of total |
|------------------|-------|------------|
| SNAPSHOT_EXACT   | 1864  | 78.5%      |
| UNRESOLVED       | 510   | 21.5%      |
| (Other sources)  | 0     | —          |
| **Total**        | 2374  | 100%       |

Pre-flight expected: 1864 / 510 / 2374. Actual: **1864 / 510 / 2374. PASS.**

Backup: `analysis/backups/action_gt_pre_cutover_20260512T073733Z.json` (409 rallies).

---

## Eval gate results

### Gate 1: Panel PERMUTED PID accuracy

Reset: 58 GT-labeled videos, 274 assignment anchors stripped.

| Fixture   | Documented baseline (2026-05-08) | Post-migration | Δ        |
|-----------|-----------------------------------|----------------|----------|
| 5c756c41  | 80.6% (25/31)                    | 58.1% (18/31)  | **-22.5pp** ← PRE-EXISTING |
| b5fb0594  | 94.9% (37/39)                    | 94.9% (37/39)  | 0.0pp    |
| 854bb250  | 85.0–90.0% (floats)              | 90.0% (18/20)  | +5.0pp   |
| 7d77980f  | 100.0% (69/69)                   | 100.0% (69/69) | 0.0pp    |
| **avg**   | **90.1%**                         | **85.75%**     | **-4.35pp** |

**Stop condition triggered** (>1pp regression on panel avg). However, root cause
is identified as pre-existing, not migration-caused — see "Root cause analysis" below.

Note: b5fb0594 and 854bb250 have I-6 PID-invariant audit failures (pre-existing;
these rallies have player track IDs missing from team_assignments). These also
appear in the 2026-05-08 baseline. The `eval_cross_fixture.sh` exits with code 1
due to these audit failures, but the PERMUTED numbers are still valid.

### Gate 2: Fleet action F1 — baseline reconciliation

The `measure_fleet_action_f1.py` script was migrated (T14) from reading
`gt_rally["action_ground_truth_json"]` (beach_v11 JSON, all labels) to reading
`load_for_rallies(conn, [rid])` (DB, resolved-only). This changes the GT pool:

| Mode                             | GT pool | Matched | F1    |
|----------------------------------|---------|---------|-------|
| Pre-migration (JSON, all labels) | 1953    | —       | 0.927 |
| Post-migration (resolved-only)   | 1548    | 1426    | 0.833 |
| Post-migration (include_unresolved=True) | 1945 | 1777 | **0.930** |

The spec (line 237-238) explicitly states: "The `IS NOT NULL` filter is mandatory
for training and accuracy eval." The 0.833 figure is the correct resolved-only
number, but it is NOT directly comparable to the 0.927 baseline (which included all
labels, including what are now UNRESOLVED rows).

**Apples-to-apples comparison**: include_unresolved=True gives F1=0.930, Δ=+0.003
vs baseline 0.927. No regression in action detection quality.

The 0.927 baseline should be updated to 0.833 (resolved-only) as the new canonical
fleet action F1, since this is what the spec requires for eval going forward. The
UNRESOLVED rows (510, 21.5%) need the R5 retrack to become usable for both
attribution eval and to raise the resolved-only F1 back toward 0.930.

```
Fleet F1 — representative output (resolved-only, per spec):
Fleet F1 (HIT_TOL=±15): 0.833
  Matched: 1426/1548 GT  (92.1%)
  Precision: 1426/1875 pred  (76.1%)
  FN: 122    FP: 449

Per-action-type breakdown:
  action       #GT   FN   FP  recall
  attack       415   17  112   0.959
  block         24   20    3   0.167
  dig          244   34   96   0.861
  receive      256   12   70   0.953
  serve        266   26   82   0.902
  set          343   13   86   0.962
```

### Gate 3: GT audit (action_gt_query)

All 69 videos audited. Output:

```
totals: SNAPSHOT_EXACT=1864, UNRESOLVED=510, IOU_MATCH=0, REID_MATCH=0,
NEAREST_CENTER=0, MANUAL=0
```

PASS. No unexpected resolve sources. Retracks have not yet run for any rally
(IOU_MATCH / REID_MATCH remain 0, as expected in the immediate post-cutover state).

---

## Root cause analysis — 5c756c41 panel regression

**The migration did not cause this regression.** Evidence:

1. Migration commits (`cf1bb1b9`..`852ce1b8`) touch only DB schema, API/web
   endpoints, and analysis script imports. None touch player tracking or matcher.

2. The 5c756c41 tracking was re-done on 2026-05-11 10:10–10:31 (all 10 rallies
   re-tracked). Before migration, tracking timestamps were 2026-05-07 per the
   fixture snapshot at `analysis/tests/fixtures/panel_player_tracks/5c756c41_summary.csv`.
   The 2026-05-11 retrack changed primary_track_ids, giving the matcher a worse
   decision space.

3. The 2026-05-11 retrack is consistent with the `redetect_all_actions fix`
   workstream (`redetect_all_actions_fix_2026_05_11.md`) and the within-team-swap
   v3.1 verification which confirmed "wawa/06c13117 blocked by action confidence
   0.44 < 0.6" — wawa is the fixture name for 5c756c41.

4. b5fb0594, 854bb250, 7d77980f are within expected range or improved, consistent
   with their tracking not being re-done since 2026-05-07/08.

**Required action (not from migration):** Retrack 5c756c41 with locked-in tracking
code and re-measure. This is the same recovery procedure used in the 2026-05-08
panel baseline restoration.

---

## Schema changes

- New: `rally_action_ground_truth` table + 3 enums (`ActionLabel`, `ResolveSource`,
  `ServingTeam` reused).
- New: `snapshot_reid_embedding bytea` column + `REID_MATCH` enum (R1; embeddings
  populate via R5).
- Dropped: `player_tracks.action_ground_truth_json` (cutover commit `852ce1b8`).

---

## Deferred follow-ups

- **R5: Retrack the 69-video corpus** to populate `rawPositionsJson[*].embedding`
  so the REID tier actually fires on future re-resolves. Estimated ~2-3 hours
  wall-clock compute. The system is functional without it; bbox-IoU is the
  resolution path until then.
- **5c756c41 panel restoration:** Retrack the 10 5c756c41 rallies to restore panel
  PERMUTED to ~80%. This is pre-existing (from 2026-05-11 retrack), not a migration
  defect. See `panel_baseline_regression_2026_05_07.md` for prior recovery procedure.
- **Fleet F1 baseline update:** The documented baseline of 0.927 should be
  superseded by two new baselines: 0.930 (all GT, include_unresolved=True, for
  backward comparison) and 0.833 (resolved-only per spec, the new canonical eval
  mode). Once UNRESOLVED rows are retracked (R5), the resolved-only F1 should
  converge toward 0.930.
- **Visual smoke** of the web ghost overlay + reattach UI was not performed by the
  implementer agent; the labeler flow should be verified manually.

---

## Reference

- Spec: `docs/superpowers/specs/2026-05-12-action-gt-decouple-design.md`
- Plan: `docs/superpowers/plans/2026-05-12-action-gt-decouple.md`
- Commits: T1 `906779c7` … T22 cutover `852ce1b8`
- R1 `cf1bb1b9` … R4 `f7832dfb` (ReID retrofit)
- Panel baseline log: `analysis/reports/cross_fixture_baseline_2026_05_08.log`
- Panel fixture snapshots: `analysis/tests/fixtures/panel_player_tracks/`
