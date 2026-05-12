# Contact-Detection FN-Reduction — `RELAX_CONTACT_DIR_CHANGE` A/B (2026-05-12)

First A/B test of Plan Task 9 from
`docs/superpowers/plans/2026-05-12-contact-detection-fn-reduction.md`.

Flag tested: `RELAX_CONTACT_DIR_CHANGE=1` — relaxes
`min_direction_change_deg` from strict default to `min_direction_change_deg_relaxed`
(see `rallycut/tracking/contact_detector.py:_resolve_effective_config`).

The probe (`reports/contact_detection_fn/probe_2026_05_12.md`) predicted this
flag as the top winner — 68.8% of FN cases fail the dir-change gate.

## Verdict: **FAIL** (G-A — no recall improvement)

No STOP triggered (recall did not regress, precision did not drop > 5pp,
C-1 did not increase > 10%). Snapshot was restored.

## Method

1. Snapshotted DB `contacts_json` + `actions_json` for 820 rallies with ball positions
   into `/tmp/dir_change_pre_ab_snapshot.jsonl`.
2. Ran `RELAX_CONTACT_DIR_CHANGE=1 uv run python -u scripts/redetect_all_actions.py --apply`
   — 820 rallies reprocessed, 0 errors, 49.3 min wall.
3. Re-measured with the flag still ON via
   `RELAX_CONTACT_DIR_CHANGE=1 uv run python -u scripts/measure_contact_recall_full.py
    --label dir_change_on
    --output reports/contact_detection_fn/measurement_dir_change_on_2026_05_12.json`.
4. Verified env-flag is wired in (`detect_contacts` → `_resolve_effective_config` →
   reads `RELAX_CONTACT_DIR_CHANGE`) and that the redetect did mutate the DB
   (481/820 rallies have different `actions_json`, 230/820 different
   `contacts_json`; total contacts went 3668 → 3670 across the fleet).
5. Restored snapshot, re-ran measurement (no flag) — recall returned to
   baseline 0.8909, C-1=112, confirming a clean rollback.

## Headline numbers vs baseline

| Metric          | Baseline | Flag ON | Delta            | Gate                  | Verdict |
| --------------- | -------- | ------- | ---------------- | --------------------- | ------- |
| Recall          | 0.8909   | 0.8909  | 0.0000           | G-A: ≥ +3pp absolute  | **FAIL** |
| Precision proxy | 0.9346   | 0.9346  | 0.0000           | G-B: ≤ -3pp absolute  | PASS    |
| C-1 violations  | 112      | 104     | -7.1% relative   | G-C: ≤ +5% relative   | PASS (improved) |
| C-2 violations  | 333      | 302     | -9.3% relative   | (no gate)             | PASS (improved) |
| C-3 violations  | 2        | 2       | 0                | (no gate)             | PASS    |

## Per-type recall vs baseline (G-E)

| Action  | Baseline | Flag ON | Delta  | Gate                 | Verdict |
| ------- | -------- | ------- | ------ | -------------------- | ------- |
| attack  | 0.9342   | 0.9342  | 0.0000 | ≤ -2pp absolute      | PASS    |
| block   | 0.1081   | 0.1081  | 0.0000 | ≤ -2pp absolute      | PASS    |
| dig     | 0.8355   | 0.8355  | 0.0000 | ≤ -2pp absolute      | PASS    |
| receive | 0.9194   | 0.9194  | 0.0000 | ≤ -2pp absolute      | PASS    |
| serve   | 0.8627   | 0.8627  | 0.0000 | ≤ -2pp absolute      | PASS    |
| set     | 0.9336   | 0.9336  | 0.0000 | ≤ -2pp absolute      | PASS    |

Every per-type recall fraction is byte-identical to baseline. The match
counts (`n_recalled / n_gt`) are bit-for-bit unchanged.

## Unit tests (G-D)

`RELAX_CONTACT_DIR_CHANGE=1 uv run pytest tests/unit/test_contact_detector_relaxations.py
 tests/unit -k contact_detector -v` → **107 passed, 0 failed**. PASS.

## Per-gate summary

| Gate                                     | Verdict     |
| ---------------------------------------- | ----------- |
| G-A: Recall ≥ +3pp absolute              | **FAIL**    |
| G-B: Precision drops ≤ 3pp absolute      | PASS        |
| G-C: C-1 violations ≤ +5% relative       | PASS (improved -7.1%) |
| G-D: Unit tests pass                     | PASS        |
| G-E: No per-type recall drops > 2pp abs  | PASS        |

## STOP-condition check

| STOP condition                            | Triggered? |
| ----------------------------------------- | ---------- |
| Action recall regresses below baseline    | No (0.8909 == 0.8909) |
| Action precision drops > 5pp absolute     | No (0.9346 == 0.9346) |
| C-1 violations increase > 10% relative    | No (-7.1% relative) |

Workstream continues — no escalation needed.

## Surprises / interpretation

The flag took effect (DB mutated 481/820 rallies, fleet-wide contact total
shifted +2) but the GT-labeled subset (409 rallies, 2374 GT actions) saw
**zero change** in recall, precision, OR per-type counts. Every action-type
fraction is bit-identical to baseline.

The probe predicted 68.8% of FN cases fail the dir-change gate, but lowering
the threshold did not promote any of those FN candidates into TP-matched
actions on the GT subset. Possible reasons:

1. The dir-change gate is one of several consecutive filters. Candidates that
   were pruned by dir-change at the strict threshold are still being pruned
   by a later filter (player-radius / inflection / velocity / GBM acceptance).
2. The relaxed threshold (`min_direction_change_deg_relaxed = 12.0` vs the
   default 15.0) may be too small a step to surface candidates that were just
   barely below the strict gate.
3. The newly-promoted contacts may be landing on frames outside the ±10
   match-tolerance window vs GT, so they neither become TPs nor displace
   existing TPs — but they DO shift the action lists in non-GT rallies
   (consistent with the 481/820 fleet diff).

Coherence improvements (C-1 -7.1%, C-2 -9.3%) suggest the new contacts ARE
making the action sequences more game-rule-coherent without affecting the
GT-subset's recall/precision.

**Recommendation for the plan**: This flag does not move the recall needle
and so does not pass G-A. The next A/B should test a DIFFERENT flag
(e.g. `RELAX_CONTACT_VELOCITY` or `RELAX_CONTACT_INFLECTION`) per the plan,
or consider tightening the relaxed `min_direction_change_deg_relaxed` constant
further if the assumption is that the gate has more headroom — but the
evidence here is that this single-flag relaxation alone is inert against GT.

## Snapshot rollback confirmation

After applying snapshot restore via `UPDATE player_tracks SET contacts_json,
actions_json WHERE rally_id = ?` (820 rallies), re-measured with no flag set:

```
recall:           0.8909  (2115/2374)
precision proxy:  0.9346  (2115/2263)
coherence:        C1=112  C2=333  C3=2
```

Matches the committed baseline (`measurement_baseline_2026_05_12.md`) exactly.
DB is back to pre-A/B state.

## References

- Spec: `docs/superpowers/specs/2026-05-12-contact-detection-fn-reduction-design.md`
- Plan: `docs/superpowers/plans/2026-05-12-contact-detection-fn-reduction.md` (Task 9)
- Baseline: `reports/contact_detection_fn/measurement_baseline_2026_05_12.md`
- Probe: `reports/contact_detection_fn/probe_2026_05_12.md`
- New JSON (gitignored): `reports/contact_detection_fn/measurement_dir_change_on_2026_05_12.json`
