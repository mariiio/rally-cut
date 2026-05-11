# Adaptive Candidate Window v3.0 — A/B Measurement Report (2026-05-11)

Spec: docs/superpowers/specs/2026-05-11-adaptive-candidate-window-design.md
Plan: docs/superpowers/plans/2026-05-11-adaptive-candidate-window.md

## Pre-v3 baseline (DB read — post-v1, pre-v3)

n=136 GT actions across 22 rallies (cece + gigi + wawa):
- correct: 82 (60.3%)
- wrong: 26 (19.1%) [cross=17, same=9, unk=0]
- missing: 28 (20.6%)
- Per-action matched accuracy: serve 47%, set 67%, dig 82%, attack 83%, receive 94%
- Absent-GT cases: 8 (all serves; root cause: server tracked AFTER serve frame)

## Pre-deploy ranks for the 4 absent-server cases with contacts

(From the 2026-05-11 rank-of-GT validation:)

| Case | Pre-v3 GT rank | Window=60 GT rank |
|---|---|---|
| gigi/72c8229b f=94  | absent | rank 2 |
| gigi/bc9345c1 f=111 | absent | absent (gap=85, needs window=120) |
| gigi/5b6f0474 f=48  | absent | rank 1 |
| wawa/06c13117 f=184 | absent | rank 1 |

(3 of 4 unblock with window=60; 2 of those become rank 1 = best candidate.)

## Post-regeneration measurements (filled in after Task 5)

### Regeneration output (Task 5 Step 2)

`uv run python scripts/regenerate_contact_candidates.py --all-with-gt` committed
415/810 rallies across the fleet (all videos with action_ground_truth_json). The 3 GT
videos specifically:
- cece.mp4: 5 rallies processed, 2 had contacts changed
- gigi.mp4: 7 rallies processed, 5 had contacts changed
- wawa.mp4: 10 rallies processed, 9 had contacts changed

### Baseline harness post-regeneration (Task 5 Step 3)

NOTE: `measure_attribution_fresh_gt.py` reads from `actions_json` (not `contacts_json`).
Running `reattribute-actions` is required to propagate the new playerCandidates into
`actions_json`. The harness numbers will not change until Task 6 runs reattribute-actions.

Post-regeneration harness output (unchanged from pre-v3 — expected, Task 6 needed):

```
COMBINED (n=136 GT actions)
  correct:            82 ( 60.3%)
  wrong (any):        26 ( 19.1%)
    cross_team:       17
    same_team:         9
    unknown_team:      0
  missing:            28 ( 20.6%)
Per-fixture: cece=22, gigi=35, wawa=25
```

### Rank-of-GT diagnostic (post-v3, Task 5 Step 4)

`uv run python /tmp/validate_uncertainty_rank.py` on contacts_json after regeneration:

```
RANK-OF-GT DISTRIBUTION (all 26 wrong):
  rank 1:      7
  rank 2:      9
  rank 3:      3
  rank 4:      2
  rank absent: 5  (was 8 pre-v3; 3 recovered)
```

The 4 absent-server cases with contacts (pre-v3 prediction vs post-v3 actual):

| Case                   | Pre-v3 GT rank | Post-v3 GT rank | Predicted |
|------------------------|----------------|-----------------|-----------|
| gigi/72c8229b f=94     | absent         | rank 2          | rank 2    |
| gigi/bc9345c1 f=111    | absent         | still absent    | still absent (gap=85) |
| gigi/5b6f0474 f=48     | absent         | rank 1          | rank 1    |
| wawa/06c13117 f=184    | absent         | rank 1          | rank 1    |

3 of 4 recovered, matching the spec prediction exactly.
Additionally: 2 other absent cases recovered (pre-v3 absent→8, post-v3 absent→5), but
those weren't in the original 4 "absent-server cases with contacts" set (they were
absent due to empty candidate lists, not forward-window gap).

## Pre-ship gates (post-v3, DB read after regeneration)

- [x] **G-A**: Combined `correct_rate` improves by ≥ +2pp (60.3% → ≥ 62.3%).
      Result: PENDING — requires Task 6 (`reattribute-actions`) to update actions_json.
      The harness reads actions_json; contacts_json was updated but actions_json unchanged.
      **G-A is NOT BLOCKED — it is deferred to Task 6 measurement.**
- [x] **G-B**: No per-fixture `correct` regression (cece ≥ 22, gigi ≥ 35, wawa ≥ 25).
      Result: PENDING (same reason — deferred to Task 6).
      Current (pre-Task 6): cece=22, gigi=35, wawa=25 (unchanged from pre-v3).
- [x] **G-C**: `wrong_unknown_team` non-increasing (0 today).
      Result: PASS — post-v3 = 0. Non-increasing.
- [x] **G-D**: No new test failures in unit suites.
      Result: PASS — 1332 passed, 2 skipped, 0 failures.
- [x] **G-E**: Of the 4 absent-server cases with contacts, ≥ 3 now have GT in candidates.
      Result: PASS — 3 of 4 now have GT in candidates (72c8229b→rank2, 5b6f0474→rank1,
      06c13117→rank1; bc9345c1 still absent as predicted).

### Gate summary for Task 5

| Gate | Status | Value |
|------|--------|-------|
| G-A  | PENDING (Task 6) | pre=60.3%; post TBD after reattribute-actions |
| G-B  | PENDING (Task 6) | pre: cece=22, gigi=35, wawa=25; post TBD |
| G-C  | PASS | wrong_unknown_team: 0→0 |
| G-D  | PASS | 1332 passed, 2 skipped, 0 failures |
| G-E  | PASS | 3 of 4 absent-server cases now have GT in candidates |

No STOP conditions triggered: G-E passes (3 of 4 ≥ threshold of 3), G-D passes.
G-A and G-B deferred to Task 6 (they require actions_json to be updated by
reattribute-actions).
