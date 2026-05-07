# Remove Upstream Ref-Crop Matcher Path

**Date:** 2026-05-07
**Status:** Design — pending implementation plan
**Origin memo:** `memory/remove_ref_crop_matching_workstream.md`

## Goal

Remove the upstream "ref-crop" player-matcher path across analysis, API, and web, leaving the blind matcher as the only path. Preserve the recent blind-path fixes (`a790f08` iter-0 orientation, `0e06fd7` late-arriver classification, `5ae71f8` chimera-stitching). Keep DB and S3 ref-crop data dormant; the future post-hoc cluster-pick UX (out of scope for this spec) may repurpose them.

## Motivation

From `memory/remove_ref_crop_matching_workstream.md`:

- **Code-quality:** dual matcher paths (frozen-profile vs blind) cost double on every architectural change. Bit us during the 2026-05-07 late-arriver investigation — `frozen_player_ids` parameter and frozen-vs-blind branching in `_classify_track_sides` and `match_players.py` complicated diagnosis.
- **Adoption-reality:** 421 ref-crop rows across 22 videos, latest 2026-04-27, all clustered around 2026-04-23 ship date — internal testing, not organic adoption. Production users won't label crops upfront (`feedback_blind_regime_goal.md`).
- **Accuracy-reality:** today's blind PERMUTED panel ceiling is ~93.8%; the original +61.4pp ref-crop lift (`player_attribution_day4_2026_04_23.md`) was measured pre-seed-partition / pre-post-switch / pre-chimera / pre-Pattern-E / pre-late-arriver / pre-iter-0. Realistic blind-vs-ref-crop gap on today's matcher state is single-digit pp — not worth the maintenance cost.

## Scope

**In scope (single PR, ~5 commits):**
- Remove all upstream ref-crop code in analysis CLI + tracker
- Remove API endpoints and ref-crop service
- Remove web UI and store/state for ref-crops
- Drop the canonical-map priority branch from the web resolver
- One-shot null-out of `Video.canonicalPidMapJson` rows
- Bump `MATCHER_VERSION` v7 → v8 to invalidate `assignmentAnchor` caches

**Out of scope (future, separate brainstorm):**
- New post-hoc cluster-pick UX (web/api workstream per the memo)
- DB schema decisions about whether to repurpose `player_reference_crops` or design a new schema
- Dropping the dormant DB table / S3 prefix

## Final-state architecture

After cleanup, there is ONE matcher path. No `useRefCrops` flag, no `reference_profiles` parameter, no `frozen_player_ids` attribute, no canonical-map writes. The blind path is THE path.

### Data flow

**Before:**

```
Tracking complete (Modal webhook)
  → matchAnalysisService.runMatchAnalysis(videoId, useRefCrops=false)
  → CLI match-players (blind branch)
  → match_analysis_json

User opens PlayerReferenceCropDialog → uploads crops → "Re-run Matching"
  → matchAnalysisService.runMatchAnalysis(videoId, useRefCrops=true)
  → CLI match-players: _load_db_reference_crops → frozen_profiles → tracker(frozen_player_ids={1,2,3,4})
  → match_analysis_json + canonicalPidMapJson

Web display: canonicalPid.ts: canonicalRallyMap → appliedFullMapping → sortOrder
```

**After:**

```
Tracking complete (Modal webhook)
  → matchAnalysisService.runMatchAnalysis(videoId)
  → CLI match-players (one path, no flags)
  → match_analysis_json

(no second path; no dialog)

Web display: canonicalPid.ts: appliedFullMapping → sortOrder
```

## Component inventory

### Files deleted (5)

- `analysis/rallycut/cli/commands/reference_crops.py`
- `analysis/rallycut/cli/commands/relabel_with_crops.py`
- `analysis/rallycut/tracking/crop_guided_identity.py`
- `api/src/services/referenceCropsService.ts`
- `web/src/components/PlayerReferenceCropDialog.tsx`

### Files modified

| File | Change |
|---|---|
| `analysis/rallycut/cli/main.py` | Drop `reference_crops` and `relabel_with_crops` CLI registrations |
| `analysis/rallycut/cli/commands/match_players.py` | Drop `--reference-crops-json`, `--no-ref-crops`, `--use-existing-profiles` flags; delete `_load_db_reference_crops` (L113-238); delete JSON branch (L454-566); delete canonical-map writing (L834-893); delete `use_existing_profiles` fallback (L589+) |
| `analysis/rallycut/tracking/match_tracker.py` | Drop `reference_profiles` parameter from `MatchPlayerTracker.__init__` and `match_players_across_rallies`; delete `frozen_player_ids` attribute; collapse ~17 `frozen_player_ids` gates; delete `match_players_across_rallies_relabel` helper (L3905-3989); bump `MATCHER_VERSION` v7 → v8 |
| `analysis/rallycut/tracking/team_identity.py`, `match_solver.py` | Inspect for any `frozen_player_ids`-aware logic; collapse if present (currently appears docstring-only) |
| `api/src/routes/videos.ts` | Delete 5 ref-crop endpoints (L577-861) |
| `api/src/services/matchAnalysisService.ts` | Drop `useRefCrops` parameter, conditional crop-loading (L240-354), `canonical_pid_map_json` nulling (L651-730) |
| `web/src/services/api.ts` | Drop 5 ref-crop methods + `useRefCrops` param on `runMatchAnalysis` |
| `web/src/stores/playerTrackingStore.ts` | Drop `referenceCrops` state + 3 methods (L72-76, L945) |
| `web/src/utils/canonicalPid.ts` | Drop canonical-map priority branch; resolver becomes appliedFullMapping → sort fallback |
| `web/src/components/ActionOverlay.tsx`, `ActionLabelingMode.tsx` | Drop canonical-map plumbing |
| `analysis/scripts/reset_matcher_state.py` | Extend to also null `Video.canonicalPidMapJson` |

### Preserved (untouched)

- `analysis/rallycut/tracking/reid_embeddings.py` (DINOv2 backbone)
- `analysis/rallycut/tracking/reid_general.py` (OSNet model)
- `analysis/rallycut/tracking/player_features.py` (HSV / `PlayerAppearanceProfile`)
- `analysis/scripts/measure_pid_accuracy.py`, `eval_cross_fixture.sh` (path-agnostic)
- `analysis/scripts/retrack_single_rally.py` (uses blind path)

### DB data — kept dormant

- `player_reference_crops` table: 421 rows untouched; no Prisma migration; no S3 deletion
- `Video.canonicalPidMapJson` column: untouched; one-shot `UPDATE videos SET canonical_pid_map_json = NULL` so display is uniform across old + new videos
- Both can be repurposed by the future post-hoc cluster-pick UX, or formally dropped in a later schema migration

## Tracker collapse — `frozen_player_ids` reduction

After collapse, `frozen_player_ids` is always-empty, so each gate has a single live branch:

| Pattern | Lines | Collapse |
|---|---|---|
| `not self.frozen_player_ids` (boolean guard) | 917, 984, 1029, 1668, 3183 | Always `True` → drop the guard, keep the gated body |
| `use_side_penalty=not self.frozen_player_ids` | 1010, 1019, 3254, 3266, 3978 | Becomes `use_side_penalty=True` |
| Early-return `if self.frozen_player_ids: return ...` | 1668, 2751, 2912, 4271 | Delete the if-block; only the post-block path remains |
| `if player_id in self.frozen_player_ids` | 2163 | Delete the conditional path; keep the else |
| `perm.get(pid, pid) for pid in self.frozen_player_ids` | 2679-2680 | Delete (no-op when set is empty) |
| `tracker.frozen_player_ids = set(...)` (init) | 805-809, 3946 | Delete entire assignment |
| Logging `"has_reference_profiles"` | 4161 | Drop key |
| `match_players_across_rallies_relabel` helper | 3905-3989 | Delete entire function |

All line numbers in this spec reflect 2026-05-07 state of the named files. Verify with `grep` / `Read` before each edit during implementation — they will drift as commits land.

## Implementation sequence (one PR, ~5 commits)

Each commit eval-gates on the cross-fixture panel before the next.

1. **Drop entry-point flags + dead callsites.** Remove `--reference-crops-json`, `--no-ref-crops`, `--use-existing-profiles`; delete `_load_db_reference_crops`, JSON branch, `use_existing_profiles` fallback, canonical-map writing in CLI. Tracker still accepts `reference_profiles=None`. Diagnostic AB scripts touched here (since they're downstream of these CLI surfaces) — update or delete as encountered.
2. **Collapse `frozen_player_ids` boolean guards + early-returns** in `match_tracker.py`. Eval gate.
3. **Remove `reference_profiles` parameter + `frozen_player_ids` attribute + `match_players_across_rallies_relabel` helper.** Eval gate.
4. **Delete `crop_guided_identity.py`, `relabel_with_crops.py`, `reference_crops.py` (CLI), CLI registrations.** Eval gate.
5. **API + web layer removal** + canonicalPidMapJson resolver branch removal + `reset_matcher_state.py` script extension (lands the null-out tool; running it against the dev DB happens here, against prod is per Q2) + `MATCHER_VERSION` v7 → v8. Final eval gate.

Why staged commits inside one PR: the user can review/bisect each collapse step, and any single eval regression localizes to one well-bounded change.

## MATCHER_VERSION v7 → v8

Lands in commit 5. Existing `assignmentAnchor` rows on `match_analysis_json` carry the v7 version tag and auto-invalidate; first run on each video re-solves cleanly under the collapsed code path. No explicit Prisma migration — the cache invalidation is the migration.

## Validation strategy

### Eval gates

**Cross-fixture PERMUTED panel** is the load-bearing gate. Run before each commit's collapse step on the 4-fixture panel and confirm byte-identical numbers vs `main`:

```bash
scripts/eval_cross_fixture.sh
# Expected (cross_fixture_gt_baseline_2026_05_05):
#   5c756c41 = 86.7
#   b5fb0594 = 100
#   854bb250 = 100
#   7d77980f = 88.4
#   panel avg = 93.8 PERMUTED
```

Reset state via `scripts/reset_matcher_state.py --all-with-gt` before each measurement (per `feedback_validation_clean_state.md`).

**After MATCHER_VERSION bump (commit 5):** the bump invalidates anchors, so the first run after the bump re-solves all rallies. The PERMUTED panel must still match the baseline — this proves the collapsed code path is mathematically equivalent to the v7-blind path under empty `frozen_player_ids`.

### Type / lint / unit tests

Per commit:
- `cd analysis && uv run mypy rallycut/` — clean
- `cd analysis && uv run ruff check rallycut/` — clean
- `cd analysis && uv run pytest tests` — clean. Tests exercising `reference_profiles`, `frozen_player_ids`, `crop_guided_identity` get deleted alongside their target.
- `cd api && npx tsc --noEmit && npm run lint` — clean
- `cd web && npx tsc --noEmit && npm run lint` — clean
- `cd api && npx prisma validate` (post-commit-5) — schema unchanged but sanity-checked

### Smoke test (commit 5, end-to-end)

`make dev` → upload sample video → "Analyze Match" → verify:
- No "Reference Crops" UI surface anywhere in the editor
- Match analysis runs without errors
- Player overlays display with correct PIDs (1-4)
- Re-running match analysis produces the same PIDs (anchor cache works under v8)
- Network panel shows no calls to `/v1/videos/:id/player-reference-crops*`

### Diagnostic / AB-test scripts

Files flagged by grep (`ab_test_blind_track_split.py`, `eval_dinov2_clean.py`, `eval_reid_matching.py`, `probe_*` etc.) are reviewed individually in commit 1 or 4. Each is either updated to remove dead `reference_profiles` arguments OR deleted if obsolete. None are production paths.

### Stop-conditions

Halt the merge and investigate if any of:
- Panel PERMUTED drifts even 1pp from baseline on any fixture
- A test fails for a reason other than "tests the deleted ref-crop machinery"
- mypy/tsc surfaces a non-trivial error chain
- Smoke test shows different PID labeling than pre-cleanup

## Risks

**R1 — Hidden coupling in `frozen_player_ids` gates.**
17+ call sites; some may have subtle interactions (e.g., `_classify_track_sides` consumers via `team_identity.py` / `match_solver.py`). Mitigation: staged-commit approach eval-gates after each batch; a regression localizes to one collapse step.

**R2 — Existing 22 videos have populated `canonicalPidMapJson`.**
After the resolver loses its canonical-map priority branch but BEFORE the column is nulled, those rows still hold stale data the resolver no longer reads. Mitigation: the null-out runs in the SAME commit (5) as the resolver edit.

**R3 — `match_players_across_rallies_relabel` callers.**
Verify zero callers outside the deleted ref-crop path before deletion (commit 3 prep).

**R4 — Diagnostic scripts referencing deleted symbols.**
Bounded — not production paths. If any are imported by other live tooling, deletion could cascade. Pass through each in commit 1 or 4.

**R5 — `playerProfiles` JSON column on Rally rows.**
`_load_db_reference_crops` and `use_existing_profiles` both read/write `playerProfiles`. Verify in commit 1 prep whether the blind path still writes this column for assignment-anchor reasons. If it does, leave the column alone. If it was ref-crop-exclusive, document as dormant alongside `canonicalPidMapJson`.

## Open questions

**Q1 — Branch name + PR shape.**
Suggest `cleanup/remove-ref-crop-matcher-path`. Single PR with 5 commits as outlined.

**Q2 — Production DB null-out timing.**
Dev DB nulls happen via the cleanup commit's script extension. Production DB has the same 421 rows. Recommended: run the same script against prod when the PR merges, since the resolver no longer reads canonical maps post-merge.

**Q3 — Memory updates.**
After merge:
- Move `remove_ref_crop_matching_workstream.md` to SHIPPED in `MEMORY.md` index
- Add a "dormant data" note under "Key facts" so a future cluster-pick design knows the table exists with stranded data (421 rows, 22 videos)

**Q4 — Documentation grep.**
Check `analysis/CLAUDE.md`, `web/CLAUDE.md`, `api/CLAUDE.md` for ref-crop mentions during commit 5; remove or update.

## Out-of-scope (future workstream)

The post-hoc cluster-pick UX:

- After blind matcher assigns PIDs 1-4 across all rallies, surface representative crops per PID (4 clusters per video)
- User picks which cluster is "me", "my partner", etc.
- Selections become a personalization layer that filters action views (e.g., "show only my contacts") — they do NOT influence matching
- Users who don't pick clusters still get correct stats labeled by PID 1-4

That work gets its own design when the web/api workstream picks it up. At that point: decide whether to repurpose `player_reference_crops` + `canonicalPidMapJson` or design a new schema.
