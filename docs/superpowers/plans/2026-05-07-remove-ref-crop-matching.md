# Remove Upstream Ref-Crop Matcher Path — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the upstream "ref-crop" player-matcher path across analysis + API + web, leaving the blind path as the only matcher. Keep DB and S3 ref-crop data dormant for a future post-hoc cluster-pick UX.

**Architecture:** Single feature branch with ~7 phases of accumulating edits committed in 5-6 commits. Each phase eval-gates on the cross-fixture PERMUTED panel. The blind path's preserved fixes (`a790f08`, `0e06fd7`, `5ae71f8`) are untouched; what collapses is the `frozen_player_ids` dual-path machinery and its callers.

**Tech Stack:** Python (Typer CLI, mypy, pytest, ruff), TypeScript (Express, Prisma, Next.js, Zustand), bash eval scripts.

**Spec:** `docs/superpowers/specs/2026-05-07-remove-ref-crop-matching-design.md`

---

## Spec corrections discovered during planning

While verifying the spec, three corrections were found:

1. **Function name:** the spec calls the relabel helper `match_players_across_rallies_relabel`. The actual function in `match_tracker.py:3910` is `replay_refine_from_scratchpad`. Use the correct name.
2. **Additional file to delete:** `analysis/rallycut/tracking/relabel.py` (211 lines) — separate orchestration module the spec missed. Imported by `relabel_with_crops.py` (CLI), `reference_crops.py` (CLI), and `test_relabel.py`. No other callers.
3. **R5 resolved:** `playerProfiles` JSON column on Rally rows IS written by the blind path (`match_players.py:732,734,824`). Leave the column alone; it caches profiles for assignment-anchor logic. Only the READ at `match_players.py:601` (inside the deleted `use_existing_profiles` fallback) goes away.

These corrections are baked into the tasks below. The spec is left as-is — the plan is the implementation reference of record.

## Pre-implementation context (read once)

| Concept | Location |
|---|---|
| Blind path entry | `analysis/rallycut/cli/commands/match_players.py:254` (`match_players` Typer command) |
| Frozen-profile mechanism | `analysis/rallycut/tracking/match_tracker.py` — `MatchPlayerTracker.__init__` accepts `reference_profiles`, sets `self.frozen_player_ids` (L805-809). 17+ branches gate on this set being empty (blind) vs non-empty (frozen). |
| Ref-crop CLI subcommands | `analysis/rallycut/cli/commands/reference_crops.py` (validate + suggest) and `relabel_with_crops.py` |
| Ref-crop helper module | `analysis/rallycut/tracking/relabel.py` (used only by deleted CLI commands) |
| Ref-crop appearance prototypes | `analysis/rallycut/tracking/crop_guided_identity.py` (entire module) |
| API endpoints | `api/src/routes/videos.ts:577-861` (5 endpoints) + `api/src/services/referenceCropsService.ts` |
| API match dispatcher | `api/src/services/matchAnalysisService.ts` — has `useRefCrops` parameter and `canonical_pid_map_json` nulling logic |
| Web UI | `web/src/components/PlayerReferenceCropDialog.tsx` (entire) + state in `web/src/stores/playerTrackingStore.ts` + API client in `web/src/services/api.ts` |
| Web display resolver | `web/src/utils/canonicalPid.ts` — has 3-priority canonical → applied → sort fallback |
| Eval gate | `analysis/scripts/eval_cross_fixture.sh` — runs `reset_matcher_state.py` + per-fixture PERMUTED measurement |
| MATCHER_VERSION constant | `analysis/rallycut/tracking/match_tracker.py:3907` (currently `"v7"`) |

**Baseline panel (must remain byte-identical):**
```
5c756c41 = 86.7
b5fb0594 = 100
854bb250 = 100
7d77980f = 88.4
panel avg = 93.8 PERMUTED
```

**Verification protocol before/after every phase:** reset state + run panel eval.
```bash
cd analysis
uv run python scripts/reset_matcher_state.py --all-with-gt
scripts/eval_cross_fixture.sh
```

---

## Phase 0: Branch setup

### Task 0.1: Create feature branch

**Files:** none (git operation)

- [ ] **Step 1: Confirm working tree is clean enough to branch**

Run: `git status`
Expected: existing untracked memory/report files are OK; no uncommitted edits to source files in `analysis/`, `api/`, `web/`.

- [ ] **Step 2: Create branch off `main`**

Run: `git checkout -b cleanup/remove-ref-crop-matcher-path`
Expected: `Switched to a new branch 'cleanup/remove-ref-crop-matcher-path'`

- [ ] **Step 3: Capture baseline panel numbers**

Run:
```bash
cd analysis
uv run python scripts/reset_matcher_state.py --all-with-gt
scripts/eval_cross_fixture.sh 2>&1 | tee /tmp/refcrop_cleanup_baseline.log
cd ..
```
Expected: log shows `panel avg = 93.8` (5c756c41=86.7, b5fb0594=100, 854bb250=100, 7d77980f=88.4). Save the log; subsequent phases compare to this exact run, not just the memory baseline (controls for any panel-state drift since the memory was written).

---

## Phase 1: Drop CLI entry-point flags + ref-crop callsites

End-of-phase commit message:
```
refactor(match-players): drop ref-crop CLI flags and dead callsites

- Remove --reference-crops-json, --no-ref-crops, --use-existing-profiles
- Delete _load_db_reference_crops and JSON-file branch
- Delete canonical-map writing from CLI
- Tracker still accepts reference_profiles=None (collapse in phase 2-3)
```

### Task 1.1: Remove the three CLI flags from `match_players` command

**Files:**
- Modify: `analysis/rallycut/cli/commands/match_players.py:275-315` (function signature)

- [ ] **Step 1: Read the current signature**

Read `analysis/rallycut/cli/commands/match_players.py:254-330` to see the full Typer signature with `--reference-crops-json`, `--no-ref-crops`, `--use-existing-profiles`.

- [ ] **Step 2: Delete the three flag declarations**

Use `Edit` to remove the entire blocks for:
- `reference_crops_json: Path | None = typer.Option(...)` (around L275)
- `no_ref_crops: bool = typer.Option(...)` (around L289)
- `use_existing_profiles: bool = typer.Option(...)` (around L280)

Leave all other flags untouched.

- [ ] **Step 3: Type-check**

Run: `cd analysis && uv run mypy rallycut/cli/commands/match_players.py`
Expected: errors about `reference_crops_json`, `no_ref_crops`, `use_existing_profiles` being undefined inside the function body (these go away in 1.2 - 1.4). Note them.

### Task 1.2: Delete `_load_db_reference_crops` function

**Files:**
- Modify: `analysis/rallycut/cli/commands/match_players.py:113-238` (function body)

- [ ] **Step 1: Locate exact function bounds**

Run: `grep -n "^def _load_db_reference_crops\|^def [a-z]" analysis/rallycut/cli/commands/match_players.py`
Expected: confirms line of `_load_db_reference_crops` and the next top-level def.

- [ ] **Step 2: Delete the function**

Use `Edit` (or `Read` + `Edit` block) to remove the function definition and its docstring entirely. Verify no `_load_db_reference_crops` references remain in the file.

- [ ] **Step 3: Verify no leftover callers**

Run: `grep -rn "_load_db_reference_crops" analysis/`
Expected: no output.

### Task 1.3: Delete the DB-load + JSON-file branches inside `match_players`

**Files:**
- Modify: `analysis/rallycut/cli/commands/match_players.py:437-566` (dispatch block)

- [ ] **Step 1: Read the dispatch block**

Read lines 430-580 of `match_players.py` to understand the structure: an `if reference_crops_json is None and not no_ref_crops:` branch (DB load) and an `elif reference_crops_json is not None:` branch (JSON file).

- [ ] **Step 2: Delete both branches and the surrounding flag dispatch**

Replace the entire `if/elif` chain with the equivalent blind-path code: just initialize `reference_profiles = None` (the variable is consumed downstream). Leave the downstream `if reference_profiles:` checks alone for now — phase 2 collapses those.

- [ ] **Step 3: Verify locally**

Run: `cd analysis && uv run mypy rallycut/cli/commands/match_players.py`
Expected: errors about `reference_crops_json`, `no_ref_crops` are gone. New errors (if any) point at downstream uses, which is expected and gets cleaned in subsequent tasks.

### Task 1.4: Delete the `use_existing_profiles` fallback

**Files:**
- Modify: `analysis/rallycut/cli/commands/match_players.py:589-610` (fallback block)

- [ ] **Step 1: Read the fallback block**

Read lines 585-615 of `match_players.py`. The block reads `playerProfiles` from a stored Rally row when `use_existing_profiles and reference_profiles is None`.

- [ ] **Step 2: Delete the entire `if use_existing_profiles and reference_profiles is None:` block**

Use `Edit` to remove the block. The downstream code that consumes `reference_profiles` will receive `None`, which is now the only path.

- [ ] **Step 3: Type-check the file**

Run: `cd analysis && uv run mypy rallycut/cli/commands/match_players.py`
Expected: clean (or only errors that point at downstream `reference_profiles` use, addressed in 1.5).

### Task 1.5: Delete canonical-map writing in CLI

**Files:**
- Modify: `analysis/rallycut/cli/commands/match_players.py:834-893` (canonical-map block)
- Modify: `analysis/rallycut/cli/commands/match_players.py` imports for `build_anchors_from_crops` / `compute_canonical_pid_map`

- [ ] **Step 1: Read the canonical-map block**

Read lines 830-900 of `match_players.py`. It checks for 4-pid coverage, calls `build_anchors_from_crops()` and `compute_canonical_pid_map()`, writes `Video.canonical_pid_map_json`.

- [ ] **Step 2: Delete the entire block**

Use `Edit` to remove the block. The match result is written without populating canonical map. Existing rows in DB are unaffected — they'll get nulled in phase 6.

- [ ] **Step 3: Remove now-dead imports**

Search the file for any imports of `build_anchors_from_crops`, `compute_canonical_pid_map` from `crop_guided_identity` and remove them. (`crop_guided_identity` module gets fully deleted in phase 4.)

- [ ] **Step 4: Type-check**

Run: `cd analysis && uv run mypy rallycut/cli/commands/match_players.py`
Expected: clean.

### Task 1.6: Update / delete diagnostic scripts that referenced removed CLI surfaces

**Files:**
- Possibly delete or modify: `analysis/scripts/probe_phase0_replay.py`, `analysis/scripts/ab_test_blind_track_split.py`, `analysis/scripts/ab_test_blind_track_split_v2.py`, `analysis/scripts/eval_dinov2_clean.py`, `analysis/scripts/eval_reid_matching.py`, `analysis/scripts/probe_rere_rally6_diagnose.py`, `analysis/scripts/probe_scratchpad_determinism.py`, `analysis/scripts/ab_test_pose_anchored.py`, `analysis/scripts/pipeline_capture_match.py`

- [ ] **Step 1: Re-grep for `reference_profiles` / `frozen_player_ids` / `replay_refine` in scripts**

Run: `grep -rln "reference_profiles\|frozen_player_ids\|replay_refine_from_scratchpad" analysis/scripts/`
Expected: list of scripts.

- [ ] **Step 2: Categorize each script**

For each file, open it and decide:
- **Delete** if it's a one-off probe/AB diagnostic for a closed workstream (most are).
- **Update** if it's a live, generally useful diagnostic. Replace `reference_profiles=...` calls with `reference_profiles=None` (or remove the kwarg entirely after phase 3).

`probe_phase0_replay.py` calls `replay_refine_from_scratchpad` — this function is deleted in phase 3, so the script is dead. **Delete it.**

- [ ] **Step 3: Delete the obsolete scripts**

For each "delete" decision, run `git rm analysis/scripts/<name>.py`.

- [ ] **Step 4: Type-check `analysis/`**

Run: `cd analysis && uv run mypy rallycut/`
Expected: clean.

### Task 1.7: Run analysis test suite

- [ ] **Step 1: Run pytest**

Run: `cd analysis && uv run pytest tests/ -x`
Expected: tests targeting deleted code paths fail. Note which.

- [ ] **Step 2: Decide on failing tests**

Failing tests in `test_match_tracker.py` that reference `replay_refine_from_scratchpad`, in `test_relabel.py`, in `test_crop_guided_identity.py`, and in `test_canonical_pid_determinism.py` are EXPECTED to fail — their targets are being removed. Mark them with `@pytest.mark.skip(reason="removed in ref-crop cleanup phase X")` for now; they get deleted in phase 4.

For any test failing for a DIFFERENT reason: STOP and investigate before continuing.

- [ ] **Step 3: Re-run with skips**

Run: `cd analysis && uv run pytest tests/`
Expected: clean (skipped tests counted, no failures).

### Task 1.8: Eval gate + commit phase 1

- [ ] **Step 1: Reset matcher state**

Run: `cd analysis && uv run python scripts/reset_matcher_state.py --all-with-gt`

- [ ] **Step 2: Run cross-fixture eval**

Run: `cd analysis && scripts/eval_cross_fixture.sh 2>&1 | tee /tmp/refcrop_cleanup_phase1.log`

- [ ] **Step 3: Diff against baseline**

Run: `diff /tmp/refcrop_cleanup_baseline.log /tmp/refcrop_cleanup_phase1.log`
Expected: no PERMUTED differences. Logging-only or timestamp diffs are fine.

If panel numbers drift even 1pp on any fixture: STOP, the dual path's "blind branch" wasn't quite what you thought. Bisect within the phase.

- [ ] **Step 4: Stage and commit**

Run:
```bash
git add analysis/rallycut/cli/commands/match_players.py analysis/scripts/
git commit -m "$(cat <<'EOF'
refactor(match-players): drop ref-crop CLI flags and dead callsites

- Remove --reference-crops-json, --no-ref-crops, --use-existing-profiles
- Delete _load_db_reference_crops and JSON-file branch
- Delete canonical-map writing from CLI
- Delete obsolete diagnostic scripts targeting removed surfaces
- Tracker still accepts reference_profiles=None; collapse in phase 2-3

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 2: Collapse `frozen_player_ids` boolean guards in tracker

End-of-phase commit message:
```
refactor(match-tracker): collapse frozen_player_ids dual-path branches

frozen_player_ids is now always-empty (no ref-crop callers since
previous commit). Collapse 17+ gates to their non-frozen branch.
```

### Task 2.1: Re-locate exact line numbers

Line numbers in spec/plan are 2026-05-07; previous phase did not edit `match_tracker.py` so they should still be accurate, but reverify.

- [ ] **Step 1: Confirm line numbers**

Run: `grep -n "frozen_player_ids" analysis/rallycut/tracking/match_tracker.py`
Expected: matches the list from the spec (805, 809, 917, 984, 1010, 1019, 1029, 1668, 2163, 2679, 2680, 2745, 2751, 2870, 2912, 3183, 3254, 3266, 3946, 3978, 4016, 4271). If lines drift, use the new numbers.

### Task 2.2: Collapse boolean guards (`not self.frozen_player_ids` → `True`)

**Files:**
- Modify: `analysis/rallycut/tracking/match_tracker.py` lines 917, 984, 1029, 1668, 3183

- [ ] **Step 1: Edit each guard site**

For each of the 5 lines, read 5 lines of context and apply:

L917 region — `if self.rally_count == 1 and not self.frozen_player_ids:`
becomes `if self.rally_count == 1:`

L984 region — `if self.rally_count <= 1 and not self.frozen_player_ids:`
becomes `if self.rally_count <= 1:`

L1029 region — `if self.rally_count > 1 or self.frozen_player_ids:`
becomes `if self.rally_count > 1:` (since `or False` collapses)

L1668 region — `if not getattr(self, "frozen_player_ids", None):`
becomes a no-op guard — DELETE the if and unindent its body.

L3183 region — `if not self.frozen_player_ids:`
becomes a no-op guard — DELETE the if and unindent its body.

- [ ] **Step 2: Type-check**

Run: `cd analysis && uv run mypy rallycut/tracking/match_tracker.py`
Expected: clean.

### Task 2.3: Collapse `use_side_penalty=not self.frozen_player_ids`

**Files:**
- Modify: `analysis/rallycut/tracking/match_tracker.py` lines 1010, 1019, 3254, 3266, 3978

- [ ] **Step 1: Edit each call site**

For each of the 5 lines, change `use_side_penalty=not self.frozen_player_ids` to `use_side_penalty=True`.

(Note: L3978 is inside `replay_refine_from_scratchpad`, which is deleted in phase 3. You can leave that single instance for phase 3 to drop with the function, OR include it here for consistency. Prefer the second — keeps the diff coherent.)

- [ ] **Step 2: Decide whether to inline `use_side_penalty=True`**

Read the called function (likely `_assign_tracks_to_players_global` or similar) to see if `use_side_penalty` is read elsewhere. If `True` is the only value passed anywhere, the parameter itself can be dropped in phase 3. For now, leave the kwarg explicit at call sites — the parameter cleanup is a phase-3 follow-up.

- [ ] **Step 3: Type-check**

Run: `cd analysis && uv run mypy rallycut/tracking/match_tracker.py`
Expected: clean.

### Task 2.4: Delete early-return blocks `if self.frozen_player_ids:`

**Files:**
- Modify: `analysis/rallycut/tracking/match_tracker.py` lines 2751, 2912, 4271

- [ ] **Step 1: For each site, read 15 lines of context**

Each site is an `if self.frozen_player_ids:` block that early-returns or short-circuits. Verify the pattern.

- [ ] **Step 2: Delete the if-block at each site**

For sites that early-return, the post-block code becomes the only path — delete the `if` and its body, leave the rest.

For L2912 specifically, check the surrounding logic (it's an identity-flip detection gate per the L2870 docstring): the docstring says "frozen_player_ids is empty" is the live case; deleting the gate exposes the body unconditionally — which is what we want.

- [ ] **Step 3: Type-check + lint**

Run: `cd analysis && uv run mypy rallycut/tracking/match_tracker.py && uv run ruff check rallycut/tracking/match_tracker.py`
Expected: clean.

### Task 2.5: Delete `if player_id in self.frozen_player_ids` conditional

**Files:**
- Modify: `analysis/rallycut/tracking/match_tracker.py:2163`

- [ ] **Step 1: Read context**

Read lines 2155-2175 of `match_tracker.py`. The site is a conditional that gates assignment differently for frozen vs non-frozen pids.

- [ ] **Step 2: Delete the conditional path**

Replace the `if player_id in self.frozen_player_ids: ... else: ...` with just the `else:` body.

- [ ] **Step 3: Type-check**

Run: `cd analysis && uv run mypy rallycut/tracking/match_tracker.py`
Expected: clean.

### Task 2.6: Delete the perm-update no-op

**Files:**
- Modify: `analysis/rallycut/tracking/match_tracker.py:2679-2680`

- [ ] **Step 1: Read context**

Read lines 2670-2690. The block is `self.frozen_player_ids = {perm.get(pid, pid) for pid in self.frozen_player_ids}` — a no-op when the set is empty.

- [ ] **Step 2: Delete the assignment**

Remove the line. The surrounding logic (permutation handling) keeps working without it.

- [ ] **Step 3: Type-check**

Run: `cd analysis && uv run mypy rallycut/tracking/match_tracker.py`
Expected: clean.

### Task 2.7: Run unit tests

- [ ] **Step 1: Run pytest**

Run: `cd analysis && uv run pytest tests/unit/test_match_tracker.py -x`
Expected: previously skipped tests stay skipped; non-skipped tests pass. If a non-skipped test fails: STOP and investigate.

### Task 2.8: Eval gate + commit phase 2

- [ ] **Step 1: Reset + eval**

```bash
cd analysis
uv run python scripts/reset_matcher_state.py --all-with-gt
scripts/eval_cross_fixture.sh 2>&1 | tee /tmp/refcrop_cleanup_phase2.log
diff /tmp/refcrop_cleanup_baseline.log /tmp/refcrop_cleanup_phase2.log
cd ..
```
Expected: no PERMUTED drift. This is the most critical gate — the boolean-guard collapse is where mathematical equivalence is proved.

- [ ] **Step 2: Stage and commit**

```bash
git add analysis/rallycut/tracking/match_tracker.py
git commit -m "$(cat <<'EOF'
refactor(match-tracker): collapse frozen_player_ids dual-path branches

frozen_player_ids is now always-empty (no ref-crop callers since
previous commit). Collapse 17+ gates to their non-frozen branch:
- 5 boolean guards → drop guard, keep body
- 5 use_side_penalty kwargs → True
- 3 early-return if-blocks → delete block
- 1 if/else conditional → keep else
- 1 perm-update no-op → delete

Cross-fixture PERMUTED panel byte-identical (5c756c41=86.7,
b5fb0594=100, 854bb250=100, 7d77980f=88.4).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 3: Remove `reference_profiles` parameter, `frozen_player_ids` attribute, `replay_refine_from_scratchpad`

End-of-phase commit message:
```
refactor(match-tracker): drop reference_profiles parameter, frozen_player_ids attribute, replay helper

The dual-path machinery is now unreachable. Remove the parameter
and attribute fully; delete replay_refine_from_scratchpad which only
existed for ref-crop relabel.
```

### Task 3.1: Delete `tracker.frozen_player_ids = set(...)` initializations

**Files:**
- Modify: `analysis/rallycut/tracking/match_tracker.py:805-809` (in `MatchPlayerTracker.__init__`)
- Modify: `analysis/rallycut/tracking/match_tracker.py:3946` (in `replay_refine_from_scratchpad`, deleted in 3.5)

- [ ] **Step 1: Read `__init__`**

Read lines 780-820 to see the full init.

- [ ] **Step 2: Delete the `frozen_player_ids` block in `__init__`**

The block is:
```python
self.frozen_player_ids: set[int] = set()
if reference_profiles:
    for pid, profile in reference_profiles.items():
        self.frozen_player_ids.add(pid)
```

Delete it entirely.

- [ ] **Step 3: Skip L3946 for now**

The L3946 assignment lives inside `replay_refine_from_scratchpad`, which gets fully deleted in 3.5. Don't edit it in isolation.

- [ ] **Step 4: Type-check**

Run: `cd analysis && uv run mypy rallycut/tracking/match_tracker.py`
Expected: errors about `self.frozen_player_ids` being undefined at remaining read sites (4016, 3946, …). Note these for follow-up tasks.

### Task 3.2: Remove remaining `frozen_player_ids` reads

**Files:**
- Modify: `analysis/rallycut/tracking/match_tracker.py` line 4016, plus any leftover read

- [ ] **Step 1: Re-grep**

Run: `grep -n "frozen_player_ids" analysis/rallycut/tracking/match_tracker.py`
Expected: a small set of remaining reads (likely L4016 — `"frozenPlayerIds": sorted(int(pid) for pid in tracker.frozen_player_ids)`, and any inside `replay_refine_from_scratchpad` to be deleted in 3.5).

- [ ] **Step 2: For L4016 (scratchpad serialization)**

Open lines 4010-4025. Delete the `"frozenPlayerIds": ...` key from the returned dict. Existing scratchpad files in DB may have this key; the deserializer in `replay_refine_from_scratchpad` will be deleted in 3.5, so consumers go away.

- [ ] **Step 3: Type-check**

Run: `cd analysis && uv run mypy rallycut/tracking/match_tracker.py`
Expected: only errors inside `replay_refine_from_scratchpad` (L3946+), addressed in 3.5.

### Task 3.3: Remove `reference_profiles` parameter from `MatchPlayerTracker.__init__`

**Files:**
- Modify: `analysis/rallycut/tracking/match_tracker.py:790-810` (init signature + body)

- [ ] **Step 1: Read the init signature**

Read lines 780-810 of `match_tracker.py`.

- [ ] **Step 2: Remove the parameter**

Delete `reference_profiles: dict[int, PlayerAppearanceProfile] | None = None,` from the parameter list and the corresponding paragraph in the docstring (around L799).

Inside the body: the `if reference_profiles:` block was already deleted in 3.1. Confirm no other in-init use of `reference_profiles` remains.

- [ ] **Step 3: Type-check**

Run: `cd analysis && uv run mypy rallycut/tracking/match_tracker.py`
Expected: callers of `MatchPlayerTracker(...)` that pass `reference_profiles=...` flagged. Note them.

### Task 3.4: Remove `reference_profiles` parameter from `match_players_across_rallies`

**Files:**
- Modify: `analysis/rallycut/tracking/match_tracker.py:4083-4180` (top-level `match_players_across_rallies` function)
- Modify: `analysis/rallycut/cli/commands/match_players.py` (the lone caller after phase 1)

- [ ] **Step 1: Read the function signature + early body**

Read lines 4080-4180 of `match_tracker.py`. The function takes `reference_profiles` and at L4142-4147 has:
```python
if not extract_reid and reference_profiles:
    if any(p.reid_embedding is not None for p in reference_profiles.values()):
        ...
if reference_profiles:
    f"{sorted(reference_profiles.keys())}"
```

And at L4161:
```python
"has_reference_profiles": bool(reference_profiles),
```

And at L4169 it passes `reference_profiles=reference_profiles` into the inner tracker.

- [ ] **Step 2: Remove the parameter and all usages**

Delete:
- The parameter from the signature (L4088)
- The docstring paragraph about it (L4109)
- The `if not extract_reid and reference_profiles:` block (L4142-4145)
- The `if reference_profiles:` log block (L4147-4150)
- The `"has_reference_profiles": bool(reference_profiles),` log key (L4161)
- The `reference_profiles=reference_profiles` kwarg in the inner tracker constructor call (L4169)

- [ ] **Step 3: Update the caller in `match_players.py`**

Run: `grep -n "match_players_across_rallies" analysis/rallycut/cli/commands/match_players.py`
Find the call site and remove the `reference_profiles=...` kwarg.

- [ ] **Step 4: Type-check `analysis/`**

Run: `cd analysis && uv run mypy rallycut/`
Expected: clean (or only errors inside `replay_refine_from_scratchpad`, addressed in 3.5).

### Task 3.5: Delete `replay_refine_from_scratchpad`

**Files:**
- Modify: `analysis/rallycut/tracking/match_tracker.py:3910-3998` (full function body, def to next def)

- [ ] **Step 1: Read function bounds**

Read lines 3905-4000 of `match_tracker.py` to confirm the function spans 3910 to roughly 3996.

- [ ] **Step 2: Delete the function entirely**

Use `Edit` to remove the `def replay_refine_from_scratchpad(...)` definition + its body + its docstring. The `scratchpad_to_dict` function (L3998) stays — that's an unrelated serializer.

- [ ] **Step 3: Re-grep for callers**

Run: `grep -rn "replay_refine_from_scratchpad" analysis/`
Expected: only test files (`test_match_tracker.py:2030+`) and possibly diagnostic scripts. Production callers should be zero (the lone production caller `relabel_with_crops.py` is deleted in phase 4).

- [ ] **Step 4: Mark dependent tests as skip (cleanup happens in phase 4)**

In `analysis/tests/unit/test_match_tracker.py`, find the test class containing `replay_refine_from_scratchpad` (starts around L2030 with docstring `"replay_refine_from_scratchpad reproduces refine_assignments stages 1+2"`). Add `@pytest.mark.skip(reason="replay function removed in ref-crop cleanup")` to that class. The class gets fully deleted in phase 4.

- [ ] **Step 5: Type-check + lint**

Run: `cd analysis && uv run mypy rallycut/ && uv run ruff check rallycut/`
Expected: clean.

### Task 3.6: Run unit tests

- [ ] **Step 1: Run pytest**

Run: `cd analysis && uv run pytest tests/ -x`
Expected: clean (with skips for ref-crop-machinery tests).

### Task 3.7: Eval gate + commit phase 3

- [ ] **Step 1: Reset + eval**

```bash
cd analysis
uv run python scripts/reset_matcher_state.py --all-with-gt
scripts/eval_cross_fixture.sh 2>&1 | tee /tmp/refcrop_cleanup_phase3.log
diff /tmp/refcrop_cleanup_baseline.log /tmp/refcrop_cleanup_phase3.log
cd ..
```
Expected: no PERMUTED drift.

- [ ] **Step 2: Stage and commit**

```bash
git add analysis/rallycut/tracking/match_tracker.py analysis/rallycut/cli/commands/match_players.py analysis/tests/unit/test_match_tracker.py
git commit -m "$(cat <<'EOF'
refactor(match-tracker): drop reference_profiles param, frozen_player_ids attr, replay helper

The dual-path machinery is now unreachable. Remove:
- reference_profiles parameter from MatchPlayerTracker.__init__ and
  match_players_across_rallies
- frozen_player_ids attribute (initialization + reads)
- replay_refine_from_scratchpad (only existed for ref-crop relabel)
- Logging key "has_reference_profiles" and scratchpad key
  "frozenPlayerIds"

Cross-fixture PERMUTED panel byte-identical.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 4: Delete ref-crop modules and CLI subcommands

End-of-phase commit message:
```
refactor: delete ref-crop modules, CLI subcommands, and tests

- Delete tracking/crop_guided_identity.py
- Delete tracking/relabel.py
- Delete cli/commands/relabel_with_crops.py
- Delete cli/commands/reference_crops.py
- Drop CLI registrations
- Delete tests targeting deleted code
```

### Task 4.1: Delete `crop_guided_identity.py`

**Files:**
- Delete: `analysis/rallycut/tracking/crop_guided_identity.py`

- [ ] **Step 1: Verify no imports remain**

Run: `grep -rn "crop_guided_identity\|build_anchors_from_crops\|compute_canonical_pid_map\|validate_prototypes" analysis/`
Expected: only the file itself and possibly tests in `test_crop_guided_identity.py`.

- [ ] **Step 2: Delete the file**

Run: `git rm analysis/rallycut/tracking/crop_guided_identity.py`

### Task 4.2: Delete `tracking/relabel.py`

**Files:**
- Delete: `analysis/rallycut/tracking/relabel.py`

- [ ] **Step 1: Verify no imports remain (after `relabel_with_crops.py` and `reference_crops.py` are deleted in 4.3, 4.4)**

Run: `grep -rn "from rallycut.tracking.relabel\|from rallycut.tracking import relabel" analysis/`
Expected: only `relabel_with_crops.py`, `reference_crops.py`, and `test_relabel.py`. These get deleted in 4.3, 4.4, 4.6.

- [ ] **Step 2: Delete the file**

Run: `git rm analysis/rallycut/tracking/relabel.py`

### Task 4.3: Delete `relabel_with_crops.py` CLI

**Files:**
- Delete: `analysis/rallycut/cli/commands/relabel_with_crops.py`

- [ ] **Step 1: Delete the file**

Run: `git rm analysis/rallycut/cli/commands/relabel_with_crops.py`

### Task 4.4: Delete `reference_crops.py` CLI

**Files:**
- Delete: `analysis/rallycut/cli/commands/reference_crops.py`

- [ ] **Step 1: Delete the file**

Run: `git rm analysis/rallycut/cli/commands/reference_crops.py`

### Task 4.5: Drop CLI registrations from `cli/main.py`

**Files:**
- Modify: `analysis/rallycut/cli/main.py:19-26, 56-58`

- [ ] **Step 1: Read the current main.py**

Already read. Imports at L19-26 and registrations at L56-58.

- [ ] **Step 2: Remove the imports**

Delete lines 19-21 (`suggest_reference_crops` import), 22-24 (`validate_reference_crops` import), and 26 (`relabel_with_crops_cmd` import) from `analysis/rallycut/cli/main.py`.

- [ ] **Step 3: Remove the registrations**

Delete lines 56 (`validate-reference-crops`), 57 (`suggest-reference-crops`), and 58 (`relabel-with-crops`) from `analysis/rallycut/cli/main.py`.

- [ ] **Step 4: Type-check**

Run: `cd analysis && uv run mypy rallycut/cli/main.py`
Expected: clean.

### Task 4.6: Delete dead test files

**Files:**
- Delete: `analysis/tests/unit/test_crop_guided_identity.py`
- Delete: `analysis/tests/unit/test_relabel.py`
- Modify: `analysis/tests/unit/test_match_tracker.py` (drop the skipped `replay_refine_from_scratchpad` test class)
- Possibly delete or modify: `analysis/tests/integration/test_canonical_pid_determinism.py`

- [ ] **Step 1: Delete `test_crop_guided_identity.py`**

Run: `git rm analysis/tests/unit/test_crop_guided_identity.py`

- [ ] **Step 2: Delete `test_relabel.py`**

Run: `git rm analysis/tests/unit/test_relabel.py`

- [ ] **Step 3: Open `test_match_tracker.py` and delete the `replay_refine_from_scratchpad` test class**

Find the class around L2030 and delete it (about 200 lines). Use `Read` first to confirm exact boundaries.

- [ ] **Step 4: Open `test_canonical_pid_determinism.py`**

Read the file. If it tests the canonical-map writing path (now deleted), delete the file (`git rm`). If it tests something more general about determinism that's still meaningful, edit out the canonical-map assertions.

- [ ] **Step 5: Run pytest**

Run: `cd analysis && uv run pytest tests/`
Expected: clean.

### Task 4.7: Verify analysis is clean

- [ ] **Step 1: Re-grep for any residual ref-crop terms**

Run: `grep -rn "reference_profiles\|frozen_player_ids\|crop_guided_identity\|replay_refine_from_scratchpad\|relabel_with_crops" analysis/rallycut/`
Expected: no output.

Run: `grep -rn "reference_crops\|ref_crops" analysis/rallycut/`
Expected: empty (or only docstring mentions in unrelated functions, which are fine).

- [ ] **Step 2: Type-check + lint**

Run: `cd analysis && uv run mypy rallycut/ && uv run ruff check rallycut/`
Expected: clean.

### Task 4.8: Eval gate + commit phase 4

- [ ] **Step 1: Reset + eval**

```bash
cd analysis
uv run python scripts/reset_matcher_state.py --all-with-gt
scripts/eval_cross_fixture.sh 2>&1 | tee /tmp/refcrop_cleanup_phase4.log
diff /tmp/refcrop_cleanup_baseline.log /tmp/refcrop_cleanup_phase4.log
cd ..
```
Expected: no PERMUTED drift.

- [ ] **Step 2: Stage and commit**

```bash
git add -A analysis/
git commit -m "$(cat <<'EOF'
refactor: delete ref-crop modules, CLI subcommands, and tests

- Delete tracking/crop_guided_identity.py (490 lines)
- Delete tracking/relabel.py (211 lines)
- Delete cli/commands/relabel_with_crops.py (199 lines)
- Delete cli/commands/reference_crops.py (478 lines)
- Drop CLI registrations from main.py
- Delete tests/unit/test_crop_guided_identity.py
- Delete tests/unit/test_relabel.py
- Drop replay_refine_from_scratchpad test class from
  tests/unit/test_match_tracker.py
- Resolve tests/integration/test_canonical_pid_determinism.py

Cross-fixture PERMUTED panel byte-identical.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 5: API and web layer removal

End-of-phase commit message:
```
refactor(api,web): remove ref-crop endpoints, service, dialog, store

- Delete 5 endpoints in api/src/routes/videos.ts
- Delete api/src/services/referenceCropsService.ts
- Drop useRefCrops parameter and canonical-map nulling in matchAnalysisService
- Delete web/src/components/PlayerReferenceCropDialog.tsx
- Drop referenceCrops state + 3 methods from playerTrackingStore
- Drop 5 ref-crop methods + useRefCrops from web api client
- Drop canonical-map priority branch from canonicalPid resolver
- Drop canonical-map plumbing from ActionOverlay/ActionLabelingMode
```

### Task 5.1: Delete the 5 ref-crop endpoints in `api/src/routes/videos.ts`

**Files:**
- Modify: `api/src/routes/videos.ts:577-861` (5 endpoint handlers)

- [ ] **Step 1: Read the endpoint block**

Read lines 575-865 of `api/src/routes/videos.ts`. The 5 endpoints are GET, POST, POST validate, POST suggest, POST pre-assign, DELETE — confirm exact line ranges.

- [ ] **Step 2: Delete the entire range**

Use `Edit` to remove all 5 handler blocks. Adjust trailing route registrations if needed.

- [ ] **Step 3: Remove dead imports at top of file**

Search the top of `videos.ts` for imports from `referenceCropsService` and remove. Search for `PlayerReferenceCrop` Prisma type imports — DON'T remove (the model still exists in schema; just no API uses it).

- [ ] **Step 4: Type-check**

Run: `cd api && npx tsc --noEmit`
Expected: clean.

### Task 5.2: Delete `referenceCropsService.ts`

**Files:**
- Delete: `api/src/services/referenceCropsService.ts`

- [ ] **Step 1: Verify no imports remain**

Run: `grep -rn "referenceCropsService\|from.*referenceCropsService" api/src/`
Expected: only the file itself.

- [ ] **Step 2: Delete the file**

Run: `git rm api/src/services/referenceCropsService.ts`

- [ ] **Step 3: Type-check**

Run: `cd api && npx tsc --noEmit`
Expected: clean.

### Task 5.3: Drop `useRefCrops` parameter and crop-loading from `matchAnalysisService.ts`

**Files:**
- Modify: `api/src/services/matchAnalysisService.ts` — `useRefCrops` parameter and conditional crop-loading (around L240-354)

- [ ] **Step 1: Read the service**

Read `api/src/services/matchAnalysisService.ts` (full file, ~700 lines).

- [ ] **Step 2: Remove `useRefCrops` from `runMatchAnalysis` signature**

Drop the parameter from the function signature and any internal references. The CLI invocation (which uses `--reference-crops-json` / `--no-ref-crops`) becomes a single unconditional call without those flags.

- [ ] **Step 3: Remove conditional crop-loading block**

Around L240-354, there's a block that conditionally loads ref-crops from DB and passes them to the CLI as `--reference-crops-json`. Delete the entire block. The CLI no longer accepts those flags (phase 1).

- [ ] **Step 4: Remove canonical-map nulling logic**

Around L651-730, find the code that nulls `Video.canonicalPidMapJson` after match analysis. Delete it. (Existing rows get nulled by the script in 5.10.)

- [ ] **Step 5: Type-check**

Run: `cd api && npx tsc --noEmit`
Expected: callers of `runMatchAnalysis(..., useRefCrops)` flagged.

### Task 5.4: Update callers of `runMatchAnalysis`

**Files:**
- Modify: any file in `api/src/` that calls `runMatchAnalysis` with `useRefCrops`

- [ ] **Step 1: Find callers**

Run: `grep -rn "runMatchAnalysis" api/src/`

- [ ] **Step 2: Drop `useRefCrops` argument from each call site**

For each caller, remove the `useRefCrops` argument. The lone caller that previously passed `true` was the (deleted) ref-crop POST endpoint — gone in 5.1. Other callers (Modal webhook, etc.) passed `false`; they just drop the arg.

- [ ] **Step 3: Type-check**

Run: `cd api && npx tsc --noEmit`
Expected: clean.

- [ ] **Step 4: Lint**

Run: `cd api && npm run lint`
Expected: clean.

### Task 5.5: Delete `PlayerReferenceCropDialog.tsx`

**Files:**
- Delete: `web/src/components/PlayerReferenceCropDialog.tsx`

- [ ] **Step 1: Find consumers**

Run: `grep -rn "PlayerReferenceCropDialog" web/src/`
Expected: parent component(s) that mount the dialog. Note them.

- [ ] **Step 2: Remove dialog mount points from parents**

For each consumer, remove the `<PlayerReferenceCropDialog ... />` JSX usage and any state/handlers that opened it (e.g., a "Reference Crops" button in `PlayerTrackingToolbar` or similar).

- [ ] **Step 3: Delete the dialog file**

Run: `git rm web/src/components/PlayerReferenceCropDialog.tsx`

- [ ] **Step 4: Type-check**

Run: `cd web && npx tsc --noEmit`
Expected: clean.

### Task 5.6: Drop `referenceCrops` state and 3 methods from `playerTrackingStore.ts`

**Files:**
- Modify: `web/src/stores/playerTrackingStore.ts:72-76, 945` (and method definitions wherever they are)

- [ ] **Step 1: Read the store**

Read `web/src/stores/playerTrackingStore.ts` to map state + methods.

- [ ] **Step 2: Remove `referenceCrops` state slice and the 3 methods**

Delete:
- The `referenceCrops` field from the state shape
- `loadReferenceCrops(videoId)` method
- `addReferenceCrop(...)` method
- `removeReferenceCrop(...)` method
- Any reset/clear logic that touches `referenceCrops`

- [ ] **Step 3: Find consumers that use these state/methods**

Run: `grep -rn "referenceCrops\|loadReferenceCrops\|addReferenceCrop\|removeReferenceCrop" web/src/`
Expected: only the store file (and the deleted dialog, which was the lone consumer).

- [ ] **Step 4: Type-check**

Run: `cd web && npx tsc --noEmit`
Expected: clean.

### Task 5.7: Drop 5 ref-crop methods + `useRefCrops` from `web/src/services/api.ts`

**Files:**
- Modify: `web/src/services/api.ts` — methods around L2125+ and `runMatchAnalysis` around L2639+

- [ ] **Step 1: Locate methods**

Run: `grep -n "PlayerReferenceCrop\|referenceCrop\|useRefCrops" web/src/services/api.ts`

- [ ] **Step 2: Delete the 5 methods**

Delete:
- `getPlayerReferenceCrops`
- `uploadPlayerReferenceCrop`
- `deletePlayerReferenceCrop`
- `validateReferenceCrops`
- `suggestReferenceCrops`

Plus any related TypeScript types (`PlayerReferenceCrop` payload shapes, etc.) declared at the top of the file.

- [ ] **Step 3: Update `runMatchAnalysis` signature**

Drop the `useRefCrops` option from the `runMatchAnalysis` method's options parameter. Update any internal pass-through to the API call.

- [ ] **Step 4: Type-check**

Run: `cd web && npx tsc --noEmit`
Expected: clean.

### Task 5.8: Drop the canonical-map priority branch from `canonicalPid.ts`

**Files:**
- Modify: `web/src/utils/canonicalPid.ts` (full file)

- [ ] **Step 1: Read the file**

The current resolver has 3-priority: canonicalRallyMap → appliedFullMapping → playerNumberMap.

- [ ] **Step 2: Drop the canonicalRallyMap branch**

Edit `resolveCanonicalPid` and `pidToTrackId` to remove the `canonicalRallyMap` parameter and the branch that consults it. The new resolver is 2-priority: appliedFullMapping → playerNumberMap.

`canonicalRallyMapFor` helper can be deleted entirely.

The `CanonicalRallyMap` type can be deleted.

- [ ] **Step 3: Update consumers**

Run: `grep -rn "canonicalRallyMap\|resolveCanonicalPid\|pidToTrackId\|canonicalPidMap" web/src/`
For each consumer, remove the `canonicalRallyMap` argument from the resolver call and any state/prop that supplied it (`Video.canonicalPidMap`, `MatchAnalysis.canonicalPidMap` field reads).

This will touch `ActionOverlay.tsx`, `ActionLabelingMode.tsx`, possibly other components.

- [ ] **Step 4: Drop `MatchAnalysis.canonicalPidMap` field consumption**

In `web/src/services/api.ts`, find the `MatchAnalysis` type definition and remove the `canonicalPidMap` field. The API still returns it (DB column is dormant); the client just stops reading it.

- [ ] **Step 5: Type-check**

Run: `cd web && npx tsc --noEmit`
Expected: clean.

### Task 5.9: Lint web

- [ ] **Step 1: Run lint**

Run: `cd web && npm run lint`
Expected: clean. If unused-imports complaints appear (e.g., dead `Video` field), clean them up.

### Task 5.10: Extend `reset_matcher_state.py` to null `canonicalPidMapJson` and run against dev DB

**Files:**
- Modify: `analysis/scripts/reset_matcher_state.py`

- [ ] **Step 1: Read the current script**

Read `analysis/scripts/reset_matcher_state.py` to understand its current resets (assignmentAnchor + likely canonical_pid_map_json already).

- [ ] **Step 2: Confirm or add canonical-map nulling**

If it already nulls `canonical_pid_map_json`: confirm it's still wired correctly.

If not: add a `UPDATE videos SET canonical_pid_map_json = NULL` step to the reset path. Use the existing Prisma-or-psql pattern that the script uses.

- [ ] **Step 3: Run against dev DB**

Run: `cd analysis && uv run python scripts/reset_matcher_state.py --all-with-gt`
Expected: confirms anchors + canonical maps cleared. Verify with a Prisma query or psql:
```bash
cd api && npx prisma studio
# or:
psql -h localhost -p 5436 -U postgres -d rallycut -c "SELECT COUNT(*) FROM videos WHERE canonical_pid_map_json IS NOT NULL;"
```
Expected: count = 0 on dev DB.

### Task 5.11: Bump `MATCHER_VERSION` v7 → v8

**Files:**
- Modify: `analysis/rallycut/tracking/match_tracker.py:3907`

- [ ] **Step 1: Edit**

Change `MATCHER_VERSION = "v7"` to `MATCHER_VERSION = "v8"`.

- [ ] **Step 2: Type-check**

Run: `cd analysis && uv run mypy rallycut/`
Expected: clean.

### Task 5.12: Final eval gate

- [ ] **Step 1: Reset state (the script now also nulls canonical maps, plus the MATCHER_VERSION bump invalidates anchors)**

```bash
cd analysis
uv run python scripts/reset_matcher_state.py --all-with-gt
```

- [ ] **Step 2: Run cross-fixture eval**

```bash
scripts/eval_cross_fixture.sh 2>&1 | tee /tmp/refcrop_cleanup_phase5.log
diff /tmp/refcrop_cleanup_baseline.log /tmp/refcrop_cleanup_phase5.log
cd ..
```
Expected: no PERMUTED drift. This is the load-bearing final gate — it proves the v8 collapsed code path is mathematically equivalent to v7-blind under empty `frozen_player_ids`.

If panel drifts: STOP, investigate which phase introduced it (binary search via `git stash` / `git checkout` on intermediate commits).

### Task 5.13: Stage and commit phase 5

- [ ] **Step 1: Stage**

```bash
git add api/src/ web/src/ analysis/scripts/reset_matcher_state.py analysis/rallycut/tracking/match_tracker.py
```

- [ ] **Step 2: Commit**

```bash
git commit -m "$(cat <<'EOF'
refactor(api,web): remove ref-crop endpoints, service, dialog, store

- Delete 5 endpoints in api/src/routes/videos.ts
- Delete api/src/services/referenceCropsService.ts
- Drop useRefCrops parameter + crop-loading + canonical-map nulling
  in matchAnalysisService
- Delete web/src/components/PlayerReferenceCropDialog.tsx
- Drop referenceCrops state + 3 methods from playerTrackingStore
- Drop 5 ref-crop methods + useRefCrops from web api client
- Drop canonical-map priority branch from canonicalPid resolver
- Drop canonical-map plumbing from ActionOverlay / ActionLabelingMode
- Extend reset_matcher_state.py to null canonical_pid_map_json
- Bump MATCHER_VERSION v7 → v8 (auto-invalidates assignmentAnchor)

DB schema unchanged: player_reference_crops table and
Video.canonicalPidMapJson column kept dormant for future post-hoc
cluster-pick UX repurposing.

Cross-fixture PERMUTED panel byte-identical post-v8.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Phase 6: Smoke test + memory updates

### Task 6.1: End-to-end smoke test

- [ ] **Step 1: Start dev environment**

Run: `make dev`
Expected: services start (PostgreSQL :5436, MinIO :9000, API :3001, web :3000).

- [ ] **Step 2: Open the editor in a browser**

Open http://localhost:3000.

- [ ] **Step 3: Upload a sample video and analyze**

Upload a short volleyball clip. Trigger "Analyze Match". Observe progress through detect → track → match-analysis → stats.

- [ ] **Step 4: Verify no ref-crop UI surfaces**

Check the editor for any "Reference Crops" button, dialog, or menu entry. Expected: none.

- [ ] **Step 5: Verify network panel**

Open browser DevTools → Network tab → filter for `player-reference-crops`. Expected: zero requests.

- [ ] **Step 6: Verify player overlays render with PIDs 1-4**

Play the rally; PlayerOverlay should display 4 boxes labeled 1-4 (or pX names). Expected: PIDs visible, no "no canonical map" empty states.

- [ ] **Step 7: Re-run match analysis**

Trigger a second match analysis pass on the same video. Expected: anchor cache works under v8 (run is fast on subsequent invocations); PIDs identical run-to-run.

- [ ] **Step 8: Stop dev**

Run: `make stop`

### Task 6.2: Update memory

**Files:**
- Modify: `/Users/mario/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY.md`
- Modify: `/Users/mario/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/remove_ref_crop_matching_workstream.md`
- Modify: `/Users/mario/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY_ARCHIVE.md`

- [ ] **Step 1: Update workstream memory to SHIPPED**

Edit `remove_ref_crop_matching_workstream.md`'s frontmatter `name` field to `Clean up ref-crop flow SHIPPED 2026-05-XX` and add a "Shipped" section at the top documenting the actual commit shas.

- [ ] **Step 2: Move from "Current workstreams" to archive in `MEMORY.md`**

In `MEMORY.md`, remove the workstream entry from "Current workstreams" and add it to the archive section (or move the file to MEMORY_ARCHIVE.md per the index convention).

- [ ] **Step 3: Add a "Key facts" entry about dormant data**

In `MEMORY.md` under "Key facts (not derivable from code)", add:
```
- player_reference_crops table: 421 rows across 22 videos (latest 2026-04-27); `Video.canonicalPidMapJson` column nulled. Both kept dormant after ref-crop matcher removal (2026-05-XX); future post-hoc cluster-pick UX may repurpose them.
```

- [ ] **Step 4: Verify memory hygiene**

Run a wc on `MEMORY.md` to confirm it stays under 200 lines (per `feedback_memory_hygiene.md`).
Run: `wc -l /Users/mario/.claude/projects/-Users-mario-Personal-Projects-RallyCut/memory/MEMORY.md`

### Task 6.3: Production DB null-out (post-merge)

This task runs AFTER the PR merges to `main` and gets deployed.

- [ ] **Step 1: Connect to production DB**

Use the team's standard prod-DB access pattern (e.g., `make dev-prod` env or direct psql via VPN).

- [ ] **Step 2: Run the canonical-map null-out**

Either:
- Run `reset_matcher_state.py --all-with-gt` against the prod DB (preferred — same path as dev), OR
- Issue: `UPDATE videos SET canonical_pid_map_json = NULL;` directly via psql

Expected: prod `videos.canonical_pid_map_json` column shows NULL across all rows.

- [ ] **Step 3: Verify**

`SELECT COUNT(*) FROM videos WHERE canonical_pid_map_json IS NOT NULL;`
Expected: 0.

### Task 6.4: Open the PR

- [ ] **Step 1: Push the branch**

Run: `git push -u origin cleanup/remove-ref-crop-matcher-path`

- [ ] **Step 2: Open PR**

Run:
```bash
gh pr create --title "cleanup: remove upstream ref-crop matcher path" --body "$(cat <<'EOF'
## Summary

- Remove the upstream "ref-crop" player-matcher path across analysis, API, and web
- Collapse the dual-path (`frozen_player_ids` / `reference_profiles`) machinery in the tracker; blind path is now THE path
- Bump `MATCHER_VERSION` v7 → v8 (auto-invalidates `assignmentAnchor` cache)
- Keep `player_reference_crops` table and `Video.canonicalPidMapJson` column dormant for a future post-hoc cluster-pick UX

Spec: `docs/superpowers/specs/2026-05-07-remove-ref-crop-matching-design.md`
Plan: `docs/superpowers/plans/2026-05-07-remove-ref-crop-matching.md`

## Test plan

- [x] Cross-fixture PERMUTED panel byte-identical (5c756c41=86.7, b5fb0594=100, 854bb250=100, 7d77980f=88.4) at every phase
- [x] mypy + ruff clean across analysis/
- [x] tsc + lint clean across api/ and web/
- [x] pytest clean (with deletions for ref-crop test files)
- [x] Smoke test: upload → analyze → no ref-crop UI, no `/player-reference-crops` requests, PIDs render correctly
- [ ] Post-merge: run `reset_matcher_state.py --all-with-gt` against production DB

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

- [ ] **Step 3: Capture PR URL** for the user to review.

---

## Self-review checklist (run before handing this plan off)

- [x] Spec coverage: each spec section maps to a phase or task
- [x] Phase 1 covers spec's "drop entry-point flags + dead callsites"
- [x] Phase 2 covers spec's "collapse boolean guards + early-returns"
- [x] Phase 3 covers spec's "remove parameter + attribute + relabel helper"
- [x] Phase 4 covers spec's "delete CLI commands + module"
- [x] Phase 5 covers spec's "API + web layer removal + canonicalPidMapJson null-out + MATCHER_VERSION bump"
- [x] Eval gate runs at every phase end
- [x] No placeholders ("TBD", "implement later")
- [x] All file paths absolute or repo-relative; no ambiguity
- [x] Function name corrected: `replay_refine_from_scratchpad` (not `match_players_across_rallies_relabel`)
- [x] Additional file `analysis/rallycut/tracking/relabel.py` flagged for deletion
- [x] R5 resolved: `playerProfiles` is shared with blind path, leave column alone
- [x] Open question Q4 (CLAUDE.md ref-crop mentions) resolved: grep returned empty, no doc edits needed
- [x] Smoke test concrete (specific UI assertions, network filter, playback verification)
- [x] Memory updates explicit and bounded
- [x] Production DB null-out explicit (Q2 from spec)
