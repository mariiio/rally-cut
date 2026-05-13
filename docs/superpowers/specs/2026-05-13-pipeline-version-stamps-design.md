# Pipeline version stamps for `actions_json` / `contacts_json`

**Status:** Design — 2026-05-13
**Owner:** Mario
**Related memory:**
- [panel_baseline_regression_2026_05_07](../../../memory/panel_baseline_regression_2026_05_07.md) — 21pp panel drop from mixed-vintage tracking
- [coherence_repair_sub_2b_2026_05_13](../../../memory/coherence_repair_sub_2b_2026_05_13.md) — F1+F2 cleared by `redetect_all_actions` (action-vintage, not real)
- [knowledge_state_2026_04_26](../../../memory/knowledge_state_2026_04_26.md) — 04-25/26 measurement contamination
- Implements the same pattern as `MATCHER_VERSION` (analysis/rallycut/tracking/match_tracker.py:3888)

## Problem

`PlayerTrack.actionsJson` and `PlayerTrack.contactsJson` are produced by `action_classifier.py` and `contact_detector.py`, written once at tracking time, and consumed many places (audits, stats, evals) without any indication of which code-vintage produced them. At least three workstreams have been derailed by this:

1. **Panel baseline regression** — A 21pp PERMUTED drop on the panel that turned out to be 12 panel rallies running on stale tracking from a prior code vintage. The matcher eval read DB content, never re-ran tracking, and produced misleading numbers for two days before the cause was found.
2. **Coherence repair Sub-2.B Phase 2** — C-4 violations F1 + F2 cleared after `redetect_all_actions` alone (no code change). The signal was action-vintage, not a real coherence gap. Two days of investigation wasted on a chimerical "regression."
3. **04-25/26 measurement contamination** — A whole batch of measurements partially invalid because actions/contacts in DB were from an older classifier than the metric being reported.

All three failed the same way: silent mixed-vintage data. The `MATCHER_VERSION` pattern already solves this for `match_analysis_json.rallies[].assignmentAnchor` — when the version on a cached anchor differs from the current constant, it's silently stripped with a counter log. The discipline holds (7 bumps in 5 days, 2026-05-02 to 2026-05-07, each named in the commit subject) because the cost of *not* bumping is immediately visible: the next eval run produces visibly wrong PERMUTED numbers.

For `actions_json` / `contacts_json` there is no equivalent feedback loop. Staleness is silent — which is exactly the bug.

## Goal

Mirror the `MATCHER_VERSION` pattern at producer and consumer for `actions_json` and `contacts_json`. Make stale-vintage data **visible** at audit time. Backstop the discipline with a pre-commit hook so the constants can't silently rot.

## Non-goals

These are explicitly out of scope for this workstream:

- **Auto-invalidation** — when a producer's version bumps, no automated downstream re-run. Manual fleet refresh (see §6) is the workflow.
- **Audit dashboard / cron** — the audit CLIs already exist; we don't add scheduling or dashboards here.
- **Audit-CLI consolidation** — `audit-coherence-invariants` and `audit-pid-invariants` stay separate.
- **`canonical_pid_map_json` cleanup** — column is dormant, see [memory/MEMORY.md](../../../memory/MEMORY.md) anchor section.
- **Sub-2.B Phase 2 coherence repair** — that workstream is parked. This design ships the *infrastructure* that would have prevented the chimerical F1+F2 signal in the first place.

## Architecture

```
                        ┌──────────────────┐
                        │ action_classifier│  ACTION_PIPELINE_VERSION
                        │ contact_detector │  CONTACT_PIPELINE_VERSION
                        └────────┬─────────┘
                                 │ (stamp at producer)
              ┌──────────────────┼──────────────────┐
              │                  │                  │
   ┌──────────▼─────────┐  ┌─────▼──────────┐  ┌────▼──────────────┐
   │ Modal tracking     │  │ reattribute-   │  │ redetect_all_     │
   │ webhook → API      │  │ actions CLI    │  │ actions script    │
   │ (TS saveTrackingResult)│ │ (direct SQL)  │  │ (direct SQL)      │
   └──────────┬─────────┘  └─────┬──────────┘  └────┬──────────────┘
              │                  │                  │
              ▼                  ▼                  ▼
              ┌─────────────────────────────────────┐
              │  player_tracks.actions_pipeline_version │
              │  player_tracks.contacts_pipeline_version│
              └────────────────┬────────────────────┘
                               │
              ┌────────────────┼────────────────┐
              │                │                │
   ┌──────────▼──────┐  ┌──────▼────────┐  ┌────▼────────────┐
   │ audit-coherence-│  │ audit-pid-    │  │ compute-match-  │
   │ invariants      │  │ invariants    │  │ stats (later)   │
   │ (skip + counter)│  │ (skip+counter)│  │                 │
   └─────────────────┘  └───────────────┘  └─────────────────┘
```

Three independent components keep the pattern clean:

1. **Producer-side:** Two version constants in the producer modules; every code path that *writes* an `actions_json` or `contacts_json` blob also stamps the corresponding version column.
2. **Consumer-side:** Audit CLIs read the column alongside the JSON. If the column is older than the current constant, the rally is skipped for invariants whose inputs depend on the stale data, and a counter is reported.
3. **Enforcement:** Pre-commit hook fails the commit when the producer file is modified without the constant being bumped (escape hatch via commit-message marker).

## Data model

### Schema migration

`api/prisma/schema.prisma`:

```prisma
model PlayerTrack {
  // ... existing fields ...
  contactsJson              Json?     @map("contacts_json")
  actionsJson               Json?     @map("actions_json")
  contactsPipelineVersion   String?   @map("contacts_pipeline_version")
  actionsPipelineVersion    String?   @map("actions_pipeline_version")
  // ... existing fields ...
}
```

Both columns are nullable. The semantic mapping is:
- `contacts_json IS NOT NULL` ⇒ `contacts_pipeline_version IS NOT NULL` (post-deploy invariant)
- `contacts_json IS NULL`     ⇒ `contacts_pipeline_version IS NULL` (no content, no version)
- Same for actions.

### Migration SQL

```sql
ALTER TABLE player_tracks
  ADD COLUMN contacts_pipeline_version TEXT,
  ADD COLUMN actions_pipeline_version  TEXT;

-- Backfill existing rows with the sentinel.
UPDATE player_tracks SET contacts_pipeline_version = 'v0'
  WHERE contacts_json IS NOT NULL;
UPDATE player_tracks SET actions_pipeline_version = 'v0'
  WHERE actions_json IS NOT NULL;
```

`v0` is a sentinel reserved for the backfill — never written by code.

## Python constants

### `analysis/rallycut/tracking/action_classifier.py`

Add near the top of the module (mirror placement of `MATCHER_VERSION` at `match_tracker.py:3888`):

```python
# ACTION_PIPELINE_VERSION:
#  - v0: 2026-05-13 — sentinel for backfilled rows (never written by code).
#  - v1: 2026-05-13 — initial release of pipeline-version stamping. Bump
#          on any change that affects RallyActions output: classify_rally,
#          repair_action_sequence, viterbi_decode_actions, reattribute_players,
#          assign_court_side_from_teams, propagate_court_side, the
#          synthetic-serve placement helpers, or any classifier dependency
#          that materially changes the serialized output.
ACTION_PIPELINE_VERSION = "v1"
```

### `analysis/rallycut/tracking/contact_detector.py`

```python
# CONTACT_PIPELINE_VERSION:
#  - v0: 2026-05-13 — sentinel for backfilled rows (never written by code).
#  - v1: 2026-05-13 — initial release. Bump on any change that affects
#          ContactSequence output: detect_contacts, the candidate gates,
#          seq-anchored rescue, deduplication, _resolve_court_side, the
#          GBM classifier wiring, or any helper that changes the
#          serialized output.
CONTACT_PIPELINE_VERSION = "v1"
```

The two constants are **independent**. A change to contact detection bumps only `CONTACT_PIPELINE_VERSION`; a change to action classification bumps only `ACTION_PIPELINE_VERSION`. This matters because `reattribute_actions` (a producer for `actions_json`) does not touch `contacts_json` — a stale `contacts_pipeline_version` next to a fresh `actions_pipeline_version` is a meaningful state.

## Producer plumbing

Four producer sites — each must stamp the version column it writes.

### Producer 1 — Modal tracking webhook (Python → TS)

The tracking pipeline assembles a `RallyTrackerResult` dict (or equivalent) inside the Python runner that posts to `POST /v1/webhooks/tracking-rally-complete`. At the assembly site, add two top-level fields alongside `contacts` and `actions`:

```python
result = {
    ...
    "contacts": contact_sequence.to_dict() if contact_sequence else None,
    "actions": rally_actions.to_dict() if rally_actions else None,
    "contactsPipelineVersion": CONTACT_PIPELINE_VERSION if contact_sequence else None,
    "actionsPipelineVersion": ACTION_PIPELINE_VERSION if rally_actions else None,
    ...
}
```

The exact file is the one that assembles `trackerResult` for the Modal webhook (likely `analysis/rallycut/tracking/player_tracker.py` or a CLI command wrapping it; writing-plans will pin the exact line).

The version field is `None` when the corresponding content is `None` (no contacts ⇒ no contact version).

### Producer 2 — `reattribute_actions` CLI (Python direct SQL)

`analysis/rallycut/cli/commands/reattribute_actions.py:994`:

```python
# Before
cur.execute(
    "UPDATE player_tracks SET actions_json = %s WHERE id = %s",
    (json.dumps(new_actions_json), pt_id),
)

# After
cur.execute(
    "UPDATE player_tracks SET actions_json = %s, actions_pipeline_version = %s WHERE id = %s",
    (json.dumps(new_actions_json), ACTION_PIPELINE_VERSION, pt_id),
)
```

This writer touches only `actions_json` — `contacts_json` and `contacts_pipeline_version` are intentionally untouched. Semantically: reattribution may run on top of older contact detections; the contact version stays at whatever it was on the row.

### Producer 3 — `redetect_all_actions` script

`analysis/scripts/redetect_all_actions.py:184`:

```python
# Before
cur.execute(
    "UPDATE player_tracks SET contacts_json = %s, actions_json = %s WHERE id = %s",
    (json.dumps(new_contacts_json), json.dumps(new_actions_json), pt_id),
)

# After
cur.execute(
    "UPDATE player_tracks SET contacts_json = %s, actions_json = %s, "
    "contacts_pipeline_version = %s, actions_pipeline_version = %s WHERE id = %s",
    (json.dumps(new_contacts_json), json.dumps(new_actions_json),
     CONTACT_PIPELINE_VERSION, ACTION_PIPELINE_VERSION, pt_id),
)
```

### Producer 4 — TS `saveTrackingResult`

`api/src/services/playerTrackingService.ts` — three touchpoints in the file:

1. **`PlayerTrackerOutput` interface (~line 142)** — add the fields to the input shape:
   ```ts
   interface PlayerTrackerOutput {
     // ... existing ...
     contacts?: ContactsData;
     actions?: ActionsData;
     contactsPipelineVersion?: string | null;
     actionsPipelineVersion?: string | null;
   }
   ```
   The other two interface declarations in the same file at ~line 160 and ~line 281 also need the new fields if they describe the same wire shape.

2. **Modal webhook re-assembly (~lines 470–491)** — when contacts/actions are reconstituted from the webhook payload, propagate the version fields onto the rebuilt `PlayerTrackerOutput` so step 3 can persist them.

3. **`saveTrackingResult` Prisma writes (lines 725–726 and 746–747)** — extend the Prisma `data` payload with:
   ```ts
   contactsPipelineVersion: trackerResult.contacts ? (trackerResult.contactsPipelineVersion ?? null) : null,
   actionsPipelineVersion:  trackerResult.actions  ? (trackerResult.actionsPipelineVersion  ?? null) : null,
   ```

**Out of scope for stamping (line 1576–1577):** the track-swap path reads existing `contactsJson` and `actionsJson` and writes them back with mutated `playerTrackId` fields. It does not re-run any classifier. Leave the version columns unchanged in this update (omit them from the `data` payload entirely so the existing column value is preserved). This is consistent with the swap being a manual user action, not a classifier output — analogous to a GT edit.

## Consumer behavior

Both audit CLIs gain "skip + counter" semantics for stale-version rallies. The CLIs themselves stay thin shells — the logic lives in the orchestrator modules.

### `coherence_invariants.run_all` (analysis/rallycut/tracking/coherence_invariants.py)

The orchestrator already excludes rallies that fail I-1 / I-3 / I-6 (upstream issues). Add a stale-version exclusion pass:

1. When reading per-rally data, also read `actions_pipeline_version`. (Coherence rules C-1..C-4 all depend on `actions_json` only; `contacts_pipeline_version` is irrelevant here.)
2. If `actions_pipeline_version != ACTION_PIPELINE_VERSION`, skip the rally and increment a counter.
3. **"Stale" semantics:** strict inequality — any column value not equal to the current constant is treated as stale (including unexpected values from a future-but-uncoordinated bump). This mirrors `MATCHER_VERSION`'s consumer check (`anchor.get("matcherVersion") != MATCHER_VERSION`).
4. Return both the violation list AND a `StaleVersionReport` summary:
   ```python
   @dataclass(frozen=True)
   class StaleVersionReport:
       total_rallies: int
       skipped_stale_actions: int          # rallies with actions_pipeline_version < current
       skipped_stale_contacts: int         # rallies with contacts_pipeline_version < current
       current_actions_version: str
       current_contacts_version: str
       observed_actions_versions: dict[str, int]   # e.g., {"v0": 8, "v1": 100}
       observed_contacts_versions: dict[str, int]
   ```
5. The audit-CLI shell prints the stale-version block at the top of the report:
   ```
   12 of 144 rallies skipped due to stale pipeline version
     - 12 stale actions_pipeline_version (observed: {"v0": 8, "v1": 100, "v2": 24};  current: v3)
   Run scripts/redetect_all_actions.py --apply to refresh.
   ```
6. Exit code: 1 if any error-severity violation; 0 otherwise. Nonzero stale count is reported in the header as a warning but does not fail the audit, so ad-hoc runs aren't blocked (the prominent "X of Y rallies skipped" line is itself the call-to-action).

### `pid_invariants.run_all` (analysis/rallycut/tracking/pid_invariants.py)

Same pattern, but the per-invariant skip is narrower:
- **I-3** depends on `actions_json` → skip if `actions_pipeline_version` is stale.
- **I-4** depends on `contacts_json` → skip if `contacts_pipeline_version` is stale.
- **I-1, I-2, I-5, I-6, I-7, I-8** depend on `primary_track_ids` / `track_to_player` / `team_assignments`, not on the actions/contacts content → unaffected.

In practice this means a single rally can be skipped for I-3 but still checked for I-1; the `StaleVersionReport` reports total rallies skipped per invariant family.

### CLI shell changes

`audit_coherence_invariants_cmd` and `audit_pid_invariants_cmd` already format a `rich.table`. Extend to also format the `StaleVersionReport` above the violations table. Reuse the same Violation list shape; no breaking changes to the CLI return contract.

## Pre-commit enforcement

The project's existing pre-commit infrastructure is a Claude Code `PreToolUse` hook on Bash calls (`.claude/hooks/pre-commit-check.sh`), wired in `.claude/settings.json`. Extend that hook — single source of truth, fires on `git commit` regardless of caller.

### Hook logic

Append a new check block to `.claude/hooks/pre-commit-check.sh`:

```bash
# Pipeline version-bump enforcement
# When action_classifier.py or contact_detector.py is staged, require either:
#  - the corresponding *_PIPELINE_VERSION constant changes in the same commit, OR
#  - the commit message includes the marker [no-version-bump]
for ENTRY in \
  "analysis/rallycut/tracking/action_classifier.py:ACTION_PIPELINE_VERSION" \
  "analysis/rallycut/tracking/contact_detector.py:CONTACT_PIPELINE_VERSION"; do
  FILE="${ENTRY%:*}"
  CONST="${ENTRY##*:}"
  if echo "$STAGED" | grep -qFx "$FILE"; then
    DIFF=$(cd "$PROJECT_DIR" && git diff --cached -- "$FILE")
    # Look for an addition line containing the constant assignment.
    if ! echo "$DIFF" | grep -qE "^\+${CONST}\s*=\s*\"v[0-9]+\""; then
      # Check for the escape-hatch marker in the commit -m argument.
      if ! echo "$COMMAND" | grep -qF '[no-version-bump]'; then
        ERRORS="${ERRORS}${FILE} modified without bumping ${CONST}. Add '[no-version-bump]' to the commit message if behavior is unchanged (docstring/typo/comment-only).\n"
      fi
    fi
  fi
done
```

### Design notes

- **Scope:** Only the two constant-owning files are watched. Helpers in other modules that `action_classifier.py` imports are not watched directly — the discipline is that if you change a helper whose output affects `RallyActions.to_dict()`, you must also touch the owning file (at minimum to bump the constant + add a comment line). Code review enforces the "also" part. This trades some false negatives for a much smaller, clearer watched set.
- **Escape hatch:** `[no-version-bump]` in the commit message. Required when the change is genuinely cosmetic (docstring fix, comment edit, refactor that produces byte-identical output). The marker forces a moment of thought — and is searchable in `git log` for future review.
- **False positives:** A commit that bumps the constant in a *different* file (e.g., bumps `CONTACT_PIPELINE_VERSION` but only `action_classifier.py` is staged) still blocks. This is the right behavior — the file you touched needs its own bump.

## Fleet refresh + reference baseline

Post-deploy ordering matters because between schema migration and fleet refresh, the audit will skip MOST rallies (everything is `v0`, current is `v1`). That's expected and short-lived.

### Step sequence

1. **Land the PR** (schema migration + producer + consumer + hook in one commit).
2. **Apply the migration** in production DB.
3. **Run fleet refresh:**
   ```bash
   cd analysis
   uv run python scripts/redetect_all_actions.py --apply
   ```
   This re-runs `detect_contacts` and `classify_rally_actions` against current code for every rally in DB, stamps `v1` on all rows. Expected runtime: comparable to a full fleet retrack (script already exists, see [memory/redetect_all_actions_fix_2026_05_11](../../../memory/redetect_all_actions_fix_2026_05_11.md)).
4. **Snapshot the reference baseline** using the existing fleet orchestrator:
   ```bash
   uv run python analysis/scripts/catalog_c4_violations.py
   ```
   This produces `analysis/reports/coherence_c4_catalog/<today>.csv` + `<today>_summary.md` (same format as the existing 2026-05-13 catalog from Sub-2.A).
5. **Commit the baseline snapshot** under `analysis/reports/coherence_c4_catalog/2026-05-14_baseline.csv` + `2026-05-14_baseline_summary.md` (use the actual deploy date). The previous 2026-05-13 catalogs stay in place as the pre-stamp reference for historical comparison.

### Validation gate

After step 3, the audit's `StaleVersionReport` should report `skipped_stale_actions = 0` and `skipped_stale_contacts = 0`. If anything remains stale, the fleet refresh failed for those rallies — investigate before committing the baseline.

## `analysis/CLAUDE.md` checklist

Add a new section under "Running Diagnostics & Long Processes" (the top-level CLAUDE.md already references "ML skills include detailed versions of these rules"):

```markdown
## Post-classifier-change checklist

When you change `analysis/rallycut/tracking/action_classifier.py` or
`analysis/rallycut/tracking/contact_detector.py` in a way that affects
serialized output:

1. **Bump the constant** in the same commit:
   - `ACTION_PIPELINE_VERSION` in `action_classifier.py`
   - `CONTACT_PIPELINE_VERSION` in `contact_detector.py`
2. **Add a version-history comment line** above the constant explaining
   what changed and why (mirror the `MATCHER_VERSION` style in
   `analysis/rallycut/tracking/match_tracker.py:3870-3888`).
3. **Pre-commit hook** will reject commits that touch these files without a
   version bump. Add `[no-version-bump]` to the commit message if the
   change is genuinely cosmetic (docstring/typo/comment-only).
4. **After merge, refresh the fleet:**
   ```bash
   uv run python scripts/redetect_all_actions.py --apply
   ```
5. **Re-run the audit** to confirm zero stale rallies and refresh the
   reference baseline if behavior changed:
   ```bash
   uv run rallycut audit-coherence-invariants
   ```

The same discipline applies to `MATCHER_VERSION` in `match_tracker.py`
(see `tests/unit/test_assignment_anchor_versioning.py`), though that
constant has additional consumer-side feedback (stale anchors visibly
break the next PERMUTED panel).
```

## Test pinning

Add `analysis/tests/unit/test_pipeline_version_versioning.py` (mirror `test_assignment_anchor_versioning.py`):

- Assert `ACTION_PIPELINE_VERSION` and `CONTACT_PIPELINE_VERSION` are non-empty strings.
- Assert neither equals `"v0"` (sentinel reserved for backfill).
- Assert neither equals values from a hard-coded `LEGACY_VERSIONS` set (initially empty after first release; grows on each future bump). This catches accidental reverts.
- Doc string explains the discipline of bumping on producer changes.

Producer-stamping tests:
- `test_action_classifier_stamps_version`: call `classify_rally_actions` end-to-end on a minimal contact sequence, verify the returned dict carries the current constant. (If stamping happens at a higher assembly site, the test is at that site.)
- `test_reattribute_actions_stamps_version`: invoke the CLI on a fixture and assert the DB row's `actions_pipeline_version` is the current constant.
- `test_redetect_all_actions_stamps_both`: same for the script.

Consumer-skipping tests:
- `test_coherence_audit_skips_stale_actions`: insert a player_track with `actions_pipeline_version = 'v0'`, run `coherence_invariants.run_all`, assert the rally is in `skipped_stale_actions` and not in the violation list.
- `test_pid_audit_skips_stale_actions_for_i3_only`: same setup, verify I-3 skips but I-1/I-2 still run.

Pre-commit hook tests: bash test scenarios (mock `git diff --cached` output) verifying the hook blocks unbumped modifications and passes when the marker is present. Lives in `analysis/tests/unit/test_pre_commit_hook.sh` or similar; reuses the existing hook test pattern if one exists.

## Risks & mitigations

| Risk | Mitigation |
|------|------------|
| Hook false-positives on docstring-only commits | `[no-version-bump]` marker. Documented in CLAUDE.md. |
| Hook false-negatives when a helper module changes | Code-review discipline; the helper change can't be merged without the reviewer either approving the no-bump or requesting the owning file get a bump too. Same trade as MATCHER_VERSION. |
| Reattribute writes `v1` over a downstream-inconsistent older `actions_json` content | Acceptable: reattribute's output reflects the *current* `action_classifier.py` code (it calls `reattribute_players` from that module). Version represents the code-vintage of the most recent producer, not the originating classifier. |
| Audit confused by mixed-vintage rows (`actions_pipeline_version != contacts_pipeline_version`) | Independent versions handle this correctly. The audit per-invariant skip uses the appropriate column. |
| Migration locks `player_tracks` table | The ADD COLUMN is metadata-only in Postgres ≥11. The two UPDATE statements scan all rows but lock per-row — acceptable for our scale (~ a few thousand player_tracks). |
| TS-side type drift if a future contributor adds a new write path that doesn't stamp the column | Add a Prisma-level audit (eslint rule or grep CI) checking every `player_tracks` update that touches `contactsJson` or `actionsJson` also touches the matching version column. Out of scope for v1; revisit if drift happens. |

## Implementation order

The order below is **logical, not commit-by-commit** — the whole infrastructure should land in a single PR (and ideally a single commit) so the hook never sees a half-installed state. If split across commits, ensure step 3 ships *after* step 2 in the same PR, and that the commit introducing producer plumbing for `action_classifier.py` / `contact_detector.py` already includes the version-constant addition line (so the hook's diff check passes).

1. Schema migration (Prisma).
2. Python constants (with version-history comments).
3. Pre-commit hook addition to `.claude/hooks/pre-commit-check.sh`.
4. Producer plumbing: TS `saveTrackingResult` + `reattribute_actions` CLI + `redetect_all_actions` script + Python trackerResult assembly site.
5. Consumer skip-counter logic in `coherence_invariants` and `pid_invariants`.
6. Audit CLI shell updates (report rendering).
7. Tests (unit + integration).
8. analysis/CLAUDE.md checklist.
9. Deploy → migrate → fleet refresh → snapshot baseline → commit baseline.

## Acceptance criteria

- [ ] Both columns exist on `player_tracks` with `v0` backfill for existing non-null content.
- [ ] `ACTION_PIPELINE_VERSION` and `CONTACT_PIPELINE_VERSION` constants exist with `v1` values and version-history comment blocks.
- [ ] Every producer write path stamps the matching column (verified by unit tests).
- [ ] Audit CLIs report `StaleVersionReport` and skip stale rallies for content-dependent invariants.
- [ ] Pre-commit hook blocks modifications to the producer files without a version bump, accepts the `[no-version-bump]` marker as escape hatch.
- [ ] Post-fleet-refresh audit reports zero stale rallies.
- [ ] Reference baseline CSV + summary committed.
- [ ] analysis/CLAUDE.md has the post-classifier-change checklist.
