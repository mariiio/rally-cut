# PID Invariant I-8: Cross-Rally Majority Vote Cleanup CLI

**Date:** 2026-05-10
**Status:** Design — pending implementation plan
**Workstream context:** Sub-1.1.E. Companion to Sub-1.1.B (`cleanup-team-assignments`) and Sub-1.1.C (`cleanup-stale-attribution`). Closes legacy I-8 violations (scrambled team partitions) on existing data without re-tracking, using cross-rally consistency. Complements the future producer fix in `classify_teams`.

## Goal

Provide a CLI command — `rallycut cleanup-team-labels-by-majority <video-id>` — that walks every rally of a given video, computes the per-PID majority team label across all rallies, and rewrites scrambled partitions (1A+3B, 3A+1B, etc.) to a valid 2v2 partition that matches an existing partition pattern in the video. Closes 80%+ of legacy I-8 violations without introducing the silent-corruption risk that a side-switch-detecting algorithm would carry.

## Motivation

The I-8 invariant (added 2026-05-10, commit `5a6968b`) surfaced 112 fleet-wide rallies where the team partition is not 2v2 — a structural impossibility for beach volleyball. Visual verification on `b097dd2a` confirmed the canonical PIDs are stable across rallies (P1 is always the same physical player) but the team labels for those PIDs vary per rally. Since a player's team is invariant across a match, the deviant rallies' labels are wrong.

`classify_teams` produces these errors when:
- A primary track arrives mid-rally (insufficient early-frame Y data for median-Y classification).
- Two primary tracks have very similar early-frame Y near the court split line (ambiguous classification).

A producer-side fix in `classify_teams` is the proper long-term solution but requires re-tracking all videos to take effect on legacy data. This cleanup CLI is the lightweight alternative that fixes existing data in place.

## Scope

**In scope:**
- New CLI command `rallycut cleanup-team-labels-by-majority <video-id>` (Typer + Rich, mirrors `cleanup-team-assignments` and `cleanup-stale-attribution` patterns).
- Per-video per-PID majority computation across all rallies with valid A/B labels.
- Per-rally fix attempt with strict 2v2 safety check + "must match an existing valid partition" gate.
- `--dry-run` flag for safe preview.
- Skip semantics for rallies that don't apply (already 2v2, missing data, etc.).
- One DB transaction per rally for atomicity.
- Registration in `cli/main.py`.

**Out of scope:**
- Per-segment side-switch detection (rejected — silent-corruption risk outweighs marginal coverage; producer fix supersedes).
- Modifying `classify_teams` itself (separate workstream — producer fix).
- Modifying `positions_json`, `actions_json["actions"]`, `contacts_json`, or any other field besides `actions_json["teamAssignments"]`.
- Modifying `match_analysis_json` on the videos table.
- Re-tracking.
- Unit tests for the CLI itself (logic is "compute majority, validate, write" with predicates already tested elsewhere; `--dry-run` is the verification tool).
- Sub-2 / Sub-3 work.

## Architecture

### Algorithm

For a given video:

1. **Load all rallies** (rally_id, primary_track_ids, actions_json["teamAssignments"]) ordered by start_ms.
2. **Compute valid partitions seen** — iterate rallies; for each rally where the partition over primary tracks is exactly 2A + 2B, record the partition shape (frozen mapping of primary tid → label).
3. **Compute per-PID majority** — for each canonical PID 1..4 (only those present as primary in some rally), count A vs B labels across all rallies where the PID has a valid label. Determine majority (A, B, or "tie" if equal).
4. **Per-rally fix loop** — for each rally:
   a. If partition is already 2A + 2B → no-op (`[noop]`).
   b. If primary_track_ids is empty or has size != 4 → skip (`[skip]`, I-1's domain).
   c. If team_assignments is None or any primary lacks a valid A/B label → skip (`[skip]`, I-6's domain).
   d. Build `candidate` by replacing each primary's label with its per-PID majority. If any PID's majority is "tie" → skip (`[skip-tie]`).
   e. **2v2 safety check**: if candidate is not 2A+2B → skip (`[ambiguous]`).
   f. **Existing-partition safety check**: if candidate's frozen mapping is not in the set of valid partitions seen elsewhere in the video → skip (`[ambiguous]`). Note: a rally's candidate equaling its own (invalid) partition can't happen because invalid partitions aren't recorded as valid.
   g. Otherwise → write back via single transaction (`[fix]`).
5. **Summary**: counts of fixed / no-op / skipped / ambiguous.

### Why the safety checks matter

- **2v2 safety check** ensures we never commit a non-2v2 partition. Without this, a video where per-PID majority happens to be all-A or all-B would produce 4A+0B "fixes."
- **Existing-partition check** ensures we don't invent a partition shape no rally in the video has. This prevents committing fixes on side-switched videos where per-PID majority is meaningless (the existing partitions form two distinct 2v2 patterns, neither of which is the per-PID majority).

Together: cleanups only fire when the data unambiguously points to a single valid 2v2 partition. Conservative by design.

### Per-PID majority semantics

- Counts only labels for the PID across rallies where the PID has a valid label (A or B).
- Ties (equal count of A and B) are treated as "no majority" — those PIDs cannot be flipped, blocking the rally's fix.
- This makes side-switched videos safe by default: PIDs that are 50% A / 50% B (because of the side switch) have no majority, so the rally won't get fixed. It stays as a visible I-8 violation.

### Idempotence

Re-running on already-clean data:
- Step 4(a) catches every rally as 2v2 → no-op.
- No DB writes.

### Why this won't introduce new I-8 violations

A "fix" only commits when:
- Primary set is exactly 4
- All primaries have valid A/B labels
- Per-PID majority is decisive for every primary
- Candidate is 2v2
- Candidate matches an existing partition in the video

The chain of safety gates makes silent corruption impossible. Worst case is "leaves rally unchanged."

### `--dry-run` semantics

Identical to Sub-1.1.C: runs the full algorithm, prints per-rally outcome with `[DRY]` prefix, makes no DB writes, prints summary with `[DRY RUN]` marker.

## Rollout sequence

1. Land the CLI command + `main.py` registration (single commit).
2. **Dry-run pass on the fleet** — loop the 42 videos with I-8 violations, capture counts.
3. Sanity-check totals: expected ~80% of I-8 violations marked as "fix" candidates, the rest as "ambiguous" (mostly side-switched videos).
4. **Real run** on the same 42 videos.
5. **Re-audit fleet**. Expected: ~80% reduction in I-8 count.
6. Update memory with post-cleanup baseline.

## Done criteria

- `uv run rallycut cleanup-team-labels-by-majority --help` shows the command + `--dry-run` flag.
- Dry-run mode produces non-empty diff output and does NOT mutate the DB (verifiable by audit before/after).
- After execute pass, fleet I-8 count drops substantially. Remaining I-8 violations are concentrated on side-switched or otherwise ambiguous videos.
- No new I-1 / I-2 / I-3 / I-4 / I-5 / I-6 / I-7 violations introduced (regression check).

## Risks

1. **Per-PID majority is wrong for the whole video.** If `classify_teams` systematically miscategorized a player across most rallies of a video (not the typical "a few rallies are off"), the majority for that PID is wrong, and "fixing" rallies would write wrong labels in a different way. **Mitigation**: the existing-partition safety check provides a second gate — if the per-PID majority produces a partition that no other rally has, we don't commit. So even with a wrong majority, we only commit fixes that align with an established pattern. Worst case is "no fix made on a video where every rally is wrong" — true but rare.

2. **Side-switched videos.** Per-PID majority for a 50/50 split is "tie" and the rally won't get fixed. These remain visible I-8 violations. This is acknowledged scope, not a bug.

3. **Edge case: video with only 1 valid 2v2 rally.** Per-PID majority is computed from a single rally, which is enough to determine each PID's label. Existing-partition set is the singleton of that rally's partition. Candidate fixes will only commit if they match it. Conservative behavior preserved.

4. **DB write failure mid-rally.** Per-rally transactions; failure leaves the rally untouched but doesn't abort the video. Standard pattern from prior cleanup CLIs.

5. **Convention drift.** "A"/"B" ↔ team identity is hardcoded. If `action_classifier.py:138` changes the convention, cleanup CLIs (B, C, E) all need updating together. Documented in spec.

## File-change summary

- New (1): `analysis/rallycut/cli/commands/cleanup_team_labels_by_majority.py` (~120 LOC)
- Modified (1): `analysis/rallycut/cli/main.py` (~2 lines: import + `app.command`)
- **Total:** 2 files, ~122 LOC.
