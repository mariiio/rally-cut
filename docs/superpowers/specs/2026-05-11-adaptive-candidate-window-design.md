# Adaptive Candidate-Generation Window (v3.0)

**Date:** 2026-05-11
**Status:** Design — pending implementation plan
**Workstream context:** Third iteration of the action-attribution improvement ladder. v1 (`attribution_team_chain_v1_2026_05_11.md`) relaxed the local nearest-guard predicate. v2 (`joint-attribution-v2-design.md`, NO-GO) proved hard-rule joint search amplifies upstream signal noise. v3 pivots to **upstream**: validated data shows candidate-generation is the bottleneck for the dominant absent-GT error class (servers tracked AFTER the serve contact, missing from `_find_nearest_players`' ±15 frame window).

## Goal

Recover 3-4 same-team-server attribution errors on the 3 fresh-GT videos by extending `_find_nearest_players`' search window forward when the standard ±15 frame window returns fewer than `len(primary_track_ids)` candidates. Surgical change to two functions in `contact_detector.py`. Env-flag rollback via `ADAPTIVE_PLAYER_SEARCH_WINDOW`.

## Motivation

### The validated bottleneck

After v1 ship + v2 NO-GO, the rank-of-GT validation on the 26 panel errors revealed:

- 31% rank-2 (uncertainty / coherence-repair sweet spot, ~+8 fixable via v3.2 work)
- 19% rank-1 (algorithm overrode the right answer)
- 19% rank-3/4 (hard but reachable)
- **31% ABSENT** — GT player not in `Contact.player_candidates` at all (UNREACHABLE by ANY attribution-layer fix)

The 8 absent cases are concentrated in same-team errors (67% of same-team errors are absent-GT). All 8 are serves. Investigation showed **7 of 8 are caused by the GT server being tracked only AFTER the serve contact frame** — the player tracker doesn't detect them at the moment of serve (off-screen behind baseline, tossing the ball, etc.) but picks them up after they step into the play.

Frame-gap distribution (server first detected at, serve frame):

| Case | Gap (frames after serve) |
|---|---:|
| gigi/39e866fd | 3 |
| cece/79ecfb2d | 7 |
| gigi/5b6f0474 | 19 |
| wawa/06c13117 | 40 |
| gigi/72c8229b | 49 |
| gigi/bc9345c1 | 85 |
| wawa/7094136a | 104 |
| wawa/8c49e480 | (static-track mislabel, separate issue) |

### Validated fix

Offline re-computation with widened windows:

| Window | Cases recovered (of 4 with contacts) |
|---|---:|
| ±15 (current) | 0 of 4 |
| ±30 | 1 of 4 |
| **±60** | **3 of 4** |
| ±120 | 4 of 4 |

(3 of 7 cases have NO contact at the serve frame at all — synthetic serves handled by `_attribute_synthetic_serves`, separate code path.)

Window=60 catches 3 of 4 cases that have a contact to expand from. 2 of those become **rank 1** with the wider window (gigi/5b6f0474, wawa/06c13117) — meaning the GT server becomes the proximity-best candidate and would be attributed correctly by the existing pipeline.

### Why this is NOT v2-style architectural overcommitment

v2 tried to be ambitious (joint solver over the whole rally) and failed because hard rules amplified upstream noise. v3 is the opposite: **a surgical fix to one function, validated by data, with a clear ceiling and a clean rollback.**

This is also independent of and orthogonal to the v3.2 coherence-repair-loop work that targets the rank-2 cross-team errors. The two ship paths don't interact.

## Scope

### In scope

- Modify `_find_nearest_players` in `analysis/rallycut/tracking/contact_detector.py` (line ~600).
- Add a forward-only fallback window when standard ±15 search returns fewer than `len(primary_track_ids)` candidates.
- Expand: `+60 frames forward` from the contact, lower bound stays at contact_frame.
- Merge second-pass results into the candidate list, keeping best-per-track-id entry.
- Sort + return top-N as before. No change to caller-visible behavior except the candidate list is richer.
- Apply the same fallback to `_find_nearest_player` (singular) for symmetry — same logic, returns the single nearest including from the expanded window.
- Env flag `ADAPTIVE_PLAYER_SEARCH_WINDOW` (default `"1"` = ON). Read at call time inside both functions. Setting `"0"` restores standard ±15 behavior.
- Unit tests at `analysis/tests/unit/test_contact_detector_adaptive_window.py`:
  - Truth-table: standard window returns 4 candidates → no expansion.
  - Standard window returns 2 candidates, expansion adds 2 more → returns 4.
  - Forward-only direction: a player tracked BEFORE the contact gets excluded from the expansion (they were already in the standard window).
  - Env flag OFF: behavior is byte-identical to pre-v3.
  - Integration test on a constructed rally mimicking gigi/5b6f0474 pattern.
- A/B harness on the 3 GT videos: re-run baseline measurement with env flag OFF then ON; expect +2-3pp combined accuracy gain.
- Deploy: `reattribute-actions` re-run on the 3 GT videos + fleet (~70 videos).
- Memory entry post-ship.

### Out of scope (deferred to other workstreams)

- **Cross-rally server identity** (v3.1 candidate): vote on server pid across consecutive same-team-serving rallies. Would catch the 4 wrong-of-two-teammates same-team-server errors PLUS provide a fallback for cases 4-7 above. Larger workstream (~1 week).
- **Coherence-repair loop** (v3.2 candidate): C-2 violation detection + local repair. Targets the rank-2 cross-team errors (8 of 17). Independent of this v3.
- **Synthetic-serve attribution improvements**: cases cece/79ecfb2d, gigi/39e866fd, wawa/7094136a have NO contact at the serve frame at all — synthetic serves. Handled by `_attribute_synthetic_serves` in `cli/commands/reattribute_actions.py`. Separate workstream.
- **Static-track mislabel detection** (wawa/8c49e480 case): a non-player track was assigned as a primary. Belongs to the player-tracking workstream.
- **Multi-stage expansion** (60 then 120): catches +1 additional case (gigi/bc9345c1). Add as v3.0.1 if v3.0 measurement justifies it.

## Design

### §1 — Where the change lives

`analysis/rallycut/tracking/contact_detector.py`, inside two functions:

- `_find_nearest_player` at line ~525 (singular — returns single best candidate)
- `_find_nearest_players` at line ~600 (plural — returns ranked top-N)

The change is identical in both: after computing the standard ±15 result, check if it's underfull, and if so run a forward-only second pass and merge.

### §2 — Adaptive logic

Both functions share this control flow (pseudocode; the implementation plan picks between "refactor into a `_collect_candidates` helper called twice" vs "inline the second pass with a tight loop"):

```python
def _find_nearest_players(
    frame: int,
    ball_x: float,
    ball_y: float,
    player_positions: list[PlayerPosition],
    search_frames: int = 15,
    max_candidates: int = 4,
    ...
    primary_track_ids: list[int] | None = None,
):
    # Pass 1: standard ±search_frames window (existing logic, unchanged).
    best_per_track_p1 = _collect_best_per_track(
        player_positions, frame, search_frames,
        ball_x, ball_y, primary_track_ids,
        lower_bound_frame=frame - search_frames,
    )

    # Adaptive fallback: if Pass 1 found fewer than max_candidates tracks
    # AND env flag is on, expand forward to capture late-tracked players
    # (validated cause of absent-GT serves).
    enabled = os.environ.get("ADAPTIVE_PLAYER_SEARCH_WINDOW", "1") != "0"
    if enabled and len(best_per_track_p1) < max_candidates:
        best_per_track_p2 = _collect_best_per_track(
            player_positions, frame,
            search_frames=_ADAPTIVE_FORWARD_FRAMES,
            ball_x=ball_x, ball_y=ball_y,
            primary_track_ids=primary_track_ids,
            lower_bound_frame=frame,  # forward-only — never look earlier than contact
        )
        # Merge: prefer Pass 1 entries (closer to contact frame). Pass 2
        # only contributes track_ids that Pass 1 missed entirely.
        merged = dict(best_per_track_p1)
        for tid, entry in best_per_track_p2.items():
            if tid not in merged:
                merged[tid] = entry
        best_per_track = merged
    else:
        best_per_track = best_per_track_p1

    # ... existing ranking + return logic operates on best_per_track ...
```

Where:
- `_ADAPTIVE_FORWARD_FRAMES = 60` (new module-level constant).
- `_collect_best_per_track` is either a new helper (if we refactor the existing inline collection logic) or the existing inline logic called twice (if we don't refactor). The implementation plan decides; both are acceptable. The current `_find_nearest_players` body at `contact_detector.py:600` has its collection inlined, so refactoring would extract those ~10 lines into a helper.
- **Merge semantics**: Pass 1 entries take precedence by construction (they're closer to the contact frame). Pass 2 only fills in track_ids missing from Pass 1. This means the existing standard-window result is exactly preserved for every track present in Pass 1; only NEW track_ids are added from the forward window.

### §3 — Forward-only window

`lower_bound_frame=frame` (the contact frame) ensures the second pass looks ONLY at positions at-or-after the contact, never before. Rationale:

- The validated failure mode is "server is tracked AFTER serve". Backward-extension doesn't help.
- Forward-only also prevents capturing players who were briefly present BEFORE the rally started (warmup, walk-ons) and then left.

This is asymmetry by design, justified by data.

### §4 — Why a fallback, not unconditional widening

Two equivalent-ish options were considered:

| Option | Pro | Con |
|---|---|---|
| **Unconditional ±60** | Simpler code | Adds compute on every contact (95% of which already have full 4 candidates); risks "late-arrival" noise on attack/dig contacts |
| **Fallback when underfull** (chosen) | Zero behavior change on the happy path; fires only when needed | Slightly more complex control flow |

The fallback approach makes the change **provably non-regressive** for any contact that already has full candidates — by construction, the fallback doesn't fire, and the function returns the identical result.

### §5 — Env flag for rollback

`ADAPTIVE_PLAYER_SEARCH_WINDOW=0` short-circuits the fallback at line 1, restoring byte-identical pre-v3 behavior. Read at call time, monkeypatch-able from tests. Default `"1"` (ON).

### §6 — Why this composes cleanly with v1 (team-chain) and future v3.x

- v1's `_team_chain_override_allowed` predicate consumes `Contact.player_candidates`. Wider candidates = more potential override targets, but v1's gates still filter strictly. No regression risk.
- Future v3.1 (cross-rally server identity) operates on a different signal entirely (cross-rally pid voting). Independent.
- Future v3.2 (coherence-repair) operates on the SAME `Contact.player_candidates`. Wider candidates = more repair targets. v3.0 strictly enables v3.2.

## Validation

### Pre-ship gates (3 GT videos, A/B in-memory)

- **G-A** Combined `correct_rate` improves by **≥ +2pp** (60.3% → ≥ 62.3%). Modest target reflecting the validated ceiling (3-4 of 8 absent cases recoverable).
- **G-B** Per-fixture `correct` non-regressing (cece ≥ 22, gigi ≥ 35, wawa ≥ 25).
- **G-C** `wrong_unknown_team` non-increasing (0 today — no synthetic-serve attributions changed by this fix).
- **G-D** No new test failures in existing `test_contact_detector.py` / `test_action_classifier.py` suites.
- **G-E** Re-run rank-of-GT diagnostic: of the 4 absent-server cases that have a contact, at least 3 should now have GT in candidates.

### Fleet validation

- Run `reattribute-actions` fleet-wide (~70 videos). Compare per-video re-attribution counts pre/post.
- Re-run `audit-coherence-invariants` fleet-wide. C-2 violation count should NOT increase fleet-wide (this fix doesn't change action attribution per se, only widens the candidate pool — but if any fix-driven swap creates a new C-2 violation we should know).

## Risk

| Risk | Mitigation |
|---|---|
| Adding late-tracked candidates creates false positives on non-serve actions (a player who entered the area 60 frames later being picked as the contact actor) | Fallback fires ONLY when standard window is underfull. Most non-serve contacts have full 4 candidates already → fallback never fires for them. Worst case: the late-tracked candidate appears in the list but ranks far behind the genuine actor (who's tracked at the contact frame) — ranking by distance puts them at rank 3+, where existing attribution logic ignores them. |
| The fallback's ±60 frames is too narrow for cases 4-7 (gaps of 40-104 frames) | Single-stage fallback catches 3 of 4 panel cases with contacts. Multi-stage (60 → 120) catches all 4 but adds complexity. Add as v3.0.1 if first measurement justifies. |
| Env-flag default-ON regresses production on some video class not in the 3 GT panel | Env flag provides instant rollback. `ADAPTIVE_PLAYER_SEARCH_WINDOW=0` restores prior behavior without code revert. Fleet-deploy verification (coherence audit pre/post) gates the ship. |
| `_merge_best_per_track` deduplication has a bug | Unit-tested directly. Construction is straightforward (`dict[int, (best_dist, best_y)]`); not algorithmically risky. |

## Roll-out

1. Land code + tests behind env-flag-ON default.
2. Run A/B harness on 3 GT videos. Validate G-A through G-E.
3. If green: deploy via `reattribute-actions` on 3 GT videos. Capture DB snapshot pre-deploy for rollback. Re-measure post-deploy.
4. If still green: fleet deploy on ~70 videos. Re-run coherence audit fleet-wide.
5. Memory entry summarizing measured impact + commits.

## Non-goals (explicit)

- NOT a rewrite of player tracking. The tracker still misses the server at the serve frame — this fix just looks past that gap when generating candidates.
- NOT a fix for synthetic-serve attribution. Those cases have NO contact and bypass `_find_nearest_players` entirely.
- NOT the final word on same-team errors. v3.1 (cross-rally server identity) addresses the 4 wrong-of-two-teammates cases + the 3-4 absent cases with gaps > 60.

## Open questions for the implementation plan

- Whether to refactor the existing collection logic into a `_collect_best_per_track` helper or inline the second pass. Suggested: refactor (cleaner, easier to test the helper in isolation). The diff is small either way.
- Whether to also propagate the adaptive behavior to other candidate-generation callers (e.g., `_find_player_motion_candidates`, `_find_post_serve_receive_candidate`). v3.0 scope: only the two `_find_nearest_player[s]` functions. Others can be added in a v3.0.1 if needed.
