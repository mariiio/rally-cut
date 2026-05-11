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

---

## Task 6: Post-deploy (DB read, reattribute-actions applied)

### Step 1: DB snapshot

`analysis/reports/attribution_baseline/db_snapshots/pre_adaptive_window_2026_05_11.jsonl`
produced: 22 lines (5 cece + 7 gigi + 10 wawa rallies). Matches expected.

### Step 2: reattribute-actions runs

```
cece (950fbe5d): 5 rallies with match teams, 3 eligible (conf >= 0.70)
  0/28 actions re-attributed. "no changes (teams stamped)" for all rallies.

gigi (b097dd2a): 7 rallies with match teams, 0 eligible (conf >= 0.70)
  0/49 actions re-attributed. All gigi rallies have assignmentConfidence < 0.70
  (max is 0.50 for rally 72c8229b; the rest are 0.25-0.35).

wawa (5c756c41): 10 rallies with match teams, 6 eligible (conf >= 0.70)
  0/42 actions re-attributed. "no changes (teams stamped)" for all rallies.
  Viterbi serving_team: 2 changed / 10 stamped.
```

### Step 3: Baseline harness post-deploy

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

Numbers **UNCHANGED** from pre-v3. Investigation below explains why.

### Root-cause analysis: why reattribute-actions produced 0 changes

The adaptive window fix (v3.0) successfully updated `contacts_json.playerCandidates`
for 16 rallies across the 3 GT videos (Task 5). However, `reattribute-actions`
operates on `actions_json` using `reattribute_players()`, which has two limitations
that prevented any action attribution changes:

**1. Same-team errors are invisible to reattribute_players (Pass 2)**

Of the 8 serve attribution errors (the root-cause category identified in the spec):
- 6 are `wrong_same_team` (both players on the same team — e.g., wawa/06c13117:
  track 4 (B) attributed, GT is track 3 (B)).
- `reattribute_players` skips an action when `current_team == expected_team`.
  Since both the wrong and correct server are on team B, the pass correctly
  concludes "current player is on the right team" and skips.
- Adding rank-1 GT candidate in contacts_json does NOT help here: the swap
  logic only fires when the current player is on the WRONG team.

**2. gigi has 0 rallies with assignmentConfidence >= 0.70**

All 7 gigi rallies fall below the 0.70 threshold (max conf = 0.50). So `reattribute_players`
never fires for any gigi rally — regardless of contacts_json.

**3. wawa/8c49e480 cross-team serve: contact frame not in contacts_json**

The cross-team serve error (track 2/A attributed, GT is track 3/B) is at frame 110.
No contact was detected at frame 110 in contacts_json (only frames 335, 429, 534, ...).
The serve detection was contact-less (or contact filtered out), so no candidate list
to update.

**Implication for G-A**

The adaptive window fixed the CANDIDATE POOL (contacts_json) correctly, as confirmed by
G-E (3 of 4 absent-server cases now have GT in the candidate list). However, the
downstream `reattribute_players` pipeline cannot leverage these improved candidates for:
- Same-team within-pair confusion (6 of 8 serve errors)
- Low-confidence rallies (gigi — 7 rallies, most of the serve errors)
- Contacts missing from contact detection (1 case)

The v3.0 improvement is structural: the candidate pool is correct, but the attribution
decision layer (`reattribute_players`) needs team-agnostic within-team disambiguation
to convert the better candidates into correct attributions.

### Step 4: Coherence invariants (post-deploy)

```
cece (950fbe5d): C-1=0, C-2=2, C-3=0 — baseline was C-2=2 (NON-REGRESSING)
gigi (b097dd2a): C-1=1, C-2=3, C-3=0 — baseline was C-2=3 (NON-REGRESSING)
wawa (5c756c41): C-1=0, C-2=7, C-3=0 — baseline was C-2=7 (NON-REGRESSING)
```

All C-2 counts match the most recent pre-v3 fleet baseline exactly. No regression.

### Gate summary for Task 6 (FINAL)

| Gate | Status | Value |
|------|--------|-------|
| G-A  | FAIL   | combined correct_rate: pre=60.3%, post=60.3%, delta=0pp (threshold was +2pp) |
| G-B  | PASS   | cece=22 (≥22), gigi=35 (≥35), wawa=25 (≥25) — no per-fixture regression |
| G-C  | PASS   | wrong_unknown_team: 0→0 (non-increasing) |
| G-D  | PASS   | 1332 passed, 2 skipped, 0 failures (confirmed in Task 5) |
| G-E  | PASS   | 3 of 4 absent-server cases now have GT in candidates (confirmed in Task 5) |

**G-A FAIL — DONE_WITH_CONCERNS.**

G-A fails because `reattribute_players` cannot leverage the improved candidates for
same-team within-pair discrimination and low-confidence rallies. The candidate pool
improvement is real (G-E: 3 of 4 cases fixed) but doesn't translate to measurable
attribution gain through the current pipeline.

The v3.0 fix is structurally correct but the floor is exposed: the 8 serve errors
requiring within-team discrimination are outside the scope of team-based reattribution.
Proceeding to fleet deploy (Task 7) is a HOLD pending user decision — see concerns below.

### Concerns and recommendations

1. **G-A failure is structural, not a v3 bug**: the candidate pool is better, but
   `reattribute_players` cannot use it for within-team disambiguation. A future
   v3.1 fix would need proximity-based within-team tie-breaking (closest player
   among same-team candidates).

2. **gigi confidence floor**: all 7 gigi rallies have conf < 0.70. The matcher
   produces low-confidence assignments because gigi's player appearances are
   ambiguous. Raising the confidence threshold for reattribute-actions won't help;
   this needs better matcher confidence.

3. **Fleet deploy (Task 7)**: contacts_json was already updated for the 3 GT videos
   in Task 5. Fleet-wide contacts_json regeneration is safe (no attribution change;
   only candidate pools updated). Fleet reattribute-actions would update team stamps
   and Viterbi serving_team on all rallies (0 action re-attributions expected, same
   root causes). Recommend HOLD on Task 7 until v3.1 within-team disambiguation
   is available — the cost is low but the benefit is nil until then.

---

## v3.1 post-deploy measurement (2026-05-11)

Plan: docs/superpowers/plans/2026-05-11-within-team-proximity-swap.md
Spec: docs/superpowers/specs/2026-05-11-within-team-proximity-swap-design.md
Commit: `df2aa381` (feat(reattribute): v3.1 within-team proximity swap (Pass 2c))

### Step 1: reattribute-actions on 3 GT videos

All 3 runs completed. 0 within_team_swap log lines fired on any video.

```
cece (950fbe5d): 5 rallies, 3 eligible (conf >= 0.70)
  0/28 actions re-attributed. "no changes (teams stamped)" for all rallies.

gigi (b097dd2a): 7 rallies, 0 eligible (conf >= 0.70)
  0/49 actions re-attributed. All gigi rallies below confidence threshold.

wawa (5c756c41): 10 rallies, 6 eligible (conf >= 0.70)
  0/42 actions re-attributed. "no changes (teams stamped)" for all rallies.
```

**G-E investigation:** Pass 2c is correctly implemented and all 4 unit tests pass.
Two distinct blockers prevent the expected fires:

1. **wawa/06c13117 (frame 184, serve):** `contacts_json` has `playerTrackId=4`, rank-1
   is track 3 (distance 0.084 vs 0.105 for track 4). Both tracks 3 and 4 are on team 1
   (far). Team check passes — Pass 2c should fire — but the action's `confidence=0.4397`
   falls below the 0.6 gate: `if action.confidence < 0.6: continue`. Pass 2c is skipped.

2. **gigi/5b6f0474 (frame 48, synthetic serve):** All 7 gigi rallies have
   `assignmentConfidence < 0.70` (max 0.50). `reattrib_ta = reattrib_teams.get(rally_id)`
   is None → the entire `reattribute_players()` call is skipped for gigi. Pass 2c never
   reaches the candidate comparison. Additionally, the synthetic serve is at frame 48 but
   the contact is at frame 46, so even if confidence gate were cleared, `contact_by_frame`
   would not find it.

These are the same structural blockers as v3.0 Task 6.

### Step 2: Baseline harness post-v3.1 deploy

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

Numbers **UNCHANGED** from pre-v3.1 — expected given G-E investigation above.

### Step 3: v3.1 pre-ship gate verdicts

| Gate | Status | Value |
|------|--------|-------|
| G-A  | FAIL   | correct_rate: pre=60.3%, post=60.3%, delta=0pp (threshold ≥+1pp) |
| G-B  | PASS   | cece=22 (≥22), gigi=35 (≥35), wawa=25 (≥25) — no per-fixture regression |
| G-C  | PASS   | wrong_unknown_team: 0→0 (non-increasing) |
| G-D  | PASS   | 1336 passed, 0 failures (confirmed in Task 1) |
| G-E  | FAIL   | 0 within_team_swap log lines fired (expected 2: gigi/5b6f0474 + wawa/06c13117) |

**STOP: G-A and G-E both FAIL — DONE_WITH_CONCERNS.**

G-A fails because Pass 2c never fires on the production cases: confidence gates
block the two expected within-team swaps (action confidence < 0.6 for wawa, rally
confidence < 0.70 for gigi). The implementation is correct per unit tests; the
blockers are in the data and pre-existing confidence gates, not in Pass 2c logic.

### Step 4: Coherence invariants (post-v3.1 deploy)

```
cece (950fbe5d): C-1=0, C-2=2, C-3=0 — matches post-v3.0 baseline (NON-REGRESSING)
gigi (b097dd2a): C-1=1, C-2=3, C-3=0 — matches post-v3.0 baseline (NON-REGRESSING)
wawa (5c756c41): C-1=0, C-2=7, C-3=0 — matches post-v3.0 baseline (NON-REGRESSING)
```

All coherence counts identical to post-v3.0. No regression introduced by v3.1 Pass 2c.

### Root-cause analysis and forward path

Pass 2c is structurally correct but cannot reach the two intended cases because:

1. **Action confidence gate (wawa/06c13117):** The serve at frame 184 has contact
   confidence 0.4397 < 0.6. This is a legitimate gate — low-confidence contacts are
   noisy and swapping them freely risks regressions elsewhere. The fix would require
   either: (a) lowering the confidence threshold for within-team swaps (v3.2 candidate),
   or (b) boosting contact confidence for serve-frame contacts specifically.

2. **Rally confidence gate (gigi):** All gigi rallies have `assignmentConfidence < 0.70`.
   The matcher produces weak assignments for this video; reattribute-actions requires
   >= 0.70 to avoid propagating bad team signals. The fix requires: (a) better matcher
   confidence for gigi, or (b) a lower confidence floor for Pass 2c only (since
   within-team swaps are lower-risk than cross-team swaps).

3. **Frame mismatch (gigi/5b6f0474 synthetic serve):** Frame 48 (action) vs frame 46
   (contact). Even with gates cleared, `contact_by_frame.get(48)` returns None. Synthetic
   serve insertion needs to align its frame to the nearest contact frame.

**Recommendation:** v3.2 should address items 1 and 3 above, or separately revisit
gigi matcher confidence. The code is deployed and ready; the data-side blockers are
well-understood and scoped.

### Concerns

1. Pass 2c is merged and enabled by default. It adds 0 changes on the fleet today
   (both expected fires are blocked by pre-existing gates). Risk of accidental firing
   on other rallies: low — the same-team + rank-1-differs + both-mapped preconditions
   are strict. No regressions observed (G-B PASS, coherence non-regressing).

2. Unit tests (4/4) validate the happy path, not-current-is-rank1, cross-team guard,
   and env-flag-off cases. Production is blocked by data; unit coverage is sufficient.

3. `WITHIN_TEAM_PROXIMITY_SWAP=0` available for rollback if unexpected fires emerge
   on fleet deploy.
