# Action Detection Architecture — Three-Pattern Investigation (2026-04-17)

## Context

After Phase 1 GT-integrity repair (see `gt_integrity_orphaned_rallies_2026_04_17.md` + `action_corpus_delta_pre_vs_post_gt_repair.md`), the clean post-repair corpus is 1351 errors over 364 rallies (254 clean / 110 gt_orphaned): 589 FN_contact, 293 wrong_action, 469 wrong_player.

User visual review flagged three fixable patterns. Tracing them against the post-repair corpus reframes the architectural story.

**Headline finding**: the corpus builder (`build_action_error_corpus.py:332-335`) calls `classify_rally_actions` without `sequence_probs`. That single wiring gap disables *both* (a) the in-detector two-signal rescue gate at `contact_detector.py:2135-2144` and (b) the post-classification MS-TCN++ override via `apply_sequence_override`. Patterns 1 and 2 share this root cause. Pattern 3 (attribution) is mostly resolved by GT repair; one genuine attribution bug remains from the user's flagged set.

---

## Pattern 1 — "ball-occluded" FN_contacts (actually near-threshold classifier rejections)

### Traced user-flagged examples (post-repair)

| Rally | Frame | User tag | Post-repair state |
|---|---|---|---|
| `37e14e1e` | 216 | ball_occluded_fixable | **wrong_action** now (pred=set, gt=receive), conf=0.375, ball_gap=0 |
| `37e14e1e` | 396 | looks_fixable | FN `rejected_by_classifier`, conf=0.161, ball_gap=0 |
| `fb7f9c23` | 230 | ball_occluded_fixable | FN `rejected_by_classifier`, conf=0.092, ball_gap=0 |
| `fb7f9c23` | 302 | looks_fixable | FN `rejected_by_classifier`, conf=0.241, ball_gap=0 |
| `99a01ce4` | 371 | ball_occluded_fixable | FN `rejected_by_classifier`, conf=0.126, ball_gap=0 |
| `99a01ce4` | 813 | ball_occluded_fixable | FN `no_player_nearby`, conf=0.162, **ball_gap=2** |
| `71c5d769` | 234 | ball_occluded_fixable | FN `rejected_by_classifier`, conf=0.172, ball_gap=0 |

Of 5 user-flagged "ball-occluded" cases, only **one** has an actual ball gap at the GT frame (`99a01ce4:813`, gap=2). The rest have the ball detected at the GT frame but the GBM's confidence is below the 0.35 classifier gate. User's visual impression of "brief occlusion" was the player's hand approaching a visible ball — not a detection dropout.

### `rejected_by_classifier` bucket shape (n=268)

| Statistic | Value |
|---|---|
| `ball_gap == 0` | 249 (93%) |
| `ball_gap ∈ [1,3]` | 17 (6%) |
| `ball_gap ∈ [4,8]` | 2 |
| `ball_gap ≥ 9` | 0 |
| conf p25 / median / p75 | 0.136 / 0.205 / 0.263 |
| near-threshold (conf > 0.2 AND player present) | 142 (53%) |
| GT action: attack | 124 (46%) |
| GT action: set / receive / dig | 41 / 40 / 38 |

### Root cause

At `contact_detector.py:2135-2144`, a two-signal rescue gate already exists:

```python
if (not is_validated
    and seq_peak_nonbg is not None
    and _has_sequence_support(frame)):
    if confidence >= SEQ_RECOVERY_CLF_FLOOR:
        is_validated = True
```

This rescues `rejected_by_classifier` FN_contacts when:
1. Classifier confidence ≥ `SEQ_RECOVERY_CLF_FLOOR` (~0.10-0.15), AND
2. MS-TCN++ has a non-background peak ≥ `SEQ_RECOVERY_TAU` within ±5 frames.

Both conditions require `sequence_probs` to be passed into `detect_contacts(...)`. In production-CLI paths (`track_player.py:925-989`), `sequence_probs = get_sequence_probs(...)` is computed once per rally and threaded through. In the corpus/eval path (`build_action_error_corpus.py:321-330`), it's not computed or passed. The rescue gate never fires.

### Recommended architecture

**Single change**: make the corpus builder (and any caller) follow the production pattern. Compute `sequence_probs` from `rallycut.tracking.sequence_action_runtime.get_sequence_probs` once per rally, pass to `detect_contacts`. The existing two-signal rescue gate handles the rest — no new features, no new thresholds, no new generators.

Why this is complete (not a partial):
- The rescue gate uses the classifier's *existing* 25-dim GBM output + MS-TCN++ agreement as the context signal. That's the "context-dependent threshold" architectural pattern the plan called for; it's already implemented and production-proven on CLI paths.
- Ball-gap-aware rescue is *not* required. The real problem is 142 near-threshold rejections (conf ∈ [0.2, 0.35]), and MS-TCN++ co-endorsement is the proven discriminator for those per `memory/fn_sequence_signal_2026_04.md`.
- `99a01ce4:813` (the one true gap case) is caught if MS-TCN++ endorses the frame — this case has a player near enough that `no_player_nearby` reflects the hand-tuned fallback gate, not the classifier branch. Wiring `sequence_probs` routes it through the rescue branch instead.

### Expected lift

Of 268 `rejected_by_classifier` FNs:
- ~181 are expected to flip to TP per `memory/fn_sequence_signal_2026_04.md` ("The Session 3 pool showed MS-TCN++ peak ≥ 0.80 within ±5 frames for 181/268 rejected FNs.").
- Remaining 87 lack either MS-TCN++ endorsement or the confidence floor.

Net: FN_contact 589 → ~408 (−181), action_acc recovers ~3-4pp.

### Critical files to modify

- `analysis/scripts/build_action_error_corpus.py:301-330` — compute `sequence_probs = get_sequence_probs(ball_positions, player_positions, court_split_y, frame_count, match_teams, calibrator=None)` once per rally, pass to `detect_contacts(sequence_probs=sequence_probs, ...)`.
- `analysis/rallycut/tracking/sequence_action_runtime.py:88-133` — `get_sequence_probs` (no change; import only).

No changes to `contact_detector.py` or `contact_classifier.py`. This pattern is a wiring fix, not an architectural change.

---

## Pattern 2 — Action-type confusions fixable by MS-TCN++ override

### Traced user-flagged examples (post-repair)

| Rally | Frame | User tag | GT / pred | conf | Rally state |
|---|---|---|---|---|---|
| `37e14e1e` | 305 | sequence_would_help | set / attack | 0.406 | clean |
| `4ea1bfa2` | 199 | sequence_would_help | set / receive | 0.900 | gt_orphaned |
| `ab1fbbaa` | 174 | sequence_would_help | set / receive | 0.700 | clean |
| `55c2c6e5` | 353 | sequence_would_help | dig / receive | 0.673 | gt_orphaned |

Observations:
- 2 of 4 rallies are `gt_orphaned` (can't be rescued by any fix — but the action-type comparison is still valid).
- All 4 have the ball detected (ball_gap=0). Bug is purely action-type classification.
- GBM confidence ∈ [0.406, 0.900] — classifier is often *confident and wrong* on these.

### Top set-adjacent confusions (full post-repair corpus)

| Confusion | Count |
|---|---|
| set→receive | 52 |
| dig→attack | 33 |
| attack→set | 30 |
| dig→set | 30 |
| set→attack | 21 |
| attack→receive | 20 |
| attack→dig | 16 |
| receive→set | 13 |

set-adjacent cluster: 215 contacts.

### Root cause

`classify_rally_actions` (`action_classifier.py:3361-3523`) runs the GBM classifier followed by Viterbi smoothing and per-contact heuristics. It does **not** call `apply_sequence_override`. The override lives in `sequence_action_runtime.py:201-303` and is only invoked from CLI paths (`track_player.py:1001`, `analyze.py:141-146`). The corpus/eval path bypasses it.

`apply_sequence_override` has three guards (`OVERRIDE_RELATIVE_CONF_K=1.2`, `ATTACK_PRESERVE_RATIO=2.5`, `DIG_GUARD_RATIO=2.5`) tuned on the 1668-contact probe set to deliver +1.68pp action_acc without receive-F1 regression. These guards assume `action.confidence` reflects the GBM's top-1 — i.e., they only ever need the override-side inputs, no pipeline changes.

### Recommended architecture

**Add a `sequence_probs` parameter to `classify_rally_actions` and call `apply_sequence_override` after Viterbi.** Pattern 1 already threads `sequence_probs` through `detect_contacts`; Pattern 2 threads the same array through `classify_rally_actions` so one rally-level compute serves both effects.

```python
def classify_rally_actions(
    contact_sequence: ContactSequence,
    ...,
    sequence_probs: np.ndarray | None = None,  # new
) -> RallyActions:
    ...
    result.actions = viterbi_decode_actions(result.actions)
    result.actions = validate_action_sequence(result.actions, rally_id)
    if sequence_probs is not None:
        from rallycut.tracking.sequence_action_runtime import apply_sequence_override
        apply_sequence_override(result, sequence_probs)
    # ... remainder unchanged
```

CLI callers that currently do `classify_rally_actions(...)` followed by `apply_sequence_override(rally_actions, sequence_probs)` collapse into one call. Double-application is idempotent (argmax is deterministic), so migration is safe — remove the caller-side `apply_sequence_override` at `track_player.py:1001`, `analyze.py:141-146`, `analyze.py:308-313` in the same PR.

### Expected lift

`sequence_action_classifier.md` documents +4.4pp action_acc from the override. In terms of the corpus:
- Best case: 215 set-adjacent confusions → −60 to −90 (ratio-gate guards protect the GBM's correct calls).
- Worst case: +2-3pp action_acc if the override runs into GBM's high-confidence-but-wrong predictions (confidence ≥ 0.7 × 1.2 = 0.84 blocks the override on 2 of 4 flagged sequence cases).

### Critical files to modify

- `analysis/rallycut/tracking/action_classifier.py:3361-3523` — add `sequence_probs` parameter and the override call after `viterbi_decode_actions` at line 3464.
- `analysis/rallycut/cli/commands/track_player.py:991-1001` — pass `sequence_probs` into `classify_rally_actions` (already computed at line 925), remove the now-duplicate caller-side call.
- `analysis/rallycut/cli/commands/analyze.py:141-146`, `analyze.py:308-313` — same.
- `analysis/scripts/build_action_error_corpus.py:332-335` — already getting `sequence_probs` per Pattern 1; also thread it into `classify_rally_actions` here.

---

## Pattern 3 — Attribution

### Traced user-flagged examples (post-repair)

| Rally | Frame | User tag | Post-repair | Notes |
|---|---|---|---|---|
| `99a01ce4` | 688 | clearly_correct_pred | **wrong_player** (GT=1, pred=3) | Genuine attribution error on a clean rally. |
| `fb7f9c23` | 397 | obvious_mistake | wrong_action (pred=receive, gt=attack) | Reclassified by Phase 1; not an attribution bug. |
| `ac84c527` | 155 | obvious_mistake | wrong_action (pred=set, gt=receive) | Same — action-type error, not attribution. |
| `5e8af221` | 157 | serve attribution differs | Serve on orphan rally (see note) | Formation-based serve path is by design. |
| `0102cbba` | 108 | serve attribution differs | Same | Same. |

Observations:
- The "clearly_correct_pred" case was a dashboard artifact. The dashboard rendered canonical IDs as `T1/T3`, hiding a raw-ID pred-vs-GT mismatch. Post-repair, it surfaces as a real wrong_player (GT=1, pred=3).
- Two "obvious_mistake" attribution flags are actually action-type errors post-repair — Pattern 2 covers them.
- Serve-path cases: `action_classifier.py:2086-2212` uses formation-based server detection (`_find_server_by_position` 308-367). Documented architectural choice; not a bug.

### Root cause for the one genuine attribution bug (`99a01ce4:688`)

`_find_nearest_player` at `contact_detector.py:488-519` picks the closest player to `(ball_x, ball_y)` using wrist keypoint (fallback: bbox upper-quarter). It operates on the raw ball position at the contact frame. When the ball position is imprecise (filter-interpolated or a low-confidence detection), the "nearest" pick can flip to the wrong teammate.

Follow-on trace needed: load `99a01ce4` raw data, measure the actual player distances at frame 688, check whether filtered vs raw ball position would pick player 1.

### Recommended architecture

**No blanket fix proposed.** The attribution surface is small (1 genuine bug in the user's flagged set) once Phase 1 and Pattern 2 are applied. Two refinements to consider in follow-up plans:

1. **Pass the filtered ball position** to `_find_nearest_player`. Currently the contact uses raw `(ball_x, ball_y)` at the frame; the ball filter (`ball_filter.py`) has an interpolated signal that's more stable across brief dropouts. Low risk — single parameter threading; easy to A/B.
2. **Investigate the broader `wrong_player` bucket (469 cases)** separately. Top subdivisions: attribution on attacks (hard — small bbox, high velocity, many near-neighbors), on sets (usually off by one teammate). A dedicated attribution workstream is warranted but out of scope here.

Serve-path attribution is by-design and correct at 72-76% per `action_classifier.py:230-245`.

### Expected lift

Pattern 3 as scoped here is measurement-only — confirm post-Phase-1-and-2 that `99a01ce4:688` is the only remaining bug in the flagged set. A follow-up plan can size the full wrong_player bucket.

---

## Implementation order and gating

Both Patterns 1 and 2 reduce to *one unified wiring change*: thread `sequence_probs` through the eval/corpus path. Pattern 2 adds a parameter to `classify_rally_actions` + one new call site; Pattern 1 adds one new call site only (the corpus builder's `detect_contacts` call already accepts the parameter).

Phase 3 of the plan (`/Users/mario/.claude/plans/ultrathink-we-ran-a-floofy-porcupine.md`) implements both in one feature commit:

1. Add `sequence_probs` parameter to `classify_rally_actions` + conditional `apply_sequence_override` call.
2. In `build_action_error_corpus.py`, compute `sequence_probs` per rally and pass it to both `detect_contacts` and `classify_rally_actions`.
3. Remove the now-duplicate caller-side `apply_sequence_override` calls in `track_player.py`, `analyze.py` (idempotent — safe to migrate).
4. Unit test: fixture rally with known GBM outputs; assert `use_sequence=True` vs explicit `sequence_probs=None` differ in expected ways; assert idempotency.
5. Regression: re-run `build_action_error_corpus.py`; confirm FN_contact drops by ~150-180 and set-adjacent confusions drop by ~40-80.

Pattern 3 follow-up is a separate plan — no code changes this round beyond confirming the user's flagged set resolves as predicted.
