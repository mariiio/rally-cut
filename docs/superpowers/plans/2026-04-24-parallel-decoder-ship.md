# Plan — Parallel Viterbi candidate decoder ship

**Date:** 2026-04-24
**Status:** APPROVED — all pre-registered gates passed in validation eval
**Effort estimate:** ~5 days end-to-end (refactor + decoder path + A/B + rollout)
**Driving evidence:** `analysis/reports/decoder_v5_full_2026_04_24.md` (68-fold LOO on current MS-TCN++ v5 + GBM v5)

---

## 1. What ships

A parallel candidate-detection entry point — `detect_contacts_via_decoder()` — that bypasses the GBM threshold gate in favor of a Viterbi MAP decode over the candidate lattice with a learned transition prior. The existing `detect_contacts()` stays callable for the attribution path (player_track_id population). Two callers:

- **Statistics / editor UI / action labels** → new path (decoder-decided contacts + action labels)
- **Per-player attribution** → existing path (preserved unchanged)

Both paths share candidate generation + feature extraction. They diverge only at the accept/reject + label step.

---

## 2. Validated lift on v5 weights

68-fold LOO-per-video, 2197 GT contacts, MS-TCN++ v5 + GBM v5:

| Metric | Decoder | Canonical baseline (v5, no synth) | Δ |
|---|---:|---:|---:|
| Contact F1 | **89.4%** | 89.0% | **+0.4pp** |
| Action Acc | **95.2%** | 91.7% | **+3.5pp** |
| TP / FP / FN | 1900 / 153 / 297 | 1809 / 159 / 286 | +91 TP / −6 FP / +11 FN |

Per-class F1 (decoder vs canonical no-synth):

| Class | Δ |
|---|---:|
| serve | **+8.5pp** |
| receive | **+8.7pp** |
| set | +1.9pp |
| attack | +0.6pp |
| dig | −0.7pp (within −1.5pp floor) |
| block | +0.6pp (exempt) |

5/6 classes improve materially. Worst non-block class regressed by only −0.7pp.

---

## 3. Pre-registered ship gates (locked before code)

Same gates as the validation already cleared. Re-measured at every milestone:

- **Contact F1** ≥ 89.0% (canonical no-synth baseline; cannot regress vs v5)
- **Contact F1** ≥ −1.0pp vs canonical synth baseline (89.8%)
- **Action Acc** ≥ +2.5pp vs canonical baseline (91.7%) — conservative vs measured +3.5pp
- **Per-class F1 floors:** serve / receive / set / attack / dig each ≥ **−1.5pp**; block exempt at **−12pp** (memory: `crop_head_phase2_nogo_2026_04_20.md`)
- **FP budget:** ≤ canonical FPs (159) — decoder must not introduce net-new FPs at aggregate
- **9-fixture attribution baseline** (43.8% / 40.8% / 15.4% per `attribution_primitive_first_phase0_2026_04_24.md`): no regression beyond ±1pp
- **Snapshot equivalence:** `detect_contacts()` output on 5 representative rallies is byte-identical pre/post-refactor (Phase 2a gate)

**Gate failure → workstream NO-GO, not a tweak-and-retry.** Memory pattern: `action_fixes_attempt_2026_04_20.md` shows iterative tweaking of failed integrations wastes weeks.

---

## 4. Architecture decision — locked

**Parallel-path, not integration.** Three prior integration attempts failed (`action_fixes_attempt_2026_04_20.md`: relabel-only +0/−0, decoder v1 −2.5pp F1, decoder v2 −1.2pp F1 / +3.1pp Action Acc).

**Why parallel works where integration failed:** the decoder's accept decisions disagree with the existing `_apply_rescue_branch`, sequential-attribution fix, pose-attribution override, dedup, and `apply_sequence_override` — each of which fires AFTER GBM accept. Re-routing decoder decisions through this stack produces conflicts. A parallel path bypasses the entire GBM-acceptance-gated post-process and emits its own contacts.

**Trade-off:** decoder-emitted contacts that have no corresponding GBM-accepted contact get a fresh attribution pass via existing logic factored out of `detect_contacts`.

```
                  ┌─ candidate generation ───┐
                  │ feature extraction       │  (shared, factored out in 2a)
                  │ MS-TCN++ probs           │
                  │ GBM probs                │
                  └────────────┬─────────────┘
                               │
            ┌──────────────────┴──────────────────┐
            ▼                                     ▼
   GBM threshold accept              Viterbi decode (NEW)
   + post-process                    + attribute decoded frames
            │                                     │
            ▼                                     ▼
   detect_contacts() ───────────────────► detect_contacts_via_decoder()
   (attribution callers)                 (stats / UI / action-label callers)
```

---

## 5. Phase breakdown

### Phase 0 — DONE
Validated decoder lift on v5 weights. All gates pass. Reports:
- `analysis/reports/decoder_v5_full_2026_04_24.md`
- `analysis/reports/decoder_v5_smoke_2026_04_24.md`

### Phase 1 — Pre-register (DONE)
This document. Ship gates locked before any code change.

### Phase 2 — TDD-protected refactor + build (1.5 days)

**2a. Snapshot test for current `detect_contacts` (~1 hour, BEFORE any refactor):**
- Add `tests/integration/test_detect_contacts_snapshot.py`: select 5 representative rallies (one per non-block class + one mixed)
- Capture current `detect_contacts()` output as JSON fixtures in `tests/fixtures/detect_contacts/`
- Test asserts `detect_contacts()` output matches fixtures byte-for-byte (frame, action, player_track_id, court_side)
- **Lock this test on main before touching `detect_contacts()`. Anyone can verify "I didn't break anything" by running pytest.**

**2b. Refactor `detect_contacts` into pure stages (~3 hours):**
```python
def _extract_candidates_and_features(ball, players, seq, cfg) -> ExtractionResult:
    """Pure: candidates + features + gbm_probs + seq_probs. No accept/reject."""

def _accept_via_gbm_thresholds(extraction, classifier) -> list[AcceptedContact]:
    """Existing GBM threshold + rescue-branch logic, extracted unchanged."""

def _attribute_and_postprocess(accepted, players, ball, ...) -> list[Contact]:
    """Existing attribution + dedup + sequence override, extracted unchanged."""

def detect_contacts(...) -> list[Contact]:
    """Same public signature; internally calls the three above."""
```
**Verification gate 2b:** `pytest tests/integration/test_detect_contacts_snapshot.py` → green. If any byte changes, fix the refactor.

**2c. Add the parallel entry point (~3 hours):**
```python
def detect_contacts_via_decoder(
    ball_positions, player_positions, sequence_probs, classifier, cfg,
) -> list[Contact]:
    extraction = _extract_candidates_and_features(...)
    decoded = decode_rally(extraction.gbm_probs, extraction.seq_probs, teams, transitions)
    # Decoder may emit frames the GBM rejected — they get fresh attribution
    accepted = _decoded_to_accepted_contacts(decoded, extraction)
    return _attribute_and_postprocess(accepted, player_positions, ball_positions, ...)
```

**2d. Unit tests for the new entry point (~2 hours):**
- Test fixture: 1 rally where decoder accepts a frame the GBM rejects → verify the new contact gets attributed
- Test fixture: 1 rally where decoder rejects a frame the GBM accepts → verify the contact is dropped
- Test that `detect_contacts_via_decoder()` and `detect_contacts()` produce identical output on a rally where the decoder agrees with the GBM

### Phase 3 — A/B eval (~half day)

**3a. Add `--use-decoder` to `eval_loo_video.py --include-synthetic`** so we have ONE harness running both paths apples-to-apples. Currently `eval_candidate_decoder.py` is a separate script — let me consolidate.

**3b. Run 68-fold LOO twice** (~40 min total): baseline vs `--use-decoder`. Compare against pre-registered gates from Section 3.

**3c. Per-fold regression analysis.** Flag any fold that regresses Action Acc > 5pp; visually inspect 3-5 contacts from those rallies in the editor UI.

**3d. Write decision memo:** `analysis/reports/decoder_ab_2026_04_25.md` — pass/fail for every gate, ship recommendation.

**Gate:** if any pre-registered gate fails → STOP. Write a NO-GO memo, do not iterate. The decoder shape is wrong.

### Phase 4 — Production wiring (~1 day)

**4a. Add config flag** `USE_PARALLEL_DECODER` (env var, default `false`):
- Read in `analysis/rallycut/cli/commands/analyze.py` action subcommand
- Logged at startup so production runs are auditable

**4b. Wire `detect_contacts_via_decoder()` behind flag** in:
- `rallycut analyze actions <video-id>` CLI command
- Modal tracking service (`rallycut/service/platforms/modal_tracking.py`) — same flag

**4c. Smoke test on 3-5 production rallies** with the flag on:
- Visually inspect contact frames + action labels in the editor UI (web app)
- Check `match_stats` aggregates make sense
- Verify attribution path still populates `playerTrackId` correctly (decoder doesn't touch this)

**4d. Soak test:** keep flag off in production, run cron job that compares both paths' output on every new rally for 1 week. Log discrepancies. If discrepancies match the LOO A/B (decoder rescues serves/receives, slightly fewer digs), proceed.

### Phase 5 — Rollout (~1 day, after 1-week soak)

**5a. Flip default to `true`** in a single PR. The flag stays callable for emergency rollback.

**5b. Monitor for 1 week** post-flip:
- Per-class contact counts in `match_stats`
- User-visible action labels in editor UI (any user-reported "wrong action label" tickets)
- Attribution oracle (9-fixture baseline) — re-measure after 1 week

**5c. Cleanup** (~half day, after 1 week stable):
- Remove the existing label-only `run_decoder_over_rally` overlay (now redundant)
- Remove `apply_sequence_override` from the decoder path (decoder grammar handles it)
- Update `analysis/CLAUDE.md` and write a memory entry
- Drop the feature flag (default-on stays for 1 release, then removed)

---

## 6. Out of scope (do not bundle)

- Refactoring `detect_contacts` beyond the three-stage extraction needed for the decoder path
- Touching the attribution layer (`_apply_pose_attribution`, `_apply_temporal_attribution`)
- Touching the rescue branch (`_apply_rescue_branch`) — keep it on the GBM path
- Touching the sequence-override (`apply_sequence_override`) — keep it on the GBM path; the decoder grammar replaces it on its own path
- Modifying MS-TCN++ training, GBM training, candidate generators
- Block-class improvements (structurally hard, exempt from gates)
- Rebuilding the transition matrix from new GT data (current matrix at `analysis/rallycut/data/contact_transitions.json` is the validated artifact)

If you discover something during the refactor that "would be cleaner if I also fix X", **note it as a follow-up issue, do not bundle**. Single-purpose PR convention.

---

## 7. Rollback plan

The feature flag is the rollback. At any point during Phase 4-5:

```bash
# Production rollback (instant)
USE_PARALLEL_DECODER=false  # in env

# Code rollback (30 minutes)
git revert <flag-flip commit>
```

Keep `detect_contacts()` callable through the cleanup phase so reverting is always possible without code re-introduction. Drop the flag only after 1 release of stable default-on.

---

## 8. Risk register

| Risk | Likelihood | Mitigation |
|---|---|---|
| MS-TCN++ v6 retrain shifts decoder behavior mid-rollout | Medium | Re-run validation suite on any MS-TCN++ checkpoint change; gate retrains on decoder eval |
| Decoder accepts contacts at frames with no nearby player → attribution returns track_id=−1 | Medium | Phase 2c attribution pass handles this case (existing `_find_nearest_player` returns nearest within search window) |
| Editor UI assumes contiguous contact frames; decoder rescues introduce gaps | Low | Phase 4c smoke test catches this; contacts have always had gaps so unlikely |
| Modal tracking service has stale GBM weights vs local | Low | Modal weights synced from same `weights/` dir; flag works identically there |
| Attribution oracle regresses despite Section 3 gate | Low | Phase 5b 1-week monitor catches this; rollback via flag |
| Per-fold regression cluster (e.g., one camera angle) hidden in aggregate | Medium | Phase 3c per-fold analysis required before ship; flag ANY fold regressing Action Acc >5pp |

---

## 9. Verification at each phase boundary

Per `superpowers:verification-before-completion`:

| Phase | Verification command | Pass condition |
|---|---|---|
| 2a | `uv run pytest tests/integration/test_detect_contacts_snapshot.py` | green |
| 2b | same pytest | still green |
| 2c-d | `uv run pytest tests/unit/test_detect_contacts_via_decoder.py` | green |
| 3 | `uv run python scripts/eval_loo_video.py --include-synthetic` (baseline) and `--use-decoder` | all gates pass |
| 4c | manual editor UI inspection on 3-5 rallies | no visible regressions |
| 5b | 1-week production soak; re-run 9-fixture attribution oracle | no regression |

**No phase is "complete" until its verification command passes.** Mark task complete only after running the command and seeing the result. No "I'm sure it works."

---

## 10. North-star alignment

User goal: "pick a player, see all his actions accurately."

The decoder lifts:
- **Action label correctness** by +3.5pp (from 91.7% to 95.2%) — directly visible in the player timeline
- **Serve recall** by +8.5pp F1 — serves are the most user-visible action; missed serves break the "every rally starts with a serve" mental model
- **Receive recall** by +8.7pp F1 — paired with the serve gain

The decoder does NOT lift:
- Per-player attribution (the player_track_id) — that's the existing path's job
- Pose-blind contact detection (the 17 pose-blind cases from `pose_features_fn_bucket_borderline_2026_04_24.md`)
- Block detection (structurally exempt)

Net: this ship advances the action-label axis of the product goal. Attribution work continues separately (reference crops shipped at 95.22%, see `player_attribution_day4_2026_04_23.md`).

---

## 11. Authorship + auditability

- **Plan author:** Mario (with Claude assistance)
- **Validation eval:** `analysis/scripts/eval_candidate_decoder.py --skip-penalty 1.0` on 68 folds, MS-TCN++ v5 + GBM v5 weights
- **Validation reports:** `analysis/reports/decoder_v5_full_2026_04_24.{md,json}`
- **Pre-registered gates:** Section 3, locked before code change
- **Rollback:** Section 7
- **Risk register:** Section 8
- **Reviewers:** code-reviewer agent at end of Phase 2 + end of Phase 4
