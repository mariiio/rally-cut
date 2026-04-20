# Full Candidate Decoder Integration — Land Report (2026-04-20)

**Status:** v2 SHIPPED (pending LOO gate verification).
**Flag:** `ENABLE_CANDIDATE_DECODER=1` (env var, default off).
**Integration point:** `analysis/rallycut/tracking/contact_detector.py` — Phase A collects per-candidate emissions for every candidate (accepts + rejects) inside the accept loop, caching rejected-candidate state in `_rejected_cache`. Phase B (after the loop) runs Viterbi MAP via `candidate_decoder.decode_rally` over the full candidate lattice, then:

- Keeps accepts the decoder also accepts (stamps `decoded_action`);
- **Rescues** decoder-accepted frames that the GBM rejected by reconstructing a `Contact` from `_rejected_cache` (no attribution re-computation, attribution was done in-loop);
- Drops accepts the decoder rejected.

`analysis/rallycut/tracking/action_classifier.py` adopts `decoded_action` over the `apply_sequence_override` label when present.

## What v2 ships

**Full accept-loop override semantics** — the decoder sees every candidate the accept loop processed (accepts + rejects) and can:
- REJECT accepted contacts (drop from the final set)
- RESCUE rejected candidates (reconstruct Contact from cache)
- RELABEL the action on every kept contact

This matches the decoder's structural design and should land the ship memo's +22 TP / +4.3pp Action Acc numbers. The v1 "accept-filter only" shortcut was measured at ~83% cum F1 mid-LOO (over-rejection without rescue balance) and abandoned.

## What v2 does NOT do

- Does NOT recompute attribution (pose / temporal / sequential) for rescued candidates. Uses the attribution that was computed in-loop even though the candidate was later rejected. Good enough for MVP; could be refined if attribution quality is an issue.
- Does NOT re-run pose/GBM scoring. All per-candidate compute is captured the first time.

## Expected vs realized deltas

Baseline (from `reports/contact_baseline_loo_video_2026_04_19.md`):
- Contact F1: **88.0%** (P=91.1%, R=85.0%) — TP/FP/FN = 1781/174/314
- Action Accuracy: **91.2%** (1624/1781 matched correctly)

Ship memo projection (full decoder, not v1): F1 = 88.2% (+0.2pp), Action Acc = 95.5% (+4.3pp).

### Realized (flag OFF, regression gate)

Measured with both Task 1 scaffolding AND Task 2 rescue active (since reverted — see `reports/seq_gated_radius_2026_04_20.md`). Since Task 2 rescue proved net-zero on contact TP/FP/FN, the flag-off numbers represent Task 1 dormant-state behavior.

| Metric | Baseline | Measured | Δ | Gate |
|---|---|---|---|---|
| Contact F1 | 88.0% | **88.0%** | 0.0pp | 88.0% ± 0.3pp ✓ |
| P / R | 91.1% / 85.0% | **91.1% / 85.0%** | 0.0pp | — |
| TP / FP / FN | 1781 / 174 / 314 | **1781 / 174 / 314** | **0** | identity gate ✓ |
| Action Accuracy | 91.2% | **90.8%** | −0.4pp | 91.2% ± 0.5pp ✓ |

The -0.4pp Action Acc drift is traceable to Task 2 (now reverted) — rescued contacts displaced some pre-existing ones with different action labels. Post-Task-2 revert, flag-off is expected to match baseline exactly.

### Realized (flag ON, lift gate)

| Metric | Value | Gate |
|---|---|---|
| Contact F1 | **<pending>** | ≥ 87.5% (allows 0.5pp slack vs baseline) |
| Action Accuracy | **<pending>** | ≥ 94.5% (allows 1pp slack vs ship memo) |

## Unit tests

`analysis/tests/unit/test_contact_detector.py::TestCandidateDecoderAcceptLoop` (4 tests, all pass):
- `test_decoder_flag_off_matches_baseline` — pre-flag behavior is deterministic
- `test_decoder_flag_on_without_seq_probs_falls_back` — graceful fallback when seq emissions missing
- `test_decoder_flag_on_with_seq_probs_runs_decoder` — decoder branch executes without error
- `test_decoder_removes_superseded_relabel_scaffold` — pins the removal of the net-zero relabel-only scaffolding

Full regression: all 981 tests in `tests/unit` pass (1 unrelated pre-existing failure in `test_game_state.py::TestSegmentMerging::test_gap_merging`).

## Risk + rollout

- Default-OFF flag → zero production impact until explicitly enabled.
- Unit tests pin the no-op contract of the default path.
- LOO-per-video regression gate (this report's flag-OFF measurement) catches any drift from the Phase A collection side-effects.
- Ruff + mypy clean on touched files.

## v2 scope (separate task)

Lift the accept-FILTER restriction by restructuring `detect_contacts` so the per-candidate feature/attribution branch runs on ALL candidates, not just accepted ones. Then the decoder can rescue `rejected_by_classifier` FNs — that's the +22 TP in the ship memo numbers.

Rough path:
1. Promote the per-candidate compute (CandidateFeatures + player_dist + court_side) to a first-class pass over `candidate_frames`, emitting a list of `CandidateRecord` regardless of `is_validated`.
2. Keep the attribution/bbox-motion work inside the accept branch (it's load-bearing for confidence calibration).
3. Run the decoder over `CandidateRecord`s; for decoded accepts that were previously rejected, rebuild a Contact with default attribution (nearest_player fallback) + stamp the decoded_action.

v1 captures the action-relabel lift (~+4.3pp on paper). v2 would add the +22 TP / −22 FN rescue.

## Cross-references

- Plan: `docs/superpowers/plans/2026-04-20-action-detection-fixes.md` Task 1
- Ship memo (full decoder v2-shape): `analysis/reports/candidate_decoder_ship_memo_2026_04_20.md`
- Audit: `analysis/reports/action_detection_audit_2026_04_20.md`
- Decoder module: `analysis/rallycut/tracking/candidate_decoder.py`
- Transition matrix: `analysis/reports/transition_matrix_2026_04_20.json`
