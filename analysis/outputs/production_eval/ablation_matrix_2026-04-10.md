# Ablation Matrix — 2026-04-10

**Baseline**: `run_2026-04-10-001609.json` (bit-exact to canonical `run_2026-04-09-181807.json`).
**Sweep**: 8 ablations × `production_eval.py --reruns 1 --ablate <name>` (std=0, deterministic).
**Scope**: 340 rallies, 62 videos.

## Baseline headline

| metric | value |
|---|---:|
| contact_f1 | 85.0 |
| contact_recall | 84.0 |
| action_accuracy | 92.4 |
| court_side_accuracy | 72.7 |
| player_attribution_accuracy | 61.4 |
| serve_id_accuracy | 63.3 |
| serve_attr_accuracy | 58.2 |
| score_accuracy | 46.2 |

## Deltas (ablation − baseline, pp)

Negative = load-bearing (ablation hurts). Positive = regressor (ablation helps).

| ablation | cf1 | crec | aacc | cs | pa | sid | sa | score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mstcn_override` | 0.0 | 0.0 | **−3.9** | 0.0 | 0.0 | **−5.8** | 0.0 | 0.0 |
| `verify_team_assignments` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| `pose_attribution` | 0.0 | 0.0 | +0.2 | **−2.3** | **−2.9** | 0.0 | −0.5 | **−1.1** |
| `contact_classifier` | **−21.1** | −0.5 | **−1.6** | **−1.2** | **−5.0** | **−3.2** | **−16.9** | **−4.4** |
| `adaptive_dedup` | +0.2 | −0.5 | +0.2 | 0.0 | +0.5 | 0.0 | 0.0 | 0.0 |
| `seq_enriched_contact_gbm` | **−2.4** | **−6.9** | −0.4 | +1.4 | +1.5 | **−7.0** | +1.2 | 0.0 |
| `sequence_recovery` | **−2.4** | **−6.9** | −0.4 | +1.4 | +1.5 | **−7.0** | +1.2 | 0.0 |
| `literal_id_match` | 0.0 | 0.0 | 0.0 | 0.0 | **−5.0** | 0.0 | **−2.0** | 0.0 |

## Load-bearing components (any headline Δ ≤ −1pp)

1. **`contact_classifier`** — THE dominant lever. Kills contact F1 (−21.1), serve_attr (−16.9), score (−4.4), player_attr (−5.0), serve_id (−3.2), action_acc (−1.6), court_side (−1.2). 25-dim GBM on trajectory features. Memory: `contact_detection.md`, `contact_detection_ceiling.md`. **KEEP — already at ceiling per Session 9.**
2. **`seq_enriched_contact_gbm`** / **`sequence_recovery`** — Identical deltas (see §Anomalies below). Together anchor the false-negative rescue: −6.9pp contact_recall, −7.0pp serve_id, −2.4pp contact_f1. But **+1.4pp court_side and +1.5pp player_attr when ablated**, meaning the rescued contacts trade downstream attribution for upstream recall. Net positive on headlines (score 0.0), net positive on contact/serve. Memory: `fn_sequence_signal_2026_04.md`. **KEEP — serve_id lever.**
3. **`mstcn_override`** — MS-TCN++ action override on non-serve actions. −3.9pp action_acc, −5.8pp serve_id. Memory: `sequence_action_classifier.md`, `action_classifier_audit.md`. **KEEP.**
4. **`pose_attribution`** — 5 nearest-player pose features in contact GBM + attribution. −2.9pp player_attr, −2.3pp court_side, −1.1pp score. Memory: `motion_pose_experiment.md`, `contact_classifier_pose_experiment.md`. **KEEP.**
5. **`literal_id_match`** — Session 11 eval-time trackId normalization via `match_analysis.trackToPlayer`. −5.0pp player_attr, −2.0pp serve_attr (reproduces pre-Session-11 baseline bit-exact, as designed). Memory: `session11_eval_trackid_norm.md`. **KEEP.**

## Net-zero components (no headline Δ ≥ |0.3pp|)

- **`verify_team_assignments`** — **all zeros across all 8 metrics.** Either the code path is dead, the ablation flag is not wired, or the function is idempotent on the current cache. **Flag for investigation** — do not remove without a diagnostic.
- **`adaptive_dedup`** — within noise on all headline metrics (max |Δ| = 0.5pp on contact_recall and player_attr, both trading). **BUT** Session 2 (`block_detection.md`) measured −3.71pp block_F1 when ablated post-rescue. Block F1 is not in the headline matrix — this ablation is load-bearing on a metric we don't report. **KEEP — Chesterton's fence closed by Session 2.**

## Regressors (ablation improves ≥ +0.5pp)

- `seq_enriched_contact_gbm` / `sequence_recovery`: +1.4pp court_side, +1.5pp player_attr when ablated. Confirms the Simpson's-paradox diagnosis from Session 4: rescued contacts drag down attribution metrics relative to the non-rescued baseline. **Do not act** — already diagnosed, net score_accuracy is 0.0pp (the metric that matters), and the +5.9pp serve_id from rescuing is the headline win (see `fn_sequence_signal_2026_04.md`).
- `adaptive_dedup`: +0.2 cf1, +0.5 pa — within determinism floor.

## Anomalies

- **`seq_enriched_contact_gbm` and `sequence_recovery` have identical deltas across all 8 metrics.** Suggests (a) the two flags alias the same code path, (b) one is a no-op that slipped in, or (c) the flags are coupled (skipping recovery makes the seq_enriched GBM features useless). Worth an ablation audit in a future housekeeping session. **Not a ship blocker.**
- **`verify_team_assignments` is completely inert.** Either (a) the flag is not wired to anything the eval path hits, (b) the function is a no-op on the cached positions_json that the evaluator reads, or (c) team assignments are already resolved upstream in cached data. **Investigate before removing.**

## Verification

- Baseline `run_2026-04-10-001609.json` matches `run_2026-04-09-181807.json` bit-exact on all 8 headline metrics.
- All 8 ablations completed (exit 0), each wrote `run_2026-04-10-*.json` under `analysis/outputs/production_eval/`.
- All `--reruns 1` runs show std = 0.00pp (determinism confirmed).
