# Phase 3 Evaluation — Sequence Wiring in `classify_rally_actions` (2026-04-17)

## Change summary

`classify_rally_actions` now accepts a `sequence_probs: np.ndarray | None = None` parameter and invokes `apply_sequence_override` internally when non-None. Callers collapse from two-call (`classify_rally_actions(...)` + `apply_sequence_override(...)`) to one.

Touched files:
- `rallycut/tracking/action_classifier.py` — parameter + override call at end of `classify_rally_actions`
- `rallycut/cli/commands/track_player.py` — pass `sequence_probs`, drop caller-side override
- `rallycut/cli/commands/analyze.py` — same in both `actions` and `highlights` commands, also thread into `detect_contacts`
- `scripts/build_action_error_corpus.py` — compute `sequence_probs` per rally, pass to both `detect_contacts` and `classify_rally_actions`

## Corpus totals (364 rallies / 2098 GT contacts)

| Metric | No-seq baseline | Seq-wired | Δ | Δ% |
|---|---:|---:|---:|---:|
| Total errors | 1351 | 1142 | **−209** | −15% |
| True positives | 747 | 956 | +209 | +28% |
| FN_contact | 589 | 395 | **−194** | −33% |
| wrong_action | 293 | 170 | **−123** | −42% |
| wrong_player | 469 | 577 | +108 | +23% |

Contact recall (TP / (TP + FN)): 55.9% → 70.8% (**+14.9pp**).

Evaluable action accuracy (TP + wrong_player) / (TP + wrong_player + wrong_action): 80.6% → 90.0% (**+9.4pp**).

### Why wrong_player rose

Rescued FN_contacts now face attribution evaluation. The 108 new wrong_player errors are all contacts that were previously invisible to the attribution metric. The ceiling for Pattern 3 investigation (out of scope for this plan) is now visible at 577 wrong_player vs the 293 true action errors.

## FN subcategory breakdown

| Subcategory | No-seq | Seq-wired | Δ |
|---|---:|---:|---:|
| rejected_by_classifier | 268 | **139** | −129 |
| rejected_by_gates | 188 | 147 | −41 |
| no_player_nearby | 71 | 56 | −15 |
| deduplicated | 25 | 20 | −5 |
| no_candidate | 19 | 19 | 0 |
| ball_dropout | 18 | 14 | −4 |

The `rejected_by_classifier` bucket — Pattern 1's target — dropped by **48%**. `memory/fn_sequence_signal_2026_04.md` had estimated 181 rescuable; realized 129. The delta (52) reflects tougher SEQ_RECOVERY_CLF_FLOOR cases + post-GT-repair drift.

## Action misclassifications — set-adjacent cluster

| Confusion | No-seq | Seq-wired | Δ |
|---|---:|---:|---:|
| set→receive | 52 | 10 | −42 |
| dig→attack | 33 | 6 | −27 |
| attack→set | 30 | 20 | −10 |
| dig→set | 30 | 23 | −7 |
| set→attack | 21 | 5 | −16 |
| attack→receive | 20 | 3 | −17 |
| attack→dig | 16 | 17 | +1 |
| receive→set | 13 | 4 | −9 |
| **Subtotal** | **215** | **88** | **−127 (−59%)** |

Design-doc estimate was "−60 to −90"; actual −127 exceeds it.

## User-flagged case resolution (14 cases from review_feedback.csv)

**Fixed** (8/14):
- `37e14e1e:305` set→attack GBM → now correct (sequence_would_help ✓)
- `37e14e1e:396` FN_contact → TP (looks_fixable ✓)
- `fb7f9c23:302` FN_contact → TP (looks_fixable ✓)
- `fb7f9c23:397` wrong_action → TP (obvious_mistake ✓)
- `99a01ce4:688` wrong_player → TP (clearly_correct_pred ✓ — now genuinely correct)
- `ab1fbbaa:174` set→receive → TP (sequence_would_help ✓)
- `55c2c6e5:353` dig→receive → TP (sequence_would_help ✓)
- `ac84c527:155` receive→set → TP (obvious_mistake ✓)

**Still broken** (6/14), expected and documented:
- `37e14e1e:216` shifted from wrong_action to wrong_player — sequence fixed the action type but attribution is wrong; Pattern 3 territory
- `fb7f9c23:230` FN (conf=0.092, below SEQ_RECOVERY_CLF_FLOOR)
- `99a01ce4:371` FN (conf=0.126, same)
- `99a01ce4:813` FN (conf=0.162, `no_player_nearby` — not a classifier rescue case)
- `71c5d769:234` FN (conf=0.172, below floor)
- `4ea1bfa2:199` wrong_action pred=receive, GBM conf=0.900 — OVERRIDE_RELATIVE_CONF_K=1.2 gate blocks (0.900 × 1.2 = 1.08 is unreachable for any argmax ≤ 1.0). Matches the design-doc "high-confidence wrong" caveat.

## Test coverage

Added `TestSequenceProbsWiring` to `tests/unit/test_action_classifier.py`:
- `test_override_wired_when_sequence_probs_provided` — confirms wiring (relaxes `OVERRIDE_RELATIVE_CONF_K` via monkeypatch because the rule-based classifier emits confidence=0.9 and the production gate is unreachable at argmax ≤ 1.0 — the point of the test is to verify threading, not to duplicate `test_sequence_action_runtime.py`'s guard tests).
- `test_sequence_probs_none_leaves_result_untouched` — backward-compat guard.
- `test_override_never_produces_serve` — serve-exemption regression guard.

Full regression: `uv run pytest tests/unit/test_action_classifier.py tests/unit/test_sequence_action_runtime.py tests/unit/test_contact_detector.py` → **236 passed in 1.81s**.

Static checks: `uv run ruff check rallycut/tracking/action_classifier.py rallycut/cli/commands/track_player.py rallycut/cli/commands/analyze.py` + `uv run mypy` — all pass on touched files. Pre-existing lint in corpus-builder and diagnostic scripts was left alone (not introduced by this change).

## Spatial orphan repair (bonus, not in original plan)

After Phase 3 measurement, the user asked to repair orphan rallies where possible. `scripts/repair_orphaned_gt.py` uses GT's `ballX`/`ballY` + closest-point-on-bbox distance + ±5-frame temporal aggregation to find the track closest to the ball at the GT frame, then remaps the orphan label's `playerTrackId` when distance ≤ 0.10 (norm-coords) AND second-nearest is ≥ 0.03 farther.

Applied 100 confident repairs across 63 rallies. Impact:

| Metric | Pre-spatial-repair | Post-spatial-repair | Δ |
|---|---:|---:|---:|
| Clean rallies | 254 | 263 | +9 |
| Orphaned rallies | 110 | 101 | −9 |
| Corpus TP | 956 | 938 | −18 |
| Corpus wrong_player | 577 | 595 | +18 |
| Total matched-correct-action | 1533 | 1533 | 0 |

Neutral for total correct-action count, but surfaces 18 previously-hidden attribution mismatches on newly-evaluable labels.

Remaining:
- 244 ambiguous labels (distance > 0.10 or margin < 0.03) — flagged in `reports/gt_orphan_manual_flag.json`
- 52 labels without ballX/ballY — require manual re-label

## Files

- `outputs/action_errors/corpus_pre_repair.jsonl` — pre-Phase-1 snapshot (1228 errors)
- `outputs/action_errors/corpus_post_gt_repair_no_seq.jsonl` — post-Phase-1 baseline (1351 errors)
- `outputs/action_errors/corpus_post_phase3_pre_spatial.jsonl` — post-Phase-3, pre-spatial-repair (1142 errors)
- `outputs/action_errors/corpus.jsonl` — **current final** post-Phase-3 + spatial-repair (1160 errors: 395 FN + 170 wrong_action + 595 wrong_player)
- `outputs/action_errors/corpus_annotated.jsonl` — joined with per-rally quality signals
- `reports/gt_orphan_auto_repair.json` — the 100 applied spatial repairs (for audit)
- `reports/gt_orphan_manual_flag.json` — 296 labels flagged for manual review
