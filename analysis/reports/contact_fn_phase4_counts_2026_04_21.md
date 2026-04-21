# Contact FN Phase 4 — Corpus-wide category counts

Counts over 238 non-block FNs. Source: `outputs/fn_stage_attribution.jsonl` joined with `outputs/action_errors/corpus_eval_reconciled.jsonl`. Generator: `scripts/phase4_categorize_fns.py` (SEED=n/a, deterministic).

**Total non-block FNs analyzed: 225**

## Primary category counts

Each FN is assigned to exactly one primary category. Priority: stage-lost → most-specific feature signature. Uncategorized-within-stage buckets use prefix `u-`.

| Category | Count | % of 238 | Expected from sample | Examples (rally[:8]/frame/action) |
|---|---|---|---|---|
| `7-dedup_kill` | 74 | 32.9% | ~107 (but capped by 84 dedup-lost) | `1bfcbc4f`/76/serve, `5d35c3bf`/216/dig, `675413bb`/66/serve |
| `2-kin_underreports` | 31 | 13.8% | ~60 | `5e8af221`/326/dig, `71c5d769`/234/receive, `740ffd88`/611/dig |
| `6-ball_gap_exceeds_interp` | 31 | 13.8% | ~12 | `21a9b203`/128/serve, `21a9b203`/200/receive, `11942746`/95/serve |
| `u-classifier_other` | 21 | 9.3% | — | `99a01ce4`/986/attack, `11723e2b`/133/receive, `62b6c286`/277/dig |
| `5b-serve_cand_gen_other` | 18 | 8.0% | — | `a43fb033`/124/serve, `5447e090`/51/serve, `f1c7ec19`/41/serve |
| `4-dual_occlusion` | 10 | 4.4% | ~24 | `4b7ad71f`/505/dig, `67abc72e`/239/set, `2adb9d80`/241/serve |
| `2b-kin_moderate_gbm_rejects` | 8 | 3.6% | — | `39139435`/207/dig, `e99733e3`/362/attack, `25edb83f`/273/attack |
| `u-candidate_gen_other` | 7 | 3.1% | — | `06bafb49`/259/attack, `2e8b3ce2`/125/receive, `f3695225`/208/attack |
| `8-action_labeling` | 5 | 2.2% | — | `25edb83f`/92/serve, `3cd2782c`/102/serve, `62b6c286`/90/serve |
| `1-interp_erases_deflection` | 5 | 2.2% | ~12 | `eaaa5305`/356/attack, `67abc72e`/338/attack, `e5e4c0b7`/168/set |
| `7+4-dedup_kill_with_occlusion` | 4 | 1.8% | subset of 7 | `55565c2b`/508/dig, `5cc5127f`/151/receive, `1f4c643a`/562/attack |
| `3-kin_max_gbm_rejects` | 4 | 1.8% | ~24 | `fb6e37bf`/277/receive, `7f0f540a`/696/attack, `9501aa64`/244/set |
| `10-matcher_steal` | 3 | 1.3% | — | `21029e9f`/189/attack, `4c0f4c83`/456/dig, `a11d3733`/343/attack |
| `seq-signal-only` | 3 | 1.3% | — | `1e38daab`/602/attack, `0793ebd2`/259/attack, `6d5e04f4`/229/set |
| `5-serve_ball_dropout` | 1 | 0.4% | ~12 | `fb6e37bf`/210/serve |

## Flat category counts (include cross-cutting overlaps)

| Category | Flat count | % of 238 |
|---|---|---|
| `1-interp_erases_deflection` | 10 | 4.4% |
| `2-kin_underreports (dir<=30)` | 31 | 13.8% |
| `2b-kin_moderate_gbm_rejects (30<dir<170 AND gbm<=0.15)` | 9 | 4.0% |
| `3-kin_max_gbm_rejects (dir>=170)` | 4 | 1.8% |
| `4-dual_occlusion` | 54 | 24.0% |
| `5-serve_ball_dropout` | 1 | 0.4% |
| `5b-serve_cand_gen_other` | 18 | 8.0% |
| `6-ball_gap_exceeds_interp (gap>=4)` | 32 | 14.2% |
| `7-dedup_kill` | 83 | 36.9% |

## Per-primary stage sanity-check

Each row: primary category → how its FNs distribute across `lost_at_stage`.

| Primary | top 3 stages |
|---|---|
| `7-dedup_kill` | dedup_survived:74 |
| `2-kin_underreports` | classifier_accepted:31 |
| `6-ball_gap_exceeds_interp` | ball_tracked:31 |
| `u-classifier_other` | classifier_accepted:21 |
| `5b-serve_cand_gen_other` | candidate_generated:18 |
| `4-dual_occlusion` | classifier_accepted:10 |
| `2b-kin_moderate_gbm_rejects` | classifier_accepted:8 |
| `u-candidate_gen_other` | candidate_generated:7 |
| `8-action_labeling` | action_labeled:5 |
| `1-interp_erases_deflection` | candidate_generated:5 |
| `7+4-dedup_kill_with_occlusion` | dedup_survived:4 |
| `3-kin_max_gbm_rejects` | classifier_accepted:4 |
| `10-matcher_steal` | matched_to_gt:3 |
| `seq-signal-only` | seq_signal:3 |
| `5-serve_ball_dropout` | ball_tracked:1 |

## Per-primary action distribution

| Primary | action distribution (top 3) |
|---|---|
| `7-dedup_kill` | serve:18, dig:18, receive:14 |
| `2-kin_underreports` | dig:12, serve:9, receive:6 |
| `6-ball_gap_exceeds_interp` | serve:19, set:5, receive:4 |
| `u-classifier_other` | attack:8, dig:6, serve:4 |
| `5b-serve_cand_gen_other` | serve:18 |
| `4-dual_occlusion` | dig:3, set:3, serve:2 |
| `2b-kin_moderate_gbm_rejects` | dig:5, attack:2, serve:1 |
| `u-candidate_gen_other` | attack:3, receive:2, dig:1 |
| `8-action_labeling` | serve:4, receive:1 |
| `1-interp_erases_deflection` | attack:2, set:2, dig:1 |
| `7+4-dedup_kill_with_occlusion` | dig:2, receive:1, attack:1 |
| `3-kin_max_gbm_rejects` | receive:1, attack:1, set:1 |
| `10-matcher_steal` | attack:2, dig:1 |
| `seq-signal-only` | attack:2, set:1 |
| `5-serve_ball_dropout` | serve:1 |

## Cross-cutting: Category 7 overlap with primary categories

Category 7 is the `classifier_accepted=True AND dedup_survived=False` signature. Primary assignment is dedup_survived-stage-based, but Cat 7 can tag classifier-accepted cases outside dedup-stage if they exist.

| Primary category | Count with Cat7 tag |
|---|---|
| `7-dedup_kill` | 74 |
| `6-ball_gap_exceeds_interp` | 4 |
| `7+4-dedup_kill_with_occlusion` | 4 |
| `seq-signal-only` | 1 |

## Output artifacts

- `outputs/phase4_category_assignments.jsonl` — per-FN assignment (one line per non-block FN)
- `reports/contact_fn_phase4_counts_2026_04_21.md` — this file

Interpret via `analysis/reports/contact_fn_phase3_categories_2026_04_21.md` for signature + mechanism hypotheses.
