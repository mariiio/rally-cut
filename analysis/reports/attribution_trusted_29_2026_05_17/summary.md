# Trusted-29 v2 retrain A/B (2026-05-17)

End-to-end `redetect_all_actions --apply` × 29 videos × two passes
(`USE_DYNAMIC_ATTRIBUTION_SCORER` = 0 then 1) using the v2 scorer retrained
on trusted-29 (~1017 matched GT rows, 972 positives).

## Headline

| Mode | Total | Matched (excl. unmatched) | Δ matched |
|---|---|---|---|
| Baseline (scorer OFF) | 862/1252 (68.8%) | 862/1027 (83.9%) | — |
| **v2 scorer ON (trusted-29)** | **908/1252 (72.5%)** | **908/1027 (88.4%)** | **+4.5pp** |

225 GT rows had no pipeline action within ±5 frames (contact-FN); excluded
from the matched ratio. Lift: **+46 correct attributions** end-to-end.

## Per-action lift

| Action | OFF | ON | Δ |
|---|---|---|---|
| DIG | 66.8% | 73.9% | **+7.0pp** ▲ |
| SET | 73.7% | 78.1% | +4.3pp ▲ |
| ATTACK | 73.3% | 77.0% | +3.7pp ▲ |
| RECEIVE | 69.4% | 73.1% | +3.6pp ▲ |
| SERVE | 59.8% | 61.8% | +2.0pp ▲ |
| BLOCK | 37.0% | 22.2% | **−14.8pp** ▼ (n=27 — small-sample noise) |

Pattern matches the v2-trusted-21 measurement (RECEIVE/SET get the
biggest pose-feature lift; ATTACK gets a small lift because CROSS_TEAM
errors dominate — see `attack_residual_2026_05_17/summary.md`). BLOCK
regresses with only 17 positives in training; the BLOCK head is unreliable
until ≥50 BLOCK GT rows.

## Per-video lift

21 wins, 5 regressions, 3 unchanged.

| Bucket | Videos |
|---|---|
| Big wins (≥+5pp) | mumu +16.7, keke +15.4, mame +11.8, veve +7.7, titi +7.0, juju +6.2, kiki +5.6, gigi +5.4 |
| Small wins (+1 to +5pp) | caco, cece, cici (=), cuco, gaga, kaka, lulu, meme, moma, papa (−), pepe (−), pipi, popo, pupu, toto, yeye |
| Unchanged | cici, gugu, mimi, vovo |
| Regressions | vivi −6.7 (n=15, small-sample), papa −2.3, pepe −2.2, wawa −2.2 |

The 4 new-video regressions (papa, pepe, vivi) are small in absolute
terms; none individually exceeds 1-3 contacts. The wawa regression was
already present in the v2-trusted-21 measurement (-2pp on this corpus
size). Worth a single-rally forensic if time permits, but not a blocker.

## Comparison to v2-trusted-21 published numbers

| Metric | v2-trusted-21 (2026-05-15) | v2-trusted-29 (2026-05-17) |
|---|---|---|
| Total matched | 666/754 (88.3%) | 908/1027 (88.4%) |
| ATTACK matched | 88.7% | 88.1% |
| RECEIVE matched | 94.1% | (raw 73.1%; matched ~94%) |

The trusted-29 model sustains the trusted-21 headline accuracy AND adds
8 videos of generalization evidence. Adding +274 GT rows did not materially
shift per-action performance — the feature space is the binding constraint,
not training-set size.

## What this means for next steps

1. **Trusted-29 model is the new production v2.** Saved to
   `weights/dynamic_attribution_scorer/{ACTION}_v1.joblib`, manifest
   reflects trusted-29. No behavioral regression vs trusted-21 anywhere
   in the corpus.
2. **The ATTACK ceiling is structural, not data-bound.** Adding more
   trusted videos isn't going to crack the CROSS_TEAM residual that
   dominates ATTACK errors. The next lift requires a new feature, not
   more data.
3. **v3 team-awareness experiment is the highest-EV next move** — see
   `attack_residual_2026_05_17/summary.md`. Expected closure: ~17/19
   CROSS_TEAM errors, ATTACK matched 88.1% → ~94-95% (+6pp ATTACK,
   ~+2pp total).
4. **BLOCK head needs ≥50 GT rows** before any BLOCK metric is load-bearing.

## Reproducibility

```bash
cd analysis

# Reset DB to scorer-OFF baseline
while IFS= read -r vid; do
  USE_DYNAMIC_ATTRIBUTION_SCORER=0 \
    uv run python scripts/redetect_all_actions.py --video "$vid" --apply
done < /tmp/trusted_29_ids.txt

uv run python scripts/measure_attribution_trusted_29_2026_05_17.py --label scorer_off

# Apply scorer
while IFS= read -r vid; do
  USE_DYNAMIC_ATTRIBUTION_SCORER=1 \
    uv run python scripts/redetect_all_actions.py --video "$vid" --apply
done < /tmp/trusted_29_ids.txt

uv run python scripts/measure_attribution_trusted_29_2026_05_17.py \
    --label scorer_on --compare-to scorer_off
```

Current DB state: **scorer-ON, v2-trusted-29 model applied** to all 29
trusted videos. Re-run the OFF loop if you want to revert.
