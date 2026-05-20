# Probe N1 — `estimate_net_line` midpoint vs M4 (LOO)

Samples: 77 videos with corners + scalar net_top GT.

## Aggregate

| Estimator    | n   | med \|Δ\| | mean \|Δ\| | worst \|Δ\| | >0.025 | >0.05 | >0.10 |
|--------------|-----|-----------|------------|-------------|--------|-------|-------|
| M4 (LOO ridge) | 77 | 0.0076 | 0.0125 | 0.2221 | 5 | 1 | 1 |
| NLE midpoint | 77 | 0.0088 | 0.0118 | 0.0465 | 9 | 0 | 0 |

## NLE failures

| reason | count |
|--------|-------|
| sanity_failed | 4 |

## Winner counts (per-video, ties within ±0.005)

| winner | count |
|--------|-------|
| M4 | 30 |
| NLE | 19 |
| tie | 28 |

## Decision rule

* If NLE midpoint med \|Δ\| is within +0.002 of M4 LOO med \|Δ\| → C2/C3 viable on midpoint quality.
* If NLE midpoint med \|Δ\| > M4 LOO med \|Δ\| + 0.005 → C2/C3 die on midpoint; C1 is the path.
* If >10 NLE failures (fetch/None/sanity) → C2 needs robust M4 fallback; C3 stays viable.
