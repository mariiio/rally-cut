# CRF-0 Transition Matrix Analysis

- Rallies analysed: 364
- Consecutive-contact transitions observed: 1731
- Laplace smoothing α=0.5
- Gap buckets: ['0-5', '6-15', '16-40', '41-120', '121-+']

## Canonical transition checks

| Transition | Bucket | Cross | Expect | P(b|a,bkt,cross) | Count | Denom |
|---|---|---|---|---|---|---|
| serve→receive cross 41-120 | 41-120 | cross | HIGH | 0.948 | 63 | 64 |
| receive→set same 6-15 | 6-15 | same | HIGH | 0.000 | 0 | 0 |
| set→attack same 16-40 | 16-40 | same | HIGH | 0.900 | 31 | 32 |
| attack→dig cross 0-5 | 0-5 | cross | HIGH | 0.026 | 0 | 16 |
| attack→block cross 0-5 | 0-5 | cross | HIGH | 0.868 | 16 | 16 |
| block→dig cross 0-5 | 0-5 | cross | HIGH | 0.000 | 0 | 0 |
| attack→attack same 0-5 | 0-5 | same | LOW | 0.042 | 0 | 9 |
| serve→attack any | 0-5 | same | LOW | 0.000 | 0 | 0 |
| receive→receive same 0-5 | 0-5 | same | LOW | 0.000 | 0 | 0 |
| dig→dig same 0-5 | 0-5 | same | LOW | 0.000 | 0 | 0 |

## Marginal transition matrix P(b | a) (all gaps/cross)

| a \ b | serve | receive | set | attack | dig | block |
|---|---|---|---|---|---|---|
| serve | 0.00 | 1.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| receive | 0.00 | 0.00 | 0.87 | 0.10 | 0.04 | 0.00 |
| set | 0.00 | 0.00 | 0.00 | 0.98 | 0.01 | 0.00 |
| attack | 0.00 | 0.00 | 0.01 | 0.07 | 0.84 | 0.09 |
| dig | 0.00 | 0.00 | 0.72 | 0.23 | 0.04 | 0.00 |
| block | 0.00 | 0.00 | 0.12 | 0.06 | 0.82 | 0.00 |

## Transition informativeness

- Mean KL(P || Uniform): 0.820
- Median KL: 0.823
- Max possible: log(7) ≈ 1.945  (higher = more structure)

## Rescue-candidate transition analysis

- Total GBM-MISSes analysed: 190
- With transition support ≥ 10 events: 159 (83.7%)
- With P(rescue transition) ≥ 0.30: 177 (93.2%)
- Out-of-distribution (0 observations): 0 (0.0%)

## Verdict: PARTIAL STRUCTURE — PROCEED WITH CAUTION

- HIGH-expect transitions passing (P≥0.40): 3/6
- LOW-expect transitions passing (P≤0.05): 4/4