# Probe X-O — v9 keypoint reader vs v8 NLE vs M4

Samples: 78 videos with v9 L/R net-top GT.

## Aggregate (all 77)

| Estimator | n | med \|Δ\| | mean \|Δ\| | worst | >0.025 | >0.05 |
|-----------|---|-----------|------------|-------|--------|-------|
| M4 LOO | 78 | 0.0060 | 0.0078 | 0.0242 | 0 | 0 |
| NLE v8 midpoint | 74 | 0.0103 | 0.0126 | 0.0529 | 11 | 1 |
| v9 8-kpt midpoint | 78 | 0.0046 | 0.0066 | 0.0317 | 3 | 0 |

## Tilt direction agreement (visibly-tilted, n=22)

19/22 = 86.4%

## Gate verdict: **FAIL — investigate**
- tilt direction agreement 86.4% < 90%
