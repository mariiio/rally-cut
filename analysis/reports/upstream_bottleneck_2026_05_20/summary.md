# Upstream Bottleneck Probe — Summary (2026-05-20)

Substrate: 166 wrong-attribution contacts on trusted-32.

## Per-layer ranking

| Rank | Layer | Oracle | Realistic | Gap | Cost | Score |
|---|---|---:|---:|---:|---:|---:|
| 1 | L6 team-chain accuracy | 35 | 35 | 0.00 | 3 | 11.67 |
| 2 | L3 contact-frame regression | 37 | 37 | 0.00 | 10 | 3.70 |
| 3 | L1 player-tracker coverage | 19 | 11 | 0.42 | 3 | 2.12 |
| 4 | L2 candidate generation | 0 | 0 | 0.00 | 3 | 0.00 |
| 5 | L4 ball-tracking (overlap n=11) | 4 | 4 | 0.00 | 10 | 0.00 |

## L5 (GT-scale learning curve)

3/6 actions still sloping at frac=1.00 (slope > 0.005)


Per-action curves:

| Action | n_contacts | frac_0.25 | frac_0.5 | frac_0.75 | frac_1.0 | Δ(1.0-0.75) |
|---|---:|---:|---:|---:|---:|---:|
| SERVE | 169 | 0.907 | 0.931 | 0.963 | 0.958 | -0.005 |
| RECEIVE | 218 | 0.904 | 0.911 | 0.923 | 0.937 | +0.013 |
| SET | 319 | 0.887 | 0.933 | 0.925 | 0.923 | -0.002 |
| ATTACK | 402 | 0.889 | 0.902 | 0.869 | 0.881 | +0.012 |
| DIG | 211 | 0.920 | 0.878 | 0.920 | 0.919 | -0.002 |
| BLOCK | 26 | NaN | 0.765 | 0.770 | 0.790 | +0.020 |

## Decision rule application

**Top recommendation:** invest in **L6 team-chain accuracy** (realistic ceiling 35, cost 3, rank score 11.67).

Caveats:
- Gap ratio > 0.5 on any layer = projection-trap candidate (audit ceiling >> realistic; treat with skepticism).
- L4 reported on partial corpus when overlap < 30 rallies (rank_score forced to 0 if so).
- All ranks are confounded if a contact fails at multiple layers; aggregator does not materialize a per-contact multi-layer-fail Venn (would be a follow-up).
- L1's realistic ceiling uses the BEST of widen_pm10/widen_pm15/interpolate; L2-L6 use oracle as realistic-proxy (intervention cost reflected in COST table).
