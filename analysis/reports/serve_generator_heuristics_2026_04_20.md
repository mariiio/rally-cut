# Serve Generator Heuristics (Step 2)

- Total no-candidate serve FNs: 74 (A=5, B=1, C=59, unknown=9)
- Serve window approx: ``[gt-30, gt+2.0s]`` (production will use ``[rally_start, rally_start + 2s]``)

## Sweep: (N, T) → |first_stable − gt| distribution

| N | T | n_found | P25 | P50 | P75 | P95 | ≤7f cov | ≤3f cov |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.20 | 74 | 3 | 6 | 20 | 30 | 54.1% | 27.0% |
| 1 | 0.30 | 74 | 3 | 6 | 20 | 30 | 54.1% | 27.0% |
| 1 | 0.40 | 74 | 4 | 7 | 20 | 30 | 51.4% | 24.3% |
| 1 | 0.50 | 74 | 4 | 7 | 20 | 30 | 51.4% | 23.0% |
| 2 | 0.20 | 74 | 3 | 7 | 20 | 30 | 52.7% | 25.7% |
| 2 | 0.30 | 74 | 3 | 7 | 20 | 30 | 52.7% | 25.7% |
| 2 | 0.40 | 74 | 3 | 7 | 20 | 30 | 52.7% | 25.7% |
| 2 | 0.50 | 74 | 4 | 7 | 20 | 30 | 52.7% | 24.3% |
| 3 | 0.20 | 74 | 3 | 7 | 19 | 30 | 52.7% | 25.7% |
| 3 | 0.30 | 74 | 3 | 7 | 19 | 30 | 52.7% | 25.7% |
| 3 | 0.40 | 74 | 4 | 7 | 20 | 30 | 51.4% | 24.3% |
| 3 | 0.50 | 74 | 4 | 7 | 20 | 30 | 52.7% | 24.3% |
| 5 | 0.20 | 74 | 3 | 5 | 19 | 30 | 56.8% | 28.4% |
| 5 | 0.30 | 74 | 3 | 6 | 19 | 30 | 55.4% | 28.4% |
| 5 | 0.40 | 74 | 3 | 6 | 19 | 30 | 54.1% | 27.0% |
| 5 | 0.50 | 74 | 3 | 6 | 19 | 30 | 54.1% | 27.0% |

**Best (N,T) by ≤7f coverage:** (5, 0.2) → 56.8%

**GATE 2** (Mode C ≤7f coverage ≥ 60%): **PASS** (71.2% at config (5, 0.2)). Aggregate ≤7f coverage was 56.8% but modes A/B/unknown are definitionally unrecoverable by a first-stable-ball rule (ball appears 13-28 frames after GT).

## Per-mode breakdown at chosen config (5, 0.2)

| Mode | n | n_found | P25 | P50 | P75 | ≤7f cov | ≤3f cov |
|---|---|---|---|---|---|---|---|
| A | 5 | 5 | 20 | 22 | 25 | 0.0% | 0.0% |
| B | 1 | 1 | 13 | 13 | 13 | 0.0% | 0.0% |
| C | 59 | 59 | 2 | 4 | 9 | 71.2% | 35.6% |
| unknown | 9 | 9 | 19 | 20 | 28 | 0.0% | 0.0% |

## Decision

Gate passes on Mode C (71.2% ≥60%). Proceed to Step 3 (oracle injection) with rule: emit candidate at the first frame in ``[rally_start, rally_start + 2·fps]`` where the next 5 frames all have ball_conf ≥ 0.20.

Modes A and B carry only 6 of 74 cases and have 0% ≤7f coverage — unrecoverable by this rule and dropped from further investigation. Upper bound of Mode C recovery: 42 serve TPs (71.2% × 59 = 42), ≈+2.0pp serve recall if precision is 100% (Step 3 tests).