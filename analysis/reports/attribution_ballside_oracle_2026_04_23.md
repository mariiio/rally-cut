# Attribution ball-side oracle ceiling (2026-04-23)

**Baseline accuracy:** 71.7% (225/314)
**Oracle ceiling:** 75.5% (+3.82 pp)
**Gate verdict:** BORDERLINE — consider tighter gate

## Breakdown

- Wrong (any reason): 63
  - Same-team swap (ball-side can't help): 17
  - Cross-team miss: 46
    - **Oracle-lift-eligible: 12**
    - Ball on pipeline's side (oracle wouldn't help): 12
    - Ball ambiguous/missing: 22

## Per-video

| video | n_gt | correct | wrong | cross-team | oracle-lift-eligible |
|---|---:|---:|---:|---:|---:|
| 0a383519 | 44 | 32 | 7 | 7 | 0 |
| 7d77980f | 63 | 50 | 10 | 5 | 0 |
| 2e984c43 | 48 | 18 | 26 | 21 | 9 |
| 808a5618 | 27 | 23 | 2 | 0 | 0 |
| 950fbe5d | 20 | 15 | 3 | 3 | 0 |
| 44e89f6c | 37 | 28 | 5 | 4 | 2 |
| eb693a6f | 52 | 41 | 8 | 5 | 1 |
| 4f2bd66a | 23 | 18 | 2 | 1 | 0 |
