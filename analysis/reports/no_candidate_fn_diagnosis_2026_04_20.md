# No-Candidate FN Diagnosis

- Total no-candidate GBM-miss GT contacts: 95
- Ball-density window: ±15 frames (ball_conf ≥ 0.3)

## Categorization

| Category | Count | Fraction |
|---|---|---|
| BALL_DROPOUT | 50 | 52.6% |
| BALL_PRESENT_NO_EVENT | 31 | 32.6% |
| EDGE | 14 | 14.7% |

## Per-action breakdown

- **BALL_DROPOUT** (n=50): {'serve': 39, 'set': 5, 'attack': 3, 'dig': 2, 'receive': 1}
- **BALL_PRESENT_NO_EVENT** (n=31): {'serve': 23, 'attack': 3, 'receive': 2, 'block': 1, 'dig': 1, 'set': 1}
- **EDGE** (n=14): {'serve': 12, 'set': 2}

## Serve-specific

- Total no-candidate serve FNs: 74
- Ball-dropout: 51
- Ball-present but no event: 23

## Verdict: PARTIAL — ball-gap inpainting helps some, serve-generator helps others

- BALL_DROPOUT fraction: 52.6%