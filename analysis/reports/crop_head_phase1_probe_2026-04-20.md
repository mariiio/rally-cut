# Crop-Head Phase 1 Probe — PASS

- Date: 2026-04-20
- Device: mps
- Epochs: 15 | Batch: 64 | LR: 0.001
- Seed: 42

## Splits (video-level)
- Train: 53 videos, 3448 samples (1292 pos + 2156 neg)
- Val: 5 videos, 493 samples
- Test: 10 videos, 902 samples

## Ship gates (pre-registered)

| Gate | Threshold | Observed | PASS? |
|---|---|---|---|
| Test AUC | ≥ 0.75 | **0.8385** | YES |
| Orthogonality gap | ≥ 0.15 | **+0.5112** (GT 0.700 − NC 0.188) | YES |
| Hard-neg AUC | ≥ 0.65 | **0.7175** (n=344) | YES |

## Verdict: **PASS**

All three pre-registered gates passed. Phase 2 (architecture ablations: T window, pooling, input combinations) is scheduled as a separate plan in a subsequent session.

## Per-source breakdown (test)

| Source | n | Mean prob | P(≥0.5) |
|---|---|---|---|
| gt_positive | 313 | 0.6350 | 0.6997 |
| hard_negative | 31 | 0.4068 | 0.4194 |
| random_negative | 558 | 0.2180 | 0.1756 |

## Threshold sweep (test)

| Threshold | TP | FP | FN | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| 0.30 | 259 | 172 | 54 | 0.601 | 0.827 | 0.696 |
| 0.40 | 236 | 135 | 77 | 0.636 | 0.754 | 0.690 |
| 0.50 | 219 | 111 | 94 | 0.664 | 0.700 | 0.681 |
| 0.60 | 193 | 87 | 120 | 0.689 | 0.617 | 0.651 |
| 0.70 | 160 | 69 | 153 | 0.699 | 0.511 | 0.590 |

## Training curve

| Epoch | Train Loss | Val AUC | Time (s) |
|---|---|---|---|
| 1 | 0.7785 | 0.8361 | 38.6 |
| 2 | 0.6337 | 0.8560 | 36.2 |
| 3 | 0.6049 | 0.8521 | 35.3 |
| 4 | 0.5446 | 0.8461 | 35.5 |
| 5 | 0.5356 | 0.8548 | 35.2 |
| 6 | 0.5073 | 0.8550 | 35.9 |
| 7 | 0.4789 | 0.8602 | 35.6 |
| 8 | 0.4307 | 0.8673 | 35.1 |
| 9 | 0.4173 | 0.8703 | 35.6 |
| 10 | 0.3787 | 0.8582 | 35.1 |
| 11 | 0.3757 | 0.8656 | 35.2 |
| 12 | 0.3447 | 0.8614 | 36.7 |
| 13 | 0.3407 | 0.8623 | 35.8 |
| 14 | 0.2845 | 0.8628 | 35.8 |
| 15 | 0.2857 | 0.8573 | 35.7 |

## Test split video IDs
- `073cb11b-c7ba-4fac-8cc9-b032b3152ad6`
- `211e2a4c-c9a3-4438-9b0c-bea4e7555ad0`
- `313c6c95-e586-4585-bfce-a2d293d96815`
- `56f2739d-ff3c-4cac-8ada-4f275e57ab63`
- `6f8ca447-a244-4833-bf49-e9f949defc1a`
- `840e8b6b-8428-48d6-8af8-d239d7c64f5c`
- `950fbe5d-fdad-4862-b05d-8b374bdd5ec6`
- `ab4edcdc-7335-44d1-b944-b201feb5a262`
- `beb70f61-de8a-4805-a7ae-e8c7199d7275`
- `dd042609-e22e-4f60-83ed-038897c88c32`
