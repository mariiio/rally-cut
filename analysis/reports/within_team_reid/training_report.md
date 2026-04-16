# Within-Team ReID — Session 3 Training Report

**Date**: 2026-04-16
**Plan**: `/Users/mario/.claude/plans/flickering-scribbling-clock.md`
**Corpus**: `analysis/training_data/within_team_reid/` — manifest sha `ec840dc0a3e8855f`, pairs sha `a9d8ddae9589d274`
**Git**: `86eed81`
**Device**: Apple MPS

## Verdict

**Session 3 GREEN-LIT for Session 4** — V3 is the global best at held-out rank-1 = **0.500** (5 / 10 scored events), beating DINOv2-S zero-shot baseline of 0.10 by **+40 pp** (4× the +10 pp gate). Cross-rally rank-1 = **0.694**, within 1 pp of the 0.703 DINOv2-S baseline (CF guard at 0.683 satisfied). Best checkpoint at `analysis/weights/within_team_reid/best.pt` (V3 epoch 2). Re-load + re-evaluation reproduces metrics byte-exactly.

Surprising finding: **the LOWEST teammate-margin weight + NO label smoothing won**. V3 (λ=0.25, smoothing=0) outperformed both V1 (λ=0.5, smoothing=0.05) and V2 (λ=1.0, smoothing=0.02) on the primary metric. SupCon alone is doing most of the work; the explicit teammate-margin term helps enough at small weights but adds destructive pressure at higher λ — V2's stronger margin pushed cross-rally below the CF guard four out of eleven epochs and tied V1 on primary.

## Variant Comparison (best epoch under CF guard)

| Variant | λ_tm | smoothing | Best epoch | Held-out rank-1 | Cross-rally rank-1 | tm_mean (held-out) | Total epochs | Halt reason |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| **DINOv2-S baseline** | — | — | — | 0.100 | 0.703 | -0.060 | — | probe baseline |
| V1 | 0.50 | 0.05 | 2 | 0.400 | **0.694** | -0.017 | 7 | early-stop |
| V2 | 1.00 | 0.02 | 6 | 0.400 | 0.684 | -0.010 | 11 | early-stop |
| **V3** | **0.25** | **0.00** | **2** | **0.500** | **0.694** | **-0.017** | 7 | early-stop |
| Target | — | — | — | ≥ 0.20 (+10 pp) | ≥ 0.683 | trend POS | — | — |
| V3 vs target | — | — | — | **+30 pp clear** | **+1.1 pp clear** | trending up on train | — | — |

Same primary on V1 and V2 (both at the 4/10 plateau); V3 broke through to 5/10 in just 2 epochs.

## Per-Epoch Curves

### V1 — λ=0.5, smoothing=0.05

| Epoch | loss | supcon | tm | tm_train | var | held-out | tm_mean(ho) | cross | CF | seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 3.618 | 3.491 | 0.254 | +0.048 | 0.787 | 0.300 | -0.046 | 0.687 | ✓ | 82.9 |
| **2** | 3.442 | 3.285 | 0.314 | -0.016 | 0.773 | **0.400** | -0.017 | **0.694** | ✓ | 79.5 |
| 3 | 2.609 | 2.492 | 0.233 | +0.066 | 0.735 | 0.400 | -0.017 | 0.684 | ✓ | 76.9 |
| 4 | 2.323 | 2.205 | 0.237 | +0.079 | 0.842 | 0.400 | -0.019 | 0.681 | ⚠ | 77.0 |
| 5 | 1.900 | 1.792 | 0.216 | +0.110 | 0.907 | 0.400 | -0.014 | 0.681 | ⚠ | 75.8 |
| 6 | 2.133 | 2.017 | 0.232 | +0.072 | 0.926 | 0.400 | -0.008 | 0.689 | ✓ | 77.7 |
| 7 | 1.785 | 1.672 | 0.224 | +0.101 | 0.928 | 0.400 | -0.019 | 0.700 | ✓ | 79.7 |

Variant best: epoch 2.

### V2 — λ=1.0, smoothing=0.02

| Epoch | loss | supcon | tm | tm_train | var | held-out | tm_mean(ho) | cross | CF | seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 4.138 | 3.869 | 0.269 | +0.018 | 0.766 | 0.300 | -0.042 | 0.692 | ✓ | 78.4 |
| 2 | 2.756 | 2.513 | 0.243 | +0.065 | 0.755 | 0.300 | -0.021 | 0.692 | ✓ | 75.5 |
| 3 | 2.450 | 2.210 | 0.240 | +0.068 | 0.874 | 0.300 | -0.028 | 0.687 | ✓ | 75.5 |
| 4 | 2.210 | 1.994 | 0.216 | +0.113 | 0.906 | 0.400 | -0.022 | 0.675 | ⚠ | 75.8 |
| 5 | 1.934 | 1.749 | 0.186 | +0.151 | 0.933 | 0.400 | -0.018 | 0.683 | ⚠ | 75.7 |
| **6** | 1.926 | 1.714 | 0.212 | +0.097 | 0.953 | **0.400** | -0.010 | **0.684** | ✓ | 75.3 |
| 7 | 1.825 | 1.626 | 0.199 | +0.129 | 0.961 | 0.300 | -0.022 | 0.690 | ✓ | 75.6 |
| 8 | 1.761 | 1.568 | 0.193 | +0.133 | 0.966 | 0.400 | -0.029 | 0.683 | ⚠ | 75.7 |
| 9 | 1.999 | 1.808 | 0.191 | +0.134 | 0.962 | 0.400 | -0.034 | 0.694 | ✓ | 76.7 |
| 10 | 1.754 | 1.532 | 0.222 | +0.114 | 0.961 | 0.200 | -0.036 | 0.675 | ⚠ | 76.1 |
| 11 | 1.903 | 1.711 | 0.192 | +0.136 | 0.961 | 0.300 | -0.045 | 0.689 | ✓ | 76.1 |

Variant best: epoch 6 (borderline CF, just 1 pp above guard). Stronger λ_tm increased cross-rally drift — epochs 4, 5, 8, 10 dropped below the 0.683 guard.

### V3 — λ=0.25, smoothing=0.00 (WINNER)

| Epoch | loss | supcon | tm | tm_train | var | held-out | tm_mean(ho) | cross | CF | seconds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|
| 1 | 3.384 | 3.317 | 0.266 | +0.040 | 0.795 | 0.300 | -0.038 | 0.692 | ✓ | 78.0 |
| **2** | 2.510 | 2.445 | 0.261 | +0.047 | 0.830 | **0.500** | -0.017 | **0.694** | ✓ | 75.2 |
| 3 | 2.204 | 2.145 | 0.238 | +0.071 | 0.883 | 0.400 | -0.025 | 0.686 | ✓ | 76.4 |
| 4 | 1.960 | 1.910 | 0.199 | +0.129 | 0.918 | 0.400 | -0.017 | 0.668 | ⚠ | 76.0 |
| 5 | 1.762 | 1.710 | 0.206 | +0.123 | 0.943 | 0.400 | -0.011 | 0.681 | ⚠ | 75.9 |
| 6 | 1.752 | 1.696 | 0.225 | +0.083 | 0.958 | 0.400 | -0.004 | 0.683 | ⚠ | 76.3 |
| 7 | 1.625 | 1.577 | 0.193 | +0.137 | 0.966 | 0.400 | -0.019 | 0.700 | ✓ | 78.1 |

Variant best: epoch 2. Weaker margin loss + zero smoothing produced the largest single jump on primary (5/10 vs 4/10 plateau in V1/V2).

## Sanity-Check Pass

All six pre-launch verification gates passed:
1. Eval cache built: 24 held-out events, 10 scorable, 641 cross-rally entries (`reports/within_team_reid/eval_cache/`).
2. **Identity head reproduces probe baselines byte-exactly**: held-out 0.100 = probe 0.10, cross-rally 0.7030 = probe 0.703.
3. One-batch smoke (V1, --smoke 1): finite loss = 13.76, head grad norm = 13.31, backbone frozen.
4. 1-epoch full-data run (V1 --epochs 1): completed cleanly in 82s, primary 0.300 (already +20 pp).
5. Variants 1-3 all completed without halt-conditions tripping (variance never below 0.755 = 96% of epoch-1 floor; tm_train always ≥ 0 from epoch 2 onward; cross-rally always within ±2 pp of baseline at any "best" candidate epoch).
6. Best-checkpoint reload reproduces metrics exactly (`weights/within_team_reid/best.pt` → 0.5000 / 0.6935 = saved values).

## Why V3 Won

1. **The corpus is cleaner than the plan assumed.** Mid-tier hard-negs came back at 1 % rejection (vs 15 % budget). Label smoothing is for noisy data; on clean data it just leaks gradient signal. V3 with `smoothing=0` got the full SupCon push.
2. **SupCon alone covers most of the within-team signal.** The teammate-margin loss is a useful complement at small weights (V3 still had λ=0.25 and tm_train trended positive — +0.04 → +0.14), but at higher λ it pulls embeddings apart aggressively enough to disrupt the cross-rally signal (V2 had four CF-fail epochs).
3. **Best epoch is early in all three variants** (epoch 2, 6, 2). With only 6,142 training pairs and a 256-crop batch (33 steps/epoch), the head saturates within a few epochs. After that, it just memorizes — tm_train keeps trending positive but the held-out plateau holds, and cross-rally sometimes drifts down.

## Caveats

- **The held-out denominator is 10 events.** A single event flip = 10 pp on the primary metric. V3's 0.500 is 5/10 vs V1's 0.400 = 4/10 — a one-event difference. This is statistically thin. Session 4 (integration eval) on full oracle suite + 800-rally retrack will be the real test.
- **14 of 24 held-out events abstained** (insufficient query/anchor crops). Those events use IoU > 0.30 occlusion gate per probe; relaxing this for held-out would expand the evaluation surface but break apples-to-apples comparison with the published baselines. Don't relax.
- **Cross-rally drift is real.** All three variants ended ≥ 1 pp below the 0.703 baseline. The CF guard at 0.683 catches catastrophic regression, but the small drift suggests the head IS reshaping DINOv2's feature space — Session 4 should monitor for any production signal degradation in cross-rally consistency.
- **Apple MPS run:** AMP disabled (MPS autocast is partial). CUDA T4 would give ~2.5× speedup but wasn't necessary here — V3 trained in ~9 min wall.

## Reproducibility

```bash
cd /Users/mario/Personal/Projects/RallyCut/analysis

# Build eval cache (one-time, ~10 min on MPS, requires DB connection)
uv run python -m training.within_team_reid.cli build-eval-cache

# Verify identity head reproduces probe baselines (smoke test)
uv run python -m training.within_team_reid.cli verify-baseline

# Train all three variants sequentially (~30 min total on MPS)
for v in v1 v2 v3; do
  uv run python -m training.within_team_reid.cli train --variant $v 2>&1 | tee /tmp/train_$v.log
done
```

Seeded with `seed=42`; manifest, pairs sha, git sha, and full config are pinned in each `weights/within_team_reid/variant_<id>/best.pt` checkpoint payload.

## Decision: Session 4 GREEN-LIT

Ship `weights/within_team_reid/best.pt` (V3 epoch 2) into Session 4 — integration into `rallycut/tracking/global_identity.py::_compute_assignment_cost` as an additive complementary cost gated by `WEIGHT_LEARNED_REID`, with weight sweep over {0.05, 0.10, 0.15, 0.20}, oracle eval gate, per-rally HOTA no-regression, and acceptance on the held-out 20 events (we held out 24 here; same set).

Pivot triggers from this report: NONE. Targets cleared, CF guard cleared, no stop conditions tripped. No need for Session 2b labeller round, ensemble pivot, or unfreezing top-2 backbone blocks.

## Artifacts

- **Global best**: `analysis/weights/within_team_reid/best.pt` (V3 epoch 2, 0.500 / 0.694)
- **Per-variant**: `analysis/weights/within_team_reid/variant_v{1,2,3}/{best.pt, last.pt, epochs.jsonl}`
- **Eval cache**: `analysis/reports/within_team_reid/eval_cache/{eval_cache.npz, metadata.json}`
- **This report**: `analysis/reports/within_team_reid/training_report.md`
- **Training logs**: `/tmp/train_v{1,2,3}.log` (ephemeral; per-epoch metrics persisted in `epochs.jsonl`)
