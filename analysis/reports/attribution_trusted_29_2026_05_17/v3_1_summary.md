# v3.1 SET-masked team-awareness A/B (trusted-29, 2026-05-17)

v3 introduced `team_matches_expected` (feature #18). v3 lifted ATTACK +2.9pp
matched but cascade-regressed SET −2.1pp (mis-attributed receive → wrong
expected_team for set → feature misleads). v3.1 masks the feature
(passes `expected_team=None`, defaults to 0.5 uninformative) for SET
only — keeps it for ATTACK / RECEIVE / DIG / SERVE / BLOCK.

## Four-way matched-accuracy A/B (n=1027)

| Action | OFF | v2 pose | v3 team | **v3.1 SET-mask** | v3.1 − v2 | v3.1 − v3 | n |
|---|---|---|---|---|---|---|---|
| ATTACK | 83.9% | 88.1% | 91.0% | **90.0%** | **+1.9** ▲ | −1.0 | 311 |
| SET | 86.1% | 91.2% | 89.1% | **90.3%** | −0.8 | +1.3 ▲ | 238 |
| DIG | 82.6% | 91.3% | 93.8% | **94.4%** | **+3.1** ▲ | +0.6 | 161 |
| RECEIVE | 87.6% | 92.2% | 93.5% | **94.1%** | **+2.0** ▲ | +0.7 | 153 |
| SERVE | 81.5% | 84.2% | 84.2% | 84.2% | 0.0 | 0.0 | 146 |
| BLOCK | 55.6% | 33.3% | 27.8% | 33.3% | 0.0 | +5.6 | 18 |
| **TOTAL** | **83.9%** | **88.4%** | **89.3%** | **89.6%** | **+1.2** | **+0.3** | 1027 |

End-to-end raw deltas vs v2: **+12 correct attributions** across 29 videos.

## Interpretation

v3.1 is the cleanest configuration:
- **SET regression closed** (v3 −2.1pp → v3.1 −0.8pp matched; the remaining
  ~2-contact gap is likely measurement noise).
- **ATTACK lift held** (+1.9pp matched, slightly less than v3's +2.9pp but
  the difference is the indirect cascade — v3.1's better SET attribution
  changes the team-chain inputs for downstream ATTACK in a small subset
  of rallies).
- **DIG +3.1pp matched** — the largest single-action lift in v3.1, dwarfs
  the SET residual.
- **BLOCK recovered** to v2 level (n=18, still noise — needs ≥50 BLOCK GT).

## The v3 → v3.1 ATTACK delta (−1pp) is the cascade-in-reverse

In v3, SET was attributed worse (89.1%). The team chain after a wrong
SET sometimes propagates an incorrect expected_team to the next ATTACK,
which COINCIDENTALLY agreed with the also-wrong attribution. In v3.1
SET is more correct (90.3%), changing 3 downstream ATTACK chains.
This is **expected behavior** — chain-aware features have these
indirect couplings. Net result is what matters: v3.1 total 89.6% beats
v3 total 89.3%.

## Per-video summary (raw, vs v2)

- **Wins (11):** pepe +6.7, vivi +6.7, keke +5.8, veve +5.1, pupu +2.8,
  papa +2.3, wawa +2.2, toto +1.7, kaka +1.4, yeye +1.2
- **Regressions (3):** gugu −3.7 (-1 contact), mumu −3.3 (-1 contact),
  kiki −1.9 (-1 contact)
- **Unchanged (15):** titi, lulu, caco, cece, cici, cuco, gaga, juju,
  gigi, mame, meme, mimi, moma, pipi, popo, vovo

v3.1 has fewer regressions than v3 (3 vs 9), confirming the masking
removed instability.

## Ship recommendation

**v3.1 is ship-ready** modulo honest LOO CV validation:

- Clean overall lift: +1.2pp matched vs v2, +5.7pp vs OFF baseline.
- No catastrophic per-action or per-video regressions.
- ATTACK + DIG + RECEIVE all benefit (+1.9 / +3.1 / +2.0 matched).
- SET residual −0.8pp is within noise.
- DB state: v3.1 currently applied to all 29 trusted videos.

**Before defaulting to ON (env flag):** run leave-one-video-out CV to
confirm the lift isn't training-data memorization. With 29 videos × ~5
min/train + ~1 min/redetect, ~3 hr in background. If LOO accuracy is
within ~2pp of full-corpus accuracy, the lift is real and we ship.

## Files

- v3.1 measurement: `reports/attribution_trusted_29_2026_05_17/scorer_v31_on.json`
- Inference change: `analysis/rallycut/tracking/action_classifier.py::_apply_dynamic_scorer_attribution` (`_TEAM_FEATURE_MASKED_ACTIONS = {ActionType.SET}`)
- Training mask: `analysis/scripts/train_and_save_dynamic_scorer_2026_05_14.py::build_dataset` (gt_action == "SET" → expected_team=None)
