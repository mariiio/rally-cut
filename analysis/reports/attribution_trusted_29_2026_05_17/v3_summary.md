# v3 team-awareness A/B (trusted-29, 2026-05-17)

Adds `team_matches_expected` feature (#18) to the v2 scorer. Built from
the [[attack_residual_2026_05_17]] catalog finding that 63% of v2 ATTACK
errors are CROSS_TEAM picks. The feature is the candidate's team-chain
match against the expected attacking team, defaulting to 0.5 when the
chain is broken or team_assignments are missing.

## Matched-accuracy three-way (n=1027)

| Action | OFF | v2 pose | **v3 team** | v3 − v2 | v3 − OFF |
|---|---|---|---|---|---|
| ATTACK | 83.9% | 88.1% | **91.0%** | **+2.9** ▲ | +7.1 |
| DIG | 82.6% | 91.3% | **93.8%** | **+2.5** ▲ | +11.2 |
| RECEIVE | 87.6% | 92.2% | 93.5% | +1.3 ▲ | +5.9 |
| SERVE | 81.5% | 84.2% | 84.2% | 0.0 | +2.7 |
| SET | 86.1% | 91.2% | 89.1% | **−2.1** ▼ | +2.9 |
| BLOCK | 55.6% | 33.3% | 27.8% | −5.6 ▼ | −27.8 (n=18, noise) |
| **TOTAL** | 83.9% | 88.4% | **89.3%** | **+0.9** | +5.4 |

End-to-end raw delta vs v2: **+9 correct attributions across 29 videos.**

## Per-video summary (raw, vs v2)

- **Wins (11):** pepe +8.9, vivi +6.7, veve +5.1, keke +3.8, vovo +3.7, pipi +3.0, pupu +2.8, papa +2.3, wawa +2.2, toto +1.7, yeye +1.2
- **Regressions (9):** mame −5.9, gugu −3.7, mumu −3.3, cuco −2.1, gaga −2.1, kiki −1.9, titi −1.2 (all 1-contact deltas on small denoms; mame n=17)
- **Unchanged (9):** lulu, caco, cece, cici, kaka, juju, gigi, meme, mimi, moma, popo

## Interpretation

The hypothesis from [[attack_residual_2026_05_17]] predicted ATTACK
matched 88.1% → ~94-95% (full CROSS_TEAM closure). Actual: 88.1% →
**91.0%**, capturing ~half of the predicted closure. The team-chain
feature is helping where it should (ATTACK +2.9, DIG +2.5 — both
cross-team-sensitive actions) and hurting where the chain breaks
(SET −2.1 — a downstream cascade when the upstream receive is
mis-attributed).

The trade-off matters: ATTACK matters more than SET for user-perceived
attribution quality (attacks are the most-watched contact), but the
total accuracy lift is only +0.9pp matched (+9 contacts on 1027).

## Why not the full +6pp predicted?

Three reasons the lift was smaller than expected:

1. **Chain-breakage cascade.** When a prior contact is mis-attributed
   (e.g., wrong receive), the expected_team chain propagates the error
   to the next ATTACK. The team_match feature then "agrees with" the
   wrong-team candidate. v2 didn't have this dependency.
2. **Some CROSS_TEAM picks have legitimate ambiguity.** The catalog
   showed 11/19 picks had `arms_raised=1` and 13/19 had lower
   `wrist_to_ball` — geometrically the wrong-team pick *was* closer to
   the ball. Even with team-awareness, the pose signal pulls toward
   the blocker. The team feature has to dominate, which the GBM does
   sometimes but not always.
3. **GT-MISSING is unrecoverable** (4/30 ATTACK errors) — upstream issue
   the scorer can't touch.

## Should v3 ship?

**Pros:** ATTACK +2.9pp matched is the largest single-feature lift since
the bbox→bbox+pose v2 jump. Net +9 correct end-to-end. No catastrophic
regression anywhere.

**Cons:** SET regression −2.1pp matched is real (5 fewer correct on
n=238). BLOCK regression is noise (n=18) but worth flagging.

**Possible refinements before ship:**

1. **Per-action feature masking.** Only train `team_matches_expected`
   into ATTACK + SERVE heads (the net-crossing actions). Set it to 0.5
   for SET / RECEIVE / DIG. Avoids the cascade.
2. **Chain-confidence gating.** Skip the feature entirely when the
   serve attribution confidence is low (so the chain is untrusted).
3. **Accept the trade-off.** Net positive; ship v3 as-is.

Recommend option 1 (per-action feature masking) — keeps the ATTACK gain,
removes SET downside. Simple to implement: pass `expected_team=None`
during training+inference when action is in {SET, RECEIVE, DIG}.

## Files

- v3 OFF (baseline): `reports/attribution_trusted_29_2026_05_17/scorer_off.json`
- v2 pose:           `reports/attribution_trusted_29_2026_05_17/scorer_on.json`
- v3 team:           `reports/attribution_trusted_29_2026_05_17/scorer_v3_on.json`
- Inference module:  `analysis/rallycut/tracking/dynamic_attribution_scorer.py`
- Training script:   `analysis/scripts/train_and_save_dynamic_scorer_2026_05_14.py`
- Integration point: `analysis/rallycut/tracking/action_classifier.py::_apply_dynamic_scorer_attribution`
