# Contact-detector at-net FP diagnostic (2026-05-14)

11 user-deleted at-net non-contacts from `rally_action_ground_truth`. Each was originally emitted by `detect_contacts` at the net, classified as ATTACK, flagged by A3 as a BLOCK candidate, then inspected by the user and removed.

User-reported FP patterns: (1) near-miss block — player jumps with arms up but no contact; (2) ball-net deflection — ball hits the net after an attack, not the player.

## Per-case signals

| # | video | rally | frame | gen(s) | dc° | velocity | pdist | arc_res | conf | ball_y | net_y_est | head_y | center_y | foot_y | classified | prev → this → next |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | cucu | 7d5fbfb2 | 218 | direction_change,velocity_peak,inflection,deceleration,net_crossing | 7.3 | 0.0114 | 0.1269 | 0.0033 | 0.601 | 0.298 | 0.308 | 0.398 | 0.489 | 0.579 | attack | set(B)@f179 → attack → dig(unknown)@f248 |
| 2 | jiji | 935ead8c | 631 | direction_change,parabolic,velocity_peak,deceleration,net_crossing | 1.2 | 0.0237 | 0.1087 | 0.0069 | 0.686 | 0.425 | 0.282 | 0.404 | 0.482 | 0.560 | attack | attack(B)@f568 → attack → - |
| 3 | lala | 2eeb3ae6 | 966 | direction_change,parabolic,velocity_peak,inflection | 27.3 | 0.0275 | 0.0792 | 0.0181 | 0.358 | 0.470 | 0.235 | 0.443 | 0.530 | 0.617 | attack | set(A)@f912 → attack → - |
| 4 | lolo | 6935b412 | 249 | direction_change,parabolic,velocity_peak,inflection,net_crossing | 0.9 | 0.0338 | 0.0353 | 0.0077 | 0.792 | 0.514 | 0.386 | 0.459 | 0.612 | 0.765 | attack | set(B)@f201 → attack → attack(A)@f307 |
| 5 | titi | 4ad457f6 | 421 | direction_change,parabolic,velocity_peak,inflection,deceleration,net_crossing | 1.7 | 0.0284 | 0.0844 | 0.0049 | 0.312 | 0.536 | 0.385 | 0.489 | 0.629 | 0.768 | attack | dig(B)@f374 → attack → - |
| 6 | toto | fcc5dcba | 174 | direction_change,parabolic,velocity_peak,inflection,net_crossing | 14.3 | 0.0243 | 0.0598 | 0.0070 | 0.678 | 0.505 | 0.495 | 0.427 | 0.589 | 0.751 | attack | set(B)@f127 → attack → - |
| 7 | toto | f1f09039 | 302 | direction_change,velocity_peak,inflection,deceleration,net_crossing | 78.0 | 0.0180 | 0.0607 | 0.0113 | 0.821 | 0.365 | 0.352 | 0.366 | 0.501 | 0.636 | attack | set(B)@f257 → attack → dig(B)@f321 |
| 8 | wewe | 83790ce7 | 331 | direction_change,inflection | 19.2 | 0.0059 | 0.1114 | 0.0021 | 0.452 | 0.230 | 0.465 | 0.292 | 0.383 | 0.474 | attack | receive(A)@f290 → attack → - |
| 9 | wiwi | 7aef7188 | 436 | velocity_peak,net_crossing | 0.5 | 0.0115 | 0.0992 | 0.0030 | 0.570 | 0.220 | 0.173 | 0.300 | 0.401 | 0.502 | attack | set(B)@f336 → attack → dig(B)@f509 |
| 10 | wowo | b07b388b | 391 | direction_change,parabolic,velocity_peak,inflection,deceleration | 0.0 | 0.0180 | 0.0628 | 0.0441 | 0.415 | 0.267 | 0.376 | 0.287 | 0.451 | 0.615 | attack | set(B)@f303 → attack → attack(A)@f413 |
| 11 | yoyo | 21a9b203 | 506 | direction_change,parabolic,velocity_peak,inflection,net_crossing | 0.0 | 0.0148 | 0.0046 | 0.0306 | 0.547 | 0.307 | 0.178 | 0.327 | 0.433 | 0.540 | attack | set(B)@f435 → attack → attack(A)@f528 |

## Pattern summary

- **n = 11** at-net FPs total.
- **Direction change is dominantly LOW**: 10/11 cases have dc < 30°; 1/11 have 40°-80°; only 0/11 ≥ 80°. The user-hypothesized 'ball-net deflection' signature (sharp dc) is ESSENTIALLY ABSENT in this set — these FPs are NOT firing because the ball bent sharply at the net.
- **Velocity is mostly low**: 6/11 have velocity < 0.020. Combined with low dc, these are weak-signal contacts.
- **Player proximity distribution**: 2/11 have player_distance < 0.05; 6/11 have 0.05-0.10; 3/11 have ≥ 0.10. So roughly half are 'a player IS near the ball' (the near-miss-block pattern), but a substantial share fire even when the nearest player is comfortably off the ball — a separate failure mode.
- **Classifier validation passed for all of them**: 11/11 were `is_validated=True`. 4/11 had confidence < 0.50. The classifier accepts them despite weak signals because the COMBINATION of weak signals matches its training distribution.
- **is_at_net flag**: only 5/11 were marked `is_at_net=True` at emit time, even though all 11 are at-net by GT inspection. The detector's own at-net flag is unreliable here.
- **Player bbox vs ball**: 5/11 have player bbox-top above the ball (image-y smaller). Roughly half-and-half — this does NOT robustly discriminate FP from TP.

- **Generator concurrency**: median 5 generators fire within ±8 of the stored contact frame. Generator counts (frames matched within ±8):
  - `direction_change`: 10/11
  - `velocity_peak`: 10/11
  - `inflection`: 9/11
  - `net_crossing`: 8/11
  - `parabolic`: 7/11
  - `deceleration`: 5/11

Most FPs are emitted by MULTIPLE generators agreeing — this is the structural FP mode. No single generator stands out.

## Recommendations (no fixes, just signals for a future workstream)

Headline: **the data falsifies the user's stated FP patterns**. Only 1/11 looks like a ball-net deflection (toto/f1f09039 with dc=78°), and only 2/11 have player_distance < 0.05 ('player very close to ball'). The remaining majority are *weak-signal* contacts where the classifier sees a multi-generator agreement on a low-dc, low-velocity, moderate-arc-residual configuration and accepts. There is no single signal that cleanly separates these 11 from true at-net contacts.

1. **No simple rule-based gate will move precision much.** A ball-net distance gate or a wrist-position gate would each fire on ≤ 3 of 11 cases — well below a 92% precision-target gate. Rule tuning at the contact-detector level on this signal set is the same dead-end as the A1/A3 ladder.

2. **The actionable insight is at the classifier level, not the generator level.** All 11 are `is_validated=True`. The GBM classifier is the one accepting them. A future workstream could retrain the classifier with these 11 (+ similar mined from the fleet) as labeled negatives — they are *clean* at-net non-contact examples, exactly the failure mode the current GBM lacks training signal for.

3. **Single-generator suppression is unsafe.** Median 5 generators fire within ±8 of each FP. Disabling any one (e.g., `enable_direction_change_candidates=False`) would still leave ≥ 4 others emitting a candidate at the same frame. The merge logic would still accept it. Generator-level suppression won't help.

4. **The is_at_net flag should be re-derived at the action layer.** Only 5/11 of these (by the detector's own at-net flag) are flagged as at-net even though all 11 are at-net by GT. An action-layer re-derivation using `|ball_y - net_y_estimated| < ε` would surface more candidates for the rule-based deletion pipeline (e.g., deduplication across cross-team adjacent contacts) — though this is still a downstream FP scrubber, not a contact-detector fix.

5. **Practical next-step recipe** (cheapest path to measurable FP reduction): (a) mine ~100 more user-deleted at-net non-contacts from `rally_action_ground_truth` by joining DELETED rows or rows the user explicitly flagged; (b) sample-equal labeled-positive at-net contacts from the same fleet; (c) compute the GBM's score distribution on both sets; (d) if the score distributions overlap (the classifier doesn't separate them), retrain the classifier on the combined set; (e) if the score distributions are separable but the threshold is in the wrong place, lift the at-net threshold from 0.30 → 0.45 for `is_at_net=True` candidates and re-measure on the GT panel.
