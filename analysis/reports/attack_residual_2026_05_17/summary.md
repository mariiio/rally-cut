# ATTACK residual failure-mode catalog (trusted-21, v2 scorer)

- Total GT ATTACK rows: 278
- Correct (scorer picked GT): 226
- Wrong: 30
- Skipped (no positions / no candidates / no contact within ±15f): 22
- Attribution accuracy on classified rows: **88.3%**

## Failure-mode breakdown (wrong picks)

| Category | Count | Share |
|---|---:|---:|
| CROSS_TEAM | 19 | 63.3% |
| OTHER | 7 | 23.3% |
| GT_MISSING | 4 | 13.3% |

## Category definitions

- **GT_MISSING** — GT resolved_track_id not present in `contact.player_candidates`. Upstream tracker / candidate-collection bug; no per-action scorer can fix.
- **CROSS_TEAM** — pick and GT on different teams (team_assignments). Failure of team-chain repair / ratio-cap relaxation.
- **POSE_BLIND_SPOT** — same team, both contestants `arms_raised=1` AND `|wrist_to_ball_pick - wrist_to_ball_gt| < 0.04`. Pose features can't disambiguate; needs visual scene reasoning (DINOv2/V-JEPA/VLM).
- **DEPTH_OVERWEIGHT** — same team, pick has smaller `bbox_dist` (upper-quarter) but GT has smaller raw bbox-center distance. Depth-correction overweights near-side blocker. Try: cap depth-scale, or use raw distance as a feature.
- **TIGHT_CONTEST** — same team, `|prob_pick - prob_gt| < 0.05`. Scorer is genuinely uncertain; more GT or a new signal class needed.
- **OTHER** — everything else (rare; investigate individually).

- Per-error CSV: `reports/attack_residual_2026_05_17/errors.csv`

## Key findings

### CROSS_TEAM dominates (63% of all ATTACK errors)

The dominant ATTACK residual is **attacker-vs-blocker confusion across the net**, not within-team contests. Of the 19 cross-team errors:

- 11/19 picks have `arms_raised=1` (blocker pose at net) vs 5/19 GTs
- 13/19 picks have lower `wrist_to_ball` than GT (blockers' hands sit at ball trajectory at net)
- 16/19 picks have lower `bbox_dist` than GT (depth-correction amplifies near-side blocker)
- Pick teams roughly symmetric (10 A, 9 B) — no systematic team bias

**Pose features actively amplify the confusion** because blockers near the net look pose-similar to attackers (arms up, hands at ball). The v2 scorer treats this geometric similarity as evidence for "is the attacker" when in fact it's evidence for "is the blocker on the opposite team".

### Next-lift hypothesis: team-awareness feature

Add `team_matches_expected_attacker` (1.0 if candidate's team matches the team-chain-expected attacking team, 0.0 otherwise). During training:
- The feature would be 0.0 for all cross-team negative candidates and 1.0 for same-team candidates
- The scorer would learn that cross-team picks correlate strongly with non-GT labels

Expected lift if it closes most CROSS_TEAM errors: ATTACK 88.3% → ~94–95% (+6–7pp on ATTACK alone, ~+2pp on total).

Risk: the team-chain itself can be wrong; in that case the feature misleads the scorer. Need to gate the feature by team-chain confidence, or treat low-confidence team-chain as feature=0.5 (uninformative).

### OTHER is concentrated in keke (5/7)

5 of 7 OTHER errors are in keke video. All same-team within-team contests where the wrong player wins decisively on every pose signal. Suggests a video-specific track-stability or pose-quality issue rather than a structural scorer limitation. Worth a separate forensic on keke r3/r4/r6 ATTACK frames before generalizing.

### GT_MISSING (13%) is upstream

4 errors where the GT track is not in `contact.player_candidates`. No per-action scorer can fix this — it's a contact-detection or player-tracking issue.
