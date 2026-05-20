# Probe X-G findings: per-stage candidate audit for the 27 trusted-31 GT blocks

**Date:** 2026-05-20
**Inputs:** 27 GT block contacts on trusted-31 (`rally_action_ground_truth` joined
to `player_tracks`). Probe runs `_prepare_candidates` then `detect_contacts` with
the production GBM + sequence_probs and reads per-stage frames near GT block frame
AND near the implied attack frame (`gt - 15..gt - 4` on the opposite court side).

## Headline reframe

**The block contact is NOT the issue.** 26 of 27 GT blocks have a detected
contact within ±7 frames of the GT frame; only `kiki 0bc00f94` lands at ±9
(H-A4, just outside eval tolerance).

The ACTUAL hole is the ATTACK contact 4-15 frames before the block. Without
that attack the block rule can never fire (it requires
`last_action_type == ATTACK` and `gap ≤ 8`). 22 of 27 cases have no detected
contact in the attack window at all.

## Failure mode distribution — for the missing preceding ATTACK

Classified by where in the pipeline the attack candidate disappears:

| Pattern | Count | What it means |
|---|---|---|
| **H-A3** (GBM rejects) | **13** | Generators fire AND candidates survive merging, but the contact GBM rejects them at the final accept gate |
| **H-A1** (merger drops cross-side pair) | **8** | Generators fire but no candidate survives the merge chain — likely because a block candidate at `T+5..T+10` on the opposite side dominates the merger keyed by `min_peak_distance_frames=12` |
| **DETECTED ATTACK** | **5** | Attack contact exists in the final list but the block rule still doesn't fire (next probe needed: is action_classifier labeling them as `attack` or as `set/dig`?) |
| **H-A2** (no signal) | **1** | No generator fires in the attack window (caco cfc464a7) |
| **H-A4** (block contact at ±8-15) | **1** | Block contact itself is detected but lands outside eval ±7 tolerance (kiki 0bc00f94) |

## Cases by pattern

### H-A1 (8) — merger collapse hypothesis
kaka f33d7ac8, papa 047c5e5f, pupu e11fa028 (frame 386), titi 21029e9f,
titi 1e38daab, toto 70bd06c8, veve 4c27b635, yeye 2d3cb54b.

For each, `gen@att ≥ 1` but `cand@att == 0`. The candidate at the attack
frame was generated but lost during `_merge_candidates`. The block candidate
sits 4-10 frames later on the opposite side and likely wins the merger.

### H-A3 (13) — GBM rejects the candidate that survives merging
caco 9452ee5a, gigi 72c8229b, gigi 3e07342a, gigi b8d333ae,
juju d810943e, juju acada27e, juju c89b346b, kiki a0aba15e,
mimi f3695225, moma 753a4ec7, moma 9bb60892, popo c1052008, toto 67b3e1ad.

For each, `gen@att ≥ 1` AND `cand@att ≥ 1` AND `det@att == 0`. The candidate
makes it through merge+dedup but the GBM scores it below the 0.30 acceptance
threshold AND the seq-anchored rescue gate (seq_max_nonbg ≥ 0.95 AND
gbm < 0.10) doesn't catch it. None of these are 60fps videos — this is NOT
the documented 60fps GBM bug.

### DETECTED ATTACK (5)
juju 6022138d, juju e03ef981, kiki 0bc00f94, moma e1929103, pupu e11fa028 (176).

These have a contact in the attack window. Next probe (X-H) should look at
what action_classifier labels them and why the block rule doesn't engage.

## Hypotheses for the H-A3 root cause (need P-X-H probe to confirm)

Possible mechanisms for the GBM rejecting attack candidates that immediately
precede a block:

1. **Feature drift from the block-back**: post-attack ball trajectory features
   (velocity_ratio, direction_change) are computed over windows that extend
   PAST the attack — and the immediate block at T+5 distorts those windows.
   The GBM was trained on "isolated" attacks where the ball continues outward
   for many frames.

2. **frames_since_last contamination**: after a block contact is accepted at
   T+5, the next attack candidate's `frames_since_last` would be small,
   penalising it. But the order is: candidates pass GBM first, THEN are
   added to `prev_accepted_frame`. So this only matters when the block
   candidate is processed BEFORE the attack candidate in candidate_frames
   order (sort by frame). The block IS later, so this argument actually
   FAILS — attacks should be processed first. Worth empirical check.

3. **The "real" block-paired attack is just a soft hit** (set-up tip, roll
   shot) with low GBM probability. The GBM is trained on clear attack
   features; soft-touch attacks at the net look more like sets/digs.

## Recommendation

This is multi-day work. The plan's decision-tree branch:
- H-A1 (8) → cross-side merger relaxation with net-proximity guard
- H-A3 (13) → either a "block-paired attack rescue" rule in `detect_contacts`
  OR a contact GBM retrain with hard-negative examples of attack-immediately-
  followed-by-block. Both bigger lift than H-A1.
- DETECTED-but-unmapped (5) → action_classifier-layer investigation (touch-
  counting / GBM type classifier behaviour for "attack followed by net contact")

Phase 1 (asymmetric is_at_net, v6) is committed and is a strict superset, so
it can't regress. Phase 3 should attack the largest bucket (H-A3, 13 cases)
first.

## Numbers reproduce

```bash
cd analysis
uv run python -u scripts/probe_X_g_block_pipeline_stages.py
# Writes reports/contact_fps_2026_05_19/probe_X_g_block_stages.csv and
# prints the pattern distribution.
```
