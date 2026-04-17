# Session 9 — Individual-Identity ReID Head Probe

**Date**: 2026-04-17
**Status**: Design approved, pending implementation plan
**Predecessor**: Session 8 (NO SHIP — merged veto ceiling accepted), anchor viability probe (NOT VIABLE with team-trained head)

## Motivation

Session 8 closed the within-team merge-veto workstream after 5 sessions (S4-S8) failed to reach the ≥50% swap reduction gate. Best result: 42% with learned ReID + court-plane velocity stacked.

A follow-up probe tested "anchor-based identity propagation" — pick the rally where same-team players are most separable, use it to anchor identity across the match. Result: NOT VIABLE with the current head (median within-team cosine = 0.784, only 16% of rallies have cos < 0.5).

**Root cause**: Session 3's head was trained with SupCon using `same_team = positive pair`. It explicitly pulls teammates' embeddings together. High within-team cosine is the head working as designed — wrong objective for individual identity.

**Key signal**: even with the wrong training objective, some rallies show real separation (cos as low as 0.21). The DINOv2 backbone HAS within-team individual signal — the current head suppresses it.

**Session 9's thesis**: retrain the head with `same_player = positive pair` (individual identity, not team membership). Same architecture, same data, different labels. Then re-run the anchor probe.

## Training

### Objective change

| | Session 3 | Session 9 |
|---|---|---|
| Loss | SupCon | SupCon |
| Positive pair | same team across crops | **same player** across rallies |
| Hard negative | different team | **teammate** (same team, different player) |
| Architecture | DINOv2 ViT-S/14 → 384→192→128 MLP | identical |
| Backbone | frozen | frozen |

### Label source

`player_matching_ground_truth.json` from the dataset export. Provides player IDs 1-4 per rally per video. Map each (video_id, rally_id, track_id) → player_id via the GT match-players assignments.

Only GT videos (~43 videos × ~15 rallies × 4 players). Clean labels, no algorithmic noise.

### Training data

Reuse Session 3's crop harvest pipeline. The crops are already extracted per-track per-rally in `training/within_team_reid/`. Re-label with player_id instead of team_id.

### Hyperparameters

Same as Session 3 V3 (the shipped checkpoint):
- Backbone: DINOv2 ViT-S/14 (frozen, 384-d output)
- Head: 384→192→128 MLP with ReLU + L2 normalization
- Loss: SupCon (temperature 0.07)
- Epochs: match Session 3's stopping point
- Optimizer: Adam, same LR schedule

### Output

`weights/within_team_reid/best_individual.pt`

## Probe

Re-run `scripts/probe_anchor_viability.py` (already written in Session 8) pointing at the new checkpoint. Same methodology:

- For each of 43 GT rallies: compute median learned embedding per primary track.
- For each same-team pair: cosine similarity between medians.
- Aggregate: distribution, separation counts, per-video anchor coverage.

### Expected change

The head now pushes teammates' embeddings APART. Within-team cosine should drop significantly. If median drops from 0.784 → below 0.5, anchor-based propagation is viable.

## Gates

1. **Viable** (≥70% of videos have at least one anchor rally with cos < 0.5 between teammates): declare success, scope propagation pipeline as Session 10.
2. **Marginal** (30-70%): investigate per-video failure cases. Decide whether to pursue with partial coverage or close.
3. **Not viable** (<30%): DINOv2 backbone lacks within-team individual signal at this scale. Close the workstream definitively.

### Cross-check

Cross-rally rank-1 accuracy must not regress below Session 3's 0.694. Individual identity labels make cross-rally matching easier (different-team pairs are a strict superset of different-player pairs), so this should hold trivially.

## Files

### Modified
- `analysis/training/within_team_reid/` — re-label crops with player_id. May require a new label-mapping script or flag on the existing harvester.
- Training script (location TBD during implementation — likely `training/within_team_reid/train.py` or a new `train_individual.py`).

### Reused (no edits)
- `analysis/scripts/probe_anchor_viability.py` — Session 8 probe, already committed.
- DINOv2 ViT-S/14 backbone — frozen, same as Session 3.
- Crop harvest pipeline — same extracted crops.

### New
- `analysis/weights/within_team_reid/best_individual.pt` — trained checkpoint (gitignored).
- `analysis/reports/merge_veto/anchor_viability_probe_v2.md` — probe results with new head.

## Budget

| Task | Estimate |
|---|---|
| Re-label training data (player_id from GT) | 30 min |
| Train new head (MLP-only, backbone frozen) | 30-60 min |
| Re-run anchor viability probe | 5 min |
| Report + memory update | 15 min |
| **Total** | **~1.5 h** |

## What NOT to change

- Session 3's original checkpoint (`best.pt`) — keep as-is for team-level matching.
- Any merge-veto code (Sessions 4-8) — stays dormant.
- The `link_tracklets_by_appearance` pipeline — this session doesn't touch tracking.
- `match-players` cross-rally matching — operates independently.

## Risk

| Risk | Mitigation |
|---|---|
| DINOv2 backbone genuinely can't separate teammates | Gate #3 closes the workstream. Probe is cheap (1.5h). |
| Too few training samples per player (some players appear in only 2-3 rallies) | SupCon handles variable-size classes. Minimum 2 rallies per player for a valid positive pair. |
| Overfitting to 43 GT videos | Same corpus as Sessions 1-8. Statistical caveat acknowledged. Probe is a viability check, not a production deployment. |
| Cross-rally accuracy regresses | Cross-check gate. Individual identity is strictly finer than team identity — should only improve cross-rally. |
