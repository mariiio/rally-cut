# GT Integrity Audit — 110 Orphaned Rallies (2026-04-17)

## Summary

After running `recover_match_state.py` on 52 videos (auto-fixable candidates identified by `diagnose_gt_track_mismatch.py --classify`), the split became:

| Status | Count | |
|---|---|---|
| clean | 254 | positions_json trackIds align with GT playerTrackIds |
| auto_fixable | 0 | remap-track-ids can align positions_json and GT |
| gt_orphaned | 110 | GT references player IDs not resolvable from current tracks |

## Diagnosis of the 110 orphans

For each orphaned rally, GT labels reference canonical player IDs (e.g. `{3, 4}`) that have no corresponding track in current `positions_json`. Typical shape:

- `avail_tids = {1, 2, 5, 6}` (raw track IDs, un-mapped for the 3/4 slots)
- `gt_tids = {1, 2, 3, 4}`
- `t2p_values = {1, 2, 3, 4}` (map's target space is correct)
- But `t2p.keys ∩ avail_tids = {1, 2}` only — tracks 5 and 6 are not in the map, so remap would assign them to `101+` sentinel, not `3` or `4`.

Running `match-players` on the video produced a confident mapping for only 2/4 player profiles, likely because appearance features drifted after retracking or the specific tracks in some rallies don't match existing profiles.

## S3 Backup Restore — Not Viable

Tested against both available snapshots:

| Snapshot | Date | Orphans covered | Would resolve |
|---|---|---|---|
| `score_gt_2026_04_14` | 2026-04-14 | 110/110 | **1/110** |
| `beach_v11` | 2026-04-13 | 110/110 | **1/110** |

Both snapshots contain the same broken GT (the orphaning predates the 2026-04-09 retrack event that baked raw-track sentinels into the labels). Restore would clobber 340 rallies for +1 rally gained. Net: strongly negative. **Skipped.**

## Mitigation (this plan's scope)

`match_contacts:540-544` already marks labels where `gt.player_track_id ∉ available_track_ids` as `player_evaluable=False`. The corpus builder inherits this: wrong_player errors are not inflated on orphaned rallies. FN_contact and wrong_action measurements remain valid.

## Follow-up queue (separate plans)

Two viable repair paths outside this plan's scope:

1. **Spatial repair via GT's ballX/ballY.** The existing diagnostic already computes `find_nearest_track_at_frame(gt.frame, gt.ball_x, gt.ball_y, player_positions)`. Wrap that as a CLI (`rallycut repair-gt-by-position <video>`) that rewrites `action_ground_truth_json[i].playerTrackId` in place with the spatial match. Estimated recovery: 60-80% of 110 orphans.
2. **Manual re-label** in the web editor for rallies where the spatial match confidence is low.

## Files

- `reports/gt_integrity_diagnosis.json` — pre-recovery classification (189 clean / 163 optimistic auto_fixable / 12 needs_restore)
- `reports/gt_integrity_diagnosis_post_recover.json` — post-recovery with optimistic classifier (254 / 100 / 10)
- `reports/gt_integrity_diagnosis_refined.json` — post-recovery with strict resolvability classifier (254 / 0 / 110)
- `reports/gt_integrity_recover_20260417_131606.log` — `recover_match_state.py` log (52 videos, 0 fail, 28.6 min)
- `reports/gt_integrity_db_snapshot_20260417_131512.sql.gz` — pre-repair DB snapshot (239 MB)
