-- Cleanup script: removes same-action duplicates within ±3 frames in
-- rally_action_ground_truth. Mirrors the client and server-side dedup
-- semantics in actionGroundTruthService.ts and playerTrackingStore.ts.
--
-- Rule: when two rows in the same rally have the same `action`, are within
-- 3 frames of each other, and have the same `resolved_track_id` (or either
-- side null), keep the row with the HIGHER frame number and delete the
-- other. MANUAL pins are never touched on either side.
--
-- Reason higher-frame wins: in practice the user re-labels at slightly
-- LATER frames as they refine timing; the later label represents the most
-- recent intent. Created_at would also work but is less robust against
-- backfill ordering.
--
-- Usage (locally): PGPASSWORD=postgres psql -h localhost -p 5436 -U postgres \
--                    -d rallycut -f api/src/scripts/cleanupActionGtDuplicates.sql
--
-- Run inside a transaction so a botched cleanup can be rolled back.
BEGIN;

CREATE TEMP TABLE _gt_dupe_cleanup AS
SELECT b.id
FROM rally_action_ground_truth a
JOIN rally_action_ground_truth b
  ON a.rally_id = b.rally_id
 AND a.action = b.action
 AND a.frame > b.frame
 AND a.frame - b.frame <= 3
 AND (
   a.resolved_track_id IS NULL
   OR b.resolved_track_id IS NULL
   OR a.resolved_track_id = b.resolved_track_id
 )
WHERE a.resolved_source <> 'MANUAL' AND b.resolved_source <> 'MANUAL';

SELECT count(*) AS rows_to_delete FROM _gt_dupe_cleanup;

DELETE FROM rally_action_ground_truth
WHERE id IN (SELECT id FROM _gt_dupe_cleanup);

-- Verification: no remaining matches under the same rule.
SELECT count(*) AS remaining_after_cleanup
FROM rally_action_ground_truth a
JOIN rally_action_ground_truth b
  ON a.rally_id = b.rally_id
 AND a.action = b.action
 AND a.frame > b.frame
 AND a.frame - b.frame <= 3
 AND (
   a.resolved_track_id IS NULL
   OR b.resolved_track_id IS NULL
   OR a.resolved_track_id = b.resolved_track_id
 )
WHERE a.resolved_source <> 'MANUAL' AND b.resolved_source <> 'MANUAL';

COMMIT;
