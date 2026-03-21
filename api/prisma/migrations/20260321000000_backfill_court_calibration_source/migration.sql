-- Backfill: pre-migration videos with courtCalibrationJson but NULL source
-- were calibrated before the source field existed. Treat them as manual since
-- auto-save didn't exist yet.
UPDATE "videos"
SET "court_calibration_source" = 'manual'
WHERE "court_calibration_json" IS NOT NULL
  AND "court_calibration_source" IS NULL;
