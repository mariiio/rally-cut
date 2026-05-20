-- v9 SOTA workstream: per-endpoint net-top GT (replaces the v7/v8 scalar).
-- Adds two Float columns + a JSON column for per-endpoint visibility flags.
-- The legacy scalar `court_calibration_net_top_y` stays — every v9 write
-- recomputes it as (L+R)/2 so older readers see a consistent midpoint.
-- See analysis/docs/labeling/net_top_endpoints.md for the click convention.

ALTER TABLE "videos"
  ADD COLUMN "court_calibration_net_top_left_y"  DOUBLE PRECISION,
  ADD COLUMN "court_calibration_net_top_right_y" DOUBLE PRECISION,
  ADD COLUMN "court_calibration_net_top_endpoints_json" JSONB;

-- Backfill: for any video that already has a legacy scalar, treat it as a
-- horizontal label (left = right = scalar). Lets the v9 UI pre-seed both
-- handles at the user's prior label without losing information. Set the
-- visibility to {2,2} so the trainer treats these as confident.
UPDATE "videos"
SET "court_calibration_net_top_left_y"  = "court_calibration_net_top_y",
    "court_calibration_net_top_right_y" = "court_calibration_net_top_y",
    "court_calibration_net_top_endpoints_json" = '{"leftVisibility": 2, "rightVisibility": 2}'::jsonb
WHERE "court_calibration_net_top_y" IS NOT NULL;
