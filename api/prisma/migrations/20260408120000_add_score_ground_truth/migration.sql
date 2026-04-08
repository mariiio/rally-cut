-- Session 5: score GT. Additive nullable columns, safe.
ALTER TABLE "rallies"
  ADD COLUMN "gt_serving_team" "ServingTeam",
  ADD COLUMN "gt_point_winner" "ServingTeam";
