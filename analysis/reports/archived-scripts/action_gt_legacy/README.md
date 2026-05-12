# action_gt_legacy/

Scripts kept as references for the JSON-array era of `player_tracks.action_ground_truth_json`.
Superseded by the `rally_action_ground_truth` table introduced 2026-05-12; see
`docs/superpowers/specs/2026-05-12-action-gt-decouple-design.md`.

These scripts are no longer executable after Task 21 (`ALTER TABLE player_tracks DROP COLUMN
action_ground_truth_json`) completes. They are preserved for historical reference only.

| Script | Purpose |
|---|---|
| `backfill_action_gt_trackid.py` | Backfilled `trackId` field on legacy `action_ground_truth_json` labels; superseded by `rally_action_ground_truth` table |
| `resave_ball_for_action_gt.py` | Re-saved ball positions for rallies with `action_ground_truth_json`; superseded by `rally_action_ground_truth` table |
