-- Rename "spike" to "attack" in action ground truth and auto-detected actions JSON

-- Update action_ground_truth_json: array of {frame, action, playerTrackId, ...}
UPDATE player_tracks
SET action_ground_truth_json = (
    SELECT jsonb_agg(
        CASE
            WHEN elem->>'action' = 'spike'
            THEN jsonb_set(elem, '{action}', '"attack"')
            ELSE elem
        END
    )
    FROM jsonb_array_elements(action_ground_truth_json) AS elem
)
WHERE action_ground_truth_json IS NOT NULL
  AND action_ground_truth_json::text LIKE '%"spike"%';

-- Update actions_json: {actions: [{action, frame, ...}, ...], ...}
UPDATE player_tracks
SET actions_json = jsonb_set(
    actions_json,
    '{actions}',
    (
        SELECT jsonb_agg(
            CASE
                WHEN elem->>'action' = 'spike'
                THEN jsonb_set(elem, '{action}', '"attack"')
                ELSE elem
            END
        )
        FROM jsonb_array_elements(actions_json->'actions') AS elem
    )
)
WHERE actions_json IS NOT NULL
  AND actions_json::text LIKE '%"spike"%';
