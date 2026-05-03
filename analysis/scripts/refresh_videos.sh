#!/usr/bin/env bash
# Refresh matcher decisions on a list of videos by resetting cached state
# and re-running match-players + remap-track-ids end-to-end. Use when you
# want to see honest matcher output on the current code (e.g., to spot
# bad-pattern videos for further investigation).
#
# Why this is needed: the editor's "Re-analyze" button intentionally
# preserves the assignmentAnchor cache (so cross-rally edits don't cascade
# unexpectedly). For developer/diagnostic refreshes you want fresh matcher
# decisions, which requires resetting state first. See
# `feedback_validation_clean_state.md` and `pattern_e_corruption_2026_05_03.md`.
#
# This script is the bulk version of `eval_cross_fixture.sh` minus the
# GT-measurement step (so it works on videos without GT labels).
#
# Usage:
#   scripts/refresh_videos.sh <video_id> [<video_id> ...]
#   scripts/refresh_videos.sh --all-with-gt
#   scripts/refresh_videos.sh --measure-gt <video_id> ...    # also run measure_pid_accuracy
set -euo pipefail

usage() {
    sed -n '2,/^set -euo/p' "$0" | sed 's/^# \?//' | sed '/^set -euo/d'
    exit "${1:-0}"
}

MEASURE_GT=0
VIDEO_IDS=()
USE_GT_VIDEOS=0
for arg in "$@"; do
    case "$arg" in
        --help|-h) usage ;;
        --measure-gt) MEASURE_GT=1 ;;
        --all-with-gt) USE_GT_VIDEOS=1 ;;
        *) VIDEO_IDS+=("$arg") ;;
    esac
done

if [ "$USE_GT_VIDEOS" = "1" ]; then
    if [ ${#VIDEO_IDS[@]} -ne 0 ]; then
        echo "Error: --all-with-gt is exclusive with explicit video IDs" >&2
        exit 1
    fi
    while IFS= read -r vid; do
        VIDEO_IDS+=("$vid")
    done < <(uv run python -c "
from rallycut.evaluation.tracking.db import get_connection
with get_connection() as c, c.cursor() as cur:
    cur.execute('SELECT id FROM videos WHERE player_matching_gt_json IS NOT NULL ORDER BY id')
    for row in cur.fetchall():
        print(row[0])
")
fi

if [ ${#VIDEO_IDS[@]} -eq 0 ]; then
    echo "No video IDs provided. Use --help for usage." >&2
    exit 1
fi

echo "Refreshing matcher decisions on ${#VIDEO_IDS[@]} video(s)..."
echo ""

for vid in "${VIDEO_IDS[@]}"; do
    short=${vid:0:8}
    echo "==================== ${short} ===================="

    echo "[${short}] Resetting matcher state..."
    uv run python scripts/reset_matcher_state.py "$vid"

    echo "[${short}] Running match-players..."
    uv run rallycut match-players "$vid"

    echo "[${short}] Running remap-track-ids..."
    uv run rallycut remap-track-ids "$vid"

    if [ "$MEASURE_GT" = "1" ]; then
        echo "[${short}] Measuring PID accuracy (requires GT)..."
        uv run python scripts/measure_pid_accuracy.py "$vid" || \
            echo "  (measure failed — likely no GT for this video)"
    fi

    echo ""
done

echo "==================== Done ===================="
echo ""
echo "Refresh the editor on each video to see updated matcher results."
