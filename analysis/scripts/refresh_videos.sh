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
# Pending-edit contract: per `api/CLAUDE.md`, rally edits queue markers in
# `Video.pendingAnalysisEditsJson` and the API service `runMatchAnalysis`
# consumes them via `consumePendingEdits` before re-running stages. This
# CLI path bypasses that. To prevent silent stale state (editor showing
# extended bounds while DB AFM reflects pre-extend), the script aborts if
# the video has un-consumed `extend|create|merge|refCrop` edits — those
# need the full pipeline (retrack + repair-identities + reattribute) and
# cannot be applied here. CLI-safe edits (scalar/shorten/delete/split)
# are cleared after a successful refresh.
#
# Usage:
#   scripts/refresh_videos.sh <video_id> [<video_id> ...]
#   scripts/refresh_videos.sh --all-with-gt
#   scripts/refresh_videos.sh --measure-gt <video_id> ...    # also run measure_pid_accuracy
#   scripts/refresh_videos.sh --skip-pending-check <video_id>  # bypass the pending-edits gate (dangerous)
set -euo pipefail

usage() {
    sed -n '2,/^set -euo/p' "$0" | sed 's/^# \?//' | sed '/^set -euo/d'
    exit "${1:-0}"
}

MEASURE_GT=0
VIDEO_IDS=()
USE_GT_VIDEOS=0
SKIP_PENDING_CHECK=0
for arg in "$@"; do
    case "$arg" in
        --help|-h) usage ;;
        --measure-gt) MEASURE_GT=1 ;;
        --all-with-gt) USE_GT_VIDEOS=1 ;;
        --skip-pending-check) SKIP_PENDING_CHECK=1 ;;
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

    if [ "$SKIP_PENDING_CHECK" != "1" ]; then
        echo "[${short}] Checking pendingAnalysisEditsJson..."
        if ! uv run python scripts/check_and_consume_pending_edits.py --check "$vid"; then
            echo ""
            echo "[${short}] ABORTING: pending edits require the full match-analysis pipeline."
            echo "Use --skip-pending-check to override (will leave queue un-consumed)."
            exit 1
        fi
    fi

    echo "[${short}] Resetting matcher state..."
    uv run python scripts/reset_matcher_state.py "$vid"

    echo "[${short}] Running match-players (blind, --no-ref-crops)..."
    uv run rallycut match-players --no-ref-crops "$vid"

    echo "[${short}] Running remap-track-ids..."
    uv run rallycut remap-track-ids "$vid"

    if [ "$SKIP_PENDING_CHECK" != "1" ]; then
        echo "[${short}] Consuming CLI-safe pending edits..."
        uv run python scripts/check_and_consume_pending_edits.py --consume "$vid" || true
    fi

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
