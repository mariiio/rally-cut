#!/usr/bin/env bash
# Cross-fixture matcher evaluation — clean-state per measurement.
#
# Usage:
#   scripts/eval_cross_fixture.sh                # default panel
#   scripts/eval_cross_fixture.sh --flag-on      # ENABLE_WITHIN_RALLY_REPAIR=1
#   scripts/eval_cross_fixture.sh <vid> <vid>    # custom video list
#
# For each video:
#   1. Reset assignmentAnchor + canonical_pid_map_json (mechanical
#      enforcement of feedback_validation_clean_state.md).
#   2. Run match-players (with the configured flag).
#   3. Run remap-track-ids.
#   4. Print PERMUTED + ID-stability summary from measure_pid_accuracy.
#
# This wrapper exists because forgetting the reset step yields
# non-deterministic measurements. The reset is mandatory for any
# A/B comparison between matcher configurations — see
# `feedback_validation_clean_state.md` for the original incident.
set -euo pipefail

# Default panel: the 4 originally GT-labeled fixtures. Swap or expand
# by passing video IDs as positional args.
DEFAULT_PANEL=(
    "5c756c41-1cc1-4486-a95c-97398912cfbe"
    "b5fb0594-d64f-4a0d-bad9-de8fc36414d0"
    "854bb250-3e91-47d2-944d-f62413e3cf45"
    "7d77980f-3006-40e0-adc0-db491a5bb659"
)

FLAG_ON=0
VIDEO_IDS=()
for arg in "$@"; do
    case "$arg" in
        --flag-on)
            FLAG_ON=1
            ;;
        --help|-h)
            head -20 "$0" | sed 's/^# \?//'
            exit 0
            ;;
        *)
            VIDEO_IDS+=("$arg")
            ;;
    esac
done

if [ ${#VIDEO_IDS[@]} -eq 0 ]; then
    VIDEO_IDS=("${DEFAULT_PANEL[@]}")
fi

if [ "$FLAG_ON" = "1" ]; then
    echo "==== Configuration: ENABLE_WITHIN_RALLY_REPAIR=1 ===="
    export ENABLE_WITHIN_RALLY_REPAIR=1
else
    echo "==== Configuration: BASELINE (no flags) ===="
    unset ENABLE_WITHIN_RALLY_REPAIR
fi

for vid in "${VIDEO_IDS[@]}"; do
    echo ""
    echo "############# ${vid:0:8} #############"

    # Step 1: Reset cache state to a known-clean baseline.
    uv run python scripts/reset_matcher_state.py "$vid" > /dev/null

    # Step 2: Run match-players.
    uv run rallycut match-players "$vid" > /tmp/eval_mp.log 2>&1

    # Step 3: Run remap-track-ids.
    uv run rallycut remap-track-ids "$vid" > /tmp/eval_remap.log 2>&1

    # Step 4: Measure + print summary.
    uv run python scripts/measure_pid_accuracy.py "$vid" 2>&1 \
        | grep -E "OVERALL|permutation|QUALITY METRIC|distinct matcher PID|AVERAGE distinct"
done

echo ""
echo "==== Done. ===="
