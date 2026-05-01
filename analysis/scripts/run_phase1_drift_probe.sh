#!/usr/bin/env bash
# Phase-1 H-Profile-Drift probe: run match-players + remap + verdict tool on
# the 4-fixture panel, twice per fixture (baseline vs EXPERIMENTAL_DROP_PROFILE_EMA).
# Writes per-fixture verdict logs and probe sidecars under
# analysis/reports/profile_drift_probe/.
#
# Final restoration: re-runs match-players in baseline mode on each fixture
# so the DB ends in a clean baseline state (idempotent via pre_remap_state_json).
set -euo pipefail

cd "$(dirname "$0")/.."

OUTDIR="reports/profile_drift_probe"
mkdir -p "$OUTDIR"

VIDEOS=(
    "b5fb0594-d64f-4a0d-bad9-de8fc36414d0"
    "5c756c41-1cc1-4486-a95c-97398912cfbe"
    "854bb250-3e91-47d2-944d-f62413e3cf45"
    "7d77980f-3006-40e0-adc0-db491a5bb659"
)

run_config() {
    local video_id="$1"
    local mode="$2"   # baseline or dropema
    local flag="$3"   # 0 or 1
    local short="${video_id:0:8}"
    echo "==== $short / $mode (EXPERIMENTAL_DROP_PROFILE_EMA=$flag) ===="
    # ENABLE_BBOX_SWAP_DETECTION=0 to match locked panel state
    # (panel_visual_verdict_2026_05_01.md). W4 default is ON via env, but
    # the locked baseline ran with it explicitly OFF.
    # --no-ref-crops forces the blind path (MatchSolver). 5c756c41 and
    # 7d77980f have ref crops in DB but the user constraint
    # (feedback_blind_regime_goal.md) means production runs blind.
    MATCH_PLAYERS_PROBE=1 EXPERIMENTAL_DROP_PROFILE_EMA="$flag" \
    ENABLE_BBOX_SWAP_DETECTION=0 ENABLE_POSITION_JUMP_SWAP=0 \
        uv run rallycut match-players "$video_id" --no-ref-crops \
        > "$OUTDIR/${short}_${mode}_match_players.log" 2>&1
    ENABLE_BBOX_SWAP_DETECTION=0 ENABLE_POSITION_JUMP_SWAP=0 \
        uv run rallycut remap-track-ids "$video_id" \
        > "$OUTDIR/${short}_${mode}_remap.log" 2>&1
}

verdict_snapshot() {
    local mode="$1"
    echo "==== verdict ($mode) ===="
    uv run python scripts/panel_verdict_per_frame.py \
        > "$OUTDIR/verdict_${mode}.txt" 2>&1
    cat "$OUTDIR/verdict_${mode}.txt"
}

# Phase 1a — baseline pass (all 4 fixtures, flag OFF)
for v in "${VIDEOS[@]}"; do
    run_config "$v" "baseline" "0"
done
verdict_snapshot "baseline"

# Phase 1b — counterfactual pass (all 4 fixtures, flag ON)
for v in "${VIDEOS[@]}"; do
    run_config "$v" "dropema" "1"
done
verdict_snapshot "dropema"

# Phase 1c — restore DB to baseline (so leaving session in clean state)
for v in "${VIDEOS[@]}"; do
    run_config "$v" "baseline_restore" "0"
done
verdict_snapshot "baseline_restore"

echo "==== Phase 1 complete. Outputs under $OUTDIR/ ===="
ls "$OUTDIR/" | sort
