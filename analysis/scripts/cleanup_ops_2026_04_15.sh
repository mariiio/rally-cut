#!/usr/bin/env bash
# One-off ops cleanup — 2026-04-15
# Aligns DB stored labels with the day's shipped fixes:
#   - Options 1+2 (54605c6: serve hygiene + threshold alignment)
#   - Override guards (b59e264)
#   - F3b track-id prevention (1436811)
#
# Three steps with hard bail-out gates. If any step fails its gate, the
# script stops; nothing in the next step runs.
#
# What this DOESN'T do:
#   - Re-track jaja video (do that via web UI or separately — F3b will
#     skip locked rallies anyway, so it's a partial op best done by hand)
#   - Re-run the F3b/repair migration on more rallies (we're staying in
#     v4 state; F3c will handle the rest with proper safeguards)
#
# Usage:
#   cd analysis
#   bash scripts/cleanup_ops_2026_04_15.sh
#
# Total wall-clock: ~1.5–2 hours, mostly waiting on production_eval.

set -uo pipefail
# Note: NOT using `set -e` because we want to handle individual rally
# failures and keep going up to the bail-out threshold.

cd "$(dirname "$0")/.."  # analysis/

LOG_DIR="outputs"
TIMESTAMP="$(date +%Y-%m-%d-%H%M%S)"
SUMMARY_LOG="${LOG_DIR}/cleanup_summary_${TIMESTAMP}.log"

mkdir -p "${LOG_DIR}"

log() {
  echo "$@" | tee -a "${SUMMARY_LOG}"
}

log "=== RallyCut ops cleanup, started ${TIMESTAMP} ==="
log ""

# ---------------------------------------------------------------------- #
# Step 1: Bulk reattribute-actions
# ---------------------------------------------------------------------- #
log "=== Step 1: Bulk reattribute-actions (refresh stored team labels at 0.70 threshold) ==="
STEP1_LOG="${LOG_DIR}/bulk_reattribute_${TIMESTAMP}.log"

VIDEO_IDS="$(PGPASSWORD=postgres psql -h localhost -p 5436 -U postgres -d rallycut -t -A -c \
    "SELECT id FROM videos WHERE match_analysis_json IS NOT NULL ORDER BY id" 2>&1)"

if [ -z "${VIDEO_IDS}" ]; then
  log "  No videos with match_analysis_json found. Skipping step 1."
else
  TOTAL=$(printf "%s\n" "${VIDEO_IDS}" | grep -c '.')
  log "  ${TOTAL} videos to process"

  SUCCESS=0
  FAIL=0
  FAIL_LIST=""
  i=0
  for vid in ${VIDEO_IDS}; do
    i=$((i+1))
    printf "  [%d/%d] %s ... " "$i" "$TOTAL" "$vid" | tee -a "${STEP1_LOG}"
    if uv run rallycut reattribute-actions "$vid" --quiet >> "${STEP1_LOG}" 2>&1; then
      echo "OK" | tee -a "${STEP1_LOG}"
      SUCCESS=$((SUCCESS+1))
    else
      echo "FAIL" | tee -a "${STEP1_LOG}"
      FAIL=$((FAIL+1))
      FAIL_LIST="${FAIL_LIST} $vid"
    fi
  done

  log ""
  log "  Step 1 complete: ${SUCCESS} success, ${FAIL} failed"
  log "  Full log: ${STEP1_LOG}"
  if [ "${FAIL}" -gt 3 ]; then
    log ""
    log "  BAIL-OUT: more than 3 failures in step 1. Stopping."
    log "  Failed videos:${FAIL_LIST}"
    log ""
    log "  Investigate ${STEP1_LOG} before re-running. Steps 2 and 3 NOT executed."
    exit 1
  fi
fi

# ---------------------------------------------------------------------- #
# Step 2: Match-players on template-missing videos
# ---------------------------------------------------------------------- #
log ""
log "=== Step 2: Match-players on template-missing videos ==="
STEP2_LOG="${LOG_DIR}/match_players_missing_${TIMESTAMP}.log"

TEMPLATE_MISSING="$(PGPASSWORD=postgres psql -h localhost -p 5436 -U postgres -d rallycut -t -A -c \
    "SELECT id FROM videos
     WHERE match_analysis_json IS NULL
        OR NOT (match_analysis_json ? 'teamTemplates')
     ORDER BY id" 2>&1)"

if [ -z "${TEMPLATE_MISSING}" ]; then
  log "  No videos missing teamTemplates. Skipping step 2."
else
  TOTAL2=$(printf "%s\n" "${TEMPLATE_MISSING}" | grep -c '.')
  log "  ${TOTAL2} videos missing teamTemplates"

  SUCCESS2=0
  FAIL2=0
  FAIL_LIST2=""
  i=0
  for vid in ${TEMPLATE_MISSING}; do
    i=$((i+1))
    printf "  [%d/%d] %s ... " "$i" "$TOTAL2" "$vid" | tee -a "${STEP2_LOG}"
    if uv run rallycut match-players "$vid" --quiet >> "${STEP2_LOG}" 2>&1; then
      echo "OK" | tee -a "${STEP2_LOG}"
      SUCCESS2=$((SUCCESS2+1))
    else
      echo "FAIL" | tee -a "${STEP2_LOG}"
      FAIL2=$((FAIL2+1))
      FAIL_LIST2="${FAIL_LIST2} $vid"
    fi
  done

  log ""
  log "  Step 2 complete: ${SUCCESS2} success, ${FAIL2} failed"
  log "  Full log: ${STEP2_LOG}"
  if [ "${FAIL2}" -gt 1 ]; then
    log ""
    log "  BAIL-OUT: more than 1 failure in step 2. Stopping."
    log "  Failed videos:${FAIL_LIST2}"
    log ""
    log "  Investigate ${STEP2_LOG} before re-running. Step 3 NOT executed."
    exit 1
  fi
fi

# ---------------------------------------------------------------------- #
# Step 3: Final verification — re-audit + production_eval
# ---------------------------------------------------------------------- #
log ""
log "=== Step 3: Final verification ==="

AUDIT_OUT="${LOG_DIR}/action_anomaly_audit_post_cleanup_${TIMESTAMP}.md"
log "  Running audit -> ${AUDIT_OUT}"
if uv run python scripts/audit_action_sequence_anomalies.py \
    --skip-session 6f599a0e-b8ea-4bf0-a331-ce7d9ef88164 \
    --output "${AUDIT_OUT}" >> "${SUMMARY_LOG}" 2>&1; then
  log "  Audit OK"
else
  log "  Audit FAILED. Check ${SUMMARY_LOG} for details."
fi

EVAL_OUT="${LOG_DIR}/production_eval_post_cleanup_${TIMESTAMP}.log"
log "  Running production_eval (this takes ~5-10 min) -> ${EVAL_OUT}"
if uv run python scripts/production_eval.py --reruns 1 > "${EVAL_OUT}" 2>&1; then
  log "  production_eval OK"
else
  log "  production_eval FAILED. Check ${EVAL_OUT} for details."
fi

log ""
log "=== Cleanup complete. Summary: ${SUMMARY_LOG} ==="
log ""
log "Next steps:"
log "  - Compare ${AUDIT_OUT} against the pre-cleanup audit output to see bucket shifts."
log "  - Compare ${EVAL_OUT} latest run JSON in outputs/production_eval/ against the"
log "    pre-cleanup baseline to see metric shifts (use the python A/B snippet from MEMORY)."
log ""
log "What was NOT done in this script:"
log "  - jaja re-track (run via web UI 're-track all rallies' button or as a separate"
log "    track-players invocation)."
log "  - F3a / ambiguous-set / unlocked repair (deferred to F3c per memory)."
