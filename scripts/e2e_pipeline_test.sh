#!/usr/bin/env bash
# e2e_pipeline_test.sh — Full end-to-end pipeline test for IMG_2313.MOV
#
# Exercises: rally detection → player tracking → ball tracking →
#            contact detection → action classification → match analysis
#
# Usage: bash scripts/e2e_pipeline_test.sh
# Prerequisites: make dev (services running)

set -euo pipefail

# ─── Constants ────────────────────────────────────────────────────────────────
DB_URL="postgresql://postgres:postgres@localhost:5436/rallycut"
API_URL="http://localhost:3001"
MINIO_URL="http://localhost:9000"
VIDEO_SHORT_ID="0a383519"
EXPECTED_RALLIES=39

# ─── Helpers ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

phase_start_time=0
pipeline_start_time=$(date +%s)
phase_times=()

log()   { echo -e "${CYAN}[E2E]${NC} $*"; }
ok()    { echo -e "${GREEN}[OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; }
phase() {
  if [[ $phase_start_time -ne 0 ]]; then
    local elapsed=$(( $(date +%s) - phase_start_time ))
    phase_times+=("${prev_phase}|${elapsed}")
  fi
  prev_phase="$1"
  phase_start_time=$(date +%s)
  echo ""
  echo -e "${BOLD}════════════════════════════════════════════════════════════════${NC}"
  echo -e "${BOLD}  $1${NC}"
  echo -e "${BOLD}════════════════════════════════════════════════════════════════${NC}"
}

elapsed_since_phase() {
  echo $(( $(date +%s) - phase_start_time ))
}

psql_cmd() {
  psql "$DB_URL" -t -A -c "$1" 2>/dev/null
}

# ─── Phase 0: Preflight Checks ───────────────────────────────────────────────
phase "Phase 0: Preflight Checks"

# Check required tools
for tool in psql curl jq node; do
  if ! command -v "$tool" &>/dev/null; then
    fail "$tool not found in PATH"
    exit 1
  fi
done
ok "Required tools available (psql, curl, jq, node)"

# Test PostgreSQL
if psql_cmd "SELECT 1" | grep -q 1; then
  ok "PostgreSQL connected"
else
  fail "Cannot connect to PostgreSQL at localhost:5436"
  exit 1
fi

# Test API
if curl -sf "${API_URL}/health" >/dev/null 2>&1; then
  ok "API healthy at ${API_URL}"
else
  fail "API not responding at ${API_URL}"
  exit 1
fi

# Test MinIO
if curl -sf "${MINIO_URL}/minio/health/live" >/dev/null 2>&1; then
  ok "MinIO healthy at ${MINIO_URL}"
else
  warn "MinIO not responding (may be fine if using S3)"
fi

# Look up video
VIDEO_ID=$(psql_cmd "SELECT id FROM videos WHERE id LIKE '${VIDEO_SHORT_ID}%' LIMIT 1")
if [[ -z "$VIDEO_ID" ]]; then
  fail "Video with short ID '${VIDEO_SHORT_ID}' not found in DB"
  exit 1
fi
ok "Video found: ${VIDEO_ID}"

# Get content hash
CONTENT_HASH=$(psql_cmd "SELECT content_hash FROM videos WHERE id = '${VIDEO_ID}'")
log "Content hash: ${CONTENT_HASH}"

# Get current state
CURRENT_STATUS=$(psql_cmd "SELECT status FROM videos WHERE id = '${VIDEO_ID}'")
CURRENT_RALLIES=$(psql_cmd "SELECT COUNT(*) FROM rallies WHERE video_id = '${VIDEO_ID}'")
log "Current state: status=${CURRENT_STATUS}, rallies=${CURRENT_RALLIES}"

# Derive visitor ID and user ID
USER_ID=$(psql_cmd "SELECT user_id FROM videos WHERE id = '${VIDEO_ID}'")
if [[ -z "$USER_ID" ]]; then
  fail "Could not determine user_id for video"
  exit 1
fi
VISITOR_ID=$(psql_cmd "SELECT visitor_id FROM anonymous_identities WHERE user_id = '${USER_ID}' LIMIT 1")
if [[ -z "$VISITOR_ID" ]]; then
  fail "Could not find visitor_id for user ${USER_ID}"
  exit 1
fi
ok "User ID: ${USER_ID}"
ok "Visitor ID: ${VISITOR_ID}"

# Read JWT secret from api/.env
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
JWT_SECRET=$(grep '^AUTH_JWT_SECRET=' "${PROJECT_DIR}/api/.env" | cut -d= -f2-)
if [[ -z "$JWT_SECRET" ]]; then
  fail "AUTH_JWT_SECRET not found in api/.env"
  exit 1
fi

# Generate JWT
JWT=$(cd "${PROJECT_DIR}/api" && node -e "
  const jwt = require('jsonwebtoken');
  process.stdout.write(jwt.sign({sub:'${USER_ID}'}, '${JWT_SECRET}', {expiresIn:'4h'}));
")
ok "JWT generated (expires in 4h)"

# ─── Phase 1: Database Reset ─────────────────────────────────────────────────
phase "Phase 1: Database Reset"

log "Deleting child records for video ${VIDEO_ID}..."

# Player tracks
COUNT=$(psql_cmd "DELETE FROM player_tracks WHERE rally_id IN (SELECT id FROM rallies WHERE video_id = '${VIDEO_ID}') RETURNING 1" | wc -l | tr -d ' ')
log "  player_tracks: ${COUNT} deleted"

# Highlight rallies
COUNT=$(psql_cmd "DELETE FROM highlight_rallies WHERE rally_id IN (SELECT id FROM rallies WHERE video_id = '${VIDEO_ID}') RETURNING 1" | wc -l | tr -d ' ')
log "  highlight_rallies: ${COUNT} deleted"

# Rally camera edits (cascades to camera_keyframes)
COUNT=$(psql_cmd "DELETE FROM rally_camera_edits WHERE rally_id IN (SELECT id FROM rallies WHERE video_id = '${VIDEO_ID}') RETURNING 1" | wc -l | tr -d ' ')
log "  rally_camera_edits: ${COUNT} deleted"

# Rallies
COUNT=$(psql_cmd "DELETE FROM rallies WHERE video_id = '${VIDEO_ID}' RETURNING 1" | wc -l | tr -d ' ')
log "  rallies: ${COUNT} deleted"

# Detection jobs (by content hash — detection caches by hash)
COUNT=$(psql_cmd "DELETE FROM rally_detection_jobs WHERE content_hash = '${CONTENT_HASH}' RETURNING 1" | wc -l | tr -d ' ')
log "  rally_detection_jobs: ${COUNT} deleted"

# Batch tracking jobs
COUNT=$(psql_cmd "DELETE FROM batch_tracking_jobs WHERE video_id = '${VIDEO_ID}' RETURNING 1" | wc -l | tr -d ' ')
log "  batch_tracking_jobs: ${COUNT} deleted"

# Rally confirmations
COUNT=$(psql_cmd "DELETE FROM rally_confirmations WHERE video_id = '${VIDEO_ID}' RETURNING 1" | wc -l | tr -d ' ')
log "  rally_confirmations: ${COUNT} deleted"

# Reset video status
psql_cmd "UPDATE videos SET status = 'UPLOADED', match_analysis_json = NULL, match_stats_json = NULL WHERE id = '${VIDEO_ID}'"
ok "Video status reset to UPLOADED"

# Verify clean state
RALLY_COUNT=$(psql_cmd "SELECT COUNT(*) FROM rallies WHERE video_id = '${VIDEO_ID}'")
TRACK_COUNT=$(psql_cmd "SELECT COUNT(*) FROM player_tracks pt JOIN rallies r ON pt.rally_id = r.id WHERE r.video_id = '${VIDEO_ID}'")
ok "Clean state verified: ${RALLY_COUNT} rallies, ${TRACK_COUNT} tracks"

# ─── Phase 2: Rally Detection ────────────────────────────────────────────────
phase "Phase 2: Rally Detection"

log "Triggering rally detection..."
DETECT_RESP=$(curl -sf -X POST "${API_URL}/v1/videos/${VIDEO_ID}/detect-rallies" \
  -H "Authorization: Bearer ${JWT}" \
  -H "x-visitor-id: ${VISITOR_ID}" \
  -H "Content-Type: application/json" \
  -d '{"model":"beach"}')

DETECT_STATUS=$(echo "$DETECT_RESP" | jq -r '.status // "unknown"')
DETECT_JOB_ID=$(echo "$DETECT_RESP" | jq -r '.jobId // "unknown"')
log "Detection triggered: jobId=${DETECT_JOB_ID}, status=${DETECT_STATUS}"

# Poll detection status
LAST_PROGRESS=""
while true; do
  sleep 10
  STATUS_RESP=$(curl -sf "${API_URL}/v1/videos/${VIDEO_ID}/detection-status" \
    -H "x-visitor-id: ${VISITOR_ID}" || echo '{}')

  JOB_STATUS=$(echo "$STATUS_RESP" | jq -r '.job.status // "unknown"')
  PROGRESS=$(echo "$STATUS_RESP" | jq -r '.job.progress // 0')
  MSG=$(echo "$STATUS_RESP" | jq -r '.job.progressMessage // ""')
  ELAPSED=$(elapsed_since_phase)

  PROGRESS_STR="[DETECT] ${PROGRESS}%"
  [[ -n "$MSG" ]] && PROGRESS_STR+=" — ${MSG}"
  PROGRESS_STR+=" (${ELAPSED}s)"

  if [[ "$PROGRESS_STR" != "$LAST_PROGRESS" ]]; then
    log "$PROGRESS_STR"
    LAST_PROGRESS="$PROGRESS_STR"
  fi

  if [[ "$JOB_STATUS" == "COMPLETED" ]]; then
    ok "Detection completed in ${ELAPSED}s"
    break
  elif [[ "$JOB_STATUS" == "FAILED" ]]; then
    ERR=$(echo "$STATUS_RESP" | jq -r '.job.errorMessage // "unknown"')
    fail "Detection failed: ${ERR}"
    exit 1
  fi
done

# Report detection results
DETECTED_RALLIES=$(psql_cmd "SELECT COUNT(*) FROM rallies WHERE video_id = '${VIDEO_ID}'")
AVG_DURATION=$(psql_cmd "SELECT ROUND(AVG((end_ms - start_ms) / 1000.0)::numeric, 1) FROM rallies WHERE video_id = '${VIDEO_ID}'")
log "Detected ${DETECTED_RALLIES} rallies (expected ~${EXPECTED_RALLIES}), avg duration ${AVG_DURATION}s"

if [[ "$DETECTED_RALLIES" -lt 30 ]]; then
  warn "Detected significantly fewer rallies than expected (${DETECTED_RALLIES} vs ${EXPECTED_RALLIES})"
elif [[ "$DETECTED_RALLIES" -gt 50 ]]; then
  warn "Detected significantly more rallies than expected (${DETECTED_RALLIES} vs ${EXPECTED_RALLIES})"
else
  ok "Rally count within expected range"
fi

# ─── Phase 3: Batch Tracking ─────────────────────────────────────────────────
phase "Phase 3: Batch Tracking (player + ball + contacts + actions)"

log "Triggering batch tracking for all ${DETECTED_RALLIES} rallies..."
TRACK_RESP=$(curl -sf -X POST "${API_URL}/v1/videos/${VIDEO_ID}/track-all-rallies" \
  -H "x-visitor-id: ${VISITOR_ID}" \
  -H "Content-Type: application/json")

TRACK_JOB_ID=$(echo "$TRACK_RESP" | jq -r '.jobId // "unknown"')
TOTAL_RALLIES=$(echo "$TRACK_RESP" | jq -r '.totalRallies // 0')
log "Batch tracking triggered: jobId=${TRACK_JOB_ID}, totalRallies=${TOTAL_RALLIES}"

# Poll batch tracking status
LAST_PROGRESS=""
while true; do
  sleep 30
  STATUS_RESP=$(curl -sf "${API_URL}/v1/videos/${VIDEO_ID}/batch-tracking-status" \
    -H "x-visitor-id: ${VISITOR_ID}" || echo '{}')

  BATCH_STATUS=$(echo "$STATUS_RESP" | jq -r '.status // "unknown"')
  COMPLETED=$(echo "$STATUS_RESP" | jq -r '.completedRallies // 0')
  FAILED=$(echo "$STATUS_RESP" | jq -r '.failedRallies // 0')
  CURRENT=$(echo "$STATUS_RESP" | jq -r '.currentRallyId // ""')
  ELAPSED=$(elapsed_since_phase)
  DONE=$(( COMPLETED + FAILED ))

  # Compute per-rally timing
  RATE=""
  if [[ "$DONE" -gt 0 ]]; then
    PER_RALLY=$(( ELAPSED / DONE ))
    REMAINING=$(( (TOTAL_RALLIES - DONE) * PER_RALLY ))
    RATE=" (~${PER_RALLY}s/rally, ~${REMAINING}s remaining)"
  fi

  PROGRESS_STR="[TRACK] ${DONE}/${TOTAL_RALLIES} done, ${FAILED} failed | ${ELAPSED}s elapsed${RATE}"

  if [[ "$PROGRESS_STR" != "$LAST_PROGRESS" ]]; then
    log "$PROGRESS_STR"
    LAST_PROGRESS="$PROGRESS_STR"
  fi

  if [[ "$BATCH_STATUS" == "completed" || "$BATCH_STATUS" == "failed" ]]; then
    if [[ "$FAILED" -gt 0 ]]; then
      warn "Batch tracking finished in ${ELAPSED}s: ${COMPLETED} completed, ${FAILED} failed"
    else
      ok "Batch tracking completed in ${ELAPSED}s: ${COMPLETED}/${TOTAL_RALLIES} rallies"
    fi
    break
  fi
done

# Wait for match analysis (runs asynchronously after batch job status updates)
log "Waiting for match analysis to complete..."
MA_WAIT=0
MA_MAX=180  # 3 minutes max
while [[ $MA_WAIT -lt $MA_MAX ]]; do
  HAS_MA=$(psql_cmd "SELECT CASE WHEN match_analysis_json IS NOT NULL THEN 'YES' ELSE 'NO' END FROM videos WHERE id = '${VIDEO_ID}'")
  if [[ "$HAS_MA" == "YES" ]]; then
    ok "Match analysis completed (waited ${MA_WAIT}s)"
    break
  fi
  sleep 10
  MA_WAIT=$(( MA_WAIT + 10 ))
  log "Waiting for match analysis... (${MA_WAIT}s)"
done
if [[ "$HAS_MA" != "YES" ]]; then
  warn "Match analysis not generated after ${MA_MAX}s — may need manual investigation"
fi

# ─── Phase 4: Summary Report ─────────────────────────────────────────────────
phase "Phase 4: Summary Report"

echo ""
echo -e "${BOLD}┌──────────────────────────────────────────────────┐${NC}"
echo -e "${BOLD}│              E2E Pipeline Results                │${NC}"
echo -e "${BOLD}└──────────────────────────────────────────────────┘${NC}"

# Detection summary
echo ""
echo -e "${BOLD}1. Rally Detection${NC}"
RALLY_STATS=$(psql_cmd "
  SELECT COUNT(*),
         ROUND(AVG((end_ms - start_ms) / 1000.0)::numeric, 1),
         ROUND(MIN((end_ms - start_ms) / 1000.0)::numeric, 1),
         ROUND(MAX((end_ms - start_ms) / 1000.0)::numeric, 1)
  FROM rallies WHERE video_id = '${VIDEO_ID}'
")
IFS='|' read -r R_COUNT R_AVG R_MIN R_MAX <<< "$RALLY_STATS"
echo "   Rallies detected:  ${R_COUNT} (expected ~${EXPECTED_RALLIES})"
echo "   Duration (avg):    ${R_AVG}s (min ${R_MIN}s, max ${R_MAX}s)"

# Player tracking summary
echo ""
echo -e "${BOLD}2. Player Tracking${NC}"
TRACK_STATS=$(psql_cmd "
  SELECT
    COUNT(*) FILTER (WHERE pt.status = 'COMPLETED'),
    COUNT(*) FILTER (WHERE pt.status = 'FAILED'),
    COUNT(*) FILTER (WHERE pt.status = 'PENDING')
  FROM rallies r
  LEFT JOIN player_tracks pt ON pt.rally_id = r.id
  WHERE r.video_id = '${VIDEO_ID}'
")
IFS='|' read -r T_COMPLETED T_FAILED T_PENDING <<< "$TRACK_STATS"
echo "   Completed: ${T_COMPLETED}  |  Failed: ${T_FAILED}  |  Pending: ${T_PENDING}"

# Player tracking metrics from dedicated columns
PLAYER_METRICS=$(psql_cmd "
  SELECT
    ROUND(AVG(detection_rate)::numeric, 3),
    ROUND(AVG(avg_player_count)::numeric, 1),
    ROUND(AVG(processing_time_ms / 1000.0)::numeric, 1)
  FROM player_tracks pt
  JOIN rallies r ON pt.rally_id = r.id
  WHERE r.video_id = '${VIDEO_ID}' AND pt.status = 'COMPLETED'
")
IFS='|' read -r P_DETRATE P_AVGPLAYERS P_AVGTIME <<< "$PLAYER_METRICS"
echo "   Avg detection rate:  ${P_DETRATE:-N/A}"
echo "   Avg players/frame:   ${P_AVGPLAYERS:-N/A}"
echo "   Avg processing time: ${P_AVGTIME:-N/A}s/rally"

# Ball tracking summary
echo ""
echo -e "${BOLD}3. Ball Tracking${NC}"
BALL_STATS=$(psql_cmd "
  SELECT
    COUNT(*) FILTER (WHERE ball_positions_json IS NOT NULL AND ball_positions_json::text NOT IN ('null', '[]')),
    COUNT(*)
  FROM player_tracks pt
  JOIN rallies r ON pt.rally_id = r.id
  WHERE r.video_id = '${VIDEO_ID}' AND pt.status = 'COMPLETED'
")
IFS='|' read -r B_WITH B_TOTAL <<< "$BALL_STATS"
echo "   Rallies with ball data: ${B_WITH}/${B_TOTAL}"

# Avg ball positions per rally
AVG_BALL_POS=$(psql_cmd "
  SELECT ROUND(AVG(jsonb_array_length(ball_positions_json))::numeric, 0)
  FROM player_tracks pt
  JOIN rallies r ON pt.rally_id = r.id
  WHERE r.video_id = '${VIDEO_ID}' AND pt.status = 'COMPLETED'
    AND ball_positions_json IS NOT NULL AND ball_positions_json::text NOT IN ('null', '[]')
")
echo "   Avg ball positions/rally: ~${AVG_BALL_POS:-N/A}"

# Contact & action summary
echo ""
echo -e "${BOLD}4. Contacts & Actions${NC}"
# contacts_json is {numContacts, contacts: [...]}
# actions_json is {numContacts, actions: [{action, frame, ...}, ...]}
CONTACT_STATS=$(psql_cmd "
  SELECT
    COUNT(*) FILTER (WHERE actions_json IS NOT NULL AND actions_json::text != 'null'),
    COALESCE(SUM((actions_json->>'numContacts')::int) FILTER (WHERE actions_json IS NOT NULL AND actions_json::text != 'null'), 0)
  FROM player_tracks pt
  JOIN rallies r ON pt.rally_id = r.id
  WHERE r.video_id = '${VIDEO_ID}' AND pt.status = 'COMPLETED'
")
IFS='|' read -r C_RALLIES C_TOTAL <<< "$CONTACT_STATS"
echo "   Rallies with actions: ${C_RALLIES}"
echo "   Total contacts: ${C_TOTAL:-0}"

# Action distribution from actions_json->'actions' array
if [[ "${C_TOTAL:-0}" -gt 0 ]]; then
  echo "   Action distribution:"
  psql_cmd "
    SELECT action_type, COUNT(*) as cnt
    FROM (
      SELECT jsonb_array_elements(actions_json->'actions')->>'action' as action_type
      FROM player_tracks pt
      JOIN rallies r ON pt.rally_id = r.id
      WHERE r.video_id = '${VIDEO_ID}' AND pt.status = 'COMPLETED'
        AND actions_json IS NOT NULL AND actions_json::text != 'null'
    ) actions
    WHERE action_type IS NOT NULL
    GROUP BY action_type
    ORDER BY cnt DESC
  " | while IFS='|' read -r ACTION CNT; do
    printf "     %-12s %s\n" "$ACTION" "$CNT"
  done
fi

# Match analysis
echo ""
echo -e "${BOLD}5. Match Analysis${NC}"
MATCH_ANALYSIS=$(psql_cmd "
  SELECT
    CASE WHEN match_analysis_json IS NOT NULL THEN 'YES' ELSE 'NO' END,
    CASE WHEN match_stats_json IS NOT NULL THEN 'YES' ELSE 'NO' END
  FROM videos WHERE id = '${VIDEO_ID}'
")
IFS='|' read -r HAS_ANALYSIS HAS_STATS <<< "$MATCH_ANALYSIS"
echo "   match_analysis_json: ${HAS_ANALYSIS}"
echo "   match_stats_json:    ${HAS_STATS}"

# Failed rallies
echo ""
echo -e "${BOLD}6. Failed Rallies${NC}"
FAILED_LIST=$(psql_cmd "
  SELECT r.id, pt.error
  FROM rallies r
  JOIN player_tracks pt ON pt.rally_id = r.id
  WHERE r.video_id = '${VIDEO_ID}' AND pt.status = 'FAILED'
")
if [[ -z "$FAILED_LIST" ]]; then
  ok "No failed rallies"
else
  echo "$FAILED_LIST" | while IFS='|' read -r RID ERR; do
    warn "  ${RID}: ${ERR}"
  done
fi

# Timing
echo ""
echo -e "${BOLD}7. Timing${NC}"
# Record final phase
if [[ $phase_start_time -ne 0 ]]; then
  elapsed=$(( $(date +%s) - phase_start_time ))
  phase_times+=("${prev_phase}|${elapsed}")
fi
TOTAL_ELAPSED=$(( $(date +%s) - pipeline_start_time ))

for pt in "${phase_times[@]}"; do
  IFS='|' read -r PNAME PSEC <<< "$pt"
  PMIN=$(( PSEC / 60 ))
  PSEC_R=$(( PSEC % 60 ))
  printf "   %-45s %dm %02ds\n" "$PNAME" "$PMIN" "$PSEC_R"
done
echo "   ─────────────────────────────────────────────────"
TMIN=$(( TOTAL_ELAPSED / 60 ))
TSEC=$(( TOTAL_ELAPSED % 60 ))
printf "   %-45s ${BOLD}%dm %02ds${NC}\n" "Total" "$TMIN" "$TSEC"

# Final status
echo ""
if [[ "${T_FAILED}" -gt 0 ]]; then
  warn "Pipeline completed with ${T_FAILED} failed rallies"
elif [[ "${HAS_ANALYSIS}" == "NO" ]]; then
  warn "Pipeline completed but match analysis was not generated"
else
  ok "Pipeline completed successfully!"
fi

echo ""
log "Verify in web UI: open http://localhost:3000 and check video for rally overlays"
