#!/bin/bash
# Sub-lever 1 A/B driver: v11 baseline vs v12 chain-context-fallback ON.
#
# Phase 1: redetect all 32 trusted videos with SCORER_CHAIN_FALLBACK=0 (v11 baseline)
# Phase 2: measure attribution -> v11_baseline.json
# Phase 3: redetect all 32 with SCORER_CHAIN_FALLBACK=1 (v12 treatment)
# Phase 4: measure attribution -> v12_fallback_on.json + compare-to v11_baseline
# Phase 5: coherence audit on both states (uses trusted_29 script — fine, covers 29/32)
# Phase 6: re-run oracle decomposition for sanity check
#
# Runs from analysis/ dir; expects DB DSN in env or default localhost:5436.
# Total wall time estimate: ~60-90 min on this corpus.

set -euo pipefail

cd "$(dirname "$0")/.."  # analysis/

# Resolve trusted-32 names -> UUIDs once.
TRUSTED_NAMES="titi toto lulu wawa caco cece cici cuco gaga gigi kaka kiki keke koko kuku juju yeye gugu mame meme mimi moma mumu papa pepe pipi popo pupu veve vivi vovo haha"
echo "[setup] Resolving 32 video UUIDs..."
UUIDS=$(uv run python -c "
import os, psycopg
dsn = os.environ.get('DATABASE_URL', 'postgresql://postgres:postgres@localhost:5436/rallycut')
names = '$TRUSTED_NAMES'.split()
with psycopg.connect(dsn) as c:
    rows = c.execute('SELECT id FROM videos WHERE name = ANY(%s) ORDER BY name', [names]).fetchall()
print('\n'.join(str(r[0]) for r in rows))
")
N_UUIDS=$(echo "$UUIDS" | wc -l | tr -d ' ')
echo "[setup] Resolved $N_UUIDS UUIDs"
if [ "$N_UUIDS" -ne 32 ]; then
    echo "[ERROR] Expected 32 UUIDs, got $N_UUIDS" >&2
    exit 1
fi

run_redetect_loop () {
    local fallback_flag="$1"
    local label="$2"
    echo
    echo "========================================="
    echo "[$label] redetect 32 videos, SCORER_CHAIN_FALLBACK=$fallback_flag"
    echo "========================================="
    local i=0
    while IFS= read -r uid; do
        i=$((i+1))
        echo "  [$i/32] redetect $uid"
        SCORER_CHAIN_FALLBACK=$fallback_flag \
            USE_DYNAMIC_ATTRIBUTION_SCORER=1 \
            PYTHONUNBUFFERED=1 \
            uv run python -u scripts/redetect_all_actions.py --video "$uid" --apply
    done <<< "$UUIDS"
}

# Phase 1: v11 baseline redetect (SCORER_CHAIN_FALLBACK=0)
run_redetect_loop 0 "PHASE 1: v11 baseline"

# Phase 2: measure v11 baseline
echo
echo "========================================="
echo "[PHASE 2] measure attribution -> v11_baseline"
echo "========================================="
PYTHONUNBUFFERED=1 uv run python -u scripts/measure_attribution_trusted_31_2026_05_20.py --label v11_baseline

# Phase 3: v12 treatment redetect (SCORER_CHAIN_FALLBACK=1)
run_redetect_loop 1 "PHASE 3: v12 fallback ON"

# Phase 4: measure v12 + compare
echo
echo "========================================="
echo "[PHASE 4] measure attribution -> v12_fallback_on (compare to v11_baseline)"
echo "========================================="
PYTHONUNBUFFERED=1 uv run python -u scripts/measure_attribution_trusted_31_2026_05_20.py \
    --label v12_fallback_on --compare-to v11_baseline

# Phase 5: coherence audit on current (v12) state vs prior (v11) snapshot
# The audit script reads from DB so we capture v12 first, then would need to
# reset DB to v11 to capture that — skip and just note current state.
echo
echo "========================================="
echo "[PHASE 5] coherence audit (current state = v12)"
echo "========================================="
PYTHONUNBUFFERED=1 uv run python -u scripts/audit_coherence_trusted_29_2026_05_17.py \
    --label v12_fallback_on_coherence

# Phase 6: re-run oracle decomposition for sanity check
echo
echo "========================================="
echo "[PHASE 6] oracle decomposition re-run (current state = v12)"
echo "========================================="
PYTHONUNBUFFERED=1 uv run python -u scripts/probe_violation_oracle_decomp_2026_05_20.py

echo
echo "========================================="
echo "A/B complete. Check reports/attribution_trusted_31_2026_05_20/ for compare output."
echo "========================================="
