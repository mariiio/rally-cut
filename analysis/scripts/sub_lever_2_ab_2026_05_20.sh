#!/bin/bash
# Sub-lever 2 A/B driver: 2x2 flag grid for v14 chain-walker.
#
# Phase 1: cfg_00 redetect on 32 trusted videos (v13-equivalent baseline)
# Phase 2: measure attribution -> v14_cfg00.json
# Phase 3: cfg_10 redetect (B.1 ON)
# Phase 4: measure -> v14_cfg10.json + compare-to v14_cfg00
# Phase 5: cfg_01 redetect (B.2 ON)
# Phase 6: measure -> v14_cfg01.json + compare-to v14_cfg00
# Phase 7: cfg_11 redetect (both ON)
# Phase 8: measure -> v14_cfg11.json + compare-to v14_cfg00
# Phase 9: coherence audit (current state = cfg_11) for sanity check
#
# Runs from analysis/ dir; expects DB DSN in env or default localhost:5436.
# Total wall time estimate: ~3-4 hours on this corpus (4 full redetect
# cycles + 4 measurements).

set -euo pipefail

cd "$(dirname "$0")/.."  # analysis/

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
echo "[setup] Resolved $N_UUIDS UUIDs (>=32 expected; duplicates by name OK)"
if [ "$N_UUIDS" -lt 32 ]; then
    echo "[ERROR] Expected at least 32 UUIDs, got $N_UUIDS" >&2
    exit 1
fi

run_redetect_loop () {
    local block_flag="$1"
    local verifier_flag="$2"
    local label="$3"
    echo
    echo "========================================="
    echo "[$label] redetect 32 videos, BLOCK_COND=$block_flag, BALL_TRAJ=$verifier_flag"
    echo "========================================="
    local i=0
    while IFS= read -r uid; do
        i=$((i+1))
        echo "  [$i/$N_UUIDS] redetect $uid"
        WALKER_BLOCK_CONDITIONAL=$block_flag \
            WALKER_BALL_TRAJECTORY_VERIFIER=$verifier_flag \
            USE_DYNAMIC_ATTRIBUTION_SCORER=1 \
            PYTHONUNBUFFERED=1 \
            uv run python -u scripts/redetect_all_actions.py --video "$uid" --apply
    done <<< "$UUIDS"
}

# Phase 1+2: cfg_00 baseline
run_redetect_loop 0 0 "PHASE 1: cfg_00 (v13 baseline)"
echo
echo "========================================="
echo "[PHASE 2] measure -> v14_cfg00"
echo "========================================="
PYTHONUNBUFFERED=1 uv run python -u scripts/measure_attribution_trusted_31_2026_05_20.py --label v14_cfg00

# Phase 3+4: cfg_10 (B.1 only)
run_redetect_loop 1 0 "PHASE 3: cfg_10 (B.1 only)"
echo
echo "========================================="
echo "[PHASE 4] measure -> v14_cfg10 (compare to v14_cfg00)"
echo "========================================="
PYTHONUNBUFFERED=1 uv run python -u scripts/measure_attribution_trusted_31_2026_05_20.py \
    --label v14_cfg10 --compare-to v14_cfg00

# Phase 5+6: cfg_01 (B.2 only)
run_redetect_loop 0 1 "PHASE 5: cfg_01 (B.2 only)"
echo
echo "========================================="
echo "[PHASE 6] measure -> v14_cfg01 (compare to v14_cfg00)"
echo "========================================="
PYTHONUNBUFFERED=1 uv run python -u scripts/measure_attribution_trusted_31_2026_05_20.py \
    --label v14_cfg01 --compare-to v14_cfg00

# Phase 7+8: cfg_11 (both ON)
run_redetect_loop 1 1 "PHASE 7: cfg_11 (B.1 + B.2)"
echo
echo "========================================="
echo "[PHASE 8] measure -> v14_cfg11 (compare to v14_cfg00)"
echo "========================================="
PYTHONUNBUFFERED=1 uv run python -u scripts/measure_attribution_trusted_31_2026_05_20.py \
    --label v14_cfg11 --compare-to v14_cfg00

# Phase 9: coherence audit on current state (cfg_11)
echo
echo "========================================="
echo "[PHASE 9] coherence audit (current state = cfg_11)"
echo "========================================="
PYTHONUNBUFFERED=1 uv run python -u scripts/audit_coherence_trusted_29_2026_05_17.py \
    --label v14_cfg11_coherence

echo
echo "========================================="
echo "A/B complete. Check reports/attribution_trusted_31_2026_05_20/v14_cfg*.json"
echo "and the compare-to outputs for the 3 non-baseline configs."
echo "========================================="
