#!/usr/bin/env bash
# A/B test: ENABLE_BBOX_SWAP_DETECTION on the 4-fixture panel.
# Captures verdict snapshots for OFF (locked baseline) vs ON.
#
# Phase 2 anchor cache invalidates naturally when W4 changes top_tracks
# (synth ids enter / leave), so no --reset-anchors needed for correctness;
# we use --reset-anchors anyway to ensure both passes are clean fresh
# solves with no anchor noise across the toggle.
set -euo pipefail
cd "$(dirname "$0")/.."

OUTDIR="reports/w4_ab_2026_05_01"
mkdir -p "$OUTDIR"

VIDEOS=(
    "b5fb0594-d64f-4a0d-bad9-de8fc36414d0"
    "5c756c41-1cc1-4486-a95c-97398912cfbe"
    "854bb250-3e91-47d2-944d-f62413e3cf45"
    "7d77980f-3006-40e0-adc0-db491a5bb659"
)

run_pass() {
    local mode="$1"   # off | on
    local w4="$2"     # 0 | 1
    for v in "${VIDEOS[@]}"; do
        local short="${v:0:8}"
        echo "==== $short / w4=$mode ===="
        ENABLE_BBOX_SWAP_DETECTION="$w4" ENABLE_POSITION_JUMP_SWAP=0 \
            uv run rallycut match-players "$v" --no-ref-crops --reset-anchors \
            > "$OUTDIR/${short}_w4${mode}_match.log" 2>&1
        ENABLE_BBOX_SWAP_DETECTION="$w4" ENABLE_POSITION_JUMP_SWAP=0 \
            uv run rallycut remap-track-ids "$v" \
            > "$OUTDIR/${short}_w4${mode}_remap.log" 2>&1
    done
    uv run python scripts/panel_verdict_per_frame.py \
        > "$OUTDIR/verdict_w4${mode}.txt" 2>&1
    echo "  result:"
    grep -E "^AGREES|^[a-f0-9]+/r" "$OUTDIR/verdict_w4${mode}.txt" || true
}

echo "=== Pass A: W4 OFF (locked baseline reproduction) ==="
run_pass "off" "0"
echo
echo "=== Pass B: W4 ON ==="
run_pass "on" "1"

echo
echo "=== Diff: per-rally derived_shape changes ==="
uv run python - <<'PYEOF'
from pathlib import Path
import re

def parse(path):
    out = {}
    if not Path(path).exists():
        return out
    for line in Path(path).read_text().splitlines():
        if "/" in line[:20] and ("PNL" in line or "CTRL" in line):
            parts = line.split(maxsplit=8)
            if len(parts) >= 6:
                out[parts[0]] = {
                    "actual": parts[3],
                    "agree": parts[4],
                    "shape": parts[8] if len(parts) > 8 else "",
                }
    return out

a = parse("reports/w4_ab_2026_05_01/verdict_w4off.txt")
b = parse("reports/w4_ab_2026_05_01/verdict_w4on.txt")

agrees_off = sum(1 for r in a.values() if r["agree"] == "YES")
agrees_on = sum(1 for r in b.values() if r["agree"] == "YES")
print(f"AGREES: off={agrees_off}/{len(a)} on={agrees_on}/{len(b)}")
print()
print("| rally | off actual | off shape | on actual | on shape | flipped |")
for rally in sorted(set(a) | set(b)):
    ra = a.get(rally, {})
    rb = b.get(rally, {})
    flip = ""
    if ra.get("actual") != rb.get("actual"):
        flip = f"{ra.get('actual', '?')}→{rb.get('actual', '?')}"
    elif ra.get("shape") != rb.get("shape"):
        flip = "shape-only"
    if flip:
        print(f"  {rally}: off={ra.get('actual')}({ra.get('shape', '')[:30]}) "
              f"on={rb.get('actual')}({rb.get('shape', '')[:30]}) {flip}")
PYEOF
