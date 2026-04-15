#!/usr/bin/env bash
# Generate manufactured-tilt fixtures for Project C Component B tests.
# Usage: bash analysis/scripts/generate_tilt_fixtures.sh
set -euo pipefail

SRC_DIR="$HOME/Desktop/rallies/Matches"
OUT_DIR="$HOME/Desktop/rallies/Negative"
SRC="$SRC_DIR/match.mp4"

if [[ ! -f "$SRC" ]]; then
  echo "ERROR: source fixture not found: $SRC"
  exit 1
fi
mkdir -p "$OUT_DIR"

for deg in 3 6 10 15; do
  out="$OUT_DIR/tilt_${deg}deg.mp4"
  echo "Generating ${out} (tilt ${deg}°)..."
  ffmpeg -y -i "$SRC" \
    -vf "rotate=${deg}*PI/180:ow=iw:oh=ih:c=black" \
    -c:a copy \
    -t 120 \
    "$out"
done
echo "Done. 4 fixtures at $OUT_DIR/tilt_*deg.mp4"
