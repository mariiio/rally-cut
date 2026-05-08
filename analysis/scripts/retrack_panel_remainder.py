"""Retrack the 12 panel rallies that are still on pre-2026-05-07 tracking.

Companion to retrack_panel_stale.py and retrack_b5fb0594_only.py. Refreshes
rallies that Phase 0 didn't touch and iter-1/iter-2 didn't cover, so the full
panel is on today's player_tracker code.

Usage:
    cd analysis
    PYTHONUNBUFFERED=1 uv run python scripts/retrack_panel_remainder.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrack_panel_stale import _retrack_rally, _save_to_db  # type: ignore[import-not-found]  # noqa: E402

REMAINDER = [
    # 5c756c41 (2 stale from 2026-05-06)
    "7f0f540a-aa6b-4c37-b2e6-eec3b7e74da7",
    "06c13117-ecb9-4469-8143-cf9c6f65c573",
    # 854bb250 (4 stale from 2026-03 / 2026-05-01)
    "38d9e796-bb62-4d46-9a71-c17ddaadaeda",
    "e4a5349e-1f7f-4a5d-a8e3-1147204962f5",
    "c039ef17-b7dd-4162-a384-75c7f5160d08",
    "40b6b832-f985-43bf-9361-1d77bf729ecf",
    # 7d77980f (6 stale from 2026-03 / 2026-04)
    "30ffb876-ef82-4310-8642-03b204a8b1d1",
    "689854e3-60a5-4932-828c-02c4bc6baf06",
    "613fde44-1b6b-4e32-8b45-805dbf15b09d",
    "1cafde0e-91fe-4b58-8afd-e814277d3f3f",
    "1f9ce33a-76d5-407a-8a80-6dcd3d103457",
    "6e4a9de1-71f1-46db-a4f6-21ff5953f097",
]


def main() -> int:
    total = len(REMAINDER)
    wall_start = time.time()
    print(f"Retracking {total} remaining stale panel rallies", flush=True)
    for idx, rally_id in enumerate(REMAINDER, 1):
        t0 = time.time()
        print(f"[{idx}/{total}] {rally_id[:8]}  retracking...", flush=True)
        try:
            result = _retrack_rally(rally_id)
        except Exception as e:
            print(f"[{idx}/{total}] {rally_id[:8]}  ERROR: {e}", flush=True)
            return 1
        elapsed = time.time() - t0
        pti = result["primary_track_ids"]
        print(
            f"[{idx}/{total}] {rally_id[:8]}  primary_track_ids={pti}  "
            f"positions={len(result['positions'])}  elapsed={elapsed:.0f}s",
            flush=True,
        )
        _save_to_db(rally_id, result)
    wall = time.time() - wall_start
    print(f"\nDone. {total} rallies in {wall / 60:.1f} min total.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
