"""Retrack all 11 b5fb0594 rallies (best-of-N investigation 2026-05-07).

Reuses _retrack_rally + _save_to_db from retrack_panel_stale.py. Run multiple
times to sample BoT-SORT non-determinism on the same video.

Usage:
    cd analysis
    PYTHONUNBUFFERED=1 uv run python scripts/retrack_b5fb0594_only.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

# Ensure scripts/ is on sys.path so we can import retrack_panel_stale.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from retrack_panel_stale import _retrack_rally, _save_to_db  # type: ignore[import-not-found]  # noqa: E402

B5FB0594_RALLIES = [
    "be3134ba-9c9c-41fc-8114-7a2eb4a7c564",
    "57a588b2-d2d2-4cb7-879b-99312555e427",
    "cc2b967b-51dd-43e1-a824-481b5903eccf",
    "59832407-1923-4574-a850-a834101d7be6",
    "3e61838c-b086-4d81-b333-9178a3e58f1e",
    "3655eb69-b01f-431b-8d99-911ef4c414d7",
    "9d5f5152-7772-4e09-b677-7cb4c8f9862b",
    "0a9ac90a-2c8e-44ff-a3da-3112cb99410e",
    "2e95067d-099f-4818-a55c-6d429f96c3ce",
    "185d87f6-6aae-43e9-8081-07ae08e8f041",
    "26165b51-ea99-4a43-8380-edb3a4c959ed",
]


def main() -> int:
    total = len(B5FB0594_RALLIES)
    wall_start = time.time()
    print(f"Retracking {total} b5fb0594 rallies", flush=True)
    for idx, rally_id in enumerate(B5FB0594_RALLIES, 1):
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
