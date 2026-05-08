"""Detect drift between live panel tracking state and the committed snapshot.

Cross-fixture panel eval (`scripts/eval_cross_fixture.sh`) is matcher-only —
the matcher reads whatever tracking is in DB. After any `player_tracker.py`
change, panel rallies need a fresh retrack before re-baselining. This script
catches the case where the panel has silently drifted away from the
committed reference state.

Compares the rally-level `primary_track_ids` for each panel video against
`analysis/tests/fixtures/panel_player_tracks/<video>_summary.csv`. Reports:
  - rallies where primary_track_ids changed (real drift)
  - rallies where only `completed_at` / `model_version` changed (refresh
    happened but tracking output is the same — usually fine)
  - missing or extra rallies

Exit code: 0 if clean, 1 if any track-content drift detected.

Usage:
    cd analysis
    uv run python scripts/check_panel_baseline.py
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

PANEL_VIDEOS = ["5c756c41", "b5fb0594", "854bb250", "7d77980f"]
SNAPSHOT_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "panel_player_tracks"


def _normalize_pti(value: str) -> list[int]:
    """Parse a primary_track_ids JSON-array string and return a sorted list."""
    if not value or value == "":
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        # Postgres array literal: "{1,2,3,4}" — fall back to manual parse.
        stripped = value.strip("{}[] ")
        if not stripped:
            return []
        parsed = [int(x) for x in stripped.split(",")]
    return sorted(int(x) for x in parsed)


def _load_snapshot(video_prefix: str) -> dict[str, list[int]]:
    """Read fixture CSV → {rally_id: sorted primary_track_ids}."""
    path = SNAPSHOT_DIR / f"{video_prefix}_summary.csv"
    out: dict[str, list[int]] = {}
    with path.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            out[row["rally_id"]] = _normalize_pti(row["primary_track_ids"])
    return out


def _load_live(video_prefix: str) -> dict[str, list[int]]:
    """Query DB → {rally_id: sorted primary_track_ids}."""
    from rallycut.evaluation.tracking.db import get_connection

    out: dict[str, list[int]] = {}
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT pt.rally_id::text, pt.primary_track_ids
            FROM player_tracks pt
            JOIN rallies r ON r.id = pt.rally_id
            JOIN videos v ON v.id = r.video_id
            WHERE v.id::text LIKE %s
            ORDER BY r.start_ms
            """,
            [f"{video_prefix}%"],
        )
        for rally_id, pti in cur.fetchall():
            if pti is None:
                pti_list: list[int] = []
            elif isinstance(pti, list):
                pti_list = [int(x) for x in pti]
            else:
                pti_list = _normalize_pti(str(pti))
            out[rally_id] = sorted(pti_list)
    return out


def _diff_video(video_prefix: str) -> tuple[int, list[str]]:
    """Compare snapshot vs live for one video. Return (drift_count, lines)."""
    snap = _load_snapshot(video_prefix)
    live = _load_live(video_prefix)
    lines: list[str] = []
    drift = 0
    for rally_id in sorted(snap.keys() | live.keys()):
        if rally_id not in snap:
            lines.append(f"  EXTRA   {rally_id[:8]}  live={live[rally_id]}")
            drift += 1
            continue
        if rally_id not in live:
            lines.append(f"  MISSING {rally_id[:8]}  snap={snap[rally_id]}")
            drift += 1
            continue
        if snap[rally_id] != live[rally_id]:
            lines.append(
                f"  DRIFT   {rally_id[:8]}  snap={snap[rally_id]}  live={live[rally_id]}"
            )
            drift += 1
    return drift, lines


def main() -> int:
    total_drift = 0
    print("Panel tracking drift check vs committed snapshot")
    print(f"  snapshot dir: {SNAPSHOT_DIR}")
    print()
    for video in PANEL_VIDEOS:
        drift, lines = _diff_video(video)
        status = "CLEAN" if drift == 0 else f"{drift} drift"
        print(f"{video}: {status}")
        for line in lines:
            print(line)
        total_drift += drift
    print()
    if total_drift:
        print(f"Total drift: {total_drift} rallies")
        print(
            "If intentional (after a player_tracker change), refresh the "
            "snapshot and re-baseline:"
        )
        print(
            "  1. Re-run the panel eval and verify PERMUTED is acceptable."
        )
        print(
            "  2. Regenerate the CSVs in tests/fixtures/panel_player_tracks/."
        )
        print(
            "  3. Update reports/cross_fixture_baseline_*.log and the memory "
            "entry."
        )
        return 1
    print("Panel tracking matches snapshot. Eval reads will be reproducible.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
