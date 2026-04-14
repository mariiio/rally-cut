"""Surgical restore of positions_json for the 60 repair-target rallies.

Reads pristine positions_json from the rallycut_restore DB (populated from
pre_retrack_20260409_093101.dump) and writes it back to the live rallycut DB.
Only touches the 60 rallies listed in /tmp/repair_rally_ids.txt.
"""

from __future__ import annotations

import os
from pathlib import Path

import psycopg
from psycopg.types.json import Jsonb

from rallycut.evaluation.db import get_connection  # live DB

# Build restore URL by swapping DB name
LIVE_URL = os.environ.get("DATABASE_URL") or ""
if "rallycut" not in LIVE_URL:
    # fall through — get_connection loads .env; derive restore url the same way
    from dotenv import dotenv_values

    env = {
        **dotenv_values(Path(__file__).parents[2] / "api" / ".env"),
        **os.environ,
    }
    LIVE_URL = env.get("DATABASE_URL", "")

# Strip prisma query string
if "?" in LIVE_URL:
    LIVE_URL = LIVE_URL.split("?", 1)[0]
RESTORE_URL = LIVE_URL.replace("/rallycut", "/rallycut_restore")

rally_ids = Path("/tmp/repair_rally_ids.txt").read_text().strip().split("\n")
print(f"Restoring positions_json for {len(rally_ids)} rallies")
print(f"  source: {RESTORE_URL}")
print(f"  target: {LIVE_URL}")

with psycopg.connect(RESTORE_URL) as src:
    with src.cursor() as cur:
        cur.execute(
            "SELECT rally_id, positions_json FROM player_tracks "
            "WHERE rally_id = ANY(%s) AND positions_json IS NOT NULL",
            (rally_ids,),
        )
        rows = cur.fetchall()

print(f"  fetched {len(rows)} rows from restore")

with get_connection() as dst:
    with dst.cursor() as cur:
        for rid, positions in rows:
            cur.execute(
                "UPDATE player_tracks SET positions_json = %s WHERE rally_id = %s",
                (Jsonb(positions), rid),
            )
            if cur.rowcount != 1:
                print(f"  WARN {rid}: rowcount={cur.rowcount}")
    dst.commit()

print(f"[ok] restored {len(rows)} rallies")
