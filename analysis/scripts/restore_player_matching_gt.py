"""Restore videos.player_matching_gt_json from a backup file.

The emergency lifeline if the bbox-keyed GT migration goes sideways.
Writes each row back verbatim from the backup JSON produced by
`backup_player_matching_gt.py`.

Usage:
    uv run python analysis/scripts/restore_player_matching_gt.py <backup.json>
    uv run python analysis/scripts/restore_player_matching_gt.py <backup.json> --apply

Defaults to dry-run: prints which video IDs would be overwritten and refuses
to touch the DB unless --apply is passed.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from psycopg.types.json import Json

from rallycut.evaluation.db import get_connection

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backup", type=Path, help="Backup JSON produced by backup_player_matching_gt.py")
    parser.add_argument("--apply", action="store_true", help="Actually write to DB (default: dry-run)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if not args.backup.exists():
        logger.error("Backup file not found: %s", args.backup)
        sys.exit(1)

    with open(args.backup) as f:
        payload: dict[str, Any] = json.load(f)

    rows = payload.get("rows", [])
    if not rows:
        logger.error("Backup file has no rows. Refusing to restore.")
        sys.exit(1)

    logger.info("Backup: %s", args.backup)
    logger.info("  backed_up_at: %s", payload.get("backed_up_at"))
    logger.info("  row_count:    %d", len(rows))

    if not args.apply:
        logger.info("")
        logger.info("DRY RUN — would restore %d rows:", len(rows))
        for row in rows:
            logger.info("  %s", row["video_id"])
        logger.info("")
        logger.info("Re-run with --apply to actually write.")
        return

    logger.warning("APPLYING: overwriting player_matching_gt_json for %d videos", len(rows))
    with get_connection() as conn:
        with conn.cursor() as cur:
            for row in rows:
                vid = row["video_id"]
                gt_json = row["player_matching_gt_json"]
                cur.execute(
                    "UPDATE videos SET player_matching_gt_json = %s WHERE id = %s",
                    (Json(gt_json), vid),
                )
                if cur.rowcount != 1:
                    logger.error(
                        "UPDATE for %s affected %d rows (expected 1). Rolling back.",
                        vid,
                        cur.rowcount,
                    )
                    conn.rollback()
                    sys.exit(1)
                logger.info("  restored %s", vid)
        conn.commit()

    logger.info("Restore complete.")


if __name__ == "__main__":
    main()
