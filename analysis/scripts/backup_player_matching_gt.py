"""Dump every row of videos.player_matching_gt_json to a timestamped JSON file.

Run this BEFORE any migration of the GT format. The resulting file is the
only lifeline if the bbox-keyed GT migration corrupts the DB.

Usage:
    uv run python analysis/scripts/backup_player_matching_gt.py
    uv run python analysis/scripts/backup_player_matching_gt.py --out custom/path.json

The script fails loudly if zero rows are found (that almost certainly means
the wrong DATABASE_URL is pointed at).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from rallycut.evaluation.db import get_connection

logger = logging.getLogger(__name__)

DEFAULT_BACKUP_DIR = Path(__file__).parent.parent / "backups" / "player_matching_gt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output file path (default: analysis/backups/player_matching_gt/<ts>.json)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    logger.warning(
        "=" * 72
        + "\nBACKUP ABOUT TO READ player_matching_gt_json FROM THE CONFIGURED DB."
        + "\nDO NOT RETRACK ANY GT VIDEO until migration+validation complete."
        + "\n"
        + "=" * 72
    )

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM videos WHERE player_matching_gt_json IS NOT NULL"
            )
            row = cur.fetchone()
            assert row is not None
            total = int(cast(tuple[int], row)[0])

            cur.execute(
                "SELECT id, player_matching_gt_json FROM videos "
                "WHERE player_matching_gt_json IS NOT NULL ORDER BY id"
            )
            rows = cur.fetchall()

    if total == 0 or not rows:
        logger.error(
            "No rows with player_matching_gt_json found. Refusing to write an "
            "empty backup — check your DATABASE_URL."
        )
        sys.exit(1)

    if len(rows) != total:
        logger.error(
            "Row count mismatch: COUNT=%d but fetched %d rows. Aborting.",
            total,
            len(rows),
        )
        sys.exit(1)

    payload: dict[str, Any] = {
        "backed_up_at": datetime.now(UTC).isoformat(),
        "row_count": total,
        "rows": [
            {"video_id": str(vid), "player_matching_gt_json": gt_json}
            for vid, gt_json in rows
        ],
    }

    out_path = args.out
    if out_path is None:
        ts = datetime.now(UTC).strftime("%Y-%m-%d-%H%M%S")
        out_path = DEFAULT_BACKUP_DIR / f"{ts}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    logger.info("Backed up %d rows to %s", total, out_path)
    logger.info("Video IDs:")
    for row in payload["rows"]:
        logger.info("  %s", row["video_id"])
    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. git add %s && git commit", out_path)
    logger.info("  2. Verify row count above matches your expectations")
    logger.info("  3. Only then proceed with migration")


if __name__ == "__main__":
    main()
