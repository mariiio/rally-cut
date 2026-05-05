"""Re-run match-players + remap-track-ids on rallies with corrupted primary_track_ids.

Identifies rallies where `primary_track_ids` contains an ID outside
match-players' PID space (1-4) — the editor symptom reported on
dd042609 r19/r21 (junk "PID 7" / "PID 6" labels). Resets the matcher
state for the affected videos, then re-runs the full match pipeline
so the contract enforcement in `remap-track-ids` (2026-05-04, see
chimera_stitching_dd042609_2026_05_04 memo) drops the orphan track
ids and surfaces the issue via WARNING logs.

Conservative by default — `--apply` to write changes, dry-run prints
the affected video/rally inventory + the per-video reset/match-players
plan without mutating anything.

Usage:
    cd analysis

    # Inventory affected rallies (no mutation):
    uv run python scripts/recover_chimera_corruption.py

    # Apply (resets state + re-runs match-players + remap):
    uv run python scripts/recover_chimera_corruption.py --apply

    # Limit to a specific video (e.g. dd042609 to validate the user-reported case first):
    uv run python scripts/recover_chimera_corruption.py --apply --video dd042609-e22e-4f60-83ed-038897c88c32

Validation gate per `feedback_validation_clean_state.md`: for each
affected video, anchor cache and canonical PID map are reset before
match-players runs. Bumped `MATCHER_VERSION = "v6"` would auto-invalidate
anchors anyway; the explicit reset is belt-and-suspenders for the
audit trail.
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from rallycut.evaluation.tracking.db import get_connection

ANALYSIS_ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _find_corrupted_rallies(
    video_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Inventory rallies with primary_track_ids containing IDs > 4."""
    sql = """
        SELECT
            r.id AS rally_id,
            r.video_id,
            r."order" AS rally_order,
            v.filename AS video_filename,
            pt.primary_track_ids
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        JOIN videos v ON v.id = r.video_id
        WHERE pt.primary_track_ids IS NOT NULL
          AND EXISTS (
            SELECT 1 FROM jsonb_array_elements_text(pt.primary_track_ids) AS x(val)
            WHERE x.val::int > 4
          )
    """
    params: tuple[object, ...] = ()
    if video_filter:
        sql += " AND r.video_id = %s"
        params = (video_filter,)
    sql += " ORDER BY r.video_id, r.\"order\""

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(sql, params)
        rows = cur.fetchall()

    return [
        {
            "rally_id": str(row[0]),
            "video_id": str(row[1]),
            "rally_order": int(row[2]) if row[2] is not None else -1,
            "video_filename": str(row[3]) if row[3] else "",
            "primary_track_ids": list(row[4]) if row[4] else [],
        }
        for row in rows
    ]


def _reset_state(video_id: str, dry_run: bool) -> None:
    """Reset assignment anchors + canonical_pid_map_json for one video.

    Delegates to the canonical reset helper to stay in sync with the
    matcher-state contract (`feedback_validation_clean_state.md`).
    """
    cmd = [
        "uv", "run", "python", "scripts/reset_matcher_state.py",
        video_id,
    ]
    if dry_run:
        cmd.append("--dry-run")
    logger.info("Reset state for %s%s", video_id[:8], " [dry-run]" if dry_run else "")
    subprocess.run(cmd, cwd=str(ANALYSIS_ROOT), check=True)


def _run_match_pipeline(video_id: str) -> None:
    """Re-run match-players + remap-track-ids for one video.

    The contract-enforcement fix in `remap-track-ids` (2026-05-04)
    drops orphan positions and emits a WARNING log when it does so —
    the log line names the dropped track ids, so the audit trail is
    self-documenting.
    """
    logger.info("match-players %s", video_id[:8])
    subprocess.run(
        ["uv", "run", "rallycut", "match-players", video_id],
        cwd=str(ANALYSIS_ROOT),
        check=True,
    )
    logger.info("remap-track-ids %s", video_id[:8])
    subprocess.run(
        ["uv", "run", "rallycut", "remap-track-ids", video_id],
        cwd=str(ANALYSIS_ROOT),
        check=True,
    )


def _verify(video_id: str) -> list[dict[str, Any]]:
    """Re-query the affected rallies for this video; should be empty post-fix."""
    return _find_corrupted_rallies(video_filter=video_id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually mutate state. Default is inventory-only (dry-run).",
    )
    parser.add_argument(
        "--video", type=str, default=None,
        help="Limit to one video_id. Useful to validate the fix on dd042609 first.",
    )
    args = parser.parse_args()

    print("Scanning for rallies with primary_track_ids containing IDs > 4...")
    affected = _find_corrupted_rallies(video_filter=args.video)

    if not affected:
        print("No corrupted rallies found.")
        return 0

    by_video: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in affected:
        by_video[r["video_id"]].append(r)

    print(f"\nFound {len(affected)} rallies across {len(by_video)} videos:")
    for vid, rallies in by_video.items():
        filename = rallies[0]["video_filename"]
        print(f"  {vid[:8]} ({filename}): {len(rallies)} rallies")
        for r in rallies:
            print(
                f"    r{r['rally_order']:>3} ({r['rally_id'][:8]}): "
                f"primary_track_ids={r['primary_track_ids']}"
            )

    if not args.apply:
        print("\nDry run — no changes. Pass --apply to repair.")
        return 0

    print("\nApplying repair...")
    failures: list[str] = []
    for vid, rallies in by_video.items():
        try:
            _reset_state(vid, dry_run=False)
            _run_match_pipeline(vid)
            remaining = _verify(vid)
            if remaining:
                logger.error(
                    "Verification FAILED for %s: %d rallies still have "
                    "out-of-range primary_track_ids: %s",
                    vid[:8],
                    len(remaining),
                    [r["rally_id"][:8] for r in remaining],
                )
                failures.append(vid)
            else:
                logger.info(
                    "Verified clean: %s — all %d affected rallies fixed",
                    vid[:8], len(rallies),
                )
        except subprocess.CalledProcessError as exc:
            logger.error("Pipeline failed for %s: %s", vid[:8], exc)
            failures.append(vid)

    print()
    if failures:
        print(f"FAILED on {len(failures)} videos: {[v[:8] for v in failures]}")
        return 1
    print(f"SUCCESS — {len(by_video)} videos repaired, {len(affected)} rallies cleaned.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
