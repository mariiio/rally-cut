"""Reset cross-rally matcher caches to a known-clean state for validation.

The cross-rally matcher's behavior depends on TWO persistent caches:

  - `match_analysis_json[].assignmentAnchor` — per-rally PID assignment
    pinned by `(trackStatsHash, matcherVersion)`. Skips MatchSolver
    re-decision when the fingerprint matches.
  - `videos.canonical_pid_map_json` — wins over `trackToPlayer` in
    remap; downstream remap behaves differently when this carries
    stale entries.

Without resetting both before each measurement, sequential validation
runs are non-deterministic relative to a "clean baseline." See
`feedback_validation_clean_state.md` for the original incident
(2026-05-03) — three failed iterations chasing a "regression" that
turned out to be stale state from a prior experiment.

Usage:
    uv run python scripts/reset_matcher_state.py <video_id> [<video_id> ...]
    uv run python scripts/reset_matcher_state.py --all-with-gt
    uv run python scripts/reset_matcher_state.py --dry-run <video_id>

The `--all-with-gt` shorthand resets every video that has a
`player_matching_gt_json` (i.e., the GT-labeled validation panel).
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import Any, cast

from rallycut.evaluation.tracking.db import get_connection


def _list_gt_labeled_videos() -> list[str]:
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM videos WHERE player_matching_gt_json IS NOT NULL"
        )
        return [str(row[0]) for row in cur.fetchall()]


def _reset_video(
    video_id: str, *, dry_run: bool,
) -> tuple[int, bool]:
    """Returns (anchors_stripped, canonical_cleared)."""
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT match_analysis_json, canonical_pid_map_json "
            "FROM videos WHERE id = %s",
            [video_id],
        )
        row = cur.fetchone()
        if row is None:
            print(f"  {video_id[:8]}: video not found")
            return 0, False
        ma_raw, cm_raw = row
        if ma_raw is None:
            print(f"  {video_id[:8]}: no match_analysis_json — skipping")
            return 0, False
        ma = ma_raw if isinstance(ma_raw, dict) else json.loads(cast(str, ma_raw))
        ma_dict = cast(dict[str, Any], ma)

        anchors_stripped = 0
        for entry in ma_dict.get("rallies", []):
            if entry.pop("assignmentAnchor", None) is not None:
                anchors_stripped += 1

        canonical_will_clear = cm_raw is not None

        if dry_run:
            print(
                f"  [dry-run] {video_id[:8]}: would strip "
                f"{anchors_stripped} anchor(s); "
                f"canonical_pid_map_json {'will be cleared' if canonical_will_clear else 'already null'}"
            )
            return anchors_stripped, canonical_will_clear

        cur.execute(
            "UPDATE videos SET match_analysis_json = %s, "
            "canonical_pid_map_json = NULL WHERE id = %s",
            [json.dumps(ma_dict), video_id],
        )
        conn.commit()
    return anchors_stripped, canonical_will_clear


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video_ids", nargs="*",
                    help="Video IDs to reset (or use --all-with-gt)")
    ap.add_argument("--all-with-gt", action="store_true",
                    help="Reset every video with player_matching_gt_json")
    ap.add_argument("--dry-run", action="store_true",
                    help="Report what would change; don't write")
    args = ap.parse_args()

    if args.all_with_gt:
        if args.video_ids:
            sys.exit("--all-with-gt is exclusive with explicit video_ids")
        video_ids = _list_gt_labeled_videos()
        print(f"Resetting {len(video_ids)} GT-labeled video(s):")
    elif args.video_ids:
        video_ids = list(args.video_ids)
        print(f"Resetting {len(video_ids)} video(s):")
    else:
        sys.exit("Provide video_ids or use --all-with-gt")

    total_anchors = 0
    canonical_cleared_count = 0
    for vid in video_ids:
        a, c = _reset_video(vid, dry_run=args.dry_run)
        total_anchors += a
        if c:
            canonical_cleared_count += 1

    verb = "would strip" if args.dry_run else "stripped"
    canonical_verb = (
        "would clear" if args.dry_run else "cleared"
    )
    print(
        f"\nSummary: {verb} {total_anchors} assignmentAnchor(s); "
        f"{canonical_verb} {canonical_cleared_count} canonical_pid_map_json."
    )


if __name__ == "__main__":
    main()
