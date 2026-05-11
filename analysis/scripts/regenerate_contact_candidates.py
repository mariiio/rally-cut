"""Regenerate Contact.player_candidates for stored rallies using the v3.0
adaptive candidate-generation window.

Reads `positions_json` + `contacts_json` from `player_tracks`; for each
contact, re-computes `playerCandidates` via the new
`_find_nearest_players` (with ADAPTIVE_PLAYER_SEARCH_WINDOW=1) and writes
the updated `contacts_json` back. Other Contact fields (ball position,
frame, court_side, etc.) are preserved.

This is the deploy companion for the v3.0 spec — without it, existing
rallies still carry the pre-v3 narrow-window candidates and the adaptive
fallback has no effect on attribution.

Run from analysis/:
    uv run python scripts/regenerate_contact_candidates.py <video-id> [--dry-run]
    uv run python scripts/regenerate_contact_candidates.py --all-with-gt [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, cast

from rallycut.evaluation.db import get_connection
from rallycut.tracking.contact_detector import _find_nearest_players
from rallycut.tracking.player_tracker import PlayerPosition


def _reconstruct_positions(positions_json: list[dict[str, Any]] | None) -> list[PlayerPosition]:
    if not positions_json:
        return []
    out: list[PlayerPosition] = []
    for p in positions_json:
        if not isinstance(p, dict):
            continue
        out.append(PlayerPosition(
            frame_number=p.get("frameNumber", 0),
            track_id=p.get("trackId", -1),
            x=p.get("x", 0.5),
            y=p.get("y", 0.5),
            width=p.get("width", 0.05),
            height=p.get("height", 0.10),
            confidence=p.get("confidence", 1.0),
            keypoints=p.get("keypoints"),
        ))
    return out


def _regenerate_one_rally(
    contacts_json: dict[str, Any],
    positions_json: list[dict[str, Any]],
    primary_track_ids: list[int] | None,
    max_candidates: int = 4,
) -> tuple[dict[str, Any], int]:
    """Return (new_contacts_json, n_contacts_changed).

    A contact is "changed" if its playerCandidates list differs (in set or
    order) from the pre-regeneration version.
    """
    # Ensure env flag is ON for the regeneration regardless of caller's env.
    os.environ["ADAPTIVE_PLAYER_SEARCH_WINDOW"] = "1"

    positions = _reconstruct_positions(positions_json)
    new_contacts_json = dict(contacts_json or {})
    new_list = []
    n_changed = 0

    for c in (contacts_json or {}).get("contacts", []):
        frame = c.get("frame", 0)
        ball_x = c.get("ballX", 0.5)
        ball_y = c.get("ballY", 0.5)

        new_candidates = _find_nearest_players(
            frame=frame, ball_x=ball_x, ball_y=ball_y,
            player_positions=positions,
            search_frames=15,
            max_candidates=max_candidates,
            court_calibrator=None,  # caller already applied calibration upstream
            primary_track_ids=primary_track_ids,
        )

        # Build the playerCandidates list in the same shape as the original.
        new_pc = [
            [tid, dist if dist != float("inf") else None]
            for tid, dist, _y in new_candidates
        ]

        old_pc = c.get("playerCandidates") or []
        if [list(x) for x in new_pc] != [list(x) for x in old_pc]:
            n_changed += 1

        c_copy = dict(c)
        c_copy["playerCandidates"] = new_pc
        new_list.append(c_copy)

    new_contacts_json["contacts"] = new_list
    return new_contacts_json, n_changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("video_id", nargs="?", default=None)
    parser.add_argument(
        "--all-with-gt",
        action="store_true",
        help="Regenerate for all videos with action_ground_truth_json (the 3 GT videos).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.video_id and not args.all_with_gt:
        sys.exit("Provide either <video-id> or --all-with-gt")

    with get_connection() as conn, conn.cursor() as cur:
        if args.all_with_gt:
            cur.execute(
                """SELECT DISTINCT v.id, v.filename FROM videos v
                   JOIN rallies r ON r.video_id = v.id
                   JOIN player_tracks pt ON pt.rally_id = r.id
                   WHERE pt.action_ground_truth_json IS NOT NULL
                   ORDER BY v.filename"""
            )
            video_rows = cur.fetchall()
        else:
            cur.execute("SELECT id, filename FROM videos WHERE id = %s", (args.video_id,))
            video_rows = cur.fetchall()

        total_rallies = 0
        total_changed = 0
        for vid, filename in video_rows:
            cur.execute(
                """SELECT pt.id, pt.contacts_json, pt.positions_json, pt.primary_track_ids
                   FROM player_tracks pt JOIN rallies r ON r.id = pt.rally_id
                   WHERE r.video_id = %s
                     AND pt.contacts_json IS NOT NULL
                     AND pt.positions_json IS NOT NULL
                   ORDER BY r.start_ms""",
                (vid,),
            )
            rallies = cur.fetchall()
            print(f"[{filename}] {len(rallies)} rallies", flush=True)

            for row in rallies:
                pt_id = str(row[0])
                contacts_j: dict[str, Any] = cast(dict[str, Any], row[1]) if row[1] else {}
                positions_j: list[dict[str, Any]] = cast(list[dict[str, Any]], row[2]) if row[2] else []
                primary_tids: list[int] | None = (
                    [int(x) for x in cast(list[Any], row[3])]
                    if row[3] else None
                )
                total_rallies += 1
                new_contacts_json, n_changed = _regenerate_one_rally(
                    contacts_j,
                    positions_j,
                    primary_tids,
                )
                if n_changed > 0:
                    total_changed += 1
                    print(f"  pt_id={pt_id}: {n_changed} contacts changed", flush=True)
                    if not args.dry_run:
                        cur.execute(
                            "UPDATE player_tracks SET contacts_json = %s WHERE id = %s",
                            [json.dumps(new_contacts_json), pt_id],
                        )
        if not args.dry_run:
            conn.commit()
    print(f"\nTotal: {total_changed}/{total_rallies} rallies had at least one "
          f"contact's playerCandidates updated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
