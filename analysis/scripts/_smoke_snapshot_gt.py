"""Snapshot rally_action_ground_truth labels per rally for a video.

Helper for verifying the action-GT stability fix: run before and after a
match-players + remap-track-ids cycle; `trackId` must be identical.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from rallycut.evaluation.db import get_connection  # noqa: E402
from rallycut.training.action_gt_query import load_for_videos  # noqa: E402


def main(video_id: str, out_path: Path) -> None:
    rows: list[dict[str, object]] = []
    with get_connection() as conn:
        gt_by_rally = load_for_videos(conn, [video_id], include_unresolved=True)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, start_ms FROM rallies WHERE video_id = %s ORDER BY start_ms",
                [video_id],
            )
            rally_meta = {str(row[0]): int(row[1]) for row in cur.fetchall()}

    for rid, labels in sorted(
        gt_by_rally.items(), key=lambda kv: rally_meta.get(kv[0], 0)
    ):
        if rid not in rally_meta:
            continue
        rows.append(
            {
                "rally_id": rid,
                "start_ms": rally_meta[rid],
                "n_labels": len(labels),
                "labels": [
                    {
                        "frame": label.get("frame"),
                        "action": label.get("action"),
                        "trackId": label.get("trackId"),
                        "playerTrackId": label.get("playerTrackId"),
                    }
                    for label in labels
                ],
            }
        )

    out_path.write_text(json.dumps(rows, indent=2))
    total = sum(r["n_labels"] for r in rows)
    with_tid = sum(
        1 for r in rows for label in r["labels"] if label.get("trackId") is not None
    )
    print(
        f"snapshot: {len(rows)} rallies, {total} labels ({with_tid} with trackId) → {out_path}"
    )


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: _smoke_snapshot_gt.py <video-id> <out-path>", file=sys.stderr)
        raise SystemExit(2)
    main(sys.argv[1], Path(sys.argv[2]))
