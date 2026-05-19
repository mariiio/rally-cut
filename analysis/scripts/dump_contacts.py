"""Dump per-rally contact frames + timestamps for inspection.

Usage:
    cd analysis
    uv run python scripts/dump_contacts.py --video <video_id_or_name>
    uv run python scripts/dump_contacts.py --rally <rally_id>

Prints one block per rally with frame numbers and seconds-since-rally-start,
plus the contacts_pipeline_version stamp. Used during fps-agnostic A/B to
diff per-rally contact lists pre/post-fix.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, cast

from rallycut.evaluation.db import get_connection


def _resolve_video_id(cur: Any, ident: str) -> str | None:
    cur.execute(
        "SELECT id FROM videos WHERE id::text = %s OR name ILIKE %s LIMIT 1",
        (ident, f"%{ident}%"),
    )
    row = cur.fetchone()
    return str(row[0]) if row else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--video", help="Video id or name fragment")
    g.add_argument("--rally", help="Rally id")
    args = ap.parse_args()

    with get_connection() as conn:
        with conn.cursor() as cur:
            if args.rally:
                where = "r.id::text = %s"
                params: tuple[Any, ...] = (args.rally,)
            else:
                vid = _resolve_video_id(cur, args.video)
                if not vid:
                    print(f"No video matched: {args.video}", file=sys.stderr)
                    return 1
                where = "r.video_id::text = %s"
                params = (vid,)

            cur.execute(
                f"""
                SELECT r.id, v.name,
                       pt.contacts_json, pt.frame_count,
                       pt.contacts_pipeline_version,
                       COALESCE(pt.fps, v.fps) AS resolved_fps
                FROM rallies r
                JOIN player_tracks pt ON pt.rally_id = r.id
                JOIN videos v ON r.video_id = v.id
                WHERE {where}
                ORDER BY r.start_ms
                """,
                params,
            )
            for row in cur:
                rally_id = str(row[0])
                video_name = row[1]
                contacts_json = row[2]
                frame_count = cast(int | None, row[3]) or 0
                version = row[4] or "?"
                fps = cast(float | None, row[5]) or 30.0

                if contacts_json is None:
                    contacts: list[dict[str, Any]] = []
                elif isinstance(contacts_json, str):
                    contacts = json.loads(contacts_json).get("contacts", [])
                else:
                    contacts = cast(dict[str, Any], contacts_json).get("contacts", [])

                frames = [c.get("frame") for c in contacts]
                ts = [round((f or 0) / fps, 3) for f in frames]
                duration = frame_count / fps if frame_count else 0.0
                density = len(contacts) / duration if duration > 0 else 0.0
                print(
                    f"rally={rally_id[:8]}  video={video_name}  "
                    f"fps={fps:.1f}  frames={frame_count}  "
                    f"dur={duration:.1f}s  contacts={len(contacts)}  "
                    f"density={density:.3f}/s  ver={version}"
                )
                print(f"    frames={frames}")
                print(f"    ts_s ={ts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
