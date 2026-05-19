"""Per-video contact density with 30fps/60fps cohort averages.

Usage:
    cd analysis
    uv run python scripts/contact_density_cohort.py \\
        --videos haha,kuku,koko,lulu,wawa,titi,toto,jaja

Prints per-video density/s (contacts per second of rally play) and a
60fps vs 30fps cohort summary. Used to verify the fps-agnostic A/B
success criteria: 60fps cohort density should rise into 0.39-0.49/s
post-fix; 30fps cohort should stay unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from typing import Any, cast

from rallycut.evaluation.db import get_connection


def _resolve_video_ids(cur: Any, names: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for n in names:
        cur.execute(
            "SELECT id FROM videos WHERE name ILIKE %s ORDER BY name LIMIT 1",
            (f"%{n}%",),
        )
        row = cur.fetchone()
        if row:
            out[n] = str(row[0])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--videos", required=True,
        help="Comma-separated video name fragments",
    )
    ap.add_argument(
        "--min-duration", type=float, default=10.0,
        help="Skip rallies shorter than this (seconds). Default 10s.",
    )
    args = ap.parse_args()

    names = [n.strip() for n in args.videos.split(",") if n.strip()]

    with get_connection() as conn:
        with conn.cursor() as cur:
            ids = _resolve_video_ids(cur, names)
            missing = set(names) - set(ids)
            if missing:
                print(
                    f"WARNING: no match for: {sorted(missing)}",
                    file=sys.stderr,
                )

            results: list[tuple[str, float, int, int, float, float]] = []
            cohort_total: dict[str, list[float]] = defaultdict(list)

            for name, vid in ids.items():
                cur.execute(
                    """
                    SELECT pt.contacts_json, pt.frame_count,
                           COALESCE(pt.fps, v.fps) AS resolved_fps
                    FROM rallies r
                    JOIN player_tracks pt ON pt.rally_id = r.id
                    JOIN videos v ON r.video_id = v.id
                    WHERE r.video_id = %s
                    """,
                    (vid,),
                )
                rallies = 0
                total_contacts = 0
                total_seconds = 0.0
                fps_seen: float | None = None
                for row in cur:
                    fps = cast(float | None, row[2]) or 30.0
                    fps_seen = fps
                    fc = cast(int | None, row[1]) or 0
                    duration = fc / fps
                    if duration < args.min_duration:
                        continue
                    contacts_json = row[0]
                    if contacts_json is None:
                        contacts: list[Any] = []
                    elif isinstance(contacts_json, str):
                        contacts = json.loads(contacts_json).get("contacts", [])
                    else:
                        contacts = cast(dict[str, Any], contacts_json).get(
                            "contacts", []
                        )
                    rallies += 1
                    total_contacts += len(contacts)
                    total_seconds += duration

                density = total_contacts / total_seconds if total_seconds > 0 else 0.0
                results.append((
                    name, fps_seen or 0.0, rallies, total_contacts,
                    total_seconds, density,
                ))
                cohort = "60fps" if (fps_seen or 0.0) > 40.0 else "30fps"
                if rallies > 0:
                    cohort_total[cohort].append(density)

            header = (
                f"{'video':<12} {'fps':>5} {'rallies':>8} {'contacts':>9} "
                f"{'seconds':>9} {'density/s':>10}"
            )
            print(header)
            for v_name, v_fps, v_rallies, v_contacts, v_secs, v_density in results:
                print(
                    f"{v_name:<12} {v_fps:>5.1f} {v_rallies:>8d} {v_contacts:>9d} "
                    f"{v_secs:>9.1f} {v_density:>10.3f}"
                )
            print()
            for cohort in ("60fps", "30fps"):
                ds = cohort_total[cohort]
                if ds:
                    avg = sum(ds) / len(ds)
                    print(
                        f"COHORT {cohort}: avg density {avg:.3f}/s (n={len(ds)})"
                    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
