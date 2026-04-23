"""Phase 0.0 — resolve 9 fixture shortnames → video_ids.

For each shortname in FIXTURES, find matching video by Video.filename LIKE '%{name}%'.
Report: video_id, filename, #rallies, #rallies with action_ground_truth_json populated.

Output: reports/attribution_rebuild/fixture_video_ids_2026_04_24.json

Run:
    uv run python scripts/phase0_resolve_fixtures.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from rallycut.evaluation.db import get_connection

FIXTURES = ["tata", "toto", "lala", "lulu", "wawa", "cece", "cuco", "rere", "yeye"]

OUT_PATH = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "attribution_rebuild"
    / "fixture_video_ids_2026_04_24.json"
)


def main() -> int:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    failures: list[str] = []

    with get_connection() as conn, conn.cursor() as cur:
        for name in FIXTURES:
            # Match by filename containing the nickname.
            cur.execute(
                """
                SELECT v.id, v.filename, v.content_hash
                FROM videos v
                WHERE LOWER(v.filename) LIKE %s
                ORDER BY v.created_at DESC
                """,
                (f"%{name.lower()}%",),
            )
            rows = cur.fetchall()
            print(f"[{name}] filename matches: {len(rows)}")
            for vid, fn, ch in rows:
                print(f"   {vid}  {fn!r}  hash={ch[:12] if ch else None}")

            if not rows:
                failures.append(name)
                continue

            # Use most-recent match (first row, ordered DESC).
            video_id, filename, content_hash = rows[0]

            # Count rallies + rallies with action GT.
            cur.execute(
                """
                SELECT
                    COUNT(*) AS n_rallies,
                    COUNT(*) FILTER (
                        WHERE pt.action_ground_truth_json IS NOT NULL
                        AND jsonb_array_length(pt.action_ground_truth_json::jsonb) > 0
                    ) AS n_with_gt
                FROM rallies r
                LEFT JOIN player_tracks pt ON pt.rally_id = r.id
                WHERE r.video_id = %s
                """,
                (video_id,),
            )
            n_rallies, n_with_gt = cur.fetchone()

            results[name] = {
                "video_id": video_id,
                "filename": filename,
                "content_hash": content_hash,
                "n_rallies": n_rallies,
                "n_rallies_with_action_gt": n_with_gt,
                "disambig_count": len(rows),
            }
            print(
                f"   → picked {video_id} ({filename}): "
                f"{n_rallies} rallies, {n_with_gt} with action_gt"
            )

    if failures:
        print(f"\nFAILED to resolve: {failures}", file=sys.stderr)

    payload = {
        "generated_at": "2026-04-24",
        "fixtures": results,
        "failures": failures,
    }
    OUT_PATH.write_text(json.dumps(payload, indent=2))
    print(f"\nwrote {OUT_PATH}")

    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
