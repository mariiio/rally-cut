"""Diff DB state for a fixture between operations.

Captures hashes of all match-players-relevant DB columns (videos.match_analysis_json,
videos.canonical_pid_map_json, player_tracks.{positions_json, raw_positions_json,
primary_track_ids, contacts_json, actions_json, pre_remap_state_json,
ball_positions_json, score_ground_truth_json}) and reports
which fields change between snapshots.

Usage:
    uv run python scripts/probe_db_state_drift.py b5fb0594-d64f-4a0d-bad9-de8fc36414d0 \\
        --label baseline_run1
    # ... run match-players + remap ...
    uv run python scripts/probe_db_state_drift.py b5fb0594-d64f-4a0d-bad9-de8fc36414d0 \\
        --label baseline_run2 --diff-against baseline_run1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from rallycut.evaluation.tracking.db import get_connection

OUTDIR = Path(__file__).resolve().parents[1] / "reports" / "profile_drift_probe" / "db_snapshots"

VIDEO_FIELDS = ["match_analysis_json", "canonical_pid_map_json"]
PLAYER_TRACK_FIELDS = [
    "positions_json",
    "raw_positions_json",
    "primary_track_ids",
    "pre_remap_state_json",
    "contacts_json",
    "actions_json",
    "ball_positions_json",
]


def _hash_value(v: Any) -> str | None:
    if v is None:
        return None
    s = json.dumps(v, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def capture(video_id: str, label: str) -> dict[str, Any]:
    snap: dict[str, Any] = {"video_id": video_id, "label": label}
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT id, {','.join(VIDEO_FIELDS)} FROM videos WHERE id = %s",
                [video_id],
            )
            row = cur.fetchone()
            if not row:
                raise SystemExit(f"video not found: {video_id}")
            snap["video_hashes"] = {
                f: _hash_value(row[i + 1]) for i, f in enumerate(VIDEO_FIELDS)
            }

            # Per-rally player_tracks fields
            cur.execute(
                """
                SELECT r.id, pt.id, """
                + ",".join(f"pt.{f}" for f in PLAYER_TRACK_FIELDS)
                + """
                FROM rallies r
                JOIN player_tracks pt ON pt.rally_id = r.id
                WHERE r.video_id = %s
                ORDER BY r.start_ms
                """,
                [video_id],
            )
            per_rally: list[dict[str, Any]] = []
            for r in cur.fetchall():
                rally_id = str(r[0])
                rally_short = rally_id[:8]
                hashes = {f: _hash_value(r[i + 2]) for i, f in enumerate(PLAYER_TRACK_FIELDS)}
                per_rally.append({"rally_id": rally_short, "hashes": hashes})
            snap["rallies"] = per_rally
    return snap


def diff(a: dict[str, Any], b: dict[str, Any]) -> list[str]:
    out: list[str] = []
    # Video-level fields
    for f, h_a in a["video_hashes"].items():
        h_b = b["video_hashes"].get(f)
        if h_a != h_b:
            out.append(f"video.{f}: {h_a} → {h_b}")

    # Per-rally
    a_by_id = {r["rally_id"]: r["hashes"] for r in a["rallies"]}
    b_by_id = {r["rally_id"]: r["hashes"] for r in b["rallies"]}
    rally_ids = sorted(set(a_by_id) | set(b_by_id))
    for rid in rally_ids:
        ha = a_by_id.get(rid, {})
        hb = b_by_id.get(rid, {})
        for f in PLAYER_TRACK_FIELDS:
            ha_v, hb_v = ha.get(f), hb.get(f)
            if ha_v != hb_v:
                out.append(f"rally[{rid}].{f}: {ha_v} → {hb_v}")
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("video_id")
    p.add_argument("--label", required=True)
    p.add_argument("--diff-against", default=None)
    args = p.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    short = args.video_id[:8]
    snap = capture(args.video_id, args.label)
    out_path = OUTDIR / f"{short}_{args.label}.json"
    out_path.write_text(json.dumps(snap, indent=2))
    print(f"snapshot written: {out_path}")

    if args.diff_against:
        prior_path = OUTDIR / f"{short}_{args.diff_against}.json"
        if not prior_path.exists():
            print(f"prior snapshot missing: {prior_path}", file=sys.stderr)
            sys.exit(2)
        prior = json.loads(prior_path.read_text())
        diffs = diff(prior, snap)
        if not diffs:
            print(f"DIFF {args.diff_against} → {args.label}: NONE (DB byte-identical)")
        else:
            print(f"DIFF {args.diff_against} → {args.label}: {len(diffs)} changes")
            for d in diffs:
                print(f"  {d}")


if __name__ == "__main__":
    main()
