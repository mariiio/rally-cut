"""backfill_action_gt_trackid.py — populate `trackId` on existing action GT.

Action-GT rows stored before the trackId-anchoring fix carry only
`playerTrackId` (canonical pid, 1-4). That field drifts across pipeline
re-runs. This script best-effort-fills a stable `trackId` (raw BoT-SORT id,
pre-remap) on each row so future renders can derive the display pid via
`appliedFullMapping[trackId]`.

Algorithm per rally:
  1. Load `appliedFullMapping` for the rally from
     `videos.match_analysis_json.rallies[]` (written by remap-track-ids).
     Build `pidToRaw = {canonical_pid → raw_trackId}` (inverse).
  2. For each GT entry with `playerTrackId=pid`:
       a. If `ballX, ballY` are present AND primary positions exist for the
          frame, pick the primary track whose bbox center at the GT frame is
          nearest the ball, and use that track's raw id as `trackId`.
          Rationale: spatial anchor is robust even when the current mapping
          has drifted, provided the pid at label time still matched the
          visible body at the ball.
       b. Otherwise, fall back to `pidToRaw[pid]` (assumes current mapping
          ≈ mapping at label time; known to be wrong for drifted fixtures).
  3. Write `trackId` alongside the existing `playerTrackId` (non-destructive).

Acknowledged limits: no history of prior `appliedFullMapping` is stored.
For fixtures with per-rally drift (cece, yeye, rere, tata), the heuristic
fills with a best-effort guess; re-labeling in the fixed editor remains the
ground truth. For lulu the audit found a clean fixture-wide permutation
`{1→2, 2→1, 3→4, 4→3}` — pass `--lulu-perm` to apply it before inversion.

Usage:
    cd analysis
    uv run python scripts/backfill_action_gt_trackid.py --dry-run
    uv run python scripts/backfill_action_gt_trackid.py --video <video-id>
    uv run python scripts/backfill_action_gt_trackid.py --fixture lulu --lulu-perm
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from rallycut.evaluation.db import get_connection  # noqa: E402

# lulu's audit-confirmed fixture-wide canonical permutation:
# labeled-pid → physical-pid at label time. Applied before pid→raw inversion
# when --lulu-perm is set.
LULU_PERMUTATION: dict[int, int] = {1: 2, 2: 1, 3: 4, 4: 3}


@dataclass
class RallyBackfill:
    rally_id: str
    video_id: str
    rows_total: int = 0
    rows_with_trackid: int = 0
    rows_backfilled_spatial: int = 0
    rows_backfilled_mapping: int = 0
    rows_unresolvable: int = 0
    notes: list[str] = field(default_factory=list)


def load_applied_mapping(
    match_analysis: dict[str, Any] | None,
    rally_id: str,
) -> tuple[dict[int, int], str]:
    """Return (pidToRaw, source) for a rally.

    Prefers `appliedFullMapping` (raw → canonical). Falls back to
    `trackToPlayer` when it's NOT identity (pre-remap state). Returns
    ({}, 'none') when neither is usable.
    """
    if not match_analysis:
        return {}, "none"
    rallies = match_analysis.get("rallies") or []
    entry = next(
        (
            r
            for r in rallies
            if r.get("rallyId") == rally_id or r.get("rally_id") == rally_id
        ),
        None,
    )
    if not entry:
        return {}, "none"

    afm = entry.get("appliedFullMapping")
    if isinstance(afm, dict) and afm:
        pid_to_raw: dict[int, int] = {}
        for raw_str, pid in afm.items():
            try:
                raw = int(raw_str)
                p = int(pid)
            except (TypeError, ValueError):
                continue
            pid_to_raw.setdefault(p, raw)
        if pid_to_raw:
            return pid_to_raw, "appliedFullMapping"

    ttp = entry.get("trackToPlayer") or entry.get("track_to_player")
    if isinstance(ttp, dict) and ttp:
        is_identity = all(str(v) == k for k, v in ttp.items())
        if not is_identity:
            pid_to_raw = {}
            for raw_str, pid in ttp.items():
                try:
                    raw = int(raw_str)
                    p = int(pid)
                except (TypeError, ValueError):
                    continue
                pid_to_raw.setdefault(p, raw)
            if pid_to_raw:
                return pid_to_raw, "trackToPlayer"

    return {}, "none"


def spatial_anchor(
    positions_json: dict[str, Any] | list[Any] | None,
    primary_track_ids: list[int] | None,
    frame: int,
    ball_x: float | None,
    ball_y: float | None,
) -> int | None:
    """Return the primary track id whose bbox center at `frame` is nearest
    to (ball_x, ball_y). Returns None when the ball coords are missing or no
    primary track has a position within ±2 frames of the target frame."""
    if ball_x is None or ball_y is None or not positions_json:
        return None
    # positions_json may be either a list or a dict {positions:[...]}
    positions: list[dict[str, Any]]
    if isinstance(positions_json, dict):
        positions = cast(list[dict[str, Any]], positions_json.get("positions") or [])
    elif isinstance(positions_json, list):
        positions = cast(list[dict[str, Any]], positions_json)
    else:
        return None
    if not positions:
        return None

    primary = set(primary_track_ids or [])
    best_tid: int | None = None
    best_dist = float("inf")
    for p in positions:
        tid = p.get("trackId")
        if tid is None:
            continue
        if primary and int(tid) not in primary:
            continue
        f = p.get("frameNumber") if "frameNumber" in p else p.get("frame")
        if f is None:
            continue
        if abs(int(f) - frame) > 2:
            continue
        x = p.get("x")
        y = p.get("y")
        w = p.get("width") if "width" in p else p.get("w")
        h = p.get("height") if "height" in p else p.get("h")
        if x is None or y is None or w is None or h is None:
            continue
        cx = float(x) + float(w) / 2
        cy = float(y) + float(h) * 0.25  # upper-quarter, matches editor heuristic
        dist = ((cx - ball_x) ** 2 + (cy - ball_y) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_tid = int(tid)
    return best_tid


def backfill_rally(
    gt_labels: list[dict[str, Any]],
    pid_to_raw: dict[int, int],
    positions_json: dict[str, Any] | list[Any] | None,
    primary_track_ids: list[int] | None,
    permute: dict[int, int] | None,
) -> tuple[list[dict[str, Any]], RallyBackfill]:
    stats = RallyBackfill(rally_id="", video_id="", rows_total=len(gt_labels))
    out: list[dict[str, Any]] = []
    for label in gt_labels:
        new_label = dict(label)
        if new_label.get("trackId") is not None:
            stats.rows_with_trackid += 1
            out.append(new_label)
            continue

        pid = new_label.get("playerTrackId")
        if pid is None:
            stats.rows_unresolvable += 1
            out.append(new_label)
            continue
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            stats.rows_unresolvable += 1
            out.append(new_label)
            continue

        effective_pid = permute.get(pid, pid) if permute else pid

        frame = new_label.get("frame")
        bx = new_label.get("ballX")
        by = new_label.get("ballY")

        # Spatial anchor first — robust under drift when ball coords exist.
        spatial = (
            spatial_anchor(positions_json, primary_track_ids, int(frame), bx, by)
            if frame is not None
            else None
        )
        if spatial is not None:
            new_label["trackId"] = int(spatial)
            stats.rows_backfilled_spatial += 1
        else:
            raw = pid_to_raw.get(effective_pid)
            if raw is None:
                stats.rows_unresolvable += 1
            else:
                new_label["trackId"] = int(raw)
                stats.rows_backfilled_mapping += 1
        out.append(new_label)
    return out, stats


def load_fixture_registry() -> dict[str, str]:
    """Return {fixture_short_name: video_id}. Empty dict if registry missing."""
    path = (
        _ANALYSIS_DIR
        / "reports"
        / "attribution_rebuild"
        / "fixture_video_ids_2026_04_24.json"
    )
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    # Registry shape: {fixture: {video_id, ...}} or {fixture: video_id}.
    out: dict[str, str] = {}
    for fixture, val in raw.items():
        if isinstance(val, dict) and "video_id" in val:
            out[fixture] = val["video_id"]
        elif isinstance(val, str):
            out[fixture] = val
    return out


def resolve_target_videos(args: argparse.Namespace) -> list[str]:
    if args.video:
        return list(args.video)
    if args.fixture:
        registry = load_fixture_registry()
        missing = [f for f in args.fixture if f not in registry]
        if missing:
            raise SystemExit(
                f"fixture(s) not found in registry: {missing}. "
                f"Known: {sorted(registry.keys())}"
            )
        return [registry[f] for f in args.fixture]
    # All videos with any action GT
    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            """
            SELECT DISTINCT r.video_id
              FROM rallies r
              JOIN player_tracks pt ON pt.rally_id = r.id
             WHERE pt.action_ground_truth_json IS NOT NULL
               AND jsonb_array_length(pt.action_ground_truth_json::jsonb) > 0
            """
        )
        return [str(row[0]) for row in cur.fetchall()]


def run(args: argparse.Namespace) -> int:
    video_ids = resolve_target_videos(args)
    if not video_ids:
        print("No videos with action GT found.")
        return 0

    print(f"Target videos: {len(video_ids)}")
    if args.dry_run:
        print("DRY RUN — no DB writes.")

    totals = RallyBackfill(rally_id="TOTAL", video_id="")
    unresolvable_details: list[tuple[str, str, RallyBackfill]] = []

    permute = LULU_PERMUTATION if args.lulu_perm else None
    if permute:
        print(f"Applying lulu permutation: {permute}")

    with get_connection() as conn:
        for i, vid in enumerate(video_ids, 1):
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT match_analysis_json FROM videos WHERE id = %s",
                    [vid],
                )
                row = cur.fetchone()
            match_analysis: dict[str, Any] | None = (
                cast(dict[str, Any], row[0]) if row and row[0] else None
            )

            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT r.id,
                           pt.id,
                           pt.action_ground_truth_json,
                           pt.positions_json,
                           pt.primary_track_ids
                      FROM rallies r
                      JOIN player_tracks pt ON pt.rally_id = r.id
                     WHERE r.video_id = %s
                       AND pt.action_ground_truth_json IS NOT NULL
                       AND jsonb_array_length(pt.action_ground_truth_json::jsonb) > 0
                     ORDER BY r.start_ms
                    """,
                    [vid],
                )
                rally_rows = cur.fetchall()

            print(
                f"[{i}/{len(video_ids)}] video {vid}: "
                f"{len(rally_rows)} rallies with GT"
            )

            for rid, pt_id, gt_json, positions_json, primary_ids in rally_rows:
                rid_str = str(rid)
                pid_to_raw, source = load_applied_mapping(match_analysis, rid_str)
                gt_labels = cast(
                    list[dict[str, Any]],
                    gt_json if isinstance(gt_json, list) else [],
                )
                positions_arg: dict[str, Any] | list[Any] | None
                if isinstance(positions_json, (dict, list)):
                    positions_arg = positions_json
                else:
                    positions_arg = None
                primary_list: list[int] | None = None
                if primary_ids is not None:
                    try:
                        primary_list = [int(x) for x in cast(list[Any], primary_ids)]
                    except TypeError:
                        # Legacy rows may have stored this column as a scalar
                        # or comma-joined string — skip primary-track filtering
                        # rather than crashing the whole backfill.
                        primary_list = None
                new_labels, stats = backfill_rally(
                    gt_labels,
                    pid_to_raw,
                    positions_arg,
                    primary_list,
                    permute,
                )
                stats.rally_id = rid_str
                stats.video_id = vid
                stats.notes.append(f"mapping_source={source}")
                # `mapping_source` is rally-level; `spatial/mapping/unresolvable`
                # are per-row. Print them on separate qualifiers so a reader
                # doesn't conflate "no mapping found" with "no spatial anchor".
                print(
                    f"  {rid_str[:8]}: total={stats.rows_total} "
                    f"have={stats.rows_with_trackid} "
                    f"spatial_rescue={stats.rows_backfilled_spatial} "
                    f"via_mapping={stats.rows_backfilled_mapping} "
                    f"unresolvable={stats.rows_unresolvable} "
                    f"[rally_mapping={source}]"
                )
                totals.rows_total += stats.rows_total
                totals.rows_with_trackid += stats.rows_with_trackid
                totals.rows_backfilled_spatial += stats.rows_backfilled_spatial
                totals.rows_backfilled_mapping += stats.rows_backfilled_mapping
                totals.rows_unresolvable += stats.rows_unresolvable
                if stats.rows_unresolvable:
                    unresolvable_details.append((vid, rid_str, stats))

                if not args.dry_run and new_labels != gt_labels:
                    with conn.cursor() as wcur:
                        wcur.execute(
                            "UPDATE player_tracks SET action_ground_truth_json = %s "
                            "WHERE id = %s",
                            [json.dumps(new_labels), pt_id],
                        )
        if not args.dry_run:
            conn.commit()

    print("\n=== TOTAL ===")
    print(
        f"rows_total={totals.rows_total} "
        f"already_have_trackId={totals.rows_with_trackid} "
        f"backfilled_spatial={totals.rows_backfilled_spatial} "
        f"backfilled_mapping={totals.rows_backfilled_mapping} "
        f"unresolvable={totals.rows_unresolvable}"
    )
    if unresolvable_details:
        print("\nUnresolvable rows (manual review):")
        for vid, rid, s in unresolvable_details:
            print(f"  {vid[:8]} {rid[:8]}: {s.rows_unresolvable} rows ({s.notes})")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--video",
        action="append",
        help="Video id(s) to process (repeatable). Default: all videos with GT.",
    )
    parser.add_argument(
        "--fixture",
        action="append",
        help=(
            "Fixture short name(s) from the attribution registry (repeatable). "
            "Resolves via reports/attribution_rebuild/fixture_video_ids_2026_04_24.json."
        ),
    )
    parser.add_argument(
        "--lulu-perm",
        action="store_true",
        help="Apply lulu's audit-confirmed permutation {1→2, 2→1, 3→4, 4→3} "
        "before inverting pid→raw. Use with --fixture lulu.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan without writing to the DB.",
    )
    args = parser.parse_args()
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
