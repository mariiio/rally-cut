"""check_gt_integrity.py — flag action-GT rows whose stored `trackId`
anchor disagrees with the nearest-track-to-ball at the contact frame.

This is a spot-check aid, not a correctness oracle. Nearest-to-ball is the
same heuristic the spatial backfill uses; disagreements usually mean either
(a) the anchor is stale (mapping-rescued from a drifted pid), or (b) the
action genuinely wasn't the nearest player (block/dig away from ball, set
across the court). You eyeball the listed rows in the editor to decide.

Usage:
    cd analysis
    uv run python scripts/check_gt_integrity.py --fixture tata
    uv run python scripts/check_gt_integrity.py --all
    uv run python scripts/check_gt_integrity.py --fixture yeye \\
        --out reports/gt_integrity/yeye.md
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

_ANALYSIS_DIR = Path(__file__).resolve().parents[1]
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

from rallycut.evaluation.db import get_connection  # noqa: E402
from rallycut.training.action_gt_query import load_for_videos  # noqa: E402

FIXTURE_REGISTRY = (
    _ANALYSIS_DIR
    / "reports"
    / "attribution_rebuild"
    / "fixture_video_ids_2026_04_24.json"
)


@dataclass
class Mismatch:
    rally_id: str
    frame: int
    action: str
    stored_track_id: int | None
    stored_player_track_id: int | None
    nearest_track_id: int | None
    nearest_dist: float
    stored_pid_display: int | None
    nearest_pid_display: int | None


def _positions_at_frame(
    positions: list[dict[str, Any]],
    frame: int,
    primary: set[int] | None,
    tolerance: int = 2,
) -> dict[int, dict[str, Any]]:
    """Return {trackId: position} at frame (best-effort within ±tolerance)."""
    by_tid: dict[int, dict[str, Any]] = {}
    by_tid_delta: dict[int, int] = {}
    for p in positions:
        tid = p.get("trackId")
        if tid is None:
            continue
        tid_int = int(tid)
        if primary and tid_int not in primary:
            continue
        f = p.get("frameNumber") if "frameNumber" in p else p.get("frame")
        if f is None:
            continue
        delta = abs(int(f) - frame)
        if delta > tolerance:
            continue
        if tid_int not in by_tid or delta < by_tid_delta[tid_int]:
            by_tid[tid_int] = p
            by_tid_delta[tid_int] = delta
    return by_tid


def _nearest_to_ball(
    positions_at_frame: dict[int, dict[str, Any]],
    ball_x: float,
    ball_y: float,
) -> tuple[int | None, float]:
    """Use upper-quarter bbox center (torso/arms) to match editor heuristic."""
    best_tid: int | None = None
    best_dist = float("inf")
    for tid, p in positions_at_frame.items():
        x = p.get("x")
        y = p.get("y")
        w = p.get("width") if "width" in p else p.get("w")
        h = p.get("height") if "height" in p else p.get("h")
        if x is None or y is None or w is None or h is None:
            continue
        cx = float(x) + float(w) / 2
        cy = float(y) + float(h) * 0.25
        dist = ((cx - ball_x) ** 2 + (cy - ball_y) ** 2) ** 0.5
        if dist < best_dist:
            best_dist = dist
            best_tid = tid
    return best_tid, best_dist


def _applied_full_mapping(
    match_analysis: dict[str, Any] | None,
    rally_id: str,
) -> dict[int, int]:
    """Return raw-trackId → canonical-pid for the rally, or {}."""
    if not match_analysis:
        return {}
    for entry in match_analysis.get("rallies", []) or []:
        rid = entry.get("rallyId") or entry.get("rally_id")
        if rid != rally_id:
            continue
        afm = entry.get("appliedFullMapping") or {}
        try:
            return {int(k): int(v) for k, v in afm.items()}
        except (TypeError, ValueError):
            return {}
    return {}


def check_video(video_id: str) -> tuple[int, list[Mismatch]]:
    """Return (total_rows_checked, mismatches) for one video."""
    mismatches: list[Mismatch] = []
    total = 0

    with get_connection() as conn, conn.cursor() as cur:
        cur.execute(
            "SELECT match_analysis_json FROM videos WHERE id = %s",
            [video_id],
        )
        row = cur.fetchone()
    match_analysis: dict[str, Any] | None = (
        cast(dict[str, Any], row[0]) if row and row[0] else None
    )

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT r.id,
                       pt.positions_json,
                       pt.primary_track_ids
                  FROM rallies r
                  JOIN player_tracks pt ON pt.rally_id = r.id
                 WHERE r.video_id = %s
                   AND EXISTS (
                       SELECT 1 FROM rally_action_ground_truth gt WHERE gt.rally_id = r.id
                   )
                 ORDER BY r.start_ms
                """,
                [video_id],
            )
            rows = cur.fetchall()

        rally_ids = [str(row[0]) for row in rows]
        gt_by_rally = load_for_videos(conn, [video_id], include_unresolved=True)

    for rid, positions_json, primary_ids in rows:
        rid_str = str(rid)
        gt: list[dict[str, Any]] = gt_by_rally.get(rid_str, [])
        if not gt:
            continue

        positions: list[dict[str, Any]]
        if isinstance(positions_json, list):
            positions = cast(list[dict[str, Any]], positions_json)
        elif isinstance(positions_json, dict):
            positions = cast(
                list[dict[str, Any]],
                positions_json.get("positions") or [],
            )
        else:
            positions = []
        primary: set[int] | None = None
        if primary_ids is not None:
            try:
                primary = {int(x) for x in cast(list[Any], primary_ids)}
            except TypeError:
                primary = None
        afm = _applied_full_mapping(match_analysis, rid_str)

        for label in gt:
            total += 1
            frame = label.get("frame")
            action = label.get("action", "?")
            ball_x = label.get("ballX")
            ball_y = label.get("ballY")
            stored_tid = label.get("trackId")
            stored_ptid = label.get("playerTrackId")
            if frame is None:
                continue
            if ball_x is None or ball_y is None:
                # No ball coords → can't check via this heuristic
                continue

            at_frame = _positions_at_frame(positions, int(frame), primary)
            nearest_tid, nearest_dist = _nearest_to_ball(
                at_frame, float(ball_x), float(ball_y)
            )

            stored_tid_int = int(stored_tid) if stored_tid is not None else None
            if stored_tid_int is None or nearest_tid is None:
                continue
            if stored_tid_int == nearest_tid:
                continue

            stored_pid_display = afm.get(stored_tid_int)
            nearest_pid_display = afm.get(nearest_tid)
            mismatches.append(
                Mismatch(
                    rally_id=rid_str,
                    frame=int(frame),
                    action=str(action),
                    stored_track_id=stored_tid_int,
                    stored_player_track_id=(
                        int(stored_ptid) if stored_ptid is not None else None
                    ),
                    nearest_track_id=nearest_tid,
                    nearest_dist=nearest_dist,
                    stored_pid_display=stored_pid_display,
                    nearest_pid_display=nearest_pid_display,
                )
            )

    return total, mismatches


def load_fixture_registry() -> dict[str, str]:
    if not FIXTURE_REGISTRY.exists():
        return {}
    raw = json.loads(FIXTURE_REGISTRY.read_text())
    entries = raw.get("fixtures") if isinstance(raw.get("fixtures"), dict) else raw
    out: dict[str, str] = {}
    for name, val in entries.items():
        if isinstance(val, dict) and "video_id" in val:
            out[name] = val["video_id"]
    return out


def render_markdown(
    results: list[tuple[str, str, int, list[Mismatch]]],
) -> str:
    lines: list[str] = []
    lines.append("# GT integrity report")
    lines.append("")
    lines.append(
        "Rows where the stored `trackId` anchor disagrees with the nearest "
        "primary track to the ball at the contact frame. Disagreements are "
        "**candidates for review**, not proof of error — e.g. blocks/digs "
        "away from the ball and long sets commonly flag here despite being "
        "correctly labeled."
    )
    lines.append("")
    total_rows = sum(x[2] for x in results)
    total_mm = sum(len(x[3]) for x in results)
    lines.append(
        f"**Total**: {total_mm} / {total_rows} labels flagged across "
        f"{len(results)} fixture(s)."
    )
    lines.append("")
    for fixture, video_id, total, mismatches in results:
        lines.append(f"## {fixture} (`{video_id[:8]}`)")
        lines.append(
            f"{len(mismatches)} / {total} labels flagged"
            + (" — all clean ✓" if not mismatches else "")
        )
        if not mismatches:
            lines.append("")
            continue
        lines.append("")
        lines.append(
            "| rally | frame | action | stored trackId → display | "
            "nearest trackId → display | dist(px) |"
        )
        lines.append(
            "| --- | --- | --- | --- | --- | --- |"
        )
        for m in mismatches:
            stored_disp = (
                f"P{m.stored_pid_display}"
                if m.stored_pid_display is not None
                else "—"
            )
            nearest_disp = (
                f"P{m.nearest_pid_display}"
                if m.nearest_pid_display is not None
                else "—"
            )
            lines.append(
                f"| `{m.rally_id[:8]}` | {m.frame} | {m.action} | "
                f"`{m.stored_track_id}` → {stored_disp} | "
                f"`{m.nearest_track_id}` → {nearest_disp} | "
                f"{m.nearest_dist:.0f} |"
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", action="append", help="Fixture short name(s).")
    parser.add_argument("--video", action="append", help="Video UUID(s).")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run over every fixture in the registry.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional path for a markdown report. Also printed to stdout.",
    )
    args = parser.parse_args()

    registry = load_fixture_registry()
    targets: list[tuple[str, str]] = []  # (fixture_name, video_id)
    if args.all:
        targets = sorted(registry.items())
    else:
        for name in args.fixture or []:
            vid = registry.get(name)
            if not vid:
                print(f"fixture {name!r} not in registry", file=sys.stderr)
                return 2
            targets.append((name, vid))
        for vid in args.video or []:
            targets.append((vid, vid))
    if not targets:
        print("specify --fixture, --video, or --all", file=sys.stderr)
        return 2

    results: list[tuple[str, str, int, list[Mismatch]]] = []
    for name, vid in targets:
        total, mismatches = check_video(vid)
        results.append((name, vid, total, mismatches))
        print(
            f"{name} ({vid[:8]}): {len(mismatches)} / {total} flagged",
            file=sys.stderr,
        )
        for m in mismatches:
            print(
                f"  {m.rally_id[:8]} f{m.frame} {m.action}: "
                f"stored trackId={m.stored_track_id} "
                f"(P{m.stored_pid_display}) "
                f"vs nearest trackId={m.nearest_track_id} "
                f"(P{m.nearest_pid_display}), dist={m.nearest_dist:.0f}px",
                file=sys.stderr,
            )

    md = render_markdown(results)
    print(md)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(md)
        print(f"wrote {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
