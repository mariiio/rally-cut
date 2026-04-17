"""Spatial GT repair for orphaned action labels.

After the 2026-04-17 Phase 1 GT-integrity repair, 110 rallies remained
`gt_orphaned`: their `action_ground_truth_json[i].playerTrackId` values
don't resolve against the current `positions_json` trackIds even after
`remap-track-ids`. Neither S3 backup snapshot contains recoverable GT.

This script performs spatial repair: for each orphaned GT label that has
`ballX` + `ballY` set, find the current track whose bbox at the GT frame
is nearest to the ball position. If confident, emit a suggested remap.

Confident ≡ best-track distance ≤ MAX_REPAIR_DIST (default 0.08 norm-coords,
roughly one bbox width) AND the second-nearest track is ≥ MIN_MARGIN farther
(default 0.03). Ambiguous cases are flagged for manual review.

Usage:
    cd analysis
    uv run python scripts/repair_orphaned_gt.py
        # writes reports/gt_orphan_auto_repair.json + gt_orphan_manual_flag.json
    uv run python scripts/repair_orphaned_gt.py --apply
        # applies confident repairs to action_ground_truth_json in-place
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from rich.console import Console
from rich.table import Table

from rallycut.evaluation.db import get_connection
from rallycut.tracking.player_tracker import PlayerPosition
from scripts.diagnose_gt_track_mismatch import _classify_rally
from scripts.eval_action_detection import (
    _load_track_to_player_maps,
    load_rallies_with_action_gt,
)

console = Console()

MAX_REPAIR_DIST = 0.10   # closest-point-on-bbox dist in norm coords (~1 bbox width)
MIN_MARGIN = 0.03        # gap between best and second-best track (~0.5 bbox widths)
SEARCH_FRAMES = 5


@dataclass
class RepairSuggestion:
    rally_id: str
    video_id: str
    gt_frame: int
    gt_action: str
    original_player_track_id: int
    suggested_player_track_id: int
    best_dist: float
    second_dist: float | None
    confident: bool
    reason: str


def _find_best_track_at_frame(
    frame: int,
    ball_x: float,
    ball_y: float,
    player_positions: list[PlayerPosition],
    search_frames: int = SEARCH_FRAMES,
) -> list[tuple[int, float]]:
    """Return [(track_id, min_distance), ...] sorted ascending.

    Aggregates per-track minimum distance across ±search_frames around
    `frame` (to smooth single-frame detection noise). Uses *closest-point-
    on-bbox* for the distance metric because contact-time the ball is
    typically at a hand extending beyond the player's bbox — a centroid
    distance penalises tall players (bbox height ~0.3) more than the
    physical ball-to-hand distance warrants.
    """
    by_track: dict[int, float] = {}
    for p in player_positions:
        if abs(p.frame_number - frame) > search_frames:
            continue
        # Closest point on bbox to ball — zero inside, normal distance outside.
        bx = max(p.x, min(ball_x, p.x + p.width))
        by = max(p.y, min(ball_y, p.y + p.height))
        dist = math.sqrt((ball_x - bx) ** 2 + (ball_y - by) ** 2)
        prev = by_track.get(p.track_id)
        if prev is None or dist < prev:
            by_track[p.track_id] = dist
    return sorted(by_track.items(), key=lambda kv: kv[1])


def _resolved_canonical(
    track_id: int,
    t2p: dict[int, int] | None,
    avail_tids: set[int],
) -> int:
    """Resolve a raw track_id to its canonical player id (1-4) if mapped.

    Falls back to the raw id when the map doesn't contain it — that's
    the 'track is live but unmapped by match-players' case. Callers
    decide whether to accept unmapped ids.
    """
    if t2p and track_id in t2p:
        return t2p[track_id]
    if track_id in avail_tids:
        return track_id
    return track_id


def _build_positions(rally_positions_json: list[dict]) -> list[PlayerPosition]:
    return [
        PlayerPosition(
            frame_number=pp["frameNumber"],
            track_id=pp["trackId"],
            x=pp["x"],
            y=pp["y"],
            width=pp["width"],
            height=pp["height"],
            confidence=pp.get("confidence", 1.0),
        )
        for pp in rally_positions_json
    ]


def run_repair(
    auto_out: Path,
    manual_out: Path,
    apply: bool = False,
) -> None:
    rallies = load_rallies_with_action_gt()
    if not rallies:
        console.print("[red]No rallies found.[/red]")
        return

    video_ids = {r.video_id for r in rallies}
    t2p_maps = _load_track_to_player_maps(video_ids)

    orphaned = []
    for rally in rallies:
        cls = _classify_rally(rally, t2p_maps.get(rally.rally_id))
        if cls["status"] == "gt_orphaned":
            orphaned.append(rally)

    console.print(f"[bold]Spatial GT repair — {len(orphaned)} orphaned rallies[/bold]\n")

    auto_suggestions: list[RepairSuggestion] = []
    manual_flags: list[RepairSuggestion] = []
    stats: dict[str, int] = defaultdict(int)

    for rally in orphaned:
        t2p = t2p_maps.get(rally.rally_id) or {}
        positions = _build_positions(rally.positions_json or [])
        avail_tids = {p.track_id for p in positions}

        for gt in rally.gt_labels:
            # Only target orphaned labels — those whose playerTrackId is
            # unresolvable from current tracks. Leaves already-resolvable
            # labels untouched.
            original = gt.player_track_id
            resolvable = original in avail_tids
            if not resolvable and t2p:
                resolvable = original in set(t2p.values()) and any(
                    t2p.get(t) == original for t in avail_tids
                )
            if resolvable:
                stats["already_resolvable"] += 1
                continue

            if gt.ball_x is None or gt.ball_y is None:
                manual_flags.append(RepairSuggestion(
                    rally_id=rally.rally_id,
                    video_id=rally.video_id,
                    gt_frame=gt.frame,
                    gt_action=gt.action,
                    original_player_track_id=original,
                    suggested_player_track_id=-1,
                    best_dist=float("inf"),
                    second_dist=None,
                    confident=False,
                    reason="no ball position in GT label",
                ))
                stats["no_ball_position"] += 1
                continue

            ranked = _find_best_track_at_frame(
                gt.frame, gt.ball_x, gt.ball_y, positions,
            )
            if not ranked:
                manual_flags.append(RepairSuggestion(
                    rally_id=rally.rally_id,
                    video_id=rally.video_id,
                    gt_frame=gt.frame,
                    gt_action=gt.action,
                    original_player_track_id=original,
                    suggested_player_track_id=-1,
                    best_dist=float("inf"),
                    second_dist=None,
                    confident=False,
                    reason="no tracks near GT frame",
                ))
                stats["no_tracks_near_frame"] += 1
                continue

            best_tid, best_dist = ranked[0]
            second_dist = ranked[1][1] if len(ranked) > 1 else None
            suggested = _resolved_canonical(best_tid, t2p, avail_tids)

            margin_ok = second_dist is None or (second_dist - best_dist) >= MIN_MARGIN
            dist_ok = best_dist <= MAX_REPAIR_DIST
            confident = dist_ok and margin_ok

            suggestion = RepairSuggestion(
                rally_id=rally.rally_id,
                video_id=rally.video_id,
                gt_frame=gt.frame,
                gt_action=gt.action,
                original_player_track_id=original,
                suggested_player_track_id=suggested,
                best_dist=round(best_dist, 4),
                second_dist=round(second_dist, 4) if second_dist is not None else None,
                confident=confident,
                reason=(
                    "confident spatial match"
                    if confident
                    else (
                        f"distance {best_dist:.3f} > {MAX_REPAIR_DIST}"
                        if not dist_ok
                        else f"margin {(second_dist or 0) - best_dist:.3f} < {MIN_MARGIN}"
                    )
                ),
            )
            if confident:
                auto_suggestions.append(suggestion)
                stats["confident_repair"] += 1
            else:
                manual_flags.append(suggestion)
                stats["ambiguous"] += 1

    auto_out.parent.mkdir(parents=True, exist_ok=True)
    manual_out.parent.mkdir(parents=True, exist_ok=True)
    auto_out.write_text(json.dumps([asdict(s) for s in auto_suggestions], indent=2))
    manual_out.write_text(json.dumps([asdict(s) for s in manual_flags], indent=2))

    table = Table(title="Spatial Repair Results")
    table.add_column("Bucket")
    table.add_column("Count", justify="right")
    for k, v in stats.items():
        table.add_row(k, str(v))
    console.print(table)
    console.print(f"\n  → {auto_out}")
    console.print(f"  → {manual_out}")

    if apply and auto_suggestions:
        _apply_suggestions(auto_suggestions)


def _apply_suggestions(suggestions: list[RepairSuggestion]) -> None:
    """Apply confident repairs to `action_ground_truth_json` in-place.

    Groups by rally_id so each rally receives a single UPDATE. Matches
    labels by `(frame, action, original_player_track_id)` to avoid
    accidental rewrites when multiple contacts share a frame.
    """
    by_rally: dict[str, list[RepairSuggestion]] = defaultdict(list)
    for s in suggestions:
        by_rally[s.rally_id].append(s)

    updated_rallies = 0
    updated_labels = 0
    with get_connection() as conn:
        with conn.cursor() as cur:
            for rally_id, sugs in by_rally.items():
                cur.execute(
                    "SELECT action_ground_truth_json FROM player_tracks "
                    "WHERE rally_id = %s",
                    (rally_id,),
                )
                row = cur.fetchone()
                if not row or not row[0]:
                    continue
                raw_labels = row[0]
                assert isinstance(raw_labels, list)
                labels: list[dict] = list(raw_labels)
                index = {
                    (s.gt_frame, s.gt_action, s.original_player_track_id): s
                    for s in sugs
                }
                changed = 0
                for label in labels:
                    frame = label.get("frame")
                    action = label.get("action")
                    tid = label.get("playerTrackId")
                    if not (isinstance(frame, int) and isinstance(action, str)
                            and isinstance(tid, int)):
                        continue
                    key = (frame, action, tid)
                    if key in index:
                        label["playerTrackId"] = index[key].suggested_player_track_id
                        changed += 1
                if changed:
                    cur.execute(
                        "UPDATE player_tracks SET action_ground_truth_json = %s::jsonb "
                        "WHERE rally_id = %s",
                        (json.dumps(labels), rally_id),
                    )
                    updated_rallies += 1
                    updated_labels += changed
        conn.commit()

    console.print(
        f"\n[bold green]Applied {updated_labels} repairs across "
        f"{updated_rallies} rallies[/bold green]"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--auto-out",
        type=Path,
        default=Path("reports/gt_orphan_auto_repair.json"),
        help="Where to write confident repair suggestions (JSON)",
    )
    parser.add_argument(
        "--manual-out",
        type=Path,
        default=Path("reports/gt_orphan_manual_flag.json"),
        help="Where to write ambiguous cases flagged for manual review (JSON)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply confident repairs to action_ground_truth_json in the DB",
    )
    args = parser.parse_args()

    run_repair(args.auto_out, args.manual_out, apply=args.apply)


if __name__ == "__main__":
    main()
