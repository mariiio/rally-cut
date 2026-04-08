"""Shared loader for videos.player_matching_gt_json.

Bbox-keyed GT only. Labels are anchored to a frame + bbox in the video's
player_tracks.positions_json coordinate system, so the same label resolves
to the current track id via IoU matching regardless of how many times
tracking has been re-run.

Format on disk:

    {
        "rallies": {
            "<rally_id>": {
                "labels": [
                    {"playerId": 1, "frame": 42,
                     "cx": 0.41, "cy": 0.63, "w": 0.08, "h": 0.28},
                    ...  // typically 4 labels per rally (one per player)
                ]
            }
        },
        "sideSwitches": [3, 11],
        "excludedRallies": []
    }

Runtime shape callers consume:

    NormalizedGt = {
        "rallies": {rally_id: {track_id_str: player_id_int}},
        "sideSwitches": list[int],
        "excludedRallies": list[str],
    }

Each label is resolved to the current track id by finding the position at
`frame` with the highest IoU against the label bbox. Labels whose bbox
fails to IoU-match any current track above IOU_THRESHOLD are dropped with
a warning (the rally still loads — partial GT is better than no GT).

Callers supply per-rally positions via a `PositionsLookup` callable that
takes a rally_id and returns the list of PlayerPosition dicts, or None if
tracking data is unavailable for that rally.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

logger = logging.getLogger(__name__)

IOU_THRESHOLD = 0.3

# Shape of a PlayerPosition dict read from player_tracks.positions_json.
Position = Mapping[str, Any]
# Per-rally positions lookup. Return None if tracking data is missing.
PositionsLookup = Callable[[str], Sequence[Position] | None]


@dataclass
class NormalizedGt:
    """Legacy-shaped GT consumed by every eval/diagnostic script."""

    rallies: dict[str, dict[str, int]] = field(default_factory=dict)
    side_switches: list[int] = field(default_factory=list)
    excluded_rallies: list[str] = field(default_factory=list)
    # Per-rally warnings raised during label resolution.
    warnings: list[str] = field(default_factory=list)


def _iou(a_cx: float, a_cy: float, a_w: float, a_h: float,
         b_cx: float, b_cy: float, b_w: float, b_h: float) -> float:
    ax1, ay1 = a_cx - a_w / 2, a_cy - a_h / 2
    ax2, ay2 = a_cx + a_w / 2, a_cy + a_h / 2
    bx1, by1 = b_cx - b_w / 2, b_cy - b_h / 2
    bx2, by2 = b_cx + b_w / 2, b_cy + b_h / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, a_w) * max(0.0, a_h)
    area_b = max(0.0, b_w) * max(0.0, b_h)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _resolve_label(
    label: Mapping[str, Any],
    positions: Sequence[Position],
) -> int | None:
    """Return the current trackId for a bbox label, or None if no match."""
    frame = int(label["frame"])
    cx, cy = float(label["cx"]), float(label["cy"])
    w, h = float(label["w"]), float(label["h"])

    best_iou = 0.0
    best_track: int | None = None
    for pos in positions:
        if int(pos["frameNumber"]) != frame:
            continue
        iou = _iou(
            cx, cy, w, h,
            float(pos["x"]), float(pos["y"]),
            float(pos["width"]), float(pos["height"]),
        )
        if iou > best_iou:
            best_iou = iou
            best_track = int(pos["trackId"])

    if best_track is None or best_iou < IOU_THRESHOLD:
        return None
    return best_track


def load_player_matching_gt(
    gt_json: Mapping[str, Any] | str | None,
    positions_lookup: PositionsLookup | None = None,
) -> NormalizedGt:
    """Parse bbox-keyed GT into the normalized runtime shape.

    Args:
        gt_json: dict (already parsed), JSON string, or None.
        positions_lookup: required. Called with a rally_id; must return the
            PlayerPosition list (same shape as player_tracks.positions_json)
            or None if unavailable.
    """
    out = NormalizedGt()
    if gt_json is None:
        return out
    if isinstance(gt_json, str):
        gt_json = cast(Mapping[str, Any], json.loads(gt_json))

    out.side_switches = list(
        gt_json.get("sideSwitches", gt_json.get("side_switches", []))
    )
    out.excluded_rallies = list(gt_json.get("excludedRallies", []))

    raw_rallies = cast(Mapping[str, Any], gt_json.get("rallies") or {})
    if not raw_rallies:
        return out

    if positions_lookup is None:
        out.warnings.append(
            "GT requires positions_lookup but none was provided"
        )
        return out

    for rid, entry in raw_rallies.items():
        rid_str = str(rid)
        if not isinstance(entry, Mapping):
            continue
        labels = entry.get("labels") or []

        positions = positions_lookup(rid_str)
        if positions is None:
            msg = f"{rid_str}: no positions available, skipping rally"
            logger.warning(msg)
            out.warnings.append(msg)
            continue

        rally_map: dict[str, int] = {}
        for label in labels:
            track_id = _resolve_label(label, positions)
            if track_id is None:
                msg = (
                    f"{rid_str}: label playerId={label.get('playerId')} "
                    f"frame={label.get('frame')} had no track with "
                    f"IoU>={IOU_THRESHOLD}; dropping"
                )
                logger.warning(msg)
                out.warnings.append(msg)
                continue
            track_key = str(track_id)
            if track_key in rally_map:
                msg = (
                    f"{rid_str}: two labels resolved to track {track_id} "
                    f"(existing playerId={rally_map[track_key]}, new "
                    f"playerId={label['playerId']}); keeping the first, "
                    f"dropping the second — labels likely anchored on "
                    f"overlapping frames"
                )
                logger.warning(msg)
                out.warnings.append(msg)
                continue
            rally_map[track_key] = int(label["playerId"])

        if rally_map:
            out.rallies[rid_str] = rally_map

    return out


def build_positions_lookup_from_db(cursor: Any) -> PositionsLookup:
    """Build a PositionsLookup that reads player_tracks.positions_json on demand.

    Lazy per-rally fetch with a small cache. Prefer `prefetch_positions` for
    bulk eval/diagnostic workloads — it issues a single batched SELECT
    instead of one query per label resolution.

    The cursor is only used for lookups; callers must not interleave
    other queries on it.
    """
    cache: dict[str, Sequence[Position] | None] = {}

    def lookup(rally_id: str) -> Sequence[Position] | None:
        if rally_id in cache:
            return cache[rally_id]
        cursor.execute(
            "SELECT positions_json FROM player_tracks WHERE rally_id = %s",
            (rally_id,),
        )
        row = cursor.fetchone()
        if row is None:
            cache[rally_id] = None
            return None
        raw = row[0]
        if isinstance(raw, str):
            raw = json.loads(raw)
        positions = cast(Sequence[Position], raw or [])
        cache[rally_id] = positions
        return positions

    return lookup


def prefetch_positions(
    cursor: Any,
    rally_ids: Sequence[str],
) -> PositionsLookup:
    """Load positions for many rallies in a single query and return a lookup.

    Use this when you already know which rally ids you need (e.g., bulk
    GT eval across every video in the DB). Avoids the N+1 round-trips
    `build_positions_lookup_from_db` would issue under per-label calls.
    """
    cache: dict[str, Sequence[Position] | None] = {rid: None for rid in rally_ids}
    if rally_ids:
        cursor.execute(
            "SELECT rally_id, positions_json FROM player_tracks "
            "WHERE rally_id = ANY(%s)",
            (list(rally_ids),),
        )
        for rid, raw in cursor.fetchall():
            if isinstance(raw, str):
                raw = json.loads(raw)
            cache[str(rid)] = cast(Sequence[Position], raw or [])

    def lookup(rally_id: str) -> Sequence[Position] | None:
        return cache.get(rally_id)

    return lookup


@dataclass
class DbGtRow:
    video_id: str
    gt: NormalizedGt


def load_all_from_db(
    cursor: Any,
    *,
    video_id_prefix: str | None = None,
) -> list[DbGtRow]:
    """Read every row with player_matching_gt_json from the DB and normalize.

    Replaces duplicated SELECT + parse boilerplate across eval scripts.
    """
    query = (
        "SELECT id, player_matching_gt_json FROM videos "
        "WHERE player_matching_gt_json IS NOT NULL"
    )
    params: list[str] = []
    if video_id_prefix:
        query += " AND id::text LIKE %s"
        params.append(f"{video_id_prefix}%")
    query += " ORDER BY id"

    cursor.execute(query, params)
    rows = cursor.fetchall()

    # Collect every rally id up front so we can batch-fetch positions in
    # one SELECT, avoiding N+1 round-trips during label resolution.
    rally_ids: list[str] = []
    parsed_rows: list[tuple[str, Mapping[str, Any] | None]] = []
    for vid, gt_json in rows:
        if gt_json is None:
            parsed_rows.append((str(vid), None))
            continue
        if isinstance(gt_json, str):
            gt_json = cast(Mapping[str, Any], json.loads(gt_json))
        parsed_rows.append((str(vid), gt_json))
        rally_ids.extend(
            str(rid) for rid in cast(Mapping[str, Any], gt_json.get("rallies") or {}).keys()
        )

    lookup = prefetch_positions(cursor, rally_ids)
    results: list[DbGtRow] = []
    for video_id, gt_json in parsed_rows:
        gt = load_player_matching_gt(gt_json, positions_lookup=lookup)
        results.append(DbGtRow(video_id=video_id, gt=gt))
    return results
