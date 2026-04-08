"""Convert videos.player_matching_gt_json to the pure bbox-keyed format.

This is a one-shot migration. It reads whatever is currently on disk —
either legacy v1 `{trackIdStr: playerId}` mappings or an intermediate
hybrid v2 with `legacyTrackId` — and writes canonical pure v2:

    {
      "rallies": {
        "<rid>": {
          "labels": [
            {"playerId": 1, "frame": 42,
             "cx": 0.41, "cy": 0.63, "w": 0.08, "h": 0.28},
            ...
          ]
        }
      },
      "sideSwitches": [...],
      "excludedRallies": [...]
    }

For each (rally, trackId, playerId) triple extracted from the input, the
script picks an ISOLATED frame for that track — one where the track's
bbox has low IoU with every other track at that frame — so the label's
bbox unambiguously maps back to the same track at load time. Isolation
is prioritized over raw visibility to prevent IoU ambiguity when nearby
players overlap (e.g. serve-receive clusters).

Labels whose track id is not in the current positions_json are DROPPED
with a warning. They were already dead — eval_match_players could not
look them up — so nothing of value is lost.

Defaults to --dry-run. Requires --apply to write.

Usage:
    uv run python analysis/scripts/migrate_gt_to_bbox_format.py
    uv run python analysis/scripts/migrate_gt_to_bbox_format.py --apply
    uv run python analysis/scripts/migrate_gt_to_bbox_format.py --video-id abc
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, cast

from psycopg.types.json import Json

from rallycut.evaluation.db import get_connection
from rallycut.evaluation.gt_loader import _iou

logger = logging.getLogger(__name__)

LOG_DIR = Path(__file__).parent.parent / "backups" / "player_matching_gt"

# Isolation threshold: if the chosen frame has any other track with IoU >=
# this value against the target track's bbox, the frame is considered
# ambiguous and we search for a better one.
ISOLATION_MAX_IOU = 0.1


@dataclass
class RallyMigration:
    rally_id: str
    source_triples: list[tuple[int, int]]  # (trackId, playerId)
    labels: list[dict[str, Any]]
    dropped: list[int]  # trackIds with no resolvable bbox


@dataclass
class VideoMigration:
    video_id: str
    rallies: list[RallyMigration]
    side_switches: list[int]
    excluded_rallies: list[str]

    @property
    def has_drops(self) -> bool:
        return any(r.dropped for r in self.rallies)

    def to_pure_v2(self) -> dict[str, Any]:
        return {
            "rallies": {
                r.rally_id: {"labels": r.labels}
                for r in self.rallies
                if r.labels
            },
            "sideSwitches": self.side_switches,
            "excludedRallies": self.excluded_rallies,
        }


def _extract_triples(
    rally_entry: Any,
) -> list[tuple[int, int]]:
    """Pull (trackId, playerId) pairs from any known historical shape.

    Handles three cases:
      - v1: rallies[rid] = {"1": 2, "3": 4}                          (dict[str,int])
      - hybrid v2 with legacyTrackId: rallies[rid] = {"labels": [...]}
      - pure v2: rallies[rid] = {"labels": [{..., cx, cy, ...}]}     (already migrated)

    For pure v2 labels (no legacyTrackId), the trackId is unknown at this
    point — we'd need to IoU-resolve against current positions. That's
    idempotent with the rest of the pipeline, so we return an empty list
    and the caller skips the rally (it's already migrated).
    """
    if not isinstance(rally_entry, Mapping):
        return []

    labels = rally_entry.get("labels")
    if isinstance(labels, list):
        # Hybrid or pure v2. Only legacyTrackId entries are extractable.
        out: list[tuple[int, int]] = []
        for label in labels:
            if not isinstance(label, Mapping):
                continue
            legacy = label.get("legacyTrackId")
            if legacy is None:
                continue
            out.append((int(legacy), int(label["playerId"])))
        return out

    # v1 shape: {trackIdStr: playerId}
    return [(int(k), int(v)) for k, v in rally_entry.items()]


def _pick_isolated_frame(
    positions: list[dict[str, Any]],
    track_id: int,
) -> dict[str, Any] | None:
    """Return the best frame for `track_id` — large bbox + low IoU with peers.

    Scoring: prefer frames where the track's bbox is large AND no other
    track's bbox at that frame overlaps it heavily. Falls back to
    largest-area if no isolated frame exists.
    """
    track_positions = [p for p in positions if int(p.get("trackId", -1)) == track_id]
    if not track_positions:
        return None

    # Index positions by frame for peer lookups.
    by_frame: dict[int, list[dict[str, Any]]] = {}
    for p in positions:
        by_frame.setdefault(int(p["frameNumber"]), []).append(p)

    best: dict[str, Any] | None = None
    best_score = -1.0
    fallback: dict[str, Any] | None = None
    fallback_area = 0.0

    for tpos in track_positions:
        area = float(tpos.get("width", 0.0)) * float(tpos.get("height", 0.0))
        if area > fallback_area:
            fallback_area = area
            fallback = tpos

        # Compute worst-case IoU against peers at this frame.
        peers = by_frame.get(int(tpos["frameNumber"]), [])
        max_peer_iou = 0.0
        for peer in peers:
            if int(peer["trackId"]) == track_id:
                continue
            iou = _iou(
                float(tpos["x"]), float(tpos["y"]),
                float(tpos["width"]), float(tpos["height"]),
                float(peer["x"]), float(peer["y"]),
                float(peer["width"]), float(peer["height"]),
            )
            if iou > max_peer_iou:
                max_peer_iou = iou

        if max_peer_iou >= ISOLATION_MAX_IOU:
            continue  # ambiguous frame, skip for isolation pass

        # Score = area (higher is better). All surviving frames are isolated.
        if area > best_score:
            best_score = area
            best = tpos

    return best or fallback


def _plan_video(
    video_id: str,
    gt_data: dict[str, Any],
    rally_positions: dict[str, list[dict[str, Any]]],
) -> VideoMigration:
    rallies: list[RallyMigration] = []
    raw_rallies = cast(Mapping[str, Any], gt_data.get("rallies") or {})
    for rid, entry in raw_rallies.items():
        triples = _extract_triples(entry)
        positions = rally_positions.get(rid) or []

        labels: list[dict[str, Any]] = []
        dropped: list[int] = []
        for track_id, player_id in triples:
            pos = _pick_isolated_frame(positions, track_id) if positions else None
            if pos is None:
                dropped.append(track_id)
                continue
            labels.append({
                "playerId": player_id,
                "frame": int(pos["frameNumber"]),
                "cx": float(pos["x"]),
                "cy": float(pos["y"]),
                "w": float(pos["width"]),
                "h": float(pos["height"]),
            })

        rallies.append(RallyMigration(
            rally_id=rid,
            source_triples=triples,
            labels=labels,
            dropped=dropped,
        ))

    return VideoMigration(
        video_id=video_id,
        rallies=rallies,
        side_switches=list(
            gt_data.get("sideSwitches", gt_data.get("side_switches", []))
        ),
        excluded_rallies=list(gt_data.get("excludedRallies", [])),
    )


def _load_all_positions(cursor: Any, rally_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    if not rally_ids:
        return {}
    cursor.execute(
        "SELECT rally_id, positions_json FROM player_tracks "
        "WHERE rally_id = ANY(%s)",
        (rally_ids,),
    )
    out: dict[str, list[dict[str, Any]]] = {}
    for rid, positions in cursor.fetchall():
        if isinstance(positions, str):
            positions = json.loads(positions)
        out[str(rid)] = list(positions or [])
    return out


def _fmt_bbox(label: dict[str, Any]) -> str:
    return (
        f"f{label['frame']:04d} "
        f"cx={label['cx']:.3f} cy={label['cy']:.3f} "
        f"w={label['w']:.3f} h={label['h']:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="Actually write to DB (default: dry-run)")
    parser.add_argument("--video-id", type=str, default=None,
                        help="Only migrate this video id (prefix match allowed)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    with get_connection() as conn:
        with conn.cursor() as cur:
            query = (
                "SELECT id, player_matching_gt_json FROM videos "
                "WHERE player_matching_gt_json IS NOT NULL"
            )
            params: list[str] = []
            if args.video_id:
                query += " AND id::text LIKE %s"
                params.append(f"{args.video_id}%")
            query += " ORDER BY id"
            cur.execute(query, params)
            rows = cur.fetchall()

            if not rows:
                logger.error("No videos with player_matching_gt_json.")
                sys.exit(1)

            plans: list[tuple[VideoMigration, dict[str, Any]]] = []
            already_pure: list[str] = []
            for vid, gt_json in rows:
                video_id = str(vid)
                gt_data = cast(dict[str, Any], gt_json)

                rally_ids = list(
                    cast(dict[str, Any], gt_data.get("rallies") or {}).keys()
                )
                rally_positions = _load_all_positions(cur, rally_ids)
                plan = _plan_video(video_id, gt_data, rally_positions)

                # If every rally has empty source triples, the input is
                # already pure v2 (bbox-only, no legacyTrackId). Skip.
                if all(not r.source_triples for r in plan.rallies):
                    already_pure.append(video_id)
                    continue

                plans.append((plan, gt_data))

            if already_pure:
                logger.info(
                    "Skipping %d videos already in pure v2 format.",
                    len(already_pure),
                )

            if not plans:
                logger.info("Nothing to migrate.")
                return

            total_rallies = 0
            total_labels = 0
            total_dropped = 0
            videos_with_drops: list[str] = []
            for plan, _gt_data in plans:
                total_rallies += len(plan.rallies)
                for r in plan.rallies:
                    total_labels += len(r.labels)
                    total_dropped += len(r.dropped)
                if plan.has_drops:
                    videos_with_drops.append(plan.video_id)

            logger.info("=" * 72)
            logger.info("Migration plan: %d videos, %d rallies, %d labels",
                        len(plans), total_rallies, total_labels)
            logger.info("  dropped (stale track id):      %d", total_dropped)
            logger.info("  videos with any drops:         %d", len(videos_with_drops))
            logger.info("=" * 72)

            for plan, _ in plans:
                tag = "lossy  " if plan.has_drops else "ok     "
                logger.info(
                    "[%s] %s  rallies=%d  labels=%d  dropped=%d",
                    tag, plan.video_id, len(plan.rallies),
                    sum(len(r.labels) for r in plan.rallies),
                    sum(len(r.dropped) for r in plan.rallies),
                )
                if plan.has_drops:
                    for r in plan.rallies:
                        if r.dropped:
                            logger.info(
                                "    %s  triples=%s  dropped_track_ids=%s",
                                r.rally_id, r.source_triples, r.dropped,
                            )

            first = plans[0][0]
            sample_rally = next((r for r in first.rallies if r.labels), None)
            if sample_rally:
                logger.info("")
                logger.info("Sample labels for %s / %s:",
                            first.video_id, sample_rally.rally_id)
                for lb in sample_rally.labels:
                    logger.info("  P%d %s", lb["playerId"], _fmt_bbox(lb))

            if not args.apply:
                logger.info("")
                logger.info("DRY RUN — pass --apply to write %d videos.", len(plans))
                return

            ts = datetime.now(UTC).strftime("%Y-%m-%d-%H%M%S")
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            log_path = LOG_DIR / f"migration_log_{ts}.jsonl"
            applied = 0
            with open(log_path, "w") as logf:
                for plan, gt_data_orig in plans:
                    payload = plan.to_pure_v2()
                    cur.execute(
                        "UPDATE videos SET player_matching_gt_json = %s "
                        "WHERE id = %s",
                        (Json(payload), plan.video_id),
                    )
                    if cur.rowcount != 1:
                        logger.error(
                            "UPDATE for %s affected %d rows — aborting, rollback.",
                            plan.video_id, cur.rowcount,
                        )
                        conn.rollback()
                        sys.exit(1)
                    logf.write(json.dumps({
                        "video_id": plan.video_id,
                        "old": gt_data_orig,
                        "new": payload,
                        "dropped_track_ids_by_rally": {
                            r.rally_id: r.dropped for r in plan.rallies if r.dropped
                        },
                    }) + "\n")
                    applied += 1
                    logger.info("  applied %s (%d labels)",
                                plan.video_id,
                                sum(len(r.labels) for r in plan.rallies))
                conn.commit()

            logger.info("")
            logger.info("Applied %d videos. Per-row log: %s", applied, log_path)


if __name__ == "__main__":
    main()
