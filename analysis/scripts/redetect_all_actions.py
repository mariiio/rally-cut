"""Re-run contact detection + action classification for all tracked rallies.

Uses stored ball/player positions (from tracking) to re-detect contacts with
the current pipeline (classifier + threshold) and re-classify actions. Saves
updated contacts_json and actions_json back to the database.

This fixes stale stored actions from older pipeline versions.

Usage:
    cd analysis
    uv run python scripts/redetect_all_actions.py                # Dry run
    uv run python scripts/redetect_all_actions.py --apply        # Write to DB
    uv run python scripts/redetect_all_actions.py --video <id>   # Single video
    uv run python scripts/redetect_all_actions.py --rally-id <id>  # Single rally
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any, cast

from rallycut.court.calibration import CourtCalibrator
from rallycut.court.net_line_estimator import NetLine, estimate_net_line_from_s3
from rallycut.court.net_top_keypoint_reader import NetTopLine, read_net_top_from_s3
from rallycut.evaluation.db import get_connection
from rallycut.evaluation.tracking.db import load_court_calibration
from rallycut.tracking.action_classifier import ACTION_PIPELINE_VERSION, classify_rally_actions
from rallycut.tracking.ball_tracker import BallPosition as BallPos
from rallycut.tracking.contact_detector import CONTACT_PIPELINE_VERSION, detect_contacts
from rallycut.tracking.match_tracker import build_match_team_assignments
from rallycut.tracking.player_tracker import PlayerPosition as PlayerPos
from rallycut.tracking.sequence_action_runtime import get_sequence_probs


def _resolve_video_s3(
    video_id: str,
    cache: dict[str, tuple[str | None, int | None, int | None]],
) -> tuple[str | None, int | None, int | None]:
    """Return (s3_key, width, height) for the video. Cached per-process."""
    if video_id in cache:
        return cache[video_id]
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT proxy_s3_key, original_s3_key, width, height "
                "FROM videos WHERE id = %s",
                (video_id,),
            )
            row = cur.fetchone()
    if not row:
        cache[video_id] = (None, None, None)
        return cache[video_id]
    proxy_key = cast(str | None, row[0])
    original_key = cast(str | None, row[1])
    width = cast(int | None, row[2])
    height = cast(int | None, row[3])
    s3_key = proxy_key or original_key
    cache[video_id] = (s3_key, width, height)
    return cache[video_id]


def _resolve_net_top_line(
    video_id: str,
    cache: dict[str, NetTopLine | None],
    video_s3_cache: dict[str, tuple[str | None, int | None, int | None]],
    *,
    quiet: bool = False,
) -> NetTopLine | None:
    """v9: read net-top from the 8-keypoint court model. Cached per-process
    by video_id. Pulls the proxy MP4 from S3/MinIO via
    `read_net_top_from_s3` so the NetTopLine is computed once on the
    FULL source video (per the v9 contract).
    """
    if video_id in cache:
        return cache[video_id]
    s3_key, width, height = _resolve_video_s3(video_id, video_s3_cache)
    if not s3_key or not width or not height:
        cache[video_id] = None
        return None
    try:
        ntl = read_net_top_from_s3(
            s3_key=s3_key,
            video_id=video_id,
            use_cache=True,
        )
    except Exception as exc:
        if not quiet:
            print(f"  [net-top] {video_id[:8]} failed ({type(exc).__name__}); falling back to v8 NLE")
        ntl = None
    cache[video_id] = ntl
    return ntl


def _resolve_net_line(
    video_id: str,
    cache: dict[str, NetLine | None],
    video_s3_cache: dict[str, tuple[str | None, int | None, int | None]],
    *,
    quiet: bool = False,
) -> NetLine | None:
    """Compute (or recall) `estimate_net_line` for `video_id`. Cached
    per-process by video_id. Pulls the proxy MP4 from S3/MinIO via
    `estimate_net_line_from_s3` so the NetLine is computed once on the
    FULL source video, not per-rally clip (v8 contract).
    """
    if video_id in cache:
        return cache[video_id]
    s3_key, width, height = _resolve_video_s3(video_id, video_s3_cache)
    if not s3_key or not width or not height:
        cache[video_id] = None
        return None
    try:
        nl = estimate_net_line_from_s3(
            s3_key=s3_key,
            video_id=video_id,
            image_width=int(width),
            image_height=int(height),
            use_cache=True,
        )
    except Exception as exc:
        if not quiet:
            print(f"  [net-line] {video_id[:8]} failed ({type(exc).__name__}); falling back to M4")
        nl = None
    cache[video_id] = nl
    return nl


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-detect contacts + actions for all rallies")
    parser.add_argument("--apply", action="store_true", help="Write changes to DB (default: dry run)")
    parser.add_argument("--video", type=str, help="Only process this video ID")
    parser.add_argument("--rally-id", type=str, help="Only process this rally ID")
    args = parser.parse_args()

    # Load match team assignments and rally data
    where_clauses = ["pt.ball_positions_json IS NOT NULL"]
    params: list[str] = []
    if args.video:
        where_clauses.append("r.video_id = %s")
        params.append(args.video)
    if args.rally_id:
        where_clauses.append("r.id = %s")
        params.append(args.rally_id)
    where_sql = " AND ".join(where_clauses)

    with get_connection() as conn:
        with conn.cursor() as cur:
            # Pre-load per-rally positions so build_match_team_assignments can
            # use _teams_from_positions + verify_team_assignments (positional
            # path) instead of the legacy pid-pairing fallback. Without
            # positions, the fallback assumes PIDs 1,2 are on the near side —
            # silently wrong on the ~18% of fleet videos where the matcher's
            # blind PID anchor put 1,2 on far instead (kuku, koko, ~10
            # others). See side_switch_kuku_koko_diagnostic_2026_05_19.
            cur.execute(
                "SELECT rally_id, positions_json FROM player_tracks "
                "WHERE positions_json IS NOT NULL"
            )
            rally_positions_by_rid: dict[str, list[PlayerPos]] = {}
            for rid_raw, pos_raw in cur.fetchall():
                rid_str = str(rid_raw)
                pos_list = pos_raw if isinstance(pos_raw, list) else []
                rally_positions_by_rid[rid_str] = [
                    PlayerPos(
                        frame_number=p.get("frameNumber", 0),
                        track_id=p.get("trackId", 0),
                        x=p.get("x", 0),
                        y=p.get("y", 0),
                        width=p.get("width", 0),
                        height=p.get("height", 0),
                        confidence=p.get("confidence", 0),
                    )
                    for p in pos_list
                    if isinstance(p, dict)
                ]

            cur.execute(
                "SELECT v.id, v.match_analysis_json FROM videos v "
                "WHERE v.match_analysis_json IS NOT NULL"
            )
            # Derive per-rally team assignments via the canonical match_tracker
            # helper. min_confidence=0.0 mirrors `reattribute-actions`'s "all_teams"
            # path, which is what stamps teamAssignments onto every rally.
            # The previous implementation here was reading `match_analysis_json.
            # team_assignments` which is a different (legacy) schema and now
            # returns 0 rallies — leaving teamAssignments null fleet-wide.
            match_teams_by_rally: dict[str, dict[int, int]] = {}
            for vid, mj_raw in cur.fetchall():
                mj = cast(dict[str, Any], mj_raw)
                if not mj:
                    continue
                match_teams_by_rally.update(
                    build_match_team_assignments(
                        mj, min_confidence=0.0,
                        rally_positions=rally_positions_by_rid,
                    )
                )

            cur.execute(f"""
                SELECT r.id, r.video_id, pt.id as pt_id,
                       pt.ball_positions_json, pt.positions_json,
                       pt.frame_count, pt.court_split_y,
                       pt.actions_json
                FROM rallies r
                JOIN player_tracks pt ON pt.rally_id = r.id
                WHERE {where_sql}
                ORDER BY r.video_id, r.start_ms
            """, params)
            rows = cur.fetchall()

    print(f"Found {len(rows)} rallies with ball positions")
    if not args.apply:
        print("  DRY RUN — use --apply to write changes to DB\n")

    # Load court calibrations
    calibrators: dict[str, CourtCalibrator | None] = {}
    # Shared per-video S3 metadata cache (proxy_s3_key, width, height)
    # so the two net-top resolvers below don't double-query the DB.
    video_s3_cache: dict[str, tuple[str | None, int | None, int | None]] = {}
    # v9: per-video NetTopLine cache (8-keypoint direct observation —
    # new primary in the cascade). Computed once per video via
    # `read_net_top_from_s3`.
    net_top_lines: dict[str, NetTopLine | None] = {}
    # v8: per-video NetLine cache (solvePnP — fallback below v9).
    net_lines: dict[str, NetLine | None] = {}

    t_start = time.monotonic()
    updated = 0
    skipped = 0
    errors = 0

    for i, row in enumerate(rows):
        rally_id = str(row[0])
        video_id = str(row[1])
        pt_id = cast(int, row[2])
        ball_json = cast(list[dict[str, Any]], row[3])
        positions_json = cast(list[dict[str, Any]] | None, row[4])
        frame_count = cast(int | None, row[5])
        court_split_y = cast(float | None, row[6])
        existing_actions_json = cast(dict[str, Any] | None, row[7]) or {}

        # Load court calibration (cached per video)
        if video_id not in calibrators:
            corners = load_court_calibration(video_id)
            if corners and len(corners) == 4:
                cal = CourtCalibrator()
                cal.calibrate([(c["x"], c["y"]) for c in corners])
                calibrators[video_id] = cal
            else:
                calibrators[video_id] = None

        # Convert DB dicts to typed objects
        ball_positions = [
            BallPos(
                frame_number=bp["frameNumber"], x=bp["x"], y=bp["y"],
                confidence=bp.get("confidence", 1.0),
            )
            for bp in ball_json
            if bp.get("x", 0) > 0 or bp.get("y", 0) > 0
        ]

        if not ball_positions:
            skipped += 1
            continue

        player_positions = []
        if positions_json:
            for pp in positions_json:
                pos = PlayerPos(
                    frame_number=pp["frameNumber"], track_id=pp["trackId"],
                    x=pp["x"], y=pp["y"], width=pp["width"], height=pp["height"],
                    confidence=pp.get("confidence", 1.0),
                )
                # Preserve pose keypoints + embedding so the dynamic-feature
                # scorer can compute pose features at inference time. Without
                # this the scorer sees default-zero pose features and the
                # training-inference distribution mismatches.
                kps = pp.get("keypoints")
                if kps and len(kps) >= 17:
                    pos.keypoints = kps
                emb = pp.get("embedding")
                if emb:
                    pos.embedding = emb
                player_positions.append(pos)

        match_teams = match_teams_by_rally.get(rally_id)

        try:
            # Compute MS-TCN++ probs once. Required for v1.1 synthetic-serve
            # placement AND for apply_sequence_override (the post-Viterbi
            # action-type correction). Returns None if weights are missing —
            # in that case both effects degrade gracefully.
            sequence_probs = get_sequence_probs(
                ball_positions, player_positions, court_split_y,
                frame_count or 0, match_teams,
                calibrator=calibrators.get(video_id),
            )

            # v9: resolve NetTopLine + v8 NetLine once per video.
            # Both passed through the cascade — `_prepare_candidates`
            # picks net_top_line first, then net_line, then M4, then v6.
            net_top_line = _resolve_net_top_line(
                video_id, net_top_lines, video_s3_cache,
            )
            net_line = _resolve_net_line(
                video_id, net_lines, video_s3_cache,
            )

            contacts = detect_contacts(
                ball_positions=ball_positions,
                player_positions=player_positions,
                frame_count=frame_count or None,
                court_calibrator=calibrators.get(video_id),
                net_line=net_line,
                net_top_line=net_top_line,
                team_assignments=match_teams,
                sequence_probs=sequence_probs,
            )

            rally_actions = classify_rally_actions(
                contacts, rally_id,
                use_classifier=True,
                team_assignments=match_teams,
                match_team_assignments=match_teams,
                sequence_probs=sequence_probs,
            )

            # Serialize via RallyActions.to_dict() so teamAssignments + servingTeam
            # are emitted from the current pipeline output (was previously dropped:
            # this script only wrote {"actions": [...]} which stripped both fields
            # on every deploy). Layer over the existing actions_json so any
            # downstream-stamped fields (formation hints, etc.) survive.
            new_contacts_json = contacts.to_dict()
            new_actions_json = {**existing_actions_json, **rally_actions.to_dict()}

            n_contacts = len(contacts.contacts)
            n_actions = len(rally_actions.actions)

            if args.apply:
                with get_connection() as wconn:
                    with wconn.cursor() as wcur:
                        wcur.execute(
                            "UPDATE player_tracks SET "
                            "contacts_json = %s, actions_json = %s, "
                            "contacts_pipeline_version = %s, "
                            "actions_pipeline_version = %s "
                            "WHERE id = %s",
                            (
                                json.dumps(new_contacts_json),
                                json.dumps(new_actions_json),
                                CONTACT_PIPELINE_VERSION,
                                ACTION_PIPELINE_VERSION,
                                pt_id,
                            ),
                        )
                    wconn.commit()

            updated += 1
            elapsed = time.monotonic() - t_start
            print(
                f"  [{i+1}/{len(rows)}] {rally_id[:8]}: "
                f"{n_contacts} contacts, {n_actions} actions ({elapsed:.1f}s)"
            )

        except Exception as e:
            errors += 1
            print(f"  ERROR {rally_id[:8]}: {e}")

    elapsed = time.monotonic() - t_start
    print(f"\nDone: {updated} updated, {skipped} skipped, {errors} errors ({elapsed:.1f}s)")
    if not args.apply:
        print("  DRY RUN — no changes written. Use --apply to write.")


if __name__ == "__main__":
    main()
