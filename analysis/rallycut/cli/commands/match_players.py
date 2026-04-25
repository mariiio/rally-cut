"""Match players across rallies for consistent IDs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import typer
from rich.console import Console
from rich.table import Table

from rallycut.cli.utils import handle_errors

console = Console()


# Schema version for canonicalPidMapJson. Bump only on breaking shape changes
# (read-side helpers fall back to legacy fields when version is unknown).
CANONICAL_PID_MAP_VERSION = 1


def _canonical_pid_map_payload(
    canonical_map: dict[str, dict[int, int]],
    crop_rows: list[tuple[Any, ...]],
) -> dict[str, Any]:
    """Build the persisted canonicalPidMapJson payload.

    `sourceRefCropsSha` digests the (player_id, frame_ms, bbox) tuples sorted
    by player_id then frame_ms. Any change to a ref crop yields a different
    sha — Phase 5's invalidation can compare without re-fetching the crops.
    Hashing the tuples (not the JPEG bytes) costs zero S3 reads and is
    sufficient because the tuple uniquely identifies the crop content.
    """
    sortable = sorted(
        (
            (
                int(r[0]),  # player_id
                int(r[1]),  # frame_ms
                float(r[2]), float(r[3]), float(r[4]), float(r[5]),  # bbox
            )
            for r in crop_rows
        ),
        key=lambda t: (t[0], t[1]),
    )
    sha = hashlib.sha256(json.dumps(sortable, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "version": CANONICAL_PID_MAP_VERSION,
        "sourceRefCropsSha": sha,
        "rallies": {
            rid: {str(tid): pid for tid, pid in rally_map.items()}
            for rid, rally_map in canonical_map.items()
        },
    }


def _load_db_reference_crops(
    video_id: str,
    video_path: Path,
    quiet: bool,
) -> tuple[list[tuple[Any, ...]], Any, dict[int, list[Any]]]:
    """Load reference crops from DB and build frozen HSV + ReID profiles.

    Returns:
        (crop_rows, reference_profiles, bgr_crops_by_player). ``crop_rows`` is
        the raw DB rows ``(player_id, frame_ms, bbox_x, bbox_y, bbox_w, bbox_h)``
        used downstream to compute the canonical-pid-map source sha and to
        rebuild :class:`IdentityAnchors`. ``bgr_crops_by_player`` is the
        per-player list of full-resolution BGR crops extracted from the source
        video; reusing it for the canonical-map path saves a second video pass.
        Empty containers when no crops exist.
    """
    from rallycut.evaluation.db import get_connection

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT player_id, frame_ms, bbox_x, bbox_y, bbox_w, bbox_h
                   FROM player_reference_crops
                   WHERE video_id = %s
                   ORDER BY player_id, created_at""",
                [video_id],
            )
            crop_rows = cur.fetchall()

    if not crop_rows:
        return [], None, {}

    crop_infos: list[dict[str, Any]] = []
    for r in crop_rows:
        pid_val, fms_val, bx_val, by_val, bw_val, bh_val = r
        crop_infos.append({
            "player_id": pid_val, "frame_ms": fms_val,
            "bbox_x": bx_val, "bbox_y": by_val,
            "bbox_w": bw_val, "bbox_h": bh_val,
        })

    import cv2
    import numpy as np

    from rallycut.tracking.player_features import (
        build_profiles_from_crops,
        extract_appearance_features,
        extract_bbox_crop,
    )

    # Build HSV profiles from reference crops (existing appearance pipeline).
    # Also collect BGR crops for DINOv2 ReID embedding extraction.
    cap = cv2.VideoCapture(str(video_path))
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    features_by_player: dict[int, list[Any]] = {}
    bgr_crops_by_player: dict[int, list[Any]] = {}
    for info in sorted(crop_infos, key=lambda c: int(c["frame_ms"])):
        pid = int(info["player_id"])
        frame_ms = int(info["frame_ms"])

        if fw > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, float(frame_ms))
            ret, frame = cap.read()
            if ret and frame is not None:
                frame_arr: Any = np.asarray(frame)
                bx = float(info["bbox_x"])
                by = float(info["bbox_y"])
                bw = float(info["bbox_w"])
                bh = float(info["bbox_h"])
                features = extract_appearance_features(
                    frame=frame_arr,
                    track_id=0,
                    frame_number=0,
                    bbox=(bx, by, bw, bh),
                    frame_width=fw,
                    frame_height=fh,
                )
                features_by_player.setdefault(pid, []).append(features)

                # Extract BGR crop for ReID embedding
                crop = extract_bbox_crop(frame_arr, (bx, by, bw, bh), fw, fh)
                if crop is not None:
                    bgr_crops_by_player.setdefault(pid, []).append(crop)

    cap.release()

    # Extract DINOv2 embeddings from reference crops (per-player average).
    # Falls back to HSV-only if DINOv2 fails (network, OOM, etc).
    reid_embeddings_by_player: dict[int, Any] = {}
    if bgr_crops_by_player:
        try:
            from rallycut.tracking.reid_embeddings import extract_backbone_features

            for pid, crops in bgr_crops_by_player.items():
                if not crops:
                    continue
                embeddings = extract_backbone_features(crops)  # (N, 384)
                mean_emb = embeddings.mean(axis=0)
                norm = np.linalg.norm(mean_emb)
                if norm > 0:
                    mean_emb /= norm
                reid_embeddings_by_player[pid] = mean_emb
        except Exception:
            import logging

            logging.getLogger(__name__).warning(
                "DINOv2 embedding extraction failed, falling back to HSV-only",
                exc_info=True,
            )
            reid_embeddings_by_player = {}

    reference_profiles = build_profiles_from_crops(
        features_by_player,
        reid_embeddings_by_player=reid_embeddings_by_player or None,
    )

    if not quiet:
        for pid in sorted(reference_profiles):
            n = len(features_by_player.get(pid, []))
            reid_str = " + ReID" if pid in reid_embeddings_by_player else ""
            console.print(f"  Reference profile P{pid}: {n} crop(s){reid_str}")
        console.print()

    return crop_rows, reference_profiles, bgr_crops_by_player


def _reverse_rally_positions(
    rally: Any,
    inverse: dict[int, int],
) -> None:
    """Reverse remapped positions and primaryTrackIds in a RallyTrackData in place."""
    for p in rally.positions:
        if p.track_id in inverse:
            p.track_id = inverse[p.track_id]
    rally.primary_track_ids = [
        inverse.get(tid, tid) for tid in rally.primary_track_ids
    ]


@handle_errors
def match_players(
    video_id: str = typer.Argument(
        ...,
        help="Video ID to match players across rallies",
    ),
    output: Path | None = typer.Option(
        None,
        "--output", "-o",
        help="Output JSON file for player matching results",
    ),
    num_samples: int = typer.Option(
        12,
        "--num-samples",
        help="Number of frames to sample per track for appearance extraction",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet", "-q",
        help="Suppress progress output",
    ),
    reference_crops_json: Path | None = typer.Option(
        None,
        "--reference-crops-json",
        help="JSON file with user-selected reference crops [{playerId, cropPath}, ...]",
    ),
    use_existing_profiles: bool = typer.Option(
        False,
        "--use-existing-profiles",
        help=(
            "Use existing player profiles from DB as frozen anchors instead of "
            "rebuilding. For single-rally retrack: classifies new tracks against "
            "established profiles from previous full match-players run."
        ),
    ),
) -> None:
    """Match players across rallies for consistent player IDs (1-4).

    Uses appearance features (skin tone, jersey color, body proportions)
    to assign consistent player IDs across all rallies in a video.
    Detects side switches based on appearance mismatch.

    Example:
        rallycut match-players abc123
        rallycut match-players abc123 -o result.json
    """
    from rallycut.cli.commands.remap_track_ids import _invert_mapping, _should_reverse
    from rallycut.court.calibration import CourtCalibrator
    from rallycut.evaluation.db import get_connection
    from rallycut.evaluation.tracking.db import get_video_path, load_rallies_for_video
    from rallycut.tracking.match_tracker import MatchPlayersResult, match_players_across_rallies

    if not quiet:
        console.print(f"[bold]Cross-Rally Player Matching:[/bold] video {video_id[:8]}...")

    # Load rallies from DB
    rallies = load_rallies_for_video(video_id)
    if not rallies:
        console.print("[red]Error:[/red] No tracked rallies found for this video.")
        console.print("[dim]Hint: Run player tracking first with 'rallycut track-players'[/dim]")
        raise typer.Exit(1)

    if not quiet:
        console.print(f"  Loaded {len(rallies)} rallies")

    # Load existing match analysis to check for appliedFullMapping
    old_match_analysis: dict[str, Any] | None = None
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT match_analysis_json FROM videos WHERE id = %s",
                [video_id],
            )
            row = cur.fetchone()
            if row and row[0]:
                old_match_analysis = cast(dict[str, Any], row[0])

    # Reverse remapped positions so matching always sees original tracker IDs.
    # Persist the reversal to DB so remap-track-ids starts from original IDs.
    if old_match_analysis:
        old_entries_by_id: dict[str, dict[str, Any]] = {}
        for entry in old_match_analysis.get("rallies", []):
            rid = entry.get("rallyId") or entry.get("rally_id", "")
            if rid:
                old_entries_by_id[rid] = entry

        reversed_ids: set[str] = set()
        for rally in rallies:
            old_entry = old_entries_by_id.get(rally.rally_id)
            if not old_entry:
                continue
            if not old_entry.get("remapApplied", False):
                continue
            applied_raw = old_entry.get("appliedFullMapping")
            if not applied_raw:
                continue
            applied = {int(k): int(v) for k, v in applied_raw.items()}
            pos_dicts = [{"trackId": p.track_id} for p in rally.positions]
            if _should_reverse(pos_dicts, applied):
                inverse = _invert_mapping(applied)
                _reverse_rally_positions(rally, inverse)
                reversed_ids.add(rally.rally_id)

        # Persist reversed positions to DB so remap-track-ids sees original IDs.
        # All updates are committed atomically.
        if reversed_ids:
            with get_connection() as conn:
                with conn.cursor() as cur:
                    for rally in rallies:
                        if rally.rally_id not in reversed_ids:
                            continue
                        pos_data = [
                            {"trackId": p.track_id, "frameNumber": p.frame_number,
                             "x": p.x, "y": p.y, "width": p.width, "height": p.height,
                             "confidence": p.confidence}
                            for p in rally.positions
                        ]
                        cur.execute(
                            "UPDATE player_tracks SET positions_json = %s, "
                            "primary_track_ids = %s WHERE rally_id = %s",
                            [json.dumps(pos_data),
                             json.dumps([int(t) for t in rally.primary_track_ids]),
                             rally.rally_id],
                        )
                conn.commit()

        if reversed_ids and not quiet:
            console.print(
                f"  Reversed previous remap on {len(reversed_ids)} rallies"
            )

    # Resolve video path
    video_path = get_video_path(video_id)
    if video_path is None:
        console.print("[red]Error:[/red] Could not resolve video file.")
        raise typer.Exit(1)

    # Load court calibration for authoritative near/far side classification.
    # Requires 4 court corners to compute a valid homography.
    court_calibrator: CourtCalibrator | None = None
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT court_calibration_json FROM videos WHERE id = %s",
                [video_id],
            )
            cal_row = cur.fetchone()
    if cal_row and cal_row[0] and isinstance(cal_row[0], list) and len(cal_row[0]) == 4:
        court_calibrator = CourtCalibrator()
        court_calibrator.calibrate([(c["x"], c["y"]) for c in cal_row[0]])
        if not court_calibrator.is_calibrated:
            court_calibrator = None

    if not quiet:
        cal_str = " (court calibration)" if court_calibrator else ""
        console.print(f"  Video: {video_path.name}{cal_str}")
        console.print()

    # Load reference crop profiles: from JSON file, or from DB
    reference_profiles = None
    # Carried forward into the canonical-pid-map computation. Only the DB
    # path populates these; the JSON-file path leaves canonicalPidMapJson
    # alone.
    canonical_crop_rows: list[tuple[Any, ...]] = []
    canonical_bgr_crops: dict[int, list[Any]] = {}

    # Try DB reference crops first (unless JSON file is explicitly provided)
    if reference_crops_json is None:
        canonical_crop_rows, reference_profiles, canonical_bgr_crops = (
            _load_db_reference_crops(video_id, video_path, quiet)
        )

    if reference_crops_json is not None:
        import cv2
        import numpy as np

        from rallycut.tracking.player_features import (
            build_profiles_from_crops,
            extract_appearance_features,
            extract_bbox_crop,
        )

        with open(reference_crops_json) as f:
            crop_entries = json.load(f)

        # Extract features from the original video at the stored bbox coordinates.
        # This produces features directly comparable to extract_rally_appearances(),
        # unlike using the small JPEG thumbnails which include padding/background.
        # Also collect BGR crops for DINOv2 ReID embedding extraction.
        cap = cv2.VideoCapture(str(video_path))
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        crops_by_player: dict[int, list[Any]] = {}
        bgr_crops_by_player_json: dict[int, list[Any]] = {}
        for entry in crop_entries:
            pid = entry["playerId"]
            bbox = entry.get("bbox")
            frame_ms = entry.get("frameMs")

            if bbox and frame_ms is not None and fw > 0:
                # Seek video and extract features at full resolution
                cap.set(cv2.CAP_PROP_POS_MSEC, frame_ms)
                ret, frame = cap.read()
                if ret and frame is not None:
                    frame_arr: Any = np.asarray(frame)
                    bx, by, bw, bh = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
                    features = extract_appearance_features(
                        frame=frame_arr,
                        track_id=0,
                        frame_number=0,
                        bbox=(bx, by, bw, bh),
                        frame_width=fw,
                        frame_height=fh,
                    )
                    crops_by_player.setdefault(pid, []).append(features)
                    # Extract BGR crop for ReID
                    crop = extract_bbox_crop(
                        frame_arr, (bx, by, bw, bh), fw, fh,
                    )
                    if crop is not None:
                        bgr_crops_by_player_json.setdefault(pid, []).append(crop)
                    continue

            # Fallback: read the JPEG crop file
            crop_path = entry.get("cropPath", "")
            img = cv2.imread(crop_path) if crop_path else None
            if img is not None:
                crops_by_player.setdefault(pid, []).append(img)
                bgr_crops_by_player_json.setdefault(pid, []).append(img)
            else:
                console.print(f"[yellow]Warning:[/yellow] Could not extract crop for P{pid}")

        cap.release()

        # Extract DINOv2 embeddings from reference crops.
        # Falls back to HSV-only if DINOv2 fails.
        reid_emb_json: dict[int, Any] = {}
        if bgr_crops_by_player_json:
            try:
                from rallycut.tracking.reid_embeddings import extract_backbone_features

                for pid_j, crops_j in bgr_crops_by_player_json.items():
                    if not crops_j:
                        continue
                    embeddings = extract_backbone_features(crops_j)
                    mean_emb = embeddings.mean(axis=0)
                    norm = np.linalg.norm(mean_emb)
                    if norm > 0:
                        mean_emb /= norm
                    reid_emb_json[pid_j] = mean_emb
            except Exception:
                import logging

                logging.getLogger(__name__).warning(
                    "DINOv2 embedding extraction failed, falling back to HSV-only",
                    exc_info=True,
                )
                reid_emb_json = {}

        reference_profiles = build_profiles_from_crops(
            crops_by_player,
            reid_embeddings_by_player=reid_emb_json or None,
        )
        if not quiet:
            for pid in sorted(reference_profiles):
                n = len(crops_by_player.get(pid, []))
                reid_str = " + ReID" if pid in reid_emb_json else ""
                console.print(f"  Reference profile P{pid}: {n} crop(s){reid_str}")
            console.print()

    # Load general ReID model if no reference profiles (priority cascade)
    general_reid_model = None
    if reference_profiles is None:
        from rallycut.tracking.reid_general import WEIGHTS_PATH as REID_WEIGHTS_PATH

        if REID_WEIGHTS_PATH.exists():
            from rallycut.tracking.reid_general import GeneralReIDModel

            general_reid_model = GeneralReIDModel(weights_path=REID_WEIGHTS_PATH)
            if not quiet:
                console.print("  Using general ReID model")
                console.print()

    # Load existing profiles as frozen anchors for single-rally retrack.
    # Uses established profiles from previous full match-players run instead
    # of rebuilding from scratch — prevents ambiguous first-rally assignments
    # that can cause player teleports.
    if use_existing_profiles and reference_profiles is None:
        from rallycut.tracking.player_features import PlayerAppearanceProfile

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT match_analysis_json FROM videos WHERE id = %s",
                    [video_id],
                )
                row = cur.fetchone()

        if row and row[0] and isinstance(row[0], dict):
            stored_profiles = row[0].get("playerProfiles", {})
            if stored_profiles:
                reference_profiles = {}
                for pid_str, profile_data in stored_profiles.items():
                    try:
                        profile = PlayerAppearanceProfile.from_dict(profile_data)
                        reference_profiles[int(pid_str)] = profile
                    except (KeyError, ValueError) as e:
                        console.print(
                            f"[yellow]Warning: Failed to load profile "
                            f"for player {pid_str}: {e}[/yellow]"
                        )

                if reference_profiles and not quiet:
                    console.print(
                        f"  Using existing profiles as anchors: "
                        f"players {sorted(reference_profiles.keys())}"
                    )
            elif not quiet:
                console.print(
                    "  [yellow]No existing profiles found — "
                    "falling back to full rebuild[/yellow]"
                )

    # Run matching
    match_result: MatchPlayersResult = match_players_across_rallies(
        video_path=video_path,
        rallies=rallies,
        num_samples=num_samples,
        reference_profiles=reference_profiles,
        reid_model=general_reid_model,
        calibrator=court_calibrator,
    )
    results = match_result.rally_results

    # Print summary table
    if not quiet:
        table = Table(title="Player Matching Results", show_header=True, header_style="bold")
        table.add_column("Rally", style="dim")
        table.add_column("Confidence")
        table.add_column("Side Switch")
        table.add_column("Assignments")
        table.add_column("Server")

        for rally, result in zip(rallies, results):
            # Format assignments
            assign_str = ", ".join(
                f"T{tid}→P{pid}" for tid, pid in sorted(result.track_to_player.items())
            )

            # Confidence color
            conf = result.assignment_confidence
            if conf >= 0.7:
                conf_str = f"[green]{conf:.2f}[/green]"
            elif conf >= 0.5:
                conf_str = f"[yellow]{conf:.2f}[/yellow]"
            else:
                conf_str = f"[red]{conf:.2f}[/red]"

            switch_str = "[yellow]YES[/yellow]" if result.side_switch_detected else "no"
            server_str = f"P{result.server_player_id}" if result.server_player_id else "-"

            table.add_row(
                rally.rally_id[:8],
                conf_str,
                switch_str,
                assign_str,
                server_str,
            )

        console.print(table)

        # Summary stats
        n_switches = sum(1 for r in results if r.side_switch_detected)
        avg_conf = sum(r.assignment_confidence for r in results) / len(results) if results else 0
        console.print(f"\n  Rallies: {len(results)}")
        console.print(f"  Avg confidence: {avg_conf:.2f}")
        console.print(f"  Side switches: {n_switches}")

    # Serialize player profiles
    player_profiles_data = {
        str(pid): profile.to_dict()
        for pid, profile in match_result.player_profiles.items()
        if profile.rally_count > 0
    }

    # Build per-rally entries. Don't carry forward old remap state —
    # match-players already reversed any previous remap, so positions
    # are back to original tracker IDs. remap-track-ids will apply fresh.
    rally_entries = []
    for rally, result in zip(rallies, results):
        rally_entry = {
            "rallyId": rally.rally_id,
            "rallyIndex": result.rally_index,
            "startMs": rally.start_ms,
            "endMs": rally.end_ms,
            "trackToPlayer": {
                str(k): v for k, v in result.track_to_player.items()
            },
            "assignmentConfidence": result.assignment_confidence,
            "sideSwitchDetected": result.side_switch_detected,
            "serverPlayerId": result.server_player_id,
        }
        rally_entries.append(rally_entry)

    # Serialize team templates
    team_templates_data = None
    if match_result.team_templates is not None:
        t0, t1 = match_result.team_templates
        team_templates_data = {
            "0": t0.to_dict(),
            "1": t1.to_dict(),
        }

    # Build result JSON (same format used by batch_match_players and API)
    result_json: dict[str, Any] = {
        "videoId": video_id,
        "numRallies": len(results),
        "rallies": rally_entries,
        "playerProfiles": player_profiles_data,
    }
    if team_templates_data is not None:
        result_json["teamTemplates"] = team_templates_data
    # Phase 0: persist the per-rally scratchpad so the relabel-with-crops
    # worker can replay Pass 2 stages 1+2 with new frozen profiles without
    # re-extracting appearance from video.
    if match_result.scratchpad:
        result_json["rallyScratchpad"] = match_result.scratchpad

    # Canonical pid map (ref-crop-sourced single source of truth).
    # Activates when the DB has ≥1 reference crop for all 4 pids. The
    # IdentityAnchors path mean-pools however many crops each pid has, so
    # extra crops only improve prototype quality — there's no point in
    # restricting to exactly 4 total. Partial sets (≤3 distinct pids) fall
    # through to the legacy Hungarian path. Bit-deterministic across re-runs
    # (see tests/integration/test_canonical_pid_determinism.py); consumers
    # (production_eval, future web editor read path) prefer it over
    # matchAnalysisJson.trackToPlayer.
    canonical_payload: dict[str, Any] | None = None
    distinct_pids = {int(r[0]) for r in canonical_crop_rows}
    if distinct_pids == {1, 2, 3, 4} and canonical_bgr_crops:
        from rallycut.tracking.crop_guided_identity import build_anchors_from_crops
        from rallycut.tracking.match_tracker import compute_canonical_pid_map

        anchors = build_anchors_from_crops(canonical_bgr_crops, source="user")
        if anchors.prototypes and len(anchors.prototypes) == 4:
            canonical_map = compute_canonical_pid_map(
                video_path=video_path,
                rallies=rallies,
                anchors=anchors,
            )
            if canonical_map:
                # Canonical pids = user's ref-crop labels, full stop. We
                # explicitly do NOT permute to match legacy Hungarian's
                # output. The plan-doc's "ref crops are the identity source"
                # rule means a user upload of "pid 1 = Carlos" is the
                # contract — every consumer (display, GT, stats) reads
                # this. Earlier code aligned canonical to legacy to avoid
                # a -1.21pp score_accuracy regression in `team_templates`,
                # but that flipped GT badges across re-runs and broke the
                # workstream's user-visible promise. The team_templates +
                # score-GT recalibration is a separate follow-up.
                canonical_payload = _canonical_pid_map_payload(
                    canonical_map, canonical_crop_rows,
                )
                result_json["canonicalPidMap"] = canonical_payload
                if not quiet:
                    console.print(
                        f"  Canonical pid map: {len(canonical_map)} rallies, "
                        f"sha={canonical_payload['sourceRefCropsSha'][:12]}..."
                    )

    # Both columns are written in a single UPDATE so a re-run after
    # crop deletion atomically clears canonical_pid_map_json alongside the
    # fresh match_analysis_json — no window where the legacy and canonical
    # maps disagree. Phase 5 adds the API hook for crop-edit invalidation;
    # this CLI write is the second arm of that contract.
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE videos "
                "SET match_analysis_json = %s, canonical_pid_map_json = %s "
                "WHERE id = %s",
                [
                    json.dumps(result_json),
                    json.dumps(canonical_payload) if canonical_payload else None,
                    video_id,
                ],
            )
        conn.commit()

    if not quiet:
        console.print("  Saved to DB")

    # Write JSON file output
    if output:
        output_data = {
            "video_id": video_id,
            "num_rallies": len(results),
            "playerProfiles": player_profiles_data,
            "rallies": [
                {
                    "rally_id": rally.rally_id,
                    "rally_index": result.rally_index,
                    "start_ms": rally.start_ms,
                    "end_ms": rally.end_ms,
                    "track_to_player": {
                        str(k): v for k, v in result.track_to_player.items()
                    },
                    "assignment_confidence": result.assignment_confidence,
                    "side_switch_detected": result.side_switch_detected,
                    "server_player_id": result.server_player_id,
                }
                for rally, result in zip(rallies, results)
            ],
        }
        with open(output, "w") as f:
            json.dump(output_data, f, indent=2)

        if not quiet:
            console.print(f"\n[green]Results saved to {output}[/green]")
