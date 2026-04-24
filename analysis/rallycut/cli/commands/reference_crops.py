"""CLI commands for managing user-provided reference crops.

Exposes two commands the API invokes via subprocess:

``rallycut validate-reference-crops <video-id>`` — loads crops from the
DB, builds DINOv2 prototypes, runs the quality validator, prints a
``ValidationResult`` as JSON. Used as the UX pre-flight gate before
"Re-run Matching".

``rallycut suggest-reference-crops <video-id>`` — samples candidate crops
from tracked rallies using a diversity-aware heuristic and prints
candidate metadata as JSON. The API persists these to S3 + DB for the
dialog to render.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from rallycut.cli.utils import handle_errors

logger = logging.getLogger(__name__)
console = Console()


# ---------------------------------------------------------------------------
# validate-reference-crops
# ---------------------------------------------------------------------------


def _load_crops_from_db(
    video_id: str,
    video_path: Path,
) -> dict[int, list[Any]]:
    """Load DB-stored reference crops, extract BGR crops from the video.

    Mirrors the loader used by ``match-players`` but returns the raw crops
    (not profile objects) so we can feed them directly to the DINOv2
    backbone.
    """
    import cv2
    import numpy as np

    from rallycut.evaluation.db import get_connection
    from rallycut.tracking.player_features import extract_bbox_crop

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT player_id, frame_ms, bbox_x, bbox_y, bbox_w, bbox_h
                   FROM player_reference_crops
                   WHERE video_id = %s
                   ORDER BY player_id, created_at""",
                [video_id],
            )
            rows = cur.fetchall()

    if not rows:
        return {}

    crops_by_player: dict[int, list[Any]] = {}
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.warning("validate-reference-crops: cannot open video %s", video_path)
        return {}
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Sort by frame_ms for forward-only seeking.
    sorted_rows = sorted(rows, key=lambda r: int(r[1]))  # type: ignore[call-overload]
    for row in sorted_rows:
        pid = int(row[0])  # type: ignore[call-overload]
        frame_ms = float(row[1])  # type: ignore[arg-type]
        bx = float(row[2])  # type: ignore[arg-type]
        by = float(row[3])  # type: ignore[arg-type]
        bw = float(row[4])  # type: ignore[arg-type]
        bh = float(row[5])  # type: ignore[arg-type]
        cap.set(cv2.CAP_PROP_POS_MSEC, frame_ms)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frame_arr = np.asarray(frame, dtype=np.uint8)
        crop = extract_bbox_crop(
            frame_arr, (bx, by, bw, bh), fw, fh,
        )
        if crop is not None:
            crops_by_player.setdefault(pid, []).append(crop)
    cap.release()
    return crops_by_player


@handle_errors
def validate_reference_crops(
    video_id: str = typer.Argument(..., help="Video ID to validate reference crops for"),
    expected_players: int = typer.Option(
        4, "--expected-players",
        help="Number of player slots the validator should enforce.",
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o",
        help="If provided, write the JSON result to this path instead of stdout.",
    ),
) -> None:
    """Validate user-selected reference crops and emit a JSON result.

    Exit code 0: validation completed (ok may be true or false — inspect
    the JSON).
    Exit code 1: unrecoverable error (video missing, crops unreadable, etc).
    """
    from rallycut.evaluation.tracking.db import get_video_path
    from rallycut.tracking.crop_guided_identity import (
        build_anchors_from_crops,
        validate_prototypes,
    )

    video_path = get_video_path(video_id)
    if video_path is None:
        sys.stderr.write(
            "Error: could not resolve video path for " + video_id + "\n",
        )
        raise typer.Exit(1)

    crops_by_player = _load_crops_from_db(video_id, video_path)
    expected_ids = list(range(1, expected_players + 1))

    anchors = build_anchors_from_crops(crops_by_player, source="validate-cli")
    result = validate_prototypes(anchors, expected_player_ids=expected_ids)

    # Phase 1.2: intra-player anomaly check. For any pid with ≥3 crops,
    # surface indices that look visually distant from the rest of the
    # cohort. The API can render these as "this crop looks different —
    # confirm or replace?" warnings.
    import numpy as np

    from rallycut.tracking.relabel import detect_anomalous_crops

    embeddings_per_pid: dict[int, np.ndarray] = {}
    for pid, bgr_crops in crops_by_player.items():
        if len(bgr_crops) < 3:
            continue
        try:
            from rallycut.tracking.reid_embeddings import extract_backbone_features

            embs = extract_backbone_features(bgr_crops)
            if embs.size > 0:
                embeddings_per_pid[int(pid)] = embs
        except Exception:
            logger.warning(
                "anomaly-check: DINOv2 extraction failed for pid %s", pid,
                exc_info=True,
            )
    anomalies = detect_anomalous_crops(embeddings_per_pid)

    payload = {
        "videoId": video_id,
        **result.to_dict(),
        "cropCounts": {
            str(pid): len(crops_by_player.get(pid, []))
            for pid in expected_ids
        },
        "anomalousCrops": {
            str(pid): flagged
            for pid, flagged in anomalies.items()
            if flagged  # only include pids with at least one flag
        },
    }
    serialized = json.dumps(payload, indent=2)

    if output is not None:
        output.write_text(serialized)
    else:
        # Use stdout directly so the API can capture without rich markup.
        sys.stdout.write(serialized + "\n")


# ---------------------------------------------------------------------------
# suggest-reference-crops
# ---------------------------------------------------------------------------


def _load_rally_positions(
    video_id: str,
) -> list[dict[str, Any]]:
    """Return a list of ``{rally_id, start_ms, end_ms, positions, primary}``
    dicts for all tracked rallies belonging to the video.
    """
    from rallycut.evaluation.tracking.db import load_rallies_for_video

    rallies = load_rallies_for_video(video_id)
    out: list[dict[str, Any]] = []
    for r in rallies:
        out.append(
            {
                "rally_id": r.rally_id,
                "start_ms": r.start_ms,
                "end_ms": r.end_ms,
                "positions": r.positions,
                "primary": list(r.primary_track_ids),
            },
        )
    return out


def _sample_candidates_for_player(
    player_slot: int,
    player_slot_to_tracks: dict[int, list[tuple[str, int, int]]],
    rallies_by_id: dict[str, Any],
    video_path: Path,
    num_candidates: int,
    min_det_conf: float,
    *,
    device: str | None = None,
) -> list[dict[str, Any]]:
    """Return ranked candidate crops for one player slot.

    Takes a pre-grouped ``{player_slot: [(rally_id, rally_start_frame, track_id), ...]}``
    and ``rallies_by_id`` (already loaded). Samples positions at evenly-
    spaced frames across the match. Runs a quality gate (detection
    confidence) and farthest-point sampling on DINOv2 features.
    """
    import cv2
    import numpy as np

    from rallycut.tracking.player_features import extract_bbox_crop
    from rallycut.tracking.reid_embeddings import extract_backbone_features

    entries = player_slot_to_tracks.get(player_slot, [])
    if not entries:
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)

    # Keyed by (rally_id, frame_ms) so we don't re-sample the same frame
    # if multiple positions exist close together.
    candidates: list[dict[str, Any]] = []
    crops: list[Any] = []
    for rally_id, rally_start_frame, track_id in entries:
        rally = rallies_by_id.get(rally_id)
        if rally is None:
            continue

        # Pick up to N evenly-spaced positions for this track in this rally.
        track_positions = [
            p for p in rally.positions if p.track_id == track_id and
            p.confidence >= min_det_conf
        ]
        if not track_positions:
            continue
        track_positions.sort(key=lambda p: p.frame_number)
        stride = max(1, len(track_positions) // 4)
        samples = track_positions[::stride][:4]

        for p in samples:
            absolute_frame = rally_start_frame + p.frame_number
            frame_ms = (absolute_frame / fps) * 1000.0
            cap.set(cv2.CAP_PROP_POS_MSEC, frame_ms)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            frame_arr = np.asarray(frame, dtype=np.uint8)
            crop = extract_bbox_crop(
                frame_arr,
                (p.x, p.y, p.width, p.height),
                fw, fh,
            )
            if crop is None:
                continue
            crops.append(crop)
            candidates.append(
                {
                    "rallyId": rally_id,
                    "trackId": track_id,
                    "frameMs": int(frame_ms),
                    "bbox": {
                        "x": float(p.x), "y": float(p.y),
                        "w": float(p.width), "h": float(p.height),
                    },
                    "detectionConfidence": float(p.confidence),
                },
            )

    cap.release()
    if not crops:
        return []

    # Phase 1.3: pre-rank by clarity (detection × edge-distance × bbox-area).
    # FPS afterwards picks the most diverse subset from the cleanest pool;
    # this prevents a tiny edge-clipped crop from "winning" diversity by
    # being visually unlike the rest.
    from rallycut.tracking.relabel import compute_clarity_score

    clarity_scores: list[float] = []
    for cand in candidates:
        bbox = cand["bbox"]
        clarity_scores.append(
            compute_clarity_score(
                detection_confidence=cand["detectionConfidence"],
                x=bbox["x"], y=bbox["y"], w=bbox["w"], h=bbox["h"],
            )
        )
    # Drop candidates with clarity == 0 (clipped or tiny — unusable).
    keep_idx = [i for i, s in enumerate(clarity_scores) if s > 0.0]
    if not keep_idx:
        return []
    candidates = [candidates[i] for i in keep_idx]
    crops = [crops[i] for i in keep_idx]
    clarity_scores = [clarity_scores[i] for i in keep_idx]
    # Surface clarity in the candidate payload so the API/UI can show it.
    for cand, score in zip(candidates, clarity_scores):
        cand["clarityScore"] = round(float(score), 3)
    # Cap the pool at 3× requested before FPS — keeps diversification
    # focused on the cleaner half of the available frames.
    if len(candidates) > 3 * num_candidates:
        order = sorted(
            range(len(candidates)),
            key=lambda i: clarity_scores[i],
            reverse=True,
        )[: 3 * num_candidates]
        candidates = [candidates[i] for i in order]
        crops = [crops[i] for i in order]

    # Farthest-point sampling on DINOv2 features.
    features = extract_backbone_features(crops, device=device)
    if features.shape[0] == 0:
        return []

    n = min(num_candidates, features.shape[0])
    chosen_idx = [0]
    while len(chosen_idx) < n:
        chosen_feats = features[chosen_idx]
        dists = 1.0 - features @ chosen_feats.T
        min_dists = dists.min(axis=1)
        # Exclude already chosen.
        for i in chosen_idx:
            min_dists[i] = -1.0
        next_idx = int(np.argmax(min_dists))
        if min_dists[next_idx] < 0:
            break
        chosen_idx.append(next_idx)

    return [candidates[i] for i in chosen_idx]


@handle_errors
def suggest_reference_crops(
    video_id: str = typer.Argument(..., help="Video ID to suggest crops for"),
    num_candidates: int = typer.Option(
        6, "--num-candidates",
        help="Max candidates returned per player slot.",
    ),
    min_det_conf: float = typer.Option(
        0.70, "--min-confidence",
        help="Minimum detection confidence for a crop to be eligible.",
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o",
        help="If provided, write JSON to this path instead of stdout.",
    ),
) -> None:
    """Emit ranked candidate reference crops per player for the given video.

    Output JSON format:
        {
            "videoId": "...",
            "candidates": {
                "1": [{rallyId, trackId, frameMs, bbox, detectionConfidence}, ...],
                "2": [...],
                ...
            }
        }

    Requires that ``match-players`` has already run (so every rally has a
    ``trackToPlayer`` mapping). If no mapping is available, the command
    returns an empty candidate set — the UI falls back to client-side
    auto-sampling.
    """
    from rallycut.evaluation.db import get_connection
    from rallycut.evaluation.tracking.db import get_video_path, load_rallies_for_video

    video_path = get_video_path(video_id)
    if video_path is None:
        sys.stderr.write(
            "Error: could not resolve video path for " + video_id + "\n",
        )
        raise typer.Exit(1)

    rallies = load_rallies_for_video(video_id)
    if not rallies:
        sys.stdout.write(
            json.dumps({"videoId": video_id, "candidates": {}}) + "\n",
        )
        return

    # Load match-analysis to map track_id → player_id per rally.
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT match_analysis_json FROM videos WHERE id = %s",
                [video_id],
            )
            row = cur.fetchone()

    match_analysis = row[0] if row and row[0] else None
    if not isinstance(match_analysis, dict):
        sys.stdout.write(
            json.dumps({"videoId": video_id, "candidates": {}}) + "\n",
        )
        return

    # rally_id → {track_id: player_id}
    track_to_player_by_rally: dict[str, dict[int, int]] = {}
    for entry in match_analysis.get("rallies", []):
        rid = entry.get("rallyId") or entry.get("rally_id")
        ttp = entry.get("trackToPlayer") or entry.get("track_to_player")
        if not rid or not ttp:
            continue
        track_to_player_by_rally[rid] = {int(k): int(v) for k, v in ttp.items()}

    # Load rally metadata for frame offset lookup. Resolve FPS once per
    # video to avoid reopening the capture per rally.
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    cap.release()

    rally_meta: dict[str, tuple[int, int]] = {}
    rallies_by_id: dict[str, Any] = {}
    for r in rallies:
        rally_start_frame = int((r.start_ms / 1000.0) * fps)
        rally_meta[r.rally_id] = (rally_start_frame, int(r.end_ms))
        rallies_by_id[r.rally_id] = r

    # Group tracks by player slot (1-4) so we can sample diverse crops
    # across rallies for each player.
    player_slot_to_tracks: dict[int, list[tuple[str, int, int]]] = {}
    for rally_id, ttp in track_to_player_by_rally.items():
        if rally_id not in rally_meta:
            continue
        rally_start_frame = rally_meta[rally_id][0]
        for tid, pid in ttp.items():
            player_slot_to_tracks.setdefault(pid, []).append(
                (rally_id, rally_start_frame, tid),
            )

    candidates_by_player: dict[str, list[dict[str, Any]]] = {}
    for slot in sorted(player_slot_to_tracks.keys()):
        cands = _sample_candidates_for_player(
            player_slot=slot,
            player_slot_to_tracks=player_slot_to_tracks,
            rallies_by_id=rallies_by_id,
            video_path=video_path,
            num_candidates=num_candidates,
            min_det_conf=min_det_conf,
        )
        candidates_by_player[str(slot)] = cands

    payload = {
        "videoId": video_id,
        "candidates": candidates_by_player,
    }
    serialized = json.dumps(payload, indent=2)
    if output is not None:
        output.write_text(serialized)
    else:
        sys.stdout.write(serialized + "\n")
