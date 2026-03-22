"""Re-attribute player actions using match-level team assignments.

Runs after match-players to improve player attribution on stored actions
by leveraging cross-rally team identity (which isn't available during
initial per-rally tracking).
"""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING, Any, cast

import typer
from rich.console import Console

from rallycut.cli.utils import handle_errors
from rallycut.tracking.match_tracker import build_match_team_assignments

if TYPE_CHECKING:
    from rallycut.tracking.action_classifier import ClassifiedAction
    from rallycut.tracking.contact_detector import Contact

console = Console()
logger = logging.getLogger(__name__)


def _reconstruct_contacts(
    contacts_data: dict[str, Any],
) -> list[Contact]:
    """Reconstruct Contact objects from stored contacts_json."""
    from rallycut.tracking.contact_detector import Contact as ContactCls

    contacts: list[Contact] = []
    for c in contacts_data.get("contacts", []):
        candidates = c.get("playerCandidates", [])
        contacts.append(ContactCls(
            frame=c.get("frame", 0),
            ball_x=c.get("ballX", 0.0),
            ball_y=c.get("ballY", 0.0),
            velocity=c.get("velocity", 0.0),
            direction_change_deg=c.get("directionChangeDeg", 0.0),
            player_track_id=c.get("playerTrackId", -1),
            player_distance=c["playerDistance"] if c.get("playerDistance") is not None else float("inf"),
            player_candidates=[
                (int(p[0]), float(p[1])) for p in candidates if p[1] is not None
            ],
            court_side=c.get("courtSide", "unknown"),
            is_at_net=c.get("isAtNet", False),
            is_validated=c.get("isValidated", False),
            confidence=c.get("confidence", 0.0),
            arc_fit_residual=c.get("arcFitResidual", 0.0),
        ))
    return contacts


def _reconstruct_actions(
    actions_data: dict[str, Any],
) -> list[ClassifiedAction]:
    """Reconstruct ClassifiedAction objects from stored actions_json."""
    from rallycut.tracking.action_classifier import ActionType
    from rallycut.tracking.action_classifier import ClassifiedAction as ActionCls

    actions: list[ClassifiedAction] = []
    for a in actions_data.get("actions", []):
        try:
            action_type = ActionType(a["action"])
        except (ValueError, KeyError):
            action_type = ActionType.UNKNOWN

        actions.append(ActionCls(
            action_type=action_type,
            frame=a.get("frame", 0),
            ball_x=a.get("ballX", 0.0),
            ball_y=a.get("ballY", 0.0),
            velocity=a.get("velocity", 0.0),
            player_track_id=a.get("playerTrackId", -1),
            court_side=a.get("courtSide", "unknown"),
            confidence=a.get("confidence", 0.0),
            is_synthetic=a.get("isSynthetic", False),
            team=a.get("team", "unknown"),
        ))
    return actions


def _serialize_actions(
    actions: list[ClassifiedAction],
    original_actions_data: dict[str, Any],
) -> dict[str, Any]:
    """Serialize actions back, preserving other fields from original data."""
    result = dict(original_actions_data)
    result["actions"] = [a.to_dict() for a in actions]
    return result


def _train_reid_classifier(
    video_id: str,
    quiet: bool = False,
) -> Any:
    """Train a ReID classifier on reference crops if available.

    Returns the trained PlayerReIDClassifier, or None if no crops.
    """
    from rallycut.evaluation.tracking.db import get_connection, get_video_path

    # Check for reference crops
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
        return None

    crop_infos = [
        {
            "player_id": r[0], "frame_ms": r[1],
            "bbox_x": r[2], "bbox_y": r[3],
            "bbox_w": r[4], "bbox_h": r[5],
        }
        for r in crop_rows
    ]

    video_path = get_video_path(video_id)
    if video_path is None:
        return None

    from rallycut.tracking.reid_embeddings import (
        PlayerReIDClassifier,
        extract_crops_from_video,
    )

    try:
        crops_by_player = extract_crops_from_video(video_path, crop_infos)
        if len(crops_by_player) < 2:
            return None

        classifier = PlayerReIDClassifier()
        stats = classifier.train(crops_by_player)

        if not quiet:
            n_crops = sum(len(c) for c in crops_by_player.values())
            console.print(
                f"  ReID classifier: {n_crops} crops → "
                f"{len(crops_by_player)} players, train_acc={stats['train_acc']:.0%}"
            )

        return classifier
    except Exception:
        logger.warning(
            "ReID classifier training failed, skipping ReID re-attribution",
            exc_info=True,
        )
        return None


def _compute_reid_predictions(
    classifier: Any,
    video_cap: Any,
    contacts: list[Contact],
    positions_json: list[dict],
    track_to_player: dict[int, int],
    start_ms: int,
    video_fps: float,
) -> dict[int, dict[str, Any]]:
    """Compute ReID predictions for each contact frame.

    Returns:
        {contact_frame: {"best_tid": int, "margin": float}}
    """
    import cv2
    import numpy as np

    pos_by_frame_track: dict[tuple[int, int], dict] = {}
    for pp in positions_json:
        pos_by_frame_track[(pp["frameNumber"], pp["trackId"])] = pp

    rally_start_frame = int(start_ms / 1000.0 * video_fps)
    predictions: dict[int, dict[str, Any]] = {}

    for contact in contacts:
        if not contact.player_candidates or len(contact.player_candidates) < 2:
            continue

        # Extract candidate crops
        candidate_info: list[tuple[int, int, np.ndarray]] = []  # (tid, pid, crop)

        for cand_tid, _dist in contact.player_candidates:
            cand_pid = track_to_player.get(cand_tid, -1)
            if cand_pid < 0:
                continue

            # Find position near contact frame
            best_pos = None
            for delta in range(6):
                for fn in [contact.frame + delta, contact.frame - delta]:
                    if fn < 0:
                        continue
                    pos = pos_by_frame_track.get((fn, cand_tid))
                    if pos is not None:
                        best_pos = pos
                        break
                if best_pos is not None:
                    break

            if best_pos is None:
                continue

            abs_fn = rally_start_frame + best_pos["frameNumber"]
            video_cap.set(cv2.CAP_PROP_POS_FRAMES, abs_fn)
            ret, frame = video_cap.read()
            if not ret or frame is None:
                continue

            h, w = frame.shape[:2]
            bx, by, bw, bh = best_pos["x"], best_pos["y"], best_pos["width"], best_pos["height"]
            x1 = max(0, int((bx - bw / 2) * w))
            y1 = max(0, int((by - bh / 2) * h))
            x2 = min(w, int((bx + bw / 2) * w))
            y2 = min(h, int((by + bh / 2) * h))

            if x2 <= x1 or y2 <= y1:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0 or crop.shape[0] < 16 or crop.shape[1] < 8:
                continue

            candidate_info.append((cand_tid, cand_pid, crop))

        if len(candidate_info) < 2:
            continue

        # Classify all candidates
        crops = [c[2] for c in candidate_info]
        probs_list = classifier.predict(crops)

        # For each candidate, find the player_id with highest probability
        # Then pick the best overall assignment
        best_tid = -1
        best_prob = -1.0
        second_prob = -1.0

        for idx, (cand_tid, cand_pid, _crop) in enumerate(candidate_info):
            # Get the probability that this candidate IS player cand_pid
            prob = probs_list[idx].get(cand_pid, 0.0)
            if prob > best_prob:
                second_prob = best_prob
                best_prob = prob
                best_tid = cand_tid
            elif prob > second_prob:
                second_prob = prob

        margin = best_prob - second_prob if second_prob >= 0 else 0.0

        predictions[contact.frame] = {
            "best_tid": best_tid,
            "margin": margin,
        }

    return predictions


@handle_errors
def reattribute_actions_cmd(
    video_id: str = typer.Argument(
        ...,
        help="Video ID to re-attribute actions for",
    ),
    min_confidence: float = typer.Option(
        0.80,
        "--min-confidence",
        help="Minimum match confidence to use team assignments",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Show what would change without updating DB",
    ),
    quiet: bool = typer.Option(
        False,
        "--quiet", "-q",
        help="Suppress progress output",
    ),
    reid: bool = typer.Option(
        False,
        "--reid",
        help="Enable ReID re-attribution (Pass 3) using reference crops",
    ),
) -> None:
    """Re-attribute player actions using match-level team assignments.

    After match-players has run, this command uses the cross-rally team
    identity to fix player attributions where the nearest player is on
    the wrong team for the court side.

    Example:
        rallycut reattribute-actions abc123
        rallycut reattribute-actions abc123 --dry-run
    """
    from rallycut.evaluation.tracking.db import get_connection, get_video_path
    from rallycut.tracking.action_classifier import (
        _team_label,
        assign_court_side_from_teams,
        reattribute_players,
    )

    # Load match analysis
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT match_analysis_json FROM videos WHERE id = %s",
                [video_id],
            )
            row = cur.fetchone()

    if not row or not row[0]:
        console.print(
            "[red]Error:[/red] No match analysis found. "
            "Run 'rallycut match-players' first."
        )
        raise typer.Exit(1)

    match_analysis = cast(dict[str, Any], row[0])

    # Build team assignments at two confidence levels:
    # - all_teams (conf >= 0): used for stamping teamAssignments on all rallies
    # - reattrib_teams (conf >= min_confidence): used for player re-attribution
    all_teams = build_match_team_assignments(match_analysis, min_confidence=0.0)
    reattrib_teams = build_match_team_assignments(match_analysis, min_confidence)

    if not all_teams:
        if not quiet:
            console.print("[yellow]No rallies with match teams[/yellow]")
        return

    if not quiet:
        console.print(
            f"[bold]Re-attributing actions[/bold] for video {video_id[:8]}..."
        )
        console.print(
            f"  {len(all_teams)} rallies with match teams, "
            f"{len(reattrib_teams)} eligible for re-attribution "
            f"(conf >= {min_confidence:.2f})"
        )

    # Train ReID classifier only if explicitly requested (off by default —
    # ReID Pass 3 is currently net negative on player attribution accuracy).
    reid_classifier = _train_reid_classifier(video_id, quiet) if reid else None

    # Load contacts + actions for all rallies with match teams
    rally_ids = list(all_teams.keys())
    placeholders = ", ".join(["%s"] * len(rally_ids))
    query = f"""
        SELECT r.id, pt.id, pt.contacts_json, pt.actions_json,
               pt.positions_json, r.start_ms
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE r.id IN ({placeholders})
          AND pt.contacts_json IS NOT NULL
          AND pt.actions_json IS NOT NULL
    """

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(query, rally_ids)
            rows = cur.fetchall()

    # Build track_to_player mapping from match analysis for ReID
    track_to_player_by_rally: dict[str, dict[int, int]] = {}
    for rally_entry in match_analysis.get("rallies", []):
        rid = rally_entry.get("rallyId") or rally_entry.get("rally_id", "")
        t2p = rally_entry.get("trackToPlayer") or rally_entry.get("track_to_player", {})
        if rid and t2p:
            track_to_player_by_rally[rid] = {int(k): int(v) for k, v in t2p.items()}

    total_reattributed = 0
    total_actions = 0
    updated_tracks: list[tuple[int, str]] = []  # (pt_id, new_actions_json)

    # Open video for ReID crop extraction (if classifier is available)
    video_cap = None
    video_fps = 30.0
    if reid_classifier is not None:
        video_path = get_video_path(video_id)
        if video_path is not None:
            import cv2
            video_cap = cv2.VideoCapture(str(video_path))
            if video_cap.isOpened():
                video_fps = video_cap.get(cv2.CAP_PROP_FPS) or 30.0
            else:
                video_cap = None

    for rally_id_val, pt_id_val, contacts_json_val, actions_json_val, positions_json_val, start_ms_val in rows:
        rally_id = str(rally_id_val)
        pt_id = cast(int, pt_id_val)
        contacts_data = cast(dict[str, Any], contacts_json_val)
        actions_data = cast(dict[str, Any], actions_json_val)

        team_assignments = all_teams.get(rally_id)
        if not team_assignments:
            continue

        contacts = _reconstruct_contacts(contacts_data)
        actions = _reconstruct_actions(actions_data)

        # Build ReID predictions for this rally if classifier available
        reid_predictions: dict[int, dict[str, Any]] | None = None
        if reid_classifier is not None and video_cap is not None and positions_json_val:
            t2p = track_to_player_by_rally.get(rally_id, {})
            if t2p:
                reid_predictions = _compute_reid_predictions(
                    reid_classifier, video_cap, contacts,
                    positions_json_val, t2p,  # type: ignore[arg-type]
                    start_ms=start_ms_val or 0, video_fps=video_fps,  # type: ignore[arg-type]
                )

        # Overwrite court_side from team assignments (stored actions have
        # propagation-based court_side which is less accurate).
        reattrib_ta = reattrib_teams.get(rally_id)
        if reattrib_ta and actions:
            assign_court_side_from_teams(actions, reattrib_ta)

        # Re-attribute players if high-confidence and contacts have candidates
        has_candidates = contacts and any(c.player_candidates for c in contacts)
        n_changed = 0
        if reattrib_ta and has_candidates and actions:
            original_track_ids = [a.player_track_id for a in actions]
            reattribute_players(
                actions, contacts, reattrib_ta,
                reid_predictions=reid_predictions,
            )
            n_changed = sum(
                1 for orig, act in zip(original_track_ids, actions)
                if orig != act.player_track_id
            )

        total_actions += len(actions)
        total_reattributed += n_changed

        # Always stamp match-level teamAssignments and per-action team labels
        # (the original tracking may not have had court calibration)
        new_actions_data = _serialize_actions(actions, actions_data)
        new_actions_data["teamAssignments"] = {
            str(tid): ("A" if team == 0 else "B")
            for tid, team in team_assignments.items()
        }
        for a in new_actions_data.get("actions", []):
            tid = a.get("playerTrackId", -1)
            if tid >= 0:
                a["team"] = _team_label(tid, team_assignments)
            elif a.get("courtSide") in ("near", "far"):
                a["team"] = "A" if a["courtSide"] == "near" else "B"

        updated_tracks.append((pt_id, json.dumps(new_actions_data)))

        if n_changed > 0:
            if not quiet:
                console.print(
                    f"  {rally_id[:8]}: {n_changed}/{len(actions)} actions re-attributed"
                )
        elif not quiet:
            console.print(f"  [dim]{rally_id[:8]}: no changes (teams stamped)[/dim]")

    if video_cap is not None:
        video_cap.release()

    # Summary
    if not quiet:
        console.print(
            f"\n  Total: {total_reattributed}/{total_actions} actions "
            f"re-attributed across {len(rows)} rallies"
        )

    # Update DB
    if updated_tracks and not dry_run:
        with get_connection() as conn:
            with conn.cursor() as cur:
                for update_pt_id, new_json in updated_tracks:
                    cur.execute(
                        "UPDATE player_tracks SET actions_json = %s WHERE id = %s",
                        [new_json, update_pt_id],
                    )
            conn.commit()

        if not quiet:
            console.print(
                f"  [green]Updated {len(updated_tracks)} player tracks in DB[/green]"
            )
    elif dry_run and updated_tracks:
        if not quiet:
            console.print(
                f"  [yellow]Dry run: would update "
                f"{len(updated_tracks)} player tracks[/yellow]"
            )
