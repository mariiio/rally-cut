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


def _build_formation_semantic_flips(
    match_analysis: dict[str, Any],
    video_id: str | None = None,
) -> dict[str, bool]:
    """Compute per-rally `semantic_flip` from switch data.

    Returns `{rally_id: flipped}` where `flipped=True` when the cumulative
    side-switch count BEFORE this rally is odd. Used by the formation-based
    serving_team predictor to correct the physical-near team convention
    (team 0 = near after `verify_team_assignments`) to the semantic team
    identity on flipped rallies.

    When ``video_id`` is provided, uses the shared switch loader which
    merges pipeline-detected switches with per-rally manual overrides
    from the Score GT UI. Falls back to match_analysis_json-only when
    video_id is not available.
    """
    if video_id:
        try:
            from rallycut.evaluation.switch_loader import resolve_side_flipped
            return resolve_side_flipped({video_id}, gt_only=False)
        except Exception:
            pass  # fall through to match_analysis-only path

    rallies = match_analysis.get("rallies", [])
    if not isinstance(rallies, list):
        return {}
    result: dict[str, bool] = {}
    count = 0
    for rally_entry in rallies:
        rid = rally_entry.get("rallyId") or rally_entry.get("rally_id")
        if rid:
            result[str(rid)] = (count % 2 == 1)
        if rally_entry.get("sideSwitchDetected") or rally_entry.get(
            "side_switch_detected"
        ):
            count += 1
    return result


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
    """Reconstruct ClassifiedAction objects from stored actions_json.

    Handles two on-disk schemas:
      1. Flat: `actions_data["actions"]` is a list of action dicts
         (produced by `RallyActions.to_dict()` + `_serialize_actions`).
      2. Nested: `actions_data["actions"]` is itself a RallyActions dict
         whose `.actions` key holds the list (older tracking writes).
    """
    from rallycut.tracking.action_classifier import ActionType
    from rallycut.tracking.action_classifier import ClassifiedAction as ActionCls

    raw_actions = actions_data.get("actions", [])
    # Unwrap nested schema
    if isinstance(raw_actions, dict):
        raw_actions = raw_actions.get("actions", [])

    actions: list[ClassifiedAction] = []
    for a in raw_actions:
        if not isinstance(a, dict):
            continue
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


def _attribute_synthetic_serves(
    actions: list[ClassifiedAction],
    positions_json: list[dict[str, Any]] | Any,
    team_assignments: dict[int, int],
) -> None:
    """Attribute synthetic serves (playerTrackId=-1) to nearest server.

    Synthetic serves are injected by the action classifier when no contact
    matches the serve.  They carry court_side but no player.  This pass
    finds the nearest primary-track player on the serving side at (or near)
    the serve frame.
    """
    from rallycut.tracking.action_classifier import ActionType

    # Build per-frame position index for primary tracks only
    pos_by_frame: dict[int, list[tuple[int, float, float]]] = {}
    for p in positions_json:
        tid = p.get("trackId")
        if tid is None or team_assignments.get(tid) is None:
            continue  # skip unmapped tracks
        fn = p.get("frameNumber", -1)
        pos_by_frame.setdefault(fn, []).append((tid, p.get("x", 0.5), p.get("y", 0.5)))

    # team 0 = near (high Y), team 1 = far (low Y)
    serve_team = {"near": 0, "far": 1}

    for action in actions:
        if not action.is_synthetic:
            continue
        if action.action_type != ActionType.SERVE:
            continue
        if action.player_track_id >= 0:
            continue  # already attributed

        expected_team = serve_team.get(action.court_side)
        if expected_team is None:
            continue

        # Search serve frame ± 5 frames for player positions
        best_tid = -1
        best_dist = float("inf")
        for delta in range(6):
            for fn in [action.frame + delta, action.frame - delta]:
                for tid, px, py in pos_by_frame.get(fn, []):
                    if team_assignments.get(tid) != expected_team:
                        continue
                    # Distance from serve baseline (Y-axis is most relevant)
                    dist = abs(py - action.ball_y) + abs(px - action.ball_x) * 0.3
                    if dist < best_dist:
                        best_dist = dist
                        best_tid = tid
            if best_tid >= 0:
                break  # found at this delta, no need to search further

        if best_tid >= 0:
            action.player_track_id = best_tid


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
    visual: bool = typer.Option(
        False,
        "--visual",
        help="Enable visual attribution using VideoMAE per-player action classifier",
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

    # Per-rally semantic flip for formation-based serving_team detection.
    # Required to map physical-near (team_assignments team 0) to the
    # correct semantic team on flipped rallies. See
    # `score_tracking_architecture_2026_04.md`.
    formation_flips = _build_formation_semantic_flips(match_analysis, video_id=video_id)

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

    # Load visual attribution classifier if requested
    visual_classifier = None
    if visual:
        from rallycut.tracking.visual_attribution import load_visual_attribution_classifier
        visual_classifier = load_visual_attribution_classifier()
        if visual_classifier is None:
            console.print(
                "[yellow]Warning: No trained visual attribution model found "
                "at weights/visual_attribution/. Skipping visual attribution.[/yellow]"
            )
        elif not quiet:
            console.print("  Visual attribution classifier loaded")

    # Load contacts + actions for all rallies with match teams
    rally_ids = list(all_teams.keys())
    placeholders = ", ".join(["%s"] * len(rally_ids))
    query = f"""
        SELECT r.id, pt.id, pt.contacts_json, pt.actions_json,
               pt.positions_json, r.start_ms, pt.court_split_y
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        WHERE r.id IN ({placeholders})
          AND pt.contacts_json IS NOT NULL
          AND pt.actions_json IS NOT NULL
        ORDER BY r.start_ms
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
    # Formation-based serving_team updates for rallies where the new
    # prediction differs from the current stored value.
    # (rally_id, new_serving_team) pairs for rallies.serving_team column.
    rally_serving_updates: list[tuple[str, str]] = []
    formation_applied = 0

    # Open video for ReID/visual crop extraction (if classifier is available)
    video_cap = None
    video_fps = 30.0
    video_w = 0
    video_h = 0
    if reid_classifier is not None or visual_classifier is not None:
        video_path = get_video_path(video_id)
        if video_path is not None:
            import cv2
            video_cap = cv2.VideoCapture(str(video_path))
            if video_cap.isOpened():
                video_fps = video_cap.get(cv2.CAP_PROP_FPS) or 30.0
                video_w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                video_h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            else:
                video_cap = None

    for (
        rally_id_val, pt_id_val, contacts_json_val, actions_json_val,
        positions_json_val, start_ms_val, court_split_y_val,
    ) in rows:
        rally_id = str(rally_id_val)
        pt_id = cast(int, pt_id_val)
        contacts_data = cast(dict[str, Any], contacts_json_val)
        actions_data = cast(dict[str, Any], actions_json_val)
        court_split_y = cast("float | None", court_split_y_val)

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

        # Fix unmapped tracks (spectator/ref IDs like 101+) even on
        # low-confidence rallies — any primary player is better than a
        # non-player track, regardless of team assignment quality.
        has_candidates = contacts and any(c.player_candidates for c in contacts)
        if has_candidates and actions and not reattrib_ta:
            contact_by_frame = {c.frame: c for c in contacts}
            for action in actions:
                if team_assignments.get(action.player_track_id) is not None:
                    continue  # already mapped to a primary track
                if action.player_track_id < 0:
                    continue
                contact = contact_by_frame.get(action.frame)
                if contact is None or not contact.player_candidates:
                    continue
                best_tid = -1
                best_dist = float("inf")
                for tid, dist in contact.player_candidates:
                    if tid == action.player_track_id:
                        continue
                    if team_assignments.get(tid) is None:
                        continue  # also unmapped
                    if dist < best_dist:
                        best_tid = tid
                        best_dist = dist
                if best_tid >= 0:
                    action.player_track_id = best_tid

        # Re-attribute players if high-confidence and contacts have candidates
        n_changed = 0
        if reattrib_ta and has_candidates and actions:
            original_track_ids = [a.player_track_id for a in actions]
            reattribute_players(
                actions, contacts, reattrib_ta,
                reid_predictions=reid_predictions,
            )
            assign_court_side_from_teams(actions, reattrib_ta)

            # Visual attribution pass (overrides when confident)
            if visual_classifier is not None and video_cap is not None and positions_json_val:
                from rallycut.tracking.visual_attribution import visual_reattribute
                start_ms_int = cast(int, start_ms_val) if start_ms_val else 0
                rally_start_frame = int(start_ms_int / 1000.0 * video_fps)
                visual_reattribute(
                    actions, contacts,
                    cast(list[dict[str, Any]], positions_json_val),
                    video_cap, rally_start_frame, video_w, video_h,
                    visual_classifier, team_assignments=reattrib_ta,
                )

            n_changed = sum(
                1 for orig, act in zip(original_track_ids, actions)
                if orig != act.player_track_id
            )

        # Attribute synthetic serves (P-1) using position data.
        # Find the nearest primary player on the serving court side.
        if positions_json_val and team_assignments:
            _attribute_synthetic_serves(actions, positions_json_val, team_assignments)

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

        # Formation-based serving_team prediction. Overrides the
        # contact-based `servingTeam` when the formation signal is
        # confident. Uses semantic_flip from match_analysis_json to
        # correct for side switches. +15.8pp over contact-based on
        # canonical production_eval (see score_tracking_architecture).
        # Internal function handles a bad `net_y` via auto-split fallback.
        if positions_json_val:
            from rallycut.tracking.action_classifier import (  # noqa: PLC0415
                _find_serving_team_by_formation,
            )
            from rallycut.tracking.match_tracker import (  # noqa: PLC0415
                verify_team_assignments,
            )
            from rallycut.tracking.player_tracker import PlayerPosition  # noqa: PLC0415

            player_positions = [
                PlayerPosition(
                    frame_number=p["frameNumber"],
                    track_id=p["trackId"],
                    x=p["x"],
                    y=p["y"],
                    width=p.get("width", 0.05),
                    height=p.get("height", 0.10),
                    confidence=p.get("confidence", 1.0),
                    keypoints=p.get("keypoints"),
                )
                for p in cast(list[dict[str, Any]], positions_json_val)
            ]
            # Enforce "team 0 = physical near" convention before calling
            # the formation function — build_match_team_assignments leaves
            # the assignment inverted on videos where the cross-rally
            # matcher labeled players 1-2 on the far side at baseline.
            verified_teams = verify_team_assignments(
                team_assignments, player_positions,
            )
            formation_team, _ = _find_serving_team_by_formation(
                player_positions,
                start_frame=0,
                net_y=court_split_y if court_split_y is not None else 0.5,
                team_assignments=verified_teams,
                semantic_flip=formation_flips.get(rally_id, False),
            )
            if formation_team is not None:
                # Stamp `servingTeam` on the top-level JSON AND on any
                # nested `actions` dict (older schema) so both the API
                # reader and downstream stats see the updated value.
                prior_serving = new_actions_data.get("servingTeam")
                new_actions_data["servingTeam"] = formation_team
                nested_actions = new_actions_data.get("actions")
                if isinstance(nested_actions, dict):
                    nested_actions["servingTeam"] = formation_team
                if prior_serving != formation_team:
                    formation_applied += 1
                # Note: rally_serving_updates is now built by the Viterbi
                # post-processing step below, not per-rally. The per-rally
                # formation prediction is still stamped in actions_json for
                # backward compatibility with the API reader.

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

    # Cross-rally Viterbi scoring: override per-rally formation predictions
    # with the Viterbi-decoded serving_team sequence. This uses the physical
    # side formation signal + dual-hypothesis convention + position-based
    # switch detection for +10.9pp over per-rally predictions.
    # See score_tracking_investigation_design.md.
    from rallycut.scoring.cross_rally_viterbi import (  # noqa: PLC0415
        RallyObservation,
        decode_video_dual_hypothesis,
    )
    from rallycut.tracking.action_classifier import (  # noqa: PLC0415
        _find_serving_side_by_formation,
    )
    from rallycut.tracking.team_identity import (  # noqa: PLC0415
        TeamTemplate,
        localize_team_near,
        resolve_serving_team,
    )

    team_templates: tuple[TeamTemplate, TeamTemplate] | None = None
    templates_data = match_analysis.get("teamTemplates")
    if templates_data and isinstance(templates_data, dict):
        t0_data = templates_data.get("0")
        t1_data = templates_data.get("1")
        if t0_data and t1_data:
            t0 = TeamTemplate.from_dict(t0_data)
            t1 = TeamTemplate.from_dict(t1_data)
            team_templates = (t0, t1)

    # Build per-rally track_to_player lookup from match_analysis
    rally_t2p: dict[str, dict[int, int]] = {}
    for rally_entry in match_analysis.get("rallies", []):
        rid_entry = str(rally_entry.get("rallyId") or rally_entry.get("rally_id", ""))
        t2p = rally_entry.get("trackToPlayer") or rally_entry.get("track_to_player", {})
        if rid_entry and t2p:
            rally_t2p[rid_entry] = {int(k): int(v) for k, v in t2p.items()}

    # Build observations from positions collected during the loop.
    viterbi_observations: list[RallyObservation] = []
    viterbi_rally_ids: list[str] = []
    team_near_labels: list[str | None] = []
    for (
        rally_id_val2, _pt_id_val2, _cj, _aj,
        positions_json_val2, _sms, court_split_y_val2,
    ) in rows:
        rid2 = str(rally_id_val2)
        viterbi_rally_ids.append(rid2)
        positions_raw = positions_json_val2
        split_y = court_split_y_val2
        formation_side: str | None = None
        formation_conf = 0.0
        team_near: str | None = None
        if positions_raw:
            from rallycut.tracking.player_tracker import (  # noqa: PLC0415
                PlayerPosition as _PlayerPos,
            )
            pos_list = [
                _PlayerPos(
                    frame_number=p["frameNumber"], track_id=p["trackId"],
                    x=p["x"], y=p["y"],
                    width=p.get("width", 0.05), height=p.get("height", 0.10),
                    confidence=p.get("confidence", 1.0), keypoints=p.get("keypoints"),
                )
                for p in cast(list[dict[str, Any]], positions_raw)
            ]
            net_y = float(str(split_y)) if split_y is not None else 0.5
            formation_side, formation_conf = _find_serving_side_by_formation(
                pos_list, net_y=net_y, start_frame=0,
            )

            # Team localization: determine which team is near using
            # player IDs from track_to_player + Y positions.
            if team_templates is not None and rid2 in rally_t2p:
                team_near = localize_team_near(
                    pos_list, rally_t2p[rid2], team_templates,
                )

        viterbi_observations.append(RallyObservation(
            rally_id=rid2,
            formation_side=formation_side,
            formation_confidence=formation_conf,
        ))
        team_near_labels.append(team_near)

    # Per-rally team localization: directly determine serving team from
    # formation (which side serves) + track_to_player (which team is on
    # which side). No accumulated side-switch state needed.
    has_team_loc = any(tn is not None for tn in team_near_labels)

    if has_team_loc and team_templates is not None:
        # Convention: first rally's near team = A (production default)
        label_a = next(
            (tn for tn in team_near_labels if tn is not None), "0",
        )

        for obs, tn in zip(viterbi_observations, team_near_labels):
            team = resolve_serving_team(
                obs.formation_side, tn, team_templates, label_a,
            )
            # Keep existing prediction when localization can't determine
            rally_serving_updates.append((obs.rally_id, team or "A"))

        formation_applied = sum(
            1 for (rid2, team), (
                _r, _p, _c, aj, _pos, _s, _cs
            ) in zip(rally_serving_updates, rows)
            if isinstance(aj, dict)
            and aj.get("servingTeam") != team
        )
        if not quiet:
            console.print(
                f"  [cyan]Team localization: {len(rally_serving_updates)} rallies, "
                f"{formation_applied} serving_team changes[/cyan]"
            )
    elif viterbi_observations:
        # Fallback: no team localization → use Viterbi with side switches
        switch_indices: set[int] = set()
        for i in range(1, len(viterbi_rally_ids)):
            cur_flip = formation_flips.get(viterbi_rally_ids[i], False)
            prev_flip = formation_flips.get(viterbi_rally_ids[i - 1], False)
            if cur_flip != prev_flip:
                switch_indices.add(i)

        decoded = decode_video_dual_hypothesis(
            viterbi_observations, side_switch_rallies=switch_indices,
        )
        rally_serving_updates = [
            (dec.rally_id, dec.serving_team) for dec in decoded
        ]
        formation_applied = sum(
            1 for (rid2, team), (
                _r, _p, _c, aj, _pos, _s, _cs
            ) in zip(rally_serving_updates, rows)
            if isinstance(aj, dict)
            and aj.get("servingTeam") != team
        )
        if not quiet:
            console.print(
                f"  [cyan]Viterbi scoring: {len(decoded)} rallies decoded, "
                f"{formation_applied} serving_team changes[/cyan]"
            )

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
                # Mirror formation-based serving_team to the rallies table
                # so it surfaces in the editor UI and score_accuracy eval.
                for rally_id_upd, serving in rally_serving_updates:
                    cur.execute(
                        "UPDATE rallies SET serving_team = %s WHERE id = %s",
                        [serving, rally_id_upd],
                    )
            conn.commit()

        if not quiet:
            console.print(
                f"  [green]Updated {len(updated_tracks)} player tracks in DB[/green]"
            )
            if rally_serving_updates:
                console.print(
                    f"  [green]Viterbi serving_team: "
                    f"{formation_applied} changed / "
                    f"{len(rally_serving_updates)} stamped[/green]"
                )
    elif dry_run and updated_tracks:
        if not quiet:
            console.print(
                f"  [yellow]Dry run: would update "
                f"{len(updated_tracks)} player tracks[/yellow]"
            )
            if rally_serving_updates:
                console.print(
                    f"  [yellow]Dry run: would stamp formation "
                    f"serving_team on {len(rally_serving_updates)} rallies "
                    f"({formation_applied} changed)[/yellow]"
                )
