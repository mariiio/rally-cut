"""Match players across rallies for consistent IDs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import typer
from rich.console import Console
from rich.table import Table

from rallycut.cli.utils import handle_errors
from rallycut.tracking._subtrack import SubTrackCandidate

console = Console()


def _build_per_frame_pid_map(
    track_to_player: dict[int, int],
    sub_tracks: list[SubTrackCandidate],
) -> dict[tuple[int, int], int]:
    """Translate Hungarian + sub-track direct-assignment output into a
    per-frame label mapping.

    Returns dict keyed on ``(parent_track_id, frame_number) -> pid``.

    Conventions:
    - Real (un-split) tracks: key is ``(track_id, -1)`` — sentinel meaning
      "any frame". The CLI's per-frame writer first looks up
      ``(track_id, frame_number)`` and falls back to ``(track_id, -1)``.
    - Split tracks: each sub-track contributes
      ``(parent_track_id, f) -> pid`` for ``f`` in ``[f_start, f_end]``.
      Sub-tracks that lost the pid conflict in
      ``_apply_subtrack_assignments`` are absent from ``track_to_player``
      and therefore their frames are absent from the returned map (writer
      leaves those frames unlabeled).

    This matches the "miss > wrong" north-star: when a sub-track loses a
    conflict and its frames have no pid resolution, the writer emits no
    label rather than guessing.
    """
    sub_by_synth = {s.synthetic_track_id: s for s in sub_tracks}
    split_parents = {s.parent_track_id for s in sub_tracks}

    out: dict[tuple[int, int], int] = {}
    for tid, pid in track_to_player.items():
        if tid in sub_by_synth:
            sub = sub_by_synth[tid]
            for f in range(sub.f_start, sub.f_end + 1):
                out[(sub.parent_track_id, f)] = pid
        elif tid in split_parents:
            # Should not happen — split parents are removed from track_to_player
            # in `_apply_subtrack_assignments`. Guard defensively.
            continue
        else:
            out[(tid, -1)] = pid
    return out




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
    reset_anchors: bool = typer.Option(
        False,
        "--reset-anchors",
        help=(
            "Strip every `assignmentAnchor` from match_analysis_json before "
            "running, forcing MatchSolver to re-solve every rally from "
            "scratch. Use after upstream pipeline changes (player-tracker "
            "retune, court-calibration update) that don't change the "
            "structural fingerprint but DO change semantic input. The cache "
            "naturally invalidates on re-tracking; --reset-anchors covers "
            "the cases that don't shift track IDs."
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
        # Also NULL out `pre_remap_state_json` for the reversed rallies: the
        # cached snapshot reflected the prior tracker output and is stale
        # relative to today's reverse. Leaving it stale causes the next
        # `remap-track-ids` run to refuse rallies where the snapshot's
        # `primaryTrackIds` reference original IDs absent from the fresh
        # `trackToPlayer` (the "snapshot's primaryTrackIds contains raw
        # tracks absent from the resolved raw_mapping" fail-closed branch).
        # Nulling triggers `remap-track-ids` to auto-recapture from current
        # row state on its next run — same outcome as
        # `--reset-snapshot`. All updates committed atomically.
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
                            "primary_track_ids = %s, "
                            "pre_remap_state_json = NULL "
                            "WHERE rally_id = %s",
                            [json.dumps(pos_data),
                             json.dumps([int(t) for t in rally.primary_track_ids]),
                             rally.rally_id],
                        )
                conn.commit()

        if reversed_ids and not quiet:
            console.print(
                f"  Reversed previous remap on {len(reversed_ids)} rallies "
                f"(pre_remap_state cleared so remap-track-ids re-captures)"
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

    # Load ReID model (blind path only). Uses the fine-tuned
    # OSNet-x1.0 backbone (`GeneralReIDModel`). The DINOv2 alternative
    # (RALLYCUT_REID_BACKBONE=dinov2_vitl14) was benchmarked WORSE than
    # OSNet for beach VB (mean off-diagonal cosine 0.70 vs 0.05) and
    # removed 2026-05-03 per `dormant_flag_audit_2026_05_03.md`.
    general_reid_model: Any = None
    from rallycut.tracking.reid_general import WEIGHTS_PATH as REID_WEIGHTS_PATH

    if REID_WEIGHTS_PATH.exists():
        from rallycut.tracking.reid_general import GeneralReIDModel

        general_reid_model = GeneralReIDModel(weights_path=REID_WEIGHTS_PATH)
        if not quiet:
            console.print("  Using general ReID model")
            console.print()

    # --reset-anchors: strip prior assignmentAnchor entries before solving.
    # Caller has explicitly invalidated the cache; the next solve will
    # rebuild fresh anchors. We mutate a copy so the DB-stored prior
    # state is untouched until the new match_analysis_json overwrites it.
    prior_match_analysis_for_solver: dict[str, Any] | None = old_match_analysis
    if reset_anchors and old_match_analysis:
        from copy import deepcopy
        scrubbed = deepcopy(old_match_analysis)
        n_stripped = 0
        for entry in scrubbed.get("rallies", []):
            if entry.pop("assignmentAnchor", None) is not None:
                n_stripped += 1
        prior_match_analysis_for_solver = scrubbed
        if not quiet:
            console.print(
                f"  [yellow]--reset-anchors:[/yellow] stripped {n_stripped} "
                "assignmentAnchor entries; all rallies will re-solve."
            )

    # Run matching. `prior_match_analysis` lets the blind path read
    # per-rally `assignmentAnchor` entries from the previous run and pin
    # those rallies' MatchSolver decisions when the pre-solve hash still
    # matches (cascade fix; default ON, set ENABLE_ASSIGNMENT_ANCHORS=0
    # to disable).
    match_result: MatchPlayersResult = match_players_across_rallies(
        video_path=video_path,
        rallies=rallies,
        num_samples=num_samples,
        reid_model=general_reid_model,
        calibrator=court_calibrator,
        prior_match_analysis=prior_match_analysis_for_solver,
    )
    results = match_result.rally_results

    if not quiet and match_result.anchor_cache_total > 0:
        hits = match_result.anchor_cache_hits
        total = match_result.anchor_cache_total
        if hits > 0:
            console.print(
                f"  [green]Anchor cache:[/green] {hits}/{total} rallies pinned "
                "(skipped MatchSolver re-decision)"
            )
        else:
            console.print(
                f"  [dim]Anchor cache: 0/{total} pinned (no prior anchors "
                "or hashes mismatch — fresh solve)[/dim]"
            )

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
        # Strip synthetic sub-track ids from trackToPlayer so old consumers
        # (which only know real track ids) see a clean mapping. The full
        # picture is the union of trackToPlayer (real tracks → pid) and
        # subTracks (parent tid + frame range → pid). When no splits
        # occurred, sub_tracks_for_rally is [] and the behavior is
        # byte-identical to the pre-segmentation flow.
        sub_tracks_for_rally = result.sub_tracks or []
        sub_track_synth_ids = {s.synthetic_track_id for s in sub_tracks_for_rally}
        real_track_to_player = {
            str(k): v for k, v in result.track_to_player.items()
            if k not in sub_track_synth_ids
        }
        rally_entry: dict[str, Any] = {
            "rallyId": rally.rally_id,
            "rallyIndex": result.rally_index,
            "startMs": rally.start_ms,
            "endMs": rally.end_ms,
            "trackToPlayer": real_track_to_player,
            "assignmentConfidence": result.assignment_confidence,
            "sideSwitchDetected": result.side_switch_detected,
            "serverPlayerId": result.server_player_id,
        }
        if sub_tracks_for_rally:
            # Only emit subTracks for sub-track candidates that won their pid
            # conflict in `_apply_subtrack_assignments` (synthetic id ended up
            # in track_to_player). Losers are omitted — their frames will be
            # written unlabeled by the remap-track-ids pass.
            sub_track_entries = [
                {
                    "syntheticTrackId": s.synthetic_track_id,
                    "parentTrackId": s.parent_track_id,
                    "segmentIndex": s.segment_index,
                    "fStart": s.f_start,
                    "fEnd": s.f_end,
                    "pid": result.track_to_player.get(s.synthetic_track_id),
                    "margin": s.aggregated_margin,
                }
                for s in sub_tracks_for_rally
                if s.synthetic_track_id in result.track_to_player
            ]
            if sub_track_entries:
                rally_entry["subTracks"] = sub_track_entries
        # Persist assignmentAnchor for blind-path solves so the next run
        # can pin this rally and skip MatchSolver re-decision when
        # pre-solve state is unchanged. Confidence gate: rallies below
        # ANCHOR_MIN_CONFIDENCE re-solve every run instead of locking in
        # an uncertain assignment. Hash is only populated for the blind
        # branch — ref-crop runs leave it empty and no anchor is written.
        from rallycut.tracking.match_tracker import (
            ANCHOR_MIN_CONFIDENCE,
            MATCHER_VERSION,
        )
        ts_hash = match_result.track_stats_hashes.get(rally.rally_id)
        confidence_ok = result.assignment_confidence >= ANCHOR_MIN_CONFIDENCE
        if ts_hash and result.track_to_player and confidence_ok:
            rally_entry["assignmentAnchor"] = {
                "trackStatsHash": ts_hash,
                "matcherVersion": MATCHER_VERSION,
                "assignment": {
                    str(int(k)): int(v)
                    for k, v in result.track_to_player.items()
                    if int(k) > 0  # exclude any synthetic sub-track ids
                },
                "confidence": float(result.assignment_confidence),
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
    # Persist the per-rally scratchpad so future diagnostics can inspect
    # intermediate Pass 2 state without re-running tracking.
    if match_result.scratchpad:
        result_json["rallyScratchpad"] = match_result.scratchpad

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE videos SET match_analysis_json = %s WHERE id = %s",
                [json.dumps(result_json), video_id],
            )
        conn.commit()

    if not quiet:
        console.print("  Saved to DB")

    # Write JSON file output. Single source of truth: the same `result_json`
    # dict that was just persisted to videos.match_analysis_json. The API
    # path (api/src/services/matchAnalysisService.ts:runMatchPlayersCli)
    # parses this file and re-writes it to DB; if the shapes diverge the
    # round-trip silently strips fields (rallyScratchpad, assignmentAnchor,
    # subTracks, teamTemplates) and converts camelCase keys to snake_case.
    if output:
        with open(output, "w") as f:
            json.dump(result_json, f, indent=2)

        if not quiet:
            console.print(f"\n[green]Results saved to {output}[/green]")
