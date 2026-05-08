"""Evaluate player tracking against ground truth from database."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.table import Table

from rallycut.cli.utils import handle_errors

if TYPE_CHECKING:
    from rallycut.cli.commands.compare_tracking import BallMetrics, MOTMetrics
    from rallycut.evaluation.tracking.ball_metrics import BallTrackingMetrics
    from rallycut.evaluation.tracking.error_analysis import ErrorEvent
    from rallycut.evaluation.tracking.metrics import PerPlayerMetrics, TrackingEvaluationResult
    from rallycut.tracking.player_tracker import PlayerPosition

app = typer.Typer(help="Evaluate player tracking against ground truth")
console = Console()


def _status_icon(value: float, target: float, higher_better: bool = True) -> str:
    """Return status icon based on whether value meets target."""
    if higher_better:
        return "[green]OK[/green]" if value >= target else "[yellow]Below[/yellow]"
    else:
        return "[green]OK[/green]" if value <= target else "[yellow]Above[/yellow]"


@app.callback(invoke_without_command=True)
@handle_errors
def evaluate_tracking(
    ctx: typer.Context,
    rally_id: str = typer.Option(
        None,
        "--rally-id", "-r",
        help="Evaluate specific rally by ID",
    ),
    video_id: str = typer.Option(
        None,
        "--video-id", "-v",
        help="Evaluate all labeled rallies in a video",
    ),
    all_rallies: bool = typer.Option(
        False,
        "--all", "-a",
        help="Evaluate all labeled rallies in database",
    ),
    per_player: bool = typer.Option(
        False,
        "--per-player", "-p",
        help="Show per-player breakdown",
    ),
    analyze_errors: bool = typer.Option(
        False,
        "--analyze-errors", "-e",
        help="Show detailed error analysis",
    ),
    ball_only: bool = typer.Option(
        False,
        "--ball-only", "-b",
        help="Evaluate ball tracking only (skip player metrics)",
    ),
    iou_threshold: float = typer.Option(
        0.5,
        "--iou", "-i",
        help="Minimum IoU for matching predictions to ground truth",
    ),
    output: Path = typer.Option(
        None,
        "--output", "-o",
        help="Output metrics to JSON file",
    ),
    retrack: bool = typer.Option(
        False,
        "--retrack",
        help="Re-run tracking pipeline instead of using DB predictions",
    ),
    cached: bool = typer.Option(
        False,
        "--cached",
        help="Cache/reuse raw BoT-SORT detections (requires --retrack)",
    ),
    clear_retrack_cache: bool = typer.Option(
        False,
        "--clear-retrack-cache",
        help="Clear retrack cache before running",
    ),
    audit_out: Path = typer.Option(
        None,
        "--audit-out",
        help=(
            "Emit per-rally audit JSON (same schema as "
            "`reports/tracking_audit/reid_debug`) to this directory. "
            "Requires --retrack. Used by the Session-4 learned-ReID gate "
            "script to re-audit the in-memory retrack output without "
            "writing predictions back to DB."
        ),
    ),
) -> None:
    """Evaluate player and ball tracking predictions against ground truth.

    Loads ground truth and predictions from the database and computes
    MOT (Multi-Object Tracking) metrics including MOTA, precision, recall,
    F1, and ID switches for players. For ball tracking, computes detection
    rate, position error statistics, and accuracy buckets.

    Examples:

        # Evaluate specific rally (players + ball)
        rallycut evaluate-tracking --rally-id abc123

        # Evaluate all labeled rallies in a video
        rallycut evaluate-tracking --video-id a7ee3d38-...

        # Evaluate all labeled data
        rallycut evaluate-tracking --all

        # Ball tracking evaluation only
        rallycut evaluate-tracking --all --ball-only

        # Re-run tracking pipeline (fresh YOLO + post-processing)
        rallycut evaluate-tracking --all --retrack

        # Re-run with cached raw detections (fast post-processing only)
        rallycut evaluate-tracking --all --retrack --cached

        # Show per-player breakdown
        rallycut evaluate-tracking --rally-id abc123 --per-player

        # Export to JSON
        rallycut evaluate-tracking --rally-id abc123 -o metrics.json

        # Grid search for optimal filter parameters
        rallycut evaluate-tracking tune-filter --all --grid quick
    """
    # If a subcommand was invoked, don't run the default
    if ctx.invoked_subcommand is not None:
        return

    # Validate --cached requires --retrack
    if cached and not retrack:
        console.print("[red]Error:[/red] --cached requires --retrack")
        raise typer.Exit(1)

    # Retrack path: re-run the tracking pipeline instead of using DB predictions
    if retrack:
        _run_retrack_evaluation(
            rally_id=rally_id,
            video_id=video_id,
            all_rallies=all_rallies,
            cached=cached,
            clear_retrack_cache=clear_retrack_cache,
            iou_threshold=iou_threshold,
            per_player=per_player,
            analyze_errors=analyze_errors,
            output=output,
            audit_out=audit_out,
        )
        return

    if audit_out is not None:
        console.print("[red]Error:[/red] --audit-out requires --retrack")
        raise typer.Exit(1)

    from rallycut.evaluation.tracking.ball_metrics import (
        aggregate_ball_metrics,
        evaluate_ball_tracking,
    )
    from rallycut.evaluation.tracking.db import load_labeled_rallies
    from rallycut.evaluation.tracking.error_analysis import analyze_errors as get_errors
    from rallycut.evaluation.tracking.metrics import aggregate_results, evaluate_rally

    # Validate input
    if not rally_id and not video_id and not all_rallies:
        console.print(
            "[red]Error:[/red] Specify --rally-id, --video-id, or --all"
        )
        raise typer.Exit(1)

    # Load rallies from database
    console.print("[bold]Loading labeled rallies from database...[/bold]")

    rallies = load_labeled_rallies(
        video_id=video_id,
        rally_id=rally_id,
    )

    if not rallies:
        console.print("[yellow]No rallies with ground truth found[/yellow]")
        if rally_id:
            console.print(f"  Rally ID: {rally_id}")
        if video_id:
            console.print(f"  Video ID: {video_id}")
        raise typer.Exit(1)

    console.print(f"  Found {len(rallies)} rally(s) with ground truth\n")

    # Ball-only evaluation mode
    if ball_only:
        from rallycut.evaluation.tracking.ball_grid_search import (
            BallRawCache,
            apply_ball_filter_config,
        )
        from rallycut.tracking.ball_filter import BallFilterConfig

        raw_cache = BallRawCache()
        ball_results = []
        for rally in rallies:
            # Prefer cached raw positions + current filter over stale DB positions
            ball_cached = raw_cache.get(rally.rally_id)
            if ball_cached is not None:
                predictions = apply_ball_filter_config(
                    ball_cached.raw_ball_positions, BallFilterConfig()
                )
            elif rally.predictions is not None and rally.predictions.ball_positions:
                predictions = rally.predictions.ball_positions
            else:
                console.print(
                    f"[yellow]Rally {rally.rally_id[:8]}... has no ball predictions, skipping[/yellow]"
                )
                continue

            ball_metrics = evaluate_ball_tracking(
                ground_truth=rally.ground_truth.positions,
                predictions=predictions,
                video_width=rally.video_width,
                video_height=rally.video_height,
                video_fps=rally.video_fps,
            )
            ball_results.append((rally.rally_id, ball_metrics))

        if not ball_results:
            console.print("[red]No rallies with ball tracking data found[/red]")
            raise typer.Exit(1)

        # Display ball-only results
        if len(ball_results) == 1:
            rally_id_str, ball_metrics = ball_results[0]
            console.print(f"[bold]Ball Tracking Evaluation - Rally {rally_id_str[:8]}...[/bold]")
            console.print("=" * 50)
            _display_enhanced_ball_metrics(ball_metrics)
        else:
            console.print(f"[bold]Ball Tracking Evaluation - {len(ball_results)} Rallies[/bold]")
            console.print("=" * 50)

            # Per-rally table
            ball_table = Table(show_header=True, header_style="bold")
            ball_table.add_column("Rally")
            ball_table.add_column("Detection", justify="right")
            ball_table.add_column("Match", justify="right")
            ball_table.add_column("Mean Err", justify="right")
            ball_table.add_column("Median", justify="right")
            ball_table.add_column("P90", justify="right")
            ball_table.add_column("<20px", justify="right")
            ball_table.add_column("Offset", justify="right")

            for rally_id_str, metrics in ball_results:
                ball_table.add_row(
                    rally_id_str[:8] + "...",
                    f"{metrics.detection_rate:.1%}",
                    f"{metrics.match_rate:.1%}",
                    f"{metrics.mean_error_px:.1f}px",
                    f"{metrics.median_error_px:.1f}px",
                    f"{metrics.p90_error_px:.1f}px",
                    f"{metrics.error_under_20px_rate:.1%}",
                    f"+{metrics.frame_offset}" if metrics.frame_offset else "0",
                )

            console.print(ball_table)

            # Aggregate metrics
            console.print("\n[bold]Aggregate Metrics[/bold]")
            ball_combined = aggregate_ball_metrics([m for _, m in ball_results])
            _display_enhanced_ball_metrics(ball_combined)

        # Save to file if requested
        if output:
            if len(ball_results) == 1:
                output_data = ball_results[0][1].to_dict()
            else:
                ball_combined = aggregate_ball_metrics([m for _, m in ball_results])
                output_data = {
                    "rallies": [
                        {"rallyId": rid, **m.to_dict()}
                        for rid, m in ball_results
                    ],
                    "aggregate": ball_combined.to_dict(),
                }

            with open(output, "w") as f:
                json.dump(output_data, f, indent=2)
            console.print(f"\n[green]Metrics saved to {output}[/green]")

        return

    # Evaluate each rally (player + ball combined)
    results = []
    all_errors_list = []

    for rally in rallies:
        if rally.predictions is None:
            console.print(
                f"[yellow]Rally {rally.rally_id[:8]}... has no predictions, skipping[/yellow]"
            )
            continue

        result = evaluate_rally(
            rally_id=rally.rally_id,
            ground_truth=rally.ground_truth,
            predictions=rally.predictions,
            iou_threshold=iou_threshold,
            video_width=rally.video_width,
            video_height=rally.video_height,
        )
        results.append(result)

        if analyze_errors:
            errors = get_errors(
                rally.ground_truth,
                rally.predictions,
                iou_threshold,
            )
            all_errors_list.extend(errors)

    if not results:
        console.print("[red]No rallies with predictions found[/red]")
        raise typer.Exit(1)

    # Display results
    if len(results) == 1:
        # Single rally - show detailed output
        result = results[0]

        console.print(f"[bold]Player Tracking Evaluation - Rally {result.rally_id[:8]}...[/bold]")
        console.print("=" * 50)

        _display_aggregate_metrics(result.aggregate)

        # Display extended metrics (HOTA, track quality, position accuracy)
        _display_extended_metrics(result)

        if result.ball_metrics:
            _display_ball_metrics(result.ball_metrics)

        if per_player and result.per_player:
            _display_per_player_metrics(result.per_player)

        # Error frames summary
        error_count = len(result.error_frames)
        total_frames = len(result.per_frame)
        if error_count > 0:
            console.print(f"\n[bold]Error Frames:[/bold] {error_count} of {total_frames} frames")
            if error_count <= 10:
                console.print(f"  Frames: {', '.join(str(f) for f in result.error_frames)}")
            else:
                first_five = result.error_frames[:5]
                last_five = result.error_frames[-5:]
                console.print(
                    f"  First 5: {', '.join(str(f) for f in first_five)}"
                )
                console.print(
                    f"  Last 5: {', '.join(str(f) for f in last_five)}"
                )
        else:
            console.print("\n[green]No error frames![/green]")

        if analyze_errors and all_errors_list:
            _display_error_analysis(all_errors_list)

    else:
        # Multiple rallies - show summary
        console.print(f"[bold]Player Tracking Evaluation - {len(results)} Rallies[/bold]")
        console.print("=" * 50)

        # Per-rally table with extended metrics
        rally_table = Table(show_header=True, header_style="bold")
        rally_table.add_column("Rally")
        rally_table.add_column("HOTA", justify="right")
        rally_table.add_column("F1", justify="right")
        rally_table.add_column("IDsw", justify="right")
        rally_table.add_column("Real", justify="right")  # Real identity switches
        rally_table.add_column("ID Acc", justify="right")  # Identity accuracy
        rally_table.add_column("MT", justify="right")  # Mostly Tracked

        for result in results:
            hota_str = "-"
            if result.hota_metrics:
                hota_str = f"{result.hota_metrics.hota:.1%}"

            mt_str = "-"
            if result.track_quality:
                mt_str = f"{result.track_quality.mostly_tracked}/{result.track_quality.gt_track_count}"

            real_str = "-"
            id_acc_str = "-"
            if result.identity_metrics:
                real_str = str(result.identity_metrics.num_switches)
                id_acc_str = f"{result.identity_metrics.identity_accuracy:.1%}"

            rally_table.add_row(
                result.rally_id[:8] + "...",
                hota_str,
                f"{result.aggregate.f1:.1%}",
                str(result.aggregate.num_id_switches),
                real_str,
                id_acc_str,
                mt_str,
            )

        console.print(rally_table)

        # Aggregate metrics
        console.print("\n[bold]Aggregate Metrics[/bold]")
        combined = aggregate_results(results)
        _display_aggregate_metrics(combined)

        # Aggregate extended metrics (average across rallies)
        _display_aggregate_extended_metrics(results)

        if analyze_errors and all_errors_list:
            _display_error_analysis(all_errors_list)

    # Save to file if requested
    if output:
        if len(results) == 1:
            output_data = results[0].to_dict()
        else:
            combined = aggregate_results(results)
            output_data = {
                "rallies": [r.to_dict() for r in results],
                "aggregate": {
                    "mota": combined.mota,
                    "precision": combined.precision,
                    "recall": combined.recall,
                    "f1": combined.f1,
                    "idSwitches": combined.num_id_switches,
                },
            }

        with open(output, "w") as f:
            json.dump(output_data, f, indent=2)
        console.print(f"\n[green]Metrics saved to {output}[/green]")


def _compute_tracker_config_hash() -> str:
    """Compute a hash of the detection-phase config for cache invalidation.

    Captures everything that affects YOLO+BoT-SORT output (Phase 1) and
    whether learned-ReID embeddings were extracted in the frame loop.
    Post-processing config is deliberately excluded — that's what we iterate on.
    """
    import hashlib
    import os

    from rallycut.tracking.player_tracker import (
        DEFAULT_CONFIDENCE,
        DEFAULT_COURT_ROI,
        DEFAULT_IMGSZ,
        DEFAULT_IOU,
        DEFAULT_TRACKER,
        MODEL_NAME,
    )
    from rallycut.tracking.reid_embeddings import HEAD_SHA

    # Session 4 — learned-ReID extraction changes what lives in
    # the cache (embedding arrays + head_sha pin). Invalidate when toggled
    # or when the head checkpoint changes. Weight VALUE is NOT in the hash:
    # downstream cost uses it multiplicatively; stored embeddings are
    # weight-invariant, so a sweep over {0.05..0.20} reuses one cache.
    learned_reid_enabled = (
        float(os.environ.get("WEIGHT_LEARNED_REID", "0.0")) > 0
    )

    config_parts = [
        f"model={MODEL_NAME}",
        f"conf={DEFAULT_CONFIDENCE}",
        f"iou={DEFAULT_IOU}",
        f"imgsz={DEFAULT_IMGSZ}",
        f"tracker={DEFAULT_TRACKER}",
        f"roi={DEFAULT_COURT_ROI}",
    ]
    # Only append the learned-ReID fragment when enabled — preserves the
    # pre-integration hash byte-identically for the W=0 control so the
    # existing 54 MB retrack cache is reused.
    if learned_reid_enabled:
        config_parts.append(f"learned_reid:True:{HEAD_SHA}")
    # Session 5 — occlusion resolver runs AFTER cache load and before
    # global_identity, so it doesn't change what's cached (same positions,
    # same stores). BUT its position mutations change what's fed to the
    # downstream stages, so we mark it in the hash ONLY to segregate
    # results for apples-to-apples baseline comparison.
    if os.environ.get("ENABLE_OCCLUSION_RESOLVER", "0") == "1":
        config_parts.append("occlusion_resolver:True")
    # Session 6 — learned-head merge veto in tracklet_link runs entirely
    # during post-processing (doesn't change what's CACHED — raw positions
    # + learned_store + color_store are all threshold-invariant). A
    # threshold sweep therefore SHARES one raw cache across all values;
    # the hash only marks enabled/disabled, not the numeric threshold.
    # Default (disabled) preserves the pre-Session-6 cache key byte-
    # identically with Session-4-era caches.
    learned_merge_veto_cos = float(
        os.environ.get("LEARNED_MERGE_VETO_COS", "0.0")
    )
    if learned_merge_veto_cos > 0:
        config_parts.append("merge_veto:enabled")
    # Court-plane velocity gate (Session 6b / Session 8 Plan B) runs
    # entirely during post-processing (inside link_tracklets_by_appearance
    # → _greedy_merge → _would_create_velocity_anomaly). It uses the
    # court_calibrator passed at apply_post_processing time, NOT anything
    # stored in the cache. Different velocity thresholds share one raw
    # cache — same principle as the learned merge veto above. NOT in hash.
    config_str = "|".join(config_parts)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def _run_retrack_evaluation(
    *,
    rally_id: str | None,
    video_id: str | None,
    all_rallies: bool,
    cached: bool,
    clear_retrack_cache: bool,
    iou_threshold: float,
    per_player: bool,
    analyze_errors: bool,
    output: Path | None,
    audit_out: Path | None = None,
) -> None:
    """Re-run tracking pipeline on GT rallies and evaluate.

    This is the implementation behind --retrack and --retrack --cached.
    """
    import time as time_mod

    from rallycut.evaluation.tracking.db import get_video_path, load_labeled_rallies
    from rallycut.evaluation.tracking.metrics import aggregate_results, evaluate_rally
    from rallycut.evaluation.tracking.retrack_cache import (
        CachedRetrackData,
        RetrackCache,
    )
    from rallycut.evaluation.tracking.retrack_results import (
        RetrackRunResult,
        format_delta,
        format_delta_int,
        load_last_run,
        save_run,
    )
    from rallycut.tracking.player_filter import PlayerFilterConfig
    from rallycut.tracking.player_tracker import (
        DEFAULT_TRACKER,
        PlayerTracker,
        PlayerTrackingResult,
    )

    # Validate input
    if not rally_id and not video_id and not all_rallies:
        console.print("[red]Error:[/red] Specify --rally-id, --video-id, or --all")
        raise typer.Exit(1)

    # Setup cache
    retrack_cache: RetrackCache | None = None
    config_hash = _compute_tracker_config_hash()
    if cached:
        retrack_cache = RetrackCache()
        if clear_retrack_cache:
            count = retrack_cache.clear()
            console.print(f"[yellow]Cleared {count} retrack cache entries[/yellow]")

    # Load rallies from database (for GT + video_id + timing)
    console.print("[bold]Loading labeled rallies from database...[/bold]")
    rallies = load_labeled_rallies(
        video_id=video_id,
        rally_id=rally_id,
    )

    if not rallies:
        console.print("[yellow]No rallies with ground truth found[/yellow]")
        raise typer.Exit(1)

    console.print(f"  Found {len(rallies)} rally(s) with ground truth")

    # Group rallies by video (download each video once)
    rallies_by_video: dict[str, list] = {}
    for rally in rallies:
        vid = rally.video_id
        if vid not in rallies_by_video:
            rallies_by_video[vid] = []
        rallies_by_video[vid].append(rally)

    console.print(
        f"  {len(rallies_by_video)} video(s), "
        f"{'cached' if cached else 'full retrack'} mode\n"
    )

    # Load previous run for delta comparison
    previous_run = load_last_run()

    results = []
    current_run: dict[str, RetrackRunResult] = {}
    filter_config = PlayerFilterConfig()
    total_rallies = len(rallies)
    rally_idx = 0
    total_start = time_mod.time()

    for video_id_key, video_rallies in rallies_by_video.items():
        # Check cache hits for this video's rallies
        cache_hits = 0
        if retrack_cache is not None:
            cache_hits = sum(
                1 for r in video_rallies
                if retrack_cache.has(r.rally_id, config_hash)
            )

        # Only download video if we have cache misses
        video_path = None
        need_video = cache_hits < len(video_rallies)
        if need_video:
            video_path = get_video_path(video_id_key)
            if not video_path or not video_path.exists():
                console.print(
                    f"[yellow]Could not download video {video_id_key[:8]}... "
                    f"({len(video_rallies)} rallies skipped)[/yellow]"
                )
                rally_idx += len(video_rallies)
                continue

        # Create fresh tracker per video
        tracker: PlayerTracker | None = None
        if need_video:
            tracker = PlayerTracker(tracker=DEFAULT_TRACKER)

        for rally in video_rallies:
            rally_idx += 1
            rally_start = time_mod.time()
            rally_label = f"[{rally_idx}/{total_rallies}] {rally.rally_id[:8]}..."

            # Get ball positions from DB for filtering
            ball_positions = None
            if rally.predictions is not None:
                ball_positions = rally.predictions.ball_positions or None

            # Get court calibration if available
            court_calibrator = None
            cal_json = rally.court_calibration_json
            if cal_json and isinstance(cal_json, list) and len(cal_json) == 4:
                try:
                    from rallycut.court.calibration import CourtCalibrator

                    court_calibrator = CourtCalibrator()
                    court_calibrator.calibrate(
                        [(c["x"], c["y"]) for c in cal_json]
                    )
                    if not court_calibrator.is_calibrated:
                        court_calibrator = None
                except Exception:
                    court_calibrator = None

            tracking_result: PlayerTrackingResult | None = None
            raw_positions_for_audit: list[PlayerPosition] | None = None

            # Try cache first
            if retrack_cache is not None:
                cache_entry = retrack_cache.get(rally.rally_id, config_hash)
                if cache_entry is not None:
                    cached_data, color_store, appearance_store, learned_store = (
                        cache_entry
                    )
                    raw_positions_for_audit = list(cached_data.positions)
                    tracking_result = PlayerTracker.apply_post_processing(
                        positions=cached_data.positions,
                        raw_positions=list(cached_data.positions),
                        color_store=color_store,
                        appearance_store=appearance_store,
                        ball_positions=ball_positions,
                        video_fps=cached_data.video_fps,
                        video_width=cached_data.video_width,
                        video_height=cached_data.video_height,
                        frame_count=cached_data.frame_count,
                        start_frame=0,  # Cache stores rally-relative frames
                        filter_enabled=True,
                        filter_config=filter_config.scaled_for_fps(
                            cached_data.video_fps
                        ),
                        court_calibrator=court_calibrator,
                        learned_store=learned_store,
                    )
                    elapsed = time_mod.time() - rally_start
                    console.print(
                        f"  {rally_label} [green]cached[/green] "
                        f"({elapsed:.1f}s)"
                    )

            # Cache miss or no cache: run full tracking
            if tracking_result is None:
                if tracker is None or video_path is None:
                    console.print(
                        f"  {rally_label} [yellow]skipped (no video)[/yellow]"
                    )
                    continue

                console.print(f"  {rally_label} tracking...", end="")

                if retrack_cache is not None:
                    # Run tracking and capture raw data for caching
                    result_tuple = tracker.track_video(
                        video_path,
                        start_ms=rally.start_ms,
                        end_ms=rally.end_ms,
                        filter_enabled=True,
                        filter_config=filter_config,
                        ball_positions=ball_positions,
                        court_calibrator=court_calibrator,
                        return_raw=True,
                    )
                    tracking_result, raw_data = result_tuple
                    raw_positions_for_audit = list(raw_data.positions)

                    # Cache the raw data
                    from rallycut.tracking.reid_embeddings import HEAD_SHA

                    cache_data = CachedRetrackData(
                        rally_id=rally.rally_id,
                        video_id=rally.video_id,
                        config_hash=config_hash,
                        positions=raw_data.positions,
                        ball_positions=ball_positions or [],
                        video_fps=tracking_result.video_fps,
                        video_width=tracking_result.video_width,
                        video_height=tracking_result.video_height,
                        frame_count=tracking_result.frame_count,
                        head_sha=(
                            HEAD_SHA if raw_data.learned_store is not None else ""
                        ),
                    )
                    retrack_cache.put(
                        cache_data,
                        raw_data.color_store,
                        raw_data.appearance_store,
                        raw_data.learned_store,
                    )
                else:
                    # Run tracking without caching
                    tracking_result = tracker.track_video(
                        video_path,
                        start_ms=rally.start_ms,
                        end_ms=rally.end_ms,
                        filter_enabled=True,
                        filter_config=filter_config,
                        ball_positions=ball_positions,
                        court_calibrator=court_calibrator,
                    )

                elapsed = time_mod.time() - rally_start
                console.print(f" done ({elapsed:.1f}s)")

            # Evaluate against GT
            if tracking_result is None:
                continue

            # Session 4 — optionally emit per-rally audit JSON alongside the
            # retrack output so the gate script can re-derive identity-swap
            # counts without writing to DB. Schema matches
            # `reports/tracking_audit/reid_debug/*.json` so
            # `_load_events_from_audit` parses it unchanged.
            if audit_out is not None:
                import json as _json

                from rallycut.evaluation.tracking.audit import build_rally_audit

                audit_out.mkdir(parents=True, exist_ok=True)
                try:
                    rally_audit = build_rally_audit(
                        rally_id=rally.rally_id,
                        video_id=rally.video_id,
                        ground_truth=rally.ground_truth,
                        predictions=tracking_result,
                        raw_positions=raw_positions_for_audit,
                        iou_threshold=iou_threshold,
                    )
                    audit_path = audit_out / f"{rally.rally_id}.json"
                    with open(audit_path, "w") as f:
                        _json.dump(rally_audit.to_dict(), f, indent=2)
                except Exception as audit_exc:  # noqa: BLE001
                    console.print(
                        f"  [yellow]audit emit failed for "
                        f"{rally.rally_id[:8]}: {audit_exc}[/yellow]"
                    )

            eval_result = evaluate_rally(
                rally_id=rally.rally_id,
                ground_truth=rally.ground_truth,
                predictions=tracking_result,
                iou_threshold=iou_threshold,
                video_width=rally.video_width,
                video_height=rally.video_height,
            )
            results.append(eval_result)

            # Store for delta comparison
            current_run[rally.rally_id] = RetrackRunResult(
                rally_id=rally.rally_id,
                hota=eval_result.hota_metrics.hota if eval_result.hota_metrics else None,
                f1=eval_result.aggregate.f1,
                id_switches=eval_result.aggregate.num_id_switches,
                identity_accuracy=(
                    eval_result.identity_metrics.identity_accuracy
                    if eval_result.identity_metrics
                    else None
                ),
            )

    total_elapsed = time_mod.time() - total_start

    if not results:
        console.print("[red]No rallies were successfully tracked[/red]")
        raise typer.Exit(1)

    # Display results table with deltas
    console.print(
        f"\n[bold]Retrack Evaluation - {len(results)} Rallies "
        f"({total_elapsed:.0f}s total)[/bold]"
    )
    console.print("=" * 70)

    rally_table = Table(show_header=True, header_style="bold")
    rally_table.add_column("Rally")
    rally_table.add_column("HOTA", justify="right")
    rally_table.add_column("F1", justify="right")
    rally_table.add_column("IDsw", justify="right")
    rally_table.add_column("Real", justify="right")
    rally_table.add_column("ID Acc", justify="right")
    rally_table.add_column("MT", justify="right")
    if previous_run:
        rally_table.add_column("ΔHOTA", justify="right")
        rally_table.add_column("ΔF1", justify="right")
        rally_table.add_column("ΔIDsw", justify="right")

    for result in results:
        hota_str = "-"
        if result.hota_metrics:
            hota_str = f"{result.hota_metrics.hota:.1%}"

        mt_str = "-"
        if result.track_quality:
            mt_str = f"{result.track_quality.mostly_tracked}/{result.track_quality.gt_track_count}"

        real_str = "-"
        id_acc_str = "-"
        if result.identity_metrics:
            real_str = str(result.identity_metrics.num_switches)
            id_acc_str = f"{result.identity_metrics.identity_accuracy:.1%}"

        row = [
            result.rally_id[:8] + "...",
            hota_str,
            f"{result.aggregate.f1:.1%}",
            str(result.aggregate.num_id_switches),
            real_str,
            id_acc_str,
            mt_str,
        ]

        if previous_run:
            prev = previous_run.get(result.rally_id)
            if prev:
                row.append(
                    format_delta(
                        result.hota_metrics.hota if result.hota_metrics else 0,
                        prev.hota,
                    )
                )
                row.append(format_delta(result.aggregate.f1, prev.f1))
                row.append(
                    format_delta_int(result.aggregate.num_id_switches, prev.id_switches)
                )
            else:
                row.extend(["", "", ""])

        rally_table.add_row(*row)

    console.print(rally_table)

    # Aggregate metrics
    console.print("\n[bold]Aggregate Metrics[/bold]")
    combined = aggregate_results(results)
    _display_aggregate_metrics(combined)
    _display_aggregate_extended_metrics(results)

    # Save current run for next delta comparison
    save_run(current_run)

    # Show cache stats
    if retrack_cache is not None:
        stats = retrack_cache.stats()
        console.print(
            f"\n[dim]Cache: {stats['count']} rallies, "
            f"{stats['total_size_mb']:.1f} MB in {stats['cache_dir']}[/dim]"
        )

    # Save to file if requested
    if output:
        combined = aggregate_results(results)
        output_data = {
            "rallies": [r.to_dict() for r in results],
            "aggregate": {
                "mota": combined.mota,
                "precision": combined.precision,
                "recall": combined.recall,
                "f1": combined.f1,
                "idSwitches": combined.num_id_switches,
            },
        }
        with open(output, "w") as f:
            json.dump(output_data, f, indent=2)
        console.print(f"\n[green]Metrics saved to {output}[/green]")


@app.command(name="tune-filter")
@handle_errors
def tune_filter(
    rally_id: Annotated[
        str | None,
        typer.Option(
            "--rally-id", "-r",
            help="Tune on specific rally by ID",
        ),
    ] = None,
    video_id: Annotated[
        str | None,
        typer.Option(
            "--video-id", "-v",
            help="Tune on all labeled rallies in a video",
        ),
    ] = None,
    all_rallies: Annotated[
        bool,
        typer.Option(
            "--all", "-a",
            help="Tune on all labeled rallies in database",
        ),
    ] = False,
    grid: Annotated[
        str,
        typer.Option(
            "--grid", "-g",
            help="Grid to search: quick, full, referee, stability, merge, farside, relaxed",
        ),
    ] = "quick",
    iou_threshold: Annotated[
        float,
        typer.Option(
            "--iou", "-i",
            help="IoU threshold for matching",
        ),
    ] = 0.5,
    min_rally_f1: Annotated[
        float | None,
        typer.Option(
            "--min-rally-f1",
            help="Reject configs where any rally drops below this F1",
        ),
    ] = None,
    top_n: Annotated[
        int,
        typer.Option(
            "--top", "-n",
            help="Number of top results to show",
        ),
    ] = 10,
    cache_only: Annotated[
        bool,
        typer.Option(
            "--cache-only",
            help="Only cache raw positions, don't run grid search",
        ),
    ] = False,
    clear_cache: Annotated[
        bool,
        typer.Option(
            "--clear-cache",
            help="Clear raw position cache before starting",
        ),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o",
            help="Export full results to JSON file",
        ),
    ] = None,
) -> None:
    """Grid search for optimal PlayerFilterConfig parameters.

    Searches over filter parameters to find the configuration that
    maximizes F1 score while minimizing ID switches.

    The key insight: YOLO+ByteTrack is slow (~seconds per rally), but
    the filter pipeline is fast (~milliseconds). By caching raw positions,
    we can re-run filtering with different configs without re-running detection.

    Examples:

        # Cache raw positions first (slow, one-time)
        rallycut evaluate-tracking tune-filter --all --cache-only

        # Quick grid search (36 combinations)
        rallycut evaluate-tracking tune-filter --all --grid quick

        # Full search with constraint
        rallycut evaluate-tracking tune-filter --all --grid full --min-rally-f1 0.70

        # Export results
        rallycut evaluate-tracking tune-filter --all -o results.json
    """
    from rallycut.evaluation.tracking.db import load_labeled_rallies
    from rallycut.evaluation.tracking.grid_search import grid_search
    from rallycut.evaluation.tracking.param_grid import (
        AVAILABLE_GRIDS,
        describe_config_diff,
        get_grid,
        grid_size,
    )
    from rallycut.evaluation.tracking.raw_cache import CachedRallyData, RawPositionCache

    # Validate input
    if not rally_id and not video_id and not all_rallies:
        console.print(
            "[red]Error:[/red] Specify --rally-id, --video-id, or --all"
        )
        raise typer.Exit(1)

    if grid not in AVAILABLE_GRIDS:
        console.print(
            f"[red]Error:[/red] Unknown grid '{grid}'. "
            f"Available: {', '.join(AVAILABLE_GRIDS.keys())}"
        )
        raise typer.Exit(1)

    # Initialize cache
    raw_cache = RawPositionCache()

    if clear_cache:
        count = raw_cache.clear()
        console.print(f"[yellow]Cleared {count} cached raw position files[/yellow]")

    # Load rallies from database
    console.print("[bold]Loading labeled rallies from database...[/bold]")
    rallies = load_labeled_rallies(
        video_id=video_id,
        rally_id=rally_id,
    )

    if not rallies:
        console.print("[yellow]No rallies with ground truth found[/yellow]")
        raise typer.Exit(1)

    console.print(f"  Found {len(rallies)} rally(s) with ground truth\n")

    from rallycut.labeling.ground_truth import GroundTruthResult

    # Check for raw positions in database and build cache
    cached_rallies: list[tuple[CachedRallyData, GroundTruthResult]] = []
    rallies_without_raw: list = []

    for rally in rallies:
        if rally.predictions is None:
            console.print(
                f"[yellow]Rally {rally.rally_id[:8]}... has no predictions, skipping[/yellow]"
            )
            continue

        # First check file cache
        cached = raw_cache.get(rally.rally_id)
        if cached:
            cached_rallies.append((cached, rally.ground_truth))
            continue

        # Check if raw positions are in database
        if rally.raw_positions:
            # Use raw positions from database
            cached_data = CachedRallyData(
                rally_id=rally.rally_id,
                video_id=rally.video_id,
                raw_positions=rally.raw_positions,
                ball_positions=rally.predictions.ball_positions or [],
                video_fps=rally.video_fps,
                frame_count=rally.predictions.frame_count,
                video_width=rally.video_width,
                video_height=rally.video_height,
                start_ms=rally.start_ms,
                end_ms=rally.end_ms,
            )
            raw_cache.put(cached_data)
            cached_rallies.append((cached_data, rally.ground_truth))
        else:
            # No raw positions - need to re-run tracking
            rallies_without_raw.append(rally)

    # Report status
    if rallies_without_raw:
        console.print(
            f"\n[yellow]Warning: {len(rallies_without_raw)} rally(s) have no raw positions.[/yellow]"
        )
        console.print(
            "[dim]Re-run player tracking to store raw positions for parameter tuning.[/dim]"
        )
        for rally in rallies_without_raw:
            console.print(f"  - {rally.rally_id[:8]}...")

    console.print(f"\n  Rallies with raw positions: {len(cached_rallies)}")

    if cache_only:
        stats = raw_cache.stats()
        console.print("\n[green]Raw positions cached successfully![/green]")
        console.print(f"  Cache location: {stats['cache_dir']}")
        console.print(f"  Cached rallies: {stats['count']}")
        console.print(f"  Total size: {stats['total_size_mb']:.2f} MB")
        return

    if not cached_rallies:
        console.print("[red]No rallies available for grid search[/red]")
        raise typer.Exit(1)

    # Run grid search
    param_grid = get_grid(grid)
    num_configs = grid_size(param_grid)

    console.print()
    console.print(f"[bold]Player Filter Grid Search - {num_configs} configs, {len(cached_rallies)} rallies[/bold]")
    console.print("=" * 60)
    console.print(f"Grid: {grid}")
    console.print(f"IoU threshold: {iou_threshold}")
    if min_rally_f1 is not None:
        console.print(f"Min rally F1 constraint: [yellow]{min_rally_f1:.0%}[/yellow]")

    # Show parameters being searched
    console.print("\n[dim]Parameters being searched:[/dim]")
    for param, values in param_grid.items():
        console.print(f"  {param}: {values}")

    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Searching...", total=num_configs)

        def update_progress(current: int, total: int) -> None:
            progress.update(task, completed=current)

        result = grid_search(
            rallies=cached_rallies,
            param_grid=param_grid,
            iou_threshold=iou_threshold,
            min_rally_f1=min_rally_f1,
            progress_callback=update_progress,
        )

    # Display results
    console.print()
    console.print("[bold]Best Configuration[/bold]")
    console.print("=" * 60)

    best = result.best_config
    for param in param_grid.keys():
        value = getattr(best, param)
        default_val = getattr(type(best)(), param)
        if value != default_val:
            console.print(f"  [cyan]{param}[/cyan]: {value} [dim](default: {default_val})[/dim]")
        else:
            console.print(f"  [dim]{param}: {value}[/dim]")

    console.print()
    console.print(
        f"Results: F1=[bold green]{result.best_f1:.1%}[/bold green], "
        f"MOTA={result.best_mota:.1%}, "
        f"ID Switches={result.best_id_switches}"
    )

    if result.improvement_f1 != 0:
        improvement_color = "green" if result.improvement_f1 > 0 else "red"
        console.print(
            f"Improvement over default: [{improvement_color}]{result.improvement_f1:+.1%}[/{improvement_color}] "
            f"(default F1={result.default_f1:.1%})"
        )

    if result.rejected_count > 0:
        console.print(
            f"\n[yellow]Rejected {result.rejected_count} configs that violated constraints[/yellow]"
        )

    # Top N configurations
    console.print()
    console.print(f"[bold]Top {min(top_n, len(result.all_results))} Configurations[/bold]")

    top_table = Table(show_header=True, header_style="bold")
    top_table.add_column("Rank", justify="right")
    top_table.add_column("F1", justify="right")
    top_table.add_column("MOTA", justify="right")
    top_table.add_column("ID Sw", justify="right")
    top_table.add_column("Changes from Default")

    for i, config_result in enumerate(result.all_results[:top_n]):
        if config_result.rejected:
            continue

        rank = i + 1
        metrics = config_result.aggregate_metrics
        diff = describe_config_diff(config_result.config)

        # Truncate diff if too long
        if len(diff) > 50:
            diff = diff[:47] + "..."

        f1_style = "green" if metrics.f1 >= 0.80 else ("yellow" if metrics.f1 >= 0.60 else "red")

        top_table.add_row(
            str(rank),
            f"[{f1_style}]{metrics.f1:.1%}[/{f1_style}]",
            f"{metrics.mota:.1%}",
            str(metrics.num_id_switches),
            diff,
        )

    console.print(top_table)

    # Export to file if requested
    if output:
        with open(output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        console.print(f"\n[green]Full results exported to {output}[/green]")


def _display_aggregate_metrics(metrics: MOTMetrics) -> None:
    """Display aggregate metrics table."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_column("Target", justify="right")
    table.add_column("Status")

    table.add_row(
        "MOTA",
        f"{metrics.mota:.2%}",
        ">80%",
        _status_icon(metrics.mota, 0.80),
    )
    table.add_row(
        "Precision",
        f"{metrics.precision:.2%}",
        ">85%",
        _status_icon(metrics.precision, 0.85),
    )
    table.add_row(
        "Recall",
        f"{metrics.recall:.2%}",
        ">80%",
        _status_icon(metrics.recall, 0.80),
    )
    table.add_row(
        "F1",
        f"{metrics.f1:.2%}",
        ">80%",
        _status_icon(metrics.f1, 0.80),
    )
    table.add_row(
        "ID Switches",
        str(metrics.num_id_switches),
        "<5",
        _status_icon(metrics.num_id_switches, 5, higher_better=False),
    )

    console.print("\n[bold]Aggregate Metrics[/bold]")
    console.print(table)

    # Detection breakdown
    console.print("\n[dim]Detection breakdown:[/dim]")
    console.print(f"  Ground truth: {metrics.num_gt} objects")
    console.print(f"  Predictions: {metrics.num_pred} objects")
    console.print(f"  Matches (TP): {metrics.num_matches}")
    console.print(f"  Misses (FN): {metrics.num_misses}")
    console.print(f"  False positives (FP): {metrics.num_false_positives}")


def _display_extended_metrics(result: TrackingEvaluationResult) -> None:
    """Display extended tracking metrics (HOTA, track quality, position accuracy)."""

    # HOTA Metrics
    if result.hota_metrics:
        hota = result.hota_metrics
        table = Table(show_header=True, header_style="bold")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_column("Target", justify="right")
        table.add_column("Status")

        table.add_row(
            "HOTA",
            f"{hota.hota:.2%}",
            ">70%",
            _status_icon(hota.hota, 0.70),
        )
        table.add_row(
            "DetA (Detection)",
            f"{hota.deta:.2%}",
            ">70%",
            _status_icon(hota.deta, 0.70),
        )
        table.add_row(
            "AssA (Association)",
            f"{hota.assa:.2%}",
            ">80%",
            _status_icon(hota.assa, 0.80),
        )
        table.add_row(
            "LocA (Localization)",
            f"{hota.loca:.2%}",
            "-",
            "",
        )

        console.print("\n[bold]HOTA Metrics[/bold] (Higher Order Tracking Accuracy)")
        console.print(table)

    # Track Quality Metrics
    if result.track_quality:
        tq = result.track_quality
        table = Table(show_header=True, header_style="bold")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_column("Target", justify="right")
        table.add_column("Status")

        table.add_row(
            "Mostly Tracked",
            f"{tq.mostly_tracked}/{tq.gt_track_count} ({tq.mostly_tracked_ratio:.0%})",
            ">80%",
            _status_icon(tq.mostly_tracked_ratio, 0.80),
        )
        table.add_row(
            "Partially Tracked",
            str(tq.partially_tracked),
            "-",
            "",
        )
        table.add_row(
            "Mostly Lost",
            str(tq.mostly_lost),
            "<1",
            _status_icon(tq.mostly_lost, 1, higher_better=False),
        )
        table.add_row(
            "Fragmentation",
            str(tq.fragmentation),
            "<10%",
            _status_icon(tq.fragmentation / max(1, tq.gt_track_count), 0.10, higher_better=False),
        )
        table.add_row(
            "Avg Track Coverage",
            f"{tq.avg_track_coverage:.1%}",
            ">90%",
            _status_icon(tq.avg_track_coverage, 0.90),
        )
        table.add_row(
            "Avg Pred IDs per GT",
            f"{tq.avg_pred_ids_per_gt:.2f}",
            "<1.2",
            _status_icon(tq.avg_pred_ids_per_gt, 1.2, higher_better=False),
        )

        console.print("\n[bold]Track Quality Metrics[/bold]")
        console.print(table)

    # Position Accuracy Metrics
    if result.position_metrics and result.position_metrics.num_position_samples > 0:
        pm = result.position_metrics
        table = Table(show_header=True, header_style="bold")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_column("Target", justify="right")
        table.add_column("Status")

        # Convert normalized error to percentage of frame
        table.add_row(
            "Mean Position Error",
            f"{pm.mean_position_error * 100:.2f}%",
            "<2%",
            _status_icon(pm.mean_position_error, 0.02, higher_better=False),
        )
        table.add_row(
            "Median Position Error",
            f"{pm.median_position_error * 100:.2f}%",
            "<1.5%",
            _status_icon(pm.median_position_error, 0.015, higher_better=False),
        )
        table.add_row(
            "P90 Position Error",
            f"{pm.p90_position_error * 100:.2f}%",
            "<5%",
            _status_icon(pm.p90_position_error, 0.05, higher_better=False),
        )

        console.print("\n[bold]Position Accuracy[/bold]")
        console.print(table)
        console.print(f"  [dim]Based on {pm.num_position_samples} matched pairs[/dim]")


def _display_aggregate_extended_metrics(results: list) -> None:
    """Display aggregated extended metrics across multiple rallies."""
    # Compute average HOTA metrics
    hota_results = [r for r in results if r.hota_metrics]
    if hota_results:
        avg_hota = sum(r.hota_metrics.hota for r in hota_results) / len(hota_results)
        avg_deta = sum(r.hota_metrics.deta for r in hota_results) / len(hota_results)
        avg_assa = sum(r.hota_metrics.assa for r in hota_results) / len(hota_results)

        console.print("\n[bold]Aggregate HOTA Metrics[/bold]")
        console.print(f"  HOTA: {avg_hota:.1%}")
        console.print(f"  DetA: {avg_deta:.1%}")
        console.print(f"  AssA: {avg_assa:.1%}")

    # Compute aggregate track quality metrics
    tq_results = [r for r in results if r.track_quality]
    if tq_results:
        total_mt = sum(r.track_quality.mostly_tracked for r in tq_results)
        total_gt = sum(r.track_quality.gt_track_count for r in tq_results)
        total_frag = sum(r.track_quality.fragmentation for r in tq_results)
        avg_coverage = sum(r.track_quality.avg_track_coverage for r in tq_results) / len(tq_results)
        avg_pred_ids = sum(r.track_quality.avg_pred_ids_per_gt for r in tq_results) / len(tq_results)

        mt_ratio = total_mt / total_gt if total_gt > 0 else 0.0
        frag_ratio = total_frag / total_gt if total_gt > 0 else 0.0

        console.print("\n[bold]Aggregate Track Quality[/bold]")
        console.print(f"  Mostly Tracked: {total_mt}/{total_gt} ({mt_ratio:.0%})")
        console.print(f"  Fragmentation: {total_frag} ({frag_ratio:.0%} of tracks)")
        console.print(f"  Avg Track Coverage: {avg_coverage:.1%}")
        console.print(f"  Avg Pred IDs per GT: {avg_pred_ids:.2f}")

    # Compute aggregate position metrics
    pm_results = [r for r in results if r.position_metrics and r.position_metrics.num_position_samples > 0]
    if pm_results:
        total_samples = sum(r.position_metrics.num_position_samples for r in pm_results)
        # Weighted average by sample count
        weighted_mean = sum(
            r.position_metrics.mean_position_error * r.position_metrics.num_position_samples
            for r in pm_results
        ) / total_samples

        console.print("\n[bold]Aggregate Position Accuracy[/bold]")
        console.print(f"  Mean Position Error: {weighted_mean * 100:.2f}%")
        console.print(f"  [dim]Based on {total_samples} matched pairs across {len(pm_results)} rallies[/dim]")

    # Compute aggregate identity metrics
    id_results = [r for r in results if r.identity_metrics]
    if id_results:
        total_switches = sum(r.identity_metrics.num_switches for r in id_results)
        total_err = sum(r.identity_metrics.num_error_frames for r in id_results)
        total_frames = sum(r.identity_metrics.num_total_frames for r in id_results)
        accuracy = 1.0 - (total_err / total_frames) if total_frames > 0 else 1.0

        total_ambiguous = sum(
            r.identity_metrics.num_ambiguous_switches for r in id_results
        )
        console.print("\n[bold]Aggregate Identity[/bold]")
        ambig_str = f" (+{total_ambiguous} ambiguous)" if total_ambiguous else ""
        console.print(f"  Real Identity Switches: {total_switches}{ambig_str}")
        console.print(f"  Identity Accuracy: {accuracy:.1%} ({total_err} error frames / {total_frames})")


def _display_ball_metrics(ball_metrics: BallMetrics) -> None:
    """Display ball tracking metrics (compact version for combined eval)."""
    from rallycut.evaluation.tracking.ball_metrics import BallTrackingMetrics

    # Handle legacy BallMetrics from compare_tracking
    if not isinstance(ball_metrics, BallTrackingMetrics):
        # Legacy format - just show basic metrics
        table = Table(show_header=True, header_style="bold")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_column("Target", justify="right")
        table.add_column("Status")

        table.add_row(
            "Detection Rate",
            f"{ball_metrics.detection_rate:.2%}",
            ">60%",
            _status_icon(ball_metrics.detection_rate, 0.60),
        )
        table.add_row(
            "Mean Error",
            f"{ball_metrics.mean_error_px:.1f} px",
            "<20px",
            _status_icon(ball_metrics.mean_error_px, 20, higher_better=False),
        )

        console.print("\n[bold]Ball Tracking Metrics[/bold]")
        console.print(table)
        return

    # Enhanced metrics
    _display_enhanced_ball_metrics(ball_metrics)


def _display_enhanced_ball_metrics(metrics: BallTrackingMetrics) -> None:
    """Display enhanced ball tracking metrics with detailed statistics."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_column("Target", justify="right")
    table.add_column("Status")

    # Detection metrics
    table.add_row(
        "Detection Rate",
        f"{metrics.detection_rate:.1%}",
        ">60%",
        _status_icon(metrics.detection_rate, 0.60),
    )
    table.add_row(
        "Match Rate (<50px)",
        f"{metrics.match_rate:.1%}",
        ">50%",
        _status_icon(metrics.match_rate, 0.50),
    )

    # Error metrics
    table.add_row(
        "Mean Error",
        f"{metrics.mean_error_px:.1f} px",
        "<20px",
        _status_icon(metrics.mean_error_px, 20, higher_better=False),
    )
    table.add_row(
        "Median Error",
        f"{metrics.median_error_px:.1f} px",
        "<15px",
        _status_icon(metrics.median_error_px, 15, higher_better=False),
    )
    table.add_row(
        "P90 Error",
        f"{metrics.p90_error_px:.1f} px",
        "<50px",
        _status_icon(metrics.p90_error_px, 50, higher_better=False),
    )
    table.add_row(
        "Max Error",
        f"{metrics.max_error_px:.1f} px",
        "-",
        "",
    )

    # Accuracy buckets
    table.add_row(
        "Accuracy <20px",
        f"{metrics.error_under_20px_rate:.1%}",
        ">60%",
        _status_icon(metrics.error_under_20px_rate, 0.60),
    )
    table.add_row(
        "Accuracy <50px",
        f"{metrics.error_under_50px_rate:.1%}",
        ">80%",
        _status_icon(metrics.error_under_50px_rate, 0.80),
    )

    console.print("\n[bold]Ball Tracking Metrics[/bold]")
    console.print(table)

    # Detection breakdown
    console.print("\n[dim]Detection breakdown:[/dim]")
    console.print(f"  Ground truth frames: {metrics.num_gt_frames}")
    console.print(f"  Detected frames: {metrics.num_detected}")
    console.print(f"  Matched frames (<50px): {metrics.num_matched}")
    miss_count = metrics.num_gt_frames - metrics.num_detected
    console.print(f"  Missed frames: {miss_count}")


def _display_per_player_metrics(per_player: list[PerPlayerMetrics]) -> None:
    """Display per-player metrics table."""
    table = Table(show_header=True, header_style="bold")
    table.add_column("Player")
    table.add_column("Recall", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("F1", justify="right")
    table.add_column("ID Swaps", justify="right")
    table.add_column("GT", justify="right")
    table.add_column("Matches", justify="right")

    # Find worst player for highlighting
    worst = min(per_player, key=lambda p: p.f1) if per_player else None

    for player in per_player:
        is_worst = player == worst
        suffix = " [yellow]<- worst[/yellow]" if is_worst else ""

        table.add_row(
            player.label + suffix,
            f"{player.recall:.1%}",
            f"{player.precision:.1%}",
            f"{player.f1:.1%}",
            str(player.id_switches),
            str(player.gt_count),
            str(player.matches),
        )

    console.print("\n[bold]Per-Player Breakdown[/bold]")
    console.print(table)


def _display_error_analysis(errors: list[ErrorEvent]) -> None:
    """Display error analysis summary."""
    from rallycut.evaluation.tracking.error_analysis import ErrorType, summarize_errors

    summary = summarize_errors(errors)

    console.print("\n[bold]Error Analysis[/bold]")
    console.print(f"  Total errors: {summary.total_errors}")

    # By type
    console.print("\n  [dim]By type:[/dim]")
    for error_type in ErrorType:
        count = summary.by_type.get(error_type, 0)
        if count > 0:
            console.print(f"    {error_type.value}: {count}")

    # By player (top 5)
    if summary.by_player:
        console.print("\n  [dim]By player (most errors):[/dim]")
        sorted_players = sorted(
            summary.by_player.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]
        for label, count in sorted_players:
            console.print(f"    {label}: {count}")

    # Consecutive error frames
    if summary.consecutive_error_frames > 1:
        console.print(
            f"\n  [yellow]Max consecutive error frames: "
            f"{summary.consecutive_error_frames}[/yellow]"
        )


@app.command(name="tune-ball-filter")
@handle_errors
def tune_ball_filter(
    rally_id: Annotated[
        str | None,
        typer.Option(
            "--rally-id", "-r",
            help="Tune on specific rally by ID",
        ),
    ] = None,
    video_id: Annotated[
        str | None,
        typer.Option(
            "--video-id", "-v",
            help="Tune on all labeled rallies in a video",
        ),
    ] = None,
    all_rallies: Annotated[
        bool,
        typer.Option(
            "--all", "-a",
            help="Tune on all labeled rallies in database",
        ),
    ] = False,
    grid: Annotated[
        str,
        typer.Option(
            "--grid", "-g",
            help="Grid to search: segment-pruning, oscillation, outlier, wasb",
        ),
    ] = "segment-pruning",
    match_threshold: Annotated[
        float,
        typer.Option(
            "--match-threshold", "-t",
            help="Max distance (pixels) for a match",
        ),
    ] = 50.0,
    min_rally_detection: Annotated[
        float | None,
        typer.Option(
            "--min-rally-detection",
            help="Reject configs where any rally drops below this detection rate",
        ),
    ] = None,
    top_n: Annotated[
        int,
        typer.Option(
            "--top", "-n",
            help="Number of top results to show",
        ),
    ] = 10,
    cache_only: Annotated[
        bool,
        typer.Option(
            "--cache-only",
            help="Only cache raw positions, don't run grid search",
        ),
    ] = False,
    clear_cache: Annotated[
        bool,
        typer.Option(
            "--clear-cache",
            help="Clear raw position cache before starting",
        ),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o",
            help="Export full results to JSON file",
        ),
    ] = None,
) -> None:
    """Grid search for optimal BallFilterConfig parameters.

    Searches over ball filter parameters to find the configuration that
    maximizes match rate while minimizing position error.

    Caches raw (unfiltered) ball positions and re-runs the filter pipeline
    with different configs.

    Examples:

        # Cache raw ball positions first (requires raw_positions in predictions)
        rallycut evaluate-tracking tune-ball-filter --all --cache-only

        # Segment pruning grid (18 combinations)
        rallycut evaluate-tracking tune-ball-filter --all --grid segment-pruning

        # Oscillation pruning grid (18 combinations)
        rallycut evaluate-tracking tune-ball-filter --all --grid oscillation

        # WASB grid (1152 combinations)
        rallycut evaluate-tracking tune-ball-filter --all --grid wasb

        # Export results
        rallycut evaluate-tracking tune-ball-filter --all -o results.json
    """
    from rallycut.evaluation.tracking.ball_grid_search import (
        BallRawCache,
        CachedBallData,
        ball_grid_search,
    )
    from rallycut.evaluation.tracking.ball_param_grid import (
        BALL_AVAILABLE_GRIDS,
        ball_grid_size,
        describe_ball_config_diff,
        get_ball_grid,
    )
    from rallycut.evaluation.tracking.db import load_labeled_rallies

    # Validate input
    if not rally_id and not video_id and not all_rallies:
        console.print(
            "[red]Error:[/red] Specify --rally-id, --video-id, or --all"
        )
        raise typer.Exit(1)

    if grid not in BALL_AVAILABLE_GRIDS:
        console.print(
            f"[red]Error:[/red] Unknown grid '{grid}'. "
            f"Available: {', '.join(BALL_AVAILABLE_GRIDS.keys())}"
        )
        raise typer.Exit(1)

    # Initialize cache
    ball_cache = BallRawCache()

    if clear_cache:
        count = ball_cache.clear()
        console.print(f"[yellow]Cleared {count} cached ball position files[/yellow]")

    # Load rallies from database
    console.print("[bold]Loading labeled rallies from database...[/bold]")
    rallies = load_labeled_rallies(
        video_id=video_id,
        rally_id=rally_id,
    )

    if not rallies:
        console.print("[yellow]No rallies with ball ground truth found[/yellow]")
        raise typer.Exit(1)

    console.print(f"  Found {len(rallies)} rally(s) with ball ground truth\n")

    from rallycut.labeling.ground_truth import GroundTruthPosition

    # Check for raw ball positions and build cache
    # NOTE: Raw ball positions (before Kalman filtering) are not stored in the database.
    # They must be cached via --cache-only after re-running ball tracking with preserve_raw=True.
    # For now, we can use the filtered positions as a baseline - they'll still allow testing
    # different Kalman filter parameters, though the starting point is already filtered.
    cached_rallies: list[tuple[CachedBallData, list[GroundTruthPosition]]] = []
    rallies_using_filtered: list = []

    for rally in rallies:
        if rally.predictions is None or not rally.predictions.ball_positions:
            console.print(
                f"[yellow]Rally {rally.rally_id[:8]}... has no ball predictions, skipping[/yellow]"
            )
            continue

        # First check file cache
        cached = ball_cache.get(rally.rally_id)
        if cached:
            # Filter GT to ball positions
            gt_ball = [p for p in rally.ground_truth.positions if p.label == "ball"]
            cached_rallies.append((cached, gt_ball))
            continue

        # No cached raw positions - use filtered positions as fallback
        # This still allows testing Kalman filter parameters, though results may differ
        # from running on truly raw (unfiltered) positions
        cached_data = CachedBallData(
            rally_id=rally.rally_id,
            video_id=rally.video_id,
            raw_ball_positions=rally.predictions.ball_positions,  # Using filtered as fallback
            video_fps=rally.video_fps,
            frame_count=rally.predictions.frame_count,
            video_width=rally.video_width,
            video_height=rally.video_height,
            start_ms=rally.start_ms,
            end_ms=rally.end_ms,
        )
        ball_cache.put(cached_data)
        gt_ball = [p for p in rally.ground_truth.positions if p.label == "ball"]
        cached_rallies.append((cached_data, gt_ball))
        rallies_using_filtered.append(rally)

    # Report status
    if rallies_using_filtered:
        console.print(
            f"\n[dim]Note: {len(rallies_using_filtered)} rally(s) using already-filtered positions.[/dim]"
        )
        console.print(
            "[dim]For more accurate results, re-run ball tracking with preserve_raw=True.[/dim]"
        )

    console.print(f"\n  Rallies available for grid search: {len(cached_rallies)}")

    if cache_only:
        stats = ball_cache.stats()
        console.print("\n[green]Raw ball positions cached successfully![/green]")
        console.print(f"  Cache location: {stats['cache_dir']}")
        console.print(f"  Cached rallies: {stats['count']}")
        console.print(f"  Total size: {stats['total_size_mb']:.2f} MB")
        return

    if not cached_rallies:
        console.print("[red]No rallies available for grid search[/red]")
        raise typer.Exit(1)

    # Run grid search
    param_grid = get_ball_grid(grid)
    num_configs = ball_grid_size(param_grid)

    console.print()
    console.print(f"[bold]Ball Filter Grid Search - {num_configs} configs, {len(cached_rallies)} rallies[/bold]")
    console.print("=" * 60)
    console.print(f"Grid: {grid}")
    console.print(f"Match threshold: {match_threshold:.0f}px")
    if min_rally_detection is not None:
        console.print(f"Min rally detection constraint: [yellow]{min_rally_detection:.0%}[/yellow]")

    # Show parameters being searched
    console.print("\n[dim]Parameters being searched:[/dim]")
    for param, values in param_grid.items():
        console.print(f"  {param}: {values}")

    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Searching...", total=num_configs)

        def update_progress(current: int, total: int) -> None:
            progress.update(task, completed=current)

        result = ball_grid_search(
            rallies=cached_rallies,
            param_grid=param_grid,
            match_threshold_px=match_threshold,
            min_rally_detection_rate=min_rally_detection,
            progress_callback=update_progress,
        )

    # Display results
    console.print()
    console.print("[bold]Best Configuration[/bold]")
    console.print("=" * 60)

    best = result.best_config
    for param in param_grid.keys():
        value = getattr(best, param)
        default_val = getattr(type(best)(), param)
        if value != default_val:
            console.print(f"  [cyan]{param}[/cyan]: {value} [dim](default: {default_val})[/dim]")
        else:
            console.print(f"  [dim]{param}: {value}[/dim]")

    console.print()
    console.print(
        f"Results: Detection=[bold green]{result.best_detection_rate:.1%}[/bold green], "
        f"Match={result.best_match_rate:.1%}, "
        f"Error={result.best_mean_error_px:.1f}px"
    )

    if result.improvement_match_rate != 0:
        improvement_color = "green" if result.improvement_match_rate > 0 else "red"
        console.print(
            f"Improvement over default: [{improvement_color}]{result.improvement_match_rate:+.1%}[/{improvement_color}] "
            f"(default match={result.default_match_rate:.1%})"
        )

    if result.rejected_count > 0:
        console.print(
            f"\n[yellow]Rejected {result.rejected_count} configs that violated constraints[/yellow]"
        )

    # Top N configurations
    console.print()
    console.print(f"[bold]Top {min(top_n, len(result.all_results))} Configurations[/bold]")

    top_table = Table(show_header=True, header_style="bold")
    top_table.add_column("Rank", justify="right")
    top_table.add_column("Detection", justify="right")
    top_table.add_column("Match", justify="right")
    top_table.add_column("Error", justify="right")
    top_table.add_column("<20px", justify="right")
    top_table.add_column("Changes from Default")

    shown = 0
    for i, config_result in enumerate(result.all_results):
        if config_result.rejected:
            continue
        if shown >= top_n:
            break

        rank = shown + 1
        metrics = config_result.aggregate_metrics
        diff = describe_ball_config_diff(config_result.config)

        # Truncate diff if too long
        if len(diff) > 40:
            diff = diff[:37] + "..."

        match_style = "green" if metrics.match_rate >= 0.70 else ("yellow" if metrics.match_rate >= 0.50 else "red")

        top_table.add_row(
            str(rank),
            f"{metrics.detection_rate:.1%}",
            f"[{match_style}]{metrics.match_rate:.1%}[/{match_style}]",
            f"{metrics.mean_error_px:.1f}px",
            f"{metrics.error_under_20px_rate:.1%}",
            diff,
        )
        shown += 1

    console.print(top_table)

    # Export to file if requested
    if output:
        with open(output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        console.print(f"\n[green]Full results exported to {output}[/green]")


@app.command(name="compare-yolo-models")
@handle_errors
def compare_yolo_models(
    rally_id: Annotated[
        str | None,
        typer.Option(
            "--rally-id", "-r",
            help="Evaluate specific rally by ID",
        ),
    ] = None,
    video_id: Annotated[
        str | None,
        typer.Option(
            "--video-id", "-v",
            help="Evaluate all labeled rallies in a video",
        ),
    ] = None,
    all_rallies: Annotated[
        bool,
        typer.Option(
            "--all", "-a",
            help="Evaluate all labeled rallies in database",
        ),
    ] = False,
    output: Annotated[
        Path | None,
        typer.Option(
            "--output", "-o",
            help="Output JSON file for full results",
        ),
    ] = None,
    models: Annotated[
        str | None,
        typer.Option(
            "--models", "-m",
            help="Comma-separated list of models to test (default: yolov8n,yolo11s,yolo11m)",
        ),
    ] = None,
) -> None:
    """Compare YOLO model sizes (n/s/m/l) for player detection.

    Runs tracking with different model sizes and compares detection/tracking
    accuracy. Useful for determining if larger models improve far-side player
    recall.

    Note: This caches nothing and runs fresh tracking for each model, so it
    will take time proportional to (models × rallies × rally_duration).

    Example:
        # Compare all models on a single rally
        rallycut evaluate-tracking compare-yolo-models -r <rally-id>

        # Compare on all labeled rallies
        rallycut evaluate-tracking compare-yolo-models --all

        # Test specific models only
        rallycut evaluate-tracking compare-yolo-models --all -m yolov8s,yolov8m
    """
    from rallycut.evaluation.tracking.db import get_video_path, load_labeled_rallies
    from rallycut.evaluation.tracking.metrics import aggregate_results, evaluate_rally
    from rallycut.tracking.player_filter import PlayerFilterConfig
    from rallycut.tracking.player_tracker import (
        DEFAULT_TRACKER,
        YOLO_MODELS,
        PlayerTracker,
    )

    # Validate input
    if not rally_id and not video_id and not all_rallies:
        console.print(
            "[red]Error:[/red] Specify --rally-id, --video-id, or --all"
        )
        raise typer.Exit(1)

    # Parse models to test
    if models:
        model_ids = [m.strip() for m in models.split(",")]
        for m in model_ids:
            if m not in YOLO_MODELS:
                console.print(f"[red]Error:[/red] Unknown model '{m}'. Available: {', '.join(YOLO_MODELS.keys())}")
                raise typer.Exit(1)
    else:
        # Default: test nano, small (default), medium
        model_ids = ["yolov8n", "yolo11s", "yolo11m"]

    # Load rallies from database
    console.print("[bold]Loading labeled rallies from database...[/bold]")
    rallies = load_labeled_rallies(
        video_id=video_id,
        rally_id=rally_id,
    )

    if not rallies:
        console.print("[yellow]No labeled rallies found[/yellow]")
        raise typer.Exit(1)

    console.print(f"Found [cyan]{len(rallies)}[/cyan] labeled rallies")
    console.print(f"Testing models: [cyan]{', '.join(model_ids)}[/cyan]")

    # Results storage: model_id -> list of (rally_id, result)
    model_results: dict[str, list[tuple[str, TrackingEvaluationResult]]] = {m: [] for m in model_ids}

    # Group rallies by video for efficiency
    rallies_by_video: dict[str, list[Any]] = {}
    for rally in rallies:
        vid = rally.video_id
        if vid not in rallies_by_video:
            rallies_by_video[vid] = []
        rallies_by_video[vid].append(rally)

    total_evaluations = len(model_ids) * len(rallies)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("Comparing YOLO models...", total=total_evaluations)

        for model_id in model_ids:
            progress.update(task, description=f"Testing {model_id}...")
            filter_config = PlayerFilterConfig()

            for video_id_key, video_rallies in rallies_by_video.items():
                video_path = get_video_path(video_id_key)
                if not video_path or not video_path.exists():
                    for _ in video_rallies:
                        progress.update(task, advance=1)
                    continue

                # Create fresh tracker for each video to avoid state leakage
                # (ByteTrack/BoT-SORT maintains internal state that doesn't reset)
                tracker = PlayerTracker(
                    yolo_model=model_id,
                    tracker=DEFAULT_TRACKER,  # Use BoT-SORT for all
                )

                for rally in video_rallies:
                    try:
                        # Run tracking
                        result = tracker.track_video(
                            video_path,
                            start_ms=rally.start_ms,
                            end_ms=rally.end_ms,
                            filter_enabled=True,
                            filter_config=filter_config,
                        )

                        # track_video() already normalizes frame numbers to
                        # 0-indexed rally-relative. No further adjustment needed.

                        # Evaluate against ground truth
                        eval_result = evaluate_rally(
                            rally_id=rally.rally_id,
                            ground_truth=rally.ground_truth,
                            predictions=result,
                        )

                        model_results[model_id].append((rally.rally_id, eval_result))

                    except Exception as e:
                        console.print(f"[yellow]Warning: {model_id} failed on {rally.rally_id}: {e}[/yellow]")

                    progress.update(task, advance=1)

    # Display comparison table
    console.print("\n[bold]YOLO Model Comparison Results[/bold]\n")

    comparison_table = Table(title="Model Comparison")
    comparison_table.add_column("Model", style="cyan")
    comparison_table.add_column("MOTA", justify="right")
    comparison_table.add_column("Precision", justify="right")
    comparison_table.add_column("Recall", justify="right")
    comparison_table.add_column("F1", justify="right")
    comparison_table.add_column("ID Sw", justify="right")

    best_f1 = 0.0
    best_model = ""

    for model_id in model_ids:
        results = model_results.get(model_id, [])
        if not results:
            comparison_table.add_row(model_id, "-", "-", "-", "-", "-")
            continue

        # Aggregate results
        eval_results = [r for _, r in results]
        combined = aggregate_results(eval_results)

        # Track best
        if combined.f1 > best_f1:
            best_f1 = combined.f1
            best_model = model_id

        # Style based on scores
        f1_style = "green" if combined.f1 >= 0.90 else ("yellow" if combined.f1 >= 0.80 else "red")
        recall_style = "green" if combined.recall >= 0.90 else ("yellow" if combined.recall >= 0.80 else "red")

        comparison_table.add_row(
            model_id,
            f"{combined.mota:.1%}",
            f"{combined.precision:.1%}",
            f"[{recall_style}]{combined.recall:.1%}[/{recall_style}]",
            f"[{f1_style}]{combined.f1:.1%}[/{f1_style}]",
            str(combined.num_id_switches),
        )

    console.print(comparison_table)

    if best_model:
        console.print(f"\n[green]Best model: {best_model}[/green] (highest F1: {best_f1:.1%})")

    # Export to file if requested
    if output:
        output_data: dict[str, Any] = {
            "rallies": len(rallies),
            "models": {},
        }

        for model_id in model_ids:
            results = model_results.get(model_id, [])
            if not results:
                continue

            eval_results = [r for _, r in results]
            combined = aggregate_results(eval_results)

            output_data["models"][model_id] = {
                "aggregate": {
                    "mota": combined.mota,
                    "precision": combined.precision,
                    "recall": combined.recall,
                    "f1": combined.f1,
                    "num_id_switches": combined.num_id_switches,
                },
                "per_rally": [
                    {
                        "rally_id": rid,
                        "mota": r.aggregate.mota,
                        "precision": r.aggregate.precision,
                        "recall": r.aggregate.recall,
                        "f1": r.aggregate.f1,
                        "num_id_switches": r.aggregate.num_id_switches,
                    }
                    for rid, r in results
                ],
            }

        with open(output, "w") as f:
            json.dump(output_data, f, indent=2)
        console.print(f"\n[green]Full results exported to {output}[/green]")


@app.command(name="audit")
@handle_errors
def audit(
    rally_id: Annotated[
        str | None,
        typer.Option("--rally-id", "-r", help="Audit a specific rally by ID"),
    ] = None,
    video_id: Annotated[
        str | None,
        typer.Option("--video-id", "-v", help="Audit all labeled rallies in a video"),
    ] = None,
    all_rallies: Annotated[
        bool,
        typer.Option("--all", "-a", help="Audit every rally with player GT"),
    ] = False,
    output_dir: Annotated[
        Path,
        typer.Option(
            "--output-dir", "-o",
            help="Directory to write per-rally JSON + _summary.json",
        ),
    ] = Path("reports/tracking_audit"),
    iou_threshold: Annotated[
        float,
        typer.Option("--iou", "-i", help="IoU threshold for matching"),
    ] = 0.5,
) -> None:
    """Dump a per-GT-track audit JSON per rally FROM THE DATABASE (see warning).

    WARNING — `player_tracks.positions_json` has been manually patched in the
    live DB to fix individual problematic tracks, so counts from this path are
    systematically inflated or deflated vs. real pipeline output. For a faithful
    audit, use the retrack pathway instead:

        rallycut evaluate-tracking --all --retrack --cached \\
            --audit-out reports/tracking_audit
        uv run python scripts/write_tracking_audit_report.py

    This subcommand is kept only for DB-side forensics.

    Classifies missed frames (out-of-frame / edge / occlusion / filter-drop /
    detector-miss), lists real ID switches with cause (net-crossing / same-team
    / cross-team), reports fragmentation (distinct pred IDs per GT track), and
    flags convention drift (GT label ↔ pred track side/team mismatch).

    Output: <output-dir>/<rally_id>.json + <output-dir>/_summary.json.
    """
    import json as _json

    from rallycut.evaluation.tracking.audit import build_rally_audit
    from rallycut.evaluation.tracking.db import load_labeled_rallies

    if not rally_id and not video_id and not all_rallies:
        console.print("[red]Error:[/red] Specify --rally-id, --video-id, or --all")
        raise typer.Exit(1)

    console.print(
        "[yellow]WARNING:[/yellow] reading predictions from DB — "
        "`player_tracks.positions_json` has been manually patched over time, "
        "so counts here do NOT reflect true pipeline output. Prefer "
        "`--retrack --cached --audit-out ...` on the parent command.\n"
    )
    console.print("[bold]Loading labeled rallies from database...[/bold]")
    rallies = load_labeled_rallies(video_id=video_id, rally_id=rally_id)
    if not rallies:
        console.print("[yellow]No labeled rallies found[/yellow]")
        raise typer.Exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)
    console.print(f"  Found {len(rallies)} rally(s). Writing to [cyan]{output_dir}[/cyan]\n")

    summary: list[dict[str, Any]] = []
    for idx, rally in enumerate(rallies, start=1):
        if rally.predictions is None:
            console.print(
                f"[{idx}/{len(rallies)}] {rally.rally_id}: [yellow]no predictions in DB — skipped[/yellow]"
            )
            continue

        rally_audit = build_rally_audit(
            rally_id=rally.rally_id,
            video_id=rally.video_id,
            ground_truth=rally.ground_truth,
            predictions=rally.predictions,
            raw_positions=rally.raw_positions,
            iou_threshold=iou_threshold,
        )

        # Per-rally JSON
        rally_path = output_dir / f"{rally.rally_id}.json"
        with open(rally_path, "w") as f:
            _json.dump(rally_audit.to_dict(), f, indent=2)

        # Progress line (CLAUDE.md §Running Diagnostics rule #3)
        missed_ranges_total = sum(
            sum(len(ranges) for ranges in g.missed_by_cause.values())
            for g in rally_audit.per_gt
        )
        worst_coverage = (
            min((g.coverage for g in rally_audit.per_gt), default=1.0)
        )
        console.print(
            f"[{idx}/{len(rallies)}] {rally.rally_id}: "
            f"HOTA={rally_audit.hota or 0:.3f} "
            f"MOTA={rally_audit.mota:.3f} "
            f"realIDsw={rally_audit.aggregate_real_switches} "
            f"missedRanges={missed_ranges_total} "
            f"worstCov={worst_coverage:.2f} "
            f"convFlip={'Y' if rally_audit.convention.team_label_flip else 'n'}"
        )

        summary.append({
            "rallyId": rally.rally_id,
            "videoId": rally.video_id,
            "hota": rally_audit.hota,
            "mota": rally_audit.mota,
            "realSwitches": rally_audit.aggregate_real_switches,
            "missedRanges": missed_ranges_total,
            "worstCoverage": worst_coverage,
            "courtSideFlip": rally_audit.convention.court_side_flip,
            "teamLabelFlip": rally_audit.convention.team_label_flip,
            "perGt": [
                {
                    "gtTrackId": g.gt_track_id,
                    "coverage": g.coverage,
                    "distinctPredIds": g.distinct_pred_ids,
                    "realSwitchCount": len(g.real_switches),
                    "missCauseCounts": {
                        c.value: sum(end - start + 1 for start, end in ranges)
                        for c, ranges in g.missed_by_cause.items()
                    },
                }
                for g in rally_audit.per_gt
            ],
        })

    summary_path = output_dir / "_summary.json"
    with open(summary_path, "w") as f:
        _json.dump({"rallies": summary}, f, indent=2)
    console.print(f"\n[green]Audit summary written to {summary_path}[/green]")


