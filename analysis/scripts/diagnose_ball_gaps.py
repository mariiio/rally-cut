"""Ball-tracking gap diagnostic: per-rally overlay MP4 + failure budget JSON.

Draws GT, filtered predictions, raw WASB detections, filter-stage removals,
teleport arrows, wrong-object markers, and per-frame status onto the source
video so the user can visually inspect every gap against ground truth. Emits
failure_budget.json + events.json next to the MP4 so the HTML dashboard
(analysis/scripts/build_ball_failure_report.py) can aggregate across rallies.

Usage:
    uv run python scripts/diagnose_ball_gaps.py --rally <rally-id>
    uv run python scripts/diagnose_ball_gaps.py --all
    uv run python scripts/diagnose_ball_gaps.py --all --no-video   # fast: JSON only

Output layout (under --output-dir, default analysis/outputs/ball_gap_report):
    {rally_id}/overlay.mp4
    {rally_id}/events.json
    {rally_id}/failure_budget.json
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rallycut.core.video import Video
from rallycut.evaluation.tracking.ball_failure_modes import (
    FailureEvents,
    TeleportEvent,
    WrongObjectEvent,
    classify_rally,
)
from rallycut.evaluation.tracking.ball_grid_search import BallRawCache
from rallycut.evaluation.tracking.ball_metrics import find_optimal_frame_offset
from rallycut.evaluation.tracking.db import (
    TrackingEvaluationRally,
    get_rally_info,
    load_labeled_rallies,
)
from rallycut.evaluation.video_resolver import VideoResolver
from rallycut.labeling.ground_truth import GroundTruthPosition, GroundTruthResult
from rallycut.tracking.ball_filter import get_wasb_filter_config
from rallycut.tracking.ball_tracker import BallPosition

# Reuse the stage walker from the existing text diagnostic.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from diagnose_ball_tracking import run_pipeline_stages  # noqa: E402

logger = logging.getLogger(__name__)

# BGR colors (OpenCV)
GT_COLOR = (64, 220, 64)            # bright green
PRED_COLOR = (64, 64, 240)          # red
RAW_COLOR = (200, 130, 60)          # faint blue-ish
FILTER_KILL_COLOR = (0, 140, 255)   # orange
TELEPORT_COLOR = (0, 0, 255)        # saturated red
WRONG_OBJ_COLOR = (0, 230, 230)     # yellow
STATIONARY_COLOR = (180, 60, 200)   # purple
REVISIT_COLOR = (210, 120, 255)     # light purple (always-shown hot-spot)
TRAIL_GT = (64, 180, 64)
TRAIL_PRED = (64, 64, 200)
BANNER_BG = (32, 32, 32)
BANNER_FG = (240, 240, 240)

TRAIL_FRAMES = 30
STAGE_ORDER = [
    "0_raw",
    "1_segment_pruned",
    "2_ghost_removed",
    "3_oscillation_pruned",
    "4_outlier_removed",
    "5_blip_removed",
    "6_repruned",
    "7_interpolated",
]


@dataclass
class RallyRenderContext:
    rally_id: str
    video_path: Path
    video_width: int
    video_height: int
    video_fps: float
    start_frame: int        # video-absolute frame of rally start
    end_frame: int          # video-absolute frame of rally end (inclusive)
    gt_by_frame: dict[int, GroundTruthPosition]  # rally-relative frame → GT
    pred_by_frame: dict[int, BallPosition]       # rally-relative (GT-space) → filtered pred
    raw_by_frame: dict[int, BallPosition]        # rally-relative (GT-space) → raw WASB
    interpolated_frames: set[int]
    filter_kills_by_frame: dict[int, list[tuple[tuple[float, float], str]]]
    teleports_by_frame: dict[int, TeleportEvent]
    wrong_by_frame: dict[int, WrongObjectEvent]
    stationary_frames: set[int]
    stationary_median: tuple[float, float] | None
    revisit_frames: set[int]
    revisit_centers: list[tuple[float, float]]
    per_frame_status: dict[int, str]
    frame_offset: int
    events: FailureEvents


def _shift_to_gt(predictions: list[BallPosition], offset: int) -> dict[int, BallPosition]:
    out: dict[int, BallPosition] = {}
    for p in predictions:
        shifted = p.frame_number - offset
        existing = out.get(shifted)
        if existing is None or p.confidence > existing.confidence:
            out[shifted] = p
    return out


def _interpolated_frame_set(predictions: list[BallPosition], offset: int) -> set[int]:
    """Positions written by the filter's interpolator carry confidence=0.5."""
    return {
        p.frame_number - offset
        for p in predictions
        if abs(p.confidence - 0.5) < 1e-6
    }


def _resolve_video_path(rally_id: str) -> tuple[Path, int, int] | None:
    info = get_rally_info(rally_id)
    if info is None:
        logger.warning(f"Rally {rally_id}: no RallyVideoInfo")
        return None
    resolver = VideoResolver()
    try:
        path = resolver.resolve(info.s3_key, info.content_hash)
    except Exception as e:  # pragma: no cover — network errors surface here
        logger.error(f"Rally {rally_id}: video resolve failed: {e}")
        return None
    return path, info.start_ms, info.end_ms


def _refilter_with_current_config(
    raw_positions: list[BallPosition],
) -> list[BallPosition]:
    """Apply the current get_wasb_filter_config() to cached raw positions."""
    from rallycut.tracking.ball_filter import BallTemporalFilter

    if not raw_positions:
        return []
    return BallTemporalFilter(get_wasb_filter_config()).filter_batch(raw_positions)


def _build_context(
    rally: TrackingEvaluationRally,
    raw_positions: list[BallPosition] | None,
    *,
    prediction_source: str = "db",
    teleport_v_max_px_per_frame: float = 120.0,
    wrong_object_threshold_px: float = 100.0,
    eval_keyframes_only: bool = False,
) -> RallyRenderContext | None:
    resolved = _resolve_video_path(rally.rally_id)
    if resolved is None:
        return None
    video_path, start_ms, end_ms = resolved

    if prediction_source == "refilter":
        if not raw_positions:
            logger.warning(
                f"Rally {rally.rally_id[:8]}: refilter requested but no raw cache"
            )
            return None
        predictions = _refilter_with_current_config(raw_positions)
    else:
        predictions = rally.predictions.ball_positions if rally.predictions else []
    if not predictions:
        logger.warning(f"Rally {rally.rally_id[:8]}: no filtered predictions")
        return None

    gt_ball = [p for p in rally.ground_truth.positions if p.label == "ball"]
    if not gt_ball:
        logger.warning(f"Rally {rally.rally_id[:8]}: no ball GT")
        return None

    # Evaluation GT: either keyframes only (honest against labeling truth) or
    # linearly interpolated (per-frame overlay rendering). The overlay always
    # uses the interpolated set so a green circle draws on every frame.
    approx_frame_count = max(p.frame_number for p in gt_ball) + 1
    gt_interp = GroundTruthResult(
        positions=gt_ball,
        frame_count=approx_frame_count,
        video_width=rally.video_width,
        video_height=rally.video_height,
    ).interpolate()
    gt_interp_positions = [p for p in gt_interp.positions if p.label == "ball"]
    gt_eval_positions = gt_ball if eval_keyframes_only else gt_interp_positions

    # Compute optimal frame offset BEFORE classifying so predictions align with GT.
    offset, _ = find_optimal_frame_offset(
        gt_eval_positions,
        predictions,
        rally.video_width,
        rally.video_height,
    )

    pred_by_frame = _shift_to_gt(predictions, offset)
    raw_positions = raw_positions or []
    raw_by_frame = _shift_to_gt(raw_positions, offset)
    interpolated_frames = _interpolated_frame_set(predictions, offset)

    # Walk the filter pipeline stage-by-stage so we can attribute kills.
    # run_pipeline_stages runs on pre-offset frame numbers; we shift the output
    # (via attribute_filter_kills inside classify_rally).
    stages = (
        run_pipeline_stages(raw_positions, config=get_wasb_filter_config())
        if raw_positions
        else {name: [] for name in STAGE_ORDER}
    )

    # Player positions — used to enrich wrong_object with player-lock proximity.
    player_positions_by_frame: dict[int, list[tuple[float, float]]] = {}
    if rally.predictions and rally.predictions.positions:
        for pp in rally.predictions.positions:
            rally_frame = pp.frame_number - offset
            player_positions_by_frame.setdefault(rally_frame, []).append((pp.x, pp.y))

    events = classify_rally(
        rally_id=rally.rally_id,
        gt_ball=gt_eval_positions,
        predictions=predictions,
        raw_positions=raw_positions,
        stages=stages,
        stage_order=STAGE_ORDER,
        width=rally.video_width,
        height=rally.video_height,
        frame_offset=offset,
        teleport_v_max_px_per_frame=teleport_v_max_px_per_frame,
        wrong_object_threshold_px=wrong_object_threshold_px,
        player_positions_by_frame=player_positions_by_frame,
        gt_mode="keyframes" if eval_keyframes_only else "interpolated",
    )

    # Index per-frame event lookups for the renderer.
    filter_kills_by_frame: dict[int, list[tuple[tuple[float, float], str]]] = {}
    for fk in events.filter_kills:
        filter_kills_by_frame.setdefault(fk.frame, []).append((fk.xy, fk.stage))

    teleports_by_frame = {tp.frame: tp for tp in events.teleports}
    wrong_by_frame = {wo.frame: wo for wo in events.wrong_objects}

    stationary_frames: set[int] = set()
    stationary_median: tuple[float, float] | None = None
    for sc in events.stationary_clusters:
        for f in range(sc.start_frame, sc.end_frame + 1):
            stationary_frames.add(f)
        if stationary_median is None and sc.passed_filter:
            stationary_median = sc.median_xy

    revisit_frames: set[int] = set()
    revisit_centers: list[tuple[float, float]] = []
    for rc in events.revisit_clusters:
        for f in rc.visit_frames:
            revisit_frames.add(f)
        revisit_centers.append(rc.center_xy)

    start_frame_video = int(start_ms / 1000 * rally.video_fps)
    end_frame_video = int(end_ms / 1000 * rally.video_fps)

    # Overlay always uses the per-frame interpolated GT (so a GT circle draws
    # on every video frame); evaluation uses gt_eval_positions (keyframes or
    # interpolated, per flag) for budget accounting.
    gt_by_frame = {p.frame_number: p for p in gt_interp_positions}

    return RallyRenderContext(
        rally_id=rally.rally_id,
        video_path=video_path,
        video_width=rally.video_width,
        video_height=rally.video_height,
        video_fps=rally.video_fps,
        start_frame=start_frame_video,
        end_frame=end_frame_video,
        gt_by_frame=gt_by_frame,
        pred_by_frame=pred_by_frame,
        raw_by_frame=raw_by_frame,
        interpolated_frames=interpolated_frames,
        filter_kills_by_frame=filter_kills_by_frame,
        teleports_by_frame=teleports_by_frame,
        wrong_by_frame=wrong_by_frame,
        stationary_frames=stationary_frames,
        stationary_median=stationary_median,
        revisit_frames=revisit_frames,
        revisit_centers=revisit_centers,
        per_frame_status=events.per_frame_status,
        frame_offset=offset,
        events=events,
    )


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------


def _norm_to_px(
    xy: tuple[float, float], width: int, height: int
) -> tuple[int, int]:
    return int(xy[0] * width), int(xy[1] * height)


def _draw_trail(
    frame: np.ndarray,
    positions: list[tuple[int, tuple[float, float]]],
    current_frame: int,
    color: tuple[int, int, int],
    width: int,
    height: int,
) -> None:
    """Fading polyline from oldest (faint) to newest (bright)."""
    if len(positions) < 2:
        return
    ordered = sorted(positions, key=lambda t: t[0])
    for i in range(1, len(ordered)):
        prev_f, prev_xy = ordered[i - 1]
        curr_f, curr_xy = ordered[i]
        age = current_frame - curr_f
        alpha = max(0.2, 1.0 - age / TRAIL_FRAMES)
        c = tuple(int(v * alpha) for v in color)
        cv2.line(
            frame,
            _norm_to_px(prev_xy, width, height),
            _norm_to_px(curr_xy, width, height),
            c,
            2,
        )


def _put_banner(
    frame: np.ndarray, text: str, height: int
) -> None:
    bar_h = 40
    cv2.rectangle(frame, (0, 0), (frame.shape[1], bar_h), BANNER_BG, -1)
    cv2.putText(
        frame, text, (12, 27),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, BANNER_FG, 2,
    )


def _draw_dashed_circle(
    frame: np.ndarray,
    center: tuple[int, int],
    radius: int,
    color: tuple[int, int, int],
    segments: int = 10,
) -> None:
    """Approximate dashed circle (OpenCV has no native dashed circle)."""
    for i in range(segments):
        if i % 2 != 0:
            continue
        a0 = int(360 * i / segments)
        a1 = int(360 * (i + 1) / segments)
        cv2.ellipse(
            frame, center, (radius, radius), 0, a0, a1, color, 2,
        )


def render_frame(
    frame: np.ndarray,
    ctx: RallyRenderContext,
    rally_frame: int,
) -> np.ndarray:
    out = frame.copy()
    width = ctx.video_width
    height = ctx.video_height

    # Trails (last TRAIL_FRAMES frames of GT + pred)
    gt_trail = [
        (f, (p.x, p.y))
        for f, p in ctx.gt_by_frame.items()
        if rally_frame - TRAIL_FRAMES <= f <= rally_frame
    ]
    pred_trail = [
        (f, (p.x, p.y))
        for f, p in ctx.pred_by_frame.items()
        if rally_frame - TRAIL_FRAMES <= f <= rally_frame
    ]
    _draw_trail(out, gt_trail, rally_frame, TRAIL_GT, width, height)
    _draw_trail(out, pred_trail, rally_frame, TRAIL_PRED, width, height)

    # Stationary-cluster marker (shown for every frame inside a cluster so the
    # distractor position is always visible while it's distracting the filter).
    if rally_frame in ctx.stationary_frames and ctx.stationary_median is not None:
        cx, cy = _norm_to_px(ctx.stationary_median, width, height)
        cv2.rectangle(
            out, (cx - 18, cy - 18), (cx + 18, cy + 18), STATIONARY_COLOR, 2,
        )
        cv2.putText(
            out, "STATIC",
            (cx + 20, cy - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, STATIONARY_COLOR, 1,
        )

    # Revisit-cluster markers (persistent hot-spots like court-side flags).
    # Rendered on every frame of the rally so the user can see the distractor
    # locations; highlighted when the current frame is one of the visits.
    for cx_norm, cy_norm in ctx.revisit_centers:
        cx, cy = _norm_to_px((cx_norm, cy_norm), width, height)
        is_active = rally_frame in ctx.revisit_frames
        thickness = 3 if is_active else 1
        cv2.rectangle(
            out, (cx - 14, cy - 14), (cx + 14, cy + 14),
            REVISIT_COLOR, thickness,
        )
        if is_active:
            cv2.putText(
                out, "REVISIT",
                (cx + 16, cy - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, REVISIT_COLOR, 2,
            )

    # Filter-kill markers for the current frame
    for xy, stage in ctx.filter_kills_by_frame.get(rally_frame, []):
        px, py = _norm_to_px(xy, width, height)
        cv2.line(out, (px - 8, py - 8), (px + 8, py + 8), FILTER_KILL_COLOR, 2)
        cv2.line(out, (px - 8, py + 8), (px + 8, py - 8), FILTER_KILL_COLOR, 2)
        cv2.putText(
            out, stage,
            (px + 10, py - 6),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, FILTER_KILL_COLOR, 1,
        )

    # Raw WASB dot (faint blue)
    raw = ctx.raw_by_frame.get(rally_frame)
    if raw is not None and raw.confidence > 0:
        cv2.circle(out, _norm_to_px((raw.x, raw.y), width, height), 4, RAW_COLOR, -1)

    # Filtered prediction (red circle; dashed if interpolated)
    pred = ctx.pred_by_frame.get(rally_frame)
    if pred is not None:
        pcenter = _norm_to_px((pred.x, pred.y), width, height)
        if rally_frame in ctx.interpolated_frames:
            _draw_dashed_circle(out, pcenter, 9, PRED_COLOR)
        else:
            cv2.circle(out, pcenter, 9, PRED_COLOR, 2)

    # GT circle (green)
    gt = ctx.gt_by_frame.get(rally_frame)
    if gt is not None:
        cv2.circle(
            out, _norm_to_px((gt.x, gt.y), width, height),
            11, GT_COLOR, 2,
        )

    # Wrong-object yellow bbox around pred
    wrong = ctx.wrong_by_frame.get(rally_frame)
    if wrong is not None:
        px, py = _norm_to_px(wrong.pred_xy, width, height)
        cv2.rectangle(
            out, (px - 22, py - 22), (px + 22, py + 22), WRONG_OBJ_COLOR, 2,
        )
        lbl = f"WRONG {wrong.error_px:.0f}px"
        if wrong.nearest_player_distance_px is not None:
            lbl += f"  near-pl={wrong.nearest_player_distance_px:.0f}"
        cv2.putText(
            out, lbl,
            (px + 24, py - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, WRONG_OBJ_COLOR, 2,
        )

    # Teleport arrow
    tp = ctx.teleports_by_frame.get(rally_frame)
    if tp is not None:
        p0 = _norm_to_px(tp.prev_xy, width, height)
        p1 = _norm_to_px(tp.curr_xy, width, height)
        cv2.arrowedLine(out, p0, p1, TELEPORT_COLOR, 3, tipLength=0.15)
        cv2.putText(
            out, f"TELEPORT {tp.dv_px:.0f}px/f",
            (p1[0] + 12, p1[1] + 12),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, TELEPORT_COLOR, 2,
        )

    # Status banner
    status = ctx.per_frame_status.get(rally_frame, "")
    conf = f"conf={pred.confidence:.2f}" if pred is not None else "conf=-"
    err_txt = "err=-"
    if pred is not None and gt is not None:
        dx = (pred.x - gt.x) * width
        dy = (pred.y - gt.y) * height
        err_txt = f"err={math.sqrt(dx*dx + dy*dy):.0f}px"
    banner = (
        f"Rally {ctx.rally_id[:8]} | f={rally_frame} "
        f"(video {ctx.start_frame + rally_frame}) | {status.upper() or '-'} | "
        f"{err_txt} | {conf} | off={ctx.frame_offset}"
    )
    _put_banner(out, banner, height)

    # Legend (bottom-left)
    legend_y = height - 10
    legend = [
        ("GT", GT_COLOR),
        ("pred", PRED_COLOR),
        ("raw", RAW_COLOR),
        ("kill", FILTER_KILL_COLOR),
        ("tele", TELEPORT_COLOR),
        ("wrong", WRONG_OBJ_COLOR),
        ("static", STATIONARY_COLOR),
        ("revisit", REVISIT_COLOR),
    ]
    x = 10
    for label, color in legend:
        cv2.circle(out, (x + 6, legend_y - 6), 5, color, -1)
        cv2.putText(
            out, label,
            (x + 15, legend_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1,
        )
        x += 65

    return out


# ---------------------------------------------------------------------------
# Rendering orchestration
# ---------------------------------------------------------------------------


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def render_rally_video(
    ctx: RallyRenderContext,
    output_mp4: Path,
    progress_every: int = 60,
) -> None:
    """Write overlay.mp4 by iterating video frames in the rally's range.

    Pipes raw BGR frames to ffmpeg (libx264) when ffmpeg is on PATH so the
    output is universally playable. Falls back to OpenCV's mp4v writer
    otherwise, which some players reject.
    """
    video = Video(ctx.video_path)
    info = video.info

    use_ffmpeg = _ffmpeg_available()
    ff_proc: subprocess.Popen[bytes] | None = None
    cv_writer: cv2.VideoWriter | None = None
    if use_ffmpeg:
        ff_cmd = [
            "ffmpeg",
            "-y",
            "-v", "error",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{info.width}x{info.height}",
            "-r", f"{ctx.video_fps}",
            "-i", "-",
            "-an",
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "22",
            "-pix_fmt", "yuv420p",
            "-movflags", "+faststart",
            str(output_mp4),
        ]
        ff_proc = subprocess.Popen(ff_cmd, stdin=subprocess.PIPE)
    else:
        logger.warning("ffmpeg not on PATH — falling back to cv2.VideoWriter mp4v")
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        cv_writer = cv2.VideoWriter(
            str(output_mp4),
            fourcc,
            ctx.video_fps,
            (info.width, info.height),
        )

    try:
        total = ctx.end_frame - ctx.start_frame + 1
        written = 0
        t0 = time.time()
        for video_frame, frame in video.iter_frames(
            start_frame=ctx.start_frame,
            end_frame=ctx.end_frame + 1,
            step=1,
        ):
            rally_frame = video_frame - ctx.start_frame

            if (info.width, info.height) != (ctx.video_width, ctx.video_height):
                frame = cv2.resize(frame, (ctx.video_width, ctx.video_height))

            annotated = render_frame(frame, ctx, rally_frame)

            if annotated.shape[:2] != (info.height, info.width):
                annotated = cv2.resize(annotated, (info.width, info.height))

            if ff_proc is not None and ff_proc.stdin is not None:
                ff_proc.stdin.write(annotated.tobytes())
            elif cv_writer is not None:
                cv_writer.write(annotated)

            written += 1
            if written % progress_every == 0 or written == total:
                elapsed = time.time() - t0
                fps = written / elapsed if elapsed > 0 else 0
                logger.info(
                    f"    [{written:>4d}/{total}] "
                    f"({fps:.1f} fps rendered)"
                )
    finally:
        if ff_proc is not None:
            if ff_proc.stdin is not None:
                ff_proc.stdin.close()
            rc = ff_proc.wait()
            if rc != 0:
                raise RuntimeError(f"ffmpeg exited with status {rc}")
        if cv_writer is not None:
            cv_writer.release()


def process_rally(
    rally: TrackingEvaluationRally,
    raw_cache: BallRawCache,
    output_dir: Path,
    *,
    write_video: bool,
    prediction_source: str,
    teleport_v_max_px_per_frame: float,
    wrong_object_threshold_px: float,
    eval_keyframes_only: bool,
) -> dict[str, Any]:
    """Classify + optionally render one rally. Returns a summary dict."""
    t_start = time.time()
    rid = rally.rally_id
    short = rid[:8]
    logger.info(f"  → Processing rally {short} ({rally.video_id[:8]})")

    cached = raw_cache.get(rid)
    raw_positions = cached.raw_ball_positions if cached else []
    raw_available = bool(cached)

    ctx = _build_context(
        rally,
        raw_positions=raw_positions,
        prediction_source=prediction_source,
        teleport_v_max_px_per_frame=teleport_v_max_px_per_frame,
        wrong_object_threshold_px=wrong_object_threshold_px,
        eval_keyframes_only=eval_keyframes_only,
    )
    if ctx is None:
        return {
            "rally_id": rid,
            "status": "skipped",
            "reason": "context_build_failed",
        }

    rally_dir = output_dir / rid
    rally_dir.mkdir(parents=True, exist_ok=True)

    events_path = rally_dir / "events.json"
    budget_path = rally_dir / "failure_budget.json"

    events_dict = ctx.events.to_dict()
    events_dict["raw_available"] = raw_available
    events_dict["prediction_source"] = prediction_source
    with events_path.open("w") as f:
        json.dump(events_dict, f, indent=2)

    budget_dict = ctx.events.budget.to_dict()
    budget_dict["raw_available"] = raw_available
    budget_dict["prediction_source"] = prediction_source
    if not raw_available:
        # Without raw cache we can't split missed_no_raw vs filter_killed —
        # everything lands in missed_no_raw by construction. Mark that.
        budget_dict["attribution_note"] = (
            "raw cache unavailable — missed bucket cannot be split between "
            "model miss and filter kill"
        )
    with budget_path.open("w") as f:
        json.dump(budget_dict, f, indent=2)

    video_path: Path | None = None
    if write_video:
        video_path = rally_dir / "overlay.mp4"
        render_rally_video(ctx, video_path)

    dur = time.time() - t_start
    budget = ctx.events.budget
    match_rate = (
        (budget.matched + budget.interpolated_correct) / budget.total_gt_frames
        if budget.total_gt_frames > 0 else 0.0
    )
    logger.info(
        f"    done: match={match_rate:.1%} "
        f"miss_noraw={budget.missed_no_raw} "
        f"miss_killed={budget.missed_filter_killed} "
        f"wrong={budget.wrong_object} "
        f"teleports={budget.teleport_count} "
        f"static_clusters={budget.stationary_cluster_count} "
        f"({dur:.1f}s)"
    )
    return {
        "rally_id": rid,
        "status": "ok",
        "video_id": rally.video_id,
        "match_rate": match_rate,
        "budget": budget_dict,
        "video": str(video_path) if video_path else None,
        "events_path": str(events_path),
        "duration_s": dur,
        "raw_available": raw_available,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rally", help="Run on one rally_id")
    parser.add_argument("--all", action="store_true", help="Run on every ball-GT rally")
    parser.add_argument(
        "--output-dir",
        default="outputs/ball_gap_report",
        help="Where rally subdirs are written (relative to analysis/)",
    )
    parser.add_argument("--no-video", action="store_true", help="Skip MP4 render")
    parser.add_argument(
        "--source",
        choices=["db", "refilter"],
        default="db",
        help=(
            "Prediction source. 'db' (default) uses the stored "
            "player_tracks.ball_positions_json — what production actually "
            "serves. 'refilter' re-applies the current "
            "get_wasb_filter_config() to the cached raw WASB output and "
            "reports what today's code would produce on re-track; use this to "
            "attribute stage-level filter kills and to A/B filter changes. "
            "NB: the two can diverge materially because DB rows were written "
            "at track time with whatever filter config was deployed then."
        ),
    )
    parser.add_argument(
        "--teleport-v-max",
        type=float,
        default=120.0,
        help=(
            "Teleport threshold in pixels/frame (default 120). Spike speeds on "
            "1920x1080 beach footage top out around 100-130 px/frame; anything "
            "above this is rarely physical."
        ),
    )
    parser.add_argument(
        "--wrong-object-px",
        type=float,
        default=100.0,
        help=(
            "Wrong-object threshold in pixels (default 100). Kept above the 50 "
            "px match threshold so fast-motion interpolation mismatches don't "
            "get falsely flagged as wrong-object."
        ),
    )
    parser.add_argument(
        "--eval-keyframes-only",
        action="store_true",
        help=(
            "Evaluate against GT keyframes only (skip linear interpolation "
            "between keyframes). Removes the parabola-vs-straight-line "
            "measurement artifact that inflates wrong-object counts during "
            "fast motion — the ball's true arc is parabolic but "
            "linearly-interpolated GT drifts from it, flagging correct "
            "predictions as off-target. Overlay still renders GT on every "
            "frame via interpolation; only the budget accounting changes."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only the first N rallies (sorted by ball_gt point count, desc)",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(message)s",
    )

    if not args.rally and not args.all:
        parser.error("must pass --rally or --all")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output dir: {output_dir}")

    raw_cache = BallRawCache()

    if args.rally:
        rallies = load_labeled_rallies(rally_id=args.rally)
    else:
        rallies = load_labeled_rallies()
        rallies = [
            r for r in rallies
            if any(p.label == "ball" for p in r.ground_truth.positions)
        ]
        rallies.sort(
            key=lambda r: sum(
                1 for p in r.ground_truth.positions if p.label == "ball"
            ),
            reverse=True,
        )

    if args.limit is not None:
        rallies = rallies[: args.limit]

    if not rallies:
        logger.error("No rallies to process.")
        return 2

    logger.info(f"Diagnosing {len(rallies)} rally(s). Video render: {not args.no_video}")

    summaries: list[dict[str, Any]] = []
    t_batch = time.time()
    for i, rally in enumerate(rallies, start=1):
        logger.info(f"[{i}/{len(rallies)}] rally={rally.rally_id[:8]}")
        try:
            summary = process_rally(
                rally,
                raw_cache,
                output_dir,
                write_video=not args.no_video,
                prediction_source=args.source,
                teleport_v_max_px_per_frame=args.teleport_v_max,
                wrong_object_threshold_px=args.wrong_object_px,
                eval_keyframes_only=args.eval_keyframes_only,
            )
        except Exception as e:  # noqa: BLE001
            logger.exception(f"Rally {rally.rally_id[:8]} failed: {e}")
            summary = {
                "rally_id": rally.rally_id,
                "status": "error",
                "error": str(e),
            }
        summaries.append(summary)

    summary_path = output_dir / "summary.json"
    with summary_path.open("w") as f:
        json.dump(
            {
                "generated_s": time.time() - t_batch,
                "rallies": summaries,
            },
            f,
            indent=2,
        )
    logger.info(f"Wrote batch summary → {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
