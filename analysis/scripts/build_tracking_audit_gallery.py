"""Build a static HTML gallery from reports/tracking_audit/*.json.

Per-rally detail pages with embedded annotated MP4 clips (GT magenta outline,
pred colored by pred_id). One event card per: real IDsw, missed-range >= 12
frames, fragmentation span boundary, convention drift.

Reuses:
  rallycut.evaluation.tracking.db.get_video_path        — S3 video fetch
  rallycut.cli.commands.track_player.TRACK_COLORS       — consistent per-track color palette
  rallycut.evaluation.tracking.db.load_labeled_rallies  — GT + pred positions

Usage:
  uv run python scripts/build_tracking_audit_gallery.py
  uv run python scripts/build_tracking_audit_gallery.py --retrack   # Use real pipeline output
  uv run python scripts/build_tracking_audit_gallery.py --rally <id>
  uv run python scripts/build_tracking_audit_gallery.py --max-events-per-rally 4
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from rallycut.cli.commands.track_player import TRACK_COLORS
from rallycut.evaluation.tracking.db import (
    get_video_path,
    load_labeled_rallies,
)
from rallycut.labeling.ground_truth import GroundTruthPosition
from rallycut.tracking.player_tracker import PlayerPosition, PlayerTrackingResult

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("gallery")

MISSED_RANGE_MIN_FRAMES = 3         # ≥ 0.1s (LS keyframes can be very sparse)
FRAGMENT_MIN_GAP_FRAMES = 8         # ≥ 0.27s between pred_id handoffs
SEVERE_LOSS_COVERAGE_MAX = 0.6      # coverage ≤ 60% + ≥ 15 GT frames → "partially/severely lost" event
SEVERE_LOSS_MIN_GT_FRAMES = 15
CLIP_WINDOW_FRAMES = 45             # ±45 frames (~1.5s at 30fps, 3s at 60fps)
CLIP_FPS_OUTPUT = 30


# ---------------------------------------------------------------------------
# Event extraction
# ---------------------------------------------------------------------------


@dataclass
class Event:
    kind: str                          # "missed" | "switch" | "fragment" | "swap" | "swap_recovery" | "convention" | "severe_loss"
    gt_track_id: int | None
    gt_label: str | None
    frame: int                         # center frame for clip
    start_frame: int                   # event extent
    end_frame: int
    title: str
    detail: str
    severity: int                      # for sorting (higher = worse)
    pred_old: int | None = None        # only set for swap / fragment events
    pred_new: int | None = None


def _missed_events(per_gt: list[dict], min_len: int) -> list[Event]:
    events: list[Event] = []
    for g in per_gt:
        for cause, ranges in g.get("missedByCause", {}).items():
            for start, end in ranges:
                length = end - start + 1
                if length < min_len:
                    continue
                center = (start + end) // 2
                events.append(Event(
                    kind="missed",
                    gt_track_id=g["gtTrackId"],
                    gt_label=g["gtLabel"],
                    frame=center,
                    start_frame=start,
                    end_frame=end,
                    title=f"{g['gtLabel']}: missed {length}f ({cause})",
                    detail=f"cause={cause}, frames {start}–{end}",
                    severity=length,
                ))
    return events


def _switch_events(per_gt: list[dict]) -> list[Event]:
    events: list[Event] = []
    for g in per_gt:
        for ev in g.get("realSwitches", []):
            events.append(Event(
                kind="switch",
                gt_track_id=g["gtTrackId"],
                gt_label=g["gtLabel"],
                frame=ev["frame"],
                start_frame=ev["frame"],
                end_frame=ev["frame"],
                title=f"{g['gtLabel']}: switch pred {ev['predId']} (gt {ev['gtOld']}→{ev['gtNew']})",
                detail=f"cause={ev['cause']} at frame {ev['frame']}",
                severity=1000,  # switches are always top priority
            ))
    return events


def _fragment_events(per_gt: list[dict], min_gap: int) -> list[Event]:
    """A fragmentation event is a span boundary where a new pred_id takes over.

    Emits one event at each boundary frame between different pred_ids. If the
    incoming pred_id was previously tracking a DIFFERENT GT track, we re-label
    the event as `swap` — i.e. the two pred IDs exchanged their GT assignments,
    which my `iter_real_switch_events` can miss when the abandoned pred's
    segments are too short. This matches how a human reviewer would describe
    "pred 3 and pred 6 swapped players".
    """
    # Build a reverse index: for each pred_id, the ordered list of
    # (gt_track_id, start_frame, end_frame) where it appeared.
    pred_history: dict[int, list[tuple[int, int, int]]] = {}
    for g in per_gt:
        for s, e, pid in g.get("predIdSpans", []):
            pred_history.setdefault(pid, []).append((g["gtTrackId"], s, e))
    for h in pred_history.values():
        h.sort(key=lambda t: t[1])

    def prior_gt_of(pred_id: int, before_frame: int) -> int | None:
        """GT track_id this pred was on just before `before_frame`, if any."""
        last_gt: int | None = None
        for gt_id, s, _e in pred_history.get(pred_id, []):
            if s >= before_frame:
                break
            last_gt = gt_id
        return last_gt

    events: list[Event] = []
    for g in per_gt:
        spans = g.get("predIdSpans", [])
        if len(spans) < 2:
            continue
        for prev, cur in zip(spans, spans[1:]):
            prev_start, prev_end, prev_pred = prev
            cur_start, cur_end, cur_pred = cur
            if prev_pred == cur_pred:
                continue
            gap = cur_start - prev_end - 1

            # Disguised swap detection.
            incoming_prior_gt = prior_gt_of(cur_pred, cur_start)
            is_swap = (
                incoming_prior_gt is not None
                and incoming_prior_gt != g["gtTrackId"]
            )

            if is_swap:
                kind = "swap"
                title = f"{g['gtLabel']}: pred {cur_pred} jumped from GT {incoming_prior_gt} (prior pred {prev_pred} left)"
                detail = (
                    f"pred {cur_pred} was tracking GT {incoming_prior_gt} before frame {cur_start}; "
                    f"post-boundary it tracks {g['gtLabel']}. Prior pred {prev_pred} ended frame {prev_end}."
                )
                severity = 600 + gap  # above pure fragment severity
            else:
                kind = "fragment"
                title = f"{g['gtLabel']}: pred {prev_pred} → {cur_pred} (gap {gap}f)"
                detail = (
                    f"pred {cur_pred} is new for {g['gtLabel']} at frame {cur_start}; "
                    f"prior pred {prev_pred} ended frame {prev_end}."
                )
                severity = 200 + gap
            events.append(Event(
                kind=kind,
                gt_track_id=g["gtTrackId"],
                gt_label=g["gtLabel"],
                frame=cur_start,
                start_frame=prev_end,
                end_frame=cur_start,
                title=title,
                detail=detail,
                severity=severity,
                pred_old=prev_pred,
                pred_new=cur_pred,
            ))
    return events


def _severe_loss_events(per_gt: list[dict]) -> list[Event]:
    """Track nearly or fully lost — emit even when there's no long contiguous miss."""
    events: list[Event] = []
    for g in per_gt:
        if g.get("gtFrameCount", 0) < SEVERE_LOSS_MIN_GT_FRAMES:
            continue
        if g.get("coverage", 1.0) > SEVERE_LOSS_COVERAGE_MAX:
            continue
        # Pick the middle of the GT track's frame range as the clip center.
        miss_frames: list[int] = []
        for ranges in g.get("missedByCause", {}).values():
            for start, end in ranges:
                miss_frames.extend(range(start, end + 1))
        if miss_frames:
            center = sorted(miss_frames)[len(miss_frames) // 2]
        else:
            center = 0
        # Dominant cause for the banner
        cause_counts = {
            c: sum(e - s + 1 for s, e in rs)
            for c, rs in g.get("missedByCause", {}).items()
        }
        dominant = max(cause_counts, key=cause_counts.get) if cause_counts else "unknown"
        events.append(Event(
            kind="severe_loss",
            gt_track_id=g["gtTrackId"],
            gt_label=g["gtLabel"],
            frame=center,
            start_frame=min(miss_frames) if miss_frames else 0,
            end_frame=max(miss_frames) if miss_frames else 0,
            title=f"{g['gtLabel']}: severely lost (coverage {g['coverage']:.0%}, dominant cause: {dominant})",
            detail=f"matched {g['matchedFrames']}/{g['gtFrameCount']} frames; causes: {cause_counts}",
            severity=2000 + int((1.0 - g["coverage"]) * 1000),  # top priority
        ))
    return events


def _convention_event(audit: dict) -> list[Event]:
    conv = audit.get("convention", {})
    if not (conv.get("courtSideFlip") or conv.get("teamLabelFlip")):
        return []
    # Use the middle of the rally as the center frame.
    mid = audit["frameCount"] // 2
    flip_type = "court_side" if conv.get("courtSideFlip") else "team_label"
    return [Event(
        kind="convention",
        gt_track_id=None,
        gt_label=None,
        frame=mid,
        start_frame=0,
        end_frame=audit["frameCount"] - 1,
        title=f"Convention drift ({flip_type})",
        detail=json.dumps(conv.get("gtLabelToPredIdMode", {})),
        severity=500,
    )]


TRAJECTORY_DELTA_HAS_SIGNAL = 0.03
TRAJECTORY_WINDOW = 10


def _linear_extrapolate(
    points: list[tuple[int, float, float]],
    target_frame: int,
) -> tuple[float, float] | None:
    """Constant-velocity extrapolation from ≤ 5 most recent points."""
    if len(points) < 2:
        return None
    tail = sorted(points, key=lambda t: t[0])[-5:]
    f0, x0, y0 = tail[0]
    f1, x1, y1 = tail[-1]
    if f1 == f0:
        return (x1, y1)
    vx = (x1 - x0) / (f1 - f0)
    vy = (y1 - y0) / (f1 - f0)
    dt = target_frame - f1
    return (x1 + vx * dt, y1 + vy * dt)


def reclassify_swap_recoveries(
    events: list[Event],
    predictions: PlayerTrackingResult | None,
    swap_kinds: tuple[str, ...] = ("swap",),
) -> list[Event]:
    """Re-label `swap` events as `swap_recovery` when trajectory continuity
    indicates the tracker was correctly re-associating, not mis-swapping.

    A swap is a recovery iff pred_new's *actual* position at swap_frame is
    significantly closer to pred_old's pre-swap extrapolation than to its
    own pre-swap extrapolation (Δ ≥ 0.03 normalised).
    """
    if predictions is None:
        return events

    # Index pred positions by (pred_id, frame) for fast lookup
    positions_by_pred: dict[int, list[tuple[int, float, float]]] = {}
    for p in predictions.positions:
        positions_by_pred.setdefault(p.track_id, []).append((p.frame_number, p.x, p.y))

    def _actual(pred_id: int, frame: int) -> tuple[float, float] | None:
        best = None
        best_df = 3
        for f, x, y in positions_by_pred.get(pred_id, []):
            if f == frame:
                return (x, y)
            df = abs(f - frame)
            if df < best_df:
                best_df = df
                best = (x, y)
        return best

    def _dist(a: tuple[float, float] | None, b: tuple[float, float] | None) -> float | None:
        if a is None or b is None:
            return None
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    refined: list[Event] = []
    for ev in events:
        if ev.kind not in swap_kinds or ev.pred_old is None or ev.pred_new is None:
            refined.append(ev)
            continue
        swap_frame = ev.frame
        pred_new = ev.pred_new
        pred_old = ev.pred_old

        pre_range = range(max(0, swap_frame - TRAJECTORY_WINDOW), swap_frame)
        new_pre = [(f, x, y) for f, x, y in positions_by_pred.get(pred_new, []) if f in pre_range]
        old_pre = [(f, x, y) for f, x, y in positions_by_pred.get(pred_old, []) if f in pre_range]
        extrap_new = _linear_extrapolate(new_pre, swap_frame)
        extrap_old = _linear_extrapolate(old_pre, swap_frame)
        actual = _actual(pred_new, swap_frame)
        d_new = _dist(actual, extrap_new)
        d_old = _dist(actual, extrap_old)
        if d_new is None or d_old is None:
            refined.append(ev)
            continue
        delta = d_new - d_old  # +ve → closer to old_pre → recovery
        if delta >= TRAJECTORY_DELTA_HAS_SIGNAL:
            # Reclassify as recovery (tracker was correct, our audit over-flagged).
            refined.append(Event(
                kind="swap_recovery",
                gt_track_id=ev.gt_track_id,
                gt_label=ev.gt_label,
                frame=ev.frame,
                start_frame=ev.start_frame,
                end_frame=ev.end_frame,
                title=ev.title.replace("pred " + str(pred_new) + " jumped from GT",
                                        "pred " + str(pred_new) + " recovered GT", 1),
                detail=(
                    ev.detail + f"  [trajectory: d(actual, new_pre)={d_new:.3f} vs "
                    f"d(actual, old_pre)={d_old:.3f} — tracker correctly re-associated]"
                ),
                severity=ev.severity - 400,  # lower than real swap, above fragment
                pred_old=ev.pred_old,
                pred_new=ev.pred_new,
            ))
        else:
            refined.append(ev)
    return refined


def extract_events(audit: dict, max_events: int) -> list[Event]:
    evs: list[Event] = []
    evs.extend(_severe_loss_events(audit["perGt"]))
    evs.extend(_switch_events(audit["perGt"]))
    evs.extend(_missed_events(audit["perGt"], MISSED_RANGE_MIN_FRAMES))
    evs.extend(_fragment_events(audit["perGt"], FRAGMENT_MIN_GAP_FRAMES))
    evs.extend(_convention_event(audit))
    evs.sort(key=lambda e: e.severity, reverse=True)
    return evs[:max_events]


# ---------------------------------------------------------------------------
# Clip rendering
# ---------------------------------------------------------------------------


def _ms_at_rally_start(rally_id: str) -> int | None:
    """Look up start_ms for the rally directly from the DB."""
    from rallycut.evaluation.db import get_connection
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT start_ms FROM rallies WHERE id = %s", [rally_id])
            row = cur.fetchone()
            return int(row[0]) if row else None


def fetch_rally_meta(rally_id: str) -> dict:
    """Fetch per-rally metadata: tracker model_version, updated_at, video quality flags.

    Pulled lazily per rally because `TrackingEvaluationRally` doesn't carry it.
    """
    from rallycut.evaluation.db import get_connection
    q = """
        SELECT pt.model_version, pt.completed_at, pt.created_at,
               v.quality_report_json,
               v.court_calibration_json,
               v.name, v.filename,
               v.fps,
               r.start_ms, r.end_ms
        FROM rallies r
        JOIN player_tracks pt ON pt.rally_id = r.id
        LEFT JOIN videos v ON v.id = r.video_id
        WHERE r.id = %s
    """
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(q, [rally_id])
            row = cur.fetchone()
    if not row:
        return {}
    (model_version, completed_at, created_at, quality_json, calibration_json,
     video_name, video_filename, video_fps, start_ms, end_ms) = row
    tracked_dt = completed_at or created_at
    # Surface the quality flags we know the team cares about (if present).
    quality_json = quality_json or {}
    flags = []
    for k in ("cameraDistance", "crowdLevel", "sceneComplexity", "brightness",
              "cameraAngle", "shakiness", "tilted", "dark", "overexposed"):
        if k in quality_json:
            flags.append((k, quality_json[k]))
    return {
        "model_version": model_version,
        "tracked_at": tracked_dt.isoformat() if tracked_dt else None,
        "quality_flags": flags,
        "has_calibration": bool(calibration_json),
        "start_ms": int(start_ms) if start_ms else 0,
        "end_ms": int(end_ms) if end_ms else 0,
        "video_name": video_name,
        "video_filename": video_filename,
        "video_fps": float(video_fps) if video_fps else 30.0,
    }


def _fmt_ts(ms: float) -> str:
    """mm:ss.mmm."""
    if ms < 0:
        return "-" + _fmt_ts(-ms)
    total_s = ms / 1000.0
    m = int(total_s // 60)
    s = total_s - m * 60
    return f"{m:02d}:{s:06.3f}"


def _event_timing(event: Event, meta: dict, video_fps: float) -> dict:
    """Produce rally-relative + video-absolute timestamps for a single event,
    plus the Label Studio frame number (1-indexed at 30 fps)."""
    rally_start_ms = meta.get("start_ms", 0)
    rally_frame_ms = event.frame / video_fps * 1000.0
    abs_video_ms = rally_start_ms + rally_frame_ms
    ls_frame = int(round(abs_video_ms / 1000.0 * 30.0)) + 1  # 1-indexed LS frame
    return {
        "rally_frame": event.frame,
        "rally_t": _fmt_ts(rally_frame_ms),
        "video_t": _fmt_ts(abs_video_ms),
        "ls_frame": ls_frame,
    }


def _extract_frames(
    video_path: Path,
    start_s: float,
    duration_s: float,
    tmp_path: Path,
) -> bool:
    """FFmpeg clip extract. Outputs a segment MP4 at original fps."""
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-ss", f"{start_s:.3f}",
        "-i", str(video_path),
        "-t", f"{duration_s:.3f}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
        "-an",
        str(tmp_path),
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
        return tmp_path.exists()
    except subprocess.CalledProcessError as e:
        logger.warning(f"ffmpeg failed: {e.stderr.decode(errors='ignore')[:200]}")
        return False


GT_COLOR_BGR = (255, 0, 255)       # magenta
EVENT_HIGHLIGHT_BGR = (0, 220, 255) # bright amber — the one bbox at the center of the event
INFO_BG_BGR = (28, 28, 28)         # near-black translucent background
INFO_FG_BGR = (240, 240, 240)


def _draw_dashed_rect(
    frame: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
    thickness: int = 2,
    dash: int = 10,
) -> None:
    for (xa, ya), (xb, yb) in [((x1, y1), (x2, y1)), ((x2, y1), (x2, y2)),
                                 ((x2, y2), (x1, y2)), ((x1, y2), (x1, y1))]:
        length = int(np.hypot(xb - xa, yb - ya))
        if length == 0:
            continue
        dx = (xb - xa) / length
        dy = (yb - ya) / length
        for i in range(0, length, dash * 2):
            sx = int(xa + dx * i)
            sy = int(ya + dy * i)
            ex = int(xa + dx * min(i + dash, length))
            ey = int(ya + dy * min(i + dash, length))
            cv2.line(frame, (sx, sy), (ex, ey), color, thickness)


def _draw_gt_layer(
    frame: np.ndarray,
    gt_in_frame: list[GroundTruthPosition],
    subject_track_id: int | None,
) -> None:
    h, w = frame.shape[:2]
    for gt in gt_in_frame:
        x1 = int((gt.x - gt.width / 2) * w)
        y1 = int((gt.y - gt.height / 2) * h)
        x2 = int((gt.x + gt.width / 2) * w)
        y2 = int((gt.y + gt.height / 2) * h)
        is_subject = subject_track_id is not None and gt.track_id == subject_track_id
        color = EVENT_HIGHLIGHT_BGR if is_subject else GT_COLOR_BGR
        thick = 3 if is_subject else 2
        _draw_dashed_rect(frame, x1, y1, x2, y2, color, thickness=thick, dash=12)
        # Label ABOVE bbox (never below — avoids player-control-bar collision).
        tag = f"GT {gt.label}"
        if is_subject:
            tag += " [subject]"
        label_y = max(y1 - 10, 18)
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, label_y - th - 6), (x1 + tw + 10, label_y + 4), color, -1)
        cv2.putText(frame, tag, (x1 + 4, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


def _draw_pred_layer(
    frame: np.ndarray,
    pred_in_frame: list[PlayerPosition],
    primary_track_ids: set[int],
) -> None:
    h, w = frame.shape[:2]
    for p in pred_in_frame:
        x1 = int((p.x - p.width / 2) * w)
        y1 = int((p.y - p.height / 2) * h)
        x2 = int((p.x + p.width / 2) * w)
        y2 = int((p.y + p.height / 2) * h)
        color_idx = p.track_id % len(TRACK_COLORS) if p.track_id >= 0 else 0
        color = TRACK_COLORS[color_idx]
        thickness = 3 if p.track_id in primary_track_ids else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        tag = f"pred#{p.track_id}"
        if p.track_id in primary_track_ids:
            tag += "★"
        label_y = max(y1 - 10, 18)
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        cv2.rectangle(frame, (x1, label_y - th - 6), (x1 + tw + 10, label_y + 4), color, -1)
        cv2.putText(frame, tag, (x1 + 4, label_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)


def _draw_info_panel(
    frame: np.ndarray,
    kind: str,
    title: str,
    rally_frame: int,
    frame_count: int,
    rally_ms: int,
    is_event_frame: bool,
) -> None:
    """Persistent translucent top-left panel. Stays clear of video player controls.

    Lines:
      1: [KIND] title
      2: frame N / total · mm:ss.mmm
      3: legend — magenta dashed = GT, colored solid = pred, amber = subject
    """
    h, w = frame.shape[:2]
    panel_w = min(int(w * 0.55), 720)
    panel_h = 78
    x0, y0 = 0, 0
    # Translucent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), INFO_BG_BGR, -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, dst=frame)
    # Divider
    cv2.line(frame, (x0, y0 + panel_h), (x0 + panel_w, y0 + panel_h), (90, 90, 90), 1)

    seconds = rally_ms / 1000.0
    mm = int(seconds // 60)
    ss = seconds - mm * 60
    time_str = f"{mm:02d}:{ss:06.3f}"

    # Line 1 — kind badge + title
    badge = f"[{kind.upper()}]"
    cv2.putText(frame, badge, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 200, 255), 2)
    (bw, _), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    cv2.putText(frame, title, (10 + bw + 8, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, INFO_FG_BGR, 1)

    # Line 2 — frame + time
    frame_line = f"frame {rally_frame}/{frame_count} · rally t={time_str}"
    if is_event_frame:
        frame_line = "▶ " + frame_line + "  ← event frame"
    cv2.putText(frame, frame_line, (10, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255) if is_event_frame else INFO_FG_BGR, 1)

    # Line 3 — legend
    legend = "GT = magenta dashed  ·  pred = colored solid (★ = primary)  ·  subject = amber"
    cv2.putText(frame, legend, (10, 66),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)


def _draw_court_split(frame: np.ndarray, split_y: float | None) -> None:
    if split_y is None:
        return
    h, w = frame.shape[:2]
    y_px = int(split_y * h)
    # Dim yellow line only — no NEAR/FAR text anywhere near the bottom.
    overlay = frame.copy()
    cv2.line(overlay, (0, y_px), (w, y_px), (0, 255, 255), 1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, dst=frame)


def render_event_clip(
    video_path: Path,
    rally_start_ms: int,
    video_fps: float,
    predictions: PlayerTrackingResult,
    gt_positions: list[GroundTruthPosition],
    event: Event,
    out_path: Path,
    window: int = CLIP_WINDOW_FRAMES,
) -> bool:
    """Extract a clip for one Event and overlay GT + pred bboxes."""
    # Rally frame → absolute video frame
    rally_start_frame = int(round(rally_start_ms / 1000.0 * video_fps))
    abs_center = rally_start_frame + event.frame
    abs_start = max(0, abs_center - window)
    abs_end = abs_center + window
    duration_s = (abs_end - abs_start) / video_fps
    start_s = abs_start / video_fps

    tmp_clip = out_path.with_suffix(".raw.mp4")
    if not _extract_frames(video_path, start_s, duration_s, tmp_clip):
        return False

    # Index predictions + GT by rally-relative frame number.
    pred_by_frame: dict[int, list[PlayerPosition]] = {}
    for p in predictions.positions:
        pred_by_frame.setdefault(p.frame_number, []).append(p)
    gt_by_frame: dict[int, list[GroundTruthPosition]] = {}
    for g in gt_positions:
        gt_by_frame.setdefault(g.frame_number, []).append(g)

    cap = cv2.VideoCapture(str(tmp_clip))
    if not cap.isOpened():
        logger.warning(f"couldn't open {tmp_clip}")
        return False

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or video_fps

    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    primary_tracks = set(predictions.primary_track_ids or [])
    subject_id = event.gt_track_id  # highlight this GT in amber
    clip_frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rally_frame = abs_start + clip_frame_idx - rally_start_frame
        # Court split (dim, line only)
        _draw_court_split(frame, predictions.court_split_y)
        # Predictions (solid, colored by pred_id)
        _draw_pred_layer(frame, pred_by_frame.get(rally_frame, []), primary_tracks)
        # GT on top (magenta dashed; amber for subject)
        _draw_gt_layer(frame, gt_by_frame.get(rally_frame, []), subject_track_id=subject_id)
        # Info panel (persistent top-left; stays clear of bottom player-control bar)
        rally_ms = int(rally_frame / fps * 1000)
        _draw_info_panel(
            frame=frame,
            kind=event.kind,
            title=event.title,
            rally_frame=rally_frame,
            frame_count=predictions.frame_count,
            rally_ms=rally_ms,
            is_event_frame=(rally_frame == event.frame),
        )
        writer.write(frame)
        clip_frame_idx += 1

    cap.release()
    writer.release()
    try:
        tmp_clip.unlink()
    except OSError:
        pass
    return out_path.exists() and out_path.stat().st_size > 0


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------


_INDEX_CSS = """
<style>
  body { font-family: -apple-system, system-ui, sans-serif; margin: 2em; color: #222; }
  h1 { margin-bottom: 0.3em; }
  .sub { color: #666; margin-bottom: 1.5em; }
  table { border-collapse: collapse; width: 100%; font-size: 14px; }
  th, td { padding: 6px 10px; border-bottom: 1px solid #ddd; text-align: left; }
  th { background: #f7f7f7; cursor: pointer; }
  tr:hover td { background: #fafafa; }
  .bad { color: #c00; font-weight: 600; }
  .mid { color: #d90; }
  .ok { color: #090; }
  a { color: #06c; text-decoration: none; }
  a:hover { text-decoration: underline; }
  code { font-family: ui-monospace, SFMono-Regular, monospace; font-size: 12px; }
</style>
"""

_RALLY_CSS = _INDEX_CSS + """
<style>
  .cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(560px, 1fr)); gap: 1.2em; margin-top: 1.5em; }
  .card { border: 1px solid #ddd; border-radius: 6px; overflow: hidden; background: #fff; }
  .card .hdr { padding: 8px 12px; background: #fafafa; border-bottom: 1px solid #eee; }
  .card .hdr .k { display: inline-block; font-size: 11px; padding: 2px 6px; border-radius: 3px; background: #eef; color: #339; margin-right: 8px; }
  .card .hdr .k.switch { background: #fee; color: #c22; }
  .card .hdr .k.missed { background: #fec; color: #b70; }
  .card .hdr .k.fragment { background: #eef; color: #339; }
  .card .hdr .k.convention { background: #efe; color: #272; }
  .card .hdr .k.severe_loss { background: #fdd; color: #900; }
  .card .hdr .k.swap { background: #fcc; color: #700; font-weight: 600; }
  .card .hdr .k.swap_recovery { background: #cfc; color: #070; }
  .card video { width: 100%; display: block; background: #000; }
  .card .ft { padding: 8px 12px; font-size: 13px; color: #555; }
  .card .timing { padding: 6px 12px; background: #f4f8ff; border-top: 1px solid #e0e8f4; font-size: 12px; color: #335; }
  .card .timing code { background: #e0e8f4; padding: 1px 4px; border-radius: 3px; }
  .legend { padding: 6px 10px; background: #fffbe6; border: 1px solid #f3dc70; border-radius: 4px; font-size: 13px; margin-bottom: 1em; }
  .meta { padding: 6px 10px; background: #f4f8ff; border: 1px solid #cfdef9; border-radius: 4px; font-size: 13px; margin: 1em 0; }
  .meta > div { margin: 2px 0; }
  .meta .bad { color: #c00; }
  .meta .ok { color: #090; }
  .perGt { margin-top: 1.5em; }
  .perGt table { font-size: 13px; }
</style>
"""


def _severity_class(value: float, good: float, mid: float, higher_is_better: bool = True) -> str:
    if higher_is_better:
        if value >= good:
            return "ok"
        if value >= mid:
            return "mid"
        return "bad"
    if value <= good:
        return "ok"
    if value <= mid:
        return "mid"
    return "bad"


def render_index_html(audits: list[dict], out_path: Path) -> None:
    rows = []
    for a in audits:
        rid = a["rallyId"]
        hota = a.get("hota") or 0.0
        mota = a.get("mota") or 0.0
        switches = a.get("aggregateRealSwitches", 0)
        frag_total = sum(len(g.get("predIdSpans", [])) for g in a["perGt"])
        missed_range_total = sum(
            sum(len(r) for r in g.get("missedByCause", {}).values())
            for g in a["perGt"]
        )
        worst_cov = min((g["coverage"] for g in a["perGt"]), default=1.0)
        conv_flag = a["convention"].get("teamLabelFlip") or a["convention"].get("courtSideFlip")
        rows.append((rid, hota, mota, switches, frag_total, missed_range_total, worst_cov, conv_flag))

    # Sort by badness: real switches first, then inverse HOTA
    rows.sort(key=lambda r: (-r[3], -r[5], r[1]))

    html_rows = []
    for rid, hota, mota, sw, frag, miss, wcov, conv in rows:
        hota_cls = _severity_class(hota, 0.9, 0.75)
        mota_cls = _severity_class(mota, 0.9, 0.7)
        wcov_cls = _severity_class(wcov, 0.8, 0.5)
        sw_cls = _severity_class(sw, 0, 0, higher_is_better=False)
        conv_cls = "bad" if conv else "ok"
        html_rows.append(
            f"<tr>"
            f"<td><a href='{rid}.html'><code>{rid[:8]}</code></a></td>"
            f"<td class='{hota_cls}'>{hota:.3f}</td>"
            f"<td class='{mota_cls}'>{mota:.3f}</td>"
            f"<td class='{sw_cls}'>{sw}</td>"
            f"<td>{frag}</td>"
            f"<td>{miss}</td>"
            f"<td class='{wcov_cls}'>{wcov:.2f}</td>"
            f"<td class='{conv_cls}'>{'Y' if conv else '—'}</td>"
            f"</tr>"
        )

    body = f"""
    <h1>Tracking audit gallery</h1>
    <div class='sub'>{len(audits)} rallies · sorted by real ID-switch count, then missed ranges. Click a rally to see per-event clips.</div>
    <table>
      <thead>
        <tr>
          <th>Rally</th><th>HOTA</th><th>MOTA</th><th>Real IDsw</th>
          <th>Pred spans</th><th>Missed ranges</th><th>Worst coverage</th><th>Conv flip</th>
        </tr>
      </thead>
      <tbody>{''.join(html_rows)}</tbody>
    </table>
    """
    out_path.write_text(f"<!DOCTYPE html><html><head><meta charset='utf-8'>{_INDEX_CSS}<title>Tracking audit</title></head><body>{body}</body></html>")


def render_rally_html(
    audit: dict,
    events_with_clips: list[tuple[Event, str]],
    out_path: Path,
    meta: dict | None = None,
) -> None:
    rid = audit["rallyId"]
    conv = audit["convention"]

    # Per-GT summary table
    gt_rows = []
    for g in audit["perGt"]:
        cov_cls = _severity_class(g["coverage"], 0.8, 0.5)
        switches = len(g.get("realSwitches", []))
        sw_cls = _severity_class(switches, 0, 0, higher_is_better=False)
        miss_summary = ", ".join(
            f"{c}={sum(e-s+1 for s,e in r)}"
            for c, r in g.get("missedByCause", {}).items()
        )
        gt_rows.append(
            f"<tr>"
            f"<td>{g['gtLabel']}</td>"
            f"<td class='{cov_cls}'>{g['coverage']:.2f}</td>"
            f"<td>{g['matchedFrames']}/{g['gtFrameCount']}</td>"
            f"<td>{g['distinctPredIds']}</td>"
            f"<td class='{sw_cls}'>{switches}</td>"
            f"<td>{g['timeAtNetPct']:.2f}</td>"
            f"<td>{html.escape(miss_summary)}</td>"
            f"</tr>"
        )

    # Event cards — include video name + rally/video timestamps + LS frame so
    # the user can jump straight to the annotation tool and fix GT issues.
    video_fps = float(audit.get("videoFps") or 30.0)
    video_label = (meta or {}).get("video_name") or (meta or {}).get("video_filename") or "—"
    card_html = []
    for ev, clip_rel in events_with_clips:
        clip_tag = (
            f"<video controls muted preload='metadata' src='{html.escape(clip_rel)}'></video>"
            if clip_rel
            else "<div style='padding:20px;color:#900;background:#fee;'>clip failed to render</div>"
        )
        t = _event_timing(ev, meta or {}, video_fps)
        timing_line = (
            f"<div class='timing'>"
            f"<strong>Video</strong> <code>{html.escape(str(video_label))}</code> · "
            f"<strong>rally t</strong> <code>{t['rally_t']}</code> · "
            f"<strong>video t</strong> <code>{t['video_t']}</code> · "
            f"<strong>rally frame</strong> <code>{t['rally_frame']}</code> · "
            f"<strong>Label Studio frame (30fps)</strong> <code>{t['ls_frame']}</code>"
            f"</div>"
        )
        card_html.append(
            f"<div class='card'>"
            f"<div class='hdr'><span class='k {ev.kind}'>{ev.kind}</span>{html.escape(ev.title)}</div>"
            f"{clip_tag}"
            f"{timing_line}"
            f"<div class='ft'>{html.escape(ev.detail)}</div>"
            f"</div>"
        )

    summary = (
        f"<div><strong>HOTA</strong> {audit.get('hota') or 0:.3f} · "
        f"<strong>MOTA</strong> {audit.get('mota') or 0:.3f} · "
        f"<strong>Real IDsw</strong> {audit.get('aggregateRealSwitches', 0)} · "
        f"<strong>Frame count</strong> {audit['frameCount']} · "
        f"<strong>FPS</strong> {audit.get('videoFps', 30):.1f}</div>"
    )
    conv_summary = (
        f"<div><strong>Convention</strong>: court_side_flip={conv.get('courtSideFlip')} · "
        f"team_label_flip={conv.get('teamLabelFlip')} · "
        f"gtLabel→predId mode: {html.escape(json.dumps(conv.get('gtLabelToPredIdMode', {})))}</div>"
    )

    # Provenance + known-limitation flags — helps you ignore rallies that fall
    # under "known bad" cases (low camera angle, very crowded scenes, etc.) and
    # flag data that was tracked with an older pipeline version.
    meta_html = ""
    if meta:
        flags_inner = ", ".join(
            f"{html.escape(k)}=<code>{html.escape(str(v))}</code>"
            for k, v in meta.get("quality_flags", [])
        ) or "<em>none recorded</em>"
        calib = "<span class='ok'>yes</span>" if meta.get("has_calibration") else "<span class='bad'>no</span>"
        video_label = meta.get("video_name") or meta.get("video_filename") or "—"
        rally_span = f"{_fmt_ts(meta.get('start_ms', 0))} – {_fmt_ts(meta.get('end_ms', 0))}"
        meta_html = (
            f"<div class='meta'>"
            f"<div><strong>Source video</strong> <code>{html.escape(str(video_label))}</code> · "
            f"<strong>rally span in video</strong> <code>{rally_span}</code> · "
            f"<strong>video FPS</strong> {meta.get('video_fps', 30):.2f}</div>"
            f"<div><strong>Tracked by</strong> <code>{html.escape(meta.get('model_version') or 'unknown')}</code> · "
            f"<strong>at</strong> {html.escape(meta.get('tracked_at') or '—')} · "
            f"<strong>calibration</strong> {calib}</div>"
            f"<div><strong>Video quality flags</strong>: {flags_inner}</div>"
            f"</div>"
        )

    body = f"""
    <h1>Rally <code>{rid[:8]}</code></h1>
    <div class='sub'><a href='index.html'>← back to index</a></div>
    <div class='legend'>
      <strong>Magenta dashed outlines = GT bounding boxes.</strong>
      Solid colored bboxes = predicted tracks (color keyed by track_id, primary tracks thicker).
      Yellow horizontal line = court split (near vs far team).
    </div>
    {summary}
    {conv_summary}
    {meta_html}
    <div class='perGt'>
      <h2>Per-GT-track</h2>
      <table>
        <thead>
          <tr><th>Label</th><th>Coverage</th><th>Matched</th><th>Distinct pred IDs</th><th>Real IDsw</th><th>Time@net</th><th>Missed by cause (frames)</th></tr>
        </thead>
        <tbody>{''.join(gt_rows)}</tbody>
      </table>
    </div>
    <h2>Events ({len(events_with_clips)})</h2>
    <div class='cards'>{''.join(card_html) if card_html else '<em>No events qualified for a clip.</em>'}</div>
    """
    out_path.write_text(f"<!DOCTYPE html><html><head><meta charset='utf-8'>{_RALLY_CSS}<title>Rally {rid[:8]}</title></head><body>{body}</body></html>")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def _load_audits(audit_dir: Path, rally_filter: str | None) -> list[dict]:
    audits = []
    for path in sorted(audit_dir.glob("*.json")):
        if path.name == "_summary.json":
            continue
        with open(path) as f:
            data = json.load(f)
        if rally_filter and not data["rallyId"].startswith(rally_filter):
            continue
        audits.append(data)
    return audits


def _load_retrack_predictions(rally_id: str, ball_positions: list | None = None) -> PlayerTrackingResult | None:
    """Load predictions from the retrack cache (real pipeline output).

    Returns the fully post-processed PlayerTrackingResult, or None if the
    rally is not in the cache.
    """
    from rallycut.cli.commands.evaluate_tracking import _compute_tracker_config_hash
    from rallycut.evaluation.tracking.retrack_cache import RetrackCache
    from rallycut.tracking.player_filter import PlayerFilterConfig
    from rallycut.tracking.player_tracker import PlayerTracker

    cache = RetrackCache()
    config_hash = _compute_tracker_config_hash()
    entry = cache.get(rally_id, config_hash)
    if entry is None:
        return None
    cached_data, color_store, appearance_store, learned_store = entry
    filter_config = PlayerFilterConfig()
    return PlayerTracker.apply_post_processing(
        positions=cached_data.positions,
        raw_positions=list(cached_data.positions),
        color_store=color_store,
        appearance_store=appearance_store,
        ball_positions=ball_positions,
        video_fps=cached_data.video_fps,
        video_width=cached_data.video_width,
        video_height=cached_data.video_height,
        frame_count=cached_data.frame_count,
        start_frame=0,
        filter_enabled=True,
        filter_config=filter_config.scaled_for_fps(cached_data.video_fps),
        court_calibrator=None,
        learned_store=learned_store,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, default=Path("reports/tracking_audit"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/tracking_audit/gallery"))
    parser.add_argument("--rally", type=str, default=None, help="rally id prefix filter")
    parser.add_argument("--max-events-per-rally", type=int, default=6)
    parser.add_argument("--skip-clips", action="store_true", help="HTML only, no ffmpeg/cv2")
    parser.add_argument(
        "--retrack", action="store_true",
        help="Use retrack cache predictions (real pipeline output) instead of DB",
    )
    args = parser.parse_args()

    audits = _load_audits(args.audit_dir, args.rally)
    if not audits:
        logger.error(f"No audit JSONs found in {args.audit_dir}")
        raise SystemExit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    clips_dir = args.out_dir / "clips"
    clips_dir.mkdir(exist_ok=True)

    logger.info(f"Loaded {len(audits)} audits")
    if args.retrack:
        logger.info("Using retrack cache predictions (real pipeline output)")

    for idx, audit in enumerate(audits, start=1):
        rid = audit["rallyId"]
        vid = audit["videoId"]
        logger.info(f"[{idx}/{len(audits)}] {rid[:8]}  video={vid[:8]}")

        events = extract_events(audit, args.max_events_per_rally)
        events_with_clips: list[tuple[Event, str]] = []

        # Load rally data: GT from DB always, predictions from retrack or DB.
        _rallies_cache = load_labeled_rallies(rally_id=rid)
        _rally_cached = _rallies_cache[0] if _rallies_cache else None

        predictions: PlayerTrackingResult | None = None
        video_fps: float = 30.0
        if args.retrack:
            ball_positions = None
            if _rally_cached and _rally_cached.predictions:
                ball_positions = _rally_cached.predictions.ball_positions or None
            predictions = _load_retrack_predictions(rid, ball_positions)
            if predictions is not None:
                video_fps = predictions.video_fps
            elif _rally_cached and _rally_cached.predictions:
                logger.warning(f"  {rid[:8]} not in retrack cache, falling back to DB")
                predictions = _rally_cached.predictions
        elif _rally_cached and _rally_cached.predictions:
            predictions = _rally_cached.predictions

        if predictions is not None:
            video_fps = predictions.video_fps or video_fps

        # Re-classify swap events as recoveries where trajectory continuity
        # indicates the tracker was correctly re-associating (not failing).
        if predictions is not None:
            events = reclassify_swap_recoveries(events, predictions)

        if events and not args.skip_clips:
            rally_start_ms = _ms_at_rally_start(rid) or 0
            video_path = get_video_path(vid)
            if video_path is None:
                logger.warning(f"  no video for {vid}, skipping clips")
            elif predictions is None:
                logger.warning(f"  no predictions for {rid[:8]}, skipping clips")
            else:
                from rallycut.evaluation.tracking.metrics import smart_interpolate_gt
                gt = _rally_cached.ground_truth if _rally_cached else None
                if gt is not None and predictions.frame_count > 0:
                    gt = smart_interpolate_gt(gt, predictions, predictions.frame_count)
                if gt is not None:
                    for eidx, ev in enumerate(events, start=1):
                        clip_name = f"{rid}_{eidx:02d}_{ev.kind}.mp4"
                        clip_path = clips_dir / clip_name
                        ok = render_event_clip(
                            video_path=video_path,
                            rally_start_ms=rally_start_ms,
                            video_fps=video_fps,
                            predictions=predictions,
                            gt_positions=gt.player_positions,
                            event=ev,
                            out_path=clip_path,
                        )
                        rel = os.path.relpath(clip_path, args.out_dir)
                        events_with_clips.append((ev, rel if ok else ""))
                        logger.info(f"  clip {eidx}/{len(events)}: {ev.kind} @ {ev.frame}  {'OK' if ok else 'FAIL'}")
        else:
            events_with_clips = [(ev, "") for ev in events]

        meta = fetch_rally_meta(rid)
        rally_html = args.out_dir / f"{rid}.html"
        render_rally_html(audit, events_with_clips, rally_html, meta=meta)

    # Index last (uses all audits)
    render_index_html(audits, args.out_dir / "index.html")
    logger.info(f"\nGallery written to {args.out_dir}/index.html")


if __name__ == "__main__":
    main()
